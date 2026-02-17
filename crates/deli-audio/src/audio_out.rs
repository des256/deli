use crate::AudioError;
use libpulse_simple_binding::Simple;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use tokio::sync::mpsc;

/// Audio output playback using PulseAudio.
///
/// Plays mono S16NE audio to the selected output device via a background tokio blocking task.
/// The async `send()` method sends audio chunks to the playback task.
///
/// # Panics
///
/// `new()` panics if called outside a tokio runtime context. All tests must use `#[tokio::test]`.
///
/// # Examples
///
/// ```no_run
/// use deli_audio::AudioOut;
///
/// async fn example() {
///     // Create AudioOut with default device at 48kHz
///     let mut audio_out = AudioOut::new(None, 48000);
///
///     // Send audio data
///     let samples = vec![0i16; 4800]; // 100ms of silence at 48kHz
///     audio_out.send(&samples).await.unwrap();
/// }
/// ```
pub struct AudioOut {
    sample_rate: u32,
    device: Option<String>,
    sender: Option<mpsc::Sender<Vec<i16>>>,
    task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl std::fmt::Debug for AudioOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioOut")
            .field("sample_rate", &self.sample_rate)
            .field("device", &self.device)
            .field("sender", &self.sender.is_some())
            .field("task_handle", &self.task_handle.is_some())
            .finish()
    }
}

impl AudioOut {
    /// Create a new AudioOut instance and start playback task immediately.
    ///
    /// # Arguments
    ///
    /// * `device` - PulseAudio sink name (from `AudioDevice::name`), or `None` for default device
    /// * `sample_rate` - Sample rate in Hz (e.g., 48000, 44100)
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime context. All callers must be within a tokio runtime.
    /// Panics if `sample_rate` is 0.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use deli_audio::AudioOut;
    ///
    /// // Use default output device at 48kHz
    /// let audio_out = AudioOut::new(None, 48000);
    ///
    /// // Use specific device at 44.1kHz
    /// let audio_out_device = AudioOut::new(
    ///     Some("alsa_output.pci-0000_00_1f.3.analog-stereo"),
    ///     44100
    /// );
    /// ```
    pub fn new(device: Option<&str>, sample_rate: u32) -> Self {
        assert!(sample_rate > 0, "sample_rate must be greater than 0");
        let device_string = device.map(|s| s.to_string());
        let (sender, task_handle) = Self::start_playback(
            device_string.clone(),
            sample_rate
        );

        Self {
            sample_rate,
            device: device_string,
            sender: Some(sender),
            task_handle: Some(task_handle),
        }
    }

    /// Send audio data to the playback task.
    ///
    /// Sends a chunk of mono S16NE samples to the background playback task.
    /// If the channel is full, this method waits until space is available (backpressure).
    ///
    /// # Errors
    ///
    /// Returns `AudioError::Channel` if the sender is not initialized.
    /// Returns `AudioError::Stream` if the playback task has terminated.
    pub async fn send(&mut self, data: &[i16]) -> Result<(), AudioError> {
        let sender = self.sender.as_ref().ok_or_else(|| {
            AudioError::Channel("Sender not initialized".to_string())
        })?;

        let data_vec = data.to_vec();
        sender.send(data_vec).await.map_err(|_| {
            AudioError::Stream("playback task terminated".to_string())
        })
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the device name.
    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    /// Cancel current playback and restart fresh playback task.
    ///
    /// Stops the current playback, flushes the PulseAudio buffer, and starts a new
    /// playback task on the same device. The AudioOut is immediately ready for new send() calls.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use deli_audio::AudioOut;
    ///
    /// async fn example() {
    ///     let mut audio_out = AudioOut::new(None, 48000);
    ///     // ... send some audio ...
    ///     audio_out.cancel().await; // Stop and flush
    ///     // Ready for new audio
    /// }
    /// ```
    pub async fn cancel(&mut self) {
        // Drop sender to signal task to stop
        drop(self.sender.take());

        // Wait for task to finish
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }

        // Start new playback task with same device
        let (sender, task_handle) = Self::start_playback(
            self.device.clone(),
            self.sample_rate
        );

        self.sender = Some(sender);
        self.task_handle = Some(task_handle);
    }

    /// Select a different audio output device.
    ///
    /// Tears down the current playback stream and starts a new one on the selected device.
    ///
    /// # Arguments
    ///
    /// * `device` - PulseAudio sink name (from `AudioDevice::name`)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use deli_audio::AudioOut;
    ///
    /// async fn example() {
    ///     let mut audio_out = AudioOut::new(None, 48000);
    ///     audio_out.select("alsa_output.pci-0000_00_1f.3.analog-stereo").await;
    /// }
    /// ```
    pub async fn select(&mut self, device: &str) {
        // Drop sender to signal task to stop
        drop(self.sender.take());

        // Wait for task to finish
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }

        // Update device name
        self.device = Some(device.to_string());

        // Start new playback task
        let (sender, task_handle) = Self::start_playback(
            self.device.clone(),
            self.sample_rate
        );

        self.sender = Some(sender);
        self.task_handle = Some(task_handle);
    }

    /// Start a new playback task.
    ///
    /// Creates an mpsc channel and spawns a blocking task that runs the playback loop.
    /// Returns the sender and task handle.
    fn start_playback(
        device: Option<String>,
        sample_rate: u32,
    ) -> (mpsc::Sender<Vec<i16>>, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(4);

        let handle = tokio::task::spawn_blocking(move || {
            Self::playback_loop(device, sample_rate, rx);
        });

        (tx, handle)
    }

    /// Background task playback loop.
    ///
    /// Creates a PulseAudio Simple stream and writes audio chunks.
    /// Automatically recovers from errors by reconnecting after 100ms.
    fn playback_loop(
        device: Option<String>,
        sample_rate: u32,
        mut rx: mpsc::Receiver<Vec<i16>>,
    ) {
        // Create PulseAudio spec for mono S16NE (static, created once)
        let spec = Spec {
            format: Format::S16NE,
            channels: 1,
            rate: sample_rate,
        };

        if !spec.is_valid() {
            log::warn!("Invalid audio specification");
            return;
        }

        // Outer loop: connection/reconnection
        loop {
            // Check channel status via try_recv() before attempting connection
            let pending_data = match rx.try_recv() {
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // Sender dropped - exit immediately
                    return;
                }
                Ok(data) => Some(data),
                Err(mpsc::error::TryRecvError::Empty) => None,
            };

            // Create PulseAudio Simple stream
            let simple = match Simple::new(
                None,                              // Use default server
                "deli-audio",                      // Application name
                Direction::Playback,               // Playback direction
                device.as_deref(),                 // Device name
                "audio-playback",                  // Stream description
                &spec,                             // Sample format
                None,                              // Default channel map
                None,                              // Default buffer attributes
            ) {
                Ok(s) => s,
                Err(e) => {
                    log::warn!("Failed to connect to PulseAudio: {}", e);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            };

            // Write pending data if any
            if let Some(samples) = pending_data {
                let bytes: Vec<u8> = samples.iter()
                    .flat_map(|s| s.to_ne_bytes())
                    .collect();

                if let Err(e) = simple.write(&bytes) {
                    log::warn!("PulseAudio write error: {}", e);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            }

            // Inner loop: receiving and writing audio chunks
            loop {
                match rx.blocking_recv() {
                    Some(samples) => {
                        // Convert i16 samples to bytes
                        let bytes: Vec<u8> = samples.iter()
                            .flat_map(|s| s.to_ne_bytes())
                            .collect();

                        // Write to PulseAudio
                        if let Err(e) = simple.write(&bytes) {
                            log::warn!("PulseAudio write error: {}", e);
                            break;
                        }
                    }
                    None => {
                        // Channel closed - flush and exit
                        let _ = simple.flush();
                        return;
                    }
                }
            }

            // Simple is dropped here (stream closed)
            // Sleep before reconnecting
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

impl Drop for AudioOut {
    fn drop(&mut self) {
        // Drop the sender to signal the task to stop
        drop(self.sender.take());

        // Drop the task handle (task will exit on its own when channel closes)
        drop(self.task_handle.take());
    }
}
