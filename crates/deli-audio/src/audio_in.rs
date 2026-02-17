use crate::AudioError;
use libpulse_simple_binding::Simple;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use tokio::sync::mpsc;

/// Audio input capture using PulseAudio.
///
/// Captures mono S16NE audio from the selected input device via a background tokio blocking task.
/// The async `recv()` method returns the next audio chunk.
///
/// # Panics
///
/// `new()` panics if called outside a tokio runtime context. All tests must use `#[tokio::test]`.
///
/// # Examples
///
/// ```no_run
/// use deli_audio::AudioIn;
///
/// async fn example() {
///     // Create AudioIn with default device, 48kHz sample rate, 4800 frames per chunk (100ms)
///     let mut audio_in = AudioIn::new(None, 48000, 4800);
///
///     loop {
///         match audio_in.recv().await {
///             Ok(samples) => println!("Received {} samples", samples.len()),
///             Err(e) => {
///                 eprintln!("Error: {}", e);
///                 break;
///             }
///         }
///     }
/// }
/// ```
pub struct AudioIn {
    sample_rate: u32,
    chunk_frames: u32,
    device: Option<String>,
    receiver: Option<mpsc::Receiver<Vec<i16>>>,
    task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl std::fmt::Debug for AudioIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioIn")
            .field("sample_rate", &self.sample_rate)
            .field("chunk_frames", &self.chunk_frames)
            .field("device", &self.device)
            .field("receiver", &self.receiver.is_some())
            .field("task_handle", &self.task_handle.is_some())
            .finish()
    }
}

impl AudioIn {
    /// Create a new AudioIn instance and start capture immediately.
    ///
    /// # Arguments
    ///
    /// * `device` - PulseAudio source name (from `AudioDevice::name`), or `None` for default device
    /// * `sample_rate` - Sample rate in Hz (e.g., 48000, 44100)
    /// * `chunk_frames` - Number of frames per audio chunk (e.g., 4800 for 100ms at 48kHz)
    ///
    /// # Panics
    ///
    /// Panics if called outside a tokio runtime context. All callers must be within a tokio runtime.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use deli_audio::AudioIn;
    ///
    /// // Use default input device at 48kHz with 100ms chunks
    /// let audio_in = AudioIn::new(None, 48000, 4800);
    ///
    /// // Use specific device at 44.1kHz with 100ms chunks
    /// let audio_in_device = AudioIn::new(
    ///     Some("alsa_input.pci-0000_00_1f.3.analog-stereo"),
    ///     44100,
    ///     4410
    /// );
    /// ```
    pub fn new(device: Option<&str>, sample_rate: u32, chunk_frames: u32) -> Self {
        assert!(chunk_frames > 0, "chunk_frames must be greater than 0");
        let device_string = device.map(|s| s.to_string());
        let (receiver, task_handle) = Self::start_capture(
            device_string.clone(),
            sample_rate,
            chunk_frames
        );

        Self {
            sample_rate,
            chunk_frames,
            device: device_string,
            receiver: Some(receiver),
            task_handle: Some(task_handle),
        }
    }

    /// Receive the next audio chunk.
    ///
    /// Returns a `Vec<i16>` containing mono S16NE samples.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::Channel` if the receiver is not initialized.
    /// Returns `AudioError::Stream` if the capture task has terminated.
    pub async fn recv(&mut self) -> Result<Vec<i16>, AudioError> {
        let receiver = self.receiver.as_mut().ok_or_else(|| {
            AudioError::Channel("Receiver not initialized".to_string())
        })?;

        receiver.recv().await.ok_or_else(|| {
            AudioError::Stream("capture task terminated".to_string())
        })
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the chunk size in frames.
    pub fn chunk_frames(&self) -> u32 {
        self.chunk_frames
    }

    /// Get the device name.
    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    /// Select a different audio input device.
    ///
    /// Tears down the current capture stream and starts a new one on the selected device.
    ///
    /// # Arguments
    ///
    /// * `device` - PulseAudio source name (from `AudioDevice::name`)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use deli_audio::AudioIn;
    ///
    /// async fn example() {
    ///     let mut audio_in = AudioIn::new(None, 48000, 4800);
    ///     audio_in.select("alsa_input.pci-0000_00_1f.3.analog-stereo").await;
    /// }
    /// ```
    pub async fn select(&mut self, device: &str) {
        // Drop receiver to signal task to stop
        drop(self.receiver.take());

        // Wait for task to finish
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }

        // Update device name
        self.device = Some(device.to_string());

        // Start new capture task
        let (receiver, task_handle) = Self::start_capture(
            self.device.clone(),
            self.sample_rate,
            self.chunk_frames
        );

        self.receiver = Some(receiver);
        self.task_handle = Some(task_handle);
    }

    /// Start a new capture task.
    ///
    /// Creates an mpsc channel and spawns a blocking task that runs the capture loop.
    /// Returns the receiver and task handle.
    fn start_capture(
        device: Option<String>,
        sample_rate: u32,
        chunk_frames: u32,
    ) -> (mpsc::Receiver<Vec<i16>>, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(4);

        let handle = tokio::task::spawn_blocking(move || {
            Self::capture_loop(device, sample_rate, chunk_frames, tx);
        });

        (rx, handle)
    }

    /// Background task capture loop.
    ///
    /// Creates a PulseAudio Simple stream and reads audio chunks.
    /// Automatically recovers from errors by reconnecting after 100ms.
    fn capture_loop(
        device: Option<String>,
        sample_rate: u32,
        chunk_frames: u32,
        tx: mpsc::Sender<Vec<i16>>,
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

        // Calculate buffer size using chunk_frames (static, created once)
        let bytes_per_chunk = chunk_frames as usize * 2; // 2 bytes per i16 sample
        let mut buffer = vec![0u8; bytes_per_chunk];

        // Outer loop: connection/reconnection
        loop {
            // Check if receiver is closed before attempting connection
            if tx.is_closed() {
                break;
            }

            // Create PulseAudio Simple stream
            let simple = match Simple::new(
                None,                              // Use default server
                "deli-audio",                      // Application name
                Direction::Record,                 // Recording direction
                device.as_deref(),                 // Device name
                "audio-capture",                   // Stream description
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

            // Inner loop: reading audio chunks
            loop {
                // Read audio data
                let read_result = simple.read(&mut buffer);

                match read_result {
                    Ok(()) => {
                        // Convert bytes to i16 samples
                        let samples: Vec<i16> = buffer
                            .chunks_exact(2)
                            .map(|chunk| i16::from_ne_bytes([chunk[0], chunk[1]]))
                            .collect();

                        // Send through channel
                        match tx.try_send(samples) {
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                log::debug!("Audio chunk dropped: consumer too slow");
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                // Receiver dropped - exit task completely
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        // Read error - log and break inner loop to reconnect
                        log::warn!("PulseAudio read error: {}", e);
                        break;
                    }
                }
            }

            // Simple is dropped here (stream closed)
            // Sleep before reconnecting
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

impl Drop for AudioIn {
    fn drop(&mut self) {
        // Drop the receiver to signal the task to stop
        drop(self.receiver.take());

        // Drop the task handle (task will exit on its own when channel closes)
        drop(self.task_handle.take());
    }
}
