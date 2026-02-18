use crate::AudioError;
use crate::audiosample::{AudioData, AudioSample};
use futures_sink::Sink;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::Direction;
use libpulse_simple_binding::Simple;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

/// Audio output playback using PulseAudio.
///
/// Plays mono S16NE audio to the selected output device via a background tokio blocking task.
/// Implements `Sink<AudioSample>` for async sending of audio data.
///
/// # Panics
///
/// `new()` panics if called outside a tokio runtime context. All tests must use `#[tokio::test]`.
///
/// # Examples
///
/// ```no_run
/// use deli_audio::{AudioOut, AudioData, AudioSample};
/// use deli_base::Tensor;
/// use futures_util::SinkExt;
///
/// async fn example() {
///     let mut audio_out = AudioOut::new(None, 48000);
///
///     let samples = vec![0i16; 4800]; // 100ms of silence at 48kHz
///     let tensor = Tensor::new(vec![4800], samples).unwrap();
///     audio_out.send(AudioSample {
///         data: AudioData::Pcm(tensor),
///         sample_rate: 48000,
///     }).await.unwrap();
/// }
/// ```
pub struct AudioOut {
    sample_rate: usize,
    device: Option<String>,
    sender: Option<mpsc::Sender<Vec<i16>>>,
    task_handle: Option<tokio::task::JoinHandle<()>>,
    inflight: Option<Pin<Box<dyn Future<Output = Result<(), ()>> + Send>>>,
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
    pub fn new(device: Option<&str>, sample_rate: usize) -> Self {
        assert!(sample_rate > 0, "sample_rate must be greater than 0");
        let device_string = device.map(|s| s.to_string());
        let (sender, task_handle) = Self::start_playback(device_string.clone(), sample_rate);

        Self {
            sample_rate,
            device: device_string,
            sender: Some(sender),
            task_handle: Some(task_handle),
            inflight: None,
        }
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Get the device name.
    pub fn device(&self) -> Option<&str> {
        self.device.as_deref()
    }

    /// Cancel current playback and restart fresh playback task.
    ///
    /// Stops the current playback, flushes the PulseAudio buffer, and starts a new
    /// playback task on the same device. The AudioOut is immediately ready for new audio.
    pub async fn cancel(&mut self) {
        // Drop inflight future (releases any cloned sender)
        self.inflight = None;

        // Drop sender to signal task to stop
        drop(self.sender.take());

        // Wait for task to finish
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }

        // Start new playback task with same device
        let (sender, task_handle) = Self::start_playback(self.device.clone(), self.sample_rate);

        self.sender = Some(sender);
        self.task_handle = Some(task_handle);
    }

    /// Select a different audio output device.
    ///
    /// Tears down the current playback stream and starts a new one on the selected device.
    pub async fn select(&mut self, device: &str) {
        // Drop inflight future (releases any cloned sender)
        self.inflight = None;

        // Drop sender to signal task to stop
        drop(self.sender.take());

        // Wait for task to finish
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }

        // Update device name
        self.device = Some(device.to_string());

        // Start new playback task
        let (sender, task_handle) = Self::start_playback(self.device.clone(), self.sample_rate);

        self.sender = Some(sender);
        self.task_handle = Some(task_handle);
    }

    /// Poll an in-flight send to completion.
    fn poll_inflight(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), AudioError>> {
        if let Some(fut) = self.inflight.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Ready(Ok(())) => {
                    self.inflight = None;
                    Poll::Ready(Ok(()))
                }
                Poll::Ready(Err(())) => {
                    self.inflight = None;
                    Poll::Ready(Err(AudioError::Stream(
                        "playback task terminated".to_string(),
                    )))
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Ready(Ok(()))
        }
    }

    /// Start a new playback task.
    fn start_playback(
        device: Option<String>,
        sample_rate: usize,
    ) -> (mpsc::Sender<Vec<i16>>, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(4);

        let handle = tokio::task::spawn_blocking(move || {
            Self::playback_loop(device, sample_rate, rx);
        });

        (tx, handle)
    }

    /// Background task playback loop.
    fn playback_loop(device: Option<String>, sample_rate: usize, mut rx: mpsc::Receiver<Vec<i16>>) {
        let spec = Spec {
            format: Format::S16NE,
            channels: 1,
            rate: sample_rate as u32,
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
                    return;
                }
                Ok(data) => Some(data),
                Err(mpsc::error::TryRecvError::Empty) => None,
            };

            // Create PulseAudio Simple stream
            let simple = match Simple::new(
                None,
                "deli-audio",
                Direction::Playback,
                device.as_deref(),
                "audio-playback",
                &spec,
                None,
                None,
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
                let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_ne_bytes()).collect();

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
                        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_ne_bytes()).collect();

                        if let Err(e) = simple.write(&bytes) {
                            log::warn!("PulseAudio write error: {}", e);
                            break;
                        }
                    }
                    None => {
                        let _ = simple.flush();
                        return;
                    }
                }
            }

            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

impl Sink<AudioSample> for AudioOut {
    type Error = AudioError;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.get_mut().poll_inflight(cx)
    }

    fn start_send(self: Pin<&mut Self>, item: AudioSample) -> Result<(), Self::Error> {
        let this = self.get_mut();
        let sender = this
            .sender
            .clone()
            .ok_or_else(|| AudioError::Channel("Sender not initialized".to_string()))?;

        let AudioData::Pcm(tensor) = item.data;
        let data = tensor.data;

        this.inflight = Some(Box::pin(
            async move { sender.send(data).await.map_err(|_| ()) },
        ));

        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.get_mut().poll_inflight(cx)
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        this.inflight = None;
        this.sender = None;
        Poll::Ready(Ok(()))
    }
}

impl Drop for AudioOut {
    fn drop(&mut self) {
        self.inflight = None;
        drop(self.sender.take());
        drop(self.task_handle.take());
    }
}
