use {
    crate::*,
    base::Tensor,
    libpulse_binding::{
        callbacks::ListResult,
        context::{Context, FlagSet, State},
        def::BufferAttr,
        mainloop::standard::{IterateResult, Mainloop},
        operation::State as OperationState,
        sample::{Format, Spec},
        stream::Direction,
    },
    libpulse_simple_binding::Simple,
    std::{cell::RefCell, rc::Rc},
    tokio::sync::mpsc,
};

// capacity of the audio input channel
const CHANNEL_CAPACITY: usize = 16;

// number of seconds to wait before reconnecting to PulseAudio
const RECONNECT_SEC: u64 = 1;

// number of mainloop iterations to wait for connection to be ready
const MAX_MAINLOOP_ITERATIONS: usize = 100;

// audio input device with description
#[derive(Clone, Debug)]
pub struct AudioInDevice {
    pub name: String,
    pub description: String,
}

// audio input configuration
#[derive(Clone, Debug)]
pub struct AudioInConfig {
    pub device_name: Option<String>,
    pub sample_rate: usize,
    pub chunk_size: usize,
}

impl Default for AudioInConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            sample_rate: 16000,
            chunk_size: 1600,
        }
    }
}

// audio input
pub struct AudioIn {
    receiver: mpsc::Receiver<AudioSample>,
    new_config_sender: mpsc::Sender<AudioInConfig>,
    config: AudioInConfig,
}

impl AudioIn {
    // open audio input
    pub async fn open() -> Self {
        // channel for receiving audio samples
        let (sender, receiver) = mpsc::channel::<AudioSample>(CHANNEL_CAPACITY);

        // channel for sending new audio configurations
        let (new_config_sender, mut new_config_receiver) =
            mpsc::channel::<AudioInConfig>(CHANNEL_CAPACITY);

        // current audio configuration
        let config = AudioInConfig::default();

        // spawn separate task for audio capture loop
        tokio::task::spawn_blocking(move || {
            // wait for new audio configuration
            while let Some(config) = new_config_receiver.blocking_recv() {
                // prepare PulseAudio stream specification and buffer
                let spec = Spec {
                    format: Format::S16NE,
                    channels: 1,
                    rate: config.sample_rate as u32,
                };
                let bytes_per_chunk = config.chunk_size * 2;
                let mut buffer = vec![0u8; bytes_per_chunk];
                let buffer_attr = BufferAttr {
                    maxlength: bytes_per_chunk as u32 * 16,
                    tlength: u32::MAX,
                    prebuf: u32::MAX,
                    minreq: u32::MAX,
                    fragsize: bytes_per_chunk as u32,
                };

                // reconnect loop
                while new_config_receiver.len() == 0 {
                    // connect to PulseAudio
                    let pulse = match Simple::new(
                        None,
                        "deli-audio",
                        Direction::Record,
                        if let Some(ref name) = config.device_name {
                            Some(name.as_str())
                        } else {
                            None
                        },
                        "audio-capture",
                        &spec,
                        None,
                        Some(&buffer_attr),
                    ) {
                        Ok(pulse) => pulse,
                        Err(error) => {
                            log::warn!(
                                "Failed to connect to PulseAudion, reconnecting...: {}",
                                error
                            );
                            std::thread::sleep(std::time::Duration::from_millis(RECONNECT_SEC));
                            continue;
                        }
                    };

                    // inner loop until new configuration is requested
                    while new_config_receiver.len() == 0 {
                        // wait for next audio chunk
                        match pulse.read(&mut buffer) {
                            Ok(()) => {
                                // turn the chunk into audio sample
                                let slice = unsafe {
                                    std::slice::from_raw_parts(
                                        buffer.as_ptr() as *const i16,
                                        buffer.len() / 2,
                                    )
                                };
                                let tensor =
                                    Tensor::new(vec![slice.len()], slice.to_vec()).unwrap();
                                let sample = AudioSample {
                                    data: AudioData::Pcm(tensor),
                                    sample_rate: config.sample_rate as usize,
                                };

                                // send the sample
                                match sender.try_send(sample) {
                                    Ok(()) => {}

                                    // if channel is full, silently drop the chunk
                                    Err(mpsc::error::TrySendError::Full(_)) => {
                                        log::debug!("Audio input chunk dropped: consumer too slow");
                                    }

                                    // if channel is closed, exit everything
                                    Err(mpsc::error::TrySendError::Closed(_)) => {
                                        return;
                                    }
                                }
                            }

                            // if read error, break inner loop to reconnect
                            Err(error) => {
                                log::warn!("PulseAudio read error: {}", error);
                                std::thread::sleep(std::time::Duration::from_millis(RECONNECT_SEC));
                                break;
                            }
                        }
                    }
                }
            }
        });

        // start default configuration
        if let Err(error) = new_config_sender.send(config.clone()).await {
            log::error!("Failed to send default audio config: {}", error);
        }

        Self {
            receiver,
            new_config_sender,
            config,
        }
    }

    // get the current audio configuration
    pub fn config(&self) -> AudioInConfig {
        self.config.clone()
    }

    // select a new audio configuration
    pub async fn select(&mut self, config: AudioInConfig) {
        if let Err(error) = self.new_config_sender.send(config.clone()).await {
            log::error!("Failed to send new audio config: {}", error);
        }
        self.config = config;
    }

    // capture the next audio chunk
    pub async fn capture(&mut self) -> Result<AudioSample, AudioError> {
        match self.receiver.recv().await {
            Some(sample) => Ok(sample),
            None => Err(AudioError::Stream("Audio input channel closed".to_string())),
        }
    }

    // get list of available audio input devices
    pub async fn list_devices() -> Result<Vec<AudioInDevice>, AudioError> {
        let join_handle = tokio::task::spawn_blocking(move || {
            let mut mainloop = Mainloop::new().ok_or_else(|| {
                AudioError::Device("Failed to create PulseAudio mainloop".to_string())
            })?;
            let mut context = Context::new(&mainloop, "deli-audio").ok_or_else(|| {
                AudioError::Device("Failed to create PulseAudio context".to_string())
            })?;
            context.connect(None, FlagSet::NOFLAGS, None).map_err(|e| {
                AudioError::Device(format!("Failed to connect to PulseAudio server: {}", e))
            })?;
            let mut iterations = MAX_MAINLOOP_ITERATIONS;
            while iterations > 0 {
                match mainloop.iterate(true) {
                    IterateResult::Quit(_) | IterateResult::Err(_) => {
                        return Err(AudioError::Device(
                            "PulseAudio mainloop error during connection".to_string(),
                        ));
                    }
                    IterateResult::Success(_) => {}
                }
                match context.get_state() {
                    State::Ready => break,
                    State::Failed | State::Terminated => {
                        return Err(AudioError::Device(
                            "PulseAudio connection failed or terminated".to_string(),
                        ));
                    }
                    _ => {}
                }
                iterations -= 1;
                if iterations == 0 {
                    return Err(AudioError::Device(
                        "PulseAudio server unavailable or connection timed out".to_string(),
                    ));
                }
            }
            let devices = Rc::new(RefCell::new(Vec::<AudioInDevice>::new()));
            let devices_clone = Rc::clone(&devices);
            let introspect = context.introspect();
            let op = introspect.get_source_info_list(move |list_result| {
                if let ListResult::Item(source_info) = list_result {
                    if source_info.monitor_of_sink.is_none() {
                        if let (Some(name), Some(desc)) =
                            (&source_info.name, &source_info.description)
                        {
                            devices_clone.borrow_mut().push(AudioInDevice {
                                name: name.to_string(),
                                description: desc.to_string(),
                            });
                        }
                    }
                }
            });
            loop {
                match mainloop.iterate(true) {
                    IterateResult::Quit(_) | IterateResult::Err(_) => {
                        return Err(AudioError::Device(
                            "Mainloop error during device enumeration".to_string(),
                        ));
                    }
                    IterateResult::Success(_) => {}
                }
                match op.get_state() {
                    OperationState::Done => break,
                    OperationState::Cancelled => {
                        return Err(AudioError::Device(
                            "Device enumeration cancelled".to_string(),
                        ));
                    }
                    OperationState::Running => {}
                }
            }
            let result = devices.borrow().clone();
            Ok(result)
        });
        match join_handle.await {
            Ok(result) => result,
            Err(error) => Err(AudioError::Device(format!(
                "Failed to join audio input device enumeration task: {}",
                error
            ))),
        }
    }
}
