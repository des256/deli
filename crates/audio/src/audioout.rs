use {
    crate::*,
    base::*,
    libpulse_binding::{
        callbacks::ListResult,
        context::{Context, FlagSet, State},
        mainloop::standard::{IterateResult, Mainloop},
        operation::State as OperationState,
        sample::{Format, Spec},
        stream::Direction,
    },
    libpulse_simple_binding::Simple,
    std::{
        cell::RefCell,
        collections::VecDeque,
        rc::Rc,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    },
    tokio::sync::mpsc,
};

// capacity of the audio output channel
const CHANNEL_CAPACITY: usize = 16;

// number of seconds to wait before reconnecting to PulseAudio
const RECONNECT_SEC: u64 = 1;

// number of audio frames to be played at once
const CHUNK_SIZE: usize = 256;

// size of the audio ring buffer
const RING_BUFFER_SIZE: usize = 1024 * 1024;

// number of mainloop iterations to wait for connection to be ready
const MAX_MAINLOOP_ITERATIONS: usize = 100;

// audio output device with description
#[derive(Clone, Debug)]
pub struct AudioOutDevice {
    pub name: String,
    pub description: String,
}

// audio output configuration
#[derive(Clone, Debug)]
pub struct AudioOutConfig {
    pub device_name: Option<String>,
    pub sample_rate: usize,
}

impl Default for AudioOutConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            sample_rate: 16000,
        }
    }
}

// audio output
pub struct AudioOut {
    sender: mpsc::Sender<AudioSample>,
    new_config_sender: mpsc::Sender<AudioOutConfig>,
    cancel: Arc<AtomicBool>,
    config: AudioOutConfig,
}

impl AudioOut {
    // open audio output
    pub async fn open(config: Option<AudioOutConfig>) -> Self {
        // current audio configuration
        let config = config.unwrap_or_default();

        // channel for sending audio samples
        let (sender, mut receiver) = mpsc::channel::<AudioSample>(CHANNEL_CAPACITY);

        // channel for sending new audio configurations
        let (new_config_sender, mut new_config_receiver) =
            mpsc::channel::<AudioOutConfig>(CHANNEL_CAPACITY);

        // canceling flag
        let cancel = Arc::new(AtomicBool::new(false));

        // send initial configuration
        if let Err(error) = new_config_sender.send(config.clone()).await {
            log_fatal!("Failed to send initial audio config: {}", error);
        }

        // spawn separate task for audio playback loop
        tokio::task::spawn_blocking({
            let cancel = Arc::clone(&cancel);
            move || {
                // initialize audio ring buffer
                let mut ring_buffer = VecDeque::<i16>::with_capacity(RING_BUFFER_SIZE);
                let mut chunk = [0i16; CHUNK_SIZE];

                // wait for new audio configuration
                while let Some(config) = new_config_receiver.blocking_recv() {
                    // prepare PulseAudio stream specification
                    let spec = Spec {
                        format: Format::S16NE,
                        channels: 1,
                        rate: config.sample_rate as u32,
                    };

                    // reconnect loop
                    while new_config_receiver.len() == 0 {
                        // connect to PulseAudio
                        let pulse = match Simple::new(
                            None,
                            "deli-audio",
                            Direction::Playback,
                            if let Some(ref name) = config.device_name {
                                Some(name.as_str())
                            } else {
                                None
                            },
                            "audio-playback",
                            &spec,
                            None,
                            None,
                        ) {
                            Ok(pulse) => pulse,
                            Err(error) => {
                                log_warn!(
                                    "Failed to connect to PulseAudio, reconnecting...: {}",
                                    error
                                );
                                std::thread::sleep(std::time::Duration::from_millis(RECONNECT_SEC));
                                continue;
                            }
                        };

                        // inner loop until new configuration is requested
                        while new_config_receiver.len() == 0 {
                            // if canceling, clear out the channel and ring buffer
                            if cancel.swap(false, Ordering::Relaxed) {
                                ring_buffer.clear();
                                loop {
                                    match receiver.try_recv() {
                                        Ok(_) => {}
                                        Err(mpsc::error::TryRecvError::Empty) => {
                                            break;
                                        }
                                        Err(mpsc::error::TryRecvError::Disconnected) => {
                                            log_error!("Audio output channel disconnected");
                                            return;
                                        }
                                    }
                                }
                            }

                            // if audio is available from the channel, add it to the ring buffer
                            match receiver.try_recv() {
                                Ok(sample) => match sample.data {
                                    // NOTE: this assumes that the sample rate is the same as the configuration; should resample here (which is not trivial if the producer decides to do some sort of streaming...)
                                    AudioData::Pcm(tensor) => {
                                        ring_buffer.extend(tensor.data.iter())
                                    }
                                },
                                Err(mpsc::error::TryRecvError::Empty) => {}
                                Err(mpsc::error::TryRecvError::Disconnected) => {
                                    log_error!("Audio output channel disconnected");
                                    return;
                                }
                            }

                            // build chunk
                            let size = ring_buffer.len().min(CHUNK_SIZE);
                            for i in 0..size {
                                chunk[i] = ring_buffer.pop_front().unwrap();
                            }
                            for i in size..CHUNK_SIZE {
                                chunk[i] = 0;
                            }

                            // write to PulseAudio
                            let slice = unsafe {
                                std::slice::from_raw_parts(
                                    chunk.as_ptr() as *const u8,
                                    chunk.len() * 2,
                                )
                            };
                            if let Err(error) = pulse.write(slice) {
                                log_warn!("PulseAudio write error: {}", error);
                                std::thread::sleep(std::time::Duration::from_millis(RECONNECT_SEC));
                                break;
                            }
                        }
                    }
                }
            }
        });

        Self {
            sender,
            new_config_sender,
            cancel,
            config,
        }
    }

    // get the current audio configuration
    pub fn config(&self) -> AudioOutConfig {
        self.config.clone()
    }

    // select a new audio configuration
    pub async fn select(&mut self, config: AudioOutConfig) {
        if let Err(error) = self.new_config_sender.send(config.clone()).await {
            log_error!("Failed to send new audio config: {}", error);
        }
        self.config = config;
    }

    // play an audio sample
    pub async fn play(&self, sample: AudioSample) {
        if let Err(error) = self.sender.send(sample).await {
            log_error!("Failed to play audio sample: {}", error);
        }
    }

    // cancel audio playback
    pub async fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    // get list of available audio output devices
    pub async fn list_devices() -> Result<Vec<AudioOutDevice>, AudioError> {
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
            let devices = Rc::new(RefCell::new(Vec::<AudioOutDevice>::new()));
            let devices_clone = Rc::clone(&devices);
            let introspect = context.introspect();
            let op = introspect.get_sink_info_list(move |list_result| {
                if let ListResult::Item(sink_info) = list_result {
                    if let (Some(name), Some(desc)) = (&sink_info.name, &sink_info.description) {
                        devices_clone.borrow_mut().push(AudioOutDevice {
                            name: name.to_string(),
                            description: desc.to_string(),
                        });
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
                "Failed to join audio output device enumeration task: {}",
                error
            ))),
        }
    }
}
