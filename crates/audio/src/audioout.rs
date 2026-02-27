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
        rc::Rc,
        sync::mpsc as std_mpsc,
    },
    tokio::sync::mpsc as tokio_mpsc,
};

// number of seconds to wait before reconnecting to PulseAudio
const RECONNECT_SEC: u64 = 1;

// number of audio frames to be played at once
const CHUNK_SIZE: usize = 256;

// number of mainloop iterations to wait for connection to be ready
const MAX_MAINLOOP_ITERATIONS: usize = 100;

// capacity of the audio output status channel
const STATUS_CHANNEL_CAPACITY: usize = 16;

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

#[derive(Clone, Debug)]
pub struct AudioOutChunk<T: Clone + Send + 'static> {
    pub payload: T,
    pub data: Vec<i16>,
}

pub enum AudioOutStatus<T: Clone + Send + 'static> {
    Started(T),
    Finished { payload: T, index: usize },
    Canceled { payload: T, index: usize },
}

pub struct AudioOutHandle<T: Clone + Send + 'static> {
    output_tx: std_mpsc::Sender<Stamped<AudioOutChunk<T>>>,
    new_config_sender: std_mpsc::Sender<AudioOutConfig>,
    epoch: Epoch,
    config: AudioOutConfig,
}

pub struct AudioOutListener<T: Clone + Send + 'static> {
    status_rx: tokio_mpsc::Receiver<AudioOutStatus<T>>,
}

pub fn create_audioout<T: Clone + Send + 'static>(
    config: Option<AudioOutConfig>,
    epoch: Epoch,
) -> (AudioOutHandle<T>, AudioOutListener<T>) {
    // current audio configuration
    let config = config.unwrap_or_default();

    let (output_tx, output_rx) = std_mpsc::channel::<Stamped<AudioOutChunk<T>>>();
    let (new_config_sender, new_config_receiver) = std_mpsc::channel::<AudioOutConfig>();
    let (status_tx, status_rx) = tokio_mpsc::channel::<AudioOutStatus<T>>(STATUS_CHANNEL_CAPACITY);

    // send initial configuration
    if let Err(error) = new_config_sender.send(config.clone()) {
        log_fatal!("Failed to send initial audio config: {}", error);
    }

    // spawn separate task for audio playback loop
    std::thread::spawn({
        let epoch = epoch.clone();
        move || {
            // active chunk being played
            struct ActiveChunk<T> {
                payload: T,
                data: Vec<i16>,
                epoch: u64,
            }

            let mut current_chunk: Option<ActiveChunk<T>> = None;
            let mut current_index = 0usize;
            let mut buffer = [0i16; CHUNK_SIZE];

            // wait for new audio configuration
            while let Ok(config) = new_config_receiver.recv() {
                // prepare PulseAudio stream specification
                let spec = Spec {
                    format: Format::S16NE,
                    channels: 1,
                    rate: config.sample_rate as u32,
                };

                // reconnect loop
                while let Err(std_mpsc::TryRecvError::Empty) = new_config_receiver.try_recv() {
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
                    while let Err(std_mpsc::TryRecvError::Empty) = new_config_receiver.try_recv() {
                        // check if current chunk is stale
                        if let Some(chunk) = &current_chunk {
                            if !epoch.is_current(chunk.epoch) {
                                let chunk = current_chunk.take().unwrap();
                                if let Err(error) = status_tx.blocking_send(
                                    AudioOutStatus::Canceled {
                                        payload: chunk.payload,
                                        index: current_index,
                                    },
                                ) {
                                    log_error!("Failed to send canceled status: {}", error);
                                    return;
                                }
                                current_index = 0;
                            }
                        }

                        // build buffer for PulseAudio
                        let mut i = 0usize;
                        while i < CHUNK_SIZE {
                            if let Some(chunk) = &current_chunk {
                                let mut n = chunk.data.len() - current_index;
                                if n > CHUNK_SIZE - i {
                                    n = CHUNK_SIZE - i;
                                }
                                buffer[i..i + n]
                                    .copy_from_slice(&chunk.data[current_index..current_index + n]);
                                current_index += n;
                                i += n;
                                if current_index >= chunk.data.len() {
                                    if let Err(error) =
                                        status_tx.blocking_send(AudioOutStatus::Finished {
                                            payload: chunk.payload.clone(),
                                            index: chunk.data.len(),
                                        })
                                    {
                                        log_error!("Failed to send finished status: {}", error);
                                        return;
                                    }
                                    current_chunk = None;
                                }
                            } else {
                                match output_rx.try_recv() {
                                    Ok(stamped) => {
                                        // drop stale chunks
                                        if !epoch.is_current(stamped.epoch) {
                                            continue;
                                        }
                                        let chunk = stamped.inner;
                                        if let Err(error) = status_tx.blocking_send(
                                            AudioOutStatus::Started(chunk.payload.clone()),
                                        ) {
                                            log_error!(
                                                "Failed to send started status: {}",
                                                error
                                            );
                                            return;
                                        }
                                        current_chunk = Some(ActiveChunk {
                                            payload: chunk.payload,
                                            data: chunk.data,
                                            epoch: stamped.epoch,
                                        });
                                        current_index = 0;
                                    }
                                    Err(std_mpsc::TryRecvError::Empty) => {
                                        if i < CHUNK_SIZE {
                                            buffer[i..].fill(0);
                                            i = CHUNK_SIZE;
                                        }
                                    }
                                    Err(std_mpsc::TryRecvError::Disconnected) => {
                                        log_error!("Audio output channel disconnected");
                                        return;
                                    }
                                }
                            }
                        }

                        // write to PulseAudio
                        let slice = unsafe {
                            std::slice::from_raw_parts(
                                buffer.as_ptr() as *const u8,
                                buffer.len() * 2,
                            )
                        };
                        if let Err(error) = pulse.write(slice) {
                            // potentially blocking
                            log_warn!("PulseAudio write error: {}", error);
                            std::thread::sleep(std::time::Duration::from_millis(RECONNECT_SEC));
                            break;
                        }
                    }
                }
            }
        }
    });

    (
        AudioOutHandle {
            output_tx,
            new_config_sender,
            epoch,
            config,
        },
        AudioOutListener { status_rx },
    )
}

impl<T: Clone + Send + 'static> AudioOutHandle<T> {
    // get the current audio configuration
    pub fn config(&self) -> AudioOutConfig {
        self.config.clone()
    }

    // select a new audio configuration
    pub fn select(&mut self, config: AudioOutConfig) {
        if let Err(error) = self.new_config_sender.send(config.clone()) {
            log_error!("Failed to send new audio config: {}", error);
        }
        self.config = config;
    }

    // send audio chunk to audio output (stamped with current epoch)
    pub fn send(
        &self,
        chunk: AudioOutChunk<T>,
    ) -> Result<(), std_mpsc::SendError<Stamped<AudioOutChunk<T>>>> {
        self.output_tx.send(Stamped {
            epoch: self.epoch.current(),
            inner: chunk,
        })
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

impl<T: Clone + Send + 'static> AudioOutListener<T> {
    pub async fn recv(&mut self) -> Option<AudioOutStatus<T>> {
        self.status_rx.recv().await
    }

    pub fn try_recv(&mut self) -> Option<AudioOutStatus<T>> {
        match self.status_rx.try_recv() {
            Ok(status) => Some(status),
            _ => None,
        }
    }
}
