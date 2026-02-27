use {
    crate::*,
    base::*,
    rand_distr::{Distribution, Normal},
    std::{
        collections::HashMap,
        io::Read,
        path::Path,
        sync::{Arc, mpsc as std_mpsc},
    },
    tokio::sync::mpsc as tokio_mpsc,
};

const CONDITIONER_PATH: &str = "data/pocket/text_conditioner.onnx";
const FLOW_MAIN_PATH: &str = "data/pocket/flow_lm_main_int8.onnx";
const FLOW_STEP_PATH: &str = "data/pocket/flow_lm_flow_int8.onnx";
const DECODER_PATH: &str = "data/pocket/mimi_decoder_int8.onnx";
const TOKENIZER_PATH: &str = "data/pocket/tokenizer.json";

const MAX_TOKENS: usize = 1000;
const LATENT_DIM: usize = 32;
const CONDITIONING_DIM: usize = 1024;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_LSD_STEPS: usize = 1;
const DEFAULT_EOS_THRESHOLD: f32 = -4.0;

fn load_voice(path: impl AsRef<Path>) -> Result<Vec<f32>, InferError> {
    let path = path.as_ref();
    let mut file = std::fs::File::open(path).map_err(|e| {
        InferError::Runtime(format!(
            "Failed to open voice latents file '{}': {}",
            path.display(),
            e
        ))
    })?;

    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4)
        .map_err(|e| InferError::Runtime(format!("Failed to read latents header: {}", e)))?;
    let ndims = u32::from_le_bytes(buf4) as usize;

    let mut total_elements: usize = 1;
    for _ in 0..ndims {
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8).map_err(|e| {
            InferError::Runtime(format!("Failed to read latents dimensions: {}", e))
        })?;
        let dim = u64::from_le_bytes(buf8) as usize;
        total_elements = total_elements
            .checked_mul(dim)
            .ok_or_else(|| InferError::Runtime("Voice latents shape overflow".to_string()))?;
    }

    let mut data = vec![0f32; total_elements];
    let byte_slice = unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            total_elements * std::mem::size_of::<f32>(),
        )
    };
    file.read_exact(byte_slice)
        .map_err(|e| InferError::Runtime(format!("Failed to read latents data: {}", e)))?;

    Ok(data)
}

fn get_names(
    session: &onnx::Session,
    exclude: &[&str],
) -> Result<(Vec<String>, Vec<String>), InferError> {
    let mut input_names: Vec<String> = Vec::new();

    // get number of inputs
    let input_count = session
        .input_count()
        .map_err(|error| InferError::Runtime(format!("Failed to get input count: {}", error)))?;

    // get each name
    for i in 0..input_count {
        let name = session
            .input_name(i)
            .map_err(|error| InferError::Runtime(format!("Failed to get input name: {}", error)))?;
        if !exclude.contains(&name.as_str()) {
            input_names.push(name);
        }
    }

    // construct output names by prepending out_ in front of each input name
    let output_names = input_names
        .iter()
        .map(|name| format!("out_{}", name))
        .collect();

    Ok((input_names, output_names))
}

fn initialize_states(
    session: &onnx::Session,
    names: &[String],
) -> Result<Vec<onnx::Value>, InferError> {
    // get number of inputs
    let input_count = session
        .input_count()
        .map_err(|e| InferError::Runtime(format!("Failed to get input count: {}", e)))?;

    // build name to index map
    let mut name_to_index = HashMap::new();
    for i in 0..input_count {
        let name = session
            .input_name(i)
            .map_err(|e| InferError::Runtime(format!("Failed to get input name: {}", e)))?;
        name_to_index.insert(name, i);
    }

    // build states
    let mut states = Vec::new();
    for name in names {
        // find index
        let index = *name_to_index
            .get(name)
            .ok_or_else(|| InferError::Runtime(format!("State input '{}' not found", name)))?;

        // get tensor shape and element type
        let shape = session
            .input_shape(index)
            .map_err(|e| InferError::Runtime(format!("Failed to get shape for {}: {}", name, e)))?;
        let elem_type = session.input_element_type(index).map_err(|e| {
            InferError::Runtime(format!("Failed to get element type for {}: {}", name, e))
        })?;

        let tensor = if shape == [0] {
            // create new 0x0 tensor
            onnx::Value::from_slice::<f32>(&session.onnx, &[0], &[]).map_err(|e| {
                InferError::Runtime(format!(
                    "Failed to create initial tensor for {}: {}",
                    name, e
                ))
            })?
        } else {
            // create new empty tensor
            match elem_type {
                onnx::ffi::ONNXTensorElementDataType::Float => {
                    onnx::Value::zeros::<f32>(&session.onnx, &shape)
                }
                onnx::ffi::ONNXTensorElementDataType::Int64 => {
                    onnx::Value::zeros::<i64>(&session.onnx, &shape)
                }
                onnx::ffi::ONNXTensorElementDataType::Bool => {
                    let resolved: Vec<usize> = shape
                        .iter()
                        .map(|&d| if d < 0 { 1 } else { d as usize })
                        .collect();
                    let total: usize = resolved.iter().product();
                    let true_data = vec![true; total];
                    onnx::Value::from_slice::<bool>(&session.onnx, &resolved, &true_data)
                }
                _ => {
                    return Err(InferError::Runtime(format!(
                        "Unsupported element type {:?} for state {}",
                        elem_type, name
                    )));
                }
            }
            .map_err(|e| {
                InferError::Runtime(format!(
                    "Failed to create initial tensor for {}: {}",
                    name, e
                ))
            })?
        };

        // add to states
        states.push(tensor);
    }

    Ok(states)
}

fn prepare(text: &str) -> (String, usize) {
    // remove leading and trailing whitespace and replace newlines with spaces
    let mut prepared = text.trim().replace('\n', " ");

    // count words
    let words = prepared.split_whitespace().count();

    // specify number of frames to add afterwards
    let frames_after = if words <= 4 { 3 } else { 1 } + 2;

    // for small utterances, prepend with spaces
    if words < 5 {
        prepared = format!("        {}", prepared);
    }

    // replace spaces with low line
    let prepared = format!("\u{2581}{}", prepared.replace(' ', "\u{2581}"));

    (prepared, frames_after)
}

fn tokenize(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<i64>, InferError> {
    // tokenize sentence
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| InferError::Runtime(format!("Tokenization failed: {}", error)))?;

    Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
}

fn condition(
    conditioner: &mut onnx::Session,
    flow_main: &mut onnx::Session,
    token_ids: &[i64],
    states: &mut Vec<onnx::Value>,
    flow_main_input_names: &[String],
    flow_main_output_names: &[String],
) -> Result<(), InferError> {
    // get numbet of tokens
    let seq_len = token_ids.len();

    // create tokens tensor
    let tokens = onnx::Value::from_slice::<i64>(&conditioner.onnx, &[1, seq_len], token_ids)
        .map_err(|e| InferError::Runtime(format!("Failed to create tokens tensor: {}", e)))?;

    // run conditioner, turn tokens into embeddings
    let outputs = conditioner
        .run(&[("token_ids", &tokens)], &["embeddings"])
        .map_err(|error| InferError::Runtime(format!("Conditioning failed: {}", error)))?;

    // extract embeddings
    let embeddings = outputs[0]
        .extract_tensor::<f32>()
        .map_err(|error| InferError::Runtime(format!("Failed to extract embeddings: {}", error)))?;

    // create sequence tensor
    let sequence = onnx::Value::from_slice::<f32>(&flow_main.onnx, &[1, 0, LATENT_DIM], &[])
        .map_err(|e| InferError::Runtime(format!("Failed to create sequence tensor: {}", e)))?;

    // create embeddings tensor
    let text_embeddings = onnx::Value::from_slice::<f32>(
        &flow_main.onnx,
        &[1, seq_len, CONDITIONING_DIM],
        embeddings,
    )
    .map_err(|e| InferError::Runtime(format!("Failed to create text embeddings tensor: {}", e)))?;

    // run flow main on sequence, embeddings and flow states
    let mut inputs = vec![
        ("sequence", &sequence),
        ("text_embeddings", &text_embeddings),
    ];
    for (i, state) in states.iter().enumerate() {
        inputs.push((&flow_main_input_names[i], state));
    }
    let output_names: Vec<&str> = flow_main_output_names.iter().map(|s| s.as_str()).collect();
    let outputs = flow_main
        .run(&inputs, &output_names)
        .map_err(|error| InferError::Runtime(format!("Flow main failed: {}", error)))?;

    // store flow state
    *states = outputs.into_iter().collect();

    Ok(())
}

fn step(
    flow_main: &mut onnx::Session,
    flow_step: &mut onnx::Session,
    states: &mut Vec<onnx::Value>,
    latent_state: &mut [f32],
    flow_main_input_names: &[String],
    flow_main_output_names: &[String],
    sequence_tensor: &mut onnx::Value,
    empty_text_embeddings: &onnx::Value,
    s_tensor: &mut onnx::Value,
    t_tensor: &mut onnx::Value,
    c_tensor: &mut onnx::Value,
    x_tensor: &mut onnx::Value,
    decoder_latent_tensor: &mut onnx::Value,
) -> Result<bool, InferError> {
    // write latent state into pre-allocated sequence tensor
    sequence_tensor
        .as_slice_mut::<f32>()
        .copy_from_slice(latent_state);

    // run flow main on sequence, empty embeddings and flow state
    let mut inputs = vec![
        ("sequence", &*sequence_tensor as &onnx::Value),
        ("text_embeddings", empty_text_embeddings),
    ];
    for (i, state) in states.iter().enumerate() {
        inputs.push((&flow_main_input_names[i], state));
    }
    let mut output_names = vec!["conditioning", "eos_logit"];
    output_names.extend(flow_main_output_names.iter().map(|s| s.as_str()));
    let mut outputs = flow_main
        .run(&inputs, &output_names)
        .map_err(|error| InferError::Runtime(format!("Flow main failed: {}", error)))?;

    // store flow state, leaving conditioning and eos_logit in outputs
    *states = outputs.split_off(2);

    // extract conditioning and EOS logit
    let conditioning = outputs[0].extract_tensor::<f32>().map_err(|error| {
        InferError::Runtime(format!("Failed to extract conditioning: {}", error))
    })?;
    let eos_logit = outputs[1]
        .extract_tensor::<f32>()
        .map_err(|error| InferError::Runtime(format!("Failed to extract eos_logit: {}", error)))?
        [0];

    // prepare to run LSD steps
    let num_steps = DEFAULT_LSD_STEPS;
    let temperature = DEFAULT_TEMPERATURE;
    let mut rng = rand::thread_rng();
    let std = (temperature as f64).sqrt();
    let normal = Normal::new(0.0, std).map_err(|error| {
        InferError::Runtime(format!("Failed to create normal distribution: {}", error))
    })?;
    let mut latent = [0f32; LATENT_DIM];
    for i in 0..LATENT_DIM {
        latent[i] = normal.sample(&mut rng) as f32;
    }
    for i in 0..num_steps {
        s_tensor.as_slice_mut::<f32>()[0] = i as f32 / num_steps as f32;
        t_tensor.as_slice_mut::<f32>()[0] = (i + 1) as f32 / num_steps as f32;
        c_tensor.as_slice_mut::<f32>().copy_from_slice(conditioning);
        x_tensor.as_slice_mut::<f32>().copy_from_slice(&latent);

        // run flow step on c, s, t and x
        let outputs = flow_step
            .run(
                &[
                    ("c", &c_tensor),
                    ("s", &s_tensor),
                    ("t", &t_tensor),
                    ("x", &x_tensor),
                ],
                &["flow_dir"],
            )
            .map_err(|error| InferError::Runtime(format!("Flow step failed: {}", error)))?;

        // extract flow direction
        let flow_dir = outputs[0].extract_tensor::<f32>().map_err(|error| {
            InferError::Runtime(format!("Failed to extract flow_dir: {}", error))
        })?;

        // apply flow direction to latent
        for j in 0..LATENT_DIM {
            latent[j] += flow_dir[j] / num_steps as f32;
        }
    }

    // store latent and prepare decoder input
    latent_state.copy_from_slice(&latent);
    decoder_latent_tensor
        .as_slice_mut::<f32>()
        .copy_from_slice(&latent);

    // check if EOS
    Ok(eos_logit > DEFAULT_EOS_THRESHOLD)
}

fn decode_audio(
    decoder: &mut onnx::Session,
    decoder_latent_tensor: &onnx::Value,
    decoder_states: &mut Vec<onnx::Value>,
    decoder_input_names: &[String],
    decoder_output_names: &[String],
) -> Result<Vec<i16>, InferError> {
    // run decoder on pre-allocated latent tensor and decoder state
    let mut inputs = vec![("latent", decoder_latent_tensor)];
    for (i, state) in decoder_states.iter().enumerate() {
        inputs.push((&decoder_input_names[i], state));
    }
    let mut output_names = vec!["audio_frame"];
    output_names.extend(decoder_output_names.iter().map(|s| s.as_str()));
    let mut outputs = decoder
        .run(&inputs, &output_names)
        .map_err(|error| InferError::Runtime(format!("Decoder failed: {}", error)))?;

    // store decoder state, leaving audio_frame in outputs
    *decoder_states = outputs.split_off(1);

    // convert audio directly from extracted slice to i16
    let audio = outputs[0]
        .extract_tensor::<f32>()
        .map_err(|error| InferError::Runtime(format!("Failed to extract audio: {}", error)))?;

    Ok(audio
        .iter()
        .map(|&f| (f * 32768.0).clamp(-32768.0, 32767.0) as i16)
        .collect())
}

pub struct PocketHandle<T: Clone + Send + 'static> {
    input_tx: std_mpsc::Sender<Stamped<TtsInput<T>>>,
    epoch: Epoch,
}

pub struct PocketListener<T: Clone + Send + 'static> {
    output_rx: tokio_mpsc::Receiver<Stamped<TtsOutput<T>>>,
}

pub fn create<T: Clone + Send + 'static>(
    onnx: &Arc<onnx::Onnx>,
    executor: &onnx::Executor,
    voice_path: &Path,
    epoch: Epoch,
) -> Result<(PocketHandle<T>, PocketListener<T>), InferError> {
    // load the things
    let mut conditioner = onnx
        .create_session(
            executor,
            &onnx::OptimizationLevel::EnableAll,
            4,
            CONDITIONER_PATH,
        )
        .map_err(|e| InferError::Runtime(format!("Failed to create conditioner session: {}", e)))?;
    let mut flow_main = onnx
        .create_session(
            executor,
            &onnx::OptimizationLevel::EnableAll,
            4,
            FLOW_MAIN_PATH,
        )
        .map_err(|e| InferError::Runtime(format!("Failed to create flow main session: {}", e)))?;
    let mut flow_step = onnx
        .create_session(
            executor,
            &onnx::OptimizationLevel::EnableAll,
            4,
            FLOW_STEP_PATH,
        )
        .map_err(|e| InferError::Runtime(format!("Failed to create flow step session: {}", e)))?;
    let mut decoder = onnx
        .create_session(
            executor,
            &onnx::OptimizationLevel::EnableAll,
            4,
            DECODER_PATH,
        )
        .map_err(|e| InferError::Runtime(format!("Failed to create decoder session: {}", e)))?;
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| InferError::Runtime(format!("Failed to create tokenizer: {}", e)))?;
    let voice = load_voice(voice_path)?;

    // obtain input and output names
    let (flow_main_input_names, flow_main_output_names) =
        get_names(&flow_main, &["sequence", "text_embeddings"])?;
    let (decoder_input_names, decoder_output_names) = get_names(&decoder, &["latent"])?;

    // initialize caches
    let mut reset_flow_states = initialize_states(&flow_main, &flow_main_input_names)?;
    let mut decoder_states = initialize_states(&decoder, &decoder_input_names)?;

    // condition voice
    let latent_frames = voice.len() / 1024;
    let sequence = onnx::Value::from_slice::<f32>(&flow_main.onnx, &[1, 0, LATENT_DIM], &[])
        .map_err(|e| InferError::Runtime(format!("Failed to create sequence tensor: {}", e)))?;
    let text_embeddings = onnx::Value::from_slice::<f32>(
        &flow_main.onnx,
        &[1, latent_frames, CONDITIONING_DIM],
        &voice,
    )
    .map_err(|e| InferError::Runtime(format!("Failed to create text embeddings tensor: {}", e)))?;
    let mut inputs = vec![
        ("sequence", &sequence),
        ("text_embeddings", &text_embeddings),
    ];
    for (i, state) in reset_flow_states.iter().enumerate() {
        inputs.push((&flow_main_input_names[i], state));
    }
    let output_names: Vec<&str> = flow_main_output_names.iter().map(|s| s.as_str()).collect();
    let outputs = flow_main
        .run(&inputs, &output_names)
        .map_err(|error| InferError::Runtime(format!("Flow main failed: {}", error)))?;
    reset_flow_states = outputs;

    // prepare re-usable tensors
    let mut sequence_tensor =
        onnx::Value::zeros::<f32>(&flow_main.onnx, &[1, 1, LATENT_DIM as i64])
            .map_err(|e| InferError::Runtime(format!("Failed to create sequence tensor: {}", e)))?;
    let empty_text_embeddings =
        onnx::Value::from_slice::<f32>(&flow_main.onnx, &[1, 0, CONDITIONING_DIM], &[]).map_err(
            |e| {
                InferError::Runtime(format!(
                    "Failed to create empty text embeddings tensor: {}",
                    e
                ))
            },
        )?;
    let mut s_tensor = onnx::Value::zeros::<f32>(&flow_step.onnx, &[1, 1])
        .map_err(|e| InferError::Runtime(format!("Failed to create s tensor: {}", e)))?;
    let mut t_tensor = onnx::Value::zeros::<f32>(&flow_step.onnx, &[1, 1])
        .map_err(|e| InferError::Runtime(format!("Failed to create t tensor: {}", e)))?;
    let mut c_tensor = onnx::Value::zeros::<f32>(&flow_step.onnx, &[1, CONDITIONING_DIM as i64])
        .map_err(|e| InferError::Runtime(format!("Failed to create c tensor: {}", e)))?;
    let mut x_tensor = onnx::Value::zeros::<f32>(&flow_step.onnx, &[1, LATENT_DIM as i64])
        .map_err(|e| InferError::Runtime(format!("Failed to create x tensor: {}", e)))?;
    let mut decoder_latent_tensor =
        onnx::Value::zeros::<f32>(&decoder.onnx, &[1, 1, LATENT_DIM as i64]).map_err(|e| {
            InferError::Runtime(format!("Failed to create decoder latent tensor: {}", e))
        })?;

    // create channels
    let (input_tx, input_rx) = std_mpsc::channel::<Stamped<TtsInput<T>>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<Stamped<TtsOutput<T>>>(32);

    // spawn it
    std::thread::spawn({
        let epoch = epoch.clone();
        move || {
            loop {
                match input_rx.recv() {
                    Ok(stamped) => {
                        // skip stale input
                        if !epoch.is_current(stamped.epoch) {
                            continue;
                        }

                        let my_epoch = stamped.epoch;
                        let input = stamped.inner;

                        let t_total = std::time::Instant::now();

                        let mut current_id = 0u64;
                        let mut cache_latent = vec![f32::NAN; LATENT_DIM];
                        let t0 = std::time::Instant::now();
                        let mut cache_flow_state = match reset_flow_states
                            .iter()
                            .map(|v| v.deepclone())
                            .collect::<Result<Vec<_>, _>>()
                        {
                            Ok(states) => states,
                            Err(e) => {
                                log_error!("Failed to clone flow state: {}", e);
                                continue;
                            }
                        };

                        let (prepared, frames_after) = prepare(&input.text);
                        let token_ids = match tokenize(&tokenizer, &prepared) {
                            Ok(token_ids) => token_ids,
                            Err(e) => {
                                log_error!("Failed to tokenize text: {}", e);
                                continue;
                            }
                        };

                        if let Err(error) = condition(
                            &mut conditioner,
                            &mut flow_main,
                            &token_ids,
                            &mut cache_flow_state,
                            &flow_main_input_names,
                            &flow_main_output_names,
                        ) {
                            log_error!("Failed to condition text: {}", error);
                            continue;
                        }
                        let conditioning_ms = t0.elapsed().as_secs_f64() * 1000.0;

                        let mut eos_countdown: Option<usize> = None;
                        let mut step_count = 0usize;
                        let t_gen = std::time::Instant::now();
                        for _ in 0..MAX_TOKENS {
                            let is_eos = match step(
                                &mut flow_main,
                                &mut flow_step,
                                &mut cache_flow_state,
                                &mut cache_latent,
                                &flow_main_input_names,
                                &flow_main_output_names,
                                &mut sequence_tensor,
                                &empty_text_embeddings,
                                &mut s_tensor,
                                &mut t_tensor,
                                &mut c_tensor,
                                &mut x_tensor,
                                &mut decoder_latent_tensor,
                            ) {
                                Ok(is_eos) => is_eos,
                                Err(e) => {
                                    log_error!("Failed to step: {}", e);
                                    break;
                                }
                            };

                            let sample = match decode_audio(
                                &mut decoder,
                                &decoder_latent_tensor,
                                &mut decoder_states,
                                &decoder_input_names,
                                &decoder_output_names,
                            ) {
                                Ok(sample) => sample,
                                Err(e) => {
                                    log_error!("Failed to decode audio: {}", e);
                                    break;
                                }
                            };

                            // check if epoch has advanced (cancellation)
                            if !epoch.is_current(my_epoch) {
                                break;
                            }

                            if let Err(e) = output_tx.blocking_send(Stamped {
                                epoch: my_epoch,
                                inner: TtsOutput {
                                    payload: TtsPayload {
                                        payload: input.payload.clone(),
                                        id: current_id,
                                    },
                                    data: sample,
                                },
                            }) {
                                log_error!("Failed to send audio: {}", e);
                                break;
                            }
                            current_id += 1;
                            step_count += 1;
                            if let Some(ref mut remaining) = eos_countdown {
                                if *remaining == 0 {
                                    break;
                                }
                                *remaining -= 1;
                            } else if is_eos {
                                eos_countdown = Some(frames_after);
                            }
                        }

                        let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
                        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
                        let per_step = if step_count > 0 {
                            gen_ms / step_count as f64
                        } else {
                            0.0
                        };
                        println!(
                            "TTS: cond {:.1}ms + gen {:.1}ms ({} steps, {:.1}ms/step) = {:.1}ms total",
                            conditioning_ms, gen_ms, step_count, per_step, total_ms
                        );
                    }
                    Err(_) => {
                        log_error!("Input channel disconnected");
                        break;
                    }
                }
            }
        }
    });

    Ok((
        PocketHandle { input_tx, epoch },
        PocketListener { output_rx },
    ))
}

impl<T: Clone + Send + 'static> PocketHandle<T> {
    // send text to TTS (stamped with current epoch)
    pub fn send(&self, input: TtsInput<T>) -> Result<(), std_mpsc::SendError<Stamped<TtsInput<T>>>> {
        self.input_tx.send(Stamped {
            epoch: self.epoch.current(),
            inner: input,
        })
    }
}

impl<T: Clone + Send + 'static> PocketListener<T> {
    // receive audio from TTS
    pub async fn recv(&mut self) -> Option<Stamped<TtsOutput<T>>> {
        self.output_rx.recv().await
    }

    // try-receive audio from TTS
    pub fn try_recv(&mut self) -> Option<Stamped<TtsOutput<T>>> {
        match self.output_rx.try_recv() {
            Ok(output) => Some(output),
            _ => None,
        }
    }
}
