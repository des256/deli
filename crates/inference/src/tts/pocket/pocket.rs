use {
    crate::*,
    base::*,
    rand_distr::{Distribution, Normal},
    std::{collections::HashMap, io::Read, path::Path, sync::Arc},
    tokio::sync::mpsc,
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
    let input_count = session
        .input_count()
        .map_err(|error| InferError::Runtime(format!("Failed to get input count: {}", error)))?;
    for i in 0..input_count {
        let name = session
            .input_name(i)
            .map_err(|error| InferError::Runtime(format!("Failed to get input name: {}", error)))?;
        if !exclude.contains(&name.as_str()) {
            input_names.push(name);
        }
    }
    let output_names = input_names
        .iter()
        .map(|name| format!("out_{}", name))
        .collect();
    Ok((input_names, output_names))
}

fn initialize_cache(
    session: &onnx::Session,
    names: &[String],
) -> Result<Vec<onnx::Value>, InferError> {
    let input_count = session
        .input_count()
        .map_err(|e| InferError::Runtime(format!("Failed to get input count: {}", e)))?;
    let mut name_to_index = HashMap::new();
    for i in 0..input_count {
        let name = session
            .input_name(i)
            .map_err(|e| InferError::Runtime(format!("Failed to get input name: {}", e)))?;
        name_to_index.insert(name, i);
    }
    let mut cache = Vec::new();
    for name in names {
        let index = *name_to_index
            .get(name)
            .ok_or_else(|| InferError::Runtime(format!("State input '{}' not found", name)))?;
        let shape = session
            .input_shape(index)
            .map_err(|e| InferError::Runtime(format!("Failed to get shape for {}: {}", name, e)))?;
        let elem_type = session.input_element_type(index).map_err(|e| {
            InferError::Runtime(format!("Failed to get element type for {}: {}", name, e))
        })?;
        let tensor = if shape == [0] {
            onnx::Value::from_slice::<f32>(&session.onnx, &[0], &[]).map_err(|e| {
                InferError::Runtime(format!(
                    "Failed to create initial tensor for {}: {}",
                    name, e
                ))
            })?
        } else {
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
        cache.push(tensor);
    }
    Ok(cache)
}

fn prepare(text: &str) -> (String, usize) {
    let mut prepared = text.trim().replace('\n', " ");
    let words = prepared.split_whitespace().count();
    let frames_after = if words <= 4 { 3 } else { 1 } + 2;
    if words < 5 {
        prepared = format!("        {}", prepared);
    }
    let prepared = format!("\u{2581}{}", prepared.replace(' ', "\u{2581}"));
    (prepared, frames_after)
}

fn tokenize(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<i64>, InferError> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| InferError::Runtime(format!("Tokenization failed: {}", error)))?;
    Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
}

fn condition(
    conditioner: &mut onnx::Session,
    flow_main: &mut onnx::Session,
    token_ids: &[i64],
    cache_flow_state: &mut Vec<onnx::Value>,
    flow_main_input_names: &[String],
    flow_main_output_names: &[String],
) -> Result<(), InferError> {
    let seq_len = token_ids.len();
    let tokens = onnx::Value::from_slice::<i64>(&conditioner.onnx, &[1, seq_len], token_ids)
        .map_err(|e| InferError::Runtime(format!("Failed to create tokens tensor: {}", e)))?;
    let outputs = conditioner
        .run(&[("token_ids", &tokens)], &["embeddings"])
        .map_err(|error| InferError::Runtime(format!("Conditioning failed: {}", error)))?;
    let embeddings = outputs[0]
        .extract_tensor::<f32>()
        .map_err(|error| InferError::Runtime(format!("Failed to extract embeddings: {}", error)))?;
    let sequence = onnx::Value::from_slice::<f32>(&flow_main.onnx, &[1, 0, LATENT_DIM], &[])
        .map_err(|e| InferError::Runtime(format!("Failed to create sequence tensor: {}", e)))?;
    let text_embeddings = onnx::Value::from_slice::<f32>(
        &flow_main.onnx,
        &[1, seq_len, CONDITIONING_DIM],
        embeddings,
    )
    .map_err(|e| InferError::Runtime(format!("Failed to create text embeddings tensor: {}", e)))?;
    let mut inputs = vec![
        ("sequence", &sequence),
        ("text_embeddings", &text_embeddings),
    ];
    for (i, state) in cache_flow_state.iter().enumerate() {
        inputs.push((&flow_main_input_names[i], state));
    }
    let output_names: Vec<&str> = flow_main_output_names.iter().map(|s| s.as_str()).collect();
    let outputs = flow_main
        .run(&inputs, &output_names)
        .map_err(|error| InferError::Runtime(format!("Flow main failed: {}", error)))?;
    *cache_flow_state = outputs.into_iter().collect();
    Ok(())
}

fn step(
    flow_main: &mut onnx::Session,
    flow_step: &mut onnx::Session,
    cache_flow_state: &mut Vec<onnx::Value>,
    cache_latent: &mut Vec<f32>,
    flow_main_input_names: &[String],
    flow_main_output_names: &[String],
) -> Result<(Vec<f32>, bool), InferError> {
    let sequence_data: Vec<f32> = cache_latent.clone();
    let sequence =
        onnx::Value::from_slice::<f32>(&flow_main.onnx, &[1, 1, LATENT_DIM], &sequence_data)
            .map_err(|e| InferError::Runtime(format!("Failed to create sequence tensor: {}", e)))?;
    let text_embeddings =
        onnx::Value::from_slice::<f32>(&flow_main.onnx, &[1, 0, CONDITIONING_DIM], &[]).map_err(
            |e| InferError::Runtime(format!("Failed to create text embeddings tensor: {}", e)),
        )?;
    let mut inputs = vec![
        ("sequence", &sequence),
        ("text_embeddings", &text_embeddings),
    ];
    for (i, state) in cache_flow_state.iter().enumerate() {
        inputs.push((&flow_main_input_names[i], state));
    }
    let mut output_names = vec!["conditioning", "eos_logit"];
    output_names.extend(flow_main_output_names.iter().map(|s| s.as_str()));
    let outputs = flow_main
        .run(&inputs, &output_names)
        .map_err(|error| InferError::Runtime(format!("Flow main failed: {}", error)))?;
    let conditioning = outputs[0]
        .extract_tensor::<f32>()
        .map_err(|error| InferError::Runtime(format!("Failed to extract conditioning: {}", error)))?
        .to_vec();
    let eos_logit_tensor = outputs[1]
        .extract_tensor::<f32>()
        .map_err(|error| InferError::Runtime(format!("Failed to extract eos_logit: {}", error)))?
        .to_vec();
    let eos_logit = eos_logit_tensor[0];
    *cache_flow_state = outputs.into_iter().skip(2).collect();
    let num_steps = DEFAULT_LSD_STEPS;
    let temperature = DEFAULT_TEMPERATURE;
    let mut rng = rand::thread_rng();
    let std = (temperature as f64).sqrt();
    let normal = Normal::new(0.0, std).map_err(|error| {
        InferError::Runtime(format!("Failed to create normal distribution: {}", error))
    })?;
    let mut latent: Vec<f32> = (0..LATENT_DIM)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();
    for i in 0..num_steps {
        let s = i as f32 / num_steps as f32;
        let t = (i + 1) as f32 / num_steps as f32;
        let c_tensor =
            onnx::Value::from_slice::<f32>(&flow_step.onnx, &[1, CONDITIONING_DIM], &conditioning)
                .map_err(|e| InferError::Runtime(format!("Failed to create c tensor: {}", e)))?;
        let s_tensor = onnx::Value::from_slice::<f32>(&flow_step.onnx, &[1, 1], &[s])
            .map_err(|e| InferError::Runtime(format!("Failed to create s tensor: {}", e)))?;
        let t_tensor = onnx::Value::from_slice::<f32>(&flow_step.onnx, &[1, 1], &[t])
            .map_err(|e| InferError::Runtime(format!("Failed to create t tensor: {}", e)))?;
        let x_tensor =
            onnx::Value::from_slice::<f32>(&flow_step.onnx, &[1, LATENT_DIM], &latent)
                .map_err(|e| InferError::Runtime(format!("Failed to create x tensor: {}", e)))?;
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
        let flow_dir = outputs[0].extract_tensor::<f32>().map_err(|error| {
            InferError::Runtime(format!("Failed to extract flow_dir: {}", error))
        })?;
        for j in 0..LATENT_DIM {
            latent[j] += flow_dir[j] / num_steps as f32;
        }
    }
    *cache_latent = latent.clone();
    let is_eos = eos_logit > DEFAULT_EOS_THRESHOLD;
    Ok((latent, is_eos))
}

fn decode_audio(
    decoder: &mut onnx::Session,
    latent: &[f32],
    cache_decoder_state: &mut Vec<onnx::Value>,
    decoder_input_names: &[String],
    decoder_output_names: &[String],
) -> Result<Vec<i16>, InferError> {
    let latent_tensor = onnx::Value::from_slice::<f32>(&decoder.onnx, &[1, 1, LATENT_DIM], latent)
        .map_err(|e| InferError::Runtime(format!("Failed to create latent tensor: {}", e)))?;
    let mut inputs = vec![("latent", &latent_tensor)];
    for (i, state) in cache_decoder_state.iter().enumerate() {
        inputs.push((&decoder_input_names[i], state));
    }
    let mut output_names = vec!["audio_frame"];
    output_names.extend(decoder_output_names.iter().map(|s| s.as_str()));
    let outputs = decoder
        .run(&inputs, &output_names)
        .map_err(|error| InferError::Runtime(format!("Decoder failed: {}", error)))?;
    let audio = outputs[0]
        .extract_tensor::<f32>()
        .map_err(|error| InferError::Runtime(format!("Failed to extract audio: {}", error)))?
        .to_vec();
    *cache_decoder_state = outputs.into_iter().skip(1).collect();
    Ok(audio
        .iter()
        .map(|&f| (f * 32768.0).clamp(-32768.0, 32767.0) as i16)
        .collect())
}

pub struct Pocket {
    text_tx: mpsc::Sender<String>,
    audio_rx: mpsc::Receiver<Vec<i16>>,
}

impl Pocket {
    pub fn new(
        onnx: &Arc<onnx::Onnx>,
        executor: &onnx::Executor,
        voice_path: &Path,
    ) -> Result<Self, InferError> {
        // load the things
        let mut conditioner = onnx
            .create_session(executor, CONDITIONER_PATH)
            .map_err(|e| {
                InferError::Runtime(format!("Failed to create conditioner session: {}", e))
            })?;
        let mut flow_main = onnx.create_session(executor, FLOW_MAIN_PATH).map_err(|e| {
            InferError::Runtime(format!("Failed to create flow main session: {}", e))
        })?;
        let mut flow_step = onnx.create_session(executor, FLOW_STEP_PATH).map_err(|e| {
            InferError::Runtime(format!("Failed to create flow step session: {}", e))
        })?;
        let mut decoder = onnx
            .create_session(executor, DECODER_PATH)
            .map_err(|e| InferError::Runtime(format!("Failed to create decoder session: {}", e)))?;
        let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| InferError::Runtime(format!("Failed to create tokenizer: {}", e)))?;
        let voice = load_voice(voice_path)?;

        // obtain input and output names
        let (flow_main_input_names, flow_main_output_names) =
            get_names(&flow_main, &["sequence", "text_embeddings"])?;
        let (decoder_input_names, decoder_output_names) = get_names(&decoder, &["latent"])?;

        // initialize caches
        let mut reset_flow_state = initialize_cache(&flow_main, &flow_main_input_names)?;
        let mut cache_decoder_state = initialize_cache(&decoder, &decoder_input_names)?;

        // condition voice
        let latent_frames = voice.len() / 1024;
        let sequence = onnx::Value::from_slice::<f32>(&flow_main.onnx, &[1, 0, LATENT_DIM], &[])
            .map_err(|e| InferError::Runtime(format!("Failed to create sequence tensor: {}", e)))?;
        let text_embeddings = onnx::Value::from_slice::<f32>(
            &flow_main.onnx,
            &[1, latent_frames, CONDITIONING_DIM],
            &voice,
        )
        .map_err(|e| {
            InferError::Runtime(format!("Failed to create text embeddings tensor: {}", e))
        })?;
        let mut inputs = vec![
            ("sequence", &sequence),
            ("text_embeddings", &text_embeddings),
        ];
        for (i, state) in reset_flow_state.iter().enumerate() {
            inputs.push((&flow_main_input_names[i], state));
        }
        let output_names: Vec<&str> = flow_main_output_names.iter().map(|s| s.as_str()).collect();
        let outputs = flow_main
            .run(&inputs, &output_names)
            .map_err(|error| InferError::Runtime(format!("Flow main failed: {}", error)))?;
        reset_flow_state = outputs;

        // create channels
        let (text_tx, mut text_rx) = mpsc::channel::<String>(32);
        let (audio_tx, audio_rx) = mpsc::channel::<Vec<i16>>(32);

        // spawn it
        tokio::task::spawn_blocking({
            move || {
                while let Some(text) = text_rx.blocking_recv() {
                    let mut cache_latent = vec![f32::NAN; LATENT_DIM];
                    let mut cache_flow_state = match reset_flow_state
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
                    let (prepared, frames_after) = prepare(&text);
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
                    let mut eos_countdown: Option<usize> = None;
                    for _ in 0..MAX_TOKENS {
                        let (latent, is_eos) = match step(
                            &mut flow_main,
                            &mut flow_step,
                            &mut cache_flow_state,
                            &mut cache_latent,
                            &flow_main_input_names,
                            &flow_main_output_names,
                        ) {
                            Ok((latent, is_eos)) => (latent, is_eos),
                            Err(e) => {
                                log_error!("Failed to step: {}", e);
                                break;
                            }
                        };
                        let sample = match decode_audio(
                            &mut decoder,
                            &latent,
                            &mut cache_decoder_state,
                            &decoder_input_names,
                            &decoder_output_names,
                        ) {
                            Ok(sample) => sample,
                            Err(e) => {
                                log_error!("Failed to decode audio: {}", e);
                                break;
                            }
                        };
                        if let Err(e) = audio_tx.blocking_send(sample) {
                            log_error!("Failed to send audio: {}", e);
                            break;
                        }
                        if let Some(ref mut remaining) = eos_countdown {
                            if *remaining == 0 {
                                break;
                            }
                            *remaining -= 1;
                        } else if is_eos {
                            eos_countdown = Some(frames_after);
                        }
                    }
                }
            }
        });

        Ok(Self { text_tx, audio_rx })
    }

    pub fn text_tx(&self) -> mpsc::Sender<String> {
        self.text_tx.clone()
    }

    pub async fn send(&self, text: String) -> Result<(), InferError> {
        self.text_tx
            .send(text)
            .await
            .map_err(|error| InferError::Runtime(format!("Failed to send text: {}", error)))
    }

    pub async fn recv(&mut self) -> Option<Vec<i16>> {
        self.audio_rx.recv().await
    }

    pub fn try_recv(&mut self) -> Option<Vec<i16>> {
        match self.audio_rx.try_recv() {
            Ok(text) => Some(text),
            _ => None,
        }
    }
}
