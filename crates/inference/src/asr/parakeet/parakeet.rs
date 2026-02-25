use crate::error::Result;

const PARAKEET_ENCODER_PATH: &str = "data/parakeet/encoder.onnx";
const PARAKEET_DECODER_PATH: &str = "data/parakeet/decoder_joint.onnx";
const PARAKEET_TOKENIZER_PATH: &str = "data/parakeet/tokenizer.model";

pub struct Parakeet {
    audio_tx: mpsc::Sender<Vec<i16>>,
    text_rx: mpsc::Receiver<Transcription>,
    buffer: Vec<i16>,
}

impl Parakeet {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: &onnx::Executor) -> Result<Self> {
        let encoder = onnx.create_session(executor, PARAKEET_ENCODER_PATH)?;
        let decoder_joint = onnx.create_session(executor, PARAKEET_DECODER_PATH)?;

        Ok(Self {})
    }
}
