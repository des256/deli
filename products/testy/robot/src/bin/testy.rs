use {
    audio::{AudioIn, AudioOut},
    base::{Vec2, log},
    com::WsServer,
    image::{PixelFormat, argb_to_jpeg, rgb_to_jpeg, srggb10p_to_jpeg, yu12_to_jpeg, yuyv_to_jpeg},
    inference::Inference,
    std::sync::Arc,
    testy::*,
    video::{VideoIn, VideoInConfig, realsense::RealsenseConfig},
};

const DEFAULT_ADDR: &str = "0.0.0.0:5090";

const WHISPER_MODEL_PATH: &str = "data/whisper/tiny.en";
const WHISPER_TOKENIZER_PATH: &str = "data/whisper/tiny.en/tokenizer.json";
const WHISPER_CONFIG_PATH: &str = "data/whisper/tiny.en/config.json";

const KOKORO_MODEL_PATH: &str = "data/kokoro/kokoro-v1.0.onnx";
const KOKORO_VOICE_PATH: &str = "data/kokoro/bf_emma.npy";
const KOKORO_ESPEAK_DATA_PATH: &str = "/usr/lib/x86_64-linux-gnu/espeak-ng-data";

const QWEN3_MODEL_PATH: &str = "data/qwen3/qwen3-8b-q4_k_m.gguf";
const QWEN3_TOKENIZER_PATH: &str = "data/qwen3/tokenizer.json";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    log::info!("initializing inference");
    let inference = Inference::cuda(0)?;

    log::info!("opening audio input");
    let audioin = AudioIn::open(None).await;

    log::info!("opening audio output");
    let audioout = AudioOut::open(None).await;

    tokio::task::spawn(async move {
        log::info!("spawning audio pipeline");
        // TODO: first VAD the incoming audio stream
        // TOOD: push the VAD results to ASR
        // TODO: extract partial and final transcription results
    });

    log::info!("loading ASR model");
    let asr = inference.use_whisper(
        WHISPER_MODEL_PATH,
        WHISPER_TOKENIZER_PATH,
        WHISPER_CONFIG_PATH,
    )?;

    log::info!("loading TTS model");
    let tts = inference.use_kokoro(
        KOKORO_MODEL_PATH,
        KOKORO_VOICE_PATH,
        Some(KOKORO_ESPEAK_DATA_PATH),
    )?;

    log::info!("loading LLM");
    let llm = inference.use_qwen3(QWEN3_MODEL_PATH, QWEN3_TOKENIZER_PATH)?;

    log::info!("opening video input");
    let mut videoin = VideoIn::open(Some(VideoInConfig::Realsense(RealsenseConfig {
        color: Some(Vec2::new(640, 480)),
        frame_rate: Some(30.0),
        ..Default::default()
    })))
    .await?;

    log::info!("creating websocket server at {}", DEFAULT_ADDR);
    let server: Arc<WsServer<ToMonitor>> = Arc::new(WsServer::bind(DEFAULT_ADDR).await?);

    tokio::task::spawn({
        let server = Arc::clone(&server);
        async move {
            log::info!("spawning video pipeline");
            // TODO: detect poses from the video stream
            // TODO: magically map poses to humans without any problems

            // TODO: for now just pass the color frame to the websocket clients
            loop {
                let frame = match videoin.capture().await {
                    Ok(frame) => frame,
                    Err(error) => {
                        log::error!("video capture failed: {}", error);
                        continue;
                    }
                };
                let jpeg =
                    color_to_jpeg(frame.color.size, &frame.color.data, frame.color.format, 80);
                if let Err(error) = server.send(&ToMonitor::Jpeg(jpeg)).await {
                    log::error!("websocket send failed: {}", error);
                    continue;
                }
            }
        }
    });

    Ok(())
}

fn color_to_jpeg(size: Vec2<usize>, data: &[u8], format: PixelFormat, quality: u8) -> Vec<u8> {
    match format {
        PixelFormat::Jpeg => data.to_vec(),
        PixelFormat::Yuyv => yuyv_to_jpeg(size, data, quality),
        PixelFormat::Srggb10p => srggb10p_to_jpeg(size, data, quality),
        PixelFormat::Yu12 => yu12_to_jpeg(size, data, quality),
        PixelFormat::Rgb8 => rgb_to_jpeg(size, data, quality),
        PixelFormat::Argb8 => argb_to_jpeg(size, data, quality),
    }
}
