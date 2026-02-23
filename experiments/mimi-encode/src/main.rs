use {
    base::*,
    inference::Inference,
    std::{
        io::Write,
        path::PathBuf,
    },
};

const SAMPLE_RATE: u32 = 24000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    base::init_stdout_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <voice.wav> <output.bin>", args[0]);
        eprintln!();
        eprintln!("Encodes a voice WAV file through the mimi encoder and saves the");
        eprintln!("resulting latent tensor to a binary file.");
        eprintln!();
        eprintln!("Output format (little-endian):");
        eprintln!("  4 bytes  - number of dimensions (u32)");
        eprintln!("  N×8 bytes - each dimension (u64)");
        eprintln!("  remainder - raw f32 data");
        std::process::exit(1);
    }

    let voice_path = &args[1];
    let output_path = &args[2];

    // Load mimi encoder model
    let mimi_encoder_path = PathBuf::from("data/pocket/mimi_encoder.onnx");
    if !mimi_encoder_path.exists() {
        eprintln!("Missing: {}", mimi_encoder_path.display());
        std::process::exit(1);
    }

    // Read WAV file
    log_info!("Reading WAV: {}", voice_path);
    let mut reader = hound::WavReader::open(voice_path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        eprintln!("WAV must be mono, got {} channels", spec.channels);
        std::process::exit(1);
    }
    if spec.sample_rate != SAMPLE_RATE {
        eprintln!("WAV must be {}Hz, got {}Hz", SAMPLE_RATE, spec.sample_rate);
        std::process::exit(1);
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|val| val as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => {
            reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    log_info!(
        "WAV: {} samples, {}Hz, {} bit",
        samples.len(),
        spec.sample_rate,
        spec.bits_per_sample,
    );

    // Initialize inference and create encoder session
    log_info!("Loading mimi encoder...");
    #[cfg(feature = "cuda")]
    let inference = Inference::cuda(0)?;
    #[cfg(not(feature = "cuda"))]
    let inference = Inference::cpu()?;

    let mut encoder = inference.onnx_session(&mimi_encoder_path)?;

    // Encode: input shape [1, 1, audio_len], output "latents"
    log_info!("Encoding {} samples...", samples.len());
    let audio_tensor = onnx::Value::from_slice::<f32>(&[1, 1, samples.len()], &samples)?;
    let outputs = encoder.run(&[("audio", &audio_tensor)], &["latents"])?;

    let shape = outputs[0].tensor_shape()?;
    let latents = outputs[0].extract_tensor::<f32>()?;

    let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    log_info!("Encoded latents: shape [{}], {} floats", shape_str.join(", "), latents.len());

    // Write binary output: ndims (u32) + dims (u64 each) + raw f32 data
    let mut file = std::fs::File::create(output_path)?;
    let ndims = shape.len() as u32;
    file.write_all(&ndims.to_le_bytes())?;
    for &dim in &shape {
        file.write_all(&(dim as u64).to_le_bytes())?;
    }
    for &val in latents {
        file.write_all(&val.to_le_bytes())?;
    }

    let file_size = std::fs::metadata(output_path)?.len();
    log_info!("Wrote {} bytes to {}", file_size, output_path);

    Ok(())
}
