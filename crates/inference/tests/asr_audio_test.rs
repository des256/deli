use inference::asr::{config::N_SAMPLES, Config, load_mel_filters, pcm_to_mel};

#[test]
fn test_load_mel_filters() {
    let filters = load_mel_filters();

    // 80 mel bins × (N_FFT/2 + 1) filter coefficients
    // For N_FFT=400: (400/2 + 1) = 201
    // Total: 80 * 201 = 16080 values
    assert_eq!(filters.len(), 16080);

    // Verify filters contain non-zero values
    let non_zero_count = filters.iter().filter(|&&x| x != 0.0).count();
    assert!(
        non_zero_count > 0,
        "Mel filters should contain non-zero values"
    );
}

#[test]
fn test_pcm_to_mel_shape() {
    let config = Config::tiny_en();

    // Generate 30 seconds of silence (480000 samples at 16kHz)
    let samples = vec![0.0_f32; N_SAMPLES];

    let mel = pcm_to_mel(&config, &samples);

    // Expected output: 80 mel bins × 3000 frames = 240000 values
    assert_eq!(mel.len(), 80 * 3000);
}

#[test]
fn test_fft_with_sine_wave() {
    let config = Config::tiny_en();

    // Generate a 1-second 440Hz sine wave (A4 note)
    let sample_rate = 16000.0;
    let duration = 1.0; // second
    let freq = 440.0; // Hz
    let n_samples = (sample_rate * duration) as usize;

    let mut samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f32 / sample_rate;
        samples.push((2.0 * std::f32::consts::PI * freq * t).sin());
    }

    let mel = pcm_to_mel(&config, &samples);

    // Verify mel spectrogram has expected shape for 1 second of audio
    // N_FRAMES = N_SAMPLES / HOP_LENGTH = 16000 / 160 = 100 frames
    let n_mel_bins = 80;
    let n_frames = 100;
    assert_eq!(mel.len(), n_mel_bins * n_frames);

    // Compute average energy per mel bin across all frames
    // Mel data is in mel-bins-major order: mel[bin * n_frames + frame]
    let mut bin_energies = vec![0.0_f32; n_mel_bins];
    for bin in 0..n_mel_bins {
        for frame in 0..n_frames {
            bin_energies[bin] += mel[bin * n_frames + frame];
        }
    }
    for e in &mut bin_energies {
        *e /= n_frames as f32;
    }

    // Find the mel bin with peak energy
    let peak_bin = bin_energies
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    // 440Hz maps to ~mel bin 15-35 range (mel scale compresses higher frequencies)
    // The exact bin depends on the mel filterbank, but should be in the lower-mid range
    assert!(
        (10..40).contains(&peak_bin),
        "Peak energy bin for 440Hz sine should be in range 10-40, got bin {peak_bin}"
    );

    // Verify non-zero variance (energy present)
    let mean: f32 = mel.iter().sum::<f32>() / mel.len() as f32;
    let variance: f32 = mel.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / mel.len() as f32;
    assert!(
        variance > 0.0,
        "Mel spectrogram should contain energy for sine wave (variance = {variance})"
    );
}

#[test]
fn test_pcm_to_mel_with_real_audio_length() {
    let config = Config::tiny_en();

    // 30 seconds at 16kHz = 480000 samples
    let samples = vec![0.1_f32; N_SAMPLES];

    let mel = pcm_to_mel(&config, &samples);

    // 80 mel bins × 3000 frames
    assert_eq!(mel.len(), 240000);
}
