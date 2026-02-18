// Audio preprocessing for Whisper: FFT and mel spectrogram computation.
//
// Ported from candle-transformers (Apache-2.0/MIT)
// https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/whisper/audio.rs

use num_traits::Float;

use super::config::{Config, HOP_LENGTH, N_FFT};

const EPSILON: f32 = 1e-10;

/// Load embedded mel filter data from melfilters.bytes.
///
/// Returns a Vec<f32> with filter coefficients for 80 mel bins.
pub fn load_mel_filters() -> Vec<f32> {
    const MEL_BYTES: &[u8] = include_bytes!("melfilters.bytes");

    let mut filters = Vec::with_capacity(MEL_BYTES.len() / 4);
    for chunk in MEL_BYTES.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap();
        filters.push(f32::from_le_bytes(bytes));
    }
    filters
}

/// Discrete Fourier Transform (DFT) for a single window.
///
/// Computes the complex DFT and returns squared magnitudes.
fn dft<T: Float>(samples: &[T]) -> Vec<T> {
    let fft_size = samples.len();
    let mut magnitudes = vec![T::zero(); fft_size / 2 + 1];

    for k in 0..magnitudes.len() {
        let mut real = T::zero();
        let mut imag = T::zero();

        for (n, &sample) in samples.iter().enumerate() {
            let angle = T::from(-2.0).unwrap() * T::from(std::f64::consts::PI).unwrap()
                * T::from(k).unwrap() * T::from(n).unwrap() / T::from(fft_size).unwrap();
            real = real + sample * angle.cos();
            imag = imag + sample * angle.sin();
        }

        // Squared magnitude: real^2 + imag^2
        magnitudes[k] = real * real + imag * imag;
    }

    magnitudes
}

/// Apply Hann window to audio samples.
fn apply_hann_window<T: Float>(samples: &[T]) -> Vec<T> {
    let n = samples.len();
    let mut windowed = Vec::with_capacity(n);

    for (i, &sample) in samples.iter().enumerate() {
        let hann = T::from(0.5).unwrap()
            * (T::one()
                - T::from(2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64)
                    .unwrap()
                    .cos());
        windowed.push(sample * hann);
    }

    windowed
}

/// Compute mel spectrogram from PCM audio samples.
///
/// Single-threaded implementation (no parallel FFT as per plan).
fn log_mel_spectrogram_<T: Float>(
    samples: &[T],
    filters: &[T],
    fft_size: usize,
    fft_step: usize,
    n_mel: usize,
) -> Vec<T> {
    // Pad samples to ensure we get exactly the expected number of frames
    // For Whisper: N_SAMPLES=480000, HOP_LENGTH=160 → 3000 frames
    let expected_frames = samples.len() / fft_step;
    let padded_len = expected_frames * fft_step + fft_size;
    let mut padded_samples = samples.to_vec();
    padded_samples.resize(padded_len, T::zero());

    let mut mel = vec![T::zero(); n_mel * expected_frames];

    // For each frame, compute FFT and apply mel filters
    for (frame_idx, frame_start) in (0..padded_samples.len() - fft_size + 1)
        .step_by(fft_step)
        .take(expected_frames)
        .enumerate()
    {
        // Extract window
        let window_samples = &padded_samples[frame_start..frame_start + fft_size];

        // Apply Hann window
        let windowed = apply_hann_window(window_samples);

        // Compute DFT (squared magnitudes)
        let magnitudes = dft(&windowed);

        // Apply mel filter bank — stored in mel-bins-major order [n_mel, n_frames]
        // so that mel[mel_idx * expected_frames + frame_idx] matches [1, n_mel, n_frames] tensor layout
        for mel_idx in 0..n_mel {
            let filter_start = mel_idx * magnitudes.len();
            let filter_end = filter_start + magnitudes.len();
            let filter_slice = &filters[filter_start..filter_end];

            // Dot product of magnitudes and filter
            let mut sum = T::zero();
            for (mag, &filt) in magnitudes.iter().zip(filter_slice.iter()) {
                sum = sum + *mag * filt;
            }

            // Log-mel: log10(max(sum, epsilon))
            let log_value = sum.max(T::from(EPSILON).unwrap()).log10();
            mel[mel_idx * expected_frames + frame_idx] = log_value;
        }
    }

    // Normalize: clamp to within 8 of max, then scale to roughly [-1, 1]
    // Formula: (max(m, max_val - 8) + 4) / 4
    // Matches OpenAI Whisper / candle-transformers normalization
    let max_val = mel
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(T::zero());
    let mmax = max_val - T::from(8).unwrap();
    let four = T::from(4).unwrap();
    let one = T::one();

    for m in mel.iter_mut() {
        let v = T::max(*m, mmax);
        *m = v / four + one;
    }

    mel
}

/// Convert PCM audio samples to mel spectrogram.
///
/// # Arguments
/// * `config` - Whisper configuration (for num_mel_bins)
/// * `samples` - PCM audio as f32 samples (normalized to [-1, 1])
///
/// # Returns
/// Flattened mel spectrogram: [n_mel * n_frames] values
pub fn pcm_to_mel(config: &Config, samples: &[f32]) -> Vec<f32> {
    let filters = load_mel_filters();
    log_mel_spectrogram_(samples, &filters, N_FFT, HOP_LENGTH, config.num_mel_bins)
}
