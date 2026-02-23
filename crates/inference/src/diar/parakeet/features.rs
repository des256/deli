use crate::error::{InferError, Result};
use std::f32::consts::PI;

const SAMPLE_RATE: usize = 16000;
const WINDOW_SIZE_MS: usize = 25;
const HOP_SIZE_MS: usize = 10;
const NUM_MEL_BINS: usize = 128;
const PRE_EMPHASIS: f32 = 0.97;
const FFT_SIZE: usize = 512;
const LOG_ZERO_GUARD: f32 = 5.96e-8;

// Mel scale conversion (HTK formula)
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Compute 128-dimensional log-mel filterbank features from 16kHz PCM audio.
///
/// Input: f32 audio already normalized to [-1, 1] range.
/// Output: (features, num_frames) where features is time-first [num_frames, 128] flattened.
///
/// Unlike ASR features, diarization features:
/// - Use time-first layout [T, 128] not channels-first [128, T]
/// - Do NOT apply per-feature normalization
/// - Use LOG_ZERO_GUARD = 5.96e-8 instead of 1e-10
pub fn compute_mel_features(audio: &[f32], sample_rate: usize) -> Result<(Vec<f32>, usize)> {
    if sample_rate != SAMPLE_RATE {
        return Err(InferError::Runtime(format!(
            "compute_mel_features requires {} Hz audio, got {} Hz",
            SAMPLE_RATE, sample_rate
        )));
    }

    let window_size = (WINDOW_SIZE_MS * sample_rate) / 1000;
    let hop_size = (HOP_SIZE_MS * sample_rate) / 1000;

    if audio.len() < window_size {
        return Err(InferError::Runtime(format!(
            "Audio too short: {} samples, need at least {}",
            audio.len(),
            window_size
        )));
    }

    // Apply pre-emphasis (input is already f32 in [-1, 1])
    let mut signal: Vec<f32> = vec![0.0; audio.len()];
    signal[0] = audio[0];
    for i in 1..audio.len() {
        signal[i] = audio[i] - PRE_EMPHASIS * audio[i - 1];
    }

    // Generate Hann window
    let hann: Vec<f32> = (0..window_size)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_size - 1) as f32).cos())
        .collect();

    // Generate mel filterbank
    let mel_filters = compute_mel_filterbank(sample_rate, FFT_SIZE, NUM_MEL_BINS);

    // Frame the signal and compute features
    let num_frames = (signal.len() - window_size) / hop_size + 1;

    // Compute log-mel features in time-first layout [num_frames, 128]
    let mut features = Vec::with_capacity(num_frames * NUM_MEL_BINS);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let end = start + window_size;

        // Apply window
        let mut windowed: Vec<f32> = signal[start..end]
            .iter()
            .zip(hann.iter())
            .map(|(s, w)| s * w)
            .collect();

        // Zero-pad to FFT size
        windowed.resize(FFT_SIZE, 0.0);

        // Compute power spectrum
        let power_spectrum = compute_power_spectrum(&windowed);

        // Apply mel filterbank
        for filter in &mel_filters {
            let mut mel_energy = 0.0_f32;
            for &(freq_bin, weight) in filter {
                mel_energy += power_spectrum[freq_bin] * weight;
            }

            // Log mel energy (NO normalization, unlike ASR)
            features.push((mel_energy + LOG_ZERO_GUARD).ln());
        }
    }

    Ok((features, num_frames))
}

/// Compute mel filterbank matrix.
///
/// Returns a sparse representation: Vec of (freq_bin_idx, weight) pairs for each mel bin.
pub fn compute_mel_filterbank(
    sample_rate: usize,
    fft_size: usize,
    num_bins: usize,
) -> Vec<Vec<(usize, f32)>> {
    let nyquist = sample_rate as f32 / 2.0;
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(nyquist);

    // Create mel-spaced center frequencies
    let mel_points: Vec<f32> = (0..=num_bins + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_bins + 1) as f32)
        .map(mel_to_hz)
        .collect();

    // Convert to FFT bin indices
    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&freq| ((freq * fft_size as f32 / sample_rate as f32) + 0.5).floor() as usize)
        .collect();

    // Build triangular filters (sparse representation)
    let mut filters = Vec::with_capacity(num_bins);

    for i in 0..num_bins {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        let mut filter = Vec::new();

        // Rising edge
        if center > left {
            for bin in left..center {
                let weight = (bin - left) as f32 / (center - left) as f32;
                if weight > 0.0 {
                    filter.push((bin, weight));
                }
            }
        }

        // Falling edge
        if right > center {
            for bin in center..right {
                let weight = (right - bin) as f32 / (right - center) as f32;
                if weight > 0.0 {
                    filter.push((bin, weight));
                }
            }
        }

        filters.push(filter);
    }

    filters
}

/// Compute power spectrum from windowed signal using DFT
fn compute_power_spectrum(signal: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let mut power = vec![0.0; n / 2 + 1];

    for k in 0..=n / 2 {
        let mut real = 0.0;
        let mut imag = 0.0;

        for (t, &sample) in signal.iter().enumerate() {
            let angle = -2.0 * PI * k as f32 * t as f32 / n as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }

        power[k] = real * real + imag * imag;
    }

    power
}
