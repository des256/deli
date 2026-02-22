use crate::error::{InferError, Result};
use std::f32::consts::PI;

const REQUIRED_SAMPLE_RATE: usize = 16000;
const WINDOW_SIZE_MS: usize = 25;
const HOP_SIZE_MS: usize = 10;
const NUM_MEL_BINS: usize = 80;
const PRE_EMPHASIS: f32 = 0.97;
const FFT_SIZE: usize = 512;

// Mel scale conversion
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Compute 80-dimensional log-mel filterbank features from 16kHz PCM audio.
///
/// Follows Kaldi fbank configuration:
/// - 25ms window (400 samples at 16kHz)
/// - 10ms hop (160 samples)
/// - Pre-emphasis 0.97
/// - Hann window
/// - 80 mel bins
///
/// Returns flattened Vec<f32> of shape [num_frames, 80].
pub fn compute_features(pcm: &[i16], sample_rate: usize) -> Result<Vec<f32>> {
    if sample_rate != REQUIRED_SAMPLE_RATE {
        return Err(InferError::Runtime(format!(
            "compute_features requires {} Hz audio, got {} Hz",
            REQUIRED_SAMPLE_RATE, sample_rate
        )));
    }

    let window_size = (WINDOW_SIZE_MS * sample_rate) / 1000;
    let hop_size = (HOP_SIZE_MS * sample_rate) / 1000;

    if pcm.len() < window_size {
        return Err(InferError::Runtime(format!(
            "Audio too short: {} samples, need at least {}",
            pcm.len(),
            window_size
        )));
    }

    // Convert to f32 and apply pre-emphasis
    let mut signal: Vec<f32> = vec![0.0; pcm.len()];
    signal[0] = pcm[0] as f32 / 32768.0;
    for i in 1..pcm.len() {
        signal[i] = (pcm[i] as f32 / 32768.0) - PRE_EMPHASIS * (pcm[i - 1] as f32 / 32768.0);
    }

    // Generate Hann window
    let hann: Vec<f32> = (0..window_size)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_size - 1) as f32).cos())
        .collect();

    // Generate mel filterbank
    let mel_filters = create_mel_filterbank(sample_rate, FFT_SIZE, NUM_MEL_BINS);

    // Frame the signal and compute features
    let num_frames = (signal.len() - window_size) / hop_size + 1;
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

            // Log mel energy (add small constant to avoid log(0))
            features.push((mel_energy + 1e-10_f32).ln());
        }
    }

    Ok(features)
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

/// Create mel filterbank (triangular filters)
fn create_mel_filterbank(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_conversion() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 0.01);
    }

    #[test]
    fn test_power_spectrum_dc() {
        // DC signal (all ones) should have energy only at bin 0
        let signal = vec![1.0; 512];
        let power = compute_power_spectrum(&signal);
        assert!(power[0] > 0.0);
        // Other bins should be near zero
        for p in &power[1..10] {
            assert!(*p < 1.0);
        }
    }
}
