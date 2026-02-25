use crate::error::{InferError, Result};
use std::f32::consts::PI;
use std::path::Path;
use std::sync::OnceLock;

const REQUIRED_SAMPLE_RATE: usize = 16000;
const HOP_SIZE: usize = 160; // 10ms at 16kHz
const PRE_EMPHASIS: f32 = 0.97;

/// NeMo's log zero guard value (2^-24, smallest normal float16).
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-08;

/// Pre-computed mel filterbank from NeMo's preprocessor [128 x 257].
/// Loaded once from `mel_filterbank.bin` (exported from the .nemo checkpoint).
static MEL_FILTERBANK: OnceLock<Vec<f32>> = OnceLock::new();

/// Pre-computed symmetric Hann window from NeMo's preprocessor [400].
/// Loaded once from `hann_window.bin`.
static HANN_WINDOW: OnceLock<Vec<f32>> = OnceLock::new();

/*
/// Initialize feature extraction by loading the mel filterbank and window
/// from binary files exported from the NeMo model's preprocessor.
///
/// Must be called before `compute_features`. Pass the directory containing
/// `mel_filterbank.bin` and `hann_window.bin`.
pub fn init_features(model_dir: &Path) -> Result<()> {
    if MEL_FILTERBANK.get().is_some() {
        return Ok(());
    }

    let fb_path = model_dir.join("mel_filterbank.bin");
    let fb_data = std::fs::read(&fb_path).map_err(|e| {
        InferError::Runtime(format!("Failed to read {}: {}", fb_path.display(), e))
    })?;
    let expected_fb_bytes = NUM_MEL_BINS * SPECTRUM_BINS * 4;
    if fb_data.len() != expected_fb_bytes {
        return Err(InferError::Runtime(format!(
            "mel_filterbank.bin: expected {} bytes, got {}",
            expected_fb_bytes,
            fb_data.len()
        )));
    }
    let fb: Vec<f32> = fb_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let _ = MEL_FILTERBANK.set(fb);

    let win_path = model_dir.join("hann_window.bin");
    let win_data = std::fs::read(&win_path).map_err(|e| {
        InferError::Runtime(format!("Failed to read {}: {}", win_path.display(), e))
    })?;
    let expected_win_bytes = WINDOW_SIZE * 4;
    if win_data.len() != expected_win_bytes {
        return Err(InferError::Runtime(format!(
            "hann_window.bin: expected {} bytes, got {}",
            expected_win_bytes,
            win_data.len()
        )));
    }
    let win: Vec<f32> = win_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let _ = HANN_WINDOW.set(win);

    Ok(())
}
*/

/// Compute 128-dimensional log-mel filterbank features from 16kHz PCM audio.
///
/// Uses NeMo's exact preprocessor parameters (filterbank + window loaded from
/// binary files via `init_features`). Falls back to computed parameters if
/// `init_features` was not called.
///
/// Returns (features, num_frames) where features is channels-first [128, num_frames] flattened.
pub fn compute_features(pcm: &[i16], sample_rate: usize) -> Result<(Vec<f32>, usize)> {
    if sample_rate != REQUIRED_SAMPLE_RATE {
        return Err(InferError::Runtime(format!(
            "compute_features requires {} Hz audio, got {} Hz",
            REQUIRED_SAMPLE_RATE, sample_rate
        )));
    }

    if pcm.len() < WINDOW_SIZE {
        return Err(InferError::Runtime(format!(
            "Audio too short: {} samples, need at least {}",
            pcm.len(),
            WINDOW_SIZE
        )));
    }

    // Get pre-loaded filterbank and window, or fall back to computed versions
    let (fb, hann) = match (MEL_FILTERBANK.get(), HANN_WINDOW.get()) {
        (Some(fb), Some(win)) => (fb.as_slice(), win.as_slice()),
        _ => {
            return Err(InferError::Runtime(
                "Features not initialized. Call init_features() first.".to_string(),
            ));
        }
    };

    // Convert to f32 and apply pre-emphasis
    let mut signal: Vec<f32> = vec![0.0; pcm.len()];
    signal[0] = pcm[0] as f32 / 32768.0;
    for i in 1..pcm.len() {
        signal[i] = (pcm[i] as f32 / 32768.0) - PRE_EMPHASIS * (pcm[i - 1] as f32 / 32768.0);
    }

    let num_frames = (signal.len() - WINDOW_SIZE) / HOP_SIZE + 1;

    // Compute log-mel features in channels-first layout [128, num_frames]
    let mut features = vec![0.0f32; NUM_MEL_BINS * num_frames];

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_SIZE;

        // Apply window and zero-pad to FFT size
        let mut windowed = vec![0.0f32; FFT_SIZE];
        for i in 0..WINDOW_SIZE {
            windowed[i] = signal[start + i] * hann[i];
        }

        // Compute power spectrum
        let power_spectrum = compute_power_spectrum(&windowed);

        // Apply mel filterbank (dense matrix multiply) and log
        for bin_idx in 0..NUM_MEL_BINS {
            let fb_offset = bin_idx * SPECTRUM_BINS;
            let mut mel_energy = 0.0_f32;
            for k in 0..SPECTRUM_BINS {
                mel_energy += fb[fb_offset + k] * power_spectrum[k];
            }
            features[bin_idx * num_frames + frame_idx] = (mel_energy + LOG_ZERO_GUARD).ln();
        }
    }

    Ok((features, num_frames))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_spectrum_dc() {
        // DC signal (all ones) should have energy only at bin 0
        let signal = vec![1.0; 512];
        let power = compute_power_spectrum(&signal);
        assert!(power[0] > 0.0);
        for p in &power[1..10] {
            assert!(*p < 1.0);
        }
    }
}
