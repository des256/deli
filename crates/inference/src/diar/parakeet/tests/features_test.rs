use crate::diar::parakeet::features::{compute_mel_features, compute_mel_filterbank};
use crate::diar::parakeet::{DiarizationConfig, SpeakerSegment};

#[test]
fn test_diarization_config_callhome() {
    let config = DiarizationConfig::callhome();
    assert!((config.onset - 0.641).abs() < 1e-6, "onset should be 0.641");
    assert!((config.offset - 0.561).abs() < 1e-6, "offset should be 0.561");
}

#[test]
fn test_diarization_config_dihard3() {
    let config = DiarizationConfig::dihard3();
    assert!((config.onset - 0.680).abs() < 1e-6);
    assert!((config.offset - 0.561).abs() < 1e-6);
}

#[test]
fn test_diarization_config_custom() {
    let config = DiarizationConfig::custom(0.5, 0.4);
    assert!((config.onset - 0.5).abs() < 1e-6);
    assert!((config.offset - 0.4).abs() < 1e-6);
}

#[test]
fn test_speaker_segment_public() {
    // Verify SpeakerSegment is accessible and has expected fields
    let segment = SpeakerSegment {
        start: 1.0,
        end: 2.5,
        speaker_id: 0,
    };
    assert_eq!(segment.start, 1.0);
    assert_eq!(segment.end, 2.5);
    assert_eq!(segment.speaker_id, 0);
}

#[test]
fn test_compute_features_dimensions() {
    // 1 second of audio at 16kHz
    let audio: Vec<f32> = vec![0.0; 16000];
    let sample_rate = 16000;

    let result = compute_mel_features(&audio, sample_rate);
    assert!(result.is_ok(), "Feature extraction should succeed");

    let (features, num_frames) = result.unwrap();

    // Expected number of frames: (16000 - 400) / 160 + 1 = 98
    // Features should be time-first: num_frames * 128
    assert_eq!(features.len(), num_frames * 128, "Features should be time-first [T, 128] flattened");
    assert!(num_frames > 0, "Should produce non-zero frames");
}

#[test]
fn test_compute_features_invalid_sample_rate() {
    let audio: Vec<f32> = vec![0.0; 8000];
    let result = compute_mel_features(&audio, 8000);
    assert!(result.is_err(), "Should error on wrong sample rate");
}

#[test]
fn test_compute_features_too_short() {
    // Audio shorter than window size (400 samples)
    let audio: Vec<f32> = vec![0.0; 300];
    let result = compute_mel_features(&audio, 16000);
    assert!(result.is_err(), "Should error on audio shorter than window size");
}

#[test]
fn test_mel_filterbank_properties() {
    let filterbank = compute_mel_filterbank(16000, 512, 128);

    assert_eq!(filterbank.len(), 128, "Should have 128 mel filters");

    // Most filters should have non-zero weights (some low-frequency filters may be empty
    // due to FFT bin quantization)
    let non_empty_count = filterbank.iter().filter(|f| !f.is_empty()).count();
    assert!(
        non_empty_count >= 110,
        "At least 110/128 filters should have non-zero weights (low-freq bins may be empty due to quantization), got {}",
        non_empty_count
    );

    // All non-empty filters should have positive weights
    for filter in &filterbank {
        for &(_, weight) in filter {
            assert!(weight > 0.0, "All filter weights should be positive");
        }
    }
}

#[test]
fn test_sine_wave_energy_distribution() {
    // Generate a 440Hz sine wave (A4 note)
    // This should produce higher energy in lower mel bins than upper bins
    let sample_rate = 16000;
    let duration_samples = 16000; // 1 second
    let frequency = 440.0;

    let mut audio = Vec::with_capacity(duration_samples);
    for i in 0..duration_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = 0.5 * (2.0 * std::f32::consts::PI * frequency * t).sin();
        audio.push(sample);
    }

    let (features, num_frames) = compute_mel_features(&audio, sample_rate)
        .expect("Feature extraction should succeed");

    // Compute average energy per mel bin across all frames
    let mut bin_energies = vec![0.0f32; 128];
    for frame_idx in 0..num_frames {
        for bin_idx in 0..128 {
            bin_energies[bin_idx] += features[frame_idx * 128 + bin_idx];
        }
    }
    for energy in &mut bin_energies {
        *energy /= num_frames as f32;
    }

    // 440Hz should concentrate energy in lower mel bins (roughly bins 10-30)
    // Average energy in lower bins should be higher than upper bins
    let lower_avg: f32 = bin_energies[10..40].iter().sum::<f32>() / 30.0;
    let upper_avg: f32 = bin_energies[90..120].iter().sum::<f32>() / 30.0;

    assert!(
        lower_avg > upper_avg,
        "440Hz sine wave should have higher energy in lower mel bins (lower: {}, upper: {})",
        lower_avg,
        upper_avg
    );
}
