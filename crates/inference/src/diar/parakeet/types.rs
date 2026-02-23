/// Configuration for speaker diarization.
#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    /// Onset threshold for hysteresis binarization.
    pub onset: f32,
    /// Offset threshold for hysteresis binarization.
    pub offset: f32,
}

impl DiarizationConfig {
    /// CallHome preset (default).
    pub fn callhome() -> Self {
        Self {
            onset: 0.641,
            offset: 0.561,
        }
    }

    /// DI HARD3 preset.
    pub fn dihard3() -> Self {
        Self {
            onset: 0.680,
            offset: 0.561,
        }
    }

    /// Custom configuration.
    ///
    /// Onset must be >= offset, and both must be in [0.0, 1.0].
    pub fn custom(onset: f32, offset: f32) -> Self {
        debug_assert!(
            onset >= offset && onset >= 0.0 && onset <= 1.0 && offset >= 0.0 && offset <= 1.0,
            "Invalid thresholds: onset={onset}, offset={offset} (need 0 <= offset <= onset <= 1)"
        );
        Self { onset, offset }
    }
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self::callhome()
    }
}

/// A speaker segment with start/end times and speaker ID.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSegment {
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Speaker ID (0-3 for 4-speaker model).
    pub speaker_id: usize,
}
