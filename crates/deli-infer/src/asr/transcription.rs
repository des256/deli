use deli_base::Language;

pub enum Transcription {
    Partial {
        text: String,
        language: Language,
        confidence: f32,
    },
    Final {
        text: String,
        language: Language,
        confidence: f32,
    },
    Cancelled,
}
