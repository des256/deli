use deli_base::Language;

pub enum Transcribed {
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
