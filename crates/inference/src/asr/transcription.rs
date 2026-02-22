pub enum Transcription {
    Partial { text: String, confidence: f32 },
    Final { text: String, confidence: f32 },
    Cancelled,
}
