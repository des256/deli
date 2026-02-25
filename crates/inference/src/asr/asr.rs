use crate::error::Result;
pub enum Asr {
    Sherpa,
    Parakeet,
}

impl Asr {
    pub fn new() -> Result<Self> {
        Ok(Self::Parakeet)
    }
}
