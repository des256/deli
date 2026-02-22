use {codec::{Codec, Dart}, std::fmt};

#[derive(Debug, Clone, Copy, Codec, Dart)]
pub enum Language {
    EnglishUs,
    ChineseChina,
    KoreanKorea,
    DutchNetherlands,
    FrenchFrance,
}

impl Language {
    /// Parse an IETF BCP 47 language tag (e.g. `"en-us"`) into a `Language`.
    ///
    /// Matching is case-insensitive.
    pub fn from_ietf(tag: &str) -> Option<Language> {
        match tag.to_ascii_lowercase().as_str() {
            "en-us" => Some(Language::EnglishUs),
            "zh-cn" => Some(Language::ChineseChina),
            "ko-kr" => Some(Language::KoreanKorea),
            "nl-nl" => Some(Language::DutchNetherlands),
            "fr-fr" => Some(Language::FrenchFrance),
            _ => None,
        }
    }

    /// Return the IETF BCP 47 language tag for this language.
    pub fn to_ietf(self) -> &'static str {
        match self {
            Language::EnglishUs => "en-us",
            Language::ChineseChina => "zh-cn",
            Language::KoreanKorea => "ko-kr",
            Language::DutchNetherlands => "nl-nl",
            Language::FrenchFrance => "fr-fr",
        }
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Language::EnglishUs => write!(f, "English (US)"),
            Language::ChineseChina => write!(f, "Chinese (China)"),
            Language::KoreanKorea => write!(f, "Korean (Korea)"),
            Language::DutchNetherlands => write!(f, "Dutch (Netherlands)"),
            Language::FrenchFrance => write!(f, "French (France)"),
        }
    }
}
