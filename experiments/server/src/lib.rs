// MANUAL DART CODEGEN: Update rstypes/lib/src/language.dart when modifying this type
#[derive(codec::Codec, codec::Dart)]
pub enum Language {
    EnglishUs,
    ChineseChina,
    KoreanKorea,
    DutchNetherlands,
    FrenchFrance,
}

// MANUAL DART CODEGEN: Update rstypes/lib/src/to_monitor.dart when modifying this type
#[derive(codec::Codec, codec::Dart)]
pub enum ToMonitor {
    VideoJpeg(Vec<u8>),
    Settings { language: Language },
}

// MANUAL DART CODEGEN: Update rstypes/lib/src/from_monitor.dart when modifying this type
#[derive(codec::Codec, codec::Dart)]
pub enum FromMonitor {
    Settings { language: Language },
}
