#[derive(codec::Codec, codec::Dart)]
pub enum ToMonitor {
    VideoJpeg(Vec<u8>),
}
