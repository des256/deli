use codec::{Codec, Dart};

#[derive(Codec, Dart)]
#[dart(target = "products/testy/webui/lib/rstypes")]
pub enum ToMonitor {
    Jpeg(Vec<u8>),
}
