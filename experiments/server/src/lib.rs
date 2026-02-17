#[derive(deli_codec::Codec, deli_codec::Dart)]
pub struct Data {
    value: i32,
    flag: bool,
    frame: Vec<u8>,
}

impl Data {
    pub fn new(value: i32, flag: bool, frame: Vec<u8>) -> Self {
        Self { value, flag, frame }
    }
}
