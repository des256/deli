#[derive(deli_codec::Codec, deli_codec::Dart)]
pub struct Data {
    value: i32,
    flag: bool,
}

impl Data {
    pub fn new(value: i32, flag: bool) -> Self {
        Self { value, flag }
    }
}
