use std::path::PathBuf;

pub enum ModelSource {
    File(PathBuf),
    Memory(Vec<u8>),
}
