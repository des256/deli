use crate::{Device, InferError};
use deli_math::Tensor;
use std::collections::HashMap;
use std::path::PathBuf;

pub enum ModelSource {
    File(PathBuf),
    Memory(Vec<u8>),
}

pub trait Backend {
    fn name(&self) -> &str;
    fn load_model(
        &self,
        model: ModelSource,
        device: Device,
    ) -> Result<Box<dyn Session>, InferError>;
}

pub trait Session {
    fn run(
        &mut self,
        inputs: &[(&str, Tensor<f32>)],
    ) -> Result<HashMap<String, Tensor<f32>>, InferError>;
    fn input_names(&self) -> &[String];
    fn output_names(&self) -> &[String];
}
