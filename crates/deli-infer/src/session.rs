use crate::InferError;
use deli_math::Tensor;
use std::collections::HashMap;

pub trait Session {
    fn run(
        &mut self,
        inputs: &[(&str, Tensor<f32>)],
    ) -> Result<HashMap<String, Tensor<f32>>, InferError>;
    fn input_names(&self) -> &[String];
    fn output_names(&self) -> &[String];
}
