use crate::{Device, InferError, ModelSource, Session};

pub trait Backend {
    fn name(&self) -> &str;
    fn load_model(
        &self,
        model: ModelSource,
        device: Device,
    ) -> Result<Box<dyn Session>, InferError>;
}
