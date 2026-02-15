use crate::{InferError, ModelSource, Session};

pub trait Backend {
    fn load_model(
        &self,
        model: ModelSource,
    ) -> Result<Box<dyn Session>, InferError>;
}
