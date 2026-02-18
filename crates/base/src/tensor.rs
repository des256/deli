use std::fmt;

#[derive(Debug, PartialEq)]
pub enum TensorError {
    ShapeOverflow,
    ShapeMismatch { expected: usize, got: usize },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeOverflow => write!(f, "shape dimensions overflow when multiplied"),
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected} elements, got {got}")
            }
        }
    }
}

impl std::error::Error for TensorError {}

#[derive(Clone, PartialEq)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data", &self.data)
            .finish()
    }
}

impl<T> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Result<Self, TensorError> {
        // Compute shape product using checked_mul to detect overflow
        let mut product: usize = 1;
        for &dim in &shape {
            product = product
                .checked_mul(dim)
                .ok_or(TensorError::ShapeOverflow)?;
        }

        // Validate data length matches shape product
        if product != data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: product,
                got: data.len(),
            });
        }

        Ok(Self { shape, data })
    }

    pub fn from_scalar(value: T) -> Self {
        Self {
            shape: vec![],
            data: vec![value],
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T: Default + Clone> Tensor<T> {
    pub fn zeros(shape: Vec<usize>) -> Result<Self, TensorError> {
        // Compute shape product using checked_mul to detect overflow
        let mut product: usize = 1;
        for &dim in &shape {
            product = product
                .checked_mul(dim)
                .ok_or(TensorError::ShapeOverflow)?;
        }

        let data = vec![T::default(); product];
        Ok(Self { shape, data })
    }
}
