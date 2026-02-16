pub mod logging;
pub mod mat2;
pub mod mat3;
pub mod mat4;
pub mod pose;
pub mod rect;
pub mod quat;
pub mod tensor;
pub mod vec2;
pub mod vec3;
pub mod vec4;

pub use logging::{init_file_logger, init_stdout_logger, FileLogger, StdoutLogger};
pub use mat2::Mat2;
pub use mat3::Mat3;
pub use mat4::Mat4;
pub use pose::Pose;
pub use quat::Quat;
pub use rect::Rect;
pub use tensor::{Tensor, TensorError};
pub use vec2::Vec2;
pub use vec3::Vec3;
pub use vec4::Vec4;

// Re-export log crate so downstream crates can use deli_base::log::*
pub use log;
