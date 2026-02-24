pub(crate) mod gemma3;
pub use gemma3::Gemma3;

#[cfg(test)]
#[path = "tests/gemma3_test.rs"]
mod gemma3_test;
