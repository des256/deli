pub(crate) mod smollm3;
pub use smollm3::Smollm3;

#[cfg(test)]
#[path = "tests/smollm3_test.rs"]
mod smollm3_test;
