//! Image decoding utilities for the deli ecosystem.
//!
//! This crate provides a simple wrapper around the `image` crate to decode
//! images from memory into `Tensor<T>` from `deli-base`.
//!
//! All decoded images use HWC layout: `[height, width, channels]`.

pub mod error;
pub mod types;

pub use error::ImageError;
pub use types::DecodedImage;

use deli_base::Tensor;
use image::DynamicImage;

/// Decodes an image from raw bytes into a typed tensor.
///
/// The image format is auto-detected by the `image` crate. Returns a
/// `DecodedImage` enum that preserves the original pixel precision (u8, u16, or f32).
///
/// All tensors use HWC layout: `[height, width, channels]`.
///
/// # Errors
///
/// Returns `ImageError::Decode` if the data is invalid or the format is unsupported.
/// Returns `ImageError::Tensor` if tensor construction fails.
pub fn decode_image(data: &[u8]) -> Result<DecodedImage, ImageError> {
    let img = image::load_from_memory(data)?;

    match img {
        // 8-bit variants
        DynamicImage::ImageLuma8(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 1];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::U8(tensor))
        }
        DynamicImage::ImageLumaA8(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 2];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::U8(tensor))
        }
        DynamicImage::ImageRgb8(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 3];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::U8(tensor))
        }
        DynamicImage::ImageRgba8(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 4];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::U8(tensor))
        }
        // 16-bit variants
        DynamicImage::ImageLuma16(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 1];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::U16(tensor))
        }
        DynamicImage::ImageLumaA16(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 2];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::U16(tensor))
        }
        DynamicImage::ImageRgb16(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 3];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::U16(tensor))
        }
        DynamicImage::ImageRgba16(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 4];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::U16(tensor))
        }
        // Float variants
        DynamicImage::ImageRgb32F(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 3];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::F32(tensor))
        }
        DynamicImage::ImageRgba32F(buf) => {
            let (width, height) = buf.dimensions();
            let shape = vec![height as usize, width as usize, 4];
            let tensor = Tensor::new(shape, buf.into_raw())?;
            Ok(DecodedImage::F32(tensor))
        }
        // Future-proofing: convert unknown variants to Rgba8
        _ => {
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            let shape = vec![height as usize, width as usize, 4];
            let tensor = Tensor::new(shape, rgba.into_raw())?;
            Ok(DecodedImage::U8(tensor))
        }
    }
}
