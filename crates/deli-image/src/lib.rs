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

fn to_tensor<T>(width: u32, height: u32, channels: usize, data: Vec<T>) -> Result<Tensor<T>, ImageError> {
    let shape = vec![height as usize, width as usize, channels];
    Ok(Tensor::new(shape, data)?)
}

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
        DynamicImage::ImageLuma8(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::U8(to_tensor(w, h, 1, buf.into_raw())?)) }
        DynamicImage::ImageLumaA8(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::U8(to_tensor(w, h, 2, buf.into_raw())?)) }
        DynamicImage::ImageRgb8(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::U8(to_tensor(w, h, 3, buf.into_raw())?)) }
        DynamicImage::ImageRgba8(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::U8(to_tensor(w, h, 4, buf.into_raw())?)) }
        DynamicImage::ImageLuma16(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::U16(to_tensor(w, h, 1, buf.into_raw())?)) }
        DynamicImage::ImageLumaA16(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::U16(to_tensor(w, h, 2, buf.into_raw())?)) }
        DynamicImage::ImageRgb16(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::U16(to_tensor(w, h, 3, buf.into_raw())?)) }
        DynamicImage::ImageRgba16(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::U16(to_tensor(w, h, 4, buf.into_raw())?)) }
        DynamicImage::ImageRgb32F(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::F32(to_tensor(w, h, 3, buf.into_raw())?)) }
        DynamicImage::ImageRgba32F(buf) => { let (w, h) = buf.dimensions(); Ok(DecodedImage::F32(to_tensor(w, h, 4, buf.into_raw())?)) }
        _ => {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            Ok(DecodedImage::U8(to_tensor(w, h, 4, rgba.into_raw())?))
        }
    }
}
