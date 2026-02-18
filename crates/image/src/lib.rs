//! Image decoding utilities for the deli ecosystem.
//!
//! This crate provides a simple wrapper around the `image` crate to decode
//! images from memory into `Tensor<T>` from `deli-base`.
//!
//! All decoded images use HWC layout: `[height, width, channels]`.

pub mod error;
pub mod image;

pub use error::ImageError;
pub use image::Image;

use base::Tensor;
use crates_image::{DynamicImage, ImageEncoder};

fn to_tensor<T>(
    width: u32,
    height: u32,
    channels: usize,
    data: Vec<T>,
) -> Result<Tensor<T>, ImageError> {
    let shape = vec![height as usize, width as usize, channels];
    Ok(Tensor::new(shape, data)?)
}

fn decode_image_inner(data: &[u8]) -> Result<Image, ImageError> {
    let img = crates_image::load_from_memory(data)?;

    match img {
        DynamicImage::ImageLuma8(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::U8(to_tensor(w, h, 1, buf.into_raw())?))
        }
        DynamicImage::ImageLumaA8(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::U8(to_tensor(w, h, 2, buf.into_raw())?))
        }
        DynamicImage::ImageRgb8(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::U8(to_tensor(w, h, 3, buf.into_raw())?))
        }
        DynamicImage::ImageRgba8(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::U8(to_tensor(w, h, 4, buf.into_raw())?))
        }
        DynamicImage::ImageLuma16(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::U16(to_tensor(w, h, 1, buf.into_raw())?))
        }
        DynamicImage::ImageLumaA16(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::U16(to_tensor(w, h, 2, buf.into_raw())?))
        }
        DynamicImage::ImageRgb16(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::U16(to_tensor(w, h, 3, buf.into_raw())?))
        }
        DynamicImage::ImageRgba16(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::U16(to_tensor(w, h, 4, buf.into_raw())?))
        }
        DynamicImage::ImageRgb32F(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::F32(to_tensor(w, h, 3, buf.into_raw())?))
        }
        DynamicImage::ImageRgba32F(buf) => {
            let (w, h) = buf.dimensions();
            Ok(Image::F32(to_tensor(w, h, 4, buf.into_raw())?))
        }
        _ => {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            Ok(Image::U8(to_tensor(w, h, 4, rgba.into_raw())?))
        }
    }
}

fn encode_jpeg_inner(image: &Image, quality: u8) -> Result<Vec<u8>, ImageError> {
    let (width, height) = (image.width() as u32, image.height() as u32);
    let channels = image.channels();

    let u8_data: Vec<u8> = match image {
        Image::U8(t) => t.data.clone(),
        Image::U16(t) => t.data.iter().map(|&v| (v >> 8) as u8).collect(),
        Image::F32(t) => t
            .data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect(),
    };

    let (jpeg_data, color_type) = match channels {
        1 => (u8_data, crates_image::ExtendedColorType::L8),
        2 => {
            let stripped: Vec<u8> = u8_data.chunks(2).map(|c| c[0]).collect();
            (stripped, crates_image::ExtendedColorType::L8)
        }
        3 => (u8_data, crates_image::ExtendedColorType::Rgb8),
        4 => {
            let stripped: Vec<u8> = u8_data.chunks(4).flat_map(|c| &c[..3]).copied().collect();
            (stripped, crates_image::ExtendedColorType::Rgb8)
        }
        _ => {
            return Err(ImageError::Encode(format!(
                "unsupported channel count: {channels}"
            )));
        }
    };

    let mut buffer = Vec::new();
    let encoder = crates_image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
    encoder
        .write_image(&jpeg_data, width, height, color_type)
        .map_err(|e| ImageError::Encode(e.to_string()))?;

    Ok(buffer)
}

/// Decodes an image from raw bytes into a typed tensor.
///
/// The image format is auto-detected by the `image` crate. Returns a
/// `Image` enum that preserves the original pixel precision (u8, u16, or f32).
///
/// All tensors use HWC layout: `[height, width, channels]`.
///
/// The CPU-bound decoding work runs on tokio's blocking thread pool.
///
/// # Errors
///
/// Returns `ImageError::Decode` if the data is invalid or the format is unsupported.
/// Returns `ImageError::Tensor` if tensor construction fails.
pub async fn decode_image(data: &[u8]) -> Result<Image, ImageError> {
    let owned = data.to_vec();
    tokio::task::spawn_blocking(move || decode_image_inner(&owned))
        .await
        .map_err(|e| ImageError::Decode(e.to_string()))?
}

/// Encodes a `Image` as JPEG bytes.
///
/// The `quality` parameter controls JPEG compression (1–100, higher = better quality).
///
/// JPEG supports grayscale (1 channel) and RGB (3 channels). Images with an alpha
/// channel (2 or 4 channels) have alpha stripped automatically. U16 and F32 images
/// are converted to U8 before encoding.
///
/// The CPU-bound encoding work runs on tokio's blocking thread pool.
///
/// # Errors
///
/// Returns `ImageError::Encode` if the channel count is unsupported or encoding fails.
pub async fn encode_jpeg(image: Image, quality: u8) -> Result<Vec<u8>, ImageError> {
    tokio::task::spawn_blocking(move || encode_jpeg_inner(&image, quality))
        .await
        .map_err(|e| ImageError::Encode(e.to_string()))?
}
