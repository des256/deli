use base::Tensor;
use crates_image::ImageEncoder;
use image::{Image, decode_image, encode_jpeg};

#[tokio::test]
async fn test_encode_jpeg_rgb_roundtrip() {
    // Create a 4x4 RGB image, encode as PNG, decode, then re-encode as JPEG
    let mut png_buf = Vec::new();
    let img = crates_image::RgbImage::from_fn(4, 4, |x, y| {
        let val = ((x + y) * 32) as u8;
        crates_image::Rgb([val, val + 16, val + 32])
    });
    crates_image::codecs::png::PngEncoder::new(&mut png_buf)
        .write_image(img.as_raw(), 4, 4, crates_image::ExtendedColorType::Rgb8)
        .unwrap();

    let decoded = decode_image(&png_buf).await.unwrap();
    let jpeg_bytes = encode_jpeg(decoded, 90).await.unwrap();

    // Verify valid JPEG by decoding back
    let re_decoded = decode_image(&jpeg_bytes).await.unwrap();
    assert_eq!(re_decoded.height(), 4);
    assert_eq!(re_decoded.width(), 4);
    assert_eq!(re_decoded.channels(), 3);
}

#[tokio::test]
async fn test_encode_jpeg_grayscale() {
    let data: Vec<u8> = (0..16).collect();
    let tensor = Tensor::new(vec![4, 4, 1], data).unwrap();
    let image = Image::U8(tensor);

    let jpeg_bytes = encode_jpeg(image, 80).await.unwrap();

    let re_decoded = decode_image(&jpeg_bytes).await.unwrap();
    assert_eq!(re_decoded.height(), 4);
    assert_eq!(re_decoded.width(), 4);
    assert_eq!(re_decoded.channels(), 1);
}

#[tokio::test]
async fn test_encode_jpeg_rgba_strips_alpha() {
    let data: Vec<u8> = (0..64).collect(); // 4x4x4
    let tensor = Tensor::new(vec![4, 4, 4], data).unwrap();
    let image = Image::U8(tensor);

    let jpeg_bytes = encode_jpeg(image, 80).await.unwrap();

    let re_decoded = decode_image(&jpeg_bytes).await.unwrap();
    assert_eq!(re_decoded.height(), 4);
    assert_eq!(re_decoded.width(), 4);
    assert_eq!(re_decoded.channels(), 3);
}

#[tokio::test]
async fn test_encode_jpeg_luma_alpha_strips_alpha() {
    let data: Vec<u8> = (0..32).collect(); // 4x4x2
    let tensor = Tensor::new(vec![4, 4, 2], data).unwrap();
    let image = Image::U8(tensor);

    let jpeg_bytes = encode_jpeg(image, 80).await.unwrap();

    let re_decoded = decode_image(&jpeg_bytes).await.unwrap();
    assert_eq!(re_decoded.height(), 4);
    assert_eq!(re_decoded.width(), 4);
    assert_eq!(re_decoded.channels(), 1);
}

#[tokio::test]
async fn test_encode_jpeg_u16_converts_to_u8() {
    // U16 values: 0, 257, 514, ... → u8: 0, 1, 2, ...
    let data: Vec<u16> = (0..48).map(|v| v * 257).collect(); // 4x4x3
    let tensor = Tensor::new(vec![4, 4, 3], data).unwrap();
    let image = Image::U16(tensor);

    let jpeg_bytes = encode_jpeg(image, 90).await.unwrap();

    let re_decoded = decode_image(&jpeg_bytes).await.unwrap();
    assert_eq!(re_decoded.height(), 4);
    assert_eq!(re_decoded.width(), 4);
    assert_eq!(re_decoded.channels(), 3);
}

#[tokio::test]
async fn test_encode_jpeg_f32_converts_to_u8() {
    // F32 values in [0.0, 1.0] range
    let data: Vec<f32> = (0..48).map(|v| v as f32 / 47.0).collect(); // 4x4x3
    let tensor = Tensor::new(vec![4, 4, 3], data).unwrap();
    let image = Image::F32(tensor);

    let jpeg_bytes = encode_jpeg(image, 90).await.unwrap();

    let re_decoded = decode_image(&jpeg_bytes).await.unwrap();
    assert_eq!(re_decoded.height(), 4);
    assert_eq!(re_decoded.width(), 4);
    assert_eq!(re_decoded.channels(), 3);
}

#[tokio::test]
async fn test_encode_jpeg_f32_clamps_out_of_range() {
    // Values outside [0, 1] should be clamped
    let data: Vec<f32> = vec![[-0.5, 1.5, 0.5]; 16].into_iter().flatten().collect();
    let tensor = Tensor::new(vec![4, 4, 3], data).unwrap();
    let image = Image::F32(tensor);

    let jpeg_bytes = encode_jpeg(image, 80).await.unwrap();
    let re_decoded = decode_image(&jpeg_bytes).await.unwrap();
    assert_eq!(re_decoded.channels(), 3);
}

#[tokio::test]
async fn test_encode_jpeg_quality_affects_size() {
    let data: Vec<u8> = (0..192).map(|v| (v % 256) as u8).collect(); // 8x8x3
    let tensor = Tensor::new(vec![8, 8, 3], data).unwrap();
    let image = Image::U8(tensor.clone());

    let low_quality = encode_jpeg(image, 10).await.unwrap();
    let high_quality = encode_jpeg(Image::U8(tensor), 100).await.unwrap();

    assert!(low_quality.len() < high_quality.len());
}
