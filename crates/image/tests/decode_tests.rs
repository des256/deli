use crates_image::ImageEncoder;
use image::{Image, ImageError, decode_image};

#[tokio::test]
async fn test_decode_jpeg_rgb() {
    // Create a small 2x2 RGB JPEG image using the image crate
    let mut buffer = Vec::new();
    let img = crates_image::RgbImage::from_fn(2, 2, |x, y| {
        let val = (x + y) as u8 * 64;
        crates_image::Rgb([val, val + 16, val + 32])
    });

    crates_image::codecs::jpeg::JpegEncoder::new(&mut buffer)
        .encode_image(&img)
        .unwrap();

    // Decode with our function
    let decoded = decode_image(&buffer).await.unwrap();

    // Should be U8 variant
    match decoded {
        Image::U8(ref tensor) => {
            assert_eq!(tensor.shape, vec![2, 2, 3]);
            assert_eq!(decoded.height(), 2);
            assert_eq!(decoded.width(), 2);
            assert_eq!(decoded.channels(), 3);
        }
        _ => panic!("Expected U8 variant"),
    }
}

#[tokio::test]
async fn test_decode_png_rgba() {
    // Create a 2x2 RGBA PNG
    let mut buffer = Vec::new();
    let img = crates_image::RgbaImage::from_fn(2, 2, |x, y| {
        let val = (x + y) as u8 * 64;
        crates_image::Rgba([val, val + 16, val + 32, 255])
    });

    crates_image::codecs::png::PngEncoder::new(&mut buffer)
        .write_image(img.as_raw(), 2, 2, crates_image::ExtendedColorType::Rgba8)
        .unwrap();

    // Decode
    let decoded = decode_image(&buffer).await.unwrap();

    match decoded {
        Image::U8(ref tensor) => {
            assert_eq!(tensor.shape, vec![2, 2, 4]);
            assert_eq!(decoded.channels(), 4);
        }
        _ => panic!("Expected U8 variant"),
    }
}

#[tokio::test]
async fn test_decode_png_16bit() {
    // Create a 2x2 16-bit RGB PNG
    let mut buffer = Vec::new();
    let img =
        crates_image::ImageBuffer::<crates_image::Rgb<u16>, Vec<u16>>::from_fn(2, 2, |x, y| {
            let val = ((x + y) as u16) * 16384;
            crates_image::Rgb([val, val + 4096, val + 8192])
        });

    // Convert to raw bytes in native endian for PNG encoding
    let raw_bytes: Vec<u8> = img.as_raw().iter().flat_map(|&v| v.to_be_bytes()).collect();

    crates_image::codecs::png::PngEncoder::new(&mut buffer)
        .write_image(&raw_bytes, 2, 2, crates_image::ExtendedColorType::Rgb16)
        .unwrap();

    // Decode
    let decoded = decode_image(&buffer).await.unwrap();

    match decoded {
        Image::U16(ref tensor) => {
            assert_eq!(tensor.shape, vec![2, 2, 3]);
            assert_eq!(decoded.channels(), 3);
        }
        _ => panic!("Expected U16 variant"),
    }
}

#[tokio::test]
async fn test_decode_grayscale_png() {
    // Create a 2x2 grayscale PNG
    let mut buffer = Vec::new();
    let img =
        crates_image::GrayImage::from_fn(2, 2, |x, y| crates_image::Luma([(x + y) as u8 * 64]));

    crates_image::codecs::png::PngEncoder::new(&mut buffer)
        .write_image(img.as_raw(), 2, 2, crates_image::ExtendedColorType::L8)
        .unwrap();

    // Decode
    let decoded = decode_image(&buffer).await.unwrap();

    match decoded {
        Image::U8(ref tensor) => {
            assert_eq!(tensor.shape, vec![2, 2, 1]);
            assert_eq!(decoded.channels(), 1);
        }
        _ => panic!("Expected U8 variant"),
    }
}

#[tokio::test]
async fn test_decode_gray_alpha_png() {
    // Create a 2x2 grayscale+alpha PNG (2 channels)
    let mut buffer = Vec::new();
    let img = crates_image::GrayAlphaImage::from_fn(2, 2, |x, y| {
        crates_image::LumaA([(x + y) as u8 * 64, 255])
    });

    crates_image::codecs::png::PngEncoder::new(&mut buffer)
        .write_image(img.as_raw(), 2, 2, crates_image::ExtendedColorType::La8)
        .unwrap();

    let decoded = decode_image(&buffer).await.unwrap();

    match decoded {
        Image::U8(ref tensor) => {
            assert_eq!(tensor.shape, vec![2, 2, 2]);
            assert_eq!(decoded.channels(), 2);
        }
        _ => panic!("Expected U8 variant"),
    }
}

#[tokio::test]
async fn test_decode_invalid_data() {
    let result = decode_image(&[0xFF, 0x00, 0x12, 0x34]).await;
    assert!(result.is_err());

    match result.unwrap_err() {
        ImageError::Decode(_) => {}
        _ => panic!("Expected Decode error"),
    }
}

#[tokio::test]
async fn test_decode_empty_data() {
    let result = decode_image(&[]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_pixel_data_matches_reference() {
    // Create a simple 2x2 RGB image with known pixel values
    let mut buffer = Vec::new();
    let img = crates_image::RgbImage::from_raw(
        2,
        2,
        vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            128, 128, 128, // Gray
        ],
    )
    .unwrap();

    // Encode as PNG (lossless)
    crates_image::codecs::png::PngEncoder::new(&mut buffer)
        .write_image(img.as_raw(), 2, 2, crates_image::ExtendedColorType::Rgb8)
        .unwrap();

    // Decode with reference
    let reference_img = crates_image::load_from_memory(&buffer).unwrap();
    let reference_rgb = reference_img.to_rgb8();

    // Decode with our function
    let decoded = decode_image(&buffer).await.unwrap();

    match decoded {
        Image::U8(ref tensor) => {
            // Check each pixel matches
            for y in 0..2 {
                for x in 0..2 {
                    let pixel_ref = reference_rgb.get_pixel(x as u32, y as u32);
                    let idx = (y * 2 + x) * 3;
                    assert_eq!(
                        tensor.data[idx], pixel_ref[0],
                        "R mismatch at ({}, {})",
                        x, y
                    );
                    assert_eq!(
                        tensor.data[idx + 1],
                        pixel_ref[1],
                        "G mismatch at ({}, {})",
                        x,
                        y
                    );
                    assert_eq!(
                        tensor.data[idx + 2],
                        pixel_ref[2],
                        "B mismatch at ({}, {})",
                        x,
                        y
                    );
                }
            }
        }
        _ => panic!("Expected U8 variant"),
    }
}
