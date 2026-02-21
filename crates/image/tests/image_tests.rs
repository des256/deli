use base::Vec2;
use image::{Image, ImageError, PixelFormat};

#[test]
fn test_image_new_rgb8() {
    let size = Vec2::new(2, 3);
    let data: Vec<u8> = (0..18).collect(); // 2*3*3 = 18 bytes
    let image = Image::new(size, data.clone(), PixelFormat::Rgb8);

    assert_eq!(image.size, size);
    assert_eq!(image.data, data);
    assert!(matches!(image.format, PixelFormat::Rgb8));
}

#[test]
fn test_image_new_argb8() {
    let size = Vec2::new(2, 2);
    let data: Vec<u8> = vec![0xFF; 16]; // 2*2*4 = 16 bytes
    let image = Image::new(size, data.clone(), PixelFormat::Argb8);

    assert_eq!(image.size.x, 2);
    assert_eq!(image.size.y, 2);
    assert_eq!(image.data.len(), 16);
    assert!(matches!(image.format, PixelFormat::Argb8));
}

#[test]
fn test_image_new_yuyv() {
    let size = Vec2::new(4, 2);
    let data: Vec<u8> = vec![128; 16]; // 4*2*2 = 16 bytes
    let image = Image::new(size, data, PixelFormat::Yuyv);

    assert_eq!(image.size.x, 4);
    assert_eq!(image.size.y, 2);
    assert!(matches!(image.format, PixelFormat::Yuyv));
}

#[test]
fn test_image_new_yu12() {
    let size = Vec2::new(4, 4);
    let data: Vec<u8> = vec![128; 24]; // 4*4 + 2*2 + 2*2 = 24 bytes
    let image = Image::new(size, data, PixelFormat::Yu12);

    assert!(matches!(image.format, PixelFormat::Yu12));
    assert_eq!(image.data.len(), 24);
}

#[test]
fn test_image_new_srggb10p() {
    let size = Vec2::new(4, 2);
    let data: Vec<u8> = vec![0; 10]; // Variable packed format
    let image = Image::new(size, data, PixelFormat::Srggb10p);

    assert!(matches!(image.format, PixelFormat::Srggb10p));
}

#[test]
fn test_image_new_jpeg() {
    let size = Vec2::new(10, 10);
    let data: Vec<u8> = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG SOI marker
    let image = Image::new(size, data.clone(), PixelFormat::Jpeg);

    assert_eq!(image.format, PixelFormat::Jpeg);
    assert_eq!(image.data, data);
}

#[test]
fn test_image_clone() {
    let size = Vec2::new(2, 2);
    let data: Vec<u8> = (0..12).collect();
    let image = Image::new(size, data.clone(), PixelFormat::Rgb8);
    let cloned = image.clone();

    assert_eq!(cloned.size, image.size);
    assert_eq!(cloned.data, image.data);
    assert_eq!(cloned.format, image.format);
}

#[test]
fn test_pixel_format_copy() {
    let format1 = PixelFormat::Yuyv;
    let format2 = format1; // Copy trait

    assert!(matches!(format1, PixelFormat::Yuyv));
    assert!(matches!(format2, PixelFormat::Yuyv));
}

#[test]
fn test_image_error_from_image_error() {
    let img_err = crates_image::ImageError::Unsupported(
        crates_image::error::UnsupportedError::from_format_and_kind(
            crates_image::error::ImageFormatHint::Unknown,
            crates_image::error::UnsupportedErrorKind::Format(
                crates_image::error::ImageFormatHint::Unknown,
            ),
        ),
    );

    let err: ImageError = img_err.into();
    let err_str = format!("{}", err);
    assert!(err_str.contains("decode error"));
}

#[test]
fn test_image_error_display() {
    let err = ImageError::Decode("test error".to_string());
    assert_eq!(format!("{}", err), "decode error: test error");

    let err = ImageError::Encode("encode failed".to_string());
    assert_eq!(format!("{}", err), "encode error: encode failed");
}
