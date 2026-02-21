use base::Vec2;
use image::{Image, PixelFormat};

#[test]
fn test_rgb_to_jpeg_roundtrip() {
    let size = Vec2::new(4, 4);
    let data: Vec<u8> = (0..48).collect(); // 4x4x3

    let jpeg = image::rgb_to_jpeg(size, &data, 90);

    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]); // JPEG SOI marker
}

#[test]
fn test_rgb_to_jpeg_quality_affects_size() {
    let size = Vec2::new(8, 8);
    let data: Vec<u8> = (0..192).map(|v| (v % 256) as u8).collect(); // 8x8x3

    let low_quality = image::rgb_to_jpeg(size, &data, 10);
    let high_quality = image::rgb_to_jpeg(size, &data, 100);

    assert!(low_quality.len() < high_quality.len());
}

#[test]
fn test_jpeg_to_rgb_decode() {
    let size = Vec2::new(4, 4);
    let data: Vec<u8> = (0..48).collect();

    let jpeg = image::rgb_to_jpeg(size, &data, 90);
    let jpeg_image = Image::new(size, jpeg, PixelFormat::Jpeg);
    let decoded_image = image::jpeg_to_rgb(&jpeg_image).unwrap();

    assert!(matches!(decoded_image.format, PixelFormat::Rgb8));
    assert_eq!(decoded_image.size, size);
    assert_eq!(decoded_image.data.len(), 48);
}

#[test]
fn test_jpeg_to_rgb_rejects_invalid_data() {
    let size = Vec2::new(4, 4);
    let data = vec![0u8; 10];
    let invalid_jpeg = Image::new(size, data, PixelFormat::Jpeg);

    assert!(image::jpeg_to_rgb(&invalid_jpeg).is_err());
}

#[test]
fn test_yuyv_to_rgb() {
    let size = Vec2::new(2, 1);
    let data = vec![128u8, 128, 128, 128];

    let rgb = image::yuyv_to_rgb(size, &data);

    assert_eq!(rgb.len(), 6);
    for &v in &rgb {
        assert!((126..=130).contains(&v));
    }
}

#[test]
fn test_yuyv_to_argb() {
    let size = Vec2::new(2, 1);
    let data = vec![128u8, 128, 128, 128];

    let argb = image::yuyv_to_argb(size, &data);

    assert_eq!(argb.len(), 8);
    assert_eq!(argb[0], 0xFF);
    assert_eq!(argb[4], 0xFF);
}

#[test]
fn test_rgb_to_argb() {
    let size = Vec2::new(2, 1);
    let data = vec![255, 0, 0, 0, 255, 0]; // red, green

    let argb = image::rgb_to_argb(size, &data);

    assert_eq!(argb.len(), 8);
    assert_eq!(&argb[0..4], &[0xFF, 255, 0, 0]);
    assert_eq!(&argb[4..8], &[0xFF, 0, 255, 0]);
}

#[test]
fn test_rgb_to_u32() {
    let size = Vec2::new(2, 1);
    let data = vec![255, 0, 0, 0, 255, 0]; // red, green

    let buf = image::rgb_to_u32(size, &data);

    assert_eq!(buf.len(), 2);
    assert_eq!(buf[0], 0xFFFF0000);
    assert_eq!(buf[1], 0xFF00FF00);
}

#[test]
fn test_argb_to_u32() {
    let size = Vec2::new(2, 1);
    let data = vec![0xFF, 255, 0, 0, 0xFF, 0, 255, 0];

    let buf = image::argb_to_u32(size, &data);

    assert_eq!(buf.len(), 2);
    assert_eq!(buf[0], 0xFFFF0000);
    assert_eq!(buf[1], 0xFF00FF00);
}

#[test]
fn test_yuyv_to_u32() {
    let size = Vec2::new(2, 1);
    let data = vec![128u8, 128, 128, 128];

    let buf = image::yuyv_to_u32(size, &data);

    assert_eq!(buf.len(), 2);
    // neutral YUV (128,128,128) should produce near-grey pixels
    for &pixel in &buf {
        let r = (pixel >> 16) & 0xFF;
        let g = (pixel >> 8) & 0xFF;
        let b = pixel & 0xFF;
        assert!((126..=130).contains(&r));
        assert!((126..=130).contains(&g));
        assert!((126..=130).contains(&b));
    }
}

#[test]
fn test_yu12_to_u32() {
    let size = Vec2::new(2, 2);
    let mut data = vec![128u8; 4]; // Y plane
    data.push(128); // U plane
    data.push(128); // V plane

    let buf = image::yu12_to_u32(size, &data);

    assert_eq!(buf.len(), 4);
    for &pixel in &buf {
        let r = (pixel >> 16) & 0xFF;
        let g = (pixel >> 8) & 0xFF;
        let b = pixel & 0xFF;
        assert!((126..=130).contains(&r));
        assert!((126..=130).contains(&g));
        assert!((126..=130).contains(&b));
    }
}

#[test]
fn test_srggb10p_to_u32() {
    let size = Vec2::new(4, 2);
    let data = vec![128u8; 10];

    let buf = image::srggb10p_to_u32(size, &data);

    assert_eq!(buf.len(), 4 * 2);
}

#[test]
fn test_jpeg_to_u32() {
    let size = Vec2::new(4, 4);
    let rgb_data: Vec<u8> = (0..48).collect();
    let jpeg = image::rgb_to_jpeg(size, &rgb_data, 90);

    let buf = image::jpeg_to_u32(&jpeg);

    assert_eq!(buf.len(), 16); // 4x4
}

#[test]
fn test_yu12_to_rgb() {
    let size = Vec2::new(2, 2);
    let mut data = vec![128u8; 4]; // Y plane
    data.push(128); // U plane
    data.push(128); // V plane

    let rgb = image::yu12_to_rgb(size, &data);

    assert_eq!(rgb.len(), 12);
    for &v in &rgb {
        assert!((126..=130).contains(&v));
    }
}

#[test]
fn test_yuyv_to_jpeg() {
    let size = Vec2::new(4, 2);
    let data = vec![128u8; 16];

    let jpeg = image::yuyv_to_jpeg(size, &data, 80);

    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
}

#[test]
fn test_yu12_to_jpeg() {
    let size = Vec2::new(4, 4);
    let data = vec![128u8; 24];

    let jpeg = image::yu12_to_jpeg(size, &data, 80);

    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
}

#[test]
fn test_argb_to_jpeg() {
    let size = Vec2::new(2, 1);
    let data = vec![0xFF, 255, 0, 0, 0xFF, 0, 255, 0]; // red, green

    let jpeg = image::argb_to_jpeg(size, &data, 80);

    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
}

#[test]
fn test_srggb10p_to_rgb() {
    let size = Vec2::new(4, 2);
    let data = vec![128u8; 10];

    let rgb = image::srggb10p_to_rgb(size, &data);

    assert_eq!(rgb.len(), 4 * 2 * 3);
}

#[test]
fn test_srggb10p_to_rgb_small_image_returns_zeroed() {
    let size = Vec2::new(1, 1);
    let data = vec![0u8; 2];

    let rgb = image::srggb10p_to_rgb(size, &data);

    assert_eq!(rgb.len(), 3);
    assert!(rgb.iter().all(|&v| v == 0));
}
