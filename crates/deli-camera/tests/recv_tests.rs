use deli_base::Tensor;
use deli_camera::CameraError;
use deli_image::DecodedImage;

/// Test the MJPEG decode pipeline in isolation.
///
/// Creates a synthetic JPEG image, decodes it using deli_image,
/// and verifies the tensor shape matches expectations.
#[tokio::test]
async fn test_mjpeg_decode_pipeline() {
    let mut jpeg_buffer = Vec::new();
    let img = image::RgbImage::from_fn(16, 16, |x, y| {
        let val = ((x + y) % 256) as u8;
        image::Rgb([val, val + 10, val + 20])
    });

    image::codecs::jpeg::JpegEncoder::new(&mut jpeg_buffer)
        .encode_image(&img)
        .unwrap();

    let decoded = deli_image::decode_image(&jpeg_buffer).await.unwrap();

    let tensor: Tensor<u8> = match decoded {
        DecodedImage::U8(t) => t,
        _ => panic!("Expected U8 variant for JPEG"),
    };

    assert_eq!(tensor.shape, vec![16, 16, 3]);
    assert_eq!(tensor.data.len(), 16 * 16 * 3);
}

#[tokio::test]
async fn test_mjpeg_decode_grayscale() {
    let mut jpeg_buffer = Vec::new();
    let img = image::GrayImage::from_fn(8, 8, |x, y| {
        image::Luma([((x + y) % 256) as u8])
    });

    image::codecs::jpeg::JpegEncoder::new(&mut jpeg_buffer)
        .encode_image(&img)
        .unwrap();

    let decoded = deli_image::decode_image(&jpeg_buffer).await.unwrap();

    let tensor: Tensor<u8> = match decoded {
        DecodedImage::U8(t) => t,
        _ => panic!("Expected U8 variant"),
    };

    assert_eq!(tensor.shape, vec![8, 8, 1]);
}

#[tokio::test]
async fn test_corrupt_jpeg_produces_decode_error() {
    let corrupt_data = b"not a jpeg at all";
    let result = deli_image::decode_image(corrupt_data).await;
    assert!(result.is_err());

    // Verify it converts to CameraError::Decode
    let img_err = result.unwrap_err();
    let cam_err: CameraError = img_err.into();
    match cam_err {
        CameraError::Decode(_) => {}
        _ => panic!("Expected CameraError::Decode"),
    }
}
