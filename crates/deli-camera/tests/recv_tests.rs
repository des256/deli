#[cfg(feature = "v4l2")]
mod recv_tests {
    use deli_base::Tensor;
    use deli_image::DecodedImage;
    use image::ImageEncoder;

    /// Test the MJPEG decode pipeline in isolation.
    ///
    /// Creates a synthetic JPEG image, decodes it using deli_image,
    /// and verifies the tensor shape matches expectations.
    #[test]
    fn test_mjpeg_decode_pipeline() {
        // Create a small 16x16 RGB JPEG image
        let mut jpeg_buffer = Vec::new();
        let img = image::RgbImage::from_fn(16, 16, |x, y| {
            let val = ((x + y) % 256) as u8;
            image::Rgb([val, val + 10, val + 20])
        });

        image::codecs::jpeg::JpegEncoder::new(&mut jpeg_buffer)
            .encode_image(&img)
            .unwrap();

        // Decode using deli_image (same path used in V4l2Camera)
        let decoded = deli_image::decode_image(&jpeg_buffer).unwrap();

        // Extract tensor from DecodedImage::U8 variant
        let tensor: Tensor<u8> = match decoded {
            DecodedImage::U8(t) => t,
            _ => panic!("Expected U8 variant for JPEG"),
        };

        // Verify HWC layout: [height, width, channels]
        assert_eq!(tensor.shape, vec![16, 16, 3]);
        assert_eq!(tensor.data.len(), 16 * 16 * 3);
    }

    #[test]
    fn test_mjpeg_decode_grayscale() {
        // Test that grayscale JPEG decodes to [h, w, 1] tensor
        let mut jpeg_buffer = Vec::new();
        let img = image::GrayImage::from_fn(8, 8, |x, y| {
            image::Luma([((x + y) % 256) as u8])
        });

        image::codecs::jpeg::JpegEncoder::new(&mut jpeg_buffer)
            .encode_image(&img)
            .unwrap();

        let decoded = deli_image::decode_image(&jpeg_buffer).unwrap();

        let tensor: Tensor<u8> = match decoded {
            DecodedImage::U8(t) => t,
            _ => panic!("Expected U8 variant"),
        };

        assert_eq!(tensor.shape, vec![8, 8, 1]);
    }
}
