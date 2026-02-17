use deli_camera::{Camera, CameraError, Frame};
use deli_base::Tensor;

// Mock implementation for testing
struct MockCamera {
    frame_count: usize,
}

impl MockCamera {
    fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl Camera for MockCamera {
    async fn recv(&mut self) -> Result<Frame, CameraError> {
        self.frame_count += 1;
        // Return a dummy 2x2 RGB tensor
        let tensor = Tensor::new(vec![2, 2, 3], vec![0u8; 12])
            .map_err(|e| CameraError::Stream(e.to_string()))?;
        Ok(Frame::Rgb(tensor))
    }
}

#[tokio::test]
async fn test_camera_trait_mock_implementation() {
    let mut cam = MockCamera::new();

    // First frame
    let frame1 = cam.recv().await.unwrap();
    match &frame1 {
        Frame::Rgb(tensor) => assert_eq!(tensor.shape, vec![2, 2, 3]),
        Frame::Jpeg(_) => panic!("Expected Frame::Rgb"),
    }
    assert_eq!(cam.frame_count, 1);

    // Second frame
    let frame2 = cam.recv().await.unwrap();
    match &frame2 {
        Frame::Rgb(tensor) => assert_eq!(tensor.shape, vec![2, 2, 3]),
        Frame::Jpeg(_) => panic!("Expected Frame::Rgb"),
    }
    assert_eq!(cam.frame_count, 2);
}

#[tokio::test]
async fn test_camera_trait_polymorphism() {
    async fn capture_frames(camera: &mut impl Camera, count: usize) -> Result<Vec<Frame>, CameraError> {
        let mut frames = Vec::new();
        for _ in 0..count {
            frames.push(camera.recv().await?);
        }
        Ok(frames)
    }

    let mut cam = MockCamera::new();
    let frames = capture_frames(&mut cam, 3).await.unwrap();
    assert_eq!(frames.len(), 3);
    assert_eq!(cam.frame_count, 3);
}

#[tokio::test]
async fn test_frame_jpeg_variant() {
    struct JpegCamera;

    impl Camera for JpegCamera {
        async fn recv(&mut self) -> Result<Frame, CameraError> {
            Ok(Frame::Jpeg(vec![0xFF, 0xD8, 0xFF, 0xE0]))
        }
    }

    let mut cam = JpegCamera;
    let frame = cam.recv().await.unwrap();
    match frame {
        Frame::Jpeg(data) => {
            assert_eq!(data.len(), 4);
            assert_eq!(data[0], 0xFF); // JPEG SOI marker
        }
        Frame::Rgb(_) => panic!("Expected Frame::Jpeg"),
    }
}
