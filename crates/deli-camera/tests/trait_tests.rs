use deli_camera::{Camera, CameraError};
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
    async fn recv(&mut self) -> Result<Tensor<u8>, CameraError> {
        self.frame_count += 1;
        // Return a dummy 2x2 RGB tensor
        Tensor::new(vec![2, 2, 3], vec![0u8; 12])
            .map_err(|e| CameraError::Stream(e.to_string()))
    }
}

#[tokio::test]
async fn test_camera_trait_mock_implementation() {
    let mut cam = MockCamera::new();

    // First frame
    let frame1 = cam.recv().await.unwrap();
    assert_eq!(frame1.shape, vec![2, 2, 3]);
    assert_eq!(cam.frame_count, 1);

    // Second frame
    let frame2 = cam.recv().await.unwrap();
    assert_eq!(frame2.shape, vec![2, 2, 3]);
    assert_eq!(cam.frame_count, 2);
}

#[tokio::test]
async fn test_camera_trait_polymorphism() {
    async fn capture_frames(camera: &mut impl Camera, count: usize) -> Result<Vec<Tensor<u8>>, CameraError> {
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
