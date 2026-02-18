#[cfg(feature = "v4l2")]
mod v4l2_tests {
    use deli_video::{CameraConfig, CameraError, V4l2Camera};

    #[test]
    fn test_v4l2_camera_invalid_device() {
        let config = CameraConfig::default()
            .with_device("/dev/nonexistent_camera".to_string());

        let result = V4l2Camera::new(config);

        assert!(result.is_err());
        match result.unwrap_err() {
            CameraError::Device(msg) => {
                assert!(
                    !msg.is_empty(),
                    "Error message should describe the failure"
                );
            }
            other => panic!("Expected CameraError::Device, got {:?}", other),
        }
    }
}
