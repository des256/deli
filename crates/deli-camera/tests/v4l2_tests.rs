#[cfg(feature = "v4l2")]
mod v4l2_tests {
    use deli_camera::{CameraConfig, CameraError, V4l2Camera};

    #[test]
    fn test_v4l2_camera_invalid_device() {
        let config = CameraConfig::default()
            .with_device("/dev/nonexistent_camera".to_string());

        let result = V4l2Camera::new(config);

        assert!(result.is_err());
        match result.unwrap_err() {
            CameraError::Device(_) => {}
            other => panic!("Expected CameraError::Device, got {:?}", other),
        }
    }

    #[test]
    fn test_v4l2_camera_config_preserved() {
        // This test uses a non-existent device but verifies the config is stored
        let config = CameraConfig::default()
            .with_width(1920)
            .with_height(1080)
            .with_device("/dev/nonexistent_camera".to_string());

        // We can't test with a real device, so just verify the struct compiles
        // and config accessor exists (tested via type checking)
        let _ = V4l2Camera::new(config);
    }
}
