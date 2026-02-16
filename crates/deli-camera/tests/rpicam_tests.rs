#[cfg(feature = "rpicam")]
mod rpicam_tests {
    use deli_camera::{CameraConfig, CameraError, RPiCamera};

    #[test]
    fn test_rpicam_invalid_device() {
        // On a system without libcamera or no cameras, RPiCamera::new should fail
        let config = CameraConfig::default();
        let result = RPiCamera::new(config);

        // This will fail because libcamera is not available on the dev host,
        // or no cameras are detected. Either way, we expect a Device error.
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
