use deli_video::CameraConfig;

#[test]
fn test_config_defaults() {
    let config = CameraConfig::default();

    assert_eq!(config.device(), "/dev/video0");
    assert_eq!(config.width(), 640);
    assert_eq!(config.height(), 480);
    assert_eq!(config.fps(), 30);
    assert_eq!(config.buffer_count(), 4);
}

#[test]
fn test_config_builder() {
    let config = CameraConfig::default()
        .with_device("/dev/video1".to_string())
        .with_width(1920)
        .with_height(1080)
        .with_fps(60)
        .with_buffer_count(8);

    assert_eq!(config.device(), "/dev/video1");
    assert_eq!(config.width(), 1920);
    assert_eq!(config.height(), 1080);
    assert_eq!(config.fps(), 60);
    assert_eq!(config.buffer_count(), 8);
}

#[test]
fn test_config_partial_builder() {
    let config = CameraConfig::default()
        .with_width(1280)
        .with_height(720);

    assert_eq!(config.device(), "/dev/video0"); // default
    assert_eq!(config.width(), 1280);
    assert_eq!(config.height(), 720);
    assert_eq!(config.fps(), 30); // default
}
