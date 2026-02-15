use deli_camera::CameraError;
use deli_image::ImageError;
use std::io;

#[test]
fn test_from_io_error() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "device not found");
    let cam_err: CameraError = io_err.into();

    match cam_err {
        CameraError::Device(msg) => assert!(msg.contains("device not found")),
        _ => panic!("Expected CameraError::Device variant"),
    }
}

#[test]
fn test_from_image_error() {
    let img_err = ImageError::Decode("invalid JPEG".to_string());
    let cam_err: CameraError = img_err.into();

    match cam_err {
        CameraError::Decode(_) => {}
        _ => panic!("Expected CameraError::Decode variant"),
    }
}

#[test]
fn test_error_display() {
    let device_err = CameraError::Device("V4L2 error".to_string());
    assert!(device_err.to_string().contains("V4L2 error"));

    let stream_err = CameraError::Stream("streaming failed".to_string());
    assert!(stream_err.to_string().contains("streaming failed"));

    let channel_err = CameraError::Channel("channel closed".to_string());
    assert!(channel_err.to_string().contains("channel closed"));
}
