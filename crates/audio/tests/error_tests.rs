use audio::AudioError;
use std::io;

#[test]
fn test_from_io_error() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "device not found");
    let audio_err: AudioError = io_err.into();

    match audio_err {
        AudioError::Device(msg) => assert!(msg.contains("device not found")),
        _ => panic!("Expected AudioError::Device variant"),
    }
}

#[test]
fn test_error_display() {
    let device_err = AudioError::Device("PulseAudio error".to_string());
    assert!(device_err.to_string().contains("PulseAudio error"));

    let stream_err = AudioError::Stream("streaming failed".to_string());
    assert!(stream_err.to_string().contains("streaming failed"));

    let channel_err = AudioError::Channel("channel closed".to_string());
    assert!(channel_err.to_string().contains("channel closed"));
}

#[test]
fn test_error_variants() {
    let device_err = AudioError::Device("test".to_string());
    match device_err {
        AudioError::Device(_) => {}
        _ => panic!("Expected Device variant"),
    }

    let stream_err = AudioError::Stream("test".to_string());
    match stream_err {
        AudioError::Stream(_) => {}
        _ => panic!("Expected Stream variant"),
    }

    let channel_err = AudioError::Channel("test".to_string());
    match channel_err {
        AudioError::Channel(_) => {}
        _ => panic!("Expected Channel variant"),
    }
}
