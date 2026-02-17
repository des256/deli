use deli_audio::AudioOut;

#[tokio::test]
async fn test_audio_out_construction() {
    let audio_out = AudioOut::new(None, 48000);
    assert_eq!(audio_out.sample_rate(), 48000);
    assert_eq!(audio_out.device(), None);

    let audio_out_with_device = AudioOut::new(Some("test_device"), 44100);
    assert_eq!(audio_out_with_device.sample_rate(), 44100);
    assert_eq!(audio_out_with_device.device(), Some("test_device"));
}

#[tokio::test]
async fn test_audio_out_send_signature() {
    // Verify send() is async and returns Result<(), AudioError>
    fn assert_send_type(_: impl std::future::Future<Output = Result<(), deli_audio::AudioError>>) {}

    let mut audio_out = AudioOut::new(None, 48000);
    let data = vec![0i16; 100];
    assert_send_type(audio_out.send(&data));
}

#[tokio::test]
async fn test_audio_out_debug() {
    let audio_out = AudioOut::new(None, 48000);
    let debug_str = format!("{:?}", audio_out);
    assert!(debug_str.contains("AudioOut"));
    assert!(debug_str.contains("sample_rate: 48000"));
    assert!(debug_str.contains("sender: true"));
    assert!(debug_str.contains("task_handle: true"));
}

#[tokio::test]
async fn test_audio_out_drop() {
    // Verify Drop doesn't panic
    let audio_out = AudioOut::new(None, 48000);
    drop(audio_out);
    // If we reach here, drop succeeded without panicking
}

#[tokio::test]
async fn test_audio_out_state_management() {
    let audio_out = AudioOut::new(Some("device1"), 48000);

    // Verify initial state
    assert_eq!(audio_out.device(), Some("device1"));
    assert_eq!(audio_out.sample_rate(), 48000);

    // Verify the struct stores the device name correctly
    assert!(audio_out.device().is_some());
}

#[tokio::test]
async fn test_select_updates_device() {
    let mut audio_out = AudioOut::new(Some("device1"), 48000);

    // Verify initial device
    assert_eq!(audio_out.device(), Some("device1"));

    // Select new device
    audio_out.select("device2").await;

    // Verify device updated
    assert_eq!(audio_out.device(), Some("device2"));
}

#[tokio::test]
async fn test_cancel_preserves_device() {
    let mut audio_out = AudioOut::new(Some("device1"), 48000);

    // Verify initial device
    assert_eq!(audio_out.device(), Some("device1"));

    // Cancel playback
    audio_out.cancel().await;

    // Verify device unchanged
    assert_eq!(audio_out.device(), Some("device1"));
}
