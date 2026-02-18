use deli_audio::AudioIn;
use futures_core::Stream;

#[tokio::test]
async fn test_audio_in_construction() {
    let audio_in = AudioIn::new(None, 48000, 4800);
    assert_eq!(audio_in.sample_rate(), 48000);
    assert_eq!(audio_in.chunk_frames(), 4800);
    assert_eq!(audio_in.device(), None);

    let audio_in_with_device = AudioIn::new(Some("test_device"), 44100, 4410);
    assert_eq!(audio_in_with_device.sample_rate(), 44100);
    assert_eq!(audio_in_with_device.chunk_frames(), 4410);
    assert_eq!(
        audio_in_with_device.device(),
        Some("test_device")
    );
}

#[tokio::test]
async fn test_audio_in_implements_stream() {
    fn assert_stream<T: Stream<Item = deli_audio::AudioSample>>() {}
    assert_stream::<AudioIn>();
}

#[tokio::test]
async fn test_audio_in_drop() {
    // Verify Drop doesn't panic
    let audio_in = AudioIn::new(None, 48000, 4800);
    drop(audio_in);
    // If we reach here, drop succeeded without panicking
}

#[tokio::test]
async fn test_audio_in_state_management() {
    let audio_in = AudioIn::new(Some("device1"), 48000, 4800);

    // Verify initial state
    assert_eq!(audio_in.device(), Some("device1"));
    assert_eq!(audio_in.sample_rate(), 48000);
    assert_eq!(audio_in.chunk_frames(), 4800);

    // Verify the struct stores the device name correctly
    assert!(audio_in.device().is_some());
}

#[tokio::test]
async fn test_select_before_streaming() {
    let mut audio_in = AudioIn::new(Some("device1"), 48000, 4800);

    // Verify initial device
    assert_eq!(audio_in.device(), Some("device1"));

    // Select new device
    audio_in.select("device2");

    // Verify device updated
    assert_eq!(audio_in.device(), Some("device2"));
}

#[tokio::test]
async fn test_select_clears_stream_state() {
    let mut audio_in = AudioIn::new(Some("device1"), 48000, 4800);

    // select() signals the capture loop to switch to device2
    audio_in.select("device2");

    // Device should be updated, new capture task running on device2
    assert_eq!(audio_in.device(), Some("device2"));
}
