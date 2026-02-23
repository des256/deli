use audio::{AudioData, AudioIn, AudioInConfig};

#[tokio::test]
async fn test_audioin_open() {
    let audioin = AudioIn::open(None).await;
    let config = audioin.config();
    assert_eq!(config.device_name, None);
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.chunk_size, 1600);
}

#[tokio::test]
async fn test_audioin_select() {
    let mut audioin = AudioIn::open(None).await;
    let devices = AudioIn::list_devices().await.unwrap();
    let config = AudioInConfig {
        device_name: Some(devices[0].name.clone()),
        ..Default::default()
    };
    audioin.select(config).await;
    let config = audioin.config();
    assert_eq!(config.device_name, Some(devices[0].name.clone()));
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.chunk_size, 1600);
}

#[tokio::test]
async fn test_audioin_drop() {
    let audioin = AudioIn::open(None).await;
    drop(audioin);
}

#[tokio::test]
async fn test_audioin_capture() {
    let mut audioin = AudioIn::open(None).await;
    let sample = audioin.capture().await.unwrap();
    let length = match sample.data {
        AudioData::Pcm(tensor) => tensor.shape[0],
    };
    assert_ne!(length, 0);
}
