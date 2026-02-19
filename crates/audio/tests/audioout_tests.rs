use audio::{AudioData, AudioOut, AudioOutConfig, AudioSample};
use base::Tensor;

#[tokio::test]
async fn test_audioout_open() {
    let audioout = AudioOut::open().await;
    let config = audioout.config();
    assert_eq!(config.device_name, None);
    assert_eq!(config.sample_rate, 16000);
}

#[tokio::test]
async fn test_audioout_select() {
    let devices = AudioOut::list_devices().await.unwrap();
    let config = AudioOutConfig {
        device_name: Some(devices[0].name.clone()),
        ..Default::default()
    };
    let mut audioout = AudioOut::open().await;
    audioout.select(config).await;
    let config = audioout.config();
    assert_eq!(config.device_name, Some(devices[0].name.clone()));
    assert_eq!(config.sample_rate, 16000);
}

#[tokio::test]
async fn test_audioout_drop() {
    let audioout = AudioOut::open().await;
    drop(audioout);
}

#[tokio::test]
async fn test_audioout_play() {
    let audioout = AudioOut::open().await;
    let sample = AudioSample {
        data: AudioData::Pcm(
            Tensor::<i16>::new(vec![5], vec![32767, -32768, 32767, -32768, 0]).unwrap(),
        ),
        sample_rate: 16000,
    };
    // this sounds like an annoying pop from the speakers
    audioout.play(sample).await;
}
