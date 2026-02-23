use {base::Vec2, video::VideoIn};

#[tokio::test]
async fn test_videoin_open() {
    let mut videoin = VideoIn::open(None).await.unwrap();
    // capture a frame to ensure camera is open and config is negotiated
    let frame = videoin.capture().await.unwrap();
    assert_ne!(frame.color.size, Vec2::new(0, 0));
    assert_ne!(videoin.size(), Vec2::new(0, 0));
}

#[tokio::test]
async fn test_videoin_drop() {
    let videoin = VideoIn::open(None).await;
    drop(videoin);
}
