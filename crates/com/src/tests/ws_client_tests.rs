use {com::{WsServer, ws::WsClient}, tokio::time::{Duration, sleep, timeout}};

#[tokio::test]
async fn test_ws_client_connect() {
    let server = WsServer::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    let _client = WsClient::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;
    assert_eq!(server.client_count().await, 1);
}

#[tokio::test]
async fn test_ws_client_send() {
    let mut server = WsServer::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    let mut client = WsClient::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;

    // Client sends a message
    client.send(&42u32).await.expect("send failed");

    // Server should receive it
    let value = timeout(Duration::from_secs(5), server.recv())
        .await
        .expect("recv timed out")
        .expect("recv failed");
    assert_eq!(value, 42);
}

#[tokio::test]
async fn test_ws_client_recv() {
    let server = WsServer::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    let mut client = WsClient::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;

    // Server broadcasts a message
    server.send(&99u32).await.expect("send failed");

    // Client should receive it
    let value = timeout(Duration::from_secs(5), client.recv())
        .await
        .expect("recv timed out")
        .expect("recv failed");
    assert_eq!(value, 99);
}

#[tokio::test]
async fn test_ws_client_send_recv_string() {
    let server = WsServer::<String>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    let mut client = WsClient::<String>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;

    // Server broadcasts
    server
        .send(&"Hello".to_string())
        .await
        .expect("send failed");

    // Client receives
    let value = timeout(Duration::from_secs(5), client.recv())
        .await
        .expect("recv timed out")
        .expect("recv failed");
    assert_eq!(value, "Hello");
}
