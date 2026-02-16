use deli_com::ws::WsServer;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_ws_server_bind_creates_server() {
    let server = WsServer::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    // Server should start with zero clients
    assert_eq!(server.client_count().await, 0);
}

#[tokio::test]
async fn test_ws_server_local_addr() {
    let server = WsServer::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();
    assert_eq!(addr.ip().to_string(), "127.0.0.1");
    assert!(addr.port() > 0);
}

#[tokio::test]
async fn test_ws_accept_loop_adds_clients() {
    use tokio_websockets::ClientBuilder;

    let server = WsServer::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect a WebSocket client
    let uri = format!("ws://{}", addr);
    let (_client1, _) = ClientBuilder::from_uri(uri.parse().unwrap())
        .connect()
        .await
        .expect("connect failed");
    sleep(Duration::from_millis(50)).await; // Allow accept loop to process

    assert_eq!(server.client_count().await, 1);

    // Connect another client
    let uri = format!("ws://{}", addr);
    let (_client2, _) = ClientBuilder::from_uri(uri.parse().unwrap())
        .connect()
        .await
        .expect("connect failed");
    sleep(Duration::from_millis(50)).await;

    assert_eq!(server.client_count().await, 2);
}

#[tokio::test]
async fn test_ws_send_broadcasts_to_clients() {
    use futures_util::StreamExt;
    use tokio_websockets::ClientBuilder;

    let server = WsServer::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect two WebSocket clients
    let uri = format!("ws://{}", addr);
    let (mut client1, _) = ClientBuilder::from_uri(uri.parse().unwrap())
        .connect()
        .await
        .expect("connect failed");
    let uri = format!("ws://{}", addr);
    let (mut client2, _) = ClientBuilder::from_uri(uri.parse().unwrap())
        .connect()
        .await
        .expect("connect failed");
    sleep(Duration::from_millis(50)).await;

    // Broadcast a message
    server.send(&42u32).await.expect("send failed");

    // Both clients should receive it
    let msg1 = client1.next().await.unwrap().unwrap();
    assert!(msg1.is_binary());
    let payload1 = msg1.into_payload();
    let value1 = u32::from_le_bytes([payload1[0], payload1[1], payload1[2], payload1[3]]);
    assert_eq!(value1, 42);

    let msg2 = client2.next().await.unwrap().unwrap();
    assert!(msg2.is_binary());
    let payload2 = msg2.into_payload();
    let value2 = u32::from_le_bytes([payload2[0], payload2[1], payload2[2], payload2[3]]);
    assert_eq!(value2, 42);
}

#[tokio::test]
async fn test_ws_disconnected_client_removed_during_send() {
    use tokio_websockets::ClientBuilder;

    let server = WsServer::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect two clients
    let uri = format!("ws://{}", addr);
    let (_client1, _) = ClientBuilder::from_uri(uri.parse().unwrap())
        .connect()
        .await
        .expect("connect failed");
    let uri = format!("ws://{}", addr);
    let client2 = ClientBuilder::from_uri(uri.parse().unwrap())
        .connect()
        .await
        .expect("connect failed")
        .0;
    sleep(Duration::from_millis(50)).await;

    assert_eq!(server.client_count().await, 2);

    // Disconnect client2
    drop(client2);
    sleep(Duration::from_millis(50)).await;

    // Send a message - this should trigger removal of client2
    server.send(&99u32).await.expect("send failed");

    // Client count should drop to 1
    sleep(Duration::from_millis(50)).await;
    assert_eq!(server.client_count().await, 1);
}
