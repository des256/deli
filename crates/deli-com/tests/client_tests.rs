use deli_com::{ComError, Client, Server};
use tokio::time::{sleep, timeout, Duration};

#[tokio::test]
async fn test_receiver_connect() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect a receiver
    let _receiver = Client::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;

    // Server should see the connection
    assert_eq!(server.client_count().await, 1);
}

#[tokio::test]
async fn test_receiver_recv_gets_broadcast() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    let mut receiver = Client::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;

    // Server broadcasts a message
    server.send(&42u32).await.expect("send failed");

    // Receiver should get it
    let value = timeout(Duration::from_secs(5), receiver.recv())
        .await
        .expect("recv timed out")
        .expect("recv failed");
    assert_eq!(value, 42);
}

#[tokio::test]
async fn test_receiver_multiple_messages() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    let mut receiver = Client::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;

    // Server sends three messages
    server.send(&10u32).await.unwrap();
    server.send(&20u32).await.unwrap();
    server.send(&30u32).await.unwrap();

    // Receiver gets all three in order
    assert_eq!(timeout(Duration::from_secs(5), receiver.recv()).await.unwrap().unwrap(), 10);
    assert_eq!(timeout(Duration::from_secs(5), receiver.recv()).await.unwrap().unwrap(), 20);
    assert_eq!(timeout(Duration::from_secs(5), receiver.recv()).await.unwrap().unwrap(), 30);
}

#[tokio::test]
async fn test_receiver_connection_closed() {
    // Create a local TCP listener and immediately close it after accepting one connection
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn a task that accepts one connection then closes the socket
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        // Immediately drop the stream (connection closed)
        drop(stream);
    });

    let mut receiver = Client::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;

    // recv() should return ConnectionClosed because server closed the connection
    let result = timeout(Duration::from_secs(5), receiver.recv())
        .await
        .expect("recv timed out");
    match result {
        Err(ComError::ConnectionClosed) => {} // Expected
        other => panic!("Expected ConnectionClosed, got {:?}", other),
    }
}
