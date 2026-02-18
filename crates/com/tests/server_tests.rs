use com::Server;
use futures_util::SinkExt;
use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;
use tokio::time::{Duration, sleep, timeout};

#[tokio::test]
async fn test_sender_bind_creates_server() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    // Server should start with zero clients
    assert_eq!(server.client_count().await, 0);
}

#[tokio::test]
async fn test_accept_loop_adds_clients() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect a client
    let _client1 = TcpStream::connect(addr).await.expect("connect failed");
    sleep(Duration::from_millis(50)).await; // Allow accept loop to process

    assert_eq!(server.client_count().await, 1);

    // Connect another client
    let _client2 = TcpStream::connect(addr).await.expect("connect failed");
    sleep(Duration::from_millis(50)).await;

    assert_eq!(server.client_count().await, 2);
}

#[tokio::test]
async fn test_send_broadcasts_to_clients() {
    let mut server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect two clients
    let mut client1 = TcpStream::connect(addr).await.expect("connect failed");
    let mut client2 = TcpStream::connect(addr).await.expect("connect failed");
    sleep(Duration::from_millis(50)).await;

    // Broadcast a message
    server.send(42u32).await.expect("send failed");

    // Both clients should receive it
    let mut buf1 = vec![0u8; 8]; // 4-byte length + 4-byte u32
    timeout(Duration::from_secs(5), client1.read_exact(&mut buf1))
        .await
        .expect("read timed out")
        .expect("read failed");

    let mut buf2 = vec![0u8; 8];
    timeout(Duration::from_secs(5), client2.read_exact(&mut buf2))
        .await
        .expect("read timed out")
        .expect("read failed");

    // Verify length prefix and payload
    assert_eq!(&buf1[0..4], &[4, 0, 0, 0]); // length = 4 bytes
    assert_eq!(&buf1[4..8], &42u32.to_le_bytes()); // payload

    assert_eq!(buf2, buf1); // Both received same data
}

#[tokio::test]
async fn test_disconnected_client_removed_during_send() {
    let mut server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect two clients
    let _client1 = TcpStream::connect(addr).await.expect("connect failed");
    let client2 = TcpStream::connect(addr).await.expect("connect failed");
    sleep(Duration::from_millis(50)).await;

    assert_eq!(server.client_count().await, 2);

    // Disconnect client2
    drop(client2);
    sleep(Duration::from_millis(50)).await;

    // Send a message - this should trigger removal of client2
    server.send(99u32).await.expect("send failed");

    // Client count should drop to 1
    assert_eq!(server.client_count().await, 1);
}

#[tokio::test]
async fn test_send_continues_after_client_disconnect() {
    let mut server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect three clients
    let mut client1 = TcpStream::connect(addr).await.expect("connect failed");
    let client2 = TcpStream::connect(addr).await.expect("connect failed");
    let mut client3 = TcpStream::connect(addr).await.expect("connect failed");
    sleep(Duration::from_millis(50)).await;

    // Disconnect client2
    drop(client2);

    // Send should succeed and deliver to remaining clients
    server.send(123u32).await.expect("send failed");

    // client1 and client3 should receive the message
    let mut buf1 = vec![0u8; 8];
    timeout(Duration::from_secs(5), client1.read_exact(&mut buf1))
        .await
        .expect("read timed out")
        .expect("read failed");
    assert_eq!(&buf1[4..8], &123u32.to_le_bytes());

    let mut buf3 = vec![0u8; 8];
    timeout(Duration::from_secs(5), client3.read_exact(&mut buf3))
        .await
        .expect("read timed out")
        .expect("read failed");
    assert_eq!(&buf3[4..8], &123u32.to_le_bytes());
}
