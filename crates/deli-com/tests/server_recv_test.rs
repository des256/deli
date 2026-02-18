use deli_com::Server;
use deli_codec::Codec;
use futures_util::StreamExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::time::{sleep, timeout, Duration};

#[tokio::test]
async fn test_server_recv_from_client() {
    // Bind server
    let mut server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect a raw TCP client
    let mut client = TcpStream::connect(addr).await.expect("connect failed");
    sleep(Duration::from_millis(50)).await; // Allow accept loop to process

    // Client sends a message using the framing protocol
    let value = 42u32;
    let payload = value.to_bytes();
    let len = payload.len() as u32;

    client.write_all(&len.to_le_bytes()).await.expect("write length failed");
    client.write_all(&payload).await.expect("write payload failed");

    // Server should receive it
    let received = timeout(Duration::from_secs(5), server.next())
        .await
        .expect("recv timed out")
        .expect("stream ended")
        .expect("recv failed");

    assert_eq!(received, 42u32);
}
