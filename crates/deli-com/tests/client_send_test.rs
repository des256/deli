use deli_com::{Client, Server};
use tokio::time::{sleep, timeout, Duration};

#[tokio::test]
async fn test_client_send_to_server() {
    // Bind server
    let mut server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect client
    let mut client = Client::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await; // Allow accept loop to process

    // Client sends a message
    client.send(&42u32).await.expect("send failed");

    // Server should receive it
    let received = timeout(Duration::from_secs(5), server.recv())
        .await
        .expect("recv timed out")
        .expect("recv failed");

    assert_eq!(received, 42u32);
}
