use deli_com::{Client, Server};
use tokio::time::{sleep, timeout, Duration};

#[tokio::test]
async fn test_single_sender_single_receiver() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    let mut receiver = Client::<u32>::connect(addr)
        .await
        .expect("connect failed");

    sleep(Duration::from_millis(50)).await;

    // Send a message
    server.send(&100u32).await.expect("send failed");

    // Receiver should get it
    let value = timeout(Duration::from_secs(5), receiver.recv())
        .await
        .expect("recv timed out")
        .expect("recv failed");
    assert_eq!(value, 100);
}

#[tokio::test]
async fn test_single_sender_multiple_receivers() {
    let server = Server::<String>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect three receivers
    let mut receiver1 = Client::<String>::connect(addr).await.unwrap();
    let mut receiver2 = Client::<String>::connect(addr).await.unwrap();
    let mut receiver3 = Client::<String>::connect(addr).await.unwrap();

    sleep(Duration::from_millis(50)).await;

    // Broadcast a message
    let msg = "Hello, MCSP!".to_string();
    server.send(&msg).await.expect("send failed");

    // All three receivers should get the same message
    assert_eq!(timeout(Duration::from_secs(5), receiver1.recv()).await.unwrap().unwrap(), msg);
    assert_eq!(timeout(Duration::from_secs(5), receiver2.recv()).await.unwrap().unwrap(), msg);
    assert_eq!(timeout(Duration::from_secs(5), receiver3.recv()).await.unwrap().unwrap(), msg);
}

#[tokio::test]
async fn test_receiver_disconnect_sender_continues() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect three receivers
    let mut receiver1 = Client::<u32>::connect(addr).await.unwrap();
    let receiver2 = Client::<u32>::connect(addr).await.unwrap();
    let mut receiver3 = Client::<u32>::connect(addr).await.unwrap();

    sleep(Duration::from_millis(50)).await;

    assert_eq!(server.client_count().await, 3);

    // Disconnect receiver2
    drop(receiver2);

    // Send a message - this should detect and remove receiver2
    server.send(&42u32).await.expect("send failed");

    // receiver1 and receiver3 should get the message
    assert_eq!(timeout(Duration::from_secs(5), receiver1.recv()).await.unwrap().unwrap(), 42);
    assert_eq!(timeout(Duration::from_secs(5), receiver3.recv()).await.unwrap().unwrap(), 42);

    // Client count should drop to 2
    assert_eq!(server.client_count().await, 2);

    // Send another message to verify server still works
    server.send(&99u32).await.expect("send failed");
    assert_eq!(timeout(Duration::from_secs(5), receiver1.recv()).await.unwrap().unwrap(), 99);
    assert_eq!(timeout(Duration::from_secs(5), receiver3.recv()).await.unwrap().unwrap(), 99);
}

#[tokio::test]
async fn test_multiple_messages_arrive_in_order() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    let mut receiver = Client::<u32>::connect(addr).await.unwrap();

    sleep(Duration::from_millis(50)).await;

    // Send 5 messages in order
    for i in 0..5 {
        server.send(&(i * 10)).await.expect("send failed");
    }

    // Receiver should get all 5 in the same order
    for i in 0..5 {
        assert_eq!(timeout(Duration::from_secs(5), receiver.recv()).await.unwrap().unwrap(), i * 10);
    }
}

#[tokio::test]
async fn test_stress_many_receivers() {
    let server = Server::<u32>::bind("127.0.0.1:0")
        .await
        .expect("bind failed");

    let addr = server.local_addr();

    // Connect 20 receivers
    let mut receivers = Vec::new();
    for _ in 0..20 {
        receivers.push(Client::<u32>::connect(addr).await.unwrap());
    }

    sleep(Duration::from_millis(100)).await;

    assert_eq!(server.client_count().await, 20);

    // Broadcast a message
    server.send(&999u32).await.expect("send failed");

    // All 20 receivers should get it
    for receiver in receivers.iter_mut() {
        assert_eq!(timeout(Duration::from_secs(5), receiver.recv()).await.unwrap().unwrap(), 999);
    }
}
