use {
    crate::ComError,
    base::*,
    codec::Codec,
    futures_util::{SinkExt, StreamExt},
    std::{collections::HashMap, net::SocketAddr, sync::Arc},
    tokio::{
        net::{TcpListener, ToSocketAddrs},
        sync::{RwLock, mpsc},
        task::JoinHandle,
    },
    tokio_websockets::{Message, ServerBuilder, WebSocketStream},
};

type WsSink = futures_util::stream::SplitSink<WebSocketStream<tokio::net::TcpStream>, Message>;

pub struct WsServer<T> {
    clients: Arc<RwLock<HashMap<SocketAddr, WsSink>>>,
    rx: mpsc::Receiver<T>,
    _accept_task: JoinHandle<()>,
    local_addr: SocketAddr,
}

impl<T: Codec + Send + 'static> WsServer<T> {
    /// Bind a TCP listener and start accepting WebSocket connections.
    ///
    /// Returns a `WsServer` with a background task that accepts new connections,
    /// performs WebSocket handshake, spawns per-client reader tasks, and adds write halves to the client map.
    pub async fn bind(addr: impl ToSocketAddrs) -> Result<Self, ComError> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;

        let clients: Arc<RwLock<HashMap<SocketAddr, WsSink>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let clients_clone = clients.clone();

        let (tx, rx) = mpsc::channel(256);

        // Spawn accept loop
        let accept_task = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((tcp_stream, addr)) => {
                        // Perform WebSocket handshake
                        let ws_stream = match ServerBuilder::new().accept(tcp_stream).await {
                            Ok((_request, ws_stream)) => ws_stream,
                            Err(e) => {
                                log_warn!("WebSocket handshake failed for {}: {}", addr, e);
                                continue;
                            }
                        };

                        // Split the WebSocket stream for concurrent read/write
                        let (write_half, read_half) = ws_stream.split();
                        clients_clone.write().await.insert(addr, write_half);

                        // Spawn a reader task for this client
                        let tx = tx.clone();
                        let clients_for_cleanup = clients_clone.clone();
                        tokio::spawn(async move {
                            let mut reader = read_half;
                            loop {
                                match reader.next().await {
                                    Some(Ok(msg)) => {
                                        // Only process binary messages
                                        if msg.is_binary() {
                                            let payload = msg.into_payload();
                                            if payload.len()
                                                > crate::framing::MAX_MESSAGE_SIZE as usize
                                            {
                                                log_warn!(
                                                    "Message from {} too large: {} bytes",
                                                    addr,
                                                    payload.len()
                                                );
                                                continue;
                                            }
                                            match T::from_bytes(&payload) {
                                                Ok(value) => {
                                                    if tx.send(value).await.is_err() {
                                                        break; // Server dropped
                                                    }
                                                }
                                                Err(e) => {
                                                    log_warn!(
                                                        "Failed to decode message from {}: {}",
                                                        addr,
                                                        e
                                                    );
                                                }
                                            }
                                        }
                                        // Ignore text messages and control frames
                                    }
                                    Some(Err(e)) => {
                                        log_warn!("Client {} error: {}", addr, e);
                                        clients_for_cleanup.write().await.remove(&addr);
                                        break;
                                    }
                                    None => {
                                        log_warn!("Client {} disconnected", addr);
                                        clients_for_cleanup.write().await.remove(&addr);
                                        break;
                                    }
                                }
                            }
                        });
                    }
                    Err(e) => {
                        log_warn!("Accept error: {}", e);
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                }
            }
        });

        Ok(Self {
            clients,
            rx,
            _accept_task: accept_task,
            local_addr,
        })
    }

    /// Broadcast a message to all connected clients.
    ///
    /// Clients that fail to receive (disconnected, error) are removed from the
    /// client map and logged as warnings. Returns `Ok(())` regardless of individual
    /// client failures.
    pub async fn send(&self, value: &T) -> Result<(), ComError> {
        let payload = value.to_bytes();
        let msg = Message::binary(payload);

        let mut lock = self.clients.write().await;

        let mut failed_addrs = Vec::new();

        for (addr, writer) in lock.iter_mut() {
            if let Err(e) = writer.send(msg.clone()).await {
                log_warn!("Failed to send to {}: {}", addr, e);
                failed_addrs.push(*addr);
            }
        }

        // Remove failed clients
        for addr in failed_addrs {
            lock.remove(&addr);
        }

        Ok(())
    }

    /// Receive a message from any connected client.
    ///
    /// Messages from all clients are multiplexed through an internal channel.
    /// Each client has a dedicated reader task, so no single client blocks others.
    ///
    /// Returns `ComError::ConnectionClosed` if all clients have disconnected
    /// and the channel is empty.
    pub async fn recv(&mut self) -> Result<T, ComError> {
        self.rx.recv().await.ok_or(ComError::ConnectionClosed)
    }

    /// Return the number of currently connected clients.
    pub async fn client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    /// Return the local address the server is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
}

impl<T> Drop for WsServer<T> {
    fn drop(&mut self) {
        self._accept_task.abort();
    }
}
