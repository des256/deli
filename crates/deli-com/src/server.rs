use crate::{framing, ComError};
use deli_codec::Codec;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, ToSocketAddrs};
use tokio::net::tcp::OwnedWriteHalf;
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;

pub struct Server<T> {
    clients: Arc<RwLock<HashMap<SocketAddr, OwnedWriteHalf>>>,
    rx: mpsc::Receiver<T>,
    _accept_task: JoinHandle<()>,
    local_addr: SocketAddr,
}

impl<T: Codec + Send + 'static> Server<T> {
    /// Bind a TCP listener and start accepting client connections.
    ///
    /// Returns a `Server` with a background task that accepts new connections,
    /// spawns per-client reader tasks, and adds write halves to the client map.
    pub async fn bind(addr: impl ToSocketAddrs) -> Result<Self, ComError> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;

        let clients: Arc<RwLock<HashMap<SocketAddr, OwnedWriteHalf>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let clients_clone = clients.clone();

        let (tx, rx) = mpsc::channel(256);

        // Spawn accept loop
        let accept_task = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let (read_half, write_half) = stream.into_split();
                        clients_clone.write().await.insert(addr, write_half);

                        // Spawn a reader task for this client
                        let tx = tx.clone();
                        let clients_for_cleanup = clients_clone.clone();
                        tokio::spawn(async move {
                            let mut reader = read_half;
                            loop {
                                match framing::read_message::<T, _>(&mut reader).await {
                                    Ok(value) => {
                                        if tx.send(value).await.is_err() {
                                            break; // Server dropped
                                        }
                                    }
                                    Err(e) => {
                                        log::warn!("Client {} disconnected: {}", addr, e);
                                        clients_for_cleanup.write().await.remove(&addr);
                                        break;
                                    }
                                }
                            }
                        });
                    }
                    Err(e) => {
                        log::warn!("Accept error: {}", e);
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
        let mut lock = self.clients.write().await;

        let mut failed_addrs = Vec::new();

        for (addr, writer) in lock.iter_mut() {
            if let Err(e) = framing::write_message(writer, value).await {
                log::warn!("Failed to send to {}: {}", addr, e);
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

impl<T> Drop for Server<T> {
    fn drop(&mut self) {
        self._accept_task.abort();
    }
}
