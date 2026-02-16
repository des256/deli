use crate::{framing, ComError};
use deli_codec::Codec;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::{TcpListener, ToSocketAddrs};
use tokio::net::tcp::OwnedWriteHalf;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

pub struct SenderServer<T> {
    clients: Arc<RwLock<HashMap<SocketAddr, OwnedWriteHalf>>>,
    _accept_task: JoinHandle<()>,
    local_addr: SocketAddr,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Codec> SenderServer<T> {
    /// Bind a TCP listener and start accepting client connections.
    ///
    /// Returns a `SenderServer` with a background task that accepts new connections
    /// and adds them to the internal client map.
    pub async fn bind(addr: impl ToSocketAddrs) -> Result<Self, ComError> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;

        let clients = Arc::new(RwLock::new(HashMap::new()));
        let clients_clone = clients.clone();

        // Spawn accept loop
        let accept_task = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, addr)) => {
                        let (_, write_half) = stream.into_split();
                        clients_clone.write().await.insert(addr, write_half);
                    }
                    Err(e) => {
                        log::warn!("Accept error: {}", e);
                        // Backoff to prevent CPU spin on persistent errors
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                }
            }
        });

        Ok(Self {
            clients,
            _accept_task: accept_task,
            local_addr,
            _marker: std::marker::PhantomData,
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

    /// Return the number of currently connected clients.
    pub async fn client_count(&self) -> usize {
        self.clients.read().await.len()
    }

    /// Return the local address the server is bound to.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }
}

impl<T> Drop for SenderServer<T> {
    fn drop(&mut self) {
        self._accept_task.abort();
    }
}
