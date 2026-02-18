use crate::{framing, ComError};
use deli_codec::Codec;
use futures_core::Stream;
use futures_sink::Sink;
use std::collections::HashMap;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::net::tcp::OwnedWriteHalf;
use tokio::net::{TcpListener, ToSocketAddrs};
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;

pub struct Server<T> {
    clients: Arc<RwLock<HashMap<SocketAddr, OwnedWriteHalf>>>,
    rx: mpsc::Receiver<T>,
    write_fut: Option<Pin<Box<dyn Future<Output = Result<(), ComError>> + Send>>>,
    _accept_task: JoinHandle<()>,
    local_addr: SocketAddr,
}

impl<T: Codec + Send + 'static> Server<T> {
    /// Bind a TCP listener and start accepting client connections.
    ///
    /// Returns a `Server` that implements `Stream<Item = Result<T, ComError>>`
    /// for receiving client messages and `Sink<T, Error = ComError>` for
    /// broadcasting messages to all connected clients.
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
            write_fut: None,
            _accept_task: accept_task,
            local_addr,
        })
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

impl<T: Codec + Send + 'static> Stream for Server<T> {
    type Item = Result<T, ComError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(value)) => Poll::Ready(Some(Ok(value))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T: Codec + Send + Sync + 'static> Sink<T> for Server<T> {
    type Error = ComError;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        if let Some(fut) = this.write_fut.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => {
                    this.write_fut = None;
                    result?;
                }
            }
        }
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: T) -> Result<(), Self::Error> {
        let this = self.get_mut();
        let clients = this.clients.clone();
        this.write_fut = Some(Box::pin(async move {
            let mut lock = clients.write().await;
            let mut failed = Vec::new();
            for (addr, writer) in lock.iter_mut() {
                if let Err(e) = framing::write_message(writer, &item).await {
                    log::warn!("Failed to send to {}: {}", addr, e);
                    failed.push(*addr);
                }
            }
            for addr in failed {
                lock.remove(&addr);
            }
            Ok(())
        }));
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        if let Some(fut) = this.write_fut.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => {
                    this.write_fut = None;
                    result?;
                }
            }
        }
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        if let Some(fut) = this.write_fut.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => {
                    this.write_fut = None;
                    result?;
                }
            }
        }
        Poll::Ready(Ok(()))
    }
}

impl<T> Drop for Server<T> {
    fn drop(&mut self) {
        self._accept_task.abort();
    }
}
