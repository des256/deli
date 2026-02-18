use crate::ComError;
use codec::Codec;
use futures_util::{SinkExt, StreamExt};
use std::marker::PhantomData;
use std::net::SocketAddr;
use tokio_websockets::{ClientBuilder, MaybeTlsStream, Message, WebSocketStream};

pub struct WsClient<T> {
    stream: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
    _marker: PhantomData<T>,
}

impl<T: Codec> WsClient<T> {
    /// Connect to a WsServer and return a WsClient.
    ///
    /// The client can then call `recv()` to receive messages from the server
    /// or `send()` to send messages to the server.
    pub async fn connect(addr: SocketAddr) -> Result<Self, ComError> {
        // Build WebSocket URI from socket address
        let uri = format!("ws://{}", addr);

        // Connect via WebSocket
        let parsed_uri: http::Uri = uri.parse().map_err(|e| {
            ComError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid WebSocket URI: {e}"),
            ))
        })?;
        let (stream, _response) = ClientBuilder::from_uri(parsed_uri).connect().await?;

        Ok(Self {
            stream,
            _marker: PhantomData,
        })
    }

    /// Send a message to the server.
    ///
    /// Returns `Ok(())` on success or a `ComError` if the send fails.
    pub async fn send(&mut self, value: &T) -> Result<(), ComError> {
        let payload = value.to_bytes();
        let msg = Message::binary(payload);
        self.stream.send(msg).await?;
        Ok(())
    }

    /// Receive the next message from the server.
    ///
    /// Returns `ComError::ConnectionClosed` if the server closes the connection.
    /// Ignores non-binary messages (text, ping, pong, close).
    pub async fn recv(&mut self) -> Result<T, ComError> {
        loop {
            match self.stream.next().await {
                Some(Ok(msg)) => {
                    if msg.is_binary() {
                        let payload = msg.into_payload();
                        if payload.len() > crate::framing::MAX_MESSAGE_SIZE as usize {
                            return Err(ComError::MessageTooLarge(payload.len() as u32));
                        }
                        return T::from_bytes(&payload).map_err(ComError::from);
                    }
                    // Ignore non-binary messages, continue loop
                }
                Some(Err(e)) => return Err(ComError::from(e)),
                None => return Err(ComError::ConnectionClosed),
            }
        }
    }
}
