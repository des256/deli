use crate::{framing, ComError};
use deli_codec::Codec;
use std::marker::PhantomData;
use tokio::net::{TcpStream, ToSocketAddrs};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

pub struct Client<T> {
    reader: OwnedReadHalf,
    writer: OwnedWriteHalf,
    _marker: PhantomData<T>,
}

impl<T: Codec> Client<T> {
    /// Connect to a Server and return a Client.
    ///
    /// The client can then call `recv()` to receive messages from the server
    /// or `send()` to send messages to the server.
    pub async fn connect(addr: impl ToSocketAddrs) -> Result<Self, ComError> {
        let stream = TcpStream::connect(addr).await?;
        let (read_half, write_half) = stream.into_split();

        Ok(Self {
            reader: read_half,
            writer: write_half,
            _marker: PhantomData,
        })
    }

    /// Send a message to the server.
    ///
    /// Returns `Ok(())` on success or a `ComError` if the write fails.
    pub async fn send(&mut self, value: &T) -> Result<(), ComError> {
        framing::write_message(&mut self.writer, value).await
    }

    /// Receive the next message from the server.
    ///
    /// Returns `ComError::ConnectionClosed` if the server closes the connection.
    pub async fn recv(&mut self) -> Result<T, ComError> {
        framing::read_message(&mut self.reader).await
    }
}
