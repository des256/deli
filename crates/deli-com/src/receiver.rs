use crate::{framing, ComError};
use deli_codec::Codec;
use std::marker::PhantomData;
use tokio::net::{TcpStream, ToSocketAddrs};
use tokio::net::tcp::OwnedReadHalf;

pub struct ReceiverClient<T> {
    reader: OwnedReadHalf,
    _marker: PhantomData<T>,
}

impl<T: Codec> ReceiverClient<T> {
    /// Connect to a SenderServer and return a ReceiverClient.
    ///
    /// The client can then call `recv()` to receive messages from the server.
    pub async fn connect(addr: impl ToSocketAddrs) -> Result<Self, ComError> {
        let stream = TcpStream::connect(addr).await?;
        let (read_half, _) = stream.into_split();

        Ok(Self {
            reader: read_half,
            _marker: PhantomData,
        })
    }

    /// Receive the next message from the server.
    ///
    /// Returns `ComError::ConnectionClosed` if the server closes the connection.
    pub async fn recv(&mut self) -> Result<T, ComError> {
        framing::read_message(&mut self.reader).await
    }
}
