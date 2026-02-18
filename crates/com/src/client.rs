use crate::{ComError, framing};
use codec::Codec;
use futures_core::Stream;
use futures_sink::Sink;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpStream, ToSocketAddrs};

pub struct Client<T> {
    reader: Option<OwnedReadHalf>,
    writer: Option<OwnedWriteHalf>,
    read_fut: Option<Pin<Box<dyn Future<Output = (OwnedReadHalf, Result<T, ComError>)> + Send>>>,
    write_fut: Option<Pin<Box<dyn Future<Output = (OwnedWriteHalf, Result<(), ComError>)> + Send>>>,
    _marker: PhantomData<T>,
}

impl<T: Codec> Client<T> {
    /// Connect to a Server and return a Client.
    ///
    /// The client implements `Stream<Item = Result<T, ComError>>` for receiving
    /// messages and `Sink<T, Error = ComError>` for sending messages.
    pub async fn connect(addr: impl ToSocketAddrs) -> Result<Self, ComError> {
        let stream = TcpStream::connect(addr).await?;
        let (read_half, write_half) = stream.into_split();

        Ok(Self {
            reader: Some(read_half),
            writer: Some(write_half),
            read_fut: None,
            write_fut: None,
            _marker: PhantomData,
        })
    }
}

impl<T: Codec + Unpin + Send + 'static> Stream for Client<T> {
    type Item = Result<T, ComError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        if this.read_fut.is_none() {
            let Some(mut reader) = this.reader.take() else {
                return Poll::Ready(None);
            };
            this.read_fut = Some(Box::pin(async move {
                let result = framing::read_message(&mut reader).await;
                (reader, result)
            }));
        }

        let fut = this.read_fut.as_mut().unwrap();
        match fut.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready((reader, result)) => {
                this.read_fut = None;
                this.reader = Some(reader);
                Poll::Ready(Some(result))
            }
        }
    }
}

impl<T: Codec + Unpin + Send + Sync + 'static> Sink<T> for Client<T> {
    type Error = ComError;

    fn poll_ready(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        if let Some(fut) = this.write_fut.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready((writer, result)) => {
                    this.write_fut = None;
                    this.writer = Some(writer);
                    result?;
                }
            }
        }
        Poll::Ready(Ok(()))
    }

    fn start_send(self: Pin<&mut Self>, item: T) -> Result<(), Self::Error> {
        let this = self.get_mut();
        let Some(mut writer) = this.writer.take() else {
            return Err(ComError::ConnectionClosed);
        };
        this.write_fut = Some(Box::pin(async move {
            let result = framing::write_message(&mut writer, &item).await;
            (writer, result)
        }));
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        if let Some(fut) = this.write_fut.as_mut() {
            match fut.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready((writer, result)) => {
                    this.write_fut = None;
                    this.writer = Some(writer);
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
                Poll::Ready((writer, result)) => {
                    this.write_fut = None;
                    this.writer = Some(writer);
                    result?;
                }
            }
        }
        Poll::Ready(Ok(()))
    }
}
