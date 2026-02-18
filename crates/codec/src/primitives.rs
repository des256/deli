use crate::{Codec, DecodeError};

// Helper: read exactly N bytes from buf at pos
fn read_bytes<'a>(buf: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], DecodeError> {
    if *pos + n > buf.len() {
        return Err(DecodeError::UnexpectedEof);
    }
    let slice = &buf[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

// --- bool ---

impl Codec for bool {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.push(if *self { 1 } else { 0 });
    }

    fn decode(buf: &[u8], pos: &mut usize) -> Result<Self, DecodeError> {
        let bytes = read_bytes(buf, pos, 1)?;
        match bytes[0] {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(DecodeError::InvalidBool(v)),
        }
    }
}

// --- Integer and float types via macro ---

macro_rules! impl_codec_for_numeric {
    ($($ty:ty),*) => {
        $(
            impl Codec for $ty {
                fn encode(&self, buf: &mut Vec<u8>) {
                    buf.extend_from_slice(&self.to_le_bytes());
                }

                fn decode(buf: &[u8], pos: &mut usize) -> Result<Self, DecodeError> {
                    const SIZE: usize = std::mem::size_of::<$ty>();
                    let bytes = read_bytes(buf, pos, SIZE)?;
                    Ok(<$ty>::from_le_bytes(bytes.try_into().unwrap()))
                }
            }
        )*
    };
}

impl_codec_for_numeric!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

// --- String ---

impl Codec for String {
    fn encode(&self, buf: &mut Vec<u8>) {
        let bytes = self.as_bytes();
        (bytes.len() as u32).encode(buf);
        buf.extend_from_slice(bytes);
    }

    fn decode(buf: &[u8], pos: &mut usize) -> Result<Self, DecodeError> {
        let len = u32::decode(buf, pos)? as usize;
        let bytes = read_bytes(buf, pos, len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| DecodeError::InvalidUtf8)
    }
}

// --- Vec<T: Codec> ---

impl<T: Codec> Codec for Vec<T> {
    fn encode(&self, buf: &mut Vec<u8>) {
        (self.len() as u32).encode(buf);
        for item in self {
            item.encode(buf);
        }
    }

    fn decode(buf: &[u8], pos: &mut usize) -> Result<Self, DecodeError> {
        let len = u32::decode(buf, pos)? as usize;
        let remaining = buf.len() - *pos;
        let capacity = len.min(remaining);
        let mut vec = Vec::with_capacity(capacity);
        for _ in 0..len {
            vec.push(T::decode(buf, pos)?);
        }
        Ok(vec)
    }
}
