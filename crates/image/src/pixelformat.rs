use crate::*;

// fourcc codes
pub(crate) const FOURCC_RGB8: u32 = u32::from_le_bytes(*b"RGB8");
pub(crate) const FOURCC_ARGB8: u32 = u32::from_le_bytes(*b"ARGB");
pub(crate) const FOURCC_YUYV: u32 = u32::from_le_bytes(*b"YUYV");
pub(crate) const FOURCC_MJPG: u32 = u32::from_le_bytes(*b"MJPG");
pub(crate) const FOURCC_SRGGB10P: u32 = u32::from_le_bytes(*b"pRAA");
pub(crate) const FOURCC_YU12: u32 = u32::from_le_bytes(*b"YU12");

/// Convert a fourcc code to a readable 4-character string.
pub fn fourcc_to_string(fourcc: u32) -> String {
    String::from_utf8_lossy(&fourcc.to_le_bytes()).into_owned()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb8,
    Argb8,
    Yuyv,
    Yu12,
    Srggb10p,
    Jpeg,
}

impl PixelFormat {
    pub fn from_fourcc(fourcc: u32) -> Self {
        match u32::from_le_bytes(fourcc.to_le_bytes()) {
            FOURCC_RGB8 => PixelFormat::Rgb8,
            FOURCC_ARGB8 => PixelFormat::Argb8,
            FOURCC_YUYV => PixelFormat::Yuyv,
            FOURCC_YU12 => PixelFormat::Yu12,
            FOURCC_SRGGB10P => PixelFormat::Srggb10p,
            FOURCC_MJPG => PixelFormat::Jpeg,
            _ => panic!("Unsupported pixel format: {}", fourcc),
        }
    }

    pub fn as_fourcc(&self) -> u32 {
        match self {
            PixelFormat::Rgb8 => FOURCC_RGB8,
            PixelFormat::Argb8 => FOURCC_ARGB8,
            PixelFormat::Yuyv => FOURCC_YUYV,
            PixelFormat::Yu12 => FOURCC_YU12,
            PixelFormat::Srggb10p => FOURCC_SRGGB10P,
            PixelFormat::Jpeg => FOURCC_MJPG,
        }
    }

    pub fn ensure_format(&self, expected: PixelFormat) -> Result<(), ImageError> {
        if *self != expected {
            return Err(ImageError::Decode(format!(
                "expected {:?} format, got {:?}",
                expected, self
            )));
        }
        Ok(())
    }
}

// BT.601 YUV-to-RGB conversion for a single pixel (fixed-point, shift 8)
pub(crate) fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let u = u as i32 - 128;
    let v = v as i32 - 128;
    let r = (y + ((359 * v) >> 8)).clamp(0, 255) as u8;
    let g = (y - ((88 * u + 183 * v) >> 8)).clamp(0, 255) as u8;
    let b = (y + ((454 * u) >> 8)).clamp(0, 255) as u8;
    (r, g, b)
}
