/// Convert IEEE 754 half-precision (f16) to single-precision (f32).
pub(crate) fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // ±0
            f32::from_bits(sign << 31)
        } else {
            // Denormalized: shift mantissa until hidden bit appears
            let mut e = 0i32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = (127 - 15 + 1 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Infinity or NaN
        f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
    } else {
        // Normalized
        let f32_exp = (exponent as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}
