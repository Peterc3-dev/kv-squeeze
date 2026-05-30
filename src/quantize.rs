use half::f16;
use rayon::prelude::*;
use std::fmt;

/// Supported quantization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    FP16,
    FP8E4M3,
    INT4,
}

impl fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantMethod::FP16 => write!(f, "FP16"),
            QuantMethod::FP8E4M3 => write!(f, "FP8 (E4M3)"),
            QuantMethod::INT4 => write!(f, "INT4 (grouped)"),
        }
    }
}

impl QuantMethod {
    pub fn compression_ratio(&self) -> f64 {
        match self {
            QuantMethod::FP16 => 2.0,
            QuantMethod::FP8E4M3 => 4.0,
            QuantMethod::INT4 => 8.0, // approximate, ignoring scale overhead
        }
    }

    pub fn bits_per_element(&self) -> u32 {
        match self {
            QuantMethod::FP16 => 16,
            QuantMethod::FP8E4M3 => 8,
            QuantMethod::INT4 => 4,
        }
    }
}

/// Round-trip accuracy statistics.
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub method: QuantMethod,
    pub mse: f64,
    pub max_error: f64,
    pub mean_error: f64,
    pub num_elements: usize,
    pub original_bytes: usize,
    pub compressed_bytes: usize,
}

impl CompressionStats {
    pub fn compression_ratio(&self) -> f64 {
        self.original_bytes as f64 / self.compressed_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// FP16 quantization
// ---------------------------------------------------------------------------

/// Quantize FP32 slice to FP16 (parallel across chunks).
pub fn quantize_fp16(data: &[f32]) -> Vec<f16> {
    data.par_chunks(4096)
        .flat_map(|chunk| chunk.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>())
        .collect()
}

/// Dequantize FP16 back to FP32.
pub fn dequantize_fp16(data: &[f16]) -> Vec<f32> {
    data.par_chunks(4096)
        .flat_map(|chunk| chunk.iter().map(|v| v.to_f32()).collect::<Vec<_>>())
        .collect()
}

// ---------------------------------------------------------------------------
// FP8 E4M3 quantization (proper bit-level encoding)
//
// Layout: 1 sign | 4 exponent (bias 7) | 3 mantissa
// - Exponent 0b0000, mantissa != 0 → subnormal: (-1)^s × 2^(-6) × (mant/8)
// - Exponent 0b0000, mantissa == 0 → zero
// - Exponent 0b1111, mantissa == 0b111 → NaN (E4M3 has NO infinity)
// - Normal: exponent 1–14 → (-1)^s × 2^(exp-7) × (1 + mant/8)
// - Max finite: exp=14, mant=6 → 1.75 × 2^7 = 448 (0b_s_1110_110)
//   Note: exp=14, mant=7 (0x77 unsigned) = 1.875 × 128 = 240… wait:
//   Actually max finite is exp=14, mant=7? No: exp=15, mant=6 would be
//   normal but exp=15 is only valid with mant=7 for NaN.
//   Let's be precise: exponents 1–14 are normal. Exponent 15 is special:
//   only (15, 7) = NaN. All other (15, 0–6) are valid normals.
//   So max finite = exp=15, mant=6 = 1.75 × 2^8 = 448.
//   And exp=14, mant=7 = 1.875 × 2^7 = 240.
// ---------------------------------------------------------------------------

/// Largest finite FP8 E4M3 value: exp=15, mant=6 → (1+6/8) × 2^(15-7) = 1.75 × 256 = 448
#[allow(dead_code)]
const FP8_MAX_VAL: f32 = 448.0;

/// Encode a single f32 to FP8 E4M3 (packed u8).
///
/// Handles zeros, subnormals, normals, NaN, and infinity (mapped to NaN since
/// E4M3 has no infinity representation). Values exceeding the representable
/// range are clamped to ±448.
fn encode_fp8(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let f32_exp = ((bits >> 23) & 0xFF) as i32;
    let f32_mant = bits & 0x7F_FFFF;

    // Handle f32 special values: NaN or infinity → E4M3 NaN
    if f32_exp == 0xFF {
        return (sign << 7) | 0x7F; // exp=15, mant=7 → NaN
    }

    // Handle zero (preserves sign)
    if f32_exp == 0 && f32_mant == 0 {
        return sign << 7;
    }

    // Compute unbiased exponent (f32 bias = 127)
    let exp_unbiased = if f32_exp == 0 {
        -126i32 // f32 subnormal
    } else {
        f32_exp - 127
    };

    // Build full 24-bit mantissa with implicit bit
    let full_mant = if f32_exp == 0 {
        f32_mant << 1 // f32 subnormal: no implicit 1, shift up
    } else {
        f32_mant | 0x80_0000 // f32 normal: add implicit 1
    };

    // E4M3 bias = 7
    // Normal exponent range: stored 1–15, unbiased -6 to 8
    // But stored exp=15 with mant=7 is NaN, so max usable:
    //   exp=15, mant=6 (value 448) or exp=15, mant<7
    // Subnormal: stored exp=0, value = 2^(-6) × (mant/8)

    // Overflow: clamp to max finite (448.0 = exp=15, mant=6)
    if exp_unbiased > 8 {
        return (sign << 7) | 0x7E; // 0b_s_1111_110
    }

    // Underflow: too small even for smallest subnormal (2^-9)
    if exp_unbiased < -9 {
        return sign << 7; // flush to zero
    }

    // Subnormal E4M3 range: unbiased exponent < -6
    if exp_unbiased < -6 {
        // Subnormal: stored exp = 0, value = 2^(-6) × (mant/8)
        // Derivation: mant = full_mant × 2^(exp_unbiased - 14)
        //           = full_mant >> (14 - exp_unbiased)
        let total_shift = (14 - exp_unbiased) as u32;
        let mant_shifted = full_mant >> total_shift;
        // Round-to-nearest: check the bit just below
        let round_bit = if total_shift > 0 {
            (full_mant >> (total_shift - 1)) & 1
        } else {
            0
        };
        let mant_rounded = (mant_shifted + round_bit).min(7) as u8;
        if mant_rounded == 0 {
            return sign << 7; // rounded down to zero
        }
        // If rounding pushed mant to 8, it becomes the smallest normal
        if mant_rounded >= 8 {
            return (sign << 7) | 0x08; // exp=1, mant=0
        }
        return (sign << 7) | mant_rounded;
    }

    // Normal E4M3 range
    let stored_exp = (exp_unbiased + 7) as u8; // bias = 7

    // Extract top 3 mantissa bits from the 23-bit fractional part
    // (full_mant bit 23 is the implicit 1, bits 22..0 are fractional)
    let mant_3bit = ((full_mant >> 20) & 0x7) as u8;
    let round_bit = ((full_mant >> 19) & 1) as u8;

    let mut result_mant = mant_3bit + round_bit;
    let mut result_exp = stored_exp;

    // Mantissa overflow from rounding: carry into exponent
    if result_mant > 7 {
        result_mant = 0;
        result_exp += 1;
    }

    // Check if we'd produce NaN (exp=15, mant=7) or exceed range
    if result_exp > 15 || (result_exp == 15 && result_mant == 7) {
        // Clamp to max finite: exp=15, mant=6 (448.0)
        return (sign << 7) | 0x7E;
    }

    (sign << 7) | (result_exp << 3) | result_mant
}

/// Decode FP8 E4M3 (packed u8) back to f32.
fn decode_fp8(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = ((byte >> 3) & 0xF) as i32;
    let mant = (byte & 0x7) as u32;

    // NaN: exp=15, mant=7
    if exp == 15 && mant == 7 {
        return f32::NAN;
    }

    let sign_f = if sign == 1 { -1.0f32 } else { 1.0f32 };

    if exp == 0 {
        if mant == 0 {
            // Signed zero
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Subnormal: (-1)^s × 2^(-6) × (mant / 8)
        return sign_f * (2.0f32).powi(-6) * (mant as f32 / 8.0);
    }

    // Normal: (-1)^s × 2^(exp - 7) × (1 + mant / 8)
    sign_f * (2.0f32).powi(exp - 7) * (1.0 + mant as f32 / 8.0)
}

/// Quantize FP32 slice to FP8 E4M3 (parallel).
pub fn quantize_fp8(data: &[f32]) -> Vec<u8> {
    data.par_chunks(4096)
        .flat_map(|chunk| chunk.iter().map(|&v| encode_fp8(v)).collect::<Vec<_>>())
        .collect()
}

/// Dequantize FP8 E4M3 back to FP32.
pub fn dequantize_fp8(data: &[u8]) -> Vec<f32> {
    data.par_chunks(4096)
        .flat_map(|chunk| chunk.iter().map(|&b| decode_fp8(b)).collect::<Vec<_>>())
        .collect()
}

// ---------------------------------------------------------------------------
// INT4 quantization with per-group scaling
//
// Each group of `group_size` f32 values is quantized to 4-bit signed integers
// [-8, 7] with a single FP16 scale factor per group.
//
// Packed format: two int4 values per byte (low nibble first).
// Storage: for N elements with group_size G:
//   - ceil(N/2) bytes of packed data
//   - ceil(N/G) * 2 bytes of scales (FP16)
// ---------------------------------------------------------------------------

/// Default group size for INT4 quantization.
pub const INT4_GROUP_SIZE: usize = 32;

/// Packed INT4 quantized data.
#[derive(Debug, Clone)]
pub struct Int4Packed {
    pub data: Vec<u8>,
    pub scales: Vec<f16>,
    pub group_size: usize,
    pub num_elements: usize,
}

impl Int4Packed {
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 2
    }
}

/// Quantize FP32 slice to INT4 with per-group scaling.
pub fn quantize_int4(data: &[f32]) -> Int4Packed {
    quantize_int4_grouped(data, INT4_GROUP_SIZE)
}

/// Quantize with configurable group size.
pub fn quantize_int4_grouped(data: &[f32], group_size: usize) -> Int4Packed {
    let num_groups = data.len().div_ceil(group_size);

    // Compute scales per group (parallel)
    let scales: Vec<f16> = (0..num_groups)
        .into_par_iter()
        .map(|g| {
            let start = g * group_size;
            let end = (start + group_size).min(data.len());
            let group = &data[start..end];
            let abs_max = group.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            // Scale maps abs_max -> 7 (max positive int4 value)
            let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 7.0 };
            f16::from_f32(scale)
        })
        .collect();

    // Quantize values to int4 and pack (parallel per group, then flatten)
    let packed_groups: Vec<Vec<u8>> = (0..num_groups)
        .into_par_iter()
        .map(|g| {
            let start = g * group_size;
            let end = (start + group_size).min(data.len());
            let group = &data[start..end];
            let scale = scales[g].to_f32();
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

            // Quantize to [-8, 7]
            let quants: Vec<i8> = group
                .iter()
                .map(|&v| {
                    let q = (v * inv_scale).round() as i32;
                    q.clamp(-8, 7) as i8
                })
                .collect();

            // Pack pairs into bytes (low nibble = first element)
            let mut packed = Vec::with_capacity(quants.len().div_ceil(2));
            for pair in quants.chunks(2) {
                let lo = (pair[0] as u8) & 0x0F;
                let hi = if pair.len() > 1 {
                    (pair[1] as u8) & 0x0F
                } else {
                    0
                };
                packed.push(lo | (hi << 4));
            }
            packed
        })
        .collect();

    let data_packed: Vec<u8> = packed_groups.into_iter().flatten().collect();

    Int4Packed {
        data: data_packed,
        scales,
        group_size,
        num_elements: data.len(),
    }
}

/// Dequantize INT4 packed data back to FP32.
pub fn dequantize_int4(packed: &Int4Packed) -> Vec<f32> {
    let mut result = Vec::with_capacity(packed.num_elements);

    let mut byte_idx = 0;
    let mut elem_idx = 0;

    for (g, &scale_f16) in packed.scales.iter().enumerate() {
        let scale = scale_f16.to_f32();
        let group_start = g * packed.group_size;
        let group_end = (group_start + packed.group_size).min(packed.num_elements);
        let group_len = group_end - group_start;

        let mut group_elem = 0;
        while group_elem < group_len {
            let byte = packed.data[byte_idx];

            // Low nibble
            let lo = (byte & 0x0F) as i8;
            // Sign-extend 4-bit value
            let lo_ext = if lo & 0x08 != 0 {
                lo | !0x0F_u8 as i8
            } else {
                lo
            };
            result.push(lo_ext as f32 * scale);
            group_elem += 1;
            elem_idx += 1;

            if group_elem < group_len {
                // High nibble
                let hi = ((byte >> 4) & 0x0F) as i8;
                let hi_ext = if hi & 0x08 != 0 {
                    hi | !0x0F_u8 as i8
                } else {
                    hi
                };
                result.push(hi_ext as f32 * scale);
                group_elem += 1;
                elem_idx += 1;
            }

            byte_idx += 1;
        }
    }

    result.truncate(packed.num_elements);
    let _ = elem_idx; // suppress unused warning
    result
}

// ---------------------------------------------------------------------------
// Round-trip accuracy measurement
// ---------------------------------------------------------------------------

/// Measure round-trip accuracy for a given quantization method.
pub fn round_trip_stats(data: &[f32], method: QuantMethod) -> CompressionStats {
    if data.is_empty() {
        return CompressionStats {
            method,
            mse: 0.0,
            max_error: 0.0,
            mean_error: 0.0,
            num_elements: 0,
            original_bytes: 0,
            compressed_bytes: 0,
        };
    }

    let original_bytes = data.len() * 4;

    let (reconstructed, compressed_bytes) = match method {
        QuantMethod::FP16 => {
            let q = quantize_fp16(data);
            let bytes = q.len() * 2;
            (dequantize_fp16(&q), bytes)
        }
        QuantMethod::FP8E4M3 => {
            let q = quantize_fp8(data);
            let bytes = q.len();
            (dequantize_fp8(&q), bytes)
        }
        QuantMethod::INT4 => {
            let q = quantize_int4(data);
            let bytes = q.size_bytes();
            let r = dequantize_int4(&q);
            (r, bytes)
        }
    };

    let n = data.len() as f64;
    let mut sum_sq_err = 0.0_f64;
    let mut sum_abs_err = 0.0_f64;
    let mut max_err = 0.0_f64;

    for (orig, recon) in data.iter().zip(reconstructed.iter()) {
        let err = (*orig as f64 - *recon as f64).abs();
        sum_sq_err += err * err;
        sum_abs_err += err;
        if err > max_err {
            max_err = err;
        }
    }

    CompressionStats {
        method,
        mse: sum_sq_err / n,
        max_error: max_err,
        mean_error: sum_abs_err / n,
        num_elements: data.len(),
        original_bytes,
        compressed_bytes,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data(n: usize) -> Vec<f32> {
        // Deterministic pseudo-random data in [-1, 1]
        let mut data = Vec::with_capacity(n);
        let mut x: u32 = 42;
        for _ in 0..n {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            let f = ((x >> 16) as f32 / 32768.0) - 1.0;
            data.push(f);
        }
        data
    }

    #[test]
    fn fp16_round_trip_accuracy() {
        let data = sample_data(1024);
        let stats = round_trip_stats(&data, QuantMethod::FP16);
        // FP16 should be very accurate for values in [-1, 1]
        assert!(stats.mse < 1e-6, "FP16 MSE too high: {}", stats.mse);
        assert!(
            stats.max_error < 1e-3,
            "FP16 max error too high: {}",
            stats.max_error
        );
        assert!(
            (stats.compression_ratio() - 2.0).abs() < 0.01,
            "FP16 compression ratio wrong: {}",
            stats.compression_ratio()
        );
    }

    #[test]
    fn fp8_round_trip_accuracy() {
        let data = sample_data(1024);
        let stats = round_trip_stats(&data, QuantMethod::FP8E4M3);
        // FP8 is coarser but should still be reasonable for [-1, 1]
        assert!(stats.mse < 0.01, "FP8 MSE too high: {}", stats.mse);
        assert!(
            stats.max_error < 0.2,
            "FP8 max error too high: {}",
            stats.max_error
        );
        assert!(
            (stats.compression_ratio() - 4.0).abs() < 0.01,
            "FP8 compression ratio wrong: {}",
            stats.compression_ratio()
        );
    }

    #[test]
    fn fp8_preserves_zero() {
        // Positive and negative zero
        assert_eq!(encode_fp8(0.0), 0x00);
        assert_eq!(encode_fp8(-0.0), 0x80);

        let r_pos = decode_fp8(0x00);
        let r_neg = decode_fp8(0x80);
        assert_eq!(r_pos, 0.0);
        assert!(!r_pos.is_sign_negative());
        assert_eq!(r_neg, 0.0);
        assert!(r_neg.is_sign_negative());

        // Round-trip via public API
        let data = vec![0.0_f32, -0.0, 0.0];
        let q = quantize_fp8(&data);
        let r = dequantize_fp8(&q);
        for v in &r {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn fp8_nan_encoding() {
        // f32 NaN → E4M3 NaN
        let encoded = encode_fp8(f32::NAN);
        assert_eq!(encoded & 0x7F, 0x7F, "NaN should encode to exp=15, mant=7");

        // f32 infinity → E4M3 NaN (no infinity in E4M3)
        let enc_inf = encode_fp8(f32::INFINITY);
        assert_eq!(enc_inf & 0x7F, 0x7F, "Inf should map to NaN");
        let enc_neg_inf = encode_fp8(f32::NEG_INFINITY);
        assert_eq!(enc_neg_inf & 0x7F, 0x7F, "Neg Inf should map to NaN");
        assert_eq!(enc_neg_inf >> 7, 1, "Neg Inf NaN should preserve sign");

        // E4M3 NaN decodes to f32 NaN
        assert!(decode_fp8(0x7F).is_nan(), "0x7F should decode to NaN");
        assert!(decode_fp8(0xFF).is_nan(), "0xFF should decode to NaN");
    }

    #[test]
    fn fp8_max_value() {
        // Max finite E4M3 = 448.0 (exp=15, mant=6 → 1.75 × 2^8)
        let encoded = encode_fp8(448.0);
        assert_eq!(
            encoded, 0x7E,
            "448.0 should encode to 0x7E (exp=15, mant=6)"
        );
        let decoded = decode_fp8(0x7E);
        assert_eq!(decoded, 448.0, "0x7E should decode to 448.0");

        // Negative max
        let neg_encoded = encode_fp8(-448.0);
        assert_eq!(neg_encoded, 0xFE, "-448.0 should encode to 0xFE");
        let neg_decoded = decode_fp8(0xFE);
        assert_eq!(neg_decoded, -448.0);
    }

    #[test]
    fn fp8_overflow_clamps() {
        // Values > 448 should clamp to max finite, not become NaN
        let data = vec![1000.0_f32, -1000.0, 500.0, -600.0];
        let q = quantize_fp8(&data);
        let r = dequantize_fp8(&q);
        assert_eq!(r[0], 448.0, "1000.0 should clamp to 448.0");
        assert_eq!(r[1], -448.0, "-1000.0 should clamp to -448.0");
        assert_eq!(r[2], 448.0, "500.0 should clamp to 448.0");
        assert_eq!(r[3], -448.0, "-600.0 should clamp to -448.0");

        // Verify the encoded value is NOT NaN
        for &byte in &q {
            let exp = (byte >> 3) & 0xF;
            let mant = byte & 0x7;
            assert!(
                !(exp == 15 && mant == 7),
                "Clamped value must not produce NaN encoding"
            );
        }
    }

    #[test]
    fn fp8_subnormals() {
        // Smallest subnormal: 2^(-6) × (1/8) = 2^(-9) ≈ 0.001953125
        let min_sub = 0.001953125_f32;
        let encoded = encode_fp8(min_sub);
        assert_eq!(
            encoded, 0x01,
            "Smallest subnormal should be 0x01 (exp=0, mant=1)"
        );
        let decoded = decode_fp8(0x01);
        assert!(
            (decoded - min_sub).abs() < 1e-10,
            "Smallest subnormal decode mismatch"
        );

        // Largest subnormal: 2^(-6) × (7/8) = 0.109375
        let max_sub_decoded = decode_fp8(0x07); // exp=0, mant=7
        let expected = (2.0f32).powi(-6) * 7.0 / 8.0;
        assert!(
            (max_sub_decoded - expected).abs() < 1e-7,
            "Largest subnormal: got {}, expected {}",
            max_sub_decoded,
            expected
        );

        // Negative subnormal
        let neg_encoded = encode_fp8(-min_sub);
        assert_eq!(
            neg_encoded, 0x81,
            "Negative smallest subnormal should be 0x81"
        );
        let neg_decoded = decode_fp8(0x81);
        assert!((neg_decoded + min_sub).abs() < 1e-10);
    }

    #[test]
    fn fp8_underflow_flushes_to_zero() {
        // Values smaller than smallest subnormal (2^-9) should flush to zero
        let tiny = 1e-4_f32;
        let encoded = encode_fp8(tiny);
        let decoded = decode_fp8(encoded);
        // Should be either zero or the smallest subnormal
        assert!(
            decoded.abs() <= 0.001953125,
            "Very small value should flush to zero or smallest subnormal, got {}",
            decoded
        );

        let tinier = 1e-6_f32;
        let encoded2 = encode_fp8(tinier);
        let decoded2 = decode_fp8(encoded2);
        assert_eq!(decoded2, 0.0, "Extremely small value should flush to zero");
    }

    #[test]
    fn fp8_negative_values() {
        // Test that sign bit works correctly for normal values
        let test_vals = [1.0f32, -1.0, 2.5, -2.5, 0.5, -0.5];
        for &v in &test_vals {
            let encoded = encode_fp8(v);
            let decoded = decode_fp8(encoded);
            assert_eq!(
                decoded.is_sign_negative(),
                v.is_sign_negative(),
                "Sign mismatch for value {}",
                v
            );
            assert!(
                (decoded.abs() - v.abs()).abs() < v.abs() * 0.15,
                "Round-trip error too large for {}: got {}",
                v,
                decoded
            );
        }
    }

    #[test]
    fn fp8_known_normal_values() {
        // 1.0 = 2^(1-7+7) × (1 + 0/8) → exp=7, mant=0 → 0b_0_0111_000 = 0x38
        let enc = encode_fp8(1.0);
        assert_eq!(enc, 0x38, "1.0 should encode to 0x38");
        assert_eq!(decode_fp8(0x38), 1.0);

        // 2.0 = 2^(8-7) = 2^1 → exp=8, mant=0 → 0b_0_1000_000 = 0x40
        let enc2 = encode_fp8(2.0);
        assert_eq!(enc2, 0x40, "2.0 should encode to 0x40");
        assert_eq!(decode_fp8(0x40), 2.0);

        // 0.5 = 2^(6-7) × (1+0/8) → exp=6, mant=0 → 0b_0_0110_000 = 0x30
        let enc3 = encode_fp8(0.5);
        assert_eq!(enc3, 0x30, "0.5 should encode to 0x30");
        assert_eq!(decode_fp8(0x30), 0.5);

        // Smallest normal: exp=1, mant=0 → 2^(1-7) = 2^(-6) = 0.015625
        let enc4 = encode_fp8(0.015625);
        assert_eq!(enc4, 0x08, "Min normal should encode to 0x08");
        assert_eq!(decode_fp8(0x08), 0.015625);
    }

    #[test]
    fn fp8_all_valid_round_trip() {
        // Every valid FP8 encoding should round-trip through decode→encode
        for byte in 0u8..=255 {
            let exp = (byte >> 3) & 0xF;
            let mant = byte & 0x7;

            // Skip NaN
            if exp == 15 && mant == 7 {
                assert!(decode_fp8(byte).is_nan());
                continue;
            }

            let decoded = decode_fp8(byte);
            let re_encoded = encode_fp8(decoded);
            let re_decoded = decode_fp8(re_encoded);

            assert!(
                (decoded - re_decoded).abs() < 1e-10 || (decoded == 0.0 && re_decoded == 0.0),
                "Round-trip failed for byte 0x{:02X}: decoded={}, re_encoded=0x{:02X}, re_decoded={}",
                byte,
                decoded,
                re_encoded,
                re_decoded
            );
        }
    }

    #[test]
    fn int4_round_trip_accuracy() {
        let data = sample_data(1024);
        let stats = round_trip_stats(&data, QuantMethod::INT4);
        // INT4 is the coarsest — only 16 levels
        assert!(stats.mse < 0.05, "INT4 MSE too high: {}", stats.mse);
        assert!(
            stats.max_error < 0.5,
            "INT4 max error too high: {}",
            stats.max_error
        );
        // Compression ratio is approximate due to scale overhead
        assert!(
            stats.compression_ratio() > 5.0,
            "INT4 compression ratio too low: {}",
            stats.compression_ratio()
        );
    }

    #[test]
    fn int4_preserves_zeros() {
        let data = vec![0.0_f32; 64];
        let packed = quantize_int4(&data);
        let r = dequantize_int4(&packed);
        for v in &r {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn int4_element_count_preserved() {
        // Non-multiple of group size
        let data = sample_data(100);
        let packed = quantize_int4(&data);
        let r = dequantize_int4(&packed);
        assert_eq!(r.len(), data.len());
    }

    #[test]
    fn fp16_element_count_preserved() {
        let data = sample_data(137);
        let q = quantize_fp16(&data);
        let r = dequantize_fp16(&q);
        assert_eq!(r.len(), data.len());
    }

    #[test]
    fn int4_grouped_custom_group_size() {
        // Group size that does not evenly divide the element count exercises
        // the ragged-tail packing/unpacking path.
        let data = sample_data(70);
        let packed = quantize_int4_grouped(&data, 16);
        assert_eq!(packed.group_size, 16);
        assert_eq!(packed.num_elements, 70);
        // ceil(70/16) = 5 groups -> 5 scales.
        assert_eq!(packed.scales.len(), 5);
        let r = dequantize_int4(&packed);
        assert_eq!(r.len(), 70);
    }

    #[test]
    fn int4_packed_size_matches_formula() {
        let data = sample_data(100);
        let packed = quantize_int4_grouped(&data, 32);
        // ceil(100/2) data bytes + ceil(100/32)*2 scale bytes.
        let expected = 100usize.div_ceil(2) + 100usize.div_ceil(32) * 2;
        assert_eq!(packed.size_bytes(), expected);
    }

    #[test]
    fn quant_method_bits_and_display() {
        assert_eq!(QuantMethod::FP16.bits_per_element(), 16);
        assert_eq!(QuantMethod::FP8E4M3.bits_per_element(), 8);
        assert_eq!(QuantMethod::INT4.bits_per_element(), 4);
        assert_eq!(QuantMethod::FP16.to_string(), "FP16");
        assert_eq!(QuantMethod::FP8E4M3.to_string(), "FP8 (E4M3)");
        assert_eq!(QuantMethod::INT4.to_string(), "INT4 (grouped)");
    }

    #[test]
    fn round_trip_empty_input() {
        let stats = round_trip_stats(&[], QuantMethod::FP8E4M3);
        assert_eq!(stats.num_elements, 0);
        assert_eq!(stats.mse, 0.0);
        assert_eq!(stats.compressed_bytes, 0);
    }

    #[test]
    fn fp8_element_count_preserved() {
        let data = sample_data(137);
        let q = quantize_fp8(&data);
        let r = dequantize_fp8(&q);
        assert_eq!(r.len(), data.len());
    }
}
