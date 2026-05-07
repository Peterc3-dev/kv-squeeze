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
// FP8 E4M3 quantization (manual)
//
// Layout: 1 sign | 4 exponent | 3 mantissa
// Bias = 7, exponent range [0,15], max normal ≈ 448, smallest normal ≈ 2^-6
// We clamp inputs to representable range, no NaN/Inf encoding.
// ---------------------------------------------------------------------------

const FP8_EXP_BIAS: i32 = 7;
const FP8_MAX_EXP: i32 = 15;
const FP8_MANT_BITS: u32 = 3;

/// Largest finite FP8 E4M3 value: (1 + 7/8) * 2^(15-7) = 1.875 * 256 = 448
const FP8_MAX_VAL: f32 = 448.0;

/// Encode a single f32 to FP8 E4M3 (packed u8).
fn encode_fp8(val: f32) -> u8 {
    if val == 0.0 {
        // Preserve sign of zero
        if val.is_sign_negative() {
            return 0x80;
        }
        return 0x00;
    }

    let sign: u8 = if val < 0.0 { 1 } else { 0 };
    let abs = val.abs().min(FP8_MAX_VAL);

    // Decompose: abs = mantissa * 2^exp where 1.0 <= mantissa < 2.0
    // Use f32 bits to extract
    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127; // unbiased exponent
    let f32_frac = bits & 0x7FFFFF; // 23-bit fraction

    // Biased exponent for FP8
    let biased_exp = f32_exp + FP8_EXP_BIAS;

    if biased_exp <= 0 {
        // Subnormal or underflow — flush to zero
        return sign << 7;
    }

    let clamped_exp = biased_exp.min(FP8_MAX_EXP) as u8;

    // Round the 23-bit mantissa to 3 bits (take top 3 bits, round-to-nearest-even)
    let mant_3 = (f32_frac >> (23 - FP8_MANT_BITS)) as u8;
    let remainder = f32_frac & ((1 << (23 - FP8_MANT_BITS)) - 1);
    let halfway = 1u32 << (23 - FP8_MANT_BITS - 1);

    let rounded_mant = if remainder > halfway || (remainder == halfway && (mant_3 & 1) == 1) {
        mant_3 + 1
    } else {
        mant_3
    };

    // Handle mantissa overflow (carry into exponent)
    if rounded_mant >= (1 << FP8_MANT_BITS) {
        let new_exp = clamped_exp + 1;
        if new_exp > FP8_MAX_EXP as u8 {
            // Overflow to max representable
            return (sign << 7) | ((FP8_MAX_EXP as u8) << FP8_MANT_BITS) | 0x07;
        }
        return (sign << 7) | (new_exp << FP8_MANT_BITS) | 0x00;
    }

    (sign << 7) | (clamped_exp << FP8_MANT_BITS) | rounded_mant
}

/// Decode FP8 E4M3 (packed u8) back to f32.
fn decode_fp8(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = ((byte >> FP8_MANT_BITS) & 0x0F) as i32;
    let mant = (byte & 0x07) as u32;

    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    let val = if exp == 0 {
        // Subnormal: 0.mantissa * 2^(1 - bias)
        let frac = mant as f64 / (1u64 << FP8_MANT_BITS) as f64;
        (frac * 2.0_f64.powi(1 - FP8_EXP_BIAS)) as f32
    } else {
        // Normal: (1 + mantissa/8) * 2^(exp - bias)
        let frac = 1.0 + mant as f64 / (1u64 << FP8_MANT_BITS) as f64;
        (frac * 2.0_f64.powi(exp - FP8_EXP_BIAS)) as f32
    };

    if sign == 1 { -val } else { val }
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
    let num_groups = (data.len() + group_size - 1) / group_size;

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
            let mut packed = Vec::with_capacity((quants.len() + 1) / 2);
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
        let data = vec![0.0_f32, -0.0, 0.0];
        let q = quantize_fp8(&data);
        let r = dequantize_fp8(&q);
        for v in &r {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn fp8_clamps_large_values() {
        let data = vec![1000.0_f32, -1000.0];
        let q = quantize_fp8(&data);
        let r = dequantize_fp8(&q);
        assert!(r[0] <= FP8_MAX_VAL);
        assert!(r[1] >= -FP8_MAX_VAL);
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
    fn fp8_element_count_preserved() {
        let data = sample_data(137);
        let q = quantize_fp8(&data);
        let r = dequantize_fp8(&q);
        assert_eq!(r.len(), data.len());
    }
}
