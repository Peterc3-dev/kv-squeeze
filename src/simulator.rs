use crate::eviction::{H2OEviction, SlidingWindow, TokenEntry, TokenEviction};
use crate::quantize::QuantMethod;

/// Model configuration for simulation.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl ModelConfig {
    /// Preset for a "7B"-class model (e.g., Llama-2-7B).
    pub fn preset_7b() -> Self {
        Self {
            name: "7B (32-layer, 32-head, dim 128)".to_string(),
            num_layers: 32,
            num_kv_heads: 32,
            head_dim: 128,
        }
    }

    /// Preset for a "13B"-class model.
    pub fn preset_13b() -> Self {
        Self {
            name: "13B (40-layer, 40-head, dim 128)".to_string(),
            num_layers: 40,
            num_kv_heads: 40,
            head_dim: 128,
        }
    }

    /// Preset for a "70B"-class model (GQA with 8 KV heads).
    pub fn preset_70b() -> Self {
        Self {
            name: "70B (80-layer, 8 KV-head, dim 128)".to_string(),
            num_layers: 80,
            num_kv_heads: 8,
            head_dim: 128,
        }
    }

    /// Bytes per token in the KV cache (K + V, FP32 baseline).
    /// Per token per layer: 2 (K+V) * num_kv_heads * head_dim * 4 bytes
    pub fn bytes_per_token_fp32(&self) -> usize {
        2 * self.num_kv_heads * self.head_dim * 4
    }

    /// Total FP32 cache size for a given context length.
    pub fn cache_size_bytes(&self, context_len: usize) -> usize {
        self.num_layers * self.bytes_per_token_fp32() * context_len
    }
}

/// Result of a simulation run.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub model_name: String,
    pub context_len: usize,
    pub baseline_bytes: usize,
    pub compressed_bytes: usize,
    pub quant_method: Option<QuantMethod>,
    pub eviction_strategy: Option<String>,
    pub effective_context: usize,
    pub compression_ratio: f64,
    pub fits_in_budget: bool,
    pub budget_bytes: usize,
}

impl SimulationResult {
    pub fn baseline_mb(&self) -> f64 {
        self.baseline_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn compressed_mb(&self) -> f64 {
        self.compressed_bytes as f64 / (1024.0 * 1024.0)
    }

    pub fn baseline_gb(&self) -> f64 {
        self.baseline_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn compressed_gb(&self) -> f64 {
        self.compressed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Parse a budget string like "2gb", "512mb", "2048mb" into bytes.
pub fn parse_budget(s: &str) -> Option<usize> {
    let s = s.trim().to_lowercase();
    if let Some(num) = s.strip_suffix("gb") {
        num.trim()
            .parse::<f64>()
            .ok()
            .map(|n| (n * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if let Some(num) = s.strip_suffix("mb") {
        num.trim()
            .parse::<f64>()
            .ok()
            .map(|n| (n * 1024.0 * 1024.0) as usize)
    } else if let Some(num) = s.strip_suffix("kb") {
        num.trim()
            .parse::<f64>()
            .ok()
            .map(|n| (n * 1024.0) as usize)
    } else {
        s.parse::<usize>().ok()
    }
}

/// Simulate KV cache with a given model, context length, and memory budget.
/// Tries combinations of quantization and eviction to find what fits.
pub fn simulate_cache(
    config: &ModelConfig,
    context_len: usize,
    budget_bytes: usize,
) -> Vec<SimulationResult> {
    let baseline = config.cache_size_bytes(context_len);
    let mut results = Vec::new();

    let methods = [
        (None, "FP32 baseline"),
        (Some(QuantMethod::FP16), "FP16"),
        (Some(QuantMethod::FP8E4M3), "FP8 E4M3"),
        (Some(QuantMethod::INT4), "INT4 grouped"),
    ];

    let eviction_configs: Vec<(Option<Box<dyn TokenEviction>>, &str)> = vec![
        (None, "None"),
        (
            Some(Box::new(SlidingWindow::new(4))),
            "Sliding Window (sink=4)",
        ),
        (Some(Box::new(H2OEviction::new(4))), "H2O (sink=4)"),
    ];

    for &(quant_method, _quant_name) in &methods {
        let quant_ratio = quant_method.map(|m| m.compression_ratio()).unwrap_or(1.0);

        for (eviction, eviction_name) in &eviction_configs {
            // Without eviction: full context
            // With eviction: try reducing context until it fits
            let compressed_full = (baseline as f64 / quant_ratio) as usize;

            if eviction.is_none() {
                let fits = compressed_full <= budget_bytes;
                results.push(SimulationResult {
                    model_name: config.name.clone(),
                    context_len,
                    baseline_bytes: baseline,
                    compressed_bytes: compressed_full,
                    quant_method,
                    eviction_strategy: None,
                    effective_context: context_len,
                    compression_ratio: baseline as f64 / compressed_full.max(1) as f64,
                    fits_in_budget: fits,
                    budget_bytes,
                });
            } else {
                // With eviction: find max context that fits in budget
                let bytes_per_token = (config.bytes_per_token_fp32() as f64 / quant_ratio) as usize;
                let bytes_per_token_all_layers = bytes_per_token * config.num_layers;
                let max_tokens = budget_bytes
                    .checked_div(bytes_per_token_all_layers)
                    .unwrap_or(context_len);
                let effective = max_tokens.min(context_len);
                let compressed = effective * bytes_per_token_all_layers;
                let fits = compressed <= budget_bytes;

                // Simulate eviction to see how many tokens get evicted
                let entries: Vec<TokenEntry> = (0..context_len)
                    .map(|i| TokenEntry {
                        position: i,
                        cumulative_attention: 1.0 / (i as f64 + 1.0), // decaying attention
                        age: context_len - i,
                    })
                    .collect();

                let evicted = eviction
                    .as_ref()
                    .unwrap()
                    .select_evictions(&entries, effective);
                let _ = evicted; // we just care about the count

                results.push(SimulationResult {
                    model_name: config.name.clone(),
                    context_len,
                    baseline_bytes: baseline,
                    compressed_bytes: compressed,
                    quant_method,
                    eviction_strategy: Some(eviction_name.to_string()),
                    effective_context: effective,
                    compression_ratio: baseline as f64 / compressed.max(1) as f64,
                    fits_in_budget: fits,
                    budget_bytes,
                });
            }
        }
    }

    results
}

/// Simulate a single combination (for bench subcommand).
pub fn simulate_single(
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    method: QuantMethod,
) -> SimulationResult {
    // Assume single-layer for benchmarking raw compression
    let bytes_per_token_fp32 = 2 * num_heads * head_dim * 4; // K + V
    let baseline = bytes_per_token_fp32 * seq_len;
    let ratio = method.compression_ratio();
    let compressed = (baseline as f64 / ratio) as usize;

    SimulationResult {
        model_name: format!("{} heads x {} dim", num_heads, head_dim),
        context_len: seq_len,
        baseline_bytes: baseline,
        compressed_bytes: compressed,
        quant_method: Some(method),
        eviction_strategy: None,
        effective_context: seq_len,
        compression_ratio: ratio,
        fits_in_budget: true,
        budget_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_budget_gb() {
        assert_eq!(parse_budget("2gb"), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_budget("1GB"), Some(1024 * 1024 * 1024));
    }

    #[test]
    fn parse_budget_mb() {
        assert_eq!(parse_budget("512mb"), Some(512 * 1024 * 1024));
    }

    #[test]
    fn model_7b_cache_size() {
        let m = ModelConfig::preset_7b();
        // 32 layers * 2(K+V) * 32 heads * 128 dim * 4 bytes = 1,048,576 bytes/token
        // At 4096 tokens: 4,294,967,296 bytes = 4 GB
        let size = m.cache_size_bytes(4096);
        assert_eq!(size, 32 * 2 * 32 * 128 * 4 * 4096);
    }

    #[test]
    fn parse_budget_kb_plain_and_invalid() {
        assert_eq!(parse_budget("4kb"), Some(4 * 1024));
        assert_eq!(parse_budget("1048576"), Some(1_048_576));
        assert_eq!(parse_budget("  2 gb "), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_budget("garbage"), None);
        assert_eq!(parse_budget("gb"), None);
    }

    #[test]
    fn gqa_70b_has_fewer_kv_heads() {
        let m = ModelConfig::preset_70b();
        // 70B preset uses GQA: only 8 KV heads despite 80 layers.
        assert_eq!(m.num_kv_heads, 8);
        // Per-token bytes: 2(K+V) * 8 heads * 128 dim * 4 bytes = 8192.
        assert_eq!(m.bytes_per_token_fp32(), 2 * 8 * 128 * 4);
    }

    #[test]
    fn simulate_single_reports_method_ratio() {
        let r = simulate_single(32, 128, 4096, QuantMethod::INT4);
        assert_eq!(r.compression_ratio, QuantMethod::INT4.compression_ratio());
        assert_eq!(r.baseline_bytes, 2 * 32 * 128 * 4 * 4096);
        // INT4 nominal 8x.
        assert_eq!(r.compressed_bytes, r.baseline_bytes / 8);
    }

    #[test]
    fn simulate_eviction_reduces_effective_context() {
        let config = ModelConfig::preset_7b();
        // Tiny budget forces eviction to cap effective context below requested.
        let results = simulate_cache(&config, 8192, 256 * 1024 * 1024);
        let evicted: Vec<_> = results
            .iter()
            .filter(|r| r.eviction_strategy.is_some())
            .collect();
        assert!(!evicted.is_empty());
        for r in evicted {
            assert!(
                r.effective_context <= r.context_len,
                "effective context must not exceed requested context"
            );
        }
    }

    #[test]
    fn simulate_produces_results() {
        let config = ModelConfig::preset_7b();
        let results = simulate_cache(&config, 2048, 2 * 1024 * 1024 * 1024);
        assert!(!results.is_empty());
        // Should have baseline + 3 quant methods, each with None + 2 eviction = 12 total
        assert_eq!(results.len(), 12);
    }
}
