use clap::{Parser, Subcommand};
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Cell, Color, Table};
use std::time::Instant;

use kv_squeeze::quantize::{self, CompressionStats, QuantMethod};
use kv_squeeze::simulator::{self, ModelConfig, SimulationResult};

#[derive(Parser)]
#[command(name = "kv-squeeze")]
#[command(about = "KV cache compression benchmarks for memory-constrained LLM inference")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run compression benchmark on synthetic KV cache data.
    Bench {
        /// Number of attention heads.
        #[arg(long, default_value_t = 32)]
        heads: usize,

        /// Head dimension.
        #[arg(long, default_value_t = 128)]
        dim: usize,

        /// Sequence length (number of tokens).
        #[arg(long, default_value_t = 4096)]
        seq_len: usize,

        /// Quantization method: fp16, fp8, int4
        #[arg(long, default_value = "fp8")]
        method: String,
    },

    /// Simulate what fits in a memory budget for a given model.
    Simulate {
        /// Model size preset: 7b, 13b, 70b
        #[arg(long, default_value = "7b")]
        model: String,

        /// Context length.
        #[arg(long, default_value_t = 8192)]
        context: usize,

        /// Memory budget (e.g., "2gb", "512mb").
        #[arg(long, default_value = "2gb")]
        budget: String,
    },

    /// Compare all quantization methods on the same data.
    Compare {
        /// Number of attention heads.
        #[arg(long, default_value_t = 32)]
        heads: usize,

        /// Head dimension.
        #[arg(long, default_value_t = 128)]
        dim: usize,

        /// Sequence length.
        #[arg(long, default_value_t = 4096)]
        seq_len: usize,
    },
}

fn parse_method(s: &str) -> Result<QuantMethod, String> {
    match s.to_lowercase().as_str() {
        "fp16" => Ok(QuantMethod::FP16),
        "fp8" | "fp8e4m3" | "e4m3" => Ok(QuantMethod::FP8E4M3),
        "int4" | "i4" => Ok(QuantMethod::INT4),
        other => Err(format!(
            "Unknown method '{}'. Use: fp16, fp8, int4",
            other
        )),
    }
}

fn parse_model(s: &str) -> Result<ModelConfig, String> {
    match s.to_lowercase().as_str() {
        "7b" => Ok(ModelConfig::preset_7b()),
        "13b" => Ok(ModelConfig::preset_13b()),
        "70b" => Ok(ModelConfig::preset_70b()),
        other => Err(format!(
            "Unknown model '{}'. Use: 7b, 13b, 70b",
            other
        )),
    }
}

/// Generate deterministic pseudo-random f32 data simulating KV cache values.
/// Values are normally-distributed-ish in roughly [-2, 2] range.
fn generate_kv_data(num_elements: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(num_elements);
    let mut state: u64 = 0xDEADBEEF_CAFEBABE;
    for _ in 0..num_elements {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Map to [-2, 2] range (typical KV cache value range)
        let f = (state as f32 / u64::MAX as f32) * 4.0 - 2.0;
        data.push(f);
    }
    data
}

fn run_bench(heads: usize, dim: usize, seq_len: usize, method: QuantMethod) {
    let num_elements = 2 * heads * dim * seq_len; // K + V
    println!(
        "Generating synthetic KV cache: {} heads, dim {}, {} tokens",
        heads, dim, seq_len
    );
    println!(
        "Total elements: {} ({:.1} MB FP32)\n",
        num_elements,
        num_elements as f64 * 4.0 / (1024.0 * 1024.0)
    );

    let data = generate_kv_data(num_elements);

    let start = Instant::now();
    let stats = quantize::round_trip_stats(&data, method);
    let elapsed = start.elapsed();

    print_stats_table(&[stats.clone()]);

    println!("\nThroughput: {:.1} MB/s (compress + decompress + measure)",
        (num_elements as f64 * 4.0 / (1024.0 * 1024.0)) / elapsed.as_secs_f64()
    );
    println!("Wall time:  {:.1} ms", elapsed.as_secs_f64() * 1000.0);

    // Also show simulation result
    let sim = simulator::simulate_single(heads, dim, seq_len, method);
    println!("\nMemory impact (single layer):");
    println!(
        "  FP32 baseline: {:.1} MB",
        sim.baseline_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  {}: {:.1} MB ({:.1}x compression)",
        method,
        sim.compressed_bytes as f64 / (1024.0 * 1024.0),
        sim.compression_ratio
    );
}

fn print_stats_table(stats_list: &[CompressionStats]) {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    table.set_header(vec![
        "Method",
        "MSE",
        "Max Error",
        "Mean Error",
        "Original",
        "Compressed",
        "Ratio",
    ]);

    for stats in stats_list {
        table.add_row(vec![
            Cell::new(stats.method.to_string()),
            Cell::new(format!("{:.2e}", stats.mse)),
            Cell::new(format!("{:.4e}", stats.max_error)),
            Cell::new(format!("{:.4e}", stats.mean_error)),
            Cell::new(format_bytes(stats.original_bytes)),
            Cell::new(format_bytes(stats.compressed_bytes)),
            Cell::new(format!("{:.2}x", stats.compression_ratio())),
        ]);
    }

    println!("{table}");
}

fn run_simulate(model: ModelConfig, context: usize, budget_str: &str) {
    let budget = simulator::parse_budget(budget_str).unwrap_or_else(|| {
        eprintln!(
            "Invalid budget '{}'. Use format like '2gb' or '512mb'.",
            budget_str
        );
        std::process::exit(1);
    });

    println!("Model:   {}", model.name);
    println!("Context: {} tokens", context);
    println!("Budget:  {} ({})", budget_str, format_bytes(budget));
    println!();

    let results = simulator::simulate_cache(&model, context, budget);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    table.set_header(vec![
        "Quant",
        "Eviction",
        "Eff. Context",
        "Cache Size",
        "Ratio",
        "Fits?",
    ]);

    for r in &results {
        let quant_str = r
            .quant_method
            .map(|m| m.to_string())
            .unwrap_or_else(|| "FP32".to_string());
        let eviction_str = r
            .eviction_strategy
            .as_deref()
            .unwrap_or("None");
        let fits_cell = if r.fits_in_budget {
            Cell::new("YES").fg(Color::Green)
        } else {
            Cell::new("NO").fg(Color::Red)
        };

        table.add_row(vec![
            Cell::new(&quant_str),
            Cell::new(eviction_str),
            Cell::new(format!("{}", r.effective_context)),
            Cell::new(format_bytes(r.compressed_bytes)),
            Cell::new(format!("{:.1}x", r.compression_ratio)),
            fits_cell,
        ]);
    }

    println!("{table}");

    // Summary
    let fitting: Vec<&SimulationResult> = results.iter().filter(|r| r.fits_in_budget).collect();
    println!(
        "\n{} of {} configurations fit within {}.",
        fitting.len(),
        results.len(),
        budget_str
    );

    if let Some(best) = fitting.last() {
        println!(
            "Best compression: {:.1}x ({} + {}), effective context: {} tokens, cache: {}",
            best.compression_ratio,
            best.quant_method
                .map(|m| m.to_string())
                .unwrap_or_else(|| "FP32".to_string()),
            best.eviction_strategy.as_deref().unwrap_or("None"),
            best.effective_context,
            format_bytes(best.compressed_bytes)
        );
    }
}

fn run_compare(heads: usize, dim: usize, seq_len: usize) {
    let num_elements = 2 * heads * dim * seq_len;
    println!(
        "Comparing all methods: {} heads, dim {}, {} tokens",
        heads, dim, seq_len
    );
    println!(
        "Total elements: {} ({:.1} MB FP32)\n",
        num_elements,
        num_elements as f64 * 4.0 / (1024.0 * 1024.0)
    );

    let data = generate_kv_data(num_elements);

    let methods = [QuantMethod::FP16, QuantMethod::FP8E4M3, QuantMethod::INT4];

    let mut all_stats = Vec::new();
    let mut timings = Vec::new();

    for method in &methods {
        let start = Instant::now();
        let stats = quantize::round_trip_stats(&data, *method);
        let elapsed = start.elapsed();
        all_stats.push(stats);
        timings.push(elapsed);
    }

    // Accuracy table
    println!("=== Round-trip Accuracy ===\n");
    print_stats_table(&all_stats);

    // Timing table
    println!("\n=== Performance ===\n");
    let mut perf_table = Table::new();
    perf_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    perf_table.set_header(vec!["Method", "Time (ms)", "Throughput (MB/s)"]);

    let data_mb = num_elements as f64 * 4.0 / (1024.0 * 1024.0);
    for (i, method) in methods.iter().enumerate() {
        let ms = timings[i].as_secs_f64() * 1000.0;
        let throughput = data_mb / timings[i].as_secs_f64();
        perf_table.add_row(vec![
            Cell::new(method.to_string()),
            Cell::new(format!("{:.1}", ms)),
            Cell::new(format!("{:.0}", throughput)),
        ]);
    }

    println!("{perf_table}");

    // Memory impact table (multi-layer model)
    println!("\n=== Projected Memory Savings (32-layer model) ===\n");
    let mut mem_table = Table::new();
    mem_table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    mem_table.set_header(vec!["Method", "Cache Size", "Savings", "Ratio"]);

    let model = ModelConfig::preset_7b();
    let baseline = model.cache_size_bytes(seq_len);

    mem_table.add_row(vec![
        Cell::new("FP32 (baseline)"),
        Cell::new(format_bytes(baseline)),
        Cell::new("-"),
        Cell::new("1.0x"),
    ]);

    for method in &methods {
        let ratio = method.compression_ratio();
        let compressed = (baseline as f64 / ratio) as usize;
        let saved = baseline - compressed;
        mem_table.add_row(vec![
            Cell::new(method.to_string()),
            Cell::new(format_bytes(compressed)),
            Cell::new(format!("-{}", format_bytes(saved))),
            Cell::new(format!("{:.1}x", ratio)),
        ]);
    }

    println!("{mem_table}");
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Bench {
            heads,
            dim,
            seq_len,
            method,
        } => {
            let m = parse_method(&method).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            });
            run_bench(heads, dim, seq_len, m);
        }
        Commands::Simulate {
            model,
            context,
            budget,
        } => {
            let config = parse_model(&model).unwrap_or_else(|e| {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            });
            run_simulate(config, context, &budget);
        }
        Commands::Compare {
            heads,
            dim,
            seq_len,
        } => {
            run_compare(heads, dim, seq_len);
        }
    }
}
