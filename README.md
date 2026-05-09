# kv-squeeze

KV cache compression library and CLI for LLM inference on memory-constrained devices.

## Features

- **Quantization methods**: FP16, FP8 (E4M3), INT4 with round-trip accuracy measurement
- **Eviction strategies**: sliding window (with attention sinks), H2O (heavy-hitter oracle), random baseline
- **Memory simulator**: projects cache sizes for 7B/13B/70B model presets against a memory budget
- Compare mode: side-by-side accuracy, throughput, and projected savings across all methods
- Reports MSE, max error, mean error, compression ratio, and throughput (MB/s)
- 25 tests covering quantization round-trips, eviction logic, and simulation
- Library crate (`kv_squeeze`) usable as a dependency

## Install

```
cargo build --release
```

## Usage

```
# Benchmark a single quantization method
kv-squeeze bench --heads 32 --dim 128 --seq-len 4096 --method fp8

# Compare all methods on the same data
kv-squeeze compare --heads 32 --dim 128 --seq-len 4096

# Simulate what fits in a memory budget
kv-squeeze simulate --model 7b --context 8192 --budget 2gb
```

### Subcommands

| Command    | Description                                              |
|------------|----------------------------------------------------------|
| `bench`    | Run compression benchmark with a chosen method           |
| `compare`  | Compare FP16, FP8, INT4 accuracy, speed, and savings    |
| `simulate` | Check which quant+eviction combos fit a memory budget    |

### Running tests

```
cargo test
```

## Integration: recursive-routing-racer-rs

kv-squeeze is wired into [recursive-routing-racer-rs](../recursive-routing-racer-rs/) as a local path dependency, providing KV cache compression and eviction for the Vulkan compute Phi-4 inference engine.

CLI flags:

```
rrr --kv-compress <none|fp16|fp8|int4>    # Compression mode (default: none)
rrr --kv-evict <none|sliding|h2o>         # Eviction strategy (default: none)
rrr --kv-budget <N>                       # Max cache entries before eviction
```

When active, compression stats are printed on exit: tokens inserted/evicted, peak cache size, compression ratio, and memory saved vs FP32 baseline.

---

Built with Rust + half + rayon.
