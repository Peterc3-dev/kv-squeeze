use rand::seq::SliceRandom;
use rand::thread_rng;

/// A token entry in the KV cache with metadata for eviction decisions.
#[derive(Debug, Clone)]
pub struct TokenEntry {
    pub position: usize,
    pub cumulative_attention: f64,
    pub age: usize, // how many steps since insertion
}

/// Result of an eviction decision: which token positions to evict.
pub trait TokenEviction {
    /// Given the current cache state and a target size, return indices to evict.
    /// `entries` is the current set of cached tokens.
    /// `target_size` is how many tokens should remain after eviction.
    fn select_evictions(&self, entries: &[TokenEntry], target_size: usize) -> Vec<usize>;

    /// Name of the strategy for display.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Sliding Window
// ---------------------------------------------------------------------------

/// Keep the first `sink_count` tokens (attention sinks) and the most recent tokens
/// to fill the remaining budget.
pub struct SlidingWindow {
    pub sink_count: usize,
}

impl SlidingWindow {
    pub fn new(sink_count: usize) -> Self {
        Self { sink_count }
    }
}

impl TokenEviction for SlidingWindow {
    fn select_evictions(&self, entries: &[TokenEntry], target_size: usize) -> Vec<usize> {
        if entries.len() <= target_size {
            return vec![];
        }

        let to_evict = entries.len() - target_size;

        // Partition: sinks (first K by position) and the rest
        let mut by_position: Vec<(usize, usize)> = entries
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx, e.position))
            .collect();
        by_position.sort_by_key(|&(_, pos)| pos);

        // Protect the first `sink_count` positions
        let protected: usize = self.sink_count.min(entries.len());
        let sink_positions: Vec<usize> = by_position[..protected]
            .iter()
            .map(|&(idx, _)| idx)
            .collect();

        // Among non-sink tokens, evict the oldest (lowest position = inserted earliest,
        // but sinks excluded, so evict those with lowest position among non-sinks)
        let mut candidates: Vec<(usize, usize)> = by_position[protected..].to_vec();
        // Evict from the front of candidates (oldest non-sink tokens)
        candidates.sort_by_key(|&(_, pos)| pos);

        let evict_count = to_evict.min(candidates.len());
        let _ = sink_positions; // sinks are protected
        candidates[..evict_count]
            .iter()
            .map(|&(idx, _)| idx)
            .collect()
    }

    fn name(&self) -> &str {
        "Sliding Window"
    }
}

// ---------------------------------------------------------------------------
// H2O (Heavy Hitter Oracle)
// ---------------------------------------------------------------------------

/// Evict tokens with the lowest cumulative attention scores.
/// Always protects the first `sink_count` positions.
pub struct H2OEviction {
    pub sink_count: usize,
}

impl H2OEviction {
    pub fn new(sink_count: usize) -> Self {
        Self { sink_count }
    }
}

impl TokenEviction for H2OEviction {
    fn select_evictions(&self, entries: &[TokenEntry], target_size: usize) -> Vec<usize> {
        if entries.len() <= target_size {
            return vec![];
        }

        let to_evict = entries.len() - target_size;

        // Partition by position to identify sinks
        let mut by_position: Vec<(usize, usize)> = entries
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx, e.position))
            .collect();
        by_position.sort_by_key(|&(_, pos)| pos);

        let protected = self.sink_count.min(entries.len());
        let sink_indices: std::collections::HashSet<usize> = by_position[..protected]
            .iter()
            .map(|&(idx, _)| idx)
            .collect();

        // Among non-sink tokens, sort by cumulative attention (ascending) and evict lowest
        let mut candidates: Vec<(usize, f64)> = entries
            .iter()
            .enumerate()
            .filter(|(idx, _)| !sink_indices.contains(idx))
            .map(|(idx, e)| (idx, e.cumulative_attention))
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let evict_count = to_evict.min(candidates.len());
        candidates[..evict_count]
            .iter()
            .map(|&(idx, _)| idx)
            .collect()
    }

    fn name(&self) -> &str {
        "H2O (Heavy Hitter Oracle)"
    }
}

// ---------------------------------------------------------------------------
// Random Eviction (baseline)
// ---------------------------------------------------------------------------

/// Randomly evict tokens, always protecting sinks.
pub struct RandomEviction {
    pub sink_count: usize,
}

impl RandomEviction {
    pub fn new(sink_count: usize) -> Self {
        Self { sink_count }
    }
}

impl TokenEviction for RandomEviction {
    fn select_evictions(&self, entries: &[TokenEntry], target_size: usize) -> Vec<usize> {
        if entries.len() <= target_size {
            return vec![];
        }

        let to_evict = entries.len() - target_size;

        let mut by_position: Vec<(usize, usize)> = entries
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx, e.position))
            .collect();
        by_position.sort_by_key(|&(_, pos)| pos);

        let protected = self.sink_count.min(entries.len());
        let sink_indices: std::collections::HashSet<usize> = by_position[..protected]
            .iter()
            .map(|&(idx, _)| idx)
            .collect();

        let mut candidates: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(idx, _)| !sink_indices.contains(idx))
            .map(|(idx, _)| idx)
            .collect();

        let mut rng = thread_rng();
        candidates.shuffle(&mut rng);

        let evict_count = to_evict.min(candidates.len());
        candidates[..evict_count].to_vec()
    }

    fn name(&self) -> &str {
        "Random"
    }
}

/// Convenience enum wrapping all strategies for CLI dispatch.
pub enum EvictionStrategy {
    Sliding(SlidingWindow),
    H2O(H2OEviction),
    Random(RandomEviction),
}

impl EvictionStrategy {
    pub fn as_eviction(&self) -> &dyn TokenEviction {
        match self {
            EvictionStrategy::Sliding(s) => s,
            EvictionStrategy::H2O(h) => h,
            EvictionStrategy::Random(r) => r,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries(n: usize) -> Vec<TokenEntry> {
        (0..n)
            .map(|i| TokenEntry {
                position: i,
                cumulative_attention: (i as f64 + 1.0) * 0.1, // higher position = more attention
                age: n - i,
            })
            .collect()
    }

    #[test]
    fn sliding_window_protects_sinks() {
        let entries = make_entries(10);
        let sw = SlidingWindow::new(2);
        let evicted = sw.select_evictions(&entries, 5);
        // Should evict 5 tokens, none from positions 0 or 1
        assert_eq!(evicted.len(), 5);
        for &idx in &evicted {
            assert!(
                entries[idx].position >= 2,
                "Evicted sink at position {}",
                entries[idx].position
            );
        }
    }

    #[test]
    fn sliding_window_no_eviction_if_under_budget() {
        let entries = make_entries(5);
        let sw = SlidingWindow::new(2);
        let evicted = sw.select_evictions(&entries, 10);
        assert!(evicted.is_empty());
    }

    #[test]
    fn h2o_evicts_lowest_attention() {
        let entries = make_entries(10);
        let h2o = H2OEviction::new(2);
        let evicted = h2o.select_evictions(&entries, 5);
        assert_eq!(evicted.len(), 5);

        // None should be sinks (positions 0, 1)
        for &idx in &evicted {
            assert!(entries[idx].position >= 2);
        }

        // Evicted should be the lowest-attention non-sink tokens (positions 2, 3, 4, 5, 6)
        let mut evicted_positions: Vec<usize> = evicted.iter().map(|&i| entries[i].position).collect();
        evicted_positions.sort();
        assert_eq!(evicted_positions, vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn random_eviction_protects_sinks() {
        let entries = make_entries(20);
        let re = RandomEviction::new(3);
        let evicted = re.select_evictions(&entries, 10);
        assert_eq!(evicted.len(), 10);
        for &idx in &evicted {
            assert!(
                entries[idx].position >= 3,
                "Random evicted sink at position {}",
                entries[idx].position
            );
        }
    }

    #[test]
    fn random_eviction_correct_count() {
        let entries = make_entries(100);
        let re = RandomEviction::new(4);
        let evicted = re.select_evictions(&entries, 50);
        assert_eq!(evicted.len(), 50);
    }
}
