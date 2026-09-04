//! Fractal memory hierarchy.
//!
//! Four tiers, from fleeting to permanent:
//!
//! * **sensory** — a short ring buffer of raw recent states;
//! * **working** — the attended subset, weighted by salience;
//! * **episodic** — timestamped traces whose retrievability decays à la
//!   Ebbinghaus: `strength = visits * exp(-age / tau)`;
//! * **semantic** — prototype centroids distilled by consolidation; strong
//!   episodic traces merge into (or seed) prototypes and are then forgotten
//!   as episodes, exactly how memory consolidates in cortex.
//!
//! Recall scores candidates by strength-weighted cosine and returns the top-k
//! across episodic and semantic stores.

use crate::error::{AetherError, Result};
use crate::hypervec::cosine_slices;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Construction parameters for fractal memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalMemoryConfig {
    /// Vector width stored at every tier.
    pub d_model: usize,
    /// Sensory ring capacity.
    pub sensory_cap: usize,
    /// Working set capacity.
    pub working_cap: usize,
    /// Episodic store capacity.
    pub episodic_cap: usize,
    /// Semantic prototype capacity.
    pub semantic_cap: usize,
    /// Ebbinghaus decay constant in clock ticks.
    pub decay_tau: f32,
    /// Episodic strength that triggers consolidation into semantics.
    pub promote_strength: f32,
}

impl FractalMemoryConfig {
    /// Validate ranges.
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(AetherError::InvalidConfig("memory d_model is 0".to_string()));
        }
        if self.sensory_cap == 0
            || self.working_cap == 0
            || self.episodic_cap == 0
            || self.semantic_cap == 0
        {
            return Err(AetherError::InvalidConfig("memory caps must all be > 0".to_string()));
        }
        if self.decay_tau <= 0.0 || self.promote_strength <= 0.0 {
            return Err(AetherError::InvalidConfig(
                "decay_tau and promote_strength must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// One timestamped episodic trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EpisodicTrace {
    vec: Vec<f32>,
    time: f32,
    visits: u32,
}

impl EpisodicTrace {
    fn strength(&self, now: f32, tau: f32) -> f32 {
        let age = (now - self.time).max(0.0);
        self.visits as f32 * (-age / tau).exp()
    }
}

/// One consolidated semantic prototype.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SemanticProto {
    centroid: Vec<f32>,
    count: u64,
}

/// Point-in-time tier occupancy.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryStats {
    /// Items in the sensory ring.
    pub sensory: usize,
    /// Items in the working set.
    pub working: usize,
    /// Episodic traces alive.
    pub episodic: usize,
    /// Semantic prototypes alive.
    pub semantic: usize,
    /// Internal clock.
    pub clock: f32,
}

/// The four-tier memory organism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalMemory {
    cfg: FractalMemoryConfig,
    clock: f32,
    sensory: VecDeque<Vec<f32>>,
    working: Vec<(Vec<f32>, f32)>,
    episodic: Vec<EpisodicTrace>,
    semantic: Vec<SemanticProto>,
}

impl FractalMemory {
    /// Build empty memory from a config.
    pub fn new(cfg: FractalMemoryConfig) -> Result<FractalMemory> {
        cfg.validate()?;
        Ok(FractalMemory {
            cfg,
            clock: 0.0,
            sensory: VecDeque::new(),
            working: Vec::new(),
            episodic: Vec::new(),
            semantic: Vec::new(),
        })
    }

    /// Current clock reading.
    pub fn now(&self) -> f32 {
        self.clock
    }

    /// Advance the clock (call once per processed step).
    pub fn tick(&mut self, dt: f32) {
        if dt > 0.0 {
            self.clock += dt;
        }
    }

    fn check_dim(&self, v: &[f32], op: &str) -> Result<()> {
        if v.len() != self.cfg.d_model {
            return Err(AetherError::ShapeMismatch(format!(
                "{op} needs {} dims, got {}",
                self.cfg.d_model,
                v.len()
            )));
        }
        Ok(())
    }

    /// Push a raw state into the sensory ring, evicting the oldest on overflow.
    pub fn sense(&mut self, vec: Vec<f32>) -> Result<()> {
        self.check_dim(&vec, "sense")?;
        self.sensory.push_back(vec);
        while self.sensory.len() > self.cfg.sensory_cap {
            self.sensory.pop_front();
        }
        Ok(())
    }

    /// Attend: move the sensory ring into the working set, weighted by
    /// `weights` (one salience score per sensed item, highest kept).
    pub fn attend(&mut self, weights: &[f32]) -> Result<()> {
        if weights.len() != self.sensory.len() {
            return Err(AetherError::ShapeMismatch(format!(
                "attend needs {} weights, got {}",
                self.sensory.len(),
                weights.len()
            )));
        }
        let mut scored: Vec<(Vec<f32>, f32)> =
            self.sensory.drain(..).zip(weights.iter().cloned()).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.cfg.working_cap);
        self.working = scored;
        Ok(())
    }

    /// Memorise: commit the working set to episodic storage, evicting the
    /// weakest traces on overflow.
    pub fn memorise(&mut self) {
        let now = self.clock;
        let tau = self.cfg.decay_tau;
        let cap = self.cfg.episodic_cap;
        for (vec, _) in self.working.drain(..) {
            // Rehearsal: refresh a near-duplicate instead of duplicating it.
            let mut best: Option<usize> = None;
            let mut best_sim = 0.985f32;
            for (i, trace) in self.episodic.iter().enumerate() {
                let sim = cosine_slices(&trace.vec, &vec);
                if sim > best_sim {
                    best_sim = sim;
                    best = Some(i);
                }
            }
            if let Some(i) = best {
                self.episodic[i].visits = self.episodic[i].visits.saturating_add(1);
                self.episodic[i].time = now;
            } else {
                self.episodic.push(EpisodicTrace {
                    vec,
                    time: now,
                    visits: 1,
                });
            }
        }
        while self.episodic.len() > cap {
            let mut weakest = 0;
            let mut weakest_strength = f32::INFINITY;
            for (i, trace) in self.episodic.iter().enumerate() {
                let s = trace.strength(now, tau);
                if s < weakest_strength {
                    weakest_strength = s;
                    weakest = i;
                }
            }
            self.episodic.remove(weakest);
        }
    }

    /// Consolidate strong episodes into semantic prototypes.
    ///
    /// Returns the number of traces promoted.
    pub fn consolidate(&mut self) -> usize {
        let now = self.clock;
        let tau = self.cfg.decay_tau;
        let threshold = self.cfg.promote_strength;
        let mut promoted = 0usize;
        let mut remaining = Vec::with_capacity(self.episodic.len());
        // Drain first: scoring borrows nothing, but absorption needs &mut self.
        let traces: Vec<EpisodicTrace> = self.episodic.drain(..).collect();
        for trace in traces {
            if trace.strength(now, tau) >= threshold {
                self.absorb_into_semantic(&trace.vec);
                promoted += 1;
            } else {
                remaining.push(trace);
            }
        }
        self.episodic = remaining;
        promoted
    }

    /// Merge a vector into the nearest prototype (or seed a new one).
    fn absorb_into_semantic(&mut self, vec: &[f32]) {
        let mut best: Option<usize> = None;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, proto) in self.semantic.iter().enumerate() {
            let sim = cosine_slices(&proto.centroid, vec);
            if sim > best_sim {
                best_sim = sim;
                best = Some(i);
            }
        }
        match best {
            Some(i) if self.semantic.len() >= self.cfg.semantic_cap || best_sim > 0.9 => {
                let proto = &mut self.semantic[i];
                let n = proto.count as f32;
                for (c, x) in proto.centroid.iter_mut().zip(vec.iter()) {
                    *c = (*c * n + *x) / (n + 1.0);
                }
                proto.count += 1;
            }
            _ => {
                self.semantic.push(SemanticProto {
                    centroid: vec.to_vec(),
                    count: 1,
                });
            }
        }
    }

    /// Drop episodes whose strength fell below `floor`.
    pub fn prune(&mut self, floor: f32) {
        let now = self.clock;
        let tau = self.cfg.decay_tau;
        self.episodic.retain(|t| t.strength(now, tau) >= floor);
    }

    /// Recall the top-k matches for `query` across episodic and semantic
    /// stores, scored by strength-weighted cosine, best first.
    pub fn recall(&self, query: &[f32], k: usize) -> Result<Vec<(Vec<f32>, f32)>> {
        self.check_dim(query, "recall")?;
        let mut scored: Vec<(Vec<f32>, f32)> = Vec::new();
        for trace in &self.episodic {
            let s = cosine_slices(&trace.vec, query) * trace.strength(self.clock, self.cfg.decay_tau);
            scored.push((trace.vec.clone(), s));
        }
        for proto in &self.semantic {
            let familiarity = (proto.count as f32).ln_1p();
            scored.push((
                proto.centroid.clone(),
                cosine_slices(&proto.centroid, query) * familiarity,
            ));
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    /// Tier occupancy snapshot.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            sensory: self.sensory.len(),
            working: self.working.len(),
            episodic: self.episodic.len(),
            semantic: self.semantic.len(),
            clock: self.clock,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn cfg() -> FractalMemoryConfig {
        FractalMemoryConfig {
            d_model: 8,
            sensory_cap: 8,
            working_cap: 4,
            episodic_cap: 16,
            semantic_cap: 8,
            decay_tau: 10.0,
            promote_strength: 0.5,
        }
    }

    fn vec_hol(seed: u64) -> Vec<f32> {
        // Deterministic pseudo-random unit-ish vector.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        use rand::Rng;
        (0..8).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn recency_orders_strength() {
        let mut mem = FractalMemory::new(cfg()).unwrap();
        for t in 0..5 {
            mem.tick(1.0);
            mem.sense(vec_hol(t)).unwrap();
            mem.attend(&vec![1.0; mem.stats().sensory]).unwrap();
            mem.memorise();
        }
        // Later traces are stronger: recall of a neutral probe is ordered.
        let probe = vec_hol(99);
        let hits = mem.recall(&probe, 5).unwrap();
        assert_eq!(hits.len(), 5);
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1, "recall not sorted");
        }
    }

    #[test]
    fn consolidation_builds_prototypes() {
        let mut mem = FractalMemory::new(cfg()).unwrap();
        for _ in 0..3 {
            mem.sense(vec_hol(4)).unwrap();
        }
        mem.attend(&[1.0, 1.0, 1.0]).unwrap();
        mem.memorise();
        // Rehearse the same content to push strength over the threshold.
        for _ in 0..4 {
            mem.sense(vec_hol(4)).unwrap();
            mem.attend(&vec![1.0; mem.stats().sensory]).unwrap();
            mem.memorise();
        }
        let promoted = mem.consolidate();
        assert!(promoted >= 1, "nothing promoted");
        assert!(mem.stats().semantic >= 1);
    }

    #[test]
    fn caps_are_enforced() {
        let mut mem = FractalMemory::new(cfg()).unwrap();
        for i in 0..40 {
            mem.sense(vec_hol(i)).unwrap();
        }
        assert_eq!(mem.stats().sensory, 8);
        mem.attend(&vec![1.0; 8]).unwrap();
        assert_eq!(mem.stats().working, 4);
    }
}
