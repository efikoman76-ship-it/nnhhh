//! The AETHER mind: the full cognitive stack.
//!
//! A mind embeds tokens with sinusoidal positions, then streams them through
//! `n_layers` blocks of
//!
//! ```text
//! x -> LayerNorm -> EntangledAttention -(+)-> Resonate -(+)->
//!     LayerNorm -> SparseMoE ---------(+)-> Homeostat -> sense gist
//! ```
//!
//! After the stack, three online-learning systems fire on every forward pass:
//! the [`HebbianBank`] updates on the last state and biases memory retrieval,
//! [`FractalMemory`] runs a full sense → attend → memorise → consolidate →
//! prune → recall cycle, and the recalled gist is blended back into the
//! stream. The mind is therefore *stateful*: two identical forwards in a row
//! see different memory. Snapshots (weights + resonance + memory) persist as
//! JSON.
//!
//! Because the resonant/holographic substrate is non-differentiable by design,
//! offline optimisation is evolutionary (`EvoTrainer` in the trainer module):
//! parameters flatten
//! to one vector via [`AetherMind::flat_params`] and restore with
//! [`AetherMind::set_flat_params`].

use crate::attention::{EntangledAttention, EntangledAttentionConfig};
use crate::error::{AetherError, Result};
use crate::memory::{FractalMemory, FractalMemoryConfig, MemoryStats};
use crate::moe::{SparseMoe, SparseMoeConfig};
use crate::plasticity::{HebbianBank, Homeostat};
use crate::resonance::{ResonantLayer, ResonantSpec};
use crate::tensor::Matrix;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;

/// Whole-model hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AetherConfig {
    /// Hidden width.
    pub d_model: usize,
    /// Attention heads.
    pub n_heads: usize,
    /// Stacked blocks.
    pub n_layers: usize,
    /// MoE expert hidden width.
    pub d_moe_hidden: usize,
    /// Experts per MoE layer.
    pub n_experts: usize,
    /// Experts active per token.
    pub top_k: usize,
    /// Local attention window diameter.
    pub window: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Longest accepted sequence.
    pub max_seq: usize,
    /// Sensory ring capacity.
    pub memory_sensory: usize,
    /// Working set capacity.
    pub memory_working: usize,
    /// Episodic capacity.
    pub memory_episodic: usize,
    /// Semantic prototype capacity.
    pub memory_semantic: usize,
    /// Episodic decay constant (ticks).
    pub decay_tau: f32,
    /// Master seed for every initialisation.
    pub seed: u64,
}

impl AetherConfig {
    /// A pocket mind for tests, examples and CI (a few thousand params).
    pub fn tiny() -> AetherConfig {
        AetherConfig {
            d_model: 16,
            n_heads: 4,
            n_layers: 1,
            d_moe_hidden: 32,
            n_experts: 4,
            top_k: 2,
            window: 8,
            vocab_size: 64,
            max_seq: 32,
            memory_sensory: 16,
            memory_working: 8,
            memory_episodic: 32,
            memory_semantic: 16,
            decay_tau: 10.0,
            seed: 7,
        }
    }

    /// A small but serious mind for real play.
    pub fn small() -> AetherConfig {
        AetherConfig {
            d_model: 64,
            n_heads: 8,
            n_layers: 3,
            d_moe_hidden: 128,
            n_experts: 6,
            top_k: 2,
            window: 16,
            vocab_size: 512,
            max_seq: 128,
            memory_sensory: 32,
            memory_working: 16,
            memory_episodic: 128,
            memory_semantic: 64,
            decay_tau: 24.0,
            seed: 7,
        }
    }

    /// Check every invariant before a mind is built.
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 {
            return Err(AetherError::InvalidConfig("d_model is 0".to_string()));
        }
        if self.n_heads == 0 || self.d_model % self.n_heads != 0 {
            return Err(AetherError::InvalidConfig(format!(
                "d_model {} incompatible with n_heads {}",
                self.d_model, self.n_heads
            )));
        }
        if self.n_layers == 0 {
            return Err(AetherError::InvalidConfig("n_layers is 0".to_string()));
        }
        if self.d_moe_hidden == 0 || self.n_experts == 0 {
            return Err(AetherError::InvalidConfig(
                "d_moe_hidden and n_experts must be > 0".to_string(),
            ));
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return Err(AetherError::InvalidConfig(format!(
                "top_k {} incompatible with n_experts {}",
                self.top_k, self.n_experts
            )));
        }
        if self.window == 0 || self.vocab_size < 4 || self.max_seq == 0 {
            return Err(AetherError::InvalidConfig(
                "need window > 0, vocab_size >= 4, max_seq > 0".to_string(),
            ));
        }
        if self.memory_sensory == 0
            || self.memory_working == 0
            || self.memory_episodic == 0
            || self.memory_semantic == 0
        {
            return Err(AetherError::InvalidConfig(
                "memory caps must all be > 0".to_string(),
            ));
        }
        if self.decay_tau <= 0.0 {
            return Err(AetherError::InvalidConfig(
                "decay_tau must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Sampling knobs for generation and dreaming.
#[derive(Debug, Clone, Copy)]
pub struct SampleConfig {
    /// Softmax temperature (0 = greedy).
    pub temperature: f32,
    /// Keep the top-k logits per step (0 = off).
    pub top_k: usize,
    /// Nucleus mass to keep (outside (0,1) = off).
    pub top_p: f32,
    /// Stop generating after emitting this id.
    pub stop_id: Option<usize>,
}

impl Default for SampleConfig {
    fn default() -> SampleConfig {
        SampleConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            stop_id: None,
        }
    }
}

/// One cognitive block.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AetherBlock {
    attn: EntangledAttention,
    moe: SparseMoe,
    gamma1: Vec<f32>,
    beta1: Vec<f32>,
    gamma2: Vec<f32>,
    beta2: Vec<f32>,
    resonance: ResonantLayer,
    homeostat: Homeostat,
}

/// Everything a forward pass returns.
#[derive(Debug, Clone)]
pub struct ForwardOut {
    /// Per-position next-token logits (`T x V`).
    pub logits: Matrix,
    /// Summed MoE load-balancing loss.
    pub aux_loss: f32,
    /// Final hidden states (`T x D`).
    pub hidden: Matrix,
}

/// The living mind: weights, oscillators, plasticity and memory in one organism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AetherMind {
    cfg: AetherConfig,
    token_embed: Matrix,
    blocks: Vec<AetherBlock>,
    out_gamma: Vec<f32>,
    out_beta: Vec<f32>,
    lm_head: Matrix,
    hebb_bank: HebbianBank,
    memory: FractalMemory,
}

impl AetherMind {
    /// Build a mind from a config; every weight is deterministically seeded.
    pub fn new(cfg: AetherConfig) -> Result<AetherMind> {
        cfg.validate()?;
        let d = cfg.d_model;
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for layer in 0..cfg.n_layers {
            let base = cfg.seed.wrapping_add(layer as u64 * 1_000);
            blocks.push(AetherBlock {
                attn: EntangledAttention::new(
                    EntangledAttentionConfig {
                        d_model: d,
                        n_heads: cfg.n_heads,
                        window: cfg.window,
                    },
                    base + 11,
                )?,
                moe: SparseMoe::new(
                    SparseMoeConfig {
                        d_model: d,
                        d_hidden: cfg.d_moe_hidden,
                        n_experts: cfg.n_experts,
                        top_k: cfg.top_k,
                        noise_std: 0.05,
                    },
                    base + 101,
                )?,
                gamma1: vec![1.0; d],
                beta1: vec![0.0; d],
                gamma2: vec![1.0; d],
                beta2: vec![0.0; d],
                resonance: ResonantLayer::new(
                    ResonantSpec {
                        n_units: d,
                        ..ResonantSpec::default()
                    },
                    base + 201,
                )?,
                homeostat: Homeostat::new(d, 0.0, 1.0, 0.01)?,
            });
        }
        Ok(AetherMind {
            token_embed: Matrix::xavier_seeded(cfg.seed + 7, cfg.vocab_size, d),
            blocks,
            out_gamma: vec![1.0; d],
            out_beta: vec![0.0; d],
            lm_head: Matrix::xavier_seeded(cfg.seed + 8, d, cfg.vocab_size),
            hebb_bank: HebbianBank::new(8, d, 0.005, cfg.seed + 9)?,
            memory: FractalMemory::new(FractalMemoryConfig {
                d_model: d,
                sensory_cap: cfg.memory_sensory,
                working_cap: cfg.memory_working,
                episodic_cap: cfg.memory_episodic,
                semantic_cap: cfg.memory_semantic,
                decay_tau: cfg.decay_tau,
                promote_strength: 0.6,
            })?,
            cfg,
        })
    }

    /// Borrow the config.
    pub fn config(&self) -> &AetherConfig {
        &self.cfg
    }

    /// Snapshot of memory occupancy.
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory.stats()
    }

    /// Mutable access to memory (for trainers and tools).
    pub fn memory_mut(&mut self) -> &mut FractalMemory {
        &mut self.memory
    }

    /// Mean token embedding — the semantic fingerprint of a token span.
    pub fn embed_mean(&self, ids: &[usize]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(AetherError::EmptyInput("embed_mean got no ids".to_string()));
        }
        let d = self.cfg.d_model;
        let mut acc = vec![0.0f32; d];
        for &id in ids {
            if id >= self.cfg.vocab_size {
                return Err(AetherError::Vocab(format!(
                    "token {id} out of vocab {}",
                    self.cfg.vocab_size
                )));
            }
            for (a, x) in acc.iter_mut().zip(self.token_embed.row(id).iter()) {
                *a += *x;
            }
        }
        let inv = 1.0 / ids.len() as f32;
        for a in acc.iter_mut() {
            *a *= inv;
        }
        Ok(acc)
    }

    /// Full forward pass with online plasticity and the memory cycle.
    pub fn forward(&mut self, ids: &[usize], rng: &mut StdRng) -> Result<ForwardOut> {
        if ids.is_empty() {
            return Err(AetherError::EmptyInput("forward got no ids".to_string()));
        }
        if ids.len() > self.cfg.max_seq {
            return Err(AetherError::InvalidConfig(format!(
                "sequence len {} exceeds max_seq {}",
                ids.len(),
                self.cfg.max_seq
            )));
        }
        for &id in ids {
            if id >= self.cfg.vocab_size {
                return Err(AetherError::Vocab(format!(
                    "token {id} out of vocab {}",
                    self.cfg.vocab_size
                )));
            }
        }
        let t = ids.len();
        let d = self.cfg.d_model;

        // Embed + sinusoidal positions.
        let mut x = Matrix::zeros(t, d);
        for (i, &id) in ids.iter().enumerate() {
            let pos = sinusoidal_pos(i, d);
            let mut row = self.token_embed.row(id).to_vec();
            for (a, p) in row.iter_mut().zip(pos.iter()) {
                *a += *p;
            }
            x.set_row(i, &row)?;
        }

        self.memory.tick(1.0);
        let mut aux_total = 0.0f32;
        for block in self.blocks.iter_mut() {
            let h1 = x.layer_norm(&block.gamma1, &block.beta1, 1e-5)?;
            let attended = block.attn.forward(&h1)?;
            x.add_inplace(&attended)?;
            // Let the oscillators sing across the sequence.
            for i in 0..t {
                let mut row = x.row(i).to_vec();
                block.resonance.resonate(&mut row, 0.15)?;
                x.set_row(i, &row)?;
            }
            let h2 = x.layer_norm(&block.gamma2, &block.beta2, 1e-5)?;
            let (mixed, aux) = block.moe.forward(&h2, rng)?;
            aux_total += aux;
            x.add_inplace(&mixed)?;
            // Adaptive gain control.
            let rows: Vec<Vec<f32>> = (0..t).map(|i| x.row(i).to_vec()).collect();
            block.homeostat.observe(&rows)?;
            for i in 0..t {
                let calmed = block.homeostat.apply(x.row(i))?;
                x.set_row(i, &calmed)?;
            }
            self.memory.sense(x.mean_over_rows())?;
        }

        // Online Hebbian plasticity biases retrieval toward live structure.
        let last = x.row(t - 1).to_vec();
        let _acts = self.hebb_bank.update(&last)?;
        let (winner, _) = self.hebb_bank.winner(&last)?;
        let proto = self.hebb_bank.prototype(winner).to_vec();
        let mut query = vec![0.0f32; d];
        for i in 0..d {
            query[i] = 0.8 * last[i] + 0.2 * proto[i];
        }

        // The memory cycle: attend, memorise, consolidate, prune, recall.
        let sensed = self.memory.stats().sensory;
        self.memory.attend(&vec![1.0; sensed])?;
        self.memory.memorise();
        self.memory.consolidate();
        self.memory.prune(1e-3);
        let hits = self.memory.recall(&query, 4)?;
        if !hits.is_empty() {
            let mut weights: Vec<f32> = hits.iter().map(|(_, s)| *s).collect();
            softmax_slice(&mut weights);
            let mut blend = vec![0.0f32; d];
            for ((vec, _), w) in hits.iter().zip(weights.iter()) {
                for (b, v) in blend.iter_mut().zip(vec.iter()) {
                    *b += w * v;
                }
            }
            for i in 0..t {
                let mut row = x.row(i).to_vec();
                for (a, b) in row.iter_mut().zip(blend.iter()) {
                    *a += 0.05 * b;
                }
                x.set_row(i, &row)?;
            }
        }

        let h = x.layer_norm(&self.out_gamma, &self.out_beta, 1e-5)?;
        let logits = h.matmul(&self.lm_head)?;
        Ok(ForwardOut {
            logits,
            aux_loss: aux_total,
            hidden: x,
        })
    }

    /// Autoregressive generation, returning only the new ids.
    pub fn generate(
        &mut self,
        prompt: &[usize],
        max_new: usize,
        sample: &SampleConfig,
        rng: &mut StdRng,
    ) -> Result<Vec<usize>> {
        if prompt.is_empty() {
            return Err(AetherError::EmptyInput(
                "generate got empty prompt".to_string(),
            ));
        }
        for &id in prompt {
            if id >= self.cfg.vocab_size {
                return Err(AetherError::Vocab(format!(
                    "token {id} out of vocab {}",
                    self.cfg.vocab_size
                )));
            }
        }
        let mut context: Vec<usize> = prompt.to_vec();
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            let start = context.len().saturating_sub(self.cfg.max_seq);
            let window: Vec<usize> = context[start..].to_vec();
            let fo = self.forward(&window, rng)?;
            let next = sample_logits(fo.logits.row(window.len() - 1), sample, rng)?;
            context.push(next);
            out.push(next);
            if sample.stop_id == Some(next) {
                break;
            }
        }
        Ok(out)
    }

    /// Flatten every evolvable parameter into one vector (fixed order).
    pub fn flat_params(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.param_count());
        self.visit_ref(&mut |s| v.extend_from_slice(s));
        v
    }

    /// Restore parameters flattened by [`AetherMind::flat_params`].
    pub fn set_flat_params(&mut self, params: &[f32]) -> Result<()> {
        let mut lens = Vec::new();
        self.visit_ref(&mut |s| lens.push(s.len()));
        let total: usize = lens.iter().sum();
        if params.len() != total {
            return Err(AetherError::ShapeMismatch(format!(
                "set_flat_params needs {total} values, got {}",
                params.len()
            )));
        }
        let mut offset = 0usize;
        self.visit_mut(&mut |s| {
            s.copy_from_slice(&params[offset..offset + s.len()]);
            offset += s.len();
        });
        Ok(())
    }

    /// Number of evolvable scalars (always equals `flat_params().len()`).
    pub fn param_count(&self) -> usize {
        let mut n = 0usize;
        self.visit_ref(&mut |s| n += s.len());
        n
    }

    fn visit_ref(&self, visitor: &mut impl FnMut(&[f32])) {
        visitor(self.token_embed.as_slice());
        for b in &self.blocks {
            visitor(b.attn.w_q.as_slice());
            visitor(b.attn.w_k.as_slice());
            visitor(b.attn.w_v.as_slice());
            visitor(b.attn.w_o.as_slice());
            visitor(&b.attn.w_gate);
            visitor(std::slice::from_ref(&b.attn.b_gate));
            for e in &b.moe.experts {
                visitor(e.w1.as_slice());
                visitor(e.w3.as_slice());
                visitor(e.w2.as_slice());
            }
            visitor(b.moe.w_gate.as_slice());
            visitor(&b.gamma1);
            visitor(&b.beta1);
            visitor(&b.gamma2);
            visitor(&b.beta2);
            visitor(b.resonance.freq.as_slice());
        }
        visitor(self.hebb_bank.prototypes.as_slice());
        visitor(&self.out_gamma);
        visitor(&self.out_beta);
        visitor(self.lm_head.as_slice());
    }

    fn visit_mut(&mut self, visitor: &mut impl FnMut(&mut [f32])) {
        visitor(self.token_embed.as_mut_slice());
        for b in self.blocks.iter_mut() {
            visitor(b.attn.w_q.as_mut_slice());
            visitor(b.attn.w_k.as_mut_slice());
            visitor(b.attn.w_v.as_mut_slice());
            visitor(b.attn.w_o.as_mut_slice());
            visitor(&mut b.attn.w_gate);
            visitor(std::slice::from_mut(&mut b.attn.b_gate));
            for e in b.moe.experts.iter_mut() {
                visitor(e.w1.as_mut_slice());
                visitor(e.w3.as_mut_slice());
                visitor(e.w2.as_mut_slice());
            }
            visitor(b.moe.w_gate.as_mut_slice());
            visitor(&mut b.gamma1);
            visitor(&mut b.beta1);
            visitor(&mut b.gamma2);
            visitor(&mut b.beta2);
            visitor(b.resonance.freq.as_mut_slice());
        }
        visitor(self.hebb_bank.prototypes.as_mut_slice());
        visitor(&mut self.out_gamma);
        visitor(&mut self.out_beta);
        visitor(self.lm_head.as_mut_slice());
    }

    /// Persist the whole organism (weights + oscillators + memory) as JSON.
    pub fn save_to(&self, path: &str) -> Result<()> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| AetherError::Ser(e.to_string()))?;
        fs::write(path, json).map_err(|e| AetherError::Io(e.to_string()))?;
        Ok(())
    }

    /// Load an organism saved with [`AetherMind::save_to`].
    pub fn load_from(path: &str) -> Result<AetherMind> {
        let text = fs::read_to_string(path).map_err(|e| AetherError::Io(e.to_string()))?;
        serde_json::from_str(&text).map_err(|e| AetherError::Ser(e.to_string()))
    }
}

/// Sample one id from raw logits under a [`SampleConfig`].
pub fn sample_logits(logits: &[f32], sample: &SampleConfig, rng: &mut StdRng) -> Result<usize> {
    if logits.is_empty() {
        return Err(AetherError::EmptyInput(
            "sample_logits got no logits".to_string(),
        ));
    }
    if !sample.temperature.is_finite() || sample.temperature < 0.0 {
        return Err(AetherError::InvalidConfig(
            "temperature must be finite and >= 0".to_string(),
        ));
    }
    if sample.temperature == 0.0 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        return Ok(best);
    }
    let mut scores: Vec<f32> = logits.iter().map(|x| x / sample.temperature).collect();
    if sample.top_k > 0 && sample.top_k < scores.len() {
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let cutoff = scores[order[sample.top_k]];
        for s in scores.iter_mut() {
            if *s < cutoff {
                *s = f32::NEG_INFINITY;
            }
        }
    }
    if sample.top_p > 0.0 && sample.top_p < 1.0 {
        let mut probs = scores.clone();
        softmax_slice(&mut probs);
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0f32;
        let mut keep = order.len() - 1;
        for (rank, &i) in order.iter().enumerate() {
            cum += probs[i];
            if cum >= sample.top_p {
                keep = rank;
                break;
            }
        }
        let mut allowed = vec![false; scores.len()];
        for &i in &order[..=keep] {
            allowed[i] = true;
        }
        for (s, a) in scores.iter_mut().zip(allowed.iter()) {
            if !a {
                *s = f32::NEG_INFINITY;
            }
        }
    }
    let mut probs = scores.clone();
    softmax_slice(&mut probs);
    let mut r = rng.gen_range(0.0..1.0);
    let mut chosen = probs.len() - 1;
    for (i, p) in probs.iter().enumerate() {
        if r < *p {
            chosen = i;
            break;
        }
        r -= p;
    }
    Ok(chosen)
}

/// Classic sinusoidal position code.
fn sinusoidal_pos(pos: usize, d: usize) -> Vec<f32> {
    (0..d)
        .map(|i| {
            let pair = (i / 2 * 2) as f32;
            let angle = pos as f32 / 10_000f32.powf(pair / d as f32);
            if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            }
        })
        .collect()
}

/// Stable softmax; all-`-inf` input falls back to uniform.
fn softmax_slice(xs: &mut [f32]) {
    if xs.is_empty() {
        return;
    }
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        let u = 1.0 / xs.len() as f32;
        for x in xs.iter_mut() {
            *x = u;
        }
        return;
    }
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inv = 1.0 / (sum + 1e-12);
    for x in xs.iter_mut() {
        *x *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seeded_rng;

    fn tiny_mind() -> AetherMind {
        AetherMind::new(AetherConfig::tiny()).unwrap()
    }

    #[test]
    fn forward_shapes_and_aux() {
        let mut mind = tiny_mind();
        let mut rng = seeded_rng(1);
        let fo = mind.forward(&[1, 2, 3, 4], &mut rng).unwrap();
        assert_eq!((fo.logits.nrows(), fo.logits.ncols()), (4, 64));
        assert_eq!((fo.hidden.nrows(), fo.hidden.ncols()), (4, 16));
        assert!(fo.aux_loss.is_finite() && fo.aux_loss >= 0.0);
        assert!(fo.logits.as_slice().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn forward_rejects_bad_inputs() {
        let mut mind = tiny_mind();
        let mut rng = seeded_rng(1);
        assert!(mind.forward(&[], &mut rng).is_err());
        assert!(mind.forward(&[999], &mut rng).is_err());
        let long = vec![1usize; 33];
        assert!(mind.forward(&long, &mut rng).is_err());
    }

    #[test]
    fn greedy_generate_is_deterministic() {
        let sample = SampleConfig {
            temperature: 0.0,
            ..SampleConfig::default()
        };
        let mut a = tiny_mind();
        let mut b = tiny_mind();
        let out_a = a
            .generate(&[1, 2, 3], 6, &sample, &mut seeded_rng(5))
            .unwrap();
        let out_b = b
            .generate(&[1, 2, 3], 6, &sample, &mut seeded_rng(5))
            .unwrap();
        assert_eq!(out_a, out_b);
        assert_eq!(out_a.len(), 6);
        assert!(out_a.iter().all(|&id| id < 64));
    }

    #[test]
    fn sampler_edges() {
        let mut rng = seeded_rng(0);
        // Unique maximum: top_k = 1 must collapse to the argmax exactly.
        let logits = vec![1.0, 5.0, 2.0, 4.0];
        // Greedy takes the first argmax.
        let greedy = SampleConfig {
            temperature: 0.0,
            ..SampleConfig::default()
        };
        assert_eq!(sample_logits(&logits, &greedy, &mut rng).unwrap(), 1);
        // top_k = 1 collapses to argmax too.
        let narrow = SampleConfig {
            temperature: 1.0,
            top_k: 1,
            ..SampleConfig::default()
        };
        for _ in 0..10 {
            let id = sample_logits(&logits, &narrow, &mut rng).unwrap();
            assert_eq!(id, 1);
        }
        assert!(sample_logits(&[], &greedy, &mut rng).is_err());
    }

    #[test]
    fn flat_params_roundtrip() {
        let mut mind = tiny_mind();
        let p = mind.flat_params();
        assert_eq!(p.len(), mind.param_count());
        assert!(mind.set_flat_params(&p).is_ok());
        assert_eq!(mind.flat_params(), p);
        // Perturbing one weight changes behaviour state (params differ).
        let mut q = p.clone();
        q[0] += 1.0;
        mind.set_flat_params(&q).unwrap();
        assert_eq!(mind.flat_params(), q);
        assert!(mind.set_flat_params(&p[..p.len() - 1]).is_err());
    }

    #[test]
    fn save_load_preserves_the_organism() {
        let mut mind = tiny_mind();
        let mut rng = seeded_rng(11);
        mind.forward(&[3, 1, 4], &mut rng).unwrap();
        let path = std::env::temp_dir().join("aether_roundtrip.json");
        mind.save_to(path.to_str().unwrap()).unwrap();
        let mut back = AetherMind::load_from(path.to_str().unwrap()).unwrap();
        let l1 = mind
            .forward(&[3, 1, 4], &mut seeded_rng(21))
            .unwrap()
            .logits
            .into_vec();
        let l2 = back
            .forward(&[3, 1, 4], &mut seeded_rng(21))
            .unwrap()
            .logits
            .into_vec();
        assert_eq!(l1, l2);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_bad_configs() {
        let mut bad = AetherConfig::tiny();
        bad.top_k = 9;
        assert!(AetherMind::new(bad).is_err());
        let mut bad2 = AetherConfig::tiny();
        bad2.d_model = 15;
        assert!(AetherMind::new(bad2).is_err());
    }
}
