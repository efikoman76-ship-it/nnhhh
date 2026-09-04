//! Sparse mixture of SwiGLU experts.
//!
//! Every token is routed to `top_k` of `n_experts` feed-forward experts. The
//! router adds a little uniform noise (exploration) and keeps the top-k noisy
//! logits, renormalised with a softmax. A Switch-style auxiliary loss,
//! `E * sum(f_i * p_i)`, pressures the router toward balanced utilisation:
//! for a single noiseless token it provably lower-bounds at `top_k`, a fact
//! the tests assert directly.

use crate::error::{AetherError, Result};
use crate::tensor::Matrix;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Construction parameters for the sparse MoE layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMoeConfig {
    /// Model width.
    pub d_model: usize,
    /// Expert hidden width.
    pub d_hidden: usize,
    /// Number of experts.
    pub n_experts: usize,
    /// Experts activated per token.
    pub top_k: usize,
    /// Router exploration noise half-width (0 = deterministic).
    pub noise_std: f32,
}

impl SparseMoeConfig {
    /// Validate ranges and the `top_k <= n_experts` invariant.
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.d_hidden == 0 {
            return Err(AetherError::InvalidConfig(
                "d_model and d_hidden must be > 0".to_string(),
            ));
        }
        if self.n_experts == 0 || self.top_k == 0 {
            return Err(AetherError::InvalidConfig(
                "n_experts and top_k must be > 0".to_string(),
            ));
        }
        if self.top_k > self.n_experts {
            return Err(AetherError::InvalidConfig(format!(
                "top_k {} exceeds n_experts {}",
                self.top_k, self.n_experts
            )));
        }
        if self.noise_std < 0.0 {
            return Err(AetherError::InvalidConfig(
                "noise_std must be >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// One SwiGLU expert: `y = (silu(x W3) * (x W1)) W2`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiGluExpert {
    d_model: usize,
    d_hidden: usize,
    pub(crate) w1: Matrix,
    pub(crate) w3: Matrix,
    pub(crate) w2: Matrix,
}

impl SwiGluExpert {
    /// Xavier-initialised expert, deterministically seeded.
    pub fn new(d_model: usize, d_hidden: usize, seed: u64) -> SwiGluExpert {
        SwiGluExpert {
            d_model,
            d_hidden,
            w1: Matrix::xavier_seeded(seed, d_model, d_hidden),
            w3: Matrix::xavier_seeded(seed + 1, d_model, d_hidden),
            w2: Matrix::xavier_seeded(seed + 2, d_hidden, d_model),
        }
    }

    /// Forward pass over a single row.
    pub fn forward_row(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.d_model {
            return Err(AetherError::ShapeMismatch(format!(
                "expert needs {0} inputs, got {1}",
                self.d_model,
                x.len()
            )));
        }
        let xm = Matrix::from_vec(1, self.d_model, x.to_vec())?;
        let h1 = xm.matmul(&self.w1)?;
        let gated = xm.matmul(&self.w3)?.silu();
        let mut h = h1.into_vec();
        for (a, g) in h.iter_mut().zip(gated.as_slice().iter()) {
            *a *= *g;
        }
        let hm = Matrix::from_vec(1, self.d_hidden, h)?;
        Ok(hm.matmul(&self.w2)?.into_vec())
    }
}

/// Sparse MoE layer: noisy top-k router over SwiGLU experts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMoe {
    d_model: usize,
    n_experts: usize,
    top_k: usize,
    noise_std: f32,
    pub(crate) experts: Vec<SwiGluExpert>,
    pub(crate) w_gate: Matrix,
}

impl SparseMoe {
    /// Build the layer, deterministically seeded.
    pub fn new(cfg: SparseMoeConfig, seed: u64) -> Result<SparseMoe> {
        cfg.validate()?;
        let mut experts = Vec::with_capacity(cfg.n_experts);
        for e in 0..cfg.n_experts {
            experts.push(SwiGluExpert::new(
                cfg.d_model,
                cfg.d_hidden,
                seed + 100 + e as u64 * 7,
            ));
        }
        Ok(SparseMoe {
            d_model: cfg.d_model,
            n_experts: cfg.n_experts,
            top_k: cfg.top_k,
            noise_std: cfg.noise_std,
            experts,
            w_gate: Matrix::xavier_seeded(seed + 500, cfg.d_model, cfg.n_experts),
        })
    }

    /// Route one row to experts: `top_k` pairs of (expert, weight), weights
    /// summing to 1.
    pub fn route(&self, x: &[f32], rng: &mut StdRng) -> Result<Vec<(usize, f32)>> {
        if x.len() != self.d_model {
            return Err(AetherError::ShapeMismatch(format!(
                "router needs {} inputs, got {}",
                self.d_model,
                x.len()
            )));
        }
        let xm = Matrix::from_vec(1, self.d_model, x.to_vec())?;
        let logits = xm.matmul(&self.w_gate)?.into_vec();
        let mut noisy = logits.clone();
        if self.noise_std > 0.0 {
            for v in noisy.iter_mut() {
                *v += rng.gen_range(-self.noise_std..=self.noise_std);
            }
        }
        let mut idx: Vec<usize> = (0..self.n_experts).collect();
        idx.sort_by(|&a, &b| {
            noisy[b]
                .partial_cmp(&noisy[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(self.top_k);
        let max = idx
            .iter()
            .map(|&i| noisy[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights = Vec::with_capacity(self.top_k);
        let mut sum = 0.0f32;
        for &i in &idx {
            let e = (noisy[i] - max).exp();
            weights.push(e);
            sum += e;
        }
        let inv = 1.0 / (sum + 1e-12);
        Ok(idx
            .into_iter()
            .zip(weights.into_iter())
            .map(|(i, w)| (i, w * inv))
            .collect())
    }

    /// Forward pass over `T x D`, returning outputs and the aux balance loss.
    pub fn forward(&self, x: &Matrix, rng: &mut StdRng) -> Result<(Matrix, f32)> {
        if x.ncols() != self.d_model {
            return Err(AetherError::ShapeMismatch(format!(
                "moe needs d_model {} cols, got {}",
                self.d_model,
                x.ncols()
            )));
        }
        let t = x.nrows();
        if t == 0 {
            return Err(AetherError::EmptyInput(
                "moe got empty sequence".to_string(),
            ));
        }
        let mut out = Matrix::zeros(t, self.d_model);
        let mut routed = vec![0.0f32; self.n_experts];
        let mut prob_mass = vec![0.0f32; self.n_experts];
        for row_idx in 0..t {
            let row = x.row(row_idx);
            let xm = Matrix::from_vec(1, self.d_model, row.to_vec())?;
            let logits = xm.matmul(&self.w_gate)?.into_vec();
            let probs = softmax_vec(&logits);
            for (p, acc) in probs.iter().zip(prob_mass.iter_mut()) {
                *acc += *p;
            }
            let chosen = self.route(row, rng)?;
            let mut y = vec![0.0f32; self.d_model];
            for (expert, weight) in &chosen {
                routed[*expert] += 1.0;
                let ey = self.experts[*expert].forward_row(row)?;
                for (a, b) in y.iter_mut().zip(ey.iter()) {
                    *a += weight * b;
                }
            }
            out.set_row(row_idx, &y)?;
        }
        let inv_t = 1.0 / t as f32;
        let mut aux = 0.0f32;
        for i in 0..self.n_experts {
            aux += (routed[i] * inv_t) * (prob_mass[i] * inv_t);
        }
        aux *= self.n_experts as f32;
        Ok((out, aux))
    }

    /// Total trainable scalars (experts + router).
    pub fn param_count(&self) -> usize {
        self.experts.len() * 3 * self.d_model * self.experts_d_hidden()
            + self.d_model * self.n_experts
    }

    fn experts_d_hidden(&self) -> usize {
        self.experts.first().map(|e| e.d_hidden).unwrap_or(0)
    }
}

/// Stable softmax over a vector.
fn softmax_vec(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    let inv = 1.0 / (sum + 1e-12);
    for x in out.iter_mut() {
        *x *= inv;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn cfg() -> SparseMoeConfig {
        SparseMoeConfig {
            d_model: 12,
            d_hidden: 24,
            n_experts: 4,
            top_k: 2,
            noise_std: 0.05,
        }
    }

    #[test]
    fn routes_top_k_with_unit_weights() {
        let moe = SparseMoe::new(cfg(), 1).unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        let x = Matrix::randn_seeded(3, 1, 12).into_vec();
        let chosen = moe.route(&x, &mut rng).unwrap();
        assert_eq!(chosen.len(), 2);
        assert_ne!(chosen[0].0, chosen[1].0);
        let sum: f32 = chosen.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn forward_shape_and_aux_floor() {
        // Noiseless single token: aux = E * sum(top-k probs) >= top_k.
        let clean = SparseMoeConfig {
            noise_std: 0.0,
            ..cfg()
        };
        let moe = SparseMoe::new(clean, 1).unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        let x = Matrix::randn_seeded(3, 1, 12);
        let (y, aux) = moe.forward(&x, &mut rng).unwrap();
        assert_eq!((y.nrows(), y.ncols()), (1, 12));
        assert!(aux.is_finite());
        assert!(aux >= 2.0 - 1e-4, "aux {aux}");
    }

    #[test]
    fn forward_batch_finite() {
        let moe = SparseMoe::new(cfg(), 1).unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        let x = Matrix::randn_seeded(3, 8, 12);
        let (y, aux) = moe.forward(&x, &mut rng).unwrap();
        assert_eq!((y.nrows(), y.ncols()), (8, 12));
        assert!(y.as_slice().iter().all(|v| v.is_finite()));
        assert!(aux.is_finite() && aux >= 0.0);
    }

    #[test]
    fn rejects_bad_config() {
        let bad = SparseMoeConfig { top_k: 5, ..cfg() };
        assert!(SparseMoe::new(bad, 0).is_err());
    }
}
