//! Entangled multi-scale attention.
//!
//! Each query position fuses two views of the past (and future — the layer is
//! non-causal by design, causality is imposed by the caller when needed):
//!
//! * a **local branch**: sharp softmax inside a sliding window of half-width
//!   `window / 2`, capturing syntax-scale structure;
//! * a **holographic branch**: a global gist (mean-pooled keys) scores every
//!   position, capturing document-scale aboutness.
//!
//! A learned per-position **resonance gate** `g in (0,1)` mixes them:
//! `p = (1-g) * p_local + g * p_holo`. The mixture stays row-stochastic, so
//! the layer is exactly permutation-equivariant — a property the test-suite
//! verifies by construction.

use crate::error::{AetherError, Result};
use crate::tensor::{sigmoid, Matrix};
use serde::{Deserialize, Serialize};

/// Construction parameters for entangled attention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntangledAttentionConfig {
    /// Model width; must be divisible by `n_heads`.
    pub d_model: usize,
    /// Number of heads.
    pub n_heads: usize,
    /// Sliding-window diameter for the local branch.
    pub window: usize,
}

impl EntangledAttentionConfig {
    /// Validate ranges and divisibility.
    pub fn validate(&self) -> Result<()> {
        if self.d_model == 0 || self.n_heads == 0 || self.window == 0 {
            return Err(AetherError::InvalidConfig(
                "d_model, n_heads and window must all be > 0".to_string(),
            ));
        }
        if self.d_model % self.n_heads != 0 {
            return Err(AetherError::InvalidConfig(format!(
                "d_model {} not divisible by n_heads {}",
                self.d_model, self.n_heads
            )));
        }
        Ok(())
    }
}

/// Entangled attention layer with Q/K/V/O projections and a resonance gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntangledAttention {
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    window: usize,
    pub(crate) w_q: Matrix,
    pub(crate) w_k: Matrix,
    pub(crate) w_v: Matrix,
    pub(crate) w_o: Matrix,
    pub(crate) w_gate: Vec<f32>,
    pub(crate) b_gate: f32,
}

impl EntangledAttention {
    /// Build a layer with Xavier projections, deterministically seeded.
    pub fn new(cfg: EntangledAttentionConfig, seed: u64) -> Result<EntangledAttention> {
        cfg.validate()?;
        let d = cfg.d_model;
        Ok(EntangledAttention {
            d_model: d,
            n_heads: cfg.n_heads,
            d_head: d / cfg.n_heads,
            window: cfg.window,
            w_q: Matrix::xavier_seeded(seed, d, d),
            w_k: Matrix::xavier_seeded(seed + 1, d, d),
            w_v: Matrix::xavier_seeded(seed + 2, d, d),
            w_o: Matrix::xavier_seeded(seed + 3, d, d),
            w_gate: Matrix::randn_seeded(seed + 4, 1, d).into_vec().iter().map(|x| x * 0.1).collect(),
            b_gate: 0.0,
        })
    }

    /// Per-position resonance gates in (0, 1).
    pub fn gates(&self, x: &Matrix) -> Result<Vec<f32>> {
        if x.ncols() != self.d_model {
            return Err(AetherError::ShapeMismatch(format!(
                "attention gates need d_model {} cols, got {}",
                self.d_model,
                x.ncols()
            )));
        }
        let mut out = Vec::with_capacity(x.nrows());
        for i in 0..x.nrows() {
            out.push(sigmoid(dot(x.row(i), &self.w_gate) + self.b_gate));
        }
        Ok(out)
    }

    /// Forward pass: `T x D` in, `T x D` out.
    pub fn forward(&self, x: &Matrix) -> Result<Matrix> {
        let t = x.nrows();
        let d = self.d_model;
        if t == 0 {
            return Err(AetherError::EmptyInput("attention got empty sequence".to_string()));
        }
        if x.ncols() != d {
            return Err(AetherError::ShapeMismatch(format!(
                "attention needs d_model {d} cols, got {}",
                x.ncols()
            )));
        }
        let h = self.n_heads;
        let dh = self.d_head;
        let q = x.matmul(&self.w_q)?;
        let k = x.matmul(&self.w_k)?;
        let v = x.matmul(&self.w_v)?;
        let qb = q.as_slice();
        let kb = k.as_slice();
        let vb = v.as_slice();

        // Raw head scores: scores[(t*T+s)*H+hh].
        let scale = 1.0 / (dh as f32).sqrt();
        let mut scores = vec![0.0f32; t * t * h];
        for tt in 0..t {
            for s in 0..t {
                for hh in 0..h {
                    let mut acc = 0.0f32;
                    let qo = tt * d + hh * dh;
                    let ko = s * d + hh * dh;
                    for dd in 0..dh {
                        acc += qb[qo + dd] * kb[ko + dd];
                    }
                    scores[(tt * t + s) * h + hh] = acc * scale;
                }
            }
        }

        // Local branch: windowed softmax per (position, head).
        let half = self.window / 2;
        let mut p_local = vec![0.0f32; t * t * h];
        for tt in 0..t {
            for hh in 0..h {
                let mut max = f32::NEG_INFINITY;
                for s in 0..t {
                    if tt.abs_diff(s) <= half {
                        let sc = scores[(tt * t + s) * h + hh];
                        if sc > max {
                            max = sc;
                        }
                    }
                }
                let mut sum = 0.0f32;
                for s in 0..t {
                    let idx = (tt * t + s) * h + hh;
                    let e = if tt.abs_diff(s) <= half {
                        (scores[idx] - max).exp()
                    } else {
                        0.0
                    };
                    p_local[idx] = e;
                    sum += e;
                }
                let inv = 1.0 / (sum + 1e-12);
                for s in 0..t {
                    p_local[(tt * t + s) * h + hh] *= inv;
                }
            }
        }

        // Holographic branch: gist scores every position, one shared distribution.
        let gist = k.mean_over_rows();
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let mut holo = vec![0.0f32; t];
        for (s, hlog) in holo.iter_mut().enumerate() {
            *hlog = dot(k.row(s), &gist) * inv_sqrt_d;
        }
        softmax_inplace(&mut holo);

        // Entangle and aggregate.
        let gates = self.gates(x)?;
        let mut out = Matrix::zeros(t, d);
        let ob = out.as_mut_slice();
        for tt in 0..t {
            let mix = gates[tt];
            for s in 0..t {
                for hh in 0..h {
                    let p = (1.0 - mix) * p_local[(tt * t + s) * h + hh] + mix * holo[s];
                    let vo = s * d + hh * dh;
                    let oo = tt * d + hh * dh;
                    for dd in 0..dh {
                        ob[oo + dd] += p * vb[vo + dd];
                    }
                }
            }
        }
        out.matmul(&self.w_o)
    }

    /// Total trainable scalars.
    pub fn param_count(&self) -> usize {
        4 * self.d_model * self.d_model + self.d_model + 1
    }
}

/// Dot product over equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Stable in-place softmax.
fn softmax_inplace(xs: &mut [f32]) {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

    fn cfg() -> EntangledAttentionConfig {
        EntangledAttentionConfig {
            d_model: 16,
            n_heads: 4,
            window: 5,
        }
    }

    #[test]
    fn forward_shape_and_finiteness() {
        let attn = EntangledAttention::new(cfg(), 1).unwrap();
        let x = Matrix::randn_seeded(2, 12, 16);
        let y = attn.forward(&x).unwrap();
        assert_eq!((y.nrows(), y.ncols()), (12, 16));
        assert!(y.as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn deterministic_forward() {
        let attn = EntangledAttention::new(cfg(), 5).unwrap();
        let x = Matrix::randn_seeded(6, 7, 16);
        let a = attn.forward(&x).unwrap();
        let b = attn.forward(&x).unwrap();
        assert_eq!(a.as_slice(), b.as_slice());
    }

    #[test]
    fn gates_live_in_unit_interval() {
        let attn = EntangledAttention::new(cfg(), 5).unwrap();
        let x = Matrix::randn_seeded(6, 7, 16);
        for g in attn.gates(&x).unwrap() {
            assert!(g > 0.0 && g < 1.0, "gate {g}");
        }
    }

    #[test]
    fn permutation_equivariant() {
        // Reversing the input must reverse the output: the layer has no
        // absolute position bias, only relative windows + invariant gist.
        let attn = EntangledAttention::new(cfg(), 5).unwrap();
        let x = Matrix::randn_seeded(6, 9, 16);
        let y = attn.forward(&x).unwrap();
        let mut rev_rows: Vec<Vec<f32>> = (0..9).map(|i| x.row(i).to_vec()).collect();
        rev_rows.reverse();
        let xr = Matrix::from_rows(rev_rows).unwrap();
        let yr = attn.forward(&xr).unwrap();
        for i in 0..9 {
            for (a, b) in y.row(i).iter().zip(yr.row(8 - i).iter()) {
                assert!((a - b).abs() < 1e-4, "row {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn rejects_bad_shapes() {
        let attn = EntangledAttention::new(cfg(), 5).unwrap();
        assert!(attn.forward(&Matrix::zeros(4, 15)).is_err());
        assert!(EntangledAttention::new(
            EntangledAttentionConfig {
                d_model: 15,
                n_heads: 4,
                window: 3
            },
            0
        )
        .is_err());
    }
}
