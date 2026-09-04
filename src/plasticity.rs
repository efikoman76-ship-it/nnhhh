//! Hebbian plasticity and homeostatic control.
//!
//! The resonant/holographic substrate is non-differentiable by design, so
//! AETHER learns online as well as evolutionarily:
//!
//! * [`HebbianBank`] — a bank of prototype neurons updated with Oja's rule,
//!   `m += lr * y * (x - y * m)`, which keeps weight norms bounded while
//!   extracting the principal subspace of the hidden stream. The live mind
//!   updates it on every forward pass and lets the winning prototype bias
//!   memory retrieval.
//! * [`Homeostat`] — per-channel adaptive gain control steering activations
//!   toward a target mean/variance, preventing resonant blow-up or silence.
//! * [`stdp_window`] — the classic double-exponential spike-timing window,
//!   available to exotic training regimes.

use crate::error::{AetherError, Result};
use crate::tensor::Matrix;
use serde::{Deserialize, Serialize};

/// Bank of Oja neurons competing for the hidden stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HebbianBank {
    n: usize,
    dim: usize,
    lr: f32,
    pub(crate) prototypes: Matrix,
}

impl HebbianBank {
    /// Build a bank of `n` prototypes over `dim`-wide inputs.
    pub fn new(n: usize, dim: usize, lr: f32, seed: u64) -> Result<HebbianBank> {
        if n == 0 || dim == 0 {
            return Err(AetherError::InvalidConfig(
                "hebbian bank needs n > 0 and dim > 0".to_string(),
            ));
        }
        if lr <= 0.0 {
            return Err(AetherError::InvalidConfig(
                "hebbian lr must be > 0".to_string(),
            ));
        }
        Ok(HebbianBank {
            n,
            dim,
            lr,
            prototypes: Matrix::randn_seeded(seed, n, dim),
        })
    }

    /// Oja update on input `x`; returns the prototype activations.
    pub fn update(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "hebbian update needs {0} inputs, got {1}",
                self.dim,
                x.len()
            )));
        }
        let mut acts = vec![0.0f32; self.n];
        for i in 0..self.n {
            let row = self.prototypes.row(i);
            acts[i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        }
        for i in 0..self.n {
            let y = acts[i];
            let row = self.prototypes.row_mut(i);
            for (w, xi) in row.iter_mut().zip(x.iter()) {
                *w += self.lr * y * (*xi - y * *w);
            }
        }
        Ok(acts)
    }

    /// Activation of every prototype on `x` without learning.
    pub fn activations(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "hebbian activations need {0} inputs, got {1}",
                self.dim,
                x.len()
            )));
        }
        let mut acts = vec![0.0f32; self.n];
        for i in 0..self.n {
            acts[i] = self
                .prototypes
                .row(i)
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum();
        }
        Ok(acts)
    }

    /// Winning prototype index and activation.
    pub fn winner(&self, x: &[f32]) -> Result<(usize, f32)> {
        let acts = self.activations(x)?;
        let mut best = 0;
        for (i, a) in acts.iter().enumerate() {
            if *a > acts[best] {
                best = i;
            }
        }
        Ok((best, acts[best]))
    }

    /// Borrow the winning prototype vector.
    pub fn prototype(&self, i: usize) -> &[f32] {
        self.prototypes.row(i)
    }

    /// Row norms (Oja's rule keeps these near 1).
    pub fn norms(&self) -> Vec<f32> {
        (0..self.n)
            .map(|i| {
                self.prototypes
                    .row(i)
                    .iter()
                    .map(|w| w * w)
                    .sum::<f32>()
                    .sqrt()
            })
            .collect()
    }

    /// Number of prototypes.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when the bank holds no prototypes (never, by construction).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Per-channel homeostatic gain control toward target statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Homeostat {
    dim: usize,
    target_mean: f32,
    target_var: f32,
    rate: f32,
    mean: Vec<f32>,
    var: Vec<f32>,
}

impl Homeostat {
    /// Build a homeostat steering toward `target_mean` / `target_var`.
    pub fn new(dim: usize, target_mean: f32, target_var: f32, rate: f32) -> Result<Homeostat> {
        if dim == 0 {
            return Err(AetherError::InvalidConfig("homeostat dim is 0".to_string()));
        }
        if target_var <= 0.0 || rate <= 0.0 || rate > 1.0 {
            return Err(AetherError::InvalidConfig(
                "need target_var > 0 and rate in (0, 1]".to_string(),
            ));
        }
        Ok(Homeostat {
            dim,
            target_mean,
            target_var,
            rate,
            mean: vec![0.0; dim],
            var: vec![1.0; dim],
        })
    }

    /// Fold a batch of observations into the running statistics.
    pub fn observe(&mut self, batch: &[Vec<f32>]) -> Result<()> {
        for x in batch {
            if x.len() != self.dim {
                return Err(AetherError::ShapeMismatch(format!(
                    "homeostat observe needs {0} dims, got {1}",
                    self.dim,
                    x.len()
                )));
            }
            for i in 0..self.dim {
                let delta = x[i] - self.mean[i];
                self.mean[i] += self.rate * delta;
                let delta2 = x[i] - self.mean[i];
                self.var[i] = (1.0 - self.rate) * self.var[i] + self.rate * delta * delta2;
            }
        }
        Ok(())
    }

    /// Normalise `x` toward the target statistics.
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "homeostat apply needs {0} dims, got {1}",
                self.dim,
                x.len()
            )));
        }
        let gain = self.target_var.sqrt();
        Ok((0..self.dim)
            .map(|i| (x[i] - self.mean[i]) / (self.var[i] + 1e-6).sqrt() * gain + self.target_mean)
            .collect())
    }
}

/// Classic STDP window over a pre/post spike-time difference `dt`.
///
/// Positive `dt` (pre before post) potentiates; negative depresses.
pub fn stdp_window(dt: f32) -> f32 {
    if dt >= 0.0 {
        (-dt / 20.0).exp()
    } else {
        -0.6 * (dt / 20.0).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn oja_norms_stay_bounded() {
        let mut bank = HebbianBank::new(6, 12, 0.02, 4).unwrap();
        let mut rng = StdRng::seed_from_u64(8);
        for _ in 0..300 {
            let x: Vec<f32> = (0..12).map(|_| rng.gen_range(-1.0..1.0)).collect();
            bank.update(&x).unwrap();
        }
        for n in bank.norms() {
            assert!(n.is_finite() && n > 0.05 && n < 5.0, "norm {n}");
        }
    }

    #[test]
    fn winner_matches_max_activation() {
        let bank = HebbianBank::new(4, 8, 0.01, 1).unwrap();
        let x = vec![0.3f32; 8];
        let acts = bank.activations(&x).unwrap();
        let (w, a) = bank.winner(&x).unwrap();
        assert!((a - acts[w]).abs() < 1e-6);
        assert!(acts.iter().all(|v| *v <= a + 1e-6));
    }

    #[test]
    fn homeostat_steers_variance() {
        let mut h = Homeostat::new(4, 0.0, 1.0, 0.05).unwrap();
        let mut rng = StdRng::seed_from_u64(3);
        // Wild high-variance stream.
        for _ in 0..400 {
            let x: Vec<f32> = (0..4).map(|_| rng.gen_range(-8.0..8.0)).collect();
            h.observe(&[x]).unwrap();
        }
        let probe = vec![8.0, -8.0, 4.0, -4.0];
        let y = h.apply(&probe).unwrap();
        let mean: f32 = y.iter().sum::<f32>() / 4.0;
        let var: f32 = y.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(var < 8.0, "var {var}");
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stdp_sign_convention() {
        assert!(stdp_window(5.0) > 0.0);
        assert!(stdp_window(-5.0) < 0.0);
        assert!((stdp_window(0.0) - 1.0).abs() < 1e-6);
        assert!(stdp_window(10.0) < stdp_window(1.0));
    }
}
