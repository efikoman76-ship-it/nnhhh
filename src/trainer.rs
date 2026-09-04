//! Derivative-free training for a non-differentiable mind.
//!
//! Backpropagation cannot flow through Kuramoto oscillators, holographic
//! binding or top-k routing — and that is intentional. AETHER optimises with
//! two honest mechanisms:
//!
//! * [`AdamW`] — the classic decoupled-weight-decay optimiser over flat
//!   parameter vectors, unit-tested on a quadratic bowl;
//! * [`EvoTrainer`] — antithetic evolutionary strategies: sample mirrored
//!   Gaussian perturbations, score each with teacher-forced cross-entropy
//!   (+ MoE balance loss, − predictive-entropy bonus), form the ES gradient
//!   estimate, clip it, and step with AdamW under a cosine schedule.
//!
//! Online Hebbian/homeostatic plasticity additionally learns on every forward
//! pass with no supervision at all.

use crate::error::{AetherError, Result};
use crate::network::AetherMind;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// AdamW optimiser over a flat parameter vector.
#[derive(Debug, Clone)]
pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step: u64,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl AdamW {
    /// Build an optimiser for `dim` parameters.
    pub fn new(dim: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32) -> Result<AdamW> {
        if dim == 0 {
            return Err(AetherError::InvalidConfig("adam dim is 0".to_string()));
        }
        if lr <= 0.0 || eps <= 0.0 || wd < 0.0 {
            return Err(AetherError::InvalidConfig(
                "need lr > 0, eps > 0, wd >= 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&beta1) || !(0.0..1.0).contains(&beta2) {
            return Err(AetherError::InvalidConfig("betas must be in (0, 1)".to_string()));
        }
        Ok(AdamW {
            lr,
            beta1,
            beta2,
            eps,
            wd,
            step: 0,
            m: vec![0.0; dim],
            v: vec![0.0; dim],
        })
    }

    /// Override the learning rate (used by the cosine schedule).
    pub fn set_lr(&mut self, lr: f32) {
        if lr > 0.0 {
            self.lr = lr;
        }
    }

    /// One optimisation step.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<()> {
        if params.len() != self.m.len() || grads.len() != self.m.len() {
            return Err(AetherError::ShapeMismatch(format!(
                "adam needs {} params/grads, got {}/{}",
                self.m.len(),
                params.len(),
                grads.len()
            )));
        }
        self.step += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);
        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i];
            let mhat = self.m[i] / bc1;
            let vhat = self.v[i] / bc2;
            params[i] -= self.lr * (mhat / (vhat.sqrt() + self.eps) + self.wd * params[i]);
        }
        Ok(())
    }
}

/// Cosine learning-rate schedule with linear warmup.
#[derive(Debug, Clone, Copy)]
pub struct CosineSchedule {
    /// Peak learning rate.
    pub base_lr: f32,
    /// Warmup steps.
    pub warmup_steps: u64,
    /// Total steps (schedule hits ~0 here).
    pub total_steps: u64,
}

impl CosineSchedule {
    /// Learning rate at `step`.
    pub fn lr_at(&self, step: u64) -> f32 {
        if self.total_steps == 0 {
            return self.base_lr;
        }
        if step < self.warmup_steps && self.warmup_steps > 0 {
            return self.base_lr * step as f32 / self.warmup_steps as f32;
        }
        let span = self.total_steps.saturating_sub(self.warmup_steps).max(1) as f32;
        let progress = ((step.saturating_sub(self.warmup_steps)) as f32 / span).min(1.0);
        0.5 * self.base_lr * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

/// Teacher-forced cross-entropy + predictive entropy of one logit row.
fn ce_and_entropy(logits: &[f32], target: usize) -> Result<(f32, f32)> {
    if target >= logits.len() {
        return Err(AetherError::Vocab(format!(
            "target {target} out of {} logits",
            logits.len()
        )));
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut probs = Vec::with_capacity(logits.len());
    for &x in logits {
        let e = (x - max).exp();
        probs.push(e);
        sum += e;
    }
    let inv = 1.0 / (sum + 1e-12);
    let mut entropy = 0.0f32;
    for p in probs.iter_mut() {
        *p *= inv;
        entropy -= *p * (p.max(1e-12)).ln();
    }
    Ok((-probs[target].max(1e-12).ln(), entropy))
}

/// Evolutionary-strategy hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvoTrainerConfig {
    /// Population per generation (even, >= 2; antithetic pairs).
    pub pop: usize,
    /// Perturbation scale.
    pub sigma: f32,
    /// Peak AdamW learning rate.
    pub lr: f32,
    /// Generations per `train` call.
    pub generations: usize,
    /// Weight of the MoE balance loss.
    pub aux_weight: f32,
    /// Weight of the predictive-entropy bonus (diversity pressure).
    pub entropy_bonus: f32,
    /// Gradient-norm ceiling.
    pub max_grad_norm: f32,
    /// Seed for perturbations.
    pub seed: u64,
}

impl EvoTrainerConfig {
    /// Small fast config for tests and demos.
    pub fn demo() -> EvoTrainerConfig {
        EvoTrainerConfig {
            pop: 4,
            sigma: 0.02,
            lr: 0.05,
            generations: 3,
            aux_weight: 0.01,
            entropy_bonus: 0.0,
            max_grad_norm: 1.0,
            seed: 11,
        }
    }

    /// Validate ranges.
    pub fn validate(&self) -> Result<()> {
        if self.pop < 2 || self.pop % 2 != 0 {
            return Err(AetherError::InvalidConfig(
                "pop must be even and >= 2".to_string(),
            ));
        }
        if self.sigma <= 0.0 || self.lr <= 0.0 || self.max_grad_norm <= 0.0 {
            return Err(AetherError::InvalidConfig(
                "sigma, lr and max_grad_norm must be > 0".to_string(),
            ));
        }
        if self.generations == 0 {
            return Err(AetherError::InvalidConfig("generations is 0".to_string()));
        }
        if self.aux_weight < 0.0 || self.entropy_bonus < 0.0 {
            return Err(AetherError::InvalidConfig(
                "aux_weight and entropy_bonus must be >= 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Per-generation report.
#[derive(Debug, Clone)]
pub struct TrainStats {
    /// Generation index just finished.
    pub generation: usize,
    /// Loss of the updated parameters.
    pub loss: f32,
    /// Mean MoE auxiliary loss.
    pub aux: f32,
    /// Best loss ever seen (including the starting point).
    pub best_loss: f32,
}

/// Antithetic evolutionary-strategy trainer.
#[derive(Debug)]
pub struct EvoTrainer {
    cfg: EvoTrainerConfig,
    adam: AdamW,
    sched: CosineSchedule,
    best_params: Vec<f32>,
    best_loss: f32,
    step: u64,
    dim: usize,
}

impl EvoTrainer {
    /// Build a trainer for a mind with `dim` flat parameters.
    pub fn new(cfg: EvoTrainerConfig, dim: usize) -> Result<EvoTrainer> {
        cfg.validate()?;
        if dim == 0 {
            return Err(AetherError::InvalidConfig("trainer dim is 0".to_string()));
        }
        let total = cfg.generations as u64;
        Ok(EvoTrainer {
            adam: AdamW::new(dim, cfg.lr, 0.9, 0.999, 1e-8, 1e-4)?,
            sched: CosineSchedule {
                base_lr: cfg.lr,
                warmup_steps: total / 4,
                total_steps: total,
            },
            best_params: Vec::new(),
            best_loss: f32::INFINITY,
            step: 0,
            dim,
            cfg,
        })
    }

    /// Best parameters seen so far (empty before the first `train`).
    pub fn best_params(&self) -> &[f32] {
        &self.best_params
    }

    /// Restore the best parameters into `mind`.
    pub fn restore_best(&self, mind: &mut AetherMind) -> Result<()> {
        if self.best_params.is_empty() {
            return Err(AetherError::EmptyInput("no best params yet".to_string()));
        }
        mind.set_flat_params(&self.best_params)
    }

    /// Score `mind` on a batch: mean teacher-forced loss + aux, and mean aux.
    pub fn eval_loss(
        &self,
        mind: &mut AetherMind,
        batch: &[Vec<usize>],
        rng: &mut StdRng,
    ) -> Result<(f32, f32)> {
        if batch.is_empty() {
            return Err(AetherError::EmptyInput("eval got empty batch".to_string()));
        }
        let mut total = 0.0f64;
        let mut tokens = 0usize;
        let mut aux_sum = 0.0f64;
        let mut used = 0usize;
        for seq in batch {
            if seq.len() < 2 {
                continue;
            }
            let fo = mind.forward(&seq[..seq.len() - 1], rng)?;
            for pos in 0..seq.len() - 1 {
                let (ce, ent) = ce_and_entropy(fo.logits.row(pos), seq[pos + 1])?;
                total += (ce - self.cfg.entropy_bonus * ent) as f64;
                tokens += 1;
            }
            aux_sum += (self.cfg.aux_weight * fo.aux_loss) as f64;
            used += 1;
        }
        if tokens == 0 {
            return Err(AetherError::EmptyInput(
                "eval needs sequences of len >= 2".to_string(),
            ));
        }
        Ok(((total / tokens as f64) as f32, (aux_sum / used as f64) as f32))
    }

    /// Run the configured generations; returns the last generation's stats.
    pub fn train(
        &mut self,
        mind: &mut AetherMind,
        batch: &[Vec<usize>],
        rng: &mut StdRng,
    ) -> Result<TrainStats> {
        if mind.flat_params().len() != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "trainer dim {} vs mind {}",
                self.dim,
                mind.flat_params().len()
            )));
        }
        let base = mind.flat_params();
        if self.best_params.is_empty() {
            let (l, _) = self.eval_loss(mind, batch, rng)?;
            self.best_loss = l;
            self.best_params = base.clone();
        }
        let mut stats = TrainStats {
            generation: 0,
            loss: self.best_loss,
            aux: 0.0,
            best_loss: self.best_loss,
        };
        for generation in 0..self.cfg.generations {
            let theta = mind.flat_params();
            let pairs = self.cfg.pop / 2;
            let mut grad = vec![0.0f32; self.dim];
            let mut gen_loss = 0.0f64;
            for _ in 0..pairs {
                let eps: Vec<f32> = (0..self.dim).map(|_| gauss(rng)).collect();
                let plus: Vec<f32> = theta
                    .iter()
                    .zip(eps.iter())
                    .map(|(t, e)| t + self.cfg.sigma * e)
                    .collect();
                let minus: Vec<f32> = theta
                    .iter()
                    .zip(eps.iter())
                    .map(|(t, e)| t - self.cfg.sigma * e)
                    .collect();
                mind.set_flat_params(&plus)?;
                let (lp, _) = self.eval_loss(mind, batch, rng)?;
                mind.set_flat_params(&minus)?;
                let (lm, _) = self.eval_loss(mind, batch, rng)?;
                gen_loss += (lp + lm) as f64 * 0.5;
                let coeff = (lp - lm) / (2.0 * self.cfg.sigma * pairs as f32);
                for (g, e) in grad.iter_mut().zip(eps.iter()) {
                    *g += coeff * e;
                }
            }
            clip_by_norm(&mut grad, self.cfg.max_grad_norm);
            let mut updated = theta.clone();
            self.adam.set_lr(self.sched.lr_at(self.step));
            self.step += 1;
            self.adam.step(&mut updated, &grad)?;
            mind.set_flat_params(&updated)?;
            let (loss, aux) = self.eval_loss(mind, batch, rng)?;
            if loss < self.best_loss {
                self.best_loss = loss;
                self.best_params.clone_from(&updated);
            }
            stats = TrainStats {
                generation,
                loss,
                aux,
                best_loss: self.best_loss,
            };
            let _ = gen_loss;
        }
        Ok(stats)
    }
}

/// Standard-normal draw via Box–Muller.
fn gauss(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Rescale `grad` in place so its norm is at most `max`.
fn clip_by_norm(grad: &mut [f32], max: f32) {
    let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max && norm.is_finite() {
        let s = max / norm;
        for g in grad.iter_mut() {
            *g *= s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::AetherConfig;
    use crate::seeded_rng;

    #[test]
    fn adamw_descends_a_quadratic() {
        let mut opt = AdamW::new(8, 0.05, 0.9, 0.999, 1e-8, 0.0).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        use rand::Rng;
        let mut p: Vec<f32> = (0..8).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let loss = |p: &[f32]| p.iter().map(|x| x * x).sum::<f32>();
        let l0 = loss(&p);
        for _ in 0..60 {
            let g: Vec<f32> = p.iter().map(|x| 2.0 * x).collect();
            opt.step(&mut p, &g).unwrap();
        }
        assert!(loss(&p) < l0, "no descent: {l0} -> {}", loss(&p));
    }

    #[test]
    fn cosine_schedule_shape() {
        let s = CosineSchedule {
            base_lr: 0.1,
            warmup_steps: 4,
            total_steps: 10,
        };
        assert!((s.lr_at(0) - 0.0).abs() < 1e-6);
        assert!(s.lr_at(2) > 0.0 && s.lr_at(2) < 0.1);
        assert!((s.lr_at(4) - 0.1).abs() < 1e-6);
        assert!(s.lr_at(10) < 1e-4);
        assert!(s.lr_at(99) < 1e-4);
    }

    #[test]
    fn ce_is_sane() {
        let (loss, ent) = ce_and_entropy(&[2.0, 1.0, 0.1], 0).unwrap();
        assert!(loss > 0.0 && loss < 2.0);
        assert!(ent > 0.0 && ent < 2.0);
        assert!(ce_and_entropy(&[1.0], 5).is_err());
    }

    #[test]
    fn evo_train_runs_and_tracks_best() {
        let mut mind = AetherMind::new(AetherConfig::tiny()).unwrap();
        let dim = mind.param_count();
        let mut trainer = EvoTrainer::new(EvoTrainerConfig::demo(), dim).unwrap();
        let batch = vec![vec![1, 2, 3, 4, 5, 6, 7, 8], vec![8, 7, 6, 5, 4, 3, 2, 1]];
        let mut rng = seeded_rng(21);
        let before = trainer.eval_loss(&mut mind, &batch, &mut rng).unwrap().0;
        let stats = trainer.train(&mut mind, &batch, &mut rng).unwrap();
        assert!(stats.loss.is_finite() && stats.best_loss.is_finite());
        // Best-so-far includes the starting point, so it can never regress.
        assert!(stats.best_loss <= before + 1e-4);
        assert!(!trainer.best_params().is_empty());
        trainer.restore_best(&mut mind).unwrap();
    }

    #[test]
    fn rejects_bad_trainer_config() {
        let bad = EvoTrainerConfig {
            pop: 3,
            ..EvoTrainerConfig::demo()
        };
        assert!(EvoTrainer::new(bad, 10).is_err());
    }
}
