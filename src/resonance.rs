//! Resonant oscillatory neurons.
//!
//! Each unit is an adaptive oscillator with its own frequency, damping and
//! phase. Units couple through a Kuramoto-style mean-field term, so the layer
//! spontaneously synchronises representations that "belong together" —
//! temporal binding without any binding supervision. Amplitude relaxes toward
//! the rectified drive, so quiet channels fade and driven channels sing.
//!
//! With zero drive and zero coupling the layer reduces to a bank of pure
//! oscillators; with strong coupling the order parameter (phase coherence)
//! climbs toward 1, exactly as in the classical Kuramoto model.

use crate::error::{AetherError, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Construction parameters for a resonant layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantSpec {
    /// Number of oscillator units (usually = d_model).
    pub n_units: usize,
    /// Centre frequency in rad/s.
    pub base_freq: f32,
    /// Relative frequency spread in [0, 1).
    pub freq_jitter: f32,
    /// Amplitude relaxation rate.
    pub damping: f32,
    /// Kuramoto mean-field coupling strength.
    pub coupling: f32,
    /// Integration time step.
    pub dt: f32,
    /// Gain applied to the external drive.
    pub drive_gain: f32,
}

impl Default for ResonantSpec {
    fn default() -> ResonantSpec {
        ResonantSpec {
            n_units: 32,
            base_freq: 1.0,
            freq_jitter: 0.05,
            damping: 2.0,
            coupling: 1.5,
            dt: 0.05,
            drive_gain: 0.5,
        }
    }
}

impl ResonantSpec {
    /// Check ranges; called by every constructor downstream.
    pub fn validate(&self) -> Result<()> {
        if self.n_units == 0 {
            return Err(AetherError::InvalidConfig("resonant n_units is 0".to_string()));
        }
        if !(0.0..1.0).contains(&self.freq_jitter) {
            return Err(AetherError::InvalidConfig(format!(
                "freq_jitter {} not in [0,1)",
                self.freq_jitter
            )));
        }
        if self.damping < 0.0 || self.dt <= 0.0 || self.drive_gain < 0.0 {
            return Err(AetherError::InvalidConfig(
                "damping/dt/drive_gain must be non-negative with dt > 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// A bank of coupled adaptive oscillators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantLayer {
    spec: ResonantSpec,
    seed: u64,
    pub(crate) freq: Vec<f32>,
    phase: Vec<f32>,
    amp: Vec<f32>,
}

impl ResonantLayer {
    /// Build a layer from a spec, deterministically seeded.
    pub fn new(spec: ResonantSpec, seed: u64) -> Result<ResonantLayer> {
        spec.validate()?;
        let mut layer = ResonantLayer {
            spec,
            seed,
            freq: Vec::new(),
            phase: Vec::new(),
            amp: Vec::new(),
        };
        layer.reset();
        Ok(layer)
    }

    /// Re-initialise phases and amplitudes from the stored seed.
    pub fn reset(&mut self) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n = self.spec.n_units;
        self.freq = (0..n)
            .map(|_| {
                let jitter = rng.gen_range(-self.spec.freq_jitter..=self.spec.freq_jitter);
                self.spec.base_freq * (1.0 + jitter)
            })
            .collect();
        self.phase = (0..n).map(|_| rng.gen_range(0.0..2.0 * std::f32::consts::PI)).collect();
        self.amp = vec![0.5; n];
    }

    /// Override natural frequencies (used by tests and neuroevolution).
    pub fn set_freq(&mut self, freq: &[f32]) -> Result<()> {
        if freq.len() != self.spec.n_units {
            return Err(AetherError::ShapeMismatch(format!(
                "set_freq needs {} values, got {}",
                self.spec.n_units,
                freq.len()
            )));
        }
        self.freq = freq.to_vec();
        Ok(())
    }

    /// Kuramoto order parameter: 0 = incoherent, 1 = fully synchronised.
    pub fn order_parameter(&self) -> f32 {
        let n = self.phase.len() as f32;
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for p in &self.phase {
            re += p.cos();
            im += p.sin();
        }
        ((re * re + im * im).sqrt()) / n.max(1.0)
    }

    /// Advance one step under `drive` and return the oscillatory output.
    pub fn step(&mut self, drive: &[f32]) -> Result<Vec<f32>> {
        let n = self.spec.n_units;
        if drive.len() != n {
            return Err(AetherError::ShapeMismatch(format!(
                "resonant step needs {n} drive values, got {}",
                drive.len()
            )));
        }
        // Mean-field coupling: average pull of the whole population on each unit.
        let mut pull = vec![0.0f32; n];
        if self.spec.coupling != 0.0 {
            for i in 0..n {
                let mut acc = 0.0f32;
                for other in &self.phase {
                    acc += (other - self.phase[i]).sin();
                }
                pull[i] = self.spec.coupling * acc / n as f32;
            }
        }
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            self.phase[i] += self.spec.dt
                * (self.freq[i] + self.spec.drive_gain * drive[i] + pull[i]);
            let target = (self.spec.drive_gain * drive[i]).tanh().abs();
            self.amp[i] += self.spec.dt * self.spec.damping * (target - self.amp[i]);
            if self.amp[i] < 0.0 {
                self.amp[i] = 0.0;
            }
            out[i] = self.amp[i] * self.phase[i].sin();
        }
        Ok(out)
    }

    /// Modulate a residual stream in place: `xs += strength * step(xs)`.
    pub fn resonate(&mut self, xs: &mut [f32], strength: f32) -> Result<()> {
        let out = self.step(xs)?;
        for (x, o) in xs.iter_mut().zip(out.iter()) {
            *x += strength * o;
        }
        Ok(())
    }

    /// Number of units.
    pub fn n_units(&self) -> usize {
        self.spec.n_units
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_oscillator_advances_by_freq() {
        let spec = ResonantSpec {
            n_units: 1,
            base_freq: 2.0,
            freq_jitter: 0.0,
            damping: 0.0,
            coupling: 0.0,
            dt: 0.1,
            drive_gain: 0.0,
        };
        let mut layer = ResonantLayer::new(spec, 9).unwrap();
        layer.set_freq(&[2.0]).unwrap();
        let p0 = layer.phase[0];
        layer.step(&[0.0]).unwrap();
        assert!((layer.phase[0] - (p0 + 0.2)).abs() < 1e-5);
    }

    #[test]
    fn coupling_synchronises_population() {
        let n = 24;
        let spec = ResonantSpec {
            n_units: n,
            base_freq: 1.0,
            freq_jitter: 0.0,
            damping: 0.0,
            coupling: 2.0,
            dt: 0.05,
            drive_gain: 0.0,
        };
        let mut layer = ResonantLayer::new(spec, 7).unwrap();
        // Prototype-style frequency spread.
        let mut rng = StdRng::seed_from_u64(7);
        let mut omega = vec![0.0f32; n];
        // Box-Muller-lite: sum of uniforms approximates a normal draw.
        for w in omega.iter_mut() {
            let u: f32 = (0..6).map(|_| rng.gen_range(0.0..1.0)).sum();
            *w = 1.0 + (u - 3.0) * 0.25;
        }
        layer.set_freq(&omega).unwrap();
        let before = layer.order_parameter();
        let drive = vec![0.0f32; n];
        for _ in 0..200 {
            layer.step(&drive).unwrap();
        }
        let after = layer.order_parameter();
        assert!(after > before, "order {before:.3} -> {after:.3}");
        assert!(after > 0.6, "order {after:.3}");
    }

    #[test]
    fn outputs_stay_finite_and_bounded() {
        let mut layer = ResonantLayer::new(ResonantSpec::default(), 3).unwrap();
        let drive: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        for _ in 0..50 {
            let out = layer.step(&drive).unwrap();
            assert!(out.iter().all(|x| x.is_finite() && x.abs() <= 1.5));
        }
    }

    #[test]
    fn reset_reproduces_initial_state() {
        let mut layer = ResonantLayer::new(ResonantSpec::default(), 11).unwrap();
        let p0 = layer.phase.clone();
        let drive = vec![1.0f32; 32];
        layer.step(&drive).unwrap();
        layer.reset();
        assert_eq!(layer.phase, p0);
    }
}
