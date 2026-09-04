//! Divergent creativity engine: forests, blends and dreams.
//!
//! Where the trainer converges, this module diverges on purpose:
//!
//! * [`NoveltyArchive`] remembers everything the mind has made and scores new
//!   ideas with `novelty = 1 - max_cosine_to_archive` (pure novelty search);
//! * [`conceptual_blend`] fuses two concept vectors holographically —
//!   normalised sum plus their binding product, so the blend carries both
//!   ingredients *and* their association;
//! * [`CreativityEngine::dream`] hill-climbs `quality + novelty`: it samples a
//!   forest of continuations at several temperatures, has the mind rate its
//!   own coherence (mean self-predicted token probability), archives what it
//!   saw, and branches the next iteration from the winner.

use crate::error::{AetherError, Result};
use crate::hypervec::cosine_slices;
use crate::network::{AetherMind, SampleConfig};
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Knobs for divergent generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativityConfig {
    /// Candidates sampled per iteration.
    pub n_candidates: usize,
    /// Temperatures cycled across candidates (the divergent forest).
    pub temperatures: Vec<f32>,
    /// Weight of novelty in the dream objective.
    pub novelty_weight: f32,
    /// Weight of self-rated coherence in the dream objective.
    pub quality_weight: f32,
    /// Dream refinement iterations.
    pub max_iters: usize,
    /// Tokens per candidate.
    pub candidate_len: usize,
}

impl Default for CreativityConfig {
    fn default() -> CreativityConfig {
        CreativityConfig {
            n_candidates: 6,
            temperatures: vec![0.6, 0.9, 1.2],
            novelty_weight: 1.0,
            quality_weight: 1.0,
            max_iters: 4,
            candidate_len: 12,
        }
    }
}

impl CreativityConfig {
    /// Validate ranges.
    pub fn validate(&self) -> Result<()> {
        if self.n_candidates == 0 || self.max_iters == 0 || self.candidate_len == 0 {
            return Err(AetherError::InvalidConfig(
                "n_candidates, max_iters and candidate_len must be > 0".to_string(),
            ));
        }
        if self.temperatures.is_empty()
            || self.temperatures.iter().any(|t| !t.is_finite() || *t <= 0.0)
        {
            return Err(AetherError::InvalidConfig(
                "temperatures must be non-empty, finite and > 0".to_string(),
            ));
        }
        if self.novelty_weight < 0.0 || self.quality_weight < 0.0 {
            return Err(AetherError::InvalidConfig("weights must be >= 0".to_string()));
        }
        Ok(())
    }
}

/// Memory of everything created, for novelty search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyArchive {
    items: Vec<Vec<f32>>,
    cap: usize,
    dim: usize,
}

impl NoveltyArchive {
    /// Empty archive holding at most `cap` fingerprints.
    pub fn new(cap: usize) -> Result<NoveltyArchive> {
        if cap == 0 {
            return Err(AetherError::InvalidConfig("archive cap is 0".to_string()));
        }
        Ok(NoveltyArchive {
            items: Vec::new(),
            cap,
            dim: 0,
        })
    }

    /// Novelty of `vec`: 1 when the archive is empty (everything is new),
    /// else `1 - max_cosine`, living in [0, 2].
    pub fn novelty(&self, vec: &[f32]) -> Result<f32> {
        if self.items.is_empty() {
            return Ok(1.0);
        }
        if vec.len() != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "archive dim {} vs {}",
                self.dim,
                vec.len()
            )));
        }
        let mut best = f32::NEG_INFINITY;
        for item in &self.items {
            let sim = cosine_slices(item, vec);
            if sim > best {
                best = sim;
            }
        }
        Ok(1.0 - best)
    }

    /// Remember a fingerprint, evicting the oldest past capacity.
    pub fn add(&mut self, vec: Vec<f32>) -> Result<()> {
        if vec.is_empty() {
            return Err(AetherError::EmptyInput("archive got empty vector".to_string()));
        }
        if self.dim == 0 {
            self.dim = vec.len();
        }
        if vec.len() != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "archive dim {} vs {}",
                self.dim,
                vec.len()
            )));
        }
        self.items.push(vec);
        while self.items.len() > self.cap {
            self.items.remove(0);
        }
        Ok(())
    }

    /// Stored fingerprints.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True when nothing was ever created.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// Fuse two concept vectors: normalised sum plus their binding product.
///
/// `mix` blends from `a` (0) to `b` (1); the `0.5 * a * b` term carries the
/// holographic association of the pair, so the blend is more than an average.
pub fn conceptual_blend(a: &[f32], b: &[f32], mix: f32) -> Result<Vec<f32>> {
    if a.is_empty() || a.len() != b.len() {
        return Err(AetherError::ShapeMismatch(format!(
            "blend needs equal non-empty vectors, got {}/{}",
            a.len(),
            b.len()
        )));
    }
    if !(0.0..=1.0).contains(&mix) {
        return Err(AetherError::InvalidConfig(format!("blend mix {mix} not in [0,1]")));
    }
    let mut out = Vec::with_capacity(a.len());
    for (x, y) in a.iter().zip(b.iter()) {
        out.push((1.0 - mix) * x + mix * y + 0.5 * x * y);
    }
    let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-9 {
        return Err(AetherError::InvalidConfig("blend collapsed to zero".to_string()));
    }
    for v in out.iter_mut() {
        *v /= norm;
    }
    Ok(out)
}

/// What a dream produced.
#[derive(Debug, Clone)]
pub struct DreamReport {
    /// Winning token ids.
    pub best_ids: Vec<usize>,
    /// Its combined score.
    pub best_score: f32,
    /// Best-so-far after each iteration.
    pub history: Vec<f32>,
    /// Total candidates scored.
    pub candidates_considered: usize,
}

/// The divergent engine itself.
#[derive(Debug)]
pub struct CreativityEngine {
    cfg: CreativityConfig,
    archive: NoveltyArchive,
}

impl CreativityEngine {
    /// Build an engine with a fresh archive.
    pub fn new(cfg: CreativityConfig) -> Result<CreativityEngine> {
        cfg.validate()?;
        Ok(CreativityEngine {
            cfg,
            archive: NoveltyArchive::new(256)?,
        })
    }

    /// Fingerprints remembered so far.
    pub fn archive_len(&self) -> usize {
        self.archive.len()
    }

    /// Novelty of a fingerprint against the archive.
    pub fn novelty_of(&self, vec: &[f32]) -> Result<f32> {
        self.archive.novelty(vec)
    }

    /// Sample one forest of candidates from `prompt` at cycled temperatures.
    pub fn diverge(
        &self,
        mind: &mut AetherMind,
        prompt: &[usize],
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<usize>>> {
        if prompt.is_empty() {
            return Err(AetherError::EmptyInput("diverge got empty prompt".to_string()));
        }
        let mut forest = Vec::with_capacity(self.cfg.n_candidates);
        for i in 0..self.cfg.n_candidates {
            let sample = SampleConfig {
                temperature: self.cfg.temperatures[i % self.cfg.temperatures.len()],
                top_k: 0,
                top_p: 0.0,
                stop_id: None,
            };
            forest.push(mind.generate(prompt, self.cfg.candidate_len, &sample, rng)?);
        }
        Ok(forest)
    }

    /// Score a candidate: self-rated coherence + archive novelty.
    fn score_candidate(
        &self,
        mind: &mut AetherMind,
        prompt: &[usize],
        cand: &[usize],
        rng: &mut StdRng,
    ) -> Result<(f32, f32, f32, Vec<f32>)> {
        let emb = mind.embed_mean(cand)?;
        let nov = self.archive.novelty(&emb)?;
        let max_seq = mind.config().max_seq;
        let mut combined = Vec::with_capacity(prompt.len() + cand.len());
        combined.extend_from_slice(prompt);
        combined.extend_from_slice(cand);
        if combined.len() > max_seq {
            combined = combined[combined.len() - max_seq..].to_vec();
        }
        let mut quality = 0.0f32;
        if combined.len() >= 2 {
            let fo = mind.forward(&combined[..combined.len() - 1], rng)?;
            let positions = cand.len().min(combined.len() - 1);
            let start = combined.len() - 1 - positions;
            for row in start..combined.len() - 1 {
                let probs = softmax_owned(fo.logits.row(row));
                quality += probs[combined[row + 1]];
            }
            quality /= positions.max(1) as f32;
        }
        let total = self.cfg.quality_weight * quality + self.cfg.novelty_weight * nov;
        Ok((quality, nov, total, emb))
    }

    /// Dream: iterate forests, archive everything seen, branch from the best.
    pub fn dream(
        &mut self,
        mind: &mut AetherMind,
        prompt: &[usize],
        rng: &mut StdRng,
    ) -> Result<DreamReport> {
        if prompt.is_empty() {
            return Err(AetherError::EmptyInput("dream got empty prompt".to_string()));
        }
        let max_seq = mind.config().max_seq;
        let mut current = prompt.to_vec();
        let mut best_ids: Vec<usize> = Vec::new();
        let mut best_score = f32::NEG_INFINITY;
        let mut history = Vec::with_capacity(self.cfg.max_iters);
        let mut considered = 0usize;
        for _ in 0..self.cfg.max_iters {
            let forest = self.diverge(mind, &current, rng)?;
            considered += forest.len();
            for cand in &forest {
                let (_, _, total, emb) = self.score_candidate(mind, &current, cand, rng)?;
                self.archive.add(emb)?;
                if total > best_score {
                    best_score = total;
                    best_ids.clone_from(cand);
                }
            }
            // Branch the next forest from the prompt fused with the winner.
            let mut grown = prompt.to_vec();
            grown.extend_from_slice(&best_ids);
            if grown.len() > max_seq {
                grown = grown[grown.len() - max_seq..].to_vec();
            }
            current = grown;
            history.push(best_score);
            let _ = rng.gen::<u32>();
        }
        Ok(DreamReport {
            best_ids,
            best_score,
            history,
            candidates_considered: considered,
        })
    }
}

/// Owned stable softmax.
fn softmax_owned(xs: &[f32]) -> Vec<f32> {
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
    use crate::network::AetherConfig;
    use crate::seeded_rng;

    #[test]
    fn novelty_bounds_and_learning() {
        let mut arch = NoveltyArchive::new(4).unwrap();
        let v = vec![1.0f32, 0.0, 0.0, 0.0];
        assert_eq!(arch.novelty(&v).unwrap(), 1.0);
        arch.add(v.clone()).unwrap();
        let same = arch.novelty(&v).unwrap();
        assert!(same.abs() < 1e-5, "same {same}");
        let other = arch.novelty(&[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert!((other - 1.0).abs() < 1e-5, "orthogonal {other}");
        assert!(arch.novelty(&[-1.0, 0.0, 0.0, 0.0]).unwrap() > 1.0);
    }

    #[test]
    fn blend_is_unit_and_between() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let m = conceptual_blend(&a, &b, 0.5).unwrap();
        let norm: f32 = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        // Symmetric inputs give a symmetric blend.
        assert!((m[0] - m[1]).abs() < 1e-5);
        assert!(conceptual_blend(&a, &b, 2.0).is_err());
    }

    #[test]
    fn dream_produces_scored_best() {
        let mut mind = AetherMind::new(AetherConfig::tiny()).unwrap();
        let cfg = CreativityConfig {
            n_candidates: 3,
            candidate_len: 6,
            max_iters: 2,
            ..CreativityConfig::default()
        };
        let mut engine = CreativityEngine::new(cfg).unwrap();
        let report = engine.dream(&mut mind, &[1, 2, 3], &mut seeded_rng(9)).unwrap();
        assert_eq!(report.best_ids.len(), 6);
        assert!(report.best_score.is_finite());
        assert_eq!(report.history.len(), 2);
        assert_eq!(report.candidates_considered, 6);
        assert!(engine.archive_len() > 0);
        for w in report.history.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "best-so-far regressed");
        }
    }

    #[test]
    fn dream_is_deterministic() {
        let run = || {
            let mut mind = AetherMind::new(AetherConfig::tiny()).unwrap();
            let cfg = CreativityConfig {
                n_candidates: 2,
                candidate_len: 5,
                max_iters: 2,
                ..CreativityConfig::default()
            };
            let mut engine = CreativityEngine::new(cfg).unwrap();
            engine.dream(&mut mind, &[4, 5], &mut seeded_rng(3)).unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.best_ids, b.best_ids);
    }
}
