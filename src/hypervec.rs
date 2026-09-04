//! Hyperdimensional computing substrate.
//!
//! The holographic heart of AETHER. Concepts are represented as very wide
//! (`dim` ≈ thousands) pseudo-orthogonal vectors. Three operators give a full
//! algebra of thought:
//!
//! * **bundling** (`bundle`) — superposition / set union: the result stays
//!   similar to every input (≈0.7 cosine for pairs),
//! * **binding** (`bind`) — association / variable-value pairing: the result is
//!   dissimilar to its inputs yet exactly invertible (`unbind(bind(a,b),b)=a`),
//! * **permutation** (`permute`) — sequence order / role quoting: a cyclic shift
//!   decorrelates while preserving structure.
//!
//! `HoloMemory` stores hundreds of items superposed in one vector and retrieves
//! them by cosine cleanup, exactly as validated in the design prototype.

use crate::error::{AetherError, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// A wide holographic vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperVec {
    dim: usize,
    data: Vec<f32>,
}

impl HyperVec {
    /// Random bipolar vector, entries in {-1, +1}.
    pub fn random(rng: &mut StdRng, dim: usize) -> Result<HyperVec> {
        if dim == 0 {
            return Err(AetherError::InvalidConfig("hypervec dim is 0".to_string()));
        }
        let data: Vec<f32> = (0..dim)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        Ok(HyperVec { dim, data })
    }

    /// Deterministic random vector from a seed.
    pub fn random_seeded(seed: u64, dim: usize) -> Result<HyperVec> {
        let mut rng = StdRng::seed_from_u64(seed);
        HyperVec::random(&mut rng, dim)
    }

    /// Zero vector.
    pub fn zeros(dim: usize) -> HyperVec {
        HyperVec {
            dim,
            data: vec![0.0; dim],
        }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Raw slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    fn check(&self, other: &HyperVec, op: &str) -> Result<()> {
        if self.dim != other.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "{op}: dim {} vs {}",
                self.dim, other.dim
            )));
        }
        Ok(())
    }

    /// Cosine similarity in [-1, 1].
    pub fn cosine(&self, other: &HyperVec) -> Result<f32> {
        self.check(other, "cosine")?;
        Ok(cosine_slices(&self.data, &other.data))
    }

    /// Superpose `a` and `b`, renormalised to keep energy ~ sqrt(dim).
    pub fn bundle(a: &HyperVec, b: &HyperVec) -> Result<HyperVec> {
        a.check(b, "bundle")?;
        let mut data = Vec::with_capacity(a.dim);
        for (x, y) in a.data.iter().zip(b.data.iter()) {
            data.push(x + y);
        }
        let mut v = HyperVec { dim: a.dim, data };
        v.renormalise();
        Ok(v)
    }

    /// Superpose many vectors with per-item weights.
    pub fn weighted_bundle(items: &[(&HyperVec, f32)]) -> Result<HyperVec> {
        if items.is_empty() {
            return Err(AetherError::EmptyInput("weighted_bundle got no items".to_string()));
        }
        let dim = items[0].0.dim;
        let mut acc = vec![0.0f32; dim];
        for (v, w) in items {
            if v.dim != dim {
                return Err(AetherError::ShapeMismatch(format!(
                    "weighted_bundle: dim {dim} vs {}",
                    v.dim
                )));
            }
            for (a, x) in acc.iter_mut().zip(v.data.iter()) {
                *a += w * x;
            }
        }
        let mut out = HyperVec { dim, data: acc };
        out.renormalise();
        Ok(out)
    }

    /// Bind (associate) two vectors. Self-inverse: `unbind(bind(a,b),b) == a`.
    pub fn bind(a: &HyperVec, b: &HyperVec) -> Result<HyperVec> {
        a.check(b, "bind")?;
        let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect();
        Ok(HyperVec { dim: a.dim, data })
    }

    /// Exact inverse of [`HyperVec::bind`].
    pub fn unbind(bound: &HyperVec, b: &HyperVec) -> Result<HyperVec> {
        HyperVec::bind(bound, b)
    }

    /// Cyclic shift — encodes order/role while decorrelating.
    pub fn permute(&self, shift: usize) -> HyperVec {
        let mut data = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            data[(i + shift) % self.dim] = self.data[i];
        }
        HyperVec { dim: self.dim, data }
    }

    /// Encode an ordered n-gram: `permute` each item by its position, then bundle.
    pub fn ngram(items: &[HyperVec]) -> Result<HyperVec> {
        if items.is_empty() {
            return Err(AetherError::EmptyInput("ngram got no items".to_string()));
        }
        let dim = items[0].dim;
        let mut acc = vec![0.0f32; dim];
        for (pos, v) in items.iter().enumerate() {
            if v.dim != dim {
                return Err(AetherError::ShapeMismatch(format!(
                    "ngram: dim {dim} vs {}",
                    v.dim
                )));
            }
            for i in 0..dim {
                acc[(i + pos) % dim] += v.data[i];
            }
        }
        let mut out = HyperVec { dim, data: acc };
        out.renormalise();
        Ok(out)
    }

    /// Scale so the Euclidean norm equals `sqrt(dim)` (bipolar energy level).
    fn renormalise(&mut self) {
        let norm = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let target = (self.dim as f32).sqrt();
            for x in self.data.iter_mut() {
                *x *= target / norm;
            }
        }
    }
}

/// Cosine similarity over raw slices.
pub fn cosine_slices(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Holographic associative memory: many items superposed in one vector.
///
/// Storage is O(dim) regardless of item count; retrieval is cosine cleanup.
/// Capacity degrades gracefully instead of failing — the first property that
/// makes it a good substrate for creative memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoloMemory {
    dim: usize,
    superposition: Vec<f32>,
    count: usize,
}

impl HoloMemory {
    /// Empty memory for `dim`-wide vectors.
    pub fn new(dim: usize) -> Result<HoloMemory> {
        if dim == 0 {
            return Err(AetherError::InvalidConfig("HoloMemory dim is 0".to_string()));
        }
        Ok(HoloMemory {
            dim,
            superposition: vec![0.0; dim],
            count: 0,
        })
    }

    /// Superpose `item` with the given weight.
    pub fn store(&mut self, item: &HyperVec, weight: f32) -> Result<()> {
        if item.dim != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "HoloMemory::store: dim {} vs {}",
                self.dim, item.dim
            )));
        }
        for (s, x) in self.superposition.iter_mut().zip(item.data.iter()) {
            *s += weight * x;
        }
        self.count += 1;
        Ok(())
    }

    /// Cosine similarity of a query against the whole superposition.
    pub fn familiarity(&self, query: &HyperVec) -> Result<f32> {
        if query.dim != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "HoloMemory::familiarity: dim {} vs {}",
                self.dim, query.dim
            )));
        }
        Ok(cosine_slices(&self.superposition, &query.data))
    }

    /// Best-match cleanup: score every candidate, return indices sorted best-first.
    pub fn cleanup(&self, query: &HyperVec, candidates: &[HyperVec]) -> Result<Vec<usize>> {
        if query.dim != self.dim {
            return Err(AetherError::ShapeMismatch(format!(
                "HoloMemory::cleanup: dim {} vs {}",
                self.dim, query.dim
            )));
        }
        let mut scored: Vec<(usize, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_slices(&query.data, &c.data)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().map(|(i, _)| i).collect())
    }

    /// Number of stored items.
    pub fn len(&self) -> usize {
        self.count
    }

    /// True when nothing was stored yet.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(seed: u64, dim: usize) -> HyperVec {
        HyperVec::random_seeded(seed, dim).unwrap()
    }

    #[test]
    fn self_similarity_is_one_cross_is_zero() {
        let a = hv(1, 1024);
        let b = hv(2, 1024);
        assert!((a.cosine(&a).unwrap() - 1.0).abs() < 1e-5);
        assert!(a.cosine(&b).unwrap().abs() < 0.15);
    }

    #[test]
    fn bundle_stays_similar_to_inputs() {
        let a = hv(11, 1024);
        let b = hv(12, 1024);
        let ab = HyperVec::bundle(&a, &b).unwrap();
        assert!((ab.cosine(&a).unwrap() - 0.707).abs() < 0.1);
        assert!((ab.cosine(&b).unwrap() - 0.707).abs() < 0.1);
    }

    #[test]
    fn bind_is_self_inverting() {
        let a = hv(21, 256);
        let b = hv(22, 256);
        let bound = HyperVec::bind(&a, &b).unwrap();
        // Bound vector is dissimilar to its inputs ...
        assert!(bound.cosine(&a).unwrap().abs() < 0.2);
        // ... yet perfectly invertible.
        let back = HyperVec::unbind(&bound, &b).unwrap();
        assert!((back.cosine(&a).unwrap() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn permute_decorrelates() {
        let a = hv(31, 512);
        let p = a.permute(7);
        assert!(a.cosine(&p).unwrap().abs() < 0.2);
        // Same shift twice composes.
        assert!((a.permute(14).cosine(&p.permute(7)).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn holo_memory_recalls_superposed_items() {
        let dim = 1024;
        let mut mem = HoloMemory::new(dim).unwrap();
        let items: Vec<HyperVec> = (0..50).map(|s| hv(1000 + s, dim)).collect();
        for it in &items {
            mem.store(it, 1.0).unwrap();
        }
        // Every stored item stays positively familiar despite 49 interferers.
        for it in &items {
            assert!(mem.familiarity(it).unwrap() > 0.05);
        }
        // Cleanup finds the exact nearest candidate.
        let order = mem.cleanup(&items[7], &items).unwrap();
        assert_eq!(order[0], 7);
    }

    #[test]
    fn ngram_order_matters() {
        let a = hv(41, 256);
        let b = hv(42, 256);
        let ab = HyperVec::ngram(&[a.clone(), b.clone()]).unwrap();
        let ba = HyperVec::ngram(&[b, a]).unwrap();
        // Different order -> different encoding.
        assert!(ab.cosine(&ba).unwrap().abs() < 0.9);
    }
}
