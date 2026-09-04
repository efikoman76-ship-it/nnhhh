//! AETHER — Adaptive Entangled Thought with Holographic Emergent Resonance.
//!
//! AETHER is a from-scratch neural AI architecture written in pure, dependency-light
//! Rust. Instead of stacking vanilla transformer blocks it fuses five ideas into one
//! organism:
//!
//! * a **hyperdimensional computing substrate** (`hypervec`) that reasons with
//!   10k-wide holographic vectors (bundling / binding / permutation),
//! * **resonant oscillatory neurons** (`resonance`) whose Kuramoto-style phase
//!   coupling synchronises representations the way rhythms synchronise brains,
//! * **entangled attention** (`attention`) blending a local sliding-window branch
//!   with a holographic global gist branch through a learned resonance gate,
//! * a **sparse mixture of SwiGLU experts** (`moe`) with noisy top-k routing and a
//!   load-balancing auxiliary loss,
//! * a **fractal memory hierarchy** (`memory`) — sensory ring, working set, decaying
//!   episodic store and consolidated semantic prototypes — plus Hebbian plasticity
//!   (`plasticity`) and a derivative-free evolutionary trainer (`trainer`) because
//!   the resonant/holographic substrate is non-differentiable by design.
//!
//! A divergent **creativity engine** (`creativity`) samples forests of candidates,
//! blends concepts with holographic binding, and hill-climbs a novelty + quality
//! objective (`dream`).
//!
//! Quick start:
//!
//! ```
//! use aether::prelude::*;
//!
//! let mut mind = AetherMind::new(AetherConfig::tiny()).unwrap();
//! assert!(mind.param_count() > 0);
//!
//! let mut rng = seeded_rng(42);
//! let out = mind.forward(&[1, 2, 3, 4], &mut rng).unwrap();
//! assert_eq!(out.logits.nrows(), 4);
//! ```

pub mod attention;
pub mod creativity;
pub mod error;
pub mod hypervec;
pub mod memory;
pub mod moe;
pub mod network;
pub mod plasticity;
pub mod resonance;
pub mod tensor;
pub mod tokenizer;
pub mod trainer;

use rand::rngs::StdRng;
use rand::SeedableRng;

/// Build a deterministic RNG shared by examples, tests and the CLI.
pub fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// The most-used types, re-exported for one-line imports.
pub mod prelude {
    pub use crate::attention::{EntangledAttention, EntangledAttentionConfig};
    pub use crate::creativity::{
        conceptual_blend, CreativityConfig, CreativityEngine, DreamReport, NoveltyArchive,
    };
    pub use crate::error::{AetherError, Result};
    pub use crate::hypervec::{HoloMemory, HyperVec};
    pub use crate::memory::{FractalMemory, FractalMemoryConfig, MemoryStats};
    pub use crate::moe::{SparseMoe, SparseMoeConfig, SwiGluExpert};
    pub use crate::network::{AetherConfig, AetherMind, ForwardOut, SampleConfig};
    pub use crate::plasticity::{HebbianBank, Homeostat};
    pub use crate::resonance::{ResonantLayer, ResonantSpec};
    pub use crate::seeded_rng;
    pub use crate::tensor::Matrix;
    pub use crate::tokenizer::BpeTokenizer;
    pub use crate::trainer::{AdamW, CosineSchedule, EvoTrainer, EvoTrainerConfig, TrainStats};
}
