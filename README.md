# AETHER

> **Adaptive Entangled Thought with Holographic Emergent Resonance** — an
> entirely new neural AI architecture, written from scratch in pure Rust.

[![AETHER CI](https://github.com/efikoman76-ship-it/nnhhh/actions/workflows/ci.yml/badge.svg)](https://github.com/efikoman76-ship-it/nnhhh/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](LICENSE-MIT)
[![Rust](https://img.shields.io/badge/rust-1.81%2B-orange.svg)](https://www.rust-lang.org)

AETHER is not another transformer wrapper. It is a from-scratch cognitive
organism that fuses **hyperdimensional computing**, **oscillatory resonance**,
**entangled attention**, **sparse expertise**, **fractal memory** and
**derivative-free evolution** into one stateful mind — with zero heavyweight
ML dependencies (`rand`, `rayon`, `serde`, `thiserror` only).

```text
                    ┌─────────────────────────────────────────┐
                    │              AETHER MIND                │
  tokens ──▶ embed + sinusoidal pos                          │
                    │                                         │
                    ▼                                         │
           ┌── LayerNorm ──▶ EntangledAttention ──┐            │
           │   (local window ⊕ holographic gist,  │            │
           │    mixed by resonance gate)          │ +          │
           │                                     ▼            │
           │                          Resonant oscillators    │
           │                          (Kuramoto sync)    ──┐   │
           │                                              │ + │
           │   LayerNorm ──▶ SparseMoE (top-k SwiGLU) ────┘   │
           │                                              │ + │
           └── Homeostat (adaptive gain) ◀────────────────┘   │
                    │ sense gist                                  │
                    ▼                                         │
     ┌────────────────────────────────────────────┐           │
     │ FRACTAL MEMORY                             │           │
     │ sensory ring → working set → episodic      │──recall──▶│ blend back
     │ (Ebbinghaus decay) → semantic prototypes   │  + Hebbian│  into stream
     └────────────────────────────────────────────┘  bias     │
                    │                                         │
                    ▼                                         │
              LayerNorm ──▶ logits ◀── generate / dream ──────┘
```

## Why it is new

| Idea | What AETHER does |
|------|------------------|
| Holographic substrate | Concepts live in kilo-wide holographic vectors with bundling / binding / permutation algebra and superposition memory |
| Resonant neurons | Oscillators with Kuramoto coupling synchronise bound representations; amplitude relaxes toward drive |
| Entangled attention | Local sliding-window branch **entangled** with a global gist branch via a learned per-position gate; exactly permutation-equivariant |
| Sparse experts | Noisy top-k routed SwiGLU experts with a provable load-balance floor |
| Fractal memory | Four tiers with rehearsal, Ebbinghaus decay, consolidation and strength-weighted recall — live on every forward pass |
| Plasticity | Oja Hebbian prototypes bias retrieval; homeostatic gain prevents blow-up; STDP window available |
| Evolution, not backprop | The substrate is non-differentiable by design, so training is antithetic ES + AdamW + cosine schedule |
| Creativity as search | Divergent temperature forests + novelty archive + holographic blending + self-rated coherence hill-climbing (`dream`) |

## Quick start

```sh
cargo run --release --bin aether -- info --preset tiny
cargo run --release --bin aether -- demo
cargo run --release --bin aether -- train --gens 5 --pop 6
cargo run --release --bin aether -- dream --iters 4 --prompt 1,2,3
cargo run --release --bin aether -- tokenize --text "hello brave world"
cargo run --release --example dreaming
```

```rust
use aether::prelude::*;

let mut mind = AetherMind::new(AetherConfig::tiny())?;
let mut rng = seeded_rng(42);

// Stateful forward: plasticity + memory fire on every call.
let out = mind.forward(&[1, 2, 3, 4], &mut rng)?;
assert_eq!(out.logits.nrows(), 4);

// Generate with temperature + nucleus sampling.
let cont = mind.generate(&[1, 2, 3], 16, &SampleConfig {
    temperature: 0.8, top_k: 8, top_p: 0.9, stop_id: None,
}, &mut rng)?;

// Dream: divergent forest + novelty hill-climbing.
let mut engine = CreativityEngine::new(CreativityConfig::default())?;
let report = engine.dream(&mut mind, &[1, 2, 3], &mut rng)?;
println!("dreamt {:?} (score {:.3})", report.best_ids, report.best_score);

// Snapshot the whole organism.
mind.save_to("mind.json")?;
```

## Module map

| Module | Role |
|--------|------|
| `tensor` | Row-major f32 matrix, rayon-parallel matmul/softmax/norm/activations |
| `hypervec` | Holographic vectors + `HoloMemory` superposition store |
| `resonance` | Kuramoto oscillator bank, order parameter, residual modulation |
| `attention` | Entangled local ⊕ holographic attention + resonance gate |
| `moe` | SwiGLU experts, noisy top-k router, balance loss |
| `memory` | Sensory / working / episodic / semantic tiers + recall |
| `plasticity` | Oja bank, homeostat, STDP window |
| `network` | `AetherMind`, sampling, generation, flat params, snapshots |
| `tokenizer` | Deterministic trainable BPE, save/load |
| `trainer` | AdamW, cosine schedule, antithetic ES trainer |
| `creativity` | Novelty archive, blending, dreaming |

## Training

Because oscillators, holographic binding and top-k routing block gradients,
AETHER trains the honest way for such substrates:

1. **Online, every forward pass** — Hebbian prototypes track the hidden
   stream, the homeostat adapts gains, memory consolidates.
2. **Offline, evolutionarily** — `EvoTrainer` scores antithetic Gaussian
   perturbations with teacher-forced cross-entropy (+ MoE balance loss,
   − entropy bonus), estimates the ES gradient, clips it, and steps AdamW
   under a cosine schedule while tracking best-so-far.

## Testing & CI — all out

Every push runs the full gauntlet (see `.github/workflows/ci.yml`):

`cargo fmt --check` · `clippy -D warnings` · multi-OS build matrix
(Ubuntu / Windows / macOS) · full test suite incl. doctests · `cargo doc
-D warnings` · MSRV check · criterion benches (`--test` smoke) ·
tarpaulin coverage · and a **showcase job** that trains, dreams and
generates with the release binary, uploading the transcripts.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib --bins --tests --examples
cargo test --workspace --doc
cargo bench
```

## Roadmap

- [ ] Scripted pre-training corpora + ready-made checkpoints
- [ ] SIMD (std::simd) fast paths for matmul/softmax
- [ ] Causal masking mode for pure autoregressive pre-training
- [ ] `no_std`-friendly core behind a feature flag
- [ ] GPU offload exploration (wgpu compute kernels)

## License

MIT — see [LICENSE-MIT](LICENSE-MIT).
