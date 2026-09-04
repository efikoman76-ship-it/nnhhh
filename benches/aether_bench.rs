//! Micro-benchmarks for the hot paths.
//!
//! Run with: `cargo bench`

use aether::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_matmul(c: &mut Criterion) {
    let a = Matrix::randn_seeded(1, 128, 128);
    let b = Matrix::randn_seeded(2, 128, 128);
    c.bench_function("matmul 128x128", |bencher| {
        bencher.iter(|| black_box(a.matmul(&b).unwrap()))
    });
}

fn bench_attention(c: &mut Criterion) {
    let attn = EntangledAttention::new(
        EntangledAttentionConfig {
            d_model: 32,
            n_heads: 4,
            window: 9,
        },
        3,
    )
    .unwrap();
    let x = Matrix::randn_seeded(4, 16, 32);
    c.bench_function("entangled attention T=16 D=32", |bencher| {
        bencher.iter(|| black_box(attn.forward(&x).unwrap()))
    });
}

fn bench_hypervec(c: &mut Criterion) {
    let a = HyperVec::random_seeded(1, 4096).unwrap();
    let b = HyperVec::random_seeded(2, 4096).unwrap();
    c.bench_function("hypervec bundle+bind D=4096", |bencher| {
        bencher.iter(|| {
            let bundled = HyperVec::bundle(&a, &b).unwrap();
            black_box(HyperVec::bind(&bundled, &a).unwrap())
        })
    });
}

fn bench_moe(c: &mut Criterion) {
    let moe = SparseMoe::new(
        SparseMoeConfig {
            d_model: 32,
            d_hidden: 64,
            n_experts: 4,
            top_k: 2,
            noise_std: 0.0,
        },
        5,
    )
    .unwrap();
    let x = Matrix::randn_seeded(6, 16, 32);
    let mut rng = seeded_rng(7);
    c.bench_function("sparse moe T=16 D=32", |bencher| {
        bencher.iter(|| black_box(moe.forward(&x, &mut rng).unwrap()))
    });
}

fn bench_forward(c: &mut Criterion) {
    let mut mind = AetherMind::new(AetherConfig::tiny()).unwrap();
    let mut rng = seeded_rng(8);
    c.bench_function("mind forward T=8 tiny", |bencher| {
        bencher.iter(|| black_box(mind.forward(black_box(&[1, 2, 3, 4, 5, 6, 7, 8]), &mut rng).unwrap()))
    });
}

criterion_group!(
    benches,
    bench_matmul,
    bench_attention,
    bench_hypervec,
    bench_moe,
    bench_forward
);
criterion_main!(benches);
