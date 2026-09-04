//! Text-in, text-out demo: BPE + mind + greedy decoding.
//!
//! Run with: `cargo run --example textgen`

use aether::prelude::*;

fn main() {
    let corpus = [
        "hello world hello there",
        "world of wonders and dreams",
        "dreams are made of starlight",
        "hello starlight world",
    ];
    let refs: Vec<&str> = corpus.to_vec();
    let mut tok = BpeTokenizer::new();
    tok.train(&refs, 25).expect("bpe train");
    println!("vocab size: {}", tok.vocab_size());

    let text = "hello world";
    let ids: Vec<usize> = tok.encode(text).iter().map(|id| id % 64).collect();
    println!("prompt {text:?} -> {ids:?}");

    let mut mind = AetherMind::new(AetherConfig::tiny()).expect("mind");
    let sample = SampleConfig {
        temperature: 0.8,
        top_k: 8,
        ..SampleConfig::default()
    };
    let cont = mind
        .generate(&ids, 12, &sample, &mut seeded_rng(2))
        .expect("generate");
    println!("continuation ids: {cont:?}");
    println!(
        "decoded prompt roundtrip: {:?}",
        tok.decode(&tok.encode(text))
    );
}
