//! End-to-end: tokenizer -> mind -> trainer -> persistence.

use aether::prelude::*;

fn tiny_batch() -> Vec<Vec<usize>> {
    vec![
        vec![1, 2, 3, 4, 5, 6],
        vec![6, 5, 4, 3, 2, 1],
        vec![1, 1, 2, 2, 3, 3],
    ]
}

#[test]
fn full_loop_learns_and_persists() {
    // 1. Tokenizer learns a vocabulary.
    let mut tok = BpeTokenizer::new();
    tok.train(&["aa bb cc aa bb", "bb cc dd ee", "aa ee ff gg"], 10).unwrap();
    assert!(tok.vocab_size() > 4);

    // 2. Mind runs a forward pass and generates.
    let mut mind = AetherMind::new(AetherConfig::tiny()).unwrap();
    let mut rng = seeded_rng(100);
    let prompt = vec![1usize, 2, 3];
    let fo = mind.forward(&prompt, &mut rng).unwrap();
    assert_eq!(fo.logits.nrows(), 3);
    let cont = mind
        .generate(&prompt, 5, &SampleConfig::default(), &mut rng)
        .unwrap();
    assert_eq!(cont.len(), 5);

    // 3. Trainer improves (best-so-far) on a toy batch.
    let dim = mind.param_count();
    let mut trainer = EvoTrainer::new(EvoTrainerConfig::demo(), dim).unwrap();
    let batch = tiny_batch();
    let before = trainer.eval_loss(&mut mind, &batch, &mut rng).unwrap().0;
    let stats = trainer.train(&mut mind, &batch, &mut rng).unwrap();
    assert!(stats.best_loss <= before + 1e-4);

    // 4. The organism persists and wakes up identical.
    let path = std::env::temp_dir().join("aether_e2e.json");
    mind.save_to(path.to_str().unwrap()).unwrap();
    let mut back = AetherMind::load_from(path.to_str().unwrap()).unwrap();
    let a = mind.forward(&prompt, &mut seeded_rng(7)).unwrap().logits.into_vec();
    let b = back.forward(&prompt, &mut seeded_rng(7)).unwrap().logits.into_vec();
    assert_eq!(a, b);
    let _ = std::fs::remove_file(path);
}

#[test]
fn memory_lives_across_forwards() {
    let mut mind = AetherMind::new(AetherConfig::tiny()).unwrap();
    let mut rng = seeded_rng(3);
    mind.forward(&[1, 2, 3, 4], &mut rng).unwrap();
    let s1 = mind.memory_stats();
    mind.forward(&[5, 6, 7, 8], &mut rng).unwrap();
    let s2 = mind.memory_stats();
    assert!(s2.clock > s1.clock);
    assert!(s2.episodic + s2.semantic > 0);
}
