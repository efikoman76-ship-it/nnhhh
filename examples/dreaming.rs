//! Creative dreaming demo: evolve a mind briefly, then dream.
//!
//! Run with: `cargo run --example dreaming`

use aether::prelude::*;

fn main() {
    let mut mind = AetherMind::new(AetherConfig::tiny()).expect("mind");
    let batch = vec![vec![1, 2, 3, 4, 5, 6], vec![6, 5, 4, 3, 2, 1]];
    let dim = mind.param_count();
    let mut trainer = EvoTrainer::new(EvoTrainerConfig::demo(), dim).expect("trainer");
    let mut rng = seeded_rng(5);
    let stats = trainer.train(&mut mind, &batch, &mut rng).expect("train");
    println!(
        "trained: loss={:.4} best={:.4}",
        stats.loss, stats.best_loss
    );

    let cfg = CreativityConfig {
        n_candidates: 4,
        candidate_len: 10,
        max_iters: 3,
        ..CreativityConfig::default()
    };
    let mut engine = CreativityEngine::new(cfg).expect("engine");
    let report = engine
        .dream(&mut mind, &[1, 2, 3], &mut rng)
        .expect("dream");
    println!(
        "dream best: {:?} (score {:.4})",
        report.best_ids, report.best_score
    );
    println!("history: {:?}", report.history);
}
