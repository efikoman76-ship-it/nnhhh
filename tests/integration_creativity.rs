//! End-to-end: the creativity engine dreams with a trained-ish mind.

use aether::prelude::*;

#[test]
fn dream_end_to_end() {
    let mut mind = AetherMind::new(AetherConfig::tiny()).unwrap();
    // Give the mind a whiff of structure first.
    let batch = vec![vec![1, 2, 3, 4, 5, 6], vec![6, 5, 4, 3, 2, 1]];
    let dim = mind.param_count();
    let mut trainer = EvoTrainer::new(EvoTrainerConfig::demo(), dim).unwrap();
    trainer.train(&mut mind, &batch, &mut seeded_rng(5)).unwrap();

    let cfg = CreativityConfig {
        n_candidates: 4,
        candidate_len: 8,
        max_iters: 3,
        ..CreativityConfig::default()
    };
    let mut engine = CreativityEngine::new(cfg).unwrap();
    let report = engine.dream(&mut mind, &[2, 4, 6], &mut seeded_rng(8)).unwrap();

    assert_eq!(report.best_ids.len(), 8);
    assert!(report.best_score.is_finite());
    assert_eq!(report.history.len(), 3);
    assert_eq!(report.candidates_considered, 12);
    // Novelty archive actually filled up while dreaming.
    assert!(engine.archive_len() > 0);

    // Conceptual blending composes with dreaming: blend the dream's
    // fingerprint with the prompt's fingerprint.
    let a = mind.embed_mean(&report.best_ids).unwrap();
    let b = mind.embed_mean(&[2, 4, 6]).unwrap();
    let blended = conceptual_blend(&a, &b, 0.5).unwrap();
    let nov = engine.novelty_of(&blended).unwrap();
    assert!(nov.is_finite());
}
