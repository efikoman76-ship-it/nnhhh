//! AETHER command line: demo, train, dream, tokenize and inspect the mind.
//!
//! Examples:
//!
//! ```sh
//! aether info --preset tiny
//! aether demo
//! aether train --gens 3 --pop 4
//! aether dream --iters 3 --prompt 1,2,3
//! aether tokenize --text "hello brave world" --merges 30
//! ```

use aether::prelude::*;
use rand::rngs::StdRng;

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "info" => cmd_info(&args),
        "demo" => cmd_demo(),
        "train" => cmd_train(&args),
        "dream" => cmd_dream(&args),
        "tokenize" => cmd_tokenize(&args),
        "help" | "--help" | "-h" => {
            print_help();
            0
        }
        other => {
            eprintln!("unknown command '{other}'; try `aether help`");
            2
        }
    }
}

/// Read `--key value` or `--key=value`.
fn flag(args: &[String], key: &str) -> Option<String> {
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if arg == &format!("--{key}") {
            return iter.next().cloned();
        }
        if let Some(rest) = arg.strip_prefix(&format!("--{key}=")) {
            return Some(rest.to_string());
        }
    }
    None
}

fn preset(args: &[String]) -> AetherConfig {
    match flag(args, "preset").as_deref() {
        Some("small") => AetherConfig::small(),
        _ => AetherConfig::tiny(),
    }
}

fn print_help() {
    println!("AETHER — Adaptive Entangled Thought with Holographic Emergent Resonance");
    println!();
    println!("USAGE: aether <command> [flags]");
    println!();
    println!("COMMANDS:");
    println!("  info      [--preset tiny|small]   show config + parameter count");
    println!("  demo                             forward pass + greedy generation");
    println!("  train     [--gens N] [--pop N]   evolutionary training on toy patterns");
    println!("  dream     [--iters N] [--prompt 1,2,3]  divergent creative dreaming");
    println!("  tokenize  [--text S] [--merges N]       train BPE + roundtrip demo");
    println!("  help                             this message");
}

fn cmd_info(args: &[String]) -> i32 {
    let cfg = preset(args);
    match AetherMind::new(cfg) {
        Ok(mind) => {
            let c = mind.config();
            println!("aether mind [{} params]", mind.param_count());
            println!("  d_model={} heads={} layers={} experts={}x(top{}) hidden={}",
                c.d_model, c.n_heads, c.n_layers, c.n_experts, c.top_k, c.d_moe_hidden);
            println!("  vocab={} max_seq={} window={}", c.vocab_size, c.max_seq, c.window);
            println!("  memory sensory/working/episodic/semantic = {}/{}/{}/{}",
                c.memory_sensory, c.memory_working, c.memory_episodic, c.memory_semantic);
            0
        }
        Err(e) => {
            eprintln!("info failed: {e}");
            1
        }
    }
}

fn cmd_demo() -> i32 {
    let mut mind = match AetherMind::new(AetherConfig::tiny()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("demo failed: {e}");
            return 1;
        }
    };
    let mut rng: StdRng = seeded_rng(42);
    let prompt = vec![1usize, 2, 3, 4];
    let out = match mind.forward(&prompt, &mut rng) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("demo failed: {e}");
            return 1;
        }
    };
    let mean = out.logits.mean_all();
    let max = out.logits.as_slice().iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("forward ok: logits {}x{} mean={mean:.4} max={max:.4} aux={:.4}",
        out.logits.nrows(), out.logits.ncols(), out.aux_loss);
    let stats = mind.memory_stats();
    println!("memory: sensory={} working={} episodic={} semantic={} clock={:.0}",
        stats.sensory, stats.working, stats.episodic, stats.semantic, stats.clock);
    let sample = SampleConfig {
        temperature: 0.0,
        ..SampleConfig::default()
    };
    match mind.generate(&prompt, 16, &sample, &mut rng) {
        Ok(ids) => {
            println!("greedy continuation: {ids:?}");
            0
        }
        Err(e) => {
            eprintln!("demo failed: {e}");
            1
        }
    }
}

fn toy_batch() -> Vec<Vec<usize>> {
    vec![
        vec![1, 2, 3, 4, 5, 6, 7, 8],
        vec![8, 7, 6, 5, 4, 3, 2, 1],
        vec![1, 3, 5, 7, 5, 3, 1, 3],
        vec![2, 4, 6, 8, 6, 4, 2, 4],
    ]
}

fn cmd_train(args: &[String]) -> i32 {
    let gens: usize = flag(args, "gens").and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut pop: usize = flag(args, "pop").and_then(|s| s.parse().ok()).unwrap_or(4);
    if pop < 2 {
        pop = 2;
    }
    if pop % 2 != 0 {
        pop += 1;
    }
    let mut mind = match AetherMind::new(AetherConfig::tiny()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("train failed: {e}");
            return 1;
        }
    };
    let mut tcfg = EvoTrainerConfig::demo();
    tcfg.generations = gens;
    tcfg.pop = pop;
    let mut trainer = match EvoTrainer::new(tcfg, mind.param_count()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("train failed: {e}");
            return 1;
        }
    };
    let batch = toy_batch();
    let mut rng: StdRng = seeded_rng(21);
    let initial = match trainer.eval_loss(&mut mind, &batch, &mut rng) {
        Ok((l, _)) => l,
        Err(e) => {
            eprintln!("train failed: {e}");
            return 1;
        }
    };
    match trainer.train(&mut mind, &batch, &mut rng) {
        Ok(stats) => {
            println!("trained {gens} gens (pop {pop}): loss {initial:.4} -> {:.4} (best {:.4}, aux {:.4})",
                stats.loss, stats.best_loss, stats.aux);
            0
        }
        Err(e) => {
            eprintln!("train failed: {e}");
            1
        }
    }
}

fn parse_ids(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .collect()
}

fn cmd_dream(args: &[String]) -> i32 {
    let iters: usize = flag(args, "iters").and_then(|s| s.parse().ok()).unwrap_or(3);
    let prompt = flag(args, "prompt").map(|s| parse_ids(&s)).unwrap_or(vec![1, 2, 3]);
    if prompt.is_empty() {
        eprintln!("dream needs a non-empty --prompt like 1,2,3");
        return 2;
    }
    let mut mind = match AetherMind::new(AetherConfig::tiny()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("dream failed: {e}");
            return 1;
        }
    };
    let mut ccfg = CreativityConfig::default();
    ccfg.max_iters = iters.max(1);
    let mut engine = match CreativityEngine::new(ccfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("dream failed: {e}");
            return 1;
        }
    };
    match engine.dream(&mut mind, &prompt, &mut seeded_rng(9)) {
        Ok(report) => {
            println!("dream from {prompt:?}: best {:?} score={:.4} ({} candidates)",
                report.best_ids, report.best_score, report.candidates_considered);
            println!("history: {:?}", report.history);
            0
        }
        Err(e) => {
            eprintln!("dream failed: {e}");
            1
        }
    }
}

fn cmd_tokenize(args: &[String]) -> i32 {
    let merges: usize = flag(args, "merges").and_then(|s| s.parse().ok()).unwrap_or(30);
    let text = flag(args, "text").unwrap_or_else(|| "hello brave world".to_string());
    let corpus = [
        "hello world hello",
        "hello there world",
        "world of hello worlds",
        "brave new world of wonders",
        "dreams are made of these",
    ];
    let refs: Vec<&str> = corpus.to_vec();
    let mut tok = BpeTokenizer::new();
    if let Err(e) = tok.train(&refs, merges) {
        eprintln!("tokenize failed: {e}");
        return 1;
    }
    let ids = tok.encode(&text);
    match tok.decode(&ids) {
        Ok(back) => {
            println!("vocab={} merges={}", tok.vocab_size(), tok.num_merges());
            println!("encode({text:?}) = {ids:?}");
            println!("decode = {back:?}");
            0
        }
        Err(e) => {
            eprintln!("tokenize failed: {e}");
            1
        }
    }
}
