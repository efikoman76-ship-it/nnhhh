//! Trainable byte-pair tokenizer, dependency-free and Unicode-aware.
//!
//! `BpeTokenizer::train` learns merges from a corpus: words start as
//! characters plus an end-of-word marker, and the most frequent adjacent pair
//! merges each round (ties broken lexicographically, so training is fully
//! deterministic). Encoding replays the merges in learned order; decoding
//! concatenates pieces and restores word boundaries. Vocabularies persist as
//! JSON alongside the mind.

use crate::error::{AetherError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

/// End-of-word marker suffixed to every word.
pub const EOW: &str = "</w>";
/// Unknown token for characters never seen in training.
pub const UNK: &str = "<unk>";

/// Byte-pair tokenizer with a learned merge table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    vocab: Vec<String>,
    merges: Vec<(String, String)>,
    #[serde(skip)]
    token_to_id: HashMap<String, usize>,
    unk_id: usize,
}

impl BpeTokenizer {
    /// Empty tokenizer (only `<unk>` known).
    pub fn new() -> BpeTokenizer {
        let mut token_to_id = HashMap::new();
        token_to_id.insert(UNK.to_string(), 0);
        BpeTokenizer {
            vocab: vec![UNK.to_string()],
            merges: Vec::new(),
            token_to_id,
            unk_id: 0,
        }
    }

    /// Learn `num_merges` merges from `corpus`.
    pub fn train(&mut self, corpus: &[&str], num_merges: usize) -> Result<()> {
        if corpus.is_empty() {
            return Err(AetherError::EmptyInput("bpe train got empty corpus".to_string()));
        }
        // Split into words of character-pieces + EOW.
        let mut words: Vec<Vec<String>> = Vec::new();
        for line in corpus {
            for word in line.split_whitespace() {
                let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                if pieces.is_empty() {
                    continue;
                }
                pieces.push(EOW.to_string());
                words.push(pieces);
            }
        }
        if words.is_empty() {
            return Err(AetherError::EmptyInput("bpe train found no words".to_string()));
        }
        // Base alphabet: every char + EOW + UNK.
        let mut base: Vec<String> = words
            .iter()
            .flatten()
            .filter(|p| p.as_str() != EOW)
            .cloned()
            .collect();
        base.sort();
        base.dedup();
        let mut vocab: Vec<String> = vec![UNK.to_string()];
        vocab.push(EOW.to_string());
        vocab.extend(base);
        let mut merges: Vec<(String, String)> = Vec::new();

        for _ in 0..num_merges {
            let mut counts: HashMap<(String, String), usize> = HashMap::new();
            for word in &words {
                for pair in word.windows(2) {
                    *counts.entry((pair[0].clone(), pair[1].clone())).or_insert(0) += 1;
                }
            }
            if counts.is_empty() {
                break;
            }
            // Deterministic pick: highest count, ties broken lexicographically.
            let mut ranked: Vec<((String, String), usize)> = counts.into_iter().collect();
            ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let (best, _) = ranked.into_iter().next().expect("counts non-empty");
            let merged = format!("{}{}", best.0, best.1);
            // Rewrite every occurrence.
            for word in words.iter_mut() {
                let mut out: Vec<String> = Vec::with_capacity(word.len());
                let mut i = 0;
                while i < word.len() {
                    if i + 1 < word.len() && word[i] == best.0 && word[i + 1] == best.1 {
                        out.push(merged.clone());
                        i += 2;
                    } else {
                        out.push(word[i].clone());
                        i += 1;
                    }
                }
                *word = out;
            }
            vocab.push(merged);
            merges.push(best);
        }

        self.rebuild(vocab, merges);
        Ok(())
    }

    fn rebuild(&mut self, vocab: Vec<String>, merges: Vec<(String, String)>) {
        self.token_to_id = vocab.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        self.vocab = vocab;
        self.merges = merges;
        self.unk_id = 0;
    }

    /// Encode text into token ids (unknown chars become `<unk>`).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            if pieces.is_empty() {
                continue;
            }
            pieces.push(EOW.to_string());
            for (a, b) in &self.merges {
                let mut out: Vec<String> = Vec::with_capacity(pieces.len());
                let mut i = 0;
                while i < pieces.len() {
                    if i + 1 < pieces.len() && &pieces[i] == a && &pieces[i + 1] == b {
                        out.push(format!("{a}{b}"));
                        i += 2;
                    } else {
                        out.push(pieces[i].clone());
                        i += 1;
                    }
                }
                pieces = out;
            }
            for p in &pieces {
                ids.push(self.token_to_id.get(p).cloned().unwrap_or(self.unk_id));
            }
        }
        ids
    }

    /// Decode ids back to text.
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let mut raw = String::new();
        for &id in ids {
            let piece = self.vocab.get(id).ok_or_else(|| {
                AetherError::Vocab(format!("token id {id} out of vocab {}", self.vocab.len()))
            })?;
            if piece != UNK {
                raw.push_str(piece);
            }
        }
        Ok(raw.split(EOW).collect::<Vec<_>>().join(" ").trim().to_string())
    }

    /// Vocabulary size (includes `<unk>`).
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Number of learned merges.
    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// Id of the unknown token.
    pub fn unk_id(&self) -> usize {
        self.unk_id
    }

    /// Persist the vocabulary.
    pub fn save_to(&self, path: &str) -> Result<()> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| AetherError::Ser(e.to_string()))?;
        fs::write(path, json).map_err(|e| AetherError::Io(e.to_string()))?;
        Ok(())
    }

    /// Load a vocabulary saved with [`BpeTokenizer::save_to`].
    pub fn load_from(path: &str) -> Result<BpeTokenizer> {
        let text = fs::read_to_string(path).map_err(|e| AetherError::Io(e.to_string()))?;
        let mut tok: BpeTokenizer =
            serde_json::from_str(&text).map_err(|e| AetherError::Ser(e.to_string()))?;
        let vocab = tok.vocab.clone();
        let merges = tok.merges.clone();
        tok.rebuild(vocab, merges);
        Ok(tok)
    }
}

impl Default for BpeTokenizer {
    fn default() -> BpeTokenizer {
        BpeTokenizer::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<&'static str> {
        vec![
            "hello world hello",
            "hello there world",
            "world of hello worlds",
        ]
    }

    #[test]
    fn learns_merges_and_roundtrips() {
        let mut tok = BpeTokenizer::new();
        tok.train(&corpus(), 20).unwrap();
        assert!(tok.num_merges() > 0);
        for line in corpus() {
            let ids = tok.encode(line);
            assert!(!ids.is_empty());
            assert_eq!(tok.decode(&ids).unwrap(), line);
        }
    }

    #[test]
    fn training_is_deterministic() {
        let mut a = BpeTokenizer::new();
        let mut b = BpeTokenizer::new();
        a.train(&corpus(), 15).unwrap();
        b.train(&corpus(), 15).unwrap();
        let probe = "hello worlds";
        assert_eq!(a.encode(probe), b.encode(probe));
    }

    #[test]
    fn unknown_chars_map_to_unk() {
        let mut tok = BpeTokenizer::new();
        tok.train(&corpus(), 10).unwrap();
        let ids = tok.encode("hello zzz");
        assert!(ids.contains(&tok.unk_id()));
    }

    #[test]
    fn save_load_roundtrip() {
        let mut tok = BpeTokenizer::new();
        tok.train(&corpus(), 12).unwrap();
        let path = std::env::temp_dir().join("aether_bpe.json");
        tok.save_to(path.to_str().unwrap()).unwrap();
        let back = BpeTokenizer::load_from(path.to_str().unwrap()).unwrap();
        assert_eq!(back.encode("hello world"), tok.encode("hello world"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_empty_corpus_and_bad_ids() {
        let mut tok = BpeTokenizer::new();
        assert!(tok.train(&[], 5).is_err());
        assert!(tok.decode(&[9999]).is_err());
    }
}
