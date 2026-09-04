//! Error type shared by every AETHER subsystem.
//!
//! All fallible operations return `Result<T, AetherError>` so callers get
//! structured, testable failures instead of panics.

use thiserror::Error;

/// Every error the AETHER engine can produce.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum AetherError {
    /// Tensor / vector dimensions did not line up.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    /// An operation received an empty sequence, corpus or vocabulary.
    #[error("empty input: {0}")]
    EmptyInput(String),
    /// A configuration value is out of range or internally inconsistent.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Tokenizer vocabulary problem (unknown merge, corrupt file, ...).
    #[error("vocabulary error: {0}")]
    Vocab(String),
    /// Filesystem failure while persisting or loading a model.
    #[error("io error: {0}")]
    Io(String),
    /// Serialization / deserialization failure.
    #[error("serialization error: {0}")]
    Ser(String),
}

/// Convenience alias used across the whole crate.
pub type Result<T> = std::result::Result<T, AetherError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_render() {
        let e = AetherError::ShapeMismatch("2x3 vs 4x5".to_string());
        assert_eq!(format!("{e}"), "shape mismatch: 2x3 vs 4x5");
        let e2 = AetherError::InvalidConfig("top_k > experts".to_string());
        assert!(format!("{e2}").contains("top_k"));
    }
}
