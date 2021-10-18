//! Error types.

/// Errors that occur while working with
#[derive(Debug)]
pub enum VectorError {
    ValueError(String),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            VectorError::ValueError(s) => {
                write!(f, "VectorError::ValueError: {}", s)
            }
        }
    }
}
