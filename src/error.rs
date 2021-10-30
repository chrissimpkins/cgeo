//! Error types.

/// Errors that occur while working with [`crate::types::vector::Vector2DInt`]
/// and [`crate::types::vector::Vector2DFloat`] types
#[derive(Debug)]
pub enum VectorError {
    /// ValueError occurs when an invalid value is used in an operation
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
