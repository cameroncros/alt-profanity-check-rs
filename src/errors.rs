use std::sync::{MutexGuard, PoisonError};

use ndarray::ShapeError;
use ort::{Error, session::Session};
use thiserror::Error;

pub type ProfanityResult<T> = Result<T, ProfanityError>;

#[derive(Error, Debug)]
pub enum ProfanityError {
    #[error("Failed to get lock on session")]
    PoisonError(#[from] PoisonError<MutexGuard<'static, Session>>),
    #[error("Failed to create session")]
    SessionError(Error),
    #[error("Failed from_shape_vec")]
    ShapeError(#[from] ShapeError),
    #[error("Failed commit_from_memory")]
    CommitError(Error),
    #[error("Failed from_array")]
    FromArrayError(Error),
    #[error("Failed run")]
    RunError(Error),
    #[error("Failed extract_from_tensor")]
    ExtractError(Error),
    #[error("Unexpected format for results")]
    UnexpectedFormat,
    #[error("Couldn't find results")]
    CouldntFindResultError,
    #[error("unknown error")]
    Unknown,
}
