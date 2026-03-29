#![feature(once_cell_try)]

mod errors;

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

use ndarray::Array2;
use ort::{session::Session, value::Value};
use regex::Regex;
use serde::Deserialize;

use crate::errors::{ProfanityError, ProfanityError::*, ProfanityResult};

#[derive(Deserialize)]
struct VectorizerParams {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f32>,
    stop_words: Vec<String>,
}

static VECTORIZER_PARAMS: OnceLock<VectorizerParams> = OnceLock::new();
static MODEL: OnceLock<Mutex<Session>> = OnceLock::new();

fn get_vectorizer_params() -> &'static VectorizerParams {
    VECTORIZER_PARAMS.get_or_init(|| {
        #[cfg(feature = "docs_build")]
        let json_str = "file wont exist for docs build.";
        #[cfg(not(feature = "docs_build"))]
        let json_str = include_str!("vectorizer_params.json");
        serde_json::from_str(json_str).expect("Failed to parse vectorizer params")
    })
}

fn tokenize(text: &str) -> Vec<String> {
    // Convert to lowercase and split on non-alphanumeric characters
    let Ok(re) = Regex::new(r"\b\w+\b") else {
        unreachable!("Static regex, can't ever fail to compile.")
    };
    re.find_iter(&text.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect()
}

fn vectorize_text(text: &str) -> Vec<f32> {
    let params = get_vectorizer_params();
    let tokens = tokenize(text);

    // Count term frequencies
    let mut term_freq: HashMap<String, f32> = HashMap::new();
    let mut total_terms: f32 = 0.0;

    for token in tokens {
        if !params.stop_words.contains(&token) {
            *term_freq.entry(token).or_insert(0.0) += 1.0;
            total_terms += 1.0;
        }
    }

    // Normalize term frequencies
    for val in term_freq.values_mut() {
        *val /= total_terms.max(1.0);
    }

    // Create TF-IDF vector
    let mut tfidf_vector = vec![0.0f32; params.vocabulary.len()];

    for (term, tf) in term_freq {
        if let Some(&idx) = params.vocabulary.get(&term)
            && idx < params.idf.len()
        {
            tfidf_vector[idx] = tf * params.idf[idx];
        }
    }

    // Normalize the vector (L2 normalization)
    let norm: f32 = tfidf_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut tfidf_vector {
            *val /= norm;
        }
    }

    tfidf_vector
}

/// Predicts the probability that the given text is profane.
///
/// # Arguments
///
/// * `text` - The input text to analyze
///
/// # Returns
///
/// A probability value between 0.0 and 1.0, where higher values indicate
/// a higher likelihood of profanity.
pub fn predict_prob(text: &str) -> ProfanityResult<f32> {
    // Vectorize the text using TF-IDF
    let features = vectorize_text(text);

    // Load the model
    let model = MODEL.get_or_try_init(|| {
        #[cfg(feature = "docs_build")]
        let model_bytes = vec![];
        #[cfg(not(feature = "docs_build"))]
        let model_bytes = include_bytes!("model.onnx").to_vec();

        Ok::<Mutex<Session>, ProfanityError>(Mutex::new(
            Session::builder()
                .map_err(SessionError)?
                .commit_from_memory(&model_bytes)
                .map_err(CommitError)?,
        ))
    })?;
    // Create tensor input
    let input_array = Array2::from_shape_vec((1, features.len()), features)?;
    let input_tensor = Value::from_array(input_array).map_err(FromArrayError)?;

    // Run inference
    let mut model_lock = model.lock()?;
    let outputs = model_lock
        .run(ort::inputs!["float_input" => input_tensor])
        .map_err(RunError)?;

    // Try to get probabilities from a named output
    let prob_output = outputs
        .get("probabilities")
        .or_else(|| outputs.get("output_probability"))
        .or_else(|| {
            if outputs.len() > 1 {
                Some(&outputs[1])
            } else {
                None
            }
        })
        .ok_or(CouldntFindResultError)?;

    let (_, probabilities) = prob_output
        .try_extract_tensor::<f32>()
        .map_err(ExtractError)?;

    // Return the probability of the positive class (profane) - index 1
    // The output is a 2D array with shape (1, 2) for [not_profane, profane]
    if probabilities.len() < 2 {
        return Err(UnexpectedFormat);
    }

    Ok(probabilities[1])
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use approx::assert_relative_eq;
    use pyo3::prelude::*;

    use super::*;

    fn python_wrap_predict_prob(text: &str) -> f32 {
        let probability = Python::attach(|py| -> f32 {
            // Import the Python module
            let profanity_module = PyModule::import(py, "profanity_check").unwrap();

            // Get the predict function
            let predict = profanity_module.getattr("predict_prob").unwrap();

            // Call predict([text])
            let result = predict.call1((vec![text],)).unwrap();

            // The result is a probability
            let first: f32 = result.get_item(0).unwrap().extract().unwrap();

            first
        });
        probability
    }

    #[test]
    fn test_compare_python() {
        Command::new("python3")
            .args(["-m", "venv", ".venv"])
            .spawn()
            .unwrap()
            .wait()
            .unwrap();
        Command::new(".venv/bin/python3")
            .args(["-m", "pip", "install", "alt-profanity-check"])
            .spawn()
            .unwrap()
            .wait()
            .unwrap();
        unsafe {
            std::env::set_var("PYTHONPATH", "./.venv/lib/python3.11/site-packages");
        }
        Python::initialize();
        assert_relative_eq!(
            predict_prob("This is a test").unwrap(),
            python_wrap_predict_prob("This is a test"),
            epsilon = 1.0e-6
        );
        assert_relative_eq!(
            predict_prob("go to hell, you scum").unwrap(),
            python_wrap_predict_prob("go to hell, you scum"),
            epsilon = 1.0e-6
        );
    }
}
