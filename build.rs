use std::{path::Path, process::Command};
use walkdir::WalkDir;

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        println!("cargo:rustc-cfg=docs_build");
        return;
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/model.onnx");
    println!("cargo:rerun-if-changed=src/vectorizer_params.json");

    if Path::new("src/model.onnx").exists() && Path::new("src/vectorizer_params.json").exists() {
        return;
    }

    let venv_dir = "build_venv";
    if let Err(e) = std::fs::remove_dir_all(venv_dir) {
        eprintln!("Failed to delete build_venv - {e}");
    };

    // Create the virtual environment if it doesn't exist
    println!("Creating virtual environment...");
    let status = Command::new("python3")
        .args(["-m", "venv", venv_dir])
        .status()
        .expect("Failed to create virtual environment");

    if !status.success() {
        panic!("Failed to create virtual environment");
    }

    let python_path = if cfg!(windows) {
        format!("{venv_dir}/Scripts/python")
    } else {
        format!("{venv_dir}/bin/python")
    };

    // Ensure pip is available in the venv
    println!("Ensuring pip is available...");
    Command::new(&python_path)
        .args(["-m", "ensurepip"])
        .status()
        .expect("Failed to ensure pip");

    // Upgrade pip
    Command::new(&python_path)
        .args(["-m", "pip", "install", "--upgrade", "pip"])
        .status()
        .expect("Failed to upgrade pip");

    // Install required dependencies
    println!("Installing dependencies...");
    let dependencies = [
        "scikit-learn",
        "joblib",
        "skl2onnx",
        "onnx",
        "alt-profanity-check",
    ];
    for dep in dependencies {
        let status = Command::new(&python_path)
            .args(["-m", "pip", "install", dep])
            .status()
            .unwrap_or_else(|e| panic!("Failed to install {}: {}", dep, e));

        if !status.success() {
            panic!("Failed to install dependency: {}", dep);
        }
    }

    // Find models in venv.
    let vectoriser_dir = WalkDir::new(venv_dir)
        .into_iter()
        .find(|e| {
            if let Ok(e) = e {
                e.file_type().is_file() && e.file_name() == "vectorizer.joblib"
            } else {
                false
            }
        })
        .unwrap()
        .unwrap();
    let vectoriser = vectoriser_dir.path().to_string_lossy();
    let model_dir = WalkDir::new(venv_dir)
        .into_iter()
        .find(|e| {
            if let Ok(e) = e {
                e.file_type().is_file() && e.file_name() == "model.joblib"
            } else {
                false
            }
        })
        .unwrap()
        .unwrap();
    let model = model_dir.path().to_string_lossy();

    let python_script = format!(
        "
import sys
import joblib
import json
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the vectorizer and model
vectorizer = joblib.load('{vectoriser}')
model = joblib.load('{model}')
"
    ) + r#"
# Export vectorizer parameters for Rust implementation
vectorizer_params = {
    'vocabulary': {word: int(idx) for word, idx in vectorizer.vocabulary_.items()},
    'idf': vectorizer.idf_.tolist(),
    'stop_words': list(vectorizer.stop_words) if vectorizer.stop_words else []
}
with open('src/vectorizer_params.json', 'w') as f:
    json.dump(vectorizer_params, f)

# Define the input type (float features after vectorization)
initial_type = [('float_input', FloatTensorType([None, len(vectorizer.vocabulary_)]))]

# Convert model to ONNX (without vectorizer, we'll do that in Rust)
# Use options to force probability output as a tensor, not sequence<map>
from skl2onnx.common.data_types import FloatTensorType
options = {type(model): {'zipmap': False}}
onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12, options=options)

# Save the ONNX model
with open('src/model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print('Model and vectorizer parameters exported successfully')
"#;

    // Run Python script to convert model using venv python
    let output = Command::new(&python_path)
        .arg("-c")
        .arg(python_script)
        .output()
        .expect("Failed to execute Python script for model conversion");

    if !output.status.success() {
        eprintln!("Python script failed:");
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Model conversion to ONNX failed");
    }

    // Delete venv
    std::fs::remove_dir_all(venv_dir).unwrap();
}
