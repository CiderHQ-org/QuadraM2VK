//! The runtime's source digest: the shader cache's build directory is keyed on it beside the
//! translator's own (`metal2vulkan::cache::SOURCE_DIGEST`), so a build of other runtime sources
//! never reads another build's entries, even when no version moved. It hashes the manifest, this
//! script, and every file under `src/`, each by its path, length and bytes, in sorted order; Cargo
//! reruns the script when one of them changes.
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let mut files = vec![PathBuf::from("Cargo.toml"), PathBuf::from("build.rs")];
    collect(Path::new("src"), &mut files);
    files.sort();
    let mut hash = Sha256::new();
    for file in &files {
        println!("cargo:rerun-if-changed={}", file.display());
        let bytes =
            fs::read(file).unwrap_or_else(|error| panic!("read {}: {error}", file.display()));
        let name = file.to_string_lossy().replace('\\', "/");
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(&bytes);
    }
    println!("cargo:rerun-if-changed=src");
    println!(
        "cargo:rustc-env=M2V_RUNTIME_SOURCE_ID={}",
        hex(&hash.finalize())
    );
}

fn collect(directory: &Path, output: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read {}: {error}", directory.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            collect(&path, output);
        } else if path.is_file() {
            output.push(path);
        }
    }
}

/// `bytes` as lowercase hex, two digits a byte: the form of every digest string here.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
