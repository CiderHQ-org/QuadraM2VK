//! Locks the env-var registry invariant (cleanup Phase 1b): every `METAL2VULKAN_*` knob an accessor
//! reads is documented in `env_vars::REGISTRY` (so `metal2vulkan --help` lists it), and every knob
//! read is centralized in `env_vars.rs` (no scattered raw `std::env::var` reads elsewhere). This is
//! the regression proxy for the drift where ~19 live knobs had accessors but no registry entry, so
//! `--help` silently under-documented them and two knobs bypassed the registry with raw `env::var`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::env_vars::REGISTRY;

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Every string literal passed as the first argument to `helper` (which must end in an opening
/// quote), e.g. `present("` yields the knob name in `present("METAL2VULKAN_WHY")`.
fn literals_after(src: &str, helper: &str) -> Vec<String> {
    src.split(helper)
        .skip(1)
        .filter_map(|chunk| chunk.find('"').map(|end| chunk[..end].to_string()))
        .collect()
}

#[test]
fn every_accessor_knob_is_registered() {
    let registry: HashSet<&str> = REGISTRY.iter().map(|v| v.name).collect();
    let env_vars_src = std::fs::read_to_string(src_dir().join("env_vars.rs")).unwrap();

    // The three ways an accessor in env_vars.rs names a knob: the `present` flag helper and the
    // inline integer/path reads. Their bodies read a `name` variable (no literal), so only accessor
    // call sites — which pass a `"METAL2VULKAN_*"` literal — are captured here.
    let mut knobs = Vec::new();
    for helper in ["present(\"", "std::env::var(\"", "std::env::var_os(\""] {
        knobs.extend(literals_after(&env_vars_src, helper));
    }

    let missing: Vec<&String> = knobs
        .iter()
        .filter(|k| k.starts_with("METAL2VULKAN_") && !registry.contains(k.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "env_vars.rs accessors read knobs absent from REGISTRY (add a registry entry so \
         `--help` documents them): {missing:?}"
    );
}

#[test]
fn all_knob_reads_are_centralized_in_env_vars() {
    let mut files = Vec::new();
    rs_files(&src_dir(), &mut files);
    // The AIR front end's crate (`crates/metal2vulkan-air/`) reads no knob: its behaviour takes no
    // setting, so a read there is an offender too.
    rs_files(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-air")
            .join("src"),
        &mut files,
    );

    let mut offenders = Vec::new();
    for f in &files {
        if f.ends_with("env_vars.rs") {
            continue; // env_vars.rs is the one sanctioned home of raw reads
        }
        // The crate's test suite reads the knobs of its own harness (where its tools are,
        // `METAL2VULKAN_TEST_*`), not product settings that `--help` documents.
        if f.starts_with(src_dir().join("tests")) {
            continue;
        }
        let src = std::fs::read_to_string(f).unwrap();
        for line in src.lines() {
            if line.contains("env::var") && line.contains("\"METAL2VULKAN_") {
                offenders.push(format!("{}: {}", f.display(), line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "raw `METAL2VULKAN_*` env reads outside env_vars.rs (route through an env_vars accessor \
         instead):\n{}",
        offenders.join("\n")
    );
}

/// The dev-tools crate (the tests' fixture assemblers, built from its own directory) reads the
/// tool overrides this registry documents: it spells `METAL2VULKAN_<TOOL>` as
/// `env_vars::tool_path_override` does.
#[test]
fn the_dev_tools_read_the_registered_tool_overrides() {
    let tool = "m2v-registry-probe";
    // A variable no other test or tool reads, so setting it races with nothing.
    std::env::set_var("METAL2VULKAN_M2V_REGISTRY_PROBE", "/registry/probe");
    assert_eq!(
        crate::env_vars::tool_path_override(tool),
        Some(std::ffi::OsString::from("/registry/probe"))
    );
    assert_eq!(
        metal2vulkan_devtools::tools::tool_path_override(tool),
        crate::env_vars::tool_path_override(tool)
    );
}

#[test]
fn registry_names_are_unique_and_well_formed() {
    let mut seen = HashSet::new();
    for v in REGISTRY {
        assert!(
            v.name.starts_with("METAL2VULKAN_") && v.name.len() > "METAL2VULKAN_".len(),
            "malformed registry name: {:?}",
            v.name
        );
        assert!(seen.insert(v.name), "duplicate registry name: {:?}", v.name);
    }
}
