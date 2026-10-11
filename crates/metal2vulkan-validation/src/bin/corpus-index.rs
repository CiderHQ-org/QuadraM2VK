use metal2vulkan_validation::index::{
    check_index, default_index_path, rebuild_index, status_counts, sync_index_with_stats,
    IndexScope,
};
use metal2vulkan_validation::source::{corpus_root, private_data_dir, public_source_set};
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-index: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut destination = None;
    let mut check = false;
    let mut rebuild = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--private-corpus" => metal2vulkan_validation::source::use_private_corpus_root(
                PathBuf::from(required(&mut args, "--private-corpus")?),
            )?,
            "--index" => destination = Some(PathBuf::from(required(&mut args, "--index")?)),
            "--check" => check = true,
            "--rebuild" => rebuild = true,
            "-h" | "--help" => {
                println!("usage: corpus-index [--corpus DIR] [--private-corpus DIR] [--index PATH] [--check | --rebuild]");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let destination = destination.unwrap_or_else(|| default_index_path(&root));
    if check && rebuild {
        return Err("--check and --rebuild are mutually exclusive".into());
    }
    if check {
        check_index(&root, &destination)?;
        println!("checked {}", destination.display());
    } else if rebuild {
        let scope = rebuild_index(&root, &destination)?;
        println!("rebuilt {}", destination.display());
        println!("{}", scope_line(&root, scope));
    } else {
        let stats = sync_index_with_stats(&root, &destination)?;
        println!(
            "synchronized {} (source_shards_scanned={} source_bytes_scanned={} rebuilt={} \
             public_rows_retired={} public_rows_added={})",
            destination.display(),
            stats.source_shards_scanned,
            stats.source_bytes_scanned,
            stats.rebuilt,
            stats.public_rows_retired,
            stats.public_rows_added
        );
        println!("{}", scope_line(&root, stats.scope));
    }
    // Public modules with several entries are not rows; say which and how many.
    for line in public_source_set()?.report_lines() {
        println!("{line}");
    }
    for (state, count) in status_counts(&destination)? {
        println!("{state}\t{count}");
    }
    Ok(())
}

/// The index's scope, one `key=value` line: `scope=complete`, or for a checkout with no private
/// store, `scope=public-only` and the committed rows of private AIR it left out.
fn scope_line(root: &Path, scope: IndexScope) -> String {
    match scope {
        IndexScope::Complete => "scope=complete".into(),
        IndexScope::PublicOnly(unavailable) => format!(
            "scope=public-only (no private store at {}; rows of private AIR are not indexed) \
             private_cases={} private_metal_observations={} private_candidate_observations={}",
            private_data_dir(root).display(),
            unavailable.cases,
            unavailable.metal_observations,
            unavailable.candidate_observations
        ),
    }
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
