use metal2vulkan_validation::case::AuthoredCase;
use metal2vulkan_validation::check::safety_census::{safety_census, summarize};
use metal2vulkan_validation::check::{check_case, recheck_installed_cases};
use metal2vulkan_validation::source::{corpus_root, public_source_set};
use metal2vulkan_validation::store::CorpusStore;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("corpus-case-check: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = corpus_root();
    let mut manifest = None;
    let mut case_id = None;
    let mut install = false;
    let mut delete_air = None;
    let mut delete_name = None;
    let mut delete_entry = None::<String>;
    let mut recheck_all = false;
    let mut safety_census_requested = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--private-corpus" => metal2vulkan_validation::source::use_private_corpus_root(
                PathBuf::from(required(&mut args, "--private-corpus")?),
            )?,
            "--manifest" => manifest = Some(PathBuf::from(required(&mut args, "--manifest")?)),
            "--case-id" => case_id = Some(required(&mut args, "--case-id")?),
            "--install" => install = true,
            "--delete-air" => delete_air = Some(required(&mut args, "--delete-air")?),
            "--delete-name" => delete_name = Some(required(&mut args, "--delete-name")?),
            "--delete-entry" => delete_entry = Some(required(&mut args, "--delete-entry")?),
            "--recheck-all" => recheck_all = true,
            "--safety-census" => safety_census_requested = true,
            "-h" | "--help" => {
                println!(
                    "usage: corpus-case-check [--corpus DIR] [--private-corpus DIR] (--manifest PATH [--install] | --case-id HASH | --delete-air HASH --delete-name NAME [--delete-entry ENTRY] | --recheck-all | --safety-census)"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}")),
        }
    }
    let store = CorpusStore::new(&root);
    if safety_census_requested {
        if recheck_all || manifest.is_some() || case_id.is_some() || install || delete_air.is_some()
        {
            return Err("--safety-census takes no other selection".into());
        }
        return safety_census_report(&root, &store);
    }
    if recheck_all {
        if manifest.is_some() || case_id.is_some() || install || delete_air.is_some() {
            return Err("--recheck-all takes no other selection".into());
        }
        return recheck_all_cases(&root, &store);
    }
    match (delete_air, delete_name) {
        (Some(air), Some(name)) if manifest.is_none() && case_id.is_none() && !install => {
            let removed = store.delete_named_case(&air, delete_entry.as_deref(), &name)?;
            println!("deleted {removed}");
            return Ok(());
        }
        (None, None) if delete_entry.is_none() => {}
        _ => {
            return Err(
                "--delete-air and --delete-name (with an optional --delete-entry) must be used together and \
                 without case checking"
                    .into(),
            )
        }
    }
    let mut case = match (manifest, case_id) {
        (Some(path), None) => {
            let bytes = fs::read(&path)
                .map_err(|error| format!("read manifest {}: {error}", path.display()))?;
            serde_json::from_slice::<AuthoredCase>(&bytes)
                .map_err(|error| format!("parse manifest {}: {error}", path.display()))?
        }
        (None, Some(case_id)) => store
            .find_case(&case_id)?
            .ok_or_else(|| format!("case_id {case_id} not found"))?,
        _ => return Err("select exactly one of --manifest or --case-id".into()),
    };
    if case.case_id.is_empty() {
        case.case_id = case.computed_case_id()?;
    }
    let checked = check_case(&root, case).map_err(|errors| errors.join("\n  - "))?;
    println!(
        "checked {} input_sha256={}",
        checked.primary.case.case_id, checked.primary.input_sha256
    );
    for unresolved in checked.unresolved() {
        println!("partial: {unresolved}");
    }
    if install {
        let mut case = checked.primary.case;
        // A case over a public fixture names it, so an edit of the fixture that does not re-key
        // the case is caught by the fixture's path. Outside the identity.
        if case.public_source.is_none() {
            case.public_source = public_source_set()?.path_of(&case.air_sha256);
        }
        println!("installed {:?}", store.put_case(case)?);
    }
    Ok(())
}

/// Report every installed case the shared checker no longer accepts.
///
/// Needs the private AIR sources, so it is a developer command rather than a CI gate. It fails when
/// any case is stale, so the answer cannot be skimmed past. See
/// [`recheck_installed_cases`] for why the store needs re-reading at all.
fn recheck_all_cases(root: &Path, store: &CorpusStore) -> Result<(), String> {
    let recheck = recheck_installed_cases(root, store)?;
    for case in &recheck.stale {
        println!(
            "stale {} {}: {}",
            case.case_id,
            case.name,
            case.errors.join("; ")
        );
    }
    // Accepted, but with checks the AIR layer cannot complete: named, never silent.
    for case in &recheck.partial {
        println!(
            "partial {} {}: {}",
            case.case_id,
            case.name,
            case.unresolved.join("; ")
        );
    }
    let total = recheck.total;
    println!(
        "rechecked {total} installed cases, {} stale, {} partial",
        recheck.stale.len(),
        recheck.partial.len()
    );
    if recheck.stale.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} of {total} installed cases no longer check against their AIR",
        recheck.stale.len()
    ))
}

/// Classify every installed case as mechanically loop-free or cyclic and check it against the
/// case-schema caps (CPU only; nothing executes).
///
/// Names each cyclic case the oracle's fuel transform cannot bound, with the cycle it cannot reach.
/// Fails when an installed `loop_free` declaration is contradicted, a case cannot be classified, or
/// a case exceeds a cap, so none of them can be skimmed past.
fn safety_census_report(root: &Path, store: &CorpusStore) -> Result<(), String> {
    let rows = safety_census(root, store)?;
    for row in &rows {
        match &row.class {
            Err(error) => println!("unclassified {} {}: {error}", row.case_id, row.name),
            Ok(class) if row.contradicts_declaration() => println!(
                "mismatch {} {}: declared loop_free, {}",
                row.case_id,
                row.name,
                class.evidence().expect("contradiction is cyclic")
            ),
            Ok(_) => {}
        }
        if let Some(gap) = row.fuel_gap() {
            println!("fuel uncovered {} {}: {gap}", row.case_id, row.name);
        }
        for error in &row.cap_errors {
            println!("over cap {} {}: {error}", row.case_id, row.name);
        }
    }
    let summary = summarize(&rows);
    println!(
        "safety census: {} cases, {} loop-free, {} cyclic, {} unclassified, {} over a cap",
        summary.total, summary.loop_free, summary.cyclic, summary.unclassified, summary.over_caps
    );
    for (kind, count) in &summary.cyclic_by_evidence {
        println!("  cyclic by first evidence: {kind} {count}");
    }
    println!(
        "  declared loop_free but cyclic: {}; declared authored_bounded but loop-free: {}",
        summary.declared_loop_free_but_cyclic, summary.declared_bounded_but_loop_free
    );
    println!(
        "  cyclic and bounded by oracle fuel: {}",
        summary.cyclic_fuel_covered
    );
    for (kind, count) in &summary.fuel_uncovered_by_evidence {
        println!("  cyclic, not bounded by oracle fuel: {kind} {count}");
    }
    if summary.declared_loop_free_but_cyclic != 0
        || summary.unclassified != 0
        || summary.over_caps != 0
    {
        return Err(format!(
            "{} contradicted loop_free declaration(s), {} unclassified case(s), {} case(s) over a cap",
            summary.declared_loop_free_but_cyclic, summary.unclassified, summary.over_caps
        ));
    }
    Ok(())
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
