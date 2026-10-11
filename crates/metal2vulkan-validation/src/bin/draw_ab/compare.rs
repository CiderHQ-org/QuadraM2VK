//! `corpus-draw-ab compare`: the A/B verdict over two legs' `run` tables (the coordinator's
//! false-pass rule of 2026-10-08).
//!
//! `corpus-draw-ab compare --base DIR --change DIR --draws FILE --runs N [--expected-skips ID,...]
//! [--out FILE]` reads `DIR/<id>.tsv` of each leg for every draw of the draws file, and gives
//! each of the `N` runs of each draw one verdict:
//!
//! - **same**: both legs drew it (`OK`) and wrote the same bytes (digest and length);
//! - **different**: both drew it and wrote different bytes, or one leg drew it and the other's
//!   runtime refused it, or both refused it for different reasons (a setup refusal, the `-` row,
//!   counts for every run);
//! - **refused alike**: both legs' runtimes refused it for the same reason;
//! - **skipped**: either leg skipped the draw (`SKIP`, an input the harness does not seed);
//! - **missing**: either leg has no table for the draw, or no row for the run.
//!
//! Only the first two are compared. A draw skipped in both legs is not compared, as one
//! skipped in one leg is not: before this command the A/B scripts counted the numbered runs
//! alone, and five draws skipped in both legs of an A/B went unseen.
//!
//! A same run carries **no evidence** (the coordinator's byte-claim rule of 2026-10-10) where both
//! legs flagged every image it drew (the run table's images column: uniform, or unchanged by the
//! draw), or where a leg wrote one digest across every run of the draw, whatever their inputs: such
//! bytes agree in any two builds. The report counts those runs and names their draws; the exit
//! status does not change, since a draw may rightly write nothing (`--depth-test less`), but a
//! claim that bytes agree stands only on the others.
//!
//! It prints a line per draw, then the totals (`compared C of E`, where `E` is the draws times
//! `N`; `not compared`, split into skipped, refused alike and missing), then the verdict. With
//! `--out`, a line per run too: the draw, the run, the verdict and each leg's outcome. The exit
//! status (the harness's, `corpus-draw-ab.rs`'s module docs):
//!
//! - **0**: every run of every draw compared and the same, but those of the draws
//!   `--expected-skips` names, whose runs may go uncompared (a survey's known skips);
//! - **1**: a run differs;
//! - **5**: no run differs, but a run of a draw not on the list was not compared;
//! - **2**: the command line, a draws file or a table is wrong.

use super::{number, options, required, Failure};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

/// What a leg's table says of one run, or of the whole draw (its `-` row).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Outcome {
    /// Drawn: the digest and length of what it wrote.
    Drawn(String),
    /// The runtime refused it, for this reason.
    Refused(String),
    /// The harness skipped it, for this reason.
    Skipped(String),
}

impl Outcome {
    fn describe(&self) -> String {
        match self {
            Outcome::Drawn(bytes) => format!("OK {bytes}"),
            Outcome::Refused(why) => format!("REFUSED {why}"),
            Outcome::Skipped(why) => format!("SKIP {why}"),
        }
    }
}

/// One leg's table of one draw.
#[derive(Default)]
struct Leg {
    /// Each numbered run's outcome.
    runs: BTreeMap<u64, Outcome>,
    /// The `-` row of a draw that stopped before it ran.
    stop: Option<Outcome>,
    /// The drawn runs whose images column flags every image (`images_blank`).
    blank: BTreeSet<u64>,
}

/// Whether a run table's images column (`images=0:uniform+unchanged,1:drawn`) flags every image
/// it names, and names one.
fn images_blank(column: &str) -> bool {
    match column.strip_prefix("images=") {
        Some("none") | None => false,
        Some(images) => images.split(',').all(|image| {
            image
                .split_once(':')
                .is_some_and(|(_, flags)| flags != "drawn")
        }),
    }
}

impl Leg {
    /// The table at `dir/<id>.tsv`; `None` where there is none.
    fn read(dir: &Path, id: &str) -> Result<Option<Leg>, String> {
        let path = dir.join(format!("{id}.tsv"));
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("read {}: {error}", path.display())),
        };
        let mut leg = Leg::default();
        for line in text.lines().filter(|line| !line.is_empty()) {
            let fields: Vec<&str> = line.split('\t').collect();
            let bad = || format!("{}: not a run table's row: {line:?}", path.display());
            let (row_id, run, outcome) = match fields.as_slice() {
                [row_id, run, outcome, ..] => (*row_id, *run, *outcome),
                _ => return Err(bad()),
            };
            if row_id != id {
                return Err(bad());
            }
            let field = |at: usize| fields.get(at).copied().unwrap_or("");
            let outcome = match outcome {
                "OK" => Outcome::Drawn(format!("{} {}", field(3), field(4))),
                "REFUSED" => Outcome::Refused(field(3).to_string()),
                "SKIP" => Outcome::Skipped(field(3).to_string()),
                _ => return Err(bad()),
            };
            if run == "-" {
                leg.stop = Some(outcome);
            } else {
                let run = run.parse().map_err(|_| bad())?;
                if matches!(outcome, Outcome::Drawn(_)) && images_blank(field(6)) {
                    leg.blank.insert(run);
                }
                leg.runs.insert(run, outcome);
            }
        }
        Ok(Some(leg))
    }

    /// Run `run`'s outcome: the stop row's where the draw stopped.
    fn outcome(&self, run: u64) -> Option<&Outcome> {
        self.stop.as_ref().or_else(|| self.runs.get(&run))
    }

    /// Whether every run drew, at least two, and wrote one digest and length: the same bytes
    /// whatever the inputs.
    fn one_digest(&self) -> bool {
        let mut drawn = self.runs.values().map(|outcome| match outcome {
            Outcome::Drawn(bytes) => Some(bytes),
            _ => None,
        });
        let first = drawn.next().flatten();
        self.stop.is_none()
            && self.runs.len() > 1
            && first.is_some()
            && drawn.all(|other| other == first)
    }
}

/// One run's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Verdict {
    Same,
    Different,
    RefusedAlike,
    Skipped,
    Missing,
}

impl Verdict {
    fn name(self) -> &'static str {
        match self {
            Verdict::Same => "same",
            Verdict::Different => "different",
            Verdict::RefusedAlike => "refused alike",
            Verdict::Skipped => "skipped",
            Verdict::Missing => "missing",
        }
    }

    /// Whether the run was compared: drawn in both legs.
    fn compared(self) -> bool {
        matches!(self, Verdict::Same | Verdict::Different)
    }

    /// The verdict of a run whose legs say `base` and `change` (`None`: no table or no row).
    fn of(base: Option<&Outcome>, change: Option<&Outcome>) -> Verdict {
        use Outcome::{Drawn, Refused, Skipped};
        match (base, change) {
            (Some(Skipped(_)), _) | (_, Some(Skipped(_))) => Verdict::Skipped,
            (None, _) | (_, None) => Verdict::Missing,
            (Some(Drawn(a)), Some(Drawn(b))) if a == b => Verdict::Same,
            (Some(Refused(a)), Some(Refused(b))) if a == b => Verdict::RefusedAlike,
            _ => Verdict::Different,
        }
    }
}

/// One run's verdict, and each leg's outcome as written.
pub(super) struct Run {
    verdict: Verdict,
    base: String,
    change: String,
    /// Why a same run carries no evidence (the module docs), if it does not.
    no_evidence: Option<&'static str>,
}

/// The verdicts of every run of every draw.
pub(super) struct Report {
    /// Each draw's runs.
    pub(super) draws: Vec<(String, Vec<Run>)>,
    /// Draws whose uncompared runs are expected.
    pub(super) expected_skips: BTreeSet<String>,
}

impl Report {
    /// Compare the tables in `base` and `change` of every draw in `ids`, `runs` runs each.
    pub(super) fn of(
        base: &Path,
        change: &Path,
        ids: &[String],
        runs: u64,
        expected_skips: BTreeSet<String>,
    ) -> Result<Report, String> {
        if let Some(stray) = expected_skips.iter().find(|id| !ids.contains(id)) {
            return Err(format!(
                "--expected-skips names {stray}, not a draw of the draws file"
            ));
        }
        let mut draws = Vec::new();
        for id in ids {
            let (a, b) = (Leg::read(base, id)?, Leg::read(change, id)?);
            let described = |outcome: Option<&Outcome>, leg: &Option<Leg>| match (outcome, leg) {
                (Some(outcome), _) => outcome.describe(),
                (None, None) => "no table".to_string(),
                (None, Some(_)) => "no row".to_string(),
            };
            let one_digest =
                a.as_ref().is_some_and(Leg::one_digest) || b.as_ref().is_some_and(Leg::one_digest);
            let blank = |leg: &Option<Leg>, run: u64| {
                leg.as_ref().is_some_and(|leg| leg.blank.contains(&run))
            };
            let verdicts = (0..runs)
                .map(|run| {
                    let oa = a.as_ref().and_then(|leg| leg.outcome(run));
                    let ob = b.as_ref().and_then(|leg| leg.outcome(run));
                    let verdict = Verdict::of(oa, ob);
                    let no_evidence = match verdict {
                        Verdict::Same if blank(&a, run) && blank(&b, run) => {
                            Some("every image uniform or unchanged")
                        }
                        Verdict::Same if one_digest => Some("one digest across every run"),
                        _ => None,
                    };
                    Run {
                        verdict,
                        base: described(oa, &a),
                        change: described(ob, &b),
                        no_evidence,
                    }
                })
                .collect();
            draws.push((id.clone(), verdicts));
        }
        Ok(Report {
            draws,
            expected_skips,
        })
    }

    /// How many runs have each verdict.
    pub(super) fn totals(&self) -> BTreeMap<Verdict, u64> {
        let mut totals = BTreeMap::new();
        for (_, runs) in &self.draws {
            for run in runs {
                *totals.entry(run.verdict).or_insert(0) += 1;
            }
        }
        totals
    }

    /// How many same runs carry no evidence (the module docs).
    pub(super) fn without_evidence(&self) -> u64 {
        self.draws
            .iter()
            .flat_map(|(_, runs)| runs)
            .filter(|run| run.no_evidence.is_some())
            .count() as u64
    }

    /// The printed report: a line per draw, the totals and the verdict.
    pub(super) fn text(&self) -> String {
        let mut out = String::new();
        for (id, runs) in &self.draws {
            let mut counts: BTreeMap<Verdict, Vec<usize>> = BTreeMap::new();
            for (at, run) in runs.iter().enumerate() {
                counts.entry(run.verdict).or_default().push(at);
            }
            let named: Vec<String> = counts
                .iter()
                .map(|(verdict, at)| match verdict {
                    Verdict::Same => format!("same {}", at.len()),
                    _ => format!("{} {} (runs {:?})", verdict.name(), at.len(), at),
                })
                .collect();
            let first_other = runs.iter().find(|run| run.verdict != Verdict::Same);
            let _ = write!(out, "{id}: {}", named.join(", "));
            if let Some(run) = first_other {
                let _ = write!(
                    out,
                    "; base: {}; change: {}",
                    clip(&run.base),
                    clip(&run.change)
                );
            }
            if self.expected_skips.contains(id) {
                out.push_str(" (an expected skip)");
            }
            let blind: Vec<&Run> = runs
                .iter()
                .filter(|run| run.no_evidence.is_some())
                .collect();
            if let Some(run) = blind.first() {
                let _ = write!(
                    out,
                    "; no evidence in {} same runs ({})",
                    blind.len(),
                    run.no_evidence.unwrap_or_default()
                );
            }
            out.push('\n');
        }
        let totals = self.totals();
        let count = |verdict| totals.get(&verdict).copied().unwrap_or(0);
        let expected: u64 = totals.values().sum();
        let compared = count(Verdict::Same) + count(Verdict::Different);
        let _ = writeln!(
            out,
            "totals: compared {compared} of {expected} (same {}, different {}); not compared {} \
             (skipped {}, refused alike {}, missing {}); excused {}; same without evidence {}",
            count(Verdict::Same),
            count(Verdict::Different),
            expected - compared,
            count(Verdict::Skipped),
            count(Verdict::RefusedAlike),
            count(Verdict::Missing),
            self.excused(),
            self.without_evidence(),
        );
        let _ = writeln!(
            out,
            "verdict: {}",
            match (self.failure(), self.without_evidence()) {
                (None, 0) => "PASS".to_string(),
                (None, blind) => format!(
                    "PASS, but {blind} of the {} same runs carry no evidence of the bytes \
                     agreeing",
                    count(Verdict::Same)
                ),
                (Some(failure), _) => format!("FAIL: {}", failure.message()),
            }
        );
        out
    }

    /// Uncompared runs of the draws `--expected-skips` names.
    fn excused(&self) -> u64 {
        self.draws
            .iter()
            .filter(|(id, _)| self.expected_skips.contains(id))
            .flat_map(|(_, runs)| runs)
            .filter(|run| !run.verdict.compared())
            .count() as u64
    }

    /// Why the A/B fails, if it does (the module docs' exit statuses).
    pub(super) fn failure(&self) -> Option<Failure> {
        let differing: Vec<&str> = self
            .draws
            .iter()
            .filter(|(_, runs)| runs.iter().any(|run| run.verdict == Verdict::Different))
            .map(|(id, _)| id.as_str())
            .collect();
        if !differing.is_empty() {
            return Some(Failure::Differs(format!(
                "runs differ in {} draws: {}",
                differing.len(),
                differing.join(" ")
            )));
        }
        let uncompared: Vec<&str> = self
            .draws
            .iter()
            .filter(|(id, runs)| {
                !self.expected_skips.contains(id) && runs.iter().any(|run| !run.verdict.compared())
            })
            .map(|(id, _)| id.as_str())
            .collect();
        if !uncompared.is_empty() {
            return Some(Failure::Uncompared(format!(
                "runs not compared in {} draws not expected to skip: {}",
                uncompared.len(),
                uncompared.join(" ")
            )));
        }
        None
    }

    /// The line per run: the draw, the run, the verdict and each leg's outcome.
    pub(super) fn rows(&self) -> String {
        let mut out = String::from("draw\trun\tverdict\tbase\tchange\n");
        for (id, runs) in &self.draws {
            for (at, run) in runs.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "{id}\t{at}\t{}\t{}\t{}",
                    run.verdict.name(),
                    run.base.replace(['\t', '\n'], " "),
                    run.change.replace(['\t', '\n'], " ")
                );
            }
        }
        out
    }
}

/// `text`'s first 160 characters.
fn clip(text: &str) -> String {
    text.chars().take(160).collect()
}

/// The draws file's draw IDs, in order.
fn ids(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("read {path}: {error}"))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let value: serde_json::Value =
                serde_json::from_str(line).map_err(|error| format!("{path}: {error}"))?;
            value["id"]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{path}: a line without an \"id\""))
        })
        .collect()
}

/// `corpus-draw-ab compare` (the module docs).
pub(super) fn compare(args: &[String]) -> Result<(), Failure> {
    let options = options(args)?;
    let base = Path::new(required(&options, "base")?);
    let change = Path::new(required(&options, "change")?);
    let ids = ids(required(&options, "draws")?)?;
    let runs = number(&options, "runs", 12)?;
    if ids.is_empty() || runs == 0 {
        return Err(Failure::from(
            "nothing to compare: no draw, or no run".to_string(),
        ));
    }
    let expected_skips: BTreeSet<String> = options
        .get("expected-skips")
        .map(|list| {
            list.split(',')
                .filter(|id| !id.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let report = Report::of(base, change, &ids, runs, expected_skips)?;
    if let Some(out) = options.get("out") {
        std::fs::write(out, report.rows()).map_err(|error| format!("write {out}: {error}"))?;
    }
    print!("{}", report.text());
    match report.failure() {
        Some(failure) => Err(failure),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::{Report, Verdict};
    use std::collections::BTreeSet;
    use std::path::Path;

    /// Write `rows` (each `run\toutcome\t...`) as draw `id`'s table in `dir`.
    fn table(dir: &Path, id: &str, rows: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        let text: String = rows.iter().map(|row| format!("{id}\t{row}\n")).collect();
        std::fs::write(dir.join(format!("{id}.tsv")), text).unwrap();
    }

    /// A draw skipped in both legs, one with no table, a run missing, and refusals are not
    /// compared, and fail the A/B unless an expected skip; a run that differs fails it whatever
    /// the list says. Only runs drawn in both legs count as compared.
    #[test]
    fn a_draw_skipped_in_both_legs_is_not_compared_and_fails_the_a_b() {
        let root =
            std::env::temp_dir().join(format!("corpus-draw-ab-compare-{}", std::process::id()));
        let (base, change) = (root.join("base"), root.join("change"));
        let ran = ["0\tOK\taa\t8\tin", "1\tOK\tbb\t8\tin"];
        // Drawn alike.
        table(&base, "same", &ran);
        table(&change, "same", &ran);
        // Skipped in both legs: the hole this command closes.
        table(&base, "skip", &["-\tSKIP\tan input"]);
        table(&change, "skip", &["-\tSKIP\tan input"]);
        // No table in the change.
        table(&base, "gone", &ran);
        // A run missing in the base.
        table(&base, "short", &ran[..1]);
        table(&change, "short", &ran);
        // Refused alike at setup, and refused alike in one run.
        table(&base, "refused", &["-\tREFUSED\tno"]);
        table(&change, "refused", &["-\tREFUSED\tno"]);
        let ids: Vec<String> = ["same", "skip", "gone", "short", "refused"]
            .map(String::from)
            .to_vec();
        let report = |skips: &[&str]| {
            let skips: BTreeSet<String> = skips.iter().map(|s| s.to_string()).collect();
            Report::of(&base, &change, &ids, 2, skips).unwrap()
        };
        let all = report(&[]);
        let totals = all.totals();
        assert_eq!(totals.get(&Verdict::Same), Some(&3));
        assert_eq!(totals.get(&Verdict::Skipped), Some(&2));
        assert_eq!(totals.get(&Verdict::Missing), Some(&3));
        assert_eq!(totals.get(&Verdict::RefusedAlike), Some(&2));
        assert_eq!(totals.get(&Verdict::Different), None);
        assert!(all
            .text()
            .contains("totals: compared 3 of 10 (same 3, different 0); not compared 7"));
        assert_eq!(all.failure().map(|f| f.status()), Some(5));
        // The expected skips excuse their own draws alone.
        let excused = report(&["skip", "gone", "refused"]);
        assert_eq!(excused.failure().map(|f| f.status()), Some(5), "short");
        let excused = report(&["skip", "gone", "short", "refused"]);
        assert!(excused.failure().is_none(), "{}", excused.text());
        assert!(excused.text().contains("verdict: PASS"));
        // A difference fails whatever is excused: other bytes, or drawn against refused.
        table(&change, "gone", &["0\tOK\taa\t8\tin", "1\tREFUSED\tno"]);
        table(&change, "short", &["0\tOK\tzz\t8\tin", "1\tOK\tbb\t8\tin"]);
        let differs = report(&["skip", "gone", "short", "refused"]);
        assert_eq!(differs.failure().map(|f| f.status()), Some(1));
        assert!(differs
            .rows()
            .contains("short\t0\tdifferent\tOK aa 8\tOK zz 8\n"));
        assert!(differs
            .rows()
            .contains("gone\t1\tdifferent\tOK bb 8\tREFUSED no\n"));
        // A skip in one leg alone is not compared either.
        table(&change, "same", &["-\tSKIP\tan input"]);
        let one = report(&["skip", "gone", "short", "refused"]);
        assert!(one
            .rows()
            .contains("same\t0\tskipped\tOK aa 8\tSKIP an input\n"));
        // A list naming no draw of the file is refused.
        let stray: BTreeSet<String> = ["nope".to_string()].into();
        assert!(Report::of(&base, &change, &ids, 2, stray).is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A same run carries no evidence where both legs flagged every image, or where a leg wrote
    /// one digest across every run; the A/B still passes, but says how many runs that is. A
    /// run flagged in one leg alone, or drawn in one image of two, still counts.
    #[test]
    fn same_runs_on_blank_images_or_one_digest_carry_no_evidence() {
        let root =
            std::env::temp_dir().join(format!("corpus-draw-ab-evidence-{}", std::process::id()));
        let (base, change) = (root.join("base"), root.join("change"));
        // Every image flagged in both legs in run 0; in one leg alone in run 1; partly in run 2.
        let flagged = [
            "0\tOK\taa\t8\tin\timages=0:uniform",
            "1\tOK\tbb\t8\tin\timages=0:unchanged",
            "2\tOK\tcc\t8\tin\timages=0:uniform,1:drawn",
        ];
        table(&base, "blank", &flagged);
        table(
            &change,
            "blank",
            &[flagged[0], "1\tOK\tbb\t8\tin\timages=0:drawn", flagged[2]],
        );
        // One digest across every run, without an images column (an older table).
        let one = ["0\tOK\tzz\t8\tin", "1\tOK\tzz\t8\tin", "2\tOK\tzz\t8\tin"];
        table(&base, "one", &one);
        table(&change, "one", &one);
        // Drawn alike, different digests, no flags.
        let drawn = [
            "0\tOK\taa\t8\tin\timages=0:drawn",
            "1\tOK\tbb\t8\tin\timages=0:drawn",
            "2\tOK\tcc\t8\tin\timages=none",
        ];
        table(&base, "drawn", &drawn);
        table(&change, "drawn", &drawn);
        let ids: Vec<String> = ["blank", "one", "drawn"].map(String::from).to_vec();
        let report = Report::of(&base, &change, &ids, 3, BTreeSet::new()).unwrap();
        assert_eq!(report.totals().get(&Verdict::Same), Some(&9));
        assert_eq!(report.without_evidence(), 4);
        assert!(report.failure().is_none());
        let text = report.text();
        assert!(text.contains(
            "blank: same 3; no evidence in 1 same runs (every image uniform or unchanged)\n"
        ));
        assert!(text
            .contains("one: same 3; no evidence in 3 same runs (one digest across every run)\n"));
        assert!(text.contains("drawn: same 3\n"));
        assert!(text.contains("same without evidence 4"));
        assert!(text.contains("verdict: PASS, but 4 of the 9 same runs carry no evidence"));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
