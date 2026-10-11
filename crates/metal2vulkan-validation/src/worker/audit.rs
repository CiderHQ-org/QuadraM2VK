//! The bitcode-reader audit lane under the worker boundary.
//!
//! Each attempt of the lane runs the engine's reader on one corpus row: the decode of a row's
//! retained bitcode ([`bitcode_audit::read_blob`]). It runs in its own killable child of the
//! lane's tool (`corpus-triage`, which installs the counting allocator and calls
//! [`serve_if_requested`] first) through [`super::run_json`], under the 20-second deadline and
//! the 500-MiB process-group limit, with the 500-MiB allocator budget inside the child
//! (as every worker child serves), exactly as a translation attempt: a row whose attempt runs away is
//! stopped, measured, and reported as that row's result, and the sweep goes on. What the child
//! returns is the attempt's typed result; the lane's comparisons and summaries (`llvm-dis` and
//! its text scan, the shape histograms) stay in the parent, outside the boundary that contains
//! the product's work. (The canon lane that shared it was retired.)

use super::{alloc, run_json, Measured, WorkerFailure};
use crate::bitcode_audit::{self, Reading};
use serde::{Deserialize, Serialize};
use std::io::{BufWriter, Write as _};
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

/// The first argument that starts `corpus-triage` as an audit worker.
pub const WORKER_ARG: &str = "--audit-worker";

/// What one audit worker runs on its row's blob.
#[derive(Debug, Serialize, Deserialize)]
enum Request {
    /// The bitcode-reader lane's reading ([`bitcode_audit::read_blob`]).
    Read,
}

/// The worker's stdin.
#[derive(Debug, Serialize, Deserialize)]
struct Input {
    request: Request,
    /// The row's retained bitcode, base64 as the index keeps it.
    blob_b64: String,
}

/// The worker's result: the answer to its [`Request`].
#[derive(Debug, Serialize, Deserialize)]
enum Output {
    Read(Reading),
}

/// Run `request` on `blob_b64` in a killable child of `program` (a binary that serves
/// [`WORKER_ARG`]).
fn run(
    program: &Path,
    request: Request,
    blob_b64: &str,
) -> Result<(Output, Measured), WorkerFailure> {
    let mut command = Command::new(program);
    command.arg(WORKER_ARG);
    let run = run_json::<Output>(command, "audit-worker", |file| {
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(
            &mut writer,
            &Input {
                request,
                blob_b64: blob_b64.into(),
            },
        )
        .map_err(|error| format!("encode the audit worker's input: {error}"))?;
        writer
            .flush()
            .map_err(|error| format!("write the audit worker's input: {error}"))
    })?;
    let measured = run.measured();
    Ok((run.result, measured))
}

/// The bitcode-reader lane's reading of `blob_b64` in a killable child of `program`, and what the
/// parent measured around it.
pub fn read(program: &Path, blob_b64: &str) -> Result<(Reading, Measured), WorkerFailure> {
    let (Output::Read(reading), measured) = run(program, Request::Read, blob_b64)?;
    Ok((reading, measured))
}

/// What a lane's attempt workers measured over a sweep: how many ran, how many the boundary
/// stopped, the most resident memory and the largest counting-allocator peak any of them
/// reached, and the slowest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub workers: usize,
    pub stopped: usize,
    pub peak_resident_bytes: u64,
    pub peak_allocator_bytes: u64,
    pub slowest: Duration,
}

impl Tally {
    /// Count one attempt's worker: what the parent measured around it (`None` when none was
    /// spawned) and whether the boundary stopped it.
    pub fn record(&mut self, measured: Option<Measured>, stopped: bool) {
        self.stopped += usize::from(stopped);
        let Some(measured) = measured else {
            return;
        };
        self.workers += 1;
        self.peak_resident_bytes = self
            .peak_resident_bytes
            .max(measured.peak_resident_bytes.unwrap_or(0));
        self.peak_allocator_bytes = self
            .peak_allocator_bytes
            .max(measured.allocator_peak_bytes.unwrap_or(0));
        self.slowest = self.slowest.max(measured.elapsed);
    }

    /// The summary fields a lane prints.
    pub fn fields(&self) -> String {
        format!(
            "workers={}\tworker_stopped={}\tpeak_worker_rss_bytes={}\tpeak_worker_allocator_bytes={}\tslowest_worker_ms={}",
            self.workers,
            self.stopped,
            self.peak_resident_bytes,
            self.peak_allocator_bytes,
            self.slowest.as_millis()
        )
    }
}

/// Serve one audit attempt when this process was started with [`WORKER_ARG`], then exit (0 when a
/// result was written, 1 otherwise); return at once otherwise. Call it first in `main`.
pub fn serve_if_requested() {
    if std::env::args_os().nth(1).as_deref() != Some(std::ffi::OsStr::new(WORKER_ARG)) {
        return;
    }
    std::process::exit(super::serve_process(
        "audit worker",
        alloc::installed(),
        serve,
    ))
}

/// The worker's attempt on `input` (the bytes of an [`Input`]). Each lane's attempt keeps its own
/// meter and 20-second deadline; the parent's watchdog is the boundary around it.
fn serve(input: &[u8], _started: Instant) -> Result<Output, String> {
    let Input { request, blob_b64 } = serde_json::from_slice(input)
        .map_err(|error| format!("decode the audit worker's input: {error}"))?;
    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &blob_b64)
        .map_err(|error| format!("the audit worker's blob is not base64: {error}"))?;
    drop(blob_b64);
    Ok(match request {
        Request::Read => Output::Read(bitcode_audit::read_blob(&bytes)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tally_counts_workers_stops_and_peaks() {
        let mut tally = Tally::default();
        let measured = |elapsed_ms, rss, allocator| Measured {
            elapsed: Duration::from_millis(elapsed_ms),
            peak_resident_bytes: rss,
            allocator_peak_bytes: allocator,
        };
        tally.record(Some(measured(5, Some(300), Some(100))), false);
        tally.record(Some(measured(20_000, Some(200), None)), true);
        // A stop before any worker ran (a setup failure) is a stop without a measure.
        tally.record(None, true);
        assert_eq!(
            tally,
            Tally {
                workers: 2,
                stopped: 2,
                peak_resident_bytes: 300,
                peak_allocator_bytes: 100,
                slowest: Duration::from_secs(20),
            }
        );
        assert_eq!(
            tally.fields(),
            "workers=2\tworker_stopped=2\tpeak_worker_rss_bytes=300\tpeak_worker_allocator_bytes=100\tslowest_worker_ms=20000"
        );
    }

    /// The child answers a read with the lane's in-process result, through JSON as the parent
    /// reads it; a blob that is not base64 is the exchange's error, never a row result.
    #[test]
    fn the_worker_answers_a_read_with_the_lanes_result() {
        let bytes = b"not bitcode";
        let blob_b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes);
        let input = |request| {
            serde_json::to_vec(&Input {
                request,
                blob_b64: blob_b64.clone(),
            })
            .unwrap()
        };
        let through_json = |output: Output| -> Output {
            serde_json::from_slice(&serde_json::to_vec(&output).unwrap()).unwrap()
        };
        let Output::Read(reading) =
            through_json(serve(&input(Request::Read), Instant::now()).unwrap());
        assert_eq!(reading.read, bitcode_audit::read_blob(bytes).read);
        assert!(reading.read.is_err());

        let not_base64 = serde_json::to_vec(&Input {
            request: Request::Read,
            blob_b64: "not*base64".into(),
        })
        .unwrap();
        assert!(serve(&not_base64, Instant::now())
            .unwrap_err()
            .contains("not base64"));
    }
}
