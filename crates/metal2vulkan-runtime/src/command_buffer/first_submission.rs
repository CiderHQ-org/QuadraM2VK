//! A command buffer's first submission: Metal resolves what a command buffer's commands reach as it
//! submits it, not as it was committed, so a command buffer committed before a rebind (`rebind.rs`)
//! or a release and submitted after it behaves as one committed after it (the consumer's
//! measurement on the M3 Max, its rebind-reach tables 1 to 3).
//!
//! - **Until then, held weakly.** From the commit to the first submission the command buffer
//!   holds the pages its commands took of buffers over host pages (`rebind::Taken`) and the live
//!   buffers' table its resident set was made of (`Live`) weakly: a rebind or a release meanwhile
//!   lets the pages go, their deallocator run, where no submitted command buffer holds them
//!   (the question, answered without revoking anything). From the first submission on it holds them
//!   until it completes, so a submitted command buffer keeps the pages it took, as Metal's does
//!   (table 3: a submitted command buffer never sees the new pages). A root table recorded
//!   before then, by a rebind's encoding again (`reencode.rs`) or a host step, holds the table
//!   (`State::hold_live`) until the first submission, which lets go of it where the list changed
//!   meanwhile: at once, as Metal lets a released buffer go, but on a handler thread
//!   (`Device::let_go_elsewhere`), as its last hold of a released buffer's pages runs that
//!   buffer's deallocator, which may not run on the scheduler's thread under its locks. The
//!   deallocator's wait from the release until then, past Metal's, is announced
//!   (`host-deallocator-delayed`).
//! - **Resolved again where the list changed.** Where the live buffers' list changed since the
//!   commit (a buffer made or gone, a rebind), the submission lists again what the commit listed
//!   (`LiveBuffers::reach`, a rebind having ended its use), makes its resident set again of the
//!   list as it stands and of what its encoders made resident as it lies now, and points every
//!   root table at it: a buffer its encoders made resident reaches the pages it lies in now
//!   (case 3, `useResource:` and `useHeap:`, and after two rebinds the second's), and one it
//!   reaches only unnamed reads 0 once a rebind ended its use (cases 4 and 4x). A root table's
//!   own range whose Metal address the remade set holds anew goes: the buffer lies elsewhere.
//!   (A set of any count; before, past the search's 2^16 entries, each root table took back its own
//!   set.)
//! - **What it named, encoded again; else typed.** A command that named a buffer over host pages
//!   was resolved as it was encoded (a binding's descriptor or root-table static, a blit's
//!   operand, an attachment, a vertex, index or indirect buffer, a texture over the pages, an
//!   indirect command's resolved buffer), where Metal reaches the pages the buffer lies in at the
//!   submission (cases 1, 2 and 5). The rebind encodes the compute, blit and render encoders that
//!   named it again before this submission, which waits for them (`reencode.rs`). Where an encoder
//!   that could not be encoded again named a buffer that lies in other pages by then, each of
//!   Metal's submissions holding such a command (`checks::metal_submissions`, the split) runs
//!   nothing and the command buffer completes with `CommandBufferError::EncodedUseRebound`, every
//!   other submission running, its signals firing and its handlers running: the gap until the
//!   encoded uses resolve at submission, announced once
//!   (`Degradation::EncodedUseRebound`). A GPU-alone replay's resident set, resolved at the
//!   commit (`generated_execute.rs`), fails alike where a range of it lies elsewhere by then.
//! - **Each submission at its own batch**. The first submission finds which of Metal's submissions
//!   are rebound, against the pages and the table it fixes for every later batch; each fails only
//!   as its own batch is submitted, after that batch's submission checks (`checks.rs`). One its
//!   check failed already keeps the check's code (8 for the consumer's detach) and announces
//!   nothing, and so does one whose every encoder the commit's failure failed (`commit_failed`'s);
//!   the submissions before it run as Metal runs them, and the command buffer fails with the first
//!   failure in submission order. Any other rebound submission runs nothing and is announced,
//!   though the command buffer failed already and keeps that first code (move14: the consumer's
//!   review; Metal runs it over the new pages).

use super::checks::metal_submissions;
use super::{write_fields, CommandBufferShared, Resident, State};
use crate::buffer::HostPages;
use crate::error::{CommandBufferError, Error};
use crate::live_buffers::LiveTable;
use crate::rebind::SubmittedHolds;
use crate::residency::Residency;
use crate::upload::Upload;
use crate::used_resources::UsedResources;
use crate::Device;
use metal2vulkan::bind::{ResidentRange, ResidentSet};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ops::Range;
use std::sync::{Arc, Weak};

/// A root table a command buffer's commands read (`State::root_table`): its upload and its
/// device address.
#[derive(Clone)]
pub(crate) struct NotedTable {
    pub(crate) upload: Upload,
    pub(crate) address: u64,
}

/// The live buffers' table a command buffer's resident set was made of (the module docs).
pub(super) enum Live {
    /// Weakly, with the generation of the list it was made of: from the commit to the first
    /// submission.
    Waiting {
        table: Weak<LiveTable>,
        generation: u64,
    },
    /// From the first submission (or a host step before it) until the command buffer completes.
    Held(Arc<LiveTable>),
}

impl Live {
    /// `table`, as a command buffer holds it: weakly before its first submission (`submitted`
    /// false), unless no other holds it (a table made as the list changed meanwhile, which the
    /// device keeps not), whose last hold would let pages go here.
    pub(super) fn of(table: Arc<LiveTable>, submitted: bool) -> Live {
        if submitted || Arc::strong_count(&table) == 1 {
            return Live::Held(table);
        }
        Live::Waiting {
            generation: table.generation(),
            table: Arc::downgrade(&table),
        }
    }

    /// The table's set, where it is held.
    pub(super) fn set(&self) -> Option<&ResidentSet> {
        match self {
            Live::Held(table) => Some(table.set()),
            Live::Waiting { .. } => None,
        }
    }
}

/// What a commit listed among the live buffers (`LiveBuffers::reach`), and the generation of the
/// list after it.
pub(super) struct Listing {
    used: UsedResources,
    generation: u64,
}

impl Listing {
    /// `used`, listed on `device` just now.
    pub(super) fn new(device: &Device, used: UsedResources) -> Listing {
        Listing {
            generation: device.shared().live_buffers.generation(),
            used,
        }
    }
}

/// The Metal addresses a resident set's ranges hold, to find whether a root table's own range
/// lies in one of them: each range's start, by its order, with the furthest end of every range
/// starting there or before (ranges nest: a heap's holds its buffers').
struct Covered(Vec<(u64, u64)>);

impl Covered {
    fn of(ranges: impl IntoIterator<Item = ResidentRange>) -> Covered {
        let mut starts: Vec<(u64, u64)> = ranges
            .into_iter()
            .map(|range| (range.guest, range.guest.saturating_add(range.length.max(1))))
            .collect();
        starts.sort_unstable();
        let mut furthest = 0;
        for (_, end) in &mut starts {
            furthest = furthest.max(*end);
            *end = furthest;
        }
        Covered(starts)
    }

    /// Whether Metal address `guest` lies in one of the ranges.
    fn holds(&self, guest: u64) -> bool {
        let below = self.0.partition_point(|&(start, _)| start <= guest);
        below.checked_sub(1).is_some_and(|at| guest < self.0[at].1)
    }
}

impl State {
    /// As the command buffer commits (its segments recorded): what its commands took is held
    /// weakly until its first submission, which lists `listing` again where the list changed.
    /// The pages its commands held go to `held`, which its deferred command buffers' replays
    /// hold until they are replayed (`recorder`).
    pub(super) fn wait_for_submission(&mut self, listing: Listing, held: &mut Vec<Arc<HostPages>>) {
        for record in &mut self.ended {
            record.taken.wait(held);
        }
        let segments = self
            .recorded
            .iter_mut()
            .flat_map(|part| part.segments.iter_mut())
            .filter_map(|segment| segment.recorded.as_mut());
        for done in segments {
            done.taken.wait(held);
        }
        self.listing = Some(listing);
    }

    /// Hold the live buffers' table its resident set was made of from now on: made again where
    /// the list changed since (the module docs). A root table a host step uploads before the
    /// first submission is pointed at the set as it stands then.
    /// A table held already, by a host step's root table or an encoder encoded again at a rebind
    /// before the first submission (`reencode.rs`), is made again where the list changed since
    /// it was made: the first submission resolves the set as the list stands then.
    pub(super) fn hold_live(&mut self, device: &Device) -> Result<(), Error> {
        let now = device.shared().live_buffers.generation();
        match &self.resident {
            Some(Resident {
                live: Live::Waiting { table, generation },
                ..
            }) => {
                // The list unchanged, the device keeps the table: the upgrade is not its last
                // hold. Changed, the table is not upgraded at all: an upgrade let go of here,
                // under the scheduler's lock and this state's, could be the last hold of a gone
                // buffer's pages, whose release runs the program's code.
                let kept = (*generation == now).then(|| table.upgrade()).flatten();
                if let Some(table) = kept {
                    if let Some(resident) = &mut self.resident {
                        resident.live = Live::Held(table);
                    }
                    return Ok(());
                }
            }
            Some(Resident {
                live: Live::Held(table),
                ..
            }) if table.generation() != now => {}
            _ => return Ok(()),
        }
        let fresh = self.fresh_ranges();
        self.remake_resident(device, &fresh)
    }

    /// The ranges of what its encoders made resident, as it lies now.
    fn fresh_ranges(&self) -> Vec<ResidentRange> {
        self.residency
            .as_ref()
            .map(Residency::resident_ranges)
            .unwrap_or_default()
    }

    /// Its resident set made again of the live buffers as they stand now and `fresh`, what its
    /// encoders made resident as it lies now, and every noted root table pointed at it. A table
    /// it held goes at once, on a handler thread.
    fn remake_resident(&mut self, device: &Device, fresh: &[ResidentRange]) -> Result<(), Error> {
        let live = device.shared().live_buffers.table(device)?;
        let set = live.set();
        // A root table's own range whose Metal address the remade set holds goes: the buffer
        // there lies as the set says now.
        let covered = Covered::of(live.ranges().chain(fresh.iter().copied()));
        let mut beyond: Vec<ResidentRange> = self
            .root_ranges
            .iter()
            .filter(|range| !covered.holds(range.guest))
            .copied()
            .collect();
        beyond.extend(fresh.iter().copied());
        beyond.retain(|range| !set.holds(range));
        let (merged, address, count) = match beyond.is_empty() {
            true => (None, live.address(), set.len()),
            false => {
                let merged = set.with(beyond);
                let (address, count) = self.upload_set(device, &merged)?;
                (Some(merged), Some(address), count)
            }
        };
        for noted in &self.root_tables {
            write_fields(
                &noted.upload,
                noted.address,
                address.unwrap_or(noted.address),
                count,
            )?;
        }
        let replaced = self.resident.replace(Resident {
            live: Live::Held(live),
            merged,
            address,
            count,
        });
        // A table held already, made of the list before it changed, may hold a gone buffer's
        // pages last, whose release runs the program's code (a deallocator), which may not run
        // here: on the scheduler's thread, under its lock and this state's. It goes at once all
        // the same, as Metal lets go of a buffer no submitted command buffer holds, on a handler
        // thread.
        if let Some(Resident {
            live: Live::Held(table),
            ..
        }) = replaced
        {
            device.let_go_elsewhere(Box::new(table));
        }
        Ok(())
    }

    /// As its first batch is submitted (the module docs), on the thread that submits it, under
    /// the scheduler's lock and its own, after that batch's checks: Metal's submissions found
    /// rebound, or holding a detached buffer, go to `failing`, each to fail as its own batch is
    /// submitted (`State::failing_at`). What its commands took is judged and held under one lock of
    /// the device's buffers over host pages (`Submitting`), so a detach comes either before (and
    /// fails the submission) or after (and waits for the command buffer). Nothing here lets pages
    /// go: the weak holds become strong, and what it held strongly it retains.
    /// `Ok(false)`, changing nothing, while a rebind since its commit has encoders of it, which
    /// `buffer`'s rebinds mark, to encode again (`reencode.rs`): read under the lock its rebinds
    /// mark them under, so a rebind comes either before this submission (which waits for them)
    /// or after it (which keeps the old pages). From here it waits for no rebind's replay; its
    /// encoders' logs stay until its release (`CommandBufferShared::complete`).
    pub(super) fn first_submission(
        &mut self,
        device: &Device,
        buffer: &CommandBufferShared,
    ) -> Result<bool, Error> {
        let shared = device.shared();
        let mut verdicts = BTreeMap::new();
        let mut moved = false;
        let mut replays = false;
        let mut holds = SubmittedHolds::new(device);
        {
            let mut submitting = shared.rebinds.submitting();
            if buffer.reencode_pending() {
                return Ok(false);
            }
            self.submitted = true;
            submitting.submitted_now(&shared.rebinds, buffer);
            let State {
                ended,
                recorded,
                retained,
                ..
            } = self;
            for (at, record) in ended.iter_mut().enumerate() {
                let found = submitting.submit(&mut record.taken, retained, &mut holds);
                moved |= found.moved;
                judge(&mut verdicts, at, found.failure());
            }
            for part in recorded.iter_mut() {
                let segments = part.segments.iter_mut().filter_map(|s| s.recorded.as_mut());
                for done in segments {
                    let found = submitting.submit(&mut done.taken, retained, &mut holds);
                    moved |= found.moved;
                    judge(&mut verdicts, part.encoder, found.failure());
                    replays |= !done.taken.resolved().is_empty();
                }
            }
        }
        if !holds.is_empty() {
            self.submitted_holds.push(holds);
        }
        // The list changed since the commit: what it listed is listed again, a rebind having
        // ended its use, and its resident set made of the list as it stands.
        let changed = self
            .listing
            .as_ref()
            .is_some_and(|listing| listing.generation != shared.live_buffers.generation());
        if changed || moved {
            if let Some(listing) = &self.listing {
                shared.live_buffers.reach(&listing.used);
            }
        }
        self.hold_live(device)?;
        // The tables host steps before now resolved against go at once, off this thread, under
        // whose locks the last hold of a released buffer's pages may not go
        // (`State::keep_stepped_table`).
        for earlier in self.stepped_tables.drain(..) {
            device.let_go_elsewhere(Box::new(earlier));
        }
        // A replay's resident set holds where every range of it lies as it did at the commit.
        if replays {
            let live = shared.live_buffers.table(device)?;
            if changed || moved {
                let now: HashSet<ResidentRange> =
                    live.ranges().chain(self.fresh_ranges()).collect();
                for part in &self.recorded {
                    let stale = part
                        .segments
                        .iter()
                        .filter_map(|segment| segment.recorded.as_ref())
                        .any(|done| done.taken.resolved().iter().any(|r| !now.contains(r)));
                    if stale {
                        judge(
                            &mut verdicts,
                            part.encoder,
                            Some(CommandBufferError::EncodedUseRebound),
                        );
                    }
                }
            }
            // Its pages held from now on, as the replay's ranges lie in them.
            self.retained.push(Box::new(live));
        }
        // Found here, against the pages and the table fixed for every later batch; each fails
        // at its own batch, after its check.
        self.failing = self.failing_submissions(&verdicts);
        // Their names and resolved ranges let go; the listing released without the lock.
        let State {
            ended,
            recorded,
            retained,
            listing,
            ..
        } = self;
        for record in ended.iter_mut() {
            record.taken.judged();
        }
        let segments = recorded
            .iter_mut()
            .flat_map(|part| part.segments.iter_mut())
            .filter_map(|segment| segment.recorded.as_mut());
        for done in segments {
            done.taken.judged();
        }
        // An empty listing (a command buffer that used no no-copy buffer and no heap) holds
        // nothing to let go of.
        if let Some(Listing { used, .. }) = listing.take() {
            if used != UsedResources::default() {
                retained.push(Box::new(used));
            }
        }
        self.root_tables.clear();
        self.root_ranges.clear();
        Ok(true)
    }

    /// Metal's submissions (the split) that hold an encoder `verdicts` fails, by their encoders,
    /// each with the failure of its encoders: `DetachedBacking` where one holds a detached buffer,
    /// else `EncodedUseRebound`.
    fn failing_submissions(
        &self,
        verdicts: &BTreeMap<usize, CommandBufferError>,
    ) -> Vec<(Range<usize>, CommandBufferError)> {
        if verdicts.is_empty() {
            return Vec::new();
        }
        let encoders = self.ended.iter().map(|record| {
            (
                record.kind,
                record.event_waits_before,
                record.event_signals_before,
            )
        });
        metal_submissions(encoders)
            .into_iter()
            .filter_map(|(_, encoders)| {
                let mut failure = None;
                for encoder in encoders.clone() {
                    if let Some(error) = verdicts.get(&encoder) {
                        judge_one(&mut failure, *error);
                    }
                }
                failure.map(|failure| (encoders, failure))
            })
            .collect()
    }

    /// As batch `index` is submitted, after its checks ran: the segments a host step recorded
    /// for it since the first submission are judged and held as the first submission judged the
    /// rest (`State::first_submission`); then each of Metal's submissions found failing that
    /// begins in the batch, or that holds such a segment, runs nothing from the batch on
    /// (`State::fail_part`), unless its check failed it already, which keeps the check's code,
    /// or the commit's failure failed every encoder of it (`State::failed_at_commit`); and the
    /// command buffer fails with the first such submission's failure unless it failed already,
    /// keeping its first code. Whether one of them failed rebound here (`EncodedUseRebound`),
    /// for the caller to announce without the lock: a rebound submission runs nothing where
    /// Metal's runs, whatever the command buffer failed with before.
    pub(super) fn failing_at(&mut self, device: &Device, index: usize) -> bool {
        let range = self.batch_range(index);
        let mut later = BTreeMap::new();
        let mut holds = SubmittedHolds::new(device);
        {
            let mut submitting = device.shared().rebinds.submitting();
            let State {
                recorded, retained, ..
            } = self;
            for part in &mut recorded[range.clone()] {
                let segments = part.segments.iter_mut().filter_map(|s| s.recorded.as_mut());
                for done in segments {
                    let found = submitting.submit(&mut done.taken, retained, &mut holds);
                    judge(&mut later, part.encoder, found.failure());
                    done.taken.judged();
                }
            }
        }
        if !holds.is_empty() {
            self.submitted_holds.push(holds);
        }
        if self.failing.is_empty() && later.is_empty() {
            return false;
        }
        let here: BTreeSet<usize> = self.recorded[range.clone()]
            .iter()
            .map(|part| part.encoder)
            .collect();
        let mut due = Vec::new();
        self.failing.retain(|(encoders, failure)| {
            let begins = encoders.clone().any(|encoder| here.contains(&encoder));
            if begins {
                due.push((encoders.clone(), *failure));
            }
            !begins
        });
        due.extend(self.failing_submissions(&later));
        due.retain(|(encoders, _)| !self.check_failed(encoders) && !self.commit_failed(encoders));
        if due.is_empty() {
            return false;
        }
        due.sort_by_key(|(encoders, _)| encoders.start);
        let State {
            recorded, retained, ..
        } = self;
        let failing = |encoder: &usize| due.iter().any(|(encoders, _)| encoders.contains(encoder));
        for part in recorded[range.start..]
            .iter_mut()
            .filter(|part| failing(&part.encoder))
        {
            State::fail_part(part, retained);
        }
        self.unrunnable
            .extend(due.iter().flat_map(|(encoders, _)| encoders.clone()));
        if self.failure.is_none() {
            self.failure = Some(due[0].1);
        }
        due.iter()
            .any(|(_, failure)| *failure == CommandBufferError::EncodedUseRebound)
    }

    /// Whether the commit's failure failed every encoder of Metal's submission of `encoders`.
    fn commit_failed(&self, encoders: &Range<usize>) -> bool {
        self.failed_at_commit
            .as_ref()
            .is_some_and(|failed| encoders.clone().all(|encoder| failed.fails(encoder)))
    }
}

/// Note `failure`, an encoder's at `at`, among `verdicts`: a detached buffer's over a rebound one.
fn judge(
    verdicts: &mut BTreeMap<usize, CommandBufferError>,
    at: usize,
    failure: Option<CommandBufferError>,
) {
    if let Some(failure) = failure {
        let mut held = verdicts.remove(&at);
        judge_one(&mut held, failure);
        if let Some(held) = held {
            verdicts.insert(at, held);
        }
    }
}

/// `failure` joined to `held`: `DetachedBacking` wins over `EncodedUseRebound`.
fn judge_one(held: &mut Option<CommandBufferError>, failure: CommandBufferError) {
    let detached = |error: &CommandBufferError| *error == CommandBufferError::DetachedBacking;
    if held
        .as_ref()
        .is_none_or(|held| !detached(held) && detached(&failure))
    {
        *held = Some(failure);
    }
}

impl State {
    /// Refuse (internal) to submit batch `range`'s parts where one of an encoder that recorded
    /// nothing for a command (`Taken::dead`, of its own or of a segment of it), or of one whose
    /// submission failed rebound or detached (`unrunnable`), still holds work: the structural
    /// hold, that a command buffer holding a detached or rebound use never submits that use's
    /// Vulkan work. A build with debug assertions stops there.
    pub(super) fn runs_nothing_unrunnable(&self, range: Range<usize>) -> Result<(), Error> {
        let held = self.recorded[range].iter().any(|part| {
            let dead = self
                .ended
                .get(part.encoder)
                .is_some_and(|record| record.taken.dead());
            (dead || self.unrunnable.contains(&part.encoder)) && !part.raws.is_empty()
        });
        debug_assert!(!held, "a detached or rebound use's work submitted");
        match held {
            true => Err(crate::error::internal(
                "a detached or rebound use's work submitted",
            )),
            false => Ok(()),
        }
    }
}
