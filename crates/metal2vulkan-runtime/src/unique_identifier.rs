//! `uniqueIdentifier`, Apple's private property of pipeline states and indirect command buffers,
//! and the identifier an indirect command's `getPipelineStateUniqueIdentifier` answers.
//!
//! Metal's identifiers are addresses of its own allocations: a pipeline state's measured by the
//! consumer on an M3 Max under macOS 26.7, an indirect command buffer's by our probes
//! (an M4 under 27.0.1 and the M3 Max):
//!
//! - **A state made without `supportIndirectCommandBuffers`** answers an address Metal allocates
//!   as any memory, which differs from run to run. What a program can rely on is how they relate:
//!   each state answers one nonzero value for its life, a render and a compute state different
//!   ones, and a second state made from an equal descriptor a new one. The runtime answers a
//!   number of one process-wide count from [`PLAIN_FIRST`], above every value below.
//! - **A state made with it** answers `2^36` ([`INDIRECT_BASE`]) plus an address in Metal's own
//!   allocation of such states, the same in every run that makes and releases the same states in
//!   the same order, which it reuses once a state goes: a compute state made after one was
//!   released answers the released one's, the last released first, and a render state takes
//!   another kind of allocation. The steps are the function's (one kernel's states `0xc0` apart,
//!   the first of another `0x300` on, render states `0x200` apart) and the base what the process
//!   made before and how it was started (`0x100` apart between a test binary and the same binary
//!   from a shell). The runtime keeps the rule, the last released taken again first, at a step
//!   of the smallest measured (`0xc0` compute, `0x200` render); a compute state's from one step
//!   into the range ([`COMPUTE_FIRST`]; it once started from `0x280`, the first such compute state
//!   of one test binary on an M3 Max under macOS 26.7, which the same binary started from a shell
//!   answered `0x100` past, so the base matched one workload, not a rule), up to the indirect
//!   command buffers' range, and past their range once that is full; a render state's past them
//!   (where Metal's lie is not measured). No state's offset is Metal's: Metal's follow its
//!   allocator, its classes by function and the process's history.
//! - **An indirect command buffer** answers `2^36` plus its private `gpuAddress`, in a range of
//!   Metal's for them: the lowest gap from `0x18000` that holds its `allocatedSize` and one 16-KiB
//!   page more, at a multiple of 32 KiB (`0x18000`, `0x20000`, `0x28000` for buffers of four
//!   commands), free again as the buffer goes. The runtime answers that rule. Metal's own work
//!   takes ranges there too, which the runtime does not (on the M4 the process's first blit
//!   encoder that copies, resets or fills takes `0x48000` after the live buffers, and the first
//!   render encoder about `0x298000`; on the M3 Max a first copy or reset about `0x4a8000`, a fill
//!   `0x48000`, a render encoder that makes one resident or draws about `0x4c8000`), so a buffer
//!   made after such work answers below Metal's; and Metal keeps a buffer's range while a command
//!   buffer that executes it may run, where the runtime frees it as the buffer's last holder goes.
//!
//! **Which answers are Metal's**. An indirect command buffer made before its device opened any
//! command encoder answers Metal's number (the rule above; on an M4, on the M3 Max, gaps and
//! releases included): Metal's own encoders take ranges from their first, by kind and by machine
//! (on an M4 and the M3 Max package of the same probe): a compute encoder that dispatches or
//! executes a buffer takes a slot, and one that is empty, makes a buffer resident or is never
//! committed none; a blit encoder that copies takes a range on both, an empty one on the M4 only; a
//! render encoder takes one, never committed too, of another size on each machine. The runtime
//! counts every encoder, as no rule tells the encoders that take nothing on every machine (a
//! resource-state encoder's is not measured). No state's answer is measured to be Metal's. The
//! first answer of a program's call that is not Metal's number
//! (`ComputePipelineState::unique_identifier`, `RenderPipelineState::unique_identifier`,
//! `IndirectCommandBuffer::unique_identifier`, an indirect command's state identifier but 0)
//! announces, once a device, one of two kinds by how Metal's own answer stands: a plain state's,
//! which differs from run to run on Metal, `plain-state-unique-identifier`
//! ([`Degradation::PlainStateUniqueIdentifier`], one of Metal's answers); a state's with indirect
//! command buffer support, or a buffer's made after an encoder, which Metal answers alike in every
//! run of one program started alike, `unmeasured-unique-identifier`
//! ([`Degradation::UnmeasuredUniqueIdentifier`], a near-equivalent). A program that reads only the
//! first buffers' identifiers announces nothing. The allocations and the encoder flag are the
//! device's: Metal's measured space is one process's one device, which a device of the runtime
//! stands for.
//!
//! No run reads an identifier: a translated function never sees one.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::degradation::Degradation;
use crate::device::lock;
use crate::error::{Error, MemoryKind};
use crate::resource_ids::StateKind;
use crate::Device;

/// The base of Metal's identifiers of indirect command buffers and of states made with
/// `supportIndirectCommandBuffers`: `2^36`.
pub(crate) const INDIRECT_BASE: u64 = 1 << 36;

/// The first identifier of a state made without `supportIndirectCommandBuffers`: past every
/// identifier from [`INDIRECT_BASE`].
pub(crate) const PLAIN_FIRST: u64 = 1 << 37;

/// Where the first indirect command buffer lies in Metal's range for them (measured: a fresh
/// process's first buffer answers `0x1000018000`).
pub(crate) const BUFFER_FIRST: u64 = 0x1_8000;

/// Metal's page: a buffer's range holds one more past its `allocatedSize`.
const BUFFER_PAGE: u64 = 0x4000;

/// The alignment of a buffer's range.
const BUFFER_SLOT: u64 = 0x8000;

/// The end of the buffers' range, from [`INDIRECT_BASE`], where the compute states' starts.
pub(crate) const BUFFERS_END: u64 = 1 << 34;

/// Where the first compute state with indirect command buffer support lies: one step into the
/// range, which no rule of Metal's places (the module docs). The `0x280` was one test binary's
/// first on an M3 Max, `0x100` below the same binary's from a shell.
pub(crate) const COMPUTE_FIRST: u64 = COMPUTE_STEP;

/// The compute states' step: the smallest measured.
pub(crate) const COMPUTE_STEP: u64 = 0xc0;

/// The render states' step, as measured.
pub(crate) const RENDER_STEP: u64 = 0x200;

/// The next plain identifier, of every device.
static PLAIN: AtomicU64 = AtomicU64::new(PLAIN_FIRST);

/// One kind of state's identifiers: in steps of `step` through `ranges` in order (each from its
/// first below its end), those given back taken again, the last first.
#[derive(Debug)]
struct Slots {
    step: u64,
    ranges: &'static [(u64, u64)],
    /// The range `next` lies in.
    range: usize,
    next: u64,
    free: Vec<u64>,
}

impl Slots {
    fn new(ranges: &'static [(u64, u64)], step: u64) -> Slots {
        Slots {
            step,
            ranges,
            range: 0,
            next: ranges.first().map_or(0, |&(first, _)| first),
            free: Vec::new(),
        }
    }

    fn take(&mut self) -> Option<u64> {
        if let Some(offset) = self.free.pop() {
            return Some(offset);
        }
        loop {
            let &(_, end) = self.ranges.get(self.range)?;
            if let Some(next) = self.next.checked_add(self.step).filter(|&next| next <= end) {
                let offset = self.next;
                self.next = next;
                return Some(offset);
            }
            self.range += 1;
            self.next = self.ranges.get(self.range)?.0;
        }
    }
}

/// The compute states' ranges: from Metal's first below the buffers' range, then past it.
const COMPUTE_RANGES: &[(u64, u64)] = &[
    (COMPUTE_FIRST, BUFFER_FIRST),
    (BUFFERS_END, 2 * BUFFERS_END),
];

/// The render states' range, past the compute states'.
const RENDER_RANGES: &[(u64, u64)] = &[(2 * BUFFERS_END, 4 * BUFFERS_END)];

/// One device's allocations of the identifiers from [`INDIRECT_BASE`], as offsets from it.
#[derive(Debug)]
struct Allocations {
    compute: Slots,
    render: Slots,
    /// The live indirect command buffers' ranges: start, then bytes.
    buffers: BTreeMap<u64, u64>,
}

/// One device's identifiers: its allocations, and whether it opened a command encoder, from
/// which on Metal's own encoders take ranges of the buffers' (the module docs).
#[derive(Debug)]
pub(crate) struct UniqueIdentifiers {
    allocations: Mutex<Allocations>,
    encoded: AtomicBool,
}

impl Default for UniqueIdentifiers {
    fn default() -> UniqueIdentifiers {
        UniqueIdentifiers {
            allocations: Mutex::new(Allocations {
                compute: Slots::new(COMPUTE_RANGES, COMPUTE_STEP),
                render: Slots::new(RENDER_RANGES, RENDER_STEP),
                buffers: BTreeMap::new(),
            }),
            encoded: AtomicBool::new(false),
        }
    }
}

impl UniqueIdentifiers {
    /// Note that the device opened a command encoder: an indirect command buffer made from now
    /// on answers no number measured to be Metal's.
    pub(crate) fn encoder_opened(&self) {
        // Relaxed: only the flag's value matters, read by the buffers made after the program's
        // call that opened the encoder.
        self.encoded.store(true, Ordering::Relaxed);
    }
}

/// What an identifier from [`INDIRECT_BASE`] was allocated as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Allocation {
    State(StateKind),
    Buffer,
}

impl Allocations {
    fn slots(&mut self, kind: StateKind) -> &mut Slots {
        match kind {
            StateKind::Compute => &mut self.compute,
            StateKind::Render => &mut self.render,
        }
    }

    /// The offset of a new buffer of Metal's `allocated_size`: the lowest gap from
    /// [`BUFFER_FIRST`] that holds it and one page more, at a multiple of the slot.
    fn take_buffer(&mut self, allocated_size: u64) -> Option<u64> {
        let footprint = allocated_size
            .checked_add(BUFFER_PAGE)?
            .checked_next_multiple_of(BUFFER_SLOT)?;
        let mut offset = BUFFER_FIRST;
        for (&start, &bytes) in &self.buffers {
            if start >= offset.checked_add(footprint)? {
                break;
            }
            offset = offset.max(start + bytes);
        }
        if offset.checked_add(footprint)? > BUFFERS_END {
            return None;
        }
        self.buffers.insert(offset, footprint);
        Some(offset)
    }

    fn give(&mut self, allocation: Allocation, offset: u64) {
        match allocation {
            Allocation::State(kind) => self.slots(kind).free.push(offset),
            Allocation::Buffer => {
                self.buffers.remove(&offset);
            }
        }
    }
}

/// A `uniqueIdentifier`, held while its state or buffer lives: one from [`INDIRECT_BASE`] goes
/// back to its device's allocation as the holder goes.
#[derive(Debug)]
pub(crate) struct UniqueIdentifier {
    value: u64,
    /// The kind its answer announces where it is not measured to be Metal's number (the module
    /// docs); none where it is.
    degradation: Option<Degradation>,
    back: Option<(Arc<UniqueIdentifiers>, Allocation)>,
}

impl UniqueIdentifier {
    /// The identifier of a new pipeline state of `kind` on the device of `ids`: from the kind's
    /// allocation where it supports indirect command buffers (`indirect`), else plain. Refused
    /// `OutOfMemory` where the kind's range is full of live states.
    pub(crate) fn state(
        ids: &Arc<UniqueIdentifiers>,
        kind: StateKind,
        indirect: bool,
    ) -> Result<UniqueIdentifier, Error> {
        if !indirect {
            return Ok(UniqueIdentifier {
                value: PLAIN.fetch_add(1, Ordering::Relaxed),
                degradation: Some(Degradation::PlainStateUniqueIdentifier),
                back: None,
            });
        }
        let offset = lock(&ids.allocations)
            .slots(kind)
            .take()
            .ok_or(Error::OutOfMemory(MemoryKind::Device))?;
        Ok(UniqueIdentifier {
            value: INDIRECT_BASE + offset,
            degradation: Some(Degradation::UnmeasuredUniqueIdentifier),
            back: Some((Arc::clone(ids), Allocation::State(kind))),
        })
    }

    /// The identifier of a new indirect command buffer of Metal's `allocated_size` on the device
    /// of `ids`. Refused `OutOfMemory` where no gap of the buffers' range holds it.
    pub(crate) fn buffer(
        ids: &Arc<UniqueIdentifiers>,
        allocated_size: u64,
    ) -> Result<UniqueIdentifier, Error> {
        let offset = lock(&ids.allocations)
            .take_buffer(allocated_size)
            .ok_or(Error::OutOfMemory(MemoryKind::Device))?;
        let encoded = ids.encoded.load(Ordering::Relaxed);
        Ok(UniqueIdentifier {
            value: INDIRECT_BASE + offset,
            degradation: encoded.then_some(Degradation::UnmeasuredUniqueIdentifier),
            back: Some((Arc::clone(ids), Allocation::Buffer)),
        })
    }

    /// Its value.
    pub(crate) fn value(&self) -> u64 {
        self.value
    }

    /// The kind its answer announces where it is not measured to be Metal's number.
    pub(crate) fn degradation(&self) -> Option<Degradation> {
        self.degradation
    }

    /// Its value, as a program's call on `device` answers it: announced where it is not measured
    /// to be Metal's number ([`UniqueIdentifier::degradation`]), `object` naming the holder.
    pub(crate) fn answer(&self, device: &Device, object: &str) -> u64 {
        if let Some(kind) = self.degradation() {
            degrade_identifier(device, kind, object);
        }
        self.value
    }
}

/// The kind an indirect command's answer of a state's identifier `value` announces: none for 0
/// (it names no state, Metal's answer); a plain state's at [`PLAIN_FIRST`] and past it; any other
/// state's below it.
pub(crate) fn state_value_degradation(value: u64) -> Option<Degradation> {
    match value {
        0 => None,
        value if value >= PLAIN_FIRST => Some(Degradation::PlainStateUniqueIdentifier),
        _ => Some(Degradation::UnmeasuredUniqueIdentifier),
    }
}

/// Announce `kind`, once a device, for a program's call that answered an identifier of `object`
/// not measured to be Metal's number (the module docs).
pub(crate) fn degrade_identifier(device: &Device, kind: Degradation, object: &str) {
    device.degrade(kind, || {
        let reason = match kind {
            Degradation::PlainStateUniqueIdentifier => {
                "one value of the runtime's count, unique and fixed for the state's life, where \
                 Metal's is an address that differs from run to run"
            }
            _ => {
                "unique and fixed for its holder's life, and reused as Metal reuses it, but not \
                 the number Metal answers there, which follows how the process was started, \
                 what it made and Metal's own encoders' allocations"
            }
        };
        (
            format!("the uniqueIdentifier of {object}"),
            reason.to_owned(),
        )
    });
}

impl Drop for UniqueIdentifier {
    fn drop(&mut self) {
        if let Some((ids, allocation)) = &self.back {
            lock(&ids.allocations).give(*allocation, self.value - INDIRECT_BASE);
        }
    }
}
