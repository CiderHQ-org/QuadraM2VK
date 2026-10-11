//! Fuel-instrumented AIR for the Metal oracle.
//!
//! The Metal oracle runs a case that is not mechanically loop-free only through AIR this module
//! instruments. The transform works on the decode-level IR (`metal2vulkan_air::ir`) and hands the
//! result to the writer ([`crate::air_writer`]), whose text `metal-as` assembles back into AIR for
//! `metal-opt`/`metallib`. It is validation code and never on a product path.
//!
//! # The model (the fuel model, which the oracle keeps for itself)
//!
//! The entry function gains one device-buffer parameter, the status buffer, in the oracle's own
//! layout ([`status`]), bound at a Metal buffer slot the executor reserves
//! through `NativeBufferSlots`. At its start the entry allocates a thread-private two-word state
//! (`fuel`, `forced`) and loads `fuel` once from the buffer's `invocation_fuel` word. Every defined
//! function that holds a loop, or calls one that does, takes the state and the status buffer as
//! two extra trailing parameters, so calls inside loops draw from the same per-invocation fuel.
//! Every call that passes them loses its `tail` marker: the marker promises that the callee
//! accesses none of the caller's allocas, and the Metal toolchain's optimizer otherwise deletes
//! the state's initialization.
//!
//! - **Loops.** The loop nesting forest of each function (`graph::LoopForest`, reducible and
//!   irreducible) names, for every loop, its header and the edges into the header from inside the
//!   loop. Every cycle of a CFG passes through such an edge of some loop, so a check spliced on each
//!   of them bounds every cycle. On a **non-barrier** loop the check is the saturating fuel check:
//!   at zero take the forced exit, else decrement, and every `2^10` local iterations test the
//!   status buffer's `EXHAUSTED` bit and draw a chunk from the shared pool.
//! - **Counted nests need no check**. A loop whose header visits per entry a constant bounds, read
//!   from its structure (`counted`), ends by itself. The oracle leaves such a loop unchecked when
//!   the product of header visits along every chain of unchecked loops it belongs to, across calls,
//!   stays within [`ORACLE_NEST_LIMIT`]: per entry of its outermost loop, such a chain runs its
//!   innermost body at most that many times, as a full unroll of the nest would. Metal's compiler
//!   does not finish some modules whose fixed-count loops carry checks (the FFT cases), and
//!   finishes them with these left as written.
//! - **Barrier loops** (a loop whose blocks reach a barrier) draw neither fuel nor the pool (an
//!   atomic result is not workgroup-uniform). Each gets a thread-private counter reset to `max(2,
//!   ⌊F_b^(1/d)⌋)` on every edge entering the loop, decremented by its check, where `d` is the
//!   longest chain of nested barrier loops through it across calls. A call the transform cannot see
//!   into (an indirect call whose callee is not closed, [`crate::call_targets`], a declaration that
//!   is not an intrinsic, an unmodeled intrinsic) counts as reaching a barrier: a counter bounds
//!   any loop, and only the barrier case needs the uniform one. An irreducible barrier loop is
//!   refused.
//! - **The forced exit** sets `EXHAUSTED` with an atomic `or`, sets the thread's `forced` word, and
//!   returns from the function holding the loop (a zero value when it returns one). After every
//!   call to an instrumented function the caller tests `forced` and returns the same way, so an
//!   exhausted invocation leaves the entry without running any more of its code, and no caller uses
//!   the zero value a forced return produced.
//!
//! Per invocation, the fuel checks passed are at most `fuel` and barrier-loop iterations at most
//! `F_b` per chain; per status buffer, checked non-barrier iterations are at most
//! `2^10 · (pool_chunks + invocations)`; and between two checks, or before the first, a chain of
//! unchecked counted loops passes its innermost header at most [`ORACLE_NEST_LIMIT`] times per
//! entry. Exhaustion is never evidence: the executor reads the status word before any output and
//! reports [`status::LoopFuelExhausted`].
//!
//! # What is refused (typed, [`FuelUnsupported`])
//!
//! A function that needs the state must be reached only through direct calls from the entry:
//! a loop in an address-taken function, in an initializer the runtime runs itself, or behind a
//! call cycle cannot be threaded and is refused, as is one that makes a `musttail` call (its
//! prototype grows) and a module whose entry is not a `!air.kernel`/`!air.vertex`/`!air.fragment`
//! function. Loops in functions nothing executes (other
//! entry points of the library, dead helpers) are left alone.

mod counted;
mod edit;
mod emit;
mod plan;
pub mod status;
#[cfg(test)]
mod tests;

use std::fmt;

use metal2vulkan::Budget;
use metal2vulkan_air::bitcode::EnumAttrTable;
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::ir::Module;

use crate::air_writer::{decode_as_written, metal_as, write, AssembleError, Pointees, WriteError};

pub use counted::COUNTED_LOOP_LIMIT;
pub use status::{LoopFuelExhausted, DEFAULT_INVOCATION_FUEL};

/// How the Metal oracle bounds one execution of a case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleBound {
    /// The case is mechanically loop-free: the AIR runs as compiled.
    LoopFree,
    /// The AIR runs fuel-instrumented with `invocation_fuel` loop checks per invocation.
    Fuel { invocation_fuel: u32 },
}

/// The compile-time inputs of the transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuelConfig {
    /// The Metal buffer slot (`air.location_index`) of the status buffer argument.
    pub status_slot: u32,
    /// `F_b`, the barrier-loop budget.
    pub barrier_fuel: u32,
    /// Whose loops get a check: [`LoopSelection::Oracle`] for the oracle.
    pub loops: LoopSelection,
}

impl FuelConfig {
    /// The oracle's configuration: the status buffer at `status_slot`, the barrier budget
    /// [`ORACLE_BARRIER_FUEL`], and every loop checked but the counted nests
    /// ([`LoopSelection::Oracle`]).
    pub fn oracle(status_slot: u32) -> FuelConfig {
        FuelConfig {
            status_slot,
            barrier_fuel: ORACLE_BARRIER_FUEL,
            loops: LoopSelection::Oracle,
        }
    }
}

/// The loops the transform checks. The oracle's is [`LoopSelection::Oracle`], which bounds every
/// cycle. [`LoopSelection::All`] bounds every cycle too; it was the oracle's before v48. The
/// others leave loops unbounded, so they exist only for `corpus-metal --compile-probe`, which
/// times pipeline creation of a module and never runs it. The four kinds select among the loops of
/// every function the entry reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopSelection {
    /// Every loop of every function the entry reaches but those in counted nests within
    /// [`ORACLE_NEST_LIMIT`] (the module documentation).
    Oracle,
    /// Every loop of every function the entry reaches.
    All,
    /// Only the entry function's own loops; its callees run as compiled.
    Entry,
    /// Only the loops of the functions the entry calls, directly or not; the entry's own loops run
    /// as compiled, and it still passes the state to the callees that take it.
    Callees,
    /// Only barrier loops (a loop whose blocks reach a barrier, or a call the transform cannot see
    /// into).
    Barrier,
    /// Only the loops that are not barrier loops.
    NonBarrier,
    /// Only counted loops: those whose header visits per entry a constant bounds, at most
    /// [`COUNTED_LOOP_LIMIT`] (the `counted` module).
    Counted,
    /// Only the loops that are not counted.
    Uncounted,
}

/// The oracle's `F_b`: `2^16` barrier-loop iterations along any chain of nested barrier loops.
pub const ORACLE_BARRIER_FUEL: u32 = 1 << 16;

/// The most header visits of its innermost loop a chain of unchecked counted loops may make per
/// entry of its outermost: `2^10`, the chunk, the most fuel checks a thread passes before it looks
/// at the shared `EXHAUSTED` bit again.
pub const ORACLE_NEST_LIMIT: u64 = 1 << 10;

/// What one instrumentation did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FuelReport {
    /// Functions that take the fuel state (the entry included).
    pub functions: u32,
    /// Loops instrumented with the fuel check.
    pub loops: u32,
    /// Loops instrumented with a uniform barrier counter.
    pub barrier_loops: u32,
    /// Calls that pass the state and so lost their `tail` marker.
    pub tail_calls: u32,
    /// Loops left unchecked as counted ([`LoopSelection::Oracle`]).
    pub exempt_loops: u32,
    /// The largest product of header visits along a chain of those loops (0 when there is none).
    pub exempt_nest: u64,
}

/// An instrumented module, the pointees of its functions, and its enum attributes, ready for
/// [`write()`].
pub struct Instrumented {
    pub module: Module,
    pub pointees: Pointees,
    pub enums: EnumAttrTable,
    pub report: FuelReport,
}

/// A structure the transform cannot bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FuelUnsupported {
    /// The module defines no function named as the entry.
    MissingEntry { entry: String },
    /// The entry is not listed in `!air.kernel`, `!air.vertex`, or `!air.fragment` with an
    /// argument list, so the status buffer cannot be declared.
    EntryNotAStage { entry: String },
    /// The entry is also called, or its address is taken, inside the module.
    EntryCalled { entry: String },
    /// The direct call graph reachable from the entry has a cycle through `function`.
    CallCycle { function: String },
    /// `function` needs the fuel state but is reached in a way that cannot pass it: its address
    /// is taken, or the runtime runs it itself (an initializer section or `llvm.global_ctors`).
    UnthreadableLoop { function: String },
    /// An irreducible loop in `function` reaches a barrier; its entry edges cannot reset one
    /// uniform counter.
    IrreducibleBarrierLoop { function: String },
    /// `function` takes the fuel state and makes a `musttail` call. Its prototype grows, so it no
    /// longer matches the callee's as `musttail` requires; and a `musttail` call that passed the
    /// state would promise that the callee accesses none of the caller's allocas, which the state
    /// is, and return before the caller's forced test.
    MustTailCall { function: String },
    /// The module already declares a runtime intrinsic the transform calls with another type.
    IntrinsicSignature { name: String },
}

impl fmt::Display for FuelUnsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FuelUnsupported::MissingEntry { entry } => write!(f, "the module defines no @{entry}"),
            FuelUnsupported::EntryNotAStage { entry } => {
                write!(
                    f,
                    "@{entry} has no stage argument list to add the status buffer to"
                )
            }
            FuelUnsupported::EntryCalled { entry } => {
                write!(f, "@{entry} is called or address-taken inside the module")
            }
            FuelUnsupported::CallCycle { function } => {
                write!(f, "the call graph has a cycle through @{function}")
            }
            FuelUnsupported::UnthreadableLoop { function } => write!(
                f,
                "@{function} needs fuel but is reached other than by direct calls from the entry"
            ),
            FuelUnsupported::IrreducibleBarrierLoop { function } => {
                write!(
                    f,
                    "@{function} has an irreducible loop that reaches a barrier"
                )
            }
            FuelUnsupported::MustTailCall { function } => {
                write!(f, "@{function} needs fuel but makes a musttail call")
            }
            FuelUnsupported::IntrinsicSignature { name } => {
                write!(f, "@{name} is already declared with another type")
            }
        }
    }
}

/// Why fuel-instrumented AIR could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FuelError {
    /// A typed refusal: the structure is outside what the transform bounds.
    Unsupported(FuelUnsupported),
    /// The input bitcode does not decode.
    Decode(String),
    /// The instrumented module could not be written.
    Write(WriteError),
    /// `metal-as` could not assemble the written text.
    Assemble(AssembleError),
    /// The rewrite produced a module the IR verifier refuses (a transform bug).
    Internal(String),
}

impl fmt::Display for FuelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FuelError::Unsupported(u) => write!(f, "oracle fuel unsupported: {u}"),
            FuelError::Decode(e) => write!(f, "oracle fuel: decode: {e}"),
            FuelError::Write(e) => write!(f, "oracle fuel: {e}"),
            FuelError::Assemble(e) => write!(f, "oracle fuel: {e}"),
            FuelError::Internal(e) => write!(f, "oracle fuel internal error: {e}"),
        }
    }
}

impl std::error::Error for FuelError {}

impl From<FuelUnsupported> for FuelError {
    fn from(u: FuelUnsupported) -> FuelError {
        FuelError::Unsupported(u)
    }
}

/// Instrument `module` (with its function pointees and enum attributes) for `entry`; see the
/// module documentation.
pub fn instrument(
    module: Module,
    pointees: Pointees,
    enums: EnumAttrTable,
    entry: &str,
    config: &FuelConfig,
) -> Result<Instrumented, FuelError> {
    let mut meter = Meter::new(&Budget::standard());
    let plan = plan::plan(
        &module,
        entry,
        config.barrier_fuel,
        config.loops,
        &mut meter,
    )?;
    emit::emit(module, pointees, enums, &plan, config, &mut meter)
}

/// Decode typed AIR `bitcode`, instrument it for `entry`, write it, and assemble it with
/// `metal-as`: the AIR the oracle hands to `metal-opt`/`metallib`.
pub fn instrument_air(
    bitcode: &[u8],
    entry: &str,
    config: &FuelConfig,
) -> Result<(Vec<u8>, FuelReport), FuelError> {
    let decoded = decode_as_written(bitcode, &mut Meter::new(&Budget::standard()))
        .map_err(|e| FuelError::Decode(e.to_string()))?;
    let pointees = Pointees::from_table(&decoded.pointees, &decoded.module);
    let instrumented = instrument(decoded.module, pointees, decoded.enum_attrs, entry, config)?;
    let text = write(
        &instrumented.module,
        &instrumented.pointees,
        &instrumented.enums,
    )
    .map_err(FuelError::Write)?;
    let air = metal_as(&text).map_err(FuelError::Assemble)?;
    Ok((air, instrumented.report))
}
