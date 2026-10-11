//! Native library loading, function specialization, and linked-function ownership.

use super::buffers::new_buffer_from_slice;
use super::indirect_commands::NativeCommandResources;
use super::{Buffer, Function, Library, NativeBufferSlots};
use crate::case::{AuthoredCase, ScalarType};
use crate::companion::{Companion, Spelling};
use crate::gpu::compile::{self, xcrun};
use crate::library_module::{LibraryModuleRow, ResolvedLinkedFunctions};
use crate::metal::{OracleSource, OracleTemplate, ShaderAir};
use crate::oracle_fuel::{self, FuelConfig, LoopSelection, OracleBound};
use crate::source::SourceRow;
use crate::ScratchDir;
use base64::Engine as _;
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLDataType, MTLDevice, MTLFunction, MTLFunctionConstantValues, MTLLibrary,
    MTLLinkedFunctions, MTLResource, MTLStorageMode,
};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

pub(super) struct MetalLinkedFunctions {
    pub(super) descriptor: Option<Retained<MTLLinkedFunctions>>,
    functions: HashMap<(String, String), Function>,
}

/// The oracle's status buffer after a fuel-instrumented run, and the buffers the run binds as
/// [`FuelStatus::arm`] saw them before it, for the exhaustion report.
pub(super) struct FuelStatus {
    buffer: Buffer,
    bound: Vec<BoundBuffer>,
}

/// One buffer a fuel-instrumented run binds, with a digest of its contents before the run.
struct BoundBuffer {
    slot: u32,
    buffer: Buffer,
    digest: Option<String>,
}

/// The bytes of a host-readable buffer; `None` for one in private storage.
fn shared_bytes(buffer: &Buffer) -> Option<&[u8]> {
    (buffer.storageMode() != MTLStorageMode::Private).then(|| {
        // Shared and managed storage are host-readable; the caller reads before the command
        // buffer is committed or after it completed.
        unsafe {
            std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), buffer.length())
        }
    })
}

impl FuelStatus {
    /// Record, just before the run is committed, every buffer `buffers` binds with a digest of
    /// its contents, so an exhausted run can say which of them it wrote.
    pub(super) fn arm(&mut self, buffers: &[(u32, Buffer)]) {
        self.bound = buffers
            .iter()
            .map(|(slot, buffer)| BoundBuffer {
                slot: *slot,
                buffer: buffer.clone(),
                digest: shared_bytes(buffer).map(crate::hash::sha256_bytes),
            })
            .collect();
    }

    /// Each armed buffer's slot, GPU address range, and whether the run changed its contents:
    /// where an exhausted run's writes landed, against the status buffer's own address.
    fn bound_report(&self) -> String {
        let mut report = format!("status buffer gpu {:#x}", self.buffer.gpuAddress());
        for bound in &self.bound {
            let changed = match (&bound.digest, shared_bytes(&bound.buffer)) {
                (Some(before), Some(after)) => {
                    if *before == crate::hash::sha256_bytes(after) {
                        "unchanged"
                    } else {
                        "changed"
                    }
                }
                _ => "unread",
            };
            report.push_str(&format!(
                "; slot {} gpu {:#x}+{} {changed}",
                bound.slot,
                bound.buffer.gpuAddress(),
                bound.buffer.length()
            ));
        }
        report
    }
}

/// Read the status buffer after the command buffer completed, before any output: exhaustion
/// means the outputs are not Metal's semantics. The exhaustion error names every
/// status word and every armed buffer ([`FuelStatus::arm`]).
pub(super) fn check_fuel(status: Option<&FuelStatus>) -> Result<(), String> {
    let Some(status) = status else {
        return Ok(());
    };
    let bytes = shared_bytes(&status.buffer)
        .ok_or_else(|| "the oracle fuel status buffer is not host-readable".to_string())?;
    oracle_fuel::status::check_status(bytes)?
        .map_err(|exhausted| format!("{exhausted}; {}", status.bound_report()))
}

/// Arm `status` (when the run is fuel-instrumented) with the buffers it binds.
pub(super) fn arm_fuel(status: Option<&mut FuelStatus>, buffers: &[(u32, Buffer)]) {
    if let Some(status) = status {
        status.arm(buffers);
    }
}

/// Load the case's own library from the AIR the oracle compiles for it. The indirect-command
/// guard and, under [`OracleBound::Fuel`], the fuel status buffer take native buffer slots no case
/// resource uses; both buffers join `buffers`, so they are bound with the case's own.
pub(super) fn load_producer_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &OracleSource<'_>,
    commands: &NativeCommandResources,
    interface: &crate::check::interface::AirInterface,
    buffers: &mut Vec<(u32, Buffer)>,
    reserved: &[u32],
) -> Result<(Library, Option<FuelStatus>), String> {
    if commands.policy.is_none() && source.bound == OracleBound::LoopFree {
        return Ok((load_library(device, &source.air, None)?, None));
    }
    let mut occupied = NativeBufferSlots::from_interface(interface)?;
    for index in reserved
        .iter()
        .copied()
        .chain(buffers.iter().map(|&(index, _)| index))
    {
        occupied.reserve(index, 1)?;
    }
    let slots = producer_slots(occupied, commands.policy.is_some(), source.bound)?;
    let library = producer_source(
        &source.air,
        &source.row.entry,
        slots.fuel,
        LoopSelection::Oracle,
    )?;
    let guard = match (&commands.policy, slots.guard) {
        (Some((policy, buffer)), Some(slot)) => Some((policy.registration_count(), slot, buffer)),
        _ => None,
    };
    let fuel = match (source.bound, slots.fuel) {
        (OracleBound::Fuel { invocation_fuel }, Some(slot)) => {
            let initial = oracle_fuel::status::initial_status(invocation_fuel);
            let buffer = new_buffer_from_slice(device, &initial, "oracle fuel status")?;
            Some((slot, buffer))
        }
        _ => None,
    };
    let library = load_verified_library(
        device,
        &library.air,
        guard.map(|(count, binding, _)| (count, binding)),
        library.verification,
    )?;
    if let Some((_, binding, buffer)) = guard {
        buffers.push((binding, buffer.clone()));
    }
    Ok((
        library,
        fuel.map(|(slot, buffer)| {
            buffers.push((slot, buffer.clone()));
            FuelStatus {
                buffer,
                bound: Vec::new(),
            }
        }),
    ))
}

/// The native buffer slots the producer library adds beyond the case's own.
#[derive(Debug, PartialEq, Eq)]
struct ProducerSlots {
    /// The indirect-command guard's slot, when the case has command buffers.
    guard: Option<u32>,
    /// The fuel status buffer's slot, under [`OracleBound::Fuel`].
    fuel: Option<u32>,
}

/// Allocate the guard slot, then the fuel status slot, from the slots `slots` leaves free: the
/// shared buffer-slot rule (`executor_contract::buffer_slots`) over the entry's interface and the
/// case's buffers.
/// The case's own library for the compile probe: as compiled (`loops` `None`), or fuel-instrumented
/// as the oracle instruments it with only the selected `loops` checked, its status buffer at the
/// slot the oracle gives it. Returns the library and what the transform did. A probe never runs
/// what it loads (an unselected loop is unbounded).
pub(super) fn load_probe_library(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    source: &OracleSource<'_>,
    interface: &crate::check::interface::AirInterface,
    loops: Option<LoopSelection>,
) -> Result<(Library, Option<oracle_fuel::FuelReport>), String> {
    let Some(loops) = loops else {
        return Ok((load_library(device, &source.air, None)?, None));
    };
    let bound = OracleBound::Fuel {
        invocation_fuel: oracle_fuel::DEFAULT_INVOCATION_FUEL,
    };
    let slots = producer_slots(NativeBufferSlots::for_case(case, interface)?, false, bound)?;
    let library = producer_source(&source.air, &source.row.entry, slots.fuel, loops)?;
    let loaded = load_verified_library(device, &library.air, None, library.verification)?;
    Ok((loaded, library.fuel))
}

fn producer_slots(
    mut slots: NativeBufferSlots,
    guard: bool,
    bound: OracleBound,
) -> Result<ProducerSlots, String> {
    let guard = if guard { Some(slots.allocate()?) } else { None };
    let fuel = match bound {
        OracleBound::LoopFree => None,
        OracleBound::Fuel { .. } => Some(slots.allocate()?),
    };
    Ok(ProducerSlots { guard, fuel })
}

/// The library the oracle compiles for one case.
struct ProducerLibrary<'a> {
    air: Cow<'a, [u8]>,
    /// The level `metal-opt` verifies `air` at.
    verification: Verification,
    /// The fuel transform's report, when `air` is fuel-instrumented.
    fuel: Option<oracle_fuel::FuelReport>,
}

/// The library the oracle compiles for the case's `air`: as compiled, or, with a status slot,
/// fuel-instrumented for `entry` with its status buffer at that slot and the `loops` checked (the
/// oracle's selection, or a compile probe's). Instrumented AIR must pass `metal-opt` at the
/// strictest level the case's own AIR passes, so the transform and the writer are held to
/// everything the toolchain checks of the input.
fn producer_source<'a>(
    air: &'a [u8],
    entry: &str,
    fuel_slot: Option<u32>,
    loops: LoopSelection,
) -> Result<ProducerLibrary<'a>, String> {
    let Some(slot) = fuel_slot else {
        return Ok(ProducerLibrary {
            air: Cow::Borrowed(air),
            verification: Verification::Strictest,
            fuel: None,
        });
    };
    let level = {
        let scratch = ScratchDir::new("metal-fuel-verify")?;
        let input = scratch.path().join("input.air");
        fs::write(&input, air).map_err(|error| format!("write native AIR: {error}"))?;
        verify_air(
            &input,
            &scratch.path().join("input.ll"),
            Verification::Strictest,
        )?
    };
    let config = FuelConfig {
        loops,
        ..FuelConfig::oracle(slot)
    };
    let (instrumented, report) =
        oracle_fuel::instrument_air(air, entry, &config).map_err(|error| error.to_string())?;
    Ok(ProducerLibrary {
        air: Cow::Owned(instrumented),
        verification: Verification::At(level),
        fuel: Some(report),
    })
}

/// A `metal-opt` AIR verification level (`--air-verification-level`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AirLevel {
    /// Every check, including each AIR intrinsic declaration against the toolchain's table.
    Full,
    /// The compiler's checks, without the intrinsic-table comparison.
    Compiler,
}

impl AirLevel {
    fn flag(self) -> &'static str {
        match self {
            AirLevel::Full => "--air-verification-level=full",
            AirLevel::Compiler => "--air-verification-level=compiler",
        }
    }
}

/// How strictly `metal-opt` verifies a library before it is packaged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verification {
    /// AIR as compiled: the strictest level it passes. `full` compares every AIR intrinsic
    /// declaration with the toolchain's own table, which lacks forms the Metal compiler itself
    /// emits (`air.simdgroup_matrix_16x16x16_*`, 8x8 fp8 `air.simdgroup_matrix_*`,
    /// `air.wait_simdgroup_events` in Q2's cases), so AIR as compiled may pass only `compiler`.
    Strictest,
    /// Exactly this level: the strictest the input of a transform passed.
    At(AirLevel),
}

/// Verify `air` with `metal-opt`, writing its text to `typed`; the level it passed.
fn verify_air(air: &Path, typed: &Path, verification: Verification) -> Result<AirLevel, String> {
    let verify = |level: AirLevel| {
        native_tool(
            "metal-opt",
            &["-S", "-passes=verify", level.flag()],
            air,
            typed,
        )
    };
    match verification {
        Verification::At(level) => verify(level).map(|()| level),
        Verification::Strictest => match verify(AirLevel::Full) {
            Ok(()) => Ok(AirLevel::Full),
            Err(full) => verify(AirLevel::Compiler)
                .map(|()| AirLevel::Compiler)
                .map_err(|compiler| format!("{full}\n{compiler}")),
        },
    }
}

/// What the oracle hands to Metal for one case, built without a device.
#[derive(Debug)]
pub(crate) struct PreparedLibrary {
    /// The bound the case's compiled module runs under.
    pub(crate) bound: OracleBound,
    /// The fuel transform's report, under [`OracleBound::Fuel`].
    pub(crate) fuel: Option<oracle_fuel::FuelReport>,
    /// The native buffer slot of the fuel status buffer, under [`OracleBound::Fuel`].
    pub(crate) status_slot: Option<u32>,
    /// The case's own packaged metallib's size.
    pub(crate) metallib_bytes: u64,
    /// The SHA-256 of every metallib built, in build order, the case's own first.
    pub(crate) library_sha256: Vec<String>,
}

/// The CPU-side part of [`load_producer_library`]: the same slot allocation, by the shared rule
/// over the entry's interface and the case (`NativeBufferSlots::for_case`, whose buffers stand for
/// the ones the executor uploads), the same fuel transform, the same indirect-command guard,
/// `metal-opt`, and `metallib`, stopping before a device opens. The guard's code depends only on
/// its slot and the registration count, one registration per authored command buffer (the
/// executor registers each and refuses a duplicate); the device's resource ids are the policy
/// table's contents, which the run uploads.
pub(crate) fn prepare_producer_library(
    case: &AuthoredCase,
    source: &OracleSource<'_>,
    interface: &crate::check::interface::AirInterface,
    reserved: &[u32],
) -> Result<PreparedLibrary, String> {
    let guard = !case.command_buffers.is_empty();
    let registrations = u32::try_from(case.command_buffers.len())
        .map_err(|_| "the case authors more command buffers than a policy table registers")?;
    let slots = if guard || source.bound != OracleBound::LoopFree {
        let mut occupied = NativeBufferSlots::for_case(case, interface)?;
        for &slot in reserved {
            occupied.reserve(slot, 1)?;
        }
        producer_slots(occupied, guard, source.bound)?
    } else {
        ProducerSlots {
            guard: None,
            fuel: None,
        }
    };
    let library = producer_source(
        &source.air,
        &source.row.entry,
        slots.fuel,
        LoopSelection::Oracle,
    )?;
    let guard = slots.guard.map(|slot| (registrations, slot));
    let (_scratch, metallib) = build_library(&library.air, guard, library.verification)?;
    let metallib_bytes = fs::metadata(&metallib)
        .map_err(|error| format!("stat {}: {error}", metallib.display()))?
        .len();
    Ok(PreparedLibrary {
        bound: source.bound,
        fuel: library.fuel,
        status_slot: slots.fuel,
        metallib_bytes,
        library_sha256: vec![metallib_sha256(&metallib)?],
    })
}

/// The SHA-256 of the metallib at `path`.
fn metallib_sha256(path: &Path) -> Result<String, String> {
    crate::hash::sha256_file(path).map_err(|error| format!("read {}: {error}", path.display()))
}

/// The CPU-side part of every other library the oracle loads for a case: its generated
/// `companion`'s, each pipeline template's, as compiled, with its companion's, and each distinct
/// linked module's, of the case and of its templates ([`load_linked_functions`]). Returns the
/// SHA-256 of each metallib it built, in build order.
pub(crate) fn prepare_dependency_libraries(
    companion: Option<&Companion>,
    linked: &ResolvedLinkedFunctions,
    templates: &BTreeMap<String, OracleTemplate<'_>>,
) -> Result<Vec<String>, String> {
    let mut built = Vec::new();
    let mut build = |air: &[u8]| -> Result<(), String> {
        let (_scratch, metallib) = build_library(air, None, Verification::Strictest)?;
        built.push(metallib_sha256(&metallib)?);
        Ok(())
    };
    let companions = companion.into_iter().chain(
        templates
            .values()
            .filter_map(|template| template.shader.companion.as_ref()),
    );
    for companion in companions {
        build(&companion_air(companion)?)?;
    }
    for template in templates.values() {
        build(&template.air)?;
    }
    let mut modules = HashSet::new();
    let tables = std::iter::once(linked).chain(
        templates
            .values()
            .map(|template| &template.shader.linked_functions),
    );
    for tables in tables {
        for library in tables.libraries() {
            if modules.insert(&library.module.module_sha256) {
                build(&linked_air(library.module)?)?;
            }
        }
    }
    Ok(built)
}

/// The AIR bitcode the oracle compiles for `row`. A public fixture is compiled on this host from
/// its typed AIR text (`metal-as`) or, without one, its Metal source (`metal -c`); any other row
/// loads its retained blob, the module the checker classified.
pub(crate) fn shader_air(row: &SourceRow) -> Result<ShaderAir, String> {
    if let Some(bitcode) = public_air(&row.label) {
        return Ok(ShaderAir {
            bitcode: bitcode?,
            checked: false,
        });
    }
    let blob = row
        .blob_b64
        .as_deref()
        .ok_or("native source requires original typed AIR or a public fixture")?;
    let bitcode = base64::engine::general_purpose::STANDARD
        .decode(blob)
        .map_err(|error| format!("decode native AIR: {error}"))?;
    Ok(ShaderAir {
        bitcode,
        checked: true,
    })
}

/// The AIR bitcode the oracle compiles for a generated companion: its typed spelling assembled by
/// this host's `metal-as`. The library it loads is then verified and packaged as the case's own is
/// ([`load_library`]).
pub(crate) fn companion_air(companion: &Companion) -> Result<Vec<u8>, String> {
    let scratch = ScratchDir::new("metal-companion")?;
    let text = scratch.path().join("companion.ll");
    fs::write(&text, companion.air_text(Spelling::Typed))
        .map_err(|error| format!("write the generated companion: {error}"))?;
    let air = scratch.path().join("companion.air");
    native_tool("metal-as", &[], &text, &air)?;
    fs::read(&air).map_err(|error| format!("read the generated companion's AIR: {error}"))
}

/// The AIR bitcode of the public module labelled `label` (`public/...`), compiled on this host
/// from its typed AIR text (`metal-as`) or, without one, its Metal source (`metal -c`); `None`
/// for a label that is not a public module's.
fn public_air(label: &str) -> Option<Result<Vec<u8>, String>> {
    let (input, typed) = crate::source::public_metal_input(label)?;
    Some(native_air(&input, typed))
}

/// `input` compiled on this host: typed AIR text with `metal-as`, Metal source with `metal -c`.
fn native_air(input: &Path, typed: bool) -> Result<Vec<u8>, String> {
    let (tool, flags): (_, &[&str]) = if typed {
        ("metal-as", &[])
    } else {
        ("metal", &["-c"])
    };
    let scratch = ScratchDir::new("metal-fixture")?;
    let air = scratch.path().join("fixture.air");
    native_tool(tool, flags, input, &air)?;
    fs::read(&air).map_err(|error| format!("read native AIR: {error}"))
}

/// A linked module's AIR: its retained blob, or a public linked module's typed text assembled
/// on this host, as a public source row's is.
fn linked_air(module: &LibraryModuleRow) -> Result<Vec<u8>, String> {
    if module.is_public() {
        return public_air(&module.label).unwrap_or_else(|| {
            Err(format!(
                "public linked module {} names no public fixture",
                module.label
            ))
        });
    }
    base64::engine::general_purpose::STANDARD
        .decode(&module.blob_b64)
        .map_err(|error| format!("decode linked AIR module {}: {error}", module.module_sha256))
}

/// One host-tool invocation, by what decides its result: the tool, its flags, and its input's
/// bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ToolCall {
    tool: String,
    flags: Vec<String>,
    input_sha256: String,
}

/// What a finished host tool left: its output file's bytes, or why it failed.
type ToolResult = Result<Vec<u8>, String>;

thread_local! {
    /// The host-tool results of the case running on this thread, while [`reusing_tool_results`]
    /// holds them (`None` otherwise).
    static TOOL_RESULTS: RefCell<Option<HashMap<ToolCall, ToolResult>>> =
        const { RefCell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// How many host tools [`native_tool`] ran on this thread.
    static TOOL_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Run `body` with every host tool it runs on this thread run once per distinct invocation (the
/// same tool, flags, and input bytes): a repeat writes the first one's output, or fails as it did,
/// without running the tool again. A case's three qualification executions each load the same
/// libraries (its own, its companion's, its templates', and its linked modules'), so each is
/// built once per case, not once per execution. The tools are deterministic in their input, so a
/// reused result is the one a rerun would give. Results are held only while `body` runs, and only
/// those of its own case; a call inside another one shares the outer one's.
pub(crate) fn reusing_tool_results<R>(body: impl FnOnce() -> R) -> R {
    /// Ends the results this call opened, also when `body` unwinds.
    struct Close(bool);
    impl Drop for Close {
        fn drop(&mut self) {
            if self.0 {
                TOOL_RESULTS.with(|results| *results.borrow_mut() = None);
            }
        }
    }
    let opened = TOOL_RESULTS.with(|results| {
        let mut results = results.borrow_mut();
        let opened = results.is_none();
        if opened {
            *results = Some(HashMap::new());
        }
        opened
    });
    let _close = Close(opened);
    body()
}

/// Run `xcrun <tool> <flags> <input> -o <output>` under the compile deadline
/// ([`crate::gpu::compile`]), or, inside [`reusing_tool_results`], write the result of the same
/// invocation that already ran.
fn native_tool(tool: &str, flags: &[&str], input: &Path, output: &Path) -> Result<(), String> {
    let call = match TOOL_RESULTS.with(|results| results.borrow().is_some()) {
        true => Some(ToolCall {
            tool: tool.into(),
            flags: flags.iter().map(|flag| flag.to_string()).collect(),
            input_sha256: crate::hash::sha256_bytes(
                &fs::read(input)
                    .map_err(|error| format!("read {tool}'s input {}: {error}", input.display()))?,
            ),
        }),
        false => None,
    };
    if let Some(call) = &call {
        let reused = TOOL_RESULTS.with(|results| {
            results
                .borrow()
                .as_ref()
                .and_then(|results| results.get(call))
                .map(|result| match result {
                    Ok(bytes) => fs::write(output, bytes).map_err(|error| {
                        format!("write {tool}'s output {}: {error}", output.display())
                    }),
                    Err(failure) => Err(failure.clone()),
                })
        });
        if let Some(reused) = reused {
            return reused;
        }
    }
    let args = flags.iter().map(OsStr::new).chain([
        input.as_os_str(),
        OsStr::new("-o"),
        output.as_os_str(),
    ]);
    // A tool that could not run, or that outlasted its deadline, is no result of its input: it
    // fails the case and is never reused.
    #[cfg(test)]
    TOOL_RUNS.with(|runs| runs.set(runs.get() + 1));
    let result = xcrun(tool, args).map_err(|error| error.to_string())?;
    let finished = if result.success {
        Ok(())
    } else {
        Err(format!("{tool} failed: {}{}", result.stdout, result.stderr))
    };
    if let Some(call) = call {
        let kept = match &finished {
            Ok(()) => Ok(fs::read(output)
                .map_err(|error| format!("read {tool}'s output {}: {error}", output.display()))?),
            Err(failure) => Err(failure.clone()),
        };
        TOOL_RESULTS.with(|results| {
            if let Some(results) = results.borrow_mut().as_mut() {
                results.insert(call, kept);
            }
        });
    }
    finished
}

/// Load AIR bitcode as a native library, verified at the strictest level it passes, with the
/// indirect-command guard applied when `guard` names one.
pub(crate) fn load_library(
    device: &ProtocolObject<dyn MTLDevice>,
    air: &[u8],
    guard: Option<(u32, u32)>,
) -> Result<Library, String> {
    load_verified_library(device, air, guard, Verification::Strictest)
}

fn load_verified_library(
    device: &ProtocolObject<dyn MTLDevice>,
    air: &[u8],
    guard: Option<(u32, u32)>,
    verification: Verification,
) -> Result<Library, String> {
    let (_scratch, metallib) = build_library(air, guard, verification)?;
    let path = NSString::from_str(metallib.to_str().ok_or("non-UTF-8 native library path")?);
    device
        .newLibraryWithURL_error(&NSURL::fileURLWithPath(&path))
        .map_err(|error| format!("load native library: {error}"))
}

/// The CPU-side part of [`load_library`]: verify the AIR with `metal-opt`, apply the
/// indirect-command guard, and package it with `metallib`. No device is involved. The returned
/// scratch directory holds the metallib and is removed when dropped.
fn build_library(
    air: &[u8],
    guard: Option<(u32, u32)>,
    verification: Verification,
) -> Result<(ScratchDir, PathBuf), String> {
    let scratch = ScratchDir::new("metal-library")?;
    let input = scratch.path().join("input.air");
    fs::write(&input, air).map_err(|error| format!("write native AIR: {error}"))?;
    let typed = scratch.path().join("typed.ll");
    verify_air(&input, &typed, verification)?;
    {
        let source =
            fs::read_to_string(&typed).map_err(|error| format!("read native AIR: {error}"))?;
        let guarded = crate::metal::command_guard::native_guarded_source(&source, guard)?;
        if matches!(guarded, std::borrow::Cow::Owned(_)) {
            fs::write(&typed, guarded.as_bytes())
                .map_err(|error| format!("write guarded native AIR: {error}"))?;
            native_tool("metal-as", &[], &typed, &input)?;
        }
    }
    let metallib = scratch.path().join("library.metallib");
    native_tool("metallib", &[], &input, &metallib)?;
    Ok((scratch, metallib))
}

/// The `MTLLinkedFunctions` a stage linking `tables` takes: every function of
/// [`ResolvedLinkedFunctions::libraries`] in `functions`, each specialized with the case's
/// `constants` (`newFunctionWithName:constantValues:`, which takes the values of the constants the
/// function declares and ignores the others), and each by `(module, name)` for the function
/// tables.
pub(super) fn load_linked_functions(
    device: &ProtocolObject<dyn MTLDevice>,
    tables: &ResolvedLinkedFunctions,
    constants: &[crate::literal::LiteralFunctionConstant],
) -> Result<MetalLinkedFunctions, String> {
    if tables.is_empty() {
        return Ok(MetalLinkedFunctions {
            descriptor: None,
            functions: HashMap::new(),
        });
    }
    // The one grouping both executors link by: each module loaded once, each of its linked
    // functions specialized with the case's constants.
    let mut functions = HashMap::<(String, String), Function>::new();
    for linked in tables.libraries() {
        let module = linked.module;
        let library = load_library(device, &linked_air(module)?, None)?;
        for function_name in linked.functions {
            let name = NSString::from_str(function_name);
            let function = make_function(&library, &name, constants).map_err(|error| {
                format!(
                    "load linked function {:?} from module {}: {error}",
                    function_name, module.module_sha256
                )
            })?;
            functions.insert(
                (module.module_sha256.clone(), function_name.to_string()),
                function,
            );
        }
    }
    let linked_array =
        NSArray::from_retained_slice(&functions.values().cloned().collect::<Vec<Function>>());
    let descriptor = MTLLinkedFunctions::linkedFunctions();
    descriptor.setFunctions(Some(&linked_array));
    Ok(MetalLinkedFunctions {
        descriptor: Some(descriptor),
        functions,
    })
}

pub(super) fn linked_function<'a>(
    linked: &'a MetalLinkedFunctions,
    entry: &crate::library_module::ResolvedFunctionEntry,
) -> Result<&'a ProtocolObject<dyn MTLFunction>, String> {
    linked
        .functions
        .get(&(entry.module.module_sha256.clone(), entry.function.clone()))
        .map(|function| &**function)
        .ok_or_else(|| {
            format!(
                "linked function {:?} from module {} was not loaded",
                entry.function, entry.module.module_sha256
            )
        })
}

pub(super) fn make_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &NSString,
    constants: &[crate::literal::LiteralFunctionConstant],
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLFunction>>, String> {
    let values = MTLFunctionConstantValues::new();
    for constant in constants {
        let pointer = NonNull::new(constant.bytes.as_ptr().cast_mut().cast::<c_void>())
            .ok_or_else(|| format!("function constant {} pointer is null", constant.index))?;
        unsafe {
            values.setConstantValue_type_atIndex(
                pointer,
                metal_data_type(constant.scalar_type, constant.lanes)?,
                constant.index as usize,
            );
        }
    }
    compile::bounded("function specialization", || {
        library.newFunctionWithName_constantValues_error(entry, &values)
    })
    .map_err(|error| format!("specialize Metal function {entry:?}: {error}"))
}

fn metal_data_type(scalar_type: ScalarType, lanes: u32) -> Result<MTLDataType, String> {
    let types = match scalar_type {
        ScalarType::Bool => [
            MTLDataType::Bool,
            MTLDataType::Bool2,
            MTLDataType::Bool3,
            MTLDataType::Bool4,
        ],
        ScalarType::U8 => [
            MTLDataType::UChar,
            MTLDataType::UChar2,
            MTLDataType::UChar3,
            MTLDataType::UChar4,
        ],
        ScalarType::I8 => [
            MTLDataType::Char,
            MTLDataType::Char2,
            MTLDataType::Char3,
            MTLDataType::Char4,
        ],
        ScalarType::U16 => [
            MTLDataType::UShort,
            MTLDataType::UShort2,
            MTLDataType::UShort3,
            MTLDataType::UShort4,
        ],
        ScalarType::I16 => [
            MTLDataType::Short,
            MTLDataType::Short2,
            MTLDataType::Short3,
            MTLDataType::Short4,
        ],
        ScalarType::F16 => [
            MTLDataType::Half,
            MTLDataType::Half2,
            MTLDataType::Half3,
            MTLDataType::Half4,
        ],
        ScalarType::U32 => [
            MTLDataType::UInt,
            MTLDataType::UInt2,
            MTLDataType::UInt3,
            MTLDataType::UInt4,
        ],
        ScalarType::I32 => [
            MTLDataType::Int,
            MTLDataType::Int2,
            MTLDataType::Int3,
            MTLDataType::Int4,
        ],
        ScalarType::F32 => [
            MTLDataType::Float,
            MTLDataType::Float2,
            MTLDataType::Float3,
            MTLDataType::Float4,
        ],
        ScalarType::U64 => [
            MTLDataType::ULong,
            MTLDataType::ULong2,
            MTLDataType::ULong3,
            MTLDataType::ULong4,
        ],
        ScalarType::I64 => [
            MTLDataType::Long,
            MTLDataType::Long2,
            MTLDataType::Long3,
            MTLDataType::Long4,
        ],
        ScalarType::F64 => return Err("Metal has no double function-constant type".into()),
    };
    lanes
        .checked_sub(1)
        .and_then(|index| types.get(index as usize))
        .copied()
        .ok_or_else(|| format!("Metal function-constant lane count {lanes} is invalid"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::CheckedShader;
    use crate::library_module::ResolvedFunctionReference;
    use std::sync::Arc;

    fn have_xcrun(test: &str) -> bool {
        let ok = compile::has_tool("metal-opt");
        if !ok {
            eprintln!("SKIP {test}: xcrun metal-opt not found");
        }
        ok
    }

    /// `kernel` compiled to AIR text with `xcrun metal -S -emit-llvm`.
    fn compiled_text(scratch: &ScratchDir, kernel: &str) -> String {
        let source = scratch.path().join("k.metal");
        let text = scratch.path().join("k.ll");
        fs::write(
            &source,
            format!("#include <metal_stdlib>\nusing namespace metal;\n{kernel}"),
        )
        .unwrap();
        let args = ["-S", "-emit-llvm", "-c"]
            .map(OsStr::new)
            .into_iter()
            .chain([source.as_os_str(), OsStr::new("-o"), text.as_os_str()]);
        let result = xcrun("metal", args).unwrap();
        assert!(result.success, "{}", result.stderr);
        fs::read_to_string(&text).unwrap()
    }

    fn assemble(scratch: &ScratchDir, name: &str, text: &str) -> std::path::PathBuf {
        let air = scratch.path().join(format!("{name}.air"));
        fs::write(&air, crate::air_writer::metal_as(text).unwrap()).unwrap();
        air
    }

    const KERNEL: &str = "kernel void k(device uint* out [[buffer(0)]], \
                          const device uint* n [[buffer(1)]]) { out[0] = n[0] + 1; }\n";

    /// A public linked module retains no blob: the oracle links its typed `.air.ll` assembled with
    /// `metal-as` on this host, which builds as a linked library does
    /// ([`prepare_dependency_libraries`]).
    #[test]
    fn a_public_linked_module_links_its_typed_text_assembled_here() {
        if !have_xcrun("a_public_linked_module_links_its_typed_text_assembled_here") {
            return;
        }
        let modules = crate::library_module::public_library_modules().unwrap();
        assert!(!modules.is_empty());
        for module in modules {
            let air = linked_air(&module).unwrap();
            let (typed, is_typed) = crate::source::public_metal_input(&module.label).unwrap();
            assert!(is_typed, "{}", module.label);
            assert_eq!(
                air,
                crate::air_writer::metal_as(&fs::read_to_string(typed).unwrap()).unwrap()
            );
            build_library(&air, None, Verification::Strictest).unwrap();
        }
    }

    /// A case's three executions load the same libraries: inside [`reusing_tool_results`] each
    /// distinct tool invocation runs once, a repeat writes its output (or fails as it did), a
    /// nested call shares the outer results, and nothing is held after the call.
    #[test]
    fn a_case_builds_each_library_once_for_all_its_executions() {
        if !have_xcrun("a_case_builds_each_library_once_for_all_its_executions") {
            return;
        }
        let scratch = ScratchDir::new("metal-tool-reuse").unwrap();
        let air = fs::read(assemble(
            &scratch,
            "plain",
            &compiled_text(&scratch, KERNEL),
        ))
        .unwrap();
        let runs = || TOOL_RUNS.with(std::cell::Cell::get);
        let built = |air: &[u8]| {
            let (_scratch, metallib) = build_library(air, None, Verification::Strictest).unwrap();
            fs::read(metallib).unwrap()
        };
        let failed = || match build_library(b"not AIR", None, Verification::Strictest) {
            Ok(_) => panic!("metal-opt verified bytes that are not AIR"),
            Err(failure) => failure,
        };

        let start = runs();
        built(&air);
        let per_build = runs() - start;
        assert!(
            per_build >= 2,
            "metal-opt and metallib at least: {per_build}"
        );
        built(&air);
        assert_eq!(
            runs() - start,
            2 * per_build,
            "outside the call every build runs"
        );

        reusing_tool_results(|| {
            let start = runs();
            let once = built(&air);
            assert_eq!(runs() - start, per_build);
            assert_eq!(built(&air), once, "a repeat writes the first output");
            assert_eq!(runs() - start, per_build, "a repeat runs no tool");
            let failure = failed();
            let after_failure = runs();
            assert!(after_failure > start + per_build);
            assert_eq!(failed(), failure, "a failure is reused as it was");
            assert_eq!(runs(), after_failure);
            reusing_tool_results(|| assert_eq!(built(&air), once));
            assert_eq!(
                runs(),
                after_failure,
                "a nested call shares the outer results"
            );
        });

        let start = runs();
        built(&air);
        assert_eq!(runs() - start, per_build, "nothing is held after the call");
        assert!(TOOL_RESULTS.with(|results| results.borrow().is_none()));
    }

    /// `full` compares every AIR intrinsic declaration with the toolchain's own table, which
    /// lacks forms the Metal compiler emits (Q2: `air.wait_simdgroup_events`,
    /// `air.simdgroup_matrix_16x16x16_*`). AIR as compiled is verified at the strictest level it
    /// passes, and instrumented AIR at exactly its input's level, so `full` still holds the
    /// transform's own declarations to the table whenever the input passes it.
    #[test]
    fn verification_takes_the_strictest_level_the_input_passes() {
        if !have_xcrun("verification_takes_the_strictest_level_the_input_passes") {
            return;
        }
        let scratch = ScratchDir::new("metal-verify-levels").unwrap();
        let text = compiled_text(&scratch, KERNEL);
        let plain = assemble(&scratch, "plain", &text);
        let typed = scratch.path().join("typed.ll");
        assert_eq!(
            verify_air(&plain, &typed, Verification::Strictest),
            Ok(AirLevel::Full)
        );
        let events = text.replacen(
            "define ",
            "%struct._simdgroup_event_t = type opaque\n\
             declare void @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t**)\n\
             define void @wait(%struct._simdgroup_event_t** %e) {\n  \
             call void @air.wait_simdgroup_events(i32 1, %struct._simdgroup_event_t** %e)\n  \
             ret void\n}\n\
             define ",
            1,
        );
        let events = assemble(&scratch, "events", &events);
        assert_eq!(
            verify_air(&events, &typed, Verification::Strictest),
            Ok(AirLevel::Compiler)
        );
        assert!(verify_air(&events, &typed, Verification::At(AirLevel::Full)).is_err());
        // What `full` exists for here: an atomic declared at the wrong pointee.
        let atomic = text.replacen(
            "define ",
            "declare i32 @air.atomic.global.load.i32(i8 addrspace(1)*, i32, i32, i1)\n\
             define i32 @peek(i8 addrspace(1)* %p) {\n  \
             %v = call i32 @air.atomic.global.load.i32(i8 addrspace(1)* %p, i32 0, i32 2, i1 true)\n  \
             ret i32 %v\n}\n\
             define ",
            1,
        );
        let atomic = assemble(&scratch, "atomic", &atomic);
        assert!(verify_air(&atomic, &typed, Verification::At(AirLevel::Full)).is_err());
    }

    const LOOPING: &str = "kernel void k(device atomic_uint* out [[buffer(0)]], \
                           const device uint* n [[buffer(1)]]) {\n\
                           for (uint i = 0; i < n[0]; i++) {\n\
                             if (atomic_fetch_add_explicit(out, i, memory_order_relaxed) == n[1]) \
                               break;\n\
                           }\n}\n";

    /// The AIR-layer interface of `bitcode`'s `entry`.
    fn interface_of(bitcode: &[u8], entry: &str) -> crate::check::interface::AirInterface {
        let decoded = crate::check::loop_free::DecodedModule::from_bitcode(entry, bitcode).unwrap();
        crate::check::interface::AirInterface::new(
            decoded.module(),
            decoded.abi().unwrap(),
            entry,
            &BTreeMap::new(),
        )
        .unwrap()
    }

    fn row(entry: &str) -> SourceRow {
        SourceRow {
            air_sha256: String::new(),
            stage: "Kernel".into(),
            entry: entry.into(),
            air_ll: String::new(),
            blob_b64: None,
            lib_sha256s: vec![],
            label: "synthetic".into(),
        }
    }

    /// `prepare` builds the case's library as `execute` loads it: as compiled when loop-free, and
    /// fuel-instrumented under a fuel bound.
    #[test]
    fn the_case_library_prepares_as_compiled_or_under_fuel() {
        if !have_xcrun("the_case_library_prepares_as_compiled_or_under_fuel") {
            return;
        }
        let scratch = ScratchDir::new("metal-prepare-producer").unwrap();
        let case = crate::case::vector_function_constant_test_case(String::new(), "k".into());
        let row = row("k");
        let flat = fs::read(assemble(&scratch, "flat", &compiled_text(&scratch, KERNEL))).unwrap();
        let interface = interface_of(&flat, "k");
        let source = OracleSource {
            row: &row,
            entry_ll: &row.air_ll,
            air: flat.into(),
            bound: OracleBound::LoopFree,
            companion: None,
        };
        let prepared = prepare_producer_library(&case, &source, &interface, &[]).unwrap();
        assert_eq!(
            (prepared.bound, prepared.fuel, prepared.library_sha256.len()),
            (OracleBound::LoopFree, None, 1)
        );
        assert_eq!(prepared.status_slot, None);
        assert!(prepared.metallib_bytes > 0);
        let looping = fs::read(assemble(
            &scratch,
            "looping",
            &compiled_text(&scratch, LOOPING),
        ))
        .unwrap();
        let interface = interface_of(&looping, "k");
        let bound = OracleBound::Fuel { invocation_fuel: 7 };
        let source = OracleSource {
            row: &row,
            entry_ll: &row.air_ll,
            air: looping.into(),
            bound,
            companion: None,
        };
        // The entry and the case occupy slots 0 and 1; the status buffer takes the highest free.
        let prepared = prepare_producer_library(&case, &source, &interface, &[30]).unwrap();
        assert_eq!(prepared.bound, bound);
        assert_eq!(
            prepared.fuel.map(|report| (report.functions, report.loops)),
            Some((1, 1))
        );
        assert_eq!(prepared.status_slot, Some(29));
    }

    /// A public fixture the oracle compiles from its MSL (`shader_air`, as for a case that
    /// links or runs it) is AIR for the pinned deployment target, whatever the host's release, so
    /// it decodes and bounds as on the reference device: its atomics keep their macOS 26 operands,
    /// which the decoder checks.
    #[test]
    fn a_public_fixture_compiles_for_the_pinned_deployment_target() {
        if !have_xcrun("a_public_fixture_compiles_for_the_pinned_deployment_target") {
            return;
        }
        let entry = "kernel_atomic_xor_and_subtract";
        let row = SourceRow {
            label: format!("public/{entry}.ll"),
            ..row(entry)
        };
        let compiled = shader_air(&row).unwrap();
        let decoded =
            crate::check::loop_free::DecodedModule::from_bitcode(entry, &compiled.bitcode).unwrap();
        let module = decoded.module();
        let triple = module
            .parts()
            .triple
            .and_then(|sym| module.symbols().get(sym))
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_default();
        let target = crate::gpu::compile::ORACLE_DEPLOYMENT_TARGET;
        assert!(
            triple.ends_with(&format!("-apple-macosx{target}.0")),
            "{triple}"
        );
        interface_of(&compiled.bitcode, entry);
    }

    /// The indirect-command guard's code depends only on its slot and the registration count,
    /// one per authored command buffer, so `prepare` builds it: a module that encodes native
    /// commands packages only with it.
    #[test]
    fn the_case_library_prepares_with_the_indirect_command_guard() {
        if !have_xcrun("the_case_library_prepares_with_the_indirect_command_guard") {
            return;
        }
        let entry = "kernel_icb_encode_compute_commands";
        let row = SourceRow {
            label: format!("public/{entry}.ll"),
            ..row(entry)
        };
        let compiled = shader_air(&row).unwrap();
        assert!(!compiled.checked);
        let interface = interface_of(&compiled.bitcode, entry);
        let source = OracleSource {
            row: &row,
            entry_ll: &row.air_ll,
            air: compiled.bitcode.into(),
            bound: OracleBound::LoopFree,
            companion: None,
        };
        let mut case = crate::case::vector_function_constant_test_case(String::new(), entry.into());
        let unguarded = prepare_producer_library(&case, &source, &interface, &[]).err();
        assert!(
            unguarded
                .as_deref()
                .is_some_and(|why| why.contains("require a policy binding")),
            "{unguarded:?}"
        );
        case.command_buffers
            .push(crate::case::indirect_commands::CommandBufferResource {
                buffer_binding: 0,
                field_offset: 0,
                capacity: 1,
                execution_range: [0, 1],
                inherit_pipeline_state: false,
                inherit_buffers: false,
                descriptor: crate::case::indirect_commands::CommandDescriptor::Compute {
                    dispatch: crate::case::indirect_commands::ComputeDispatch::Both,
                    max_buffer_bind_count: 0,
                    max_threadgroup_memory_bind_count: 0,
                    support_dynamic_attribute_stride: false,
                },
                commands: vec![],
                execute: None,
            });
        let prepared = prepare_producer_library(&case, &source, &interface, &[]).unwrap();
        assert_eq!(
            (prepared.bound, prepared.status_slot),
            (OracleBound::LoopFree, None)
        );
        assert!(prepared.metallib_bytes > 0);
    }

    /// The guard slot, then the status slot, come from the highest slots the shared rule leaves
    /// free.
    #[test]
    fn producer_slots_come_from_slots_no_case_buffer_occupies() {
        let occupied = |taken: &[u32]| {
            let mut slots = NativeBufferSlots::default();
            for &slot in taken {
                slots.reserve(slot, 1).unwrap();
            }
            slots
        };
        let bound = OracleBound::Fuel { invocation_fuel: 1 };
        assert_eq!(
            producer_slots(occupied(&[30, 28]), true, bound).unwrap(),
            ProducerSlots {
                guard: Some(29),
                fuel: Some(27)
            }
        );
        assert_eq!(
            producer_slots(occupied(&[0]), false, OracleBound::LoopFree).unwrap(),
            ProducerSlots {
                guard: None,
                fuel: None
            }
        );
    }

    /// The public fixture `kernel_vector_function_constant`, checked as a pipeline template.
    fn checked_template(scratch: &ScratchDir) -> CheckedShader<Arc<SourceRow>> {
        let ll = include_str!("../../../fixtures/public/kernel_vector_function_constant.ll");
        let source = Arc::new(SourceRow {
            air_sha256: crate::hash::sha256_bytes(ll.as_bytes()),
            air_ll: ll.into(),
            stage: "Kernel".into(),
            entry: "kernel_vector_function_constant".into(),
            blob_b64: None,
            lib_sha256s: vec![],
            label: "public/kernel_vector_function_constant.ll".into(),
        });
        let mut case = crate::case::vector_function_constant_test_case(
            source.air_sha256.clone(),
            source.entry.clone(),
        );
        case.case_id = case.computed_case_id().unwrap();
        let checked = crate::check::check_case_against_source(scratch.path(), case, &source)
            .unwrap()
            .primary;
        CheckedShader {
            case: checked.case,
            source: source.clone(),
            interface: checked.interface,
            reflection: checked.reflection,
            companion: checked.companion,
            requirements: checked.requirements,
            linked_functions: checked.linked_functions,
            input_sha256: checked.input_sha256,
            loop_class: checked.loop_class,
            fuel_scope: checked.fuel_scope,
            entry_projection: checked.entry_projection,
        }
    }

    /// A public fixture is compiled on this host, apart from the row the checker classified; any
    /// other row loads its retained blob, the checked module itself. `prepare` builds every other
    /// library the oracle loads: each template's, and each distinct linked module's once.
    #[test]
    fn dependency_libraries_build_each_template_and_each_linked_module_once() {
        if !have_xcrun("dependency_libraries_build_each_template_and_each_linked_module_once") {
            return;
        }
        let scratch = ScratchDir::new("metal-prepare-dependencies").unwrap();
        let template = checked_template(&scratch);
        let compiled = shader_air(&template.source).unwrap();
        assert!(!compiled.checked);
        let blob = SourceRow {
            blob_b64: Some(base64::engine::general_purpose::STANDARD.encode(&compiled.bitcode)),
            ..row("kernel_vector_function_constant")
        };
        let retained = shader_air(&blob).unwrap();
        assert!(retained.checked);
        assert_eq!(retained.bitcode, compiled.bitcode);

        let visible = compiled_text(
            &scratch,
            "[[visible]] uint twice(uint x) { return 2 * x; }
",
        );
        let visible = fs::read(assemble(&scratch, "visible", &visible)).unwrap();
        let module = LibraryModuleRow {
            module_sha256: "linked".into(),
            air_ll: String::new(),
            blob_b64: base64::engine::general_purpose::STANDARD.encode(visible),
            lib_sha256s: vec![],
            label: "linked".into(),
        };
        let reference = ResolvedFunctionReference {
            function: "twice".into(),
            module,
        };
        let linked = ResolvedLinkedFunctions {
            references: vec![reference.clone(), reference],
            visible: vec![],
            intersection: vec![],
        };
        let templates = BTreeMap::from([(
            "template".to_string(),
            OracleTemplate {
                shader: &template,
                air: compiled.bitcode,
            },
        )]);
        assert_eq!(
            prepare_dependency_libraries(None, &linked, &templates).map(|built| built.len()),
            Ok(2)
        );
        assert_eq!(
            prepare_dependency_libraries(
                None,
                &ResolvedLinkedFunctions::default(),
                &BTreeMap::new()
            ),
            Ok(vec![])
        );
    }

    /// The oracle loads a generated companion as it loads the case's own library: its typed
    /// spelling assembled by `metal-as`, then `metal-opt` at the strictest level and `metallib`, so
    /// `prepare` builds it beside the case's.
    #[test]
    fn a_generated_companion_prepares_as_the_oracle_loads_it() {
        use crate::case::{Stage, TextureFormat, VertexObservation};
        use crate::companion::tests::{case, derived};
        use metal2vulkan_devtools::{fixtures_dir, llvm_assemble};
        if !have_xcrun("a_generated_companion_prepares_as_the_oracle_loads_it") {
            return;
        }
        let Ok(triangle) = llvm_assemble(&fixtures_dir().join("vertex_varying_triangle.ll")) else {
            eprintln!("SKIPPED a_generated_companion_prepares_as_the_oracle_loads_it: llvm-as");
            return;
        };
        let mut observed = case(
            Stage::Vertex,
            "vertex_varying_triangle",
            TextureFormat::Rgba32Float,
        );
        observed.vertex_observation = Some(VertexObservation::Position);
        let observer = derived(&observed, &triangle).unwrap().unwrap();
        assert!(!companion_air(&observer).unwrap().is_empty());
        assert_eq!(
            prepare_dependency_libraries(
                Some(&observer),
                &ResolvedLinkedFunctions::default(),
                &BTreeMap::new()
            )
            .map(|built| built.len()),
            Ok(1)
        );
    }
}
