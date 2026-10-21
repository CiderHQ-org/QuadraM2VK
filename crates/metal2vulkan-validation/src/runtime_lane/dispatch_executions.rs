//! Executions of indirect command buffers the GPU may write, on the GPU alone by indirect
//! dispatches (`runtime/dispatch_execute.rs`), on the device the cfg selects, under the validation
//! layer. Every test opens its device with `VK_EXT_device_generated_commands` withheld
//! (`DeviceOptions::withhold_generated_commands`), so the dispatches' recording runs on every
//! device, MoltenVK's among them. The records are the CPU's, of two states made with
//! `supportIndirectCommandBuffers` whose kernels do not observe their threadgroup partition
//! (`flat_sum` of `indirect_dispatches` and [`FLAT_MARK`]), and an encoder makes the buffer
//! resident for writing, so the GPU may write them: each execution is total, and runs with no
//! host step. Each runs what Metal runs:
//!
//! - each command its own state, group count and buffers, a command never written and a reset
//!   command nothing, in a serial and in a concurrent encoder, with commands times states indirect
//!   dispatches and nothing announced; the serial encoder's with a barrier between every two
//!   commands, the concurrent encoder's, whose records no route could set a barrier in, with none;
//! - a concurrent encoder's execution whose buffer's records the CPU sets a barrier in after the
//!   commit, before its batch is submitted (it waits for an event), recorded again with the
//!   barriers, still on the GPU alone;
//! - a range its own commands alone;
//! - inside a block of command predication, nothing where the block skips and every command where
//!   it holds, from the preprocess's gate;
//! - a checked execution, and one whose range is read from a buffer, keep the host step, announced
//!   once, and run the same.

use super::degrade_notices::assert_made_once;
use super::generated_executions::flat_template;
use super::indirect_host_steps::HOST_STEPS;
use super::shared_kinds;
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_case::{buffer_with, read_buffer};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::StatusReport;
use metal2vulkan::metal::{DispatchType, IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, RuntimeFact, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState,
    Degradation, Device, EventRef, ExecutedPipeline, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, PipelineOptions, ResourceOptions, Size,
};

/// `flat_mark(params, out, p)`: each thread adds `params[0] * 1000 + p.x` to `out[p.x]` by a
/// relaxed atomic, reading only `thread_position_in_grid`: it does not observe its threadgroup
/// partition.
const FLAT_MARK: &str = r#"define void @flat_mark(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, ptr addrspace(1) noundef "air-buffer-no-alias" %b1, <3 x i32> noundef %p) {
entry:
  %base = load i32, ptr addrspace(1) %b0, align 4
  %px = extractelement <3 x i32> %p, i64 0
  %scaled = mul i32 %base, 1000
  %v = add i32 %scaled, %px
  %px64 = zext i32 %px to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %px64
  %old = atomicrmw add ptr addrspace(1) %o, i32 %v monotonic, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @flat_mark, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b1"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"p"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The words of one command's window of `params` and of `out`.
const WINDOW: usize = 64;

/// The commands of the buffer.
const COMMANDS: u64 = 8;

/// What a command holds: nothing, or a dispatch of a state (`Sum`: `flat_sum`, `Mark`:
/// [`FLAT_MARK`]'s) of `threads` threads in all, as threadgroups or as threads.
#[derive(Clone, Copy)]
enum Command {
    /// Never written.
    Unwritten,
    /// Written, then reset.
    Reset,
    Sum(Grid),
    Mark(Grid),
}

/// A command's dispatch: `groups` threadgroups of `local` threads, or `threads` threads in
/// threadgroups of `local`.
#[derive(Clone, Copy)]
enum Grid {
    Groups { groups: u64, local: u64 },
    Threads { threads: u64, local: u64 },
}

impl Grid {
    /// The threads it runs.
    fn threads(self) -> usize {
        match self {
            Grid::Groups { groups, local } => (groups * local) as usize,
            Grid::Threads { threads, .. } => threads as usize,
        }
    }
}

/// The commands, by index: each state, a ragged grid among the threads dispatches, an unwritten
/// and a reset command.
const PLAN: [Command; COMMANDS as usize] = [
    Command::Sum(Grid::Groups {
        groups: 2,
        local: 8,
    }),
    Command::Mark(Grid::Threads {
        threads: 24,
        local: 8,
    }),
    Command::Unwritten,
    Command::Sum(Grid::Threads {
        threads: 5,
        local: 4,
    }),
    Command::Mark(Grid::Groups {
        groups: 1,
        local: 64,
    }),
    Command::Reset,
    Command::Mark(Grid::Threads {
        threads: 7,
        local: 2,
    }),
    Command::Sum(Grid::Groups {
        groups: 1,
        local: 1,
    }),
];

/// Word 0 of command `command`'s window of `params`.
fn base(command: usize) -> u32 {
    command as u32 + 2
}

/// What Metal computes into `out` where commands `ran` run once each: `flat_sum`'s thread `p` of
/// a grid of `t` adds `base + 3 p + 11 t + 13 + 17`, [`FLAT_MARK`]'s `base * 1000 + p`, each to
/// word `p` of its command's window, which starts at 0.
fn expected(ran: std::ops::Range<usize>) -> Vec<u32> {
    let mut out = vec![0u32; COMMANDS as usize * WINDOW];
    for command in ran {
        let window = &mut out[command * WINDOW..][..WINDOW];
        let b = base(command);
        match PLAN[command] {
            Command::Unwritten | Command::Reset => {}
            Command::Sum(grid) => {
                let t = grid.threads() as u32;
                for p in 0..t {
                    window[p as usize] += b + 3 * p + 11 * t + 30;
                }
            }
            Command::Mark(grid) => {
                for p in 0..grid.threads() as u32 {
                    window[p as usize] += b * 1000 + p;
                }
            }
        }
    }
    out
}

/// The test's buffers and states: the indirect command buffer of [`PLAN`]'s commands, written by
/// the CPU, `params`, `out`, and both states.
struct Setup {
    icb: IndirectCommandBuffer,
    params: Buffer,
    out: Buffer,
    /// Held: the states made with `supportIndirectCommandBuffers` alive, which the records may
    /// name.
    states: [ComputePipelineState; 2],
}

impl Setup {
    fn new(device: &Device) -> Setup {
        let sum = flat_template(device);
        let mark = mark_state(device);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
            .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        descriptor.max_kernel_buffer_bind_count = 2;
        let icb = device
            .new_indirect_command_buffer(&descriptor, COMMANDS, ResourceOptions::shared())
            .unwrap();
        let params: Vec<u8> = (0..COMMANDS as usize * WINDOW)
            .map(|word| match word % WINDOW {
                0 => base(word / WINDOW),
                _ => 0,
            })
            .flat_map(u32::to_le_bytes)
            .collect();
        let params = buffer_with(device, &params).unwrap();
        let out = buffer_with(device, &vec![0; COMMANDS as usize * WINDOW * 4]).unwrap();
        for (index, command) in PLAN.iter().enumerate() {
            let (state, grid) = match *command {
                Command::Unwritten => continue,
                Command::Reset => (
                    &sum,
                    Grid::Groups {
                        groups: 1,
                        local: 1,
                    },
                ),
                Command::Sum(grid) => (&sum, grid),
                Command::Mark(grid) => (&mark, grid),
            };
            let at = icb.indirect_compute_command_at(index as u64).unwrap();
            at.set_compute_pipeline_state(state).unwrap();
            let offset = (index * WINDOW * 4) as u64;
            at.set_kernel_buffer(Some(&params), offset, 0).unwrap();
            at.set_kernel_buffer(Some(&out), offset, 1).unwrap();
            match grid {
                Grid::Groups { groups, local } => at
                    .concurrent_dispatch_threadgroups(
                        Size::new(groups, 1, 1),
                        Size::new(local, 1, 1),
                    )
                    .unwrap(),
                Grid::Threads { threads, local } => at
                    .concurrent_dispatch_threads(Size::new(threads, 1, 1), Size::new(local, 1, 1))
                    .unwrap(),
            }
            if matches!(command, Command::Reset) {
                at.reset().unwrap();
            }
        }
        Setup {
            icb,
            params,
            out,
            states: [sum, mark],
        }
    }

    /// Make the buffer resident for writing in an encoder of `command_buffer` that writes
    /// nothing: from then on the GPU may write its records, so an execution reads them on the
    /// GPU or in a host step.
    fn writable(&self, command_buffer: &CommandBuffer) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(self.icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder.end_encoding().unwrap();
        assert!(self.icb.gpu_written());
    }

    /// Make the commands' windows resident on `encoder`.
    fn resident(&self, encoder: &mut ComputeCommandEncoder) {
        encoder
            .use_resource(self.params.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder
            .use_resource(self.out.as_resource(), ResourceUsage::WRITE)
            .unwrap();
    }

    /// `out`, read back.
    fn output(&self) -> Vec<u32> {
        read_buffer(&self.out, 0, self.out.length())
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }
}

/// [`FLAT_MARK`]'s state on `device`, made with `supportIndirectCommandBuffers`.
fn mark_state(device: &Device) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(FLAT_MARK).unwrap())
        .unwrap();
    let function = library.new_function_with_name("flat_mark").unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Run `test` on a validated device with `VK_EXT_device_generated_commands` withheld; then require
/// that the layer reported nothing.
fn on_device(test: impl FnOnce(&Device, &Messages)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.withhold_generated_commands = true;
        })
        .unwrap();
        assert!(device.compile_device().generated_commands().is_none());
        test(&device, &messages);
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// The indirect dispatches `device` replayed so far.
fn dispatches_indirect(device: &Device) -> u64 {
    device.work_counts().commands_replayed.cmd_dispatch_indirect
}

/// The pipeline barriers `device` replayed so far.
fn barriers(device: &Device) -> u64 {
    device.work_counts().commands_replayed.cmd_pipeline_barrier
}

/// Whether `command_buffer` ran a flat variant: the GPU-alone path's execution record, where the
/// host step's runs the variants its dispatches plan.
fn ran_flat(command_buffer: &CommandBuffer) -> bool {
    command_buffer
        .executions()
        .iter()
        .any(|execution| match execution.pipeline() {
            ExecutedPipeline::Compute(pipeline) => pipeline
                .dependencies
                .iter()
                .any(|dependency| dependency.fact == RuntimeFact::FlatGrid),
            ExecutedPipeline::Render(_) => false,
        })
}

/// Commit `command_buffer`, wait for it, and require it completed with a clean report.
fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(
        command_buffer.execution_report(),
        Some(StatusReport::default())
    );
}

/// Require that `device` announced `kinds` once each, and beside them only what the test's
/// `Shared` buffers force: nothing of its own for executions on the GPU alone.
fn announced(device: &Device, messages: &Messages, kinds: &[Degradation]) {
    assert_made_once(messages, kinds, &shared_kinds(device));
}

/// What the host step of [`PLAN`]'s commands announces: the step, and `split-dispatch` for the
/// ragged grid it dispatches as Vulkan does, whole workgroups (the flat variants on the GPU alone
/// take it exactly, announcing nothing).
const STEPPED: [Degradation; 2] = [HOST_STEPS[0], Degradation::SplitDispatch];

/// Every command runs its own state, group count and buffers, and the unwritten and reset ones
/// nothing, in a serial and in a concurrent encoder: on the GPU alone, one indirect dispatch per
/// command and state, the execution record listing the flat variants, nothing announced. The
/// serial encoder's commands take a barrier between every two, the concurrent encoder's none: no
/// route set a barrier in the buffer's records.
#[test]
fn each_command_runs_its_own_state_and_group_count() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let mut ordered = Vec::new();
        for dispatch_type in [DispatchType::Serial, DispatchType::Concurrent] {
            let setup = Setup::new(device);
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            let before = dispatches_indirect(device);
            let barriers_before = barriers(device);
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap();
            setup.resident(&mut encoder);
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let at = format!("{dispatch_type:?}");
            assert_eq!(setup.output(), expected(0..COMMANDS as usize), "{at}");
            assert!(ran_flat(&command_buffer), "{at}");
            let states = setup.states.len() as u64;
            assert_eq!(
                dispatches_indirect(device) - before,
                COMMANDS * states,
                "{at}: one indirect dispatch per command and state"
            );
            ordered.push(barriers(device) - barriers_before);
        }
        assert_eq!(
            ordered[0] - ordered[1],
            COMMANDS - 1,
            "a barrier between every two commands in the serial encoder alone: {ordered:?}"
        );
        announced(device, messages, &[]);
    });
}

/// A concurrent encoder's execution recorded with no barrier between its commands is recorded
/// again with them where the CPU sets a barrier in one of the buffer's records after the commit,
/// before its batch is submitted (the batch waits for an event the CPU signals after the set): the
/// commands run on the GPU alone, with as many barriers as the serial encoder's, and write what
/// Metal computes. Without the set, none.
#[test]
fn a_barrier_set_after_the_commit_records_the_commands_again() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let mut taken = Vec::new();
        for set in [false, true] {
            let setup = Setup::new(device);
            let event = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)
                .unwrap();
            setup.resident(&mut encoder);
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            encoder.end_encoding().unwrap();
            let before = barriers(device);
            command_buffer.commit().unwrap();
            if set {
                let command = setup.icb.indirect_compute_command_at(3).unwrap();
                command.set_barrier().unwrap();
            }
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(
                command_buffer.execution_report(),
                Some(StatusReport::default())
            );
            assert_eq!(setup.output(), expected(0..COMMANDS as usize), "set {set}");
            assert!(ran_flat(&command_buffer), "set {set}");
            taken.push(barriers(device) - before);
        }
        assert!(
            taken[1] >= taken[0] + COMMANDS - 1,
            "the commands recorded again with a barrier between every two: {taken:?}"
        );
        announced(device, messages, &[]);
    });
}

/// A range runs its own commands alone, on the GPU alone, one indirect dispatch per command of the
/// range and state.
#[test]
fn a_range_runs_its_commands_alone() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let setup = Setup::new(device);
        let command_buffer = queue.command_buffer().unwrap();
        setup.writable(&command_buffer);
        let before = dispatches_indirect(device);
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.resident(&mut encoder);
        encoder
            .execute_commands_in_buffer(&setup.icb, 1..5)
            .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let mut want = expected(1..5);
        // Command 0's window stays zero though the range ends past it.
        want[..WINDOW].fill(0);
        assert_eq!(setup.output(), want);
        assert!(ran_flat(&command_buffer));
        assert_eq!(dispatches_indirect(device) - before, 4 * 2);
        announced(device, messages, &[]);
    });
}

/// Inside a block of command predication an execution runs from its preprocess's gate, on the GPU
/// alone: nothing in a failing If's block, every command in its Else, and the same in a holding
/// If's block, whose Else runs nothing.
#[test]
fn predicated_executions_run_as_their_blocks_say() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        // Word 0 is 1: an If on `== 1` holds, one on `== 0` fails.
        let word = buffer_with(device, &1u32.to_le_bytes()).unwrap();
        for holds in [false, true] {
            let setup = Setup::new(device);
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.resident(&mut encoder);
            encoder
                .encode_start_if(Some(&word), 0, 0, u32::from(holds))
                .unwrap();
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            encoder.encode_start_else().unwrap();
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            assert_eq!(encoder.encode_end_if(), Ok(true));
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(
                setup.output(),
                expected(0..COMMANDS as usize),
                "holds {holds}: one of the two executions ran"
            );
            assert!(ran_flat(&command_buffer), "holds {holds}");
        }
        announced(device, messages, &[]);
    });
}

/// A checked execution (the caller's check runs in the host step), and one whose range is read
/// from a buffer, keep the host step, announced once; both run what Metal runs.
#[test]
fn checked_and_indirect_range_executions_keep_the_host_step() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        for form in ["checked", "indirect range"] {
            let setup = Setup::new(device);
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            let range = buffer_with(device, &[0, 0, 0, 0, COMMANDS as u8, 0, 0, 0]).unwrap();
            let checked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.resident(&mut encoder);
            match form {
                "checked" => {
                    let seen = std::sync::Arc::clone(&checked);
                    encoder
                        .execute_commands_in_buffer_checked(
                            &setup.icb,
                            0..COMMANDS,
                            Box::new(move |_, _| {
                                seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }),
                            metal2vulkan_runtime::IndirectCommandCheckNeeds::ALL,
                        )
                        .unwrap();
                }
                _ => encoder
                    .execute_commands_in_buffer_indirect(&setup.icb, &range, 0)
                    .unwrap(),
            }
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(setup.output(), expected(0..COMMANDS as usize), "{form}");
            assert!(!ran_flat(&command_buffer), "{form}");
            if form == "checked" {
                // The check saw each command the step recorded.
                assert!(checked.load(std::sync::atomic::Ordering::Relaxed) > 0);
            }
        }
        // The check reads the commands, which answer the runtime's unique identifiers of their
        // states, announced as the getter announces them.
        let identifiers = [Degradation::UnmeasuredUniqueIdentifier];
        let kinds: Vec<Degradation> = STEPPED.iter().chain(&identifiers).copied().collect();
        announced(device, messages, &kinds);
    });
}
