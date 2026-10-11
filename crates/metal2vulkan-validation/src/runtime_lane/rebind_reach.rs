//! A command buffer committed before a rebind of its no-copy buffer or heap
//! (`rebind_bytes_no_copy`) and first submitted after it, on the device the cfg selects, under the
//! validation layer. The rule is Metal's as the consumer measured it on the M3 Max (macOS 26.7, the
//! rebind probe's tables 2 and 3; a detach then a replace of the backing, which the rebind is):
//!
//! - **Resolved at the first submission.** What the command buffer's encoders made resident
//!   (`use_resources`, case 3; `use_heap`, case 3h) reaches the pages the buffer lies in as the
//!   command buffer is first submitted: the new pages, and after two rebinds the last.
//! - **Unnamed reaches.** A buffer the command buffer neither names nor makes resident is reached
//!   at its old listing no more: the walk reads 0, whether an earlier command buffer used it
//!   (case 4) or none did since its pages were given (case 4x).
//! - **Named uses, encoded again**. A command that names the buffer (a binding, case 1; a binding
//!   the walk also reaches, case 2; a blit's operand, case 5) was resolved as it was encoded; the
//!   rebind encodes its compute, blit or render encoder again from its log, so it reaches the new
//!   pages, as Metal's does, and the command buffer completes: a draw reading a vertex buffer over
//!   host pages draws the new pages' corners. Where the encoder cannot be encoded again (a call
//!   refused at its encoding, and so again), each of Metal's submissions of the command buffer
//!   holding such a command runs nothing and the command buffer fails
//!   `CommandBufferError::EncodedUseRebound` (metal code 1). Its other submissions run, its event
//!   signals fire, its handlers run, and the device announces that gap once
//!   (`Degradation::EncodedUseRebound`).
//! - **Submitted is kept.** A command buffer whose first batch was submitted before the rebind
//!   keeps the table and the pages it took then for every later batch (the split: a wait for an
//!   event after an encoder), named or resident (table 3: a running command buffer reads the old
//!   pages until it completes). The old pages' deallocator runs once it completed; so does a
//!   released buffer's that its table lists, while a buffer released beside it that no submitted
//!   table lists goes at once.
//!
//! The walker (`walk-a-device-pointer-chain-through-a-loop-phi`, an owned case) reaches its
//! argument buffer's member through a pointer it loads; the member holds a node whose value is
//! the page set's ([`A`], [`B`], [`C`]), and its output is calibrated over plain buffers holding
//! each node, and an all-zero one for an unreached member.

use super::degrade_notices::{assert_made_once, on_device as on_device_noticed};
use super::host_memory::{gpu_read, host_page, no_copy, not_importable, on_device, recording};
use super::host_memory::{Mapping, Seen};
use super::no_copy_heaps::{no_copy_heap, shared, HEAP};
use super::rebinds::{rebind, rebind_heap};
use super::shared_kinds;
use super::{
    bound_buffers, buffer_with, encode_case, pipeline_state, public_case, read_buffer,
    validated_device_with, CaseResources, Messages, PublicCase, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::ComputeOutput;
use crate::gpu::TEST_DEVICE;
use crate::worker::alloc::live_bytes;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    CommitFailure, Degradation, Device, EventRef, FailedEncoders, Heap, HeapType, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    StoreAction, Submission, SubmissionCheck, TextureDescriptor,
};
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Walks a pointer chain from the address its argument buffer (buffer 0) holds at offset 0.
pub(super) const POINTER_WALKER: &str = "walk-a-device-pointer-chain-through-a-loop-phi";

/// The walker's binding of its step count, a word it reads.
const STEPS: u32 = 2;

/// The node values of the old pages, the first new ones and the second, as the probe's sentinels.
pub(super) const A: u32 = 0xa0;
pub(super) const B: u32 = 0xb0;
pub(super) const C: u32 = 0xc0;

/// The kinds the gap's test takes.
pub(super) const REBOUND_USES: [Degradation; 1] = [Degradation::EncodedUseRebound];

/// Where a heap's member lies in it.
pub(super) const IN_HEAP: usize = 64 << 10;

/// A node of the walked chain: no next node, and `value`.
pub(super) fn node(value: u32) -> Vec<u8> {
    let mut bytes = vec![0; 16];
    bytes[8..12].copy_from_slice(&value.to_le_bytes());
    bytes
}

/// The walker's output binding, offset and length.
pub(super) fn output(walker: &PublicCase) -> (u32, u64, u64) {
    let ComputeOutput::Buffer {
        binding,
        offset,
        length,
    } = walker.compute.output
    else {
        panic!("{POINTER_WALKER}: a buffer output");
    };
    (binding, offset, length)
}

/// The walker's outputs over a member holding each node, and over an unreached one.
pub(super) struct Outputs {
    pub(super) a: Vec<u8>,
    pub(super) b: Vec<u8>,
    pub(super) c: Vec<u8>,
    pub(super) zero: Vec<u8>,
}

impl Outputs {
    /// Each walked over a plain buffer its encoder makes resident; distinct.
    pub(super) fn new(device: &Device, queue: &CommandQueue, walker: &PublicCase) -> Outputs {
        let over = |bytes: Vec<u8>| {
            let member = buffer_with(device, &bytes).unwrap();
            let walked = walk(device, queue, walker, &member, Uses::Resident, false, || {});
            assert_eq!(walked.command_buffer.error(), None);
            walked.produced
        };
        let outputs = Outputs {
            a: over(node(A)),
            b: over(node(B)),
            c: over(node(C)),
            zero: over(vec![0; 16]),
        };
        let all = [&outputs.a, &outputs.b, &outputs.c, &outputs.zero];
        for (at, one) in all.iter().enumerate() {
            assert!(
                all[at + 1..].iter().all(|other| other != one),
                "the walker's outputs over the nodes are not distinct"
            );
        }
        outputs
    }
}

/// What the walker's command buffer holds of the member besides the walk's reach of it.
#[derive(Clone, Copy)]
pub(super) enum Uses<'a> {
    /// Its encoder makes the member resident (`use_resources`): case 3.
    Resident,
    /// Its encoder makes the member's heap resident (`use_heap`): case 3h.
    Heap(&'a Heap),
    /// Nothing: cases 4 and 4x.
    Unnamed,
    /// The walker's step count is read from the given buffer, bound: case 1.
    Steps(&'a Buffer),
    /// The walker writes its output into the member, bound: case 2.
    Output,
    /// A blit before the walker copies the member's node out: case 5.
    Blit,
}

/// A walk's command buffer, once it completed, and what it left.
pub(super) struct Walked {
    pub(super) command_buffer: CommandBuffer,
    /// The walker's output.
    pub(super) produced: Vec<u8>,
    /// The event the command buffer signals as its last command, to 1.
    pub(super) signaled: u64,
    /// Whether its completed handler ran.
    pub(super) handled: bool,
    /// What the blit before the walker copied out of the member ([`Uses::Blit`]).
    pub(super) copied: Option<Vec<u8>>,
}

/// The walker run over `member`, held by `uses`, in a command buffer that waits for an event
/// before it where `wait` (`between`, a rebind, running once it is committed, before the event),
/// else at once; the command buffer signals an event of its own last.
pub(super) fn walk(
    device: &Device,
    queue: &CommandQueue,
    walker: &PublicCase,
    member: &Buffer,
    uses: Uses<'_>,
    wait: bool,
    between: impl FnOnce(),
) -> Walked {
    let (binding, offset, length) = output(walker);
    let state = pipeline_state(device, walker).unwrap();
    let mut holding = CaseResources::new(device, walker).unwrap();
    let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
        panic!("{POINTER_WALKER}: one argument-buffer buffer");
    };
    holding.argument_buffers.insert(field, member.clone());
    let mut buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
    match uses {
        Uses::Steps(steps) => {
            buffers.insert(STEPS, steps.clone());
        }
        Uses::Output => {
            buffers.insert(binding, member.clone());
        }
        _ => {}
    }
    if !matches!(uses, Uses::Resident) {
        // The encoder makes no member resident: the walk reaches it unnamed.
        drop(holding.argument_buffers.remove(&field));
    }
    let event = device.new_shared_event().unwrap();
    let done = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if wait {
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
    }
    let copied = match uses {
        Uses::Blit => {
            let copied = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(member, 0, &copied, 0, 16)
                .unwrap();
            blit.end_encoding().unwrap();
            Some(copied)
        }
        _ => None,
    };
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    if let Uses::Heap(heap) = uses {
        encoder.use_heap(heap).unwrap();
    }
    encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
        .encode_signal_event(EventRef::from(&done), 1)
        .unwrap();
    let handled = Arc::new(AtomicBool::new(false));
    {
        let handled = Arc::clone(&handled);
        command_buffer
            .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                handled.store(true, Ordering::SeqCst);
            }))
            .unwrap();
    }
    command_buffer.commit().unwrap();
    if wait {
        between();
        assert_eq!(
            command_buffer.kernel_start_time(),
            0.0,
            "a batch was submitted before the event"
        );
        event.set_signaled_value(1);
    }
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let produced = read_buffer(&buffers[&binding], offset, length).unwrap();
    let deadline = Instant::now() + WAIT;
    while !handled.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the completed handler never ran");
        std::thread::sleep(Duration::from_millis(1));
    }
    Walked {
        command_buffer,
        produced,
        signaled: done.signaled_value(),
        handled: handled.load(Ordering::SeqCst),
        copied: copied.map(|copied| read_buffer(&copied, 0, 16).unwrap()),
    }
}

/// `len` bytes of fresh host memory holding `bytes` at `at`, zero elsewhere.
pub(super) fn pages(len: usize, at: usize, bytes: &[u8]) -> Mapping {
    let mapping = Mapping::anonymous(len);
    mapping.write(at, bytes);
    mapping
}

/// The buffers of these tests: two of any host's pages.
pub(super) fn buffer_len() -> usize {
    2 * host_page().max(16 << 10)
}

/// A no-copy buffer over `mapping`, whose deallocator records into `seen`; `None`, printed, where
/// the device imports no host memory.
pub(super) fn buffer_over(device: &Device, mapping: Mapping, seen: &Arc<Seen>) -> Option<Buffer> {
    let pointer = mapping.pointer();
    // SAFETY: the deallocator releases the mapping, after the buffer's last use of it.
    let made = unsafe {
        no_copy(
            device,
            pointer,
            buffer_len(),
            ResourceOptions::default(),
            Some(recording(seen, mapping)),
        )
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(error) if not_importable(&error) => {
            eprintln!("REBIND-REACH: the device imports no host memory: {error:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

/// A no-copy buffer over a node of `value`, its deallocator's runs in a `Seen` of its own.
pub(super) fn member_over(device: &Device, value: u32) -> Option<(Buffer, Arc<Seen>)> {
    let seen = Arc::new(Seen::default());
    let buffer = buffer_over(device, pages(buffer_len(), 0, &node(value)), &seen)?;
    Some((buffer, seen))
}

/// Rebind `buffer` to fresh pages holding a node of `value`.
pub(super) fn rebind_to(buffer: &Buffer, value: u32) -> Arc<Seen> {
    let seen = Arc::new(Seen::default());
    rebind(buffer, pages(buffer_len(), 0, &node(value)), &seen).unwrap();
    seen
}

/// A placement heap over the caller's memory holding a node of `A` at [`IN_HEAP`], and its member
/// there; `None` where the device imports no host memory.
pub(super) fn heap_member(device: &Device) -> Option<(Heap, Buffer)> {
    let mapping = pages(HEAP, IN_HEAP, &node(A));
    let pointer = mapping.pointer();
    // SAFETY: the deallocator releases the mapping once the heap's pages go.
    let heap = unsafe {
        no_copy_heap(
            device,
            pointer,
            HeapType::Placement,
            Some(recording(&Arc::new(Seen::default()), mapping)),
        )
    }?;
    let member = heap
        .new_buffer_at_offset(256, shared(), IN_HEAP as u64)
        .unwrap();
    Some((heap, member))
}

/// Rebind `heap` to fresh pages holding a node of `value` at [`IN_HEAP`].
pub(super) fn rebind_heap_to(heap: &Heap, value: u32) {
    let seen = Arc::new(Seen::default());
    rebind_heap(heap, pages(HEAP, IN_HEAP, &node(value)), &seen).unwrap();
}

/// Wait until the deallocator `seen` records ran.
pub(super) fn deallocated(seen: &Seen) {
    let deadline = Instant::now() + WAIT;
    while seen.runs.load(Ordering::SeqCst) == 0 {
        assert!(Instant::now() < deadline, "the deallocator never ran");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Table 2, case 3 and 3h (detach+replace, 6a/6b): what a pending command buffer's encoder made
/// resident reaches the pages its buffer lies in at the first submission, B after one rebind, C
/// after two; with no rebind, A. Buffer and heap forms, `use_resources` and `use_heap`.
#[test]
fn a_pending_command_buffer_s_resident_uses_reach_the_pages_of_its_first_submission() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let held = |member: &Buffer, uses: Uses<'_>, between: &dyn Fn()| {
            let walked = walk(device, &queue, &walker, member, uses, true, between);
            assert_eq!(walked.command_buffer.error(), None);
            walked.produced
        };
        assert!(
            held(&member, Uses::Resident, &|| {}) == outputs.a,
            "3: control"
        );
        let once = held(&member, Uses::Resident, &|| drop(rebind_to(&member, B)));
        assert!(once == outputs.b, "3: one rebind");
        let twice = held(&member, Uses::Resident, &|| {
            rebind_to(&member, C);
            rebind_to(&member, B);
        });
        assert!(twice == outputs.b, "3: two rebinds, the last's pages");
        let to_c = held(&member, Uses::Resident, &|| {
            rebind_to(&member, A);
            rebind_to(&member, C);
        });
        assert!(to_c == outputs.c, "3: two rebinds, to C");

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        for (name, uses) in [("3", Uses::Resident), ("3h", Uses::Heap(&heap))] {
            rebind_heap_to(&heap, A);
            assert!(
                held(&placed, uses, &|| {}) == outputs.a,
                "heap {name}: control"
            );
            let once = held(&placed, uses, &|| rebind_heap_to(&heap, B));
            assert!(once == outputs.b, "heap {name}: one rebind");
            let twice = held(&placed, uses, &|| {
                rebind_heap_to(&heap, B);
                rebind_heap_to(&heap, C);
            });
            assert!(twice == outputs.c, "heap {name}: two rebinds");
        }
        eprintln!("REBIND-REACH resident: B after one rebind, C after two, buffer and heap");
    });
}

/// Table 2, cases 4 and 4x: a buffer a pending command buffer neither names nor makes resident
/// is reached at no pages once rebound, its walk reading 0 and its command buffer completing with
/// no error; with no rebind, one an earlier command buffer used reads A (case 4) and one none
/// used reads 0 (4x). Buffer and heap forms.
#[test]
fn a_pending_command_buffer_reaches_a_buffer_rebound_unnamed_at_no_pages() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let held = |member: &Buffer, between: &dyn Fn()| {
            let walked = walk(
                device,
                &queue,
                &walker,
                member,
                Uses::Unnamed,
                true,
                between,
            );
            assert_eq!(walked.command_buffer.error(), None);
            walked.produced
        };
        let Some((never, _)) = member_over(device, A) else {
            return;
        };
        assert!(held(&never, &|| {}) == outputs.zero, "4x: control");
        let rebound = held(&never, &|| drop(rebind_to(&never, B)));
        assert!(rebound == outputs.zero, "4x: rebound");

        let (used, _) = member_over(device, A).unwrap();
        gpu_read(&queue, &used, 0..16);
        assert!(held(&used, &|| {}) == outputs.a, "4: control");
        let rebound = held(&used, &|| drop(rebind_to(&used, B)));
        assert!(rebound == outputs.zero, "4: rebound");
        gpu_read(&queue, &used, 0..16);
        assert!(held(&used, &|| {}) == outputs.b, "4: used again");
        let rebound = held(&used, &|| {
            rebind_to(&used, C);
            rebind_to(&used, A);
        });
        assert!(rebound == outputs.zero, "4: two rebinds");

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        assert!(held(&placed, &|| {}) == outputs.zero, "heap 4x: control");
        let rebound = held(&placed, &|| rebind_heap_to(&heap, B));
        assert!(rebound == outputs.zero, "heap 4x: rebound");
        let other = heap.new_buffer_at_offset(256, shared(), 0).unwrap();
        gpu_read(&queue, &other, 0..4);
        assert!(held(&placed, &|| {}) == outputs.b, "heap 4: control");
        let rebound = held(&placed, &|| rebind_heap_to(&heap, C));
        assert!(rebound == outputs.zero, "heap 4: rebound");
        eprintln!("REBIND-REACH unnamed: 0 once rebound, buffer and heap");
    });
}

/// Table 2 cases 1, 2 and 5, as Metal runs them: a pending command buffer whose command names a
/// buffer rebound since it was encoded (bound and read, bound and walked, a blit's operand; a
/// heap's member bound or blitted) reaches the pages the buffer lies in at its first submission.
/// The rebind encodes the command buffer's compute and blit encoders that named it again from
/// their logs (`replay.rs`): the blit copies the new node, the walk reads it, the bound output
/// lands in the new pages, and the command buffer completes with no error, its signal fired and
/// its handler run; nothing of its own is announced (a device without unified memory announces
/// its first `Shared` buffer in system memory). Each once failed `EncodedUseRebound`. The device's
/// counts show the encoders encoded again. A rebind before the encoding takes the new pages as
/// before.
#[test]
fn a_pending_command_buffer_naming_a_buffer_rebound_since_reaches_the_new_pages() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let ran = |walked: &Walked, name: &str| {
            let command_buffer = &walked.command_buffer;
            assert_eq!(command_buffer.error(), None, "{name}");
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "{name}"
            );
            assert_eq!(walked.signaled, 1, "{name}: its signal");
            assert!(walked.handled, "{name}: its handler");
        };
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let held = |member: &Buffer, uses: Uses<'_>, between: &dyn Fn()| {
            walk(device, &queue, &walker, member, uses, true, between)
        };
        let before = device.work_counts();
        let blit = held(&member, Uses::Blit, &|| drop(rebind_to(&member, B)));
        ran(&blit, "5: a blit's operand");
        assert_eq!(blit.copied.as_deref(), Some(&node(B)[..]), "5: the copy");
        assert!(blit.produced == outputs.b, "5: the walk");
        let output = held(&member, Uses::Output, &|| drop(rebind_to(&member, C)));
        ran(&output, "2: bound and walked");
        assert!(
            output.produced == outputs.c,
            "2: the output in the new pages"
        );
        let steps_seen = Arc::new(Seen::default());
        let steps = buffer_over(
            device,
            pages(buffer_len(), 0, &2u32.to_le_bytes()),
            &steps_seen,
        )
        .unwrap();
        let plain = buffer_with(device, &node(A)).unwrap();
        let bound = held(&plain, Uses::Steps(&steps), &|| {
            let seen = Arc::new(Seen::default());
            rebind(&steps, pages(buffer_len(), 0, &2u32.to_le_bytes()), &seen).unwrap();
        });
        ran(&bound, "1: bound and read");
        assert!(bound.produced == outputs.a, "1: the walk");
        // Rebound before the encoding: the commands take the new pages.
        rebind_to(&member, A);
        let before_encoding = held(&member, Uses::Blit, &|| {});
        ran(&before_encoding, "5: rebound before the encoding");
        assert!(
            before_encoding.produced == outputs.a,
            "5: rebound before the encoding"
        );

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        let blit = held(&placed, Uses::Blit, &|| rebind_heap_to(&heap, B));
        ran(&blit, "heap 5: a blit's operand");
        assert_eq!(
            blit.copied.as_deref(),
            Some(&node(B)[..]),
            "heap 5: the copy"
        );
        let output = held(&placed, Uses::Output, &|| rebind_heap_to(&heap, C));
        ran(&output, "heap 2: bound and walked");
        assert!(output.produced == outputs.c, "heap 2: the output");
        // Nothing of its own: a device without unified memory announces its first `Shared`
        // buffer in system memory (An RTX 3080 announced it here).
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(!notices.contains_key(&Degradation::EncodedUseRebound));
        let after = device.work_counts();
        let reencoded = after.encoders_reencoded - before.encoders_reencoded;
        assert!(reencoded >= 5, "encoders encoded again: {reencoded}");
        assert_eq!(after.reencodes_refused, before.reencodes_refused);
        eprintln!(
            "CBF-MARK {}: named uses rebound before the first submission reached the new pages; \
             {reencoded} encoders encoded again",
            device.name()
        );
    });
}

/// What a rebind cannot encode again keeps the gap: a blit encoder one of whose calls was refused
/// at its encoding (a fill past the buffer) is refused again as the rebind encodes it, so the
/// encoder keeps its first recording, which named the old pages: its submission runs nothing,
/// the command buffer fails `EncodedUseRebound`, announced once, and the device counts the
/// refused encoding. Its signal fires and its handler runs.
#[test]
fn an_encoder_a_rebind_cannot_encode_again_fails_encoded_use_rebound() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let past = member.length() + 16;
        assert!(blit.fill_buffer(&member, 0..past, 0).is_err());
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let before = device.work_counts();
        drop(rebind_to(&member, B));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(
            command_buffer.error(),
            Some(CommandBufferError::EncodedUseRebound)
        );
        assert_eq!(
            read_buffer(&copied, 0, 16).unwrap(),
            [0x5a; 16],
            "ran nothing"
        );
        assert_made_once(messages, &REBOUND_USES, &shared_kinds(device));
        let after = device.work_counts();
        assert_eq!(after.reencodes_refused, before.reencodes_refused + 1);
        assert_eq!(after.encoders_reencoded, before.encoders_reencoded);
        eprintln!("REBIND-REACH refused again: EncodedUseRebound, announced once");
    });
}

/// What the logs cost: one compute encoder of [`COST_DISPATCHES`] walker dispatches, each with
/// its pipeline, buffers and resident members set again (`encode_case`), encoded three ways on
/// one device without the validation layer (whose own cost per call would hide the log's): with
/// no buffer over host pages on the device (nothing logged), with one the encoder does not name
/// (each call logged, the log let go at its end), and naming it (the log kept until the first
/// submission). Each way is encoded [`COST_ROUNDS`] times, the logged two alternating, and its
/// fastest round kept. `CBF-COST` prints each way's encode time per dispatch; `CBF-MEMORY` the
/// command buffer's memory once its encoder ended, unlogged and with the log kept: the heap bytes
/// the process's counting allocator saw it take (the runtime's own, not the driver's) and the
/// process's resident growth (Linux), beside the log's own bytes. The log is kept only where a
/// command named the buffer. The numbers are printed, not asserted: a release build gives those
/// records.
#[test]
fn a_dispatch_heavy_encoder_s_log_costs_a_push_per_call() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    let device = &device;
    let queue = device.new_command_queue().unwrap();
    let walker = public_case(POINTER_WALKER).unwrap();
    let state = pipeline_state(device, &walker).unwrap();
    let holding = CaseResources::new(device, &walker).unwrap();
    let plain = bound_buffers(device, &walker.compute, &holding).unwrap();
    let encode = |buffers: &BTreeMap<u32, Buffer>| {
        let command_buffer = queue.command_buffer().unwrap();
        let (heap, resident) = (live_bytes(), resident_bytes());
        let start = Instant::now();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        for _ in 0..COST_DISPATCHES {
            encode_case(&mut encoder, &state, &walker.compute, buffers, &holding).unwrap();
        }
        encoder.end_encoding().unwrap();
        let elapsed = start.elapsed();
        let held = Held {
            nanos: elapsed.as_secs_f64() * 1e9 / COST_DISPATCHES as f64,
            heap: live_bytes().saturating_sub(heap),
            resident: resident_bytes()
                .zip(resident)
                .map(|(now, then)| now.saturating_sub(then)),
        };
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        held
    };
    let before = device.work_counts();
    let off: Vec<Held> = (0..COST_ROUNDS).map(|_| encode(&plain)).collect();
    assert_eq!(
        device.work_counts().replay_logs_kept,
        before.replay_logs_kept,
        "a log kept on a device with no buffer over host pages"
    );
    let Some((member, _)) = member_over(device, A) else {
        return;
    };
    let mut named = plain.clone();
    let (binding, _, _) = output(&walker);
    named.insert(binding, member.clone());
    let (mut unnamed, mut kept, mut bytes) = (Vec::new(), Vec::new(), 0);
    for _ in 0..COST_ROUNDS {
        let logged = device.work_counts();
        unnamed.push(encode(&plain));
        let let_go = device.work_counts();
        assert_eq!(
            let_go.replay_logs_kept, logged.replay_logs_kept,
            "a log kept where no command named a buffer over host pages"
        );
        kept.push(encode(&named));
        let after = device.work_counts();
        assert_eq!(
            after.replay_logs_kept,
            let_go.replay_logs_kept + 1,
            "the log kept"
        );
        bytes = bytes.max(after.replay_log_bytes - let_go.replay_log_bytes);
    }
    let fastest = |held: &[Held]| held.iter().map(|h| h.nanos).fold(f64::INFINITY, f64::min);
    let least = |held: &[Held]| held.iter().map(|h| h.heap).min().unwrap_or(0);
    let resident = |held: &[Held]| {
        held.iter()
            .filter_map(|h| h.resident)
            .min()
            .map_or_else(|| "n/a".to_owned(), |bytes| bytes.to_string())
    };
    let name = device.name();
    eprintln!(
        "CBF-COST {name}: {COST_DISPATCHES} dispatches, fastest of {COST_ROUNDS}: {:.0} ns per \
         dispatch unlogged, {:.0} ns logged and let go, {:.0} ns logged and kept",
        fastest(&off),
        fastest(&unnamed),
        fastest(&kept),
    );
    eprintln!(
        "CBF-MEMORY {name}: the command buffer once its encoder ended, least of {COST_ROUNDS}: \
         heap {} bytes unlogged, {} with the log kept; resident growth {} unlogged, {} kept; the \
         log {bytes} bytes, {} per dispatch",
        least(&off),
        least(&kept),
        resident(&off),
        resident(&kept),
        bytes / COST_DISPATCHES as u64
    );
}

/// What [`a_dispatch_heavy_encoder_s_log_costs_a_push_per_call`] measures of one command buffer:
/// its encode time per dispatch, the heap bytes the process took while it encoded, and the
/// process's resident growth (where the system says).
struct Held {
    nanos: f64,
    heap: usize,
    resident: Option<usize>,
}

/// The process's resident bytes, from `/proc/self/statm` (Linux; `None` elsewhere).
fn resident_bytes() -> Option<usize> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
    // SAFETY: `sysconf` reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    Some(pages * usize::try_from(page).ok()?)
}

/// The dispatches [`a_dispatch_heavy_encoder_s_log_costs_a_push_per_call`] encodes each way.
const COST_DISPATCHES: usize = 4000;

/// The rounds [`a_dispatch_heavy_encoder_s_log_costs_a_push_per_call`] encodes each way.
const COST_ROUNDS: usize = 5;

/// A render pass committed behind an event wait, whose draw reads its corners from a vertex
/// buffer over host pages, is encoded again by a rebind before its first submission: the draw
/// reads the new pages, as Metal's does. The old pages hold corners off the target, so with no
/// rebind the draw covers nothing and the target keeps the pass's clear (blue); the new pages
/// hold a triangle over the whole target, which the fragment function writes (1/255, 0, 0, 1).
/// The fixture's pipeline (`rt6_restart.ll`) reads the corners from vertex buffer 0.
#[test]
fn a_render_pass_drawing_from_a_buffer_rebound_before_its_first_submission_reads_the_new_pages() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let pipeline = corner_pipeline(device);
        let side = 8;
        let corners =
            |points: [f32; 6]| -> Vec<u8> { points.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let off = corners([4.0, 4.0, 5.0, 4.0, 4.0, 5.0]);
        let whole = corners([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
        let drawn = |rebound: bool| -> Option<Vec<u8>> {
            let seen = Arc::new(Seen::default());
            let member = buffer_over(device, pages(buffer_len(), 0, &off), &seen)?;
            let mut target =
                TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
            target.usage = TextureUsage::RENDER_TARGET;
            target.resource_options = ResourceOptions::shared();
            let target = device.new_texture(&target).unwrap();
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
            color.store_action = StoreAction::Store;
            let event = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&pipeline).unwrap();
            encoder.set_vertex_buffer(Some(&member), 0, 0).unwrap();
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            let before = device.work_counts();
            if rebound {
                let seen = Arc::new(Seen::default());
                rebind(&member, pages(buffer_len(), 0, &whole), &seen).unwrap();
                let after = device.work_counts();
                assert_eq!(
                    after.encoders_reencoded,
                    before.encoders_reencoded + 1,
                    "the rebind encoded the render encoder again"
                );
            }
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(command_buffer.error(), None);
            let mut texels = vec![0; (side * side * 4) as usize];
            let region = Region::new_2d(0, 0, side, side);
            target
                .get_bytes(&mut texels, side * 4, 0, region, 0, 0)
                .unwrap();
            Some(texels)
        };
        let Some(kept) = drawn(false) else {
            return;
        };
        assert!(
            kept.chunks(4).all(|texel| texel == [0, 0, 255, 255]),
            "the old pages' draw covered texels: {kept:?}"
        );
        let rebound = drawn(true).unwrap();
        assert!(
            rebound.chunks(4).all(|texel| texel == [1, 0, 0, 255]),
            "the draw did not read the new pages: {rebound:?}"
        );
        eprintln!(
            "CBF-MARK {}: a render pass drawing from a buffer rebound before its first submission \
             drew the new pages",
            device.name()
        );
    });
}

/// The pipeline of `rt6_restart.ll`: corners from vertex buffer 0, each fragment
/// (1/255, 0, 0, 1) into an `RGBA8Unorm` target.
fn corner_pipeline(device: &Device) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("rt6_restart.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("rt6_restart_vertex")));
    descriptor.fragment = Some(StageSource::Function(function("rt6_restart_count")));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A pending command buffer that a submission check fails (the consumer's emulated detach,
/// `Injected(8)`, as Metal's code 8) keeps that code where its encoder also named a buffer
/// rebound since: the encoder runs nothing, and the device announces no
/// `Degradation::EncodedUseRebound`, which only a command buffer that fails with it announces.
#[test]
fn a_command_buffer_a_check_failed_keeps_its_code_and_announces_no_rebound_use() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        let check: SubmissionCheck = Arc::new(|_: &Submission<'_>| NonZeroU64::new(8));
        command_buffer.set_submission_check(check).unwrap();
        command_buffer.commit().unwrap();
        drop(rebind_to(&member, B));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(
            command_buffer.error(),
            Some(CommandBufferError::Injected(8))
        );
        assert_eq!(CommandBufferError::Injected(8).metal_code(), 8);
        assert_eq!(
            read_buffer(&copied, 0, 16).unwrap(),
            [0x5a; 16],
            "the blit ran"
        );
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(
            !notices.contains_key(&Degradation::EncodedUseRebound),
            "announced EncodedUseRebound for a command buffer failed with code 8"
        );
        eprintln!("REBIND-REACH checked: code 8 kept, no rebound-use notice");
    });
}

/// A command buffer split by an event signal into Metal's submissions A, a blit that uses no
/// member, and B, a blit that names the member, committed behind a wait; the member is rebound,
/// then detached as the consumer detaches (its submission check fails a submission that uses it
/// with code 8), while the command buffer waits. A runs; B runs nothing and the command buffer
/// keeps B's check's code 8, with no `Degradation::EncodedUseRebound` notice (Metal: A runs, B
/// fails with code 8). Rebound and not detached, B's check passes it: A runs, and B, encoded again
/// at the rebind (It failed `EncodedUseRebound` before), copies the new node, and the command
/// buffer completes, with no notice. Either way the pages are fixed at the first submission for
/// every later batch, and B's submission is decided at its own batch, after its check.
#[test]
fn a_split_command_buffer_decides_each_rebound_submission_after_its_own_check() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let split = |detached: bool| {
            let ran = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            let copied = buffer_with(device, &[0x5a; 16]).unwrap();
            let event = device.new_shared_event().unwrap();
            let between = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut first = command_buffer.blit_command_encoder().unwrap();
            first.fill_buffer(&ran, 0..16, 0xa5).unwrap();
            first.end_encoding().unwrap();
            command_buffer
                .encode_signal_event(EventRef::from(&between), 1)
                .unwrap();
            let mut second = command_buffer.blit_command_encoder().unwrap();
            second
                .copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
                .unwrap();
            second.end_encoding().unwrap();
            let gone = Arc::new(AtomicBool::new(false));
            let check: SubmissionCheck = {
                let gone = Arc::clone(&gone);
                let id = member.object_id();
                Arc::new(move |submission: &Submission<'_>| {
                    let uses = submission
                        .used_resources()
                        .buffers
                        .iter()
                        .any(|buffer| buffer.object_id() == id);
                    (uses && gone.load(Ordering::SeqCst)).then(|| NonZeroU64::new(8).unwrap())
                })
            };
            command_buffer.set_submission_check(check).unwrap();
            command_buffer.commit().unwrap();
            drop(rebind_to(&member, B));
            gone.store(detached, Ordering::SeqCst);
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(read_buffer(&ran, 0, 16).unwrap(), [0xa5; 16], "A ran");
            assert_eq!(between.signaled_value(), 1, "the signal between them");
            let copied = read_buffer(&copied, 0, 16).unwrap();
            (command_buffer.status(), command_buffer.error(), copied)
        };
        assert_eq!(
            split(true),
            (
                CommandBufferStatus::Error,
                Some(CommandBufferError::Injected(8)),
                vec![0x5a; 16]
            ),
            "B ran nothing"
        );
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(
            !notices.contains_key(&Degradation::EncodedUseRebound),
            "announced EncodedUseRebound for a submission its check failed with code 8"
        );
        assert_eq!(
            split(false),
            (CommandBufferStatus::Completed, None, node(B)),
            "B encoded again copied the new node"
        );
        assert_made_once(messages, &[], &shared_kinds(device));
        eprintln!("REBIND-REACH split: A ran, B's check's code 8 kept; unchecked, B encoded again");
    });
}

/// How [`rebound_after_a_failure`]'s command buffer fails X's submission with code 8.
#[derive(Clone, Copy)]
enum FirstFailed {
    /// As the consumer's emulated detach does: a submission check fails what names X. X's
    /// submission comes first, the fill's after it.
    Check,
    /// As `commit_failed` does, from X's encoder on (`FailedEncoders::From`). The fill's
    /// submission comes first, before that encoder.
    CommitFailed,
}

/// A command buffer split by an event signal into two of Metal's submissions, a blit that copies
/// out of X and a fill of Y, in the order `how` says, committed behind a wait; X's submission fails
/// with code 8 as `how` says, and Y is rebound while it waits. Metal fails X's submission with code
/// 8 and runs the fill over Y's new pages. The runtime keeps code 8 for the command buffer, and the
/// fill, encoded again at the rebind, writes Y's new pages, with no notice: it once ran nothing,
/// announced (`Degradation::EncodedUseRebound`), and before that a command buffer that had failed
/// already failed the fill silently.
fn rebound_after_a_failure(how: FirstFailed) {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let Some((x, _)) = member_over(device, A) else {
            return;
        };
        let Some((y, _)) = member_over(device, A) else {
            return;
        };
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let event = device.new_shared_event().unwrap();
        let between = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let copy = || {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&x, 0, &copied, 0, 16)
                .unwrap();
            blit.end_encoding().unwrap();
        };
        let fill = || {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&y, 0..16, 0x77).unwrap();
            blit.end_encoding().unwrap();
        };
        let signal = || {
            command_buffer
                .encode_signal_event(EventRef::from(&between), 1)
                .unwrap();
        };
        match how {
            FirstFailed::Check => {
                copy();
                signal();
                fill();
                let id = x.object_id();
                let check: SubmissionCheck = Arc::new(move |submission: &Submission<'_>| {
                    let uses = submission
                        .used_resources()
                        .buffers
                        .iter()
                        .any(|buffer| buffer.object_id() == id);
                    uses.then(|| NonZeroU64::new(8).unwrap())
                });
                command_buffer.set_submission_check(check).unwrap();
                command_buffer.commit().unwrap();
            }
            FirstFailed::CommitFailed => {
                fill();
                signal();
                copy();
                let mut failure = CommitFailure::new(8);
                failure.encoders = FailedEncoders::From(1);
                command_buffer.commit_failed(failure).unwrap();
            }
        }
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(
            !notices.contains_key(&Degradation::EncodedUseRebound),
            "announced before the rebind"
        );
        drop(rebind_to(&y, B));
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
        assert_eq!(
            command_buffer.error(),
            Some(CommandBufferError::Injected(8)),
            "the first code kept"
        );
        assert_eq!(
            read_buffer(&copied, 0, 16).unwrap(),
            [0x5a; 16],
            "X's copy ran"
        );
        assert_eq!(between.signaled_value(), 1, "the signal between them");
        assert_eq!(
            read_buffer(&y, 0, 16).unwrap(),
            [0x77; 16],
            "the fill ran over Y's new pages"
        );
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(!notices.contains_key(&Degradation::EncodedUseRebound));
    });
}

/// [`rebound_after_a_failure`], X's submission failed by the consumer's submission check.
#[test]
fn a_rebound_submission_after_a_checks_code_8_runs_over_the_new_pages() {
    rebound_after_a_failure(FirstFailed::Check);
    eprintln!("REBIND-REACH after a check: code 8 kept; the rebound fill ran over the new pages");
}

/// [`rebound_after_a_failure`], X's encoder failed by `commit_failed`, the fill before it.
#[test]
fn a_rebound_submission_before_commit_faileds_encoders_runs_over_the_new_pages() {
    rebound_after_a_failure(FirstFailed::CommitFailed);
    eprintln!(
        "REBIND-REACH after commit_failed: code 8 kept; the rebound fill ran over the new pages"
    );
}

/// A command buffer whose first batch was submitted before a rebind (its wait for an event after
/// the walker's encoder splits it) keeps the table and pages that submission took for its later
/// batch, which runs after the rebind: its blit, which names the buffer, copies the old node, and
/// its walker, which made the buffer resident, reads it, as the first batch's did (table 3: a
/// running command buffer reads the old pages). The old pages' deallocator waits for the command
/// buffer; a command buffer encoded after the rebind reads the new pages.
#[test]
fn a_command_buffer_first_submitted_before_a_rebind_keeps_its_pages_in_every_later_batch() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let Some((member, old)) = member_over(device, A) else {
            return;
        };
        let (binding, offset, length) = output(&walker);
        let state = pipeline_state(device, &walker).unwrap();
        let mut holding = CaseResources::new(device, &walker).unwrap();
        let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        holding.argument_buffers.insert(field, member.clone());
        let first = bound_buffers(device, &walker.compute, &holding).unwrap();
        let mut later = first.clone();
        let fresh = read_buffer(&first[&binding], 0, first[&binding].length()).unwrap();
        later.insert(binding, buffer_with(device, &fresh).unwrap());
        let copied = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &first, &holding).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &later, &holding).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let deadline = Instant::now() + WAIT;
        while command_buffer.kernel_start_time() == 0.0 {
            assert!(
                Instant::now() < deadline,
                "the first batch was never submitted"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(rebind_to(&member, B));
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            old.runs.load(Ordering::SeqCst),
            0,
            "the old pages went while a submitted command buffer held them"
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let read = |buffers: &std::collections::BTreeMap<u32, Buffer>| {
            read_buffer(&buffers[&binding], offset, length).unwrap()
        };
        assert!(read(&first) == outputs.a, "the first batch's walk");
        assert!(read(&later) == outputs.a, "the later batch's walk");
        assert_eq!(read_buffer(&copied, 0, 16).unwrap(), node(A), "its blit");
        deallocated(&old);
        drop(holding);
        let after = walk(
            device,
            &queue,
            &walker,
            &member,
            Uses::Resident,
            false,
            || {},
        );
        assert_eq!(after.command_buffer.error(), None);
        assert!(
            after.produced == outputs.b,
            "a command buffer encoded after"
        );
        eprintln!("REBIND-REACH submitted: the later batch kept the first submission's pages");
    });
}

/// A no-copy buffer released while a command buffer whose first batch was submitted holds the
/// table that lists it keeps its pages, its deallocator waiting for that command buffer to
/// complete (the submitted batches hold their table); one released beside it, which no submitted
/// table lists, has its deallocator run while that command buffer still waits. (The G344 re-land's
/// tests marked a table by hand and ran a spinning kernel on a host-written flag, which an Apple
/// GPU never saw; an event wait between two batches holds the command buffer here.)
#[test]
fn a_released_buffer_s_pages_wait_for_the_submitted_command_buffer_whose_table_lists_it() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let outputs = Outputs::new(device, &queue, &walker);
        let Some((listed, listed_seen)) = member_over(device, A) else {
            return;
        };
        gpu_read(&queue, &listed, 0..16);
        let (binding, offset, length) = output(&walker);
        let state = pipeline_state(device, &walker).unwrap();
        let mut holding = CaseResources::new(device, &walker).unwrap();
        let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        holding.argument_buffers.insert(field, listed.clone());
        let buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
        drop(holding.argument_buffers.remove(&field));
        let unrelated = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&unrelated, 0..16, 0x3c).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let deadline = Instant::now() + WAIT;
        while command_buffer.kernel_start_time() == 0.0 {
            assert!(
                Instant::now() < deadline,
                "the first batch was never submitted"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(listed);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            listed_seen.runs.load(Ordering::SeqCst),
            0,
            "ran though a submitted command buffer's table listed it"
        );
        // Beside it: a buffer listed after that submission, released; read on another queue,
        // since this one's next command buffer waits behind the held one.
        let Some((beside, beside_seen)) = member_over(device, B) else {
            return;
        };
        let other = device.new_command_queue().unwrap();
        gpu_read(&other, &beside, 0..16);
        drop(beside);
        deallocated(&beside_seen);
        assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
        assert_eq!(listed_seen.runs.load(Ordering::SeqCst), 0);
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let produced = read_buffer(&buffers[&binding], offset, length).unwrap();
        assert!(produced == outputs.a, "its walk read the listed pages");
        assert_eq!(read_buffer(&unrelated, 0, 16).unwrap(), [0x3c; 16]);
        deallocated(&listed_seen);
        assert_eq!(listed_seen.runs.load(Ordering::SeqCst), 1);
        eprintln!("REBIND-REACH released: the pages waited for the submitted table's holder");
    });
}
