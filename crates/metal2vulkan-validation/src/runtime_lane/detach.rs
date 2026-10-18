//! Detaching a no-copy buffer or heap from its host pages (`detach_backing`), on the device the cfg
//! selects, under the validation layer. The rule is Metal's as the consumer measured it on the M3
//! Max (macOS 26.7, the rebind probe's tables 1 and 3):
//!
//! - **Pending work** (table 1: committed before the detach, first submitted after). A command
//!   buffer whose command names the buffer (bound and read, case 1; bound and walked, case 2; a
//!   blit's operand, case 5) or whose encoder made it resident (`use_resources`, case 3;
//!   `use_heap`, case 3h) fails `CommandBufferError::DetachedBacking`, Metal's code 8, and runs
//!   nothing; its signal fires, its completed handler runs, and the queue runs on. One that
//!   reaches the buffer unnamed alone (case 4) reads 0 there and completes. No degradation is
//!   announced: the failure is Metal's.
//! - **Submitted work** (table 3: a command buffer submitted and not yet complete). A buffer's
//!   detach blocks until each such command buffer that took it completed, then answers `false`
//!   and detaches nothing; a heap's answers `false` at once; a buffer such a command buffer does
//!   not take is detached at once.
//! - **Encoded while detached.** A command that names a detached buffer records nothing, nor does
//!   any after it in its encoder (blit, dispatch, draw), and its command buffer fails
//!   `DetachedBacking` and runs nothing; where a rebind gave the buffer pages again before its
//!   first submission, the rebind encodes a compute or blit encoder again over them, and the
//!   command buffer completes over the new pages, as Metal's (`EncodedUseRebound` before). The
//!   layer reports nothing: no Vulkan command takes the detached buffer's stand-in.
//! - **Racing a commit.** A detach on one thread while another commits a command buffer whose
//!   command names the buffer runs the old pages' deallocator in one of the two calls, wherever
//!   the device's recorder replays the commands, and announces nothing.
//! - **No memory.** A detach places no memory of the bytes it detaches: once the device's
//!   stand-in is made, detaching a large heap, and a large buffer, allocates no device memory.
//! - **Textures**: `detach_textures.rs`.

use super::degrade_notices::{
    assert_made_once, assert_once_allowing, on_device as on_device_noticed,
};
use super::host_memory::{gpu_read, on_device, recording, Mapping, Seen};
use super::no_copy_heaps::shared;
use super::rebind_reach::{
    buffer_len, buffer_over, deallocated, heap_member, kept_member_over, member_over, node, pages,
    rebind_heap_to, rebind_to, walk, Outputs, Uses, Walked, A, B, POINTER_WALKER,
};
use super::shared_kinds;
use super::{
    bound_buffers, buffer_with, encode_case, pipeline_state, public_case, read_buffer,
    CaseResources, WAIT,
};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferError, CommandBufferStatus, CommandQueue,
    Degradation, Device, Error, EventRef, Heap, HeapDescriptor, HeapType, HostMemory, Invalid,
    LoadAction, PipelineOptions, Problem, RebindProblem, Region, RenderPassDescriptor,
    ResourceOptions, SharedEvent, StoreAction, TextureDescriptor,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long a detach that must wait is given to return early, wrongly.
const EARLY: Duration = Duration::from_millis(100);

/// The bytes of the large heap and buffer the memory row detaches.
const LARGE: usize = 64 << 20;

/// `command_buffer` failed `DetachedBacking`, Metal's code 8.
fn detached_backing(command_buffer: &CommandBuffer, name: &str) {
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Error,
        "{name}"
    );
    assert_eq!(
        command_buffer.error(),
        Some(CommandBufferError::DetachedBacking),
        "{name}"
    );
    assert_eq!(CommandBufferError::DetachedBacking.metal_code(), 8);
}

/// A walk failed `DetachedBacking`: its signal fired, its handler ran, and a command buffer
/// behind it on `queue` runs.
fn failed(device: &Device, queue: &CommandQueue, walked: Walked, name: &str) {
    detached_backing(&walked.command_buffer, name);
    assert_eq!(walked.signaled, 1, "{name}: its signal");
    assert!(walked.handled, "{name}: its handler");
    let after = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let behind = queue.command_buffer().unwrap();
    let mut blit = behind.blit_command_encoder().unwrap();
    blit.fill_buffer(&after, 0..16, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    behind.commit().unwrap();
    behind.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(behind.error(), None, "{name}: the command buffer behind it");
    assert_eq!(read_buffer(&after, 0, 16).unwrap(), [0x5a; 16]);
}

/// Detach `buffer`, which must detach at once.
fn detach(buffer: &Buffer) {
    assert_eq!(buffer.detach_backing(), Ok(true));
}

/// Detach `heap`, which must detach at once.
fn detach_heap(heap: &Heap) {
    assert_eq!(heap.detach_backing(), Ok(true));
}

/// Table 1, cases 1, 3, 3h and 5 (detach): a pending command buffer whose command names a
/// buffer detached since (bound and read, a blit's operand), or whose encoder made it resident,
/// fails `DetachedBacking` (code 8) at its first submission and runs nothing; its signal fires,
/// its handler runs, the queue runs on, and nothing is announced. Buffer and heap forms; each
/// given pages again by a rebind. (Case 2 binds the member as the walker's output, which a
/// detached buffer's `None` contents cannot be read back from.)
#[test]
fn table_1_a_pending_command_buffer_holding_a_detached_buffer_fails_with_code_8() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let Some((member, old)) = member_over(device, A) else {
            return;
        };
        let held = |member: &Buffer, uses: Uses<'_>, between: &dyn Fn()| {
            walk(device, &queue, &walker, member, uses, true, between)
        };
        let blit = held(&member, Uses::Blit, &|| detach(&member));
        failed(device, &queue, blit, "5: a blit's operand");
        deallocated(&old);
        rebind_to(&member, A);
        let resident = held(&member, Uses::Resident, &|| detach(&member));
        failed(device, &queue, resident, "3: resident");
        rebind_to(&member, A);
        let steps_seen = Arc::new(Seen::default());
        let steps = buffer_over(
            device,
            pages(buffer_len(), 0, &2u32.to_le_bytes()),
            &steps_seen,
        )
        .unwrap();
        let plain = buffer_with(device, &node(A)).unwrap();
        let bound = held(&plain, Uses::Steps(&steps), &|| detach(&steps));
        failed(device, &queue, bound, "1: bound and read");

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        let cases = [
            ("heap 5", &placed, Uses::Blit),
            ("heap 3", &placed, Uses::Resident),
            ("3h", &placed, Uses::Heap(&heap)),
            ("heap 1", &plain, Uses::Steps(&placed)),
        ];
        for (name, member, uses) in cases {
            let walked = held(member, uses, &|| detach_heap(&heap));
            failed(device, &queue, walked, name);
            rebind_heap_to(&heap, A);
        }
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(
            !notices.contains_key(&Degradation::EncodedUseRebound),
            "announced EncodedUseRebound for a detach"
        );
        eprintln!("DETACH table 1: named and resident uses fail with code 8, buffer and heap");
    });
}

/// Table 1, case 4 (detach): a pending command buffer that reaches a detached buffer unnamed
/// alone reads 0 there and completes with no error. Buffer and heap forms.
#[test]
fn table_1_a_pending_command_buffer_reaching_a_detached_buffer_unnamed_reads_zero() {
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
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        gpu_read(&queue, &member, 0..16);
        assert!(held(&member, &|| {}) == outputs.a, "4: control");
        let detached = held(&member, &|| detach(&member));
        assert!(detached == outputs.zero, "4: detached");

        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        let other = heap.new_buffer_at_offset(256, shared(), 0).unwrap();
        gpu_read(&queue, &other, 0..4);
        assert!(held(&placed, &|| {}) == outputs.a, "heap 4: control");
        let detached = held(&placed, &|| detach_heap(&heap));
        assert!(detached == outputs.zero, "heap 4: detached");
        eprintln!("DETACH table 1 unnamed: 0 once detached, buffer and heap");
    });
}

/// A command that names a detached buffer records nothing, nor does any after it in its
/// encoder or its command buffer's later encoders: a blit's copy (the fills around it unrun) and
/// a dispatch, each failing `DetachedBacking`; the old pages' deallocator ran in the detach,
/// which nothing held, and `contents` answers `None`. Encoded while detached and given pages
/// again before its first submission, a blit encoder is encoded again by the rebind (It failed
/// `EncodedUseRebound` before) and its copy reads the new pages, with no notice. A command buffer
/// encoded after the rebind reads the new pages. A buffer of the runtime's memory, or a heap's
/// buffer, is not detached (`Invalid`, `NotHostMemory`).
#[test]
fn a_command_naming_a_detached_buffer_records_nothing_and_its_command_buffer_runs_nothing() {
    on_device_noticed(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let Some((member, old)) = member_over(device, A) else {
            return;
        };
        gpu_read(&queue, &member, 0..16);
        assert!(member.contents().is_some());
        detach(&member);
        deallocated(&old);
        assert_eq!(member.contents(), None, "a detached buffer's contents");
        detach(&member);

        let before = buffer_with(device, &[0; 16]).unwrap();
        let copied = buffer_with(device, &[0x5a; 16]).unwrap();
        let after = buffer_with(device, &[0; 16]).unwrap();
        let later = buffer_with(device, &[0; 16]).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&before, 0..16, 0x11).unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.fill_buffer(&after, 0..16, 0x22).unwrap();
        blit.end_encoding().unwrap();
        let mut next = command_buffer.blit_command_encoder().unwrap();
        next.fill_buffer(&later, 0..16, 0x33).unwrap();
        next.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        detached_backing(&command_buffer, "a blit");
        for (name, buffer, held) in [
            ("the fill before", &before, [0; 16]),
            ("the copy", &copied, [0x5a; 16]),
            ("the fill after", &after, [0; 16]),
            ("the later encoder", &later, [0; 16]),
        ] {
            assert_eq!(read_buffer(buffer, 0, 16).unwrap(), held, "{name} ran");
        }

        let plain = buffer_with(device, &node(A)).unwrap();
        let walked = walk(
            device,
            &queue,
            &walker,
            &plain,
            Uses::Steps(&member),
            false,
            || {},
        );
        detached_backing(&walked.command_buffer, "a dispatch");

        // Encoded while detached, given pages before its first submission.
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let notices = assert_made_once(messages, &[], &shared_kinds(device));
        assert!(!notices.contains_key(&Degradation::EncodedUseRebound));
        let reencoded = device.work_counts().encoders_reencoded;
        rebind_to(&member, B);
        // The rebind encodes the blit encoder again over the new pages before its first
        // submission: the copy reads them, as Metal's does.
        assert!(
            device.work_counts().encoders_reencoded > reencoded,
            "the rebind encoded the blit encoder again"
        );
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        assert_eq!(read_buffer(&copied, 0, 16).unwrap(), node(B), "the copy");
        assert_made_once(messages, &[], &shared_kinds(device));
        assert!(member.contents().is_some());
        assert_eq!(gpu_read(&queue, &member, 0..16), node(B), "attached again");

        // Refused: no host pages of its own.
        let refused = Err(Error::Invalid(Invalid::new(
            "detach_backing",
            Problem::Rebind(RebindProblem::NotHostMemory),
        )));
        let ordinary = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        assert_eq!(ordinary.detach_backing(), refused);
        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        assert_eq!(placed.detach_backing(), refused);
        let descriptor = HeapDescriptor::new(HeapType::Placement, 1 << 16, shared());
        let own = device.new_heap(&descriptor).unwrap();
        assert_eq!(own.detach_backing(), refused);
        drop(heap);
        eprintln!(
            "DETACH encoded: nothing recorded runs; given pages again, encoded again over them"
        );
    });
}

/// A render pass whose draw takes a detached vertex buffer records none of its draws: its
/// command buffer fails `DetachedBacking`, its target keeps its texels (the pass's clear unrun),
/// and the layer reports nothing. The fixture's pipeline (`rt6_restart.ll`) reads the corners
/// from vertex buffer 0.
#[test]
fn a_render_pass_drawing_from_a_detached_buffer_runs_nothing() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
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
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let side = 8;
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let target = device.new_texture(&target).unwrap();
        let old = vec![0x11; (side * side * 4) as usize];
        let region = Region::new_2d(0, 0, side, side);
        target
            .replace_region(region, 0, 0, &old, side * 4, 0)
            .unwrap();
        let corners: Vec<u8> = [-1.0f32, -1.0, 3.0, -1.0, -1.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let seen = Arc::new(Seen::default());
        let Some(detached) = buffer_over(device, pages(buffer_len(), 0, &corners), &seen) else {
            return;
        };
        let plain = buffer_with(device, &corners).unwrap();
        detach(&detached);
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Clear;
        color.clear_color = ClearColor::new(0.5, 0.0, 0.0, 1.0);
        color.store_action = StoreAction::Store;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&plain), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&detached), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(&plain), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        detached_backing(&command_buffer, "a draw");
        let mut texels = vec![0; old.len()];
        target
            .get_bytes(&mut texels, side * 4, 0, region, 0, 0)
            .unwrap();
        assert!(texels == old, "the pass ran");
        eprintln!("DETACH render: a draw from a detached buffer ran nothing");
    });
}

/// A command buffer whose first batch was submitted (an event wait after its first encoder
/// splits it), held by that wait.
struct Running {
    command_buffer: CommandBuffer,
    event: SharedEvent,
}

impl Running {
    /// `encode` encodes its first encoder; a blit after the wait fills a buffer of its own.
    fn new(device: &Device, queue: &CommandQueue, encode: impl FnOnce(&CommandBuffer)) -> Running {
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        encode(&command_buffer);
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let filled = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&filled, 0..16, 0x3c).unwrap();
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
        Running {
            command_buffer,
            event,
        }
    }

    /// Let it run to its end.
    fn finish(&self) {
        self.event.set_signaled_value(1);
        self.command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(self.command_buffer.error(), None);
    }
}

/// `detach`, called on another thread while `running` waits: it must not return before the
/// command buffer is let run, and answers `false` once it completed.
fn waits_then_refuses(running: &Running, detach: impl FnOnce() -> Result<bool, Error> + Send) {
    let returned = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let detaching = scope.spawn(|| {
            let answer = detach();
            returned.store(true, Ordering::SeqCst);
            answer
        });
        std::thread::sleep(EARLY);
        assert!(
            !returned.load(Ordering::SeqCst),
            "the detach did not wait for the submitted command buffer"
        );
        running.event.set_signaled_value(1);
        assert_eq!(detaching.join().unwrap(), Ok(false));
        assert_eq!(
            running.command_buffer.status(),
            CommandBufferStatus::Completed,
            "the detach answered before the command buffer completed"
        );
    });
}

/// Table 3 (a command buffer submitted and not yet complete): a buffer its command named, or
/// its encoder made resident, is detached only once it completed: the detach blocks until then
/// and answers `false`, detaching nothing (the buffer reads its pages after). A heap one of its
/// buffers was named in answers `false` at once, the command buffer still running. A buffer it
/// does not take is detached at once.
#[test]
fn table_3_a_detach_waits_for_a_submitted_command_buffer_then_refuses() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let walker = public_case(POINTER_WALKER).unwrap();
        let Some((member, _)) = member_over(device, A) else {
            return;
        };
        let copied = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        let named = Running::new(device, &queue, |command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&member, 0, &copied, 0, 16)
                .unwrap();
            blit.end_encoding().unwrap();
        });
        // A buffer it does not take: at once.
        let (unrelated, _) = member_over(device, B).unwrap();
        detach(&unrelated);
        assert_eq!(
            named.command_buffer.status(),
            CommandBufferStatus::Committed
        );
        waits_then_refuses(&named, || member.detach_backing());
        assert_eq!(read_buffer(&copied, 0, 16).unwrap(), node(A));
        assert!(member.contents().is_some(), "detached though refused");
        assert_eq!(gpu_read(&queue, &member, 0..16), node(A));

        // Resident: the walker reaches it through what its encoder made resident.
        let state = pipeline_state(device, &walker).unwrap();
        let mut holding = CaseResources::new(device, &walker).unwrap();
        let [field] = holding.argument_buffers.keys().copied().collect::<Vec<_>>()[..] else {
            panic!("{POINTER_WALKER}: one argument-buffer buffer");
        };
        holding.argument_buffers.insert(field, member.clone());
        let buffers = bound_buffers(device, &walker.compute, &holding).unwrap();
        let resident = Running::new(device, &queue, |command_buffer| {
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encode_case(&mut encoder, &state, &walker.compute, &buffers, &holding).unwrap();
            encoder.end_encoding().unwrap();
        });
        waits_then_refuses(&resident, || member.detach_backing());
        drop(holding);
        detach(&member);

        // A heap: at once.
        let Some((heap, placed)) = heap_member(device) else {
            return;
        };
        let in_heap = Running::new(device, &queue, |command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&placed, 0, &copied, 0, 16)
                .unwrap();
            blit.end_encoding().unwrap();
        });
        assert_eq!(heap.detach_backing(), Ok(false));
        assert_eq!(
            in_heap.command_buffer.status(),
            CommandBufferStatus::Committed,
            "the heap's detach waited"
        );
        in_heap.finish();
        assert_eq!(read_buffer(&copied, 0, 16).unwrap(), node(A));
        detach_heap(&heap);
        eprintln!("DETACH table 3: a buffer waits then refuses, a heap refuses at once");
    });
}

/// A heap of `LARGE` bytes over fresh host memory, and a buffer placed at its start.
fn large_heap(device: &Device) -> Option<(Heap, Buffer)> {
    let mapping = Mapping::anonymous(LARGE);
    let pointer = mapping.pointer();
    let descriptor = HeapDescriptor::new(HeapType::Placement, LARGE as u64, shared());
    let seen = Arc::new(Seen::default());
    // SAFETY: the deallocator releases the mapping once the heap's pages go.
    let made = unsafe {
        device.new_heap_with_bytes_no_copy(
            HostMemory::new(pointer),
            &descriptor,
            Some(recording(&seen, mapping)),
            None,
        )
    };
    let heap = match made {
        Ok(heap) => heap,
        Err(error) => {
            eprintln!("DETACH memory: no heap over host memory: {error:?}");
            return None;
        }
    };
    let placed = heap.new_buffer_at_offset(1 << 20, shared(), 0).unwrap();
    Some((heap, placed))
}

/// A detach places no memory of the bytes it detaches (the coordinator's lane row): once the
/// device's stand-in is made (a small buffer's detach), detaching a heap of 64 MiB, and a
/// no-copy buffer of 64 MiB, allocates no device memory, and a command buffer that names them
/// fails `DetachedBacking` with no device memory allocated for them either.
#[test]
fn a_detach_of_a_large_heap_allocates_no_device_memory() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let Some((small, _)) = member_over(device, A) else {
            return;
        };
        detach(&small);
        let Some((heap, placed)) = large_heap(device) else {
            return;
        };
        let mapping = Mapping::anonymous(LARGE);
        let pointer = mapping.pointer();
        let seen = Arc::new(Seen::default());
        // SAFETY: the deallocator releases the mapping once the buffer's pages go.
        let large = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(pointer),
                LARGE as u64,
                ResourceOptions::default(),
                Some(recording(&seen, mapping)),
            )
        }
        .unwrap();
        let copied = device.new_buffer(16, ResourceOptions::shared()).unwrap();
        // The copies' own work warmed: a command buffer like the failing ones, over plain bytes.
        let plain = buffer_with(device, &[0; 16]).unwrap();
        let warm = queue.command_buffer().unwrap();
        let mut blit = warm.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&plain, 0, &copied, 0, 16)
            .unwrap();
        blit.end_encoding().unwrap();
        warm.commit().unwrap();
        warm.wait_until_completed_for(WAIT).unwrap();
        let allocated = || device.work_counts().device_memory_allocated;
        let made = allocated();
        detach_heap(&heap);
        detach(&large);
        assert_eq!(allocated(), made, "a detach allocated device memory");
        deallocated(&seen);
        for (name, buffer) in [("the heap's buffer", &placed), ("the buffer", &large)] {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(buffer, 0, &copied, 0, 16)
                .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            detached_backing(&command_buffer, name);
        }
        assert_eq!(
            allocated(),
            made,
            "a command buffer naming a detached buffer allocated device memory"
        );
        eprintln!(
            "DETACH memory: 0 bytes allocated detaching {} MiB twice",
            LARGE >> 20
        );
    });
}

/// The rounds of [`a_detach_racing_a_commit_runs_the_deallocator_in_one_of_the_two_calls`], for
/// each form: two sweeps of the offsets.
const RACES: u32 = 640;

/// How far apart the detach and the commit of a race start, at most, either first.
const RACE_SPAN: Duration = Duration::from_micros(40);

/// How far one race's offset is from the last's.
const RACE_STEP: Duration = Duration::from_nanos(250);

/// When a race's two calls start, after its other thread is spawned.
const RACE_LEAD: Duration = Duration::from_micros(500);

/// Spin until `at`.
fn spin_until(at: Instant) {
    while Instant::now() < at {
        std::hint::spin_loop();
    }
}

/// A detach on one thread racing the commit, on another, of a command buffer whose blit names the
/// buffer: the old pages' deallocator runs once, in one of the two calls (the detach where it lets
/// go of the pages last, the commit where its encoder held them last), wherever the device's
/// recorder replays the command buffer's commands, and nothing is announced: with a keep-alive
/// and without one. The command buffer, which an event holds back until both calls returned,
/// then fails `DetachedBacking`. The races start the two calls up to 40 us apart, either first,
/// swept in steps of 0.25 us, so that many land inside each other.
#[test]
fn a_detach_racing_a_commit_runs_the_deallocator_in_one_of_the_two_calls() {
    for (form, kept) in [("no keep-alive", false), ("keep-alive", true)] {
        on_device_noticed(|device, messages| {
            let queue = device.new_command_queue().unwrap();
            let event = device.new_shared_event().unwrap();
            let made = || match kept {
                true => kept_member_over(device, A).map(|(buffer, seen, _)| (buffer, seen)),
                false => member_over(device, A),
            };
            let detacher = std::thread::current().id();
            let (mut in_detach, mut in_commit, mut late, mut elsewhere) = (0, 0, 0, 0);
            let before = device.work_counts();
            let span = RACE_SPAN.as_nanos() as i64;
            for round in 0..RACES {
                let Some((buffer, seen)) = made() else {
                    return;
                };
                let value = u64::from(round) + 1;
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .encode_wait_for_event(EventRef::from(&event), value)
                    .unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&buffer, 0..16, 0x5a).unwrap();
                blit.end_encoding().unwrap();
                // The commit's start less the detach's, swept over the span.
                let step = RACE_STEP.as_nanos() as i64;
                let offset = (i64::from(round) * step) % (2 * span) - span;
                let start = Instant::now() + RACE_LEAD;
                let late_by = |ns: i64| Duration::from_nanos(ns.max(0) as u64);
                let (commit_at, detach_at) = (start + late_by(offset), start + late_by(-offset));
                let committer = std::thread::scope(|scope| {
                    let committing = scope.spawn(|| {
                        spin_until(commit_at);
                        command_buffer.commit().unwrap();
                        std::thread::current().id()
                    });
                    spin_until(detach_at);
                    detach(&buffer);
                    committing.join().unwrap()
                });
                let ran = *seen.thread.lock().unwrap();
                match (seen.runs.load(Ordering::SeqCst), ran) {
                    (1, Some(thread)) if thread == detacher => in_detach += 1,
                    (1, Some(thread)) if thread == committer => in_commit += 1,
                    (0, _) => late += 1,
                    _ => elsewhere += 1,
                }
                event.set_signaled_value(value);
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                detached_backing(&command_buffer, form);
                deallocated(&seen);
                assert_eq!(seen.runs.load(Ordering::SeqCst), 1, "{form}: ran once");
            }
            let inline = device.work_counts().command_buffers_replayed_inline
                - before.command_buffers_replayed_inline;
            eprintln!(
                "DETACH race {} ({form}): {RACES} races, the deallocator ran in the detach {in_detach}, \
                 in the commit {in_commit}, after both returned {late}, on another thread \
                 {elsewhere}; {inline} command buffers replayed off the recorder's thread",
                device.name()
            );
            assert_eq!(
                (late, elsewhere),
                (0, 0),
                "{form}: the deallocator ran outside the detach and the commit"
            );
            assert_once_allowing(&messages.take_notices(), &[], &shared_kinds(device));
        });
    }
}
