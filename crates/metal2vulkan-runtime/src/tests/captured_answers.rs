//! Render encoders the device's resolver encodes (`render/captured.rs`, `resolver.rs`): a frame
//! the memo answers draws the bytes the same frame encoded at the call drew, its calls' refusals
//! land at the call, and a call the memo does not key takes the encoder back to the call. Each
//! runs with the memo taught by an earlier frame, and untaught
//! (`DeviceOptions::answer_untaught`), where every call it keys is answered `Ok` unseen, so a key
//! that leaves out what decides an answer shows as `WorkCounts::answered_calls_refused`. An indexed
//! draw that reads its indices through a copy, which it announces, is encoded at the call. No
//! test's resolver delivers a notice (`WorkCounts::resolver_notices`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::Scratch;
use crate::argument_table::BUFFER_SLOTS;
use crate::device::lock;
use crate::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, DebugMessage, Degradation,
    Device, DeviceKind, DeviceOptions, Error, LoadAction, MessageSink, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ShaderCacheOptions, Size,
    StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage, TriangleFillMode};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;
use std::time::Duration;

/// The side of the target.
const SIDE: u64 = 8;

/// How long a command buffer may take to complete.
const WAIT: Duration = Duration::from_secs(30);

/// The rectangle each frame draws, corner to corner in clip space: the target's left half.
const RECT: [f32; 4] = [-1.0, -1.0, 0.0, 1.0];

/// A device of the kind the cfg selects, its memo `untaught` or not, its shader cache in
/// `cache`.
fn device(cache: &Path, untaught: bool) -> Device {
    let options = options(cache, untaught);
    Device::system_default_with(&options).unwrap()
}

/// The degrade notices a sink was handed, each with the thread that delivered it.
type Heard = Arc<Mutex<Vec<(Degradation, ThreadId)>>>;

/// A device as [`device`]'s, taught, announcing `each` at every occurrence to a sink that keeps
/// every degrade notice.
fn listening(cache: &Path, each: BTreeSet<Degradation>) -> (Device, Heard) {
    let heard = Heard::default();
    let kept = Arc::clone(&heard);
    let sink = MessageSink::new(move |message: &DebugMessage| {
        if let Some(notice) = &message.degradation {
            let thread = std::thread::current().id();
            lock(&kept).push((notice.degradation, thread));
        }
    });
    let options = DeviceOptions {
        messages: Some(sink),
        announce_each: each,
        ..options(cache, false)
    };
    (Device::system_default_with(&options).unwrap(), heard)
}

/// [`device`]'s options.
fn options(cache: &Path, untaught: bool) -> DeviceOptions {
    DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        answer_untaught: untaught,
        ..DeviceOptions::default()
    }
}

/// A state that draws `tob_rect` and `tob_solid` into `RGBA8Unorm`.
fn state(device: &Device) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/tob_renders.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library = device.new_library_with_compile_library(library);
    let vertex = library.new_function_with_name("tob_rect").unwrap();
    let fragment = library.new_function_with_name("tob_solid").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A private `RGBA8Unorm` render target.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// A pass that clears `target` to black and stores it.
fn pass(target: &Texture) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
    pass
}

/// The bytes of `values`.
fn bytes_of(values: [f32; 4]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The target's texels, copied into a shared buffer by a blit of `command_buffer`.
fn copied(command_buffer: &CommandBuffer, target: &Texture) -> Buffer {
    let device = command_buffer.device();
    let back = device
        .new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &back,
        0,
        SIDE * 4,
        SIDE * SIDE * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    back
}

/// The bytes of `back`.
fn texels(back: &Buffer) -> Vec<u8> {
    let contents = back.contents().unwrap();
    // SAFETY: the buffer's `length` bytes, which no command buffer writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), back.length() as usize) }.to_vec()
}

/// One frame: a command buffer of two passes over `target`, each drawing the rectangle in
/// `colors[i]`, the second's draw then `fill`'s mode set where given, and a blit of the target's
/// texels; committed, completed, and its texels.
fn frame(
    device: &Device,
    (state, target): (&RenderPipelineState, &Texture),
    colors: [[f32; 4]; 2],
    fill: Option<TriangleFillMode>,
) -> Vec<u8> {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for (at, color) in colors.into_iter().enumerate() {
        let mut encoder = command_buffer
            .render_command_encoder(&pass(target))
            .unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        encoder.set_vertex_bytes(&bytes_of(RECT), 0).unwrap();
        encoder.set_fragment_bytes(&bytes_of(color), 0).unwrap();
        if let (1, Some(fill)) = (at, fill) {
            encoder.set_triangle_fill_mode(fill).unwrap();
        }
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    let back = copied(&command_buffer, target);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    texels(&back)
}

/// How many frames [`taught`] makes at most.
const TEACHING: usize = 300;

/// Frames of `frame` until one of its `passes` passes is answered whole from the device's memo,
/// none taken back to the call; that frame's texels. The memo learns a draw once its own pipeline
/// is made (`device::unsettled`), which a device may make in the background while the draws take
/// its residual form, their frames completing meanwhile.
fn taught(device: &Device, passes: u64, mut frame: impl FnMut() -> Vec<u8>) -> Vec<u8> {
    for _ in 0..TEACHING {
        let before = device.work_counts();
        let texels = frame();
        let after = device.work_counts();
        let captured = moved(&before, &after, |c| c.render_encoders_captured);
        if captured == passes && moved(&before, &after, |c| c.render_encoders_demoted) == 0 {
            return texels;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("no frame answered whole from the memo in {TEACHING} frames");
}

/// `after`'s count less `before`'s.
fn moved(before: &WorkCounts, after: &WorkCounts, count: fn(&WorkCounts) -> u64) -> u64 {
    count(after) - count(before)
}

#[test]
fn a_captured_frame_draws_what_the_frame_at_the_call_drew() {
    for untaught in [false, true] {
        let scratch = Scratch::new(&format!("captured-frames-{untaught}"));
        let device = device(&scratch.0, untaught);
        let state = state(&device);
        let target = target(&device);
        let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
        // The first frames teach the memo ([`taught`]); the later ones are answered from it.
        let first = frame(&device, (&state, &target), colors, None);
        let second = taught(&device, 2, || {
            frame(&device, (&state, &target), colors, None)
        });
        assert_eq!(second, first, "{untaught}: the taught frame's texels");
        let green = [0, 255, 0, 255];
        let row = |y: usize| &first[y * SIDE as usize * 4..][..SIDE as usize * 4];
        for y in 0..SIDE as usize {
            assert_eq!(&row(y)[..4], &green, "{untaught}: the drawn half, row {y}");
            assert_eq!(
                &row(y)[row(y).len() - 4..],
                &[0, 0, 0, 255],
                "{untaught}: row {y}"
            );
        }
        let before = device.work_counts();
        for _ in 0..2 {
            let later = frame(&device, (&state, &target), colors, None);
            assert_eq!(later, first, "{untaught}: a captured frame's texels");
        }
        let after = device.work_counts();
        let captured = moved(&before, &after, |c| c.render_encoders_captured);
        assert_eq!(
            captured, 4,
            "{untaught}: each later frame's two passes captured"
        );
        assert_eq!(moved(&before, &after, |c| c.render_encoders_demoted), 0);
        assert!(
            moved(&before, &after, |c| c.calls_answered) >= 4 * 6,
            "{untaught}"
        );
        assert_eq!(moved(&before, &after, |c| c.answered_calls_refused), 0);
        assert_eq!(moved(&before, &after, |c| c.resolver_failures), 0);
        assert_eq!(device.work_counts().resolver_notices, 0, "{untaught}");
        assert_eq!(device.work_counts().resolver_drain_wakes, 0, "{untaught}");
    }
}

/// A device that encodes at the call (`DeviceOptions::encode_at_the_call`) captures no encoder of
/// the frames another device captures, starts no resolver, and draws the same.
#[test]
fn a_device_that_encodes_at_the_call_captures_nothing() {
    let scratch = Scratch::new("captured-at-the-call");
    let resolving = device(&scratch.0, false);
    let options = DeviceOptions {
        encode_at_the_call: true,
        ..options(&scratch.0, false)
    };
    let at_the_call = Device::system_default_with(&options).unwrap();
    let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
    let resolving_state = state(&resolving);
    let resolving_target = target(&resolving);
    let drawn = (&resolving_state, &resolving_target);
    let first = frame(&resolving, drawn, colors, None);
    let captured = taught(&resolving, 2, || frame(&resolving, drawn, colors, None));
    assert_eq!(captured, first, "the resolving device's captured frame");
    let at_the_call_state = state(&at_the_call);
    let at_the_call_target = target(&at_the_call);
    for at in 0..4 {
        let drawn = (&at_the_call_state, &at_the_call_target);
        let texels = frame(&at_the_call, drawn, colors, None);
        assert_eq!(texels, first, "frame {at} at the call");
    }
    let counts = at_the_call.work_counts();
    assert_eq!(counts.render_encoders_captured, 0);
    assert_eq!(
        counts.render_opens_unseen_passes, 0,
        "opens followed by a memo"
    );
    assert_eq!(counts.resolver_wakes, 0);
}

#[test]
fn a_call_the_memo_does_not_key_takes_its_encoder_back_to_the_call() {
    let scratch = Scratch::new("captured-taken-back");
    let device = device(&scratch.0, false);
    let state = state(&device);
    let target = target(&device);
    let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]];
    let first = frame(&device, (&state, &target), colors, None);
    let second = taught(&device, 2, || {
        frame(&device, (&state, &target), colors, None)
    });
    assert_eq!(second, first);
    let before = device.work_counts();
    let later = frame(
        &device,
        (&state, &target),
        colors,
        Some(TriangleFillMode::Fill),
    );
    let after = device.work_counts();
    assert_eq!(
        later, first,
        "the frame taken back to the call draws the same"
    );
    assert_eq!(moved(&before, &after, |c| c.render_encoders_captured), 2);
    assert_eq!(moved(&before, &after, |c| c.render_encoders_demoted), 1);
    assert_eq!(moved(&before, &after, |c| c.taken_back_render_state), 1);
    assert_eq!(moved(&before, &after, |c| c.resolver_failures), 0);
    // The take-back encoded what was handed in the resolver's place, or found it encoded: it
    // woke no resolver to wait for (`resolver.rs`).
    assert_eq!(moved(&before, &after, |c| c.resolver_drain_wakes), 0);
    // A draw that binds bytes of a length no key binds there is taken back by that difference.
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let before = device.work_counts();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder.set_vertex_bytes(&bytes_of(RECT), 0).unwrap();
    let mut longer = bytes_of(colors[1]);
    longer.resize(longer.len() * 2, 0);
    encoder.set_fragment_bytes(&longer, 0).unwrap();
    let wakes = device.work_counts().resolver_wakes;
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    // The draw took the encoder back: what was handed is encoded on this thread in the
    // resolver's place, or by the resolver already at work, and no sleeping resolver is woken to
    // wait for (`resolver.rs`).
    assert_eq!(
        device.work_counts().resolver_wakes,
        wakes,
        "the take-back woke the resolver"
    );
    encoder.end_encoding().unwrap();
    let back = copied(&command_buffer, &target);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    assert_eq!(texels(&back), first, "the frame of longer bytes");
    assert_eq!(moved(&before, &after, |c| c.render_encoders_demoted), 1);
    assert_eq!(
        moved(&before, &after, |c| c.taken_back_new_bytes_lengths),
        1
    );
    assert_eq!(device.work_counts().resolver_notices, 0);
    assert_eq!(device.work_counts().resolver_drain_wakes, 0);
}

#[test]
fn a_captured_call_s_argument_refusals_land_at_the_call() {
    let scratch = Scratch::new("captured-argument-refusals");
    let device = device(&scratch.0, false);
    let state = state(&device);
    let target = target(&device);
    let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
    let first = frame(&device, (&state, &target), colors, None);
    let second = taught(&device, 2, || {
        frame(&device, (&state, &target), colors, None)
    });
    assert_eq!(second, first);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let before = device.work_counts();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let slot = encoder.set_vertex_bytes(&bytes_of(RECT), BUFFER_SLOTS);
    assert!(matches!(slot, Err(Error::Invalid(_))), "{slot:?}");
    let long = encoder.set_fragment_bytes(&[0; 4097], 0);
    assert!(matches!(long, Err(Error::Invalid(_))), "{long:?}");
    encoder.set_vertex_bytes(&bytes_of(RECT), 0).unwrap();
    encoder.set_fragment_bytes(&bytes_of(colors[1]), 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let after = device.work_counts();
    assert_eq!(moved(&before, &after, |c| c.render_encoders_captured), 1);
    assert_eq!(moved(&before, &after, |c| c.render_encoders_demoted), 0);
    // A command buffer of one captured encoder commits on the resolver: committed at once.
    command_buffer.commit().unwrap();
    assert_ne!(command_buffer.status(), CommandBufferStatus::NotEnqueued);
    let late = command_buffer.add_completed_handler(Box::new(|_| {}));
    assert!(matches!(late, Err(Error::Invalid(_))), "{late:?}");
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let reading = queue.command_buffer().unwrap();
    let back = copied(&reading, &target);
    reading.commit().unwrap();
    reading.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(texels(&back), first);
    assert_eq!(device.work_counts().answered_calls_refused, 0);
    assert_eq!(device.work_counts().resolver_notices, 0);
    assert_eq!(device.work_counts().resolver_drain_wakes, 0);
}

/// A command buffer with a blit at the call before its captured pass commits without waiting for
/// the device's resolver (`CommandBuffer::commit`): handed to it behind the pass where it has work,
/// else made at the call, which finds nothing to wait for. Its texels are the frame's.
#[test]
fn a_command_buffer_with_a_blit_at_the_call_commits_without_waiting_for_the_resolver() {
    let scratch = Scratch::new("captured-blit-commit");
    let device = device(&scratch.0, false);
    let state = state(&device);
    let target = target(&device);
    let colors = [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
    let first = frame(&device, (&state, &target), colors, None);
    let second = taught(&device, 2, || {
        frame(&device, (&state, &target), colors, None)
    });
    assert_eq!(second, first);
    let filled = device.new_buffer(16, ResourceOptions::private()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    for _ in 0..4 {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&filled, 0..16, 7).unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_bytes(&bytes_of(RECT), 0).unwrap();
        encoder.set_fragment_bytes(&bytes_of(colors[1]), 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        assert_ne!(command_buffer.status(), CommandBufferStatus::NotEnqueued);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    let after = device.work_counts();
    assert_eq!(moved(&before, &after, |c| c.render_encoders_captured), 4);
    assert_eq!(moved(&before, &after, |c| c.render_encoders_demoted), 0);
    assert_eq!(
        moved(&before, &after, |c| c.resolver_drains_commits),
        0,
        "a commit waited for the resolver"
    );
    let reading = queue.command_buffer().unwrap();
    let back = copied(&reading, &target);
    reading.commit().unwrap();
    reading.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(texels(&back), first);
    assert_eq!(device.work_counts().resolver_notices, 0);
    assert_eq!(device.work_counts().resolver_drain_wakes, 0);
}

/// One frame of one pass over `target` drawing the rectangle in `color` through `count` indices of
/// `indices` from its byte 0, and a blit of the target's texels; committed, completed, and its
/// texels.
fn indexed_frame(
    device: &Device,
    (state, target): (&RenderPipelineState, &Texture),
    (indices, count): (&Buffer, u32),
    color: [f32; 4],
) -> Vec<u8> {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(target))
        .unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_bytes(&bytes_of(RECT), 0).unwrap();
    encoder.set_fragment_bytes(&bytes_of(color), 0).unwrap();
    encoder
        .draw_indexed_primitives(
            PrimitiveType::Triangle,
            count,
            IndexType::UInt16,
            indices,
            0,
            1,
            0,
            0,
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    let back = copied(&command_buffer, target);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    texels(&back)
}

/// An indexed draw whose positions run past its index buffer reads a zero-padded copy of its
/// indices and announces it (`draw_source.rs`), which the memo's key of the draw does not tell:
/// the draw takes its encoder back to the call, so the notice, asked for at every occurrence,
/// comes in the program's call at each such draw, and the same draw reading its indices in place
/// stays answered from the memo.
#[test]
fn an_indexed_draw_through_a_copy_announces_in_the_program_s_call() {
    let scratch = Scratch::new("captured-index-copies");
    let each = BTreeSet::from([Degradation::IsolatedIndexBytes]);
    let (device, heard) = listening(&scratch.0, each);
    let state = state(&device);
    let target = target(&device);
    // The rectangle's six vertices, in order.
    let indices: Vec<u8> = (0u16..6).flat_map(u16::to_le_bytes).collect();
    let indices = device
        .new_buffer_with_bytes(&indices, ResourceOptions::shared())
        .unwrap();
    let color = [0.0, 1.0, 0.0, 1.0];
    let in_place = indexed_frame(&device, (&state, &target), (&indices, 6), color);
    let again = taught(&device, 1, || {
        indexed_frame(&device, (&state, &target), (&indices, 6), color)
    });
    assert_eq!(again, in_place, "the taught frame");
    let before = device.work_counts();
    let answered = indexed_frame(&device, (&state, &target), (&indices, 6), color);
    let after = device.work_counts();
    assert_eq!(answered, in_place, "the frame answered from the memo");
    assert_eq!(moved(&before, &after, |c| c.render_encoders_captured), 1);
    assert_eq!(moved(&before, &after, |c| c.render_encoders_demoted), 0);
    let copies = |heard: &Heard| {
        (lock(heard).iter())
            .filter(|(kind, _)| *kind == Degradation::IsolatedIndexBytes)
            .map(|&(_, thread)| thread)
            .collect::<Vec<_>>()
    };
    assert!(copies(&heard).is_empty(), "{:?}", copies(&heard));
    // Nine positions: the last three, past the buffer, draw index 0 three times, a triangle that
    // covers nothing.
    for frame in 1..=3 {
        let before = device.work_counts();
        let padded = indexed_frame(&device, (&state, &target), (&indices, 9), color);
        let after = device.work_counts();
        assert_eq!(padded, in_place, "frame {frame}: the padded draw's texels");
        assert_eq!(moved(&before, &after, |c| c.render_encoders_captured), 1);
        assert_eq!(moved(&before, &after, |c| c.render_encoders_demoted), 1);
        let here = std::thread::current().id();
        assert_eq!(copies(&heard), vec![here; frame], "frame {frame}");
    }
    let before = device.work_counts();
    let answered = indexed_frame(&device, (&state, &target), (&indices, 6), color);
    let after = device.work_counts();
    assert_eq!(answered, in_place, "the frame answered from the memo again");
    assert_eq!(moved(&before, &after, |c| c.render_encoders_demoted), 0);
    assert_eq!(device.work_counts().resolver_notices, 0);
    assert_eq!(device.work_counts().resolver_drain_wakes, 0);
}
