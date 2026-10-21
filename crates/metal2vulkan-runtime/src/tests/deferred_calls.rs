//! A render encoder's calls lowered after they return (`RenderLoweringPolicy::Worker`,
//! `render/deferral.rs`) against inside each call (`AtCall`): each refusal the encoder checks at
//! the call (`render/call_checks.rs`) returns from the call under both policies, the same error,
//! and the command buffer then runs; a refusal only the lowering finds fails the command buffer,
//! naming the call (`CommandBufferError::CallRefused`), and the command buffer keeps the call's
//! error; a notice announced while a call is lowered names it, and a kind the device refuses while
//! lowering (`DeviceOptions::lowering_refuses`) refuses it; the bytes a call was lent are copied at
//! the call, and the objects it names are held from the call until its command buffer completed;
//! a buffer detached after the call refuses nothing of it.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/tob_renders.ll` (`llvm-as`).

use super::blit_holds::{busy, over_host, until, Host, Scratch, FILLED, WAIT};
use super::encoder_holds::{bits, bytes_of, pass, render_state, words_of};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, Degradation,
    DegradeNotice, Device, DeviceKind, DeviceOptions, EncoderCall, Error, MessageSink, Origin,
    RenderCommandEncoder, RenderPipelineState, ResourceOptions, ShaderCacheOptions, Size, Texture,
    TextureDescriptor, VertexAmplificationViewMapping, Viewport,
};
use metal2vulkan::device::RenderLoweringPolicy;
use metal2vulkan::metal::{IndexType, PixelFormat, PrimitiveType, TextureUsage};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// The side of the target, and the color the draws draw: `(0.2, 0.4, 0.6, 1)`, whose
/// `RGBA8Unorm` bytes are exact.
const SIDE: u64 = 16;
const COLOR: [f32; 4] = [0.2, 0.4, 0.6, 1.0];
const PACKED: u32 = u32::from_le_bytes([51, 102, 153, 255]);

/// Both policies, the default first.
const POLICIES: [RenderLoweringPolicy; 2] =
    [RenderLoweringPolicy::AtCall, RenderLoweringPolicy::Worker];

/// The notices a device announced.
type Heard = Arc<Mutex<Vec<DegradeNotice>>>;

/// A device of the kind the cfg selects that lowers render calls as `lowering` says, refusing a
/// call for a notice of a kind of `refuses` announced while it is lowered, its notices into
/// `heard`.
fn device(
    cache: &Path,
    lowering: RenderLoweringPolicy,
    refuses: &[Degradation],
    heard: &Heard,
) -> Device {
    let heard = Arc::clone(heard);
    let mut options = DeviceOptions {
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
        messages: Some(MessageSink::new(move |message| {
            if let Some(notice) = &message.degradation {
                heard.lock().unwrap().push(notice.clone());
            }
        })),
        lowering_refuses: refuses.iter().copied().collect(),
        ..DeviceOptions::default()
    };
    options.policy.render_lowering = lowering;
    Device::system_default_with(&options).unwrap()
}

/// A `Private` `RGBA8Unorm` render target.
fn target(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// The state that draws a rectangle over the whole target, whose fragments take their color
/// from fragment buffer 0 (`tob_rect`, `tob_solid`).
fn rectangle_state(device: &Device) -> RenderPipelineState {
    render_state(device, "engine/tob_renders.ll", "tob_rect", "tob_solid")
}

/// Set `state` on `encoder` with the rectangle's corners.
fn rectangle(state: &RenderPipelineState, encoder: &mut RenderCommandEncoder) {
    encoder.set_render_pipeline_state(state).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
}

/// A `Shared` buffer `command_buffer` copies `target` into, encoded after its pass.
fn copy_back(device: &Device, command_buffer: &CommandBuffer, target: &Texture) -> Buffer {
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

/// Commit `command_buffer` and wait for it: the words of `back` then.
fn committed(command_buffer: &CommandBuffer, back: &Buffer) -> Vec<u32> {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    words_of(back)
}

/// What a check at the call works on: the device's rectangle state, a small buffer of the device
/// and one of another device.
struct Fixture {
    state: RenderPipelineState,
    buffer: Buffer,
    foreign: Buffer,
}

/// A call refused at its call, after the calls it needs: its result.
type Refused = fn(&Fixture, &mut RenderCommandEncoder) -> Result<(), Error>;

/// One call for each kind of check the encoder makes at the call, each refused.
const REFUSED: &[(&str, Refused)] = &[
    ("an index past the vertex table", |f, e| {
        e.set_vertex_buffer(Some(&f.buffer), 0, 31)
    }),
    ("a buffer of another device", |f, e| {
        e.set_vertex_buffer(Some(&f.foreign), 0, 0)
    }),
    ("an offset past the buffer", |f, e| {
        e.set_fragment_buffer(Some(&f.buffer), f.buffer.length() + 4, 0)
    }),
    ("buffers and offsets of different lengths", |f, e| {
        e.set_vertex_buffers(&[Some(&f.buffer)], &[0, 0], 0)
    }),
    ("an offset where no buffer is bound", |_, e| {
        e.set_vertex_buffer_offset(0, 3)
    }),
    ("an offset past the buffer bound", |f, e| {
        e.set_fragment_buffer(Some(&f.buffer), 0, 2)?;
        e.set_fragment_buffer_offset(f.buffer.length() + 4, 2)
    }),
    ("more inline bytes than Metal takes", |_, e| {
        e.set_fragment_bytes(&[0; 4097], 0)
    }),
    ("a texture index past the table", |_, e| {
        e.set_vertex_texture(None, 128)
    }),
    ("samplers and clamps of different lengths", |_, e| {
        e.set_fragment_sampler_states(&[None], Some((&[0.0, 0.0], &[1.0])), 0)
    }),
    ("more viewports than Metal takes", |_, e| {
        let viewport = Viewport::new(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
        e.set_viewports(&[viewport; 17])
    }),
    ("fewer view mappings than the count", |_, e| {
        let mapping = VertexAmplificationViewMapping::new(0, 0);
        e.set_vertex_amplification_count(2, Some(&[mapping]))
    }),
    ("a factor buffer of another device", |f, e| {
        e.set_tessellation_factor_buffer(Some(&f.foreign), 0, 0)
    }),
    ("a draw without a pipeline state", |_, e| {
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
    }),
    ("an indexed draw of another device's indices", |f, e| {
        rectangle(&f.state, e);
        e.draw_indexed_primitives(
            PrimitiveType::Triangle,
            3,
            IndexType::UInt16,
            &f.foreign,
            0,
            1,
            0,
            0,
        )
    }),
    ("indirect arguments past the buffer", |f, e| {
        rectangle(&f.state, e);
        e.draw_primitives_indirect(PrimitiveType::Triangle, &f.buffer, f.buffer.length() - 8)
    }),
    ("indexed indirect arguments past the buffer", |f, e| {
        rectangle(&f.state, e);
        let past = f.buffer.length() - 12;
        e.draw_indexed_primitives_indirect(
            PrimitiveType::Triangle,
            IndexType::UInt16,
            &f.buffer,
            0,
            &f.buffer,
            past,
        )
    }),
    ("an amplification above the pipeline state's", |f, e| {
        rectangle(&f.state, e);
        e.set_vertex_amplification_count(2, None)?;
        e.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
    }),
    ("a resource of another device", |f, e| {
        e.use_resource(
            crate::ResourceRef::Buffer(&f.foreign),
            metal2vulkan::metal::ResourceUsage::READ,
            metal2vulkan::metal::RenderStages::VERTEX,
        )
    }),
];

/// Each refusal the encoder checks at the call returns from the call under both policies, the
/// same error; the command buffer then runs, naming no refusal.
#[test]
fn each_refusal_checked_at_the_call_returns_from_it_under_both_policies() {
    let scratch = Scratch::new("deferred-checks");
    let heard = Heard::default();
    let other = device(&scratch.0, RenderLoweringPolicy::AtCall, &[], &heard);
    let foreign = other.new_buffer(256, ResourceOptions::shared()).unwrap();
    let mut answers: Vec<Vec<Result<(), Error>>> = Vec::new();
    for policy in POLICIES {
        let device = device(&scratch.0, policy, &[], &heard);
        let fixture = Fixture {
            state: rectangle_state(&device),
            buffer: device.new_buffer(64, ResourceOptions::shared()).unwrap(),
            foreign: foreign.clone(),
        };
        let queue = device.new_command_queue().unwrap();
        let target = target(&device);
        let mut each = Vec::new();
        for (what, refused) in REFUSED {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer
                .render_command_encoder(&pass(&target))
                .unwrap();
            let answer = refused(&fixture, &mut encoder);
            assert!(answer.is_err(), "{policy:?}, {what}: {answer:?}");
            // The checks at the call refused it: the encoder still logs its calls.
            let logs = format!("{encoder:?}").contains("lowered_at_call: false");
            assert_eq!(logs, policy == RenderLoweringPolicy::Worker, "{what}");
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                (command_buffer.status(), command_buffer.refusal()),
                (CommandBufferStatus::Completed, None),
                "{policy:?}, {what}"
            );
            each.push(answer);
        }
        answers.push(each);
    }
    for (((what, _), at_call), worker) in REFUSED.iter().zip(&answers[0]).zip(&answers[1]) {
        assert_eq!(at_call, worker, "{what}");
    }
}

/// A draw of points under a state whose vertex function writes no point size, of no point class:
/// its lowering announces `UnsizedPoints` (`ShapeInputs::settings`), at the draw, call 3.
fn unsized_points(state: &RenderPipelineState, encoder: &mut RenderCommandEncoder) {
    rectangle(state, encoder);
    let color = bytes_of(&COLOR, f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
    assert_eq!(
        encoder.draw_primitives(PrimitiveType::Point, 0, 4, 1, 0),
        Ok(())
    );
}

/// The draw of [`unsized_points`].
const POINTS: EncoderCall = EncoderCall {
    selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
    index: 3,
};

/// A notice announced while a call is lowered names it under `Worker`, and none under `AtCall`,
/// where the call lowers itself; with no kind refused (the default) it reaches the sink and the
/// command buffer runs.
#[test]
fn a_notice_announced_as_a_call_is_lowered_names_it() {
    let scratch = Scratch::new("deferred-notice");
    for policy in POLICIES {
        let heard = Heard::default();
        let device = device(&scratch.0, policy, &[], &heard);
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        unsized_points(&state, &mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let heard = heard.lock().unwrap();
        let notice = heard
            .iter()
            .find(|notice| notice.degradation == Degradation::UnsizedPoints)
            .unwrap_or_else(|| panic!("{policy:?}: no notice of unsized points"));
        let named = match policy {
            RenderLoweringPolicy::Worker => Some(POINTS),
            _ => None,
        };
        assert_eq!(notice.call, named, "{policy:?}");
    }
}

/// A kind the device refuses while lowering (`lowering_refuses`) refuses the call it is announced
/// at under `Worker`: the call returned, the notice reaches no sink, and the command buffer fails
/// naming it (`CallRefused`), the notice its error, the encoder and the one after it running
/// nothing. Under `AtCall` it is announced as any other kind.
#[test]
fn a_kind_refused_while_lowering_refuses_the_call() {
    let scratch = Scratch::new("deferred-refused");
    for policy in POLICIES {
        let heard = Heard::default();
        let device = device(&scratch.0, policy, &[Degradation::UnsizedPoints], &heard);
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        unsized_points(&state, &mut encoder);
        // A call after the refused one runs nothing, and returns.
        let color = bytes_of(&COLOR, f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        let back = copy_back(&device, &command_buffer, &target);
        let words = committed(&command_buffer, &back);
        let heard = heard.lock().unwrap();
        let announced = heard
            .iter()
            .any(|notice| notice.degradation == Degradation::UnsizedPoints);
        match policy {
            RenderLoweringPolicy::Worker => {
                assert!(!announced, "a refused kind announced");
                assert_eq!(
                    command_buffer.error(),
                    Some(CommandBufferError::CallRefused(POINTS))
                );
                let Some(Error::Degraded(notice)) = command_buffer.refusal() else {
                    panic!("{:?}", command_buffer.refusal());
                };
                assert_eq!(notice.degradation, Degradation::UnsizedPoints);
                assert_eq!(notice.call, Some(POINTS));
                // Neither the pass nor the copy after it ran.
                assert!(words.iter().all(|&word| word == 0), "{:x?}", &words[..4]);
            }
            _ => {
                assert!(announced, "{policy:?}: not announced");
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                assert_eq!(command_buffer.refusal(), None);
                assert!(
                    words.iter().all(|&word| word == PACKED),
                    "{:x?}",
                    &words[..4]
                );
            }
        }
    }
}

/// The bytes a call was lent are copied at the call: a later write to them before the encoder
/// ends is not drawn.
#[test]
fn bytes_lent_to_a_call_are_copied_at_the_call() {
    let scratch = Scratch::new("deferred-bytes");
    for policy in POLICIES {
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        let mut color = bytes_of(&COLOR, f32::to_le_bytes);
        encoder.set_fragment_bytes(&color, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        color.fill(0);
        encoder.end_encoding().unwrap();
        let back = copy_back(&device, &command_buffer, &target);
        let words = committed(&command_buffer, &back);
        assert!(
            words.iter().all(|&word| word == PACKED),
            "{policy:?}: {:x?}",
            &words[..4]
        );
    }
}

/// An object a call names is held from the call: a no-copy buffer the program lets go of right
/// after binding it lives (its deallocator waits) until the command buffer completed.
#[test]
fn an_object_released_after_its_call_lives_until_its_command_buffer_completes() {
    let scratch = Scratch::new("deferred-holds");
    for policy in POLICIES {
        let host = Host::new();
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let runs = Arc::new(AtomicUsize::new(0));
        let Some(no_copy) = over_host(&device, &host, &runs) else {
            return;
        };
        host.set(0, &bits(&COLOR));
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let filled = device
            .new_buffer(FILLED, ResourceOptions::private())
            .unwrap();
        let before = busy(&queue, &filled);
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        encoder.set_fragment_buffer(Some(&no_copy), 0, 0).unwrap();
        drop(no_copy);
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        assert_eq!(
            runs.load(Ordering::SeqCst),
            0,
            "{policy:?}: let go of at the call"
        );
        encoder.end_encoding().unwrap();
        let back = copy_back(&device, &command_buffer, &target);
        command_buffer.commit().unwrap();
        if before.status() != CommandBufferStatus::Completed {
            assert_eq!(
                runs.load(Ordering::SeqCst),
                0,
                "{policy:?}: let go of before completion"
            );
        }
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let words = words_of(&back);
        assert!(
            words.iter().all(|&word| word == PACKED),
            "{policy:?}: {:x?}",
            &words[..4]
        );
        until(
            "the deallocator runs once the command buffer completed",
            || runs.load(Ordering::SeqCst) == 1,
        );
    }
}

/// A buffer detached after a call that names it refuses nothing of the call: the encoder ends
/// without a refusal, and a rebind before the command buffer's first submission and the command
/// buffer meet the encoded call under `Worker` as under `AtCall`, where it was lowered before the
/// detach.
#[test]
fn a_buffer_detached_after_its_call_refuses_nothing_of_it() {
    let scratch = Scratch::new("deferred-detach");
    let mut outcomes = Vec::new();
    for policy in POLICIES {
        let (host, again) = (Host::new(), Host::new());
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let runs = Arc::new(AtomicUsize::new(0));
        let Some(no_copy) = over_host(&device, &host, &runs) else {
            return;
        };
        host.fill(0);
        again.set(0, &bits(&COLOR));
        let state = rectangle_state(&device);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        encoder.set_fragment_buffer(Some(&no_copy), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        assert_eq!(no_copy.detach_backing(), Ok(true));
        assert_eq!(encoder.end_encoding(), Ok(()), "{policy:?}");
        let back = copy_back(&device, &command_buffer, &target);
        // SAFETY: the pages outlive the buffer (dropped after it), which alone uses them.
        let rebound = unsafe { no_copy.rebind_bytes_no_copy(again.memory(), None, None) };
        let words = committed(&command_buffer, &back);
        let drawn = words.iter().all(|&word| word == PACKED);
        outcomes.push((
            rebound,
            command_buffer.status(),
            command_buffer.error(),
            drawn,
        ));
    }
    assert_eq!(outcomes[0], outcomes[1]);
}

/// A call made while a buffer of the device is detached is lowered at the call: a draw naming the
/// detached buffer fails its command buffer as one at its call does (`DetachedBacking`).
#[test]
fn a_call_made_while_a_buffer_is_detached_is_lowered_at_the_call() {
    let scratch = Scratch::new("deferred-detached");
    let mut outcomes = Vec::new();
    for policy in POLICIES {
        let host = Host::new();
        let device = device(&scratch.0, policy, &[], &Heard::default());
        let runs = Arc::new(AtomicUsize::new(0));
        let Some(no_copy) = over_host(&device, &host, &runs) else {
            return;
        };
        let state = rectangle_state(&device);
        assert_eq!(no_copy.detach_backing(), Ok(true));
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let target = target(&device);
        let mut encoder = command_buffer
            .render_command_encoder(&pass(&target))
            .unwrap();
        rectangle(&state, &mut encoder);
        encoder.set_fragment_buffer(Some(&no_copy), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        outcomes.push((command_buffer.status(), command_buffer.error()));
    }
    assert_eq!(outcomes[0], outcomes[1]);
    assert_eq!(
        outcomes[0],
        (
            CommandBufferStatus::Error,
            Some(CommandBufferError::DetachedBacking)
        )
    );
}
