//! A command buffer's used resources (`CommandBuffer::used_resources`, a runtime extension; the
//! runtime's `used_resources.rs`) on the device the cfg selects, under the validation layer: each
//! kind of use lists what owns the memory it reaches, from each encoder's end, while the command
//! buffer is committed and waiting, and nothing once it completed.
//!
//! - **Bindings.** A kernel case's buffers and textures, bound at its slots, are listed (a buffer
//!   in a heap of its own between guards as that heap); a texture buffer over a buffer, bound at
//!   a texture slot, as the buffer.
//! - **Residency.** A buffer and a texture an encoder made resident (`use_resource`), and a heap
//!   it used (`use_heap`), are listed though no command of the encoder reads them.
//! - **Blits.** A copy's buffers; a buffer placed in a heap, filled, as its heap; a texture placed
//!   in a heap, written, as its heap; a view, read, as the texture it was made from.
//! - **Attachments.** A render pass's color attachment, cleared and stored.
//! - **What is not used.** A buffer and a texture no command uses are not listed.
//! - **One memory, two buffers.** Two no-copy buffers over one memory, which lie in one import, are
//!   two objects: each is listed where a command uses it, the other not.
//! - **By encoder** (`CommandBuffer::ended_encoders`). Each ended encoder states its kind, the
//!   event waits and signals before it, and what its own commands use; the command buffer's list is
//!   their union, and once it completed each lists nothing.

use super::host_memory::{no_copy, not_importable, on_device, run, Mapping, GUEST_PAGE};
use super::{bound_buffers, encode_case, pipeline_state, public_case, CaseResources};
use crate::candidate::engine::ComputeOutput;
use metal2vulkan::metal::{PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, Device, EncoderKind, EventRef, Heap,
    HeapDescriptor, HeapType, LoadAction, Origin, RenderPassDescriptor, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor, UsedResources,
};
use std::collections::HashSet;
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// A kernel case over buffers alone.
const KERNEL: &str = "saturating-halving-and-wide-product-integers-at-their-edges";

/// A kernel case that writes four bytes of a storage texel buffer.
const TEXEL_WRITER: &str = "write-four-bytes-into-a-storage-texel-buffer";

/// The side of the tests' textures, `RGBA8Unorm`.
const SIDE: u64 = 4;

/// What a command buffer lists, as sets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Listed {
    buffers: HashSet<Buffer>,
    textures: HashSet<Texture>,
    heaps: HashSet<Heap>,
}

impl Listed {
    fn of(used: UsedResources) -> Listed {
        Listed {
            buffers: used.buffers.into_iter().collect(),
            textures: used.textures.into_iter().collect(),
            heaps: used.heaps.into_iter().collect(),
        }
    }

    /// Expect `buffer`, as its heap where it lies in one.
    fn buffer(&mut self, buffer: &Buffer) {
        match buffer.heap() {
            Some(heap) => self.heaps.insert(heap.clone()),
            None => self.buffers.insert(buffer.clone()),
        };
    }

    /// Expect what `other` lists too.
    fn join(&mut self, other: &Listed) {
        self.buffers.extend(other.buffers.iter().cloned());
        self.textures.extend(other.textures.iter().cloned());
        self.heaps.extend(other.heaps.iter().cloned());
    }
}

/// Require `command_buffer`'s ended encoders to be of `kinds`, after one event wait and no
/// signal, each listing what `each` expects of it (none where `each` is shorter), and their
/// union to be the command buffer's list.
fn check_encoders(command_buffer: &CommandBuffer, kinds: &[EncoderKind], each: &[Listed]) {
    let ended = command_buffer.ended_encoders();
    let reported: Vec<_> = ended.iter().map(|encoder| encoder.kind).collect();
    assert_eq!(reported, kinds);
    let mut union = Listed::default();
    for (at, encoder) in ended.into_iter().enumerate() {
        assert_eq!(
            (encoder.event_waits_before, encoder.event_signals_before),
            (1, 0),
            "encoder {at}"
        );
        let listed = Listed::of(encoder.used_resources);
        let want = each.get(at).cloned().unwrap_or_default();
        assert_eq!(listed, want, "encoder {at}");
        union.join(&listed);
    }
    assert_eq!(Listed::of(command_buffer.used_resources()), union);
}

/// A `SIDE` x `SIDE` `RGBA8Unorm` texture descriptor of `usage` and `options`.
fn square(usage: TextureUsage, options: ResourceOptions) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = usage;
    descriptor.resource_options = options;
    descriptor
}

/// A `Shared` buffer of `length` zero bytes.
fn zeros(device: &Device, length: u64) -> Buffer {
    device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap()
}

/// Each kind of use lists what owns the memory it reaches (the module docs), from each encoder's
/// end through the commit, while the command buffer waits for an event; once it completed, it
/// lists nothing. Each ended encoder lists what its own commands use, its kind and the wait
/// before it, and the command buffer's list is their union.
#[test]
fn a_command_buffer_lists_each_kind_of_use_until_it_completes() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        let mut expected = Listed::default();
        assert_eq!(Listed::of(command_buffer.used_resources()), expected);
        // What each encoder's own commands use, and the kinds.
        let mut each: Vec<Listed> = Vec::new();
        let mut kinds = Vec::new();

        // Bindings: a kernel case's buffers.
        let kernel = public_case(KERNEL).unwrap();
        let pipeline = pipeline_state(device, &kernel).unwrap();
        let resources = CaseResources::new(device, &kernel).unwrap();
        let bound = bound_buffers(device, &kernel.compute, &resources).unwrap();
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut compute, &pipeline, &kernel.compute, &bound, &resources).unwrap();
        compute.end_encoding().unwrap();
        let mut own = Listed::default();
        bound.values().for_each(|buffer| own.buffer(buffer));
        expected.join(&own);
        each.push(own);
        kinds.push(EncoderKind::Compute);
        assert_eq!(Listed::of(command_buffer.used_resources()), expected);
        check_encoders(&command_buffer, &kinds, &each);

        // A texture buffer over a buffer, bound at a texture slot: the buffer.
        let writer = public_case(TEXEL_WRITER).unwrap();
        let ComputeOutput::Texture { binding, .. } = writer.compute.output else {
            panic!("{TEXEL_WRITER}: the case's output is no texture");
        };
        let literal = &writer.compute.textures[&binding];
        let length = (literal.bytes.len() as u64).next_multiple_of(256);
        let under = zeros(device, length);
        let descriptor =
            super::texture_descriptor(literal, writer.texture_usage[&binding]).unwrap();
        let texels = under.new_texture(&descriptor, 0, length).unwrap();
        let pipeline = pipeline_state(device, &writer).unwrap();
        let mut resources = CaseResources::new(device, &writer).unwrap();
        resources.textures.insert(binding, texels.clone());
        let bound = bound_buffers(device, &writer.compute, &resources).unwrap();
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        encode_case(&mut compute, &pipeline, &writer.compute, &bound, &resources).unwrap();
        compute.end_encoding().unwrap();
        let mut own = Listed::default();
        bound.values().for_each(|buffer| own.buffer(buffer));
        own.buffer(&under);
        // The case's other textures, each an image of its own, bound at their slots.
        for (_, texture) in resources.textures.iter().filter(|&(&at, _)| at != binding) {
            own.textures.insert(texture.clone());
        }
        expected.join(&own);
        each.push(own);
        kinds.push(EncoderKind::Compute);
        assert_eq!(Listed::of(command_buffer.used_resources()), expected);
        check_encoders(&command_buffer, &kinds, &each);

        // Residency: a buffer and a texture made resident, and a heap used, by an encoder that
        // dispatches nothing.
        let resident = zeros(device, 256);
        let shader_read = TextureUsage::SHADER_READ;
        let resident_texture = device
            .new_texture(&square(shader_read, ResourceOptions::private()))
            .unwrap();
        let used_heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute
            .use_resource(resident.as_resource(), ResourceUsage::READ)
            .unwrap();
        compute
            .use_resource(resident_texture.as_resource(), ResourceUsage::READ)
            .unwrap();
        compute.use_heap(&used_heap).unwrap();
        compute.end_encoding().unwrap();
        let mut own = Listed::default();
        own.buffer(&resident);
        own.textures.insert(resident_texture.clone());
        own.heaps.insert(used_heap.clone());
        expected.join(&own);
        each.push(own);
        kinds.push(EncoderKind::Compute);
        assert_eq!(Listed::of(command_buffer.used_resources()), expected);
        check_encoders(&command_buffer, &kinds, &each);

        // Blits: a copy's buffers; a heap's buffer, filled, and a heap's texture, written, as
        // their heaps; a view, read, as its texture.
        let source = zeros(device, 256);
        let destination = zeros(device, 256);
        let blit_heap = device
            .new_heap(&HeapDescriptor::new(
                HeapType::Automatic,
                1 << 20,
                ResourceOptions::private(),
            ))
            .unwrap();
        let in_heap = blit_heap
            .new_buffer(256, ResourceOptions::private())
            .unwrap();
        let heap_texture = blit_heap
            .new_texture(&square(shader_read, ResourceOptions::private()))
            .unwrap();
        let viewed = device
            .new_texture(&square(shader_read, ResourceOptions::private()))
            .unwrap();
        let view = viewed.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
        let row = 4 * SIDE;
        let whole = Size::new(SIDE, SIDE, 1);
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&source, 0, &destination, 0, 256)
            .unwrap();
        blit.fill_buffer(&in_heap, 0..256, 0x5a).unwrap();
        blit.copy_from_buffer_to_texture(
            &source,
            0,
            row,
            row * SIDE,
            whole,
            &heap_texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.copy_from_texture_to_buffer(
            &view,
            0,
            0,
            Origin::new(0, 0, 0),
            whole,
            &destination,
            0,
            row,
            row * SIDE,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        let mut own = Listed::default();
        own.buffer(&source);
        own.buffer(&destination);
        own.heaps.insert(blit_heap.clone());
        own.textures.insert(viewed.clone());
        expected.join(&own);
        each.push(own);
        kinds.push(EncoderKind::Blit);
        assert_eq!(Listed::of(command_buffer.used_resources()), expected);
        check_encoders(&command_buffer, &kinds, &each);

        // An attachment: a color attachment, cleared and stored.
        let target = device
            .new_texture(&square(
                TextureUsage::RENDER_TARGET,
                ResourceOptions::private(),
            ))
            .unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
        command_buffer
            .render_command_encoder(&pass)
            .unwrap()
            .end_encoding()
            .unwrap();
        let mut own = Listed::default();
        own.textures.insert(target.clone());
        expected.join(&own);
        each.push(own);
        kinds.push(EncoderKind::Render);

        // What no command uses is not listed.
        let idle = zeros(device, 256);
        let idle_texture = device
            .new_texture(&square(shader_read, ResourceOptions::private()))
            .unwrap();
        let listed = Listed::of(command_buffer.used_resources());
        assert!(!listed.buffers.contains(&idle) && !listed.textures.contains(&idle_texture));
        assert_eq!(listed, expected);
        check_encoders(&command_buffer, &kinds, &each);

        // Committed and waiting for the event: the same; completed: nothing, each encoder
        // keeping its kind.
        command_buffer.commit().unwrap();
        assert_eq!(Listed::of(command_buffer.used_resources()), expected);
        check_encoders(&command_buffer, &kinds, &each);
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        assert_eq!(
            Listed::of(command_buffer.used_resources()),
            Listed::default()
        );
        check_encoders(&command_buffer, &kinds, &[]);
        eprintln!(
            "USED buffers {}, textures {}, heaps {} listed while committed",
            expected.buffers.len(),
            expected.textures.len(),
            expected.heaps.len()
        );
    });
}

/// Two no-copy buffers over one memory lie in one import and stay two Metal objects: two object
/// IDs, and a command buffer lists the one its blit fills and not the other, both where it fills
/// both.
#[test]
fn two_buffers_over_one_memory_are_each_listed_where_used() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let memory = Mapping::anonymous(GUEST_PAGE);
        let import = || {
            // SAFETY: the mapping outlives both buffers.
            unsafe {
                no_copy(
                    device,
                    memory.pointer(),
                    GUEST_PAGE,
                    ResourceOptions::default(),
                    None,
                )
            }
        };
        let first = match import() {
            Err(error) if not_importable(&error) => {
                eprintln!("USED: the device imports no host memory: {error:?}");
                return;
            }
            first => first.unwrap(),
        };
        let second = import().unwrap();
        assert_ne!(first.object_id(), second.object_id());
        for filled in [vec![&first], vec![&second], vec![&first, &second]] {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            for buffer in &filled {
                blit.fill_buffer(buffer, 0..16, 0x5a).unwrap();
            }
            blit.end_encoding().unwrap();
            let listed = Listed::of(command_buffer.used_resources());
            let expected = Listed {
                buffers: filled.iter().map(|&buffer| buffer.clone()).collect(),
                ..Listed::default()
            };
            assert_eq!(listed, expected);
            run(&command_buffer);
        }
        drop((first, second));
        eprintln!("USED one memory, two buffers: each listed where used");
    });
}
