//! Tile dispatches over 2D textures whose texels lie in buffers (`render/tile_dispatch.rs`) on the
//! device the cfg selects, under the validation layer. the Metal probe's cases run on the runtime
//! with the probe's functions (the owned fixture `cfb_tile_shadows`): passes of one 32x32
//! `RGBA8Unorm` attachment in 16x16 tiles, cleared and not stored; 4x4 `RGBA8Unorm` textures at
//! byte 256 of 16384-byte buffers, rows 32 bytes apart, every other byte `7i + 3`
//! (`buffer_texture_blits`' `Held`), the source's texels set 0's; a thread or a fragment at a pixel
//! whose coordinates are multiples of 8 works on the texel at an eighth of them. Each case runs
//! over an ordinary buffer and a no-copy one, so each texture takes the aliasing or the shadow form
//! the device gives it.
//!
//! Every byte of each buffer is checked against Metal's, which the M3 Max (macOS 26.7) and an M4
//! (macOS 27.0.1) measured alike, with and without API validation:
//!
//! - A dispatch copies a texture bound in the tile table, or held in an argument buffer and made
//!   resident in the tile stage, into another, the channels reversed.
//! - It sees a draw before it in the pass that wrote the texture; a draw after it sees what it
//!   wrote, through the texture and through the buffer's bytes; a second dispatch sees the
//!   first's; a blit after the pass sees it.
//! - A dispatch that writes a texel and other bytes of its buffer keeps both (TOB's two ways):
//!   where the texture is shadowed, the runtime merges its writes and announces it once.
//!
//! Where the runtime does not give Metal's answer, the test says so: a dispatch reading through a
//! texture the bytes a draw before it in the pass wrote through the buffer reads, on Metal, the
//! texels as they were before the draw, an order Metal leaves undefined (its validation stops the
//! barrier that would order them, from the fragment stage, on Apple GPUs); the runtime orders
//! them, so its dispatch reads the draw's bytes (printed).

use super::buffer_texture_blits::{texel, Held, Kind, OFF};
use super::buffer_texture_renders::MERGES;
use super::degrade_notices::counts;
use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, Degradation, Device, Library, LoadAction,
    PipelineOptions, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceRef,
    Size, StoreAction, Texture, TextureDescriptor,
};

/// The textures' rows.
const BPR: u64 = 32;
/// The textures' extent.
const EXTENT: [u64; 2] = [4, 4];
/// The functions' `at`: the word of texel (0, 0) and the words of a row.
const AT: [u32; 2] = [OFF as u32 / 4, BPR as u32 / 4];
/// The probe's texture usage.
const USAGE: TextureUsage = TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE);

/// The lane's device, queue, the fixture's library and the passes' attachment.
struct Lane {
    device: Device,
    queue: CommandQueue,
    library: Library,
    target: Texture,
}

impl Lane {
    fn new(device: &Device) -> Lane {
        let bitcode = llvm_assemble(&engine_dir().join("cfb_tile_shadows.ll")).unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 32, 32, false);
        target.usage = TextureUsage::RENDER_TARGET;
        Lane {
            queue: device.new_command_queue().unwrap(),
            library: device.new_library_with_air(&bitcode).unwrap(),
            target: device.new_texture(&target).unwrap(),
            device: device.clone(),
        }
    }

    /// The tile state of `name`, over the attachment.
    fn tile(&self, name: &str) -> RenderPipelineState {
        let function = self.library.new_function_with_name(name).unwrap();
        let mut descriptor = TileRenderPipelineDescriptor::new(StageSource::Function(
            function.function_descriptor(),
        ));
        descriptor.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
        self.device
            .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// The fullscreen triangle's state with the fragment function `name`.
    fn draw(&self, name: &str) -> RenderPipelineState {
        let function = |name| self.library.new_function_with_name(name).unwrap();
        let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(
            function("cfb_ts_full").function_descriptor(),
        ));
        descriptor.fragment = Some(StageSource::Function(function(name).function_descriptor()));
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        self.device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap()
    }

    /// The probe's pass.
    fn pass(&self) -> RenderPassDescriptor {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(self.target.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::DontCare;
        pass.tile_width = 16;
        pass.tile_height = 16;
        pass
    }

    /// One command buffer of the probe's pass, its commands by `body`, and after it by `after`;
    /// committed and waited for.
    fn run(
        &self,
        body: impl FnOnce(&Lane, &mut RenderCommandEncoder, &mut Vec<Buffer>),
        after: impl FnOnce(&metal2vulkan_runtime::CommandBuffer),
    ) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut held = Vec::new();
        let mut encoder = command_buffer.render_command_encoder(&self.pass()).unwrap();
        body(self, &mut encoder, &mut held);
        encoder.end_encoding().unwrap();
        after(&command_buffer);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
    }

    /// A dispatch of `cfb_ts_copy` from `src` into `dst`.
    fn copy(&self, e: &mut RenderCommandEncoder, src: &Texture, dst: &Texture) {
        e.set_render_pipeline_state(&self.tile("cfb_ts_copy"))
            .unwrap();
        e.set_tile_texture(Some(src), 0).unwrap();
        e.set_tile_texture(Some(dst), 1).unwrap();
        e.dispatch_threads_per_tile(Size::new(16, 16, 1)).unwrap();
    }

    /// A draw of the fullscreen triangle with the fragment function `name`.
    fn fullscreen(&self, e: &mut RenderCommandEncoder, name: &str) {
        e.set_render_pipeline_state(&self.draw(name)).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    }

    /// A buffer of the functions' `at`.
    fn at(&self, held: &mut Vec<Buffer>) -> Buffer {
        let bytes: Vec<u8> = AT.iter().flat_map(|w| w.to_le_bytes()).collect();
        let buffer = buffer_with(&self.device, &bytes).unwrap();
        held.push(buffer.clone());
        buffer
    }

    /// A buffer of 16 words, each `0xEEEEEEEE`, the fragment functions' `o`.
    fn out(&self) -> Buffer {
        buffer_with(&self.device, &[0xEE; 64]).unwrap()
    }
}

/// A buffer of `kind` holding the probe's bytes, set 0's texels where `texels`.
fn held(device: &Device, kind: Kind, texels: bool) -> Held {
    let mut held = Held::new(device, kind).unwrap();
    if texels {
        held.fill(OFF, BPR, EXTENT, 0);
    }
    held
}

/// The texture over `held`.
fn over(held: &Held) -> Texture {
    held.texture_of(PixelFormat::RGBA8Unorm, USAGE, (OFF, BPR), EXTENT)
}

/// Set the texels of a texture over `model`'s buffer, rows `BPR` apart from `OFF`, to
/// `value(x, y)`.
fn set(model: &mut [u8], value: impl Fn(u64, u64) -> [u8; 4]) {
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let at = (OFF + y * BPR + 4 * x) as usize;
            model[at..at + 4].copy_from_slice(&value(x, y));
        }
    }
}

/// Set 0's texel (x, y), its channels reversed: what `cfb_ts_copy` writes.
fn reversed(x: u64, y: u64) -> [u8; 4] {
    let mut t = texel(x, y, 0);
    t.reverse();
    t
}

/// What `cfb_ts_write` writes at texel (x, y).
fn drawn(x: u64, y: u64) -> [u8; 4] {
    [(x * 32) as u8, (y * 32) as u8, 0xbf, 0xff]
}

/// What `cfb_ts_write_bytes` writes at texel (x, y).
fn drawn_bytes(x: u64, y: u64) -> [u8; 4] {
    (0xA533_0000u32 | ((y as u32) << 8) | x as u32).to_le_bytes()
}

/// `got` against `model`, every byte: the first that differs is named.
fn check(got: &[u8], model: &[u8], what: &str) {
    if let Some(i) = (0..model.len()).find(|&i| got[i] != model[i]) {
        panic!(
            "{what}: byte {i} is {:#04x}, Metal's {:#04x}",
            got[i], model[i]
        );
    }
}

/// The 16 words of `o` the fragment functions write, as Metal's texels at (x, y): `value(x, y)`.
fn check_out(o: &Buffer, value: impl Fn(u64, u64) -> [u8; 4], what: &str) {
    let got = read_buffer(o, 0, 64).unwrap();
    let model: Vec<u8> = (0..4)
        .flat_map(|y| (0..4).map(move |x| (x, y)))
        .flat_map(|(x, y)| value(x, y))
        .collect();
    check(&got, &model, what);
}

/// Run `test` over each kind of buffer `device` makes.
fn kinds_of(device: &Device, test: impl Fn(&Lane, Kind)) {
    let lane = Lane::new(device);
    for kind in [Kind::Ordinary, Kind::NoCopy] {
        if Held::new(device, kind).is_none() {
            eprintln!(
                "{}: no {kind:?} buffer (the device imports none)",
                device.name()
            );
            continue;
        }
        test(&lane, kind);
    }
}

/// Run `test` on a validated device of the lane over each kind of buffer it makes, then fail on
/// any validation message.
fn each_kind(test: impl Fn(&Lane, Kind)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        kinds_of(&device, test);
    }
    quiet(&messages);
}

/// Fail on any validation message.
fn quiet(messages: &Messages) {
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Probe 3's `copy` and `copy_held`: a dispatch copies a texture bound in the tile table, and
/// one held in an argument buffer and made resident in the tile stage, into another; the source
/// keeps its bytes, and no byte past the texels changes.
#[test]
fn a_tile_dispatch_copies_textures_over_buffers() {
    each_kind(|lane, kind| {
        let device = &lane.device;
        for held_src in [false, true] {
            let name = format!("copy (held {held_src}) over a {kind:?} buffer");
            let (a, mut b) = (held(device, kind, true), held(device, kind, false));
            let (src, dst) = (over(&a), over(&b));
            lane.run(
                |l, e, h| {
                    if held_src {
                        let id = src.gpu_resource_id().0.to_le_bytes();
                        let ids = buffer_with(&l.device, &id).unwrap();
                        e.set_render_pipeline_state(&l.tile("cfb_ts_copy_held"))
                            .unwrap();
                        e.set_tile_buffer(Some(&ids), 0, 0).unwrap();
                        e.set_tile_texture(Some(&dst), 1).unwrap();
                        e.use_resource(
                            ResourceRef::Texture(&src),
                            ResourceUsage::READ,
                            RenderStages::TILE,
                        )
                        .unwrap();
                        e.dispatch_threads_per_tile(Size::new(16, 16, 1)).unwrap();
                        h.push(ids);
                    } else {
                        l.copy(e, &src, &dst);
                    }
                },
                |_| {},
            );
            set(&mut b.model, reversed);
            check(&b.bytes(), &b.model, &format!("{name}: the destination"));
            check(&a.bytes(), &a.model, &format!("{name}: the source"));
        }
    });
}

/// Probe 3's order across a pass: `draw_texture_then_copy` (a draw writes the source texture,
/// and the dispatch after it copies what it wrote), `copy_then_draw_bytes` and
/// `copy_then_draw_texture` (a draw after the dispatch reads what it wrote through the buffer's
/// bytes, and through the texture), `copy_chain` (a second dispatch copies the first's), and
/// `copy_then_blit` (a blit after the pass copies the bytes it wrote). `draw_bytes_then_copy`:
/// the dispatch after a draw that wrote the source's bytes through the buffer reads the draw's
/// bytes where Metal read the texels before it (printed).
#[test]
fn tile_dispatches_and_draws_see_each_others_writes_in_order() {
    each_kind(|lane, kind| {
        let device = &lane.device;
        let name = |case: &str| format!("{case} over a {kind:?} buffer");

        let (mut a, mut b) = (held(device, kind, true), held(device, kind, false));
        let (src, dst) = (over(&a), over(&b));
        lane.run(
            |l, e, _| {
                e.set_fragment_texture(Some(&src), 0).unwrap();
                l.fullscreen(e, "cfb_ts_write");
                l.copy(e, &src, &dst);
            },
            |_| {},
        );
        set(&mut a.model, drawn);
        set(&mut b.model, |x, y| {
            let mut t = drawn(x, y);
            t.reverse();
            t
        });
        check(
            &a.bytes(),
            &a.model,
            &name("draw_texture_then_copy: the source"),
        );
        check(
            &b.bytes(),
            &b.model,
            &name("draw_texture_then_copy: the destination"),
        );

        for through_texture in [false, true] {
            let case = match through_texture {
                false => "copy_then_draw_bytes",
                true => "copy_then_draw_texture",
            };
            let (a, mut b, o) = (
                held(device, kind, true),
                held(device, kind, false),
                lane.out(),
            );
            let (src, dst) = (over(&a), over(&b));
            lane.run(
                |l, e, h| {
                    l.copy(e, &src, &dst);
                    if through_texture {
                        e.set_fragment_texture(Some(&dst), 0).unwrap();
                    } else {
                        let at = l.at(h);
                        e.set_fragment_buffer(Some(&b.buffer), 0, 0).unwrap();
                        e.set_fragment_buffer(Some(&at), 0, 1).unwrap();
                    }
                    e.set_fragment_buffer(Some(&o), 0, 2).unwrap();
                    let fragment = match through_texture {
                        false => "cfb_ts_read_bytes",
                        true => "cfb_ts_read_texture",
                    };
                    l.fullscreen(e, fragment);
                },
                |_| {},
            );
            check_out(&o, reversed, &name(&format!("{case}: what the draw read")));
            set(&mut b.model, reversed);
            check(
                &b.bytes(),
                &b.model,
                &name(&format!("{case}: the destination")),
            );
        }

        let (a, mut b, mut c) = (
            held(device, kind, true),
            held(device, kind, false),
            held(device, kind, false),
        );
        let (src, mid, dst) = (over(&a), over(&b), over(&c));
        lane.run(
            |l, e, _| {
                l.copy(e, &src, &mid);
                l.copy(e, &mid, &dst);
            },
            |_| {},
        );
        set(&mut b.model, reversed);
        set(&mut c.model, |x, y| texel(x, y, 0));
        check(
            &b.bytes(),
            &b.model,
            &name("copy_chain: the first destination"),
        );
        check(&c.bytes(), &c.model, &name("copy_chain: the second"));

        let (a, mut b) = (held(device, kind, true), held(device, kind, false));
        let (src, dst) = (over(&a), over(&b));
        let blitted = buffer_with(device, &[0xEE; 128]).unwrap();
        lane.run(
            |l, e, _| l.copy(e, &src, &dst),
            |command_buffer| {
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_buffer(&b.buffer, OFF, &blitted, 0, 128)
                    .unwrap();
                blit.end_encoding().unwrap();
            },
        );
        set(&mut b.model, reversed);
        let at = OFF as usize;
        check(
            &read_buffer(&blitted, 0, 128).unwrap(),
            &b.model[at..at + 128],
            &name("copy_then_blit: the blit's copy"),
        );

        let (mut a, b) = (held(device, kind, true), held(device, kind, false));
        let (src, dst) = (over(&a), over(&b));
        lane.run(
            |l, e, h| {
                let at = l.at(h);
                e.set_fragment_buffer(Some(&a.buffer), 0, 0).unwrap();
                e.set_fragment_buffer(Some(&at), 0, 1).unwrap();
                l.fullscreen(e, "cfb_ts_write_bytes");
                l.copy(e, &src, &dst);
            },
            |_| {},
        );
        set(&mut a.model, drawn_bytes);
        check(
            &a.bytes(),
            &a.model,
            &name("draw_bytes_then_copy: the source"),
        );
        let got = b.bytes();
        let texel_at = |x: u64, y: u64| {
            let at = (OFF + y * BPR + 4 * x) as usize;
            [got[at], got[at + 1], got[at + 2], got[at + 3]]
        };
        let ordered = (0..4).all(|y| {
            (0..4).all(|x| {
                let mut t = drawn_bytes(x, y);
                t.reverse();
                texel_at(x, y) == t
            })
        });
        let metal = (0..4).all(|y| (0..4).all(|x| texel_at(x, y) == reversed(x, y)));
        assert!(
            ordered || metal,
            "{}: the destination is neither the draw's bytes nor the texels before it",
            name("draw_bytes_then_copy")
        );
        eprintln!(
            "{}: the dispatch read {} (Metal: the texels before the draw)",
            name("draw_bytes_then_copy"),
            if ordered {
                "the draw's bytes"
            } else {
                "the texels before the draw"
            }
        );
    });
}

/// Probe 3's `two_ways_disjoint` and `two_ways_gap`: a dispatch writes texel (0, 0) and the bytes
/// of texel (1, 0), or the four bytes after row 0's texels, through the buffer; both writes stay.
/// Where the texture is shadowed (the device announces `shadowed-texture`), its writes are merged
/// and announced once ([`MERGES`]); where none is, nothing is merged.
#[test]
fn a_tile_dispatch_writing_two_ways_keeps_both() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    // What the device announced as it was made.
    messages.take_notices();
    kinds_of(&device, |lane, kind| {
        let device = &lane.device;
        let state = lane.tile("cfb_ts_two_ways");
        for (case, off) in [("two_ways_disjoint", OFF + 4), ("two_ways_gap", OFF + 16)] {
            let name = format!("{case} over a {kind:?} buffer");
            let mut b = held(device, kind, true);
            let texture = over(&b);
            lane.run(
                |l, e, h| {
                    let off = buffer_with(&l.device, &(off as u32).to_le_bytes()).unwrap();
                    e.set_render_pipeline_state(&state).unwrap();
                    e.set_tile_texture(Some(&texture), 0).unwrap();
                    e.set_tile_buffer(Some(&b.buffer), 0, 0).unwrap();
                    e.set_tile_buffer(Some(&off), 0, 1).unwrap();
                    e.dispatch_threads_per_tile(Size::new(16, 16, 1)).unwrap();
                    h.push(off);
                },
                |_| {},
            );
            let at = OFF as usize;
            b.model[at..at + 4].copy_from_slice(&[0xff, 0, 0, 0xff]);
            let off = off as usize;
            b.model[off..off + 4].copy_from_slice(&[0x5a, 0x5b, 0x5c, 0x5d]);
            check(&b.bytes(), &b.model, &name);
        }
    });
    let counted = counts(&messages.take_notices());
    let shadowed = counted.contains_key(&Degradation::ShadowedTexture);
    assert_eq!(
        counted.get(&MERGES[0]).copied(),
        shadowed.then_some(1),
        "{} merges where a texture is shadowed, once: {counted:?}",
        device.name()
    );
    drop(device);
    quiet(&messages);
}
