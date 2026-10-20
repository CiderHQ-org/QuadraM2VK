//! A blit's copy into every texel of a shadowed texture (`shadow.rs`), a one-level, one-slice 2D
//! texture of a heap over host pages whose image lies in memory of its own, replaces the
//! shadow's texels (`ShadowUse::overwrites`, `shadow::overwritten`): it takes no copy in and no
//! keep, and the encoder's one copy is the copy back of what it wrote. A copy into a part of
//! the texture still copies in, keeps and merges back. Each leaves the copied texels in the
//! texture, the others as they were, and every byte of the heap outside the texture as it was.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. A device that lays such a texture in
//! the heap's bytes with no shadow skips them, saying so.

use super::blit_holds::{device, Host, Scratch, BYTES, WAIT};
use crate::shadow::{ShadowSource, ShadowTarget};
use crate::{
    BlitOption, Buffer, CommandQueue, Device, HeapDescriptor, HeapType, Origin, Region,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// The texture's side: 64 x 64 `BGRA8Unorm`, a window surface's format and usage.
const SIDE: u64 = 64;

/// Where the texture lies in the heap.
const OFFSET: u64 = 64 << 10;

/// The part a partial copy writes: `PART` texels square from (`AT`, `AT`).
const AT: u64 = 8;
const PART: u64 = 16;

/// `length` bytes that name their place and `seed`.
fn pattern(seed: u8, length: u64) -> Vec<u8> {
    (0..length)
        .map(|i| (i as u8) ^ (i >> 8) as u8 ^ seed)
        .collect()
}

/// The shadow copies a command buffer of one blit encoder planned (`WorkCounts::shadow_copies`),
/// the texels it copied in (`shadow_texels_copied_in`) and kept (`shadow_texels_kept`), for a
/// copy of `source` (rows of `side.0` texels) into the box `side` from (`at`, `at`).
fn copy(
    device: &Device,
    queue: &CommandQueue,
    (source, texture): (&Buffer, &Texture),
    at: u64,
    side: u64,
) -> [u64; 3] {
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        source,
        0,
        side * 4,
        side * side * 4,
        Size::new(side, side, 1),
        texture,
        0,
        0,
        Origin::new(at, at, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let after = device.work_counts();
    [
        after.shadow_copies - before.shadow_copies,
        after.shadow_texels_copied_in - before.shadow_texels_copied_in,
        after.shadow_texels_kept - before.shadow_texels_kept,
    ]
}

/// The texture's texels, as `getBytes` reads them.
fn texels(texture: &Texture) -> Vec<u8> {
    let mut read = vec![0u8; (SIDE * SIDE * 4) as usize];
    let region = Region::new_2d(0, 0, SIDE, SIDE);
    texture
        .get_bytes(&mut read, SIDE * 4, SIDE * SIDE * 4, region, 0, 0)
        .unwrap();
    read
}

#[test]
fn a_blits_copy_into_every_texel_of_a_shadowed_texture_takes_no_copy_in_and_no_keep() {
    let scratch = Scratch::new("blit-cover");
    let host = Host::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let descriptor =
        HeapDescriptor::new(HeapType::Placement, BYTES as u64, ResourceOptions::shared());
    // SAFETY: the host memory outlives the heap and its texture (dropped after the device), and
    // only their commands use it.
    let heap =
        match unsafe { device.new_heap_with_bytes_no_copy(host.memory(), &descriptor, None, None) }
        {
            Ok(heap) => heap,
            Err(error) => {
                eprintln!("skipped: no heap over host pages here: {error}");
                return;
            }
        };
    let mut surface = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
    surface.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    surface.resource_options = ResourceOptions::shared();
    let span = device.heap_texture_size_and_align(&surface).unwrap().size;
    let texture = heap.new_texture_at_offset(&surface, OFFSET).unwrap();
    let kind = match texture.shadow().map(|copies| &copies.source) {
        Some(ShadowSource::Bytes {
            target: ShadowTarget::Image { .. },
            ..
        }) => "rows",
        Some(ShadowSource::Bytes {
            target: ShadowTarget::Tiled { .. },
            ..
        }) => "tiles",
        _ => {
            eprintln!("skipped: the device lays the texture in the heap's bytes, no shadow");
            return;
        }
    };
    eprintln!("BLIT-COVER: a shadow of the heap's bytes in {kind}");
    let source = |seed, side: u64| {
        device
            .new_buffer_with_bytes(&pattern(seed, side * side * 4), ResourceOptions::shared())
            .unwrap()
    };
    let (whole, part) = (source(1, SIDE), source(2, PART));

    // Every texel: no copy in and no keep; the encoder's one copy is the copy back.
    let [copies, copied_in, kept] = copy(&device, &queue, (&whole, &texture), 0, SIDE);
    eprintln!("BLIT-COVER: every texel: {copies} copies, {copied_in} texels in, {kept} kept");
    assert_eq!(
        [copies, copied_in, kept],
        [1, 0, 0],
        "a copy into every texel"
    );
    let mut expected = pattern(1, SIDE * SIDE * 4);
    assert_eq!(
        texels(&texture),
        expected,
        "the texels after a copy into every texel"
    );

    // A part: copied in, kept, merged back.
    let [copies, copied_in, kept] = copy(&device, &queue, (&part, &texture), AT, PART);
    eprintln!("BLIT-COVER: a part: {copies} copies, {copied_in} texels in, {kept} kept");
    assert_eq!(
        copies, 3,
        "a copy into a part: its copy in, keep and merge back"
    );
    // A shadow in rows counts the texels its copies move (`ShadowCopy::texels_in`).
    if kind == "rows" {
        assert!(
            copied_in > 0 && kept > 0,
            "a copy into a part copies in and keeps"
        );
    }
    let rows = pattern(2, PART * PART * 4);
    for y in 0..PART {
        let at = (((AT + y) * SIDE + AT) * 4) as usize;
        let from = (y * PART * 4) as usize;
        expected[at..at + (PART * 4) as usize]
            .copy_from_slice(&rows[from..from + (PART * 4) as usize]);
    }
    assert_eq!(
        texels(&texture),
        expected,
        "the texels after a copy into a part"
    );

    // No byte of the heap outside the texture changed: the host memory was zeroed.
    let words = host.words();
    let (first, last) = ((OFFSET / 4) as usize, ((OFFSET + span) / 4) as usize);
    assert!(words[..first].iter().all(|&word| word == 0), "bytes before");
    assert!(words[last..].iter().all(|&word| word == 0), "bytes after");
    drop(texture);
    drop(heap);
    drop(device);
}
