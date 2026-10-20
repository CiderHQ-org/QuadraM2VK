//! A blit encoder's commands over untracked memory, ordered per key and range (`tracker.rs`,
//! **Inside an encoder**): Metal runs a blit encoder's commands in encoded order, which orders
//! two commands only where their bytes can meet. Copies into disjoint bytes of a heap's untracked
//! buffers take no barrier between them, copies into overlapping bytes still do, and each leaves
//! the bytes it copied; copies into textures of a heap over host pages, two of them over the same
//! bytes, leave the last copy's texels in both and the other texture's in its own.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Host, Scratch, WAIT};
use crate::{
    BlitCommandEncoder, BlitOption, Buffer, CommandQueue, Device, HeapDescriptor, HeapType, Origin,
    Region, ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// Each heap buffer's bytes.
const LENGTH: u64 = 64 << 10;

/// The textures' side: 64 x 64 `BGRA8Unorm`, one of Metal's tiles of 4-byte elements.
const SIDE: u64 = 64;

/// `LENGTH` bytes that name their place and `seed`.
fn pattern(seed: u8, length: u64) -> Vec<u8> {
    (0..length)
        .map(|i| (i as u8) ^ (i >> 8) as u8 ^ seed)
        .collect()
}

/// Run one command buffer of a blit encoder `encode` fills; the barriers recorded between
/// ordered encoders' commands meanwhile (`WorkCounts::command_barriers`).
fn barriers(
    device: &Device,
    queue: &CommandQueue,
    encode: impl FnOnce(&mut BlitCommandEncoder),
) -> u64 {
    let before = device.work_counts().command_barriers;
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    device.work_counts().command_barriers - before
}

/// The bytes of a `Shared` buffer.
fn bytes(buffer: &Buffer, length: u64) -> Vec<u8> {
    let contents = buffer.contents().expect("a Shared buffer's contents");
    // SAFETY: the buffer is `Shared`, `length` bytes long at least, and no command uses it now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

#[test]
fn a_blits_copies_into_disjoint_bytes_of_a_heap_take_no_barrier_and_overlapping_ones_do() {
    let scratch = Scratch::new("blit-order-buffers");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    // A heap's buffers are untracked (`HazardTrackingMode::Default`).
    let descriptor =
        HeapDescriptor::new(HeapType::Placement, 4 * LENGTH, ResourceOptions::shared());
    let heap = device.new_heap(&descriptor).unwrap();
    let placed = |offset| {
        heap.new_buffer_at_offset(LENGTH, ResourceOptions::shared(), offset)
            .unwrap()
    };
    let (a, b, c) = (placed(0), placed(LENGTH), placed(LENGTH / 2));
    let source = |seed| {
        device
            .new_buffer_with_bytes(&pattern(seed, LENGTH), ResourceOptions::shared())
            .unwrap()
    };
    let (one, two, three) = (source(1), source(2), source(3));

    // Disjoint bytes of one heap: no order between them.
    let disjoint = barriers(&device, &queue, |blit| {
        blit.copy_from_buffer_to_buffer(&one, 0, &a, 0, LENGTH)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&two, 0, &b, 0, LENGTH)
            .unwrap();
    });
    assert_eq!(disjoint, 0, "copies into disjoint bytes of a heap");
    assert_eq!(bytes(&a, LENGTH), pattern(1, LENGTH));
    assert_eq!(bytes(&b, LENGTH), pattern(2, LENGTH));

    // Overlapping bytes (c's first half is a's second): the second copy after the first.
    let overlapping = barriers(&device, &queue, |blit| {
        blit.copy_from_buffer_to_buffer(&one, 0, &a, 0, LENGTH)
            .unwrap();
        blit.copy_from_buffer_to_buffer(&three, 0, &c, 0, LENGTH)
            .unwrap();
    });
    assert_eq!(overlapping, 1, "copies into overlapping bytes of a heap");
    let half = (LENGTH / 2) as usize;
    let mut expected = pattern(1, LENGTH);
    expected[half..].copy_from_slice(&pattern(3, LENGTH)[..half]);
    assert_eq!(bytes(&a, LENGTH), expected);
    assert_eq!(bytes(&c, LENGTH), pattern(3, LENGTH));
}

#[test]
fn a_blits_copies_into_textures_of_a_heap_over_host_pages_leave_each_its_texels() {
    let scratch = Scratch::new("blit-order-textures");
    let host = Host::new();
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let size = 4 * SIDE * SIDE * 4;
    let descriptor = HeapDescriptor::new(HeapType::Placement, size, ResourceOptions::shared());
    // SAFETY: the host memory outlives the heap and its textures (dropped after the device), and
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
    let placed = |offset| -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, SIDE, SIDE, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        descriptor.allow_gpu_optimized_contents = false;
        heap.new_texture_at_offset(&descriptor, offset).unwrap()
    };
    let tile = SIDE * SIDE * 4;
    // `first` and `last` lie over the same bytes, `other` over the next tile.
    let (first, other, last) = (placed(0), placed(tile), placed(0));
    let source = |seed| {
        device
            .new_buffer_with_bytes(&pattern(seed, tile), ResourceOptions::shared())
            .unwrap()
    };
    let (one, two, three) = (source(1), source(2), source(3));
    let copy = |blit: &mut BlitCommandEncoder, from: &Buffer, to: &Texture| {
        blit.copy_from_buffer_to_texture(
            from,
            0,
            SIDE * 4,
            tile,
            Size::new(SIDE, SIDE, 1),
            to,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
    };
    barriers(&device, &queue, |blit| {
        copy(blit, &one, &first);
        copy(blit, &two, &other);
        copy(blit, &three, &last);
    });
    let texels = |texture: &Texture| {
        let mut read = vec![0u8; tile as usize];
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        texture
            .get_bytes(&mut read, SIDE * 4, tile, region, 0, 0)
            .unwrap();
        read
    };
    // The last copy's texels in both textures over its bytes, the other copy's in its own.
    assert_eq!(texels(&last), pattern(3, tile));
    assert_eq!(texels(&first), pattern(3, tile));
    assert_eq!(texels(&other), pattern(2, tile));
}
