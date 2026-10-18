//! How block-compressed textures move their blocks (the consumer's x86 guests: BC textures on
//! NVIDIA showed corrupt texels), on the device the cfg selects, under the validation layer,
//! against a byte model of Metal's copies, and the model against plain Metal's own answers
//! ([`METAL_TRANSCRIPT`]: our own probe on an Apple M4, the same steps over the
//! same bytes).
//!
//! Every BC format, in five shapes whose levels end in partial blocks: a 2D texture 13 x 7 of four
//! levels (13 x 7 down to 1 x 1), a 2D array 10 x 6 of three slices and two levels, a cube 12 x 12
//! of three levels, a 3D texture 9 x 5 x 3 of two levels, and a 2D texture 8 x 8 of four levels
//! (its last two 2 x 2 and 1 x 1, smaller than a block). Per format and shape, five steps
//! ([`Step`]), each read back whole and compared with the model ([`Model`]):
//!
//! - `put`: `replaceRegion` of every level and slice whole, rows padded two blocks past their
//!   blocks and (3D) images one block past their rows, then one block replaced alone at block
//!   (1, 0) of level 0, slice 0; read back by `getBytes` of every level and slice whole, rows three
//!   blocks past and (3D) images two blocks past, into bytes of 0xcd;
//! - `getsub`: `getBytes` of each level wider than a block from texel x 4 to its edges;
//! - `out`: a blit of every level and slice into one buffer of 0xcd, each at five blocks past the
//!   last, rows one block past their blocks;
//! - `in`: a blit of a buffer into a second texture, each level and slice at three blocks past the
//!   last, rows two blocks past, then out tight;
//! - `t2t`: the first texture copied whole into a third, level by level and slice by slice, then
//!   a region of the second copied over part of it (block-aligned, to the source's edges), and the
//!   third blitted out tight.
//!
//! The bytes are the same for every format of one block size, so each step's bytes are too.
//!
//! The second lane ([`views_regions_and_large_levels_move_as_metal_s`]) adds a 2D texture of
//! 1030 x 520 and all eleven levels to the five shapes, and per format and shape ([`Step2`]):
//! every level and slice put whole, then read out through a view in the sibling format (the sRGB
//! twin, or the other member of a signed or float pair) made before the texture held anything;
//! a buffer blitted into interior regions of levels 0 and 1 to their edges; one block blitted in
//! through the view; and an interior `replaceRegion` to the edges. Plain Metal answered the same
//! 336 steps as the model ([`METAL_TRANSCRIPT_2`]).

use super::{validated_device_with, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandQueue, CpuCacheMode, Device, HazardTrackingMode,
    Origin, Region, ResourceOptions, Size, StorageMode, Texture, TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The SHA-256 of plain Metal's lines on an Apple M4 (macOS 27.0.1; our own probe
/// running these steps over these bytes), one `"<format> <shape> <step>: <n> bytes, sha <first six
/// bytes of the bytes' SHA-256>\n"` per format, shape and step in the test's order: Metal moved
/// every block as the model does.
const METAL_TRANSCRIPT: &str = "0e3f97649e3d295db1c7c44cfb39cd72da2ed1ca588c9349d9c52db420c6dfcd";

/// Every BC format and its block's bytes.
const FORMATS: [(PixelFormat, u64); 14] = [
    (PixelFormat::BC1RGBA, 8),
    (PixelFormat::BC1RGBASRGB, 8),
    (PixelFormat::BC2RGBA, 16),
    (PixelFormat::BC2RGBASRGB, 16),
    (PixelFormat::BC3RGBA, 16),
    (PixelFormat::BC3RGBASRGB, 16),
    (PixelFormat::BC4RUnorm, 8),
    (PixelFormat::BC4RSnorm, 8),
    (PixelFormat::BC5RGUnorm, 16),
    (PixelFormat::BC5RGSnorm, 16),
    (PixelFormat::BC6HRGBFloat, 16),
    (PixelFormat::BC6HRGBUfloat, 16),
    (PixelFormat::BC7RGBAUnorm, 16),
    (PixelFormat::BC7RGBAUnormSRGB, 16),
];

/// A texture's shape: its type, size (depth for 3D), slices (array length, or 6 for a cube) and
/// levels.
#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    ty: TextureType,
    size: [u64; 3],
    array: u64,
    levels: u64,
}

const SHAPES: [Shape; 5] = [
    Shape {
        name: "2d-13x7",
        ty: TextureType::Type2D,
        size: [13, 7, 1],
        array: 1,
        levels: 4,
    },
    Shape {
        name: "array-10x6x3",
        ty: TextureType::Type2DArray,
        size: [10, 6, 1],
        array: 3,
        levels: 2,
    },
    Shape {
        name: "cube-12",
        ty: TextureType::Cube,
        size: [12, 12, 1],
        array: 1,
        levels: 3,
    },
    Shape {
        name: "3d-9x5x3",
        ty: TextureType::Type3D,
        size: [9, 5, 3],
        array: 1,
        levels: 2,
    },
    Shape {
        name: "2d-8x8",
        ty: TextureType::Type2D,
        size: [8, 8, 1],
        array: 1,
        levels: 4,
    },
];

impl Shape {
    fn slices(&self) -> u64 {
        match self.ty {
            TextureType::Cube => 6,
            _ => self.array,
        }
    }

    fn three_d(&self) -> bool {
        self.ty == TextureType::Type3D
    }

    /// Level `l`'s size in texels.
    fn level(&self, l: u64) -> [u64; 3] {
        let [w, h, d] = self.size;
        [(w >> l).max(1), (h >> l).max(1), (d >> l).max(1)]
    }

    /// Level `l`'s size in blocks.
    fn blocks(&self, l: u64) -> [u64; 3] {
        let [w, h, d] = self.level(l);
        [w.div_ceil(4), h.div_ceil(4), d]
    }

    /// Every (level, slice), levels outermost.
    fn each(&self) -> Vec<(u64, u64)> {
        (0..self.levels)
            .flat_map(|l| (0..self.slices()).map(move |s| (l, s)))
            .collect()
    }

    /// The bytes between images of a level `l` copy whose rows are `row` apart, `extra` past its
    /// rows: 3D only, 0 for a 2D texture, a slice's rows for an array or cube.
    fn image(&self, l: u64, row: u64, extra: u64) -> u64 {
        match self.ty {
            TextureType::Type3D => row * self.blocks(l)[1] + extra,
            TextureType::Type2D => 0,
            _ => row * self.blocks(l)[1],
        }
    }
}

/// A splitmix64 stream (the probe's twin).
struct Rng(u64);

impl Rng {
    fn bytes(&mut self, n: u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u8
            })
            .collect()
    }
}

/// A texture's blocks as Metal's copies leave them: `(level, slice, z, y, x)` to the block's bytes.
#[derive(Clone, Default)]
struct Model(BTreeMap<(u64, u64, u64, u64, u64), Vec<u8>>);

impl Model {
    /// The blocks of `count` (x, y, z blocks) at block `at` of `(l, s)` from `bytes` at `offset`,
    /// rows `row` and images `image` apart.
    fn take(
        &mut self,
        (l, s): (u64, u64),
        at: [u64; 3],
        count: [u64; 3],
        (bytes, offset, row, image): (&[u8], u64, u64, u64),
        block: u64,
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let from = (offset + z * image + y * row + x * block) as usize;
                    let key = (l, s, at[2] + z, at[1] + y, at[0] + x);
                    self.0
                        .insert(key, bytes[from..from + block as usize].to_vec());
                }
            }
        }
    }

    /// The blocks of `count` at block `at` of `(l, s)` written over `bytes` at `offset`, rows
    /// `row` and images `image` apart.
    fn give(
        &self,
        (l, s): (u64, u64),
        at: [u64; 3],
        count: [u64; 3],
        (bytes, offset, row, image): (&mut [u8], u64, u64, u64),
        block: u64,
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let to = (offset + z * image + y * row + x * block) as usize;
                    let key = (l, s, at[2] + z, at[1] + y, at[0] + x);
                    bytes[to..to + block as usize].copy_from_slice(&self.0[&key]);
                }
            }
        }
    }

    /// `other`'s blocks of `count` at `from` of `(l, s)` copied to `to` of `(l2, s2)`.
    fn copy(
        &mut self,
        other: &Model,
        ((l, s), from): ((u64, u64), [u64; 3]),
        ((l2, s2), to): ((u64, u64), [u64; 3]),
        count: [u64; 3],
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let block = other.0[&(l, s, from[2] + z, from[1] + y, from[0] + x)].clone();
                    self.0
                        .insert((l2, s2, to[2] + z, to[1] + y, to[0] + x), block);
                }
            }
        }
    }
}

/// The steps, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Step {
    Put,
    GetSub,
    Out,
    In,
    TextureToTexture,
}

struct Lane {
    device: Device,
    queue: CommandQueue,
}

impl Lane {
    fn texture(&self, pf: PixelFormat, shape: &Shape, storage: StorageMode) -> Texture {
        self.texture_with(pf, shape, storage, TextureUsage::SHADER_READ)
    }

    fn texture_with(
        &self,
        pf: PixelFormat,
        shape: &Shape,
        storage: StorageMode,
        usage: TextureUsage,
    ) -> Texture {
        let [w, h, d] = shape.size;
        let mut desc = TextureDescriptor::texture_2d(pf, w, h, false);
        desc.texture_type = shape.ty;
        desc.depth = d;
        desc.array_length = shape.array;
        desc.mipmap_level_count = shape.levels;
        desc.usage = usage;
        desc.resource_options = ResourceOptions::new(
            storage,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Default,
        );
        self.device.new_texture(&desc).unwrap()
    }

    fn buffer(&self, bytes: &[u8]) -> Buffer {
        let buffer = self
            .device
            .new_buffer(bytes.len() as u64, ResourceOptions::shared())
            .unwrap();
        // SAFETY: the buffer holds `bytes.len()` bytes at its contents; no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().unwrap().as_ptr(),
                bytes.len(),
            );
        }
        buffer
    }

    fn bytes_of(buffer: &Buffer, n: usize) -> Vec<u8> {
        // SAFETY: the command buffers that wrote it completed; it holds at least `n` bytes.
        unsafe { std::slice::from_raw_parts(buffer.contents().unwrap().as_ptr(), n) }.to_vec()
    }

    fn run(&self, record: impl FnOnce(&mut BlitCommandEncoder)) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        record(&mut blit);
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// Every level and slice of `texture` blitted out tight, one after another from byte 0, into
    /// a buffer of 0xcd; with the model's bytes for `model`.
    fn out_tight(
        &self,
        texture: &Texture,
        shape: &Shape,
        block: u64,
        model: &Model,
    ) -> [Vec<u8>; 2] {
        let mut places = Vec::new();
        let mut at = 0;
        for (l, s) in shape.each() {
            let [bx, by, bz] = shape.blocks(l);
            let row = bx * block;
            let image = shape.image(l, row, 0);
            places.push((l, s, at, row, image));
            at += row * by * bz;
        }
        let out = self.buffer(&vec![0xcd; at as usize]);
        self.run(|blit| {
            for &(l, s, offset, row, image) in &places {
                let [w, h, d] = shape.level(l);
                blit.copy_from_texture_to_buffer(
                    texture,
                    s,
                    l,
                    Origin::new(0, 0, 0),
                    Size::new(w, h, d),
                    &out,
                    offset,
                    row,
                    image,
                    BlitOption::None,
                )
                .unwrap();
            }
        });
        let mut want = vec![0xcd; at as usize];
        for &(l, s, offset, row, image) in &places {
            let image = if shape.three_d() { image } else { 0 };
            model.give(
                (l, s),
                [0; 3],
                shape.blocks(l),
                (&mut want, offset, row, image),
                block,
            );
        }
        [Self::bytes_of(&out, at as usize), want]
    }
}

/// The steps over `pf` in `shape`: per step, the device's bytes and the model's.
fn steps(
    lane: &Lane,
    pf: PixelFormat,
    block: u64,
    shape: &Shape,
    seed: u64,
) -> Vec<(Step, Vec<u8>, Vec<u8>)> {
    let mut rng = Rng(seed);
    let mut results = Vec::new();
    let three_d = shape.three_d();
    // put: every level and slice whole, then one block alone.
    let first = lane.texture(pf, shape, StorageMode::Shared);
    let mut model = Model::default();
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 2 * block;
        let image = if three_d { row * by + block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let bytes = rng.bytes(total);
        first
            .replace_region(Region::new_3d(0, 0, 0, w, h, d), l, s, &bytes, row, image)
            .unwrap();
        model.take((l, s), [0; 3], [bx, by, bz], (&bytes, 0, row, image), block);
    }
    if shape.blocks(0)[0] >= 2 {
        let h = shape.level(0)[1].min(4);
        let bytes = rng.bytes(block);
        let image = if three_d { block } else { 0 };
        first
            .replace_region(Region::new_3d(4, 0, 0, 4, h, 1), 0, 0, &bytes, block, image)
            .unwrap();
        model.take(
            (0, 0),
            [1, 0, 0],
            [1, 1, 1],
            (&bytes, 0, block, image),
            block,
        );
    }
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 3 * block;
        let image = if three_d { row * by + 2 * block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let mut bytes = vec![0xcd; total as usize];
        first
            .get_bytes(
                &mut bytes,
                row,
                image,
                Region::new_3d(0, 0, 0, w, h, d),
                l,
                s,
            )
            .unwrap();
        got.extend_from_slice(&bytes);
        let mut expected = vec![0xcd; total as usize];
        model.give(
            (l, s),
            [0; 3],
            [bx, by, bz],
            (&mut expected, 0, row, image),
            block,
        );
        want.extend_from_slice(&expected);
    }
    results.push((Step::Put, got, want));
    // getsub: from texel x 4 to the level's edges.
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        if w <= 4 {
            continue;
        }
        let across = bx - 1;
        let row = across * block + block;
        let image = if three_d { row * by + block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let mut bytes = vec![0xcd; total as usize];
        first
            .get_bytes(
                &mut bytes,
                row,
                image,
                Region::new_3d(4, 0, 0, w - 4, h, d),
                l,
                s,
            )
            .unwrap();
        got.extend_from_slice(&bytes);
        let mut expected = vec![0xcd; total as usize];
        model.give(
            (l, s),
            [1, 0, 0],
            [across, by, bz],
            (&mut expected, 0, row, image),
            block,
        );
        want.extend_from_slice(&expected);
    }
    results.push((Step::GetSub, got, want));
    // out: each level and slice five blocks past the last, rows one block past.
    let mut places = Vec::new();
    let mut at = 0;
    for (l, s) in shape.each() {
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + block;
        let image = shape.image(l, row, block);
        at += 5 * block;
        places.push((l, s, at, row, image));
        at += if three_d { image * bz } else { row * by };
    }
    let out = lane.buffer(&vec![0xcd; at as usize]);
    lane.run(|blit| {
        for &(l, s, offset, row, image) in &places {
            let [w, h, d] = shape.level(l);
            blit.copy_from_texture_to_buffer(
                &first,
                s,
                l,
                Origin::new(0, 0, 0),
                Size::new(w, h, d),
                &out,
                offset,
                row,
                image,
                BlitOption::None,
            )
            .unwrap();
        }
    });
    let mut expected = vec![0xcd; at as usize];
    for &(l, s, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        model.give(
            (l, s),
            [0; 3],
            shape.blocks(l),
            (&mut expected, offset, row, image),
            block,
        );
    }
    results.push((Step::Out, Lane::bytes_of(&out, at as usize), expected));
    // in: a buffer into a second texture, each level and slice three blocks past the last, rows
    // two blocks past; out tight.
    let second = lane.texture(pf, shape, StorageMode::Private);
    let mut second_model = Model::default();
    let mut places = Vec::new();
    let mut at = 0;
    for (l, s) in shape.each() {
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 2 * block;
        let image = shape.image(l, row, 2 * block);
        at += 3 * block;
        places.push((l, s, at, row, image));
        at += if three_d { image * bz } else { row * by };
    }
    let source = rng.bytes(at);
    let source_buffer = lane.buffer(&source);
    lane.run(|blit| {
        for &(l, s, offset, row, image) in &places {
            let [w, h, d] = shape.level(l);
            blit.copy_from_buffer_to_texture(
                &source_buffer,
                offset,
                row,
                image,
                Size::new(w, h, d),
                &second,
                s,
                l,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
        }
    });
    for &(l, s, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        second_model.take(
            (l, s),
            [0; 3],
            shape.blocks(l),
            (&source, offset, row, image),
            block,
        );
    }
    let [got, want] = lane.out_tight(&second, shape, block, &second_model);
    results.push((Step::In, got, want));
    // t2t: the first texture whole into a third, then a region of the second over part of it.
    let third = lane.texture(pf, shape, StorageMode::Private);
    let mut third_model = Model::default();
    let (from, to, size, ls_from, ls_to) = match shape.name {
        // Blocks (1..3, 1) of level 0 to (0..2, 1): x interior, y to the edge.
        "2d-13x7" => ([4, 4, 0], [0, 4, 0], [8, 3, 1], (0, 0), (0, 0)),
        // Block (1, 1) of level 0 to (0, 0).
        "2d-8x8" => ([4, 4, 0], [0, 0, 0], [4, 4, 1], (0, 0), (0, 0)),
        // Level 1 of slice 2 whole to slice 0.
        "array-10x6x3" => ([0, 0, 0], [0, 0, 0], [5, 3, 1], (1, 2), (1, 0)),
        // Blocks (1..3, 0..3) of face 2 to face 5: to the edges.
        "cube-12" => ([4, 0, 0], [4, 0, 0], [8, 12, 1], (0, 2), (0, 5)),
        // Depth 1..3 of level 0 to depth 0..2.
        _ => ([0, 0, 1], [0, 0, 0], [9, 5, 2], (0, 0), (0, 0)),
    };
    lane.run(|blit| {
        for (l, s) in shape.each() {
            let [w, h, d] = shape.level(l);
            let at = Origin::new(0, 0, 0);
            blit.copy_from_texture_to_texture(
                &first,
                s,
                l,
                at,
                Size::new(w, h, d),
                &third,
                s,
                l,
                at,
            )
            .unwrap();
        }
        blit.copy_from_texture_to_texture(
            &second,
            ls_from.1,
            ls_from.0,
            Origin::new(from[0], from[1], from[2]),
            Size::new(size[0], size[1], size[2]),
            &third,
            ls_to.1,
            ls_to.0,
            Origin::new(to[0], to[1], to[2]),
        )
        .unwrap();
    });
    for (l, s) in shape.each() {
        third_model.copy(&model, ((l, s), [0; 3]), ((l, s), [0; 3]), shape.blocks(l));
    }
    let blocks = |v: [u64; 3]| [v[0] / 4, v[1] / 4, v[2]];
    let count = [size[0].div_ceil(4), size[1].div_ceil(4), size[2]];
    third_model.copy(
        &second_model,
        (ls_from, blocks(from)),
        (ls_to, blocks(to)),
        count,
    );
    let [got, want] = lane.out_tight(&third, shape, block, &third_model);
    results.push((Step::TextureToTexture, got, want));
    results
}

fn digest(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().take(6).map(|b| format!("{b:02x}")).collect()
}

/// Where `got` first differs from `want`, as a byte offset, and how many bytes differ.
fn first_difference(got: &[u8], want: &[u8]) -> Option<(usize, usize)> {
    let first = got.iter().zip(want).position(|(a, b)| a != b)?;
    let count = got.iter().zip(want).filter(|(a, b)| a != b).count();
    Some((first, count))
}

/// Every format's blocks move in every shape as the model of Metal's copies moves them.
#[test]
fn compressed_blocks_move_as_metal_moves_them() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
    let name = device.name().to_owned();
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        device,
    };
    let mut wrong = Vec::new();
    // The model's lines, as plain Metal printed its own.
    let mut transcript = Sha256::new();
    for (pf, block) in FORMATS {
        for (k, shape) in SHAPES.iter().enumerate() {
            let seed = 0x5eed_0000 + block * 0x100 + k as u64;
            for (step, got, want) in steps(&lane, pf, block, shape, seed) {
                let line = format!(
                    "{pf:?} {} {step:?}: {} bytes, sha {} (model {})",
                    shape.name,
                    got.len(),
                    digest(&got),
                    digest(&want)
                );
                eprintln!("{line}");
                transcript.update(format!(
                    "{pf:?} {} {step:?}: {} bytes, sha {}\n",
                    shape.name,
                    want.len(),
                    digest(&want)
                ));
                if let Some((at, count)) = first_difference(&got, &want) {
                    wrong.push(format!("{line}: {count} bytes differ, the first at {at}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
    let transcript: String = transcript
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        transcript, METAL_TRANSCRIPT,
        "the model no longer moves blocks as plain Metal on an Apple M4 moved them"
    );
    assert_eq!(
        messages.take(),
        Vec::new(),
        "the validation layer reported on {name}"
    );
}

// ---- The second lane: views, interior regions and a large texture ----

/// The SHA-256 of plain Metal's lines for [`views_regions_and_large_levels_move_as_metal_s`]
/// (an Apple M4, macOS 27.0.1; our own probe), as [`METAL_TRANSCRIPT`] is the
/// first test's: Metal moved every block as the model does.
const METAL_TRANSCRIPT_2: &str = "888ee9af670a85af8ca6b60fa45ef0aff50a47f08947827cf1bb9f5046141ddd";

/// A large 2D texture of every level, whose rows and levels end in partial blocks.
const LARGE: Shape = Shape {
    name: "2d-1030x520",
    ty: TextureType::Type2D,
    size: [1030, 520, 1],
    array: 1,
    levels: 11,
};

/// `pf`'s view format: its sRGB twin, or the other member of its signed or float pair, and
/// whether a view of it needs `PixelFormatView` usage (the sRGB twins do not).
fn sibling(pf: PixelFormat) -> (PixelFormat, bool) {
    use PixelFormat as P;
    match pf {
        P::BC1RGBA => (P::BC1RGBASRGB, false),
        P::BC1RGBASRGB => (P::BC1RGBA, false),
        P::BC2RGBA => (P::BC2RGBASRGB, false),
        P::BC2RGBASRGB => (P::BC2RGBA, false),
        P::BC3RGBA => (P::BC3RGBASRGB, false),
        P::BC3RGBASRGB => (P::BC3RGBA, false),
        P::BC4RUnorm => (P::BC4RSnorm, true),
        P::BC4RSnorm => (P::BC4RUnorm, true),
        P::BC5RGUnorm => (P::BC5RGSnorm, true),
        P::BC5RGSnorm => (P::BC5RGUnorm, true),
        P::BC6HRGBFloat => (P::BC6HRGBUfloat, true),
        P::BC6HRGBUfloat => (P::BC6HRGBFloat, true),
        P::BC7RGBAUnorm => (P::BC7RGBAUnormSRGB, false),
        _ => (P::BC7RGBAUnorm, false),
    }
}

/// The second lane's steps, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Step2 {
    /// Every level and slice put whole by `replaceRegion`, then blitted out tight through a view
    /// in the sibling format made before the texture held anything.
    ViewOut,
    /// A buffer blitted into interior regions of levels 0 and 1 of the last slice, to the
    /// level's edges; the texture out tight.
    InSub,
    /// One block blitted from a buffer through the view; the texture out tight.
    ViewIn,
    /// `replaceRegion` of an interior region of level 0, slice 0, to its edges; the texture out
    /// tight.
    PutSub,
}

/// The region from texel `(4, 4)` (and depth 1 of a 3D texture deeper than 1) of level `l` to
/// its edges, as an origin and a size in texels, and its first block; `None` where the level is
/// a block or less across or down.
fn interior(shape: &Shape, l: u64) -> Option<([u64; 3], [u64; 3], [u64; 3])> {
    let [w, h, d] = shape.level(l);
    if w <= 4 || h <= 4 {
        return None;
    }
    let z = u64::from(shape.three_d() && d > 1);
    Some(([4, 4, z], [w - 4, h - 4, d - z], [1, 1, z]))
}

/// The second lane's steps over `pf` in `shape`: per step, the device's bytes and the model's.
fn steps2(
    lane: &Lane,
    pf: PixelFormat,
    block: u64,
    shape: &Shape,
    seed: u64,
) -> Vec<(Step2, Vec<u8>, Vec<u8>)> {
    let mut rng = Rng(seed);
    let mut results = Vec::new();
    let three_d = shape.three_d();
    let (view_format, needs_usage) = sibling(pf);
    let usage = match needs_usage {
        true => TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        false => TextureUsage::SHADER_READ,
    };
    let first = lane.texture_with(pf, shape, StorageMode::Shared, usage);
    let view = first.new_texture_view(view_format).unwrap();
    let mut model = Model::default();
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 2 * block;
        let image = if three_d { row * by + block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let bytes = rng.bytes(total);
        first
            .replace_region(Region::new_3d(0, 0, 0, w, h, d), l, s, &bytes, row, image)
            .unwrap();
        model.take((l, s), [0; 3], [bx, by, bz], (&bytes, 0, row, image), block);
    }
    let [got, want] = lane.out_tight(&view, shape, block, &model);
    results.push((Step2::ViewOut, got, want));
    // insub: levels 0 and 1 of the last slice, from (4, 4) to the edges.
    let s = shape.slices() - 1;
    let mut places = Vec::new();
    let mut at = 0;
    for l in 0..shape.levels.min(2) {
        let Some((origin, size, first_block)) = interior(shape, l) else {
            continue;
        };
        let [bx, by, _] = shape.blocks(l);
        let count = [bx - first_block[0], by - first_block[1], size[2]];
        let row = count[0] * block + block;
        let image = shape.image(l, row, block);
        at += 2 * block;
        places.push((l, origin, size, first_block, count, at, row, image));
        at += if three_d {
            image * count[2]
        } else {
            row * count[1]
        };
    }
    let source = rng.bytes(at.max(1));
    let source_buffer = lane.buffer(&source);
    lane.run(|blit| {
        for &(l, origin, size, _, _, offset, row, image) in &places {
            blit.copy_from_buffer_to_texture(
                &source_buffer,
                offset,
                row,
                image,
                Size::new(size[0], size[1], size[2]),
                &first,
                s,
                l,
                Origin::new(origin[0], origin[1], origin[2]),
                BlitOption::None,
            )
            .unwrap();
        }
    });
    for &(l, _, _, first_block, count, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        model.take(
            (l, s),
            first_block,
            count,
            (&source, offset, row, image),
            block,
        );
    }
    let [got, want] = lane.out_tight(&first, shape, block, &model);
    results.push((Step2::InSub, got, want));
    // viewin: one block through the view.
    let [w, h, _] = shape.level(0);
    let one = rng.bytes(block);
    let one_buffer = lane.buffer(&one);
    lane.run(|blit| {
        blit.copy_from_buffer_to_texture(
            &one_buffer,
            0,
            block,
            if three_d { block } else { 0 },
            Size::new(w.min(4), h.min(4), 1),
            &view,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
    });
    model.take((0, 0), [0; 3], [1, 1, 1], (&one, 0, block, 0), block);
    let [got, want] = lane.out_tight(&first, shape, block, &model);
    results.push((Step2::ViewIn, got, want));
    // putsub: level 0, slice 0, from (4, 4) to the edges, rows one block past.
    if let Some((origin, size, first_block)) = interior(shape, 0) {
        let [bx, by, _] = shape.blocks(0);
        let count = [bx - first_block[0], by - first_block[1], size[2]];
        let row = count[0] * block + block;
        let image = if three_d { row * count[1] + block } else { 0 };
        let total = if three_d {
            image * count[2]
        } else {
            row * count[1]
        };
        let bytes = rng.bytes(total);
        let region = Region::new_3d(origin[0], origin[1], origin[2], size[0], size[1], size[2]);
        first
            .replace_region(region, 0, 0, &bytes, row, image)
            .unwrap();
        model.take((0, 0), first_block, count, (&bytes, 0, row, image), block);
    }
    let [got, want] = lane.out_tight(&first, shape, block, &model);
    results.push((Step2::PutSub, got, want));
    results
}

/// Every format's blocks move through a view made before the texture held anything, into
/// interior regions, and in a large texture of every level, as the model of Metal's copies moves
/// them (the module docs' second lane).
#[test]
fn views_regions_and_large_levels_move_as_metal_s() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
    let name = device.name().to_owned();
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        device,
    };
    let mut wrong = Vec::new();
    let mut transcript = Sha256::new();
    let shapes: Vec<Shape> = SHAPES.iter().copied().chain([LARGE]).collect();
    for (pf, block) in FORMATS {
        for (k, shape) in shapes.iter().enumerate() {
            let seed = 0x5eed_1000 + block * 0x100 + k as u64;
            for (step, got, want) in steps2(&lane, pf, block, shape, seed) {
                let line = format!(
                    "{pf:?} {} {step:?}: {} bytes, sha {} (model {})",
                    shape.name,
                    got.len(),
                    digest(&got),
                    digest(&want)
                );
                eprintln!("{line}");
                transcript.update(format!(
                    "{pf:?} {} {step:?}: {} bytes, sha {}\n",
                    shape.name,
                    want.len(),
                    digest(&want)
                ));
                if let Some((at, count)) = first_difference(&got, &want) {
                    wrong.push(format!("{line}: {count} bytes differ, the first at {at}"));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "on {name}:\n{}", wrong.join("\n"));
    let transcript: String = transcript
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        transcript, METAL_TRANSCRIPT_2,
        "the model no longer moves blocks as plain Metal on an Apple M4 moved them"
    );
    assert_eq!(
        messages.take(),
        Vec::new(),
        "the validation layer reported on {name}"
    );
}
