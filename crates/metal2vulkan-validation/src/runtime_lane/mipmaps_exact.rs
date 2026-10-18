//! `generate_mipmaps` computed as Metal's filter (`runtime::mipmap_filter`) on the device the cfg
//! selects, under the validation layer: every format the filter computes, level by level, is
//! `bind::MipmapFilterChain::expected_levels` from the first level as the device read it back; and
//! the 16-bit unorm and snorm formats on chains that halve exactly are the chain too, the boxes an
//! Apple M4 measured among them, on every device, never refused, and announced once
//! (`mipmaps-as-apple-m4`) exactly where an Apple M3 Max's answer may differ.
//!
//! a Metal probe measured every value of such chains of `R16Unorm`, `RG16Unorm`, `RGBA16Unorm`,
//! `R16Snorm`, `RG16Snorm` and `RGBA16Snorm` on an Apple M4 (macOS 27.0.1) and an Apple M3 Max
//! (macOS 26.7), plain and under the API validation layer: 2D textures (square, wide, tall, one
//! texel high or wide), a 2D array, a cube, and 3D textures, seeded from random codes, near
//! neighbours and the extremes. On the M4 each component of a level is the average of its box in
//! the level above (2x2, 2 where an axis is 1, 2x2x2 in 3D) rounded half up to a code, an snorm
//! code of -32768 read as -32767. The M3 Max agrees on every 3D level and every first level; below
//! a 2D chain's first level it runs its own arithmetic, the chain's tail where the first level is
//! at most 64 texels wide and tall.
//!
//! The chains here are seeded as probe 10 seeded its textures; each chain is compared with the
//! CPU's from its first level as the device read it back, and the boxes the M4 measured are
//! compared with its bytes.

use super::degrade_notices::assert_once;
use super::{validated_device, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{MipmapFilterChain, MipmapRule};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    CommandQueue, Degradation, Device, MetalLevel, Region, ResourceOptions, Texture,
    TextureDescriptor,
};

/// The kinds the computed chains take: the M4's answer where an Apple M3 Max's differs, every
/// venue, in `the_m3_note_is_announced_exactly_where_an_m3_differs`.
pub(super) const M3_MIPMAPS: [Degradation; 1] = [Degradation::MipmapsAsAppleM4];

/// The six formats and their channels.
const FORMATS: [(PixelFormat, usize); 6] = [
    (PixelFormat::R16Unorm, 1),
    (PixelFormat::RG16Unorm, 2),
    (PixelFormat::RGBA16Unorm, 4),
    (PixelFormat::R16Snorm, 1),
    (PixelFormat::RG16Snorm, 2),
    (PixelFormat::RGBA16Snorm, 4),
];

/// `n` codes from a seed: half uniform, a quarter extremes, a quarter near the previous one, as
/// probe 10 seeded its textures.
fn seeded(n: usize, seed: &mut u64) -> Vec<u16> {
    const EXTREMES: [u16; 12] = [
        0, 1, 2, 3, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0x8002, 0xfffd, 0xfffe, 0xffff,
    ];
    let mut last = 0u16;
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            let v = match *seed % 4 {
                0 | 1 => (*seed >> 16) as u16,
                2 => EXTREMES[((*seed >> 8) % EXTREMES.len() as u64) as usize],
                _ => last.wrapping_add(((*seed >> 8) % 5) as u16).wrapping_sub(2),
            };
            last = v;
            v
        })
        .collect()
}

fn bytes(codes: &[u16]) -> Vec<u8> {
    codes.iter().flat_map(|c| c.to_le_bytes()).collect()
}

fn codes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// A Shared texture of `pf` and `ty`, `extent`, `levels` and `slices` (cube faces included).
fn texture(
    device: &metal2vulkan_runtime::Device,
    pf: PixelFormat,
    ty: TextureType,
    extent: [usize; 3],
    levels: usize,
    slices: usize,
) -> Texture {
    texture_allowing(device, pf, ty, extent, levels, slices, true)
}

/// [`texture`]'s, with `allowGPUOptimizedContents` as `allow` says (off, Metal does not compress
/// it).
fn texture_allowing(
    device: &metal2vulkan_runtime::Device,
    pf: PixelFormat,
    ty: TextureType,
    extent: [usize; 3],
    levels: usize,
    slices: usize,
    allow: bool,
) -> Texture {
    let mut d = TextureDescriptor::texture_2d(pf, extent[0] as u64, extent[1] as u64, false);
    d.allow_gpu_optimized_contents = allow;
    d.texture_type = ty;
    d.depth = extent[2] as u64;
    d.mipmap_level_count = levels as u64;
    if ty == TextureType::Type2DArray {
        d.array_length = slices as u64;
    }
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::shared();
    device.new_texture(&d).expect("a texture")
}

/// The row and image strides of `extent`'s texels of `texel` bytes, tightly packed: an image
/// stride only for a texture of several images.
fn strides([w, h, d]: [usize; 3], texel: usize) -> (u64, u64) {
    let row = (w * texel) as u64;
    (row, if d > 1 { row * h as u64 } else { 0 })
}

/// Write `bytes` into `level` of `slice` of `t`, of `extent`, `texel` bytes a texel.
fn put(t: &Texture, slice: usize, level: usize, extent: [usize; 3], texel: usize, bytes: &[u8]) {
    let [w, h, d] = extent;
    let (row, image) = strides(extent, texel);
    let region = Region::new_3d(0, 0, 0, w as u64, h as u64, d as u64);
    t.replace_region(region, level as u64, slice as u64, bytes, row, image)
        .expect("the level written");
}

/// The bytes of `level` of `slice` of `t`, of `extent`, `texel` bytes a texel.
fn get(t: &Texture, slice: usize, level: usize, extent: [usize; 3], texel: usize) -> Vec<u8> {
    let [w, h, d] = extent;
    let (row, image) = strides(extent, texel);
    let mut out = vec![0u8; w * h * d * texel];
    let region = Region::new_3d(0, 0, 0, w as u64, h as u64, d as u64);
    t.get_bytes(&mut out, row, image, region, level as u64, slice as u64)
        .expect("the level's bytes");
    out
}

/// Write `codes` into `level` of `slice` of `t`, of `extent`.
fn write(t: &Texture, slice: usize, level: usize, extent: [usize; 3], ch: usize, codes: &[u16]) {
    put(t, slice, level, extent, ch * 2, &bytes(codes));
}

/// `level` of `slice` of `t`, of `extent`.
fn level(t: &Texture, slice: usize, level: usize, extent: [usize; 3], ch: usize) -> Vec<u16> {
    codes(&get(t, slice, level, extent, ch * 2))
}

/// Generate `t`'s mipmaps in one command buffer of `queue`.
fn generate(queue: &CommandQueue, t: &Texture) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.generate_mipmaps(t)
        .expect("the exact chain runs on every device");
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The lane's device answering at Metal level `level`, whose release's mipmap rule its chains
/// take (`MetalLevel::mipmap_rule`).
fn device_at(level: MetalLevel, messages: &Messages, cache: &ScratchCache) -> Device {
    let device = validated_device_with(TEST_DEVICE, messages, cache.path(), |options| {
        options.metal_level = Some(level);
    })
    .unwrap();
    assert_eq!(device.properties().metal_level, level);
    device
}

/// Whether Metal keeps a texture [`texture`] makes of `extent` losslessly compressed (`ShaderRead`
/// usage, `allowGPUOptimizedContents` on): where its shorter side is 16 texels or more
/// (`MipmapFilterChain::plan`'s `compressed`).
fn compressed(extent: [u32; 3]) -> bool {
    extent[0].min(extent[1]) >= 16
}

/// The chain of `levels` levels of `slices` layers of `pf` from a first level of `extent`, as
/// the runtime plans it on a device of generous limits: of a texture [`texture`] makes, or with
/// `allowGPUOptimizedContents` off where `allow` is false ([`texture_allowing`]).
fn chain_of(
    pf: PixelFormat,
    ty: TextureType,
    extent: [usize; 3],
    levels: usize,
    slices: usize,
    allow: bool,
    rule: MipmapRule,
) -> MipmapFilterChain {
    MipmapFilterChain::plan(
        pf,
        extent.map(|a| a as u32),
        slices as u32,
        levels as u32,
        ty == TextureType::Type3D,
        allow && compressed(extent.map(|a| a as u32)),
        rule,
        64,
        u32::MAX,
        [65535, 65535],
        None,
    )
    .expect("a chain")
}

/// Every level of halving chains of the six formats, in 2D (square, wide, tall, one texel high,
/// one wide, one past the M3 Max's tail's 64 texels), as a 2D array and a cube, in 3D, and
/// through a view of a texture's later levels, is the chain on the CPU from the first level at the
/// device's Metal level; the M4's answer is never announced (the strip past 64 texels takes the M3
/// Max's passes at macOS 26.7, its filter at 27.0.1), nor the blit chain's near-equivalent.
#[test]
fn mipmaps_of_the_16_bit_fixed_formats_on_halving_chains_are_metal_s() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let rule = device.properties().metal_level.mipmap_rule();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let chains: [(TextureType, [usize; 3], usize); 11] = [
            (TextureType::Type2D, [16, 16, 1], 1),
            (TextureType::Type2D, [64, 4, 1], 1),
            (TextureType::Type2D, [4, 32, 1], 1),
            (TextureType::Type2D, [1, 16, 1], 1),
            (TextureType::Type2D, [16, 1, 1], 1),
            (TextureType::Type2D, [128, 2, 1], 1),
            (TextureType::Type2DArray, [8, 8, 1], 3),
            (TextureType::Cube, [8, 8, 1], 6),
            (TextureType::Type3D, [4, 4, 4], 1),
            (TextureType::Type3D, [8, 8, 2], 1),
            (TextureType::Type3D, [2, 2, 8], 1),
        ];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut checked = 0usize;
        for (pf, ch) in FORMATS {
            for (ty, extent, slices) in chains {
                let levels = 1 + extent.iter().max().unwrap().ilog2() as usize;
                let t = texture(&device, pf, ty, extent, levels, slices);
                let [w, h, d] = extent;
                for slice in 0..slices {
                    let first = seeded(w * h * d * ch, &mut seed);
                    write(&t, slice, 0, extent, ch, &first);
                }
                generate(&queue, &t);
                let at = |l: usize| extent.map(|a| (a >> l).max(1));
                let read = |l: usize| -> Vec<u8> {
                    (0..slices)
                        .flat_map(|slice| bytes(&level(&t, slice, l, at(l), ch)))
                        .collect()
                };
                let chain = chain_of(pf, ty, extent, levels, slices, true, rule);
                let want = chain.expected_levels(&read(0)).expect("the levels");
                for (l, want) in want.iter().enumerate().skip(1) {
                    let got = read(l);
                    assert_eq!(&got, want, "{pf:?} {ty:?} {extent:?} level {l}");
                    checked += got.len() / 2;
                }
            }
            // A view of levels 1 to 4 of a 32x32 texture: its first level is the source.
            let t = texture(&device, pf, TextureType::Type2D, [32, 32, 1], 6, 1);
            let at = |l: usize| [(32 >> l).max(1), (32 >> l).max(1), 1];
            let first = seeded(16 * 16 * ch, &mut seed);
            write(&t, 0, 1, at(1), ch, &first);
            let past = vec![0x1234; ch];
            write(&t, 0, 5, at(5), ch, &past);
            let view = t
                .new_texture_view_with_levels_slices(pf, TextureType::Type2D, 1..5, 0..1)
                .unwrap();
            generate(&queue, &view);
            assert_eq!(
                level(&t, 0, 5, at(5), ch),
                past,
                "{pf:?}: the level past the view"
            );
            assert_eq!(
                level(&t, 0, 1, at(1), ch),
                first,
                "{pf:?}: the view's first level is only read"
            );
            let chain = chain_of(pf, TextureType::Type2D, at(1), 4, 1, true, rule);
            let want = chain.expected_levels(&bytes(&first)).expect("the levels");
            for l in 2..5 {
                let got = bytes(&level(&t, 0, l, at(l), ch));
                assert_eq!(got, want[l - 1], "{pf:?} view {l}");
                checked += got.len() / 2;
            }
        }
        eprintln!(
            "exact mipmaps on {}: {checked} components of 6 formats, every one the chain's",
            device.name()
        );
        let announced = assert_once(&messages.take_notices(), &[]);
        assert!(
            !announced.contains_key(&Degradation::MipmapsAsAppleM4)
                && !announced.contains_key(&Degradation::MipmapsWithinSpread),
            "the M3 Max's passes, no blit chain: {announced:?}"
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// The boxes an Apple M4 measured (probe 10), each the first level of a texture of the box: exact
/// halves rounded up in unorm and in snorm (towards +inf), of 2, 4 and 8 texels, and snorm boxes
/// holding -32768, which reads as -32767.
#[test]
fn the_boxes_metal_measured_mipmap_as_it_measured_them() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let (unorm, snorm) = (PixelFormat::R16Unorm, PixelFormat::R16Snorm);
        let boxes: [(PixelFormat, [usize; 3], &[u16], u16); 9] = [
            (unorm, [2, 2, 1], &[0x4e79, 0x0000, 0x8002, 0x8003], 0x53a0),
            (unorm, [1, 2, 1], &[0x7ffe, 0x7fff], 0x7fff),
            (
                unorm,
                [2, 2, 2],
                &[
                    0xb8e8, 0xb8e6, 0x8002, 0x8003, 0xc1fc, 0xc1fa, 0x987b, 0x3498,
                ],
                0x985c,
            ),
            (snorm, [2, 2, 1], &[0x4e79, 0x0000, 0x8002, 0x8003], 0xd3a0),
            (snorm, [2, 1, 1], &[0xfaa6, 0xf73f], 0xf8f3),
            (
                snorm,
                [2, 2, 2],
                &[
                    0xb8e8, 0xb8e6, 0x8002, 0x8003, 0xc1fc, 0xc1fa, 0x987b, 0x3498,
                ],
                0xb85c,
            ),
            (snorm, [2, 2, 1], &[0x8000, 0x8002, 0x7699, 0x769a], 0xfb4e),
            (snorm, [2, 1, 1], &[0x8000, 0x8002], 0x8002),
            (
                snorm,
                [2, 2, 2],
                &[
                    0x7ffe, 0x8000, 0x7b53, 0xffff, 0xd48a, 0xc7c7, 0x7ffd, 0x7ffd,
                ],
                0x22f4,
            ),
        ];
        for (pf, extent, first, measured) in boxes {
            let ty = if extent[2] > 1 {
                TextureType::Type3D
            } else {
                TextureType::Type2D
            };
            let t = texture(&device, pf, ty, extent, 2, 1);
            write(&t, 0, 0, extent, 1, first);
            generate(&queue, &t);
            assert_eq!(
                level(&t, 0, 1, [1, 1, 1], 1),
                [measured],
                "{pf:?} {first:04x?}"
            );
        }
        // One computed level each: the M3 Max's answer there is the M4's.
        let announced = assert_once(&messages.take_notices(), &[]);
        assert!(
            !announced.contains_key(&Degradation::MipmapsAsAppleM4),
            "{announced:?}"
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// At macOS 26.7 (`MetalLevel::MacOS26_7`, the M3 Max's passes), `mipmaps-as-apple-m4` is
/// announced exactly where an Apple M3 Max's mipmaps may differ from the chain's: not for a 3D
/// chain, a chain of one computed level (a 2x2 texture, a 32x32 one of
/// two levels, a view of two levels of a 32x32 texture), a 2D chain or array whose first level is
/// at most 65 texels wide and tall, nor a 2D strip past 65 texels thinner than 16 texels; once,
/// naming the texture, for a compressed 2D chain of two computed levels whose first pass starts
/// from a level 256 times wider than tall (past `PASS_ASPECT`); and for an `A1BGR5Unorm` cube of
/// three past 65 texels (a few of the format's measured values unexplained) on a device of its own.
#[test]
fn the_m3_note_is_announced_exactly_where_an_m3_differs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = device_at(MetalLevel::MacOS26_7, &messages, &cache);
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let pf = PixelFormat::RG16Unorm;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut chain = |ty: TextureType, extent: [usize; 3], levels: usize, slices: usize| {
            let t = texture(&device, pf, ty, extent, levels, slices);
            for slice in 0..slices {
                let first = seeded(extent.iter().product::<usize>() * 2, &mut seed);
                write(&t, slice, 0, extent, 2, &first);
            }
            t
        };
        let agreeing = [
            chain(TextureType::Type3D, [4, 4, 4], 3, 1),
            chain(TextureType::Type3D, [8, 8, 2], 4, 1),
            chain(TextureType::Type2D, [2, 2, 1], 2, 1),
            chain(TextureType::Type2D, [32, 32, 1], 2, 1),
            chain(TextureType::Type2D, [4, 4, 1], 3, 1),
            chain(TextureType::Type2DArray, [8, 8, 1], 4, 3),
            chain(TextureType::Type2D, [64, 64, 1], 7, 1),
            chain(TextureType::Type2D, [128, 4, 1], 3, 1),
            chain(TextureType::Type2D, [4, 128, 1], 4, 1),
        ];
        for t in &agreeing {
            generate(&queue, t);
        }
        let whole = chain(TextureType::Type2D, [32, 32, 1], 6, 1);
        let view = whole
            .new_texture_view_with_levels_slices(pf, TextureType::Type2D, 1..3, 0..1)
            .unwrap();
        generate(&queue, &view);
        let announced = assert_once(&messages.take_notices(), &[]);
        assert!(
            !announced.contains_key(&Degradation::MipmapsAsAppleM4),
            "where the M3 Max agrees: {announced:?}"
        );
        generate(&queue, &chain(TextureType::Type2D, [4096, 16, 1], 3, 1));
        let notices = messages.take_notices();
        assert_once(&notices, &M3_MIPMAPS);
        let notice = notices
            .iter()
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::MipmapsAsAppleM4)
            .unwrap();
        assert!(
            notice.object.contains("RG16Unorm") && notice.object.contains("3 levels"),
            "{notice:?}"
        );
    }
    {
        let device = device_at(MetalLevel::MacOS26_7, &messages, &cache);
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let t = texture(
            &device,
            PixelFormat::A1BGR5Unorm,
            TextureType::Cube,
            [128, 128, 1],
            3,
            6,
        );
        let mut seed = 0x0123_4567_89ab_cdefu64;
        for slice in 0..6 {
            write(
                &t,
                slice,
                0,
                [128, 128, 1],
                1,
                &seeded(128 * 128, &mut seed),
            );
        }
        generate(&queue, &t);
        assert_once(&messages.take_notices(), &M3_MIPMAPS);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Every format the filter computes, from random bytes (every bit pattern: NaNs, infinities,
/// subnormals, out-of-range codes), in a 2D texture, one texel high, a square of a power of two
/// (whose even levels read the values the level above kept), one past the M3 Max's tail's
/// extent, a 2D array, a cube and a 3D texture, halving and not: every level the device wrote is
/// the chain on the CPU from the first level as the device read it back, and the blit chain's
/// near-equivalent is never announced. The kinds the chains' own facts force are announced once
/// each: `mipmaps-as-apple-m4` where a chain's M3 Max passes are not known
/// (`MipmapFilterChain::unlike_an_m3`), `mipmap-tie-rounding` where a level may take a rounding
/// the chain does not fit (`MipmapFilterChain::near_ties`: the 8-bit sRGB formats among them).
#[test]
fn every_format_the_filter_computes_is_its_filter_level_by_level() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let rule = device.properties().metal_level.mipmap_rule();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let shapes: [(TextureType, [usize; 3], usize, usize); 7] = [
            (TextureType::Type2D, [7, 5, 1], 1, 3),
            (TextureType::Type2D, [17, 1, 1], 1, 5),
            (TextureType::Type2D, [8, 8, 1], 1, 4),
            (TextureType::Type2D, [66, 2, 1], 1, 3),
            (TextureType::Type2DArray, [5, 4, 1], 3, 3),
            (TextureType::Cube, [4, 4, 1], 6, 3),
            (TextureType::Type3D, [5, 4, 3], 1, 3),
        ];
        let mut seed = 0x51f1_5eed_cafe_f00du64;
        let (mut checked, mut formats) = (0usize, 0usize);
        let (mut unlike_an_m3, mut near_ties) = (false, false);
        for &pf in PixelFormat::ALL {
            let plan = |extent: [usize; 3], layers: usize, levels: usize, volume: bool| {
                MipmapFilterChain::plan(
                    pf,
                    extent.map(|a| a as u32),
                    layers as u32,
                    levels as u32,
                    volume,
                    compressed(extent.map(|a| a as u32)),
                    rule,
                    64,
                    u32::MAX,
                    [65535, 65535],
                    None,
                )
            };
            if plan([4, 4, 1], 1, 2, false).is_none() {
                continue;
            }
            formats += 1;
            for (ty, extent, slices, levels) in shapes {
                let volume = ty == TextureType::Type3D;
                let chain = plan(extent, slices, levels, volume).expect("a chain");
                unlike_an_m3 |= chain.unlike_an_m3;
                near_ties |= chain.near_ties;
                let at = |l: usize| extent.map(|a| (a >> l).max(1));
                let texels = |l: usize| at(l).iter().product::<usize>();
                let texel = chain.levels[0].bytes as usize / (texels(0) * slices);
                let t = texture(&device, pf, ty, extent, levels, slices);
                for slice in 0..slices {
                    let bytes: Vec<u8> = seeded(texels(0) * texel / 2 + 1, &mut seed)
                        .iter()
                        .flat_map(|c| c.to_le_bytes())
                        .take(texels(0) * texel)
                        .collect();
                    put(&t, slice, 0, at(0), texel, &bytes);
                }
                generate(&queue, &t);
                let read = |l: usize| -> Vec<u8> {
                    (0..slices)
                        .flat_map(|slice| get(&t, slice, l, at(l), texel))
                        .collect()
                };
                let want = chain.expected_levels(&read(0)).expect("the levels");
                for (l, want) in want.iter().enumerate().skip(1) {
                    let got = read(l);
                    assert_eq!(&got, want, "{pf:?} {ty:?} {extent:?} level {l}");
                    checked += got.len();
                }
            }
        }
        eprintln!(
            "the mipmap filter on {}: {formats} formats, {checked} bytes, every one the filter's",
            device.name()
        );
        assert_eq!(formats, 35, "the formats the filter computes");
        let kinds: Vec<Degradation> = [
            (Degradation::MipmapsAsAppleM4, unlike_an_m3),
            (Degradation::MipmapTieRounding, near_ties),
        ]
        .into_iter()
        .filter_map(|(kind, holds)| holds.then_some(kind))
        .collect();
        let announced = assert_once(&messages.take_notices(), &kinds);
        assert!(
            !announced.contains_key(&Degradation::MipmapsWithinSpread),
            "no blit chain: {announced:?}"
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// At macOS 26.7 (`MetalLevel::MacOS26_7`), the Apple M3 Max's passes past its tail's extent, from
/// random bytes: a 256x256 `BGRA8Unorm` 2D
/// texture of nine levels, a 640x480 `RGBA8Unorm` one of ten, a three-slice 200x150 `R8Unorm` array
/// of eight, a 128x128 `RG32Float` cube of eight, a 2048x8 `BGRA8Unorm` strip of twelve, a 256x256
/// `RGBA16Unorm` texture of nine and a 300x200 `BGR10_XR` one of nine, and textures Metal does not
/// compress (`allowGPUOptimizedContents` off): a 256x256 `BGRA8Unorm` one of nine, whose run reads
/// kept values and codes in turn, and a 1024x16 `RGBA16Float` one of eleven, are each the chain on
/// the CPU from the first level as the device read it back (passes of the M3 Max's filter and the
/// values it kept, then its tail) and announce nothing, as does a 256x256 `RG11B10Float` one of
/// nine; a 4096x16 `BGRA8Unorm` texture of thirteen, compressed, whose first pass starts from a
/// level 256 times wider than tall (past `PASS_ASPECT`), is the chain too and announces
/// `mipmaps-as-apple-m4` once.
#[test]
fn mipmaps_past_65_texels_take_the_m3_max_s_passes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = device_at(MetalLevel::MacOS26_7, &messages, &cache);
        let rule = device.properties().metal_level.mipmap_rule();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let mut seed = 0x6d69_7073_7061_7373u64;
        let mut chain = |pf: PixelFormat, ty, extent: [usize; 3], levels, slices, allow| {
            let chain = chain_of(pf, ty, extent, levels, slices, allow, rule);
            let at = |l: usize| extent.map(|a| (a >> l).max(1));
            let texels = |l: usize| at(l).iter().product::<usize>();
            let texel = chain.levels[0].bytes as usize / (texels(0) * slices);
            let t = texture_allowing(&device, pf, ty, extent, levels, slices, allow);
            for slice in 0..slices {
                let bytes: Vec<u8> = seeded(texels(0) * texel / 2 + 1, &mut seed)
                    .iter()
                    .flat_map(|c| c.to_le_bytes())
                    .take(texels(0) * texel)
                    .collect();
                put(&t, slice, 0, at(0), texel, &bytes);
            }
            generate(&queue, &t);
            let read = |l: usize| -> Vec<u8> {
                (0..slices)
                    .flat_map(|slice| get(&t, slice, l, at(l), texel))
                    .collect()
            };
            let want = chain.expected_levels(&read(0)).expect("the levels");
            for (l, want) in want.iter().enumerate().skip(1) {
                assert_eq!(&read(l), want, "{pf:?} {ty:?} {extent:?} level {l}");
            }
            chain.unlike_an_m3
        };
        use PixelFormat as P;
        use TextureType as T;
        let known = [
            chain(P::BGRA8Unorm, T::Type2D, [256, 256, 1], 9, 1, true),
            chain(P::RGBA8Unorm, T::Type2D, [640, 480, 1], 10, 1, true),
            chain(P::R8Unorm, T::Type2DArray, [200, 150, 1], 8, 3, true),
            chain(P::RG32Float, T::Cube, [128, 128, 1], 8, 6, true),
            chain(P::BGRA8Unorm, T::Type2D, [2048, 8, 1], 12, 1, true),
            chain(P::RGBA16Unorm, T::Type2D, [256, 256, 1], 9, 1, true),
            chain(P::BGR10XR, T::Type2D, [300, 200, 1], 9, 1, true),
            chain(P::BGRA8Unorm, T::Type2D, [256, 256, 1], 9, 1, false),
            chain(P::RGBA16Float, T::Type2D, [1024, 16, 1], 11, 1, false),
            chain(P::RG11B10Float, T::Type2D, [256, 256, 1], 9, 1, true),
        ];
        assert_eq!(known, [false; 10], "the M3 Max's passes");
        let announced = assert_once(&messages.take_notices(), &[]);
        assert!(
            !announced.contains_key(&Degradation::MipmapsAsAppleM4),
            "where the M3 Max's passes are known: {announced:?}"
        );
        let unknown = chain(P::BGRA8Unorm, T::Type2D, [4096, 16, 1], 13, 1, true);
        assert!(unknown, "a compressed pass past PASS_ASPECT");
        assert_once(&messages.take_notices(), &M3_MIPMAPS);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// The kind a chain takes where a level may take a rounding the chain does not fit, every venue,
/// in `mipmaps_near_ties_are_announced_where_the_rounding_is_not_fit`.
pub(super) const TIE_MIPMAPS: [Degradation; 1] = [Degradation::MipmapTieRounding];

/// At macOS 26.7 (`MetalLevel::MacOS26_7`), `mipmap-tie-rounding` is announced exactly where a
/// level may take a rounding the chain does not fit: not for `RGBA8Unorm` or `RGBA32Float`, nor for `RGBA16Float` chains with no level read
/// from kept values (one computed level, a 7x5 texture), nor for a 7x5 `RGB10A2Unorm` texture of
/// two levels (its 10-bit grid exact); once, naming the texture, for a `BGRA8Unorm_sRGB` texture of
/// two levels; and on a device of its own for a 16x16 `RGBA16Float` one of three (its second level
/// reads the first's kept values).
#[test]
fn mipmaps_near_ties_are_announced_where_the_rounding_is_not_fit() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let mut seed = 0x7469_6573_7469_6573u64;
    let mut run = |cases: &[(PixelFormat, [usize; 3], usize)], named: Option<&str>| {
        let device = device_at(MetalLevel::MacOS26_7, &messages, &cache);
        let rule = device.properties().metal_level.mipmap_rule();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        for &(pf, extent, levels) in cases {
            let chain = chain_of(pf, TextureType::Type2D, extent, levels, 1, true, rule);
            let texels = extent[0] * extent[1];
            let texel = chain.levels[0].bytes as usize / texels;
            let t = texture(&device, pf, TextureType::Type2D, extent, levels, 1);
            let bytes: Vec<u8> = seeded(texels * texel / 2 + 1, &mut seed)
                .iter()
                .flat_map(|c| c.to_le_bytes())
                .take(texels * texel)
                .collect();
            put(&t, 0, 0, extent, texel, &bytes);
            generate(&queue, &t);
            assert_eq!(
                chain.near_ties,
                named.is_some(),
                "{pf:?} {extent:?} x{levels}"
            );
        }
        let notices = messages.take_notices();
        let Some(name) = named else {
            let announced = assert_once(&notices, &[]);
            assert!(
                !announced.contains_key(&Degradation::MipmapTieRounding),
                "where the chain fits every rounding: {announced:?}"
            );
            return;
        };
        assert_once(&notices, &TIE_MIPMAPS);
        let notice = notices
            .iter()
            .filter_map(|message| message.degradation.as_ref())
            .find(|notice| notice.degradation == Degradation::MipmapTieRounding)
            .unwrap();
        assert!(notice.object.contains(name), "{notice:?}");
    };
    run(
        &[
            (PixelFormat::RGBA8Unorm, [8, 8, 1], 4),
            (PixelFormat::RGBA32Float, [16, 16, 1], 5),
            (PixelFormat::RGBA16Float, [8, 8, 1], 2),
            (PixelFormat::RGBA16Float, [7, 5, 1], 3),
            (PixelFormat::RGB10A2Unorm, [7, 5, 1], 2),
        ],
        None,
    );
    run(
        &[(PixelFormat::BGRA8UnormSRGB, [8, 8, 1], 2)],
        Some("BGRA8Unorm_sRGB"),
    );
    run(
        &[(PixelFormat::RGBA16Float, [16, 16, 1], 3)],
        Some("RGBA16Float"),
    );
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// At macOS 27.0.1 (`MetalLevel::MacOS27_0_1`), where the Apple M3 Max makes every level by its
/// filter (`MipmapRule::Filter`), from random bytes: a 256x256 `BGRA8Unorm` 2D texture of nine
/// levels, a 4096x16 one of thirteen (compressed, past the old passes' `PASS_ASPECT`), a 640x480
/// `RGBA8Unorm` one of ten, a 16x16 `RGBA16Float` one of three, an `A1BGR5Unorm` cube of 128x128
/// and three levels, a 193x193 `BGR10_XR_sRGB` texture of eight, a 16x16x6 `RGBA16Unorm` 3D
/// texture of five, and 41x61x5 `RGBA8Unorm` and `RGBA16Float` 3D textures of six (their first
/// level's width and height odd) are each the chain on the CPU from the first level as the device
/// read it back and announce nothing; a 129x129 `RGBA8Unorm_sRGB` texture of eight is the chain
/// and announces `mipmap-tie-rounding` once.
#[test]
fn mipmaps_at_macos_27_0_1_are_the_m3_max_s_filter() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = device_at(MetalLevel::MacOS27_0_1, &messages, &cache);
        let rule = device.properties().metal_level.mipmap_rule();
        assert_eq!(rule, MipmapRule::Filter);
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let mut seed = 0x6d69_7073_3237_3031u64;
        let mut chain = |pf: PixelFormat, ty, extent: [usize; 3], levels, slices| {
            let chain = chain_of(pf, ty, extent, levels, slices, true, rule);
            let at = |l: usize| extent.map(|a| (a >> l).max(1));
            let texels = |l: usize| at(l).iter().product::<usize>();
            let texel = chain.levels[0].bytes as usize / (texels(0) * slices);
            let t = texture(&device, pf, ty, extent, levels, slices);
            for slice in 0..slices {
                let bytes: Vec<u8> = seeded(texels(0) * texel / 2 + 1, &mut seed)
                    .iter()
                    .flat_map(|c| c.to_le_bytes())
                    .take(texels(0) * texel)
                    .collect();
                put(&t, slice, 0, at(0), texel, &bytes);
            }
            generate(&queue, &t);
            let read = |l: usize| -> Vec<u8> {
                (0..slices)
                    .flat_map(|slice| get(&t, slice, l, at(l), texel))
                    .collect()
            };
            let want = chain.expected_levels(&read(0)).expect("the levels");
            for (l, want) in want.iter().enumerate().skip(1) {
                assert_eq!(&read(l), want, "{pf:?} {ty:?} {extent:?} level {l}");
            }
            (chain.unlike_an_m3, chain.near_ties)
        };
        use PixelFormat as P;
        use TextureType as T;
        let quiet = [
            chain(P::BGRA8Unorm, T::Type2D, [256, 256, 1], 9, 1),
            chain(P::BGRA8Unorm, T::Type2D, [4096, 16, 1], 13, 1),
            chain(P::RGBA8Unorm, T::Type2D, [640, 480, 1], 10, 1),
            chain(P::RGBA16Float, T::Type2D, [16, 16, 1], 3, 1),
            chain(P::A1BGR5Unorm, T::Cube, [128, 128, 1], 3, 6),
            chain(P::BGR10XRSRGB, T::Type2D, [193, 193, 1], 8, 1),
            chain(P::RGBA16Unorm, T::Type3D, [16, 16, 6], 5, 1),
            chain(P::RGBA8Unorm, T::Type3D, [41, 61, 5], 6, 1),
            chain(P::RGBA16Float, T::Type3D, [41, 61, 5], 6, 1),
        ];
        assert_eq!(quiet, [(false, false); 9], "the M3 Max's filter");
        assert_once(&messages.take_notices(), &[]);
        let ties = chain(P::RGBA8UnormSRGB, T::Type2D, [129, 129, 1], 8, 1);
        assert_eq!(ties, (false, true), "an 8-bit sRGB chain");
        assert_once(&messages.take_notices(), &TIE_MIPMAPS);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Plain Metal's integer mipmap of a level from the level above (`above`, of `above_extent`),
/// `texel` bytes a texel: texel `i` is the above's texel `floor((2i + 1) W / 2w)` on each axis,
/// `W` and `w` the two levels' sizes on it (an M4 under macOS 27.0.1).
fn picked(above: &[u8], above_extent: [usize; 3], extent: [usize; 3], texel: usize) -> Vec<u8> {
    let [aw, ah, ad] = above_extent;
    let [w, h, d] = extent;
    let pick = |i: usize, big: usize, small: usize| (2 * i + 1) * big / (2 * small);
    let mut out = Vec::with_capacity(w * h * d * texel);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let at = ((pick(z, ad, d) * ah + pick(y, ah, h)) * aw + pick(x, aw, w)) * texel;
                out.extend_from_slice(&above[at..at + texel]);
            }
        }
    }
    out
}

/// `generate_mipmaps` of the integer formats, which Metal's API validation stops, on the device
/// the cfg selects under the validation layer: plain Metal's texel picks exactly, never refused
/// and never announced. Our probe's `R8Uint` textures, each texel the low byte of its
/// index, give its levels; 2D textures one texel high or wide, a 2D array of odd height, a cube,
/// 3D textures and a view of a 2D array's later levels and middle slices, of 1 to 16 bytes a
/// texel, give the rule applied to the level above as the device read it back.
#[test]
fn integer_mipmaps_pick_the_texels_plain_metal_picks() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let (t2, t3) = (TextureType::Type2D, TextureType::Type3D);
        // The probe's: its levels below the first.
        type Probed<'a> = (TextureType, [usize; 3], &'a [&'a [u8]]);
        let probed: [Probed<'_>; 6] = [
            (
                t2,
                [6, 6, 1],
                &[
                    &[0x07, 0x09, 0x0b, 0x13, 0x15, 0x17, 0x1f, 0x21, 0x23],
                    &[0x15],
                ],
            ),
            (
                t2,
                [7, 5, 1],
                &[&[0x08, 0x0a, 0x0c, 0x16, 0x18, 0x1a], &[0x18]],
            ),
            (
                t2,
                [12, 10, 1],
                &[
                    &[
                        0x0d, 0x0f, 0x11, 0x13, 0x15, 0x17, 0x25, 0x27, 0x29, 0x2b, 0x2d, 0x2f,
                        0x3d, 0x3f, 0x41, 0x43, 0x45, 0x47, 0x55, 0x57, 0x59, 0x5b, 0x5d, 0x5f,
                        0x6d, 0x6f, 0x71, 0x73, 0x75, 0x77,
                    ],
                    &[0x27, 0x2b, 0x2f, 0x57, 0x5b, 0x5f],
                    &[0x5b],
                ],
            ),
            (
                t2,
                [15, 9, 1],
                &[
                    &[
                        0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x2e, 0x30, 0x32, 0x34, 0x36,
                        0x38, 0x3a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x6a, 0x6c, 0x6e,
                        0x70, 0x72, 0x74, 0x76,
                    ],
                    &[0x30, 0x34, 0x38, 0x6c, 0x70, 0x74],
                    &[0x70],
                ],
            ),
            (
                t3,
                [4, 4, 4],
                &[&[0x15, 0x17, 0x1d, 0x1f, 0x35, 0x37, 0x3d, 0x3f], &[0x3f]],
            ),
            (
                t3,
                [6, 4, 6],
                &[
                    &[
                        0x1f, 0x21, 0x23, 0x2b, 0x2d, 0x2f, 0x4f, 0x51, 0x53, 0x5b, 0x5d, 0x5f,
                        0x7f, 0x81, 0x83, 0x8b, 0x8d, 0x8f,
                    ],
                    &[0x5d],
                ],
            ),
        ];
        let mut checked = 0usize;
        for (ty, extent, levels) in probed {
            let full = 1 + extent.iter().max().unwrap().ilog2() as usize;
            assert_eq!(levels.len() + 1, full, "{extent:?}");
            let t = texture(&device, PixelFormat::R8Uint, ty, extent, full, 1);
            let [w, h, d] = extent;
            let first: Vec<u8> = (0..w * h * d).map(|i| i as u8).collect();
            put(&t, 0, 0, extent, 1, &first);
            generate(&queue, &t);
            let at = |l: usize| extent.map(|a| (a >> l).max(1));
            let mut above = first;
            for (l, want) in (1..).zip(levels) {
                let got = get(&t, 0, l, at(l), 1);
                assert_eq!(&got, want, "{extent:?} level {l}: the probe's");
                assert_eq!(
                    got,
                    picked(&above, at(l - 1), at(l), 1),
                    "{extent:?} level {l}"
                );
                checked += got.len();
                above = got;
            }
        }
        // The other shapes, of the formats of each texel size.
        let shapes: [(PixelFormat, usize, TextureType, [usize; 3], usize); 7] = [
            (PixelFormat::RG8Sint, 2, t2, [33, 1, 1], 1),
            (PixelFormat::RG32Uint, 8, t2, [1, 9, 1], 1),
            (
                PixelFormat::RG16Uint,
                4,
                TextureType::Type2DArray,
                [9, 7, 1],
                3,
            ),
            (PixelFormat::R32Sint, 4, TextureType::Cube, [5, 5, 1], 6),
            (PixelFormat::RGBA8Uint, 4, t3, [5, 3, 7], 1),
            (PixelFormat::RGBA32Sint, 16, t2, [5, 5, 1], 1),
            (PixelFormat::RGBA16Sint, 8, t3, [8, 2, 3], 1),
        ];
        let fill = |n: usize, slice: usize| -> Vec<u8> {
            (0..n)
                .map(|i| (i * 7 + slice * 61 + i / 251) as u8)
                .collect()
        };
        for (pf, texel, ty, extent, slices) in shapes {
            let full = 1 + extent.iter().max().unwrap().ilog2() as usize;
            let t = texture(&device, pf, ty, extent, full, slices);
            let [w, h, d] = extent;
            let at = |l: usize| extent.map(|a| (a >> l).max(1));
            for slice in 0..slices {
                put(&t, slice, 0, extent, texel, &fill(w * h * d * texel, slice));
            }
            generate(&queue, &t);
            for slice in 0..slices {
                let mut above = get(&t, slice, 0, extent, texel);
                for l in 1..full {
                    let got = get(&t, slice, l, at(l), texel);
                    let want = picked(&above, at(l - 1), at(l), texel);
                    assert_eq!(
                        got, want,
                        "{pf:?} {ty:?} {extent:?} slice {slice} level {l}"
                    );
                    checked += got.len();
                    above = got;
                }
            }
        }
        // A view of levels 1 to 3 and slices 1 and 2 of a 2D array of 4 levels and slices: its
        // first level is the source; level 0 and slices 0 and 3 stay.
        let (pf, texel, extent) = (PixelFormat::RGBA8Uint, 4, [11, 7, 1]);
        let t = texture(&device, pf, TextureType::Type2DArray, extent, 4, 4);
        let at = |l: usize| extent.map(|a| (a >> l).max(1));
        let mut before = Vec::new();
        for slice in 0..4 {
            for l in 0..4 {
                let [w, h, d] = at(l);
                let bytes = fill(w * h * d * texel, slice * 4 + l);
                put(&t, slice, l, at(l), texel, &bytes);
                before.push(bytes);
            }
        }
        let view = t
            .new_texture_view_with_levels_slices(pf, TextureType::Type2DArray, 1..4, 1..3)
            .unwrap();
        generate(&queue, &view);
        for slice in 0..4 {
            let mut above = get(&t, slice, 1, at(1), texel);
            for l in 0..4 {
                let got = get(&t, slice, l, at(l), texel);
                if (1..3).contains(&slice) && l >= 2 {
                    assert_eq!(
                        got,
                        picked(&above, at(l - 1), at(l), texel),
                        "view {slice} {l}"
                    );
                    checked += got.len();
                    above = got;
                } else {
                    assert_eq!(got, before[slice * 4 + l], "past the view: {slice} {l}");
                }
            }
        }
        eprintln!(
            "integer mipmaps on {}: {checked} bytes, every one plain Metal's pick",
            device.name()
        );
        let announced = assert_once(&messages.take_notices(), &[]);
        assert!(announced.is_empty(), "{announced:?}");
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// The first level read in place recovers every code: for every format the filter computes, a 2D
/// array of two 512x256 layers whose 2x2 boxes each hold one probe texel, the 65,536 values of a
/// 16-bit pattern repeated over the texel's bytes (every code of every field: each 8-, 10-, 16- or
/// 32-bit field, every 4-, 5-, 1- and 2-bit one, every half's bits, the extended-range codes),
/// mipmapped once. Where the device read the first level in place (the work count says so), each
/// texel of the second level is the filter of one probe's four equal taps, so it is the CPU's only
/// if the fetch read back that probe's codes, up to what the filter reads alike; where the level
/// was copied in, it checks the copy. The run prints, per format, which path ran and how many of
/// its distinct probe texels the filter's answers tell apart (all of them but the ones it reads
/// alike: an snorm field's two codes of -1.0, a half's NaNs), the measurement of
/// `DriverFact::SampledFetchRecoversCodes`.
#[test]
fn a_first_level_read_in_place_recovers_every_code() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let rule = device.properties().metal_level.mipmap_rule();
        messages.take_notices();
        let queue = device.new_command_queue().unwrap();
        let (w, h, layers) = (512usize, 256usize, 2usize);
        let (mut in_place, mut copied) = (Vec::new(), Vec::new());
        for &pf in PixelFormat::ALL {
            let Some(chain) = MipmapFilterChain::plan(
                pf,
                [w as u32, h as u32, 1],
                layers as u32,
                2,
                false,
                compressed([w as u32, h as u32, 1]),
                rule,
                64,
                u32::MAX,
                [65535, 65535],
                None,
            ) else {
                continue;
            };
            let texel = chain.levels[0].bytes as usize / (w * h * layers);
            let t = texture(&device, pf, TextureType::Type2DArray, [w, h, 1], 2, layers);
            for slice in 0..layers {
                let bytes: Vec<u8> = (0..h)
                    .flat_map(|y| {
                        (0..w).flat_map(move |x| {
                            let probe =
                                (slice * (w / 2) * (h / 2) + (y / 2) * (w / 2) + x / 2) as u16;
                            probe.to_le_bytes().into_iter().cycle().take(texel)
                        })
                    })
                    .collect();
                put(&t, slice, 0, [w, h, 1], texel, &bytes);
            }
            let before = device.work_counts().mipmap_first_levels_fetched;
            generate(&queue, &t);
            let fetched = device.work_counts().mipmap_first_levels_fetched - before;
            assert!(fetched <= 1, "{pf:?}: one chain, one first level");
            let read = |l: usize, extent: [usize; 3]| -> Vec<u8> {
                (0..layers)
                    .flat_map(|slice| get(&t, slice, l, extent, texel))
                    .collect()
            };
            let first = read(0, [w, h, 1]);
            let want = chain.expected_levels(&first).expect("the levels");
            assert_eq!(
                read(1, [w / 2, h / 2, 1]),
                want[1],
                "{pf:?}: the second level, its first {}",
                if fetched == 1 {
                    "read in place"
                } else {
                    "copied in"
                }
            );
            // How many distinct probe texels the filter's answers tell apart.
            let probes: std::collections::BTreeSet<&[u8]> = first.chunks_exact(texel).collect();
            let answers: std::collections::BTreeSet<&[u8]> = want[1].chunks_exact(texel).collect();
            let row = format!("{pf:?} {}/{}", answers.len(), probes.len());
            if fetched == 1 {
                in_place.push(row);
            } else {
                copied.push(row);
            }
        }
        eprintln!(
            "first levels on {}: read in place, every code recovered (answers told apart / \
             probes): {in_place:?}; copied in: {copied:?}",
            device.name()
        );
        assert_eq!(
            in_place.len() + copied.len(),
            35,
            "the formats the filter computes"
        );
        assert!(
            !in_place.is_empty(),
            "the extended-range formats' integer carriers are read in place on every device"
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}
