//! Texture copies: Metal's rules as probe 3 measured them on an Apple M3 Max and M4, and the Vulkan
//! regions and staging layouts that carry them.

use super::super::copy_plan::{
    block_extent, check_region, copy_unit, plan_host_copy, shape, BlitOption, CopyProblem,
    CopyRefusal, CopyUnit, HostLayout, TextureRegion,
};
use super::super::image_plan::{is_compressed, plan_image, ImagePlan};
use super::super::texture_descriptor::TextureDescriptor;
use super::super::view_plan::ViewRecord;
use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use metal2vulkan::device::{Device, DeviceDescription, FormatUsage};
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::metal::format::{direct_format, vk_format, FormatMapping};
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::ImageAspectFlags;

fn lavapipe() -> (DeviceDescription, Device) {
    let (_, desc, device) = committed()
        .into_iter()
        .find(|(n, ..)| n == "lavapipe")
        .unwrap();
    (desc, device)
}

/// A texture of `desc` on lavapipe: what it presents, its image, and its copy unit for `option`.
struct Texture {
    record: ViewRecord,
    image: ImagePlan,
}

fn texture(desc: TextureDescriptor) -> Texture {
    let (d, device) = lavapipe();
    let facts = facts(&d, &device);
    let layout = desc.check(APPLE).expect("a valid descriptor");
    let image = plan_image(&layout, desc.usage, false, &facts).expect("an image");
    Texture {
        record: ViewRecord::of_texture(&layout, desc.usage, desc.swizzle),
        image,
    }
}

fn tex(pf: PixelFormat, ty: TextureType, extent: [u64; 3]) -> Texture {
    texture(TextureDescriptor {
        texture_type: ty,
        pixel_format: pf,
        width: extent[0],
        height: extent[1],
        depth: extent[2],
        ..TextureDescriptor::default()
    })
}

fn unit(t: &Texture, option: BlitOption) -> Result<CopyUnit, CopyRefusal> {
    copy_unit(t.record.pixel_format, &t.image.mapping, option)
}

fn region(origin: [u32; 3], size: [u32; 3]) -> TextureRegion {
    TextureRegion {
        level: 0,
        slice: 0,
        origin,
        size,
    }
}

#[test]
fn block_extents_name_exactly_the_compressed_formats() {
    let mut compressed = 0;
    for &pf in PixelFormat::ALL {
        let Some(block) = direct_format(pf).and_then(|f| f.block()) else {
            continue;
        };
        if block.depth.is_some() || block.stencil.is_some() {
            continue;
        }
        let extent = block_extent(pf);
        assert_eq!(extent.is_some(), is_compressed(&block), "{pf:?}");
        if let Some([w, h]) = extent {
            compressed += 1;
            // Vulkan's registry counts a PVRTC block as one texel; every other block has its
            // extent's texels.
            assert!(block.texels == w * h || block.texels == 1, "{pf:?}");
        }
    }
    assert!(compressed > 50, "{compressed}");
}

#[test]
fn copy_units_are_metals_pixels_blocks_and_aspects() {
    let (_, device) = lavapipe();
    let p = device.profile();
    let mapping = |pf| vk_format(pf, &[FormatUsage::SampledImage].into(), p).unwrap();
    let u = |pf, option| copy_unit(pf, &mapping(pf), option);
    let color = ImageAspectFlags::COLOR;
    assert_eq!(
        u(PixelFormat::RGBA8Unorm, BlitOption::None).unwrap(),
        CopyUnit {
            bytes: 4,
            extent: [1, 1],
            aspect: color
        }
    );
    assert_eq!(
        u(PixelFormat::BC1RGBA, BlitOption::None).unwrap(),
        CopyUnit {
            bytes: 8,
            extent: [4, 4],
            aspect: color
        }
    );
    let d32s8 = PixelFormat::Depth32FloatStencil8;
    let depth = u(d32s8, BlitOption::DepthFromDepthStencil).unwrap();
    assert_eq!((depth.bytes, depth.aspect), (4, ImageAspectFlags::DEPTH));
    let stencil = u(d32s8, BlitOption::StencilFromDepthStencil).unwrap();
    assert_eq!(
        (stencil.bytes, stencil.aspect),
        (1, ImageAspectFlags::STENCIL)
    );
    // Plain Metal's aspects where its validation stops the option: a combined
    // format without one copies its depth (`bo_btt_d32s8_none`, `bo_ttb_d32s8_none`), and an option
    // naming an aspect the format lacks is ignored (`bo_*_rgba8_depth`, `bo_*_rgba8_stencil`,
    // `bo_*_d32_stencil`, `bo_*_s8_depth`).
    assert_eq!(u(d32s8, BlitOption::None).unwrap(), depth);
    for option in [
        BlitOption::DepthFromDepthStencil,
        BlitOption::StencilFromDepthStencil,
    ] {
        assert_eq!(
            u(PixelFormat::RGBA8Unorm, option).unwrap(),
            u(PixelFormat::RGBA8Unorm, BlitOption::None).unwrap(),
            "{option:?}"
        );
        assert_eq!(
            u(PixelFormat::BC1RGBA, option).unwrap(),
            u(PixelFormat::BC1RGBA, BlitOption::None).unwrap(),
            "{option:?}"
        );
        let d32 = u(PixelFormat::Depth32Float, option).unwrap();
        assert_eq!((d32.bytes, d32.aspect), (4, ImageAspectFlags::DEPTH));
        let s8 = u(PixelFormat::Stencil8, option).unwrap();
        assert_eq!((s8.bytes, s8.aspect), (1, ImageAspectFlags::STENCIL));
    }
    assert_eq!(
        u(PixelFormat::Depth32Float, BlitOption::None)
            .unwrap()
            .bytes,
        4
    );
    assert_eq!(
        u(PixelFormat::Depth16Unorm, BlitOption::None)
            .unwrap()
            .bytes,
        2
    );
    assert_eq!(u(PixelFormat::Stencil8, BlitOption::None).unwrap().bytes, 1);
    // The stencil view of a combined texture copies its stencil whatever the option (a probe).
    for option in [
        BlitOption::None,
        BlitOption::StencilFromDepthStencil,
        BlitOption::DepthFromDepthStencil,
    ] {
        let view = copy_unit(PixelFormat::X32Stencil8, &mapping(d32s8), option).unwrap();
        assert_eq!(
            view,
            CopyUnit {
                bytes: 1,
                extent: [1, 1],
                aspect: ImageAspectFlags::STENCIL
            },
            "{option:?}"
        );
    }
    // 32-bit float depth held in a combined format copies the same 4 bytes.
    let mut combined: FormatMapping = mapping(PixelFormat::Depth32Float);
    combined.format = metal2vulkan::vk::Format::D32_SFLOAT_S8_UINT;
    assert_eq!(
        copy_unit(PixelFormat::Depth32Float, &combined, BlitOption::None)
            .unwrap()
            .bytes,
        4
    );
    // 24-bit depth: Vulkan's buffer layout of it is not Metal's measured one.
    let mut d24: FormatMapping = mapping(PixelFormat::Depth32Float);
    d24.format = metal2vulkan::vk::Format::D24_UNORM_S8_UINT;
    d24.aspects = ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL;
    match copy_unit(
        PixelFormat::Depth24UnormStencil8,
        &d24,
        BlitOption::DepthFromDepthStencil,
    ) {
        Err(CopyRefusal::Unsupported(u)) => {
            assert_eq!(
                (u.feature, u.needs),
                (Feature::Format, Needs::NotImplemented)
            )
        }
        other => panic!("{other:?}"),
    }
}

/// A texture with a depth twin holds `Depth32Float` in an `R32_SFLOAT` image: Metal's 4 bytes a
/// texel in its colour aspect, with any option (plain Metal copies `Depth32Float`'s depth with
/// `StencilFromDepthStencil`); no other format held in colour as depth.
#[test]
fn depth_held_in_colour_copies_metals_four_bytes() {
    let (_, device) = lavapipe();
    let sampled = [FormatUsage::SampledImage].into();
    let colour: FormatMapping =
        vk_format(PixelFormat::R32Float, &sampled, device.profile()).unwrap();
    assert_eq!(colour.aspects, ImageAspectFlags::COLOR);
    for option in [
        BlitOption::None,
        BlitOption::DepthFromDepthStencil,
        BlitOption::StencilFromDepthStencil,
    ] {
        let unit = copy_unit(PixelFormat::Depth32Float, &colour, option).unwrap();
        assert_eq!(
            (unit.bytes, unit.extent, unit.aspect),
            (4, [1, 1], ImageAspectFlags::COLOR),
            "{option:?}"
        );
    }
    match copy_unit(PixelFormat::Depth16Unorm, &colour, BlitOption::None) {
        Err(CopyRefusal::Unsupported(u)) => assert_eq!(
            (u.feature, u.needs),
            (Feature::Format, Needs::NotImplemented)
        ),
        other => panic!("{other:?}"),
    }
}

#[test]
fn host_copies_follow_metals_measured_verdicts_and_pack_rows() {
    let rgba = tex(PixelFormat::RGBA8Unorm, TextureType::Type2D, [2, 2, 1]);
    let u = unit(&rgba, BlitOption::None).unwrap();
    let full = region([0, 0, 0], [2, 2, 1]);
    let plan = |t: &Texture, r, bpr, bpi, private, over| {
        plan_host_copy(&t.record, &r, &u, bpr, bpi, private, over).map(|p| p.map(|(_, h)| h))
    };
    // `gb.private` and `rp.private`.
    assert_eq!(
        plan(&rgba, full, 8, 0, true, false),
        Err(CopyProblem::CpuAccessPrivate)
    );
    // Plain Metal's `bytesPerImage` off its rules: one image ignores it
    // (`rp.2d.bpi3`, `gb.2d.bpi3`, `rp.2darray.bpi3`, `gb.2darray.bpi4`).
    for bpi in [0, 3, 4, 16, 18] {
        let host = plan(&rgba, full, 8, bpi, false, false).unwrap().unwrap();
        assert_eq!(host.rows().map(|(h, _)| h).collect::<Vec<_>>(), [0, 8]);
        assert_eq!(host.host_len(), 16, "{bpi}");
    }
    // More than one image are exactly `bytesPerImage` apart, off the pixel and below an image
    // too (`rp.3d.bpi0`, `bpi4`, `bpi12`, `bpi18`), ...
    let volume = tex(PixelFormat::RGBA8Unorm, TextureType::Type3D, [2, 2, 2]);
    let whole = region([0, 0, 0], [2, 2, 2]);
    for (bpi, len) in [(0, 16), (4, 20), (12, 28), (18, 34)] {
        let host = plan(&volume, whole, 8, bpi, false, false).unwrap().unwrap();
        let rows: Vec<usize> = host.rows().map(|(h, _)| h).collect();
        let bpi = bpi as usize;
        assert_eq!(rows, [0, 8, bpi, bpi + 8]);
        assert_eq!(host.host_len(), len, "{bpi}");
    }
    // ... and `getBytes` writes them in turn, a later image's bytes over an earlier one's
    // (`gb.3d.bpi12`: `00..0b`, then the second image's `10..1f` from 12).
    let host = plan(&volume, whole, 8, 12, false, false).unwrap().unwrap();
    let texels: Vec<u8> = (0..16).chain(0x10..0x20).collect();
    let mut bytes = vec![0xabu8; 40];
    host.unpack(&texels, &mut bytes);
    let want: Vec<u8> = (0..12).chain(0x10..0x20).chain([0xab; 12]).collect();
    assert_eq!(bytes, want);
    // One row takes `bytesPerRow` 0 (`gb.bpr0.1row`).
    let one_row = plan(&rgba, region([0, 1, 0], [2, 1, 1]), 0, 0, false, false)
        .unwrap()
        .unwrap();
    assert_eq!((one_row.host_len(), one_row.packed_len()), (8, 8));
    // Plain Metal's pitches off its rules: a pitch of a row or more exactly, off the pixel too; 0,
    // every row at the first byte; between 0 and a row, nothing copied, one row or more.
    let odd = plan(&rgba, full, 10, 0, false, false).unwrap().unwrap();
    assert_eq!(odd.rows().map(|(h, _)| h).collect::<Vec<_>>(), [0, 10]);
    let zero = plan(&rgba, full, 0, 0, false, false).unwrap().unwrap();
    assert_eq!(zero.rows().map(|(h, _)| h).collect::<Vec<_>>(), [0, 0]);
    let src: Vec<u8> = (0..8).collect();
    let mut packed = vec![0u8; 16];
    zero.pack(&src, &mut packed);
    assert_eq!(packed, [&src[..], &src[..]].concat());
    // `getBytes` writes each row there in turn: the last one stays.
    let rows: Vec<u8> = (0..16).collect();
    let mut host = vec![0xeeu8; 8];
    zero.unpack(&rows, &mut host);
    assert_eq!(host, rows[8..]);
    for bpr in [4, 6] {
        assert_eq!(plan(&rgba, full, bpr, 0, false, false), Ok(None), "{bpr}");
        let row = region([0, 0, 0], [2, 1, 1]);
        assert_eq!(plan(&rgba, row, bpr, 0, false, false), Ok(None), "{bpr}");
    }
    // Not measured: a pitch off the rules of a texture over a buffer, and a pitch of 0 for more
    // than one row and image.
    for bpr in [0, 4, 10] {
        assert_eq!(
            plan(&rgba, full, bpr, 0, false, true),
            Err(CopyProblem::BytesPerRow),
            "{bpr}"
        );
    }
    assert_eq!(
        plan(&volume, whole, 0, 16, false, false),
        Err(CopyProblem::BytesPerRow)
    );
    // `rp.bc1`: rows of blocks.
    let bc1 = tex(PixelFormat::BC1RGBA, TextureType::Type2D, [8, 8, 1]);
    let bu = unit(&bc1, BlitOption::None).unwrap();
    let (image, host) = plan_host_copy(
        &bc1.record,
        &region([0, 0, 0], [8, 8, 1]),
        &bu,
        16,
        0,
        false,
        false,
    )
    .unwrap()
    .unwrap();
    assert_eq!((host.row_bytes, host.rows, host.packed_len()), (16, 2, 32));
    assert_eq!(image.size, [8, 8, 1]);
    // Padded rows and images pack and unpack exactly, leaving the padding alone.
    let host = plan(&volume, whole, 12, 32, false, false).unwrap().unwrap();
    assert_eq!((host.host_len(), host.packed_len()), (52, 32));
    let src: Vec<u8> = (0..52).collect();
    let mut packed = vec![0u8; 32];
    host.pack(&src, &mut packed);
    let want: Vec<u8> = [0u8, 12, 32, 44].iter().flat_map(|&o| o..o + 8).collect();
    assert_eq!(packed, want);
    let mut back = vec![0xeeu8; 52];
    host.unpack(&packed, &mut back);
    for (i, b) in back.iter().enumerate() {
        let in_row = [0usize, 12, 32, 44]
            .iter()
            .any(|&o| (o..o + 8).contains(&i));
        assert_eq!(*b, if in_row { i as u8 } else { 0xee }, "{i}");
    }
}

/// A compressed region off its blocks copies the blocks it starts in, as plain Metal does (an M4
/// under macOS 27.0.1): the origin rounded down to its block and the size
/// rounded up to whole blocks, each on its own, ending at the level's edge.
#[test]
fn compressed_regions_off_their_blocks_copy_the_blocks_they_start_in() {
    let bc1 = tex(PixelFormat::BC1RGBA, TextureType::Type2D, [16, 16, 1]);
    let u = unit(&bc1, BlitOption::None).unwrap();
    let image = |origin, size| {
        let r = check_region(&bc1.record, &region(origin, size), &u).unwrap();
        (r.origin, r.size)
    };
    // `ba_*_o1`: one block from 0; `ba_btt_o2y`: likewise down the column.
    assert_eq!(image([1, 0, 0], [3, 4, 1]), ([0, 0, 0], [4, 4, 1]));
    assert_eq!(image([0, 2, 0], [4, 2, 1]), ([0, 0, 0], [4, 4, 1]));
    // `ba_*_o5_s4`: the origin and the size each on its own (one block, not two).
    assert_eq!(image([5, 5, 0], [4, 4, 1]), ([4, 4, 0], [4, 4, 1]));
    // `ba_*_s5x4`, `ba_*_s3x3`, `ba_btt_s4x6_at4`: the size rounded up.
    assert_eq!(image([0, 0, 0], [5, 4, 1]), ([0, 0, 0], [8, 4, 1]));
    assert_eq!(image([0, 0, 0], [3, 3, 1]), ([0, 0, 0], [4, 4, 1]));
    assert_eq!(image([4, 4, 0], [4, 6, 1]), ([4, 4, 0], [4, 8, 1]));
    // The region's buffer layout is that of its blocks.
    assert_eq!(shape([5, 4, 1], &u), shape([8, 4, 1], &u));
    // A level whose edge cuts a block: the blocks end there.
    let edge = tex(PixelFormat::BC1RGBA, TextureType::Type2D, [6, 6, 1]);
    let r = check_region(&edge.record, &region([5, 1, 0], [1, 2, 1]), &u).unwrap();
    assert_eq!((r.origin, r.size), ([4, 0, 0], [2, 4, 1]));
    // An uncompressed format is left as it is; a region past the level stays refused.
    let rgba = tex(PixelFormat::RGBA8Unorm, TextureType::Type2D, [4, 4, 1]);
    let ru = unit(&rgba, BlitOption::None).unwrap();
    let r = check_region(&rgba.record, &region([1, 1, 0], [3, 2, 1]), &ru).unwrap();
    assert_eq!((r.origin, r.size), ([1, 1, 0], [3, 2, 1]));
    assert_eq!(
        check_region(&bc1.record, &region([13, 0, 0], [4, 4, 1]), &u),
        Err(CopyProblem::Region)
    );
}

/// A compressed region is bounded by its level's blocks, as plain Metal bounds it (an Apple M3 Max
/// under macOS 26.7, a probe; its validation takes a whole block out of a texture, between
/// textures and of the CPU): a whole block of a level smaller than one (4 x 4 of level 9 of a
/// 2048 x 1024 `BC3_RGBA` texture, 4 x 2, and of levels 10 and 11) copies the block, ending at the
/// level's edge, and so does a region of it from past the level's texels. A region past the
/// level's last block stays refused, and an uncompressed format keeps its texels as the bound.
#[test]
fn compressed_regions_are_bounded_by_their_levels_blocks() {
    let bc3 = texture(TextureDescriptor {
        texture_type: TextureType::Type2D,
        pixel_format: PixelFormat::BC3RGBA,
        width: 2048,
        height: 1024,
        mipmap_level_count: 12,
        ..TextureDescriptor::default()
    });
    let u = unit(&bc3, BlitOption::None).unwrap();
    let at = |level, origin, size| {
        let region = TextureRegion {
            level,
            slice: 0,
            origin,
            size,
        };
        check_region(&bc3.record, &region, &u).map(|r| (r.level, r.origin, r.size))
    };
    // `t2b_l9_4x4`, `t2b_l10_4x4`, `t2b_l11_4x4`: the one block, to the level's edge.
    assert_eq!(at(9, [0, 0, 0], [4, 4, 1]), Ok((9, [0, 0, 0], [4, 2, 1])));
    assert_eq!(at(10, [0, 0, 0], [4, 4, 1]), Ok((10, [0, 0, 0], [2, 1, 1])));
    assert_eq!(at(11, [0, 0, 0], [4, 4, 1]), Ok((11, [0, 0, 0], [1, 1, 1])));
    // `t2b_l9_o2_4x2`: from past the level's texels, the block it starts in.
    assert_eq!(at(9, [0, 2, 0], [4, 2, 1]), Ok((9, [0, 0, 0], [4, 2, 1])));
    // The region's buffer layout is the block's.
    assert_eq!(shape([4, 4, 1], &u), shape([4, 2, 1], &u));
    // Past the level's last block.
    for (origin, size) in [
        ([0, 0, 0], [4, 5, 1]),
        ([4, 0, 0], [4, 4, 1]),
        ([0, 0, 0], [8, 4, 1]),
    ] {
        assert_eq!(
            at(9, origin, size),
            Err(CopyProblem::Region),
            "{origin:?} {size:?}"
        );
    }
    // An empty region at the blocks' end, past the level's edge, is empty.
    assert_eq!(at(9, [0, 4, 0], [4, 0, 1]), Ok((9, [0, 4, 0], [4, 0, 1])));
    // An uncompressed format keeps its texels as the bound.
    let rgba = tex(PixelFormat::RGBA8Unorm, TextureType::Type2D, [4, 2, 1]);
    let ru = unit(&rgba, BlitOption::None).unwrap();
    assert_eq!(
        check_region(&rgba.record, &region([0, 0, 0], [4, 4, 1]), &ru),
        Err(CopyProblem::Region)
    );
}

/// A 4:2:2 texture's copy unit is its 2-byte texel, one texel of its `R8G8_UNORM` carrier: plain
/// Metal copies it at any x and width, its validation's 2 x 1 blocks aside.
#[test]
fn a_four_two_two_texture_copies_its_two_byte_texels() {
    for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
        let t = tex(pf, TextureType::Type2D, [18, 6, 1]);
        let u = unit(&t, BlitOption::None).unwrap();
        assert_eq!(
            u,
            CopyUnit {
                bytes: 2,
                extent: [1, 1],
                aspect: ImageAspectFlags::COLOR
            },
            "{pf:?}"
        );
        let r = check_region(&t.record, &region([1, 2, 0], [3, 1, 1]), &u).unwrap();
        assert_eq!((r.origin, r.size), ([1, 2, 0], [3, 1, 1]), "{pf:?}");
    }
}

/// A host upload's depth words hold a negative value or a NaN (`signed-depth-test`'s check) where a
/// row the layout reads holds a negative normal or infinity, or a NaN of either sign; a zero of
/// either sign, a denormal of either sign (`depth-denormals`'), a positive value, and a word
/// between rows hold none.
#[test]
fn a_host_upload_holds_a_negative_or_nan_depth_only_in_its_rows() {
    // Two rows of two words, 12 bytes apart: the third word of each row lies between them.
    let layout = HostLayout {
        row_bytes: 8,
        rows: 2,
        images: 1,
        bytes_per_row: 12,
        bytes_per_image: 24,
    };
    let bytes =
        |words: [u32; 6]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let quiet = [
        0x0000_0000,
        0x8000_0000,
        0x8000_0001,
        0x807f_ffff,
        0x3f00_0000,
        0x7f80_0000,
    ];
    for at in [0, 1, 3, 4] {
        for word in quiet {
            let mut words = [0x3e80_0000; 6];
            words[at] = word;
            assert!(
                !layout.holds_negative_or_nan(&bytes(words)),
                "{word:08x} at {at}"
            );
        }
        for word in [
            0xbf00_0000,
            0xc000_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0000,
            0x7f80_0001,
        ] {
            let mut words = [0x3e80_0000; 6];
            words[at] = word;
            assert!(
                layout.holds_negative_or_nan(&bytes(words)),
                "{word:08x} at {at}"
            );
        }
    }
    // Between the rows.
    for at in [2, 5] {
        let mut words = [0x3e80_0000; 6];
        words[at] = 0xbf00_0000;
        assert!(
            !layout.holds_negative_or_nan(&bytes(words)),
            "between rows at {at}"
        );
    }
}
