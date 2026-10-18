//! Metal's private texture fills, `BlitCommandEncoder::fill_texture` and `fill_texture_color`, on
//! the device the cfg selects, under the validation layer, against Metal's bytes.
//!
//! Each case replays one of our probes' (`tfl.swift` and `tfl3.swift`; an Apple M4 under macOS
//! 27.0.1): a private 2D or 2D array texture of the case's extent, levels and slices, every byte of
//! every level, slice and aspect set to `0xEE` by copies from a buffer, one fill, and every level,
//! slice and aspect read back by copies into a buffer; for a two-plane extended-range format, a
//! one-level texture at byte 0 of a shared placement heap of Metal's size, every byte of the heap
//! set to `0xEE` through a buffer over it and the heap's changed bytes read back. Metal's answer
//! per format: the first 16 hex digits of the SHA-256 of its cases' lines, `<case> <answer>`, each
//! answer the first 16 hex digits of the SHA-256 of the case's read-back lines as the probe printed
//! them, `unpacked` where Metal's bytes do not follow the color (one constant of its process's for
//! every color; the runtime's fill there must write what the bytes form writes for one zero byte,
//! its one constant, and the device announce `unpacked-fill-color` once: changed), or the refusal
//! the runtime gives where it has no measured layout (`unmeasured`: a heap's two-plane texture of
//! more than one level or slice, which Metal fills). A format the device holds no texture of is
//! skipped, with an `XR-TEXTURE-FILLS-SKIPPED` line, and so is `Depth32Float_Stencil8` where its
//! uploads announce that the device may store depth outside `[0, 1]` clamped
//! (`restricted-depth-upload`, checked to fire exactly where the runtime plans a depth image that
//! may clamp); a differing format prints its cases on one `XR-TEXTURE-FILLS-DIFFERS` line. The
//! two-plane formats in private textures of their own memory replay `tfl4.swift` the same way, 5
//! and 8 texels wide.

use super::degrade_notices::{assert_made_once, forced, loses_depth_denormals, on_device};
use super::degrade_notices::{
    decodes_blocks_its_way, shared_in_system_memory, twins_blocks, twins_carried,
};
use super::degrade_notices::{twins_depth, UNPACKED_FILL};
use super::extended_range_two_planes::{get, put};
use super::texture_blits::depth_images;
use super::{buffer_with, read_buffer};
use metal2vulkan::metal::format::{is_depth_stencil, padded_only_over_memory, ExtendedRange};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    AppleOnly, BlitOption, ClearColor, CommandBufferStatus, CommandQueue, Degradation, Device,
    Error, HeapDescriptor, HeapType, Origin, Region, ResourceOptions, Size, TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Metal's answers: each format's raw value, its heap's bytes (a two-plane format's), and the
/// digest of its cases' lines.
const FORMATS: [(u32, u64, &str); 50] = [
    (10, 0, "5464fe66b33895e2"),
    (12, 0, "aa81065f10178e1f"),
    (13, 0, "ed9490278981257b"),
    (20, 0, "24fe976f77bb4506"),
    (23, 0, "3c66fbe4c17d400e"),
    (25, 0, "14accca524027699"),
    (30, 0, "b5ee494f91b9b3c6"),
    (40, 0, "be84d6a6cdce7ab4"),
    (41, 0, "9ec87b84ff4de073"),
    (42, 0, "6987da912b86071a"),
    (43, 0, "07cfccb40d827c06"),
    (53, 0, "4573cb785ba0b16a"),
    (55, 0, "f830ab1fa7dd3cc5"),
    (60, 0, "b3cf44f6777186d7"),
    (65, 0, "399c23bba36179e3"),
    (70, 0, "5e21f045d0204773"),
    (71, 0, "32cba3e03801ac38"),
    (80, 0, "8bcbb14846224039"),
    (81, 0, "fbb8049269874f64"),
    (90, 0, "1448888ff150253d"),
    (92, 0, "9067c0305af37729"),
    (93, 0, "f0aa4ed5a6cabe46"),
    (94, 0, "8f3866a50999be31"),
    (103, 0, "e78f560f4df1b5d0"),
    (105, 0, "d260d8c7f05ca8ff"),
    (110, 0, "2df2cb38c53448fb"),
    (115, 0, "27f9e93fc6475f47"),
    (123, 0, "87c65dbffa375059"),
    (125, 0, "a1a38a2bc59adc90"),
    (130, 0, "eee8a299143ef6c3"),
    (134, 0, "f96803ca3cf8b50a"),
    (152, 0, "f96803ca3cf8b50a"),
    (160, 0, "be7903fe2aeebda6"),
    (162, 0, "62f023563eaba5d6"),
    (170, 0, "eee8a299143ef6c3"),
    (178, 0, "f96803ca3cf8b50a"),
    (180, 0, "eee8a299143ef6c3"),
    (187, 0, "f96803ca3cf8b50a"),
    (198, 0, "a7cd635068daae8f"),
    (212, 0, "a7cd635068daae8f"),
    (236, 0, "a7cd635068daae8f"),
    (240, 0, "76db2674be6f822f"),
    (241, 0, "76db2674be6f822f"),
    (250, 0, "25b03997269e7e5f"),
    (252, 0, "91bf73db1b60d4f9"),
    (253, 0, "3e98e4db29723b7e"),
    (260, 0, "6bad089e9ea96a78"),
    (261, 0, "dbb9a8f56e387de0"),
    (550, 180224, "d9a9085f09b969dd"),
    (551, 180224, "d9a9085f09b969dd"),
];

/// A format's unit as the probes count it: its block's bytes, width and height (a texel's for an
/// uncompressed format; a 4:2:2 format's 2 x 1 block).
fn block(raw: u32) -> (u64, u64, u64) {
    match raw {
        130 | 140 | 170 | 180 => (8, 4, 4),
        134 | 152 | 178 => (16, 4, 4),
        160 => (8, 8, 4),
        162 => (8, 4, 4),
        187 => (16, 5, 4),
        198 => (16, 10, 10),
        212 => (16, 8, 8),
        236 => (16, 12, 12),
        240 | 241 => (4, 2, 1),
        253 | 261 | 10 | 12 | 13 => (1, 1, 1),
        260 | 550 | 551 => (4, 1, 1),
        20 | 23 | 25 | 30 | 40 | 41 | 42 | 43 | 250 => (2, 1, 1),
        103 | 105 | 110 | 115 => (8, 1, 1),
        123 | 125 => (16, 1, 1),
        _ => (4, 1, 1),
    }
}

/// One probe case: the texture and the fill.
struct Case {
    array: bool,
    extent: (u64, u64),
    levels: u64,
    slices: u64,
    level: u64,
    slice: u64,
    origin: (u64, u64),
    size: (u64, u64),
    /// The bytes form's pattern length, or the color form's color and pixel format.
    fill: Fill,
}

enum Fill {
    Bytes(usize),
    Color(ClearColor, Option<u32>),
    /// A pattern of one zero byte: what a color fill packed where fills ignore the color writes.
    Zeros,
}

/// The probes' case `name` of format `raw`; `wide`, at `tfl4.swift`'s extents 8 texels wide.
fn case(raw: u32, name: &str, wide: bool) -> Case {
    let (bb, bw, bh) = block(raw);
    let pvrtc = raw == 160 || raw == 162;
    let texels = bw == 1 && bh == 1;
    let side = 4 * bw.max(bh);
    let e = if wide {
        (8, 3)
    } else if pvrtc {
        (side, side)
    } else if texels {
        (5, 3)
    } else {
        (4 * bw, 3 * bh)
    };
    let odd = if wide {
        (12, 5)
    } else if pvrtc {
        e
    } else if texels {
        (7, 5)
    } else {
        (2 * bw + bw / 2 + 1, bh + bh / 2 + 1)
    };
    let mip = if wide {
        (8, 8)
    } else if pvrtc {
        (2 * side, 2 * side)
    } else {
        (4 * bw, 4 * bh)
    };
    let plain = |origin, size, fill| Case {
        array: false,
        extent: e,
        levels: 1,
        slices: 1,
        level: 0,
        slice: 0,
        origin,
        size,
        fill,
    };
    let color = |r, g, b, a, format| Fill::Color(ClearColor::new(r, g, b, a), format);
    let bytes = |n: u64| Fill::Bytes(n as usize);
    match name {
        "b0" => plain((0, 0), e, bytes(bb)),
        "b1" => plain((bw, bh), (bw, bh), bytes(3)),
        "b2" => plain((1, 1), (1, 1), bytes(bb)),
        "b3" => plain((0, 0), (2 * bw + 1, bh), bytes(bb + 5)),
        "b4" => plain((3 * bw, 2 * bh), (100, 100), bytes(2 * bb)),
        "b5" => Case {
            extent: odd,
            ..plain((0, 0), odd, bytes(bb))
        },
        "b6" => Case {
            extent: mip,
            levels: 3,
            level: 1,
            ..plain((0, 0), (mip.0 / 2, mip.1 / 2), bytes(1))
        },
        "b7" => Case {
            extent: mip,
            levels: 3,
            level: 2,
            ..plain((0, 0), (1, 1), bytes(bb))
        },
        "b8" => Case {
            array: true,
            slices: 2,
            slice: 1,
            ..plain((0, 0), e, bytes(bb))
        },
        "b9" => plain((2, 1), (5, 3), bytes(7)),
        "b10" => plain((0, 0), e, bytes(64)),
        "b11" => plain((bw / 2, 0), (bw, 1), bytes(bb)),
        "c0" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, None)),
        "c1" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, Some(raw))),
        "c2" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, Some(70))),
        "c3" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, Some(115))),
        "c4" => plain((bw, bh), (bw, bh), color(0.25, 0.5, 0.75, 1.0, None)),
        "c5" => plain((0, 0), e, color(2.0, -1.0, 0.5, 0.3, None)),
        "c6" => plain((0, 0), e, color(0.25, 0.5, 0.75, 1.0, Some(10))),
        "c7" => plain((0, 0), e, color(0.0, 0.0, 0.0, 0.0, None)),
        _ => unreachable!("no case {name}"),
    }
}

/// Metal's answers for two-plane textures of their own memory (`tfl4.swift`):
/// each format's raw value, whether 8 texels wide (else 5), and the digest of its cases' lines.
const OWN_TWO_PLANES: [(u32, bool, &str); 4] = [
    (550, false, "6254dd68bd6a8801"),
    (550, true, "3abe0fd65c9b2fc9"),
    (551, false, "6254dd68bd6a8801"),
    (551, true, "3abe0fd65c9b2fc9"),
];

/// The cases `tfl4.swift` ran.
const OWN_CASES: [&str; 15] = [
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "c2", "c3", "c6",
];

const CASES: [&str; 20] = [
    "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "c0", "c1", "c2",
    "c3", "c4", "c5", "c6", "c7",
];

fn format(raw: u32) -> PixelFormat {
    PixelFormat::try_from(raw).unwrap()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn digest(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))[..16].to_owned()
}

/// The fill of `case` into `texture` by `blit`.
fn fill(
    blit: &mut metal2vulkan_runtime::BlitCommandEncoder,
    texture: &metal2vulkan_runtime::Texture,
    case: &Case,
) -> Result<(), Error> {
    let region = Region::new_2d(case.origin.0, case.origin.1, case.size.0, case.size.1);
    match &case.fill {
        Fill::Bytes(n) => {
            let pattern: Vec<u8> = (0..*n).map(|k| 0x80u8.wrapping_add(k as u8)).collect();
            blit.fill_texture(texture, case.level, case.slice, region, &pattern)
        }
        Fill::Color(color, pf) => blit.fill_texture_color(
            texture,
            case.level,
            case.slice,
            region,
            *color,
            pf.map(format),
        ),
        Fill::Zeros => blit.fill_texture(texture, case.level, case.slice, region, &[0]),
    }
}

/// Whether `case`, of format `raw`, is a color fill packed in a format whose fills do not follow
/// the color: a block-compressed or 4:2:2 one, depth or stencil, two-plane extended-range, or
/// padded 10-bit (`blit_fill.rs`).
fn unpacked(raw: u32, case: &Case) -> bool {
    let Fill::Color(_, packed_in) = case.fill else {
        return false;
    };
    let raw = packed_in.unwrap_or(raw);
    let (_, bw, bh) = block(raw);
    let pf = format(raw);
    bw * bh > 1
        || is_depth_stencil(pf)
        || matches!(ExtendedRange::of(pf), Some(ExtendedRange::Planes { .. }))
        || padded_only_over_memory(pf)
}

/// Case `name` of format `raw` (of a heap of `heap` bytes where a two-plane format's, else
/// `wide` as [`case`] takes it): [`two_plane_case`]'s or [`texture_case`]'s answer, and whether
/// the case is [`unpacked`]. Such a case's answer is `unpacked`, as Metal's table records it,
/// where its fill wrote what the same case's fill of one zero byte writes, else both answers (or
/// the refusal, where the fill was refused).
fn case_answer(
    device: &Device,
    queue: &CommandQueue,
    (raw, name): (u32, &str),
    heap: u64,
    wide: bool,
) -> (Option<String>, bool) {
    let run = |zeros| {
        if heap > 0 {
            Some(two_plane_case(device, queue, raw, name, heap, zeros))
        } else {
            texture_case(device, queue, (raw, name), wide, zeros)
        }
    };
    let answer = run(false);
    if !unpacked(raw, &case(raw, name, wide)) {
        return (answer, false);
    }
    let zeros = run(true);
    let answer = match (answer, zeros) {
        // A refusal stays the answer: the fill wrote nothing.
        (Some(answer), _) if answer == "unmeasured" => Some(answer),
        (Some(answer), Some(zeros)) if answer == zeros => Some("unpacked".to_owned()),
        (Some(answer), Some(zeros)) => Some(format!("unpacked-differs:{answer}/{zeros}")),
        (answer, _) => answer,
    };
    (answer, true)
}

/// The answer a refused fill gives.
fn refusal(error: Error) -> String {
    match error {
        Error::AppleOnly(AppleOnly::FillUnmeasured) => "unmeasured".to_owned(),
        other => panic!("a fill refused: {other}"),
    }
}

/// The descriptor of `case`'s texture of `raw`.
fn descriptor(raw: u32, case: &Case, options: ResourceOptions) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(format(raw), case.extent.0, case.extent.1, false);
    if case.array {
        d.texture_type = TextureType::Type2DArray;
        d.array_length = case.slices;
    }
    d.mipmap_level_count = case.levels;
    d.usage = TextureUsage::SHADER_READ;
    d.resource_options = options;
    d
}

/// Case `name` of the two-plane format `raw` in a shared heap of `heap` bytes, its fill one zero
/// byte where `zeros`: the probe's "changed" line, or the refusal.
fn two_plane_case(
    device: &Device,
    queue: &CommandQueue,
    raw: u32,
    name: &str,
    heap: u64,
    zeros: bool,
) -> String {
    let case = case(raw, name, false);
    let case = match zeros {
        true => Case {
            fill: Fill::Zeros,
            ..case
        },
        false => case,
    };
    let options = ResourceOptions::shared();
    let placement = device
        .new_heap(&HeapDescriptor::new(HeapType::Placement, heap, options))
        .unwrap();
    let texture = placement
        .new_texture_at_offset(&descriptor(raw, &case, options), 0)
        .unwrap();
    let bytes = placement.new_buffer_at_offset(heap, options, 0).unwrap();
    put(device, queue, &bytes, &vec![0xee; heap as usize]);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let filled = fill(&mut blit, &texture, &case);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    if let Err(error) = filled {
        return refusal(error);
    }
    let got = get(device, queue, &bytes, heap);
    let mut runs = Vec::new();
    let mut i = 0;
    while i < got.len() {
        if got[i] == 0xee {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < got.len() && got[j] != 0xee {
            j += 1;
        }
        let shown = hex(&got[i..j.min(i + 64)]);
        let more = if j - i > 64 {
            format!("...({})", j - i)
        } else {
            String::new()
        };
        runs.push(format!("{i}:{shown}{more}"));
        i = j;
    }
    digest(&format!("changed {} runs: {}", runs.len(), runs.join(" ")))
}

/// Case `name` of format `raw` (`wide`, as [`case`] takes it), its fill one zero byte where
/// `zeros`: the probe's read-back lines' digest, the refusal, or `None` where the device holds no
/// such texture (or Metal makes none: a 4:2:2 texture of more than one level or slice).
fn texture_case(
    device: &Device,
    queue: &CommandQueue,
    (raw, name): (u32, &str),
    wide: bool,
    zeros: bool,
) -> Option<String> {
    let case = case(raw, name, wide);
    let case = match zeros {
        true => Case {
            fill: Fill::Zeros,
            ..case
        },
        false => case,
    };
    if matches!(raw, 240 | 241) && (case.levels > 1 || case.slices > 1) {
        return None;
    }
    let texture = match device.new_texture(&descriptor(raw, &case, ResourceOptions::private())) {
        Ok(texture) => texture,
        Err(Error::Unsupported(gap)) => {
            println!("XR-TEXTURE-FILLS-SKIPPED {raw} {name}: {gap:?}");
            return None;
        }
        Err(other) => panic!("{raw} {name}: {other}"),
    };
    let (bb, bw, bh) = block(raw);
    let aspects: &[(BlitOption, u64, &str)] = match raw {
        260 => &[
            (BlitOption::DepthFromDepthStencil, 4, "depth"),
            (BlitOption::StencilFromDepthStencil, 1, "stencil"),
        ],
        261 => &[(BlitOption::StencilFromDepthStencil, 1, "stencil")],
        _ => &[(BlitOption::None, bb, "all")],
    };
    // Each level, slice and aspect: its extent, its blocks across and down, and its bytes.
    let mut parts = Vec::new();
    let mut total = 0;
    for level in 0..case.levels {
        let w = (case.extent.0 >> level).max(1);
        let h = (case.extent.1 >> level).max(1);
        let (across, down) = (w.div_ceil(bw), h.div_ceil(bh));
        for slice in 0..case.slices {
            for &(option, unit, aspect) in aspects {
                // Each part at a multiple of 16 bytes, which every copy unit divides.
                total = u64::next_multiple_of(total, 16);
                let n = across * down * unit;
                parts.push((
                    level,
                    slice,
                    option,
                    unit,
                    aspect,
                    (w, h),
                    (across, down),
                    total,
                ));
                total += n;
            }
        }
    }
    let prefill = buffer_with(device, &vec![0xee; total as usize]).unwrap();
    let readback = buffer_with(device, &vec![0; total as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for &(level, slice, option, unit, _, (w, h), (across, down), at) in &parts {
        let row = across * unit;
        blit.copy_from_buffer_to_texture(
            &prefill,
            at,
            row,
            row * down,
            Size::new(w, h, 1),
            &texture,
            slice,
            level,
            Origin::new(0, 0, 0),
            option,
        )
        .unwrap_or_else(|e| panic!("{raw} {name}: prefill: {e}"));
    }
    let filled = fill(&mut blit, &texture, &case);
    for &(level, slice, option, unit, _, (w, h), (across, down), at) in &parts {
        let row = across * unit;
        blit.copy_from_texture_to_buffer(
            &texture,
            slice,
            level,
            Origin::new(0, 0, 0),
            Size::new(w, h, 1),
            &readback,
            at,
            row,
            row * down,
            option,
        )
        .unwrap_or_else(|e| panic!("{raw} {name}: read back: {e}"));
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{raw} {name}"
    );
    if let Err(error) = filled {
        return Some(refusal(error));
    }
    let got = read_buffer(&readback, 0, total).unwrap();
    let lines: Vec<String> = parts
        .iter()
        .map(|&(level, slice, _, unit, aspect, _, (across, down), at)| {
            let n = across * down * unit;
            let bytes = &got[at as usize..(at + n) as usize];
            format!(
                "level {level} slice {slice} {aspect} {across}x{down}x{unit}: {}",
                hex(bytes)
            )
        })
        .collect();
    Some(digest(&lines.join("\n")))
}

/// Every case of every format writes Metal's bytes, or is refused where the runtime refuses it.
#[test]
fn texture_fills_write_metal_s_bytes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let mut wrong = Vec::new();
        // Whether the device announced `unpacked-fill-color`, once a device.
        let mut unpacked_announced = false;
        'formats: for &(raw, heap, metal) in &FORMATS {
            messages.take_notices();
            let mut answers = Vec::new();
            let mut unpacked_here = false;
            for name in CASES {
                let (answer, unpacked) = case_answer(device, &queue, (raw, name), heap, false);
                // A color fill packed where fills ignore the color, which the runtime ran.
                unpacked_here |= unpacked && answer.as_deref().is_some_and(|a| a != "unmeasured");
                match answer {
                    Some(answer) => answers.push(format!("{name} {answer}")),
                    // A 4:2:2 texture Metal makes none of; another skip is the device's.
                    None if matches!(raw, 240 | 241) => {}
                    None => continue 'formats,
                }
            }
            // `Depth32Float_Stencil8`'s patterns are depth outside [0, 1]: where its uploads
            // announce that the device may store such a value clamped (`restricted-depth-upload`),
            // exactly where the runtime plans a depth image that may clamp (`texture_blits.rs`),
            // the format's bytes are not Metal's to match.
            let first_unpacked = unpacked_here && !unpacked_announced;
            unpacked_announced |= first_unpacked;
            let kinds: &[Degradation] = if first_unpacked { &UNPACKED_FILL } else { &[] };
            // A read-only `Depth32Float` texture's depth twin and a BC texture's decoded twin; the
            // uploads into a depth image (the twin's texture holds its texels in colour), what
            // a float depth's may store, and the combined format's clamp; the `Shared` buffers.
            let pf = format(raw);
            let shapes: Vec<bool> = CASES
                .iter()
                .map(|name| {
                    let case = case(raw, name, false);
                    !case.array && case.levels == 1
                })
                .collect();
            let one_level = shapes.iter().any(|&one| one);
            let every_one_level = shapes.iter().all(|&one| one);
            let twin = pf == PixelFormat::Depth32Float && twins_depth(device);
            let combined = pf == PixelFormat::Depth32FloatStencil8;
            let depth_image = combined || (pf == PixelFormat::Depth32Float && !twin);
            let first = forced(&[
                (Degradation::DepthTwin, twin),
                (Degradation::DecodedBlockTwin, twins_blocks(device, pf)),
                (
                    Degradation::CarriedDecodedTwin,
                    twins_carried(device, pf)[0],
                ),
                (
                    Degradation::DecodedTexelPrecision,
                    twins_carried(device, pf)[1],
                ),
                (
                    Degradation::CompressedTexelDecode,
                    decodes_blocks_its_way(device, pf),
                ),
                (
                    Degradation::DepthDenormals,
                    depth_image && loses_depth_denormals(device, false),
                ),
                (
                    Degradation::RestrictedDepthUpload,
                    combined && depth_images(device).1,
                ),
                (
                    Degradation::SharedInSystemMemory,
                    shared_in_system_memory(device),
                ),
                // A two-plane format's cases lie in a heap: a 2D texture of one level a shadow of
                // its planes in the heap's bytes, any other apart from them.
                (Degradation::ShadowedTexture, heap > 0 && one_level),
                (
                    Degradation::HeapTwoPlaneTexture,
                    heap > 0 && !every_one_level,
                ),
            ]);
            let announced = assert_made_once(messages, kinds, &first);
            let clamped = announced.contains_key(&Degradation::RestrictedDepthUpload);
            if raw == 260 {
                assert_eq!(
                    clamped,
                    depth_images(device).1,
                    "restricted-depth-upload announced exactly where a depth image may clamp"
                );
                if clamped {
                    println!("XR-TEXTURE-FILLS-SKIPPED 260: restricted-depth-upload announced");
                    continue;
                }
            }
            let ours = digest(&answers.join("\n"));
            if ours != metal {
                println!(
                    "XR-TEXTURE-FILLS-DIFFERS {raw} {ours} (Metal's {metal}): {}",
                    answers.join("; ")
                );
                wrong.push(raw);
            }
        }
        assert!(wrong.is_empty(), "fills unlike Metal's: {wrong:?}");
    });
}

/// Every case of a two-plane texture of its own memory writes Metal's bytes, or is refused where
/// the runtime refuses it (5 texels wide, or of more than one level).
#[test]
fn own_two_plane_fills_write_metal_s_bytes() {
    on_device(|device, _messages| {
        let queue = device.new_command_queue().unwrap();
        let mut wrong = Vec::new();
        for &(raw, wide, metal) in &OWN_TWO_PLANES {
            let mut answers = Vec::new();
            for name in OWN_CASES {
                let answer = case_answer(device, &queue, (raw, name), 0, wide)
                    .0
                    .unwrap_or_else(|| panic!("no {raw} texture of its own"));
                answers.push(format!("{name} {answer}"));
            }
            let ours = digest(&answers.join("\n"));
            if ours != metal {
                println!(
                    "XR-TEXTURE-FILLS-DIFFERS {raw} wide {wide} {ours} (Metal's {metal}): {}",
                    answers.join("; ")
                );
                wrong.push((raw, wide));
            }
        }
        assert!(wrong.is_empty(), "fills unlike Metal's: {wrong:?}");
    });
}
