//! A fast-math sum in Metal's grouping (`lower::arith::reassoc`), on the device the cfg selects,
//! under the validation layer.
//!
//! The owned fixture `sum_regroup` draws Metal's fullscreen triangle over a 64x64 `R32Uint`
//! target. Each fragment samples one channel of the nine nearest texels of an `RGBA16Float`
//! texture of fixed random halfs (magnitudes in [2^-4, 2^4), either sign), keeps a neighbour only
//! inside the bounds (0, 0) to (64, 64), weighs the four edges by two, and writes the bits of two
//! `half` sums of six leaves. Metal regroups each sum by its ranks (the fixture's docs); its
//! answer on an Apple M4 is each sum in the grouping the fixture names, every
//! addition rounded to `half`, ties to even, signed zeros included. [`expected`] computes it on
//! the host from the same texels.
//!
//! The translation computes Metal's grouping and keeps it (`NoContraction`), so a device whose
//! `half` additions round correctly gives these bits exactly.

use super::buffer_with;
use super::resolves::{lane, quiet, Lane};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, LoadAction, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, StoreAction, TextureDescriptor,
};
use std::path::Path;

/// The target's and the texture's width and height.
const W: u64 = 64;

/// The constant buffer: the offset (0, 0), the bounds (0, 0) to (64, 64), and the step 1.
const CB: [f32; 8] = [0.0, 0.0, 0.0, 0.0, 64.0, 64.0, 1.0, 0.0];

/// The value of IEEE half bits `h` (finite only, as every texel and sum here is).
fn half(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f64::from(h & 0x3ff);
    match e {
        0 => sign * m * 2f64.powi(-24),
        _ => sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// `x` rounded to half bits, ties to even (finite and at most 65504 in magnitude, as every sum
/// here is); a zero keeps its sign.
fn to_half(x: f64) -> u16 {
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    let mut e = a.log2().floor() as i32;
    if a < 2f64.powi(e) {
        e -= 1;
    } else if a >= 2f64.powi(e + 1) {
        e += 1;
    }
    if e < -14 {
        // Subnormal: multiples of 2^-24.
        let m = (a * 2f64.powi(24)).round_ties_even() as u16;
        return sign | m;
    }
    let mut m = (a / 2f64.powi(e) * 1024.0).round_ties_even();
    if m >= 2048.0 {
        m /= 2.0;
        e += 1;
    }
    sign | (((e + 15) as u16) << 10) | ((m as u16).wrapping_sub(1024) & 0x3ff)
}

/// A half value (as bits) in an expression, each operation rounded to half.
#[derive(Clone, Copy)]
struct H(u16);

impl std::ops::Add for H {
    type Output = H;
    fn add(self, o: H) -> H {
        // Two halves' sum is exact in f64, so one rounding to half is the half addition's.
        H(to_half(half(self.0) + half(o.0)))
    }
}

impl std::ops::Sub for H {
    type Output = H;
    fn sub(self, o: H) -> H {
        H(to_half(half(self.0) - half(o.0)))
    }
}

impl std::ops::Neg for H {
    type Output = H;
    fn neg(self) -> H {
        H(self.0 ^ 0x8000)
    }
}

/// The texels: four random halfs each, row by row (the generator of the probe's runner).
fn texels() -> Vec<u16> {
    let mut state: u64 = 202_610_092;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as u32
    };
    (0..W * W * 4)
        .map(|_| {
            let r = next();
            let e = 11 + r % 8;
            ((((r >> 20) & 1) << 15) | (e << 10) | ((r >> 4) & 0x3ff)) as u16
        })
        .collect()
}

/// The leaves of pixel (`x`, `y`): each of the nine samples' fourth channel, zero outside the
/// bounds, the four edges doubled, in the order C, L, R, U, D, TL, TR, BL, BR.
fn leaves(texels: &[u16], x: u64, y: u64) -> [H; 9] {
    let at = |dx: i64, dy: i64| {
        let (qx, qy) = (x as i64 + dx, y as i64 + dy);
        let inside = (0..W as i64).contains(&qx) && (0..W as i64).contains(&qy);
        if !inside {
            return H(0);
        }
        H(texels[((qy as u64 * W + qx as u64) * 4 + 3) as usize])
    };
    let edge = |h: H| h + h;
    [
        at(0, 0),
        edge(at(-1, 0)),
        edge(at(1, 0)),
        edge(at(0, -1)),
        edge(at(0, 1)),
        at(-1, -1),
        at(1, -1),
        at(-1, 1),
        at(1, 1),
    ]
}

/// Metal's answer of `fragment` at pixel (`x`, `y`): its two sums in Metal's grouping (the
/// fixture's docs), the first in the low half.
fn expected(fragment: &str, texels: &[u16], x: u64, y: u64) -> u32 {
    let [_c, wl, wr, wu, wd, tl, tr, bl, br] = leaves(texels, x, y);
    let (gx, gy) = match fragment {
        "regroup" => (
            (((wr - (wl + tl)) + tr) - bl) + br,
            ((wd - ((wu + tl) + tr)) + bl) + br,
        ),
        "tied" => (
            (((-tl - (bl + wl)) + br) + wr) + tr,
            (((bl - (wu + tl)) + br) - tr) + wd,
        ),
        _ => unreachable!("a measured fragment"),
    };
    u32::from(gx.0) | (u32::from(gy.0) << 16)
}

/// The fixture's pipeline of fragment function `fragment` on `lane`'s device.
fn pipeline(lane: &Lane, fragment: &str) -> RenderPipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/engine/sum_regroup.ll");
    let library = lane
        .device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let function = |name| {
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function("sum_regroup_full")));
    descriptor.fragment = Some(StageSource::Function(function(fragment)));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::R32Uint));
    lane.device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The words of one draw of `pipeline` over `texels`, row by row.
fn drawn(lane: &Lane, pipeline: &RenderPipelineState, texels: &[u16]) -> Vec<u32> {
    let mut target = TextureDescriptor::texture_2d(PixelFormat::R32Uint, W, W, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = lane.device.new_texture(&target).unwrap();
    let mut sampled = TextureDescriptor::texture_2d(PixelFormat::RGBA16Float, W, W, false);
    sampled.usage = TextureUsage::SHADER_READ;
    sampled.resource_options = ResourceOptions::shared();
    let sampled = lane.device.new_texture(&sampled).unwrap();
    let bytes: Vec<u8> = texels.iter().flat_map(|t| t.to_le_bytes()).collect();
    sampled
        .replace_region(Region::new_2d(0, 0, W, W), 0, 0, &bytes, W * 8, 0)
        .unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.clear_color = ClearColor::new(f64::from(0xabab_abab_u32), 0.0, 0.0, 0.0);
    attachment.store_action = StoreAction::Store;
    lane.run(&descriptor, |lane, encoder, held| {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        let cb: Vec<u8> = CB.iter().flat_map(|f| f.to_le_bytes()).collect();
        let buffer = buffer_with(&lane.device, &cb).unwrap();
        encoder.set_fragment_buffer(Some(&buffer), 0, 0).unwrap();
        held.push(buffer);
        encoder.set_fragment_texture(Some(&sampled), 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
    })
    .unwrap();
    lane.read(&target, (0, 0), 4, BlitOption::None)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Each pinned sum gives Metal's bits on every pixel: its grouping, not the AIR's.
#[test]
fn a_pinned_sum_gives_metal_s_bits() {
    let (messages, _cache, lane) = lane();
    let name = lane.device.name().to_owned();
    let texels = texels();
    for fragment in ["regroup", "tied"] {
        let words = drawn(&lane, &pipeline(&lane, fragment), &texels);
        let wrong: Vec<String> = words
            .iter()
            .enumerate()
            .filter_map(|(i, &got)| {
                let (x, y) = (i as u64 % W, i as u64 / W);
                let want = expected(fragment, &texels, x, y);
                (got != want).then(|| format!("({x}, {y}) {got:#010x}, Metal's {want:#010x}"))
            })
            .collect();
        eprintln!(
            "sum regroup {fragment} on {name}: {} of {} pixels differ from Metal's",
            wrong.len(),
            words.len()
        );
        assert!(
            wrong.is_empty(),
            "{fragment} on {name}: {} pixels differ:\n  {}",
            wrong.len(),
            wrong
                .iter()
                .take(16)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }
    quiet(lane, &messages);
}

/// The texels tell the groupings apart: on some pixel the AIR's grouping of `regroup`'s first sum
/// (the frontend's) gives other bits than Metal's.
#[test]
fn the_texels_tell_the_groupings_apart() {
    let texels = texels();
    let differs = (0..W * W).any(|i| {
        let (x, y) = (i % W, i / W);
        let [_c, wl, wr, _wu, _wd, tl, tr, bl, br] = leaves(&texels, x, y);
        let air = (((wr + tr) - ((wl + tl) + bl)) + br).0;
        u32::from(air) != expected("regroup", &texels, x, y) & 0xffff
    });
    assert!(
        differs,
        "the AIR's grouping gives Metal's bits on every pixel"
    );
}
