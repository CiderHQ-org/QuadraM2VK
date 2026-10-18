//! Texture texels end to end on the CPU interpreter's image model.
//!
//! The `texture_ops` fixture's kernels run on the interpreter with bound images (each image holds
//! its texels as the four 32-bit components of its sampled type) and must compute their Metal
//! results in both scheduling orders, on every committed profile that compiles them without
//! runtime inputs:
//!
//! - texel buffers: a uniform texel buffer read, doubled, written to a storage texel buffer;
//! - a read_write texture read, written, fenced, and read again;
//! - cube and cube-array reads through the 2D-array view (face + 6 x cube), cube queries;
//! - size, array-size, level, sample-count queries, and `is_null_texture` of a bound texture;
//! - 16-bit signed texels (sign-extended to the int texel on write) and a `half` 3D read at a level;
//! - a depth read and a multisampled integer read.
//!
//! Containment: the interpreter traps any access outside an image, so a dispatch that overruns
//! its image on a profile without robust image access (for a texel buffer, without
//! `robustBufferAccess2`) completes only because the shader checks its reads and guards its
//! writes, and the in-range texels are exact. `engine_texture_oob` checks the out-of-bounds
//! values themselves.
//!
//! `llvm-as` assembles the fixture; without it the tests skip visibly.

use std::collections::BTreeMap;

use crate::device::profile::{Containment, DeviceProfile, Robustness};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription};
use crate::error::Error;
use crate::lower::select::Plan;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{engine_dir, llvm_assemble, typed_module, AssembleError};
use crate::spirv::interp::{
    run_with_images, Buffer, Config, Image, InterpError, Order, Resources, TrapKind,
};
use crate::spirv::ir::TypedModule;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("texture_ops.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected texture_ops: {why}"),
    }
}

/// `stem` with the robustness features in `drop` removed.
fn without(stem: &str, drop: &[VkFeature]) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in drop {
        desc.features.remove(f);
    }
    Device::new(desc, crate::pipeline::testing::EXACT_SAMPLING)
        .expect("device")
        .profile()
        .clone()
}

/// Every committed profile, and two without any robustness (so the shader contains itself).
fn profiles() -> Vec<(String, DeviceProfile)> {
    let mut out = crate::spirv::testing::committed_exact_profiles().expect("profiles");
    let all = [
        VkFeature::RobustImageAccess,
        VkFeature::RobustImageAccess2,
        VkFeature::RobustBufferAccess,
        VkFeature::RobustBufferAccess2,
    ];
    for stem in ["moltenvk-m3", "lavapipe"] {
        out.push((format!("{stem} unrobust"), without(stem, &all)));
    }
    out
}

/// Reads are checked and writes guarded for images on `p`.
fn contains_images(p: &DeviceProfile) -> bool {
    let f = p.fields();
    !f.images.robust_image_access && matches!(f.containment, Containment::On { .. })
}

/// Reads are checked and writes guarded for texel buffers on `p`: `robustBufferAccess` may write
/// past a texel buffer anywhere in its buffer.
fn contains_texel_buffers(p: &DeviceProfile) -> bool {
    matches!(
        p.fields().containment,
        Containment::On {
            robust: Robustness::None | Robustness::Rba
        }
    )
}

/// The module of `entry` on `p`, or `None` when `p` needs a runtime input for it (a storage
/// texture format without a formatless load/store).
fn module(bc: &[u8], entry: &str, stem: &str, p: &DeviceProfile) -> Option<(Plan, TypedModule)> {
    match typed_module(bc, entry, p) {
        Ok(m) => Some(m),
        Err(Error::NeedsInput(_)) => None,
        Err(e) => panic!("{stem}: {entry}: {e}"),
    }
}

/// Resources of a dispatch of 1x1x1 threadgroups, `[[buffer(i)]]` holding `init[i]`.
fn resources(plan: &Plan, init: &[Vec<u8>]) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    // A single-region dispatch (the region is unused) with every buffer bound at its start.
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            // 0: no channel narrower than the written word, every integer lane as written (the
            // interpreter's images hold words; the kernels write in-range values).
            (PushField::TextureIntegerChannels(_), PushLoc::Push { .. }) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let buffers: Vec<Buffer> = init
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    Resources {
        buffers,
        push,
        spec,
    }
}

fn texture(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index,
    }
}

/// Run over `grid` threadgroups of one thread in both scheduling orders; both must agree.
fn run_both(
    m: &TypedModule,
    res: &Resources,
    images: &[Image],
    grid: [u32; 3],
) -> Result<(Vec<Vec<u8>>, Vec<Image>), InterpError> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: grid,
            order,
            ..Config::default()
        };
        let (o, imgs) = run_with_images(m, &cfg, res.clone(), images.to_vec())?;
        outs.push((o.buffers, imgs));
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    Ok(outs.swap_remove(0))
}

fn fl(x: f32) -> u32 {
    x.to_bits()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `dst[i] = src[i] * 2` over a texel buffer of `N` texels; on a profile that contains texel
/// buffers, a grid of `N + 3` threads checks its reads and drops its writes past the end.
#[test]
fn texel_buffers_double_their_input() {
    const N: u32 = 5;
    let Some(bc) = fixture() else {
        return;
    };
    let mut ran = 0;
    let mut overran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "texel_buffers", &stem, &p) else {
            continue;
        };
        let mut src = Image::new(texture(0), [N, 1, 1], 1, 1, 1);
        for (i, t) in src.texels.iter_mut().enumerate() {
            let x = i as f32;
            *t = [fl(x), fl(-x), fl(0.5 + x), fl(1.0)];
        }
        let dst = Image::new(texture(1), [N, 1, 1], 1, 1, 1);
        let threads = if contains_texel_buffers(&p) {
            overran += 1;
            N + 3
        } else {
            N
        };
        let res = resources(&plan, &[]);
        let (_, imgs) = run_both(&m, &res, &[src, dst], [threads, 1, 1])
            .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        for (i, t) in imgs[1].texels.iter().enumerate() {
            let x = i as f32;
            assert_eq!(
                *t,
                [fl(2.0 * x), fl(-2.0 * x), fl(1.0 + 2.0 * x), fl(2.0)],
                "{stem} texel {i}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0 && overran > 0, "ran {ran}, overran {overran}");
}

/// A texture buffer's reads take the bound swizzle: `texel_buffers` states
/// `RuntimeFact::TextureSwizzle` for the slot it reads and not the one it writes, and under each
/// swizzle's `TextureSwizzle` specialization writes `2 * src[i]` swizzled as Metal swizzles it:
/// each channel from the source the swizzle names, 0 or 1 for `Zero` and
/// `One`. Without the specialization the read is unswizzled.
#[test]
fn a_texel_buffer_read_takes_the_bound_swizzle() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels as C};
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::{SlotClass, SlotRef};
    const N: u32 = 5;
    let Some(bc) = fixture() else {
        return;
    };
    let read = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    };
    let written = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 1,
    };
    let swizzles = [
        None,
        Some(C::new(S::Blue, S::Green, S::Red, S::One)),
        Some(C::new(S::Zero, S::One, S::Alpha, S::Red)),
        Some(C::new(S::Green, S::Green, S::Green, S::Green)),
        Some(C::new(S::Alpha, S::Blue, S::Green, S::Red)),
    ];
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "texel_buffers", &stem, &p) else {
            continue;
        };
        let facts = &plan.facts;
        assert!(
            facts.contains(&RuntimeFact::TextureSwizzle { slot: read }),
            "{stem}"
        );
        assert!(
            !facts.contains(&RuntimeFact::TextureSwizzle { slot: written }),
            "{stem}"
        );
        for swizzle in swizzles {
            let mut src = Image::new(texture(0), [N, 1, 1], 1, 1, 1);
            for (i, t) in src.texels.iter_mut().enumerate() {
                let x = i as f32;
                *t = [fl(x), fl(-x), fl(0.5 + x), fl(1.0 + x)];
            }
            let dst = Image::new(texture(1), [N, 1, 1], 1, 1, 1);
            let mut res = resources(&plan, &[]);
            if let Some(s) = swizzle {
                res.spec
                    .insert(SpecKey::TextureSwizzle(read), u64::from(s.word()));
            }
            let (_, imgs) = run_both(&m, &res, &[src, dst], [N, 1, 1])
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            let pick = |s: S, t: [f32; 4]| match s {
                S::Zero => 0.0,
                S::One => 1.0,
                S::Red => t[0],
                S::Green => t[1],
                S::Blue => t[2],
                _ => t[3],
            };
            for (i, got) in imgs[1].texels.iter().enumerate() {
                let x = i as f32;
                let t = [x, -x, 0.5 + x, 1.0 + x];
                let s = swizzle.unwrap_or(C::DEFAULT);
                let want = [s.red, s.green, s.blue, s.alpha].map(|c| fl(2.0 * pick(c, t)));
                assert_eq!(*got, want, "{stem} {swizzle:?} texel {i}");
            }
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// A swizzled texture buffer's read past its texels reads Metal's out-of-bounds texel swizzled,
/// its alpha the bound format's (1 where it has none) wherever the swizzle carries it, also where
/// the function reads only the red lane (the MoltenVK lane read 0 there): `ob1_texel_buffer` writes
/// `src.read(idx[t]).x`, so on a profile whose shader checks its texel-buffer reads it states
/// `OutOfBoundsAlpha` for the slot it reads, and under a swizzle taking red from alpha reads each
/// texel's alpha in bounds and the specialized alpha past them.
#[test]
fn a_swizzled_texel_buffer_reads_metals_out_of_bounds_alpha_in_any_lane() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels as C};
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::{SlotClass, SlotRef};
    let bc = match llvm_assemble(&engine_dir().join("ob1_texel_buffer.ll")) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected ob1_texel_buffer: {why}"),
    };
    let read = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    };
    const N: u32 = 4;
    let at: [u32; 8] = [0, 1, 2, 3, 4, 7, 64, 1 << 20];
    let idx: Vec<u8> = at
        .iter()
        .copied()
        .chain(0..8)
        .flat_map(|i: u32| i.to_le_bytes())
        .collect();
    let red_from_alpha = C::new(S::Alpha, S::Green, S::Blue, S::Alpha);
    let mut ran = 0;
    for (stem, p) in profiles() {
        if !contains_texel_buffers(&p) {
            continue;
        }
        let Some((plan, m)) = module(&bc, "ob1_texel_buffer", &stem, &p) else {
            continue;
        };
        for fact in [
            RuntimeFact::TextureSwizzle { slot: read },
            RuntimeFact::OutOfBoundsAlpha { slot: read },
        ] {
            assert!(plan.facts.contains(&fact), "{stem}: {fact:?}");
        }
        // An `R32Uint` texture buffer: each texel `(v, 0, 0, 1)`, no alpha of its own.
        let mut src = Image::new(texture(0), [N, 1, 1], 1, 1, 1);
        for (i, t) in src.texels.iter_mut().enumerate() {
            *t = [100 + i as u32, 0, 0, 1];
        }
        let dst = Image::new(texture(1), [8, 1, 1], 1, 1, 1);
        let mut res = resources(&plan, &[idx.clone(), vec![0u8; 32]]);
        res.spec.insert(
            SpecKey::TextureSwizzle(read),
            u64::from(red_from_alpha.word()),
        );
        res.spec.insert(SpecKey::OutOfBoundsAlpha(read), 1);
        let (bufs, _) =
            run_both(&m, &res, &[src, dst], [8, 1, 1]).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(words(&bufs[1]), vec![1; 8], "{stem}");
        ran += 1;
    }
    assert!(ran > 0);
}

/// `t = 2 * (t + (1, 2, 3, 4))` per texel of a 2x2 read_write texture; on a profile that contains
/// images, a 4x3 grid checks its reads and drops its writes outside the texture.
#[test]
fn read_write_fence_rereads_its_own_write() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut ran = 0;
    let mut overran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "read_write_fence", &stem, &p) else {
            continue;
        };
        let mut t = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
        for (i, x) in t.texels.iter_mut().enumerate() {
            let i = i as u32;
            *x = [i, 10 * i, u32::MAX - i, 7];
        }
        let grid = if contains_images(&p) {
            overran += 1;
            [4, 3, 1]
        } else {
            [2, 2, 1]
        };
        let res = resources(&plan, &[]);
        let (_, imgs) = run_both(&m, &res, &[t], grid).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        for (i, x) in imgs[0].texels.iter().enumerate() {
            let i = i as u32;
            let want = [
                (i + 1).wrapping_mul(2),
                (10 * i + 2).wrapping_mul(2),
                (u32::MAX - i).wrapping_add(3).wrapping_mul(2),
                22,
            ];
            assert_eq!(*x, want, "{stem} texel {i}");
        }
        ran += 1;
    }
    assert!(ran > 0 && overran > 0, "ran {ran}, overran {overran}");
}

/// A cube read of face 3 is layer 3 of the 2D-array view; a cube-array read of face 4 of cube 1
/// is layer 10; the queries see the cube's width, the cube count, and the level count.
#[test]
fn cube_faces_read_their_view_layers() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "cube_faces", &stem, &p) else {
            continue;
        };
        let face = |layer: usize| {
            [
                fl(layer as f32),
                fl(0.25),
                fl(-1.0),
                fl(100.0 + layer as f32),
            ]
        };
        let mut c = Image::new(texture(0), [4, 4, 1], 6, 3, 1);
        let mut ca = Image::new(texture(1), [4, 4, 1], 12, 1, 1);
        for img in [&mut c, &mut ca] {
            for layer in 0..img.layers {
                let i = img.index(0, layer, 0, [0, 0, 0]).expect("texel");
                img.texels[i] = face(layer as usize);
            }
        }
        let res = resources(&plan, &[vec![0; 48]]);
        let (bufs, _) =
            run_both(&m, &res, &[c, ca], [1, 1, 1]).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        let out = words(&bufs[0]);
        assert_eq!(out[0..4], face(3), "{stem}");
        assert_eq!(out[4..8], face(10), "{stem}");
        assert_eq!(out[8..12], [fl(4.0), fl(2.0), fl(3.0), fl(0.0)], "{stem}");
        ran += 1;
    }
    assert!(ran > 0);
}

/// Each query reads its image: level sizes halve, array sizes and level and sample counts are the
/// bound image's, and a bound texture is not null.
#[test]
fn queries_read_the_bound_images() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "queries", &stem, &p) else {
            continue;
        };
        let images = [
            Image::new(texture(0), [8, 4, 1], 3, 3, 1),
            Image::new(texture(1), [4, 4, 2], 1, 1, 1),
            Image::new(texture(2), [6, 4, 1], 1, 1, 4),
            Image::new(texture(3), [5, 5, 1], 1, 1, 1),
        ];
        let res = resources(&plan, &[vec![0; 36]]);
        let (bufs, _) =
            run_both(&m, &res, &images, [1, 1, 1]).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(words(&bufs[0]), [4, 2, 3, 3, 2, 4, 6, 5, 0], "{stem}");
        ran += 1;
    }
    assert!(ran > 0);
}

/// A `short` texel is the low half of its int texel: `x - 1` is written sign-extended (a negative
/// short stays negative in the int texel); a `half` read at level 1 rounds each component.
#[test]
fn short_texels_sign_extend_and_half_reads_round() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "short_texels", &stem, &p) else {
            continue;
        };
        let mut s = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
        for (i, t) in s.texels.iter_mut().enumerate() {
            let i = i as i32;
            *t = [(-5 - i) as u32, (7 + i) as u32, i16::MIN as i32 as u32, 0];
        }
        let d = Image::new(texture(1), [2, 2, 1], 1, 1, 1);
        let mut h = Image::new(texture(2), [4, 4, 2], 1, 2, 1);
        // Every thread reads its own texel of level 1 into the same output: all hold one value.
        // 1 + 2^-12 is between two halves and rounds to even (1.0).
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let at = h.index(1, 0, 0, [x, y, 0]).expect("texel");
            h.texels[at] = [fl(1.5), fl(-0.25), fl(1.0 + 1.0 / 4096.0), fl(3.0)];
        }
        let res = resources(&plan, &[vec![0; 8]]);
        let (bufs, imgs) =
            run_both(&m, &res, &[s, d, h], [2, 2, 1]).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        for (i, t) in imgs[1].texels.iter().enumerate() {
            let i = i as i32;
            assert_eq!(
                *t,
                [(-6 - i) as u32, (6 + i) as u32, i16::MAX as u32, u32::MAX],
                "{stem} texel {i}"
            );
        }
        let halves: Vec<u16> = bufs[0]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(halves, [0x3E00, 0xB400, 0x3C00, 0x4200], "{stem}");
        ran += 1;
    }
    assert!(ran > 0);
}

/// A `half` read converts its texel by the bound format's class, the slot's `TexelFormat`
/// specialization: a 32-bit float texel to nearest even, half denormals kept; a 16-bit normalized
/// texel's code value `c / 65535` correctly rounded, where rounding the `f32` value would round
/// twice (`65455` and `65519`); a format whose conversion is not measured (the default) takes every
/// other `half` texel's exact rounding, which keeps a half denormal.
#[test]
fn half_reads_convert_by_the_bound_format() {
    use crate::metal::texel_format::texel_format;
    use crate::metal::PixelFormat;
    use crate::pipeline::artifact::PrecisionNote;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::{SlotClass, SlotRef};
    let Some(bc) = fixture() else {
        return;
    };
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 2,
    };
    let unorm16 = |c: u32| (f64::from(c) / 65535.0) as f32;
    let float = [2f32.powi(-20), 65520.0, 1.0 + 1.0 / 4096.0, 3.0];
    let cases: [(Option<u32>, [f32; 4], [u16; 4]); 3] = [
        (None, float, [0x0010, 0x7C00, 0x3C00, 0x4200]),
        (
            Some(texel_format(PixelFormat::RGBA32Float).0),
            float,
            [0x0010, 0x7C00, 0x3C00, 0x4200],
        ),
        (
            Some(texel_format(PixelFormat::RGBA16Unorm).0),
            [unorm16(65455), unorm16(65519), unorm16(1), unorm16(65535)],
            [0x3BFD, 0x3BFF, 0x0100, 0x3C00],
        ),
    ];
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "short_texels", &stem, &p) else {
            continue;
        };
        assert!(
            plan.facts.contains(&RuntimeFact::TexelFormat { slot }),
            "{stem}"
        );
        assert!(
            plan.precision.contains(&PrecisionNote::HalfTexelConversion),
            "{stem}"
        );
        for (class, texel, want) in cases {
            let s = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
            let d = Image::new(texture(1), [2, 2, 1], 1, 1, 1);
            let mut h = Image::new(texture(2), [4, 4, 2], 1, 2, 1);
            for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let at = h.index(1, 0, 0, [x, y, 0]).expect("texel");
                h.texels[at] = texel.map(fl);
            }
            let mut res = resources(&plan, &[vec![0; 8]]);
            if let Some(c) = class {
                res.spec.insert(SpecKey::TexelFormat(slot), u64::from(c));
            }
            let (bufs, _) = run_both(&m, &res, &[s, d, h], [2, 2, 1])
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            let halves: Vec<u16> = bufs[0]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            assert_eq!(halves, want, "{stem} class {class:?}");
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// A `half` read keeps the texel's exact value: a half-float texel, which the device decodes to its
/// exact `f32` value, reads back bit for bit, every half subnormal, both zeros, both infinities,
/// `±65504`, and NaN payloads included; a 32-bit float texel rounds to nearest even onto the half
/// grid, subnormals kept, past `65504` to the infinity. The interpreter gives `OpQuantizeToF16`'s
/// result below the smallest normal half as the zero of the input's sign and its NaN as `0x7E00`,
/// so a bare quantize fails here.
#[test]
fn half_reads_keep_subnormals_zeros_infinities_and_nans() {
    let Some(bc) = fixture() else {
        return;
    };
    // The `f32` decode of half `h`: its value, or the NaN of its sign and payload.
    let wide = |h: u16| -> u32 {
        let sign = u32::from(h & 0x8000) << 16;
        let e = u32::from(h >> 10 & 0x1F);
        let m = u32::from(h & 0x3FF);
        sign | match e {
            0x1F => 0x7F80_0000 | m << 13,
            0 => (m as f32 * 2f32.powi(-24)).to_bits(),
            _ => (e + 112) << 23 | m << 13,
        }
    };
    let halves: [[u16; 4]; 5] = [
        [0x0001, 0x8001, 0x03FF, 0x83FF],
        [0x0000, 0x8000, 0x7C00, 0xFC00],
        [0x7BFF, 0xFBFF, 0x0400, 0x8200],
        [0x7E00, 0x7E01, 0xFD55, 0x7C01],
        [0x0155, 0x82AA, 0x7FFF, 0x3C00],
    ];
    let mut sets: Vec<([u32; 4], [u16; 4])> = halves.iter().map(|h| (h.map(wide), *h)).collect();
    // 32-bit float texels: 1.5 · 2^-24 (a tie, to the even 0x0002), 2^-25 (a tie, to +0),
    // 2^-25 one ulp up (0x0001), and 65520 (a tie past 65504, to the infinity).
    sets.push((
        [0x33C0_0000, 0x3300_0000, 0xB300_0001, 0x477F_F000],
        [0x0002, 0x0000, 0x8001, 0x7C00],
    ));
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "short_texels", &stem, &p) else {
            continue;
        };
        for (texel, want) in &sets {
            let s = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
            let d = Image::new(texture(1), [2, 2, 1], 1, 1, 1);
            let mut h = Image::new(texture(2), [4, 4, 2], 1, 2, 1);
            for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let at = h.index(1, 0, 0, [x, y, 0]).expect("texel");
                h.texels[at] = *texel;
            }
            let res = resources(&plan, &[vec![0; 8]]);
            let (bufs, _) = run_both(&m, &res, &[s, d, h], [2, 2, 1])
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            let got: Vec<u16> = bufs[0]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            assert_eq!(got, want, "{stem}: texel {texel:08x?}");
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// A depth read is the texel's first component; a multisampled read reads its sample.
#[test]
fn depth_and_multisample_reads_read_their_texel() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "depth_ms", &stem, &p) else {
            continue;
        };
        let mut dp = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
        dp.texels[0] = [fl(0.75), fl(9.0), fl(9.0), fl(9.0)];
        let mut ms = Image::new(texture(1), [2, 2, 1], 1, 1, 4);
        for sample in 0..4 {
            let i = ms.index(0, 0, sample, [0, 0, 0]).expect("texel");
            ms.texels[i] = [0, (-3 - sample as i32) as u32, 0, 0];
        }
        let res = resources(&plan, &[vec![0; 8]]);
        let (bufs, _) =
            run_both(&m, &res, &[dp, ms], [1, 1, 1]).unwrap_or_else(|e| panic!("{stem}: {e:?}"));
        assert_eq!(words(&bufs[0]), [fl(0.75), fl(-5.0)], "{stem}");
        ran += 1;
    }
    assert!(ran > 0);
}

/// A depth read with a `float` result converts its texel by the bound format's class, the slot's
/// `TexelFormat` specialization, on a profile without Metal's own sampler: a `Depth16Unorm`
/// texel the device gives as its code times the reciprocal (lavapipe) reads `c / 65535`
/// correctly rounded, Metal's, and a `Depth32Float` texel reads as given. On
/// a profile with Metal's own sampler the device's value is Metal's, and the read takes no class.
#[test]
fn a_depth_read_converts_by_the_bound_format() {
    use crate::device::requirements::VkProperty;
    use crate::metal::texel_format::texel_format;
    use crate::metal::PixelFormat;
    use crate::pipeline::facts::RuntimeFact;
    use crate::pipeline::sym::{SlotClass, SlotRef};
    let Some(bc) = fixture() else {
        return;
    };
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 0,
    };
    // A code whose reciprocal product is not its quotient.
    let code = (1..65535u16)
        .find(|&c| f32::from(c) * (1.0 / 65535.0) != f32::from(c) / 65535.0)
        .expect("a code the reciprocal misses");
    let reciprocal = f32::from(code) * (1.0 / 65535.0);
    let metal = f32::from(code) / 65535.0;
    let mut ran = 0;
    for (stem, p) in profiles() {
        let Some((plan, m)) = module(&bc, "depth_ms", &stem, &p) else {
            continue;
        };
        let metal_sampler = p
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler);
        assert_eq!(
            plan.facts.contains(&RuntimeFact::TexelFormat { slot }),
            !metal_sampler,
            "{stem}"
        );
        if metal_sampler {
            continue;
        }
        for (pf, texel, want) in [
            (PixelFormat::Depth16Unorm, reciprocal, metal),
            (PixelFormat::Depth32Float, reciprocal, reciprocal),
        ] {
            let mut dp = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
            dp.texels[0] = [fl(texel), fl(9.0), fl(9.0), fl(9.0)];
            let ms = Image::new(texture(1), [2, 2, 1], 1, 1, 4);
            let mut res = resources(&plan, &[vec![0; 8]]);
            res.spec
                .insert(SpecKey::TexelFormat(slot), u64::from(texel_format(pf).0));
            let (bufs, _) = run_both(&m, &res, &[dp, ms], [1, 1, 1])
                .unwrap_or_else(|e| panic!("{stem}: {e:?}"));
            assert_eq!(words(&bufs[0])[0], fl(want), "{stem} {pf:?} code {code}");
        }
        ran += 1;
    }
    assert!(ran > 0);
}

/// Without containment of its own, an access outside the image is the interpreter's trap, never a
/// neighbouring texel: a robust profile leaves out-of-range accesses to the device.
#[test]
fn an_uncontained_overrun_traps() {
    let Some(bc) = fixture() else {
        return;
    };
    let mut ran = 0;
    for (stem, p) in profiles() {
        if contains_images(&p) {
            continue;
        }
        let Some((plan, m)) = module(&bc, "read_write_fence", &stem, &p) else {
            continue;
        };
        let t = Image::new(texture(0), [2, 2, 1], 1, 1, 1);
        let res = resources(&plan, &[]);
        let cfg = Config {
            workgroups: [3, 2, 1],
            ..Config::default()
        };
        match run_with_images(&m, &cfg, res, vec![t]) {
            Err(InterpError::Trap(trap)) => {
                assert_eq!(trap.kind, TrapKind::ImageOutOfBounds { image: 0 }, "{stem}")
            }
            other => panic!("{stem}: {other:?}"),
        }
        ran += 1;
    }
    assert!(ran > 0);
}
