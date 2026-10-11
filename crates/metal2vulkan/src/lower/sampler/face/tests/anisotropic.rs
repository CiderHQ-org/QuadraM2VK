//! `emit::cube_footprint` through a constexpr anisotropic sampler, on the interpreter against
//! Metal's face gradients on the Apple M3 Max (the RT13-ag2-probe and RT13-ag3-probe-m3). Each
//! kernel invocation reads a direction's bits and writes the bits of the gradient the device is
//! handed for a constant explicit gradient `g`. Metal reads, on the direction's face, `σ (g[(s + 1)
//! mod 3], g[(t + 1) mod 3])`, with `s` and `t` the axes of the face's `s_c` and `t_c` and `σ` the
//! sign of `s_c`. Vulkan's cube derivative rule must read that from the `Anisotropic` form over `2
//! |m_a|`, and nothing on the normal; and the M3 Max's own reading must take it from what
//! MoltenVK's swizzle (SPIRV-Cross's `spvGradientCube`) makes of the `AnisotropicSwizzled` form.
//! Through a runtime sampler `emit::word_footprint` hands one of those forms or the isotropic one
//! by the sampler's word (`emit::by_anisotropy`).

use super::super::super::emit::{by_anisotropy, cube_footprint};
use super::super::super::CubeFootprint;
use super::{kernel_on, run_records};
use crate::metal::sampler::SamplerWord;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ir::{ConstBits, Constant, Type};

/// Vulkan's face of direction `d` ("Cube map face selection"; no ties among `d`'s magnitudes):
/// its major axis, and the axis and sign of its `s_c` and of its `t_c`. Metal's faces are the
/// same.
fn face_of(d: [f32; 3]) -> (usize, (usize, f32), (usize, f32)) {
    let m = d.map(f32::abs);
    let a = if m[2] >= m[0] && m[2] >= m[1] {
        2
    } else if m[1] >= m[0] {
        1
    } else {
        0
    };
    let positive = d[a] > 0.0;
    let (s, t) = match (a, positive) {
        (0, true) => ((2, -1.0), (1, -1.0)),
        (0, false) => ((2, 1.0), (1, -1.0)),
        (1, true) => ((0, 1.0), (2, 1.0)),
        (1, false) => ((0, 1.0), (2, -1.0)),
        (_, true) => ((0, 1.0), (1, -1.0)),
        (_, false) => ((0, -1.0), (1, -1.0)),
    };
    (a, s, t)
}

/// The face gradients the Apple M3 Max reads from gradient `g` through an anisotropic sampler
/// on the face `(s, t)` names, up to `σ` (RT13-ag2-probe).
fn m3_reading(g: [f32; 3], (s, t): (usize, usize)) -> [f32; 2] {
    [g[(s + 1) % 3], g[(t + 1) % 3]]
}

/// SPIRV-Cross's `spvGradientCube` (MoltenVK's cube-gradient swizzle on an Apple GPU) of
/// gradient `g` at direction `p`: the vector Apple's hardware is handed.
fn swizzled(p: [f32; 3], g: [f32; 3]) -> [f32; 3] {
    let a = p.map(f32::abs);
    let x_major = a[0] >= a[1].max(a[2]);
    let y_major = a[1] >= a[2];
    let permute = |v: [f32; 3]| match (x_major, y_major) {
        (true, _) => [v[1], v[2], v[0]],
        (false, true) => [v[0], v[2], v[1]],
        (false, false) => v,
    };
    let (q, dq) = (permute(p), permute(g));
    let d = [dq[0] - q[0] / q[2] * dq[2], dq[1] - q[1] / q[2] * dq[2]];
    match x_major {
        true => [d[0], d[0], d[1]],
        false => [d[0], d[1], d[0]],
    }
}

/// Directions on each of the six faces, at the face's center and off it.
fn directions() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for a in 0..3 {
        for sign in [1.0f32, -1.0] {
            for (u, v) in [(0.0f32, 0.0f32), (0.25, -0.5), (-0.75, 0.125)] {
                let mut d = [0.0f32; 3];
                d[a] = 2.0 * sign;
                d[(a + 1) % 3] = u;
                d[(a + 2) % 3] = v;
                out.push(d);
            }
        }
    }
    out
}

/// The gradients `cube_footprint` hands the device in `form` for gradient `g`, per direction of
/// `lanes`.
fn handed(form: CubeFootprint, g: [f32; 3], lanes: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let records: Vec<u32> = lanes.iter().flat_map(|d| d.map(f32::to_bits)).collect();
    let m = kernel_on(ModuleBuilder::new(), 3, &|f, p, _| {
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let vec3 = f.module().ty(Type::Vector { elem: f32_, len: 3 })?;
        let mut c = Vec::new();
        for x in g {
            let k = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            c.push(f.module().const_value(k)?);
        }
        let g = f.composite_construct(vec3, &c)?;
        Ok(cube_footprint(f, p, (g, g), form)?.0)
    })
    .expect("kernel");
    run_records(&m, &records, lanes.len())
}

const GRADIENTS: [[f32; 3]; 3] = [[1.0, 2.0, 4.0], [-3.0, 0.5, -7.25], [0.0, -1.5, 6.0]];

#[test]
fn an_anisotropic_footprint_keeps_metals_face_gradients() {
    let lanes = directions();
    for g in GRADIENTS {
        for (d, h) in lanes
            .iter()
            .zip(handed(CubeFootprint::Anisotropic, g, &lanes))
        {
            let (a, (s, s_sign), (t, t_sign)) = face_of(*d);
            assert_eq!(h[a], 0.0, "{d:?} {g:?}: the normal's component {h:?}");
            let read = [s_sign * h[s], t_sign * h[t]];
            let metal = m3_reading(g, (s, t)).map(|x| s_sign * x);
            assert_eq!(read, metal, "{d:?} {g:?}: handed {h:?}");
        }
    }
}

#[test]
fn a_swizzled_anisotropic_footprint_reaches_apples_reading_as_metals() {
    let lanes = directions();
    for g in GRADIENTS {
        for (d, h) in lanes
            .iter()
            .zip(handed(CubeFootprint::AnisotropicSwizzled, g, &lanes))
        {
            let (a, (s, _), (t, _)) = face_of(*d);
            assert_eq!(h[a], 0.0, "{d:?} {g:?}: the normal's component {h:?}");
            let read = m3_reading(swizzled(*d, h), (s, t));
            assert_eq!(read, m3_reading(g, (s, t)), "{d:?} {g:?}: handed {h:?}");
        }
    }
}

/// What `by_anisotropy` hands the device for gradient `g` through a sampler of word `word`, per
/// direction of `lanes`, choosing between the gradients `cube_footprint` hands in form `set` and in
/// form `clear`.
fn by_word(
    (set, clear): (CubeFootprint, CubeFootprint),
    word: u32,
    g: [f32; 3],
    lanes: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    let records: Vec<u32> = lanes
        .iter()
        .flat_map(|d| [d[0].to_bits(), d[1].to_bits(), d[2].to_bits(), word])
        .collect();
    let m = kernel_on(ModuleBuilder::new(), 4, &|f, p, w| {
        let f32_ = f.module().ty(Type::Float { width: 32 })?;
        let vec3 = f.module().ty(Type::Vector { elem: f32_, len: 3 })?;
        let mut c = Vec::new();
        for x in g {
            let k = f.module().constant(Constant::Scalar {
                ty: f32_,
                bits: ConstBits(u64::from(x.to_bits())),
            })?;
            c.push(f.module().const_value(k)?);
        }
        let g = f.composite_construct(vec3, &c)?;
        let a = cube_footprint(f, p, (g, g), set)?;
        let i = cube_footprint(f, p, (g, g), clear)?;
        Ok(by_anisotropy(f, w, a, i)?.0)
    })
    .expect("kernel");
    run_records(&m, &records, lanes.len())
}

/// A runtime sampler's footprint by its word (`word_footprint`) takes, on every lane and bit for
/// bit, the anisotropic arm where the word states `SamplerWord::ANISOTROPIC`, whatever its other
/// fields, and the isotropic arm where it does not. The interpreter does not model the isotropic
/// form's `Length` (an approximate instruction), so the swizzled form stands in for that arm: the
/// choice is `by_anisotropy`'s, between whatever two forms it is given.
#[test]
fn a_footprint_by_the_word_takes_the_form_its_anisotropy_states() {
    let lanes = directions();
    let others = SamplerWord::MIN_LINEAR
        | SamplerWord::MAG_LINEAR
        | (2 << SamplerWord::MIP_FILTER_SHIFT)
        | SamplerWord::LEVEL_MAX;
    let bits =
        |v: &[[f32; 3]]| -> Vec<[u32; 3]> { v.iter().map(|x| x.map(f32::to_bits)).collect() };
    let (set, clear) = (
        CubeFootprint::Anisotropic,
        CubeFootprint::AnisotropicSwizzled,
    );
    for g in GRADIENTS {
        let (when_set, when_clear) = (
            bits(&handed(set, g, &lanes)),
            bits(&handed(clear, g, &lanes)),
        );
        assert_ne!(when_set, when_clear, "{g:?}");
        for (word, want) in [
            (0, &when_clear),
            (others, &when_clear),
            (SamplerWord::ANISOTROPIC, &when_set),
            (SamplerWord::ANISOTROPIC | others, &when_set),
        ] {
            let got = bits(&by_word((set, clear), word, g, &lanes));
            assert_eq!(&got, want, "{g:?} {word:#x}");
        }
    }
}
