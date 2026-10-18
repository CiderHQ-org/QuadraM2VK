//! Metal's linear filter of the packed formats, on the CPU interpreter against Metal's own samples,
//! bit for bit.
//!
//! `packed_samples.tsv` holds 384 of the linear samples our probes took (an Apple M3 Max on macOS
//! 26.7 and an Apple M4 on macOS 27.0.1, byte-identical; random finite texel words, clamp to edge,
//! exact 1/256 fractions): 8 per format, dimension (1D, 2D, 3D), and result (`float`, `half`),
//! chosen to tell the rule from its simpler neighbours (the sum without the terms' truncation,
//! every channel on the 8-bit grid, the exact value rounded once; for a small float the sum rounded
//! to `f32`). Then 48 of our probe's (the Apple M3 Max): `RG11B10Float` words with infinities and
//! NaNs among their channels, 8 per dimension and result: a NaN tap of nonzero weight gives Metal's
//! NaN (`0x7fc00000`, `half` `0x7e00`), else an infinity gives `+inf`, a tap of zero weight
//! neither, and in 3D a tap whose product weight rounds to zero is not of zero weight. The small
//! floats' 3D rows are among those the half-float rule answers (it misses about 0.3% of their 3D
//! channels, the residue). Each row is a sample through a constexpr sampler and through a runtime
//! one, on every device value a packed channel's code reads as: exact, lavapipe's product by the
//! reciprocal, and ANV's and NVIDIA's 5- and 6-bit codes replicated to 8 bits (a small float's
//! value is exact, its NaN's payload kept).

use super::*;

/// Metal's samples, one per line: the raw format (40 `B5G6R5Unorm`, 41 `A1BGR5Unorm`, 42
/// `ABGR4Unorm`, 43 `BGR5A1Unorm`, 90 `RGB10A2Unorm`, 92 `RG11B10Float`, 93 `RGB9E5Float`, 94
/// `BGR10A2Unorm`), the dimension (1, 2, or 3), the result (`f` `float`, `h` `half`), the
/// fractions `kx`, `ky`, and `kz` in 1/256 steps, the taps' texel words (hex; 1D `i`, `i + 1`; 2D
/// `(i, j)`, `(i + 1, j)`, `(i, j + 1)`, `(i + 1, j + 1)`; 3D those of plane `l`, then of `l +
/// 1`), and the result's lanes as bits (hex).
const SAMPLES: &str = include_str!("packed_samples.tsv");

/// The file's rows in runs of `ROWS`, each of one format, dimension, and result: a texture
/// `2 · ROWS` texels wide holds a run's taps.
const ROWS: usize = 8;

/// The width of each sample's texture.
const WIDTH: u32 = 2 * ROWS as u32;

/// One of Metal's samples.
struct Sample {
    raw: u32,
    dims: u32,
    half: bool,
    k: [u32; 3],
    words: Vec<u32>,
    out: [u32; 4],
}

fn samples() -> Vec<Sample> {
    let hex = |s: &str| -> Vec<u32> {
        s.split(',')
            .map(|x| u32::from_str_radix(x, 16).expect("hex"))
            .collect()
    };
    SAMPLES
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            let num = |i: usize| -> u32 { f[i].parse().expect("number") };
            let out = hex(f[7]);
            Sample {
                raw: num(0),
                dims: num(1),
                half: f[2] == "h",
                k: [num(3), num(4), num(5)],
                words: hex(f[6]),
                out: [out[0], out[1], out[2], out[3]],
            }
        })
        .collect()
}

/// The pixel format of raw format `raw`, and its channels' fields `(lsb, bits)` red to alpha (a
/// missing alpha reads 1); `None` fields for a small float.
fn format(raw: u32) -> (PixelFormat, Option<[Option<(u32, u32)>; 4]>) {
    use PixelFormat as P;
    let f = |r, g, b, a| Some([Some(r), Some(g), Some(b), a]);
    match raw {
        40 => (P::B5G6R5Unorm, f((11, 5), (5, 6), (0, 5), None)),
        41 => (P::A1BGR5Unorm, f((11, 5), (6, 5), (1, 5), Some((0, 1)))),
        42 => (P::ABGR4Unorm, f((12, 4), (8, 4), (4, 4), Some((0, 4)))),
        43 => (P::BGR5A1Unorm, f((10, 5), (5, 5), (0, 5), Some((15, 1)))),
        90 => (
            P::RGB10A2Unorm,
            f((0, 10), (10, 10), (20, 10), Some((30, 2))),
        ),
        94 => (
            P::BGR10A2Unorm,
            f((20, 10), (10, 10), (0, 10), Some((30, 2))),
        ),
        92 => (P::RG11B10Float, None),
        93 => (P::RGB9E5Float, None),
        _ => panic!("raw format {raw}"),
    }
}

/// How a device reads a packed channel's code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decode {
    /// `c / S` correctly rounded.
    Exact,
    /// lavapipe's `c · RN(1 / S)`.
    Reciprocal,
    /// ANV's and NVIDIA's: a 4-, 5-, or 6-bit code's bits replicated to 8, over 255.
    Replicated,
}

/// The small float `bits` of `mantissa` bits (and a 5-bit exponent above them), exactly: an
/// exponent of all ones an infinity, or a NaN keeping its payload.
fn small_float(bits: u32, mantissa: u32) -> f32 {
    let (e, m) = (bits >> mantissa, bits & ((1 << mantissa) - 1));
    if e == 0x1f {
        return f32::from_bits(0x7f80_0000 | m << (23 - mantissa));
    }
    let m = f64::from(m) / f64::from(1u32 << mantissa);
    (match e {
        0 => m * 2f64.powi(-14),
        e => (1.0 + m) * 2f64.powi(e as i32 - 15),
    }) as f32
}

/// The device's texel of word `word` in raw format `raw`, read as `decode` reads its codes.
fn texel(raw: u32, word: u32, decode: Decode) -> [u32; 4] {
    let lanes: [f32; 4] = match (raw, format(raw).1) {
        (92, _) => [
            small_float(word & 0x7ff, 6),
            small_float(word >> 11 & 0x7ff, 6),
            small_float(word >> 22, 5),
            1.0,
        ],
        (93, _) => {
            let e = (word >> 27) as i32 - 24;
            std::array::from_fn(|c| match c {
                3 => 1.0,
                c => (f64::from(word >> (9 * c) & 0x1ff) * 2f64.powi(e)) as f32,
            })
        }
        (_, Some(fields)) => fields.map(|field| {
            let Some((lsb, b)) = field else {
                return 1.0;
            };
            let (c, s) = (word >> lsb & ((1 << b) - 1), (1u32 << b) - 1);
            match decode {
                Decode::Exact => (f64::from(c) / f64::from(s)) as f32,
                Decode::Reciprocal => c as f32 * (1.0 / s as f32),
                Decode::Replicated if (4..=6).contains(&b) => {
                    let r = (c << (8 - b)) | (c >> (2 * b - 8));
                    r as f32 / 255.0
                }
                Decode::Replicated => (f64::from(c) / f64::from(s)) as f32,
            }
        }),
        _ => unreachable!("raw format {raw}"),
    };
    lanes.map(f32::to_bits)
}

/// Each row's taps in texture coordinates: row `k`'s at `x` `2k` and `2k + 1`, rows and planes 0
/// and 1, in the file's order.
fn sites(dims: u32, k: u32) -> Vec<[u32; 3]> {
    let x = 2 * k;
    match dims {
        1 => vec![[x, 0, 0], [x + 1, 0, 0]],
        2 => vec![[x, 0, 0], [x + 1, 0, 0], [x, 1, 0], [x + 1, 1, 0]],
        _ => (0..8).map(|t| [x + (t & 1), t >> 1 & 1, t >> 2]).collect(),
    }
}

/// Each of Metal's samples of each packed format, through a constexpr linear clamp-to-edge
/// sampler and a runtime sampler of that state, in 1D, 2D, and 3D, with `float` and `half`
/// results, is Metal's bytes, whatever the device reads a packed code as (a small float's value
/// is exact on every device). The emulation is reached.
#[test]
fn packed_linear_samples_are_metals_bytes() {
    let p = lavapipe();
    let all = samples();
    assert_eq!(all.len(), 432);
    let runs: Vec<&[Sample]> = all.chunks(ROWS).collect();
    for run in &runs {
        let s = &run[0];
        assert!(
            run.iter()
                .all(|r| (r.raw, r.dims, r.half) == (s.raw, s.dims, s.half)),
            "a run of one format, dimension, and result"
        );
    }
    let word = runtime_word(
        SamplerAddressMode::ClampToEdge,
        false,
        SamplerBorderColor::TransparentBlack,
    );
    for dims in 1..=3u32 {
        let (form, routine) = match dims {
            1 => (Form::OneD, "sampler.linear.span"),
            2 => (Form::Plain { offset: (0, 0) }, "sampler.linear.quad"),
            _ => (Form::ThreeD { offset: (0, 0, 0) }, "sampler.linear.block"),
        };
        for half in [false, true] {
            for runtime in [false, true] {
                let sampler = (!runtime).then(|| linear(1, false, 0));
                let name = format!("{dims}D half={half} runtime={runtime}");
                let Some(bc) = assemble(&name, &kernel_of(sampler, form, !half)) else {
                    return;
                };
                let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert!(a.strategies.entries.contains_key(routine), "{name}");
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
                let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
                let mine = runs
                    .iter()
                    .filter(|r| r[0].dims == dims && r[0].half == half);
                for rows in mine {
                    let rows: &[Sample] = rows;
                    let raw = rows[0].raw;
                    let pf = format(raw).0;
                    let class = texel_format(pf).0;
                    let coords: Vec<u8> = (0u32..)
                        .zip(rows)
                        .flat_map(|(k, s)| {
                            let u = (512 * k + 128 + s.k[0]) as f32 / (256 * WIDTH) as f32;
                            let v = (128 + s.k[1]) as f32 / 512.0;
                            let w = (128 + s.k[2]) as f32 / 512.0;
                            match dims {
                                3 => vec![u, v, w, 0.0],
                                _ => vec![u, v],
                            }
                        })
                        .flat_map(f32::to_le_bytes)
                        .collect();
                    let size = if half { 8 } else { 16 };
                    let res = resources(&plan, &[coords, vec![0xab; size * ROWS]], class);
                    let res = with_word(res, &a, runtime.then_some(word));
                    for decode in [Decode::Exact, Decode::Reciprocal, Decode::Replicated] {
                        if format(raw).1.is_none() && decode != Decode::Exact {
                            continue;
                        }
                        let extent = match dims {
                            1 => [WIDTH, 1, 1],
                            2 => [WIDTH, 2, 1],
                            _ => [WIDTH, 2, 2],
                        };
                        let mut image = Image::new(combined_slot(&m), extent, 1, 1, 1);
                        for (k, s) in (0u32..).zip(rows) {
                            for (at, w) in sites(dims, k).into_iter().zip(&s.words) {
                                let i = image.index(0, 0, 0, at).expect("texel");
                                image.texels[i] = texel(raw, *w, decode);
                            }
                        }
                        let got = run(&m, &res, &image, ROWS as u32, half);
                        for (s, g) in rows.iter().zip(got) {
                            assert_eq!(
                                g, s.out,
                                "{name} {pf:?} {decode:?} k {:?} words {:x?}: {g:x?}, Metal {:x?}",
                                s.k, s.words, s.out
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Metal's 2D samples with one axis's two taps clamped onto one texel: `edge_samples.tsv`, 94 of
/// the linear samples our probes took (an Apple M3 Max on macOS 26.7 and an Apple M4 on macOS
/// 27.0.1, byte-identical; random finite texels of 64 x 64 textures past
/// either edge, clamp to edge, exact 1/256 fractions): per format (`RGBA16Float`, `RGBA32Float`,
/// `RG11B10Float`, `RGB9E5Float`), result, and clamped axis, the rows whose answer the unmerged
/// fractions change, and `RGBA32Float`'s `half` results that are a negative zero. One per line: the
/// raw format (115, 125, 92, 93), the clamped axis (`x` or `y`), the result (`f` `float`, `h`
/// `half`), the fractions `kx` and `ky` in 1/256 steps, the two distinct taps' texel words (hex,
/// the lower one first along the other axis), and the result's lanes as bits (hex).
const EDGES: &str = include_str!("edge_samples.tsv");

/// One of Metal's edge samples.
struct Edge {
    raw: u32,
    x: bool,
    half: bool,
    k: [u32; 2],
    taps: [Vec<u32>; 2],
    out: [u32; 4],
}

fn edges() -> Vec<Edge> {
    let hex = |s: &str| -> Vec<u32> {
        s.split(',')
            .map(|x| u32::from_str_radix(x, 16).expect("hex"))
            .collect()
    };
    EDGES
        .lines()
        .map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            let num = |i: usize| -> u32 { f[i].parse().expect("number") };
            let out = hex(f[7]);
            Edge {
                raw: num(0),
                x: f[1] == "x",
                half: f[2] == "h",
                k: [num(3), num(4)],
                taps: [hex(f[5]), hex(f[6])],
                out: [out[0], out[1], out[2], out[3]],
            }
        })
        .collect()
}

/// The device's texel (`f32` bits per channel) of texel words `words` in raw format `raw`: each
/// value exact.
fn edge_texel(raw: u32, words: &[u32]) -> [u32; 4] {
    let half = |h: u32| half_value(h as u16).to_bits();
    match (raw, words) {
        (115, [a, b]) => [
            half(a & 0xffff),
            half(a >> 16),
            half(b & 0xffff),
            half(b >> 16),
        ],
        (125, [a, b, c, d]) => [*a, *b, *c, *d],
        (92 | 93, [w]) => texel(raw, *w, Decode::Exact),
        _ => panic!("raw format {raw}: {words:x?}"),
    }
}

/// Each of Metal's edge samples, through a constexpr linear clamp-to-edge sampler and a runtime
/// sampler of that state, with `float` and `half` results, is Metal's bytes: the emulation's 2D
/// float sum weighs a clamped axis's taps `(256, 0)`. A run of one format, axis, and result reads
/// a texture one texel across its clamped axis, whose two taps clamp onto that texel, and two
/// texels per sample along the other.
#[test]
fn clamped_edges_of_float_formats_are_metals_bytes() {
    use crate::metal::texel_format::texel_format as class_of;
    let p = lavapipe();
    let all = edges();
    assert_eq!(all.len(), 94);
    let word = runtime_word(
        SamplerAddressMode::ClampToEdge,
        false,
        SamplerBorderColor::TransparentBlack,
    );
    let mut runs: Vec<Vec<&Edge>> = Vec::new();
    for e in &all {
        match runs.last_mut() {
            Some(r) if (r[0].raw, r[0].x, r[0].half) == (e.raw, e.x, e.half) => r.push(e),
            _ => runs.push(vec![e]),
        }
    }
    for half in [false, true] {
        for runtime in [false, true] {
            let sampler = (!runtime).then(|| linear(1, false, 0));
            let name = format!("edges half={half} runtime={runtime}");
            let form = Form::Plain { offset: (0, 0) };
            let Some(bc) = assemble(&name, &kernel_of(sampler, form, !half)) else {
                return;
            };
            let a = compile(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                a.strategies.entries.contains_key("sampler.linear.quad"),
                "{name}"
            );
            let (plan, m) = typed_module(&bc, "k", &p).expect("typed module");
            for rows in runs.iter().filter(|r| r[0].half == half) {
                let (raw, x) = (rows[0].raw, rows[0].x);
                let pf = match raw {
                    115 => PixelFormat::RGBA16Float,
                    125 => PixelFormat::RGBA32Float,
                    _ => format(raw).0,
                };
                let n = rows.len() as u32;
                // The clamped axis is one texel wide; the other holds two per sample.
                let extent = match x {
                    true => [1, 2 * n, 1],
                    false => [2 * n, 1, 1],
                };
                let coords: Vec<u8> = (0u32..)
                    .zip(rows)
                    .flat_map(|(r, e)| {
                        let along = |k: u32| (512 * r + 128 + k) as f32 / (512 * n) as f32;
                        let across = |k: u32| (128 + k) as f32 / 256.0;
                        match x {
                            true => [across(e.k[0]), along(e.k[1])],
                            false => [along(e.k[0]), across(e.k[1])],
                        }
                    })
                    .flat_map(f32::to_le_bytes)
                    .collect();
                let size = if half { 8 } else { 16 };
                let res = resources(
                    &plan,
                    &[coords, vec![0xab; size * rows.len()]],
                    class_of(pf).0,
                );
                let res = with_word(res, &a, runtime.then_some(word));
                let mut image = Image::new(combined_slot(&m), extent, 1, 1, 1);
                for (r, e) in (0u32..).zip(rows) {
                    for (t, words) in (0u32..).zip(&e.taps) {
                        let at = match x {
                            true => [0, 2 * r + t, 0],
                            false => [2 * r + t, 0, 0],
                        };
                        let i = image.index(0, 0, 0, at).expect("texel");
                        image.texels[i] = edge_texel(raw, words);
                    }
                }
                let got = run(&m, &res, &image, n, half);
                for (e, g) in rows.iter().zip(got) {
                    assert_eq!(
                        g, e.out,
                        "{name} {pf:?} x={x} k {:?} taps {:x?}: {g:x?}, Metal {:x?}",
                        e.k, e.taps, e.out
                    );
                }
            }
        }
    }
}

/// How a [`swizzle_kernel`] accesses its `texture2d` slot 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Access {
    /// `t.read(uint2(tid % 16, tid / 16))` of an `access::read` texture.
    Read,
    /// `t.sample(s, coord[tid])` through a constexpr clamp-to-edge sampler, linear or nearest.
    Sample { linear: bool },
    /// `t.gather(s, coord[tid], int2(0), component)` through a constexpr clamp-to-edge sampler.
    Gather { component: u32 },
}

/// A kernel accessing texture slot 0 as `access` (at `float2` buffer 0's lane `tid` for a
/// sample or gather), storing the `half4` (`float4` unless `half`) to buffer 1.
fn swizzle_kernel(access: Access, half: bool) -> String {
    let (texel, vec4, size, suffix) = match half {
        true => ("half", "<4 x half>", 8, "v4f16"),
        false => ("float", "<4 x float>", 16, "v4f32"),
    };
    let state = |words: u64| {
        format!(
            "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8\n\n"
        )
    };
    let (state, tex_access, call, decl) = match access {
        Access::Read => (
            String::new(),
            "read",
            format!(
                "%rs = call ptr addrspace(2) @air.get_read_sampler()\n  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)"
            ),
            format!(
                "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)"
            ),
        ),
        Access::Sample { linear } => (
            state(sampler_words(u64::from(linear), 0, 1, false, 0)),
            "sample",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"
            ),
        ),
        Access::Gather { component } => (
            state(sampler_words(0, 0, 1, false, 0)),
            "sample",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 {component}, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"
            ),
        ),
    };
    let (states, state_md) = match access {
        Access::Read => ("", ""),
        _ => (
            "!air.sampler_states = !{!7}\n",
            "!7 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
    };
    let out_ty = format!("{texel}4");
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %c, ptr addrspace(1) %out, i32 %tid) {{
  %i = zext i32 %tid to i64
  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i
  %uv = load <2 x float>, ptr addrspace(1) %p, align 8
  %x = and i32 %tid, 15
  %y = lshr i32 %tid, 4
  %g0 = insertelement <2 x i32> zeroinitializer, i32 %x, i64 0
  %gid = insertelement <2 x i32> %g0, i32 %y, i64 1
  {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

{decl}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{tex_access}", !"air.arg_type_name", !"texture2d<{texel}, {tex_access}>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"c"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
!6 = !{{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}}
{state_md}"#
    )
}

/// The images of texture slot 0 module `m` binds (its sampled image, or a combined image
/// sampler), each `extent` texels holding `texels`.
fn slot_images(m: &TypedModule, extent: [u32; 3], texels: &[[u32; 4]]) -> Vec<Image> {
    let images: Vec<Image> = m
        .globals()
        .iter()
        .filter_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::Texture | ResourceKind::CombinedTextureSampler(_),
                    index: 0,
                    ..
                },
            ) => {
                let mut image = Image::new(s, extent, 1, 1, 1);
                image.texels = texels.to_vec();
                Some(image)
            }
            _ => None,
        })
        .collect();
    assert!(!images.is_empty(), "no image of texture slot 0");
    images
}

/// A swizzled packed normalized texture's reads and linear samples, with `float` and `half`
/// results, are the unswizzled texture's, swizzled, on a device that does not sample through
/// Metal's own sampler, whatever it reads a packed code as (exact, lavapipe's product by the
/// reciprocal, ANV's and NVIDIA's replicated bits): under the slot's `swizzled_storage` variant
/// the function converts the unswizzled texel by the format's whole class and swizzles it
/// (`Zero` 0, `One` 1). The reference is the unswizzled module's answer for exact codes, which is
/// Metal's (`packed_linear_samples_are_metals_bytes` and the device lanes). Previously a swizzled
/// view's class dropped the packed field, and the device's replicated codes showed. A nearest
/// sample's and a gather's modules (four gathers, the one the swizzle names taken) validate, and
/// every module's words stay within the selection's estimate; the interpreter models no device
/// sample or gather, so their answers are the device lane's (`runtime_lane::extended_swizzles`).
#[test]
fn swizzled_packed_texels_are_the_unswizzled_swizzled() {
    use crate::lower::select::LowerInputs;
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels as C};
    use crate::pipeline::testing::{compile_with, typed_module_with};
    const THREADS: u32 = 2 * WIDTH;
    let p = lavapipe();
    let swizzles = [
        C::new(S::Green, S::Blue, S::Alpha, S::Red),
        C::new(S::Blue, S::Green, S::Red, S::Alpha),
        C::new(S::Alpha, S::Red, S::One, S::Zero),
        C::new(S::Zero, S::Alpha, S::Green, S::One),
        C::new(S::Green, S::Green, S::Green, S::Green),
    ];
    let mut rng = Lcg(0xd793);
    // Positions inside the texture and past its edges (clamped), at 1/256 steps.
    let coords: Vec<u8> = (0..THREADS)
        .flat_map(|_| {
            let u = (rng.next() % (256 * u64::from(WIDTH) + 256)) as f32 - 128.0;
            let v = (rng.next() % (256 * 2 + 256)) as f32 - 128.0;
            [u / (256 * WIDTH) as f32, v / 512.0]
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut accesses = vec![
        Access::Read,
        Access::Sample { linear: false },
        Access::Sample { linear: true },
    ];
    accesses.extend((0..4).map(|component| Access::Gather { component }));
    // The accesses the interpreter runs: a read, and a linear sample, which fetches its taps.
    let runs = |access: &Access| matches!(access, Access::Read | Access::Sample { linear: true });
    let mut swizzled_inputs = LowerInputs::new();
    swizzled_inputs.swizzled_storage.insert(TEXTURE);
    for half in [false, true] {
        let one = if half { 0x3c00 } else { 1.0f32.to_bits() };
        // Each access's module, unswizzled and swizzled.
        let mut modules = Vec::new();
        for access in &accesses {
            let name = format!("{access:?} half={half}");
            let Some(bc) = assemble(&name, &swizzle_kernel(*access, half)) else {
                return;
            };
            let (plain, m) = typed_module(&bc, "k", &p).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (plan, s) = typed_module_with(&bc, "k", &p, &swizzled_inputs)
                .unwrap_or_else(|e| panic!("{name} swizzled: {e}"));
            let a = compile_with(&bc, "k", &p, &swizzled_inputs)
                .unwrap_or_else(|e| panic!("{name} swizzled: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{name} swizzled: {e}"));
            let words = a.module.words.len() as u64;
            assert!(
                plan.estimate.words >= words,
                "{name}: estimate {} < emitted {words}",
                plan.estimate.words
            );
            modules.push((name, (plain, m), (plan, s)));
        }
        for raw in [40, 41, 42, 43] {
            let (pf, _) = format(raw);
            let class = texel_format(pf).0;
            let words: Vec<u32> = (0..THREADS).map(|_| rng.next() as u32 & 0xffff).collect();
            let texels = |decode: Decode| -> Vec<[u32; 4]> {
                words.iter().map(|w| texel(raw, *w, decode)).collect()
            };
            let extent = [WIDTH, 2, 1];
            let size = if half { 8 } else { 16 };
            let init = [coords.clone(), vec![0xab; size * THREADS as usize]];
            // The unswizzled answers of exact codes: each access's the interpreter runs.
            let reference: Vec<Vec<[u32; 4]>> = accesses
                .iter()
                .zip(&modules)
                .map(|(access, (_, (plan, m), _))| {
                    if !runs(access) {
                        return Vec::new();
                    }
                    let res = resources(plan, &init, class);
                    let images = slot_images(m, extent, &texels(Decode::Exact));
                    run_images(m, &res, &images, THREADS, half)
                })
                .collect();
            for swizzle in swizzles {
                let sources = [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha];
                // A lane's source: 0, 1, or channel `c` of the unswizzled texel.
                let pick = |source: S, x: [u32; 4]| match source {
                    S::Zero => 0,
                    S::One => one,
                    S::Red => x[0],
                    S::Green => x[1],
                    S::Blue => x[2],
                    _ => x[3],
                };
                for (k, (access, (name, _, (plan, m)))) in accesses.iter().zip(&modules).enumerate()
                {
                    if !runs(access) {
                        continue;
                    }
                    let mut res = resources(plan, &init, class);
                    res.spec
                        .insert(SpecKey::TextureSwizzle(TEXTURE), u64::from(swizzle.word()));
                    let expected: Vec<[u32; 4]> = reference[k]
                        .iter()
                        .map(|x| sources.map(|s| pick(s, *x)))
                        .collect();
                    for decode in [Decode::Exact, Decode::Reciprocal, Decode::Replicated] {
                        let images = slot_images(m, extent, &texels(decode));
                        let got = run_images(m, &res, &images, THREADS, half);
                        for (t, (g, e)) in got.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                g, e,
                                "{name} {pf:?} {swizzle:?} {decode:?} thread {t} word {:#06x}: \
                                 {g:x?}, expected {e:x?}",
                                words[t]
                            );
                        }
                    }
                }
            }
        }
    }
}
