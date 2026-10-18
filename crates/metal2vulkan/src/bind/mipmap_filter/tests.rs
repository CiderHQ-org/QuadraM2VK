use std::collections::HashMap;

use super::fetch::{components, fits, Recover};
use super::kernel::{narrow, typed_module};
use super::layout::{axis_tap, grid_tie_up, layout, mode, record, rule, Rule};
use super::reciprocal::Reciprocal;
use super::{
    model, params, FetchView, KernelStep, MipmapFilterChain, MipmapFilterKernel,
    MipmapFilterModule, MipmapRule, Step, DESTINATION, KEPT_IN, KEPT_OUT, PARAMS, SOURCE,
    SOURCE_IMAGE,
};
use crate::metal::format::candidate_formats;
use crate::metal::PixelFormat;
use crate::spirv::interp::{run, run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::{check_valid, committed_profiles};
use crate::vk::Format;

/// Every chain the M4 measured that the tests replay (the file's header states them).
const MEASURED: &str = include_str!("measured.txt");
/// Every chain the M3 Max measured that the tests replay (the file's header states them).
const MEASURED_M3: &str = include_str!("measured_m3.txt");
/// The digests of every level of the M3 Max's chains past its tail's extent (the file's header
/// states them).
const MEASURED_M3_PASSES: &str = include_str!("measured_m3_passes.txt");
/// The digests of every level of share-mips-m3b's chains of 8-bit unorm and 32-bit float
/// formats, by the texture's store (the file's header states them).
const MEASURED_M3_LAYOUTS: &str = include_str!("measured_m3_layouts.txt");

/// The sampled views through which a fetch reads the first level of `pf`: one per candidate
/// Vulkan format of its image whose fetch recovers the texel (`MipmapFilterChain::fetch_format`).
fn fetch_views(pf: PixelFormat) -> Vec<Format> {
    let Some(l) = layout(pf) else {
        return Vec::new();
    };
    let mut views: Vec<Format> = candidate_formats(pf)
        .into_iter()
        .filter_map(MipmapFilterChain::fetch_format)
        .filter(|&view| components(view).is_some_and(|c| fits(&l, &c)))
        .collect();
    views.dedup();
    views
}

/// Every kernel a chain may dispatch: each format's sampler levels (their half converted where
/// the format has one, keeping or not, with the depth taps or not) and tail levels (reading
/// codes or kept values, keeping or not); and, for each view a fetch reads its first level
/// through, the first later level's sampler level fetching from it (keeping or not, from a 2D
/// array, a 3D texture one texel deep, or a deeper one).
pub(crate) fn every_kernel() -> Vec<MipmapFilterKernel> {
    let mut out = Vec::new();
    for &pixel_format in PixelFormat::ALL {
        let Some(l) = layout(pixel_format) else {
            continue;
        };
        let mut steps = vec![
            KernelStep::Sampler { half: false },
            KernelStep::Tail { read_kept: false },
            KernelStep::Tail { read_kept: true },
        ];
        if l.half {
            steps.push(KernelStep::Sampler { half: true });
        }
        for step in steps {
            for keep in [false, true] {
                for depth in [false, true] {
                    if depth && matches!(step, KernelStep::Tail { .. }) {
                        continue;
                    }
                    out.push(MipmapFilterKernel {
                        pixel_format,
                        step,
                        keep,
                        depth,
                        fetch: None,
                    });
                }
            }
        }
        for format in fetch_views(pixel_format) {
            for keep in [false, true] {
                for (depth, volume) in [(false, false), (false, true), (true, true)] {
                    out.push(MipmapFilterKernel {
                        pixel_format,
                        step: KernelStep::Sampler { half: l.half },
                        keep,
                        depth,
                        fetch: Some(FetchView { format, volume }),
                    });
                }
            }
        }
    }
    out
}

/// Every kernel is valid on every committed profile with `shaderInt64` (the owned verifier, the
/// profile's requirements and `spirv-val`), checked on as many threads as the host has.
#[test]
fn every_filter_kernel_is_valid_on_every_committed_profile_with_int64() {
    let profiles: Vec<_> = committed_profiles()
        .expect("profiles")
        .into_iter()
        .filter(|(_, p)| p.fields().leaf.int64)
        .collect();
    assert!(!profiles.is_empty(), "no committed profile has shaderInt64");
    let kernels = every_kernel();
    let fetching = kernels.iter().filter(|k| k.fetches()).count();
    assert_eq!(
        kernels.len() - fetching,
        304,
        "the kernels of the 35 formats"
    );
    assert_eq!(
        fetching,
        6 * 39,
        "the fetching kernels of the 39 views of 30 formats"
    );
    let modules: Vec<(MipmapFilterKernel, TypedModule)> = kernels
        .iter()
        .map(|k| {
            (
                *k,
                typed_module(&k.spec().expect("a spec")).expect("module"),
            )
        })
        .collect();
    let jobs: Vec<(usize, usize)> = (0..modules.len())
        .flat_map(|m| (0..profiles.len()).map(move |p| (m, p)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let failures: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .chunks(jobs.len().div_ceil(threads))
            .map(|chunk| {
                let (modules, profiles) = (&modules, &profiles);
                scope.spawn(move || {
                    let mut failed = Vec::new();
                    for &(m, p) in chunk {
                        let (kernel, module) = &modules[m];
                        let (name, profile) = &profiles[p];
                        if let Err(e) = check_valid(module.clone(), profile) {
                            failed.push(format!("{kernel:?} on {name}: {e}"));
                        }
                    }
                    failed
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a checker thread"))
            .collect()
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every integer field sums in 32 bits at a 2D level (`kernel::narrow`): in `u32`s the unsigned
/// 16-bit codes' terms, in `i32`s the rest.
#[test]
fn every_integer_field_sums_in_32_bits_in_2d() {
    let mut fields = 0;
    for kernel in every_kernel() {
        let spec = kernel.spec().expect("a spec");
        for r in &spec.records {
            if r[record::MODE as usize] != mode::INTEGER {
                continue;
            }
            fields += 1;
            let unsigned16 =
                r[record::RULE as usize] == rule::FIXED16 && r[record::IN_LO as usize] == 0;
            assert_eq!(
                narrow(r),
                Some(!unsigned16),
                "{kernel:?} {:?}",
                r[record::RULE as usize]
            );
        }
    }
    assert!(fields > 0);
}

/// One dispatch's buffers: the level's words and workgroups, the level above's bytes, the
/// level's bytes, the level above's kept words where the level reads them, and the bytes of
/// the level's own where it keeps them.
struct Dispatch<'a> {
    params: &'a [u32],
    groups: [u32; 2],
    above: &'a [u8],
    above_bytes: u64,
    bytes: u64,
    kept_in: Option<&'a [u8]>,
    kept_out: u64,
    /// Whether the destination is zeroed first ([`MipmapFilterChain::zeroed`]); else it holds
    /// the sentinel 0xa5, which the dispatch must overwrite.
    zeroed: bool,
    /// The first level's sampled view, where the kernel fetches it (`above` is then unread).
    first: Option<&'a Image>,
}

/// A dispatch run in the interpreter, both invocation orders, over a destination zeroed or not
/// (`zeroed`) and 8 sentinel bytes past its words, which it must leave, and a kept-out buffer
/// likewise: the
/// level's bytes and the words it keeps. A buffer the level does not use is 4 bytes, which the
/// interpreter refuses to read or write past.
fn dispatch(m: &TypedModule, d: &Dispatch<'_>) -> (Vec<u8>, Vec<u8>) {
    let mut source = d.above.to_vec();
    source.resize(d.above_bytes.next_multiple_of(4) as usize, 0);
    let words = d.bytes.next_multiple_of(4) as usize;
    let mut destination = vec![if d.zeroed { 0 } else { 0xa5 }; words];
    destination.extend([0xee; 8]);
    let kept_in = d.kept_in.map_or_else(|| vec![0u8; 4], <[u8]>::to_vec);
    let mut kept_out = vec![0u8; d.kept_out.max(4) as usize];
    kept_out.extend([0xee; 8]);
    let mut results = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let part = |slot, bytes: Vec<u8>| Buffer {
            slot: Some(slot),
            address: None,
            bytes,
        };
        let res = Resources {
            buffers: vec![
                part(SOURCE, source.clone()),
                part(DESTINATION, destination.clone()),
                part(
                    PARAMS,
                    d.params.iter().flat_map(|w| w.to_le_bytes()).collect(),
                ),
                part(KEPT_IN, kept_in.clone()),
                part(KEPT_OUT, kept_out.clone()),
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: [d.groups[0], d.groups[1], 1],
            step_limit: 1 << 30,
            ..Config::default()
        };
        let buffers = match d.first {
            Some(image) => {
                let (outcome, images) =
                    run_with_images(m, &cfg, res, vec![image.clone()]).expect("run");
                assert_eq!(
                    images[0].texels, image.texels,
                    "the first level is only read"
                );
                outcome.buffers
            }
            None => run(m, &cfg, res).expect("run").buffers,
        };
        assert_eq!(buffers[0], source, "the level above is only read");
        assert_eq!(buffers[3], kept_in, "the kept values above are only read");
        let written = &buffers[1];
        assert_eq!(
            written[words..],
            [0xee; 8],
            "nothing past the level's words"
        );
        assert!(
            written[d.bytes as usize..words].iter().all(|&b| b == 0),
            "the last word's unused bytes stay 0"
        );
        let kept = &buffers[4];
        let end = kept.len() - 8;
        assert_eq!(kept[end..], [0xee; 8], "nothing past the kept words");
        results.push((
            written[..d.bytes as usize].to_vec(),
            kept[..d.kept_out as usize].to_vec(),
        ));
    }
    assert_eq!(results[0], results[1], "order independent");
    results.swap_remove(0)
}

/// The modules of the kernels the tests have run, each built once.
#[derive(Default)]
struct Modules(HashMap<MipmapFilterKernel, TypedModule>);

impl Modules {
    fn of(&mut self, kernel: MipmapFilterKernel) -> &TypedModule {
        self.0
            .entry(kernel)
            .or_insert_with(|| typed_module(&kernel.spec().expect("a spec")).expect("module"))
    }
}

/// Every level of `chain` run in the interpreter from `first`, each by its kernel from the level
/// above and the values it kept: the levels, the first included. Where the chain fetches its
/// first level ([`MipmapFilterChain::fetch`]), the first later level reads it from its view, each
/// texel what [`fetched_texel`] says a device returns.
fn run_chain(modules: &mut Modules, chain: &MipmapFilterChain, first: &[u8]) -> Vec<Vec<u8>> {
    let mut levels = vec![first.to_vec()];
    let mut kept: Vec<u8> = Vec::new();
    let view = chain.fetch.map(|format| first_view(chain, format, first));
    for level in 1..chain.levels.len() {
        let (a, l) = (&chain.levels[level - 1], &chain.levels[level]);
        let (_, words) = l.params.as_ref().expect("params");
        let m = modules.of(l.kernel.expect("a kernel"));
        let (bytes, keep) = dispatch(
            m,
            &Dispatch {
                params: words,
                groups: l.groups,
                above: &levels[level - 1],
                above_bytes: a.bytes,
                bytes: l.bytes,
                kept_in: a.kept.map(|_| &kept[..]),
                kept_out: l.kept.map_or(0, |(_, b)| b),
                zeroed: chain.zeroed,
                first: view.as_ref().filter(|_| level == 1),
            },
        );
        levels.push(bytes);
        kept = keep;
    }
    levels
}

/// What a device's fetch of a sampled view of `view` returns for a texel of `bytes`, as the
/// interpreter holds it (four 32-bit components): a unorm code `c` of `b` bits as `c / (2^b -
/// 1)` and a snorm one as `max(c / (2^(b-1) - 1), -1)`, each the `f32` nearest it; a half as its
/// own value; an integer component's bits.
fn fetched_texel(view: Format, bytes: &[u8]) -> [u32; 4] {
    let mut word = [0u8; 8];
    word[..bytes.len()].copy_from_slice(bytes);
    let all = u64::from_le_bytes(word);
    let mut out = [0, 0, 0, 1f32.to_bits()];
    for (lane, c) in components(view).expect("a fetched view").iter().enumerate() {
        let Some(c) = c else {
            continue;
        };
        let code = (all >> c.offset) & ((1u64 << c.bits) - 1);
        out[lane] = match c.recover {
            Recover::Unorm => (code as f32 / ((1u64 << c.bits) - 1) as f32).to_bits(),
            Recover::Snorm => {
                let signed = ((code << (64 - c.bits)) as i64 >> (64 - c.bits)) as f32;
                (signed / ((1u64 << (c.bits - 1)) - 1) as f32)
                    .max(-1.0)
                    .to_bits()
            }
            Recover::Half => half_value(code as u16),
            Recover::Bits => code as u32,
        };
    }
    out
}

/// The `f32` bits of the value of half-float bits `h` (a NaN keeps its payload).
fn half_value(h: u16) -> u32 {
    let sign = u32::from(h >> 15) << 31;
    let (e, m) = (u32::from((h >> 10) & 0x1f), u32::from(h & 0x3ff));
    match e {
        0 => sign | (m as f32 * 2f32.powi(-24)).to_bits(),
        31 => sign | 0x7f80_0000 | (m << 13),
        _ => sign | ((e + 112) << 23) | (m << 13),
    }
}

/// The first level of `chain`, `first` (every layer's texels), as its sampled view of `view`
/// holds it: a 2D array of every layer, or a 3D image.
fn first_view(chain: &MipmapFilterChain, view: Format, first: &[u8]) -> Image {
    let l = &chain.levels[0];
    let volume = chain.volume;
    let (extent, layers) = if volume {
        (l.extent, 1)
    } else {
        ([l.extent[0], l.extent[1], 1], chain.layers)
    };
    let mut image = Image::new(SOURCE_IMAGE, extent, layers, 1, 1);
    let bytes = layout(chain.pixel_format).expect("a layout").bytes as usize;
    for (texel, b) in image.texels.iter_mut().zip(first.chunks(bytes)) {
        *texel = fetched_texel(view, b);
    }
    image
}

/// Whether Metal keeps a texture of `extent` losslessly compressed as the tests' chains make it
/// (`allowGPUOptimizedContents`, a usage of neither `ShaderWrite` nor `PixelFormatView`): where
/// its shorter side is 16 texels or more (`MipmapFilterChain::plan`'s `compressed`).
fn compressed(extent: [u32; 3]) -> bool {
    extent[0].min(extent[1]) >= 16
}

fn plan(pf: PixelFormat, extent: [u32; 3], layers: u32, levels: u32) -> MipmapFilterChain {
    let volume = extent[2] > 1;
    MipmapFilterChain::plan(
        pf,
        extent,
        layers,
        levels,
        volume,
        compressed(extent),
        MipmapRule::AppleM3MaxPasses,
        64,
        u32::MAX,
        [65535, 65535],
        None,
    )
    .unwrap_or_else(|| panic!("a chain of {pf:?} {extent:?}"))
}

/// One measured chain: the format, the first level's extent, the layers, and every level.
struct Measured {
    pf: PixelFormat,
    extent: [u32; 3],
    layers: u32,
    levels: Vec<Vec<u8>>,
}

fn measured(table: &str) -> Vec<Measured> {
    let hex = |s: &str| {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
            .collect::<Vec<u8>>()
    };
    let mut out: Vec<Measured> = Vec::new();
    for line in table.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        match parts.first() {
            None => {}
            Some(p) if p.starts_with('#') => {}
            Some(&"chain") => {
                let pf = PixelFormat::from_name(parts[1]).expect("a format");
                let d: Vec<u32> = parts[3]
                    .split('x')
                    .map(|v| v.parse().expect("a size"))
                    .collect();
                out.push(Measured {
                    pf,
                    extent: [d[0], d[1], d[2]],
                    layers: parts[4].parse().expect("layers"),
                    levels: Vec::new(),
                });
            }
            Some(_) => out.last_mut().expect("a chain").levels.push(hex(parts[1])),
        }
    }
    out
}

/// Every chain the M4 measured, level by level from the level above as it measured it: the
/// M4's filter on the CPU and in the kernel (every level a sampler level of the M4's
/// coordinates, nothing carried through a half) gives its bytes.
#[test]
fn the_filter_gives_the_m4_s_measured_chains() {
    let mut modules = Modules::default();
    let chains = measured(MEASURED);
    assert_eq!(chains.len(), 76);
    let m4 = Step::Sampler {
        reciprocal: Reciprocal::AppleM4,
        half: false,
    };
    for c in chains {
        let chain = plan(c.pf, c.extent, c.layers, c.levels.len() as u32);
        let l = layout(c.pf).expect("a layout");
        for level in 1..c.levels.len() {
            let (a, b) = (&chain.levels[level - 1], &chain.levels[level]);
            let above = &c.levels[level - 1];
            let want = &c.levels[level];
            let what = format!("{:?} {:?} x{} level {level}", c.pf, c.extent, c.layers);
            let reciprocal = Reciprocal::AppleM4;
            let (cpu, _) = model::level(&l, above, a.extent, b.extent, c.layers, reciprocal, false);
            assert_eq!(&cpu, want, "CPU {what}");
            let texels = (b.bytes / u64::from(l.bytes)) as u32;
            let row = b.groups[0] * MipmapFilterModule::LOCAL_SIZE;
            let words = params(&l, a.extent, b.extent, texels, row, m4).expect("words");
            let kernel = MipmapFilterKernel::new(c.pf, &l, m4, false, a.extent, None);
            let (gpu, _) = dispatch(
                modules.of(kernel),
                &Dispatch {
                    params: &words,
                    groups: b.groups,
                    above,
                    above_bytes: a.bytes,
                    bytes: b.bytes,
                    kept_in: None,
                    kept_out: 0,
                    zeroed: chain.zeroed,
                    first: None,
                },
            );
            assert_eq!(&gpu, want, "kernel {what}");
        }
    }
}

/// Every chain the M3 Max measured (2D chains of first levels up to 64x64, arrays and cubes,
/// and 3D chains of the formats it carries through a half), from its first level: the chain on
/// the CPU and in the kernel gives every level's bytes.
#[test]
fn the_chain_gives_the_m3_s_measured_chains() {
    let mut modules = Modules::default();
    let chains = measured(MEASURED_M3);
    assert_eq!(chains.len(), 105);
    for c in chains {
        let chain = plan(c.pf, c.extent, c.layers, c.levels.len() as u32);
        let what = format!("{:?} {:?} x{}", c.pf, c.extent, c.layers);
        let cpu = chain.expected_levels(&c.levels[0]).expect("levels");
        for (level, (got, want)) in cpu.iter().zip(&c.levels).enumerate() {
            assert_eq!(got, want, "CPU {what} level {level}");
        }
        let gpu = run_chain(&mut modules, &chain, &c.levels[0]);
        assert_eq!(gpu, c.levels, "kernel {what}");
    }
}

/// One chain of `measured_m3_passes.txt`: its key, format, first level's extent, layers and
/// pattern, and every level's digest.
struct Digested {
    key: String,
    pf: PixelFormat,
    extent: [u32; 3],
    /// Whether the chain is a 3D texture's (`3d`, its extent `<width>x<height>x<depth>`).
    volume: bool,
    layers: u32,
    pattern: String,
    /// The texture's store (a table's store column; `private` where it has none).
    store: String,
    /// Whether Metal kept the texture compressed (`MipmapFilterChain::plan`'s `compressed`): a
    /// `private`, `shared` or `rt` (RenderTarget) store, which keeps `allowGPUOptimizedContents`
    /// on and neither `ShaderWrite` nor `PixelFormatView`, 16 texels or more on its shorter side;
    /// not `plain` (`allowGPUOptimizedContents` off), `sw` (`ShaderWrite`) or `pfv`
    /// (`PixelFormatView`).
    compressed: bool,
    digests: Vec<String>,
}

/// The chains of a table of digests (`measured_m3_passes.txt`'s rows, with a store column
/// before the digests where `stored`).
fn digested(table: &str, stored: bool) -> Vec<Digested> {
    table
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let p: Vec<&str> = line.split_whitespace().collect();
            let d: Vec<u32> = p[3]
                .split('x')
                .map(|v| v.parse().expect("a size"))
                .collect();
            let volume = p[2] == "3d";
            let extent = [d[0], d[1], if volume { d[2] } else { 1 }];
            let store = if stored { p[6] } else { "private" };
            let kept = match store {
                "private" | "shared" | "rt" => true,
                "plain" | "sw" | "pfv" => false,
                s => panic!("a store {s}"),
            };
            Digested {
                key: p[0].to_owned(),
                pf: PixelFormat::from_name(p[1]).expect("a format"),
                extent,
                volume,
                layers: p[4].parse().expect("layers"),
                pattern: p[5].to_owned(),
                store: store.to_owned(),
                compressed: kept && !volume && compressed(extent),
                digests: p[6 + usize::from(stored)..]
                    .iter()
                    .map(|&d| d.to_owned())
                    .collect(),
            }
        })
        .collect()
}

/// The first 12 hex digits of the SHA-256 of `bytes`.
fn digest(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    sha2::Sha256::digest(bytes)[..6]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The first level of every layer of a `measured_m3_passes.txt` chain of texels of `bytes`
/// bytes, as the probe made it (the file's header); of a 3D chain, every slice in turn as a layer.
fn probe_first_level(c: &Digested, bytes: u32) -> Vec<u8> {
    let mut state = c.key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let [w, h, d] = c.extent;
    let mut out = Vec::new();
    for _ in 0..if c.volume { d } else { c.layers } {
        for i in 0..w * h {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            match c.pattern.as_str() {
                "u8" => {
                    for _ in 0..bytes {
                        out.push((next() >> 56) as u8);
                    }
                }
                "h16" => {
                    for _ in 0..bytes / 2 {
                        let r = next();
                        let half = (((r >> 40) % 16) << 10 | (r & 0x3ff)) as u16;
                        out.extend(half.to_le_bytes());
                    }
                }
                "f32" => {
                    for _ in 0..bytes / 4 {
                        let r = next();
                        let float = (((100 + (r >> 40) % 28) << 23) | (r & 0x7f_ffff)) as u32;
                        out.extend(float.to_le_bytes());
                    }
                }
                "ramp" => out.extend(x.to_le_bytes().into_iter().chain(y.to_le_bytes())),
                "xramp" => out.extend(x.to_le_bytes()),
                "yramp" => out.extend(y.to_le_bytes()),
                p => panic!("a pattern {p}"),
            }
        }
    }
    out
}

/// The chain of a table's row under `rule`, as Metal stored the texture (`Digested::compressed`).
fn digested_chain(c: &Digested, rule: MipmapRule) -> MipmapFilterChain {
    MipmapFilterChain::plan(
        c.pf,
        c.extent,
        c.layers,
        c.digests.len() as u32,
        c.volume,
        c.compressed,
        rule,
        64,
        u32::MAX,
        [65535, 65535],
        None,
    )
    .unwrap_or_else(|| panic!("a chain of {} {}", c.key, c.store))
}

/// Whether level `level` of `chain` is one a near tie may miss on (the levels
/// `MipmapFilterChain::near_ties` counts): a level of the filter's of a format with an 8-bit sRGB
/// colour field, or an extended-range sRGB one the M3 Max carries through a half; or a tail
/// level read from kept values of a format with a 16-bit float field, or such an extended-range
/// sRGB one.
fn near_tie_level(chain: &MipmapFilterChain, level: usize) -> bool {
    let l = layout(chain.pixel_format).expect("a layout");
    let has = |rule: Rule| l.fields.iter().any(|f| f.rule == rule);
    let half_srgb = l.half && has(Rule::XrSrgb);
    match chain.levels[level].kernel.as_ref().map(|k| k.step) {
        Some(KernelStep::Sampler { .. }) => half_srgb || has(Rule::Srgb8),
        Some(KernelStep::Tail { read_kept: true }) => half_srgb || has(Rule::Float16),
        _ => false,
    }
}

/// What [`check_m3_chains`] found over a table's chains.
#[derive(Debug, PartialEq, Eq)]
struct Checked {
    /// The chains whose M3 Max answer the chain claims (`unlike_an_m3` false).
    known: usize,
    /// The claimed chains that miss Metal's digest from a level a near tie may miss on.
    near_ties: usize,
}

/// Every chain of a table of digests from the probe's first level, against the chain on the
/// CPU. The first level is the probe's. The chain claims the M3 Max's answer (`unlike_an_m3`
/// false) but for `A1BGR5Unorm`'s chains and `ABGR4Unorm`'s below their first level (a few
/// values unexplained). An unclaimed chain gives its first computed level (the M3 Max's filter all the
/// same) but where a near tie may miss. A claimed chain gives every level's digest down to its
/// first miss, if any, which is at a level a near tie may miss on ([`near_tie_level`]); the
/// levels below it are computed from a level unlike Metal's, and are not compared.
fn check_m3_chains(chains: &[Digested]) -> Checked {
    use PixelFormat as P;
    let mut checked = Checked {
        known: 0,
        near_ties: 0,
    };
    for c in chains {
        let what = format!("{} {}", c.key, c.store);
        let first = probe_first_level(c, layout(c.pf).expect("a layout").bytes);
        assert_eq!(digest(&first), c.digests[0], "{what} level 0");
        let chain = digested_chain(c, MipmapRule::AppleM3MaxPasses);
        let unexplained = c.pf == P::A1BGR5Unorm || (c.pf == P::ABGR4Unorm && c.digests.len() >= 3);
        assert_eq!(chain.unlike_an_m3, unexplained, "{what}");
        let cpu = chain.expected_levels(&first).expect("levels");
        let miss = (1..cpu.len()).find(|&level| digest(&cpu[level]) != c.digests[level]);
        if chain.unlike_an_m3 {
            assert!(
                miss != Some(1) || (chain.near_ties && near_tie_level(&chain, 1)),
                "{what} level 1"
            );
            continue;
        }
        checked.known += 1;
        if let Some(level) = miss {
            assert!(chain.near_ties, "{what} level {level}");
            assert!(near_tie_level(&chain, level), "{what} level {level}");
            checked.near_ties += 1;
        }
    }
    checked
}

/// Every chain the M3 Max measured past its tail's extent (2D textures, arrays and cubes whose
/// first level is wider or taller than 64 texels, Private), from the probe's first level: the
/// chain claims every one (`check_m3_chains`; the formats whose fields are not all 8-bit unorm or
/// 32-bit float), and gives every level's digest but from a near tie in four (one sRGB colour code
/// of a first level, and 16-bit float texels of levels read from kept values); the kernel's chain
/// of one of them, two passes and the tail, and of a strip, is the CPU's.
#[test]
fn the_chain_gives_the_m3_s_measured_passes() {
    let chains = digested(MEASURED_M3_PASSES, false);
    assert_eq!(chains.len(), 238);
    assert_eq!(
        check_m3_chains(&chains),
        Checked {
            known: 238,
            near_ties: 4
        }
    );
    let mut modules = Modules::default();
    for key in ["f80-2d-130x130-l8-u8", "f80-2d-1000x3-l10-u8"] {
        kernel_gives(
            &mut modules,
            &chains,
            (key, "private"),
            MipmapRule::AppleM3MaxPasses,
        );
    }
}

/// share-mips-m3b's chains of the 8-bit unorm and 32-bit float formats (strips from 66x1 to
/// 16384x1, chains to 8192x64, Private, Shared and with `allowGPUOptimizedContents` off): the
/// chain claims every one and gives every level's digest (`check_m3_chains`): the 21 compressed
/// chains 16 texels or more on their shorter side, to 128 times wider than tall, a pass's
/// one-texel-tall level and the tails from levels 4096 texels and more wide among them, the 45
/// strips, narrower or shorter than 16 texels, to 16384x1, and the 7 textures Metal does not
/// compress 16 texels or more on their shorter side (their runs).
#[test]
fn the_chain_claims_the_m3_s_passes_only_where_they_were_measured() {
    let chains = digested(MEASURED_M3_LAYOUTS, true);
    assert_eq!(chains.len(), 73);
    assert_eq!(
        check_m3_chains(&chains),
        Checked {
            known: 73,
            near_ties: 0
        }
    );
}

/// The digests of every level of share-mips-m3b's chains of every format, Private with
/// `allowGPUOptimizedContents` off, and of its 16-bit float chains (the file's header states
/// them).
const MEASURED_M3_FORMATS: &str = include_str!("measured_m3_formats.txt");
/// The digests of every level of share-mips-m3c's chains, by the texture's store: textures
/// Metal does not compress, and every format compressed and as strips (the file's header states
/// them).
const MEASURED_M3_STORES: &str = include_str!("measured_m3_stores.txt");

/// The kernel's chain of a table's row under `rule` is the CPU's, which gives every level's digest.
fn kernel_gives(
    modules: &mut Modules,
    chains: &[Digested],
    (key, store): (&str, &str),
    rule: MipmapRule,
) {
    let c = chains
        .iter()
        .find(|c| c.key == key && c.store == store)
        .expect("a chain");
    let first = probe_first_level(c, layout(c.pf).expect("a layout").bytes);
    let chain = digested_chain(c, rule);
    let cpu = chain.expected_levels(&first).expect("levels");
    for (level, (got, want)) in cpu.iter().zip(&c.digests).enumerate() {
        assert_eq!(&digest(got), want, "{key} {store} level {level}");
    }
    assert_eq!(
        run_chain(modules, &chain, &first),
        cpu,
        "kernel {key} {store}"
    );
}

/// share-mips-m3c's `BGRA8Unorm` chains of textures Metal does not compress
/// (`allowGPUOptimizedContents` off, `ShaderWrite`, `PixelFormatView`): squares from 64x64 to
/// 4096x4096, rectangles to 16384x64 and their transposes, arrays and cubes, and the compressed
/// controls (RenderTarget): the chain claims every one (`check_m3_chains`) and gives every level's
/// digest. Their runs read kept values and codes in turn, but codes only below a filtered level
/// whose shorter side is a power of two of at most 16 texels; the kernel's chain of an uncompressed
/// square is the CPU's.
#[test]
fn the_chain_gives_the_m3_s_runs_where_metal_does_not_compress() {
    let chains: Vec<Digested> = digested(MEASURED_M3_STORES, true)
        .into_iter()
        .filter(|c| c.pf == PixelFormat::BGRA8Unorm)
        .collect();
    assert_eq!(chains.len(), 93);
    assert_eq!(
        check_m3_chains(&chains),
        Checked {
            known: 93,
            near_ties: 0
        }
    );
    kernel_gives(
        &mut Modules::default(),
        &chains,
        ("f80-2d-128x128-l8-u8", "plain"),
        MipmapRule::AppleM3MaxPasses,
    );
}

/// share-mips-m3b's chains of every format the filter computes with `allowGPUOptimizedContents`
/// off, and its 16-bit float chains; share-mips-m3c's of every format but `BGRA8Unorm`,
/// compressed, as strips and with `allowGPUOptimizedContents` off: the chain claims the M3 Max's
/// answer but for the 4-, 5- and 1-bit formats' (`check_m3_chains`): 146 of share-mips-m3b's 158
/// and 107 of share-mips-m3c's 117. It gives every claimed chain's levels but from a near tie in
/// 14, of 16-bit float, 8-bit sRGB and extended-range sRGB formats. A format carried through a half
/// reads kept values at every level of a run, and the tail reads an axis of one texel alone
/// (`RG11B10Float`'s 4096x4 strip, whose one-texel levels hold infinities); the kernel's chains of
/// such strips are the CPU's.
#[test]
fn the_chain_gives_every_format_s_m3_passes() {
    let formats = digested(MEASURED_M3_FORMATS, true);
    let stores: Vec<Digested> = digested(MEASURED_M3_STORES, true)
        .into_iter()
        .filter(|c| c.pf != PixelFormat::BGRA8Unorm)
        .collect();
    assert_eq!((formats.len(), stores.len()), (158, 117));
    assert_eq!(
        (check_m3_chains(&formats), check_m3_chains(&stores)),
        (
            Checked {
                known: 146,
                near_ties: 13
            },
            Checked {
                known: 107,
                near_ties: 1
            }
        )
    );
    let mut modules = Modules::default();
    for key in ["f554-2d-1024x12-l11-u8", "f92-2d-4096x4-l13-u8"] {
        kernel_gives(
            &mut modules,
            &stores,
            (key, "private"),
            MipmapRule::AppleM3MaxPasses,
        );
    }
}

/// The digests of every level of the M3 Max's chains under macOS 27.0.1 (the file's header states
/// them).
const MEASURED_M3_27: &str = include_str!("measured_m3_27.txt");

/// Every chain of a table of digests from the probe's first level, against the chain under
/// [`MipmapRule::Filter`] on the CPU. The first level is the probe's. The chain claims the M3
/// Max's answer (`unlike_an_m3` false) for every chain, and gives every level's digest down to its
/// first miss, if any, which is at a level a near tie may miss on ([`near_tie_level`]).
fn check_filter_chains(chains: &[Digested]) -> Checked {
    let mut checked = Checked {
        known: 0,
        near_ties: 0,
    };
    for c in chains {
        let what = format!("{} {}", c.key, c.store);
        let first = probe_first_level(c, layout(c.pf).expect("a layout").bytes);
        assert_eq!(digest(&first), c.digests[0], "{what} level 0");
        let chain = digested_chain(c, MipmapRule::Filter);
        assert!(!chain.unlike_an_m3, "{what}");
        let cpu = chain.expected_levels(&first).expect("levels");
        checked.known += 1;
        if let Some(level) = (1..cpu.len()).find(|&level| digest(&cpu[level]) != c.digests[level]) {
            assert!(chain.near_ties, "{what} level {level}");
            assert!(near_tie_level(&chain, level), "{what} level {level}");
            checked.near_ties += 1;
        }
    }
    checked
}

/// The M3 Max's 2D chains, arrays and cubes under macOS 27.0.1 (`measured_m3_27.txt`: the four
/// tables above rerun, to 512x512 texels a layer): the chain under [`MipmapRule::Filter`] claims
/// every one, `A1BGR5Unorm`'s and `ABGR4Unorm`'s among them, and gives every level's digest but
/// from a near tie in 2, of 8-bit sRGB formats (one texel each: linear sums of 906.5002 and
/// 413.5002 steps stored one byte below the chain's); macOS 26.7's passes miss its levels below
/// the first computed one. The kernel's chain of one is the CPU's.
#[test]
fn the_filter_gives_the_m3_s_chains_under_macos_27_0_1() {
    let chains: Vec<Digested> = digested(MEASURED_M3_27, true)
        .into_iter()
        .filter(|c| !c.volume)
        .collect();
    assert_eq!(chains.len(), 588);
    assert_eq!(
        check_filter_chains(&chains),
        Checked {
            known: 588,
            near_ties: 2
        }
    );
    // The release moved: 26.7's chain of a 256x256 texture misses 27.0.1's second computed level.
    let c = chains
        .iter()
        .find(|c| c.key == "f80-2d-256x256-l9-u8" && c.store == "private")
        .expect("a chain");
    let first = probe_first_level(c, 4);
    let older = digested_chain(c, MipmapRule::AppleM3MaxPasses)
        .expected_levels(&first)
        .expect("levels");
    assert_eq!(digest(&older[1]), c.digests[1]);
    assert_ne!(digest(&older[2]), c.digests[2]);
    kernel_gives(
        &mut Modules::default(),
        &chains,
        ("f80-2d-130x130-l8-u8", "private"),
        MipmapRule::Filter,
    );
}

/// The M3 Max's 3D chains under macOS 27.0.1 (`measured_m3_27.txt`: 14 formats at 13 extents,
/// halving and not, `R16Float` among them): the chain under [`MipmapRule::Filter`] claims every
/// one and gives every level's digest: the weights split along y last, a float's slices summed
/// apart (`model.rs`). The kernel's chains of a 3D texture of odd sides, of 8-bit and half-float
/// codes, are the CPU's.
#[test]
fn the_filter_gives_the_m3_s_3d_chains_under_macos_27_0_1() {
    let chains: Vec<Digested> = digested(MEASURED_M3_27, true)
        .into_iter()
        .filter(|c| c.volume)
        .collect();
    assert_eq!(chains.len(), 164);
    assert_eq!(
        check_filter_chains(&chains),
        Checked {
            known: 164,
            near_ties: 0
        }
    );
    let mut modules = Modules::default();
    for key in ["f70-3d-41x61x5-l6-u8", "f25-3d-41x61x5-l6-h16"] {
        kernel_gives(&mut modules, &chains, (key, "private"), MipmapRule::Filter);
    }
}

/// `n` bytes from a seed (xorshift).
fn seeded(n: usize, mut seed: u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 24) as u8
        })
        .collect()
}

/// Every format the filter computes, over chains that halve and do not, layered, 3D, squares
/// of powers of two (whose even levels read the values the level above kept), one past the
/// tail's extent (the M4's filter below its first level, its passes not known) and one of two
/// passes (where the M3 Max's passes are known), from random bytes (every bit pattern: NaNs,
/// infinities, subnormals, out-of-range codes): the kernel's chain is the CPU's.
#[test]
fn the_kernel_is_the_chain_on_the_cpu() {
    let mut modules = Modules::default();
    let shapes: [([u32; 3], u32, u32); 7] = [
        ([7, 5, 1], 1, 3),
        ([4, 3, 1], 6, 3),
        ([5, 4, 3], 1, 3),
        ([17, 1, 1], 1, 5),
        ([8, 8, 1], 2, 4),
        ([66, 2, 1], 1, 3),
        ([272, 16, 1], 1, 5),
    ];
    let mut formats = 0;
    for (seed, &pf) in PixelFormat::ALL.iter().enumerate() {
        let Some(l) = layout(pf) else {
            continue;
        };
        formats += 1;
        for (s, &(extent, layers, levels)) in shapes.iter().enumerate() {
            let chain = plan(pf, extent, layers, levels);
            let texels = extent.iter().product::<u32>() * layers;
            let first = seeded(
                (texels * l.bytes) as usize,
                0x9e37_79b9 ^ (seed * 31 + s) as u64,
            );
            let want = chain.expected_levels(&first).expect("levels");
            let got = run_chain(&mut modules, &chain, &first);
            for (level, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{pf:?} {extent:?} x{layers} level {level}");
            }
        }
    }
    assert_eq!(formats, 35, "the formats the filter computes");
}

/// Taps the M4 measured (R32Float ramps) where the exact centre's weight,
/// rounded, is one step from Metal's: `(above, level, texel, x0, q)`.
#[test]
fn the_taps_are_the_m4_s_f32_coordinates() {
    let m4 = Reciprocal::AppleM4;
    let measured: [(u32, u32, u32, u32, u32); 26] = [
        (513, 256, 246, 493, 119),
        (645, 322, 302, 605, 113),
        (707, 353, 287, 575, 81),
        (771, 385, 289, 579, 65),
        (783, 391, 237, 475, 28),
        (845, 422, 180, 360, 238),
        (859, 429, 369, 739, 93),
        (879, 439, 225, 451, 4),
        (1023, 511, 250, 500, 254),
        (1537, 768, 211, 422, 199),
        (8191, 4095, 1191, 2382, 203),
        (8191, 4095, 2213, 4427, 11),
        (8193, 4096, 2711, 5423, 42),
        (12289, 6144, 2219, 4438, 221),
        (12289, 6144, 3466, 6933, 17),
        (12289, 6144, 4182, 8365, 47),
        (12289, 6144, 4475, 8951, 59),
        (12289, 6144, 5936, 11873, 120),
        (16383, 8191, 2701, 5402, 213),
        (16383, 8191, 5774, 11549, 53),
        (12289, 6144, 395, 790, 145),
        (12289, 6144, 419, 838, 146),
        (12289, 6144, 443, 886, 147),
        (16383, 8191, 1039, 2078, 161),
        (16383, 8191, 1071, 2142, 162),
        (16383, 8191, 1103, 2206, 163),
    ];
    for (big, small, i, x0, q) in measured {
        assert_eq!(
            axis_tap(i, big, small, m4),
            (x0, q),
            "{big} -> {small} texel {i}"
        );
    }
    // A halving axis takes its two texels at half each; an axis of one that stays one, one.
    assert_eq!(axis_tap(3, 16, 8, m4), (6, 128));
    assert_eq!(axis_tap(0, 1, 1, m4), (0, 0));
    // 3 -> 1: the centre texel.
    assert_eq!(axis_tap(0, 3, 1, m4), (1, 0));
}

/// Taps the M3 Max measured (R32Float ramps: 173,505 taps of widths, heights
/// and depths from 257 to 16383, every one its coordinates from its own reciprocal) where its
/// reciprocal moves the weight one step above the M4's: `(above, level, texel, x0, q)`.
#[test]
fn the_m3_s_taps_start_from_its_own_reciprocal() {
    let measured: [(u32, u32, u32, u32, u32); 16] = [
        (283, 141, 89, 179, 35),
        (461, 230, 141, 283, 30),
        (475, 237, 199, 399, 88),
        (491, 245, 44, 88, 175),
        (565, 282, 249, 499, 99),
        (615, 307, 156, 313, 3),
        (637, 318, 256, 513, 79),
        (661, 330, 169, 339, 4),
        (723, 361, 278, 557, 70),
        (795, 397, 374, 749, 114),
        (851, 425, 422, 845, 127),
        (855, 427, 424, 849, 127),
        (921, 460, 225, 450, 254),
        (949, 474, 467, 935, 125),
        (977, 488, 477, 955, 123),
        (981, 490, 433, 867, 99),
    ];
    for (big, small, i, x0, q) in measured {
        let m3 = axis_tap(i, big, small, Reciprocal::AppleM3Max);
        assert_eq!(m3, (x0, q), "{big} -> {small} texel {i}");
        let m4 = axis_tap(i, big, small, Reciprocal::AppleM4);
        assert_eq!(m4, (x0, q - 1), "the M4's: {big} -> {small} texel {i}");
    }
    // The M3 Max's reciprocal: one unit above the rounded quotient at
    // 41 and 61, below at 33 and its doubles, the rounded quotient at 3 and at 16384.
    let units = |n: u32| {
        let m3 = Reciprocal::AppleM3Max.of(n).to_bits();
        i64::from(m3) - i64::from((1.0f32 / n as f32).to_bits())
    };
    let deviations: Vec<i64> = [41, 61, 33, 66, 8448, 3, 16384].map(units).to_vec();
    assert_eq!(deviations, [1, 1, -1, -1, -1, 0, 0]);
}

/// The plan: levels and params at the alignment, in order; no chain where the filter is not
/// Metal's or the device's limits do not fit.
#[test]
fn the_plan_lays_the_levels_and_refuses_what_it_cannot_run() {
    let pf = PixelFormat::RGBA8Unorm;
    let chain = MipmapFilterChain::plan(
        pf,
        [8, 4, 1],
        2,
        3,
        false,
        compressed([8, 4, 1]),
        MipmapRule::AppleM3MaxPasses,
        256,
        u32::MAX,
        [65535, 65535],
        None,
    )
    .expect("a chain");
    let offsets: Vec<u64> = chain.levels.iter().map(|l| l.offset).collect();
    // 8x4, 4x2 and 2x1 texels of 4 bytes over two layers: 256, 64 and 16 bytes.
    assert_eq!(offsets, [0, 256, 512]);
    let bytes: Vec<u64> = chain.levels.iter().map(|l| l.bytes).collect();
    assert_eq!(bytes, [256, 64, 16]);
    assert!(chain.levels[0].params.is_none());
    let (at1, w1) = chain.levels[1].params.clone().expect("params");
    let (at2, w2) = chain.levels[2].params.clone().expect("params");
    assert_eq!(at1, 768);
    assert_eq!(at2, (at1 + 4 * w1.len() as u64).next_multiple_of(256));
    assert_eq!(
        chain.bytes,
        (at2 + 4 * w2.len() as u64).next_multiple_of(256)
    );
    // Level 2's header: 4 texels over two layers, one row of 64, its extent and the one above.
    assert_eq!(w2[..8], [4, 64, 2, 1, 1, 4, 2, 1]);
    // The kernels: the M3 Max's filter, then its tail from level 1's codes (not a square).
    let kernel = |step, depth| MipmapFilterKernel {
        pixel_format: pf,
        step,
        keep: false,
        depth,
        fetch: None,
    };
    let kernels: Vec<_> = chain.levels.iter().map(|l| l.kernel).collect();
    assert_eq!(
        kernels,
        [
            None,
            Some(kernel(KernelStep::Sampler { half: false }, false)),
            Some(kernel(KernelStep::Tail { read_kept: false }, false)),
        ]
    );
    assert_eq!(chain.levels[2].groups, [1, 1]);
    // A 4-byte texel's levels are written whole; a narrower texel's are zeroed and ORed into.
    assert!(!chain.zeroed);
    let narrow = MipmapFilterChain::plan(
        PixelFormat::RG8Unorm,
        [8, 4, 1],
        1,
        2,
        false,
        compressed([8, 4, 1]),
        MipmapRule::AppleM3MaxPasses,
        256,
        u32::MAX,
        [65535, 65535],
        None,
    );
    assert!(narrow.expect("a chain").zeroed);
    let none = |pf, extent, layers, levels, max_range, max_groups| {
        MipmapFilterChain::plan(
            pf,
            extent,
            layers,
            levels,
            false,
            compressed(extent),
            MipmapRule::AppleM3MaxPasses,
            64,
            max_range,
            max_groups,
            None,
        )
        .is_none()
    };
    let (big, groups) = (u32::MAX, [65535, 65535]);
    // The two 5-bit formats the rule misses, an integer format, a compressed one.
    for other in [
        PixelFormat::B5G6R5Unorm,
        PixelFormat::BGR5A1Unorm,
        PixelFormat::RGBA8Uint,
        PixelFormat::BC1RGBA,
    ] {
        assert!(none(other, [8, 8, 1], 1, 2, big, groups), "{other:?}");
    }
    // One level, no layers, an empty extent, an extent past the tap tables.
    assert!(none(pf, [8, 8, 1], 1, 1, big, groups));
    assert!(none(pf, [8, 8, 1], 0, 2, big, groups));
    assert!(none(pf, [0, 8, 1], 1, 2, big, groups));
    assert!(none(pf, [65536, 1, 1], 1, 2, big, groups));
    // A level past the storage range, a dispatch past the workgroups.
    // (64x64 texels of 4 bytes: 16384; each level's params fit in far less.)
    assert!(none(pf, [64, 64, 1], 1, 2, 16383, groups));
    assert!(!none(pf, [64, 64, 1], 1, 2, 16384, groups));
    // Level 1: 64x64 texels, 64 workgroups.
    assert!(none(pf, [128, 128, 1], 1, 2, big, [1, 63]));
    assert!(!none(pf, [128, 128, 1], 1, 2, big, [1, 64]));
}

/// Where an Apple M3 Max's answer may differ from the chain's (the module docs' "Whose
/// answer"): below the first level of a 2D chain whose passes are not known (a compressed
/// texture's pass from a level more than 128 times wider than tall), at a few values of `A1BGR5Unorm` and `ABGR4Unorm` in 2D, and in 3D for the formats
/// summed as half floats.
#[test]
fn the_chain_says_where_an_m3_s_answer_differs() {
    let unlike = |pf, extent: [u32; 3], levels| {
        let volume = extent[2] > 1;
        MipmapFilterChain::plan(
            pf,
            extent,
            1,
            levels,
            volume,
            compressed(extent),
            MipmapRule::AppleM3MaxPasses,
            64,
            u32::MAX,
            [65535, 65535],
            None,
        )
        .expect("a chain")
        .unlike_an_m3
    };
    use PixelFormat as P;
    // The M3 Max's filter for the first computed level, its tail below it up to 64x64.
    assert!(!unlike(P::RGBA8Unorm, [8, 8, 1], 2));
    assert!(!unlike(P::RGBA16Float, [7, 5, 1], 2));
    assert!(!unlike(P::RGBA8Unorm, [8, 8, 1], 4));
    assert!(!unlike(P::RGBA16Float, [64, 64, 1], 7));
    assert!(!unlike(P::BGR10XR, [61, 37, 1], 6));
    // Up to 65 texels: its filter and its tail below it.
    assert!(!unlike(P::RGBA8Unorm, [65, 8, 1], 3));
    assert!(!unlike(P::BGR10XR, [65, 65, 1], 7));
    // Past it, its passes for every format, but a compressed pass from a level more than 128 times
    // wider than tall.
    assert!(!unlike(P::RGBA8Unorm, [128, 128, 1], 2));
    assert!(!unlike(P::BGRA8Unorm, [256, 256, 1], 9));
    assert!(!unlike(P::RG32Float, [1920, 1080, 1], 11));
    assert!(!unlike(P::R8Unorm, [1024, 16, 1], 11));
    assert!(!unlike(P::BGRA8UnormSRGB, [256, 256, 1], 3));
    assert!(!unlike(P::RGBA16Float, [256, 256, 1], 3));
    assert!(!unlike(P::RGB9E5Float, [300, 200, 1], 9));
    assert!(!unlike(P::RG11B10Float, [256, 256, 1], 9));
    // A strip, narrower or shorter than 16 texels: its own passes.
    assert!(!unlike(P::RGBA8Unorm, [2048, 8, 1], 12));
    assert!(!unlike(P::R8Unorm, [1, 16384, 1], 15));
    assert!(!unlike(P::RG32Float, [1024, 12, 1], 11));
    assert!(!unlike(P::RGBA8UnormSRGB, [2048, 8, 1], 12));
    assert!(!unlike(P::BGR10XR, [8, 128, 1], 8));
    assert!(!unlike(P::RG11B10Float, [4096, 4, 1], 13));
    // A pass before the last that filters to a level one texel tall of an even width (2048x32's
    // 64x1), or wide, then the tail of its codes; up to 128 times wider than tall, the M4's filter
    // below the first level past it.
    assert!(!unlike(P::BGRA8Unorm, [2048, 32, 1], 7));
    assert!(!unlike(P::BGRA8Unorm, [32, 2048, 1], 12));
    assert!(!unlike(P::BGRA8Unorm, [4096, 64, 1], 13));
    assert!(!unlike(P::R8Unorm, [2048, 16, 1], 12));
    assert!(unlike(P::R8Unorm, [4096, 16, 1], 3));
    // A texture 16 texels or more on its shorter side that Metal does not compress
    // (`allowGPUOptimizedContents` off): its runs, as a strip's, at any aspect.
    let plain = |pf, extent: [u32; 3], levels| {
        MipmapFilterChain::plan(
            pf,
            extent,
            1,
            levels,
            false,
            false,
            MipmapRule::AppleM3MaxPasses,
            64,
            u32::MAX,
            [65535, 65535],
            None,
        )
        .expect("a chain")
        .unlike_an_m3
    };
    assert!(!plain(P::BGRA8Unorm, [256, 256, 1], 9));
    assert!(!plain(P::BGRA8Unorm, [64, 64, 1], 3));
    assert!(!plain(P::BGRA8Unorm, [16, 8, 1], 5));
    assert!(!plain(P::BGRA8Unorm, [16384, 64, 1], 15));
    assert!(!plain(P::RGBA16Float, [1024, 1024, 1], 11));
    assert!(!plain(P::RG11B10Float, [256, 256, 1], 9));
    assert!(unlike(P::BGRA8Unorm, [16384, 64, 1], 3));
    // 3D: the M3 Max's at every level, but for the half-float sums.
    assert!(!unlike(P::RGBA8Unorm, [8, 8, 8], 4));
    assert!(!unlike(P::RGBA32Float, [8, 8, 8], 4));
    assert!(!unlike(P::ABGR4Unorm, [8, 8, 8], 4));
    assert!(unlike(P::RGBA16Float, [8, 8, 8], 4));
    assert!(unlike(P::RG11B10Float, [8, 8, 8], 2));
    // The 4-bit and 1-bit fields: their tail, and A1BGR5's first level, missed a few values.
    assert!(!unlike(P::ABGR4Unorm, [8, 8, 1], 2));
    assert!(unlike(P::ABGR4Unorm, [8, 8, 1], 3));
    assert!(unlike(P::A1BGR5Unorm, [8, 8, 1], 2));
}

/// Where a level may take a rounding the chain does not fit (the module docs' "Near ties"): every
/// chain of a format with an 8-bit sRGB colour (its filtered levels), 3D too, every chain of an
/// extended-range sRGB format (carried through a half), and a 16-bit float chain with a level read
/// from kept values; nowhere else: not a 10-bit field's, nor `BGR10_XR`'s and `BGRA10_XR`'s, nor
/// `ABGR4Unorm`'s and `A1BGR5Unorm`'s.
#[test]
fn the_chain_says_where_a_rounding_is_not_fit() {
    let ties = |pf, extent: [u32; 3], levels| {
        let volume = extent[2] > 1;
        MipmapFilterChain::plan(
            pf,
            extent,
            1,
            levels,
            volume,
            compressed(extent),
            MipmapRule::AppleM3MaxPasses,
            64,
            u32::MAX,
            [65535, 65535],
            None,
        )
        .expect("a chain")
        .near_ties
    };
    use PixelFormat as P;
    for pf in [
        P::R8UnormSRGB,
        P::RG8UnormSRGB,
        P::RGBA8UnormSRGB,
        P::BGRA8UnormSRGB,
    ] {
        assert!(ties(pf, [7, 5, 1], 2), "{pf:?}");
        assert!(ties(pf, [4, 4, 4], 3), "{pf:?} 3D");
    }
    for pf in [P::R16Float, P::RG16Float, P::RGBA16Float] {
        assert!(
            ties(pf, [16, 16, 1], 3),
            "{pf:?}: level 2 reads kept values"
        );
        assert!(!ties(pf, [16, 16, 1], 2), "{pf:?}: one computed level");
        assert!(!ties(pf, [16, 8, 1], 4), "{pf:?}: no square, nothing kept");
        assert!(!ties(pf, [4, 4, 4], 3), "{pf:?} 3D");
    }
    // The 10-bit grid and the extended-range codes floor each term before the sum, and the 4-, 5-
    // and 1-bit codes filter as wider codes.
    for pf in [
        P::RGBA8Unorm,
        P::BGRA8Unorm,
        P::RGBA32Float,
        P::RGBA16Unorm,
        P::RGB10A2Unorm,
        P::BGR10A2Unorm,
        P::BGR10XR,
        P::BGRA10XR,
        P::ABGR4Unorm,
        P::A1BGR5Unorm,
    ] {
        assert!(!ties(pf, [16, 16, 1], 5), "{pf:?}");
        assert!(!ties(pf, [7, 5, 1], 2), "{pf:?}");
        assert!(!ties(pf, [4, 4, 4], 3), "{pf:?} 3D");
    }
    for pf in [P::BGR10XRSRGB, P::BGRA10XRSRGB] {
        assert!(ties(pf, [7, 5, 1], 2), "{pf:?}");
        assert!(ties(pf, [4, 4, 4], 3), "{pf:?} 3D");
    }
}

/// The texels of the M4's first computed levels where the 2-bit alpha's two candidate rules part
/// (the file's header states them).
const MEASURED_2BIT: &str = include_str!("measured_2bit.txt");

/// The 2-bit code of the four 2-bit codes `codes` (x0y0, x1y0, x0y1, x1y1) weighing `qx` and `qy`
/// (of 256) along x and y: on the 12-bit grid of 2-bit codes (`wide` false), or as the 10-bit
/// code they replicate to (`341 c`) on its grid, taken to its nearest 2-bit code (`wide`).
fn two_bit(codes: [u32; 4], qx: u32, qy: u32, wide: bool) -> u32 {
    let (qx, qy) = (i64::from(qx), i64::from(qy));
    let n = [
        (256 - qx) * (256 - qy),
        qx * (256 - qy),
        (256 - qx) * qy,
        qx * qy,
    ];
    let (m, k, den) = if wide { (341, 2, 1023) } else { (1, 10, 3) };
    let s: i64 = codes
        .iter()
        .zip(n)
        .map(|(&c, n)| i64::from(c) * m * n * 256)
        .sum();
    let q = (s + (1 << (23 - k))) >> (24 - k);
    let (code, fraction, half) = (q >> k, q & ((1 << k) - 1), 1 << (k - 1));
    let up = fraction > half || (fraction == half && grid_tie_up(code as i32, den));
    ((code + i64::from(up) + m / 2) / m) as u32
}

/// `RGB10A2Unorm`'s and `BGR10A2Unorm`'s 2-bit alpha filters as the 10-bit code its bits
/// replicate to, and stores that code's nearest 2-bit code: at every texel of the M4's designed
/// first computed levels where that and the 12-bit grid of 2-bit codes part, the M4's filter on the
/// CPU gives the code the M4 stored, the 10-bit code's; and in the kernel too, over the 257x257
/// texture's level. Each texel's taps hold the measured codes, every other texel 0.
#[test]
fn the_2_bit_field_filters_as_its_10_bit_code() {
    let m4 = Step::Sampler {
        reciprocal: Reciprocal::AppleM4,
        half: false,
    };
    let mut modules = Modules::default();
    let mut chains: Vec<(PixelFormat, [u32; 3], Vec<[u32; 4]>)> = Vec::new();
    for line in MEASURED_2BIT.lines().filter(|l| !l.starts_with('#')) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts[0] == "chain" {
            let d: Vec<u32> = parts[2].split('x').map(|v| v.parse().unwrap()).collect();
            let pf = PixelFormat::from_name(parts[1]).expect("a format");
            chains.push((pf, [d[0], d[1], 1], Vec::new()));
            continue;
        }
        let codes = u32::from_str_radix(parts[2], 4).expect("four codes");
        let row = [
            parts[0].parse().unwrap(),
            parts[1].parse().unwrap(),
            codes,
            parts[3].parse().unwrap(),
        ];
        chains.last_mut().expect("a chain").2.push(row);
    }
    let mut rows = 0;
    for (pf, big, texels) in chains {
        let small = [big[0] / 2, big[1] / 2, 1];
        let l = layout(pf).expect("a layout");
        let mut above = vec![0u8; (big[0] * big[1] * 4) as usize];
        for &[x, y, codes, _] in &texels {
            let taps = model::taps([x, y, 0], big, small, Reciprocal::AppleM4);
            for (k, &(at, _)) in taps[..4].iter().enumerate() {
                above[at as usize * 4 + 3] |= ((codes >> (6 - 2 * k)) as u8 & 3) << 6;
            }
        }
        let (cpu, _) = model::level(&l, &above, big, small, 1, Reciprocal::AppleM4, false);
        let alpha = |level: &[u8], x: u32, y: u32| {
            u32::from(level[((y * small[0] + x) * 4 + 3) as usize] >> 6)
        };
        for &[x, y, codes, stored] in &texels {
            let four = std::array::from_fn(|k| (codes >> (6 - 2 * k)) & 3);
            let qx = axis_tap(x, big[0], small[0], Reciprocal::AppleM4).1;
            let qy = axis_tap(y, big[1], small[1], Reciprocal::AppleM4).1;
            let what = format!("{pf:?} {big:?} ({x}, {y})");
            assert_eq!(
                two_bit(four, qx, qy, true),
                stored,
                "the 10-bit code's: {what}"
            );
            assert_ne!(
                two_bit(four, qx, qy, false),
                stored,
                "the rules part: {what}"
            );
            assert_eq!(alpha(&cpu, x, y), stored, "CPU {what}");
            rows += 1;
        }
        if big[0] != 257 {
            continue;
        }
        let chain = plan(pf, big, 1, 2);
        let (a, b) = (&chain.levels[0], &chain.levels[1]);
        let texel_count = (b.bytes / u64::from(l.bytes)) as u32;
        let row = b.groups[0] * MipmapFilterModule::LOCAL_SIZE;
        let words = params(&l, a.extent, b.extent, texel_count, row, m4).expect("words");
        let kernel = MipmapFilterKernel::new(pf, &l, m4, false, a.extent, None);
        let (gpu, _) = dispatch(
            modules.of(kernel),
            &Dispatch {
                params: &words,
                groups: b.groups,
                above: &above,
                above_bytes: a.bytes,
                bytes: b.bytes,
                kept_in: None,
                kept_out: 0,
                zeroed: chain.zeroed,
                first: None,
            },
        );
        assert_eq!(gpu, cpu, "kernel {pf:?} {big:?}");
    }
    assert_eq!(rows, 1061);
}

/// The values a level keeps: where the next level reads them (an even level below a square of a
/// power of two, every tail level of a format carried through a half), after its params.
#[test]
fn the_plan_keeps_what_the_next_level_reads() {
    let kept = |pf, extent: [u32; 3], levels| {
        let chain = MipmapFilterChain::plan(
            pf,
            extent,
            1,
            levels,
            false,
            compressed(extent),
            MipmapRule::AppleM3MaxPasses,
            64,
            u32::MAX,
            [9, 9],
            None,
        )
        .expect("a chain");
        for l in &chain.levels[1..] {
            if let (Some((params_at, words)), Some((at, _))) = (&l.params, l.kept) {
                assert!(at >= params_at + 4 * words.len() as u64, "after the params");
                assert_eq!(at % 64, 0, "aligned");
            }
        }
        chain
            .levels
            .iter()
            .map(|l| l.kept.map(|(_, bytes)| bytes))
            .collect::<Vec<_>>()
    };
    use PixelFormat as P;
    // 16x16: levels 2 and 4 read the 8x8 and 2x2 squares above them; 16 bytes a texel.
    assert_eq!(
        kept(P::RGBA8Unorm, [16, 16, 1], 5),
        [None, Some(1024), None, Some(64), None]
    );
    // 16x8: no square, nothing kept.
    assert_eq!(kept(P::RGBA8Unorm, [16, 8, 1], 5), [None; 5]);
    // A format carried through a half: every level a tail level reads.
    assert_eq!(
        kept(P::BGR10XR, [12, 10, 1], 4),
        [None, Some(480), Some(96), None]
    );
    // A strip of a format carried through a half: its run reads kept values.
    assert_eq!(kept(P::BGR10XR, [128, 8, 1], 3), [None, Some(4096), None]);
    // Passes: each pass's filter keeps for its second level (levels 1 and 3); the last pass's
    // tail reads the 4x4 square at an even distance below its base, level 2.
    assert_eq!(
        kept(P::RGBA8Unorm, [128, 128, 1], 8),
        [
            None,
            Some(65_536),
            None,
            Some(4096),
            None,
            Some(256),
            None,
            None
        ]
    );
}

/// Every format a fetch reads, through each of its views, over chains that halve and do not,
/// layered, 3D (one texel deep and deeper), squares of powers of two, and one past the tail's
/// extent, from random bytes (every code: snorm's -2^(b-1), half NaNs, infinities and
/// subnormals): the chain that fetches its first level from the device's values is the CPU's
/// chain of the same bytes, and lays out no bytes of the first level.
#[test]
fn a_first_level_fetched_from_its_view_is_the_chain_on_the_cpu() {
    let mut modules = Modules::default();
    let shapes: [([u32; 3], u32, u32); 6] = [
        ([7, 5, 1], 1, 3),
        ([4, 3, 1], 6, 3),
        ([5, 4, 3], 1, 3),
        ([5, 4, 1], 1, 2),
        ([8, 8, 1], 2, 4),
        ([66, 2, 1], 1, 3),
    ];
    let mut views = 0;
    for (seed, &pf) in PixelFormat::ALL.iter().enumerate() {
        let Some(l) = layout(pf) else {
            continue;
        };
        for view in fetch_views(pf) {
            views += 1;
            for (s, &(extent, layers, levels)) in shapes.iter().enumerate() {
                let volume = extent[2] > 1 || s == 3;
                let copied = MipmapFilterChain::plan(
                    pf,
                    extent,
                    layers,
                    levels,
                    volume,
                    compressed(extent),
                    MipmapRule::AppleM3MaxPasses,
                    64,
                    u32::MAX,
                    [65535, 65535],
                    None,
                )
                .expect("a chain");
                let chain = MipmapFilterChain::plan(
                    pf,
                    extent,
                    layers,
                    levels,
                    volume,
                    compressed(extent),
                    MipmapRule::AppleM3MaxPasses,
                    64,
                    u32::MAX,
                    [65535, 65535],
                    Some(view),
                )
                .expect("a chain");
                assert_eq!(chain.fetch, Some(view), "{pf:?} through {view:?}");
                let first = chain.levels[0].bytes.next_multiple_of(64);
                assert_eq!(
                    chain.bytes + first,
                    copied.bytes,
                    "no first level in the scratch"
                );
                let texels = extent.iter().product::<u32>() * layers;
                let first = seeded(
                    (texels * l.bytes) as usize,
                    0x51f1_5eed ^ (seed * 31 + s) as u64,
                );
                let want = chain.expected_levels(&first).expect("levels");
                let got = run_chain(&mut modules, &chain, &first);
                for (level, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g, w,
                        "{pf:?} through {view:?} {extent:?} x{layers} level {level}"
                    );
                }
            }
        }
    }
    assert_eq!(views, 39, "the views of the formats a fetch reads");
}

/// A fetch the chain cannot take, a format whose fetch converts or reads bits, and the view an
/// sRGB image is read through.
#[test]
fn the_plan_fetches_only_what_a_fetch_recovers() {
    let fetched = |pf, view| {
        MipmapFilterChain::plan(
            pf,
            [8, 8, 1],
            1,
            3,
            false,
            compressed([8, 8, 1]),
            MipmapRule::AppleM3MaxPasses,
            64,
            u32::MAX,
            [65535, 65535],
            view,
        )
        .expect("a chain")
        .fetch
    };
    // A view whose components are not the format's fields: none.
    assert_eq!(
        fetched(PixelFormat::RGBA8Unorm, Some(Format::R8G8B8A8_SNORM)),
        None
    );
    assert_eq!(
        fetched(PixelFormat::RGBA16Float, Some(Format::R16G16B16A16_UNORM)),
        None
    );
    // A format no fetch recovers: copied in.
    assert_eq!(
        fetched(PixelFormat::RGBA32Float, Some(Format::R32G32B32A32_SFLOAT)),
        None
    );
    assert_eq!(fetched(PixelFormat::RGBA8Unorm, None), None);
    assert_eq!(
        fetched(PixelFormat::RGBA8Unorm, Some(Format::B8G8R8A8_UNORM)),
        Some(Format::B8G8R8A8_UNORM)
    );
    assert_eq!(
        MipmapFilterChain::fetch_format(Format::B8G8R8A8_SRGB),
        Some(Format::B8G8R8A8_UNORM)
    );
    assert_eq!(MipmapFilterChain::fetch_format(Format::R32_SFLOAT), None);
    assert!(MipmapFilterChain::fetch_converts(
        Format::R16G16B16A16_SFLOAT
    ));
    assert!(MipmapFilterChain::fetch_converts(
        Format::A2B10G10R10_UNORM_PACK32
    ));
    assert!(!MipmapFilterChain::fetch_converts(Format::R32_UINT));
    assert!(!MipmapFilterChain::fetch_converts(
        Format::R16G16B16A16_UINT
    ));
}

/// share-mips-m3d's 10-bit sums (the file's header states them).
const MEASURED_10BIT: &str = include_str!("measured_10bit.txt");

/// A row of `measured_10bit.txt`, `measured_xr.txt` or `measured_widened.txt`: the GPU, the
/// format, the texture's size, the texel, the field, its four tap codes and the code the GPU
/// stored.
type MeasuredSum = (String, PixelFormat, [u32; 2], [u32; 8]);

/// Each row's four taps hold its measured codes in a texture of its size and format, every other
/// texel 0; the rows of one size, format and GPU share a level, each at its own texel. Each GPU's
/// filter on the CPU gives the code the GPU stored at every row: the number of rows.
fn measured_sums(rows: Vec<MeasuredSum>) -> usize {
    let mut groups: HashMap<(String, PixelFormat, [u32; 2]), Vec<[u32; 8]>> = HashMap::new();
    for (gpu, pf, size, row) in rows {
        groups.entry((gpu, pf, size)).or_default().push(row);
    }
    let mut rows = 0;
    for ((gpu, pf, [w, h]), group) in &groups {
        let lay = layout(*pf).expect("a layout");
        let (reciprocal, half) = match gpu.as_str() {
            "m3" => (Reciprocal::AppleM3Max, true),
            _ => (Reciprocal::AppleM4, false),
        };
        let bytes = lay.bytes as usize;
        let (w1, h1) = ((w >> 1).max(1), (h >> 1).max(1));
        let mut level = vec![0u8; *w as usize * *h as usize * bytes];
        let word = |level: &[u8], at: usize| {
            let mut v = [0u8; 16];
            v[..bytes].copy_from_slice(&level[at..at + bytes]);
            u128::from_le_bytes(v)
        };
        for &[x, y, f, c00, c01, c10, c11, _] in group {
            let field = lay.fields[f as usize];
            let (x0, _) = axis_tap(x, *w, w1, reciprocal);
            let (y0, _) = axis_tap(y, *h, h1, reciprocal);
            let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
            for (tx, ty, c) in [(x0, y0, c00), (x1, y0, c01), (x0, y1, c10), (x1, y1, c11)] {
                let at = (ty * w + tx) as usize * bytes;
                let mask = ((1u128 << field.width) - 1) << field.offset;
                let v = (word(&level, at) & !mask) | (u128::from(c) << field.offset);
                level[at..at + bytes].copy_from_slice(&v.to_le_bytes()[..bytes]);
            }
        }
        let (out, _) = model::level(&lay, &level, [*w, *h, 1], [w1, h1, 1], 1, reciprocal, half);
        for &[x, y, f, .., code] in group {
            let field = lay.fields[f as usize];
            let v = word(&out, (y * w1 + x) as usize * bytes) >> field.offset;
            assert_eq!(
                v as u32 & ((1 << field.width) - 1),
                code,
                "{gpu} {pf:?} {w}x{h} ({x}, {y}) field {f}"
            );
            rows += 1;
        }
    }
    rows
}

/// A table of measured sums' rows, `pf` the format of a row without one.
fn measured_rows(table: &str, pf: Option<PixelFormat>) -> Vec<MeasuredSum> {
    table
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|line| {
            let p: Vec<&str> = line.split_whitespace().collect();
            let (format, rest) = match pf {
                Some(pf) => (pf, &p[1..]),
                None => (PixelFormat::from_name(p[1]).expect("a format"), &p[2..]),
            };
            let (w, h) = rest[0].split_once('x').expect("a size");
            let n: Vec<u32> = rest[1..]
                .iter()
                .map(|v| v.parse().expect("a number"))
                .collect();
            (
                p[0].to_owned(),
                format,
                [w.parse().expect("w"), h.parse().expect("h")],
                n.try_into().expect("eight numbers"),
            )
        })
        .collect()
}

/// A 10-bit grid floors each term, a code times its weight, to `2^-14` code before the sum, on an
/// Apple M3 Max and an M4 alike: at every measured sum where that floor and the exact sum part (and
/// at near-tie sums where they agree), each GPU's filter on the CPU gives the code the GPU stored.
#[test]
fn the_10_bit_terms_are_floored_before_the_sum() {
    let rows = measured_rows(MEASURED_10BIT, Some(PixelFormat::RGB10A2Unorm));
    assert_eq!(measured_sums(rows), 1849);
}

/// share-mips-m3d's extended-range sums (the file's header states them).
const MEASURED_XR: &str = include_str!("measured_xr.txt");

/// An extended-range colour floors each term to `2^-13` code before the sum, and alpha to
/// `2^-15`, on an Apple M3 Max (through its half) and an M4 alike: at every measured sum where that
/// floor and the exact sum part, and at near-tie sums where they agree, each GPU's filter on the
/// CPU gives the code the GPU stored.
#[test]
fn the_extended_range_terms_are_floored_before_the_sum() {
    assert_eq!(measured_sums(measured_rows(MEASURED_XR, None)), 690);
}

/// share-mips-m3d's sums of the widened fields (the file's header states them).
const MEASURED_WIDENED: &str = include_str!("measured_widened.txt");

/// `ABGR4Unorm`'s fields filter as the 8-bit codes their bits replicate to, and `A1BGR5Unorm`'s
/// as 10-bit codes, each then stored as its own code nearest the wide grid sum, on an Apple M3 Max
/// (through its half) and an M4 alike: at every measured sum where that rule and each code on its
/// own grid part, and at near-tie sums where they agree, each GPU's filter on the CPU gives the
/// code the GPU stored.
#[test]
fn the_4_5_and_1_bit_fields_filter_as_wider_codes() {
    assert_eq!(measured_sums(measured_rows(MEASURED_WIDENED, None)), 823);
}

/// A filtered half whose sum rounds to zero is +0, on the CPU and in the kernel: a 2x2
/// `RGBA16Float` level of `-(2^-14 + 2^-24)`, `2^-14` and two +0s filters to `-2^-26`, below half
/// the least subnormal.
#[test]
fn a_filtered_half_that_rounds_to_zero_is_plus_zero() {
    let chain = plan(PixelFormat::RGBA16Float, [2, 2, 1], 1, 2);
    let mut first = Vec::new();
    for h in [0x8401u16, 0x0400, 0, 0] {
        for _ in 0..4 {
            first.extend(h.to_le_bytes());
        }
    }
    let cpu = chain.expected_levels(&first).expect("levels");
    assert_eq!(cpu[1], vec![0u8; 8], "CPU");
    assert_eq!(
        run_chain(&mut Modules::default(), &chain, &first),
        cpu,
        "kernel"
    );
}

/// The tail reads an axis of one texel alone, on the CPU and in the kernel: an 8x1 `RGBA16Float`
/// chain's 4x1 level of +inf, 1, -inf and 2 halves to `+inf` and `-inf` (the mix along x only),
/// where a mix along its one-texel y axis at a weight of 0 made each a NaN.
#[test]
fn the_tail_reads_an_axis_of_one_texel_alone() {
    let chain = plan(PixelFormat::RGBA16Float, [8, 1, 1], 1, 3);
    let mut first = Vec::new();
    for h in [
        0x7c00u16, 0x7c00, 0x3c00, 0x3c00, 0xfc00, 0xfc00, 0x4000, 0x4000,
    ] {
        for _ in 0..4 {
            first.extend(h.to_le_bytes());
        }
    }
    let cpu = chain.expected_levels(&first).expect("levels");
    let half =
        |level: &[u8], texel: usize| u16::from_le_bytes([level[8 * texel], level[8 * texel + 1]]);
    // Level 1, the filter of the first: +inf, 1, -inf, 2; level 2, its tail: +inf, -inf.
    assert_eq!(
        (half(&cpu[1], 0), half(&cpu[1], 2)),
        (0x7c00, 0xfc00),
        "level 1"
    );
    assert_eq!(
        (half(&cpu[2], 0), half(&cpu[2], 1)),
        (0x7c00, 0xfc00),
        "level 2"
    );
    assert_eq!(
        run_chain(&mut Modules::default(), &chain, &first),
        cpu,
        "kernel"
    );
}
