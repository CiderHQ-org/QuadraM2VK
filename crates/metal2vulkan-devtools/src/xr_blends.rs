//! The extended-range blend table: Metal's fixed-function blend into each of Apple's extended-range
//! formats, from `half4` and `float4` outputs, under a compact table of blend configurations over
//! fixed synthetic inputs, measured once with plain Metal and committed as
//! `validation/fixtures/public/xr_blend_table.txt`.
//!
//! Each configuration draws into its own band of [`WIDTH`] x [`BAND_HEIGHT`] texels ([`BAND`]
//! source/destination pairs) of a target whose every band holds the same raw texels
//! ([`texels`]), its fragment function returning the same outputs in every band ([`sources`]),
//! under its own scissor and blend color, in one pass that loads and stores the target. The table
//! holds each band's FNV-1a 64 ([`fnv`]) of the plane-0 bytes after the pass, and of a two-plane
//! format's alpha bytes (`round(255 a)` of a `float` read), and the outcome metal2vulkan's
//! pipeline gives each configuration: made, or the `ExtendedRangeBlend` gap it is refused with.
//!
//! The generator is the probe's twin, so both sides draw the same bytes: the product's unit test
//! runs the table through the CPU interpreter, and validation's lane draws it on a device. The
//! table records the digests of the probe's inputs, which both check first.
//!
//! The same probe built without fast math (under `-fmetal-math-mode=safe`; under `relaxed` gave
//! every band the same bytes) is committed beside it as
//! `validation/fixtures/public/xr_blend_table_safe.txt` ([`safe_table`]).
//!
//! The same probe's table of the second source's configurations (factors that read the function's
//! second output, `[[color(0), index(1)]]`, its own outputs [`second_sources`]), built without fast
//! math (under `-fmetal-math-mode=safe`; under `relaxed` gave every band the same bytes), is
//! committed as `validation/fixtures/public/xr_blend_table_dual.txt` ([`dual_table`]).
//!
//! The same configurations and inputs under fast math with the `float4` output computed (the source
//! times a loaded vector of ones), each row's outcome made, `announced` (made,
//! announcing `extended-range-float-blend-source`), or a gap, are committed as
//! `validation/fixtures/public/xr_blend_table_computed.txt` ([`computed_table`]).
//!
//! The second source's function drawn with blending disabled under every channel (its second output
//! read by no blend), one band per format and output type over the same inputs
//! (`-fmetal-math-mode=safe`), is committed as
//! `validation/fixtures/public/xr_blend_table_dual_unblended.txt` ([`dual_unblended_table`]).

use std::path::PathBuf;

/// A band's width in texels.
pub const WIDTH: usize = 64;
/// A band's height in texels.
pub const BAND_HEIGHT: usize = 4;
/// A band's texels: its source/destination pairs.
pub const BAND: usize = WIDTH * BAND_HEIGHT;

/// The probe's generator (xorshift64*, as validation's `two_plane_copies::Rng`).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut state = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x632B_E59B_D9B4_E019);
        if state == 0 {
            state = 1;
        }
        let mut rng = Rng(state);
        for _ in 0..8 {
            rng.next();
        }
        rng
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2_685_821_657_736_338_717)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn in_range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.below(hi - lo + 1)
    }
}

/// The special `half`s: NaN, the infinities, the zeros, the largest finite values, a negative NaN.
const SPECIAL: [u16; 8] = [
    0x7E00, 0x7C00, 0xFC00, 0x8000, 0x0000, 0x7BFF, 0xFBFF, 0xFE00,
];

/// A color component's `half` bits: mostly in the range Metal clamps a blend's source to, near
/// its ends, beyond it, tiny, or special.
fn half_color(r: &mut Rng) -> u16 {
    let k = r.below(100);
    let bits = if k < 30 {
        r.in_range(0x3400, 0x3D00)
    } else if k < 45 {
        r.in_range(0x2000, 0x33FF)
    } else if k < 65 {
        0x8000 | r.in_range(0x3000, 0x3A0A)
    } else if k < 72 {
        let sign = r.below(2) << 15;
        sign | r.below(0x400)
    } else if k < 80 {
        match r.below(2) {
            0 => r.in_range(0x3CFF, 0x3D03),
            _ => r.in_range(0xBA04, 0xBA08),
        }
    } else if k < 90 {
        match r.below(2) {
            0 => r.in_range(0xBA0B, 0xC200),
            _ => r.in_range(0x3D02, 0x4600),
        }
    } else {
        return special(r);
    };
    bits as u16
}

/// An alpha component's `half` bits: mostly in `[0, 1]`, near its ends, beyond it, or special.
fn half_alpha(r: &mut Rng) -> u16 {
    let k = r.below(100);
    let bits = if k < 40 {
        r.in_range(0x3000, 0x3C00)
    } else if k < 55 {
        r.in_range(0x1C00, 0x2FFF)
    } else if k < 62 {
        r.below(0x400)
    } else if k < 70 {
        match r.below(2) {
            0 => 0x0000,
            _ => 0x3C00,
        }
    } else if k < 80 {
        r.in_range(0x3BF0, 0x3C10)
    } else if k < 90 {
        match r.below(2) {
            0 => 0x8000 | r.in_range(0x2000, 0x3C00),
            _ => r.in_range(0x3C01, 0x4400),
        }
    } else {
        return special(r);
    };
    bits as u16
}

fn special(r: &mut Rng) -> u16 {
    let i = r.below(SPECIAL.len() as u64) as usize;
    SPECIAL.get(i).copied().unwrap_or_default()
}

/// The `f32` of `half` bits `h`, exactly (a NaN keeps its payload, as a conversion does).
pub fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exponent = u32::from(h >> 10) & 0x1F;
    let fraction = u32::from(h) & 0x3FF;
    match exponent {
        0 => {
            let magnitude = fraction as f32 * (1.0 / 16_777_216.0);
            f32::from_bits(sign | magnitude.to_bits())
        }
        0x1F => f32::from_bits(sign | 0x7F80_0000 | fraction << 13),
        _ => f32::from_bits(sign | (exponent + 112) << 23 | fraction << 13),
    }
}

/// A `float4` output's component: one in twenty an `f32` within two of an edge of Metal's raw
/// clamp (`-384 / 510`, `638 / 510`) or `+-1e30`; otherwise a `half`'s value, three times in four
/// moved within the gap to the next `half` away from zero.
fn float_bits(r: &mut Rng, alpha: bool) -> u32 {
    if r.below(20) == 0 {
        const EDGE: [f64; 4] = [-384.0 / 510.0, 638.0 / 510.0, 1e30, -1e30];
        let i = r.below(EDGE.len() as u64) as usize;
        let edge = EDGE.get(i).copied().unwrap_or_default() as f32;
        let d = r.below(5) as u32;
        return edge.to_bits().wrapping_add(d).wrapping_sub(2);
    }
    let h = match alpha {
        true => half_alpha(r),
        false => half_color(r),
    };
    let v = half_to_f32(h);
    let mut bits = v.to_bits();
    let k = r.below(4);
    if v.is_finite() && v != 0.0 && k != 0 {
        bits += r.below(1 << 13) as u32;
    }
    bits
}

/// A band's outputs' bytes: [`BAND`] `half4`s (8 bytes each, seed 81) or `float4`s (16 bytes
/// each, seed 83), alpha the fourth component.
pub fn sources(half: bool) -> Vec<u8> {
    outputs(if half { 81 } else { 83 }, half)
}

/// A band's second outputs' bytes (the second source's, `[[color(0), index(1)]]`), drawn as
/// [`sources`] from seeds 85 (`half4`) and 87 (`float4`).
pub fn second_sources(half: bool) -> Vec<u8> {
    outputs(if half { 85 } else { 87 }, half)
}

/// [`BAND`] outputs drawn from `seed`: `half4`s or `float4`s.
fn outputs(seed: u64, half: bool) -> Vec<u8> {
    let mut r = Rng::new(seed);
    let mut out = Vec::with_capacity(BAND * if half { 8 } else { 16 });
    for _ in 0..BAND {
        for c in 0..4 {
            match half {
                true => {
                    let h = match c {
                        3 => half_alpha(&mut r),
                        _ => half_color(&mut r),
                    };
                    out.extend(h.to_le_bytes());
                }
                false => out.extend(float_bits(&mut r, c == 3).to_le_bytes()),
            }
        }
    }
    out
}

/// How a format of Metal's number `raw` lays a texel out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// `BGRA10_XR` (552, 553): four 16-bit fields, blue first, each a code shifted up 6.
    Bgra10,
    /// `BGR10_XR` (554, 555): a word of three codes, red in bits 20 to 29, two top bits.
    Bgr10,
    /// `RGB10A8_2P_XR10` (550, 551): `Bgr10`'s word in plane 0 and an alpha byte in plane 1.
    Planes,
}

impl Layout {
    /// The layout of the extended-range format of Metal's number `raw`.
    pub fn of(raw: u32) -> Option<Layout> {
        match raw {
            550 | 551 => Some(Layout::Planes),
            552 | 553 => Some(Layout::Bgra10),
            554 | 555 => Some(Layout::Bgr10),
            _ => None,
        }
    }

    /// A texel's plane-0 bytes.
    pub fn bytes(self) -> usize {
        match self {
            Layout::Bgra10 => 8,
            _ => 4,
        }
    }
}

fn code(r: &mut Rng, alpha: bool) -> u32 {
    let k = r.below(16);
    let c = if alpha && k < 11 {
        r.in_range(384, 894)
    } else if k < 2 {
        r.below(3)
    } else if k < 4 {
        r.in_range(1021, 1023)
    } else if k < 6 {
        r.in_range(380, 388)
    } else {
        r.below(1024)
    };
    c as u32
}

/// A band's raw texels of the format of Metal's number `raw` (seed `9000 + raw`): plane 0's bytes
/// and a two-plane format's alpha bytes (empty for another).
pub fn texels(raw: u32) -> (Vec<u8>, Vec<u8>) {
    let layout = Layout::of(raw).expect("an extended-range format");
    let mut r = Rng::new(9000 + u64::from(raw));
    let (mut plane, mut alpha) = (Vec::with_capacity(BAND * layout.bytes()), Vec::new());
    for _ in 0..BAND {
        match layout {
            Layout::Bgra10 => {
                for c in 0..4 {
                    let k = code(&mut r, c == 3);
                    let field = (k << 6 | r.below(64) as u32) as u16;
                    plane.extend(field.to_le_bytes());
                }
            }
            _ => {
                let top = (r.below(4) as u32) << 30;
                let (a, b, c) = (
                    code(&mut r, false),
                    code(&mut r, false),
                    code(&mut r, false),
                );
                plane.extend((top | a << 20 | b << 10 | c).to_le_bytes());
                if layout == Layout::Planes {
                    alpha.push(match r.below(8) {
                        0 => 0,
                        1 => 255,
                        _ => r.below(256) as u8,
                    });
                }
            }
        }
    }
    (plane, alpha)
}

/// FNV-1a 64 of `bytes`.
pub fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

/// One configuration: Metal's numbers of the color and alpha factors and operations, the write
/// mask (red 8, green 4, blue 2, alpha 1), and the blend color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Source and destination color factors, color operation, source and destination alpha
    /// factors, alpha operation (`MTLBlendFactor`, `MTLBlendOperation`).
    pub equation: [u64; 6],
    /// `MTLColorWriteMask`.
    pub mask: u64,
    /// `setBlendColorRed:green:blue:alpha:`.
    pub color: [f32; 4],
}

/// One case's inputs' digests: the format's texels (plane 0 and alpha bytes, 0 where it has
/// none) and the outputs, and in the second source's table its second outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inputs {
    pub format: u32,
    pub half: bool,
    pub plane: u64,
    pub alpha: u64,
    pub sources: u64,
    /// FNV-1a 64 of [`second_sources`] (the second source's table alone).
    pub seconds: Option<u64>,
}

/// One configuration's band in one case: Metal's digests and metal2vulkan's outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    /// Metal's number of the format.
    pub format: u32,
    /// Whether the output is `half4` (else `float4`).
    pub half: bool,
    /// The configuration's index.
    pub config: usize,
    /// FNV-1a 64 of the band's plane-0 bytes after the pass.
    pub plane: u64,
    /// FNV-1a 64 of the band's alpha bytes after the pass (0 for a format without them).
    pub alpha: u64,
    /// `made`, or the name of the `ExtendedRangeBlend` gap metal2vulkan refuses it with.
    pub ours: String,
}

/// The committed table.
#[derive(Clone, Debug, Default)]
pub struct Table {
    pub configs: Vec<Config>,
    pub inputs: Vec<Inputs>,
    pub rows: Vec<Row>,
}

/// `validation/fixtures/public/xr_blend_table.txt`.
pub fn table_path() -> PathBuf {
    crate::fixtures_dir().join("xr_blend_table.txt")
}

fn hex64(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn number(s: &str) -> u64 {
    s.parse().unwrap_or_else(|e| panic!("{s}: {e}"))
}

/// `validation/fixtures/public/xr_blend_table_safe.txt`.
pub fn safe_table_path() -> PathBuf {
    crate::fixtures_dir().join("xr_blend_table_safe.txt")
}

/// The committed table, parsed ([`parse`]).
pub fn table() -> Table {
    parse(table_path())
}

/// The committed table of the probe built without fast math, parsed ([`parse`]).
pub fn safe_table() -> Table {
    parse(safe_table_path())
}

/// `validation/fixtures/public/xr_blend_table_dual.txt`.
pub fn dual_table_path() -> PathBuf {
    crate::fixtures_dir().join("xr_blend_table_dual.txt")
}

/// The committed table of the second source's configurations without fast math, parsed
/// ([`parse`]).
pub fn dual_table() -> Table {
    parse(dual_table_path())
}

/// `validation/fixtures/public/xr_blend_table_computed.txt`.
pub fn computed_table_path() -> PathBuf {
    crate::fixtures_dir().join("xr_blend_table_computed.txt")
}

/// The committed table of a computed `float4` output under fast math, parsed ([`parse`]).
pub fn computed_table() -> Table {
    parse(computed_table_path())
}

/// `validation/fixtures/public/xr_blend_table_dual_unblended.txt`.
pub fn dual_unblended_table_path() -> PathBuf {
    crate::fixtures_dir().join("xr_blend_table_dual_unblended.txt")
}

/// The committed table of the second source's function drawn unblended, parsed ([`parse`]).
pub fn dual_unblended_table() -> Table {
    parse(dual_unblended_table_path())
}

/// The table at `path`, parsed: `cfg <index> <6 equation numbers> <mask> <4 f32 bits>`, `inputs
/// <format> <h|f> <plane> <alpha> <sources> [<second sources>]`, `row <format> <h|f> <index>
/// <plane> <alpha> <ours>`; `#` starts a comment.
pub fn parse(path: PathBuf) -> Table {
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut t = Table::default();
    for line in text
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let at = |i: usize| *f.get(i).unwrap_or_else(|| panic!("{line}: field {i}"));
        match at(0) {
            "cfg" => {
                assert_eq!(number(at(1)) as usize, t.configs.len(), "{line}");
                t.configs.push(Config {
                    equation: std::array::from_fn(|k| number(at(2 + k))),
                    mask: number(at(8)),
                    color: std::array::from_fn(|k| f32::from_bits(hex64(at(9 + k)) as u32)),
                });
            }
            "inputs" => t.inputs.push(Inputs {
                format: number(at(1)) as u32,
                half: at(2) == "h",
                plane: hex64(at(3)),
                alpha: hex64(at(4)),
                sources: hex64(at(5)),
                seconds: f.get(6).map(|s| hex64(s)),
            }),
            "row" => t.rows.push(Row {
                format: number(at(1)) as u32,
                half: at(2) == "h",
                config: number(at(3)) as usize,
                plane: hex64(at(4)),
                alpha: hex64(at(5)),
                ours: at(6).to_owned(),
            }),
            other => panic!("{}: unknown line kind {other}", path.display()),
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator draws the probe's inputs (the probe's digests in the table, in the table
    /// without fast math, in the second source's, the second outputs' too, and its `float4` ones in
    /// the computed table), and each table names every configuration of every case once, the same
    /// configurations in each but the second source's.
    #[test]
    fn the_inputs_are_the_probes() {
        let (fast, safe, dual) = (table(), safe_table(), dual_table());
        // The second source's table: the same texels and outputs, and its second outputs.
        assert!(dual.inputs.iter().all(|i| i.seconds.is_some()));
        let firsts: Vec<Inputs> = dual
            .inputs
            .iter()
            .map(|i| Inputs {
                seconds: None,
                ..*i
            })
            .collect();
        assert_eq!(firsts, fast.inputs);
        the_table_is_the_probes(&dual);
        // A blend color may be a NaN: the configurations compare by bits.
        let configs = |t: &Table| {
            t.configs
                .iter()
                .map(|c| (c.equation, c.mask, c.color.map(f32::to_bits)))
                .collect::<Vec<_>>()
        };
        assert_eq!(configs(&fast), configs(&safe));
        assert_eq!(fast.inputs, safe.inputs);
        // The computed table: the same configurations, and the `float4` cases' inputs.
        let computed = computed_table();
        assert_eq!(configs(&fast), configs(&computed));
        let floats: Vec<Inputs> = fast.inputs.iter().filter(|i| !i.half).copied().collect();
        assert_eq!(floats, computed.inputs);
        // The unblended second outputs: the second source's inputs, one configuration.
        let unblended = dual_unblended_table();
        assert_eq!(unblended.inputs, dual.inputs);
        assert_eq!(unblended.configs.len(), 1);
        the_table_is_the_probes(&unblended);
        for t in [fast, safe, computed] {
            the_table_is_the_probes(&t);
        }
    }

    fn the_table_is_the_probes(t: &Table) {
        assert!(!t.configs.is_empty() && !t.inputs.is_empty());
        for i in &t.inputs {
            let (plane, alpha) = texels(i.format);
            assert_eq!(fnv(&plane), i.plane, "{} plane 0", i.format);
            let a = if alpha.is_empty() { 0 } else { fnv(&alpha) };
            assert_eq!(a, i.alpha, "{} alpha", i.format);
            assert_eq!(fnv(&sources(i.half)), i.sources, "half {}", i.half);
            if let Some(s) = i.seconds {
                assert_eq!(fnv(&second_sources(i.half)), s, "second, half {}", i.half);
            }
            let rows = t
                .rows
                .iter()
                .filter(|r| r.format == i.format && r.half == i.half);
            let mut seen: Vec<usize> = rows.map(|r| r.config).collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..t.configs.len()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn halves_convert_exactly() {
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xBA06), -1542.0 / 2048.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(half_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(half_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(half_to_f32(0x7E00).to_bits(), 0x7FC0_0000);
        assert_eq!(half_to_f32(0xFE00).to_bits(), 0xFFC0_0000);
        assert_eq!(half_to_f32(0x7BFF), 65504.0);
    }
}
