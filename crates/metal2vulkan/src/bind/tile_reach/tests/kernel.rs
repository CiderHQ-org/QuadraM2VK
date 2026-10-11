//! The kernel against the reference function, through the CPU interpreter: built and valid
//! where a device has `shaderInt64` and refused where it does not; its arithmetic Rust's (the
//! division, the snap and a point's extent); and its mask and counts the rule's on every measured
//! case (the M4's tiles themselves where the case's answer is its primitives' reach) and on
//! seeded random passes.

use std::collections::BTreeSet;
use std::path::PathBuf;

use super::reference::{
    clip, point_extent, reach, snap_clamped, Cull, Flags, Primitive, Raster, Winding,
    POINT_SIZE_LIMIT,
};
use super::{clip_in, measured, points, Draw, Other, Row};
use crate::bind::tile_reach::kernel::{
    arithmetic_module, clip_module, typed_module, ARITHMETIC_WORDS, CLIP_WORDS, PASS, POLYGON,
    PRIMITIVES, TILES,
};
use crate::bind::tile_reach::{tile_reach_module, TileReachModule as M};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, Needs};
use crate::pipeline::sym::ResourceSlot;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::assert_valid;

/// Every committed device (`validation/profiles/*.json`), derived with the default policy, by
/// file stem.
fn committed_devices() -> Vec<(String, Device)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("profiles") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("read");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let device = Device::new(desc, DevicePolicy::default()).expect("device");
        out.push((stem, device));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// On every committed device the module is built, valid and within the profile where the device
/// has `shaderInt64`, and refused as `Feature::TileReach` needing it where it does not.
#[test]
fn the_module_is_valid_with_int64_and_refused_without() {
    let (mut built, mut refused) = (0, 0);
    for (name, device) in committed_devices() {
        let profile = device.profile();
        let int64 = profile
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderInt64);
        match tile_reach_module(&device) {
            Ok(m) => {
                assert!(int64, "{name}");
                assert!(!m.spirv.is_empty(), "{name}");
                assert!(profile.check(&m.requirements).is_ok(), "{name}");
                assert_valid(typed_module().expect("module"), profile);
                built += 1;
            }
            Err(Error::Unsupported(u)) => {
                assert!(!int64, "{name}");
                assert_eq!(u.feature, Feature::TileReach, "{name}");
                let needs = vec![vec![Capability::Feature(VkFeature::ShaderInt64)]];
                assert_eq!(u.needs, Needs::AnyOf(needs), "{name}");
                refused += 1;
            }
            Err(other) => panic!("{name}: {other}"),
        }
    }
    assert!(built > 0 && refused > 0, "{built} built, {refused} refused");
}

/// A splitmix64 stream.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn word(&mut self) -> u32 {
        self.next() as u32
    }

    /// A number in `[0, n)`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// True `k` times in `n`.
    fn chance(&mut self, k: u64, n: u64) -> bool {
        self.below(n) < k
    }

    pub(super) fn pick<X: Copy>(&mut self, xs: &[X]) -> X {
        xs[self.below(xs.len() as u64) as usize]
    }

    /// A number in `[lo, hi)` on a grid of `1 / steps`.
    fn grid(&mut self, lo: f64, hi: f64, steps: f64) -> f64 {
        let n = ((hi - lo) * steps).max(1.0) as u64;
        lo + self.below(n) as f64 / steps
    }
}

/// Bits the arithmetic meets at its edges: zeros, subnormals, normals' bounds, the snap's limit
/// and its whole-number threshold, infinities and a NaN.
const EDGE_BITS: [u32; 20] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0001,
    0x0040_0000,
    0x007f_ffff,
    0x0080_0000,
    0x0080_0001,
    0x3f00_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3fc0_0000,
    0x4a00_0000,
    0x49ff_ffff,
    0x4e00_0000,
    0x4dff_ffff,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
];

/// A point size in `[0, 511]` the extent's rounding meets: on its ties (an odd number of units,
/// an odd number of 1/32 pixels), beside them, or anywhere.
fn size_bits(rng: &mut Rng) -> u32 {
    let size = match rng.below(5) {
        0 => (2 * rng.below(256 * 255) + 1) as f32 / 256.0,
        1 => (2 * rng.below(16 * 511) + 1) as f32 / 32.0,
        2 => {
            let tie = (2 * rng.below(16 * 511) + 1) as f32 / 32.0;
            let s = f32::from_bits(tie.to_bits().wrapping_add(rng.pick(&[1, u32::MAX])));
            s.clamp(0.0, POINT_SIZE_LIMIT)
        }
        3 => f32::from_bits(rng.word() & 0x43ff_ffff).min(POINT_SIZE_LIMIT),
        _ => rng.grid(0.0, 511.0, 4096.0) as f32,
    };
    size.to_bits()
}

/// The kernel's division, multiplication and addition give Rust's `/`, `*` and `+` (a NaN for a
/// NaN), and its snap and point extent the reference's, bit for bit, on edge operands (zeros,
/// subnormals, the largest finite values, infinities, a NaN) and on 6000 random ones (any bits;
/// near-equal exponents, where a sum cancels; coordinates on and beside the snap's ties).
#[test]
fn the_kernels_arithmetic_is_the_references() {
    let mut rng = Rng(0x7ea1_0d71);
    let mut records: Vec<[u32; 3]> = Vec::new();
    for x in EDGE_BITS {
        for y in EDGE_BITS {
            records.push([x, y, size_bits(&mut rng)]);
        }
    }
    for _ in 0..6000 {
        let x = match rng.below(4) {
            0 => rng.word(),
            // A coordinate on a tie of the snap (k + 1/2 units), or one bit beside one.
            1 => {
                let tie = (rng.below(1 << 22) as f64 * 2.0 + 1.0) / 512.0;
                let tie = tie as f32 * rng.pick(&[1.0, -1.0]);
                tie.to_bits().wrapping_add(rng.pick(&[0, 1, u32::MAX]))
            }
            // Exponents about the snap's thresholds.
            2 => ((rng.below(40) as u32 + 115) << 23) | (rng.word() & 0x807f_ffff),
            _ => (rng.grid(-1e5, 1e5, 65536.0) as f32).to_bits(),
        };
        let y = match rng.below(3) {
            0 => rng.word(),
            1 => (x & 0xff80_0000) ^ (rng.word() & 0x807f_ffff),
            _ => (rng.grid(0.01, 16.0, 1024.0) as f32).to_bits(),
        };
        records.push([x, y, size_bits(&mut rng)]);
    }
    let words = ARITHMETIC_WORDS as usize;
    let mut bytes = Vec::with_capacity(records.len() * words * 4);
    for r in &records {
        for w in 0..words {
            bytes.extend(r.get(w).copied().unwrap_or(0).to_le_bytes());
        }
    }
    let m = arithmetic_module().expect("module");
    let cfg = Config {
        workgroups: [(records.len() as u32).div_ceil(M::LOCAL_SIZE), 1, 1],
        step_limit: 1 << 40,
        ..Config::default()
    };
    let res = Resources {
        buffers: vec![buffer(PRIMITIVES, bytes)],
        push: Vec::new(),
        spec: Default::default(),
    };
    let out = run(&m, &cfg, res).expect("run").buffers.swap_remove(0);
    let mut wrong = Vec::new();
    for (i, r) in records.iter().enumerate() {
        let w = |k: usize| word(&out, i * words + k);
        let (x, y, size) = (
            f32::from_bits(r[0]),
            f32::from_bits(r[1]),
            f32::from_bits(r[2]),
        );
        for (what, at, want) in [("/", 3, x / y), ("*", 4, x * y), ("+", 5, x + y)] {
            let got = f32::from_bits(w(at));
            if !(want.to_bits() == got.to_bits() || (want.is_nan() && got.is_nan())) {
                wrong.push(format!(
                    "{x:e} {what} {y:e} ({:#x} {what} {:#x}): {got:e} ({:#x}), not {want:e} \
                     ({:#x})",
                    r[0],
                    r[1],
                    w(at),
                    want.to_bits()
                ));
            }
        }
        let snapped = (i64::from(w(7)) << 32) | i64::from(w(6));
        let measured = w(8) == 1;
        if (snapped, measured) != snap_clamped(x) {
            let want = snap_clamped(x);
            wrong.push(format!("snap {x:e}: {snapped} {measured}, not {want:?}"));
        }
        let extent = (i64::from(w(9)), i64::from(w(10)));
        if extent != point_extent(size) {
            let want = point_extent(size);
            wrong.push(format!("point size {size:e}: {extent:?}, not {want:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The clipper holds the most vertices a polygon reaches ([`POLYGON`], 9): a hexagon whose
/// vertices lie by turns in front of the eye and behind it clips by the eye's plane to 9, which
/// only a polygon the clipper's rounding left not convex can be; and on seeded polygons of 2 to 6
/// vertices, by each plane, the clipper's polygon is the reference's, bit for bit.
#[test]
fn the_clipper_holds_its_most_vertices() {
    let planes: [fn(&[f32; 4]) -> f32; 3] = [|v| v[2], |v| v[3] - v[2], |v| v[3]];
    let hexagon: Vec<[f32; 4]> = (0..6)
        .map(|j| {
            let a = j as f32 * std::f32::consts::FRAC_PI_3;
            [a.cos(), a.sin(), 0.5, if j % 2 == 0 { 1.0 } else { -0.5 }]
        })
        .collect();
    assert_eq!(clip(&hexagon, planes[2]).len(), POLYGON as usize);
    let mut cases = vec![(2, hexagon)];
    let mut rng = Rng(0xc11b_0009);
    for _ in 0..600 {
        let n = rng.pick(&[2, 3, 4, 5, 6]);
        let plane = rng.below(3) as u32;
        let polygon = (0..n)
            .map(|_| [0; 4].map(|_| rng.grid(-2.0, 2.0, 64.0) as f32))
            .collect();
        cases.push((plane, polygon));
    }
    let words = CLIP_WORDS as usize;
    let mut bytes = Vec::with_capacity(cases.len() * words * 4);
    for (plane, polygon) in &cases {
        let mut record = vec![0u32; words];
        record[0] = *plane;
        record[1] = polygon.len() as u32;
        for (j, v) in polygon.iter().enumerate() {
            for (c, x) in v.iter().enumerate() {
                record[2 + 4 * j + c] = x.to_bits();
            }
        }
        bytes.extend(record.iter().flat_map(|w| w.to_le_bytes()));
    }
    let m = clip_module().expect("module");
    let cfg = Config {
        workgroups: [(cases.len() as u32).div_ceil(M::LOCAL_SIZE), 1, 1],
        step_limit: 1 << 40,
        ..Config::default()
    };
    let res = Resources {
        buffers: vec![buffer(PRIMITIVES, bytes)],
        push: Vec::new(),
        spec: Default::default(),
    };
    let out = run(&m, &cfg, res).expect("run").buffers.swap_remove(0);
    let (mut wrong, mut most) = (Vec::new(), 0);
    for (i, (plane, polygon)) in cases.iter().enumerate() {
        let w = |k: usize| word(&out, i * words + k);
        let want = clip(polygon, planes[*plane as usize]);
        let got: Vec<[f32; 4]> = (0..w(1) as usize)
            .map(|j| [0, 1, 2, 3].map(|c| f32::from_bits(w(2 + 4 * j + c))))
            .collect();
        let bits =
            |p: &[[f32; 4]]| -> Vec<u32> { p.iter().flatten().map(|x| x.to_bits()).collect() };
        if bits(&got) != bits(&want) {
            wrong.push(format!(
                "plane {plane} of {polygon:?}: {got:?}, not {want:?}"
            ));
        }
        most = most.max(got.len());
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(most, POLYGON as usize);
}

fn buffer(slot: ResourceSlot, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: Some(slot),
        address: None,
        bytes,
    }
}

/// Word `i` of little-endian bytes.
fn word(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().expect("a word"))
}

/// One primitive of a pass: its reference form, the kind the kernel reads (its own, or one that
/// is none), the index of its state and its layer.
#[derive(Clone, Copy, Debug)]
pub(super) struct Drawn {
    prim: Primitive,
    kind: u32,
    state: u32,
    layer: u32,
}

impl Drawn {
    pub(super) fn of(prim: Primitive, state: u32, layer: u32) -> Drawn {
        let kind = match prim {
            Primitive::Point(..) => M::POINT,
            Primitive::Line(_) => M::LINE,
            Primitive::Triangle(_) => M::TRIANGLE,
        };
        Drawn {
            prim,
            kind,
            state,
            layer,
        }
    }

    pub(super) fn record(&self) -> [u8; 64] {
        let none = [0.0; 4];
        let (size, vertices) = match self.prim {
            Primitive::Point(v, size) => (size, [v, none, none]),
            Primitive::Line([a, b]) => (1.0, [a, b, none]),
            Primitive::Triangle(t) => (1.0, t),
        };
        M::primitive(self.kind, self.state, self.layer, size, vertices)
    }
}

/// A pass the kernel bins: its render area, samples and layers, the states its primitives were
/// drawn under (their area and samples the pass's), and its primitives.
#[derive(Clone, Debug)]
pub(super) struct Pass {
    pub(super) area: [u32; 2],
    pub(super) samples: u32,
    pub(super) layers: u32,
    pub(super) states: Vec<Raster>,
    pub(super) drawn: Vec<Drawn>,
}

/// What a pass binned: the tiles `[x, y]` each slice reached, and the primitives counted near-far
/// and unmeasured.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Binned {
    pub(super) slices: Vec<BTreeSet<[u32; 2]>>,
    pub(super) near_far: u32,
    pub(super) unmeasured: u32,
}

pub(super) fn cull(c: Cull) -> u32 {
    match c {
        Cull::None => 0,
        Cull::Front => 1,
        Cull::Back => 2,
    }
}

pub(super) fn winding(w: Winding) -> u32 {
    match w {
        Winding::Clockwise => 0,
        Winding::CounterClockwise => 1,
    }
}

impl Pass {
    /// The tiles of a slice's row and column.
    pub(super) fn grid(&self) -> [u32; 2] {
        let th = if self.samples == 4 { 16 } else { 32 };
        [self.area[0].div_ceil(32), self.area[1].div_ceil(th)]
    }

    /// The rule's answer: each primitive's reach by the reference function, into the slice it
    /// renders (a layer past the pass's renders its first); a state past the pass's or a kind
    /// that is none reach nothing, flagged unmeasured.
    pub(super) fn rule(&self) -> Binned {
        let mut out = Binned {
            slices: vec![BTreeSet::new(); self.layers as usize],
            near_far: 0,
            unmeasured: 0,
        };
        for d in &self.drawn {
            let mut flags = Flags::default();
            match self.states.get(d.state as usize) {
                Some(_) if d.kind > M::TRIANGLE => flags.unmeasured = true,
                None => flags.unmeasured = true,
                Some(state) => {
                    let mut raster = *state;
                    raster.area = self.area;
                    raster.samples = self.samples;
                    let tiles = reach(&raster, &d.prim, &mut flags);
                    let slice = if d.layer < self.layers { d.layer } else { 0 };
                    out.slices[slice as usize].extend(tiles);
                }
            }
            out.near_far += u32::from(flags.near_far);
            out.unmeasured += u32::from(flags.unmeasured);
        }
        out
    }

    /// The kernel's answer, through the interpreter in both invocation orders (alike).
    fn kernel(&self, m: &TypedModule) -> Binned {
        let prims: Vec<u8> = self.drawn.iter().flat_map(|d| d.record()).collect();
        let count = self.drawn.len() as u32;
        let mut pass = M::header(self.area, self.samples, self.layers, count).to_vec();
        for s in &self.states {
            pass.extend(M::state(
                s.viewport,
                s.scissor,
                cull(s.cull),
                winding(s.winding),
                u32::from(s.clamp),
            ));
        }
        let words = M::mask_words(self.area, self.samples, self.layers).max(1) as usize;
        let mut outs = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![
                    buffer(PRIMITIVES, prims.clone()),
                    buffer(PASS, pass.clone()),
                    buffer(TILES, vec![0; 4 * words]),
                ],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [count.div_ceil(M::LOCAL_SIZE).max(1), 1, 1],
                step_limit: 1 << 40,
                ..Config::default()
            };
            outs.push(run(m, &cfg, res).expect("run").buffers);
        }
        assert_eq!(outs[0], outs[1], "order independent");
        let buffers = &outs[0];
        let [gx, gy] = self.grid().map(|g| g as usize);
        let mut slices = vec![BTreeSet::new(); self.layers as usize];
        for i in 0..words {
            let w = word(&buffers[2], i);
            for bit in (0..32).filter(|b| w >> b & 1 == 1) {
                let tile = 32 * i + bit;
                let (slice, at) = (tile / (gx * gy), tile % (gx * gy));
                assert!(slice < slices.len(), "a bit past the pass's tiles");
                slices[slice].insert([(at % gx) as u32, (at / gx) as u32]);
            }
        }
        Binned {
            slices,
            near_far: word(&buffers[1], M::NEAR_FAR_WORD as usize),
            unmeasured: word(&buffers[1], M::UNMEASURED_WORD as usize),
        }
    }

    /// Where the kernel and the rule differ: each primitive that differs alone, else the pass.
    pub(super) fn differences(&self, m: &TypedModule) -> Vec<String> {
        if self.kernel(m) == self.rule() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for d in &self.drawn {
            let alone = Pass {
                drawn: vec![*d],
                ..self.clone()
            };
            let (kernel, rule) = (alone.kernel(m), alone.rule());
            if kernel != rule {
                let state = self.states.get(d.state as usize);
                out.push(format!(
                    "{d:?} under {state:?} (area {:?}, {} samples): kernel {kernel:?}, rule {rule:?}",
                    self.area, self.samples
                ));
            }
        }
        if out.is_empty() {
            out.push(format!("the pass {self:?} differs, no primitive alone"));
        }
        out
    }
}

/// The pass of a measured case: its draw's primitives, under its one state.
fn pass_of(row: &Row) -> Pass {
    Pass {
        area: row.raster.area,
        samples: row.raster.samples,
        layers: row.slices,
        states: vec![row.raster],
        drawn: row
            .primitives()
            .into_iter()
            .map(|p| Drawn::of(p, 0, row.layer))
            .collect(),
    }
}

/// On every measured case drawn directly or indirectly, the kernel bins what the rule does, and,
/// where the case's answer is its primitives' reach (no attachment cleared and stored), the
/// tiles the M4 reached.
#[test]
fn the_kernel_reaches_every_measured_answer() {
    let m = typed_module().expect("module");
    let mut rows = measured::rows();
    rows.extend(measured::later_rows());
    rows.extend(points::rows());
    let mut wrong = Vec::new();
    let mut checked = 0;
    for row in rows
        .iter()
        .filter(|r| matches!(r.draw, Draw::Direct | Draw::Indirect))
    {
        let pass = pass_of(row);
        let kernel = pass.kernel(&m);
        let rule = pass.rule();
        if kernel != rule {
            wrong.push(format!(
                "{}/{}: kernel {kernel:?}, rule {rule:?}",
                row.probe, row.name
            ));
        }
        if matches!(row.other, Other::None | Other::ColorClearDontCare)
            && kernel.slices != row.measured()
        {
            wrong.push(format!(
                "{}/{}: kernel {:?}, measured {:?}",
                row.probe,
                row.name,
                kernel.slices,
                row.measured()
            ));
        }
        checked += 1;
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert!(checked > 300, "{checked} cases");
}

/// A random state over a render area of `area`: Metal's defaults, or a viewport and a scissor of
/// its own (past the area too, the viewport flipped at times), any cull mode and winding, depth
/// clipped or clamped.
fn state(rng: &mut Rng, area: [u32; 2]) -> Raster {
    let mut r = Raster::over(area);
    let [w, h] = area.map(f64::from);
    if rng.chance(1, 2) {
        let ox = rng.grid(-w / 4.0, w, 512.0);
        let oy = rng.grid(-h / 4.0, h, 512.0);
        let vw = rng.grid(1.0, 1.5 * w + 1.0, 512.0);
        let vh = rng.grid(1.0, 1.5 * h + 1.0, 512.0) * rng.pick(&[1.0, 1.0, 1.0, -1.0]);
        r.viewport = [ox, oy, vw, vh].map(|c| c as f32);
    }
    if rng.chance(1, 2) {
        let sx = rng.below(u64::from(area[0]) + 8) as u32;
        let sy = rng.below(u64::from(area[1]) + 8) as u32;
        let sw = rng.below(u64::from(area[0]) + 8) as u32;
        let sh = rng.below(u64::from(area[1]) + 8) as u32;
        r.scissor = [sx, sy, sw, sh];
    }
    r.cull = rng.pick(&[Cull::None, Cull::Front, Cull::Back]);
    r.winding = rng.pick(&[Winding::Clockwise, Winding::CounterClockwise]);
    r.clamp = rng.chance(1, 3);
    r
}

/// A random pixel position about a render area of `area`: on or beside a tile's corner (within
/// 1/64 of a pixel, on 1/512 steps), anywhere about the area on 1/1024 steps, or far past it.
fn position(rng: &mut Rng, area: [u32; 2], tile: [u32; 2]) -> [f64; 2] {
    let near = |rng: &mut Rng, extent: u32, size: u32| {
        let k = rng.below(u64::from(extent / size) + 3) as f64 - 1.0;
        k * f64::from(size) + (rng.below(17) as f64 - 8.0) / 512.0
    };
    let [w, h] = area.map(f64::from);
    match rng.below(10) {
        0..=3 => [near(rng, area[0], tile[0]), near(rng, area[1], tile[1])],
        4..=8 => [
            rng.grid(-40.0, w + 40.0, 1024.0),
            rng.grid(-40.0, h + 40.0, 1024.0),
        ],
        _ => [
            rng.grid(-2e5, 2e5, 4.0) * rng.pick(&[1.0, 1e-3]),
            rng.grid(-2e5, 2e5, 4.0) * rng.pick(&[1.0, 1e-3]),
        ],
    }
}

/// The clip-space vertex at pixel `p` of a frame of `area` and depth `z`, at `w` (its
/// coordinates scaled by it), each rounded to `f32`.
fn vertex(area: [u32; 2], p: [f64; 2], z: f64, w: f64) -> [f32; 4] {
    let [x, y, _, _] = clip_in(area, p[0], p[1], z);
    [
        (f64::from(x) * w) as f32,
        (f64::from(y) * w) as f32,
        (z * w) as f32,
        w as f32,
    ]
}

/// A random depth and `w` for a primitive's vertices: within the depth range at `w = 1` mostly,
/// at times in perspective or across the near or far plane.
fn depth(rng: &mut Rng) -> (f64, f64) {
    let z = match rng.chance(1, 5) {
        true => rng.grid(-1.0, 2.0, 256.0),
        false => rng.grid(0.0, 1.0, 256.0),
    };
    let w = match rng.chance(1, 3) {
        true => rng.pick(&[0.5, 2.0, 3.0, 0.3, 7.25, 1.0e3]) * rng.grid(0.5, 2.0, 64.0),
        false => 1.0,
    };
    (z, w)
}

/// `v` with one component at times replaced by a value the rule flags: not finite, subnormal,
/// huge, or a `w` at or behind the eye.
fn spoiled(rng: &mut Rng, mut v: [f32; 4]) -> [f32; 4] {
    if rng.chance(1, 30) {
        let c = rng.below(4) as usize;
        v[c] = rng.pick(&[
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0e-40,
            -1.0e-40,
            1.0e30,
            -1.0e30,
            0.0,
            -1.0,
        ]);
    }
    v
}

/// A random point size: fractional, on the extent's ties, zero, past the largest, or flagged
/// (negative, not a number, subnormal).
fn point_size(rng: &mut Rng) -> f32 {
    match rng.below(8) {
        0 => rng.pick(&[0.0, 0.25, 1.0, 511.0, 511.5, 600.0, f32::INFINITY]),
        1 => rng.pick(&[f32::NAN, -1.0, 1.0e-40, -0.0]),
        2 => (2 * rng.below(2048) + 1) as f32 / 32.0,
        3 => (2 * rng.below(8192) + 1) as f32 / 256.0,
        _ => rng.grid(0.0, 80.0, 512.0) as f32,
    }
}

/// A random primitive about a render area of `area`: a point, a line (at times of no length or
/// at 45 degrees), or a triangle (at times small, for the small-primitive cull).
fn primitive(rng: &mut Rng, area: [u32; 2], tile: [u32; 2]) -> Primitive {
    let (z, w) = depth(rng);
    let at = |rng: &mut Rng, p: [f64; 2]| {
        let (dz, dw) = match rng.chance(1, 4) {
            true => depth(rng),
            false => (z, w),
        };
        spoiled(rng, vertex(area, p, dz, dw))
    };
    let p = position(rng, area, tile);
    match rng.below(10) {
        0..=2 => Primitive::Point(at(rng, p), point_size(rng)),
        3..=5 => {
            let q = match rng.below(6) {
                0 => p,
                1 => {
                    let d = rng.grid(-40.0, 40.0, 256.0);
                    [p[0] + d, p[1] + d * rng.pick(&[1.0, -1.0])]
                }
                _ => position(rng, area, tile),
            };
            Primitive::Line([at(rng, p), at(rng, q)])
        }
        _ => {
            let small = rng.chance(1, 2);
            let near = |rng: &mut Rng| match small {
                true => [
                    p[0] + rng.grid(-4.0, 4.0, 64.0),
                    p[1] + rng.grid(-4.0, 4.0, 64.0),
                ],
                false => position(rng, area, tile),
            };
            let (b, c) = (near(rng), near(rng));
            Primitive::Triangle([at(rng, p), at(rng, b), at(rng, c)])
        }
    }
}

/// A random pass of `area` and `samples`: a few states; each primitive in a slice of its own
/// (slice 0 holding those whose layer is past the pass's), a few under a state past the pass's
/// or of a kind that is none.
pub(super) fn random_pass(rng: &mut Rng, area: [u32; 2], samples: u32, count: u32) -> Pass {
    let states: Vec<Raster> = (0..1 + rng.below(3)).map(|_| state(rng, area)).collect();
    let tile = [32, if samples == 4 { 16 } else { 32 }];
    let layers = count + 1;
    let drawn = (0..count)
        .map(|i| {
            let prim = primitive(rng, area, tile);
            let state = match rng.chance(1, 40) {
                true => states.len() as u32 + rng.below(3) as u32,
                false => rng.below(states.len() as u64) as u32,
            };
            let layer = match rng.chance(1, 20) {
                true => layers + rng.below(4) as u32,
                false => i + 1,
            };
            let mut d = Drawn::of(prim, state, layer);
            if rng.chance(1, 40) {
                d.kind = rng.pick(&[3, 4, u32::MAX]);
            }
            d
        })
        .collect();
    Pass {
        area,
        samples,
        layers,
        states,
        drawn,
    }
}

/// On seeded random passes (render areas of a pixel to a few tiles, 1, 2 and 4 samples and
/// other counts, viewports, scissors, culls and windings, depth clipped and clamped; points,
/// lines and triangles about tile edges, in perspective, across the near and far planes, with
/// values the rule flags), the kernel bins what the rule does.
#[test]
fn the_kernel_bins_what_the_rule_does_on_random_passes() {
    let m = typed_module().expect("module");
    let mut rng = Rng(0x0d67_1a2b);
    let areas = [
        [128, 128],
        [96, 64],
        [200, 72],
        [33, 31],
        [256, 40],
        [1, 1],
        [64, 200],
        [160, 96],
    ];
    let mut wrong = Vec::new();
    for _ in 0..96 {
        let area = rng.pick(&areas);
        let samples = rng.pick(&[1, 1, 2, 4, 4, 3, 8]);
        let pass = random_pass(&mut rng, area, samples, 32);
        wrong.extend(pass.differences(&m));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// On the largest render areas (16384 pixels a side and others large), small primitives about
/// tile edges anywhere in the area, at every sample count, bin as the rule does.
#[test]
fn the_kernel_bins_what_the_rule_does_on_large_areas() {
    let m = typed_module().expect("module");
    let mut rng = Rng(0x1a29_e5ee);
    let mut wrong = Vec::new();
    for (area, samples) in [([16384, 16384], 1), ([16384, 9000], 4), ([12000, 16384], 2)] {
        let tile = [32, if samples == 4 { 16 } else { 32 }];
        let states = vec![Raster::over(area), state(&mut rng, area)];
        let drawn = (0..6)
            .map(|i| {
                let [w, h] = area.map(f64::from);
                let p = match rng.chance(1, 2) {
                    true => [
                        (rng.below(u64::from(area[0] / tile[0])) * u64::from(tile[0])) as f64
                            + (rng.below(17) as f64 - 8.0) / 512.0,
                        (rng.below(u64::from(area[1] / tile[1])) * u64::from(tile[1])) as f64
                            + (rng.below(17) as f64 - 8.0) / 512.0,
                    ],
                    false => [rng.grid(0.0, w, 512.0), rng.grid(0.0, h, 512.0)],
                };
                let near = |rng: &mut Rng| {
                    [
                        p[0] + rng.grid(-24.0, 24.0, 256.0),
                        p[1] + rng.grid(-24.0, 24.0, 256.0),
                    ]
                };
                let v = |q: [f64; 2]| vertex(area, q, 0.5, 1.0);
                let prim = match i % 3 {
                    0 => Primitive::Point(v(p), point_size(&mut rng).min(64.0)),
                    1 => Primitive::Line([v(p), v(near(&mut rng))]),
                    _ => Primitive::Triangle([v(p), v(near(&mut rng)), v(near(&mut rng))]),
                };
                Drawn::of(prim, (i % 2) as u32, i as u32)
            })
            .collect();
        let pass = Pass {
            area,
            samples,
            layers: 6,
            states,
            drawn,
        };
        wrong.extend(pass.differences(&m));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
