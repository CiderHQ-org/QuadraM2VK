//! [`super::axis_gradients`] on the interpreter: the pair it hands a device gives Metal's
//! anisotropic level under the Vulkan rule at every anisotropy up to 16, and its ratio never
//! passes 16. The interpreter leaves `Exp2` unmodelled (its precision is the device's), so the
//! kernel reads the bias's scale `2^B`, rounded once, through [`super::scaled_axis_gradients`].

use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, BinOp, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};

/// Words a case reads: `dx`, `dy` (two floats each), the bias's scale `2^B`, and the plane's size.
const IN: u32 = 7;
/// Words a case writes: the two gradients.
const OUT: u32 = 4;

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn storage_buffer(b: &mut ModuleBuilder, index: u32, elem: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(slot(index), StorageClass::StorageBuffer, block)
}

/// Word `k` of invocation `i`'s record of `per` words in `buf`.
fn word(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value, per: u32, k: u32) -> BResult<Value> {
    let per = f.u32_const(per)?;
    let per = f.module().const_value(per)?;
    let at = f.binary(BinOp::IMul, i, per)?;
    let k = f.u32_const(k)?;
    let k = f.module().const_value(k)?;
    let at = f.binary(BinOp::IAdd, at, k)?;
    let zero = f.u32_const(0)?;
    let zero = f.module().const_value(zero)?;
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, at])
}

/// A kernel whose invocation `i` reads case `i` from buffer 0 and writes its pair to buffer 1.
fn kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let read = |f: &mut FunctionBuilder<'_>, k: u32| -> BResult<Value> {
        let at = word(f, input, i, IN, k)?;
        f.load(at, MemAccess::default())
    };
    let mut floats = Vec::new();
    for k in 0..5 {
        let w = read(&mut f, k)?;
        floats.push(f.convert(ConvOp::Bitcast, w, f32_)?);
    }
    let (w, h) = (read(&mut f, 5)?, read(&mut f, 6)?);
    let dx = f.composite_construct(vec2, &[floats[0], floats[1]])?;
    let dy = f.composite_construct(vec2, &[floats[2], floats[3]])?;
    let size = f.composite_construct(uvec2, &[w, h])?;
    let scale = floats[4];
    let (gx, gy) = super::scaled_axis_gradients(&mut f, (dx, dy), |_| Ok(scale), size, 2)?;
    for (k, (v, lane)) in [(gx, 0), (gx, 1), (gy, 0), (gy, 1)].into_iter().enumerate() {
        let x = f.composite_extract(v, &[lane])?;
        let bits = f.convert(ConvOp::Bitcast, x, u32_)?;
        let at = word(&mut f, output, i, OUT, k as u32)?;
        f.store(at, bits, MemAccess::default())?;
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The length in texels of normalized gradient `g` on a plane of `size` texels.
fn texel_len(g: [f32; 2], size: [u32; 2]) -> f64 {
    let x = f64::from(g[0]) * f64::from(size[0]);
    let y = f64::from(g[1]) * f64::from(size[1]);
    (x * x + y * y).sqrt()
}

/// The pair's anisotropic level under the Vulkan rule at anisotropy `n`:
/// `log2(ρmax / min(ρmax / ρmin, n))`.
fn vulkan_level(gx: [f32; 2], gy: [f32; 2], size: [u32; 2], n: f64) -> f64 {
    let (a, b) = (texel_len(gx, size), texel_len(gy, size));
    let (max, min) = (a.max(b), a.min(b));
    let eta = if min > 0.0 { (max / min).min(n) } else { n };
    (max / eta).log2()
}

/// Metal's level of the footprint `(dx, dy)` biased by `B` through anisotropy `n`: `max(log2
/// minor + B, log2 major - log2 n)`.
fn metal_level(dx: [f32; 2], dy: [f32; 2], bias: f32, size: [u32; 2], n: f64) -> f64 {
    let (a, b) = (texel_len(dx, size), texel_len(dy, size));
    let (major, minor) = (a.max(b), a.min(b));
    let shifted = if minor > 0.0 {
        minor.log2() + f64::from(bias)
    } else {
        f64::NEG_INFINITY
    };
    shifted.max(major.log2() - n.log2())
}

/// Every pair the axis gradients hand a device keeps a ratio within 16, and gives, under the
/// Vulkan rule at anisotropies 1 to 16, Metal's level of the biased footprint: footprints of
/// ratios 1 to `2^20` and of a vanishing minor axis, on square and oblong planes, at biases across
/// Metal's range (lavapipe read levels far below Metal's past ratios of 4,564).
#[test]
fn the_axis_gradients_keep_metals_level_within_a_ratio_of_16() {
    let mut rng = Rng(0x5eed_0552_a715_0007);
    let sizes = [[1024u32, 1024u32], [1024, 64], [37, 2048]];
    let biases = [
        -32.0f32, -25.48, -20.5, -15.0, -9.375, -1.0, -0.015625, 0.0, 0.5, 3.0, 15.984375,
        31.984375,
    ];
    let mut cases: Vec<([f32; 2], [f32; 2], f32, [u32; 2])> = Vec::new();
    for n in 0..1200 {
        let size = sizes[n % sizes.len()];
        let bias = biases[(n / sizes.len()) % biases.len()];
        // The major axis's length in texels, 2^-4 to 2^20, its direction, and the minor's.
        let major = 2f64.powf(-4.0 + 24.0 * rng.unit());
        let ratio = match n % 4 {
            0 => 1.0 + 15.0 * rng.unit(),
            1 => 2f64.powf(20.0 * rng.unit()),
            2 => f64::INFINITY,
            _ => 2f64.powf(4.0 + 16.0 * rng.unit()),
        };
        let th = std::f64::consts::TAU * rng.unit();
        let skew = 0.5 + rng.unit();
        let minor = if ratio.is_finite() {
            major / ratio
        } else {
            0.0
        };
        let to_uv = |x: f64, y: f64| {
            [
                (x / f64::from(size[0])) as f32,
                (y / f64::from(size[1])) as f32,
            ]
        };
        let gmaj = to_uv(major * th.cos(), major * th.sin());
        let gmin = to_uv(-minor * (th + skew).sin(), minor * (th + skew).cos());
        let (dx, dy) = if n % 2 == 0 {
            (gmaj, gmin)
        } else {
            (gmin, gmaj)
        };
        cases.push((dx, dy, bias, size));
    }
    let m = kernel().expect("kernel");
    let bytes: Vec<u8> = cases
        .iter()
        .flat_map(|(dx, dy, bias, size)| {
            [
                dx[0].to_bits(),
                dx[1].to_bits(),
                dy[0].to_bits(),
                dy[1].to_bits(),
                (2f64.powf(f64::from(*bias)) as f32).to_bits(),
                size[0],
                size[1],
            ]
        })
        .flat_map(u32::to_le_bytes)
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes,
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: vec![0xAB; cases.len() * OUT as usize * 4],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(cases.len()).expect("cases"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    let mut wrong = Vec::new();
    for (i, (dx, dy, bias, size)) in cases.iter().enumerate() {
        let w = |k: usize| {
            let at = 4 * (i * OUT as usize + k);
            f32::from_bits(u32::from_le_bytes(
                out.buffers[1][at..at + 4].try_into().expect("word"),
            ))
        };
        let (gx, gy) = ([w(0), w(1)], [w(2), w(3)]);
        let (a, b) = (texel_len(gx, *size), texel_len(gy, *size));
        let ratio = a.max(b) / a.min(b);
        if ratio.is_nan() || ratio > 16.0 * (1.0 + 1e-5) {
            wrong.push(format!(
                "case {i}: ratio {ratio} ({dx:?}, {dy:?}, B {bias}, {size:?})"
            ));
        }
        for n in [1.0, 2.0, 4.0, 8.0, 16.0] {
            let (got, want) = (
                vulkan_level(gx, gy, *size, n),
                metal_level(*dx, *dy, *bias, *size, n),
            );
            if (got - want).abs() > 1e-4 {
                wrong.push(format!(
                    "case {i} at {n}x: level {got}, Metal's {want} ({dx:?}, {dy:?}, B {bias}, {size:?})"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        cases.len(),
        wrong[..wrong.len().min(20)].join("\n")
    );
}
