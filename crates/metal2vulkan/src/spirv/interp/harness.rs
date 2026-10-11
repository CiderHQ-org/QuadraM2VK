//! The out-of-bounds trap harness with wrapped offsets.
//!
//! Containment code (the leaf encodings that lowering emits) is checked by running it on the
//! interpreter at offsets around the end of a buffer and near `2^32` and `2^64`, where a check
//! written as `off + n <= size` wraps. The interpreter never wraps an address, so an access that
//! escapes its buffer traps instead of aliasing; a correct containment encoding therefore never
//! traps, and yields the in-bounds bytes exactly when the normative check [`in_bounds`] holds and
//! zero otherwise.
//!
//! A probe kernel reads `(base, offset, size)` from a parameter buffer, lets an [`AccessFn`] emit
//! the `n`-byte access of the physical buffer at `base + offset` (with `offset` and `size` in the
//! probe's [`OffsetWidth`]), and stores the result. [`raw_access`] is the uncontained access and
//! [`checked_access`] the normative guard; lowerings pass their own emitters to
//! [`check_containment`].

use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, LayoutRule, MemAccess,
    StorageClass, Type, TypedModule,
};

use super::{run, Buffer, Config, InterpError, Resources, TrapKind};

/// The width of the offset and size arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OffsetWidth {
    U32,
    U64,
}

impl OffsetWidth {
    fn max(self) -> u64 {
        match self {
            OffsetWidth::U32 => u64::from(u32::MAX),
            OffsetWidth::U64 => u64::MAX,
        }
    }
}

/// One access: `n` bytes at byte `offset` of the buffer, computed in `width`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Probe {
    pub offset: u64,
    pub n: u32,
    pub width: OffsetWidth,
}

/// The access sizes the tests cover.
pub const ACCESS_SIZES: [u32; 4] = [1, 2, 4, 16];

/// The normative bounds check: `size >= n && off <= size - n`, which never wraps.
pub fn in_bounds(size: u64, off: u64, n: u64) -> bool {
    size >= n && off <= size - n
}

/// Probes for a buffer of `size` bytes: every access size and offset width, at offsets around the
/// start and end of the buffer, at `0xFFFF_FFF0..=0xFFFF_FFFF`, and (64-bit) at the top of the
/// address space.
pub fn probes(size: u64) -> Vec<Probe> {
    let mut offs: Vec<u64> = vec![0, 1, 2, 3];
    for d in 0..=17u64 {
        offs.push(size.saturating_sub(d));
        offs.push(size.saturating_add(d));
    }
    offs.extend(0xFFFF_FFF0u64..=0xFFFF_FFFF);
    let top: Vec<u64> = (0..16u64).map(|d| u64::MAX - d).collect();
    offs.sort_unstable();
    offs.dedup();
    let mut out = Vec::new();
    for width in [OffsetWidth::U32, OffsetWidth::U64] {
        for n in ACCESS_SIZES {
            let extra: &[u64] = match width {
                OffsetWidth::U32 => &[],
                OffsetWidth::U64 => &top,
            };
            for off in offs.iter().chain(extra) {
                if *off <= width.max() {
                    out.push(Probe {
                        offset: *off,
                        n,
                        width,
                    });
                }
            }
        }
    }
    out
}

/// What an [`AccessFn`] gets to emit one access.
#[derive(Clone, Copy, Debug)]
pub struct AccessArgs {
    /// The buffer's device address (`u64`).
    pub base: Value,
    /// The byte offset, in the probe's width.
    pub offset: Value,
    /// The buffer's byte size, in the probe's width.
    pub size: Value,
    /// The access size in bytes.
    pub n: u32,
    pub width: OffsetWidth,
    /// The offset/size integer type.
    pub int: TypeId,
    /// The accessed type: `u8`, `u16`, `u32`, or `uvec4` for `n` = 1, 2, 4, 16.
    pub elem: TypeId,
    /// `PhysicalStorageBuffer` pointer to `elem`.
    pub ptr: TypeId,
}

/// Emits one access in the open block and returns the loaded `elem` value.
pub type AccessFn<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, &AccessArgs) -> BResult<Value>;

/// The physical address `base + offset` (the offset zero-extended) as a pointer to `elem`.
pub fn address(fb: &mut FunctionBuilder<'_>, a: &AccessArgs) -> BResult<Value> {
    let u64_ = fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let off = match a.width {
        OffsetWidth::U32 => fb.convert(ConvOp::UConvert, a.offset, u64_)?,
        OffsetWidth::U64 => a.offset,
    };
    let addr = fb.binary(BinOp::IAdd, a.base, off)?;
    fb.convert(ConvOp::ConvertUToPtr, addr, a.ptr)
}

/// A byte-aligned load (probes are not aligned to `n`).
fn load(fb: &mut FunctionBuilder<'_>, p: Value) -> BResult<Value> {
    fb.load(
        p,
        MemAccess {
            aligned: Some(1),
            ..MemAccess::default()
        },
    )
}

/// The uncontained access: it traps for every out-of-bounds probe.
pub fn raw_access(fb: &mut FunctionBuilder<'_>, a: &AccessArgs) -> BResult<Value> {
    let p = address(fb, a)?;
    load(fb, p)
}

/// An integer constant of the offset type.
pub fn int_const(fb: &mut FunctionBuilder<'_>, a: &AccessArgs, v: u64) -> BResult<Value> {
    let c = fb.module().constant(Constant::Scalar {
        ty: a.int,
        bits: ConstBits(v),
    })?;
    fb.module().const_value(c)
}

/// The normative containment: `inb = size >= n && off <= size - n` in the offset's width; load when
/// `inb`, else zero.
pub fn checked_access(fb: &mut FunctionBuilder<'_>, a: &AccessArgs) -> BResult<Value> {
    let n = int_const(fb, a, u64::from(a.n))?;
    let fits = fb.binary(BinOp::UGreaterThanEqual, a.size, n)?;
    let room = fb.binary(BinOp::ISub, a.size, n)?;
    let within = fb.binary(BinOp::ULessThanEqual, a.offset, room)?;
    let inb = fb.binary(BinOp::LogicalAnd, fits, within)?;
    let null = fb.module().constant(Constant::Null { ty: a.elem })?;
    let var = fb.local_var(a.elem, Some(null))?;
    let result = fb.var_ptr(var)?;
    let args = *a;
    let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let p = address(fb, &args)?;
        let v = load(fb, p)?;
        fb.store(result, v, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), inb, then, None)?;
    fb.load(result, MemAccess::default())
}

/// The device address the probed buffer is placed at: far from zero, so offsets near `2^32`
/// cannot land in it by accident.
pub const DATA_ADDRESS: u64 = 0x0000_0040_0000_0000;

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// A `Block { elem data[]; }` storage buffer at slot `index`.
fn array_buffer(b: &mut ModuleBuilder, index: u32, elem: TypeId, stride: u32) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(stride),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(slot(index), StorageClass::StorageBuffer, block)
}

/// The probe kernel for access size `n` in `width`: params (`u64` base, offset, size) at slot 1,
/// the result at element 0 of slot 0.
pub fn probe_module(n: u32, width: OffsetWidth, access: AccessFn<'_>) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let elem = match n {
        1 | 2 => b.ty(Type::Int {
            width: (n * 8) as u8,
            signed: false,
        })?,
        4 => u32_,
        _ => b.ty(Type::Vector { elem: u32_, len: 4 })?,
    };
    let out = array_buffer(&mut b, 0, elem, n)?;
    let params = array_buffer(&mut b, 1, u64_, 8)?;
    let ptr = b.ty(Type::Pointer {
        class: StorageClass::PhysicalStorageBuffer,
        pointee: elem,
    })?;
    let int = match width {
        OffsetWidth::U32 => u32_,
        OffsetWidth::U64 => u64_,
    };
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = f.u32_const(0)?;
    let zero = f.module().const_value(zero)?;
    let param = |f: &mut FunctionBuilder<'_>, i: u32| -> BResult<Value> {
        let k = f.u32_const(i)?;
        let k = f.module().const_value(k)?;
        let base = f.module().global_value(params)?;
        let p = f.access_chain(base, &[zero, k])?;
        let v = f.load(p, MemAccess::default())?;
        match width {
            OffsetWidth::U32 if i > 0 => f.convert(ConvOp::UConvert, v, u32_),
            _ => Ok(v),
        }
    };
    let args = AccessArgs {
        base: param(&mut f, 0)?,
        offset: param(&mut f, 1)?,
        size: param(&mut f, 2)?,
        n,
        width,
        int,
        elem,
        ptr,
    };
    let v = access(&mut f, &args)?;
    let ob = f.module().global_value(out)?;
    let p = f.access_chain(ob, &[zero, zero])?;
    f.store(p, v, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The byte pattern of the probed buffer.
pub fn pattern(size: u64) -> Vec<u8> {
    (0..size)
        .map(|i| i.wrapping_mul(7).wrapping_add(3) as u8)
        .collect()
}

/// What one probe did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// The access completed and stored these `n` bytes.
    Loaded(Vec<u8>),
    /// The access trapped.
    Trapped(TrapKind),
}

/// Run `m` (a [`probe_module`] for `probe.n` and `probe.width`) over a `size`-byte buffer.
/// Errors other than traps are returned as they are.
pub fn run_probe(m: &TypedModule, size: u64, probe: &Probe) -> Result<ProbeOutcome, InterpError> {
    let params: Vec<u8> = [DATA_ADDRESS, probe.offset, size]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes: vec![0; probe.n as usize],
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: params,
            },
            Buffer {
                slot: None,
                address: Some(DATA_ADDRESS),
                bytes: pattern(size),
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    match run(m, &Config::default(), res) {
        Ok(out) => Ok(ProbeOutcome::Loaded(
            out.buffers.into_iter().next().unwrap_or_default(),
        )),
        Err(InterpError::Trap(t)) => Ok(ProbeOutcome::Trapped(t.kind)),
        Err(e) => Err(e),
    }
}

/// What a contained access must produce: the in-bounds bytes, or zero.
pub fn expected(size: u64, probe: &Probe) -> Vec<u8> {
    let n = u64::from(probe.n);
    if in_bounds(size, probe.offset, n) {
        let data = pattern(size);
        data.get(probe.offset as usize..(probe.offset + n) as usize)
            .map(<[u8]>::to_vec)
            .unwrap_or_default()
    } else {
        vec![0; probe.n as usize]
    }
}

/// Check that `access` contains every probe of a `size`-byte buffer: no trap, the in-bounds
/// bytes when [`in_bounds`] holds, zero otherwise. Returns every failing probe.
pub fn check_containment(access: AccessFn<'_>, size: u64) -> Result<(), String> {
    let mut failures = Vec::new();
    let all = probes(size);
    for width in [OffsetWidth::U32, OffsetWidth::U64] {
        for n in ACCESS_SIZES {
            if width == OffsetWidth::U32 && size > width.max() {
                continue;
            }
            let m = probe_module(n, width, access).map_err(|e| format!("build: {e:?}"))?;
            for p in all.iter().filter(|p| p.n == n && p.width == width) {
                let got = run_probe(&m, size, p).map_err(|e| format!("{p:?}: {e}"))?;
                let want = ProbeOutcome::Loaded(expected(size, p));
                if got != want {
                    failures.push(format!("{p:?}: got {got:?}, want {want:?}"));
                }
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}
