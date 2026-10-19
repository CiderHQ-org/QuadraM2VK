//! The device's checks of three shader-converter defects (`DriverWorkaround`'s
//! `CompositeInsertsWrittenInPlace`, `LoadsForwardedIntoLoops` and
//! `LocalizedLoadsForwardedPastAtomicsAndCalls`): each kernel holds the shape the workaround's
//! encoding guards, serialized as a profile without the workaround writes it (`serialize`, no
//! copy and no materialization), and leaves words that only that shape's misreading changes. A
//! device whose converter reads the module as SPIR-V states leaves [`CheckKernel::expected`]; one
//! with the defect leaves other words (MoltenVK 1.4.1, whose SPIRV-Cross `adec7acb` has all three:
//! its MSL for these shapes is `spirv::serialize`'s tests' `loop_reads` and `localized_loads`).
//!
//! The shapes are the serializer tests' kernels, with inputs whose answers tell the readings
//! apart:
//!
//! - **An insert written in place** ([`composite_inserts_check`]): `v` loaded before a loop of
//!   two iterations whose body stores `v` with `v.x + 1` inserted, then loads the element back and
//!   stores it with `y + 1` inserted. Every iteration inserts into the `v` loaded before the loop,
//!   so an element `(a, b)` ends `(a + 1, b + 2)`; written in place, the second iteration reads the
//!   first's `v.x` and the element ends `(a + 2, b + 2)`.
//! - **A load forwarded into a loop** ([`loads_forwarded_check`]): `v` and its `y` loaded before a
//!   loop of two iterations that stores `v.x + y + x.y` (`x` loaded in the loop) to both lanes, so
//!   an element first stored `(a + 1, a + 1)` ends `5a + 5` in both; read again in the loop, `v`
//!   and `y` see the first iteration's store, and it ends `9a + 9` (`7a + 7` with one of them).
//! - **A load forwarded past an atomic and a call** ([`localized_loads_check`]): `a = w` (a
//!   `Workgroup` word), then `w += 10` atomically; `c = p` (a `Private` word), then a call that
//!   stores 15 to `p`; `out[0] = a + c`. Read where the module loads them, `out[0]` is `w`'s first
//!   value plus `p`'s initializer; read past the atomic or the call, it is 10 or `p`'s 15 off.
//!
//! The insert's shape also misreads where only its load is forwarded (the second iteration then
//! reads the first's store), so a converter with only the load defect answers the insert's check
//! wrong too: the device keeps both encodings, each of which is exact on every device.

use super::kernel::{finish, LOCAL_SIZE};
use super::CheckKernel;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, LoopCtx, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, IoDir, LayoutRule,
    MemAccess, Scope, Semantics, StorageClass, Type, TypedModule,
};

/// The elements of the loop checks' buffer: one per invocation.
const ELEMENTS: u32 = LOCAL_SIZE;

/// The first lane of element `i` in the loop checks' buffer, a small whole `float` (exact in every
/// sum the kernels make).
fn first_lane(i: u32) -> f32 {
    (3 * i + 1) as f32
}

/// The second lane of element `i`.
fn second_lane(i: u32) -> f32 {
    (5 * i + 2) as f32
}

/// The loop checks' buffer before the dispatch: element `i` is `(first_lane(i),
/// second_lane(i))`.
fn pairs() -> Vec<u32> {
    (0..ELEMENTS)
        .flat_map(|i| [first_lane(i).to_bits(), second_lane(i).to_bits()])
        .collect()
}

/// The check of `DriverWorkaround::CompositeInsertsWrittenInPlace` on `device` (the module
/// docs): one storage buffer of [`CheckKernel::LOCAL_SIZE`] `float2` elements, read back; element `(a, b)`
/// must end `(a + 1, b + 2)`.
pub fn composite_inserts_check(device: &Device) -> Result<CheckKernel, Error> {
    let expected = (0..ELEMENTS)
        .flat_map(|i| {
            [
                (first_lane(i) + 1.0).to_bits(),
                (second_lane(i) + 2.0).to_bits(),
            ]
        })
        .collect();
    let spirv = finish(device, inserts_kernel()?, "composite insert check")?;
    Ok(CheckKernel::new(spirv, 1, vec![pairs()], 0, expected))
}

/// The check of `DriverWorkaround::LoadsForwardedIntoLoops` on `device` (the module docs): one
/// storage buffer of [`CheckKernel::LOCAL_SIZE`] `float2` elements, read back; element `(a, b)` must end
/// `(5a + 5, 5a + 5)`.
pub fn loads_forwarded_check(device: &Device) -> Result<CheckKernel, Error> {
    let expected = (0..ELEMENTS)
        .flat_map(|i| {
            let t = (5.0 * first_lane(i) + 5.0).to_bits();
            [t, t]
        })
        .collect();
    let spirv = finish(device, loads_kernel()?, "forwarded load check")?;
    Ok(CheckKernel::new(spirv, 1, vec![pairs()], 0, expected))
}

/// The `Workgroup` word's first value in [`localized_loads_check`], which the kernel stores from
/// the buffer's last word.
const FIRST: u32 = 7;

/// The `Private` word's initializer in [`localized_loads_check`].
const SEED: u32 = 0x5eed;

/// The check of `DriverWorkaround::LocalizedLoadsForwardedPastAtomicsAndCalls` on `device` (the
/// module docs): one storage buffer of four words `[0, 0, 0, FIRST]`, one invocation, read back;
/// it must end `[FIRST + SEED, FIRST + 10, 0, FIRST]` (`out[1]` is `w` after the atomic, which a
/// routine with no atomic and no call stores).
pub fn localized_loads_check(device: &Device) -> Result<CheckKernel, Error> {
    let spirv = finish(device, localized_kernel()?, "localized load check")?;
    Ok(CheckKernel::new(
        spirv,
        1,
        vec![vec![0, 0, 0, FIRST]],
        0,
        vec![FIRST + SEED, FIRST + 10, 0, FIRST],
    ))
}

/// The binding symbol of storage buffer `index`.
fn buffer_slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// A `Block { elem data[]; }` storage buffer (std430) at buffer slot `index`.
fn buffer(b: &mut ModuleBuilder, index: u32, elem: TypeId, stride: u32) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(stride),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(buffer_slot(index), StorageClass::StorageBuffer, block)
}

/// A constant of type `ty` with bits `bits`, as a value.
fn constant(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })?;
    f.module().const_value(c)
}

/// A pointer to element `i` of storage buffer `g`.
fn element(f: &mut FunctionBuilder<'_>, g: GlobalId, u32_: TypeId, i: Value) -> BResult<Value> {
    let zero = constant(f, u32_, 0)?;
    let base = f.module().global_value(g)?;
    f.access_chain(base, &[zero, i])
}

/// `gl_GlobalInvocationID.x`.
fn invocation(f: &mut FunctionBuilder<'_>, u32_: TypeId) -> BResult<Value> {
    let uvec3 = f.module().ty(Type::Vector { elem: u32_, len: 3 })?;
    let g = f
        .module()
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let p = f.module().global_value(g)?;
    let v = f.load(p, MemAccess::default())?;
    f.composite_extract(v, &[0])
}

/// A loop body of [`twice`].
type Body<'a> = Box<dyn FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a>;

/// A loop of two iterations around `body` (a function-local counter, its test first, as the
/// serializer tests' loops).
fn twice(f: &mut FunctionBuilder<'_>, u32_: TypeId, body: Body<'_>) -> BResult<()> {
    let counter = f.local_var(u32_, None)?;
    let counter = f.var_ptr(counter)?;
    let zero = constant(f, u32_, 0)?;
    f.store(counter, zero, MemAccess::default())?;
    let (one, two) = (constant(f, u32_, 1)?, constant(f, u32_, 2)?);
    f.loop_(
        &mut (),
        LoopSpec::default(),
        Box::new(
            move |fb: &mut FunctionBuilder<'_>, ctx: &mut (), lc: &LoopCtx| {
                let n = fb.load(counter, MemAccess::default())?;
                let done = fb.binary(BinOp::UGreaterThanEqual, n, two)?;
                fb.if_(
                    ctx,
                    done,
                    Box::new(|fb: &mut FunctionBuilder<'_>, _: &mut ()| {
                        lc.break_(fb)?;
                        Ok(ArmEnd::Exits)
                    }),
                    None,
                )?;
                let n1 = fb.binary(BinOp::IAdd, n, one)?;
                fb.store(counter, n1, MemAccess::default())?;
                body(fb)
            },
        ),
    )
}

/// The compute entry `main` of `function`, at local size `(local, 1, 1)`, finalized.
fn compute_entry(mut b: ModuleBuilder, function: FuncId, local: u32) -> BResult<TypedModule> {
    b.entry(
        ExecModel::GLCompute,
        function,
        vec![ExecMode::LocalSize {
            x: local,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// [`composite_inserts_check`]'s kernel: the serializer test's `loop_reads::kernel`.
fn inserts_kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let vectors = buffer(&mut b, 0, vec2, 8)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let i = invocation(&mut f, u32_)?;
    let at = element(&mut f, vectors, u32_, i)?;
    let one = constant(&mut f, f32_, u64::from(1.0f32.to_bits()))?;
    // An insert outside every loop, which runs once.
    let w = f.load(at, MemAccess::default())?;
    let y = f.composite_extract(w, &[1])?;
    let y = f.binary(BinOp::FAdd, y, one)?;
    let w = f.composite_insert(y, w, &[1])?;
    f.store(at, w, MemAccess::default())?;
    // The composite the loop inserts into, defined before it; nothing reads it after the loop.
    let v = f.load(at, MemAccess::default())?;
    twice(
        &mut f,
        u32_,
        Box::new(move |fb| {
            let e = fb.composite_extract(v, &[0])?;
            let e = fb.binary(BinOp::FAdd, e, one)?;
            let stepped = fb.composite_insert(e, v, &[0])?;
            fb.store(at, stepped, MemAccess::default())?;
            // An insert into a composite the loop defines.
            let x = fb.load(at, MemAccess::default())?;
            let y = fb.composite_extract(x, &[1])?;
            let y = fb.binary(BinOp::FAdd, y, one)?;
            let x = fb.composite_insert(y, x, &[1])?;
            fb.store(at, x, MemAccess::default())
        }),
    )?;
    f.ret(None)?;
    let main = f.finish()?;
    compute_entry(b, main, LOCAL_SIZE)
}

/// [`loads_forwarded_check`]'s kernel: the serializer test's `loop_reads::loads_kernel`.
fn loads_kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let vectors = buffer(&mut b, 0, vec2, 8)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let i = invocation(&mut f, u32_)?;
    let at = element(&mut f, vectors, u32_, i)?;
    let one = constant(&mut f, f32_, u64::from(1.0f32.to_bits()))?;
    // A value read only before the loop.
    let u = f.load(at, MemAccess::default())?;
    let ux = f.composite_extract(u, &[0])?;
    let ux = f.binary(BinOp::FAdd, ux, one)?;
    let pair = f.composite_construct(vec2, &[ux, ux])?;
    f.store(at, pair, MemAccess::default())?;
    // Values read from memory before the loop that the loop reads.
    let v = f.load(at, MemAccess::default())?;
    let y = f.composite_extract(v, &[1])?;
    twice(
        &mut f,
        u32_,
        Box::new(move |fb| {
            let x = fb.load(at, MemAccess::default())?;
            let xy = fb.composite_extract(x, &[1])?;
            let vx = fb.composite_extract(v, &[0])?;
            let t = fb.binary(BinOp::FAdd, vx, y)?;
            let t = fb.binary(BinOp::FAdd, t, xy)?;
            let pair = fb.composite_construct(vec2, &[t, t])?;
            fb.store(at, pair, MemAccess::default())
        }),
    )?;
    f.ret(None)?;
    let main = f.finish()?;
    compute_entry(b, main, LOCAL_SIZE)
}

/// [`localized_loads_check`]'s kernel: the serializer test's `localized_loads::kernel`, with the
/// `Workgroup` word first stored from the buffer's last word, at one invocation.
fn localized_kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let out = buffer(&mut b, 0, u32_, 4)?;
    let w = b.plain_global(StorageClass::Workgroup, u32_, None)?;
    let seed = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(SEED)),
    })?;
    let p = b.plain_global(StorageClass::Private, u32_, Some(seed))?;
    let void = b.ty(Type::Void)?;
    // A routine that writes the `Private` word.
    let bump = {
        let mut f = b.routine(void)?;
        let pp = f.module().global_value(p)?;
        let fifteen = constant(&mut f, u32_, 15)?;
        f.store(pp, fifteen, MemAccess::default())?;
        f.ret(None)?;
        f.finish()?
    };
    // A routine that loads the `Workgroup` word with no atomic and no call: `out[1] = w`.
    let record = {
        let mut f = b.routine(void)?;
        let wp = f.module().global_value(w)?;
        let v = f.load(wp, MemAccess::default())?;
        let one = constant(&mut f, u32_, 1)?;
        let at = element(&mut f, out, u32_, one)?;
        f.store(at, v, MemAccess::default())?;
        f.ret(None)?;
        f.finish()?
    };
    let mut f = b.function(void)?;
    let wp = f.module().global_value(w)?;
    let pp = f.module().global_value(p)?;
    let three = constant(&mut f, u32_, 3)?;
    let last = element(&mut f, out, u32_, three)?;
    let first = f.load(last, MemAccess::default())?;
    f.store(wp, first, MemAccess::default())?;
    let a = f.load(wp, MemAccess::default())?;
    let ten = constant(&mut f, u32_, 10)?;
    f.atomic(
        AtomicOp::IAdd,
        wp,
        Scope::Workgroup,
        Semantics::RELAXED,
        ten,
    )?;
    let c = f.load(pp, MemAccess::default())?;
    f.call(bump, &[])?;
    f.call(record, &[])?;
    let s = f.binary(BinOp::IAdd, a, c)?;
    let zero = constant(&mut f, u32_, 0)?;
    let at = element(&mut f, out, u32_, zero)?;
    f.store(at, s, MemAccess::default())?;
    f.ret(None)?;
    let main = f.finish()?;
    compute_entry(b, main, 1)
}

#[cfg(test)]
mod tests {
    use super::{inserts_kernel, loads_kernel, localized_kernel};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// Each kernel is valid on every committed profile.
    #[test]
    fn the_converter_check_kernels_are_valid_on_every_committed_profile() {
        for (stem, profile) in committed_profiles().expect("profiles") {
            for (name, kernel) in [
                ("inserts", inserts_kernel as fn() -> _),
                ("loads", loads_kernel),
                ("localized", localized_kernel),
            ] {
                let m = kernel().unwrap_or_else(|e| panic!("{name}: {e:?}"));
                let _ = (stem.as_str(), assert_valid(m, &profile));
            }
        }
    }
}
