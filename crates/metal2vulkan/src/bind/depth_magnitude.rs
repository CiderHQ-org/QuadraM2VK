//! A 32-bit float depth's magnitude stand-in, and the merge back of a pass rendered over it.
//!
//! Metal's depth test compares a fragment's depth with the stored value's magnitude, as the
//! integers of their low 31 bits: a stored -0.5 passes `Equal` for a fragment at 0.5, and a stored
//! NaN orders above +inf. Vulkan's compares the stored value as it is. A consumer whose depth image
//! holds such a value renders a pass that loads it over a stand-in image instead: before the pass
//! it copies the depth's words into the *exact* buffer, one dispatch of
//! [`depth_magnitude_module`] writes each word's stand-in into the *stand-in* buffer, and it copies
//! that into the stand-in image, which the pass loads, tests and writes. After a pass that may have
//! written the depth, it copies the stand-in image into the stand-in buffer, one dispatch merges it
//! into the exact buffer, and it copies that back into the depth.
//!
//! **The stand-in** of a word `w`, the bits of a `float`: its magnitude `w & 0x7fffffff`; a NaN's
//! is +inf's, `0x7f800000`; and where the push's `flush` word is not 0, a nonzero denormal's is the
//! smallest normal's, `0x00800000`, for a device whose depth test compares a stored denormal as
//! zero. A fragment's depth is never negative or NaN, so Vulkan's test against the stand-in gives
//! Metal's answer against the word, but where the fragment lies at exactly 2^-126 against a
//! denormal's stand-in.
//!
//! **The merge**: Metal stores each texel of a pass that sets a depth/stencil state with depth
//! writes without its sign, `w & 0x7fffffff` (a NaN's payload and a denormal kept), where no
//! fragment wrote it, and the fragment's depth where one did (our probe `depthstore`, the M3 Max
//! under macOS 27.0.1). Where the stand-in buffer's word (the pass's) equals the stand-in of the
//! exact buffer's, the pass left the texel as it loaded it, and the exact word without its sign
//! replaces it; elsewhere the pass's word does. A pass that writes exactly a word's stand-in over a
//! word whose stand-in is not that word without its sign (+inf over a NaN, the smallest normal
//! over a denormal) leaves the word without its sign, where Metal stores the pass's.
//!
//! **The strip**: after a pass with depth writes that renders the depth in place on a device
//! whose depth test is Metal's own but which leaves the stored words as they are where no draw ran
//! under such a state, every word without its sign, as Metal stores them.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`DepthMagnitudeModule::LOCAL_SIZE`], 1, 1), [`DepthMagnitudeModule::PUSH_BYTES`] of `COMPUTE`
//! push constants at offset 0 ([`DepthMagnitudeModule::shape`]), and two storage buffers in set 0,
//! each a whole number of 4-byte words:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`DepthMagnitudeModule::EXACT_BINDING`] | the depth's words; a merge writes them |
//! | [`DepthMagnitudeModule::STAND_IN_BINDING`] | their stand-ins; in a merge, the pass's |
//!
//! The shape's words, in order: `count`, the words; `flush`, whether a denormal's stand-in is the
//! smallest normal; `merge`, 0 for a stand-in and 1 for a merge; and `strip`, in a merge 1 for a
//! strip ([`DepthMagnitudeModule::strip`]), which stores every exact word without its sign and
//! reads no stand-in. Invocation
//! `i` below `count` takes word `i` of both buffers; one past `count` does nothing. A consumer
//! whose words exceed `maxStorageBufferRange` or `maxComputeWorkGroupCount[0]` workgroups binds
//! them in windows, one dispatch each: no word's result reads another word.
//!
//! **Synchronization** (consumer): the copies' `TRANSFER_WRITE` → `SHADER_STORAGE_READ` and
//! `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the dispatch; its writes → `TRANSFER_READ`
//! before a buffer is copied into an image.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// The crate-owned depth magnitude module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DepthMagnitudeModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl DepthMagnitudeModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The depth's words, a storage buffer a merge writes.
    pub const EXACT_BINDING: u32 = 0;
    /// Their stand-ins, a storage buffer a stand-in writes; the pass's words in a merge.
    pub const STAND_IN_BINDING: u32 = 1;
    /// Invocations of one workgroup, along x: one 4-byte word each.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of a dispatch over `count` words: a merge where `merge`, else a stand-in,
    /// with a denormal's stand-in the smallest normal where `flush` (the module docs).
    pub fn shape(count: u32, merge: bool, flush: bool) -> [u8; 16] {
        Self::words(count, u32::from(flush), u32::from(merge), 0)
    }

    /// The push bytes of a strip over `count` words: each exact word stored without its sign,
    /// the stand-in buffer unread (the module docs).
    pub fn strip(count: u32) -> [u8; 16] {
        Self::words(count, 0, 1, 1)
    }

    /// The shape's words as push bytes.
    fn words(count: u32, flush: u32, merge: u32, strip: u32) -> [u8; 16] {
        let words = [count, flush, merge, strip];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The stand-in of the word `bits` (the module docs), as the module computes it: its
    /// magnitude, a NaN's +inf's, and where `flush` a nonzero denormal's the smallest normal's.
    pub fn stand_in(bits: u32, flush: bool) -> u32 {
        let magnitude = bits & 0x7fff_ffff;
        if magnitude > 0x7f80_0000 {
            return 0x7f80_0000;
        }
        if flush && magnitude != 0 && magnitude < 0x0080_0000 {
            return 0x0080_0000;
        }
        magnitude
    }
}

/// The binding symbol of the exact buffer in the typed module.
const EXACT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the stand-in buffer in the typed module.
const STAND_IN: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

/// The module as a typed module, before serialization: the module [`depth_magnitude_module`]
/// serializes, which the CPU interpreter runs. Its bindings are the symbols `EXACT` and
/// `STAND_IN`, and its push field the shape, a `uvec4` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    let exact = words_buffer(&mut b, EXACT, u32_)?;
    let stand_in = words_buffer(&mut b, STAND_IN, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let sign_off = u32_value(&mut f, 0x7fff_ffff)?;
    let infinity = u32_value(&mut f, 0x7f80_0000)?;
    let smallest = u32_value(&mut f, 0x0080_0000)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let i = f.composite_extract(invocation, &[0])?;
    let exact = f.module().global_value(exact)?;
    let stand_in = f.module().global_value(stand_in)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let count = f.composite_extract(shape, &[0])?;
    let flush = f.composite_extract(shape, &[1])?;
    let merge = f.composite_extract(shape, &[2])?;
    let strip = f.composite_extract(shape, &[3])?;
    let merging = f.binary(BinOp::INotEqual, merge, zero)?;
    let stripping = f.binary(BinOp::INotEqual, strip, zero)?;
    let flushing = f.binary(BinOp::INotEqual, flush, zero)?;
    let inside = f.binary(BinOp::ULessThan, i, count)?;
    let word: ArmFn<'_, ()> = Box::new(move |f, _| {
        let at_exact = f.access_chain(exact, &[zero, i])?;
        let held = f.load(at_exact, MemAccess::default())?;
        // The stand-in of the word (`DepthMagnitudeModule::stand_in`).
        let unsigned = f.binary(BinOp::BitwiseAnd, held, sign_off)?;
        let nan = f.binary(BinOp::UGreaterThan, unsigned, infinity)?;
        let magnitude = f.select(nan, infinity, unsigned)?;
        let nonzero = f.binary(BinOp::INotEqual, magnitude, zero)?;
        let below = f.binary(BinOp::ULessThan, magnitude, smallest)?;
        let denormal = f.binary(BinOp::LogicalAnd, nonzero, below)?;
        let raised = f.binary(BinOp::LogicalAnd, denormal, flushing)?;
        let standing = f.select(raised, smallest, magnitude)?;
        let at_stand_in = f.access_chain(stand_in, &[zero, i])?;
        let into: ArmFn<'_, ()> = Box::new(move |f, _| {
            // A strip: every word without its sign; the stand-in buffer is not read.
            let bare: ArmFn<'_, ()> = Box::new(move |f, _| {
                f.store(at_exact, unsigned, MemAccess::default())?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            let merged: ArmFn<'_, ()> = Box::new(move |f, _| {
                let drawn = f.load(at_stand_in, MemAccess::default())?;
                // A word the pass left as it loaded it is the exact word without its sign, as
                // Metal stores it after a pass with depth writes; any other is the pass's.
                let changed = f.binary(BinOp::INotEqual, drawn, standing)?;
                let merged = f.select(changed, drawn, unsigned)?;
                f.store(at_exact, merged, MemAccess::default())?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), stripping, bare, Some(merged))?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        let out: ArmFn<'_, ()> = Box::new(move |f, _| {
            f.store(at_stand_in, standing, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), merging, into, Some(out))?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, word, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: DepthMagnitudeModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(DepthMagnitudeModule::SET),
        Symbol::Binding(s) if *s == EXACT => Some(DepthMagnitudeModule::EXACT_BINDING),
        Symbol::Binding(s) if *s == STAND_IN => Some(DepthMagnitudeModule::STAND_IN_BINDING),
        _ => None,
    }
}

/// The depth magnitude module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`DepthMagnitudeModule`].
pub fn depth_magnitude_module(device: &Device) -> Result<DepthMagnitudeModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the depth magnitude module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(DepthMagnitudeModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, DepthMagnitudeModule, EXACT, STAND_IN};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// The 18 stored values of the `depthkeep` probe (the M3 Max under macOS 27.0.1): ±2, ±0.5,
    /// the smallest and largest denormal of each sign, ±0, the smallest normal, 0.25, 0.5, 1, the
    /// `float` after 1, 1.5, ±inf, and the quiet NaN.
    const PROBED: [u32; 18] = [
        0x4000_0000,
        0xc000_0000,
        0x3f00_0000,
        0xbf00_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0000_0000,
        0x8000_0000,
        0x0080_0000,
        0x3e80_0000,
        0x3f80_0000,
        0x3f80_0001,
        0x3fc0_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
    ];

    #[test]
    fn the_magnitude_kernel_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(typed_module().expect("module"), &p);
        }
    }

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect()
    }

    /// Run the kernel over `exact` and `stand_in`, in `groups` workgroups of the shape of
    /// `count` words, both invocation orders: both buffers after it.
    fn ran(
        exact: &[u32],
        stand_in: &[u32],
        count: u32,
        merge: bool,
        flush: bool,
        groups: u32,
    ) -> (Vec<u32>, Vec<u32>) {
        let push = DepthMagnitudeModule::shape(count, merge, flush);
        ran_with(exact, stand_in, push, groups)
    }

    /// Run the kernel over `exact` and `stand_in` with the push bytes `push`, as [`ran`] does.
    fn ran_with(
        exact: &[u32],
        stand_in: &[u32],
        push: [u8; 16],
        groups: u32,
    ) -> (Vec<u32>, Vec<u32>) {
        let m = typed_module().expect("module");
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let part = |slot, words: &[u32]| Buffer {
                slot: Some(slot),
                address: None,
                bytes: bytes(words),
            };
            let res = Resources {
                buffers: vec![part(EXACT, exact), part(STAND_IN, stand_in)],
                push: push.to_vec(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [groups, 1, 1],
                ..Config::default()
            };
            let buffers = run(&m, &cfg, res).expect("run").buffers;
            out.push((words(&buffers[0]), words(&buffers[1])));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// Each probed value's stand-in is its magnitude, a NaN's +inf, and, flushing, a denormal's
    /// the smallest normal: the kernel's, as the host computes it; the exact buffer is only read.
    #[test]
    fn each_probed_value_stands_in_as_its_magnitude() {
        for flush in [false, true] {
            let want: Vec<u32> = PROBED
                .iter()
                .map(|&w| DepthMagnitudeModule::stand_in(w, flush))
                .collect();
            let (exact, stand_in) = ran(&PROBED, &[0; 18], 18, false, flush, 1);
            assert_eq!(exact, PROBED, "the exact words are only read");
            assert_eq!(stand_in, want, "flush {flush}");
            for (&held, &standing) in PROBED.iter().zip(&want) {
                let magnitude = held & 0x7fff_ffff;
                let expected = match magnitude {
                    m if m > 0x7f80_0000 => 0x7f80_0000,
                    m if flush && m != 0 && m < 0x0080_0000 => 0x0080_0000,
                    m => m,
                };
                assert_eq!(standing, expected, "{held:08x} flush {flush}");
            }
        }
    }

    /// A merge stores each exact word whose stand-in the pass left without its sign, as Metal
    /// stores it (a NaN's payload and a denormal kept), and the pass's word wherever it differs; a
    /// word the pass wrote with its stand-in's value is the exact word without its sign, which
    /// for a negative one is the pass's (|D| over -D), and for a NaN not (the module docs' corner).
    #[test]
    fn a_merge_stores_the_words_the_pass_left_without_their_signs() {
        let exact = [
            0xbf00_0000,
            0xffc0_0001,
            0x3e80_0000,
            0x8000_0001,
            0xc000_0000,
            0x8000_0000,
            0x7fc0_0001,
            0x8000_0002,
        ];
        // The pass left the first, second, sixth and eighth, wrote 0.75 over the third, 0 over the
        // fourth and +inf over the NaN, and wrote 2's stand-in, 2, over -2.
        let drawn = [
            0x3f00_0000,
            0x7f80_0000,
            0x3f40_0000,
            0x0000_0000,
            0x4000_0000,
            0x0000_0000,
            0x7f80_0000,
            0x0080_0000,
        ];
        let (merged, stand_in) = ran(&exact, &drawn, 8, true, true, 1);
        assert_eq!(stand_in, drawn, "the pass's words are only read");
        assert_eq!(
            merged,
            [
                0x3f00_0000,
                0x7fc0_0001,
                0x3f40_0000,
                0x0000_0000,
                0x4000_0000,
                0x0000_0000,
                0x7fc0_0001,
                0x0000_0002
            ]
        );
        // Not flushing, a denormal's stand-in is its magnitude, so 0 and 2^-126 there are writes.
        let (merged, _) = ran(&exact, &drawn, 8, true, false, 1);
        assert_eq!(merged[3], 0x0000_0000);
        assert_eq!(merged[7], 0x0080_0000);
    }

    /// A strip stores every exact word without its sign, a NaN's payload and a denormal kept, and
    /// leaves the stand-in buffer as it was; a word past the count stays.
    #[test]
    fn a_strip_stores_every_word_without_its_sign() {
        let exact = [
            0xbf00_0000,
            0xffc0_0001,
            0x8000_0001,
            0x8000_0000,
            0x3e80_0000,
            0xff80_0000,
            0xbf80_0000,
        ];
        let junk = [0xeeee_eeee; 7];
        let (stripped, stand_in) = ran_with(&exact, &junk, DepthMagnitudeModule::strip(6), 1);
        assert_eq!(stand_in, junk, "the stand-in buffer is not written");
        assert_eq!(
            stripped,
            [
                0x3f00_0000,
                0x7fc0_0001,
                0x0000_0001,
                0x0000_0000,
                0x3e80_0000,
                0x7f80_0000,
                0xbf80_0000
            ]
        );
    }

    /// Invocations past the words write nothing, and every word of several workgroups is taken.
    #[test]
    fn every_word_stands_in_across_workgroups_and_none_past_them() {
        let count = 64 * 3 + 5;
        let exact: Vec<u32> = (0..count + 7).map(|i| 0x8000_0000 | (i * 0x0101)).collect();
        let (_, stand_in) = ran(
            &exact,
            &vec![0xeeee_eeee; exact.len()],
            count,
            false,
            false,
            4,
        );
        for (i, &w) in stand_in.iter().enumerate() {
            match (i as u32) < count {
                true => assert_eq!(w, exact[i] & 0x7fff_ffff, "word {i}"),
                false => assert_eq!(w, 0xeeee_eeee, "word {i} past the count"),
            }
        }
    }
}
