//! Imageblock slice writes (`air.write_imageblock_slice_to_texture_2d[_array]`).
//!
//! `texture.write(img.slice(cell->field[, size]), coord[, layer])` copies one field of a block of
//! the kernel's explicit imageblock (`access::imageblock`) into the texture, one texel per cell.
//! The operands are the texture, the field pointer, an explicit-size flag, the block's origin, its
//! size, the texture coordinate, the layer (arrays), the level, a trailing flag, and the access.
//! Metal's semantics, as the authored cases measured them:
//!
//! - the block starts at the imageblock's own origin; the pointer selects the **field** (its byte
//!   offset within the cell) and its cell is ignored;
//! - the block is the size operand when the flag is set, else the whole imageblock
//!   (`width · height`): an implicit slice is the imageblock's extent, not the threadgroup's;
//! - cell `(x, y)` of the block lands at `coord + (x, y)` on the level, and a texel outside the
//!   texture is discarded, like any write (the guard `write_guard` decides); a level other than
//!   the constant 0 is written through the slot's level views, and a level the texture lacks
//!   drops the slice;
//! - the element `E` the intrinsic names (`f32`, `f16`, `i32`, `i16`, 1, 2, or 4 lanes) is the
//!   texel's leading lanes and every later lane is 0, alpha included, in every pixel format: a
//!   4-lane element is the whole texel, a 1-lane one `(v, 0, 0, 0)`, a 2-lane one `(x, y, 0, 0)`,
//!   which the format then keeps as a texture write keeps any texel (an `R` format the first
//!   lane, `A8Unorm` the alpha of 0). Measured on an M4 and the M3 Max (a probe `slice`) for
//!   `half`, `float`, `short` and `int` elements of each width into formats of one, two and four
//!   components, through a texture slot and through an argument buffer, with Metal's validation
//!   layer on and off alike.
//!
//! The copy is one counted loop over the block's cells in the writing invocation (a builder
//! loop), each iteration a contained arena leaf load
//! (`access::imageblock::load`) and a texture write. A block past the imageblock's extent in
//! either axis is clipped to it (the cells past it are not the imageblock's; Metal leaves them
//! undefined). The counter starts at 0 on every execution of the write.
//!
//! **The threadgroup's copy** ([`Cooperative`], [`cooperate`]). A kernel's slice is usually written
//! by one thread (`if (thread_index == 0)`), whose loop then holds every other thread of its
//! threadgroup at the next barrier while it copies `width · height` cells one at a time. Where
//! every invocation of the threadgroup can take the write's part, the whole threadgroup copies the
//! block instead, each cell once: the write is a synchronization of the predicated walk
//! (`simd::divergent`), which every invocation reaches in workgroup-uniform control flow with its
//! own live flag. There a workgroup barrier makes every invocation's cells visible, invocation `i`
//! of the `n` threads copies the cells whose texel `(u, v)` has `(u + v · W) mod n = i` (`W` the
//! imageblock's width: the cells of row-major index `m` over rows `W` cells wide with
//! `m ≡ i - c (mod n)`, `c` the block's first texel's `u + v · W`), and a workgroup barrier keeps
//! every later access of the cells after the copy. Each texel is converted and stored by exactly
//! the code the one-invocation copy runs. In a block some invocations skip, the writing
//! invocations publish the write's operands to the others first: a workgroup vote on the live
//! flags (where no invocation is live, nothing is written), then each word of an operand computed
//! in a block some invocations skip stored to the vote word by the live invocations, which all
//! hold it alike, and read back by every invocation (`simd::workgroup::publish`); every
//! invocation ran a uniform block's operands as the writers did. Several live invocations write
//! one block (their operands are equal), as each would in Metal.
//!
//! The form is taken in a kernel (not a tile function) for a slice to a static texture slot whose
//! every operand (coordinate, layer, level, size) is a constant, a parameter, or a value
//! `simd::uniform::Uniformity` proves the threadgroup holds alike, each published one a `u32` or a
//! vector of them, where the slot is written only by slices and never read, sampled, or updated by
//! an atomic in the function, and is not volatile; every slice to a slot takes it or none does.
//! Then every texel of that slot a dispatch writes is written by the invocation its `(u, v)` names,
//! in that invocation's program order, as the one-invocation copy writes them all in its own: two
//! slices that overlap land in program order without image memory ordering between invocations.
//! Any other slice keeps the one-invocation copy.
//!
//! Only these are pinned (else `Unsupported{Imageblocks, NotImplemented}`): the origin operand the
//! constant zero, the trailing flag the constant `false`, the flag a constant, a field pointer
//! whose roots are all imageblock cells at one exact in-cell offset (in a tile function whose pass
//! keeps fewer bytes of its cells, `LoweringInputs::kept_imageblock_bytes`, a field within them),
//! and a static texture slot or a heap element (a texture loaded from an argument buffer), not a
//! switch. An integer element spells no signedness (neither the AIR symbol nor the texture
//! argument's ABI metadata carries one): its texel class is the slot's from another operation, else
//! the numeric type of the format bound there, the slot's `StorageTextureFormat` input on every
//! profile (a formatless choice keeps the type, `RuntimeFact::StorageTextureNumericType`).
//!
//! A heap element is written as a heap `write` writes it: through its group's formatless storage
//! class, at level 0, the write dropped through the null element, a narrower element as a slot's is
//! (the texel does not depend on the member's format, so no binder check is needed). An integer
//! element's class is the group's from another operation (no slot has a format input to give it).

use std::collections::{BTreeMap, BTreeSet};

use crate::air::intrinsics::{Elem, TexRole, Texture};
use crate::air::ir::{BlockId, InstId, Value, ValueId};
use crate::budget::Meter;
use crate::error::{Error, Feature, RuntimeInput, Unsupported};
use crate::lower::access::imageblock;
use crate::lower::arith::legal::held_on;
use crate::lower::arith::{FloatFmt, Form, Held};
use crate::lower::select::SelectCx;
use crate::lower::simd::uniform::Uniformity;
use crate::lower::strategy::Strategy;
use crate::memory::leaf::{Leaf, Path};
use crate::memory::provenance::imageblock_cell;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::StorageImageClass;
use crate::pipeline::sym::{SampledType, SlotClass, SlotRef, Stage};
use crate::spirv::ids::{DemoteTy, ScalarTy};

use super::{
    coord, image_access, is_zero, mark, observe, slot, storage_level, Coord, ImageRef, Operands,
    TexelClass, TexelConv, TexelForm, TextureStrategy,
};

/// A slice write of the kernel's imageblock to a texture slot or heap element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slice {
    pub image: ImageRef,
    /// The texture coordinate of the block's first cell.
    pub at: Coord,
    /// The level of the slot's level views when it is not the constant 0.
    pub level: Option<ValueId>,
    /// The block's size (a held `ushort2`) when the slice names one; else the imageblock's.
    pub size: Option<ValueId>,
    /// The field's byte offset within a cell.
    pub field: u32,
    pub cell_bytes: u32,
    /// The element in the cell.
    pub leaf: Leaf,
    /// The element's texel conversion; its lanes are the element's.
    pub texel: TexelForm,
    /// Discard a texel outside the texture in the shader (no robustness discards it).
    pub guard: bool,
    /// The cell loads are contained.
    pub contained: bool,
    /// The slot's storage class (`super::storage_class`): a converting one (`RotateArgb`,
    /// `HalfTowardZero`) rounds each texel to Metal's code of each channel.
    pub class: StorageImageClass,
    /// A `float` slice through a heap handle on a device not measured to store it in a
    /// half-float texture as Metal does: the class each texel takes where the handle's texel
    /// class is a half-float format's (`super::Write::heap_half`).
    pub heap_half: Option<StorageImageClass>,
    /// A `float` slice through a heap handle: each texel's NaN lanes quieted where the handle's
    /// texel class is a small float's (`super::Write::heap_small_floats`).
    pub heap_small_floats: bool,
    /// How a `half` element's NaN is written (`super::half_nans`).
    pub half_nans: super::HalfNans,
    /// Each texel of a half-float member stores its bits through the heap group's unsigned
    /// integer twin where the handle's class names one (`super::Write::heap_bits`).
    pub heap_bits: bool,
    /// The whole threadgroup copies the block ([`cooperate`]); `None`, the writing invocation.
    pub cooperative: Option<Cooperative>,
}

/// A slice the whole threadgroup copies (the module docs): a synchronization of the predicated
/// walk, every invocation taking a share of the cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cooperative {
    /// The operands that the writing invocations publish where only some invocations write, each
    /// with its `u32` lanes: those among the coordinate's plane, its layer, the level, and the
    /// size, in that order, that are results of instructions in blocks `Uniformity` does not prove
    /// workgroup uniform. Every invocation holds a constant or a parameter alike, and the one
    /// result of an instruction in a uniform block, which every invocation ran as the writers did.
    pub published: [Option<(ValueId, u8)>; 4],
}

impl Cooperative {
    /// The `u32` words the published operands take.
    pub fn words(&self) -> u32 {
        self.published
            .iter()
            .flatten()
            .map(|&(_, lanes)| u32::from(lanes))
            .sum()
    }
}

/// Whether `s` is a slice the whole threadgroup copies, a synchronization of the predicated walk
/// (`simd::divergent`).
pub(crate) fn is_cooperative(s: &Strategy) -> bool {
    matches!(
        s,
        Strategy::Texture(TextureStrategy::ImageblockSlice(sl)) if sl.cooperative.is_some()
    )
}

impl Slice {
    pub(super) fn words(&self) -> u32 {
        self.leaf.cost()
            + if self.contained { 64 } else { 0 }
            + if self.guard { 48 } else { 0 }
            + super::emit::class_write_words(self.class)
            + self.heap_half.map_or(0, |c| {
                super::HEAP_CLASS_WORDS + super::emit::class_write_words(c) + 32
            })
            + match self.heap_small_floats {
                true => super::heap_half_write_words(StorageImageClass::SmallFloatQuietNan),
                false => 0,
            }
            + self.half_nans.words()
            + if self.heap_bits {
                super::heap_bits_write_words()
            } else {
                0
            }
            + 320
            // The vote and barriers, the share's mapping, and per published word its store under
            // the live flag, two barriers, its read, and its rebuilt lane.
            + self.cooperative.map_or(0, |c| 160 + 40 * c.words())
    }
}

fn imageblocks() -> Error {
    Unsupported::not_implemented(Feature::Imageblocks).into()
}

/// Whether `v` is the constant `i1` `b`.
fn is_flag(cx: &SelectCx<'_>, v: Value, b: bool) -> bool {
    super::const_int(cx, v) == Some(u64::from(b))
}

/// The texel class of an integer element written to `slot`: the slot's from another operation,
/// else the numeric type of the format bound there, which the caller states in the slot's
/// `StorageImageChoice` (an input on every profile: `NeedsInput(StorageTextureFormat)` without
/// one). The choice's declared format gives the type, or, for a formatless choice, the type it
/// keeps (`StorageImageChoice::numeric`); the function then depends on it
/// (`RuntimeFact::StorageTextureNumericType`), so the binder keys the slot's choice even where a
/// formatless declaration serves its format. A bound float format leaves the element's texels
/// undefined: refused.
fn integer_class(cx: &mut SelectCx<'_>, slot: SlotRef) -> Result<TexelClass, Error> {
    let seen = cx
        .families
        .texture
        .slots
        .get(&slot.index)
        .and_then(|s| s.texel);
    let class = match seen {
        Some(c) => Some(c),
        None => {
            let Some(choice) = cx.inputs.storage_images.get(&slot) else {
                return Err(Error::NeedsInput(RuntimeInput::StorageTextureFormat {
                    slot,
                }));
            };
            let numeric = if choice.vk_format == crate::vk::Format::UNDEFINED {
                choice.numeric
            } else {
                crate::bind::neutral::numeric_type(choice.vk_format)
            };
            cx.facts
                .insert(RuntimeFact::StorageTextureNumericType { slot });
            match numeric {
                Some(SampledType::Sint) => Some(TexelClass::Sint),
                Some(SampledType::Uint) => Some(TexelClass::Uint),
                Some(SampledType::Uint64) => Some(TexelClass::Uint64),
                Some(SampledType::Float) | None => None,
            }
        }
    };
    match class {
        Some(c @ (TexelClass::Sint | TexelClass::Uint)) => Ok(c),
        _ => Err(imageblocks()),
    }
}

/// The texel class of an integer element written to heap group `group`: the group's from another
/// operation (its class is then that numeric type's). A group no other operation types has no
/// format input to give it one: refused.
fn heap_integer_class(cx: &SelectCx<'_>, group: u32) -> Result<TexelClass, Error> {
    match cx.families.texture.heap.get(&group).and_then(|h| h.texel) {
        Some(c @ (TexelClass::Sint | TexelClass::Uint)) => Ok(c),
        _ => Err(imageblocks()),
    }
}

/// Select a slice write to `image`, a texture slot or a heap element.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    t: &Texture,
    ops: &Operands<'_>,
    image: ImageRef,
) -> Result<TextureStrategy, Error> {
    let elem = t
        .texel
        .ok_or_else(|| cx.invariant("a slice write without an element type"))?
        .ty;
    let operand = |role| {
        ops.get(role)
            .ok_or_else(|| cx.invariant("a slice write without an operand"))
    };
    let (flag, origin, size, tail, ptr) = (
        operand(TexRole::Opaque(2))?,
        operand(TexRole::Opaque(3))?,
        operand(TexRole::Opaque(4))?,
        operand(TexRole::Opaque(8))?,
        operand(TexRole::Imageblock)?,
    );
    if !is_zero(cx, origin) || !is_flag(cx, tail, false) {
        return Err(imageblocks());
    }
    let size = if is_flag(cx, flag, true) {
        Some(imageblock::operand(cx, size)?)
    } else if is_flag(cx, flag, false) {
        None
    } else {
        return Err(imageblocks());
    };
    let level = storage_level(cx, ops, image)?;
    let cell_bytes = imageblock::use_imageblock(cx, imageblock::CellUse::Read)?;
    let roots_ok = cx
        .memory
        .roots_of(ptr)
        .is_some_and(|r| !r.is_empty() && r.iter().all(|r| imageblock_cell(cx.entry, *r)));
    let lanes = u8::try_from(elem.lanes).map_err(|_| imageblocks())?;
    let (held, conv, lane_bytes) = match elem.elem {
        Elem::F32 => (Held::Float(FloatFmt::F32), Some(TexelConv::F32), 4),
        Elem::F16 => (Held::Float(FloatFmt::Half), Some(TexelConv::F16), 2),
        Elem::I32 => (
            Held::Int {
                bits: 32,
                container: 32,
            },
            None,
            4,
        ),
        Elem::I16 => (
            Held::Int {
                bits: 16,
                container: 32,
            },
            None,
            2,
        ),
        _ => return Err(imageblocks()),
    };
    let bytes = lane_bytes * u32::from(lanes);
    // A tile function's slice of words its pass does not keep is unmeasured
    // (`access::tile::KeptCells`).
    let limit = cx
        .inputs
        .kept_imageblock_bytes
        .map_or(cell_bytes, |kept| kept.min(cell_bytes));
    let field = cx
        .memory
        .offset(ptr)
        .and_then(|o| o.exact())
        .and_then(|o| u32::try_from(o).ok())
        .filter(|o| roots_ok && o.checked_add(bytes).is_some_and(|end| end <= limit))
        .ok_or_else(imageblocks)?;
    let slot = match image {
        ImageRef::Slot(index) => Some(SlotRef {
            stage: cx.stage,
            class: SlotClass::Texture,
            index,
        }),
        ImageRef::Heap { .. } => None,
    };
    let (conv, class) = match conv {
        Some(c) => (c, TexelClass::Float),
        None => {
            let class = match (slot, image) {
                (Some(slot), _) => integer_class(cx, slot)?,
                (None, ImageRef::Heap { group, .. }) => heap_integer_class(cx, group)?,
                (None, ImageRef::Slot(_)) => return Err(cx.invariant("a slot without its ref")),
            };
            let conv = match elem.elem {
                Elem::I32 => TexelConv::I32,
                _ => TexelConv::I16 {
                    signed: class == TexelClass::Sint,
                },
            };
            (conv, class)
        }
    };
    observe(cx, image, t.image, Some(class))?;
    if !image_access(cx, image)?.0 {
        return Err(imageblocks());
    }
    // A slice written to a storage slot bound to a swizzled texture is not measured.
    if super::swizzled_storage(cx, image)? {
        return Err(crate::error::Unsupported::not_implemented(
            crate::error::Feature::TextureSwizzle,
        )
        .into());
    }
    slot::stage_stores(cx)?;
    let at = coord(cx, ops, t.image)?;
    // Every cell's offset is `region + k · cell_bytes + field`, the region 16-byte aligned.
    let align = 1u32 << (cell_bytes | field | 16).trailing_zeros();
    let path = if align >= 4 {
        Path::Aligned
    } else if bytes <= 2 && bytes <= align {
        Path::OneWord
    } else {
        Path::Bytes
    };
    let leaf = Leaf {
        form: Form {
            held: held_on(cx.profile, held)?,
            lanes,
        },
        bytes,
        path,
    };
    let guard = slot::write_guard(cx, false);
    let class = super::storage_class(cx, image)?;
    let heap_half = match image.heap() && conv == TexelConv::F32 {
        true => super::heap_write_class(cx),
        false => None,
    };
    let heap_small_floats = image.heap() && conv == TexelConv::F32;
    cx.families.access.heap_class_index |= heap_half.is_some() || heap_small_floats;
    // A half-float member's bits through the group's unsigned integer twin, where the handle's
    // class names one.
    let heap_bits = super::heap_bits(cx, image, t.image, (conv, false));
    if conv == TexelConv::F32 {
        super::half_storage_facts(cx, image, class, (heap_half, heap_bits))?;
    }
    let half_nans = match conv == TexelConv::F16 {
        true => super::half_nans(cx, image, (false, heap_bits))?,
        false => super::HalfNans::Held,
    };
    mark(cx, image, |u| u.write = true);
    Ok(TextureStrategy::ImageblockSlice(Box::new(Slice {
        image,
        at,
        level,
        size,
        field,
        cell_bytes,
        leaf,
        texel: TexelForm { conv, lanes },
        guard,
        contained: imageblock::contained(cx),
        class,
        heap_half,
        heap_small_floats,
        half_nans,
        heap_bits,
        cooperative: None,
    })))
}

/// Decide which slice writes of the selected body `strategy` the whole threadgroup copies (the
/// module docs), and set their [`Slice::cooperative`]. `given` is the body's uniformity where the
/// caller holds it; it is computed here when a slice may take the form, and returned for the
/// caller's later passes, which release its ledger bytes.
pub(crate) fn cooperate(
    cx: &mut SelectCx<'_>,
    strategy: &mut [Option<Strategy>],
    given: Option<Uniformity>,
    m: &mut Meter,
) -> Result<Option<Uniformity>, Error> {
    let is_slice = |s: &Option<Strategy>| {
        matches!(
            s,
            Some(Strategy::Texture(TextureStrategy::ImageblockSlice(_)))
        )
    };
    if cx.stage != Stage::Kernel
        || crate::lower::access::tile::is_tile(cx)
        || !strategy.iter().any(is_slice)
    {
        return Ok(given);
    }
    let u = match given {
        Some(u) => u,
        None => Uniformity::of(cx.entry, cx.body, m)?,
    };
    // The slots some slice of which keeps the one-invocation copy: one through a switch over
    // several candidate textures does.
    let mut serial: BTreeSet<u32> = BTreeSet::new();
    for s in strategy.iter().flatten() {
        let Strategy::Texture(TextureStrategy::Switch(sw)) = s else {
            continue;
        };
        for (_, arm) in &sw.arms {
            if let TextureStrategy::ImageblockSlice(sl) = arm {
                if let ImageRef::Slot(index) = sl.image {
                    serial.insert(index);
                }
            }
        }
    }
    let homes = homes(cx, strategy, m)?;
    let mut verdicts: Vec<(usize, u32, Option<Cooperative>)> = Vec::new();
    for (i, s) in strategy.iter().enumerate() {
        m.tick(1)?;
        let Some(Strategy::Texture(TextureStrategy::ImageblockSlice(sl))) = s else {
            continue;
        };
        let ImageRef::Slot(index) = sl.image else {
            continue;
        };
        let verdict = match cx.families.texture.slots.get(&index) {
            Some(plan) if slice_only(plan) => cooperative(cx, (&u, &homes), sl),
            _ => None,
        };
        if verdict.is_none() {
            serial.insert(index);
        }
        verdicts.push((i, index, verdict));
    }
    for (i, index, verdict) in verdicts {
        if serial.contains(&index) {
            continue;
        }
        if let Some(Some(Strategy::Texture(TextureStrategy::ImageblockSlice(sl)))) =
            strategy.get_mut(i)
        {
            sl.cooperative = verdict;
            cx.families.texture.cooperative_slices = true;
        }
    }
    Ok(Some(u))
}

/// Whether slot `plan` is reached in the function only by slice writes and queries: no texel
/// write, read, sample, gather, or atomic, and it is not volatile.
fn slice_only(plan: &super::slot::SlotPlan) -> bool {
    let u = &plan.uses;
    !(plan.volatile || u.texel_write || u.read || u.sample || u.compare || u.atomic)
}

/// The operands of slice `sl`: its coordinate's plane, layer and face, its level, and its size.
fn operands(sl: &Slice) -> [Option<ValueId>; 5] {
    [
        Some(sl.at.plane),
        sl.at.layer,
        sl.at.face,
        sl.level,
        sl.size,
    ]
}

/// The block of each instruction whose result some slice write of `strategy` takes as an operand.
fn homes(
    cx: &SelectCx<'_>,
    strategy: &[Option<Strategy>],
    m: &mut Meter,
) -> Result<BTreeMap<InstId, BlockId>, Error> {
    let mut wanted: BTreeSet<InstId> = BTreeSet::new();
    for s in strategy.iter().flatten() {
        let Strategy::Texture(TextureStrategy::ImageblockSlice(sl)) = s else {
            continue;
        };
        for v in operands(sl).into_iter().flatten() {
            if let Some(Value::Inst(at)) = cx.values.value(v) {
                wanted.insert(at);
            }
        }
    }
    let mut homes = BTreeMap::new();
    if wanted.is_empty() {
        return Ok(homes);
    }
    for (b, block) in cx.body.blocks.iter() {
        for id in block.inst_ids() {
            m.tick(1)?;
            if wanted.contains(&id) {
                homes.insert(id, b);
            }
        }
    }
    Ok(homes)
}

/// The threadgroup's copy of slice `sl`, when every operand is one the threadgroup holds alike
/// (`u`) and each one it publishes (the result of an instruction whose block in `homes` is not
/// uniform) is a `u32` or a vector of them.
fn cooperative(
    cx: &SelectCx<'_>,
    (u, homes): (&Uniformity, &BTreeMap<InstId, BlockId>),
    sl: &Slice,
) -> Option<Cooperative> {
    let mut published = [None; 4];
    let mut n = 0;
    for v in operands(sl).into_iter().flatten() {
        let value = cx.values.value(v)?;
        if !u.is_uniform_value(cx.entry, value) {
            return None;
        }
        // A constant or a parameter is held by every invocation where the write runs, and so is
        // the result of an instruction in a uniform block.
        let Value::Inst(at) = value else {
            continue;
        };
        if u.is_uniform(*homes.get(&at)?) || published.iter().flatten().any(|&(p, _)| p == v) {
            continue;
        }
        let ty = cx.body.insts.get(at)?.ty;
        let lanes = match cx.lowered_repr(value, ty).ok()? {
            DemoteTy::Scalar(ScalarTy::Int { bits: 32 }) => 1,
            DemoteTy::Vector {
                elem: ScalarTy::Int { bits: 32 },
                len,
            } => len,
            _ => return None,
        };
        *published.get_mut(n)? = Some((v, lanes));
        n += 1;
    }
    Some(Cooperative { published })
}
