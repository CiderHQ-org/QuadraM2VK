//! `DataLayout`: the module's typed layout rules and the one source of layout truth.
//!
//! [`DataLayout::parse`] turns the module's layout string into typed rules; unknown specifiers
//! are an error. [`TypeLayouts`] applies them to every type of a [`TypeTable`] in one forward
//! sweep (children precede parents, so there is no recursion) and answers `alloc_size`,
//! `store_size`, `abi_align`, and struct field offsets. The rules are LLVM's: an integer without
//! an exact entry takes the next larger entry (else the largest), a vector without an entry is
//! naturally aligned to its store size rounded up to a power of two, and a struct is aligned to
//! the larger of its fields and the aggregate (`a`) entry unless packed. AIR's `v96:128:128`
//! therefore makes `float3` occupy 16 bytes.

use std::collections::BTreeMap;

use super::id::{Idx, TypeId};
use super::types::{Type, TypeTable};
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{BitcodeError, InputError, InputErrorKind, Location};

/// Byte order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Endian {
    /// `e`: little-endian, LLVM's default.
    #[default]
    Little,
    /// `E`: big-endian.
    Big,
}

/// ABI and preferred alignment, in bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AlignSpec {
    /// The ABI (required) alignment.
    pub abi_bits: u32,
    /// The preferred alignment, at least the ABI one.
    pub pref_bits: u32,
}

/// One address space's pointer layout, in bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PointerSpec {
    /// The pointer's size.
    pub size_bits: u32,
    /// The ABI (required) alignment.
    pub abi_bits: u32,
    /// The preferred alignment, at least the ABI one.
    pub pref_bits: u32,
    /// The width of address arithmetic (GEP offsets); the size when not given.
    pub index_bits: u32,
}

/// Function pointer alignment (`F`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FnPtrAlign {
    /// `Fi<n>`: independent of function alignment.
    Independent(u32),
    /// `Fn<n>`: a multiple of function alignment.
    MultipleOfFunctionAlign(u32),
}

/// The typed rules of a `target datalayout` string.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DataLayout {
    /// The byte order (`e` or `E`).
    pub endian: Endian,
    /// Keyed by address space; space 0 is the fallback for spaces without an entry.
    pub pointers: BTreeMap<u32, PointerSpec>,
    /// Keyed by bit width.
    pub ints: BTreeMap<u32, AlignSpec>,
    /// Floating-point alignments (`f`), keyed by bit width.
    pub floats: BTreeMap<u32, AlignSpec>,
    /// Vector alignments (`v`), keyed by the vector's total bit width.
    pub vectors: BTreeMap<u32, AlignSpec>,
    /// The `a` entry; an ABI alignment of 0 means 1 byte.
    pub aggregate: AlignSpec,
    /// Native integer widths (`n`).
    pub native_ints: Box<[u32]>,
    /// Natural stack alignment in bits (`S`), if specified.
    pub stack_align_bits: Option<u32>,
    /// The address space of `alloca`s (`A`), 0 when not given.
    pub alloca_space: u32,
    /// The address space of functions (`P`), 0 when not given.
    pub program_space: u32,
    /// The default address space of globals (`G`), 0 when not given.
    pub globals_space: u32,
    /// The mangling mode character (`m:`), diagnostics only.
    pub mangling: Option<u8>,
    /// Function pointer alignment (`F`), if specified.
    pub fn_ptr_align: Option<FnPtrAlign>,
    /// Non-integral address spaces (`ni:`).
    pub non_integral: Box<[u32]>,
}

impl Default for DataLayout {
    /// LLVM's defaults (the rules an empty layout string denotes).
    fn default() -> DataLayout {
        let spec = |abi_bits, pref_bits| AlignSpec {
            abi_bits,
            pref_bits,
        };
        DataLayout {
            endian: Endian::Little,
            pointers: BTreeMap::from([(
                0,
                PointerSpec {
                    size_bits: 64,
                    abi_bits: 64,
                    pref_bits: 64,
                    index_bits: 64,
                },
            )]),
            ints: BTreeMap::from([
                (1, spec(8, 8)),
                (8, spec(8, 8)),
                (16, spec(16, 16)),
                (32, spec(32, 32)),
                (64, spec(32, 64)),
            ]),
            floats: BTreeMap::from([
                (16, spec(16, 16)),
                (32, spec(32, 32)),
                (64, spec(64, 64)),
                (128, spec(128, 128)),
            ]),
            vectors: BTreeMap::from([(64, spec(64, 64)), (128, spec(128, 128))]),
            aggregate: spec(0, 64),
            native_ints: Box::new([]),
            stack_align_bits: None,
            alloca_space: 0,
            program_space: 0,
            globals_space: 0,
            mangling: None,
            fn_ptr_align: None,
            non_integral: Box::new([]),
        }
    }
}

fn bad_layout() -> Error {
    Error::InvalidInput(InputError::new(
        InputErrorKind::Bitcode(BitcodeError::BadDataLayout),
        Location::module(),
    ))
}

fn num(s: &str) -> Result<u32, Error> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad_layout());
    }
    s.parse::<u32>().map_err(|_| bad_layout())
}

/// A byte-granular power-of-two alignment in bits (`0` allowed only where `zero_ok`).
fn align_bits(bits: u32, zero_ok: bool) -> Result<u32, Error> {
    if (bits == 0 && zero_ok)
        || (bits >= 8 && bits.is_multiple_of(8) && (bits / 8).is_power_of_two())
    {
        Ok(bits)
    } else {
        Err(bad_layout())
    }
}

fn align_spec(fields: &[&str], zero_ok: bool) -> Result<AlignSpec, Error> {
    let (abi, pref) = match fields {
        [abi] => (num(abi)?, None),
        [abi, pref] => (num(abi)?, Some(num(pref)?)),
        _ => return Err(bad_layout()),
    };
    let abi_bits = align_bits(abi, zero_ok)?;
    let pref_bits = align_bits(pref.unwrap_or(abi), zero_ok)?;
    if pref_bits < abi_bits {
        return Err(bad_layout());
    }
    Ok(AlignSpec {
        abi_bits,
        pref_bits,
    })
}

fn space(s: &str) -> Result<u32, Error> {
    if s.is_empty() {
        Ok(0)
    } else {
        num(s)
    }
}

impl DataLayout {
    /// Parse a `target datalayout` string. Every specifier LLVM defines is recognized; any other
    /// is [`BitcodeError::BadDataLayout`], as is any malformed or non-byte-granular value.
    pub fn parse(s: &str) -> Result<DataLayout, Error> {
        let mut dl = DataLayout::default();
        if s.is_empty() {
            return Ok(dl);
        }
        for spec in s.split('-') {
            let mut chars = spec.chars();
            let head = chars.next().ok_or_else(bad_layout)?;
            let rest = chars.as_str();
            match head {
                'e' | 'E' if rest.is_empty() => {
                    dl.endian = if head == 'e' {
                        Endian::Little
                    } else {
                        Endian::Big
                    };
                }
                'm' => {
                    let mode = rest.strip_prefix(':').ok_or_else(bad_layout)?;
                    let &[mode] = mode.as_bytes() else {
                        return Err(bad_layout());
                    };
                    if !b"emoxwla".contains(&mode) {
                        return Err(bad_layout());
                    }
                    dl.mangling = Some(mode);
                }
                'S' => dl.stack_align_bits = Some(align_bits(num(rest)?, true)?),
                'A' => dl.alloca_space = num(rest)?,
                'P' => dl.program_space = num(rest)?,
                'G' => dl.globals_space = num(rest)?,
                'F' => {
                    let (kind, n) = rest.split_at_checked(1).ok_or_else(bad_layout)?;
                    let n = align_bits(num(n)?, false)?;
                    dl.fn_ptr_align = Some(match kind {
                        "i" => FnPtrAlign::Independent(n),
                        "n" => FnPtrAlign::MultipleOfFunctionAlign(n),
                        _ => return Err(bad_layout()),
                    });
                }
                'n' if rest.starts_with('i') => {
                    let spaces = rest.strip_prefix("i:").ok_or_else(bad_layout)?;
                    dl.non_integral = spaces.split(':').map(num).collect::<Result<_, _>>()?;
                    if dl.non_integral.contains(&0) {
                        return Err(bad_layout());
                    }
                }
                'n' => {
                    dl.native_ints = rest.split(':').map(num).collect::<Result<_, _>>()?;
                    if dl.native_ints.contains(&0) {
                        return Err(bad_layout());
                    }
                }
                'p' => {
                    let mut fields = rest.split(':');
                    let sp = space(fields.next().ok_or_else(bad_layout)?)?;
                    let fields: Vec<&str> = fields.collect();
                    let (size, abi, pref, index) = match fields.as_slice() {
                        [size, abi] => (num(size)?, num(abi)?, None, None),
                        [size, abi, pref] => (num(size)?, num(abi)?, Some(num(pref)?), None),
                        [size, abi, pref, index] => {
                            (num(size)?, num(abi)?, Some(num(pref)?), Some(num(index)?))
                        }
                        _ => return Err(bad_layout()),
                    };
                    let abi_bits = align_bits(abi, false)?;
                    let pref_bits = align_bits(pref.unwrap_or(abi), false)?;
                    let index_bits = index.unwrap_or(size);
                    if size == 0
                        || !size.is_multiple_of(8)
                        || pref_bits < abi_bits
                        || index_bits == 0
                        || index_bits > size
                    {
                        return Err(bad_layout());
                    }
                    dl.pointers.insert(
                        sp,
                        PointerSpec {
                            size_bits: size,
                            abi_bits,
                            pref_bits,
                            index_bits,
                        },
                    );
                }
                'a' => {
                    // `a:abi[:pref]`; the legacy `a0:` spelling carries a size that must be 0.
                    let fields: Vec<&str> = rest.split(':').collect();
                    let (size, fields) = fields.split_first().ok_or_else(bad_layout)?;
                    if !(size.is_empty() || *size == "0") {
                        return Err(bad_layout());
                    }
                    dl.aggregate = align_spec(fields, true)?;
                }
                'i' | 'f' | 'v' => {
                    let fields: Vec<&str> = rest.split(':').collect();
                    let (size, fields) = fields.split_first().ok_or_else(bad_layout)?;
                    let size = num(size)?;
                    if size == 0 || size >= 1 << 24 {
                        return Err(bad_layout());
                    }
                    let spec = align_spec(fields, false)?;
                    let table = match head {
                        'i' => {
                            if size == 8 && spec.abi_bits != 8 {
                                return Err(bad_layout());
                            }
                            &mut dl.ints
                        }
                        'f' => &mut dl.floats,
                        _ => &mut dl.vectors,
                    };
                    table.insert(size, spec);
                }
                _ => return Err(bad_layout()),
            }
        }
        Ok(dl)
    }

    /// The pointer layout of address space `space` (space 0's when it has none).
    pub fn pointer(&self, space: u32) -> PointerSpec {
        self.pointers
            .get(&space)
            .or_else(|| self.pointers.get(&0))
            .copied()
            .unwrap_or(PointerSpec {
                size_bits: 64,
                abi_bits: 64,
                pref_bits: 64,
                index_bits: 64,
            })
    }

    /// ABI alignment in bytes of an integer of `bits`.
    fn int_align(&self, bits: u32) -> u64 {
        let spec = self
            .ints
            .range(bits..)
            .next()
            .or_else(|| self.ints.iter().next_back())
            .map(|(_, s)| s.abi_bits);
        bytes_of(spec.unwrap_or(8))
    }

    /// ABI alignment in bytes of a float of `bits` (natural when the layout has no entry).
    fn float_align(&self, bits: u32) -> u64 {
        self.floats.get(&bits).map_or_else(
            || natural(u64::from(bits).div_ceil(8)),
            |s| bytes_of(s.abi_bits),
        )
    }

    /// ABI alignment in bytes of a vector of `bits` total (natural when the layout has no entry).
    fn vector_align(&self, bits: u64, store: u64) -> u64 {
        u32::try_from(bits)
            .ok()
            .and_then(|b| self.vectors.get(&b))
            .map_or_else(|| natural(store), |s| bytes_of(s.abi_bits))
    }
}

fn bytes_of(bits: u32) -> u64 {
    (u64::from(bits) / 8).max(1)
}

/// Natural alignment: the store size rounded up to a power of two.
fn natural(store: u64) -> u64 {
    store.max(1).checked_next_power_of_two().unwrap_or(1 << 63)
}

fn round_up(value: u64, align: u64) -> Option<u64> {
    let rem = value.checked_rem(align)?;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(align - rem)
    }
}

/// The layout of one sized type, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    /// Bytes an allocation of the type occupies (store size rounded up to `abi_align`).
    pub alloc_size: u64,
    /// Bytes a store of the type writes.
    pub store_size: u64,
    /// ABI alignment.
    pub abi_align: u64,
    /// Size in bits of a scalar or vector (0 for aggregates).
    size_bits: u64,
}

/// Layouts of every type in a [`TypeTable`] under one [`DataLayout`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeLayouts {
    /// `None` for unsized types and for sized types whose size overflows `u64`.
    layouts: Vec<Option<Layout>>,
    /// Struct field offsets, keyed by struct type, in field order.
    offsets: BTreeMap<TypeId, Box<[u64]>>,
}

impl TypeLayouts {
    /// Lay out every type of `types`.
    pub fn new(
        dl: &DataLayout,
        types: &TypeTable,
        meter: &mut Meter,
    ) -> Result<TypeLayouts, Error> {
        let mut layouts: Vec<Option<Layout>> = Vec::new();
        meter.reserve(&mut layouts, types.len())?;
        let mut offsets = BTreeMap::new();
        for (id, ty) in types.iter() {
            meter.tick(1)?;
            let get = |t: TypeId| layouts.get(t.index()).copied().flatten();
            let layout = match ty {
                Type::Int(bits) => scalar(u64::from(*bits), dl.int_align(u32::from(*bits))),
                Type::Half | Type::BFloat | Type::Float | Type::Double => {
                    let bits = ty.float_bits().unwrap_or(32);
                    scalar(u64::from(bits), dl.float_align(bits))
                }
                Type::Ptr { space } => {
                    let p = dl.pointer(space.raw());
                    scalar(u64::from(p.size_bits), bytes_of(p.abi_bits))
                }
                Type::Vector { elem, len } => get(*elem).and_then(|e| {
                    let bits = e.size_bits.checked_mul(u64::from(*len))?;
                    let store = bits.div_ceil(8);
                    let align = dl.vector_align(bits, store);
                    Some(Layout {
                        alloc_size: round_up(store, align)?,
                        store_size: store,
                        abi_align: align,
                        size_bits: bits,
                    })
                }),
                Type::Array { elem, len } => get(*elem).and_then(|e| {
                    let size = e.alloc_size.checked_mul(*len)?;
                    Some(Layout {
                        alloc_size: size,
                        store_size: size,
                        abi_align: e.abi_align,
                        size_bits: 0,
                    })
                }),
                Type::Struct { fields, packed, .. } => {
                    let mut end = 0u64;
                    let mut align = if *packed {
                        1
                    } else {
                        bytes_of(dl.aggregate.abi_bits)
                    };
                    let mut field_offsets = Vec::new();
                    meter.reserve(&mut field_offsets, fields.len())?;
                    let mut ok = true;
                    for &f in fields.iter() {
                        let Some(fl) = get(f) else {
                            ok = false;
                            break;
                        };
                        let fa = if *packed { 1 } else { fl.abi_align };
                        let Some(off) = round_up(end, fa) else {
                            ok = false;
                            break;
                        };
                        field_offsets.push(off);
                        let Some(next) = off.checked_add(fl.alloc_size) else {
                            ok = false;
                            break;
                        };
                        end = next;
                        align = align.max(fa);
                    }
                    match (ok, round_up(end, align)) {
                        (true, Some(size)) => {
                            offsets.insert(id, field_offsets.into_boxed_slice());
                            Some(Layout {
                                alloc_size: size,
                                store_size: size,
                                abi_align: align,
                                size_bits: 0,
                            })
                        }
                        _ => None,
                    }
                }
                Type::Void
                | Type::Label
                | Type::Metadata
                | Type::OpaqueStruct { .. }
                | Type::Function { .. } => None,
            };
            layouts.push(layout);
        }
        Ok(TypeLayouts { layouts, offsets })
    }

    /// The layout of `ty`, if it is sized and its size fits `u64`.
    pub fn get(&self, ty: TypeId) -> Option<Layout> {
        self.layouts.get(ty.index()).copied().flatten()
    }

    /// Bytes an allocation of `ty` occupies.
    pub fn alloc_size(&self, ty: TypeId) -> Option<u64> {
        self.get(ty).map(|l| l.alloc_size)
    }

    /// Bytes a store of `ty` writes.
    pub fn store_size(&self, ty: TypeId) -> Option<u64> {
        self.get(ty).map(|l| l.store_size)
    }

    /// ABI alignment of `ty` in bytes.
    pub fn abi_align(&self, ty: TypeId) -> Option<u64> {
        self.get(ty).map(|l| l.abi_align)
    }

    /// Byte offsets of a struct's fields.
    pub fn field_offsets(&self, ty: TypeId) -> Option<&[u64]> {
        self.offsets.get(&ty).map(|o| &**o)
    }

    /// The bytes these tables hold, as a ledger charges them: the layout vector, and per struct its
    /// offsets plus a 48-byte map-entry charge.
    pub fn retained_bytes(&self) -> usize {
        self.layouts.capacity() * std::mem::size_of::<Option<Layout>>()
            + self
                .offsets
                .values()
                .map(|o| o.len() * std::mem::size_of::<u64>() + 48)
                .sum::<usize>()
    }
}

fn scalar(bits: u64, align: u64) -> Option<Layout> {
    let store = bits.div_ceil(8);
    Some(Layout {
        alloc_size: round_up(store, align)?,
        store_size: store,
        abi_align: align,
        size_bits: bits,
    })
}
