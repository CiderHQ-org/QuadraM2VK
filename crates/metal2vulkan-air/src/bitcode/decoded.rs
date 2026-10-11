//! `Decoded` assembly: a verified module, its AIR metadata ABI, and the decode-only pointee table.

use super::{decode_module, EnumAttrTable, ModuleTables};
use crate::budget::Meter;
use crate::error::Error;
use crate::ir::{Const, FuncId, Idx, Module, TypeId};
use crate::{abi, container};

/// A decoded, verified bitcode module with its AIR metadata ABI; canonicalization (`specialize`)
/// consumes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    /// The verified IR module.
    pub module: Module,
    /// The module's metadata ABI, decoded and cross-checked against `module` and `pointees`
    /// ([`abi::decode`]).
    pub abi: abi::AirAbi,
    /// The pointees of the bitcode's typed pointer parameters and returns, which the ABI's
    /// cross-check read; nothing after canonicalization may read it.
    pub pointees: PointeeTable,
    /// The enum attributes the IR does not interpret, by position, when the decode asked for
    /// them ([`DecodeOptions::enum_attrs`]); empty otherwise. Nothing after decode reads them
    /// but a writer that hands the module back.
    pub enum_attrs: EnumAttrTable,
}

/// What a decode keeps beyond what translation reads. The default keeps neither, which is what
/// translation uses: neither changes what the module computes, and neither is free (each costs
/// memory or a pass over every instruction).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Record the enum attributes the IR does not interpret, by function and call site
    /// ([`Decoded::enum_attrs`]), for a writer that must hand Metal's compiler the module as
    /// written.
    pub enum_attrs: bool,
    /// Remove each marker declaration (`llvm.lifetime.*`, `llvm.assume`,
    /// `llvm.experimental.noalias.scope.decl`) that no use survives once its calls are dropped,
    /// with its address constant, every constant only the dropped calls named, and the types only
    /// the removed used, renumbering what remains in order: what is left of such a declaration and
    /// its calls is how an LLVM major spells the marker, so a decoded module compared across majors
    /// (a golden printout, the typed and opaque spellings of one program) needs it gone.
    /// Translation does not: entry pruning keeps only what the entry reaches.
    pub drop_unused_markers: bool,
}

/// The typed-pointer pointees the IR erases, kept for decode-time ABI cross-checks only (parameter
/// roles against types, `air.arg_type_size` and `air.struct_type_info` against the DataLayout).
/// Nothing after `air::canon` may read it.
///
/// For every function, the pointee of each parameter (and of the return value) that the
/// bitcode spells as a typed pointer, as an erased IR type; `None` for non-pointers and opaque
/// pointers. Pointees are per function, from the function's own bitcode type, never looked up
/// through the erased `Function::ty` (which functions differing only in pointees share).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PointeeTable {
    funcs: Vec<FnPointees>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FnPointees {
    ret: Option<TypeId>,
    params: Box<[Option<TypeId>]>,
}

impl PointeeTable {
    /// The pointee of parameter `index` of `func`, when it is a typed pointer.
    pub fn param(&self, func: FuncId, index: usize) -> Option<TypeId> {
        self.funcs
            .get(func.index())?
            .params
            .get(index)
            .copied()
            .flatten()
    }

    /// The pointees of every parameter of `func` (`None` for an unknown function).
    pub fn params(&self, func: FuncId) -> Option<&[Option<TypeId>]> {
        self.funcs.get(func.index()).map(|f| &*f.params)
    }

    /// The pointee of `func`'s return value, when it is a typed pointer.
    pub fn ret(&self, func: FuncId) -> Option<TypeId> {
        self.funcs.get(func.index())?.ret
    }

    /// Whether any pointee is recorded (false for opaque-pointer bitcode).
    pub fn is_empty(&self) -> bool {
        self.funcs
            .iter()
            .all(|f| f.ret.is_none() && f.params.iter().all(Option::is_none))
    }

    fn build(t: &ModuleTables, m: &mut Meter) -> Result<PointeeTable, Error> {
        let mut funcs = Vec::new();
        m.reserve(&mut funcs, t.parts.funcs.len())?;
        funcs.resize(t.parts.funcs.len(), FnPointees::default());
        let erased = |idx: u32| -> Option<TypeId> {
            let pointee = t.types.pointee(u64::from(idx))?;
            t.types.ty(u64::from(pointee))
        };
        for v in &t.values {
            m.tick(1)?;
            let Some(Const::FuncAddr(f)) =
                v.id.and_then(|c| t.parts.consts.get(c)).map(|c| &c.value)
            else {
                continue;
            };
            let Some(fn_bc) = v.object_bc_type else {
                continue;
            };
            let Some(slot) = funcs.get_mut(f.index()) else {
                continue;
            };
            let bc_params = t.types.params(u64::from(fn_bc)).unwrap_or(&[]);
            let mut params = Vec::new();
            m.reserve(&mut params, bc_params.len())?;
            params.extend(bc_params.iter().map(|&p| erased(p)));
            *slot = FnPointees {
                ret: t.types.ret(u64::from(fn_bc)).and_then(erased),
                params: params.into_boxed_slice(),
            };
        }
        Ok(PointeeTable { funcs })
    }
}

/// Decode and verify one bitcode module, raw (`BC 0xC0DE`) or wrapped (`0x0B17C0DE`), then
/// decode its AIR metadata ABI and cross-check it against the module ([`abi::decode`]). Keeps
/// what translation reads ([`DecodeOptions::default`]).
pub fn decode(bitcode: &[u8], m: &mut Meter) -> Result<Decoded, Error> {
    decode_with(bitcode, DecodeOptions::default(), m)
}

/// [`decode`], keeping what `opts` asks for beyond what translation reads.
pub fn decode_with(bitcode: &[u8], opts: DecodeOptions, m: &mut Meter) -> Result<Decoded, Error> {
    let (_, stream) = container::bitcode_stream(bitcode)?;
    let tables = decode_module(stream, opts, m)?;
    let pointees = PointeeTable::build(&tables, m)?;
    let ModuleTables {
        parts, enum_attrs, ..
    } = tables;
    let module = Module::new(parts, m)?;
    let abi = abi::decode(&module, &pointees, m)?;
    Ok(Decoded {
        module,
        abi,
        pointees,
        enum_attrs,
    })
}
