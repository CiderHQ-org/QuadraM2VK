//! Globals, functions, bodies, and the parts a module is assembled from.

use super::attrs::{FnAttrs, ParamAttrs};
use super::constant::ConstTable;
use super::id::{Arena, BlockId, ConstId, FuncId, GlobalId, InstId, MdId, Sym, TypeId};
use super::inst::{Align, CallConv, Inst};
use super::layout::DataLayout;
use super::metadata::{Md, NamedMd};
use super::symbols::Symbols;
use super::types::{AddrSpace, TypeTable};
use crate::budget::Meter;
use crate::error::Error;

/// Linkage of a global or function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Linkage {
    /// `external` (no keyword): visible to other modules.
    #[default]
    External,
    /// `available_externally`: a copy of another module's definition, for inlining only.
    AvailableExternally,
    /// `linkonce`: merged with same-named definitions; may be dropped when unreferenced.
    LinkOnceAny,
    /// `linkonce_odr`: `linkonce` where every definition is equivalent.
    LinkOnceOdr,
    /// `weak`: merged with same-named definitions, but kept when unreferenced.
    WeakAny,
    /// `weak_odr`: `weak` where every definition is equivalent.
    WeakOdr,
    /// `appending`: an array concatenated with same-named arrays at link time.
    Appending,
    /// `internal`: local to the module.
    Internal,
    /// `private`: local to the module and absent from its symbol table.
    Private,
    /// `extern_weak`: a weak reference to an external symbol, null when unresolved.
    ExternalWeak,
    /// `common`: a zero-initialized tentative definition, merged at link time.
    Common,
}

impl Linkage {
    /// Every linkage.
    #[cfg(test)]
    pub const ALL: [Linkage; 11] = [
        Linkage::External,
        Linkage::AvailableExternally,
        Linkage::LinkOnceAny,
        Linkage::LinkOnceOdr,
        Linkage::WeakAny,
        Linkage::WeakOdr,
        Linkage::Appending,
        Linkage::Internal,
        Linkage::Private,
        Linkage::ExternalWeak,
        Linkage::Common,
    ];

    /// The LLVM keyword, for the printer (empty for external).
    pub fn keyword(self) -> &'static str {
        match self {
            Linkage::External => "",
            Linkage::AvailableExternally => "available_externally",
            Linkage::LinkOnceAny => "linkonce",
            Linkage::LinkOnceOdr => "linkonce_odr",
            Linkage::WeakAny => "weak",
            Linkage::WeakOdr => "weak_odr",
            Linkage::Appending => "appending",
            Linkage::Internal => "internal",
            Linkage::Private => "private",
            Linkage::ExternalWeak => "extern_weak",
            Linkage::Common => "common",
        }
    }
}

/// The visibility of a global or function. The translation never reads it; it is kept so a writer
/// reproduces the module. A local symbol has the default, as LLVM's reader gives it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Visibility {
    /// `default`: visible to other modules.
    #[default]
    Default,
    /// `hidden`: not visible outside the linked image.
    Hidden,
    /// `protected`: visible, but references from the linked image bind to its own definition.
    Protected,
}

impl Visibility {
    /// The LLVM keyword (empty for the default), for validation's AIR writer.
    pub fn keyword(self) -> &'static str {
        match self {
            Visibility::Default => "",
            Visibility::Hidden => "hidden",
            Visibility::Protected => "protected",
        }
    }
}

/// Whether the address of a global or function is significant. The translation never reads it;
/// it is kept so a writer reproduces the module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UnnamedAddr {
    /// The address is significant.
    #[default]
    Significant,
    /// `local_unnamed_addr`: the address is not significant within the module.
    Local,
    /// `unnamed_addr`: the address is not significant.
    Global,
}

impl UnnamedAddr {
    /// The LLVM keyword (empty when the address is significant), for validation's AIR writer.
    pub fn keyword(self) -> &'static str {
        match self {
            UnnamedAddr::Significant => "",
            UnnamedAddr::Local => "local_unnamed_addr",
            UnnamedAddr::Global => "unnamed_addr",
        }
    }
}

/// The section of a global or function. The two AIR ABI sections are decoded from their stable
/// `air.*` names; any other section is kept for diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Section {
    /// `air.fc_initializer`.
    FcInitializer,
    /// `air.static_init`.
    StaticInit,
    /// Any other section, by name.
    Other(Sym),
}

/// A module-level global variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Global {
    /// The symbol name, without the `@`.
    pub name: Sym,
    /// The type of the object the global's address points at.
    pub value_ty: TypeId,
    /// The address space the object lives in, and so the space of the global's address.
    pub space: AddrSpace,
    /// `constant`: the object is never written.
    pub constant: bool,
    /// The initializer; `None` for a declaration.
    pub init: Option<ConstId>,
    /// `externally_initialized`: the object may be given its value from outside the module
    /// before execution starts.
    pub externally_initialized: bool,
    /// The linkage.
    pub linkage: Linkage,
    /// The visibility.
    pub visibility: Visibility,
    /// Whether the global's address is significant.
    pub unnamed_addr: UnnamedAddr,
    /// The object's alignment; `None` when the record gives none.
    pub align: Option<Align>,
    /// The section the global is placed in, if any.
    pub section: Option<Section>,
}

/// A function parameter's attributes and diagnostic name. Its type is the function type's.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Param {
    /// The parameter's attributes.
    pub attrs: ParamAttrs,
    /// The name from the function's value symbol table; `None` for an unnamed parameter.
    pub name: Option<Sym>,
}

/// A function definition (with a body) or declaration.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Function {
    /// The symbol name, without the `@`.
    pub name: Sym,
    /// A [`super::Type::Function`].
    pub ty: TypeId,
    /// The linkage.
    pub linkage: Linkage,
    /// The visibility.
    pub visibility: Visibility,
    /// Whether the function's address is significant.
    pub unnamed_addr: UnnamedAddr,
    /// The calling convention.
    pub cconv: CallConv,
    /// The function attributes.
    pub attrs: FnAttrs,
    /// The return-value attributes.
    pub ret_attrs: ParamAttrs,
    /// One entry per parameter of `ty`.
    pub params: Box<[Param]>,
    /// The body of a definition; `None` for a declaration.
    pub body: Option<Body>,
    /// The section the function is placed in, if any.
    pub section: Option<Section>,
    /// The function's alignment; `None` when the record gives none.
    pub align: Option<Align>,
}

/// A function body: blocks in layout order over one instruction arena. Block 0 is the entry.
/// Each block owns a contiguous, non-empty run of instructions (PHIs first, the terminator
/// last), and the runs tile the arena in block order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Body {
    /// The blocks, in layout order.
    pub blocks: Arena<BlockId, Block>,
    /// Every instruction, block after block.
    pub insts: Arena<InstId, Inst>,
}

/// A basic block: the half-open instruction run `start..end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Block {
    /// The block's first instruction.
    pub start: InstId,
    /// One past the block's last instruction.
    pub end: InstId,
    /// The label from the function's value symbol table; `None` for an unnamed block.
    pub name: Option<Sym>,
}

impl Block {
    /// The block's instruction ids in order.
    pub fn inst_ids(&self) -> impl DoubleEndedIterator<Item = InstId> {
        (self.start.raw()..self.end.raw()).map(InstId::new)
    }

    /// The number of instructions.
    pub fn len(&self) -> usize {
        self.end.raw().saturating_sub(self.start.raw()) as usize
    }

    /// Whether the range holds no instruction.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The last instruction, which is the terminator in a verified body.
    pub fn last(&self) -> Option<InstId> {
        self.inst_ids().next_back()
    }
}

impl Body {
    /// An empty body with room for exactly `insts` instructions and `blocks` blocks, charged to
    /// the ledger: for a pass that knows the size of the body it builds before it pushes (the
    /// inliner's exact counts, a rewrite's survivors), so the body holds no arena slack beside the
    /// body it replaces.
    pub fn with_capacity(insts: usize, blocks: usize, m: &mut Meter) -> Result<Body, Error> {
        let mut body = Body::default();
        body.insts.reserve_exact(insts, m)?;
        body.blocks.reserve_exact(blocks, m)?;
        Ok(body)
    }

    /// Append `inst` to the instruction arena, charging the ledger for what the body now holds
    /// for it: its boxed payload ([`super::InstKind::heap_bytes`]) and the arena capacity the
    /// slot adds.
    ///
    /// Every pass that builds a body from instructions it clones or constructs appends them here,
    /// so a body is on the ledger with its calls, attributes, and operand lists, not only its
    /// slots. The decoder alone appends with [`Arena::push`]: it charges each list as it reads it
    /// from the record, before allocating it.
    pub fn push_inst(&mut self, inst: Inst, m: &mut Meter) -> Result<InstId, Error> {
        m.charge(inst.kind.heap_bytes())?;
        self.insts.push(inst, m)
    }

    /// The heap bytes the body holds: its arenas' capacity and every instruction's boxed payload.
    /// Building a body charged at least this much: [`Body::push_inst`] charges each payload and
    /// the arenas charge their capacity, and the decoder charges each list it reads at least at
    /// its final length. In-place edits only shrink a payload (a PHI losing entries), never grow
    /// one.
    pub fn footprint(&self) -> usize {
        self.insts.values().fold(
            self.blocks
                .capacity_bytes()
                .saturating_add(self.insts.capacity_bytes()),
            |sum, inst| sum.saturating_add(inst.kind.heap_bytes()),
        )
    }

    /// Free the body and return its [`Body::footprint`] to the ledger, where a pass drops a body
    /// it has replaced. Only for a body that was charged as it was built, decoded or appended
    /// through [`Body::push_inst`]: the bytes returned are then at most the bytes charged, so the
    /// ledger keeps counting at least what is live. The walk is one pass over instructions whose
    /// building already ticked.
    pub fn release(self, m: &mut Meter) {
        let bytes = self.footprint();
        drop(self);
        m.release(bytes);
    }

    /// The instructions of `block`, in order.
    pub fn block_insts(&self, block: BlockId) -> impl Iterator<Item = (InstId, &Inst)> + '_ {
        self.blocks
            .get(block)
            .into_iter()
            .flat_map(Block::inst_ids)
            .filter_map(|id| Some((id, self.insts.get(id)?)))
    }

    /// The terminator of `block`.
    pub fn terminator(&self, block: BlockId) -> Option<&Inst> {
        self.insts.get(self.blocks.get(block)?.last()?)
    }

    /// The successors of `block` in terminator operand order (duplicates kept). Empty for a
    /// block id out of range.
    pub fn successors(&self, block: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.terminator(block)
            .into_iter()
            .flat_map(|t| t.kind.successors())
    }
}

/// Everything a module consists of. [`super::Module::new`] verifies parts into a module; canon
/// passes take a module apart with `into_parts`, rebuild, and verify again.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModuleParts {
    /// The string interner every `Sym` refers into.
    pub symbols: Symbols,
    /// The type interner.
    pub types: TypeTable,
    /// The constant interner.
    pub consts: ConstTable,
    /// The global variables.
    pub globals: Arena<GlobalId, Global>,
    /// The functions, defined and declared.
    pub funcs: Arena<FuncId, Function>,
    /// The metadata nodes, numbered as in the bitcode.
    pub metadata: Arena<MdId, Md>,
    /// The named metadata lists, in bitcode order.
    pub named_md: Vec<NamedMd>,
    /// The target triple; `None` when the module gives none.
    pub triple: Option<Sym>,
    /// The `source_filename`; `None` when the module gives none.
    pub source_filename: Option<Sym>,
    /// The target data layout.
    pub data_layout: DataLayout,
}
