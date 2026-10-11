//! A typed IR builder for tests.
//!
//! Canonicalization, ABI, and lowering tests construct IR through this builder instead of bytes.
//! It never panics: the first construction error is kept and returned by
//! [`ModuleBuilder::finish`], which verifies the module like the decoder does. Instructions may
//! be appended to blocks in any order; `FunctionBuilder::finish` lays each block's run out
//! contiguously and renumbers instruction operands to match.

use super::constant::{Const, Constant};
use super::id::{BlockId, ConstId, FuncId, GlobalId, Idx, InstId, MdId, Sym, TypeId};
use super::inst::{Inst, InstKind, Value};
use super::layout::DataLayout;
use super::metadata::{Md, NamedMd};
use super::module::{Block, Body, Function, Global, Linkage, ModuleParts, Param};
use super::types::{AddrSpace, Type};
use super::verify::Module;
use super::{attrs, inst};
use crate::budget::{Budget, Meter};
use crate::error::Error;

/// Builds a [`Module`].
#[derive(Debug)]
pub struct ModuleBuilder {
    parts: ModuleParts,
    meter: Meter,
    error: Option<Error>,
}

impl Default for ModuleBuilder {
    fn default() -> ModuleBuilder {
        ModuleBuilder::new()
    }
}

/// An id standing in for a failed construction; [`ModuleBuilder::finish`] reports the failure.
const FAILED: u32 = u32::MAX;

impl ModuleBuilder {
    /// An empty module with LLVM's default data layout and no deadline.
    pub fn new() -> ModuleBuilder {
        ModuleBuilder {
            parts: ModuleParts::default(),
            meter: Meter::new(&Budget::default()),
            error: None,
        }
    }

    fn record<T>(&mut self, r: Result<T, Error>, failed: T) -> T {
        match r {
            Ok(v) => v,
            Err(e) => {
                self.error.get_or_insert(e);
                failed
            }
        }
    }

    /// Replace the data layout.
    pub fn set_data_layout(&mut self, dl: DataLayout) {
        self.parts.data_layout = dl;
    }

    /// Intern a string.
    pub fn sym(&mut self, s: &str) -> Sym {
        let r = self.parts.symbols.intern(s.as_bytes(), &mut self.meter);
        self.record(r, Sym::new(FAILED))
    }

    /// Intern a type.
    pub fn ty(&mut self, ty: Type) -> TypeId {
        let r = self.parts.types.intern(ty, &mut self.meter);
        self.record(r, TypeId::new(FAILED))
    }

    /// Intern `void`.
    pub fn void(&mut self) -> TypeId {
        self.ty(Type::Void)
    }

    /// Intern the integer type of `bits` bits.
    pub fn int(&mut self, bits: u8) -> TypeId {
        self.ty(Type::Int(bits))
    }

    /// Intern `float`.
    pub fn float(&mut self) -> TypeId {
        self.ty(Type::Float)
    }

    /// Intern the pointer type into `space`.
    pub fn ptr(&mut self, space: AddrSpace) -> TypeId {
        self.ty(Type::Ptr { space })
    }

    /// Intern the vector of `len` lanes of `elem`.
    pub fn vector(&mut self, elem: TypeId, len: u32) -> TypeId {
        self.ty(Type::Vector { elem, len })
    }

    /// Intern the array of `len` elements of `elem`.
    pub fn array(&mut self, elem: TypeId, len: u64) -> TypeId {
        self.ty(Type::Array { elem, len })
    }

    /// A literal (unnamed, unpacked) struct.
    pub fn struct_ty(&mut self, fields: &[TypeId]) -> TypeId {
        self.ty(Type::Struct {
            fields: fields.into(),
            packed: false,
            name: None,
        })
    }

    /// Intern the non-variadic function type returning `ret` and taking `params`.
    pub fn fn_ty(&mut self, ret: TypeId, params: &[TypeId]) -> TypeId {
        self.ty(Type::Function {
            ret,
            params: params.into(),
            vararg: false,
        })
    }

    /// Intern a constant.
    pub fn constant(&mut self, ty: TypeId, value: Const) -> ConstId {
        let r = self
            .parts
            .consts
            .intern(Constant { ty, value }, &mut self.meter);
        self.record(r, ConstId::new(FAILED))
    }

    /// An integer constant operand.
    pub fn const_int(&mut self, ty: TypeId, v: u64) -> Value {
        Value::Const(self.constant(ty, Const::Int(v)))
    }

    /// A global's address as an operand.
    pub fn global_addr(&mut self, g: GlobalId) -> Value {
        let space = self.parts.globals.get(g).map(|g| g.space);
        let Some(space) = space else {
            return Value::Const(ConstId::new(FAILED));
        };
        let ptr = self.ptr(space);
        Value::Const(self.constant(ptr, Const::GlobalAddr(g)))
    }

    /// Add a global.
    pub fn global(&mut self, g: Global) -> GlobalId {
        let r = self.parts.globals.push(g, &mut self.meter);
        self.record(r, GlobalId::new(FAILED))
    }

    /// Add an external global named `name` of `value_ty` in `space`, with an optional
    /// initializer (internal linkage when initialized).
    pub fn simple_global(
        &mut self,
        name: &str,
        value_ty: TypeId,
        space: AddrSpace,
        init: Option<ConstId>,
    ) -> GlobalId {
        let name = self.sym(name);
        self.global(Global {
            name,
            value_ty,
            space,
            constant: false,
            init,
            externally_initialized: false,
            linkage: if init.is_some() {
                Linkage::Internal
            } else {
                Linkage::External
            },
            visibility: Default::default(),
            unnamed_addr: Default::default(),
            align: None,
            section: None,
        })
    }

    fn push_function(&mut self, name: &str, ty: TypeId) -> FuncId {
        let name = self.sym(name);
        let nparams = match self.parts.types.get(ty) {
            Some(Type::Function { params, .. }) => params.len(),
            _ => 0,
        };
        let f = Function {
            name,
            ty,
            linkage: Linkage::External,
            visibility: Default::default(),
            unnamed_addr: Default::default(),
            cconv: inst::CallConv::C,
            attrs: attrs::FnAttrs::default(),
            ret_attrs: attrs::ParamAttrs::default(),
            params: vec![Param::default(); nparams].into(),
            body: None,
            section: None,
            align: None,
        };
        let r = self.parts.funcs.push(f, &mut self.meter);
        self.record(r, FuncId::new(FAILED))
    }

    /// Declare a function.
    pub fn declare(&mut self, name: &str, ty: TypeId) -> FuncId {
        self.push_function(name, ty)
    }

    /// Start defining a function; its entry block is current.
    pub fn define(&mut self, name: &str, ty: TypeId) -> FunctionBuilder<'_> {
        let func = self.push_function(name, ty);
        FunctionBuilder {
            m: self,
            func,
            blocks: vec![Vec::new()],
            current: 0,
            tmp: Vec::new(),
        }
    }

    /// Mutable access to a function, for attributes and linkage.
    pub fn func_mut(&mut self, f: FuncId) -> Option<&mut Function> {
        self.parts.funcs.get_mut(f)
    }

    /// Add a metadata entry.
    pub fn md(&mut self, md: Md) -> MdId {
        let r = self.parts.metadata.push(md, &mut self.meter);
        self.record(r, MdId::new(FAILED))
    }

    /// A metadata string.
    pub fn md_string(&mut self, s: &str) -> MdId {
        let s = self.sym(s);
        self.md(Md::String(s))
    }

    /// A (uniqued) metadata tuple.
    pub fn md_node(&mut self, ops: &[Option<MdId>]) -> MdId {
        self.md(Md::Node {
            distinct: false,
            ops: ops.into(),
        })
    }

    /// Add a named metadata list.
    pub fn named_md(&mut self, name: &str, ops: &[MdId]) {
        let name = self.sym(name);
        self.parts.named_md.push(NamedMd {
            name,
            ops: ops.into(),
        });
    }

    /// The parts built so far, unverified (for verifier tests that corrupt them).
    pub fn finish_parts(self) -> Result<ModuleParts, Error> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(self.parts),
        }
    }

    /// Verify and return the module.
    pub fn finish(mut self) -> Result<Module, Error> {
        if let Some(e) = self.error {
            return Err(e);
        }
        Module::new(self.parts, &mut self.meter)
    }
}

/// Builds one function body. Instruction values handed out are provisional until
/// [`FunctionBuilder::finish`] renumbers them in block layout order.
#[derive(Debug)]
pub struct FunctionBuilder<'m> {
    m: &'m mut ModuleBuilder,
    func: FuncId,
    /// Per block, the provisional ids of its instructions in order.
    blocks: Vec<Vec<u32>>,
    current: usize,
    /// Provisional instructions, by provisional id.
    tmp: Vec<Inst>,
}

impl FunctionBuilder<'_> {
    /// The module under construction (for types and constants).
    pub fn module(&mut self) -> &mut ModuleBuilder {
        self.m
    }

    /// The function being defined.
    pub fn id(&self) -> FuncId {
        self.func
    }

    /// Parameter `i`.
    pub fn arg(&self, i: u32) -> Value {
        Value::Arg(super::ArgIdx::new(i))
    }

    /// A new, empty block (not current).
    pub fn block(&mut self) -> BlockId {
        self.blocks.push(Vec::new());
        BlockId::from_index(self.blocks.len() - 1).unwrap_or(BlockId::new(FAILED))
    }

    /// Append subsequent instructions to `b`.
    pub fn switch_to(&mut self, b: BlockId) {
        self.current = b.index();
    }

    /// Append an instruction of result type `ty` to the current block.
    pub fn push(&mut self, ty: TypeId, kind: InstKind) -> Value {
        let id = u32::try_from(self.tmp.len()).unwrap_or(FAILED);
        self.tmp.push(Inst { ty, kind });
        if let Some(block) = self.blocks.get_mut(self.current) {
            block.push(id);
        }
        Value::Inst(InstId::new(id))
    }

    /// Append a void instruction (store, fence, terminator, void call).
    pub fn push_void(&mut self, kind: InstKind) {
        let void = self.m.void();
        self.push(void, kind);
    }

    /// The type of an operand built so far.
    pub fn value_ty(&self, v: Value) -> Option<TypeId> {
        match v {
            Value::Arg(a) => {
                let ty = self.m.parts.funcs.get(self.func)?.ty;
                match self.m.parts.types.get(ty)? {
                    Type::Function { params, .. } => params.get(a.index()).copied(),
                    _ => None,
                }
            }
            Value::Inst(i) => self.tmp.get(i.index()).map(|i| i.ty),
            Value::Const(c) => self.m.parts.consts.ty(c),
        }
    }

    /// `a op b`, typed as `a`.
    pub fn bin(&mut self, op: inst::BinOp, a: Value, b: Value) -> Value {
        let ty = self.value_ty(a).unwrap_or(TypeId::new(FAILED));
        self.push(
            ty,
            InstKind::BinOp {
                op,
                flags: inst::IntFlags::NONE,
                fmf: inst::FastMath::NONE,
                a,
                b,
            },
        )
    }

    /// `ret` / `ret v`.
    pub fn ret(&mut self, v: Option<Value>) {
        self.push_void(InstKind::Ret(v));
    }

    /// `br dest`.
    pub fn br(&mut self, dest: BlockId) {
        self.push_void(InstKind::Br { dest, hints: None });
    }

    /// `br c, t, f`.
    pub fn cond_br(&mut self, c: Value, t: BlockId, f: BlockId) {
        self.push_void(InstKind::CondBr {
            c,
            t,
            f,
            hints: None,
        });
    }

    /// Lay the blocks out, renumber instruction operands, and attach the body.
    pub fn finish(self) -> FuncId {
        let FunctionBuilder {
            m,
            func,
            blocks,
            current: _,
            tmp,
        } = self;
        let mut final_id = vec![FAILED; tmp.len()];
        let mut next = 0u32;
        let mut run_len = Vec::with_capacity(blocks.len());
        for run in &blocks {
            let mut len = 0usize;
            for &t in run {
                if let Some(slot) = final_id.get_mut(t as usize) {
                    *slot = next;
                    next = next.saturating_add(1);
                    len += 1;
                }
            }
            run_len.push(len);
        }
        // Each temporary id sits in at most one block run, so its final id is unique; ids in no
        // block keep `FAILED` and are dropped. Sorting by final id yields the layout order.
        let mut laid: Vec<(u32, Inst)> = final_id
            .iter()
            .copied()
            .zip(tmp)
            .filter(|&(id, _)| id != FAILED)
            .collect();
        laid.sort_unstable_by_key(|&(id, _)| id);
        let mut laid = laid.into_iter();
        let mut body = Body::default();
        for len in run_len {
            let start = InstId::new(u32::try_from(body.insts.len()).unwrap_or(FAILED));
            for (_, mut inst) in laid.by_ref().take(len) {
                inst.kind.for_each_value_mut(|v| {
                    if let Value::Inst(i) = v {
                        let id = final_id.get(i.index()).copied().unwrap_or(FAILED);
                        *v = Value::Inst(InstId::new(id));
                    }
                });
                let r = body.push_inst(inst, &mut m.meter);
                m.record(r.map(|_| ()), ());
            }
            let end = InstId::new(u32::try_from(body.insts.len()).unwrap_or(FAILED));
            let r = body.blocks.push(
                Block {
                    start,
                    end,
                    name: None,
                },
                &mut m.meter,
            );
            m.record(r.map(|_| ()), ());
        }
        if let Some(f) = m.parts.funcs.get_mut(func) {
            f.body = Some(body);
            f.linkage = Linkage::External;
        }
        func
    }
}
