//! Typed-pointer types as unification terms.
//!
//! The IR keeps pointer types with an address space only, so the writer rebuilds a typed type for
//! every value: each erased type becomes a term whose structure is the erased type's and whose
//! pointer pointees are variables. Uses equate terms (a load equates its pointer's pointee with its
//! result, a call equates arguments with the callee's parameters, and so on), and first-order
//! unification solves the equations. An identified struct is one nominal term whose fields form a
//! per-struct template, as in LLVM, so a struct that points at itself stays finite. A pointee no
//! use constrains is written as `i8`.
//!
//! Every equation holds in the typed module the IR was decoded from, so for typed-pointer input
//! a solution exists and unification finds the most general one. Input that has none (an
//! opaque-pointer module that uses one pointer at two pointee types) is a typed conflict.

use std::collections::HashMap;

use metal2vulkan_air::ir::{Idx, Symbols, Type, TypeId, TypeTable};

use super::{Unspellable, WriteError};

/// A term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct N(u32);

#[derive(Clone, Debug)]
enum Node {
    /// An unconstrained pointee.
    Var,
    /// A pointer-free type other than an identified struct, spelled as the erased type.
    Ground(TypeId),
    /// An identified (nominal) struct, opaque or not.
    Named(TypeId),
    /// An integer the writer chose for a pointee no use constrains, or one a contract fixes.
    Int(u8),
    /// `void`, as the result of a function type a contract fixes, when the module interns no
    /// `void` (an interned type is always its [`Node::Ground`] term).
    Void,
    Ptr {
        space: u32,
        pointee: N,
    },
    Vector {
        len: u32,
        elem: N,
    },
    Array {
        len: u64,
        elem: N,
    },
    Struct {
        packed: bool,
        fields: Box<[N]>,
    },
    Func {
        ret: N,
        params: Box<[N]>,
        vararg: bool,
    },
}

/// The term store: nodes, a union-find forest over them, and the per-type caches.
pub(super) struct Terms<'m> {
    types: &'m TypeTable,
    symbols: &'m Symbols,
    nodes: Vec<Node>,
    parent: Vec<u32>,
    /// Whether an erased type contains a pointer outside identified structs.
    has_ptr: Vec<bool>,
    ground: HashMap<TypeId, N>,
    templates: HashMap<TypeId, Box<[N]>>,
    text: HashMap<N, String>,
}

impl<'m> Terms<'m> {
    pub(super) fn new(types: &'m TypeTable, symbols: &'m Symbols) -> Terms<'m> {
        let mut has_ptr = Vec::new();
        for (_, ty) in types.iter() {
            let child = |t: &TypeId| has_ptr.get(t.index()).copied().unwrap_or(false);
            let p = match ty {
                Type::Ptr { .. } => true,
                Type::Struct { name: Some(_), .. } => false,
                Type::Vector { elem, .. } | Type::Array { elem, .. } => child(elem),
                Type::Struct { fields, .. } => fields.iter().any(child),
                Type::Function { ret, params, .. } => child(ret) || params.iter().any(child),
                _ => false,
            };
            has_ptr.push(p);
        }
        Terms {
            types,
            symbols,
            nodes: Vec::new(),
            parent: Vec::new(),
            has_ptr,
            ground: HashMap::new(),
            templates: HashMap::new(),
            text: HashMap::new(),
        }
    }

    fn push(&mut self, node: Node) -> N {
        let id = u32::try_from(self.nodes.len()).expect("term count fits u32");
        self.nodes.push(node);
        self.parent.push(id);
        N(id)
    }

    fn ty(&self, t: TypeId) -> Result<&'m Type, WriteError> {
        self.types
            .get(t)
            .ok_or_else(|| WriteError::Malformed(format!("type {t:?} out of range")))
    }

    /// Whether `t` contains a pointer (identified structs count as leaves).
    pub(super) fn has_ptr(&self, t: TypeId) -> bool {
        self.has_ptr.get(t.index()).copied().unwrap_or(false)
    }

    /// A fresh term for erased type `t`: its structure, with a new variable per pointee.
    pub(super) fn fresh(&mut self, t: TypeId) -> Result<N, WriteError> {
        let ty = self.ty(t)?;
        match ty {
            Type::Struct { name: Some(_), .. } | Type::OpaqueStruct { name: Some(_) } => {
                return Ok(self.cached(t, Node::Named(t)))
            }
            Type::OpaqueStruct { name: None } => {
                return Err(WriteError::Unsupported(Unspellable::UnnamedOpaqueStruct))
            }
            _ => {}
        }
        if !self.has_ptr(t) {
            return Ok(self.cached(t, Node::Ground(t)));
        }
        Ok(match ty {
            Type::Ptr { space } => {
                let pointee = self.push(Node::Var);
                self.push(Node::Ptr {
                    space: space.raw(),
                    pointee,
                })
            }
            Type::Vector { elem, len } => {
                let elem = self.fresh(*elem)?;
                self.push(Node::Vector { len: *len, elem })
            }
            Type::Array { elem, len } => {
                let elem = self.fresh(*elem)?;
                self.push(Node::Array { len: *len, elem })
            }
            Type::Struct { fields, packed, .. } => {
                let fields = fields
                    .iter()
                    .map(|&f| self.fresh(f))
                    .collect::<Result<Box<[N]>, _>>()?;
                self.push(Node::Struct {
                    packed: *packed,
                    fields,
                })
            }
            Type::Function {
                ret,
                params,
                vararg,
            } => {
                let ret = self.fresh(*ret)?;
                let params = params
                    .iter()
                    .map(|&p| self.fresh(p))
                    .collect::<Result<Box<[N]>, _>>()?;
                self.push(Node::Func {
                    ret,
                    params,
                    vararg: *vararg,
                })
            }
            _ => return Err(WriteError::Malformed(format!("type {t:?} has no pointer"))),
        })
    }

    /// A pointer into `space` at `pointee`.
    pub(super) fn ptr_to(&mut self, space: u32, pointee: N) -> N {
        self.push(Node::Ptr { space, pointee })
    }

    /// The integer type `i<width>`.
    pub(super) fn int(&mut self, width: u8) -> N {
        match self.interned(|ty| *ty == Type::Int(width)) {
            Some(t) => self.cached(t, Node::Ground(t)),
            None => self.push(Node::Int(width)),
        }
    }

    /// The function type `void ()`. A pointer-free type the module interns is one ground term,
    /// so the interned `void ()` is used when there is one.
    pub(super) fn void_fn(&mut self) -> N {
        let is_void_fn = |ty: &Type, types: &TypeTable| match ty {
            Type::Function {
                ret,
                params,
                vararg: false,
            } => params.is_empty() && matches!(types.get(*ret), Some(Type::Void)),
            _ => false,
        };
        let types = self.types;
        if let Some(t) = self.interned(|ty| is_void_fn(ty, types)) {
            return self.cached(t, Node::Ground(t));
        }
        let ret = match self.interned(|ty| *ty == Type::Void) {
            Some(t) => self.cached(t, Node::Ground(t)),
            None => self.push(Node::Void),
        };
        self.push(Node::Func {
            ret,
            params: Box::new([]),
            vararg: false,
        })
    }

    /// The first interned type `pred` holds for.
    fn interned(&self, pred: impl Fn(&Type) -> bool) -> Option<TypeId> {
        self.types.iter().find(|(_, ty)| pred(ty)).map(|(t, _)| t)
    }

    fn cached(&mut self, t: TypeId, node: Node) -> N {
        if let Some(&n) = self.ground.get(&t) {
            return n;
        }
        let n = self.push(node);
        self.ground.insert(t, n);
        n
    }

    /// The field terms of identified struct `t` (created on first use).
    fn template(&mut self, t: TypeId) -> Result<Box<[N]>, WriteError> {
        if let Some(fields) = self.templates.get(&t) {
            return Ok(fields.clone());
        }
        let fields = match self.ty(t)? {
            Type::Struct { fields, .. } => fields
                .iter()
                .map(|&f| self.fresh(f))
                .collect::<Result<Box<[N]>, _>>()?,
            _ => Box::new([]),
        };
        self.templates.insert(t, fields.clone());
        Ok(fields)
    }

    /// The representative of `n`.
    pub(super) fn find(&mut self, n: N) -> N {
        let mut root = n.0;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        let mut at = n.0;
        while self.parent[at as usize] != root {
            let next = self.parent[at as usize];
            self.parent[at as usize] = root;
            at = next;
        }
        N(root)
    }

    fn node(&mut self, n: N) -> Node {
        let r = self.find(n);
        self.nodes[r.0 as usize].clone()
    }

    /// Whether `n` is still an unconstrained variable.
    pub(super) fn is_var(&mut self, n: N) -> bool {
        matches!(self.node(n), Node::Var)
    }

    /// Bind the unconstrained variable `n` to the integer type `iW`.
    pub(super) fn bind_int(&mut self, n: N, width: u8) {
        let r = self.find(n);
        if matches!(self.nodes[r.0 as usize], Node::Var) {
            self.nodes[r.0 as usize] = Node::Int(width);
            // Spellings computed so far may have spelled this variable `i8`.
            self.text.clear();
        }
    }

    /// The pointee of pointer term `n`.
    pub(super) fn pointee(&mut self, n: N) -> Result<N, WriteError> {
        match self.node(n) {
            Node::Ptr { pointee, .. } => Ok(pointee),
            other => Err(WriteError::Malformed(format!(
                "expected a pointer term, found {other:?}"
            ))),
        }
    }

    /// The return and parameter terms of function term `n`.
    pub(super) fn signature(&mut self, n: N) -> Result<(N, Box<[N]>, bool), WriteError> {
        match self.node(n) {
            Node::Func {
                ret,
                params,
                vararg,
            } => Ok((ret, params, vararg)),
            Node::Ground(t) => match self.ty(t)? {
                Type::Function {
                    ret,
                    params,
                    vararg,
                } => {
                    let ret = self.fresh(*ret)?;
                    let params = params
                        .iter()
                        .map(|&p| self.fresh(p))
                        .collect::<Result<Box<[N]>, _>>()?;
                    Ok((ret, params, *vararg))
                }
                _ => Err(WriteError::Malformed("expected a function type".into())),
            },
            other => Err(WriteError::Malformed(format!(
                "expected a function term, found {other:?}"
            ))),
        }
    }

    /// Element or field `i` of aggregate or vector term `n`.
    pub(super) fn child(&mut self, n: N, i: u64) -> Result<N, WriteError> {
        let bad = || WriteError::Malformed(format!("no member {i} in an aggregate term"));
        match self.node(n) {
            Node::Vector { elem, .. } | Node::Array { elem, .. } => Ok(elem),
            Node::Struct { fields, .. } => fields
                .get(usize::try_from(i).map_err(|_| bad())?)
                .copied()
                .ok_or_else(bad),
            Node::Named(t) => self
                .template(t)?
                .get(usize::try_from(i).map_err(|_| bad())?)
                .copied()
                .ok_or_else(bad),
            Node::Ground(t) => {
                let child = match self.ty(t)? {
                    Type::Vector { elem, .. } | Type::Array { elem, .. } => *elem,
                    Type::Struct { fields, .. } => *fields
                        .get(usize::try_from(i).map_err(|_| bad())?)
                        .ok_or_else(bad)?,
                    _ => return Err(bad()),
                };
                self.fresh(child)
            }
            _ => Err(bad()),
        }
    }

    /// Equate `a` and `b`.
    pub(super) fn unify(&mut self, a: N, b: N) -> Result<(), String> {
        let mut work = vec![(a, b)];
        while let Some((a, b)) = work.pop() {
            let (ra, rb) = (self.find(a), self.find(b));
            if ra == rb {
                continue;
            }
            let (na, nb) = (
                self.nodes[ra.0 as usize].clone(),
                self.nodes[rb.0 as usize].clone(),
            );
            match (&na, &nb) {
                (Node::Var, _) => {
                    self.parent[ra.0 as usize] = rb.0;
                    continue;
                }
                (_, Node::Var) => {
                    self.parent[rb.0 as usize] = ra.0;
                    continue;
                }
                (Node::Ground(x), Node::Ground(y)) | (Node::Named(x), Node::Named(y)) if x == y => {
                }
                (Node::Int(x), Node::Int(y)) if x == y => {}
                (Node::Void, Node::Void) => {}
                (
                    Node::Ptr {
                        space: sa,
                        pointee: pa,
                    },
                    Node::Ptr {
                        space: sb,
                        pointee: pb,
                    },
                ) if sa == sb => work.push((*pa, *pb)),
                (Node::Vector { len: la, elem: ea }, Node::Vector { len: lb, elem: eb })
                    if la == lb =>
                {
                    work.push((*ea, *eb))
                }
                (Node::Array { len: la, elem: ea }, Node::Array { len: lb, elem: eb })
                    if la == lb =>
                {
                    work.push((*ea, *eb))
                }
                (
                    Node::Struct {
                        packed: pa,
                        fields: fa,
                    },
                    Node::Struct {
                        packed: pb,
                        fields: fb,
                    },
                ) if pa == pb && fa.len() == fb.len() => {
                    work.extend(fa.iter().copied().zip(fb.iter().copied()))
                }
                (
                    Node::Func {
                        ret: ra2,
                        params: pa,
                        vararg: va,
                    },
                    Node::Func {
                        ret: rb2,
                        params: pb,
                        vararg: vb,
                    },
                ) if va == vb && pa.len() == pb.len() => {
                    work.push((*ra2, *rb2));
                    work.extend(pa.iter().copied().zip(pb.iter().copied()));
                }
                _ => {
                    let (ta, tb) = (self.text(a), self.text(b));
                    // The spellings above are of a partial solution: do not keep them.
                    self.text.clear();
                    return Err(format!(
                        "{} vs {}",
                        ta.unwrap_or_else(|_| format!("{na:?}")),
                        tb.unwrap_or_else(|_| format!("{nb:?}"))
                    ));
                }
            }
            self.parent[ra.0 as usize] = rb.0;
        }
        Ok(())
    }

    /// The LLVM typed-pointer spelling of `n`, unconstrained pointees as `i8`.
    pub(super) fn text(&mut self, n: N) -> Result<String, WriteError> {
        let mut in_progress = Vec::new();
        self.text_rec(n, &mut in_progress)
    }

    fn text_rec(&mut self, n: N, in_progress: &mut Vec<N>) -> Result<String, WriteError> {
        let r = self.find(n);
        if let Some(t) = self.text.get(&r) {
            return Ok(t.clone());
        }
        if in_progress.contains(&r) || in_progress.len() > 4096 {
            return Err(WriteError::Unsupported(Unspellable::InfinitePointee));
        }
        in_progress.push(r);
        let node = self.nodes[r.0 as usize].clone();
        let t = match node {
            Node::Var => "i8".to_owned(),
            Node::Int(w) => format!("i{w}"),
            Node::Void => "void".to_owned(),
            Node::Ground(t) => self.erased_text(t)?,
            Node::Named(t) => self.struct_name(t)?,
            Node::Ptr { space, pointee } => {
                let p = self.text_rec(pointee, in_progress)?;
                if space == 0 {
                    format!("{p}*")
                } else {
                    format!("{p} addrspace({space})*")
                }
            }
            Node::Vector { len, elem } => {
                format!("<{len} x {}>", self.text_rec(elem, in_progress)?)
            }
            Node::Array { len, elem } => {
                format!("[{len} x {}]", self.text_rec(elem, in_progress)?)
            }
            Node::Struct { packed, fields } => {
                let mut f = Vec::with_capacity(fields.len());
                for &x in fields.iter() {
                    f.push(self.text_rec(x, in_progress)?);
                }
                struct_body(packed, &f)
            }
            Node::Func {
                ret,
                params,
                vararg,
            } => {
                let ret = self.text_rec(ret, in_progress)?;
                let mut p = Vec::with_capacity(params.len() + 1);
                for &x in params.iter() {
                    p.push(self.text_rec(x, in_progress)?);
                }
                if vararg {
                    p.push("...".to_owned());
                }
                format!("{ret} ({})", p.join(", "))
            }
        };
        in_progress.pop();
        self.text.insert(r, t.clone());
        Ok(t)
    }

    /// The spelling of a pointer-free erased type.
    pub(super) fn erased_text(&mut self, t: TypeId) -> Result<String, WriteError> {
        Ok(match self.ty(t)? {
            Type::Void => "void".into(),
            Type::Label => "label".into(),
            Type::Metadata => "metadata".into(),
            Type::Int(w) => format!("i{w}"),
            Type::Half => "half".into(),
            Type::BFloat => "bfloat".into(),
            Type::Float => "float".into(),
            Type::Double => "double".into(),
            Type::Vector { elem, len } => format!("<{len} x {}>", self.erased_text(*elem)?),
            Type::Array { elem, len } => format!("[{len} x {}]", self.erased_text(*elem)?),
            Type::Struct { name: Some(_), .. } | Type::OpaqueStruct { .. } => {
                self.struct_name(t)?
            }
            Type::Struct { fields, packed, .. } => {
                let mut f = Vec::with_capacity(fields.len());
                for &x in fields.iter() {
                    f.push(self.erased_text(x)?);
                }
                struct_body(*packed, &f)
            }
            Type::Function {
                ret,
                params,
                vararg,
            } => {
                let ret = self.erased_text(*ret)?;
                let mut p = Vec::with_capacity(params.len() + 1);
                for &x in params.iter() {
                    p.push(self.erased_text(x)?);
                }
                if *vararg {
                    p.push("...".to_owned());
                }
                format!("{ret} ({})", p.join(", "))
            }
            Type::Ptr { .. } => {
                return Err(WriteError::Malformed(
                    "a pointer type spelled without its pointee".into(),
                ))
            }
        })
    }

    /// `%"name"` of identified struct `t`.
    pub(super) fn struct_name(&self, t: TypeId) -> Result<String, WriteError> {
        match self.ty(t)? {
            Type::Struct { name: Some(n), .. } | Type::OpaqueStruct { name: Some(n) } => {
                let bytes = self
                    .symbols
                    .get(*n)
                    .ok_or_else(|| WriteError::Malformed(format!("symbol {n:?} out of range")))?;
                Ok(format!("%{}", super::text::quoted(bytes)))
            }
            _ => Err(WriteError::Unsupported(Unspellable::UnnamedOpaqueStruct)),
        }
    }

    /// The definition body of identified struct `t` (`{ ... }`, `<{ ... }>`, or `opaque`).
    pub(super) fn struct_definition(&mut self, t: TypeId) -> Result<String, WriteError> {
        match self.ty(t)? {
            Type::Struct { packed, .. } => {
                let packed = *packed;
                let fields = self.template(t)?;
                let mut f = Vec::with_capacity(fields.len());
                for &x in fields.iter() {
                    f.push(self.text(x)?);
                }
                Ok(struct_body(packed, &f))
            }
            _ => Ok("opaque".to_owned()),
        }
    }
}

fn struct_body(packed: bool, fields: &[String]) -> String {
    let body = if fields.is_empty() {
        String::new()
    } else {
        format!(" {} ", fields.join(", "))
    };
    if packed {
        format!("<{{{body}}}>")
    } else {
        format!("{{{body}}}")
    }
}
