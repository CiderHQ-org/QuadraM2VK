//! The guard of the device checks' buffer accesses: on a device without robust buffer access an
//! access past a buffer is undefined (llvmpipe faults), so every check module must stay inside its
//! buffers for every invocation its draw or dispatch can launch, whatever the device enables.
//!
//! The guard reads each module a check builder finishes (`bind::device_checks::kernel::finished`),
//! on every committed profile, and bounds each buffer index over the module's SSA values (the
//! interpreter, `spirv::interp`, runs no fragment entry and stops at the first operation outside
//! its exact model, such as the denormal check's `InverseSqrt`, so it cannot evaluate every
//! module; an index needs only its own arithmetic):
//!
//! - **A fragment stage** (the color writes', the raster checks', the color outputs' and the depth
//!   writes' fragments): its helper invocations shade the uncovered pixels of a quad (llvmpipe
//!   shades 4x4 blocks) past the target's last row and column, where an index computed from the
//!   position lies past a buffer sized to the target. Each read of a storage or uniform buffer
//!   takes a constant index or one below the buffer's own length (`OpArrayLength`, through
//!   `OpSelect` on `OpULessThan`, `UMin`, `UMod` or the length less one), and an image fetch or
//!   read a constant coordinate. A helper invocation's stores and atomics have no effect, so only
//!   reads are held. - **A compute check kernel** ([`CheckKernel`]): every access of each
//!   invocation of its dispatch
//!   of whole workgroups (its `workgroups` along x of its local size) lies inside the buffer the
//!   kernel describes, or below that buffer's own length.
//!
//! Not covered: vertex stages (no helper invocations; a draw's vertices are the runtime's), and
//! the compute modules whose dispatch and buffers the runtime sizes (the storage writes, texel and
//! sampled fetches, stored depths, cube gradients kernel and descriptor checks, and the sweeps,
//! whose inputs a push constant bounds). A new check's
//! builder joins [`fragment_checks`] or [`kernel_checks`].

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::bind::device_checks::kernel::finished;
use crate::bind::{
    barycentric_pair_check_fragment, canonical_nans_check, center_varyings_check_fragment,
    color_outputs_check_fragment, color_write_modules, composite_inserts_check,
    coverage_count_check_fragment, cube_gradients_check_fragment, depth_writes_check_fragment,
    early_tests_check_fragment, float_denormals_check, fragcoord_depth_check_fragment,
    half_quantize_check, half_round_trip_check, loads_forwarded_check, localized_loads_check,
    narrow_mask_check, CheckKernel, MathMode,
};
use crate::device::{Device, DevicePolicy};
use crate::error::Error;
use crate::spirv::ids::{BlockId, FuncId, GlobalId, Value};
use crate::spirv::ir::{
    BinOp, ConstKind, Constant, ConvOp, ExecMode, ExecModel, GlobalKind, GlslOp, Inst,
    StorageClass, Type, TypedModule, ValueOrigin,
};
use crate::spirv::testing::committed_descriptions;

/// The width of the raster checks' targets: not a multiple of 4, so a quad or a 4x4 block reaches
/// past the last column too.
const RASTER_WIDTH: u32 = 7;

/// What is known of an integer value over every invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    /// At most this value.
    AtMost(u64),
    /// The length (`OpArrayLength`) of this buffer's runtime array.
    Length(GlobalId),
    /// Below that length: an element of the buffer whatever its size (which holds one at least).
    Inside(GlobalId),
    /// Nothing.
    Unknown,
}

/// The bound of a value that is one of two values.
fn join(a: Bound, b: Bound) -> Bound {
    use Bound::*;
    match (a, b) {
        (AtMost(x), AtMost(y)) => AtMost(x.max(y)),
        (Inside(g), Inside(h)) if g == h => Inside(g),
        (Inside(g), AtMost(0)) | (AtMost(0), Inside(g)) => Inside(g),
        (Length(g), Length(h)) if g == h => Length(g),
        _ => Unknown,
    }
}

/// Who runs a module's entry.
#[derive(Clone, Copy, Debug)]
enum Launch {
    /// A draw's fragments, helper invocations past the target included: no bound of the position.
    Fragment,
    /// A dispatch of `groups` workgroups of `local` invocations each.
    Dispatch { groups: [u32; 3], local: [u32; 3] },
}

/// One module under the guard.
struct Guard<'m> {
    m: &'m TypedModule,
    launch: Launch,
    /// Each buffer's bytes by binding, for a dispatch.
    bytes: Vec<u64>,
    insts: HashMap<Value, &'m Inst>,
    phis: HashMap<Value, &'m [(Value, BlockId)]>,
    memo: HashMap<Value, Bound>,
    open: HashSet<Value>,
}

impl<'m> Guard<'m> {
    fn new(m: &'m TypedModule, launch: Launch, bytes: Vec<u64>) -> Guard<'m> {
        let mut insts = HashMap::new();
        let mut phis = HashMap::new();
        for block in m.blocks() {
            for node in &block.insts {
                if let Some(result) = node.result {
                    insts.insert(result, &node.inst);
                }
            }
            for phi in &block.phis {
                phis.insert(phi.result, phi.incoming.as_slice());
            }
        }
        Guard {
            m,
            launch,
            bytes,
            insts,
            phis,
            memo: HashMap::new(),
            open: HashSet::new(),
        }
    }

    /// The bound of `v` (a value that depends on itself through a loop is [`Bound::Unknown`]).
    fn bound(&mut self, v: Value) -> Bound {
        if let Some(b) = self.memo.get(&v) {
            return *b;
        }
        if !self.open.insert(v) {
            return Bound::Unknown;
        }
        let b = self.bound_of(v);
        self.open.remove(&v);
        self.memo.insert(v, b);
        b
    }

    fn bound_of(&mut self, v: Value) -> Bound {
        let Ok(def) = self.m.value(v) else {
            return Bound::Unknown;
        };
        match def.origin {
            ValueOrigin::Const(c) => match self.m.consts().get(c).map(|d| &d.kind) {
                Ok(ConstKind::Plain(Constant::Scalar { bits, .. })) => Bound::AtMost(bits.0),
                Ok(ConstKind::Plain(Constant::Null { .. })) => Bound::AtMost(0),
                _ => Bound::Unknown,
            },
            ValueOrigin::Phi(_) => {
                let incoming: Vec<Value> = match self.phis.get(&v) {
                    Some(incoming) => incoming.iter().map(|(value, _)| *value).collect(),
                    None => return Bound::Unknown,
                };
                let mut bounds = incoming.into_iter().map(|value| self.bound(value));
                let first = bounds.next().unwrap_or(Bound::Unknown);
                bounds.fold(first, join)
            }
            ValueOrigin::Inst(_) => match self.insts.get(&v).copied() {
                Some(inst) => {
                    let b = self.inst_bound(inst);
                    self.fit(v, b)
                }
                None => Bound::Unknown,
            },
            _ => Bound::Unknown,
        }
    }

    /// `b`, or [`Bound::Unknown`] where it passes the largest value of `v`'s integer type (the
    /// arithmetic wrapped).
    fn fit(&self, v: Value, b: Bound) -> Bound {
        let Bound::AtMost(x) = b else {
            return b;
        };
        let width = self
            .m
            .value(v)
            .ok()
            .and_then(|d| self.m.types().get(d.ty).ok())
            .and_then(|t| match t {
                Type::Int { width, .. } => Some(u32::from(*width)),
                _ => None,
            });
        match width {
            Some(w) if w < 64 && x >> w != 0 => Bound::Unknown,
            Some(_) => b,
            None => Bound::Unknown,
        }
    }

    /// `v`'s value where it is a plain constant.
    fn constant(&mut self, v: Value) -> Option<u64> {
        let def = self.m.value(v).ok()?;
        match (def.origin, self.bound(v)) {
            (ValueOrigin::Const(_), Bound::AtMost(x)) => Some(x),
            _ => None,
        }
    }

    fn inst_bound(&mut self, inst: &'m Inst) -> Bound {
        use Bound::*;
        match inst {
            Inst::Binary { op, a, b, .. } => self.binary(*op, *a, *b),
            Inst::Select { cond, t, f } => {
                let t = self.taken(*cond, *t, true);
                let f = self.taken(*cond, *f, false);
                join(t, f)
            }
            Inst::Convert {
                op: ConvOp::Bitcast | ConvOp::UConvert,
                a,
                ..
            } => self.bound(*a),
            Inst::ArrayLength { ptr, .. } => match self.m.value(*ptr).map(|d| d.origin) {
                Ok(ValueOrigin::Global(g)) => Length(g),
                _ => Unknown,
            },
            Inst::Glsl {
                op: GlslOp::UMin,
                args,
                ..
            } if args.len() == 2 => match (self.bound(args[0]), self.bound(args[1])) {
                (AtMost(x), AtMost(y)) => AtMost(x.min(y)),
                (AtMost(x), _) | (_, AtMost(x)) => AtMost(x),
                (Inside(g), _) | (_, Inside(g)) => Inside(g),
                _ => Unknown,
            },
            Inst::Glsl {
                op: GlslOp::UClamp,
                args,
                ..
            } if args.len() == 3 => self.bound(args[2]),
            Inst::CompositeExtract { composite, indices } if indices.len() == 1 => {
                self.lane(*composite, indices[0])
            }
            Inst::Load { ptr, .. } => self.builtin_load(*ptr),
            _ => Unknown,
        }
    }

    fn binary(&mut self, op: BinOp, a: Value, b: Value) -> Bound {
        use Bound::*;
        let (x, y) = (self.bound(a), self.bound(b));
        match (op, x, y) {
            (BinOp::IAdd, AtMost(x), AtMost(y)) => x.checked_add(y).map_or(Unknown, AtMost),
            (BinOp::IMul, AtMost(x), AtMost(y)) => x.checked_mul(y).map_or(Unknown, AtMost),
            (BinOp::ShiftLeftLogical, AtMost(x), AtMost(s)) if s < 64 => {
                x.checked_mul(1 << s).map_or(Unknown, AtMost)
            }
            (BinOp::ShiftRightLogical | BinOp::UDiv, AtMost(x), _) => AtMost(x),
            (BinOp::UMod, _, Length(g)) => Inside(g),
            (BinOp::UMod, AtMost(x), AtMost(n)) if n >= 1 => AtMost(x.min(n - 1)),
            (BinOp::UMod, _, AtMost(n)) if n >= 1 => AtMost(n - 1),
            (BinOp::UMod, AtMost(x), _) => AtMost(x),
            (BinOp::BitwiseAnd, AtMost(x), AtMost(y)) => AtMost(x.min(y)),
            (BinOp::BitwiseAnd, AtMost(x), _) | (BinOp::BitwiseAnd, _, AtMost(x)) => AtMost(x),
            (BinOp::ISub, Length(g), _) if self.constant(b) == Some(1) => Inside(g),
            _ => Unknown,
        }
    }

    /// The bound of `arm`, the value an `OpSelect` on `cond` takes where `cond` is `taken`: below
    /// `n` where that is what `cond` states of it.
    fn taken(&mut self, cond: Value, arm: Value, taken: bool) -> Bound {
        use Bound::*;
        let plain = self.bound(arm);
        let Some(&Inst::Binary { op, a, b, .. }) = self.insts.get(&cond).copied() else {
            return plain;
        };
        let below = match (op, taken) {
            (BinOp::ULessThan, true) | (BinOp::UGreaterThanEqual, false) if a == arm => b,
            (BinOp::UGreaterThan, true) | (BinOp::ULessThanEqual, false) if b == arm => a,
            _ => return plain,
        };
        match (plain, self.bound(below)) {
            (_, Length(g)) => Inside(g),
            (AtMost(x), AtMost(n)) if n >= 1 => AtMost(x.min(n - 1)),
            (_, AtMost(n)) if n >= 1 => AtMost(n - 1),
            _ => plain,
        }
    }

    /// The bound of lane `k` of the vector `composite`.
    fn lane(&mut self, composite: Value, k: u32) -> Bound {
        let m = self.m;
        match self.insts.get(&composite).copied() {
            Some(Inst::Load { ptr, .. }) => match self.builtin_of(*ptr) {
                Some(builtin) => self.builtin_lane(builtin, k),
                None => Bound::Unknown,
            },
            Some(Inst::CompositeConstruct { parts }) => {
                let scalars = parts.iter().all(|p| {
                    m.value(*p)
                        .ok()
                        .and_then(|d| m.types().get(d.ty).ok())
                        .is_some_and(|t| matches!(t, Type::Int { .. }))
                });
                match parts.get(k as usize) {
                    Some(part) if scalars => self.bound(*part),
                    _ => Bound::Unknown,
                }
            }
            _ => Bound::Unknown,
        }
    }

    /// The bound of a load of a scalar builtin, or of a lane of a vector builtin through an access
    /// chain.
    fn builtin_load(&mut self, ptr: Value) -> Bound {
        if let Some(builtin) = self.builtin_of(ptr) {
            return self.builtin_lane(builtin, 0);
        }
        match self.insts.get(&ptr).copied() {
            Some(Inst::AccessChain { base, indices, .. }) if indices.len() == 1 => {
                let (base, index) = (*base, indices[0]);
                match (self.builtin_of(base), self.constant(index)) {
                    (Some(builtin), Some(k)) => match u32::try_from(k) {
                        Ok(k) => self.builtin_lane(builtin, k),
                        Err(_) => Bound::Unknown,
                    },
                    _ => Bound::Unknown,
                }
            }
            _ => Bound::Unknown,
        }
    }

    /// The input builtin whose variable `ptr` is.
    fn builtin_of(&self, ptr: Value) -> Option<crate::spv::BuiltIn> {
        let ValueOrigin::Global(g) = self.m.value(ptr).ok()?.origin else {
            return None;
        };
        let global = self.m.globals().get(g.index())?;
        match (global.class, &global.kind) {
            (StorageClass::Input, GlobalKind::BuiltIn { builtin, .. }) => Some(*builtin),
            _ => None,
        }
    }

    /// The bound of lane `k` of `builtin` over the launch's invocations.
    fn builtin_lane(&self, builtin: crate::spv::BuiltIn, k: u32) -> Bound {
        use crate::spv::BuiltIn as B;
        let Launch::Dispatch { groups, local } = self.launch else {
            return Bound::Unknown;
        };
        let lane = |v: [u32; 3]| v.get(k as usize).map(|n| u64::from(*n));
        let at_most = match builtin {
            B::GlobalInvocationId => lane(groups)
                .zip(lane(local))
                .map(|(g, l)| (g * l).saturating_sub(1)),
            B::LocalInvocationId => lane(local).map(|l| l.saturating_sub(1)),
            B::WorkgroupId => lane(groups).map(|g| g.saturating_sub(1)),
            B::NumWorkgroups => lane(groups),
            B::LocalInvocationIndex if k == 0 => Some(
                local
                    .iter()
                    .map(|n| u64::from(*n))
                    .product::<u64>()
                    .saturating_sub(1),
            ),
            _ => None,
        };
        at_most.map_or(Bound::Unknown, Bound::AtMost)
    }

    /// The functions the entry runs: itself and every function a call reaches.
    fn reached(&self) -> Vec<FuncId> {
        let mut reached = vec![self.m.entry().func];
        let mut at = 0;
        while let Some(&func) = reached.get(at) {
            at += 1;
            let Some(function) = self.m.functions().get(func.index()) else {
                continue;
            };
            for block in &function.blocks {
                let Some(block) = self.m.blocks().get(block.index()) else {
                    continue;
                };
                for node in &block.insts {
                    if let Inst::Call { func, .. } = &node.inst {
                        if !reached.contains(func) {
                            reached.push(*func);
                        }
                    }
                }
            }
        }
        reached
    }

    /// Each access of the reached functions that leaves its buffer (or that the guard cannot
    /// bound), in a few words.
    fn escapes(&mut self) -> Vec<String> {
        let mut found = Vec::new();
        let m = self.m;
        for func in self.reached() {
            let Some(function) = m.functions().get(func.index()) else {
                continue;
            };
            for block in &function.blocks {
                let Some(block) = m.blocks().get(block.index()) else {
                    continue;
                };
                for node in &block.insts {
                    found.extend(self.escape(&node.inst));
                }
            }
        }
        found
    }

    /// How `inst` leaves its buffer, where it accesses one and does.
    fn escape(&mut self, inst: &'m Inst) -> Option<String> {
        let fragment = matches!(self.launch, Launch::Fragment);
        match inst {
            Inst::Load { ptr, .. } => self.pointer(*ptr, "read"),
            Inst::Store { ptr, .. } if !fragment => self.pointer(*ptr, "write"),
            Inst::ImageFetch { coord, .. } | Inst::ImageRead { coord, .. } if fragment => {
                self.coordinate(*coord)
            }
            _ if !fragment => inst
                .atomic_pointer()
                .and_then(|ptr| self.pointer(ptr, "atomic")),
            _ => None,
        }
    }

    /// How an image read at `coord` may leave the image: a fragment's coordinate must be constant.
    fn coordinate(&mut self, coord: Value) -> Option<String> {
        if matches!(self.m.value(coord).ok()?.origin, ValueOrigin::Const(_)) {
            return None;
        }
        Some(
            "a fragment stage's image read at a coordinate that is not constant (the guard \
             bounds no coordinate against an image's size)"
                .into(),
        )
    }

    /// How the access `what` through `ptr` may leave its buffer.
    fn pointer(&mut self, ptr: Value, what: &str) -> Option<String> {
        let m = self.m;
        let mut indices = Vec::new();
        let mut at = ptr;
        let global = loop {
            let def = m.value(at).ok()?;
            match def.origin {
                ValueOrigin::Global(g) => break g,
                ValueOrigin::Inst(_) => match self.insts.get(&at).copied() {
                    Some(Inst::AccessChain {
                        base, indices: i, ..
                    }) => {
                        indices.splice(0..0, i.iter().copied());
                        at = *base;
                    }
                    _ => return self.untraced(def.ty, what),
                },
                ValueOrigin::Var(_) => return None,
                _ => return self.untraced(def.ty, what),
            }
        };
        let g = m.globals().get(global.index())?;
        let GlobalKind::Resource(slot) = g.kind else {
            return None;
        };
        if !matches!(g.class, StorageClass::StorageBuffer | StorageClass::Uniform) {
            return None;
        }
        let buffer = format!("buffer {}", slot.index);
        let mut ty = g.pointee;
        let mut offset = 0u64;
        for index in indices {
            let bound = self.bound(index);
            let t = m.types().get(ty).ok()?;
            match t {
                Type::Struct { members, layout } => {
                    let member = self.constant(index).and_then(|k| usize::try_from(k).ok());
                    let Some((member, k)) = member.and_then(|k| Some((*members.get(k)?, k))) else {
                        return Some(format!("a {what} of {buffer} at a member not constant"));
                    };
                    offset += layout
                        .as_ref()
                        .and_then(|l| l.members.get(k))
                        .map_or(0, |l| u64::from(l.offset));
                    ty = member;
                }
                Type::RuntimeArray {
                    elem,
                    stride: Some(stride),
                } => {
                    let (stride, elem) = (u64::from(*stride), *elem);
                    let elements = self
                        .bytes
                        .get(slot.index as usize)
                        .map(|b| b.saturating_sub(offset) / stride.max(1));
                    if let Some(why) = self.element(global, index, bound, elements) {
                        return Some(format!("a {what} of {buffer} at {why}"));
                    }
                    ty = elem;
                    offset = 0;
                }
                Type::Array { elem, len, .. } => {
                    let len = match m.consts().get(*len).map(|d| &d.kind) {
                        Ok(ConstKind::Plain(Constant::Scalar { bits, .. })) => bits.0,
                        _ => return Some(format!("a {what} of {buffer} in an array of no length")),
                    };
                    if !matches!(bound, Bound::AtMost(x) if x < len) {
                        return Some(format!(
                            "a {what} of {buffer} at an index of {bound:?} into {len} elements"
                        ));
                    }
                    ty = *elem;
                }
                Type::Vector { elem, len } => {
                    if !matches!(bound, Bound::AtMost(x) if x < u64::from(*len)) {
                        return Some(format!("a {what} of {buffer} at a lane of {bound:?}"));
                    }
                    ty = *elem;
                }
                Type::Matrix { col, cols } => {
                    if !matches!(bound, Bound::AtMost(x) if x < u64::from(*cols)) {
                        return Some(format!("a {what} of {buffer} at a column of {bound:?}"));
                    }
                    ty = *col;
                }
                _ => return Some(format!("a {what} of {buffer} into a type the guard skips")),
            }
        }
        None
    }

    /// Why element `index` (bound `bound`) of `global`'s runtime array, of `elements` elements
    /// where the launch describes them, may lie past it.
    fn element(
        &mut self,
        global: GlobalId,
        index: Value,
        bound: Bound,
        elements: Option<u64>,
    ) -> Option<String> {
        match (self.launch, bound, elements) {
            (_, Bound::Inside(g), _) if g == global => None,
            (Launch::Fragment, _, _) if self.constant(index).is_some() => None,
            (Launch::Fragment, _, _) => Some(
                "an index bounded by neither a constant nor the buffer's length (OpArrayLength)"
                    .into(),
            ),
            (Launch::Dispatch { .. }, Bound::AtMost(x), Some(n)) if x < n => None,
            (Launch::Dispatch { .. }, Bound::AtMost(x), Some(n)) => {
                Some(format!("up to element {x} of the {n} the kernel describes"))
            }
            (Launch::Dispatch { .. }, _, None) => {
                Some("a binding the kernel does not describe".into())
            }
            (Launch::Dispatch { .. }, b, Some(n)) => Some(format!(
                "an index of {b:?} the dispatch does not bound below {n} elements"
            )),
        }
    }

    /// An access through a pointer of type `ty` the guard does not trace to its variable: a
    /// failure where it may point into a buffer.
    fn untraced(&self, ty: crate::spirv::ids::TypeId, what: &str) -> Option<String> {
        match self.m.types().get(ty).ok()? {
            Type::Pointer {
                class:
                    StorageClass::StorageBuffer
                    | StorageClass::Uniform
                    | StorageClass::PhysicalStorageBuffer,
                ..
            } => Some(format!(
                "a {what} through a buffer pointer the guard does not trace to its variable"
            )),
            _ => None,
        }
    }
}

/// `build`'s result and the modules it finished, where it builds on the device (a check that
/// does not build there does not run there).
fn built<T>(build: impl FnOnce() -> Result<T, Error>) -> Option<(T, Vec<TypedModule>)> {
    let (built, modules) = finished(build);
    built.ok().map(|b| (b, modules))
}

/// The modules of every check that draws a fragment stage on `device`, by check.
fn fragment_checks(device: &Device) -> Vec<(&'static str, Vec<TypedModule>)> {
    let w = RASTER_WIDTH;
    let checks = [
        (
            "color writes",
            built(|| color_write_modules(device)).map(|b| b.1),
        ),
        (
            "fragment's depth",
            built(|| fragcoord_depth_check_fragment(device, w)).map(|b| b.1),
        ),
        (
            "centre varyings",
            built(|| center_varyings_check_fragment(device, w)).map(|b| b.1),
        ),
        (
            "coverage counts",
            built(|| coverage_count_check_fragment(device, w)).map(|b| b.1),
        ),
        (
            "early tests",
            built(|| early_tests_check_fragment(device, w)).map(|b| b.1),
        ),
        (
            "barycentric pair",
            built(|| barycentric_pair_check_fragment(device, true)).map(|b| b.1),
        ),
        (
            "barycentric control",
            built(|| barycentric_pair_check_fragment(device, false)).map(|b| b.1),
        ),
        (
            "cube gradients",
            built(|| cube_gradients_check_fragment(device)).map(|b| b.1),
        ),
        (
            "color outputs",
            built(|| color_outputs_check_fragment(device, 3)).map(|b| b.1),
        ),
        (
            "depth writes",
            built(|| depth_writes_check_fragment(device, w)).map(|b| b.1),
        ),
    ];
    checks
        .into_iter()
        .filter_map(|(what, modules)| Some((what, modules?)))
        .collect()
}

/// Every compute check kernel of `device` and its module, by check.
fn kernel_checks(device: &Device) -> Vec<(String, CheckKernel, Vec<TypedModule>)> {
    let mut checks: Vec<(String, Option<(CheckKernel, Vec<TypedModule>)>)> = vec![
        (
            "half quantize".into(),
            built(|| half_quantize_check(device)),
        ),
        ("narrow mask".into(), built(|| narrow_mask_check(device))),
        (
            "composite inserts".into(),
            built(|| composite_inserts_check(device)),
        ),
        (
            "forwarded loads".into(),
            built(|| loads_forwarded_check(device)),
        ),
        (
            "localized loads".into(),
            built(|| localized_loads_check(device)),
        ),
    ];
    for mode in MathMode::of(device) {
        checks.push((
            format!("canonical NaNs ({mode:?})"),
            built(|| canonical_nans_check(device, mode)),
        ));
        checks.push((
            format!("float denormals ({mode:?})"),
            built(|| float_denormals_check(device, mode)),
        ));
        checks.push((
            format!("half round trip ({mode:?})"),
            built(|| half_round_trip_check(device, mode)),
        ));
    }
    checks
        .into_iter()
        .filter_map(|(what, built)| built.map(|(kernel, modules)| (what, kernel, modules)))
        .collect()
}

/// The local size of `m`'s compute entry, where a mode states it.
fn local_size(m: &TypedModule) -> Option<[u32; 3]> {
    m.entry().modes.iter().find_map(|mode| match mode {
        ExecMode::LocalSize { x, y, z } => Some([*x, *y, *z]),
        _ => None,
    })
}

/// Every check module stays inside its buffers on a device without robust buffer access, for
/// every invocation its draw or dispatch can launch (the module docs): each fragment stage's
/// reads, a helper invocation's past the target's last row and column included, and each access
/// of a check kernel's dispatch of whole workgroups.
#[test]
fn every_check_module_stays_inside_its_buffers() {
    let mut escapes: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut kernels = 0usize;
    let mut fragments = 0usize;
    for (stem, desc) in committed_descriptions().expect("profiles") {
        let device = Device::new(desc.clone(), DevicePolicy::default()).expect(stem);
        let mut found = |what: &str, why: String| {
            escapes
                .entry(format!("{what}: {why}"))
                .or_default()
                .push(stem.clone());
        };
        for (what, modules) in fragment_checks(&device) {
            for m in &modules {
                if m.entry().model != ExecModel::Fragment {
                    continue;
                }
                fragments += 1;
                for why in Guard::new(m, Launch::Fragment, Vec::new()).escapes() {
                    found(what, why);
                }
            }
        }
        for (what, kernel, modules) in kernel_checks(&device) {
            let [m] = modules.as_slice() else {
                found(
                    what.as_str(),
                    format!("{} modules where one was expected", modules.len()),
                );
                continue;
            };
            let Some(local) = local_size(m) else {
                found(what.as_str(), "a kernel without a local size".into());
                continue;
            };
            kernels += 1;
            let launch = Launch::Dispatch {
                groups: [kernel.workgroups, 1, 1],
                local,
            };
            let bytes = kernel.buffers.iter().map(|b| 4 * b.len() as u64).collect();
            for why in Guard::new(m, launch, bytes).escapes() {
                found(what.as_str(), why);
            }
        }
    }
    assert!(
        fragments > 0 && kernels > 0,
        "{fragments} fragments, {kernels} kernels"
    );
    let report: Vec<String> = escapes
        .iter()
        .map(|(why, stems)| format!("{why} (on {})", stems.join(", ")))
        .collect();
    assert!(report.is_empty(), "{}", report.join("\n"));
}
