//! Buffers shorter than the elements every thread indexes by its grid position.
//!
//! A compute dispatch of a grid of `n` threads along an axis runs every grid position `0..n` on
//! it. When a kernel's entry loads or stores `p[gid.c]`, a GEP over buffer argument `p` whose
//! first index is component `c` of `[[thread_position_in_grid]]` (through zero or sign extensions)
//! and whose other indices are constants, in a block every path from the entry's first block to an
//! exit passes through, thread `n - 1` reaches byte `(n - 1) * stride + reach` of the buffer:
//! `stride` is the indexed type's allocation size and `reach` the end of the access within the
//! element (its constant offset there plus its store size). A case whose authored buffer is
//! shorter accesses past it on every run, and what such a read returns depends on the memory
//! around the buffer, so the checker refuses it unless the case declares the buffer
//! an out-of-bounds target, whose accesses past the end are its subject. Only a direct compute
//! dispatch of the case's grid is judged: a tile dispatch (`imageblock`) runs another grid.
//!
//! Nothing else is judged: an access in a branch (a guard against the length), in a callee, at
//! an index the code computes otherwise, or through a pointer a phi or select chooses reaches
//! what its data decides.

use super::interface::{gep_offset, AirInterface, Param};
use crate::case::{AuthoredCase, ResourceRole, Stage};
use base64::Engine as _;
use metal2vulkan::Budget;
use metal2vulkan_air::abi::{Builtin, ParamRole};
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::graph::{Cfg, PostDomTree};
use metal2vulkan_air::ir::{
    AddrSpace, CastOp, Const, FuncId, Idx, InstKind, Module, Type, TypeId, TypeLayouts, Value,
};
use std::collections::{BTreeMap, BTreeSet};

/// The farthest one buffer argument is accessed at the element a grid component names, by every
/// thread of the entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GridReach {
    /// The buffer argument's entry parameter.
    pub param: u32,
    /// The `[[thread_position_in_grid]]` component that indexes it: 0, 1, or 2.
    pub axis: u32,
    /// The bytes between two threads' elements: the indexed type's allocation size.
    pub stride: u64,
    /// One past the last byte the farthest such access touches in thread 0's element.
    pub reach: u64,
}

/// Every [`GridReach`] of `entry` in `module`, whose parameters are `params`: at most one per
/// (parameter, axis, stride), the farthest. Empty when the entry declares no grid position, or its
/// type layouts or graph do not build.
pub(super) fn scan(module: &Module, entry: FuncId, params: &[Param]) -> Vec<GridReach> {
    let position = params
        .iter()
        .filter(|param| {
            matches!(
                param.binding.role,
                ParamRole::Builtin(Builtin::ThreadPositionInGrid)
            )
        })
        .map(|param| param.binding.param)
        .collect::<BTreeSet<_>>();
    let buffers = params
        .iter()
        .filter(|param| !param.table)
        .filter(|param| {
            matches!(&param.binding.role, ParamRole::Buffer(buffer)
                if matches!(buffer.space, AddrSpace::Device | AddrSpace::Constant))
        })
        .map(|param| param.binding.param)
        .collect::<BTreeSet<_>>();
    if position.is_empty() || buffers.is_empty() {
        return Vec::new();
    }
    let Some(body) = module.func(entry).and_then(|f| f.body.as_ref()) else {
        return Vec::new();
    };
    let Some((first, _)) = body.blocks.iter().next() else {
        return Vec::new();
    };
    let mut meter = Meter::new(&Budget::default());
    let Ok(layouts) = TypeLayouts::new(module.data_layout(), module.types(), &mut meter) else {
        return Vec::new();
    };
    let Ok(cfg) = Cfg::build(body, &mut meter) else {
        return Vec::new();
    };
    let Ok(pdom) = PostDomTree::compute(&cfg, &mut meter) else {
        return Vec::new();
    };
    let scan = Scan {
        module,
        entry,
        layouts: &layouts,
        position: &position,
        buffers: &buffers,
    };
    let mut farthest = BTreeMap::<(u32, u32, u64), u64>::new();
    for (block, _) in body.blocks.iter() {
        if !pdom.post_dominates(block, first) {
            continue;
        }
        for (_, inst) in body.block_insts(block) {
            let (ptr, ty) = match &inst.kind {
                InstKind::Load { ptr, .. } => (*ptr, Some(inst.ty)),
                InstKind::Store { val, ptr, .. } | InstKind::AtomicRmw { ptr, val, .. } => {
                    (*ptr, module.value_ty(entry, *val))
                }
                _ => continue,
            };
            let Some(size) = ty.and_then(|ty| layouts.store_size(ty)) else {
                continue;
            };
            if let Some((param, axis, stride, offset)) = scan.element(ptr) {
                let reach = offset.saturating_add(size);
                let held = farthest.entry((param, axis, stride)).or_default();
                *held = (*held).max(reach);
            }
        }
    }
    farthest
        .into_iter()
        .map(|((param, axis, stride), reach)| GridReach {
            param,
            axis,
            stride,
            reach,
        })
        .collect()
}

/// What [`scan`] reads of the entry.
struct Scan<'a> {
    module: &'a Module,
    entry: FuncId,
    layouts: &'a TypeLayouts,
    /// The entry parameters that are `[[thread_position_in_grid]]`.
    position: &'a BTreeSet<u32>,
    /// The entry's device and constant buffer parameters (not tables).
    buffers: &'a BTreeSet<u32>,
}

impl Scan<'_> {
    /// The (buffer parameter, axis, stride, byte offset in the element) `ptr` addresses: a chain
    /// of constant GEPs over one GEP whose base is a buffer parameter and whose first index is a
    /// grid component.
    fn element(&self, ptr: Value) -> Option<(u32, u32, u64, u64)> {
        let body = self.module.func(self.entry)?.body.as_ref()?;
        let mut offset = 0u64;
        let mut value = ptr;
        loop {
            let Value::Inst(id) = value else {
                return None;
            };
            let InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            } = &body.insts.get(id)?.kind
            else {
                return None;
            };
            if let Some(step) = gep_offset(self.module, self.layouts, *src_ty, indices) {
                offset = offset.checked_add(step)?;
                value = *base;
                continue;
            }
            let (index, inner) = indices.split_first()?;
            let axis = self.axis(*index)?;
            let Value::Arg(arg) = base else {
                return None;
            };
            let param = u32::try_from(arg.index()).ok()?;
            if !self.buffers.contains(&param) {
                return None;
            }
            let stride = self.layouts.alloc_size(*src_ty)?;
            let within = self.inner_offset(*src_ty, inner)?;
            return Some((param, axis, stride, offset.checked_add(within)?));
        }
    }

    /// The grid component `index` is: the scalar position, or a constant lane of the vector one,
    /// through zero and sign extensions.
    fn axis(&self, index: Value) -> Option<u32> {
        let types = self.module.types();
        match index {
            Value::Arg(arg) => {
                let param = u32::try_from(arg.index()).ok()?;
                let ty = self.module.value_ty(self.entry, index)?;
                (self.position.contains(&param) && matches!(types.get(ty)?, Type::Int(_)))
                    .then_some(0)
            }
            Value::Inst(id) => {
                let body = self.module.func(self.entry)?.body.as_ref()?;
                match &body.insts.get(id)?.kind {
                    InstKind::Cast {
                        op: CastOp::ZExt | CastOp::SExt,
                        v,
                    } => self.axis(*v),
                    InstKind::ExtractElement {
                        vec: Value::Arg(arg),
                        idx,
                    } => {
                        let param = u32::try_from(arg.index()).ok()?;
                        let lane = self.constant(*idx)?;
                        (self.position.contains(&param) && lane < 3).then_some(lane)
                    }
                    _ => None,
                }
            }
            Value::Const(_) => None,
        }
    }

    /// The value of the integer constant `value`, when it fits a lane number.
    fn constant(&self, value: Value) -> Option<u32> {
        let Value::Const(id) = value else {
            return None;
        };
        match self.module.consts().get(id)?.value {
            Const::Int(raw) => u32::try_from(raw).ok(),
            Const::Null => Some(0),
            _ => None,
        }
    }

    /// The byte offset the constant `indices` after a GEP's first select inside one element of
    /// `src_ty`.
    fn inner_offset(&self, src_ty: TypeId, indices: &[Value]) -> Option<u64> {
        let mut offset = 0u64;
        let mut ty = src_ty;
        for index in indices {
            let step = u64::from(self.constant(*index)?);
            match self.module.types().get(ty)? {
                Type::Struct { fields, .. } => {
                    let field = usize::try_from(step).ok()?;
                    offset = offset.checked_add(*self.layouts.field_offsets(ty)?.get(field)?)?;
                    ty = *fields.get(field)?;
                }
                Type::Array { elem, .. } | Type::Vector { elem, .. } => {
                    offset =
                        offset.checked_add(step.checked_mul(self.layouts.alloc_size(*elem)?)?)?;
                    ty = *elem;
                }
                _ => return None,
            }
        }
        Some(offset)
    }
}

/// Refuse each authored buffer of a kernel `case` shorter than what every thread of its grid
/// reaches through `interface`'s [`GridReach`]es (the module documentation).
pub(super) fn validate(case: &AuthoredCase, interface: &AirInterface, errors: &mut Vec<String>) {
    use crate::case::out_of_bounds::OutOfBoundsResource;
    let Some(dispatch) = &case.dispatch else {
        return;
    };
    if case.stage != Stage::Kernel || case.imageblock.is_some() {
        return;
    }
    let declared = case
        .out_of_bounds
        .iter()
        .flat_map(|declaration| declaration.targets.iter())
        .filter_map(|target| match target.resource {
            OutOfBoundsResource::Buffer { binding } => Some(binding),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    for reach in &interface.grid_reaches {
        let Some((slot, _)) = interface
            .params
            .iter()
            .find(|param| param.binding.param == reach.param && param.live)
            .and_then(Param::slots)
        else {
            continue;
        };
        if declared.contains(&slot) {
            continue;
        }
        let Some(buffer) = case.buffers.iter().find(|buffer| buffer.binding == slot) else {
            continue;
        };
        if buffer.rasterization_rate_map.is_some() {
            continue;
        }
        let encoded = match buffer.role {
            ResourceRole::Input => buffer.bytes_b64.as_deref(),
            ResourceRole::Output | ResourceRole::InOut => buffer.initial_bytes_b64.as_deref(),
        };
        let Some(Ok(bytes)) =
            encoded.map(|encoded| base64::engine::general_purpose::STANDARD.decode(encoded))
        else {
            continue;
        };
        let threads = u64::from(dispatch.grid[reach.axis as usize]);
        if threads == 0 {
            continue;
        }
        let needed = (threads - 1)
            .saturating_mul(reach.stride)
            .saturating_add(reach.reach);
        let held = bytes.len() as u64;
        if held < needed {
            errors.push(format!(
                "buffer {slot} holds {held} bytes, but every thread of the grid accesses it at the \
                 {}-byte element its `[[thread_position_in_grid]]` component {} names, so thread \
                 {} reaches byte {needed}: author at least {needed} bytes, or declare the buffer \
                 an out_of_bounds target",
                reach.stride,
                ["x", "y", "z"][reach.axis as usize],
                threads - 1
            ));
        }
    }
}

#[cfg(test)]
mod tests;
