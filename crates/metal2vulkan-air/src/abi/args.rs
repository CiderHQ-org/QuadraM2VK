//! Argument, output, and argument-buffer member nodes.
//!
//! A node is `[<param or id>] [air.function_constant, !pred] <role key> <role operands>
//! <attribute>*`. Each attribute key has a fixed arity and value type; each role accepts a fixed
//! set of attributes and requires some of them.

use super::keys::{Key, BUFFER_ACCESS, TEXTURE_ACCESS};
use super::md::{kept_bytes, Ops, Site, Tok};
use super::structs::Structs;
use super::{
    AbiInt, AccelerationStructureKind, Access, ArgTypeInfo, BufferArg, Builtin, ControlPointField,
    DepthQualifier, ImageblockArg, ImageblockKind, IndirectArg, Interpolation, MeshField,
    MeshFieldRole, MeshOutputs, MeshTopology, ObjectArg, OutputBinding, OutputRole, ParamBinding,
    ParamRole, PatchInput, Predicate, Sampling, Slot, Stage, StructRef, TextureArg, UserKey,
    Varying,
};
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{AbiError, DiagName};
use crate::ir::{AddrSpace, MdId, Module, Type, TypeId};

/// The decoder state shared by every node: the module, the meter, and the layout arena.
pub(super) struct Cx<'m, 'a> {
    pub(super) m: &'m Module,
    pub(super) meter: &'a mut Meter,
    pub(super) structs: Structs,
}

/// Which list a node sits in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Context {
    /// An entry parameter (`!inputs`).
    Input,
    /// A stage output (`!outputs`).
    Output,
    /// An argument-buffer member (`air.indirect_argument`).
    Indirect,
}

/// Attributes every argument or output may carry.
const COMMON: &[Key] = &[Key::ArgTypeName, Key::ArgName, Key::ArgUnused];
const ACCESS: &[Key] = &[Key::Read, Key::Write, Key::ReadWrite];
const INTERPOLATION: &[Key] = &[
    Key::Center,
    Key::Centroid,
    Key::Sample,
    Key::Perspective,
    Key::NoPerspective,
    Key::Flat,
];
const BUFFER: &[Key] = &[
    Key::LocationIndex,
    Key::BufferSize,
    Key::AddressSpace,
    Key::StructTypeInfo,
    Key::ArgTypeSize,
    Key::ArgTypeAlignSize,
];
const IMAGEBLOCK: &[Key] = &[
    Key::ImageblockDataSize,
    Key::StructTypeInfo,
    Key::ImageblockMaster,
    Key::AliasImplicitImageblock,
    Key::ArgTypeSize,
    Key::ArgTypeAlignSize,
];

/// The attributes of one node, as read.
#[derive(Default)]
struct Attrs {
    seen: Vec<Key>,
    location: Option<Slot>,
    buffer_size: Option<u32>,
    access: Option<Access>,
    space: Option<u32>,
    /// The `air.struct_type_info` layout node, resolved by [`Node::layouts`].
    layout: Option<MdId>,
    size: Option<u32>,
    align: Option<u32>,
    type_name: Option<DiagName>,
    name: Option<DiagName>,
    unused: bool,
    raster_order_group: Option<u32>,
    sampling: Option<Sampling>,
    perspective: Option<bool>,
    flat: bool,
    data_size: Option<u32>,
    /// The `air.imageblock_master` layout node, resolved by [`Node::layouts`].
    master: Option<MdId>,
    alias_implicit: bool,
    shared: bool,
    invariant: bool,
    clip_array_size: Option<u32>,
    depth: Option<DepthQualifier>,
}

impl Attrs {
    fn has(&self, k: Key) -> bool {
        self.seen.contains(&k)
    }

    /// Every attribute read is in `allowed` (or `COMMON`); anything else is a key this role does
    /// not define.
    fn allow(&self, meter: &mut Meter, site: &Site, allowed: &[&[Key]]) -> Result<(), Error> {
        for k in &self.seen {
            meter.tick(1)?;
            if !COMMON.contains(k) && !allowed.iter().any(|set| set.contains(k)) {
                return Err(site.err(AbiError::UnknownKey, Some(k.name())));
            }
        }
        Ok(())
    }

    fn ty(&self) -> ArgTypeInfo {
        ArgTypeInfo {
            size: self.size,
            align: self.align,
            name: self.type_name.clone(),
        }
    }

    fn location(&self, site: &Site) -> Result<Slot, Error> {
        self.location
            .ok_or_else(|| site.err(AbiError::MissingKey, Some(Key::LocationIndex.name())))
    }

    /// A read/write/read_write access (never `air.sample`).
    fn rw_access(&self, site: &Site) -> Result<Access, Error> {
        match self.access {
            Some(a @ (Access::Read | Access::Write | Access::ReadWrite)) => Ok(a),
            Some(Access::Sample) => Err(site.err(AbiError::UnknownKey, Some(Key::Sample.name()))),
            None => Err(site.err(AbiError::MissingKey, Some(BUFFER_ACCESS))),
        }
    }

    /// The interpolation keys, if any were given.
    fn interpolation(&self, site: &Site) -> Result<Option<Interpolation>, Error> {
        let sampling = match (self.sampling, self.access) {
            (Some(s), None) => Some(s),
            (None, Some(Access::Sample)) => Some(Sampling::Sample),
            (None, None) => None,
            _ => return Err(site.err(AbiError::BadValue, Some(Key::Sample.name()))),
        };
        match (self.flat, sampling, self.perspective) {
            (true, None, None) => Ok(Some(Interpolation::Flat)),
            (false, Some(sampling), Some(perspective)) => Ok(Some(Interpolation::Smooth {
                sampling,
                perspective,
            })),
            (false, None, None) => Ok(None),
            _ => Err(site.err(AbiError::BadValue, Some(Key::Flat.name()))),
        }
    }

    fn require_interpolation(&self, site: &Site) -> Result<Interpolation, Error> {
        self.interpolation(site)?
            .ok_or_else(|| site.err(AbiError::MissingKey, Some(Key::Center.name())))
    }
}

/// The layouts a node names, resolved to the arena.
#[derive(Clone, Copy, Default)]
pub(super) struct Layouts {
    layout: Option<StructRef>,
    master: Option<StructRef>,
}

/// The decoded parts of one node. Reading a node never decodes a layout: the layout nodes it
/// names are resolved afterwards ([`Node::layouts`]), so the layout decoder can walk nested
/// layouts with an explicit stack instead of recursing through argument-buffer members.
pub(super) struct Node {
    index: Option<u32>,
    predicate: Option<Predicate>,
    role: Key,
    attrs: Attrs,
    site: Site,
    /// Role operands: the varying user key or imageblock kind.
    word: Option<Box<[u8]>>,
    /// Role operands: the render-target attachment and index.
    target: Option<(AbiInt, Option<u32>)>,
    /// Role operands: the control-point function node and field nodes, or a mesh argument's
    /// `air.mesh_type_info` node.
    nodes: Vec<MdId>,
}

impl Node {
    /// The layout nodes this node names, in attribute order.
    pub(super) fn nested(&self) -> impl Iterator<Item = MdId> + '_ {
        self.attrs.seen.iter().filter_map(|k| match k {
            Key::StructTypeInfo => self.attrs.layout,
            Key::ImageblockMaster => self.attrs.master,
            _ => None,
        })
    }

    /// The site of this node's diagnostics (and of the layouts it names).
    pub(super) fn site(&self) -> &Site {
        &self.site
    }

    /// Resolve the layout nodes this node names with `get`, in attribute order.
    pub(super) fn layouts(
        &self,
        mut get: impl FnMut(MdId, &Site) -> Result<StructRef, Error>,
    ) -> Result<Layouts, Error> {
        let mut l = Layouts::default();
        for k in &self.attrs.seen {
            match (k, self.attrs.layout, self.attrs.master) {
                (Key::StructTypeInfo, Some(id), _) => l.layout = Some(get(id, &self.site)?),
                (Key::ImageblockMaster, _, Some(id)) => l.master = Some(get(id, &self.site)?),
                _ => {}
            }
        }
        Ok(l)
    }
}

impl<'m> Cx<'m, '_> {
    /// Decode one parameter node of an entry of `stage`, whose function's parameter types are
    /// `types` (empty where its type is no function's; the signature check refuses it then).
    pub(super) fn param(
        &mut self,
        id: MdId,
        entry: &DiagName,
        stage: &Stage,
        types: &[TypeId],
    ) -> Result<ParamBinding, Error> {
        let node = self.node(id, Context::Input, entry)?;
        let param = node.index.unwrap_or_default();
        let pointer = types
            .get(param as usize)
            .and_then(|&ty| match self.m.types().get(ty) {
                Some(Type::Ptr { space }) => Some(*space),
                _ => None,
            });
        let layouts = node.layouts(|id, site| Structs::get(self, id, site))?;
        let role = self.role(&node, layouts, Context::Input, pointer)?;
        if !input_in_stage(&role, stage) {
            return Err(node.site.err(AbiError::UnknownKey, Some(node.role.name())));
        }
        Ok(ParamBinding {
            param,
            role,
            unused: node.attrs.unused,
            predicate: node.predicate,
            ty: node.attrs.ty(),
            name: node.attrs.name.clone(),
        })
    }

    /// Decode one output node of an entry of `stage`, the returned value's field `field`.
    pub(super) fn output(
        &mut self,
        id: MdId,
        field: u32,
        entry: &DiagName,
        stage: &Stage,
    ) -> Result<OutputBinding, Error> {
        let node = self.node(id, Context::Output, entry)?;
        let layouts = node.layouts(|id, site| Structs::get(self, id, site))?;
        let role = self.output_role(&node, layouts)?;
        if !output_in_stage(&role, stage) {
            return Err(node.site.err(AbiError::UnknownKey, Some(node.role.name())));
        }
        if node.attrs.unused {
            return Err(node
                .site
                .err(AbiError::UnknownKey, Some(Key::ArgUnused.name())));
        }
        Ok(OutputBinding {
            field,
            role,
            shared: node.attrs.shared,
            predicate: node.predicate,
            ty: node.attrs.ty(),
            name: node.attrs.name.clone(),
        })
    }

    /// Read one argument-buffer member node. The layouts it names ([`Node::nested`]) are
    /// decoded by the caller before [`Cx::indirect`] builds the binding.
    pub(super) fn indirect_node(&mut self, id: MdId, site: &Site) -> Result<Node, Error> {
        let entry = site.entry.clone().unwrap_or_else(|| DiagName::new(""));
        self.node(id, Context::Indirect, &entry)
    }

    /// Build one argument-buffer member binding from its node and resolved layouts.
    pub(super) fn indirect(&mut self, node: &Node, layouts: Layouts) -> Result<IndirectArg, Error> {
        let role = self.role(node, layouts, Context::Indirect, None)?;
        if node.attrs.unused {
            return Err(node
                .site
                .err(AbiError::UnknownKey, Some(Key::ArgUnused.name())));
        }
        Ok(IndirectArg {
            ordinal: node.index.unwrap_or_default(),
            role,
            ty: node.attrs.ty(),
            name: node.attrs.name.clone(),
        })
    }

    /// Read a node's index, predicate, role key, role operands, and attributes.
    fn node(&mut self, id: MdId, cx: Context, entry: &DiagName) -> Result<Node, Error> {
        self.meter.tick(1)?;
        let site = Site {
            entry: Some(entry.clone()),
            arg: None,
        };
        let mut ops = Ops::node(self.m, id, site)?;
        let index = match cx {
            Context::Input | Context::Indirect => {
                let i = ops.u32(None)?;
                ops.site.arg = Some(i);
                Some(i)
            }
            Context::Output => None,
        };
        let mut role = key(&mut ops)?;
        let mut predicate = None;
        if role == Key::FunctionConstant && cx != Context::Indirect {
            predicate = Some(self.predicate(&mut ops)?);
            role = key(&mut ops)?;
        }
        let mut node = Node {
            index,
            predicate,
            role,
            attrs: Attrs::default(),
            site: ops.site.clone(),
            word: None,
            target: None,
            nodes: Vec::new(),
        };
        let role_key = Some(role.name());
        match (role, cx) {
            (Key::FragmentInput, Context::Input) | (Key::VertexOutput, Context::Output) => {
                node.word = Some(ops.string(role_key)?.into());
            }
            (Key::Imageblock, Context::Input) => node.word = Some(ops.string(role_key)?.into()),
            (Key::RenderTarget, Context::Output) => {
                node.target = Some((ops.abi_int(role_key)?, Some(ops.u32(role_key)?)));
            }
            (Key::RenderTarget, Context::Input) => {
                node.target = Some((ops.abi_int(role_key)?, None))
            }
            (Key::Mesh, Context::Input) => {
                let child = ops.child(role_key)?;
                self.meter.reserve(&mut node.nodes, 1)?;
                node.nodes.push(child);
            }
            (Key::PatchControlPointInput, Context::Input) => {
                while !ops.is_done() {
                    self.meter.tick(1)?;
                    let child = ops.child(role_key)?;
                    self.meter.reserve(&mut node.nodes, 1)?;
                    node.nodes.push(child);
                }
            }
            _ => {}
        }
        let vertex_input = (role, cx) == (Key::VertexInput, Context::Input);
        self.attrs(&mut ops, &mut node.attrs, vertex_input)?;
        ops.finish()?;
        node.site = ops.site;
        Ok(node)
    }

    /// `air.function_constant, !{<predicate global>, <type>, <name>}`.
    fn predicate(&mut self, ops: &mut Ops<'m>) -> Result<Predicate, Error> {
        let key = Some(Key::FunctionConstant.name());
        let child = ops.child(key)?;
        let mut p = Ops::node(self.m, child, ops.site.clone())?;
        let global = p.global(key)?;
        // The MSL type name is not kept (the global's IR type is checked instead).
        p.string(key)?;
        let name = p.diag(key, self.meter)?;
        p.finish()?;
        Ok(Predicate { global, name })
    }

    /// Read attributes until the node ends. A `vertex_input` node's location may be followed by
    /// its attribute's user key, which AIR 2.2 (macOS 13's compiler) writes there as
    /// `generated(..)`: a vertex input binds by its location alone, so the key is read and not
    /// kept. Any other node's non-key string is `UnknownKey`.
    fn attrs(&mut self, ops: &mut Ops<'m>, a: &mut Attrs, vertex_input: bool) -> Result<(), Error> {
        while !ops.is_done() {
            self.meter.tick(1)?;
            let k = key(ops)?;
            if a.has(k) {
                return Err(ops.site.err(AbiError::BadValue, Some(k.name())));
            }
            let name = Some(k.name());
            match k {
                Key::LocationIndex => {
                    a.location = Some(Slot {
                        index: ops.abi_int(name)?,
                        count: ops.abi_int(name)?,
                    });
                    if vertex_input
                        && matches!(ops.peek(), Some(Tok::Str(s)) if Key::parse(s).is_none())
                    {
                        ops.string(name)?;
                    }
                }
                Key::BufferSize => a.buffer_size = Some(ops.u32(name)?),
                Key::Read => set(&mut a.access, Access::Read, ops, k)?,
                Key::Write => set(&mut a.access, Access::Write, ops, k)?,
                Key::ReadWrite => set(&mut a.access, Access::ReadWrite, ops, k)?,
                Key::Sample => set(&mut a.access, Access::Sample, ops, k)?,
                Key::AddressSpace => a.space = Some(ops.u32(name)?),
                Key::StructTypeInfo => {
                    a.layout = Some(ops.child(name)?);
                }
                Key::ArgTypeSize => a.size = Some(ops.u32(name)?),
                Key::ArgTypeAlignSize => a.align = Some(ops.u32(name)?),
                Key::ArgTypeName => a.type_name = Some(ops.diag(name, self.meter)?),
                Key::ArgName => a.name = Some(ops.diag(name, self.meter)?),
                Key::ArgUnused => a.unused = true,
                Key::RasterOrderGroup => a.raster_order_group = Some(ops.u32(name)?),
                Key::Center => set(&mut a.sampling, Sampling::Center, ops, k)?,
                Key::Centroid => set(&mut a.sampling, Sampling::Centroid, ops, k)?,
                Key::Perspective => set(&mut a.perspective, true, ops, k)?,
                Key::NoPerspective => set(&mut a.perspective, false, ops, k)?,
                Key::Flat => a.flat = true,
                Key::ImageblockDataSize => a.data_size = Some(ops.u32(name)?),
                Key::ImageblockMaster => {
                    a.master = Some(ops.child(name)?);
                }
                Key::AliasImplicitImageblock => a.alias_implicit = true,
                Key::Shared => a.shared = true,
                Key::Invariant => a.invariant = true,
                Key::ClipDistanceArraySize => a.clip_array_size = Some(ops.u32(name)?),
                Key::DepthQualifier => {
                    a.depth = Some(match key(ops)? {
                        Key::Any => DepthQualifier::Any,
                        Key::Less => DepthQualifier::Less,
                        Key::Greater => DepthQualifier::Greater,
                        _ => return Err(ops.site.err(AbiError::BadValue, name)),
                    })
                }
                _ => return Err(ops.site.err(AbiError::UnknownKey, name)),
            }
            self.meter.reserve(&mut a.seen, 1)?;
            a.seen.push(k);
        }
        Ok(())
    }

    /// The role of an input or argument-buffer node; `pointer` is the address space of an input's
    /// parameter where its IR type is a pointer.
    fn role(
        &mut self,
        n: &Node,
        l: Layouts,
        cx: Context,
        pointer: Option<AddrSpace>,
    ) -> Result<ParamRole, Error> {
        let a = &n.attrs;
        let site = &n.site;
        let indirect = cx == Context::Indirect;
        let builtin = |meter: &mut Meter, b: Builtin| -> Result<ParamRole, Error> {
            if indirect {
                return Err(site.err(AbiError::UnknownKey, Some(n.role.name())));
            }
            a.allow(meter, site, &[])?;
            Ok(ParamRole::Builtin(b))
        };
        let object = |meter: &mut Meter, allowed: &[Key]| -> Result<ObjectArg, Error> {
            a.allow(meter, site, &[allowed, ACCESS])?;
            Ok(ObjectArg {
                location: a.location(site)?,
                access: a.access,
            })
        };
        Ok(match n.role {
            Key::Buffer | Key::IndirectBuffer => {
                a.allow(
                    self.meter,
                    site,
                    &[BUFFER, ACCESS, &[Key::RasterOrderGroup]],
                )?;
                let space = match (a.space, pointer) {
                    (Some(raw), _) => AddrSpace::from_raw(raw).ok(),
                    // An older compiler's buffer node states no address space (the AIR 2.2 of
                    // macOS 13's system libraries): the parameter's pointer type is the only
                    // carrier, and a node that states one must state that one
                    // (`check::role_type`), so it is the same answer. A thread-space pointer is an
                    // `array_ref` of buffers, whose type does not carry its elements' space, and
                    // an argument-buffer member has no parameter: both still need the key.
                    (
                        None,
                        Some(
                            s @ (AddrSpace::Device | AddrSpace::Constant | AddrSpace::Threadgroup),
                        ),
                    ) => Some(s),
                    (None, _) => {
                        return Err(site.err(AbiError::MissingKey, Some(Key::AddressSpace.name())))
                    }
                };
                let space = match space {
                    Some(
                        s @ (AddrSpace::Device | AddrSpace::Constant | AddrSpace::Threadgroup),
                    ) => s,
                    _ => return Err(site.err(AbiError::BadValue, Some(Key::AddressSpace.name()))),
                };
                // `air.indirect_argument` is an argument buffer's member key: Metal writes it only
                // in an `air.indirect_buffer`'s layout, so an `air.buffer` layout holding one at any
                // depth is malformed, whatever its member's role.
                if n.role == Key::Buffer && l.layout.is_some_and(|r| self.structs.holds_indirect(r))
                {
                    return Err(site.err(AbiError::UnknownKey, Some(Key::IndirectArgument.name())));
                }
                let arg = BufferArg {
                    location: a.location(site)?,
                    access: a.rw_access(site)?,
                    space,
                    buffer_size: a.buffer_size,
                    layout: l.layout,
                    raster_order_group: a.raster_order_group,
                };
                match (n.role, space) {
                    (Key::IndirectBuffer, AddrSpace::Threadgroup) => {
                        return Err(site.err(AbiError::BadValue, Some(Key::AddressSpace.name())))
                    }
                    (Key::IndirectBuffer, _) => ParamRole::ArgumentBuffer(arg),
                    (_, AddrSpace::Threadgroup) if !indirect => ParamRole::Threadgroup(arg),
                    (_, AddrSpace::Threadgroup) => {
                        return Err(site.err(AbiError::BadValue, Some(Key::AddressSpace.name())))
                    }
                    _ => ParamRole::Buffer(arg),
                }
            }
            Key::Texture => {
                a.allow(
                    self.meter,
                    site,
                    &[
                        &[Key::LocationIndex, Key::RasterOrderGroup, Key::Sample],
                        ACCESS,
                    ],
                )?;
                let access = a
                    .access
                    .ok_or_else(|| site.err(AbiError::MissingKey, Some(TEXTURE_ACCESS)))?;
                ParamRole::Texture(TextureArg {
                    location: a.location(site)?,
                    access,
                    raster_order_group: a.raster_order_group,
                })
            }
            Key::Sampler => {
                a.allow(self.meter, site, &[&[Key::LocationIndex]])?;
                ParamRole::Sampler {
                    location: a.location(site)?,
                }
            }
            Key::InstanceAccelerationStructure => ParamRole::AccelerationStructure {
                kind: AccelerationStructureKind::Instance,
                arg: object(self.meter, &[Key::LocationIndex])?,
            },
            Key::PrimitiveAccelerationStructure => ParamRole::AccelerationStructure {
                kind: AccelerationStructureKind::Primitive,
                arg: object(self.meter, &[Key::LocationIndex])?,
            },
            Key::VisibleFunctionTable => {
                ParamRole::FunctionTable(object(self.meter, &[Key::LocationIndex])?)
            }
            Key::IntersectionFunctionTable => {
                ParamRole::IntersectionTable(object(self.meter, &[Key::LocationIndex])?)
            }
            Key::IndirectConstant
            | Key::CommandBuffer
            | Key::ComputePipelineState
            | Key::RenderPipelineState
            | Key::DepthStencilState
                if indirect =>
            {
                match n.role {
                    Key::IndirectConstant => {
                        a.allow(
                            self.meter,
                            site,
                            &[&[Key::LocationIndex, Key::ArgTypeSize, Key::ArgTypeAlignSize]],
                        )?;
                        ParamRole::IndirectConstant {
                            location: a.location(site)?,
                        }
                    }
                    Key::CommandBuffer => {
                        ParamRole::CommandBuffer(object(self.meter, &[Key::LocationIndex])?)
                    }
                    Key::ComputePipelineState => {
                        ParamRole::ComputePipelineState(object(self.meter, &[Key::LocationIndex])?)
                    }
                    Key::RenderPipelineState => {
                        ParamRole::RenderPipelineState(object(self.meter, &[Key::LocationIndex])?)
                    }
                    _ => ParamRole::DepthStencilState(object(self.meter, &[Key::LocationIndex])?),
                }
            }
            Key::StageIn | Key::VertexInput | Key::PatchInput if !indirect => {
                a.allow(self.meter, site, &[&[Key::LocationIndex]])?;
                let location = a.location(site)?;
                match n.role {
                    Key::StageIn => ParamRole::StageIn { location },
                    Key::VertexInput => ParamRole::VertexInput { location },
                    _ => ParamRole::PatchInput(PatchInput::Patch { location }),
                }
            }
            Key::FragmentInput if !indirect => {
                a.allow(self.meter, site, &[INTERPOLATION])?;
                let key = kept_bytes(n.word.as_deref().unwrap_or_default(), self.meter)?;
                ParamRole::FragmentInput(Varying {
                    key: UserKey(key),
                    interpolation: a.require_interpolation(site)?,
                })
            }
            Key::RenderTarget if !indirect => {
                a.allow(self.meter, site, &[&[Key::RasterOrderGroup]])?;
                let (index, _) = n
                    .target
                    .ok_or_else(|| site.err(AbiError::MissingKey, Some(n.role.name())))?;
                ParamRole::ColorInput {
                    index,
                    raster_order_group: a.raster_order_group,
                }
            }
            Key::Imageblock | Key::ImageblockData if !indirect => {
                a.allow(self.meter, site, &[IMAGEBLOCK])?;
                let kind = match (n.role, n.word.as_deref()) {
                    (Key::ImageblockData, None) => ImageblockKind::Data,
                    (Key::Imageblock, Some(b"explicit")) => ImageblockKind::Explicit,
                    (Key::Imageblock, Some(b"implicit")) => ImageblockKind::Implicit,
                    _ => return Err(site.err(AbiError::BadValue, Some(n.role.name()))),
                };
                ParamRole::Imageblock(imageblock(kind, a, l))
            }
            Key::PatchControlPointInput if !indirect => {
                a.allow(self.meter, site, &[])?;
                ParamRole::PatchInput(self.control_points(n)?)
            }
            Key::Payload if !indirect => {
                a.allow(
                    self.meter,
                    site,
                    &[&[Key::StructTypeInfo, Key::ArgTypeSize, Key::ArgTypeAlignSize]],
                )?;
                if a.size.is_none() {
                    return Err(site.err(AbiError::MissingKey, Some(Key::ArgTypeSize.name())));
                }
                ParamRole::Payload { layout: l.layout }
            }
            Key::MeshGridProperties if !indirect => {
                a.allow(self.meter, site, &[])?;
                ParamRole::MeshGridProperties
            }
            Key::Mesh if !indirect => {
                a.allow(self.meter, site, &[])?;
                let info = n.nodes.first().copied().ok_or_else(|| {
                    site.err(AbiError::MissingKey, Some(Key::MeshTypeInfo.name()))
                })?;
                ParamRole::Mesh(Box::new(self.mesh_type_info(info, site)?))
            }
            Key::Position if !indirect => {
                a.allow(self.meter, site, &[INTERPOLATION])?;
                ParamRole::Builtin(Builtin::Position(a.interpolation(site)?))
            }
            Key::BarycentricCoord if !indirect => {
                a.allow(self.meter, site, &[INTERPOLATION])?;
                ParamRole::Builtin(Builtin::BarycentricCoord(a.require_interpolation(site)?))
            }
            Key::ThreadPositionInGrid => builtin(self.meter, Builtin::ThreadPositionInGrid)?,
            Key::ThreadPositionInThreadgroup => {
                builtin(self.meter, Builtin::ThreadPositionInThreadgroup)?
            }
            Key::ThreadgroupPositionInGrid => {
                builtin(self.meter, Builtin::ThreadgroupPositionInGrid)?
            }
            Key::ThreadsPerThreadgroup => builtin(self.meter, Builtin::ThreadsPerThreadgroup)?,
            Key::ThreadsPerGrid => builtin(self.meter, Builtin::ThreadsPerGrid)?,
            Key::ThreadgroupsPerGrid => builtin(self.meter, Builtin::ThreadgroupsPerGrid)?,
            Key::ThreadIndexInThreadgroup => {
                builtin(self.meter, Builtin::ThreadIndexInThreadgroup)?
            }
            Key::ThreadIndexInSimdgroup => builtin(self.meter, Builtin::ThreadIndexInSimdgroup)?,
            Key::ThreadIndexInQuadgroup => builtin(self.meter, Builtin::ThreadIndexInQuadgroup)?,
            Key::SimdgroupIndexInThreadgroup => {
                builtin(self.meter, Builtin::SimdgroupIndexInThreadgroup)?
            }
            Key::QuadgroupIndexInThreadgroup => {
                builtin(self.meter, Builtin::QuadgroupIndexInThreadgroup)?
            }
            Key::ThreadsPerSimdgroup => builtin(self.meter, Builtin::ThreadsPerSimdgroup)?,
            Key::SimdgroupsPerThreadgroup => {
                builtin(self.meter, Builtin::SimdgroupsPerThreadgroup)?
            }
            Key::QuadgroupsPerThreadgroup => {
                builtin(self.meter, Builtin::QuadgroupsPerThreadgroup)?
            }
            Key::DispatchThreadsPerThreadgroup => {
                builtin(self.meter, Builtin::DispatchThreadsPerThreadgroup)?
            }
            Key::DispatchSimdgroupsPerThreadgroup => {
                builtin(self.meter, Builtin::DispatchSimdgroupsPerThreadgroup)?
            }
            Key::StageInGridSize => builtin(self.meter, Builtin::StageInGridSize)?,
            Key::StageInGridOrigin => builtin(self.meter, Builtin::StageInGridOrigin)?,
            Key::VertexId => builtin(self.meter, Builtin::VertexId)?,
            Key::InstanceId => builtin(self.meter, Builtin::InstanceId)?,
            Key::BaseVertex => builtin(self.meter, Builtin::BaseVertex)?,
            Key::BaseInstance => builtin(self.meter, Builtin::BaseInstance)?,
            Key::AmplificationId => builtin(self.meter, Builtin::AmplificationId)?,
            Key::AmplificationCount => builtin(self.meter, Builtin::AmplificationCount)?,
            Key::PositionInPatch => builtin(self.meter, Builtin::PositionInPatch)?,
            Key::PatchId => builtin(self.meter, Builtin::PatchId)?,
            Key::FrontFacing => builtin(self.meter, Builtin::FrontFacing)?,
            Key::PointCoord => builtin(self.meter, Builtin::PointCoord)?,
            Key::PrimitiveId => builtin(self.meter, Builtin::PrimitiveId)?,
            Key::SampleId => builtin(self.meter, Builtin::SampleId)?,
            Key::SampleMaskIn => builtin(self.meter, Builtin::SampleMaskIn)?,
            Key::RenderTargetArrayIndex => builtin(self.meter, Builtin::RenderTargetArrayIndex)?,
            Key::ViewportArrayIndex => builtin(self.meter, Builtin::ViewportArrayIndex)?,
            other => return Err(site.err(AbiError::UnknownKey, Some(other.name()))),
        })
    }

    /// `air.patch_control_point_input, !{air.patch_control_point_function, @fn}, !field...`.
    fn control_points(&mut self, n: &Node) -> Result<PatchInput, Error> {
        let key = Some(Key::PatchControlPointInput.name());
        let (first, rest) = n
            .nodes
            .split_first()
            .ok_or_else(|| n.site.err(AbiError::MissingKey, key))?;
        let mut f = Ops::node(self.m, *first, n.site.clone())?;
        let fn_key = Some(Key::PatchControlPointFunction.name());
        if key_of(&mut f)? != Key::PatchControlPointFunction {
            return Err(n.site.err(AbiError::MissingKey, fn_key));
        }
        let function = match f.next(fn_key)? {
            Tok::Func(func) => func,
            _ => return Err(n.site.err(AbiError::BadValue, fn_key)),
        };
        f.finish()?;
        let mut fields = Vec::new();
        self.meter.reserve(&mut fields, rest.len())?;
        for &id in rest {
            self.meter.tick(1)?;
            let mut ops = Ops::node(self.m, id, n.site.clone())?;
            let predicate = match ops.peek() {
                Some(Tok::Str(s)) if Key::parse(s) == Some(Key::FunctionConstant) => {
                    key_of(&mut ops)?;
                    Some(self.predicate(&mut ops)?)
                }
                _ => None,
            };
            let mut a = Attrs::default();
            self.attrs(&mut ops, &mut a, false)?;
            a.allow(self.meter, &n.site, &[&[Key::LocationIndex]])?;
            fields.push(ControlPointField {
                location: a.location(&n.site)?,
                predicate,
                ty: a.ty(),
                name: a.name.clone(),
                unused: a.unused,
                gated_off: false,
            });
        }
        Ok(PatchInput::ControlPoints { function, fields })
    }

    /// `!{"air.mesh_type_info", !{vertex field...}, !{primitive field...}, i32 NV, i32 NP,
    /// "air.point"|"air.line"|"air.triangle"}`.
    fn mesh_type_info(&mut self, id: MdId, site: &Site) -> Result<MeshOutputs, Error> {
        let key = Some(Key::MeshTypeInfo.name());
        let mut ops = Ops::node(self.m, id, site.clone())?;
        if key_of(&mut ops)? != Key::MeshTypeInfo {
            return Err(site.err(AbiError::MissingKey, key));
        }
        let vertex_list = ops.child(key)?;
        let primitive_list = ops.child(key)?;
        let max_vertices = ops.u32(key)?;
        let max_primitives = ops.u32(key)?;
        let topology = match key_of(&mut ops)? {
            Key::Point => MeshTopology::Point,
            Key::Line => MeshTopology::Line,
            Key::Triangle => MeshTopology::Triangle,
            _ => return Err(site.err(AbiError::BadValue, key)),
        };
        ops.finish()?;
        let vertex = self.mesh_fields(vertex_list, true, site)?;
        let primitive = self.mesh_fields(primitive_list, false, site)?;
        Ok(MeshOutputs {
            vertex,
            primitive,
            max_vertices,
            max_primitives,
            topology,
        })
    }

    /// The field nodes of a mesh type's vertex (`vertex`) or primitive struct.
    fn mesh_fields(
        &mut self,
        list: MdId,
        vertex: bool,
        site: &Site,
    ) -> Result<Vec<MeshField>, Error> {
        let key = Some(Key::MeshTypeInfo.name());
        let ids =
            super::md::node_ops(self.m, list).ok_or_else(|| site.err(AbiError::BadValue, key))?;
        let mut fields = Vec::new();
        self.meter.reserve(&mut fields, ids.len())?;
        for &id in ids {
            self.meter.tick(1)?;
            let id = id.ok_or_else(|| site.err(AbiError::BadValue, key))?;
            let mut ops = Ops::node(self.m, id, site.clone())?;
            // A member under `[[function_constant(..)]]` (the VFX framework's particle meshes).
            let predicate = match ops.peek() {
                Some(Tok::Str(s)) if Key::parse(s) == Some(Key::FunctionConstant) => {
                    key_of(&mut ops)?;
                    Some(self.predicate(&mut ops)?)
                }
                _ => None,
            };
            let role_key = key_of(&mut ops)?;
            let name = Some(role_key.name());
            let data = |ops: &mut Ops<'m>, meter: &mut Meter| -> Result<(u32, UserKey), Error> {
                let slot = ops.u32(name)?;
                let key = kept_bytes(ops.string(name)?, meter)?;
                Ok((slot, UserKey(key)))
            };
            let pending = match (role_key, vertex) {
                (Key::Position, true) => MeshFieldRole::Position { invariant: false },
                (Key::PointSize, true) => MeshFieldRole::PointSize,
                (Key::ClipDistance, true) => MeshFieldRole::ClipDistance { array_size: None },
                (Key::MeshVertexData, true) => {
                    let (slot, key) = data(&mut ops, self.meter)?;
                    MeshFieldRole::VertexData { slot, key }
                }
                (Key::PrimitiveCulled, false) => MeshFieldRole::PrimitiveCulled,
                (Key::PrimitiveId, false) => MeshFieldRole::PrimitiveId,
                (Key::RenderTargetArrayIndex, false) => MeshFieldRole::RenderTargetArrayIndex,
                (Key::ViewportArrayIndex, false) => MeshFieldRole::ViewportArrayIndex,
                (Key::MeshPrimitiveData, false) => {
                    let (slot, key) = data(&mut ops, self.meter)?;
                    MeshFieldRole::PrimitiveData { slot, key }
                }
                _ => return Err(site.err(AbiError::UnknownKey, name)),
            };
            let mut a = Attrs::default();
            self.attrs(&mut ops, &mut a, false)?;
            ops.finish()?;
            let role = match pending {
                MeshFieldRole::Position { .. } => {
                    a.allow(self.meter, site, &[&[Key::Invariant]])?;
                    MeshFieldRole::Position {
                        invariant: a.invariant,
                    }
                }
                MeshFieldRole::ClipDistance { .. } => {
                    a.allow(self.meter, site, &[&[Key::ClipDistanceArraySize]])?;
                    MeshFieldRole::ClipDistance {
                        array_size: a.clip_array_size,
                    }
                }
                other => {
                    a.allow(self.meter, site, &[])?;
                    other
                }
            };
            if a.unused {
                return Err(site.err(AbiError::UnknownKey, Some(Key::ArgUnused.name())));
            }
            fields.push(MeshField {
                role,
                predicate,
                ty: a.ty(),
                name: a.name.clone(),
                gated_off: false,
            });
        }
        Ok(fields)
    }

    /// The role of an output node.
    fn output_role(&mut self, n: &Node, l: Layouts) -> Result<OutputRole, Error> {
        let a = &n.attrs;
        let site = &n.site;
        let simple = |meter: &mut Meter, role: OutputRole| -> Result<OutputRole, Error> {
            a.allow(meter, site, &[&[Key::Shared]])?;
            Ok(role)
        };
        Ok(match n.role {
            Key::Position => {
                a.allow(self.meter, site, &[&[Key::Invariant, Key::Shared]])?;
                OutputRole::Position {
                    invariant: a.invariant,
                }
            }
            Key::VertexOutput => {
                a.allow(self.meter, site, &[&[Key::Shared]])?;
                let key = kept_bytes(n.word.as_deref().unwrap_or_default(), self.meter)?;
                OutputRole::VertexOutput(UserKey(key))
            }
            Key::PointSize => simple(self.meter, OutputRole::PointSize)?,
            Key::ClipDistance => {
                a.allow(
                    self.meter,
                    site,
                    &[&[Key::Shared, Key::ClipDistanceArraySize]],
                )?;
                OutputRole::ClipDistance {
                    array_size: a.clip_array_size,
                }
            }
            Key::RenderTargetArrayIndex => simple(self.meter, OutputRole::RenderTargetArrayIndex)?,
            Key::ViewportArrayIndex => simple(self.meter, OutputRole::ViewportArrayIndex)?,
            Key::RenderTarget => {
                a.allow(self.meter, site, &[&[Key::RasterOrderGroup]])?;
                let (attachment, index) = n
                    .target
                    .ok_or_else(|| site.err(AbiError::MissingKey, Some(n.role.name())))?;
                OutputRole::RenderTarget {
                    attachment,
                    index: index.unwrap_or_default(),
                    raster_order_group: a.raster_order_group,
                }
            }
            Key::Depth => {
                a.allow(self.meter, site, &[&[Key::DepthQualifier]])?;
                OutputRole::Depth(a.depth.ok_or_else(|| {
                    site.err(AbiError::MissingKey, Some(Key::DepthQualifier.name()))
                })?)
            }
            Key::SampleMask => {
                a.allow(self.meter, site, &[])?;
                OutputRole::SampleMask
            }
            Key::Stencil => {
                a.allow(self.meter, site, &[])?;
                OutputRole::Stencil
            }
            Key::ImageblockData => {
                a.allow(self.meter, site, &[IMAGEBLOCK])?;
                OutputRole::ImageblockData(imageblock(ImageblockKind::Data, a, l))
            }
            other => return Err(site.err(AbiError::UnknownKey, Some(other.name()))),
        })
    }
}

// The stages a role belongs to, as bits.
const KERNEL: u8 = 1;
const VERTEX: u8 = 2;
const POST_TESSELLATION: u8 = 4;
const FRAGMENT: u8 = 8;
const OBJECT: u8 = 16;
const MESH: u8 = 32;
const VERTICES: u8 = VERTEX | POST_TESSELLATION;
/// The stages that run as threadgroups of a grid: kernels, object and mesh functions.
const GRIDS: u8 = KERNEL | OBJECT | MESH;
const ANY: u8 = KERNEL | VERTICES | FRAGMENT | OBJECT | MESH;

fn stage_bit(stage: &Stage) -> u8 {
    match stage {
        Stage::Kernel { .. } => KERNEL,
        Stage::Vertex { patch: None } => VERTEX,
        Stage::Vertex { patch: Some(_) } => POST_TESSELLATION,
        Stage::Fragment { .. } => FRAGMENT,
        Stage::Object { .. } => OBJECT,
        Stage::Mesh { .. } => MESH,
    }
}

/// Whether an entry of `stage` may take a parameter of `role` (the MSL attribute tables of each
/// function kind). Resources bind in every stage; a stage input, varying, or builtin belongs to
/// the stages that define it. Tile functions are kernels, so the render-target array and
/// viewport indices and imageblocks are also kernel inputs; fragment functions may take
/// imageblocks, threadgroup memory, and the SIMD-lane builtins. Object and mesh functions take the
/// grid builtins as kernels do, except the dispatch and stage-in ones, and a mesh function its
/// vertex amplification's id and count (the VFX framework's particle meshes take
/// `[[amplification_id]]` under a function constant).
fn input_in_stage(role: &ParamRole, stage: &Stage) -> bool {
    let stages = match role {
        ParamRole::Buffer(_)
        | ParamRole::ArgumentBuffer(_)
        | ParamRole::Texture(_)
        | ParamRole::Sampler { .. }
        | ParamRole::AccelerationStructure { .. }
        | ParamRole::FunctionTable(_)
        | ParamRole::IntersectionTable(_)
        | ParamRole::IndirectConstant { .. }
        | ParamRole::CommandBuffer(_)
        | ParamRole::ComputePipelineState(_)
        | ParamRole::RenderPipelineState(_)
        | ParamRole::DepthStencilState(_) => ANY,
        // An object function takes threadgroup buffers; a mesh function only threadgroup
        // variables (Metal's compiler refuses `[[threadgroup(n)]]` there).
        ParamRole::Threadgroup(_) => KERNEL | FRAGMENT | OBJECT,
        ParamRole::Imageblock(_) => KERNEL | FRAGMENT,
        ParamRole::StageIn { .. } => KERNEL,
        ParamRole::Payload { .. } => OBJECT | MESH,
        ParamRole::MeshGridProperties => OBJECT,
        ParamRole::Mesh(_) => MESH,
        ParamRole::VertexInput { .. } => VERTICES,
        ParamRole::FragmentInput(_) | ParamRole::ColorInput { .. } => FRAGMENT,
        ParamRole::PatchInput(_) => POST_TESSELLATION,
        ParamRole::Builtin(b) => match b {
            Builtin::ThreadPositionInGrid
            | Builtin::ThreadPositionInThreadgroup
            | Builtin::ThreadgroupPositionInGrid
            | Builtin::ThreadsPerThreadgroup
            | Builtin::ThreadsPerGrid
            | Builtin::ThreadgroupsPerGrid
            | Builtin::ThreadIndexInThreadgroup
            | Builtin::SimdgroupIndexInThreadgroup
            | Builtin::QuadgroupIndexInThreadgroup
            | Builtin::SimdgroupsPerThreadgroup
            | Builtin::QuadgroupsPerThreadgroup => GRIDS,
            Builtin::DispatchThreadsPerThreadgroup
            | Builtin::DispatchSimdgroupsPerThreadgroup
            | Builtin::StageInGridSize
            | Builtin::StageInGridOrigin => KERNEL,
            Builtin::ThreadIndexInSimdgroup
            | Builtin::ThreadIndexInQuadgroup
            | Builtin::ThreadsPerSimdgroup => GRIDS | FRAGMENT,
            Builtin::VertexId | Builtin::InstanceId | Builtin::BaseInstance => VERTICES,
            // Metal's compiler refuses `[[base_vertex]]` in a post-tessellation vertex function.
            Builtin::BaseVertex => VERTEX,
            Builtin::AmplificationId | Builtin::AmplificationCount => VERTICES | FRAGMENT | MESH,
            Builtin::PositionInPatch | Builtin::PatchId => POST_TESSELLATION,
            Builtin::Position(_)
            | Builtin::FrontFacing
            | Builtin::PointCoord
            | Builtin::PrimitiveId
            | Builtin::SampleId
            | Builtin::SampleMaskIn
            | Builtin::BarycentricCoord(_) => FRAGMENT,
            Builtin::RenderTargetArrayIndex | Builtin::ViewportArrayIndex => KERNEL | FRAGMENT,
        },
    };
    stages & stage_bit(stage) != 0
}

/// Whether an entry of `stage` may return an output of `role`: vertex functions return the
/// vertex interface, fragment functions the attachments. Kernels return nothing (checked by the
/// entry decode).
fn output_in_stage(role: &OutputRole, stage: &Stage) -> bool {
    let stages = match role {
        OutputRole::Position { .. }
        | OutputRole::VertexOutput(_)
        | OutputRole::PointSize
        | OutputRole::ClipDistance { .. }
        | OutputRole::RenderTargetArrayIndex
        | OutputRole::ViewportArrayIndex => VERTICES,
        OutputRole::RenderTarget { .. }
        | OutputRole::Depth(_)
        | OutputRole::SampleMask
        | OutputRole::Stencil
        | OutputRole::ImageblockData(_) => FRAGMENT,
    };
    stages & stage_bit(stage) != 0
}

fn imageblock(kind: ImageblockKind, a: &Attrs, l: Layouts) -> ImageblockArg {
    ImageblockArg {
        kind,
        data_size: a.data_size,
        layout: l.layout,
        master: l.master,
        alias_implicit: a.alias_implicit,
    }
}

/// Set a single-valued attribute; a second key of the same group is `BadValue`.
fn set<T>(slot: &mut Option<T>, v: T, ops: &Ops<'_>, k: Key) -> Result<(), Error> {
    if slot.is_some() {
        return Err(ops.site.err(AbiError::BadValue, Some(k.name())));
    }
    *slot = Some(v);
    Ok(())
}

/// The next operand as a known key.
fn key(ops: &mut Ops<'_>) -> Result<Key, Error> {
    key_of(ops)
}

pub(super) fn key_of(ops: &mut Ops<'_>) -> Result<Key, Error> {
    let s = ops.string(None)?;
    Key::parse(s).ok_or_else(|| ops.site.err(AbiError::UnknownKey, None))
}
