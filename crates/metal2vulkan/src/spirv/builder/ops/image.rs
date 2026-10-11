//! Image, sampler, and derivative constructors.
//!
//! Operand legality is checked per opcode and image type: `Bias` only with implicit LOD, `Lod` or
//! `Grad` (exactly one) with explicit LOD, `Dref` only on depth images, `ConstOffset` a constant,
//! `Offset` only on gathers and samples, both offsets signed 32-bit integers (SPIRV-Cross hands an
//! unsigned one to MSL, whose `sample` and `gather` reject it), `Sample` only (and always) on
//! multisampled images, `Lod` never with multisampling, `OpImageRead`/`OpImageWrite` only on
//! storage images, `OpImageFetch` only on sampled images, and the Vulkan memory model's texel
//! operands ([`TexelMemory`], which only a storage read or write carries) never on subpass data.
//! Implicit-LOD sampling, `OpImageQueryLod`, and derivatives need implicit derivatives, which
//! `finalize` checks against the entry's execution model.
//!
//! Vulkan admits `Offset` outside a gather only with the `maintenance8` feature
//! (VUID-RuntimeSpirv-Offset-10213). The builder is profile-free: a sample with `Offset` is
//! well formed, and requirement derivation records that the module needs `maintenance8`, which
//! the profile check refuses on a device without it.

use crate::spirv::builder::function::FunctionBuilder;
use crate::spirv::builder::ops::{err, Num};
use crate::spirv::error::{BResult, ImageRule, OperandRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    DerivOp, Dim, ImageOperandKind as K, ImageOperands, ImageQuery, ImageType, ImageUsage, Inst,
    SampleForm, TexelMemory, Type, ValueOrigin,
};

/// Which image operands an opcode admits.
#[derive(Clone, Copy, Debug, Default)]
struct Allowed {
    bias: bool,
    /// `Some(true)`: a float `Lod`; `Some(false)`: an integer `Lod`.
    lod: Option<bool>,
    grad: bool,
    const_offset: bool,
    offset: bool,
    sample: bool,
    min_lod: bool,
}

fn rule(op: crate::spv::Op, r: ImageRule) -> crate::spirv::error::BuildError {
    err(op, OperandRule::Image(r))
}

/// Coordinate components without the array layer or projection.
fn plane(dim: Dim) -> u8 {
    match dim {
        Dim::D1 | Dim::Buffer => 1,
        Dim::D2 | Dim::SubpassData => 2,
        Dim::D3 | Dim::Cube => 3,
    }
}

impl FunctionBuilder<'_> {
    fn image_of_ty(&self, ty: TypeId) -> BResult<Option<ImageType>> {
        Ok(match self.mb().types().get(ty)? {
            Type::Image(img) => Some(*img),
            _ => None,
        })
    }

    /// Operand `i` must be an image value.
    fn image_value(&self, op: crate::spv::Op, i: u8, v: Value) -> BResult<ImageType> {
        self.image_of_ty(self.value_ty(v)?)?
            .ok_or(err(op, OperandRule::Category(i)))
    }

    /// Operand 0 must be a sampled image created by `OpSampledImage` in the open block (Vulkan:
    /// a sampled image is consumed in the block that creates it).
    fn sampled_image_value(&self, op: crate::spv::Op, v: Value) -> BResult<ImageType> {
        let image = match self.mb().types().get(self.value_ty(v)?)? {
            Type::SampledImage { image } => *image,
            _ => return Err(err(op, OperandRule::Category(0))),
        };
        let here = match self.origin(v)? {
            ValueOrigin::Inst(b) => Some(b) == self.current_block(),
            _ => false,
        };
        if !here {
            return Err(err(op, OperandRule::SampledImageBlock));
        }
        self.image_of_ty(image)?
            .ok_or(err(op, OperandRule::Category(0)))
    }

    /// `coord` must be a 32-bit float (`float`) or 32-bit integer scalar/vector of `n` components.
    fn check_coord(&self, op: crate::spv::Op, coord: Value, float: bool, n: u8) -> BResult<()> {
        match self.num(coord)? {
            Some(c)
                if c.n == n && c.width == 32 && (if float { c.is_float() } else { c.is_int() }) =>
            {
                Ok(())
            }
            _ => Err(rule(op, ImageRule::Coordinate)),
        }
    }

    fn check_ops(
        &self,
        op: crate::spv::Op,
        img: &ImageType,
        ops: &ImageOperands,
        allowed: Allowed,
    ) -> BResult<()> {
        let bad = |k| Err(rule(op, ImageRule::OperandType(k)));
        let f32_scalar = |v: Value| -> BResult<bool> {
            Ok(self.num(v)?.is_some_and(|n| is_f32(&n) && n.n == 1))
        };
        let offset_n = plane(img.dim);
        if let Some(v) = ops.bias {
            if !allowed.bias {
                return Err(rule(op, ImageRule::NotAllowed(K::Bias)));
            }
            if !f32_scalar(v)? {
                return bad(K::Bias);
            }
        }
        if let Some(v) = ops.lod {
            let Some(float) = allowed
                .lod
                .filter(|_| !img.multisampled && img.dim != Dim::Buffer)
            else {
                return Err(rule(op, ImageRule::NotAllowed(K::Lod)));
            };
            let ok = match self.num(v)? {
                Some(n) if n.n == 1 && float => is_f32(&n),
                Some(n) if n.n == 1 => n.is_int() && n.width == 32,
                _ => false,
            };
            if !ok {
                return bad(K::Lod);
            }
        }
        if let Some((dx, dy)) = ops.grad {
            if !allowed.grad {
                return Err(rule(op, ImageRule::NotAllowed(K::Grad)));
            }
            for g in [dx, dy] {
                if !self.num(g)?.is_some_and(|n| is_f32(&n) && n.n == offset_n) {
                    return bad(K::Grad);
                }
            }
        }
        for (v, kind, admitted) in [
            (ops.const_offset, K::ConstOffset, allowed.const_offset),
            (ops.offset, K::Offset, allowed.offset),
        ] {
            let Some(v) = v else { continue };
            if !admitted || img.dim == Dim::Cube || img.dim == Dim::Buffer {
                return Err(rule(op, ImageRule::NotAllowed(kind)));
            }
            // Signed, as Metal's offsets are (`int2`, `int3`): SPIR-V admits either signedness,
            // but SPIRV-Cross passes the operand's own type to MSL's `sample` and `gather`, which
            // take only a signed vector.
            let typed = self
                .num(v)?
                .is_some_and(|n| n.is_sint() && n.width == 32 && n.n == offset_n);
            if !typed || (kind == K::ConstOffset && !self.is_plain_const(v)?) {
                return bad(kind);
            }
        }
        if ops.const_offset.is_some() && ops.offset.is_some() {
            return Err(rule(op, ImageRule::NotAllowed(K::Offset)));
        }
        match ops.sample {
            Some(v) => {
                if !allowed.sample || !img.multisampled {
                    return Err(rule(op, ImageRule::NotAllowed(K::Sample)));
                }
                if !self
                    .num(v)?
                    .is_some_and(|n| n.is_int() && n.width == 32 && n.n == 1)
                {
                    return bad(K::Sample);
                }
            }
            None if allowed.sample && img.multisampled => {
                return Err(rule(op, ImageRule::Missing(K::Sample)));
            }
            None => {}
        }
        if let Some(v) = ops.min_lod {
            if !allowed.min_lod {
                return Err(rule(op, ImageRule::NotAllowed(K::MinLod)));
            }
            if !f32_scalar(v)? {
                return bad(K::MinLod);
            }
        }
        Ok(())
    }

    /// `vec4` of the image's sampled type.
    fn texel_ty(&mut self, img: &ImageType) -> BResult<TypeId> {
        self.vec_of(img.sampled_type, 4)
    }

    /// `OpSampledImage` of a sampled, non-buffer image and a sampler. Consume the result in the
    /// same block.
    pub fn sampled_image(&mut self, image: Value, sampler: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::SampledImage;
        let image_ty = self.value_ty(image)?;
        let img = self.image_value(OP, 0, image)?;
        if img.usage != ImageUsage::Sampled || matches!(img.dim, Dim::Buffer | Dim::SubpassData) {
            return Err(rule(OP, ImageRule::ImageType));
        }
        if !matches!(
            self.mb().types().get(self.value_ty(sampler)?)?,
            Type::Sampler
        ) {
            return Err(err(OP, OperandRule::Category(1)));
        }
        let ty = self.module().ty(Type::SampledImage { image: image_ty })?;
        self.emit(Inst::SampledImage { image, sampler }, ty)
    }

    /// `OpImage`: the image of a sampled image.
    pub fn image(&mut self, sampled_image: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::Image;
        self.sampled_image_value(OP, sampled_image)?;
        let ty = match self.mb().types().get(self.value_ty(sampled_image)?)? {
            Type::SampledImage { image } => *image,
            _ => return Err(err(OP, OperandRule::Category(0))),
        };
        self.emit(Inst::Image(sampled_image), ty)
    }

    /// One of the eight `OpImageSample*` opcodes. The LOD form follows from `ops`: with `Lod` or
    /// `Grad` it is explicit, else implicit. `proj` adds a projective coordinate (1D/2D/3D,
    /// non-arrayed images); `dref` compares against a 32-bit float on a depth image (not 3D).
    /// `coord` is a 32-bit float with one component per plane coordinate, plus the array layer,
    /// plus the projection. `ConstOffset` or a dynamic `Offset` (which needs `maintenance8`) on a
    /// non-cube image. Result: a scalar of the sampled type with `dref`, else a `vec4`.
    pub fn sample(
        &mut self,
        sampled_image: Value,
        coord: Value,
        dref: Option<Value>,
        proj: bool,
        ops: &ImageOperands,
    ) -> BResult<Value> {
        self.sample_form(sampled_image, coord, dref, proj, ops, false)
    }

    /// One of the four `OpImageSparseSample*` opcodes: [`FunctionBuilder::sample`] without
    /// projection, whose result is the struct of the residency code (`u32`) and the texel
    /// ([`FunctionBuilder::texels_resident`] reads the code).
    pub fn sparse_sample(
        &mut self,
        sampled_image: Value,
        coord: Value,
        dref: Option<Value>,
        ops: &ImageOperands,
    ) -> BResult<Value> {
        self.sample_form(sampled_image, coord, dref, false, ops, true)
    }

    fn sample_form(
        &mut self,
        sampled_image: Value,
        coord: Value,
        dref: Option<Value>,
        proj: bool,
        ops: &ImageOperands,
        sparse: bool,
    ) -> BResult<Value> {
        let form = SampleForm {
            implicit: ops.lod.is_none() && ops.grad.is_none(),
            proj,
            dref: dref.is_some(),
            sparse,
        };
        let op = form.to_spirv();
        let img = self.sampled_image_value(op, sampled_image)?;
        if img.multisampled
            || (proj && (img.arrayed || !matches!(img.dim, Dim::D1 | Dim::D2 | Dim::D3)))
        {
            return Err(rule(op, ImageRule::ImageType));
        }
        if let Some(d) = dref {
            let float_sampled =
                matches!(self.mb().types().get(img.sampled_type)?, Type::Float { .. });
            if !img.depth || img.dim == Dim::D3 || !float_sampled {
                return Err(rule(op, ImageRule::ImageType));
            }
            if !self.num(d)?.is_some_and(|n| is_f32(&n) && n.n == 1) {
                return Err(rule(op, ImageRule::Value));
            }
        }
        let n = plane(img.dim) + u8::from(img.arrayed) + u8::from(proj);
        self.check_coord(op, coord, true, n)?;
        if !form.implicit && ops.lod.is_some() && ops.grad.is_some() {
            return Err(rule(op, ImageRule::NotAllowed(K::Grad)));
        }
        let allowed = if form.implicit {
            Allowed {
                bias: true,
                const_offset: true,
                offset: true,
                min_lod: true,
                ..Allowed::default()
            }
        } else {
            Allowed {
                lod: Some(true),
                grad: true,
                const_offset: true,
                offset: true,
                min_lod: ops.grad.is_some(),
                ..Allowed::default()
            }
        };
        self.check_ops(op, &img, ops, allowed)?;
        let ty = if dref.is_some() {
            img.sampled_type
        } else {
            self.texel_ty(&img)?
        };
        let ty = self.sparse_result(ty, sparse)?;
        self.emit(
            Inst::ImageSample {
                form,
                sampled_image,
                coord,
                dref,
                ops: *ops,
            },
            ty,
        )
    }

    /// `OpImageFetch` from a sampled (`Sampled = 1`) image that is not a cube: `coord` is a 32-bit
    /// integer (plane + layer); operands `Lod` (integer; not buffer or multisampled),
    /// `ConstOffset`, and `Sample` (multisampled images, required). Result: `vec4`.
    pub fn fetch(&mut self, image: Value, coord: Value, ops: &ImageOperands) -> BResult<Value> {
        self.fetch_form(image, coord, ops, false)
    }

    /// `OpImageSparseFetch`: [`FunctionBuilder::fetch`] whose result is the struct of the residency
    /// code (`u32`) and the texel.
    pub fn sparse_fetch(
        &mut self,
        image: Value,
        coord: Value,
        ops: &ImageOperands,
    ) -> BResult<Value> {
        self.fetch_form(image, coord, ops, true)
    }

    fn fetch_form(
        &mut self,
        image: Value,
        coord: Value,
        ops: &ImageOperands,
        sparse: bool,
    ) -> BResult<Value> {
        let op = if sparse {
            crate::spv::Op::ImageSparseFetch
        } else {
            crate::spv::Op::ImageFetch
        };
        let img = self.image_value(op, 0, image)?;
        // A sparse fetch reads no texel buffer (SPIR-V: `Dim` must not be `Buffer`).
        if img.usage != ImageUsage::Sampled
            || matches!(img.dim, Dim::Cube | Dim::SubpassData)
            || (sparse && img.dim == Dim::Buffer)
        {
            return Err(rule(op, ImageRule::ImageType));
        }
        self.check_coord(op, coord, false, plane(img.dim) + u8::from(img.arrayed))?;
        self.check_ops(
            op,
            &img,
            ops,
            Allowed {
                lod: Some(false),
                const_offset: true,
                sample: true,
                ..Allowed::default()
            },
        )?;
        let ty = self.texel_ty(&img)?;
        let ty = self.sparse_result(ty, sparse)?;
        self.emit(
            Inst::ImageFetch {
                image,
                coord,
                ops: *ops,
                sparse,
            },
            ty,
        )
    }

    fn check_gather(
        &self,
        op: crate::spv::Op,
        sampled_image: Value,
        coord: Value,
        depth: bool,
        ops: &ImageOperands,
    ) -> BResult<ImageType> {
        let img = self.sampled_image_value(op, sampled_image)?;
        if !matches!(img.dim, Dim::D2 | Dim::Cube) || img.multisampled || (depth && !img.depth) {
            return Err(rule(op, ImageRule::ImageType));
        }
        self.check_coord(op, coord, true, plane(img.dim) + u8::from(img.arrayed))?;
        self.check_ops(
            op,
            &img,
            ops,
            Allowed {
                const_offset: true,
                offset: true,
                ..Allowed::default()
            },
        )?;
        Ok(img)
    }

    /// `OpImageGather` of `component` (0..=3) from a 2D or cube image; operands `ConstOffset` or
    /// `Offset` only. Result: `vec4`.
    pub fn gather(
        &mut self,
        sampled_image: Value,
        coord: Value,
        component: u32,
        ops: &ImageOperands,
    ) -> BResult<Value> {
        self.gather_form(sampled_image, coord, component, ops, false)
    }

    /// `OpImageSparseGather`: [`FunctionBuilder::gather`] whose result is the struct of the
    /// residency code (`u32`) and the texel.
    pub fn sparse_gather(
        &mut self,
        sampled_image: Value,
        coord: Value,
        component: u32,
        ops: &ImageOperands,
    ) -> BResult<Value> {
        self.gather_form(sampled_image, coord, component, ops, true)
    }

    fn gather_form(
        &mut self,
        sampled_image: Value,
        coord: Value,
        component: u32,
        ops: &ImageOperands,
        sparse: bool,
    ) -> BResult<Value> {
        let op = if sparse {
            crate::spv::Op::ImageSparseGather
        } else {
            crate::spv::Op::ImageGather
        };
        let img = self.check_gather(op, sampled_image, coord, false, ops)?;
        if component > 3 {
            return Err(rule(op, ImageRule::Value));
        }
        let component_id = self.u32_const(component)?;
        let ty = self.texel_ty(&img)?;
        let ty = self.sparse_result(ty, sparse)?;
        self.emit(
            Inst::ImageGather {
                sampled_image,
                coord,
                component,
                component_id,
                ops: *ops,
                sparse,
            },
            ty,
        )
    }

    /// `OpImageDrefGather` from a 2D or cube depth image; `dref` is a 32-bit float. Result:
    /// `vec4`.
    pub fn dref_gather(
        &mut self,
        sampled_image: Value,
        coord: Value,
        dref: Value,
        ops: &ImageOperands,
    ) -> BResult<Value> {
        self.dref_gather_form(sampled_image, coord, dref, ops, false)
    }

    /// `OpImageSparseDrefGather`: [`FunctionBuilder::dref_gather`] whose result is the struct of
    /// the residency code (`u32`) and the texel.
    pub fn sparse_dref_gather(
        &mut self,
        sampled_image: Value,
        coord: Value,
        dref: Value,
        ops: &ImageOperands,
    ) -> BResult<Value> {
        self.dref_gather_form(sampled_image, coord, dref, ops, true)
    }

    fn dref_gather_form(
        &mut self,
        sampled_image: Value,
        coord: Value,
        dref: Value,
        ops: &ImageOperands,
        sparse: bool,
    ) -> BResult<Value> {
        let op = if sparse {
            crate::spv::Op::ImageSparseDrefGather
        } else {
            crate::spv::Op::ImageDrefGather
        };
        let img = self.check_gather(op, sampled_image, coord, true, ops)?;
        if !self.num(dref)?.is_some_and(|n| is_f32(&n) && n.n == 1) {
            return Err(rule(op, ImageRule::Value));
        }
        let ty = self.texel_ty(&img)?;
        let ty = self.sparse_result(ty, sparse)?;
        self.emit(
            Inst::ImageDrefGather {
                sampled_image,
                coord,
                dref,
                ops: *ops,
                sparse,
            },
            ty,
        )
    }

    /// Checks shared by `OpImageRead` and `OpImageWrite`: a storage image (`Sampled = 2`), a
    /// 32-bit integer coordinate (cubes take 3 components, arrayed or not; subpass data takes a
    /// constant `ivec2`), `Sample` exactly on multisampled images, and the Vulkan memory model's
    /// texel operands never on subpass data, which no invocation writes.
    fn check_storage_access(
        &self,
        op: crate::spv::Op,
        image: Value,
        coord: Value,
        ops: &ImageOperands,
        mem: TexelMemory,
        write: bool,
    ) -> BResult<ImageType> {
        let img = self.image_value(op, 0, image)?;
        if img.usage != ImageUsage::Storage || (write && img.dim == Dim::SubpassData) {
            return Err(rule(op, ImageRule::ImageType));
        }
        let n = match img.dim {
            Dim::Cube | Dim::SubpassData => plane(img.dim),
            _ => plane(img.dim) + u8::from(img.arrayed),
        };
        self.check_coord(op, coord, false, n)?;
        if img.dim == Dim::SubpassData && !self.is_plain_const(coord)? {
            return Err(rule(op, ImageRule::Coordinate));
        }
        self.check_ops(
            op,
            &img,
            ops,
            Allowed {
                sample: true,
                ..Allowed::default()
            },
        )?;
        if img.dim == Dim::SubpassData {
            let present = match mem {
                TexelMemory::Made { .. } => Some(K::MakeTexelVisible),
                TexelMemory::NonPrivate { .. } => Some(K::NonPrivateTexel),
                TexelMemory::Private { volatile: true } => Some(K::VolatileTexel),
                TexelMemory::Private { volatile: false } => None,
            };
            if let Some(kind) = present {
                return Err(rule(op, ImageRule::NotAllowed(kind)));
            }
        }
        Ok(img)
    }

    /// `OpImageRead` from a storage image or subpass input, with the texel operands `mem` (a
    /// `Made` scope is `MakeTexelVisible`). Result: `vec4`.
    pub fn read(
        &mut self,
        image: Value,
        coord: Value,
        ops: &ImageOperands,
        mem: TexelMemory,
    ) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::ImageRead;
        let img = self.check_storage_access(OP, image, coord, ops, mem, false)?;
        let ty = self.texel_ty(&img)?;
        self.emit(
            Inst::ImageRead {
                image,
                coord,
                ops: *ops,
                sparse: false,
                mem,
            },
            ty,
        )
    }

    /// `OpImageSparseRead` from a storage image that is no subpass input or texel buffer:
    /// [`FunctionBuilder::read`] whose result is the struct of the residency code (`u32`) and the
    /// texel.
    pub fn sparse_read(
        &mut self,
        image: Value,
        coord: Value,
        ops: &ImageOperands,
        mem: TexelMemory,
    ) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::ImageSparseRead;
        let img = self.check_storage_access(OP, image, coord, ops, mem, false)?;
        if matches!(img.dim, Dim::SubpassData | Dim::Buffer) {
            return Err(rule(OP, ImageRule::ImageType));
        }
        let ty = self.texel_ty(&img)?;
        let ty = self.sparse_result(ty, true)?;
        self.emit(
            Inst::ImageRead {
                image,
                coord,
                ops: *ops,
                sparse: true,
                mem,
            },
            ty,
        )
    }

    /// `OpImageSparseTexelsResident` of residency code `code`, a `u32` taken from a sparse
    /// access's result. Result: `bool`, true when every texel the access read was resident.
    pub fn texels_resident(&mut self, code: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::ImageSparseTexelsResident;
        if !self
            .num(code)?
            .is_some_and(|n| n.is_int() && n.width == 32 && n.n == 1)
        {
            return Err(err(OP, OperandRule::Category(0)));
        }
        let b = self.module().ty(Type::Bool)?;
        self.emit(Inst::ImageSparseTexelsResident(code), b)
    }

    /// The result type of an access whose plain result is `texel`: `texel` itself, or for a
    /// sparse access the struct of the `u32` residency code and `texel`.
    fn sparse_result(&mut self, texel: TypeId, sparse: bool) -> BResult<TypeId> {
        if !sparse {
            return Ok(texel);
        }
        let code = self.uint_ty(32)?;
        self.module().ty(Type::Struct {
            members: Box::new([code, texel]),
            layout: None,
        })
    }

    /// `OpImageWrite` of a `vec4` of the sampled type to a storage image, with the texel
    /// operands `mem` (a `Made` scope is `MakeTexelAvailable`).
    pub fn write(
        &mut self,
        image: Value,
        coord: Value,
        texel: Value,
        ops: &ImageOperands,
        mem: TexelMemory,
    ) -> BResult<()> {
        const OP: crate::spv::Op = crate::spv::Op::ImageWrite;
        let img = self.check_storage_access(OP, image, coord, ops, mem, true)?;
        let texel_ok = self
            .num(texel)?
            .is_some_and(|n| n.elem == img.sampled_type && n.n == 4);
        if !texel_ok {
            return Err(rule(OP, ImageRule::Value));
        }
        self.emit_void(Inst::ImageWrite {
            image,
            coord,
            texel,
            ops: *ops,
            mem,
        })
    }

    /// An image query on an image value (Vulkan rules):
    ///
    /// - `SizeLod(level)`: a sampled, single-sampled 1D/2D/3D/cube image; `level` a 32-bit
    ///   integer scalar.
    /// - `Size`: a buffer, multisampled, or storage image.
    /// - `Levels`: a sampled, single-sampled 1D/2D/3D/cube image.
    /// - `Samples`: a multisampled 2D image.
    ///
    /// Result: `u32` components, one per size dimension (cubes 2) plus the layer count, for the
    /// size queries; `u32` for levels and samples.
    pub fn query(&mut self, query: ImageQuery, image: Value) -> BResult<Value> {
        let op = query.to_spirv();
        let img = self.image_value(op, 0, image)?;
        let mip_dims = matches!(img.dim, Dim::D1 | Dim::D2 | Dim::D3 | Dim::Cube);
        let ok = match query {
            ImageQuery::SizeLod(_) | ImageQuery::Levels => {
                mip_dims && !img.multisampled && img.usage == ImageUsage::Sampled
            }
            ImageQuery::Size => {
                img.dim != Dim::SubpassData
                    && (img.dim == Dim::Buffer
                        || img.multisampled
                        || img.usage == ImageUsage::Storage)
            }
            ImageQuery::Samples => img.dim == Dim::D2 && img.multisampled,
        };
        if !ok {
            return Err(rule(op, ImageRule::ImageType));
        }
        if let ImageQuery::SizeLod(level) = query {
            if !self
                .num(level)?
                .is_some_and(|n| n.is_int() && n.width == 32 && n.n == 1)
            {
                return Err(rule(op, ImageRule::OperandType(K::Lod)));
            }
        }
        let u32_ty = self.uint_ty(32)?;
        let ty = match query {
            ImageQuery::SizeLod(_) | ImageQuery::Size => {
                let size = match img.dim {
                    Dim::Cube => 2,
                    d => plane(d),
                };
                self.vec_of(u32_ty, size + u8::from(img.arrayed))?
            }
            ImageQuery::Levels | ImageQuery::Samples => u32_ty,
        };
        self.emit(Inst::ImageQuery { query, image }, ty)
    }

    /// `OpImageQueryLod` on a 1D/2D/3D/cube sampled image; `coord` is a 32-bit float with the
    /// plane coordinate count. Result: `vec2` of `f32`.
    pub fn query_lod(&mut self, sampled_image: Value, coord: Value) -> BResult<Value> {
        const OP: crate::spv::Op = crate::spv::Op::ImageQueryLod;
        let img = self.sampled_image_value(OP, sampled_image)?;
        if !matches!(img.dim, Dim::D1 | Dim::D2 | Dim::D3 | Dim::Cube) || img.multisampled {
            return Err(rule(OP, ImageRule::ImageType));
        }
        self.check_coord(OP, coord, true, plane(img.dim))?;
        let f = self.f32_ty()?;
        let ty = self.vec_of(f, 2)?;
        self.emit(
            Inst::ImageQueryLod {
                sampled_image,
                coord,
            },
            ty,
        )
    }

    /// A derivative of a 32-bit float scalar or vector (Vulkan). Result: the operand's type.
    pub fn derivative(&mut self, op: DerivOp, a: Value) -> BResult<Value> {
        let o = op.to_spirv();
        let x = self.want(o, 0, a, Num::is_float)?;
        if x.width != 32 {
            return Err(err(o, OperandRule::Width(0)));
        }
        self.emit(Inst::Derivative { op, a }, x.ty)
    }
}

fn is_f32(n: &Num) -> bool {
    n.is_float() && n.width == 32
}
