//! Kernels whose values SPIRV-Cross would forward past a write that overwrites what they read, for
//! the tests that run them through SPIRV-Cross: the product's, through the `spirv-cross` on `PATH`
//! with MoltenVK's options, and the validation crate's, through MoltenVK's pinned SPIRV-Cross. Each
//! module is valid SPIR-V; the tests decide what its MSL must declare.
//!
//! Validation reads each kernel's module as committed SPIR-V assembly
//! (`validation/fixtures/public/msl_portability/<kernel>.spvasm`): the words the serializer writes
//! for it, linked densely ([`super::link_dense`]) and relocated, which `spirv-dis --raw-id` prints.
//! This module's test holds the two equal.

use crate::spv::ImageFormat;

use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, Dim, ExecMode, ExecModel, ImageOperands,
    ImageQuery, ImageType, ImageUsage, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule, UnOp,
};

/// A kernel's module.
pub struct Kernel {
    pub module: TypedModule,
}

fn slot(kind: ResourceKind, index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind,
        index,
    }
}

/// A `Block { elem data[]; }` storage buffer (std430) in buffer slot `index`.
fn storage_buffer(
    b: &mut ModuleBuilder,
    index: u32,
    elem: TypeId,
    stride: u32,
) -> BResult<GlobalId> {
    let block = block_of(b, elem, stride)?;
    b.resource(
        slot(ResourceKind::Buffer, index),
        StorageClass::StorageBuffer,
        block,
    )
}

fn block_of(b: &mut ModuleBuilder, elem: TypeId, stride: u32) -> BResult<TypeId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(stride),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })
}

fn scalar(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })?;
    f.module().const_value(c)
}

/// A pointer to element `i` of storage buffer `g`.
fn element(f: &mut FunctionBuilder<'_>, g: GlobalId, i: Value) -> BResult<Value> {
    let u32_ = f.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let zero = scalar(f, u32_, 0)?;
    let base = f.module().global_value(g)?;
    f.access_chain(base, &[zero, i])
}

/// `gl_GlobalInvocationID.x`.
fn invocation(f: &mut FunctionBuilder<'_>) -> BResult<Value> {
    let u32_ = f.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = f.module().ty(Type::Vector { elem: u32_, len: 3 })?;
    let g = f
        .module()
        .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let p = f.module().global_value(g)?;
    let v = f.load(p, MemAccess::default())?;
    f.composite_extract(v, &[0])
}

/// The compute module of `b` whose entry runs `body`.
fn finish(
    mut b: ModuleBuilder,
    body: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()>,
) -> BResult<TypedModule> {
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    body(&mut f)?;
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// Four quantizations of a loaded `vec4 x`, read once each in a shape SPIRV-Cross reads once
/// though the SPIR-V names the value more often, then `x.x += 4`, which SPIRV-Cross performs in
/// place on `x`, then the four stored: a swizzle (`OpVectorShuffle q q 2 1 0 3`, every component
/// from the first operand), a splat (`OpCompositeConstruct s s s s` of a quantized `x.x`), two
/// adjacent scalar extracts in a constructor (`vec4(m.x, m.y, 0, 1)`), and a quantization stored
/// whole that also has a scalar extract nothing reads. Buffer slot 0: the `vec4` elements.
pub fn quantized_after_a_step() -> BResult<Kernel> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let vectors = storage_buffer(&mut b, 0, vec4, 16)?;
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let i = invocation(f)?;
        let at = element(f, vectors, i)?;
        let x = f.load(at, MemAccess::default())?;
        let q = f.unary(UnOp::QuantizeToF16, x)?;
        let swizzled = f.vector_shuffle(q, q, &[2, 1, 0, 3])?;
        let xx = f.composite_extract(x, &[0])?;
        let s = f.unary(UnOp::QuantizeToF16, xx)?;
        let splat = f.composite_construct(vec4, &[s, s, s, s])?;
        let m = f.unary(UnOp::QuantizeToF16, x)?;
        let mx = f.composite_extract(m, &[0])?;
        let my = f.composite_extract(m, &[1])?;
        let zero = scalar(f, f32_, 0)?;
        let one = scalar(f, f32_, u64::from(1f32.to_bits()))?;
        let merged = f.composite_construct(vec4, &[mx, my, zero, one])?;
        let whole = f.unary(UnOp::QuantizeToF16, x)?;
        f.composite_extract(whole, &[2])?;
        let four = scalar(f, f32_, u64::from(4f32.to_bits()))?;
        let e = f.composite_extract(x, &[0])?;
        let e = f.binary(BinOp::FAdd, e, four)?;
        let stepped = f.composite_insert(e, x, &[0])?;
        for v in [stepped, swizzled, splat, merged, whole] {
            let dst = element(f, vectors, i)?;
            f.store(dst, v, MemAccess::default())?;
        }
        Ok(())
    };
    Ok(Kernel {
        module: finish(b, body)?,
    })
}

/// An image and a buffer chosen from descriptor arrays by `h.x`, for a loaded `uvec2 h`: a fetch,
/// a size, a level count, a sample, a fetch through `OpImage` of the sampled image, and an array
/// length through them, each read once after `h.x += 1`, which SPIRV-Cross performs in place on
/// `h`. Buffer slot 0: the `uvec2` elements;
/// texture slot 0: four sampled 2D images; sampler slot 0; buffer slot 1: four `uint[]` blocks;
/// buffer slot 2: `vec4` results; buffer slot 3: `uint` results.
pub fn images_after_their_index_steps() -> BResult<Kernel> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let i32_ = b.ty(Type::Int {
        width: 32,
        signed: true,
    })?;
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let ivec2 = b.ty(Type::Vector { elem: i32_, len: 2 })?;
    let four = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(4),
    })?;
    let pairs = storage_buffer(&mut b, 0, uvec2, 8)?;
    let image = b.ty(Type::Image(ImageType {
        sampled_type: f32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: ImageFormat::Unknown,
    }))?;
    let images = b.ty(Type::Array {
        elem: image,
        len: four,
        stride: None,
    })?;
    let textures = b.resource(
        slot(ResourceKind::Texture, 0),
        StorageClass::UniformConstant,
        images,
    )?;
    let sampler_ty = b.ty(Type::Sampler)?;
    let sampler = b.resource(
        slot(ResourceKind::Sampler, 0),
        StorageClass::UniformConstant,
        sampler_ty,
    )?;
    let words = block_of(&mut b, u32_, 4)?;
    let blocks = b.ty(Type::Array {
        elem: words,
        len: four,
        stride: None,
    })?;
    let buffers = b.resource(
        slot(ResourceKind::Buffer, 1),
        StorageClass::StorageBuffer,
        blocks,
    )?;
    let texels = storage_buffer(&mut b, 2, vec4, 16)?;
    let counts = storage_buffer(&mut b, 3, u32_, 4)?;
    let body = |f: &mut FunctionBuilder<'_>| -> BResult<()> {
        let i = invocation(f)?;
        let at = element(f, pairs, i)?;
        let h = f.load(at, MemAccess::default())?;
        let index = f.composite_extract(h, &[0])?;
        let base = f.module().global_value(textures)?;
        let p = f.access_chain(base, &[index])?;
        let img = f.load(p, MemAccess::default())?;
        let zero = scalar(f, i32_, 0)?;
        let origin = f.composite_construct(ivec2, &[zero, zero])?;
        let level = ImageOperands {
            lod: Some(zero),
            ..ImageOperands::default()
        };
        let fetched = f.fetch(img, origin, &level)?;
        let size = f.query(ImageQuery::SizeLod(zero), img)?;
        let width = f.composite_extract(size, &[0])?;
        let levels = f.query(ImageQuery::Levels, img)?;
        let s = f.module().global_value(sampler)?;
        let s = f.load(s, MemAccess::default())?;
        let si = f.sampled_image(img, s)?;
        let half = scalar(f, f32_, u64::from(0.5f32.to_bits()))?;
        let middle = f.composite_construct(vec2, &[half, half])?;
        let fzero = scalar(f, f32_, 0)?;
        let explicit = ImageOperands {
            lod: Some(fzero),
            ..ImageOperands::default()
        };
        let sampled = f.sample(si, middle, None, false, &explicit)?;
        let through = f.image(si)?;
        let refetched = f.fetch(through, origin, &level)?;
        let base = f.module().global_value(buffers)?;
        let buffer = f.access_chain(base, &[index])?;
        let length = f.array_length(buffer, 0)?;
        let one = scalar(f, u32_, 1)?;
        let next = f.binary(BinOp::IAdd, index, one)?;
        let stepped = f.composite_insert(next, h, &[0])?;
        let dst = element(f, pairs, i)?;
        f.store(dst, stepped, MemAccess::default())?;
        for v in [fetched, sampled, refetched] {
            let dst = element(f, texels, i)?;
            f.store(dst, v, MemAccess::default())?;
        }
        for v in [width, levels, length] {
            let dst = element(f, counts, i)?;
            f.store(dst, v, MemAccess::default())?;
        }
        Ok(())
    };
    Ok(Kernel {
        module: finish(b, body)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv::serialize::{relocate, serialize};
    use metal2vulkan_devtools::{fixtures_dir, spirv_assemble, AssembleError};

    /// Each kernel's committed assembly is the module the serializer writes for it, linked densely
    /// and relocated: every word but the header's generator, which names the assembler. A
    /// serializer change that moves a kernel's words fails here; its message holds the words, for
    /// `spirv-dis --raw-id` to write the new assembly.
    #[test]
    fn the_committed_assembly_is_each_kernels_module() {
        let kernels: [(&str, fn() -> BResult<Kernel>); 2] = [
            ("forwarding_quantized_after_a_step", quantized_after_a_step),
            (
                "forwarding_images_after_their_index_steps",
                images_after_their_index_steps,
            ),
        ];
        for (name, kernel) in kernels {
            let relocatable = serialize(kernel().expect(name).module).expect("serialize");
            let table = super::super::link_dense(&relocatable);
            let words = relocate(&relocatable, &table).expect("relocate");
            let hex = words
                .iter()
                .map(|word| format!("{word:08x}"))
                .collect::<Vec<_>>()
                .join(" ");
            let path = fixtures_dir()
                .join("msl_portability")
                .join(format!("{name}.spvasm"));
            let committed = match spirv_assemble(&path) {
                Ok(committed) => committed,
                Err(AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return;
                }
                Err(AssembleError::Rejected(why)) => {
                    panic!("{name}: {why}\nthe kernel's words: {hex}")
                }
            };
            let without_generator = |words: &[u32]| {
                let mut words = words.to_vec();
                if let Some(generator) = words.get_mut(2) {
                    *generator = 0;
                }
                words
            };
            assert_eq!(
                without_generator(&committed),
                without_generator(&words),
                "{name}: the committed assembly is not the kernel's module; its words: {hex}"
            );
        }
    }
}
