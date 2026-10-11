//! Builder rules that `spirv-val` enforces per execution model or per scope: each refused shape is
//! refused by the builder, and its permitted neighbour serializes to a module `spirv-val` accepts.

use super::families::{tys, Tys};
use super::families_image::{image_ty, ivec_const, Img};
use super::{assert_spirv_valid, cval, elem_ptr, slot, ssbo};
use crate::pipeline::sym::ResourceKind;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, EntryRule, GlobalRule, OperandRule};
use crate::spirv::ids::GlobalId;
use crate::spirv::ir::{
    AtomicOp, Dim, ExecMode, ExecModel, ImageOperands, ImageUsage, MemAccess, MemoryOrder, Scope,
    Semantics, StorageClass, TexelMemory, TypedModule,
};
use crate::spv::{ImageFormat, Op};

const MODELS: [ExecModel; 3] = [ExecModel::GLCompute, ExecModel::Vertex, ExecModel::Fragment];

/// A module for `model` whose entry runs `body`, given an SSBO of `u32`s.
fn module_for(
    model: ExecModel,
    body: impl FnOnce(&mut FunctionBuilder<'_>, &Tys, GlobalId) -> BResult<()>,
) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let buf = ssbo(&mut b, 0, t.u32_, 4);
    let mut f = b.function(t.void)?;
    body(&mut f, &t, buf)?;
    f.ret(None)?;
    let main = f.finish()?;
    let modes = match model {
        ExecModel::GLCompute => vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        _ => vec![],
    };
    b.entry(model, main, modes)?;
    b.finalize()
}

fn acq_rel_uniform() -> Semantics {
    Semantics {
        order: MemoryOrder::AcquireRelease,
        uniform_memory: true,
        ..Semantics::RELAXED
    }
}

fn model_error(op: Op) -> Option<BuildError> {
    Some(BuildError::Entry(EntryRule::InstructionForModel(op)))
}

#[test]
fn workgroup_scopes_are_compute_only() {
    type Body = fn(&mut FunctionBuilder<'_>, &Tys, GlobalId) -> BResult<()>;
    let cases: [(Op, Body); 4] = [
        (Op::ControlBarrier, |f, _, _| {
            f.control_barrier(Scope::Workgroup, Scope::Workgroup, Semantics::RELAXED)
        }),
        (Op::ControlBarrier, |f, _, _| {
            f.control_barrier(Scope::Subgroup, Scope::Workgroup, Semantics::RELAXED)
        }),
        (Op::MemoryBarrier, |f, _, _| {
            f.memory_barrier(Scope::Workgroup, acq_rel_uniform())
        }),
        (Op::AtomicIAdd, |f, t, buf| {
            let zero = cval(f, t.u32_, 0);
            let p = elem_ptr(f, buf, zero);
            f.atomic(
                AtomicOp::IAdd,
                p,
                Scope::Workgroup,
                Semantics::RELAXED,
                zero,
            )
            .map(|_| ())
        }),
    ];
    for (op, body) in cases {
        for model in MODELS {
            let m = module_for(model, body);
            if model == ExecModel::GLCompute {
                assert_spirv_valid(m.expect("compute accepts workgroup scope"));
            } else {
                assert_eq!(m.err(), model_error(op), "{op:?} in {model:?}");
            }
        }
    }
    // The subgroup and device scopes stay available to every model.
    for model in MODELS {
        let m = module_for(model, |f, t, buf| {
            f.control_barrier(Scope::Subgroup, Scope::Subgroup, Semantics::RELAXED)?;
            f.control_barrier(Scope::Subgroup, Scope::Device, acq_rel_uniform())?;
            f.memory_barrier(Scope::Device, acq_rel_uniform())?;
            let zero = cval(f, t.u32_, 0);
            let p = elem_ptr(f, buf, zero);
            f.atomic(AtomicOp::IAdd, p, Scope::Device, Semantics::RELAXED, zero)?;
            Ok(())
        });
        assert_spirv_valid(m.expect("non-workgroup scopes"));
    }
}

#[test]
fn invocation_scope_takes_no_semantics() {
    let semantics = Some(BuildError::Operand {
        op: Op::MemoryBarrier,
        rule: OperandRule::Semantics,
    });
    let m = module_for(ExecModel::GLCompute, |f, _, _| {
        f.memory_barrier(Scope::Invocation, acq_rel_uniform())
    });
    assert_eq!(m.err(), semantics);
    let m = module_for(ExecModel::GLCompute, |f, t, buf| {
        let zero = cval(f, t.u32_, 0);
        let p = elem_ptr(f, buf, zero);
        f.atomic(
            AtomicOp::IAdd,
            p,
            Scope::Invocation,
            acq_rel_uniform(),
            zero,
        )
        .map(|_| ())
    });
    assert_eq!(
        m.err(),
        Some(BuildError::Operand {
            op: Op::AtomicIAdd,
            rule: OperandRule::Semantics,
        })
    );
    let m = module_for(ExecModel::GLCompute, |f, t, buf| {
        let zero = cval(f, t.u32_, 0);
        let p = elem_ptr(f, buf, zero);
        f.atomic(
            AtomicOp::IAdd,
            p,
            Scope::Invocation,
            Semantics::RELAXED,
            zero,
        )?;
        f.control_barrier(Scope::Subgroup, Scope::Invocation, Semantics::RELAXED)
    });
    assert_spirv_valid(m.expect("relaxed invocation scope"));
}

#[test]
fn a_relaxed_ordering_names_no_storage_class() {
    let m = module_for(ExecModel::GLCompute, |f, t, buf| {
        let zero = cval(f, t.u32_, 0);
        let p = elem_ptr(f, buf, zero);
        let relaxed_uniform = Semantics {
            uniform_memory: true,
            ..Semantics::RELAXED
        };
        f.atomic(AtomicOp::IAdd, p, Scope::Device, relaxed_uniform, zero)
            .map(|_| ())
    });
    assert_eq!(
        m.err(),
        Some(BuildError::Operand {
            op: Op::AtomicIAdd,
            rule: OperandRule::Semantics,
        })
    );
}

const SUBPASS: Img = Img {
    dim: Dim::SubpassData,
    depth: false,
    arrayed: false,
    ms: false,
    usage: ImageUsage::Storage,
    format: ImageFormat::Unknown,
};

/// A module for `model` that reads input attachment 1.
fn subpass_read(model: ExecModel) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let ty = image_ty(&mut b, t.f32_, SUBPASS);
    let att = b.resource(
        slot(ResourceKind::InputAttachment(1), 0),
        StorageClass::UniformConstant,
        ty,
    )?;
    let mut f = b.function(t.void)?;
    let p = f.module().global_value(att)?;
    let img = f.load(p, MemAccess::default())?;
    let c = ivec_const(&mut f, 2, 0);
    f.read(img, c, &ImageOperands::default(), TexelMemory::PRIVATE)?;
    f.ret(None)?;
    let main = f.finish()?;
    let modes = match model {
        ExecModel::GLCompute => vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        _ => vec![],
    };
    b.entry(model, main, modes)?;
    b.finalize()
}

#[test]
fn subpass_reads_are_fragment_only() {
    let words = assert_spirv_valid(subpass_read(ExecModel::Fragment).expect("fragment read"));
    let text = crate::spirv::testing::spirv_dis(&words);
    assert!(text.contains("InputAttachmentIndex 1"), "{text}");
    for model in [ExecModel::GLCompute, ExecModel::Vertex] {
        assert_eq!(subpass_read(model).err(), model_error(Op::ImageRead));
    }
}

#[test]
fn input_attachments_are_exactly_the_subpass_images() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let subpass = image_ty(&mut b, t.f32_, SUBPASS);
    let storage = image_ty(
        &mut b,
        t.f32_,
        Img {
            dim: Dim::D2,
            format: ImageFormat::Rgba32f,
            ..SUBPASS
        },
    );
    let resource_type = Some(BuildError::Global(GlobalRule::ResourceType));
    let texture = slot(ResourceKind::Texture, 0);
    let attachment = slot(ResourceKind::InputAttachment(0), 0);
    assert_eq!(
        b.resource(texture, StorageClass::UniformConstant, subpass)
            .err(),
        resource_type
    );
    assert_eq!(
        b.resource(attachment, StorageClass::UniformConstant, storage)
            .err(),
        resource_type
    );
}
