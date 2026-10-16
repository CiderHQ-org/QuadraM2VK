//! `lower::sync::memory_model`: the module's memory-model choice and LLVM `fence` instructions.
//!
//! - **The choice.** Every module is `GLSL450` (Metal's default: non-atomic device writes are not
//!   coherent across threadgroups within a dispatch), unless it contains a decoded `*_coherent`
//!   access on a profile with `vulkanMemoryModel` and its device scope, whose lowering calls
//!   [`require_vulkan`] ([`coherent_access`]). The model is the family's plan
//!   ([`MemoryModelPlan::model`]); `sync::barrier::memory_model` reads it when it emits barrier and
//!   fence semantics (`MakeAvailable`/`MakeVisible` only under `Vulkan`).
//! - **Coherent accesses without the Vulkan model** ([`coherent_access`]). On a profile without
//!   those features (both optional in Vulkan 1.2) a buffer's coherent access keeps `GLSL450`, with
//!   plain operands, and the module decorates every storage buffer and storage image `Coherent`
//!   ([`declare`], [`MemoryModelPlan::coherent_resources`]), as an interlocked body's are. Metal
//!   promises such an access device coherence, "memory writes are visible to all threads on the
//!   device, that is, threads across threadgroups" (the Metal Shading Language specification,
//!   4.8 "Memory Coherency"). Vulkan's memory model describes the `Coherent` decoration of legacy
//!   SPIR-V as per-instruction availability and visibility: "shaders using fragment shader
//!   interlock will declare the relevant resources as coherent to get implicit per-instruction
//!   availability and visibility operations", and "GLSL to SPIR-V compilers should map all
//!   (non-atomic) operations on a coherent variable to Make{Pointer,Texel}{Available}{Visible}
//!   flags in this model" (Memory Model appendix). Their scope is the `Device` scope a `GLSL450`
//!   barrier names, which there is the queue family scope instance: "This scope is identified in
//!   SPIR-V as the QueueFamily Scope if the vulkanMemoryModel feature is enabled, or if not, the
//!   Device Scope, which can be used as a Memory Scope for barrier and atomic operations"
//!   (Shaders, "Queue Family" scope). A dispatch's or draw's threadgroups all run on one queue, so
//!   they lie in one such instance: every thread Metal's device coherence names. The two forms
//!   differ only toward work of another queue family running at the same time, which the Vulkan
//!   model's `Device` scope also reaches; Metal's promise holds "if you properly synchronize"
//!   the threads, and work on another queue is ordered with a dispatch only by events and
//!   fences, whose Vulkan synchronization makes the writes available and visible itself. The
//!   decoration covers every shared resource, not only the ones the coherent accesses reach: a
//!   stronger guarantee than Metal's default threadgroup coherence, which Metal permits. A
//!   coherent access through a device address has no variable to decorate, so it needs the Vulkan
//!   model on every profile.
//! - **Accesses under the Vulkan model** ([`coherence`]). The model orders only `NonPrivatePointer`
//!   accesses between invocations, so every ordinary word access of shared memory (buffer views,
//!   device addresses, the threadgroup arena) carries it, keeping the ordering barriers give it
//!   under `GLSL450`; a coherent access adds `MakePointerAvailable` (a store) or
//!   `MakePointerVisible` (a load) at `Device` scope, which is why the model also needs the device
//!   scope. A storage-image access takes the texel operands of the same coherence
//!   ([`Coherence::texel`]). Outside an interlocked body, a function with a storage image (a
//!   storage slot's, a staged attachment's, or a heap storage class's) and a coherent access is not
//!   implemented: its `NonPrivateTexel` accesses are not measured.
//! - **An interlocked body**: every access of memory another invocation may write is made available
//!   and visible at `Device` scope, which is what a fragment-interlock critical section orders
//!   (`VK_EXT_fragment_shader_interlock`: only coherent accesses inside it are ordered with the
//!   other invocations' at the same pixel, as Metal's raster order group orders all of its memory).
//!   Under `GLSL450` the module decorates every storage buffer and storage image `Coherent`
//!   ([`declare`]); under the Vulkan model (which bans that decoration) each buffer,
//!   device-address, and storage-image access carries `MakePointer*`/`MakeTexel*` at `Device`
//!   scope ([`coherence`]). A body that reaches memory through a device address (a slot lowered
//!   to its device address, or a wide pointer), which no decoration covers, selects the Vulkan
//!   model once the slots are placed ([`finish_interlock`]); a device without
//!   `vulkanMemoryModel` and its device scope refuses it, naming the feature.
//! - **Requirements** ([`finish`], [`check`]). The Vulkan model needs `vulkanMemoryModel`; under
//!   it, every `Device`-scope operation (a device fence, a device atomic, and the two relaxed
//!   device-scope word atomics of a sub-word buffer store) also needs
//!   `vulkanMemoryModelDeviceScope`. Both are refused at selection, naming the feature, rather
//!   than left to the finalize verifier. The function's device scope is structural: a family
//!   that emits a `Device`-scope operation reports it ([`note_device_scope`]), and every slot the
//!   function writes counts (its stores may compose sub-word bytes with device atomics).
//! - **LLVM `fence`** (`InstKind::Fence`): `syncscope("singlethread")` orders nothing another
//!   invocation observes and emits nothing. A system-scope fence is `OpMemoryBarrier` at
//!   `Device` scope (the widest scope of one logical device, as for Metal's legacy all-devices
//!   fence) over every memory a stage can reach (`UniformMemory`, `ImageMemory`, and
//!   `WorkgroupMemory` in a kernel), with the ordering mapped as `air.atomic.fence`'s (`seq_cst`
//!   is `AcquireRelease`, and a `seq_cst` fence is also ordered with the command buffer's other
//!   sequentially consistent fences, `barrier::seq_cst_order`). `unordered` and `monotonic` are
//!   not fence orderings in LLVM and never decode here; a relaxed ordering would emit nothing.

use crate::air::intrinsics::MemFlags;
use crate::air::ir::{Inst, InstId, InstKind, Ordering, SyncScope};
use crate::device::requirements::VkFeature;
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::{FamilyPlans, Plan, SelectCx, SlotChoice};
use crate::lower::strategy::FamilyStrategy;
use crate::lower::sync::barrier::{self, BarrierModel};
use crate::memory::leaf::Coherence;
use crate::pipeline::interface::BufferForm;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ir::{ExecMode, MemoryOrder, Scope, Semantics};
use std::borrow::Cow;

/// The sync::memory_model strategies.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryModelStrategy {
    /// A system-scope LLVM `fence`: `OpMemoryBarrier(Device, order | memories)`; a `seq_cst` one
    /// is also ordered with the command buffer's other sequentially consistent fences
    /// (`barrier::seq_cst_order`).
    Fence {
        flags: MemFlags,
        order: MemoryOrder,
        seq_cst: bool,
    },
    /// A single-thread fence: nothing is emitted.
    NoFence,
}

impl FamilyStrategy for MemoryModelStrategy {
    fn name(&self, _: &FamilyPlans) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            MemoryModelStrategy::Fence { seq_cst: true, .. } => "memory_model.fence.seq_cst",
            MemoryModelStrategy::Fence { .. } => "memory_model.fence",
            MemoryModelStrategy::NoFence => "memory_model.fence.none",
        })
    }

    fn words(&self) -> u32 {
        match self {
            // The instruction and its scope and semantics constants; a sequentially consistent
            // one's status word address and atomic.
            MemoryModelStrategy::Fence { seq_cst, .. } => {
                12 + barrier::SEQ_CST_WORDS * u32::from(*seq_cst)
            }
            MemoryModelStrategy::NoFence => 0,
        }
    }
}

/// The memory-model family's part of the per-function plan (`select::FamilyPlans`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryModelPlan {
    /// The module's memory model.
    pub model: BarrierModel,
    /// Some operation of the function has `Device` scope.
    pub device_scope: bool,
    /// The function's coherent accesses take the `GLSL450` form: every storage buffer and storage
    /// image the module declares is `Coherent` ([`coherent_access`], [`declare`]).
    pub coherent_resources: bool,
}

impl Default for MemoryModelPlan {
    fn default() -> MemoryModelPlan {
        MemoryModelPlan {
            model: BarrierModel::Glsl450,
            device_scope: false,
            coherent_resources: false,
        }
    }
}

/// Select the Vulkan memory model for the function (it holds a coherent access, whose
/// `MakePointerAvailable`/`MakePointerVisible` operands have `Device` scope).
pub fn require_vulkan(cx: &mut SelectCx<'_>) {
    cx.families.memory_model.model = BarrierModel::Vulkan;
    note_device_scope(cx);
}

/// The coherence of a buffer's `*_coherent` access on `cx`'s profile (module docs): under the
/// Vulkan model where the profile has `vulkanMemoryModel` and `vulkanMemoryModelDeviceScope`
/// ([`require_vulkan`]: `MakePointerAvailable` or `MakePointerVisible` at `Device` scope), and
/// otherwise the `GLSL450` form, plain operands through `Coherent` resources.
pub fn coherent_access(cx: &mut SelectCx<'_>) -> Coherence {
    if vulkan_model(cx.profile) {
        require_vulkan(cx);
        Coherence::Device
    } else {
        cx.families.memory_model.coherent_resources = true;
        Coherence::Plain
    }
}

/// Whether `p` enables the Vulkan memory model with its device scope.
fn vulkan_model(p: &DeviceProfile) -> bool {
    p.fields().vulkan_memory_model
        && p.fields()
            .enabled
            .features
            .contains(&VkFeature::VulkanMemoryModelDeviceScope)
}

/// The coherence of `plan`'s ordinary accesses of shared memory (buffers, device addresses, the
/// threadgroup arena, and storage images): under the Vulkan model `NonPrivate*`, so barriers
/// order them with other invocations' accesses as under `GLSL450`, and inside an interlocked body
/// `Device`; plain under `GLSL450`, where an interlocked body's resources are `Coherent` instead
/// ([`declare`]).
pub fn coherence(plan: &Plan) -> Coherence {
    match plan.families.memory_model.model {
        BarrierModel::Vulkan if interlocked(&plan.families) => Coherence::Device,
        BarrierModel::Vulkan => Coherence::NonPrivate,
        BarrierModel::Glsl450 => Coherence::Plain,
    }
}

/// The function's body runs inside a fragment-interlock critical section.
fn interlocked(families: &FamilyPlans) -> bool {
    families.fragment.interlock.is_some()
}

/// Record that the function emits a `Device`-scope operation.
pub fn note_device_scope(cx: &mut SelectCx<'_>) {
    cx.families.memory_model.device_scope = true;
}

/// Whether `p` enables the features `plan` needs: `vulkanMemoryModel` under the Vulkan model,
/// and `vulkanMemoryModelDeviceScope` when a `Device`-scope operation is emitted under it, or when
/// coherent accesses that took the `GLSL450` form (on a profile short of either) meet something
/// else that needs the Vulkan model, which bans their `Coherent` decorations.
pub fn check(plan: &MemoryModelPlan, p: &DeviceProfile) -> Result<(), Error> {
    if plan.model != BarrierModel::Vulkan {
        return Ok(());
    }
    let needs = |f: VkFeature| -> Error {
        Unsupported::needs(Feature::CoherentMemory, Capability::Feature(f)).into()
    };
    if !p.fields().vulkan_memory_model {
        return Err(needs(VkFeature::VulkanMemoryModel));
    }
    let device_scope = p
        .fields()
        .enabled
        .features
        .contains(&VkFeature::VulkanMemoryModelDeviceScope);
    if (plan.device_scope || plan.coherent_resources) && !device_scope {
        return Err(needs(VkFeature::VulkanMemoryModelDeviceScope));
    }
    Ok(())
}

/// Complete the memory-model family's plan once every instruction and parameter is selected: the
/// function's device scope includes every written buffer slot (sub-word stores are device-scope
/// word atomics), and the plan's features are checked against the profile.
pub fn finish(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx
        .memory
        .slots
        .iter()
        .any(|s| s.window.is_some() && s.access.write)
    {
        note_device_scope(cx);
    }
    // Outside an interlocked body, storage-image accesses under the Vulkan model are not
    // implemented (module docs), nor beside coherent accesses in their `GLSL450` form, so that a
    // function's outcome does not hang on which form the profile takes. A heap storage class is
    // no per-stage descriptor, so it is asked of the texture family.
    let storage_images =
        cx.descriptors.storage_images > 0 || cx.families.texture.heap_storage_images();
    let model = &cx.families.memory_model;
    if (model.model == BarrierModel::Vulkan || model.coherent_resources)
        && storage_images
        && !interlocked(&cx.families)
    {
        return Err(not_implemented());
    }
    check(&cx.families.memory_model, cx.profile)
}

/// Complete an interlocked body's model once its slots are placed (`slots`): a body that reaches
/// memory through a device address (a slot lowered to its device address, or a wide pointer through
/// the root table) selects the Vulkan model, whose per-access operands make those accesses
/// coherent; `GLSL450` has no decoration for them.
pub fn finish_interlock(cx: &mut SelectCx<'_>, slots: &[Option<SlotChoice>]) -> Result<(), Error> {
    if !interlocked(&cx.families) || cx.families.memory_model.model == BarrierModel::Vulkan {
        return Ok(());
    }
    let addressed = cx.families.access.dynamic.used
        || slots.iter().flatten().any(|s| s.form == BufferForm::Psb);
    if !addressed {
        return Ok(());
    }
    require_vulkan(cx);
    check(&cx.families.memory_model, cx.profile)
}

fn not_implemented() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::MemoryModel)).into()
}

/// The memory-model family's module-scope objects (none: the serializer derives the
/// `OpMemoryModel` operand from the semantics the module contains).
#[derive(Clone, Debug, Default)]
pub struct Declared {}

/// Declare an interlocked body's resources, and those of a function whose coherent accesses took
/// the `GLSL450` form, `Coherent` under `GLSL450` (module docs).
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    _modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let model = &plan.families.memory_model;
    if (interlocked(&plan.families) || model.coherent_resources)
        && model.model == BarrierModel::Glsl450
    {
        mb.coherent_resources();
    }
    Ok(Declared::default())
}

fn fence_order(o: Ordering) -> Option<MemoryOrder> {
    match o {
        Ordering::Unordered | Ordering::Monotonic => None,
        Ordering::Acquire => Some(MemoryOrder::Acquire),
        Ordering::Release => Some(MemoryOrder::Release),
        Ordering::AcqRel | Ordering::SeqCst => Some(MemoryOrder::AcquireRelease),
    }
}

/// Select the strategy of instruction `inst` of this family.
pub fn select(
    cx: &mut SelectCx<'_>,
    _at: InstId,
    inst: &Inst,
) -> Result<MemoryModelStrategy, Error> {
    let InstKind::Fence(a) = &inst.kind else {
        return Err(
            Unsupported::not_implemented(Feature::Lowering(LowerFamily::MemoryModel)).into(),
        );
    };
    let Some(order) = fence_order(a.ordering) else {
        return Ok(MemoryModelStrategy::NoFence);
    };
    match a.scope {
        SyncScope::SingleThread => Ok(MemoryModelStrategy::NoFence),
        SyncScope::System => {
            note_device_scope(cx);
            let seq_cst = a.ordering == Ordering::SeqCst;
            if seq_cst {
                barrier::require_seq_cst(cx)?;
            }
            Ok(MemoryModelStrategy::Fence {
                seq_cst,
                flags: MemFlags {
                    device: true,
                    threadgroup: cx.stage.has_workgroup(),
                    texture: true,
                    threadgroup_imageblock: false,
                    object_data: false,
                },
                order,
            })
        }
    }
}

/// Emit strategy `s` of instruction `at` into the open block.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    _at: InstId,
    s: &MemoryModelStrategy,
) -> BResult<()> {
    match s {
        MemoryModelStrategy::Fence {
            flags,
            order,
            seq_cst,
        } => {
            let sem: Semantics = barrier::semantics(*flags, *order, barrier::memory_model(cx.plan));
            fb.memory_barrier(Scope::Device, sem)?;
            match seq_cst {
                true => barrier::seq_cst_order(cx, fb, Scope::Device, *flags),
                false => Ok(()),
            }
        }
        MemoryModelStrategy::NoFence => Ok(()),
    }
}

#[cfg(test)]
mod tests;
