//! Metal's refusal of a pipeline state that supports indirect command buffers whose functions use
//! a direct texture or sampler argument.
//!
//! - **What Metal refuses** (on an M3 Max under macOS 26.7 and an M4 under 27.0.1, alike but for
//!   the error's domain). `newRenderPipelineStateWithDescriptor:`,
//!   `newRenderPipelineStateWithMeshDescriptor:` and `newComputePipelineStateWithDescriptor:`
//!   with `supportIndirectCommandBuffers` answer nil with an `NSError` where a function uses a
//!   `[[texture(n)]]` or `[[sampler(n)]]` argument: a read, a sample, a write, a texture array
//!   alike. Unused ones, an argument buffer's textures and samplers, and constexpr samplers are
//!   made. The state is never made, so no indirect command can name it, and an encoder's state
//!   an indirect command buffer inherits must support indirect command buffers too (Metal's GPU
//!   faults otherwise, `render/executions.rs`), so no indirect command reads a direct texture.
//! - **The answer.** [`Error::MetalError`] with [`Problem::IndirectCommandsDirectArgument`]
//!   naming the stage Metal names, and Metal's error: code 3, the stage's description, and
//!   `userInfo` holding only `NSLocalizedDescription`. A render state names its fragment function
//!   before its vertex function; a mesh state its fragment function, then its mesh function, then
//!   its object function ([`RENDER`]).
//! - **Once linked** (A vertex function's texture that reaches no fragment input was refused where
//!   Metal makes the state). Which stages use one is the compile layer's answer
//!   (`RenderPipelineDescriptor::stages_using_direct_arguments`, its mesh form): an object function
//!   where its reflection lists a used texture or sampler; a vertex, mesh or fragment function
//!   where an instruction Metal's linking keeps reads it, so a texel that reaches only a varying no
//!   fragment input reads, or a fragment function's color the pipeline has no attachment for or
//!   writes no channel of, is dropped, as Metal drops it (on an M4 under macOS 27.0.1), though
//!   Metal's reflection still lists the texture as used.
//! - **Decided from roles** (From the whole reflection, so an argument whose reflection Metal's
//!   answer has not been measured for, such as a threadgroup `uint`, refused an ordinary state).
//!   Whether a function uses a direct texture or sampler is decided from its arguments' roles
//!   (`[[texture(n)]]`, `[[sampler(n)]]`) and use alone
//!   (`ComputePipelineDescriptor::uses_direct_arguments`, and the render and mesh forms above),
//!   never from the types the reflection derives.
//! - **Undecided** (Such a state was made). Where the compile layer refuses that answer (the
//!   decode's and specialization's own refusals, a stage it already compiled that kept no
//!   reflection), whether Metal refuses the state is not decided, and the answer is
//!   [`Error::Unproven`] with [`Problem::IndirectCommandsDirectArgumentUndecided`] carrying the
//!   refusal: the runtime makes no state Metal may have refused. Without
//!   `supportIndirectCommandBuffers` nothing is asked, and the state is made.
//! - **Proven or not**. A vertex or mesh function's use only a rule kept beyond Metal keeps, where
//!   Metal's answer is unmeasured (`DirectArgumentUse::gap`), is not Metal's refusal: the first
//!   stage in Metal's order that uses one decides, and where its use is unproven the answer is
//!   [`Error::Unproven`] with [`Problem::IndirectCommandsDirectArgumentUnproven`] naming the stage
//!   and the rule, not Metal's error (the stage Metal would name is not proven either).
//! - **The error's domain is the GPU driver's class**, so it follows the device: Metal's on the
//!   M3 Max is `AGXMetalG15X_M1`, on the M4 `AGXMetalG16G_B0`. The runtime answers the M3 Max's
//!   ([`DOMAIN`]), the closest measured to the consumer's guest (macOS 26.6.2); a consumer may
//!   substitute its own device's.
//! - **Tile states never support indirect command buffers**: Metal's tile descriptor has no
//!   `supportIndirectCommandBuffers` (both devices).

use crate::error::{Error, MetalError, Problem, Unproven};
use metal2vulkan::error::Error as CompileError;
use metal2vulkan::pipeline::Stage;
use metal2vulkan::reflection::DirectArgumentUse;

/// The domain of Metal's error: its GPU driver's class on the M3 Max under macOS 26.7 (the M4's
/// under 27.0.1 is `AGXMetalG16G_B0`; the module docs).
pub(super) const DOMAIN: &str = "AGXMetalG15X_M1";

/// The code of Metal's error, on both devices measured.
pub(super) const CODE: i64 = 3;

/// A render or mesh state's stages in the order Metal names them, the first that uses a direct
/// argument named (the fragment function before the vertex function; before the mesh function,
/// which goes before the object function), each with Metal's `localizedDescription` for it.
const RENDER: [(Stage, &str); 4] = [
    (
        Stage::Fragment,
        "Fragment shader cannot be used with indirect command buffers",
    ),
    (
        Stage::Vertex,
        "Vertex shader cannot be used with indirect command buffers",
    ),
    (
        Stage::Mesh,
        "Mesh shader cannot be used with indirect command buffers",
    ),
    (
        Stage::Object,
        "Object shader cannot be used with indirect command buffers",
    ),
];

/// Metal's `localizedDescription` for a compute state's kernel.
const COMPUTE: &str = "Compute function cannot be used with indirect command buffers";

/// Metal's refusal in `call` of a function of `stage` that uses a direct argument, `description`
/// Metal's for it.
fn refusal(call: &'static str, stage: Stage, description: &'static str) -> Error {
    Error::MetalError(MetalError::new(
        call,
        Problem::IndirectCommandsDirectArgument { stage },
        DOMAIN,
        CODE,
        description,
    ))
}

/// [`Error::Unproven`] in `call`: the compile layer's `refusal` to say whether the state's
/// functions use a direct texture or sampler argument (the module docs).
fn undecided(call: &'static str, refusal: CompileError) -> Error {
    Error::Unproven(Unproven::new(
        call,
        Problem::IndirectCommandsDirectArgumentUndecided {
            refusal: Box::new(refusal),
        },
    ))
}

/// Refused in `call` (the module docs): a render or mesh state, of a descriptor with
/// `supportIndirectCommandBuffers`, whose functions use a direct texture or sampler once linked
/// (`uses`), the first Metal names named: Metal's refusal where its use is proven, else
/// [`Error::Unproven`]. `uses` is the descriptor's answer under the creation's budget; a refused
/// one is [`Problem::IndirectCommandsDirectArgumentUndecided`] (It decided nothing).
pub(crate) fn check_render(
    call: &'static str,
    uses: Result<Vec<DirectArgumentUse>, CompileError>,
) -> Result<(), Error> {
    let uses = uses.map_err(|refusal| undecided(call, refusal))?;
    let first = RENDER.into_iter().find_map(|(stage, description)| {
        uses.iter()
            .find(|used| used.stage == stage)
            .map(|used| (used, description))
    });
    match first {
        Some((used, description)) => Err(match used.gap {
            None => refusal(call, used.stage, description),
            Some(reason) => Error::Unproven(Unproven::new(
                call,
                Problem::IndirectCommandsDirectArgumentUnproven {
                    stage: used.stage,
                    reason,
                },
            )),
        }),
        None => Ok(()),
    }
}

/// Refused in `call` (the module docs): a compute state (of a descriptor with
/// `supportIndirectCommandBuffers`) whose kernel `uses` a direct texture or sampler, as
/// `ComputePipelineDescriptor::uses_direct_arguments` answers under the creation's budget
/// (From the arguments' roles, not the whole reflection). A refused answer is
/// [`Problem::IndirectCommandsDirectArgumentUndecided`] (It decided nothing).
pub(crate) fn check_compute(
    call: &'static str,
    uses: Result<bool, CompileError>,
) -> Result<(), Error> {
    match uses.map_err(|refusal| undecided(call, refusal))? {
        true => Err(refusal(call, Stage::Kernel, COMPUTE)),
        false => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::error::{Feature, Unsupported};

    /// A refused answer is `Unproven`, carrying the refusal, for a render and a compute state
    /// alike: the state is not made.
    #[test]
    fn a_refused_answer_is_undecided_and_makes_no_state() {
        let refusal =
            || CompileError::Unsupported(Unsupported::not_implemented(Feature::Reflection));
        let undecided = |call| {
            Err(Error::Unproven(Unproven::new(
                call,
                Problem::IndirectCommandsDirectArgumentUndecided {
                    refusal: Box::new(refusal()),
                },
            )))
        };
        assert_eq!(
            check_render("new_render_pipeline_state", Err(refusal())),
            undecided("new_render_pipeline_state")
        );
        assert_eq!(
            check_compute("new_compute_pipeline_state", Err(refusal())),
            undecided("new_compute_pipeline_state")
        );
        assert_eq!(
            check_compute("new_compute_pipeline_state", Ok(false)),
            Ok(())
        );
        assert_eq!(
            check_render("new_render_pipeline_state", Ok(Vec::new())),
            Ok(())
        );
    }
}
