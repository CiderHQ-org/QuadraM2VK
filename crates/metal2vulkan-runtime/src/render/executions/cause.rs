//! Why a pass whose executions' records the GPU may write takes the host step rather than the
//! GPU form (`gpu.rs`): the first rule that sent it there, which its notice names
//! (`Degradation::HostReplayedIndirectCommands`, [`DeferredPass::degrade`](super::DeferredPass::degrade)).

use std::fmt;

use metal2vulkan::pipeline::CommandWrites;

/// The rule that sent a pass to the host step (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HostStepCause {
    /// A rule of the call, the buffer, the pass or the state, as the notice words it.
    Rule(&'static str),
    /// The caller's check looks for what the records may hold: draws of points (`points`), and
    /// commands under a state that may encode indirect commands (`encodes`), the needs the
    /// buffer's records did not rule out.
    Check { points: bool, encodes: bool },
    /// The records may hold operations past draws, buffers, resets and cleared barriers: the
    /// `CommandWrites` operation bits past those.
    Operations(u32),
    /// The records' draws may pass more than one primitive type (one the writes do not state
    /// counts as all five): the `CommandWrites` primitive-type bits.
    PrimitiveTypes(u8),
    /// The inherited state's replay variant, or the call's check of it, was refused so.
    Refused(String),
    /// What the records may hold grew after the commit's decision, before the pass's batch was
    /// ready (`DeferredPass::outgrown`).
    Outgrown,
}

impl fmt::Display for HostStepCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostStepCause::Rule(rule) => f.write_str(rule),
            HostStepCause::Check { points, encodes } => {
                let needs = match (points, encodes) {
                    (true, true) => {
                        "draws of points and commands under a state that may encode \
                                     indirect commands"
                    }
                    (true, false) => "draws of points",
                    _ => "commands under a state that may encode indirect commands",
                };
                write!(
                    f,
                    "the execution's check looks for {needs}, which the buffer's records may hold"
                )
            }
            HostStepCause::Operations(bits) => {
                f.write_str("the buffer's records may hold ")?;
                let names = OPERATIONS
                    .iter()
                    .filter(|(bit, _)| bits & bit != 0)
                    .map(|(_, name)| *name);
                for (at, name) in names.enumerate() {
                    if at > 0 {
                        f.write_str(", ")?;
                    }
                    f.write_str(name)?;
                }
                Ok(())
            }
            HostStepCause::PrimitiveTypes(bits) => write!(
                f,
                "the buffer's draws may pass more than one primitive type (MTLPrimitiveType bits \
                 {bits:#07b})"
            ),
            HostStepCause::Refused(error) => {
                write!(f, "the inherited state's replay was refused: {error}")
            }
            HostStepCause::Outgrown => f.write_str(
                "what the buffer's records may hold grew after the commit, before the pass's \
                 batch was ready (a CPU setter, or a route encoded since)",
            ),
        }
    }
}

/// The operations past a GPU form's, by their `CommandWrites` bit.
const OPERATIONS: [(u32, &str); 11] = [
    (CommandWrites::DRAW_INDEXED, "indexed draws"),
    (CommandWrites::DRAW_PATCHES, "draws of patches"),
    (CommandWrites::DRAW_MESH, "mesh draws"),
    (CommandWrites::DISPATCH, "dispatches"),
    (CommandWrites::PIPELINE, "pipeline states"),
    (CommandWrites::BARRIER, "barriers"),
    (CommandWrites::RENDER_STATE, "render state setters"),
    (
        CommandWrites::THREADGROUP_MEMORY,
        "threadgroup memory lengths",
    ),
    (CommandWrites::STAGE_IN_REGION, "stage-in regions"),
    (CommandWrites::IMAGEBLOCK_SIZE, "imageblock sizes"),
    (CommandWrites::COPY, "copied commands"),
];
