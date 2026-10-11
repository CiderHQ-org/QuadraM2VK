//! A command buffer's execution record: what each of its dispatches and draws executed, for
//! validation's executor, which runs authored cases through the runtime and records what ran.
//!
//! Hidden, and minimal: exactly what that executor reads. Each dispatch or draw an encoder took
//! (it passed every check, so its commands are in the command buffer) adds one [`Execution`]:
//! the compiled pipeline its plan settled on (the stage modules the census reads, the content the
//! artifact digest covers, the SPIR-V the row hashes), the bind shape it was planned for (the
//! artifact digest plans the pipeline for it again), how long the driver took to create each
//! `VkPipeline` the command records (the pipeline-creation ceiling), and how many index positions
//! of a direct indexed draw lie past where Metal's fetch stops reading its index buffer, which it
//! read from a zero-padded copy (`draw_source.rs`; the executor holds them to the case's
//! containment). A dispatch of no threads and
//! a draw of no vertices record no Vulkan command, but they settled a pipeline, so they are
//! executions too: a dispatch's list of created pipelines is then empty. The record holds no
//! Vulkan object, and stays with the command buffer after it completed.

use metal2vulkan::bind::BindShape;
use metal2vulkan::pipeline::{ComputePipeline, RenderPipeline};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// The compiled pipeline one dispatch or draw ran.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub enum ExecutedPipeline {
    Compute(Arc<ComputePipeline>),
    Render(Arc<RenderPipeline>),
}

/// One dispatch or draw of a command buffer (the module's record).
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Execution {
    pipeline: ExecutedPipeline,
    shape: Arc<BindShape>,
    /// The creation times, which the record's clones share: set with the command, or once the
    /// `VkPipeline` it records is made, after the command (`render/pending.rs`).
    creation: Arc<OnceLock<Vec<Duration>>>,
    index_padding: Option<u64>,
}

impl Execution {
    /// A dispatch or draw of `pipeline` planned for `shape`, recording the `VkPipeline`s whose
    /// creation took `creation`, in the command's order.
    pub(super) fn new(
        pipeline: ExecutedPipeline,
        shape: Arc<BindShape>,
        creation: Vec<Duration>,
    ) -> Execution {
        Execution {
            pipeline,
            shape,
            creation: Arc::new(OnceLock::from(creation)),
            index_padding: None,
        }
    }

    /// A draw of `pipeline` planned for `shape` whose `VkPipeline` is still being made: the
    /// making sets `creation`, its creation's and those of the other `VkPipeline`s the draw
    /// records in the draw's order, once it is made (`render/pending.rs`).
    pub(super) fn made_later(
        pipeline: ExecutedPipeline,
        shape: Arc<BindShape>,
        creation: Arc<OnceLock<Vec<Duration>>>,
    ) -> Execution {
        Execution {
            pipeline,
            shape,
            creation,
            index_padding: None,
        }
    }

    /// The execution, a direct indexed draw that read `padding` index positions past where
    /// Metal's fetch stops reading its index buffer from a zero-padded copy.
    pub(super) fn with_index_padding(self, padding: Option<u64>) -> Execution {
        Execution {
            index_padding: padding,
            ..self
        }
    }

    /// The compiled pipeline the command's plan settled on.
    pub fn pipeline(&self) -> &ExecutedPipeline {
        &self.pipeline
    }

    /// The bind shape the command was planned for.
    pub fn shape(&self) -> &BindShape {
        &self.shape
    }

    /// How long the driver took to create each `VkPipeline` the command records, in its order
    /// (a memoized one's creation, when another command made it); empty for a command that
    /// records none. A draw whose `VkPipeline` the device's pool makes after the draw has its times
    /// once it is made, by its command buffer's completion at the latest, in every copy of the
    /// record taken before too; empty until then, and for a making that failed.
    pub fn pipeline_creation(&self) -> &[Duration] {
        self.creation.get().map_or(&[], Vec::as_slice)
    }

    /// How many index positions of a direct indexed draw lie past where Metal's fetch stops
    /// reading its index buffer (its length rounded up to 4 bytes), which the draw read from a
    /// zero-padded copy, each as index 0 (`draw_source.rs`): `None` for a draw with none, and for
    /// every other command.
    pub fn index_padding(&self) -> Option<u64> {
        self.index_padding
    }
}
