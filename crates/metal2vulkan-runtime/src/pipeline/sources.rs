//! Where a pipeline state's compiles look before compiling: the device's memo, then the creation's
//! binary archives in order, then the device's shader cache on disk. What a compile makes is kept
//! in the memo and on disk; what an archive or the disk gives, in the memo. A creation for a binary
//! archive's add also notes every artifact it used.

use super::memo::Remembered;
use super::render::RenderVariant;
use super::{Kernel, PipelineOptions, Variant};
use crate::binary_archive::{Artifact, BinaryArchive, Used};
use crate::Device;
use metal2vulkan::cache::PipelineKey;
use metal2vulkan::library::{CompiledFunction, FunctionCompileKey};
use metal2vulkan::pipeline::{
    ComputePipeline, FunctionDescriptor, LoweringInputs, RenderPipeline, StageEntries,
};
use metal2vulkan::Budget;
use std::sync::Arc;

/// Where one creation's (or one later variant's) compiles look, and its budget.
pub(crate) struct Sources<'a> {
    device: &'a Device,
    /// The creation's options: its archives and budget; `None` for a variant a dispatch or
    /// draw needs after creation, which no archive holds and which compiles under the standard
    /// budget.
    options: Option<&'a PipelineOptions>,
    /// What an add's creation used.
    used: Option<&'a Used>,
}

impl<'a> Sources<'a> {
    /// The sources of a creation under `options`, noting what it uses in `used`.
    pub(crate) fn creation(
        device: &'a Device,
        options: &'a PipelineOptions,
        used: Option<&'a Used>,
    ) -> Sources<'a> {
        Sources {
            device,
            options: Some(options),
            used,
        }
    }

    /// The sources of a variant a dispatch or draw needs after creation.
    pub(crate) fn later(device: &'a Device) -> Sources<'a> {
        Sources {
            device,
            options: None,
            used: None,
        }
    }

    pub(crate) fn device(&self) -> &'a Device {
        self.device
    }

    /// The budget of one compile attempt starting now.
    pub(crate) fn budget(&self) -> Budget {
        match self.options {
            Some(options) => options.budget(),
            None => Budget::standard(),
        }
    }

    fn archives(&self) -> &'a [BinaryArchive] {
        self.options
            .map_or(&[], |options| options.binary_archives.as_slice())
    }

    fn note(&self, artifact: Artifact) {
        if let Some(used) = self.used {
            used.borrow_mut().push(artifact);
        }
    }

    /// `function` compiled under `inputs`, whose key is `key`: found, else compiled under
    /// `budget`, reading its specialized entry from `entries` where given (A render creation's,
    /// which its checks and key filled). A refusal is the compile layer's.
    pub(crate) fn function(
        &self,
        function: &FunctionDescriptor,
        inputs: &LoweringInputs,
        (key, entries): (&FunctionCompileKey, Option<&StageEntries>),
        budget: &Budget,
    ) -> Result<CompiledFunction, metal2vulkan::error::Error> {
        let compile = self.device.compile_device();
        let cache = &self.device.shared().shader_cache;
        let compiled = match cache.memo.function(key) {
            Some(compiled) => compiled,
            None => {
                let found = self
                    .archives()
                    .iter()
                    .find_map(|archive| archive.function(compile, key))
                    .or_else(|| cache.load_function(compile, key));
                let (compiled, bytes) = match found {
                    Some(found) => found,
                    None => {
                        let compiled = match entries {
                            Some(entries) => compile
                                .compile_function_from_entries(function, inputs, budget, entries)?,
                            None => compile.compile_function(function, inputs, budget)?,
                        };
                        let bytes = cache.store_function(compile, &compiled);
                        (compiled, bytes)
                    }
                };
                cache.memo.keep_function(&compiled, bytes);
                compiled
            }
        };
        self.note(Artifact::Function(compiled.clone()));
        Ok(compiled)
    }

    /// The compute pipeline of `descriptor`, a compute or tile descriptor whose kernel or tile
    /// function is compiled: the variant of it the memo knows live, or the pipeline found or
    /// linked now under `budget` (`compile_compute_pipeline` or `compile_tile_pipeline`, each
    /// under its own key) and held to the device's compiler word bound from `functions`, the same
    /// descriptor holding the function, compiled under `inputs` ([`Kernel::fit`]). A refusal is
    /// the compile layer's.
    pub(crate) fn kernel(
        &self,
        descriptor: &Kernel,
        (functions, inputs): (&Kernel, &LoweringInputs),
        budget: &Budget,
    ) -> Result<Remembered<ComputePipeline, Variant>, metal2vulkan::error::Error> {
        let compile = self.device.compile_device();
        let cache = &self.device.shared().shader_cache;
        let key = match descriptor {
            Kernel::Compute(descriptor) => compile.compute_pipeline_key(descriptor),
            Kernel::Tile(descriptor) => compile.tile_pipeline_key(descriptor),
        };
        let found = match cache.memo.compute(&key) {
            Some(found) => found,
            None => {
                let found = self
                    .archives()
                    .iter()
                    .find_map(|archive| archive.compute_pipeline(compile, &key))
                    .or_else(|| cache.load_compute_pipeline(compile, &key));
                let (pipeline, bytes) = match found {
                    Some(found) => found,
                    None => {
                        let pipeline = match descriptor {
                            Kernel::Compute(descriptor) => {
                                compile.compile_compute_pipeline(descriptor, budget)?
                            }
                            Kernel::Tile(descriptor) => {
                                compile.compile_tile_pipeline(descriptor, budget)?
                            }
                        };
                        // Held to the driver's compiler word bound, translated again from the
                        // function where its module passes it; it keeps its key, so the cache
                        // stores and finds it as compiled.
                        let pipeline = functions.fit(compile, inputs, budget, pipeline)?;
                        let bytes = cache.store_compute_pipeline(compile, &pipeline);
                        (pipeline, bytes)
                    }
                };
                let pipeline = Arc::new(pipeline);
                cache.memo.keep_compute(&pipeline, bytes);
                Remembered::Compiled(pipeline)
            }
        };
        self.note(Artifact::Compute(match &found {
            Remembered::Live(variant) => Arc::clone(&variant.pipeline),
            Remembered::Compiled(pipeline) => Arc::clone(pipeline),
        }));
        Ok(found)
    }

    /// The render pipeline of `key`, found: the variant of it the memo knows live, or the
    /// pipeline an archive or the disk cache holds.
    pub(crate) fn render(
        &self,
        key: &PipelineKey,
    ) -> Option<Remembered<RenderPipeline, RenderVariant>> {
        let compile = self.device.compile_device();
        let cache = &self.device.shared().shader_cache;
        let found = match cache.memo.render(key) {
            Some(found) => found,
            None => {
                let (pipeline, bytes) = self
                    .archives()
                    .iter()
                    .find_map(|archive| archive.render_pipeline(compile, key))
                    .or_else(|| cache.load_render_pipeline(compile, key))?;
                let pipeline = Arc::new(pipeline);
                cache.memo.keep_render(&pipeline, bytes);
                Remembered::Compiled(pipeline)
            }
        };
        self.note(Artifact::Render(match &found {
            Remembered::Live(variant) => Arc::clone(variant.shared_pipeline()),
            Remembered::Compiled(pipeline) => Arc::clone(pipeline),
        }));
        Some(found)
    }

    /// Keep `pipeline`, linked now, in the memo and on disk.
    pub(crate) fn linked_render(&self, pipeline: RenderPipeline) -> Arc<RenderPipeline> {
        let compile = self.device.compile_device();
        let cache = &self.device.shared().shader_cache;
        let pipeline = Arc::new(pipeline);
        let bytes = cache.store_render_pipeline(compile, &pipeline);
        cache.memo.keep_render(&pipeline, bytes);
        self.note(Artifact::Render(Arc::clone(&pipeline)));
        pipeline
    }
}
