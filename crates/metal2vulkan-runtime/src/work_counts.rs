//! Counts of work a device's commit path does that the lanes hold to bounds: each counts a piece of
//! work the runtime once repeated at every commit or every change, so a lane fails when it is
//! repeated again; and the textures that took their storage form, which a lane holds to once each;
//! and the buffers made with the device-address usage, which every encoder that reads pointers made
//! resident one by one on MoltenVK. Not part of the API.

use super::degradation::Degradation;
use super::device::lock;
use super::Device;
use ash::vk;
use std::sync::atomic::{AtomicU64, Ordering};

/// A device's counters (`DeviceShared::counts`), each only ever added to.
#[derive(Default)]
pub(crate) struct Counters {
    command_pools_made: AtomicU64,
    live_tables_uploaded: AtomicU64,
    timeline_waits: AtomicU64,
    timestamp_slots_taken: AtomicU64,
    storage_forms_taken: AtomicU64,
    addressable_buffers_made: AtomicU64,
    mipmap_first_levels_fetched: AtomicU64,
    pipelines_pooled: AtomicU64,
    shader_modules_made: AtomicU64,
    shader_modules_shared: AtomicU64,
    variant_pipelines_made: AtomicU64,
    variants_taken_back: AtomicU64,
    unlisted_argument_sampler_failures: AtomicU64,
    host_step_sampler_sets_made: AtomicU64,
    creation_pipelines_made: AtomicU64,
    creation_pipelines_drawn: AtomicU64,
    sampler_sets_made: AtomicU64,
    sampler_sets_shared: AtomicU64,
    sampler_sets_destroyed: AtomicU64,
    ahead_render_passes_made: AtomicU64,
    ahead_framebuffers_made: AtomicU64,
    kept_backings_made: AtomicU64,
    ahead_nanos: AtomicU64,
    unused_bits_load_passes: AtomicU64,
    unused_bits_load_texels: AtomicU64,
    capture_reach_passes: AtomicU64,
    capture_reach_primitives: AtomicU64,
    capture_reach_tiles_found: AtomicU64,
    capture_reach_tiles_bound: AtomicU64,
    capture_reach_bound_passes: AtomicU64,
    capture_reach_narrower_passes: AtomicU64,
    driver_diagnostics_withheld: AtomicU64,
    stand_ins_made: AtomicU64,
    stand_in_bytes: AtomicU64,
    stale_detached_reaches: AtomicU64,
    detached_storage_reaches: AtomicU64,
    heap_textures_tiled: AtomicU64,
    heap_texture_lossless_layouts: AtomicU64,
    heap_texture_block_compressed_layouts: AtomicU64,
    heap_texture_untiled_layouts: AtomicU64,
    fills_recorded: AtomicU64,
    fills_by_transfers: AtomicU64,
    attachment_clears_recorded: AtomicU64,
    image_blits_recorded: AtomicU64,
    query_result_copies_recorded: AtomicU64,
    visibility_queries_begun: AtomicU64,
    indirect_draws_recorded: AtomicU64,
    internal_submissions: AtomicU64,
    images_started_on_the_host: AtomicU64,
    host_texel_copies: AtomicU64,
    staging_submissions: AtomicU64,
    uncached_host_images: AtomicU64,
    cpu_block_decodes: AtomicU64,
    block_decode_passes: AtomicU64,
    queue_drains: AtomicU64,
    stand_in_copies_made: AtomicU64,
    stand_in_copies_made_at_draws: AtomicU64,
    draw_shapes_made: AtomicU64,
    draws_settled: AtomicU64,
    bound_notices_made: AtomicU64,
    argument_tables_retained: AtomicU64,
    graphics_pipelines_looked_up: AtomicU64,
    linear_passes_in_place: AtomicU64,
    linear_passes_through_copies: AtomicU64,
    draw_states_recorded: AtomicU64,
    host_deallocators_delayed: AtomicU64,
    replay_logs_kept: AtomicU64,
    replay_log_bytes: AtomicU64,
    encoders_reencoded: AtomicU64,
    reencodes_refused: AtomicU64,
    pipeline_libraries_made: AtomicU64,
    fragment_shader_libraries_made: AtomicU64,
    pipelines_linked: AtomicU64,
    dispatch_pipelines_pooled: AtomicU64,
}

/// A command the runtime records that MoltenVK 1.4.1 encodes on Apple's GPUs through a pipeline it
/// compiles per command pool, inside `vkQueueSubmit`, at the pool's first such command
/// (`MVKCommandEncodingPool`): counted, so a run shows which of them it records.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PoolCompiled {
    /// `vkCmdClearAttachments`: a render pipeline per pool and attachment formats.
    AttachmentClear,
    /// `vkCmdBlitImage`: a render pipeline per pool, format and filter where the blit scales.
    ImageBlit,
    /// `vkCmdCopyQueryPoolResults`: a compute pipeline per pool.
    QueryResultCopy,
    /// A visibility result mode set, beginning an occlusion query: MoltenVK accumulates the
    /// queries of each Metal render pass that ran one by a compute pipeline per pool.
    VisibilityQuery,
    /// `vkCmdDrawIndirect` and `vkCmdDrawIndexedIndirect`: a compute conversion per pool where the
    /// pipeline tessellates or the pass is multiview.
    IndirectDraw,
}

impl Counters {
    /// `count` commands of `kind` were recorded (`PoolCompiled`).
    pub(crate) fn pool_compiled(&self, kind: PoolCompiled, count: u64) {
        let counter = match kind {
            PoolCompiled::AttachmentClear => &self.attachment_clears_recorded,
            PoolCompiled::ImageBlit => &self.image_blits_recorded,
            PoolCompiled::QueryResultCopy => &self.query_result_copies_recorded,
            PoolCompiled::VisibilityQuery => &self.visibility_queries_begun,
            PoolCompiled::IndirectDraw => &self.indirect_draws_recorded,
        };
        counter.fetch_add(count, Ordering::Relaxed);
    }

    /// A fill was recorded (`word_fill.rs`): as transfers where `by_transfers`, else as
    /// `vkCmdFillBuffer`.
    pub(crate) fn fill_recorded(&self, by_transfers: bool) {
        let counter = match by_transfers {
            true => &self.fills_by_transfers,
            false => &self.fills_recorded,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A `VkCommandPool` was made for a command buffer, none of the device's spares taken.
    pub(crate) fn command_pool_made(&self) {
        self.command_pools_made.fetch_add(1, Ordering::Relaxed);
    }

    /// The live buffers' table was uploaded to a buffer made for it, no spare reused.
    pub(crate) fn live_table_uploaded(&self) {
        self.live_tables_uploaded.fetch_add(1, Ordering::Relaxed);
    }

    /// The completion thread waited for a timeline's value, not at a fence
    /// (`command_buffer/completion.rs`).
    pub(crate) fn timeline_wait(&self) {
        self.timeline_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A committed command buffer took a slot of one of the device's timestamp pools, to write
    /// its GPU start and end into (`gpu_time.rs`).
    pub(crate) fn timestamp_slot_taken(&self) {
        self.timestamp_slots_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture took its storage form (`texture/storage_form.rs`).
    pub(crate) fn storage_form_taken(&self) {
        self.storage_forms_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// A `VkBuffer` of `usage` was made: counted where the usage names device addresses.
    pub(crate) fn buffer_made(&self, usage: vk::BufferUsageFlags) {
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            self.addressable_buffers_made
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A `generate_mipmaps` read its texture's first level in place, through a sampled view,
    /// rather than copying it in (`mipmap_filter.rs`).
    pub(crate) fn mipmap_first_level_fetched(&self) {
        self.mipmap_first_levels_fetched
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw handed the making of its `VkPipeline` to the device's pool and went on
    /// (`render/pending.rs`).
    pub(crate) fn pipeline_pooled(&self) {
        self.pipelines_pooled.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline state's variant made a `VkShaderModule` (`pipeline/objects.rs`).
    pub(crate) fn shader_module_made(&self) {
        self.shader_modules_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline state's variant took the device's `VkShaderModule` of its stage's SPIR-V
    /// (`pipeline/objects.rs`).
    pub(crate) fn shader_module_shared(&self) {
        self.shader_modules_shared.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline state's variant made a `VkPipeline` (`pipeline/objects.rs`,
    /// `pipeline/graphics.rs`).
    pub(crate) fn variant_pipeline_made(&self) {
        self.variant_pipelines_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A graphics pipeline library of `part` was made (`pipeline/libraries.rs`).
    pub(crate) fn pipeline_library_made(&self, part: crate::pipeline::libraries::Part) {
        self.pipeline_libraries_made.fetch_add(1, Ordering::Relaxed);
        if part == crate::pipeline::libraries::Part::FragmentShader {
            self.fragment_shader_libraries_made
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A dispatch handed the making of a region's `VkPipeline` to the device's pool and went on
    /// (`pipeline/dispatch_making.rs`).
    pub(crate) fn dispatch_pipeline_pooled(&self) {
        self.dispatch_pipelines_pooled
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A render variant's `VkPipeline` was linked from libraries (`pipeline/graphics.rs`).
    pub(crate) fn pipeline_linked(&self) {
        self.pipelines_linked.fetch_add(1, Ordering::Relaxed);
    }

    /// A variant made again took back a released one's objects (`pipeline/kept.rs`).
    pub(crate) fn variant_taken_back(&self) {
        self.variants_taken_back.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer committed whose commands' plans could not list the sampler states live
    /// then (`sampler_listing.rs`), or whose host step's command left a live state out: it fails
    /// `CommandBufferError::UnlistedArgumentSampler`.
    pub(crate) fn unlisted_argument_sampler_failure(&self) {
        self.unlisted_argument_sampler_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A host step's variant made immutable samplers no live set had (`pipeline/objects.rs`).
    pub(crate) fn host_step_sampler_set_made(&self) {
        self.host_step_sampler_sets_made
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A state's creation made a `VkPipeline` (`pipeline/predicted.rs`).
    pub(crate) fn creation_pipeline_made(&self) {
        self.creation_pipelines_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw took a `VkPipeline` a state's creation made, its first draw.
    pub(crate) fn creation_pipeline_drawn(&self) {
        self.creation_pipelines_drawn
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A variant made an immutable sampler set: the device's map's, or a host step's or a pool
    /// piece's own (`pipeline/objects.rs`).
    pub(crate) fn sampler_set_made(&self) {
        self.sampler_sets_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A variant took the device's set of its immutable samplers, made before.
    pub(crate) fn sampler_set_shared(&self) {
        self.sampler_sets_shared.fetch_add(1, Ordering::Relaxed);
    }

    /// The device's map destroyed a set of immutable samplers: at a settle, a sweep or the
    /// teardown (`pipeline/objects.rs`).
    pub(crate) fn sampler_set_destroyed(&self) {
        self.sampler_sets_destroyed.fetch_add(1, Ordering::Relaxed);
    }

    /// A commit made ahead, for its render pass's host step, `passes` render passes and
    /// `framebuffers` framebuffers, taking `nanos` nanoseconds (`render/executions/ahead.rs`).
    pub(crate) fn made_ahead(&self, passes: u64, framebuffers: u64, nanos: u64) {
        self.ahead_render_passes_made
            .fetch_add(passes, Ordering::Relaxed);
        self.ahead_framebuffers_made
            .fetch_add(framebuffers, Ordering::Relaxed);
        self.ahead_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// A memoryless attachment's texture made a backing to keep (`attachment_stand_in.rs`).
    pub(crate) fn kept_backing_made(&self) {
        self.kept_backings_made.fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder recorded the load pass of an extended-range or padded color attachment over a
    /// region of `texels` texels (`render/unused_bits_load.rs`).
    pub(crate) fn unused_bits_load_recorded(&self, texels: u64) {
        self.unused_bits_load_passes.fetch_add(1, Ordering::Relaxed);
        self.unused_bits_load_texels
            .fetch_add(texels, Ordering::Relaxed);
    }

    /// An encoder recorded a pass's capture replay over `primitives` primitives
    /// (`render/capture.rs`).
    pub(crate) fn capture_reach_recorded(&self, primitives: u64) {
        self.capture_reach_passes.fetch_add(1, Ordering::Relaxed);
        self.capture_reach_primitives
            .fetch_add(primitives, Ordering::Relaxed);
    }

    /// A captured pass's command buffer completed: its primitives reach `found` tiles, of the
    /// `bound` its draws' viewports and scissors reach.
    pub(crate) fn capture_reach_tallied(&self, found: u64, bound: u64) {
        self.capture_reach_tiles_found
            .fetch_add(found, Ordering::Relaxed);
        self.capture_reach_tiles_bound
            .fetch_add(bound, Ordering::Relaxed);
        match found >= bound {
            true => &self.capture_reach_bound_passes,
            false => &self.capture_reach_narrower_passes,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    /// A driver call reported `withheld` diagnostics its driver facts measure it to report, which
    /// the device's sink was not handed (`driver_facts.rs`).
    pub(crate) fn driver_diagnostics_withheld(&self, withheld: u64) {
        self.driver_diagnostics_withheld
            .fetch_add(withheld, Ordering::Relaxed);
    }

    /// An internal submission was made (`device/submit.rs`): one a host call waits for.
    pub(crate) fn internal_submission(&self) {
        self.internal_submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission on the staging queue was made (`device/staging.rs`).
    pub(crate) fn staging_submission(&self) {
        self.staging_submissions.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's host-transfer image was placed in memory the host does not cache
    /// (`memory::host_cached_placement`).
    pub(crate) fn uncached_host_image(&self) {
        self.uncached_host_images.fetch_add(1, Ordering::Relaxed);
    }

    /// A host call's write into a texture with a decoded twin was decoded on the CPU into the
    /// twin (`Image::decode_written`, on the device's decode thread).
    pub(crate) fn cpu_block_decode(&self) {
        self.cpu_block_decodes.fetch_add(1, Ordering::Relaxed);
    }

    /// A GPU decode pass into a decoded twin was made (`Image::decode_pass`).
    pub(crate) fn block_decode_pass(&self) {
        self.block_decode_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// The device's queues were drained before a failed wait's command buffers let go of what
    /// they used (`Device::drain_queues`).
    pub(crate) fn queue_drain(&self) {
        self.queue_drains.fetch_add(1, Ordering::Relaxed);
    }

    /// A pipeline a depth stand-in's copies run was made (`depth_copy.rs`: the depth split, the
    /// draw back from the stand-in), by a draw's encoding where `at_draw`, else by a render
    /// pipeline state's creation.
    pub(crate) fn stand_in_copy_made(&self, at_draw: bool) {
        self.stand_in_copies_made.fetch_add(1, Ordering::Relaxed);
        if at_draw {
            self.stand_in_copies_made_at_draws
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A draw made the bind shape of its bind state (`LiveBind::shape`).
    pub(crate) fn draw_shape_made(&self) {
        self.draw_shapes_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw settled its variant and plan from its bind shape (`GraphicsState::settle_shape`).
    pub(crate) fn draw_settled(&self) {
        self.draws_settled.fetch_add(1, Ordering::Relaxed);
    }

    /// A bind of a variant made the notices it announces at every occurrence
    /// (`VariantObjects::announce_bound`).
    pub(crate) fn bound_notices_made(&self) {
        self.bound_notices_made.fetch_add(1, Ordering::Relaxed);
    }

    /// A command retained what an argument table binds (`ArgumentTable::retain`).
    pub(crate) fn argument_table_retained(&self) {
        self.argument_tables_retained
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw looked its `VkPipeline` up in its variant's memo by key
    /// (`RenderVariant::begin_graphics_pipeline`).
    pub(crate) fn graphics_pipeline_looked_up(&self) {
        self.graphics_pipelines_looked_up
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A pass over a bound linear attachment rendered them all in place, or one through the
    /// image its texture keeps (`through_copies`; `linear_attachment.rs`).
    pub(crate) fn linear_pass(&self, through_copies: bool) {
        let counter = match through_copies {
            false => &self.linear_passes_in_place,
            true => &self.linear_passes_through_copies,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Draws recorded `count` state commands into their secondary command buffers
    /// (`render::Recording::record`).
    pub(crate) fn draw_states_recorded(&self, count: u64) {
        if count > 0 {
            self.draw_states_recorded
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    /// A no-copy buffer's deallocator, given no keep-alive, waits past its release for the
    /// import of its pages, which something else still holds (`buffer.rs`).
    pub(crate) fn host_deallocator_delayed(&self) {
        self.host_deallocators_delayed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer kept an ended encoder's log of `bytes` bytes (`replay.rs`).
    pub(crate) fn replay_log_kept(&self, bytes: usize) {
        self.replay_logs_kept.fetch_add(1, Ordering::Relaxed);
        self.replay_log_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// A rebind encoded an ended encoder again (`command_buffer/reencode.rs`).
    pub(crate) fn encoder_reencoded(&self) {
        self.encoders_reencoded.fetch_add(1, Ordering::Relaxed);
    }

    /// A rebind could not encode an ended encoder again, which keeps its first recording.
    pub(crate) fn reencode_refused(&self) {
        self.reencodes_refused.fetch_add(1, Ordering::Relaxed);
    }

    /// A new image was moved to `GENERAL` and zeroed on the host (`texture/host_copy.rs`).
    pub(crate) fn image_started_on_the_host(&self) {
        self.images_started_on_the_host
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's CPU copy ran on the host (`texture/host_copy.rs`).
    pub(crate) fn host_texel_copy(&self) {
        self.host_texel_copies.fetch_add(1, Ordering::Relaxed);
    }

    /// A stand-in of `bytes` was made for textures over detached bytes (`texture/stand_in.rs`).
    pub(crate) fn stand_in_made(&self, bytes: u64) {
        self.stand_ins_made.fetch_add(1, Ordering::Relaxed);
        self.stand_in_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// A stand-in of `bytes` went.
    pub(crate) fn stand_in_released(&self, bytes: u64) {
        self.stand_in_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// A texture detached after it was resident was announced reachable
    /// (`Degradation::StaleDetachedReach`).
    pub(crate) fn stale_detached_reach(&self) {
        self.stale_detached_reaches.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over detached bytes whose stand-in has a sink was announced reachable
    /// (`Degradation::DetachedStorageKeepsWrites`).
    pub(crate) fn detached_storage_reach(&self) {
        self.detached_storage_reaches
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture in a heap over host pages was made tiled as Metal lays it out
    /// (`texture/heap_tiled.rs`).
    pub(crate) fn heap_texture_tiled(&self) {
        self.heap_textures_tiled.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture in a heap over host pages was met as laid out otherwise than Metal tiles it, as
    /// `kind` (`heap-texture-lossless-layout`, `heap-texture-block-compressed-layout` or
    /// `heap-texture-untiled-layout`; announced where it is the first of its kind).
    pub(crate) fn heap_layout_degraded(&self, kind: Degradation) {
        let counter = match kind {
            Degradation::HeapTextureLosslessLayout => &self.heap_texture_lossless_layouts,
            Degradation::HeapTextureBlockCompressedLayout => {
                &self.heap_texture_block_compressed_layouts
            }
            _ => &self.heap_texture_untiled_layouts,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// The objects counted of `kind`, for a kind met once per object (`Device::degraded_objects`):
    /// its counter's value; `None` for any other kind.
    pub(crate) fn degraded_objects(&self, kind: Degradation) -> Option<u64> {
        let counter = match kind {
            Degradation::StaleDetachedReach => &self.stale_detached_reaches,
            Degradation::DetachedStorageKeepsWrites => &self.detached_storage_reaches,
            Degradation::HeapTextureLosslessLayout => &self.heap_texture_lossless_layouts,
            Degradation::HeapTextureBlockCompressedLayout => {
                &self.heap_texture_block_compressed_layouts
            }
            Degradation::HeapTextureUntiledLayout => &self.heap_texture_untiled_layouts,
            _ => return None,
        };
        Some(counter.load(Ordering::Relaxed))
    }
}

/// What a device's commit path did so far (`Device::work_counts`), for the lanes: not part of the
/// API.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkCounts {
    /// `VkCommandPool`s made for command buffers, where no spare one was free.
    pub command_pools_made: u64,
    /// Buffers made to upload a live buffers' table to, where no spare one held it.
    pub live_tables_uploaded: u64,
    /// The completion thread's waits for a timeline's value, where the batch it waited for had
    /// no fence.
    pub timeline_waits: u64,
    /// Command buffers that took a slot of the device's timestamp pools: only those that encoded
    /// work, on a device that times them by GPU timestamps.
    pub timestamp_slots_taken: u64,
    /// Textures that took their storage form: a fetched render target's image made again with
    /// `STORAGE`, once per texture (`texture/storage_form.rs`).
    pub storage_forms_taken: u64,
    /// `VkBuffer`s made with the device-address usage: one per memory block a buffer with an
    /// address is placed in (`memory.rs`), and one per buffer that takes its own address (a
    /// no-copy buffer's, a preprocess buffer's). None per buffer placed in the runtime's memory:
    /// MoltenVK makes each such buffer resident in every encoder whose pipeline reads pointers.
    pub addressable_buffers_made: u64,
    /// `generate_mipmaps` calls that read the first level in place through a sampled view, not
    /// copied in (`mipmap_filter.rs`): the lane that measures the fetch holds itself to them.
    pub mipmap_first_levels_fetched: u64,
    /// Draws that handed the making of their `VkPipeline` to the device's pool and went on,
    /// rather than make it at the draw.
    pub pipelines_pooled: u64,
    /// `VkShaderModule`s pipeline states' variants made: none for a state made again whose
    /// released variant's objects were kept (`pipeline/kept.rs`).
    pub shader_modules_made: u64,
    /// Variants' stages that took the device's `VkShaderModule` of their SPIR-V, made before by
    /// another variant or state.
    pub shader_modules_shared: u64,
    /// `VkPipeline`s pipeline states' variants made (compute regions and their emulated mesh
    /// stages, indirect-bindable replays, and draws): none for one a released variant kept.
    pub variant_pipelines_made: u64,
    /// Variants made again that took back a released variant's objects and `VkPipeline`s.
    pub variants_taken_back: u64,
    /// Command buffers committed whose commands' plans could not list the sampler states live
    /// as they committed (`sampler_listing.rs`): a state past the sampler arrays, more states
    /// than the plan's tables hold, or a shape its checks refuse; or one of whose commands a host
    /// step recorded after the commit left a live state out, past the sampler arrays. Each fails
    /// `CommandBufferError::UnlistedArgumentSampler` unless another error fails it first, and
    /// is counted once whether or not its functions reach the state left out.
    pub unlisted_argument_sampler_failures: u64,
    /// Immutable sampler sets a command buffer's host step made, counting samplers on the
    /// scheduler's thread, because the device's map held no set of its variant's samplers: the
    /// open break (a render execution's step drawing a state made after its commit, on the variant
    /// for the encoder's sRGB or YCbCr textures, that bakes samplers no variant of the program's
    /// made).
    pub host_step_sampler_sets_made: u64,
    /// `VkPipeline`s states' creations made: each one's predicted key (`pipeline/predicted.rs`).
    pub creation_pipelines_made: u64,
    /// Of those, the ones a draw took so far: the rest are makings no draw has used yet, the
    /// speculative part of making at creation.
    pub creation_pipelines_drawn: u64,
    /// Immutable sampler sets variants made (the device's map's, and a host step's or a pool
    /// piece's own): none for a variant without immutable samplers.
    pub sampler_sets_made: u64,
    /// Variants that took the device's set of their immutable samplers, made before by another
    /// variant or state.
    pub sampler_sets_shared: u64,
    /// Immutable sampler sets destroyed while the device lives: a set of the map's once no
    /// variant shares it (a settle), and a host step's or a pool piece's own once nothing holds
    /// it (a sweep or a settle). A set whose creation failed part way was never made, and
    /// counts in neither.
    pub sampler_sets_destroyed: u64,
    /// Render passes commits made ahead for their render passes' host steps: the runs' passes
    /// of instances the encoder gave none, and those of instances over memoryless backings.
    pub ahead_render_passes_made: u64,
    /// Framebuffers commits made ahead so: one per instance over memoryless backings.
    pub ahead_framebuffers_made: u64,
    /// Backings memoryless attachments' textures made and keep, one per texture, subresource
    /// and queue whatever the passes (`attachment_stand_in::kept_backing`): the memory a
    /// memoryless attachment costs here where its passes run as several, which Metal's costs
    /// none.
    pub kept_backings_made: u64,
    /// Nanoseconds commits spent making ahead for their render passes' host steps, kept
    /// backings included.
    pub ahead_nanos: u64,
    /// Load passes render encoders recorded: one per extended-range or padded 10-bit color
    /// attachment that a pass which draws loads and stores, whose unused bits a dispatch made
    /// canonical in the tiles the pass's draws reach before the pass, as Metal's pass does
    /// (`render/unused_bits_load.rs`).
    pub unused_bits_load_passes: u64,
    /// The texels of those passes' regions (the render area rounded up to 32 texels a side,
    /// within the texture, of every layer the pass renders).
    pub unused_bits_load_texels: u64,
    /// Passes whose draws' tile reach capture replay found before the pass, from the positions
    /// their own vertex functions compute (`render/capture.rs`): the load pass rewrites the tiles
    /// those reach, as Metal's pass does.
    pub capture_reach_passes: u64,
    /// The primitives those passes' capture draws assembled and binned.
    pub capture_reach_primitives: u64,
    /// The tiles those passes' primitives reach, as their masks held them once their command
    /// buffers completed (every tile of the pass where its capture fell back, a primitive's
    /// reach not measured): those the load passes rewrote.
    pub capture_reach_tiles_found: u64,
    /// The tiles their draws' viewports and scissors reach (`clear_reach::DrawnArea`): those the
    /// load passes would rewrite without the capture.
    pub capture_reach_tiles_bound: u64,
    /// Those passes whose primitives reach every tile their viewports and scissors reach (a
    /// full-screen draw's, say): the capture changed nothing the load pass rewrote, but for the
    /// announcement.
    pub capture_reach_bound_passes: u64,
    /// Those passes whose primitives reach fewer: the tiles between were kept, as Metal keeps
    /// them, where the load pass would have rewritten them.
    pub capture_reach_narrower_passes: u64,
    /// Driver diagnostics withheld from the device's message sink: messages a driver diagnostic
    /// fact (`DriverFact::pipeline_creation_diagnostic`) measures the driver to report inside a
    /// call the runtime made, reported there (`driver_facts.rs`). MoltenVK 1.4.1's warning of a
    /// pipeline whose restart is dynamic, once per such pipeline it makes.
    pub driver_diagnostics_withheld: u64,
    /// Bytes of `VkDeviceMemory` the runtime allocated so far for memory of its own (its
    /// blocks, pooled or dedicated, freed since or not): none for host memory it imports. A
    /// detach allocates none for the bytes it detaches (`detach.rs`).
    pub device_memory_allocated: u64,
    /// Stand-ins made for textures over detached bytes (`texture/stand_in.rs`): one per distinct
    /// image plan among the textures detached at once.
    pub stand_ins_made: u64,
    /// Bytes of memory the live stand-ins hold now.
    pub stand_in_bytes: u64,
    /// Textures detached after a command buffer made their pages resident that a later
    /// submission could reach unnamed, each counted once (`Degradation::StaleDetachedReach`).
    pub stale_detached_reaches: u64,
    /// Textures of storage usage over detached bytes that a later commit could reach unnamed,
    /// each counted once (`Degradation::DetachedStorageKeepsWrites`).
    pub detached_storage_reaches: u64,
    /// Textures made in heaps over host pages laid out as Metal tiles them, each a shadow of its
    /// elements in the heap's bytes (`texture/heap_tiled.rs`).
    pub heap_textures_tiled: u64,
    /// Textures made in heaps over host pages that Metal keeps losslessly compressed
    /// (`Degradation::HeapTextureLosslessLayout`).
    pub heap_texture_lossless_layouts: u64,
    /// Block-compressed textures made in heaps over host pages whose placement is not measured
    /// (`Degradation::HeapTextureBlockCompressedLayout`).
    pub heap_texture_block_compressed_layouts: u64,
    /// Other textures made in heaps over host pages laid out otherwise than Metal tiles them
    /// (`Degradation::HeapTextureUntiledLayout`).
    pub heap_texture_untiled_layouts: u64,
    /// Fills recorded as `vkCmdFillBuffer` (`word_fill.rs`): every fill but on a driver measured
    /// to compile its fill per command pool.
    pub fills_recorded: u64,
    /// Fills recorded as transfers, `vkCmdUpdateBuffer` and copies of what it wrote, on a driver
    /// measured to compile its fill per command pool (`DriverFact::FillCompiledPerCommandPool`,
    /// MoltenVK 1.4.1), which records no `vkCmdFillBuffer`.
    pub fills_by_transfers: u64,
    // The commands MoltenVK 1.4.1 compiles a pipeline for per command pool on Apple's GPUs
    // (`PoolCompiled`), each counted as the runtime records it: a run's survey of which can
    // stall a commit behind a compile storm, as the fill did.
    /// `vkCmdClearAttachments` recorded (`clear_reach.rs`).
    pub attachment_clears_recorded: u64,
    /// `vkCmdBlitImage` recorded (`generate_mipmaps`, `mipmap.rs`).
    pub image_blits_recorded: u64,
    /// `vkCmdCopyQueryPoolResults` recorded (`visibility.rs`).
    pub query_result_copies_recorded: u64,
    /// Visibility result modes set other than `Disabled`, each beginning an occlusion query.
    pub visibility_queries_begun: u64,
    /// `vkCmdDrawIndirect` and `vkCmdDrawIndexedIndirect` recorded (`render.rs`).
    pub indirect_draws_recorded: u64,
    /// Internal submissions made (`device/submit.rs`), each one a host call waits for: a host
    /// copy through the GPU, a clear of a live buffer, a neutral's making.
    pub internal_submissions: u64,
    /// New `Shared` and `Managed` textures' images moved to `GENERAL` and zeroed on the host, with
    /// no submission (`texture/host_copy.rs`).
    pub images_started_on_the_host: u64,
    /// `replace_region` and `get_bytes` copies that ran on the host, with no submission
    /// (`texture/host_copy.rs`).
    pub host_texel_copies: u64,
    /// Submissions on the device's staging queue (`device/staging.rs`), each one a host call
    /// waits for: a `Shared` or `Managed` texture's first move, `replace_region` or `get_bytes`
    /// on a device that copies none on the host from memory the host caches.
    pub staging_submissions: u64,
    /// New `Shared` and `Managed` textures' images with host-transfer usage placed in memory the
    /// host does not cache, where the device has none the image may take
    /// (`texture/host_copy.rs`): their host copies read at uncached memory's speed (a 1080p
    /// `get_bytes` 108 ms on a Radeon 780M), and none elsewhere.
    pub uncached_host_images: u64,
    /// `replace_region` calls into a texture with a decoded twin (`texture/decoded_twin.rs`)
    /// whose region was decoded on the CPU and copied into the twin: every such call whose blocks
    /// were copied on the host or the staging queue, counted once the device's decode thread
    /// finished it (`texture/twin_decode.rs`).
    pub cpu_block_decodes: u64,
    /// GPU decode passes made into decoded twins (`texture/decoded_twin.rs`, `Image::decode_pass`):
    /// a blit's or a copy's into a BC texture, a heap texture's start, a write through
    /// submissions. None for a twin that takes CPU copies, whose texels only the CPU decodes.
    pub block_decode_passes: u64,
    /// Drains of the device's queues: one each time a wait for command buffers' work failed (a
    /// lost device), before those command buffers let go of what they used, so that the
    /// driver's own completion of the work, which can still read their pools, has run
    /// (`command_buffer/completion.rs`).
    pub queue_drains: u64,
    /// Pipelines a depth stand-in's copies run (the depth split, the draw back), made: once per
    /// device each, by the creation of the first render pipeline state whose draws test the depth
    /// themselves on a device that copies the stand-in that way (`depth_stand_in.rs`).
    pub stand_in_copies_made: u64,
    /// Of them, those a draw's encoding made: none where every such state's creation made them.
    pub stand_in_copies_made_at_draws: u64,
    /// Bind shapes draws made of their bind states: none for a draw whose bound slots keep their
    /// shapes and whose other bind state is its encoder's last draw's.
    pub draw_shapes_made: u64,
    /// Draws that settled their variant and plan from their bind shapes: none for a draw whose
    /// bind shape is the one its pipeline state last settled in the same buffer forms.
    pub draws_settled: u64,
    /// Notices of a variant's that a bind announces at every occurrence
    /// (`DeviceOptions::announce_each`), made for a bind: once per variant until the
    /// specializations or the formats its notices name change; each bind announces them.
    pub bound_notices_made: u64,
    /// Argument tables a dispatch or draw retained what they bind from: none for a table that is
    /// unchanged since its encoder's last command retained it.
    pub argument_tables_retained: u64,
    /// Draws that looked their `VkPipeline` up in their variant's memo by key: none for a draw
    /// whose key is the one its pipeline state's last draw found made, while that entry stays.
    pub graphics_pipelines_looked_up: u64,
    /// Passes over color attachments the device renders linear only under
    /// `VK_NV_linear_color_attachment`'s rule (bound ones, `linear_attachment.rs`) that rendered
    /// every one in place.
    pub linear_passes_in_place: u64,
    /// Those that rendered one through the image its texture keeps: copied in and its changed
    /// words merged back, or cleared there and copied back.
    pub linear_passes_through_copies: u64,
    /// State commands draws recorded into their secondary command buffers: pipeline binds,
    /// descriptor set binds, their push constants, vertex and index buffer binds, and dynamic
    /// states. None for a command that sets what its command buffer already holds from an
    /// earlier draw there.
    pub draw_states_recorded: u64,
    /// No-copy buffers and heaps released with a deallocator and no keep-alive whose deallocator
    /// waited past the release for the import of their pages, which something else held (a
    /// command buffer submitted before a texture over them went, through the texture's image):
    /// it runs once the import is freed (`host-deallocator-delayed`). With a keep-alive the
    /// deallocator runs at the release, and none is counted.
    pub host_deallocators_delayed: u64,
    /// Ended compute and blit encoders whose logs their command buffers kept, as a rebind before
    /// the first submission may encode them again (`replay.rs`), and the bytes of those logs.
    pub replay_logs_kept: u64,
    /// The bytes of the logs counted in `replay_logs_kept`.
    pub replay_log_bytes: u64,
    /// Ended encoders a rebind encoded again before their command buffer's first submission
    /// (`command_buffer/reencode.rs`).
    pub encoders_reencoded: u64,
    /// Ended encoders a rebind could not encode again (a call refused the second time, or a
    /// recording unlike the first), which keep their first recording and fail
    /// `EncodedUseRebound`.
    pub reencodes_refused: u64,
    /// Fences made to end command buffers, where none was idle (`command_buffer/fences.rs`):
    /// none on a driver that reads no command buffer's objects after its value, whose queues'
    /// failures are not relayed.
    pub fences_made: u64,
    /// Graphics pipeline libraries made, one per distinct part of the render pipelines' state,
    /// where the device links its render pipelines from libraries (`pipeline/libraries.rs`):
    /// none elsewhere.
    pub pipeline_libraries_made: u64,
    /// Those of them of a fragment shader: each a compile of a fragment function, the costly part.
    pub fragment_shader_libraries_made: u64,
    /// Render variants' `VkPipeline`s linked from libraries (each counted in
    /// `variant_pipelines_made` too).
    pub pipelines_linked: u64,
    /// Dispatches that handed the making of a region's `VkPipeline` to the device's pool and
    /// went on, rather than make it at the dispatch: each counted once per region it handed.
    pub dispatch_pipelines_pooled: u64,
}

impl Device {
    /// What its commit path did so far (`work_counts.rs`): counts the lanes hold to bounds. No
    /// Metal counterpart; not part of the API.
    #[doc(hidden)]
    pub fn work_counts(&self) -> WorkCounts {
        let counters = &self.shared().counts;
        WorkCounts {
            command_pools_made: counters.command_pools_made.load(Ordering::Relaxed),
            live_tables_uploaded: counters.live_tables_uploaded.load(Ordering::Relaxed),
            timeline_waits: counters.timeline_waits.load(Ordering::Relaxed),
            timestamp_slots_taken: counters.timestamp_slots_taken.load(Ordering::Relaxed),
            storage_forms_taken: counters.storage_forms_taken.load(Ordering::Relaxed),
            addressable_buffers_made: counters.addressable_buffers_made.load(Ordering::Relaxed),
            mipmap_first_levels_fetched: counters
                .mipmap_first_levels_fetched
                .load(Ordering::Relaxed),
            pipelines_pooled: counters.pipelines_pooled.load(Ordering::Relaxed),
            shader_modules_made: counters.shader_modules_made.load(Ordering::Relaxed),
            shader_modules_shared: counters.shader_modules_shared.load(Ordering::Relaxed),
            variant_pipelines_made: counters.variant_pipelines_made.load(Ordering::Relaxed),
            variants_taken_back: counters.variants_taken_back.load(Ordering::Relaxed),
            unlisted_argument_sampler_failures: counters
                .unlisted_argument_sampler_failures
                .load(Ordering::Relaxed),
            host_step_sampler_sets_made: counters
                .host_step_sampler_sets_made
                .load(Ordering::Relaxed),
            creation_pipelines_made: counters.creation_pipelines_made.load(Ordering::Relaxed),
            creation_pipelines_drawn: counters.creation_pipelines_drawn.load(Ordering::Relaxed),
            sampler_sets_made: counters.sampler_sets_made.load(Ordering::Relaxed),
            sampler_sets_shared: counters.sampler_sets_shared.load(Ordering::Relaxed),
            sampler_sets_destroyed: counters.sampler_sets_destroyed.load(Ordering::Relaxed),
            ahead_render_passes_made: counters.ahead_render_passes_made.load(Ordering::Relaxed),
            ahead_framebuffers_made: counters.ahead_framebuffers_made.load(Ordering::Relaxed),
            kept_backings_made: counters.kept_backings_made.load(Ordering::Relaxed),
            ahead_nanos: counters.ahead_nanos.load(Ordering::Relaxed),
            unused_bits_load_passes: counters.unused_bits_load_passes.load(Ordering::Relaxed),
            unused_bits_load_texels: counters.unused_bits_load_texels.load(Ordering::Relaxed),
            capture_reach_passes: counters.capture_reach_passes.load(Ordering::Relaxed),
            capture_reach_primitives: counters.capture_reach_primitives.load(Ordering::Relaxed),
            capture_reach_tiles_found: counters.capture_reach_tiles_found.load(Ordering::Relaxed),
            capture_reach_tiles_bound: counters.capture_reach_tiles_bound.load(Ordering::Relaxed),
            capture_reach_bound_passes: counters.capture_reach_bound_passes.load(Ordering::Relaxed),
            capture_reach_narrower_passes: counters
                .capture_reach_narrower_passes
                .load(Ordering::Relaxed),
            driver_diagnostics_withheld: counters
                .driver_diagnostics_withheld
                .load(Ordering::Relaxed),
            device_memory_allocated: lock(&self.shared().memory).allocated(),
            stand_ins_made: counters.stand_ins_made.load(Ordering::Relaxed),
            stand_in_bytes: counters.stand_in_bytes.load(Ordering::Relaxed),
            stale_detached_reaches: counters.stale_detached_reaches.load(Ordering::Relaxed),
            detached_storage_reaches: counters.detached_storage_reaches.load(Ordering::Relaxed),
            heap_textures_tiled: counters.heap_textures_tiled.load(Ordering::Relaxed),
            heap_texture_lossless_layouts: counters
                .heap_texture_lossless_layouts
                .load(Ordering::Relaxed),
            heap_texture_block_compressed_layouts: counters
                .heap_texture_block_compressed_layouts
                .load(Ordering::Relaxed),
            heap_texture_untiled_layouts: counters
                .heap_texture_untiled_layouts
                .load(Ordering::Relaxed),
            fills_recorded: counters.fills_recorded.load(Ordering::Relaxed),
            fills_by_transfers: counters.fills_by_transfers.load(Ordering::Relaxed),
            attachment_clears_recorded: counters.attachment_clears_recorded.load(Ordering::Relaxed),
            image_blits_recorded: counters.image_blits_recorded.load(Ordering::Relaxed),
            query_result_copies_recorded: counters
                .query_result_copies_recorded
                .load(Ordering::Relaxed),
            visibility_queries_begun: counters.visibility_queries_begun.load(Ordering::Relaxed),
            indirect_draws_recorded: counters.indirect_draws_recorded.load(Ordering::Relaxed),
            internal_submissions: counters.internal_submissions.load(Ordering::Relaxed),
            images_started_on_the_host: counters.images_started_on_the_host.load(Ordering::Relaxed),
            host_texel_copies: counters.host_texel_copies.load(Ordering::Relaxed),
            staging_submissions: counters.staging_submissions.load(Ordering::Relaxed),
            uncached_host_images: counters.uncached_host_images.load(Ordering::Relaxed),
            cpu_block_decodes: counters.cpu_block_decodes.load(Ordering::Relaxed),
            block_decode_passes: counters.block_decode_passes.load(Ordering::Relaxed),
            queue_drains: counters.queue_drains.load(Ordering::Relaxed),
            stand_in_copies_made: counters.stand_in_copies_made.load(Ordering::Relaxed),
            stand_in_copies_made_at_draws: counters
                .stand_in_copies_made_at_draws
                .load(Ordering::Relaxed),
            draw_shapes_made: counters.draw_shapes_made.load(Ordering::Relaxed),
            draws_settled: counters.draws_settled.load(Ordering::Relaxed),
            bound_notices_made: counters.bound_notices_made.load(Ordering::Relaxed),
            argument_tables_retained: counters.argument_tables_retained.load(Ordering::Relaxed),
            graphics_pipelines_looked_up: counters
                .graphics_pipelines_looked_up
                .load(Ordering::Relaxed),
            linear_passes_in_place: counters.linear_passes_in_place.load(Ordering::Relaxed),
            linear_passes_through_copies: counters
                .linear_passes_through_copies
                .load(Ordering::Relaxed),
            draw_states_recorded: counters.draw_states_recorded.load(Ordering::Relaxed),
            host_deallocators_delayed: counters.host_deallocators_delayed.load(Ordering::Relaxed),
            replay_logs_kept: counters.replay_logs_kept.load(Ordering::Relaxed),
            replay_log_bytes: counters.replay_log_bytes.load(Ordering::Relaxed),
            encoders_reencoded: counters.encoders_reencoded.load(Ordering::Relaxed),
            reencodes_refused: counters.reencodes_refused.load(Ordering::Relaxed),
            fences_made: self.shared().fences.made(),
            pipeline_libraries_made: counters.pipeline_libraries_made.load(Ordering::Relaxed),
            fragment_shader_libraries_made: counters
                .fragment_shader_libraries_made
                .load(Ordering::Relaxed),
            pipelines_linked: counters.pipelines_linked.load(Ordering::Relaxed),
            dispatch_pipelines_pooled: counters.dispatch_pipelines_pooled.load(Ordering::Relaxed),
        }
    }
}
