//! Counts of work a device's commit path does that the lanes hold to bounds: each counts a piece of
//! work the runtime once repeated at every commit or every change, so a lane fails when it is
//! repeated again; and the textures that took their storage form, which a lane holds to once each;
//! and the buffers made with the device-address usage, which every encoder that reads pointers made
//! resident one by one on MoltenVK; and the completion waits that missed their wake, which a test
//! holds to none. Not part of the API.

use super::degradation::Degradation;
use super::device::lock;
use super::Device;
use ash::vk;
use std::sync::atomic::{AtomicU64, Ordering};

/// A device's counters (`DeviceShared::counts`), each only ever added to.
#[derive(Default)]
pub(crate) struct Counters {
    command_pools_made: AtomicU64,
    spare_command_pools_taken: AtomicU64,
    spare_command_pools_reset: AtomicU64,
    live_tables_uploaded: AtomicU64,
    timeline_waits: AtomicU64,
    missed_completion_wakes: AtomicU64,
    completed_where_seen: AtomicU64,
    timestamp_slots_taken: AtomicU64,
    calibrations: AtomicU64,
    calibration_readings: AtomicU64,
    calibration_deviation_most: AtomicU64,
    calibration_disagreements: AtomicU64,
    storage_forms_taken: AtomicU64,
    addressable_buffers_made: AtomicU64,
    mipmap_first_levels_fetched: AtomicU64,
    pipelines_pooled: AtomicU64,
    shader_modules_made: AtomicU64,
    shader_modules_shared: AtomicU64,
    variant_pipelines_made: AtomicU64,
    flights_followed: AtomicU64,
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
    color_resolve_passes_made: AtomicU64,
    color_resolve_passes_kept: AtomicU64,
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
    queue_submissions: AtomicU64,
    command_buffers_submitted: AtomicU64,
    command_buffers_begun: AtomicU64,
    encoder_command_buffers_begun: AtomicU64,
    barrier_buffers_kept: AtomicU64,
    stamped_buffers_kept: AtomicU64,
    stamped_buffers_reused: AtomicU64,
    finales_folded: AtomicU64,
    finales_apart: AtomicU64,
    completed_waits: AtomicU64,
    bare_starts: AtomicU64,
    prefixed_starts: AtomicU64,
    unrun_instances_ended: AtomicU64,
    inline_passes: AtomicU64,
    inline_passes_split: AtomicU64,
    inline_split_instances: AtomicU64,
    commits_handed_off: AtomicU64,
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
    depth16_sided_draws: AtomicU64,
    bound_notices_made: AtomicU64,
    argument_tables_retained: AtomicU64,
    graphics_pipelines_looked_up: AtomicU64,
    linear_passes_in_place: AtomicU64,
    linear_passes_through_copies: AtomicU64,
    kept_bytes_passes: AtomicU64,
    signed_depth_switch_ons: AtomicU64,
    magnitude_stand_in_passes: AtomicU64,
    magnitude_write_backs: AtomicU64,
    sign_strips: AtomicU64,
    color_group_passes: AtomicU64,
    color_group_twins: AtomicU64,
    color_group_draws_skipped: AtomicU64,
    draw_states_recorded: AtomicU64,
    host_deallocators_delayed: AtomicU64,
    host_keep_alives_kept: AtomicU64,
    host_keep_alives_covered: AtomicU64,
    dma_buf_imports: AtomicU64,
    dma_buf_memories_live: AtomicU64,
    dma_buf_image_memories_live: AtomicU64,
    host_imports_released: AtomicU64,
    host_imports_reimported: AtomicU64,
    host_imports_resident: AtomicU64,
    host_imports_lost: AtomicU64,
    host_import_fds_kept: AtomicU64,
    dma_buf_sparse_buffers: AtomicU64,
    dma_buf_images: AtomicU64,
    dma_buf_image_fallbacks: AtomicU64,
    pitched_host_images: AtomicU64,
    pitched_host_image_refusals: AtomicU64,
    pitched_host_image_refusals_recalled: AtomicU64,
    first_move_keeps: AtomicU64,
    first_move_kept_bytes: AtomicU64,
    first_move_restored_bytes: AtomicU64,
    shared_alias_images: AtomicU64,
    shared_alias_moves_waited: AtomicU64,
    modifier_feature_queries: AtomicU64,
    shadowed_textures: AtomicU64,
    shadows_zeroed: AtomicU64,
    shadow_copies: AtomicU64,
    shadow_texels_copied_in: AtomicU64,
    shadow_texels_copied_back: AtomicU64,
    shadow_copies_back_dispatched: AtomicU64,
    bounded_shadow_copies: AtomicU64,
    views_through_swizzle: AtomicU64,
    view_shadows: AtomicU64,
    attachment_reads_through_copies: AtomicU64,
    attachment_read_splits: AtomicU64,
    reading_draws_snapshotted: AtomicU64,
    snapshot_inline_splits: AtomicU64,
    own_pixel_reads: AtomicU64,
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

    /// A command buffer took one of the device's spare command pools (`spares.rs`).
    pub(crate) fn spare_command_pool_taken(&self) {
        self.spare_command_pools_taken
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer took a spare command pool that one of its command buffers may still be
    /// recording in, and reset it first (`spares.rs`).
    pub(crate) fn spare_command_pool_reset(&self) {
        self.spare_command_pools_reset
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The live buffers' table was uploaded to a buffer made for it, no spare reused.
    pub(crate) fn live_table_uploaded(&self) {
        self.live_tables_uploaded.fetch_add(1, Ordering::Relaxed);
    }

    /// A queue's waiter waited for a timeline's value, not at a fence
    /// (`command_buffer/completion.rs`).
    pub(crate) fn timeline_wait(&self) {
        self.timeline_waits.fetch_add(1, Ordering::Relaxed);
    }

    /// A completion waiter's wait ran out its slice although what it waited for was signaled when
    /// the waiter looked again (`command_buffer/completion.rs`).
    pub(crate) fn missed_completion_wake(&self) {
        self.missed_completion_wakes.fetch_add(1, Ordering::Relaxed);
    }

    /// A command buffer's transition to `Completed` or `Error` ran on the queue waiter that saw
    /// it complete (`CommandBufferShared::finish_here`).
    pub(crate) fn completed_where_seen(&self) {
        self.completed_where_seen.fetch_add(1, Ordering::Relaxed);
    }

    /// A committed command buffer took a slot of one of the device's timestamp pools, to write
    /// its GPU start and end into (`gpu_time.rs`).
    pub(crate) fn timestamp_slot_taken(&self) {
        self.timestamp_slots_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// A calibration pair was taken of `readings` readings, its deviation `deviation`
    /// nanoseconds (`gpu_time.rs`, `Timing::pair`).
    pub(crate) fn calibrated(&self, readings: u32, deviation: u64) {
        self.calibrations.fetch_add(1, Ordering::Relaxed);
        self.calibration_readings
            .fetch_add(u64::from(readings), Ordering::Relaxed);
        self.calibration_deviation_most
            .fetch_max(deviation, Ordering::Relaxed);
    }

    /// A command buffer's calibrated GPU times fell outside its life, and it was placed by its
    /// completion (`gpu_time.rs`, `GpuClock::span`).
    pub(crate) fn calibration_disagreed(&self) {
        self.calibration_disagreements
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture took its storage form (`texture/storage_form.rs`).
    pub(crate) fn storage_form_taken(&self) {
        self.storage_forms_taken.fetch_add(1, Ordering::Relaxed);
    }

    /// The textures that took their storage form so far, as [`Device::work_counts`] reads it.
    #[cfg(test)]
    pub(crate) fn storage_forms_taken(&self) -> u64 {
        self.storage_forms_taken.load(Ordering::Relaxed)
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

    /// A variant's making waited for its twin's in the air before its own (`pipeline/flight.rs`).
    pub(crate) fn flight_followed(&self) {
        self.flights_followed.fetch_add(1, Ordering::Relaxed);
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

    /// A pass made the objects of its exact color resolves (`color_resolve.rs`).
    pub(crate) fn color_resolve_pass_made(&self) {
        self.color_resolve_passes_made
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A pass took the objects of its exact color resolves that an image kept
    /// (`color_resolve::KeptResolves`).
    pub(crate) fn color_resolve_pass_kept(&self) {
        self.color_resolve_passes_kept
            .fetch_add(1, Ordering::Relaxed);
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

    /// A `vkQueueSubmit` or `vkQueueBindSparse` call was made with `command_buffers`
    /// `VkCommandBuffer`s among its `VkSubmitInfo`s.
    pub(crate) fn queue_submitted(&self, command_buffers: usize) {
        self.queue_submissions.fetch_add(1, Ordering::Relaxed);
        (self.command_buffers_submitted).fetch_add(command_buffers as u64, Ordering::Relaxed);
    }

    /// A `VkCommandBuffer` was begun (`vkBeginCommandBuffer`).
    pub(crate) fn command_buffer_begun(&self) {
        self.command_buffers_begun.fetch_add(1, Ordering::Relaxed);
    }

    /// An encoder began a `VkCommandBuffer` as it recorded (`vkBeginCommandBuffer`): its own, a
    /// segment's, one it went on in, or a render encoder's draws' secondary.
    pub(crate) fn encoder_command_buffer_begun(&self) {
        self.command_buffers_begun.fetch_add(1, Ordering::Relaxed);
        (self.encoder_command_buffers_begun).fetch_add(1, Ordering::Relaxed);
    }

    /// A batch's barrier command buffer was recorded to be kept and submitted again
    /// (`command_buffer/kept_barriers.rs`).
    pub(crate) fn barrier_buffer_kept(&self) {
        self.barrier_buffers_kept.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch's barrier command buffer that writes a command buffer's timestamps was recorded
    /// to be kept for its slot and submitted again (`command_buffer/kept_barriers.rs`).
    pub(crate) fn stamped_buffer_kept(&self) {
        self.stamped_buffers_kept.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch took such a kept command buffer again (`command_buffer/kept_barriers.rs`).
    pub(crate) fn stamped_buffer_reused(&self) {
        self.stamped_buffers_reused.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch with work was submitted with its end in its last command buffer
    /// (`command_buffer/submit.rs`).
    pub(crate) fn finale_folded(&self) {
        self.finales_folded.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch with work was submitted with its end in a command buffer of its own.
    pub(crate) fn finale_apart(&self) {
        self.finales_apart.fetch_add(1, Ordering::Relaxed);
    }

    /// A batch with work was submitted waiting for `waits` timelines at values the host saw them
    /// reach (`command_buffer/submit.rs`).
    pub(crate) fn completed_waited(&self, waits: usize) {
        self.completed_waits
            .fetch_add(waits as u64, Ordering::Relaxed);
    }

    /// A batch with work was submitted with no command buffer of barriers before its first
    /// encoder.
    pub(crate) fn bare_start(&self) {
        self.bare_starts.fetch_add(1, Ordering::Relaxed);
    }

    /// A timed command buffer's first batch was submitted with its start written by its first
    /// encoder's own command buffer (`gpu_time.rs`, `command_buffer/submit.rs`).
    pub(crate) fn prefixed_start(&self) {
        self.prefixed_starts.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass's first instance began inline in a primary command buffer of its encoder's
    /// (`render/inline_first.rs`).
    pub(crate) fn inline_pass(&self) {
        self.inline_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass whose first instance recorded inline ended with more than one instance.
    pub(crate) fn inline_pass_split(&self) {
        self.inline_passes_split.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass instance after a split of such a pass began inline in the same primary
    /// command buffer (`render/inline_splits.rs`).
    pub(crate) fn inline_split_instance(&self) {
        self.inline_split_instances.fetch_add(1, Ordering::Relaxed);
    }

    /// A render pass instance its pass replaced or gave up before running it had its draws
    /// recording ended (`render.rs`, `RenderCommandEncoder::end_unrun`).
    pub(crate) fn unrun_instance_ended(&self) {
        self.unrun_instances_ended.fetch_add(1, Ordering::Relaxed);
    }

    /// A commit left its batches to the scheduler's thread (`scheduler/inbox.rs`).
    pub(crate) fn commit_handed_off(&self) {
        self.commits_handed_off.fetch_add(1, Ordering::Relaxed);
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

    /// A draw ran the variant whose fragment stage declares its `Depth16Unorm` depth bias's side
    /// (`VariantKey::depth16_bias_raises`).
    pub(crate) fn depth16_sided_draw(&self) {
        self.depth16_sided_draws.fetch_add(1, Ordering::Relaxed);
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

    /// A pass rendered a texture over a buffer it does not store through the image its texture
    /// keeps (`kept_bytes.rs`).
    pub(crate) fn kept_bytes_pass(&self) {
        self.kept_bytes_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A render target's 32-bit float depth was switched on to render over its magnitude
    /// stand-in, or to have its words stripped of their signs (`texture/signed_depth.rs`): once
    /// per texture image.
    pub(crate) fn signed_depth_switch_on(&self) {
        self.signed_depth_switch_ons.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass rendered a depth over its magnitude stand-in (`texture/signed_depth.rs`), and
    /// wrote what it changed back into the depth where `written_back`.
    pub(crate) fn magnitude_stand_in_pass(&self, written_back: bool) {
        self.magnitude_stand_in_passes
            .fetch_add(1, Ordering::Relaxed);
        if written_back {
            self.magnitude_write_backs.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A depth's words were stored without their signs after a pass with depth writes whose
    /// stores the device's pass left signed (`texture/signed_depth.rs`, `SignStrip`).
    pub(crate) fn sign_strip(&self) {
        self.sign_strips.fetch_add(1, Ordering::Relaxed);
    }

    /// A render encoder drew one of its pass's later color groups (`render/color_groups.rs`).
    pub(crate) fn color_group_pass(&self) {
        self.color_group_passes.fetch_add(1, Ordering::Relaxed);
    }

    /// A pass of several color groups made the twin of its depth/stencil attachment its later
    /// groups render against (`render/color_groups.rs`).
    pub(crate) fn color_group_twin(&self) {
        self.color_group_twins.fetch_add(1, Ordering::Relaxed);
    }

    /// A later color group recorded nothing for a draw that writes none of its colors, no depth
    /// and no stencil (`render/color_groups.rs`).
    pub(crate) fn color_group_draw_skipped(&self) {
        self.color_group_draws_skipped
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Draws recorded `count` state commands into their secondary command buffers
    /// (`render::Recording::record`).
    pub(crate) fn draw_states_recorded(&self, count: u64) {
        if count > 0 {
            self.draw_states_recorded
                .fetch_add(count, Ordering::Relaxed);
        }
    }

    /// A no-copy buffer's deallocator waits past its release: given no keep-alive, for the import
    /// of its pages, which something else still holds, or for the batches in flight as that
    /// import's memory is freed (`freed_memory.rs`); or for a live buffers' table that lists its
    /// pages, which a command buffer holds (`buffer.rs`).
    pub(crate) fn host_deallocator_delayed(&self) {
        self.host_deallocators_delayed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A no-copy buffer's keep-alive is kept past its release by the import of its pages, which
    /// something else still holds and no other buffer lies in (`buffer.rs`).
    pub(crate) fn host_keep_alive_kept(&self) {
        self.host_keep_alives_kept.fetch_add(1, Ordering::Relaxed);
    }

    /// The import that kept a keep-alive was freed, and dropped it.
    pub(crate) fn host_keep_alive_freed(&self) {
        self.host_keep_alives_kept.fetch_sub(1, Ordering::Relaxed);
    }

    /// A no-copy buffer's keep-alive was dropped at its release: another buffer lying in the
    /// import of its pages, or the keep-alive the import keeps, keeps them mapped until it is
    /// freed (`buffer.rs`).
    pub(crate) fn host_keep_alive_covered(&self) {
        self.host_keep_alives_covered
            .fetch_add(1, Ordering::Relaxed);
    }

    /// `chunks` chunks of host pages a caller's source made were imported as dma-buf memory, and
    /// a buffer bound in them, where the device refused the pages' pointer (`dma_buf.rs`).
    pub(crate) fn dma_buf_imported(&self, chunks: usize) {
        self.dma_buf_imports
            .fetch_add(chunks as u64, Ordering::Relaxed);
    }

    /// A `VkDeviceMemory` was imported from a dma-buf of a caller's source and is held: an
    /// image's dedicated import where `of_image`, else a chunk of a buffer's pages (`dma_buf.rs`).
    pub(crate) fn dma_buf_memory_imported(&self, of_image: bool) {
        self.dma_buf_live(of_image).fetch_add(1, Ordering::Relaxed);
    }

    /// `count` such imports were freed.
    pub(crate) fn dma_buf_memories_freed(&self, of_image: bool, count: usize) {
        (self.dma_buf_live(of_image)).fetch_sub(count as u64, Ordering::Relaxed);
    }

    /// The live gauge of dma-buf imports of images where `of_image`, else of buffers' chunks.
    fn dma_buf_live(&self, of_image: bool) -> &AtomicU64 {
        match of_image {
            true => &self.dma_buf_image_memories_live,
            false => &self.dma_buf_memories_live,
        }
    }

    /// A releasable import was released while idle (`host_residency.rs`).
    pub(crate) fn host_import_released(&self) {
        self.host_imports_released.fetch_add(1, Ordering::Relaxed);
    }

    /// A released import was imported again.
    pub(crate) fn host_import_reimported(&self) {
        self.host_imports_reimported.fetch_add(1, Ordering::Relaxed);
    }

    /// A releasable import holds its memory from now on (`resident`), or no longer.
    pub(crate) fn host_import_resident(&self, resident: bool) {
        match resident {
            true => self.host_imports_resident.fetch_add(1, Ordering::Relaxed),
            false => self.host_imports_resident.fetch_sub(1, Ordering::Relaxed),
        };
    }

    /// A read found a released import that the device could not import again: the device is
    /// lost.
    pub(crate) fn host_import_lost(&self) {
        self.host_imports_lost.fetch_add(1, Ordering::Relaxed);
    }

    /// Keep `count` more descriptors of dma-bufs for imports the device may release, where the
    /// kept ones stay within `limit`: whether they do (counted then).
    pub(crate) fn host_import_fds_keep(&self, count: u64, limit: u64) -> bool {
        let kept = &self.host_import_fds_kept;
        let fits = |held: u64| held.checked_add(count).filter(|after| *after <= limit);
        kept.fetch_update(Ordering::AcqRel, Ordering::Acquire, fits)
            .is_ok()
    }

    /// `count` such descriptors were closed.
    pub(crate) fn host_import_fds_closed(&self, count: u64) {
        self.host_import_fds_kept
            .fetch_sub(count, Ordering::Relaxed);
    }

    /// A no-copy buffer's (or heap's, or rebind's) `VkBuffer` was assembled from several chunks'
    /// imports as a sparse buffer (`dma_buf.rs`).
    pub(crate) fn dma_buf_sparse_buffer_made(&self) {
        self.dma_buf_sparse_buffers.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's image over dma-buf pages was made and bound to a dedicated import of a dma-buf
    /// of its own pages (`dma_buf.rs`).
    pub(crate) fn dma_buf_image_made(&self) {
        self.dma_buf_images.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over dma-buf pages that such an image would alias took the path of a texture
    /// over memory no image aliases instead: the device made no such image of it (`dma_buf.rs`).
    pub(crate) fn dma_buf_image_fell_back(&self) {
        self.dma_buf_image_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's image over a no-copy buffer's host pages imported by their pointer was made a
    /// `DRM_FORMAT_MOD_LINEAR` image at Metal's row pitch and bound there (`texture.rs`'s
    /// `host_pitched_image`).
    pub(crate) fn pitched_host_image_made(&self) {
        self.pitched_host_images.fetch_add(1, Ordering::Relaxed);
    }

    /// The driver refused a `DRM_FORMAT_MOD_LINEAR` image at Metal's row pitch (`texture.rs`'s
    /// `host_pitched_image`).
    #[cfg(unix)]
    pub(crate) fn pitched_host_image_refused(&self) {
        self.pitched_host_image_refusals
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture's image at Metal's row pitch was not asked of the driver, which refused the same
    /// image before (`texture/pitch_refusals.rs`).
    #[cfg(unix)]
    pub(crate) fn pitched_host_image_refusal_recalled(&self) {
        self.pitched_host_image_refusals_recalled
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The host kept the bytes `count` first moves' images are bound to across them, `bytes` of
    /// them in all (`texture/first_move.rs`'s `Keep`).
    pub(crate) fn first_move_kept(&self, (count, bytes): (usize, u64)) {
        if count > 0 {
            self.first_move_keeps
                .fetch_add(count as u64, Ordering::Relaxed);
            self.first_move_kept_bytes
                .fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// The host stored `bytes` of the kept bytes again, those first moves changed
    /// (`texture/first_move.rs`'s `KeptBytes::write_back`).
    pub(crate) fn first_move_restored(&self, bytes: u64) {
        if bytes > 0 {
            self.first_move_restored_bytes
                .fetch_add(bytes, Ordering::Relaxed);
        }
    }

    /// A texture over host pages took the `VkImage` another texture of the same image over the
    /// same bytes made (`texture/shared_alias.rs`).
    pub(crate) fn shared_alias_taken(&self) {
        self.shared_alias_images.fetch_add(1, Ordering::Relaxed);
    }

    /// A submission left a shared image's first move waiting: no texture over the image lived
    /// (`texture/shared_alias.rs`).
    pub(crate) fn shared_alias_move_waited(&self) {
        self.shared_alias_moves_waited
            .fetch_add(1, Ordering::Relaxed);
    }

    /// The driver was asked a format's `DRM_FORMAT_MOD_LINEAR` tiling features (`dma_buf.rs`'s
    /// `modifier_features`).
    #[cfg(unix)]
    pub(crate) fn modifier_features_asked(&self) {
        self.modifier_feature_queries
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture over a buffer's bytes was made a shadow of them (`shadow.rs`), its image zeroed
    /// first where `zeroed` (its copies leave an aspect out).
    pub(crate) fn shadowed_texture_made(&self, zeroed: bool) {
        self.shadowed_textures.fetch_add(1, Ordering::Relaxed);
        if zeroed {
            self.shadows_zeroed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// An encoder planned `n` copies between shadowed textures' truths and their shadows
    /// (`shadow.rs`), whose copies in and back of buffers' textures' images move `texels_in` and
    /// `texels_back` texels.
    pub(crate) fn shadow_copies_planned(&self, n: usize, texels_in: u64, texels_back: u64) {
        self.shadow_copies.fetch_add(n as u64, Ordering::Relaxed);
        self.shadow_texels_copied_in
            .fetch_add(texels_in, Ordering::Relaxed);
        self.shadow_texels_copied_back
            .fetch_add(texels_back, Ordering::Relaxed);
    }

    /// An encoder planned `n` copies in of the texels only writes within a rectangle changed
    /// since a shadowed texture's copy in (`shadow.rs`, `shadow::Within`).
    pub(crate) fn bounded_shadow_copies_planned(&self, n: usize) {
        self.bounded_shadow_copies
            .fetch_add(n as u64, Ordering::Relaxed);
    }

    /// An encoder recorded a shadowed texture's copy back as a dispatch
    /// (`shadow_copy_back.rs`).
    pub(crate) fn shadow_copy_back_dispatched(&self) {
        self.shadow_copies_back_dispatched
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A texture view was made the image itself to its bindings, through a swizzle over the
    /// image's own format (`texture/view_shadow.rs`, `ViewRecord::read_as`).
    pub(crate) fn view_through_swizzle_made(&self) {
        self.views_through_swizzle.fetch_add(1, Ordering::Relaxed);
    }

    /// A texture view's shadow image was made (`texture/view_shadow.rs`): the view's own, or
    /// the render shadow of a view its bindings present through a swizzle.
    pub(crate) fn view_shadow_made(&self) {
        self.view_shadows.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples, at a slot, a texture that presents a color attachment of its render
    /// pass read the texture's copy, where its shadow is served (`texture/attached_reads.rs`).
    pub(crate) fn attachment_read_through_copy(&self) {
        self.attachment_reads_through_copies
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture after a memory barrier of its render
    /// pass recorded into a new instance (`texture/attached_reads.rs`).
    pub(crate) fn attachment_read_split(&self) {
        self.attachment_read_splits.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture after a memory barrier of its render
    /// pass read it through a snapshot inside its instance (`render/snapshot.rs`).
    pub(crate) fn reading_draw_snapshotted(&self) {
        self.reading_draws_snapshotted
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A draw read through a snapshot recorded into a new instance, as its own was recorded
    /// inline (`render/snapshot.rs`).
    pub(crate) fn snapshot_inline_split(&self) {
        self.snapshot_inline_splits.fetch_add(1, Ordering::Relaxed);
    }

    /// A draw that samples a color attachment's texture reads it at its fragments' own pixels
    /// alone, and its copy in moves the texels around its scissors alone
    /// (`texture/attached_reads.rs`).
    pub(crate) fn own_pixel_read(&self) {
        self.own_pixel_reads.fetch_add(1, Ordering::Relaxed);
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
    /// Spare `VkCommandPool`s command buffers took, made by earlier ones. The pool is not reset
    /// where the command buffer that gave it back completed and ended every command buffer of it
    /// it was handed: each is reset as it is begun again.
    pub spare_command_pools_taken: u64,
    /// The spare `VkCommandPool`s among those taken that were reset first, as one of their
    /// command buffers may still have been recording: the command buffer that gave the pool back
    /// failed, was dropped unsubmitted, or left a command buffer of it unended (a recording it did
    /// not use).
    pub spare_command_pools_reset: u64,
    /// Buffers made to upload a live buffers' table to, where no spare one held it.
    pub live_tables_uploaded: u64,
    /// The queues' waiters' waits for a timeline's value, where the batch one waited for had
    /// no fence.
    pub timeline_waits: u64,
    /// The completion waiters' waits that ran out their 100 ms slice although the fence or the
    /// timeline value they waited for was signaled when the waiter looked again: a wake the wait
    /// missed, which finishes its batch a slice late (MoltenVK 1.4.1's wait for any of several
    /// timelines lost them; `command_buffer/completion.rs`), or a signal in the slice's last
    /// moment.
    pub missed_completion_wakes: u64,
    /// Command buffers whose transition to `Completed` or `Error` ran on the queue waiter that
    /// saw their work complete, with no handler thread woken: those with no completed handler, no
    /// pass that captured its tiles' reach, and no earlier transition of their queue that runs
    /// the program's code or runs on a handler thread then, while the runtime held none of the
    /// program's objects a release drops or runs (`command_buffer/completion.rs`).
    pub completed_where_seen: u64,
    /// Command buffers that took a slot of the device's timestamp pools: only those that encoded
    /// work, on a device that times them by GPU timestamps.
    pub timestamp_slots_taken: u64,
    /// Calibration pairs the device took to place its command buffers' GPU times on the host's
    /// clock (one a commit at most every 100 ms, on a device with calibrated timestamps).
    pub calibrations: u64,
    /// The `vkGetCalibratedTimestampsKHR` calls those pairs took: each pair is read again while
    /// each reading halves the deviation the one before it stated, four readings at most.
    pub calibration_readings: u64,
    /// The largest deviation, in nanoseconds, of the pairs taken: how far apart the GPU and host
    /// values of the pair that placed the times may stand, as its call stated.
    pub calibration_deviation_most: u64,
    /// Command buffers whose calibration pair placed a GPU time outside their life (before their
    /// first batch's submission began, or after their completion was seen) by more than the
    /// pair's stated deviation and the clocks' drift allow: a pair that disagrees with the
    /// device's own timestamps. Each was placed by its completion, as on an uncalibrated device.
    pub calibration_disagreements: u64,
    /// Textures that took their storage form: a fetched render target's image made again with
    /// `STORAGE`, once per texture (`texture/storage_form.rs`). Counted by the time the
    /// committing command buffer is scheduled, and by the time a command buffer made meanwhile
    /// is returned.
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
    /// Of those, the makings that waited for a twin's making in the air, of the same functions
    /// and specialization, to land before their own (`pipeline/flight.rs`).
    pub flights_followed: u64,
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
    /// Passes that made the descriptor sets and framebuffers of their exact color resolves
    /// (`color_resolve.rs`).
    pub color_resolve_passes_made: u64,
    /// Passes that took the descriptor sets and framebuffers of their exact color resolves from
    /// the image they resolve into, which a pass of the same views made
    /// (`color_resolve::KeptResolves`).
    pub color_resolve_passes_kept: u64,
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
    /// The runtime's live pooled `VkDeviceMemory` blocks with a placement in them (`memory.rs`):
    /// with the blocks below and every dma-buf import, the allocations ANV on i915 lists in each
    /// of its kernel submissions, beside its own buffer objects.
    pub memory_blocks_live: u64,
    /// Its live pooled blocks with no placement, kept for the next.
    pub empty_memory_blocks_live: u64,
    /// Its live dedicated blocks: a placement larger than half a block, or one placed apart.
    pub dedicated_memory_blocks_live: u64,
    /// Its live imports of the caller's host memory, a block each (`Memory::import_host`).
    pub host_imports_live: u64,
    /// Its blocks that went while batches submitted before were in flight, on a device whose
    /// driver fails pending work as memory a device-address buffer lay in is freed
    /// (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`): each block's memory waited
    /// for those batches to complete before it was freed (`freed_memory.rs`). 0 elsewhere.
    pub freed_memory_parked: u64,
    /// Of those, the blocks whose memory waits now.
    pub freed_memory_waiting: u64,
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
    /// `vkQueueSubmit` and `vkQueueBindSparse` calls the runtime made on the device's queues once
    /// it was made: every batch's (its new resources' first work rides in the same call, in a
    /// `VkSubmitInfo` of its own before the batch's, `texture/first_move.rs`), internal and
    /// staging submissions (an internal submission's first work likewise), sparse binds, and
    /// fences submitted with no work. Each is a
    /// submission of its own to the driver; Mesa's drivers merge the `VkSubmitInfo`s of one call
    /// where no signal lies between a command and a later one, and no further (`vk_queue.c`).
    pub queue_submissions: u64,
    /// The `VkCommandBuffer`s those calls submitted.
    pub command_buffers_submitted: u64,
    /// `VkCommandBuffer`s the runtime began once the device was made: each encoder's, a render
    /// encoder's draws' secondary ones, segments', a batch's barrier and first-work ones, internal
    /// and staging submissions'. A driver may clear a command buffer's state as it begins one
    /// (ANV, Mesa 26.2.4: `anv_cmd_state_init` at each begin after a reset).
    pub command_buffers_begun: u64,
    /// Of `command_buffers_begun`, those encoders began as they recorded: each encoder's own, a
    /// segment's, one an encoder went on in, and a render encoder's draws' secondaries. The others
    /// are begun as batches are submitted (their barriers and ends), on whichever thread submits
    /// them, and by the device's own submissions; a lane that reads its encoding thread's steps
    /// counts these alone there.
    pub encoder_command_buffers_begun: u64,
    /// Barrier command buffers of batches recorded once, kept, and submitted again by later
    /// batches once the batch that last took one completed (`command_buffer/kept_barriers.rs`):
    /// each is among `command_buffers_begun` once.
    pub barrier_buffers_kept: u64,
    /// Of a batch's barrier command buffers, those that also write a timed command buffer's
    /// timestamps (the start after resetting its slot, or the end), recorded once per list of
    /// barriers and slot of the device's timestamp pools and kept: each is among
    /// `command_buffers_begun` once.
    pub stamped_buffers_kept: u64,
    /// The times a batch took such a kept command buffer again, for a later command buffer
    /// holding the same slot, once the batch that last took it completed and the slot's
    /// timestamps were read: no command buffer begun for it.
    pub stamped_buffers_reused: u64,
    /// Batches with work submitted with their end (the barrier to the host, and in a timed
    /// command buffer's last batch its end timestamp) recorded into their last command buffer as
    /// they ended, the end's own command buffer saved (`command_buffer/submit.rs`).
    pub finales_folded: u64,
    /// Batches with work submitted with their end in a command buffer of its own: one that
    /// signals an event, leaves its submission a segment or sparse update, names a texture over
    /// the caller's image that moves between layouts, or whose last command buffer does not go
    /// to the queue as recorded.
    pub finales_apart: u64,
    /// The waits batches with work made for timelines at values the host had seen them reach,
    /// which make what completed work wrote visible to them: one per timeline whose completed
    /// value moved since a batch of the same `VkQueue` last waited for it.
    pub completed_waits: u64,
    /// Batches with work submitted with no command buffer of barriers before their first
    /// encoder: nothing planned before it, and no start timestamp to write there (none, or one
    /// its first encoder wrote, `WorkCounts::prefixed_starts`).
    pub bare_starts: u64,
    /// Timed command buffers whose first batch took the start its first encoder's own command
    /// buffer wrote as it opened, with no command buffer of barriers before that encoder for the
    /// timestamp (`gpu_time.rs`): those batches start bare where nothing else is planned.
    pub prefixed_starts: u64,
    /// Render pass instances a pass replaced or gave up before running them (a draw that needed
    /// another shape before any draw recorded, a reroute's instance left behind, a capture that
    /// recorded nothing, a first instance whose draws record inline instead), whose draws
    /// recording the runtime ended: their command pool goes back to the device's spares unreset
    /// (`WorkCounts::spare_command_pools_reset` counts the others). A pass's first instance that
    /// may record inline begins its draws' secondary only where a command records into it, so
    /// one replaced or recording inline before then began none, ended none, and is not counted.
    pub unrun_instances_ended: u64,
    /// Render passes whose first render pass instance recorded its draws inline in a primary
    /// command buffer of its encoder's, begun at its first draw, in place of a secondary
    /// (`render/inline_first.rs`).
    pub inline_passes: u64,
    /// Those of them that ended with more than one instance.
    pub inline_passes_split: u64,
    /// The instances after a split of such passes that recorded their draws inline too, in the
    /// same primary command buffer after the instance before, in place of a secondary
    /// (`render/inline_splits.rs`); the others record in secondaries, after them.
    pub inline_split_instances: u64,
    /// Commits that left their batches to the device's scheduler thread and returned at once,
    /// which submits them in commit order at its next pass (`scheduler/inbox.rs`): every commit,
    /// as no commit submits anything itself.
    pub commits_handed_off: u64,
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
    /// bind shape is one of the latest its pipeline state settled in the same buffer forms.
    pub draws_settled: u64,
    /// Draws that ran the variant whose fragment stage adds a `Depth16Unorm` depth's bias itself
    /// and declares its side (`DepthGreater` or `DepthLess`), where the bias's side is known: the
    /// device may then test the depth before the stage runs, which a stage that writes its depth
    /// otherwise keeps it from.
    pub depth16_sided_draws: u64,
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
    /// Passes that rendered a color attachment over a buffer's bytes, which they do not store
    /// (`DontCare`, `MultisampleResolve`), through the image its texture keeps, so that the buffer
    /// keeps its bytes (`kept_bytes.rs`).
    pub kept_bytes_passes: u64,
    /// Render targets' 32-bit float depths switched on to render over their magnitude stand-in,
    /// or to have their words stripped of their signs after a pass with depth writes whose stores
    /// the device's pass leaves signed: a write that may store a negative value, a NaN, or a
    /// denormal the device's depth test compares as zero, once per texture image
    /// (`texture/signed_depth.rs`). None where no such write was made: every other pass renders
    /// its depth in place.
    pub signed_depth_switch_ons: u64,
    /// Passes that loaded such a depth and rendered over its magnitude stand-in: its words copied
    /// out, their stand-ins written by a dispatch and copied into the stand-in before the pass.
    pub magnitude_stand_in_passes: u64,
    /// Those written back into the depth, a pass that stores it and may have written it (a
    /// depth/stencil state with depth writes, or an indirect command buffer's own states):
    /// the stand-in copied out after the pass, merged into the depth's words by a dispatch, and
    /// copied back.
    pub magnitude_write_backs: u64,
    /// Passes after which such a depth, rendered in place or through another image copied back
    /// into it, not over its magnitude stand-in, was stored without its signs, as Metal stores
    /// it: the pass set a depth/stencil state with depth writes, and the device's pass left the
    /// words signed (every such pass, or, on a device whose depth test and stores are Metal's
    /// own, one that drew under none). A single-sample depth's words copied out, stripped by a
    /// dispatch and copied back; a multisampled one's samples copied out, stripped by a
    /// sample-rate draw of each layer and copied back, or copied out and drawn back stripped.
    pub sign_strips: u64,
    /// Passes a render encoder drew past its first for a pass of more color attachments than the
    /// device draws at once, one per later color group, each of that group's colors over a twin
    /// of the depth and stencil (`render/color_groups.rs`).
    pub color_group_passes: u64,
    /// The twins of a depth/stencil attachment such passes made: one per pass of more groups
    /// with a depth/stencil attachment, loaded by a copy or a clear.
    pub color_group_twins: u64,
    /// Draws a later color group recorded nothing for: they write none of its colors, no depth
    /// and no stencil.
    pub color_group_draws_skipped: u64,
    /// State commands draws recorded into their secondary command buffers: pipeline binds,
    /// descriptor set binds, their push constants, vertex and index buffer binds, and dynamic
    /// states. None for a command that sets what its command buffer already holds from an
    /// earlier draw there.
    pub draw_states_recorded: u64,
    /// No-copy buffers and heaps released with a deallocator whose deallocator waited past the
    /// release (`host-deallocator-delayed`), counted at each wait: given no keep-alive, for the
    /// import of their pages, which something else held (a command buffer submitted before a
    /// texture over them went, through the texture's image), until the import is freed, or, on a
    /// device whose driver has `RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`, for the
    /// batches submitted before the import's memory went, until they completed (with a keep-alive
    /// it runs at the release there); or, given none, for a live buffers' table that listed their
    /// pages, which a command buffer that may reach them unnamed held, until it let go of it
    /// (`live_buffers.rs`; with a keep-alive it runs at the release there too).
    pub host_deallocators_delayed: u64,
    /// Keep-alives of no-copy buffers and heaps (`KeepAlive`) held now past their release: each
    /// by the import of its pages, whose last buffer it was while something else (a texture's
    /// image a submitted command buffer may still read) held the import, or that a live buffers'
    /// table a command buffer that may reach them unnamed holds listed as it was released, until
    /// the import is freed. One per import at most: a later keep-alive over the same import is
    /// dropped at its release (`host_keep_alives_covered`).
    pub host_keep_alives_kept: u64,
    /// Keep-alives dropped at their buffer's release, after its deallocator ran, because the
    /// import of its pages stays mapped without them: another no-copy buffer made over the same
    /// bytes lies in it, or it keeps another's keep-alive already.
    pub host_keep_alives_covered: u64,
    /// Chunks of host pages imported as dma-buf memory from a caller's source
    /// (`HostMemory::with_dma_buf`), where the device refused the pages' pointer: no-copy
    /// buffers', heaps' and rebinds' alike, one per chunk. None where every pointer was imported.
    pub dma_buf_imports: u64,
    /// Those chunks' imports live now: each a `VkDeviceMemory` the device holds until the last
    /// buffer, texture or command buffer over its pages lets go. RADV lists each, as every
    /// imported memory, in every kernel submission of the device (Mesa 26.2.4's global BO list),
    /// and amdgpu's work per submission grows with them (`runtime_lane::submit_per_import`).
    pub dma_buf_memories_live: u64,
    /// The dedicated dma-buf imports of images over such memory live now (`dma_buf_images`): one
    /// per live image, listed in every kernel submission as those above.
    pub dma_buf_image_memories_live: u64,
    /// No-copy buffers' imports of dma-bufs released while idle, their `VkBuffer` destroyed and
    /// their memory freed, the dma-bufs kept: a runtime-internal residency, which nothing of
    /// Metal's observes (`HostMemory::with_dma_buf` documents it). Their chunks leave
    /// `dma_buf_memories_live`.
    pub host_imports_released: u64,
    /// Released imports imported again from the dma-bufs kept, each at the first blit that read
    /// the buffer after its release (on the encoding thread).
    pub host_imports_reimported: u64,
    /// The imports the device may release that hold their memory now.
    pub host_imports_resident: u64,
    /// Reads other than a blit's that found an import released and the device unable to import
    /// it again: the device was lost then. None but out of memory.
    pub host_imports_lost: u64,
    /// The descriptors of dma-bufs the device keeps open now to import released imports again: at
    /// most a quarter of the process's descriptor limit.
    pub host_import_fds_kept: u64,
    /// Of those buffers, heaps and rebinds, the ones whose `VkBuffer` is a sparse buffer
    /// assembled from several chunks.
    pub dma_buf_sparse_buffers: u64,
    /// Images of textures over such dma-buf memory (over a no-copy buffer, placed in a no-copy
    /// heap, a view of either), each a `DRM_FORMAT_MOD_LINEAR` image bound to a dedicated import
    /// of a dma-buf of its own pages from the caller's source, aliasing the buffer's bytes: one per
    /// image made, a rebind's too. Not counted in `dma_buf_imports`.
    pub dma_buf_images: u64,
    /// Textures over such dma-buf memory that such an image would alias but the device made none
    /// (the source failed, the format, usage, extent or plane layout is not supported, or the
    /// driver refused the import): each takes the path of a texture over memory no image aliases
    /// (`shadowed-texture`).
    pub dma_buf_image_fallbacks: u64,
    /// Images of textures over no-copy buffers' host pages imported by their pointer made
    /// `DRM_FORMAT_MOD_LINEAR` images at Metal's row pitch, where the driver's own linear image
    /// lays the rows out otherwise (ANV packs them tight), each bound in the buffer's import and
    /// aliasing its bytes: one per image made, a rebind's too.
    pub pitched_host_images: u64,
    /// Images at Metal's row pitch the driver refused (with
    /// `VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT`, by laying the plane out at another
    /// offset or pitch, or by asking a dedicated allocation of it): one per image asked of the
    /// driver, and a texture asks up to two (with a usage it did not declare, then without).
    pub pitched_host_image_refusals: u64,
    /// Images at Metal's row pitch not asked of the driver, which refused the same image (format,
    /// extent, usage, flags, view formats and row pitch) before (`texture/pitch_refusals.rs`).
    pub pitched_host_image_refusals_recalled: u64,
    /// First moves of images over a no-copy buffer's host pages (or a rebind's new pages) across
    /// which the host kept the bytes the image is bound to, which a move from `UNDEFINED` may
    /// discard: read through the host pointer once the submissions before completed, the move
    /// submitted alone and waited for, the bytes it changed written back. One per image moved, a
    /// rebind's too.
    pub first_move_keeps: u64,
    /// The bytes those moves kept, each read once, and read again after the move.
    pub first_move_kept_bytes: u64,
    /// The kept bytes a move changed that the host stored again: none where the driver's moves
    /// change no byte of the memory bound to the image.
    pub first_move_restored_bytes: u64,
    /// Textures over a no-copy buffer's host pages that took the `VkImage` another texture of the
    /// same image over the same bytes of the same import made, which the import or a texture
    /// holds, rather than make one (and move it, keep its texels or, over dma-buf pages, ask the
    /// pages' source and import one).
    pub shared_alias_images: u64,
    /// Submissions that found a shared image's first move with no texture over the image alive,
    /// and left it waiting for a submission while one is (one a submission and image): the move
    /// holds the host pages of a texture over the image while it runs.
    pub shared_alias_moves_waited: u64,
    /// Formats whose `DRM_FORMAT_MOD_LINEAR` tiling features the driver was asked, for an image
    /// over dma-bufs or one at Metal's row pitch over host pages: each format once at most, as
    /// the answer is kept.
    pub modifier_feature_queries: u64,
    /// Textures over a buffer's bytes (a 2D texture or array over a buffer, a no-copy buffer or an
    /// IOSurface plane, or a view of another layout of one) that no image aliases, each an image
    /// of its own that the encoders copy the bytes into and back (`shadowed-texture`).
    pub shadowed_textures: u64,
    /// Of `shadowed_textures`, those whose image was zeroed as it was made: its copies leave an
    /// aspect out (a depth texture over a surface's plane, whose image's stencil reads zero).
    /// Every other shadow's copy in writes each texel before any command reads it, so its image
    /// starts as its memory lies.
    pub shadows_zeroed: u64,
    /// Copies between a shadowed texture's truth and its shadow that encoders planned
    /// (`shadow.rs`): copies in before a command, copies back after one or as the encoder ends,
    /// and the keeps and merges of a command that writes the bytes two ways. Each is a command of
    /// its encoder over the texture's texels.
    pub shadow_copies: u64,
    /// The texels those copies in move from a 2D texture's buffer's bytes to its image (a texture
    /// over a buffer, a no-copy buffer or an IOSurface plane): every texel, or those inside the
    /// rectangle bounding a pass's draws' scissors over an attachment it loads and takes in no
    /// other way.
    pub shadow_texels_copied_in: u64,
    /// The texels those copies back move from a 2D texture's image to its buffer's bytes (a
    /// texture over a buffer, a no-copy buffer or an IOSurface plane): every texel, or those
    /// inside the rectangle bounding a pass's draws' scissors over an attachment it loads.
    pub shadow_texels_copied_back: u64,
    /// Of those copies back, the ones recorded as a compute dispatch that stores the texels'
    /// words, where the device runs one faster than its copy
    /// (`DriverFact::ShadowCopyBackDispatchFaster`): the same bytes either way.
    pub shadow_copies_back_dispatched: u64,
    /// Of `shadow_copies`, the copies in of the texels inside the rectangle bounding where the
    /// bytes changed since the image's copy in, where only writes inside rectangles of its texels
    /// changed them: a render pass instance's draws into a color attachment it loads, over a
    /// buffer's bytes, inside their scissors, before the next instance whose draws sample the
    /// attachment's texture (`texture/attached_reads.rs`). Each holds the same texels as a copy
    /// of every one; the rest of the image holds the bytes still.
    pub bounded_shadow_copies: u64,
    /// Texture views made the image itself to their bindings, through a swizzle over the image's
    /// own format: a view in other channels of its image's bits (an `A8Unorm` view of an
    /// `R8Unorm` texture), which neither copies nor announces (`texture/view_shadow.rs`).
    pub views_through_swizzle: u64,
    /// Shadow images of texture views made (`texture/view_shadow.rs`, `shadowed-texture-view`):
    /// a view's own, or the render shadow of one of `views_through_swizzle`, made at the first
    /// pass that renders into it.
    pub view_shadows: u64,
    /// Draws that sample, at a slot, a texture that presents a color attachment of their own
    /// render pass, each reading the texture's copy, taken before its render pass instance, in
    /// the attachment's place where the texture's shadow is served (`texture/attached_reads.rs`;
    /// the copies count in `shadow_copies`).
    pub attachment_reads_through_copies: u64,
    /// Draws that sample a color attachment's texture after a memory barrier of their render
    /// pass, each recorded into a new instance (a barrier inside an instance orders one pixel
    /// alone), whose copy holds what the draws before the barrier wrote
    /// (`texture/attached_reads.rs`).
    pub attachment_read_splits: u64,
    /// Draws that sample a color attachment's texture after a memory barrier of their render
    /// pass, each reading it through a snapshot inside its instance in place of a split: drawn
    /// twice, first copying each fragment's texel into the snapshot, then reading the snapshot
    /// (`render/snapshot.rs`). They count in neither `attachment_reads_through_copies` nor
    /// `attachment_read_splits`.
    pub reading_draws_snapshotted: u64,
    /// Of `reading_draws_snapshotted`, the draws recorded into a new instance because their own
    /// was recorded inline, its render pass instance begun without the feedback loop layout the
    /// snapshot needs (`render/snapshot.rs`, `render/inline_first.rs`).
    pub snapshot_inline_splits: u64,
    /// Of `attachment_reads_through_copies`, the draws that read the texture at their fragments'
    /// own pixels alone (`StageModule::own_pixel_textures`) over a texture whose copy is cut to a
    /// rectangle: the copy in before their instance moves only the texels inside the rectangle
    /// bounding their scissors, a pixel wider each way (`texture/attached_reads.rs`).
    pub own_pixel_reads: u64,
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
        let (allocated, census, (parked, waiting)) = {
            let memory = lock(&self.shared().memory);
            (memory.allocated(), memory.census(), memory.parked())
        };
        WorkCounts {
            command_pools_made: counters.command_pools_made.load(Ordering::Relaxed),
            spare_command_pools_taken: counters.spare_command_pools_taken.load(Ordering::Relaxed),
            spare_command_pools_reset: counters.spare_command_pools_reset.load(Ordering::Relaxed),
            live_tables_uploaded: counters.live_tables_uploaded.load(Ordering::Relaxed),
            timeline_waits: counters.timeline_waits.load(Ordering::Relaxed),
            missed_completion_wakes: counters.missed_completion_wakes.load(Ordering::Relaxed),
            completed_where_seen: counters.completed_where_seen.load(Ordering::Relaxed),
            timestamp_slots_taken: counters.timestamp_slots_taken.load(Ordering::Relaxed),
            calibrations: counters.calibrations.load(Ordering::Relaxed),
            calibration_readings: counters.calibration_readings.load(Ordering::Relaxed),
            calibration_deviation_most: (counters.calibration_deviation_most)
                .load(Ordering::Relaxed),
            calibration_disagreements: (counters.calibration_disagreements).load(Ordering::Relaxed),
            storage_forms_taken: counters.storage_forms_taken.load(Ordering::Relaxed),
            addressable_buffers_made: counters.addressable_buffers_made.load(Ordering::Relaxed),
            mipmap_first_levels_fetched: counters
                .mipmap_first_levels_fetched
                .load(Ordering::Relaxed),
            pipelines_pooled: counters.pipelines_pooled.load(Ordering::Relaxed),
            shader_modules_made: counters.shader_modules_made.load(Ordering::Relaxed),
            shader_modules_shared: counters.shader_modules_shared.load(Ordering::Relaxed),
            variant_pipelines_made: counters.variant_pipelines_made.load(Ordering::Relaxed),
            flights_followed: counters.flights_followed.load(Ordering::Relaxed),
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
            color_resolve_passes_made: counters.color_resolve_passes_made.load(Ordering::Relaxed),
            color_resolve_passes_kept: counters.color_resolve_passes_kept.load(Ordering::Relaxed),
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
            device_memory_allocated: allocated,
            memory_blocks_live: census.pooled,
            empty_memory_blocks_live: census.empty,
            dedicated_memory_blocks_live: census.dedicated,
            host_imports_live: census.imported,
            freed_memory_parked: parked,
            freed_memory_waiting: waiting,
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
            queue_submissions: counters.queue_submissions.load(Ordering::Relaxed),
            command_buffers_submitted: counters.command_buffers_submitted.load(Ordering::Relaxed),
            command_buffers_begun: counters.command_buffers_begun.load(Ordering::Relaxed),
            encoder_command_buffers_begun: (counters.encoder_command_buffers_begun)
                .load(Ordering::Relaxed),
            barrier_buffers_kept: counters.barrier_buffers_kept.load(Ordering::Relaxed),
            stamped_buffers_kept: counters.stamped_buffers_kept.load(Ordering::Relaxed),
            stamped_buffers_reused: counters.stamped_buffers_reused.load(Ordering::Relaxed),
            finales_folded: counters.finales_folded.load(Ordering::Relaxed),
            finales_apart: counters.finales_apart.load(Ordering::Relaxed),
            completed_waits: counters.completed_waits.load(Ordering::Relaxed),
            bare_starts: counters.bare_starts.load(Ordering::Relaxed),
            prefixed_starts: counters.prefixed_starts.load(Ordering::Relaxed),
            unrun_instances_ended: counters.unrun_instances_ended.load(Ordering::Relaxed),
            inline_passes: counters.inline_passes.load(Ordering::Relaxed),
            inline_passes_split: counters.inline_passes_split.load(Ordering::Relaxed),
            inline_split_instances: counters.inline_split_instances.load(Ordering::Relaxed),
            commits_handed_off: counters.commits_handed_off.load(Ordering::Relaxed),
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
            depth16_sided_draws: counters.depth16_sided_draws.load(Ordering::Relaxed),
            bound_notices_made: counters.bound_notices_made.load(Ordering::Relaxed),
            argument_tables_retained: counters.argument_tables_retained.load(Ordering::Relaxed),
            graphics_pipelines_looked_up: counters
                .graphics_pipelines_looked_up
                .load(Ordering::Relaxed),
            linear_passes_in_place: counters.linear_passes_in_place.load(Ordering::Relaxed),
            linear_passes_through_copies: counters
                .linear_passes_through_copies
                .load(Ordering::Relaxed),
            kept_bytes_passes: counters.kept_bytes_passes.load(Ordering::Relaxed),
            signed_depth_switch_ons: counters.signed_depth_switch_ons.load(Ordering::Relaxed),
            magnitude_stand_in_passes: counters.magnitude_stand_in_passes.load(Ordering::Relaxed),
            magnitude_write_backs: counters.magnitude_write_backs.load(Ordering::Relaxed),
            sign_strips: counters.sign_strips.load(Ordering::Relaxed),
            color_group_passes: counters.color_group_passes.load(Ordering::Relaxed),
            color_group_twins: counters.color_group_twins.load(Ordering::Relaxed),
            color_group_draws_skipped: counters.color_group_draws_skipped.load(Ordering::Relaxed),
            draw_states_recorded: counters.draw_states_recorded.load(Ordering::Relaxed),
            host_deallocators_delayed: counters.host_deallocators_delayed.load(Ordering::Relaxed),
            host_keep_alives_kept: counters.host_keep_alives_kept.load(Ordering::Relaxed),
            host_keep_alives_covered: counters.host_keep_alives_covered.load(Ordering::Relaxed),
            dma_buf_imports: counters.dma_buf_imports.load(Ordering::Relaxed),
            dma_buf_memories_live: counters.dma_buf_memories_live.load(Ordering::Relaxed),
            dma_buf_image_memories_live: counters
                .dma_buf_image_memories_live
                .load(Ordering::Relaxed),
            host_imports_released: counters.host_imports_released.load(Ordering::Relaxed),
            host_imports_reimported: counters.host_imports_reimported.load(Ordering::Relaxed),
            host_imports_resident: counters.host_imports_resident.load(Ordering::Relaxed),
            host_imports_lost: counters.host_imports_lost.load(Ordering::Relaxed),
            host_import_fds_kept: counters.host_import_fds_kept.load(Ordering::Relaxed),
            dma_buf_sparse_buffers: counters.dma_buf_sparse_buffers.load(Ordering::Relaxed),
            dma_buf_images: counters.dma_buf_images.load(Ordering::Relaxed),
            dma_buf_image_fallbacks: counters.dma_buf_image_fallbacks.load(Ordering::Relaxed),
            pitched_host_images: counters.pitched_host_images.load(Ordering::Relaxed),
            pitched_host_image_refusals: counters
                .pitched_host_image_refusals
                .load(Ordering::Relaxed),
            pitched_host_image_refusals_recalled: counters
                .pitched_host_image_refusals_recalled
                .load(Ordering::Relaxed),
            first_move_keeps: counters.first_move_keeps.load(Ordering::Relaxed),
            first_move_kept_bytes: counters.first_move_kept_bytes.load(Ordering::Relaxed),
            first_move_restored_bytes: counters.first_move_restored_bytes.load(Ordering::Relaxed),
            shared_alias_images: counters.shared_alias_images.load(Ordering::Relaxed),
            shared_alias_moves_waited: counters.shared_alias_moves_waited.load(Ordering::Relaxed),
            modifier_feature_queries: counters.modifier_feature_queries.load(Ordering::Relaxed),
            shadowed_textures: counters.shadowed_textures.load(Ordering::Relaxed),
            views_through_swizzle: counters.views_through_swizzle.load(Ordering::Relaxed),
            view_shadows: counters.view_shadows.load(Ordering::Relaxed),
            attachment_reads_through_copies: counters
                .attachment_reads_through_copies
                .load(Ordering::Relaxed),
            attachment_read_splits: counters.attachment_read_splits.load(Ordering::Relaxed),
            reading_draws_snapshotted: counters.reading_draws_snapshotted.load(Ordering::Relaxed),
            snapshot_inline_splits: counters.snapshot_inline_splits.load(Ordering::Relaxed),
            own_pixel_reads: counters.own_pixel_reads.load(Ordering::Relaxed),
            shadows_zeroed: counters.shadows_zeroed.load(Ordering::Relaxed),
            shadow_copies: counters.shadow_copies.load(Ordering::Relaxed),
            shadow_texels_copied_in: counters.shadow_texels_copied_in.load(Ordering::Relaxed),
            shadow_texels_copied_back: counters.shadow_texels_copied_back.load(Ordering::Relaxed),
            shadow_copies_back_dispatched: counters
                .shadow_copies_back_dispatched
                .load(Ordering::Relaxed),
            bounded_shadow_copies: counters.bounded_shadow_copies.load(Ordering::Relaxed),
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
