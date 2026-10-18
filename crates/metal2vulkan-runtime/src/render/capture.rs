//! Capture replay: the tiles a pass's draws reach, from the positions its own vertex functions
//! compute, before the pass.
//!
//! An Apple GPU loads and stores only the tiles a pass's primitives reach (`bind::tile_reach`),
//! which decides what the pass's extended-range load rewrites (`render/unused_bits_load.rs`).
//! The host decides that reach where the draws reach nothing or every tile, and bounds it by
//! their viewports and scissors (`clear_reach.rs`); elsewhere the runtime announces
//! `unreached-tiles-rewritten`. Capture replay finds the tiles themselves:
//!
//! - **Kept.** In a pass that consumes its reach (a color attachment the load pass takes, of one
//!   sample, Metal's default 32 by 32 tiles, one layer, and no rotated texture's frame among
//!   them: `unused_bits_load::consumes_capture`), each draw is kept with the encoder's
//!   state at it (`reroute::Rerecord`) while every draw of the pass can run again before it: a
//!   direct draw of a triangle list or strip, indexed or not, of any instance count, of one
//!   amplification and no views, through a pipeline without tessellation or mesh stages whose
//!   vertex function writes no memory and chooses no viewport, under solid fill, outside a
//!   rasterization-rate map, a visibility query and custom sample positions, its indices read in
//!   place; at most [`MAX_DRAWS`] draws of at most [`MAX_RECORDS`] vertices. Any other draw, a
//!   tile dispatch or an execution of render commands ends the keeping for the pass.
//! - **Recorded again.** As the pass ends, before its steps, each kept draw records again with
//!   its state, through the capture form of its variant (`GraphicsState::capture_variant`: the
//!   vertex stage alone, writing each vertex's clip position to a record), into a pass of no
//!   attachments under empty scissors, so no fragment runs. Every vertex function's position is
//!   invariant, so the records hold the positions the pass's own draws rasterize. The capture form
//!   is translated on the device's pool and its objects and pipeline made at a later pass (the
//!   HITCH rule: nothing waits on the program's thread); until each of the pass's draws has both,
//!   the pass stays on the host's reach and announces as before.
//! - **Binned.** Before the pass, the capture draws run, then `bind::capture_assembly_module`
//!   assembles their records into primitives as Metal's input assembly does (a strip's restarts
//!   and parity too), `bind::tile_reach_module` bins them by the measured rule into a tile mask,
//!   and `bind::capture_settle_module` makes it every tile where a primitive crossed the near or
//!   far plane or lay past what was measured, which it reports through the command buffer's
//!   status words: the command buffer announces `unreached-tiles-rewritten` once it completes
//!   (`StatusReport::captured_reach_unmeasured`). The load pass rewrites the tiles the mask
//!   names. Exact otherwise: the rule is Metal's, measured, and its arithmetic integer.
//! - **Counted.** The mask is also copied where the host reads it once the command buffer
//!   completes, which counts its tiles against those the draws' viewports and scissors reach
//!   (`WorkCounts::capture_reach_tiles_found` and `capture_reach_tiles_bound`), and the pass as
//!   one whose primitives reach every tile of that bound (`capture_reach_bound_passes`: a
//!   full-screen draw's, the capture changing nothing rewritten) or fewer
//!   (`capture_reach_narrower_passes`).

use super::bound::Bound;
use super::reroute::Rerecord;
use super::shadows::InstanceShadows;
use super::{Assembly, Instance, InterlockedStorage, RenderCommandEncoder};
use crate::attachmentless;
use crate::binding::CommandUses;
use crate::clear_reach::{DrawnArea, DrawnTiles, PassReach};
use crate::command_buffer::CaptureTally;
use crate::draw_source::DrawSource;
use crate::owned_kernel::{OwnedKernel, OwnedModule, OwnedSet, OwnedSets};
use crate::render_pass::PassLayout;
use crate::render_target::{self, Feedback};
use crate::tracker::Barrier;
use crate::tracker::{GeometryWrites, IndirectReads};
use crate::values::ResourceOptions;
use ash::vk;
use metal2vulkan::bind::{
    CaptureAssemblyModule, CaptureSettleModule, DrawCall, ExecutionStatus, TileReachModule,
    ViewSetup,
};
use metal2vulkan::metal::{CullMode, IndexType, PrimitiveType};
use metal2vulkan::pipeline::CaptureForm;
use std::sync::Arc;

/// The most draws a pass keeps for capture replay: past them its reach is the host's.
pub(super) const MAX_DRAWS: usize = 256;

/// The most capture records a pass's kept draws write (their vertices or indices, each instance
/// counted): 8 MiB of records, and their primitives' 16 MiB.
pub(super) const MAX_RECORDS: u64 = 1 << 18;

/// What a pass keeps for capture replay (the module docs).
pub(super) enum Keeping {
    /// The draws kept so far, and the records they write.
    Draws(Vec<Rerecord>, u64),
    /// None: the pass does not consume its reach, or one of its draws cannot run again before
    /// it.
    Stopped,
    /// The kept draws record again as capture draws, the pass ending: the one recording now.
    Recording(Capturing),
}

impl Keeping {
    /// What a pass of `layout` keeps from its start.
    pub(super) fn of(layout: &PassLayout) -> Keeping {
        match consumes(layout) {
            true => Keeping::Draws(Vec::new(), 0),
            false => Keeping::Stopped,
        }
    }
}

/// A capture draw recording (`RenderCommandEncoder::draw_resident` in capture mode).
pub(super) struct Capturing {
    /// The device address of the draw's first record.
    pub(super) records: u64,
    /// The device address of the draw's first index, where it is indexed.
    pub(super) indices: Option<u64>,
    /// What the capture draws use, as the one command they record in.
    pub(super) uses: CommandUses,
    /// A capture draw recorded nothing: its capture form's objects or pipeline were not made
    /// yet, or it could not record.
    pub(super) missed: bool,
}

/// Whether a pass of `layout` consumes its draws' tile reach where capture replay can give it:
/// the load pass takes one of its color attachments and no rotated frame
/// (`unused_bits_load::consumes_capture`), at one sample and one layer, in Metal's default 32 by
/// 32 tiles, which the reach kernel bins (`bind::tile_reach`).
fn consumes(layout: &PassLayout) -> bool {
    layout.samples == 1
        && layout.layers == 1
        && layout.tile.tile == Some([32, 32])
        && super::unused_bits_load::consumes_capture(layout, false)
}

/// Whether capture replay assembles draws of `assembly`.
pub(super) fn assembles(assembly: Assembly) -> bool {
    matches!(
        assembly,
        Assembly::Primitives(PrimitiveType::Triangle | PrimitiveType::TriangleStrip)
    )
}

/// The capture form of a direct draw `draw`: of its vertices, or of its indices of their type.
pub(super) fn form(draw: &DrawCall) -> Option<CaptureForm> {
    match *draw {
        DrawCall::Primitives { .. } => Some(CaptureForm::Vertices),
        DrawCall::Indexed { index_type, .. } => match index_type {
            t if t == metal2vulkan::vk::IndexType::UINT16 => {
                Some(CaptureForm::Indexed(IndexType::UInt16))
            }
            t if t == metal2vulkan::vk::IndexType::UINT32 => {
                Some(CaptureForm::Indexed(IndexType::UInt32))
            }
            _ => None,
        },
        _ => None,
    }
}

/// The vertices or indices of each instance of a direct draw `draw`, and its instances.
fn counts(draw: &DrawCall) -> Option<(u64, u64)> {
    match *draw {
        DrawCall::Primitives {
            vertex_count,
            instance_count,
            ..
        } => Some((u64::from(vertex_count), u64::from(instance_count))),
        DrawCall::Indexed {
            index_count,
            instance_count,
            ..
        } => Some((u64::from(index_count), u64::from(instance_count))),
        _ => None,
    }
}

/// The reach the pass's capture draws give (`RenderCommandEncoder::capture_reach`): the tile
/// mask, recorded before the pass's other steps, in the scratch the command buffer holds.
#[derive(Debug)]
pub(super) struct CapturedReach {
    /// The mask's `VkBuffer`, its offset there, and its bytes: one bit a tile, as
    /// `bind::TileReachModule` numbers them.
    pub(super) mask: (vk::Buffer, u64, u64),
}

/// One kept draw's place among the pass's capture records and tile reach states.
struct Placed {
    /// Its first record: instance 0's vertex 0.
    first: u64,
    /// Its records an instance, and its instances.
    count: u64,
    instances: u64,
    topology: PrimitiveType,
    /// It restarts at flagged records: an indexed strip.
    restarts: bool,
    /// Its cull mode is front or back.
    culls: bool,
}

impl RenderCommandEncoder {
    /// Keep the draw just recorded (`call` of `assembly` and `draw`, its indices from `index`)
    /// for capture replay where the pass keeps its draws and `capturable` (the module docs);
    /// otherwise the pass keeps none from here on. A draw recorded again (a reroute's, or a
    /// capture draw) was kept at its first recording.
    pub(super) fn keep_capture(
        &mut self,
        capturable: bool,
        recorded: (&'static str, Assembly, DrawCall),
        index: Option<&DrawSource<'_>>,
    ) {
        if self.rerouting {
            return;
        }
        let Keeping::Draws(draws, records) = &self.keeping else {
            return;
        };
        let added = counts(&recorded.2).and_then(|(count, instances)| count.checked_mul(instances));
        let total = added.and_then(|added| added.checked_add(*records));
        let room = draws.len() < MAX_DRAWS && total.is_some_and(|total| total <= MAX_RECORDS);
        let kept = match capturable && room {
            true => self.kept_draw(recorded, index),
            false => None,
        };
        match (kept, total, &mut self.keeping) {
            (Some(kept), Some(total), Keeping::Draws(draws, records)) => {
                draws.push(kept);
                *records = total;
            }
            _ => self.keeping = Keeping::Stopped,
        }
    }

    /// The capture draw recording now: its record and index addresses, where the encoder
    /// records one (`draw_resident`).
    pub(super) fn capture_addresses(&self) -> Option<(u64, Option<u64>)> {
        match &self.keeping {
            Keeping::Recording(capturing) => Some((capturing.records, capturing.indices)),
            _ => None,
        }
    }

    /// Note that the capture draw recording now records nothing (its capture form's objects or
    /// pipeline not made yet): whether one is recording.
    pub(super) fn capture_missed(&mut self) -> bool {
        match &mut self.keeping {
            Keeping::Recording(capturing) => {
                capturing.missed = true;
                true
            }
            _ => false,
        }
    }

    /// Note `uses`, the capture draw's recording now, among the capture draws'.
    pub(super) fn capture_uses(&mut self, uses: CommandUses) {
        if let Keeping::Recording(capturing) = &mut self.keeping {
            capturing.uses.merge(uses);
        }
    }

    /// As the pass ends, before its steps: record its kept draws again as capture draws and,
    /// before the pass, the steps that bin their primitives into a tile mask (the module docs),
    /// where the pass's draws reach some tiles and a load pass rewrites them. `None` where it
    /// does not, where a kept draw's capture form or pipeline is not made yet, or where the
    /// device cannot: the pass's reach is then the host's, announced as before.
    pub(super) fn capture_reach(&mut self) -> Option<CapturedReach> {
        let reach = PassReach::of(&self.layout, self.called, self.drew);
        let drawn = self.drawn.tiles(&self.layout);
        let reaches = drawn.as_ref().is_none_or(|tiles| !tiles.is_empty());
        let consumed = super::unused_bits_load::consumes_capture(&self.layout, true);
        if !self.rewrites_unused_bits || reach != PassReach::Draws || !reaches || !consumed {
            return None;
        }
        let Keeping::Draws(kept, records) = &self.keeping else {
            return None;
        };
        if kept.is_empty() {
            return None;
        }
        let records = *records;
        let mut placed = Vec::with_capacity(kept.len());
        let mut states = Vec::with_capacity(kept.len() * TileReachModule::STATE_WORDS as usize * 4);
        let mut first = 0u64;
        for draw in kept {
            let (count, instances) = counts(&draw.draw())?;
            let Assembly::Primitives(topology) = draw.assembly() else {
                return None;
            };
            let state = draw.state();
            states.extend(state.reach_state()?);
            placed.push(Placed {
                first,
                count,
                instances,
                topology,
                restarts: matches!(draw.draw(), DrawCall::Indexed { .. })
                    && topology == PrimitiveType::TriangleStrip,
                culls: state.cull_mode() != CullMode::None,
            });
            first += count * instances;
        }
        if first != records {
            return None;
        }
        // The entries, one an instance of each draw of some primitives, and the primitives'.
        let mut entries = Vec::new();
        let mut total = 0u32;
        let mut count = 0u32;
        for (state, draw) in placed.iter().enumerate() {
            let records = u32::try_from(draw.count).ok()?;
            let primitives = CaptureAssemblyModule::primitives(draw.topology, records);
            if primitives == 0 {
                continue;
            }
            for n in 0..draw.instances {
                let at = u32::try_from(draw.first + n * draw.count).ok()?;
                entries.extend(CaptureAssemblyModule::draw(
                    total,
                    at,
                    records,
                    draw.topology,
                    draw.restarts,
                    draw.culls,
                    u32::try_from(state).ok()?,
                ));
                total = total.checked_add(primitives)?;
                count += 1;
            }
        }
        if total == 0 {
            return None;
        }
        let steps = self.capture_steps(records, total)?;
        let recorded = self.record_captures(&steps)?;
        let mut draws = CaptureAssemblyModule::header(count, total).to_vec();
        draws.extend(entries);
        let area = self.layout.extent;
        let mut pass = TileReachModule::header(area, 1, 1, total).to_vec();
        pass.extend(states);
        // The tiles the host's reach names, which the load pass would rewrite without the
        // capture: counted against the mask's once the command buffer completes.
        let grid = u64::from(area[0].div_ceil(32)) * u64::from(area[1].div_ceil(32));
        let bound = drawn.as_ref().map_or(grid, DrawnTiles::count);
        self.record_capture_steps(steps, recorded, (&draws, &pass), (grid, bound))
    }

    /// The scratch and kernels of a pass's capture of `records` records and `total`
    /// primitives: `None` where the device makes no kernel (the tile reach's `shaderInt64`, among
    /// others) or gives no device address.
    fn capture_steps(&mut self, records: u64, total: u32) -> Option<CaptureSteps> {
        let device = self.core.device().clone();
        let kernels = [
            OwnedModule::CaptureAssembly,
            OwnedModule::TileReach,
            OwnedModule::CaptureSettle,
        ]
        .map(|which| device.owned_kernel(which).ok());
        let [Some(assembly), Some(reach), Some(settle)] = kernels else {
            return None;
        };
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(32);
        let mask_words = TileReachModule::mask_words(self.layout.extent, 1, 1);
        let groups = |n: u32, local: u32| n.div_ceil(local).max(1);
        let most = limits.max_compute_work_group_count[0];
        if groups(total, CaptureAssemblyModule::LOCAL_SIZE) > most
            || groups(total, TileReachModule::LOCAL_SIZE) > most
            || groups(mask_words, CaptureSettleModule::LOCAL_SIZE) > most
        {
            return None;
        }
        let records_bytes = records * u64::from(CaptureAssemblyModule::RECORD_WORDS) * 4;
        let primitives_bytes = u64::from(total) * u64::from(TileReachModule::PRIMITIVE_WORDS) * 4;
        let mask_bytes = u64::from(mask_words) * 4;
        let range = u64::from(limits.max_storage_buffer_range);
        if records_bytes > range || primitives_bytes > range || mask_bytes > range {
            return None;
        }
        let primitives_at = records_bytes.next_multiple_of(align);
        let mask_at = (primitives_at + primitives_bytes).next_multiple_of(align);
        let scratch = device
            .own_buffer(mask_at + mask_bytes, ResourceOptions::private())
            .ok()?;
        let (raw, at) = scratch.at(0);
        let address = scratch.device_address()?;
        if !at.is_multiple_of(align) || !address.is_multiple_of(32) {
            return None;
        }
        Some(CaptureSteps {
            scratch,
            raw,
            records: (at, records_bytes, address),
            primitives: (at + primitives_at, primitives_bytes),
            mask: (at + mask_at, mask_bytes),
            kernels: [assembly, reach, settle],
            total,
            mask_words,
        })
    }

    /// Record the pass's kept draws again as capture draws, each through its capture form into
    /// its records in `steps`' scratch, into an instance of a pass without attachments: the
    /// instance and the draws' uses, or `None` where a draw records nothing (its capture form's
    /// objects or pipeline not made yet, or a refusal). The encoder's current instance stays
    /// the pass's; the pass keeps no draw from here on.
    fn record_captures(&mut self, steps: &CaptureSteps) -> Option<(Instance, CommandUses)> {
        let device = self.core.device().clone();
        let shape = attachmentless::shape(1, 0);
        let compatible = Arc::new(render_target::compatible_pass(&device, &shape).ok()?);
        let draws = self.core.command_buffer().begin_draws(&compatible).ok()?;
        let instance = Instance {
            draws,
            shape,
            compatible,
            recorded: false,
            shadow: false,
            shader_tests: false,
            imageblock: false,
            interlocked_writes: InterlockedStorage::NONE,
            indirect_reads: IndirectReads::new(),
            geometry_writes: GeometryWrites::new(),
            tile_memory: false,
            barriers: false,
            views: ViewSetup::default(),
            pieces: Vec::new(),
            shadows: InstanceShadows::default(),
            before: Vec::new(),
            reformats: Vec::new(),
            positioned: None,
            rerecord: None,
            bound: Bound::default(),
            drawn: DrawnArea::default(),
            inline: false,
            continues: false,
            kept: None,
            planned: false,
            feedback: Feedback::NONE,
            storage_colors: 0,
        };
        let (_, _, base) = steps.records;
        let recording = Keeping::Recording(Capturing {
            records: base,
            indices: None,
            uses: CommandUses::default(),
            missed: false,
        });
        // The kept draws leave the pass's keeping for the capture draws' recording, the pass's
        // instance for the capture's: each a state the encoder moves on to, given back below.
        let Keeping::Draws(mut kept, _) = std::mem::replace(&mut self.keeping, recording) else {
            return None;
        };
        let current = std::mem::replace(&mut self.current, instance);
        let mut first = 0u64;
        let mut recorded = true;
        for draw in kept.iter_mut() {
            let Some((count, instances)) = counts(&draw.draw()) else {
                recorded = false;
                break;
            };
            let indices = match draw.index() {
                None => Some(None),
                Some(index) => index.address().map(Some),
            };
            let (Some(indices), Keeping::Recording(capturing)) = (indices, &mut self.keeping)
            else {
                recorded = false;
                break;
            };
            capturing.records = base + first * u64::from(CaptureAssemblyModule::RECORD_WORDS) * 4;
            capturing.indices = indices;
            let again = self.record_again(draw);
            let missed = matches!(&self.keeping, Keeping::Recording(c) if c.missed);
            if again.is_err() || missed {
                recorded = false;
                break;
            }
            first += count * instances;
        }
        let instance = std::mem::replace(&mut self.current, current);
        // The pass ends: it keeps no draw from here on.
        let Keeping::Recording(capturing) = std::mem::replace(&mut self.keeping, Keeping::Stopped)
        else {
            self.end_unrun(instance);
            return None;
        };
        if !recorded {
            // A capture that records nothing is never executed.
            self.end_unrun(instance);
            return None;
        }
        Some((instance, capturing.uses))
    }

    /// Record, into the encoder's command buffer before the pass, the capture draws' instance
    /// `recorded` and the kernels over `steps`' scratch: the records the draws write, assembled
    /// into the primitives (the draws' `draws` bytes), binned into the mask under the pass's
    /// header and states (`pass`), and settled.
    fn record_capture_steps(
        &mut self,
        steps: CaptureSteps,
        (instance, uses): (Instance, CommandUses),
        (draws, pass): (&[u8], &[u8]),
        (grid, bound): (u64, u64),
    ) -> Option<CapturedReach> {
        let device = self.core.device().clone();
        // SAFETY: only this encoder records the secondary command buffer.
        unsafe { self.core.command_buffer().end_raw(instance.draws) }.ok()?;
        let framebuffer = render_target::attachmentless_framebuffer(
            &device,
            &instance.compatible,
            self.layout.extent,
        )
        .ok()?;
        let (draws_upload, pass_upload, status, tally) = {
            let mut state = self.core.command_buffer().state();
            let draws_upload = state
                .uploads
                .upload(&device, draws, draws.len() as u64)
                .ok()?;
            let pass_upload = state
                .uploads
                .upload(&device, pass, pass.len() as u64)
                .ok()?;
            let status = state.status_words(&device).ok()?;
            let tally = state.uploads.upload(&device, &[], steps.mask.1).ok()?;
            (draws_upload, pass_upload, status, tally)
        };
        let region = |upload: &crate::upload::Upload| (upload.raw(), upload.offset(), upload.len());
        let (records_at, records_bytes, _) = steps.records;
        let records = (steps.raw, records_at, records_bytes);
        let primitives = (steps.raw, steps.primitives.0, steps.primitives.1);
        let mask = (steps.raw, steps.mask.0, steps.mask.1);
        let status = (status.raw(), status.offset(), ExecutionStatus::BYTES as u64);
        let [assembly, reach, settle] = steps.kernels;
        let sets = OwnedSets::write_each(
            &device,
            &[assembly, reach, settle],
            &[
                buffers(vec![records, region(&draws_upload), primitives]),
                buffers(vec![primitives, region(&pass_upload), mask]),
                buffers(vec![status, region(&pass_upload), mask]),
            ],
        )
        .ok()?;
        let descriptor_sets = sets.sets.clone();
        let groups = [
            steps
                .total
                .div_ceil(CaptureAssemblyModule::LOCAL_SIZE)
                .max(1),
            steps.total.div_ceil(TileReachModule::LOCAL_SIZE).max(1),
            steps
                .mask_words
                .div_ceil(CaptureSettleModule::LOCAL_SIZE)
                .max(1),
        ];
        let [width, height] = self.layout.extent;
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let (pass_raw, framebuffer_raw, capture_draws) =
            (instance.compatible.raw, framebuffer.raw, instance.draws);
        let stages = uses.iter().fold(CAPTURE_STAGES, |all, used| {
            all | used.read.stages | used.write.stages
        });
        let (mask_raw, mask_at, mask_bytes) = mask;
        let tally_copy = vk::BufferCopy {
            src_offset: mask_at,
            dst_offset: tally.offset(),
            size: mask_bytes,
        };
        let tally_raw = tally.raw();
        self.core
            .command_reaching(stages, &uses, uses.resident(), |raw, command_buffer| {
                // SAFETY: the encoder's command buffer is recording outside a render pass (the
                // pass records after it), and only the encoder records it; the pass without
                // attachments, its framebuffer, the capture draws' secondary command buffer
                // (recorded against that pass), the scratch, the uploads, the status words and
                // the sets are live until the command buffer completes (retained below, the
                // uploads and status words with the command buffer); every binding lies inside
                // its buffer, and each step reads what the one before it wrote past a barrier.
                unsafe {
                    Barrier::everything().record(raw, command_buffer);
                    raw.cmd_fill_buffer(command_buffer, mask_raw, mask_at, mask_bytes, 0);
                    let begin = vk::RenderPassBeginInfo::default()
                        .render_pass(pass_raw)
                        .framebuffer(framebuffer_raw)
                        .render_area(render_area);
                    raw.cmd_begin_render_pass(
                        command_buffer,
                        &begin,
                        vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
                    );
                    raw.cmd_execute_commands(command_buffer, &[capture_draws]);
                    raw.cmd_end_render_pass(command_buffer);
                    for ((kernel, set), groups) in
                        steps.kernels.iter().zip(&descriptor_sets).zip(groups)
                    {
                        Barrier::everything().record(raw, command_buffer);
                        raw.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.pipeline,
                        );
                        raw.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.layout,
                            0,
                            std::slice::from_ref(set),
                            &[],
                        );
                        raw.cmd_dispatch(command_buffer, groups, 1, 1);
                    }
                    Barrier::everything().record(raw, command_buffer);
                    // The mask where the host counts it once the command buffer completes.
                    raw.cmd_copy_buffer(command_buffer, mask_raw, tally_raw, &[tally_copy]);
                }
            });
        self.core.retain(steps.scratch);
        self.core.retain(sets);
        self.core.retain(framebuffer);
        self.core.retain(instance.compatible);
        device
            .shared()
            .counts
            .capture_reach_recorded(u64::from(steps.total));
        let tally = CaptureTally {
            mask: tally,
            grid,
            bound,
        };
        self.core
            .command_buffer()
            .state()
            .capture_tallies
            .push(tally);
        Some(CapturedReach { mask })
    }
}

/// A set of storage buffers alone, `(buffer, offset, range)` in binding order.
fn buffers(buffers: Vec<(vk::Buffer, u64, u64)>) -> OwnedSet {
    OwnedSet {
        images: Vec::new(),
        storage_images: Vec::new(),
        buffers,
    }
}

/// The stages the capture steps run in besides their draws' uses'.
const CAPTURE_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::TRANSFER.as_raw()
        | vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw(),
);

/// A pass's capture scratch and kernels (`RenderCommandEncoder::capture_steps`).
struct CaptureSteps {
    scratch: crate::Buffer,
    /// The scratch's `VkBuffer`.
    raw: vk::Buffer,
    /// The records' offset in `raw`, their bytes, and their device address.
    records: (u64, u64, u64),
    /// The primitives' offset in `raw` and bytes.
    primitives: (u64, u64),
    /// The mask's offset in `raw` and bytes.
    mask: (u64, u64),
    /// The assembly's, the tile reach's and the settle's kernels.
    kernels: [OwnedKernel; 3],
    /// The primitives, and the mask's words.
    total: u32,
    mask_words: u32,
}
