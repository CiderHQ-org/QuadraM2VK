//! A pass's first render pass instance recorded inline: its draws record straight into a
//! primary command buffer of the encoder's, in a render pass instance begun at the pass's first
//! draw, in place of a secondary command buffer the pass executes at its end.
//!
//! A secondary per instance costs GPU time and CPU on every driver measured: on ms01's Iris Xe
//! (ANV) about 11 us of GPU and 8 us of recording CPU per pass, on a Radeon 780M (RADV) and an
//! RTX 5080 about 1 to 3 us of each, whatever the draw count. Everything else of the pass records
//! as before, at its end; only where the first instance's commands go changes.
//!
//! - **The command buffer.** The pass takes the encoder's own command buffer where nothing has
//!   recorded into it yet, else a new one after it (`EncoderCore::hold_pass`). Until the pass
//!   ends, the encoder's commands before the pass (a draw's copy or index pass, and what the end
//!   records before the pass: clears, copies into stand-ins and shadows, the visibility reset)
//!   record into another command buffer, begun as the first comes, which goes before the pass's
//!   in submission order: where they go before a pass recorded at its end. The pass's end, its
//!   later instances and the steps between and after them record into the pass's command buffer
//!   after the first instance (`EncoderCore::release_pass`). So every command keeps its place in
//!   submission order.
//! - **At the open: the layout allows it** ([`opens_inline`]). The first instance's render pass,
//!   framebuffer and clear values must be what the pass's end would make: no step the end decides
//!   from the whole pass may change them. So: one layer of one sample, no depth or stencil
//!   attachment, no rasterization-rate map, custom sample positions or visibility buffer, and
//!   every color attachment the texture's own image or its image over a buffer's bytes
//!   (`Texture::image_over_buffer`; not a texture buffer, a shadow, a stand-in while the bytes
//!   are detached, or a bound linear image), not memoryless, of no extended-range, padded or
//!   bfloat format, with no resolve or stand-in, loaded `Load` or `Clear` (a `Clear` reaching
//!   no farther than the render area, and of no NaN the device must clear apart) and stored
//!   `Store`. Then the end changes none of the first instance's load and
//!   store operations (`render_target::final_pass` stores every attachment of an instance that
//!   is not the last; so does the last here, and `kept_bytes.rs` keeps a stored attachment in
//!   place), its attachments or its clear values.
//! - **Bytes over host pages** that an attachment lies in may be detached or rebound while the
//!   pass is open (`detach.rs`, `rebind.rs`), which swaps its images out from under the
//!   framebuffer the first instance recorded with. The pass holds those bytes from its open, as
//!   a command that names them does: a rebind meanwhile is refused (`EncodedUses`), and a
//!   detach's old images and views go with the old pages, which the hold keeps until the command
//!   buffer goes. So the command buffer the pass records into stays in the recording state:
//!   "Command buffers in any state other than the pending state will transition to the invalid
//!   state if any memory or object, other than VkCommandPool, that may be accessed when the
//!   command buffer is accessed (e.g. an object bound to the command buffer) is destroyed or
//!   freed", and then only "A command buffer in the invalid state can be reset or freed" (the
//!   command buffer lifecycle). A command buffer whose pass named detached bytes never runs
//!   (`EncoderCore::skips`).
//! - **No secondary at the open.** Such a pass begins no secondary for its first instance as it
//!   opens: one begins where a command records into that instance's secondary (a draw that does
//!   not record inline, a visibility query, an execution of render commands) or a split keeps the
//!   instance before the next ([`RenderCommandEncoder::begin_current_draws`]). An instance a new
//!   one replaces, or that the pass ends with nothing recorded, never begins one, and its render
//!   pass instance runs its loads and stores alone. A secondary begun and ended unused cost every
//!   such pass a begin, which on ANV resets the command buffer's whole command state.
//! - **At the first draw that records: the instance allows it**
//!   ([`RenderCommandEncoder::inline_begin`]). Its pipeline is made; the instance is the pass's
//!   first, of no fetched (storage) attachment, view of another format, multiview or
//!   shader-side test; nothing the pass's end records inside it is pending: no draw whose
//!   pipeline is being made, execution of render commands, tile dispatch, fragment imageblock or
//!   shader-side stencil or depth, and no stand-in arrives with the draw, which reads no texture
//!   of the pass through a snapshot (`render/snapshot.rs`). Otherwise the pass
//!   records as before, with its first instance in a secondary.
//! - **Later draws.** A draw that needs a new instance splits the pass: the new instance records
//!   inline too where it can, after the inline one (`render/inline_splits.rs`); else as before,
//!   the inline instances staying open in their command buffer, and the pass's end ends the last
//!   there before the steps after it. A draw whose pipeline is being made, or an execution of
//!   render commands, which run as secondaries between an instance's own draws, records into a
//!   new instance after the inline one, as after a split. The inline instance records no draw
//!   again (`render/reroute.rs`): a draw that would reroute it splits the pass, as on a device
//!   that splits such passes. A draw's steps before the first instance (`Before`) and its
//!   shadowed textures' copies record before the pass, in the command buffer before the pass's,
//!   as the end records them before the first instance.
//! - **A pass that executes render commands** records later, as a segment of its encoder
//!   (`render/executions.rs`), which records at the commit what runs before the pass (the
//!   executions' copies of bytes the pass's fragments wrote, an emulated mesh command's
//!   dispatches) and then its instances. The pass's command buffer, its first instance and the
//!   steps after it recorded, ends as the encoder ends the pass and goes over to the segment
//!   (`EncoderCore::hand_over_pass`), which places it after what it records before the pass and
//!   before the later instances (`EncoderCore::cut_before`): the first instance runs after them,
//!   as where the segment records it.
//! - **A stand-in that arrives later** (a fetched attachment whose image lacks the usage the
//!   fetch needs) stands in from the next instance on: the end copies each attachment into its
//!   stand-in right after the inline instance, which rendered the attachment itself, not before
//!   the pass, and back after the last as before.
//! - **Counted**: `WorkCounts::inline_passes` (passes whose first instance recorded inline) and
//!   `WorkCounts::inline_passes_split` (those of more than one instance), and
//!   `WorkCounts::inline_split_instances` (`render/inline_splits.rs`).

use super::extended_shadow::ExtendedShadows;
use super::inline_splits::Later;
use super::{Instance, Next, RenderCommandEncoder, ViewSetup};
use crate::device::Device;
use crate::error::Error;
use crate::linear_attachment;
use crate::nan_clear::keeps_nan_sign;
use crate::render_pass::PassLayout;
use crate::render_target::{
    self, Feedback, FramebufferObject, PassShape, Position, RenderPassObject,
};
use crate::render_values::{LoadAction, StoreAction};
use ash::vk;
use metal2vulkan::metal::format::ExtendedRange;
use std::sync::Arc;

/// Whether a pass's first instance records inline (the module docs).
pub(super) enum Inline {
    /// It does not: the pass's layout or its first draw ruled it out, or the encoder encodes an
    /// ended encoder's calls again.
    Never,
    /// The layout allows it; the pass's first draw that records decides. `splits`: the layout
    /// lets its instances after a split record inline too (`render/inline_splits.rs`).
    Open { splits: bool },
    /// It records inline, as [`Begun`] holds.
    Begun(Box<Begun>),
}

/// A pass whose first instance records inline: its render pass and framebuffer, which the
/// encoder retains, and what an instance after a split that records inline takes
/// (`render/inline_splits.rs`).
pub(super) struct Begun {
    pub(super) pass: RenderPassObject,
    pub(super) framebuffer: FramebufferObject,
    /// The first instance's shape, which such an instance has.
    pub(super) shape: PassShape,
    /// The layout lets such instances record inline.
    pub(super) splits: bool,
    /// Their render pass, made at the first.
    pub(super) later: Option<Later>,
}

impl Begun {
    /// The objects the pass's inline instances recorded with, which the encoder retains until
    /// the command buffer completes: the first's render pass and framebuffer, and the later
    /// ones' render pass where one was made.
    pub(super) fn objects(self) -> (RenderPassObject, FramebufferObject, Option<Later>) {
        (self.pass, self.framebuffer, self.later)
    }
}

/// What the first draw that records an inline instance made for it before recording starts
/// ([`RenderCommandEncoder::inline_begin`]): its render pass and framebuffer, the command
/// buffer the encoder holds for it, and the instance's shape and the layout's `splits`
/// ([`Begun`]).
pub(super) struct InlineBegin {
    pass: RenderPassObject,
    framebuffer: FramebufferObject,
    held: vk::CommandBuffer,
    shape: PassShape,
    splits: bool,
}

/// Whether a pass of `layout` on `device`, rate-mapped where `rate_mapped` says, with the
/// extended-range shadows `extended`, lets its first instance record inline (the module docs).
pub(super) fn opens_inline(
    device: &Device,
    layout: &PassLayout,
    rate_mapped: bool,
    extended: &ExtendedShadows,
) -> bool {
    let plain = !rate_mapped
        && layout.sample_positions.is_empty()
        && layout.layers <= 1
        && layout.samples == 1
        && layout.visibility.is_none()
        && layout.depth_stencil.is_none()
        && !extended.keeps_draws();
    plain
        && layout.colors.iter().flatten().next().is_some()
        && layout.colors.iter().flatten().all(|color| {
            let target = &color.target;
            let texture = &target.texture;
            let cleared = match color.load {
                LoadAction::Load => true,
                LoadAction::Clear => {
                    let signed = keeps_nan_sign(target, true);
                    let format = texture.pixel_format();
                    crate::clear_reach::cleared(layout, target.extent, true) == layout.extent
                        && crate::clear_value::nan_texel(format, &color.clear, signed).is_none()
                }
                LoadAction::DontCare => false,
            };
            cleared
                && color.store == StoreAction::Store
                && !color.late_store
                && !color.memoryless
                && !super::memoryless(target)
                && color.resolve.is_none()
                && color.copied_resolve.is_none()
                && color.stand_in.is_none()
                && color.planes.is_none()
                && color.straight_clear.is_none()
                && (!texture.over_buffer() || texture.image_over_buffer())
                && ExtendedRange::of(texture.pixel_format()).is_none()
                && linear_attachment::bound(device, target).is_ok_and(|bound| !bound)
        })
}

impl RenderCommandEncoder {
    /// Begin the current instance's secondary where the pass left it unbegun (`Instance::draws`):
    /// a pass that may record its first instance inline begins none at its opening, and one only
    /// where a command records into that instance's secondary or a split keeps the instance. An
    /// instance that records inline takes none: the pass's first, recording into the pass's
    /// command buffer, and one after a split whose commands are kept (`render/inline_splits.rs`).
    pub(super) fn begin_current_draws(&mut self) -> Result<(), Error> {
        if self.current.draws == vk::CommandBuffer::null() && !self.current.inline {
            let compatible = Arc::clone(&self.current.compatible);
            self.current.draws = self.core.command_buffer().begin_draws(&compatible)?;
        }
        Ok(())
    }

    /// Where the pass's first instance may record inline (`Inline::Open`) and the draw about to
    /// record into `next` is its first that records (`records`), decide: where the instance
    /// allows it (the module docs; `made` its pipeline, `capture` a capture draw, `stood` a
    /// stand-in arriving with it), its render pass and framebuffer, and the command buffer the
    /// encoder holds for it, for [`begin_inline`](Self::begin_inline) once the draw installed
    /// its instance; else `None`, and the pass records as before: so for a draw whose `feedback`
    /// colors its instance holds in the feedback loop layout, which the pass's first render pass
    /// instance would begin without (`render/snapshot.rs`). Refused only where an object
    /// cannot be made. Where it does not record inline, the current instance's secondary, unbegun
    /// at the pass's opening (`Instance::draws`), begins here where the draw records into that
    /// instance, or a split keeps it before the next; a new instance that takes its place
    /// leaves it unbegun.
    pub(super) fn inline_begin(
        &mut self,
        next: &Next,
        stood: Option<&PassLayout>,
        (records, made, capture): (bool, bool, bool),
        feedback: Feedback,
    ) -> Result<Option<InlineBegin>, Error> {
        let begin = match feedback == Feedback::NONE {
            true => self.decide_inline(next, stood, (records, made, capture))?,
            false => {
                // Decided at this draw: the pass records its first instance in a secondary.
                if matches!(self.inline, Inline::Open { .. }) && records && !capture {
                    self.inline = Inline::Never;
                }
                None
            }
        };
        let keeps = match next {
            Next::Current => records,
            Next::New { split, .. } => *split,
        };
        if begin.is_none() && keeps {
            self.begin_current_draws()?;
        }
        Ok(begin)
    }

    /// [`inline_begin`](Self::inline_begin)'s decision.
    fn decide_inline(
        &mut self,
        next: &Next,
        stood: Option<&PassLayout>,
        (records, made, capture): (bool, bool, bool),
    ) -> Result<Option<InlineBegin>, Error> {
        let Inline::Open { splits } = self.inline else {
            return Ok(None);
        };
        if !records || capture {
            return Ok(None);
        }
        // Decided at this draw, either way.
        self.inline = Inline::Never;
        let instance: &Instance = match next {
            Next::Current => &self.current,
            Next::New {
                instance,
                split: false,
            } => instance,
            Next::New { split: true, .. } => return Ok(None),
        };
        let plain = made
            && stood.is_none()
            && !self.replaying
            && !self.rerouting
            && self.earlier.is_empty()
            && self.pending.is_empty()
            && self.executions.is_empty()
            && self.dispatches.is_empty()
            && !self.dispatched
            && self.imageblock.is_none()
            && !self.tile_imageblock
            && self.shadow.is_none()
            && self.depth_stand_in.is_none()
            && !self.visibility.active()
            && !instance.recorded
            && instance.pieces.is_empty()
            && instance.shape.storage.is_empty()
            && instance.reformats.is_empty()
            && instance.views == ViewSetup::default()
            && !instance.shadow
            && !instance.shader_tests;
        if !plain {
            return Ok(None);
        }
        let device = self.core.device().clone();
        self.kept_attachments = None;
        let at = Position {
            first: true,
            last: false,
        };
        // The pass has no depth/stencil attachment (`opens_inline`): every attachment in
        // `GENERAL`.
        let (none, general) = (Feedback::NONE, vk::ImageLayout::GENERAL);
        let shape = &instance.shape;
        let pass = render_target::final_pass(&device, &self.layout, shape, at, none, general)?;
        let held = self.core.hold_pass()?;
        Ok(Some(InlineBegin {
            pass,
            framebuffer,
            held,
            shape: instance.shape.clone(),
            splits,
        }))
    }

    /// Begin the pass's first instance inline as [`inline_begin`](Self::inline_begin) made it,
    /// the draw's instance installed: its render pass instance begins in the held command
    /// buffer, whose subpass runs its draws inline, and the secondary it was given, where one
    /// began (a new instance's, `Instance::draws`), ends unused.
    pub(super) fn begin_inline(&mut self, begin: InlineBegin) {
        let InlineBegin {
            pass,
            framebuffer,
            held,
            shape,
            splits,
        } = begin;
        let device = self.core.device().clone();
        let raw = device.shared().vk();
        if self.current.draws != vk::CommandBuffer::null() {
            // SAFETY: the secondary is recording, only this encoder records it, and nothing
            // recorded into it: no draw, and no query (the pass has no visibility buffer).
            let ended = unsafe { self.core.command_buffer().end_raw(self.current.draws) };
            if ended.is_ok() {
                device.shared().counts.unrun_instance_ended();
            }
        }
        let [width, height] = self.layout.extent;
        let clear_values = render_target::clear_values(&self.layout);
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(pass.raw)
            .framebuffer(framebuffer.raw)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            })
            .clear_values(clear_values.as_slice());
        // SAFETY: `held` is recording outside a render pass, held for this pass alone; the pass
        // and framebuffer are live until the command buffer completes (the encoder holds them
        // to its end, which retains them), and every image rests in `GENERAL`.
        unsafe { raw.cmd_begin_render_pass(held, &info, vk::SubpassContents::INLINE) };
        self.current.draws = held;
        self.current.inline = true;
        self.current.rerecord = None;
        self.inline = Inline::Begun(Box::new(Begun {
            pass,
            framebuffer,
            shape,
            splits,
            later: None,
        }));
        device.shared().counts.inline_pass();
    }
}
