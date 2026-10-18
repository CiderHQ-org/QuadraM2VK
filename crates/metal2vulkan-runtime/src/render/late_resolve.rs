//! Late resolve targets: the render encoder's `set_color_resolve_target`,
//! `set_depth_resolve_target` and `set_stencil_resolve_target`, which replace an attachment's
//! resolve texture, slice, depth plane and level before the pass ends.
//!
//! Metal has them only as private calls (`setColorResolveTexture:slice:depthPlane:level:yInvert:
//! atIndex:` and its depth and stencil twins). Measured on an M4 with macOS 27.0.1 (Metal probes,
//! with and without API validation): the texture set last before `endEncoding` is resolved into, at
//! its slice and level, and holds every draw of the pass, those after the call too; the
//! descriptor's resolve texture keeps its texels; a depth or stencil resolve is written whether or
//! not the descriptor named one; `yInvert` changes nothing; and Metal's validation stops a resolve
//! texture of another pixel format or size than the attachment, and an index without an attachment.
//! A color resolve set where the descriptor named none writes nothing.
//!
//! These setters are a generic extension, not copies of Apple's calls: they take no `yInvert`,
//! and each does what it says, so a color target set where the descriptor named none is resolved
//! into (keeping Metal's quirk is the consumer's). `None` sets no target. A target is checked as
//! the descriptor's resolve texture is (`resolve.rs`): against the attachment, and against its
//! store action where that is known (one that resolves needs a target, and a target one that
//! resolves); a late store action that resolves needs the target set when it is, and one that does
//! not drops the attachment's resolve, as plain Metal leaves a resolve texture under such an action
//! untouched. Each refusal is the descriptor's, typed, and leaves the target set
//! before.
//!
//! **The mechanism.** The encoder begins the pass at `end_encoding` (the deferred pass begin,
//! `render.rs`), so a target only replaces the attachment's resolve in the pass's layout, and
//! `end_encoding` gives every instance's shape the resolves the pass ends with ([`settle`]). The
//! draws recorded against compatible passes of the resolves before: Vulkan ignores resolve
//! attachments when it compares passes of one subpass, and every instance's pass has one, a
//! depth/stencil resolve the device makes one aspect at a time resolving its second aspect in a
//! pass of its own after them (`render_target.rs`).

use super::RenderCall;
use super::RenderCommandEncoder;
use crate::argument_table::check_index;
use crate::command_buffer::CommandBuffer;
use crate::error::{invalid, Error, Problem};
use crate::render_pass::{
    Attachment, DepthStencilTarget, RenderPassProblem, MAX_COLOR_ATTACHMENTS,
};
use crate::render_target::PassShape;
use crate::render_values::StoreAction;
use crate::resolve::{self, DepthStencilResolve, Wanted};
use crate::texture::Texture;

/// Where an attachment resolves: a single-sample texture at `level`, `slice` and `depth_plane`,
/// as a pass descriptor's resolve texture is named. A generic extension standing for Apple's
/// private `setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:` and its depth and
/// stencil twins, without `yInvert`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ResolveTarget {
    /// The single-sample texture resolved into (`resolveTexture`).
    pub texture: Texture,
    /// Its slice or cube face (`resolveSlice`).
    pub slice: u64,
    /// Its depth plane, for a 3D texture (`resolveDepthPlane`).
    pub depth_plane: u64,
    /// Its mipmap level (`resolveLevel`).
    pub level: u64,
}

impl ResolveTarget {
    /// `texture` at its level 0, slice 0 and depth plane 0.
    pub fn new(texture: Texture) -> ResolveTarget {
        ResolveTarget {
            texture,
            slice: 0,
            depth_plane: 0,
            level: 0,
        }
    }

    /// `texture` at `(level, slice, depth_plane)`, as a descriptor's resolve fields name it.
    pub(crate) fn named(
        texture: &Option<Texture>,
        (level, slice, depth_plane): (u64, u64, u64),
    ) -> Option<ResolveTarget> {
        texture.as_ref().map(|texture| ResolveTarget {
            texture: texture.clone(),
            slice,
            depth_plane,
            level,
        })
    }
}

/// The resolve `target` names for an attachment stored by `store`.
fn wanted(target: Option<&ResolveTarget>, store: StoreAction) -> Wanted<'_> {
    Wanted {
        texture: target.map(|t| &t.texture),
        level: target.map_or(0, |t| t.level),
        slice: target.map_or(0, |t| t.slice),
        depth_plane: target.map_or(0, |t| t.depth_plane),
        store,
    }
}

impl RenderCommandEncoder {
    /// The color attachment at `index` resolves into `target` (`None`: nowhere), whatever the
    /// descriptor named (the module docs). Refused (`Invalid`): an index past the attachments,
    /// an attachment without a texture, and a target the descriptor's resolve texture would be
    /// refused as (Metal's validation stops another pixel format or size, measured).
    pub fn set_color_resolve_target(
        &mut self,
        target: Option<ResolveTarget>,
        index: u32,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::ColorResolveTarget(target.clone().map(Box::new), index));
        const CALL: &str = "set_color_resolve_target";
        check_index(CALL, index, MAX_COLOR_ATTACHMENTS as u32)?;
        let which = Attachment::Color(index);
        let maker = self.core.command_buffer().clone();
        // In a pass of several color groups, no draw read the target before
        // (`render/color_groups.rs`).
        self.check_group_resolve(target.as_ref().map(|t| &t.texture))?;
        // Its color group's, where the pass draws in groups; a later group's encoder sets only
        // its own colors.
        let later = self.later_group_encoder();
        let color = match self.color_slot(index) {
            Some(Some(color)) => color,
            None if later => return Ok(()),
            _ => return Err(problem(CALL, RenderPassProblem::NoAttachment(which))),
        };
        // A late target keeps the rules Metal's validation states (plain Metal's answer is
        // unmeasured): a resolving store action needs a target, and a target a store action that
        // resolves or is `Unknown`. The descriptor's color resolve takes either, as plain Metal
        // does (`resolve::resolves`).
        match (&target, color.store) {
            (None, store) if store.resolves() => {
                return Err(problem(
                    CALL,
                    RenderPassProblem::ResolveWithoutTexture(which),
                ));
            }
            (Some(_), store) if !store.resolves() && store != StoreAction::Unknown => {
                return Err(problem(CALL, RenderPassProblem::ResolveStoreAction(which)));
            }
            _ => {}
        }
        let resolve = resolve::color(
            CALL,
            &maker,
            which,
            &color.target,
            &wanted(target.as_ref(), color.store),
            (color.load, color.clear),
        )?;
        color.resolve = resolve;
        // A single-sampled attachment's copied resolve is replaced too (a late target of one is
        // refused, `ResolveSourceSamples`: plain Metal's answer is unmeasured).
        color.copied_resolve = None;
        Ok(())
    }

    /// The depth attachment resolves into `target` (`None`: nowhere), whatever the descriptor
    /// named. Refused as [`RenderCommandEncoder::set_color_resolve_target`] is, and as the
    /// descriptor's depth resolve is where the stencil attachment is the same texture (both
    /// aspects into one texture, the stencil stored while the depth resolves).
    pub fn set_depth_resolve_target(&mut self, target: Option<ResolveTarget>) -> Result<(), Error> {
        self.log(|_| RenderCall::DepthResolveTarget(target.clone().map(Box::new)));
        self.set_aspect_resolve_target("set_depth_resolve_target", Attachment::Depth, target)
    }

    /// The stencil attachment resolves into `target` (`None`: nowhere). Refused as
    /// [`RenderCommandEncoder::set_depth_resolve_target`] is, the aspects swapped.
    pub fn set_stencil_resolve_target(
        &mut self,
        target: Option<ResolveTarget>,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::StencilResolveTarget(target.clone().map(Box::new)));
        self.set_aspect_resolve_target("set_stencil_resolve_target", Attachment::Stencil, target)
    }

    /// The depth or stencil attachment (`which`) resolves into `target`: the attachment's
    /// resolve made again from both aspects' targets and the descriptor's filters.
    fn set_aspect_resolve_target(
        &mut self,
        call: &'static str,
        which: Attachment,
        target: Option<ResolveTarget>,
    ) -> Result<(), Error> {
        // A later color group's twin of the depth and stencil resolves nothing; in a pass of
        // several groups, no draw read the target before (`render/color_groups.rs`).
        if self.later_group_encoder() {
            return Ok(());
        }
        self.check_group_resolve(target.as_ref().map(|t| &t.texture))?;
        let maker = self.core.command_buffer().clone();
        // The pass's hidden depth and stencil are no attachment of Metal's (`render.rs`).
        let Some(ds) = self.layout.depth_stencil.as_mut().filter(|ds| !ds.hidden) else {
            return Err(problem(call, RenderPassProblem::NoAttachment(which)));
        };
        let stencil = which == Attachment::Stencil;
        let attached = if stencil {
            ds.stencil.is_some()
        } else {
            ds.depth.is_some()
        };
        if !attached {
            return Err(problem(call, RenderPassProblem::NoAttachment(which)));
        }
        // A late target keeps the rule Metal's validation states, as a color one does (plain
        // Metal's answer is unmeasured): a target needs a store action that resolves or is
        // `Unknown`. The descriptor's resolve texture under one that does not is left untouched,
        // as plain Metal leaves it (`resolve::resolves`).
        let store = if stencil {
            ds.stencil.map(|s| s.store)
        } else {
            ds.depth.map(|d| d.store)
        };
        let unresolving = store.is_some_and(|s| !s.resolves() && s != StoreAction::Unknown);
        if target.is_some() && unresolving {
            return Err(problem(call, RenderPassProblem::ResolveStoreAction(which)));
        }
        let (depth_target, stencil_target) = if stencil {
            (ds.depth_resolve_target.as_ref(), target.as_ref())
        } else {
            (target.as_ref(), ds.stencil_resolve_target.as_ref())
        };
        let depth = ds.depth.map(|d| wanted(depth_target, d.store));
        let stencil_wanted = ds.stencil.map(|s| wanted(stencil_target, s.store));
        let resolve = resolve::depth_stencil(
            call,
            &maker,
            &ds.target,
            ds.separate.as_ref(),
            depth.as_ref(),
            stencil_wanted.as_ref(),
            ds.depth_filter,
            ds.stencil_filter,
        )?;
        if stencil {
            ds.stencil_resolve_target = target;
        } else {
            ds.depth_resolve_target = target;
        }
        ds.resolve = resolve;
        Ok(())
    }
}

fn problem(call: &'static str, p: RenderPassProblem) -> Error {
    invalid(call, Problem::RenderPass(p))
}

/// The resolve of the depth/stencil attachment `ds` made again from both aspects' targets and
/// the descriptor's filters, with `stores` as the depth's and the stencil's store actions (`None`
/// for an aspect without an attachment): a late store action that does not resolve drops its
/// aspect's resolve (`RenderCommandEncoder::set_depth_store_action`).
pub(super) fn remade(
    call: &'static str,
    maker: &CommandBuffer,
    ds: &DepthStencilTarget,
    (depth, stencil): (Option<StoreAction>, Option<StoreAction>),
) -> Result<Option<DepthStencilResolve>, Error> {
    let depth = depth.map(|store| wanted(ds.depth_resolve_target.as_ref(), store));
    let stencil = stencil.map(|store| wanted(ds.stencil_resolve_target.as_ref(), store));
    resolve::depth_stencil(
        call,
        maker,
        &ds.target,
        ds.separate.as_ref(),
        depth.as_ref(),
        stencil.as_ref(),
        ds.depth_filter,
        ds.stencil_filter,
    )
}

/// Give each of a pass's instance shapes the resolves of `ending`, the shape of the pass's
/// layout as it ends: the targets set last (the module docs).
pub(super) fn settle<'a>(shapes: impl Iterator<Item = &'a mut PassShape>, ending: &PassShape) {
    for shape in shapes {
        shape.color_resolves.clone_from(&ending.color_resolves);
        shape.depth_stencil_resolve = ending.depth_stencil_resolve;
    }
}
