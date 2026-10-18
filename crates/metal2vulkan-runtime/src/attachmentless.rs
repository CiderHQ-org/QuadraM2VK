//! The render passes a device keeps for its passes without attachments: one per sample count and
//! view mask (a multiview instance's), made when an encoder of such a pass first needs it and
//! destroyed with the device. Every encoder of a pass without attachments records its draws against
//! its sample count's kept pass, makes its pipelines against it, and begins it.
//!
//! Such a pass rasterizes at its `defaultRasterSampleCount` (a Metal probe on an M4), the sample
//! count the encoder checks each pipeline's `rasterSampleCount` against (`render.rs`). A Vulkan
//! subpass without attachments rasterizes at its pipelines' `rasterizationSamples`, but MoltenVK
//! takes the count from its subpass's default sample count, which only creating a pipeline against
//! that very `VkRenderPass` sets (`MVKGraphicsPipeline` in MoltenVK 1.4.1's `MVKPipeline.mm`; a
//! `MVKRenderSubpass` begins with 1). A pass made per encoder, after the pipelines it draws were
//! made against another, compatible one, rasterized one sample: at 2x each pixel held sample 0
//! alone. The kept pass is the one its pipelines were made against, on every
//! device, so it is the one its subpass's default was set on; the others rasterize the pipelines'
//! count either way. A pass of no attachments has no load or store operation, so the one pass
//! serves both as the compatible pass and as the pass begun.
//!
//! A pass whose subpass leaves out attachments the pass still describes (the sample planes', with
//! their anchor, `render_target.rs`) is no pass without attachments here: only a shape with no
//! attachment at all is kept, so the kept passes number at most the sample counts times the
//! view masks drawn.

use super::color_list::ColorList;
use super::device::lock;
use super::error::Error;
use super::render_target::{self, PassShape, RenderPassObject};
use super::Device;
use crate::release;
use ash::vk;
use std::collections::BTreeMap;

/// The device's kept passes, by sample count and view mask.
#[derive(Default)]
pub(super) struct KeptPasses {
    passes: BTreeMap<(u32, u32), vk::RenderPass>,
}

impl KeptPasses {
    /// Destroy every kept pass; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        for &pass in self.passes.values() {
            // SAFETY: the device is idle, so no pending command uses the pass, and a pipeline
            // made with it does not keep it.
            unsafe { release::object(raw, pass) };
        }
    }
}

/// Whether a device keeps the pass of `shape`: a shape with no attachment at all.
pub(super) fn keeps(shape: &PassShape) -> bool {
    *shape == self::shape(shape.samples, shape.view_mask)
}

/// The shape of a pass without attachments of `samples` samples whose subpasses have
/// `view_mask`.
pub(super) fn shape(samples: u32, view_mask: u32) -> PassShape {
    PassShape {
        colors: ColorList::new(),
        depth_stencil: None,
        samples,
        color_resolves: ColorList::new(),
        depth_stencil_resolve: None,
        fetch_inputs: false,
        no_inputs: ColorList::new(),
        storage: ColorList::new(),
        view_mask,
        locations: ColorList::new(),
    }
}

/// The pass `device` keeps for `shape` (made now if this is its first use), where `shape` has no
/// attachment at all; `None` for any other shape.
pub(super) fn kept(device: &Device, shape: &PassShape) -> Result<Option<RenderPassObject>, Error> {
    if !keeps(shape) {
        return Ok(None);
    }
    let mut kept = lock(&device.shared().attachmentless);
    let key = (shape.samples, shape.view_mask);
    let raw = match kept.passes.get(&key) {
        Some(&raw) => raw,
        None => {
            let raw = render_target::made_compatible_pass(device, shape)?.into_kept();
            kept.passes.insert(key, raw);
            raw
        }
    };
    Ok(Some(RenderPassObject::kept(device, raw)))
}
