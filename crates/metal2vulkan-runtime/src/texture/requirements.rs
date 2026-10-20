//! The memory requirements of the images a heap texture's plan asks about and never binds
//! (`heap.rs`'s `planned_requirements`): each query makes an image, asks its requirements and
//! destroys it, on the thread that makes the texture. A window server that makes its heap textures
//! each frame asked the same ones each frame.
//!
//! Vulkan (Resource Memory Association) fixes the answer by the creation parameters:
//!
//! - "The size member is identical for all VkImage objects created with the same combination of
//!   creation parameters specified in VkImageCreateInfo and its pNext chain."
//! - "For images created with a color format, the memoryTypeBits member is identical for all
//!   VkImage objects created with the same combination of values for the tiling member, the
//!   VK_IMAGE_CREATE_SPARSE_BINDING_BIT bit and VK_IMAGE_CREATE_PROTECTED_BIT bit of the flags
//!   member, the VK_IMAGE_CREATE_SPLIT_INSTANCE_BIND_REGIONS_BIT bit of the flags member, [...]
//!   handleTypes member of VkExternalMemoryImageCreateInfo, and the
//!   VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT of the usage member", and for a depth/stencil format
//!   the same with the format member too.
//! - "If the maintenance4 feature is enabled, then the alignment member is identical for all
//!   VkImage objects created with the same combination of values for the flags, imageType,
//!   format, extent, mipLevels, arrayLayers, samples, tiling and usage members in the
//!   VkImageCreateInfo structure passed to vkCreateImage."
//!
//! So on a device that enables `maintenance4`, an image of the same creation parameters has the
//! same requirements, and a query of parameters asked before takes the answer kept for them
//! ([`RequirementAnswers`]); elsewhere each query makes its image, as alignment is not fixed. A
//! plan and a tiling fix every parameter of the image a query makes (`ImagePlan::create_info`, the
//! device's sharing, and the tiling's initial layout), so they are the key. The device's limits for
//! the image are checked at every query all the same.

use crate::image_plan::ImagePlan;
use ash::vk;
use std::collections::VecDeque;

/// The answers kept at most: the least recently asked goes first.
const KEPT: usize = 64;

/// The memory requirements of planned images by their plan and tiling (the module docs), kept
/// only where the device enables `maintenance4`.
pub(crate) struct RequirementAnswers {
    keeps: bool,
    /// The most recently asked first.
    answers: VecDeque<(ImagePlan, vk::ImageTiling, vk::MemoryRequirements)>,
}

impl RequirementAnswers {
    /// The answers of a device that enables `maintenance4` where `maintenance4`.
    pub(crate) fn new(maintenance4: bool) -> RequirementAnswers {
        RequirementAnswers {
            keeps: maintenance4,
            answers: VecDeque::new(),
        }
    }

    /// The requirements kept for an image of `plan` and `tiling`, now the most recently asked.
    pub(crate) fn answer(
        &mut self,
        plan: &ImagePlan,
        tiling: vk::ImageTiling,
    ) -> Option<vk::MemoryRequirements> {
        let at = (self.answers.iter())
            .position(|(kept, kept_tiling, _)| *kept_tiling == tiling && kept == plan)?;
        let entry = self.answers.remove(at)?;
        let requirements = entry.2;
        self.answers.push_front(entry);
        Some(requirements)
    }

    /// Keep `requirements`, the answer for an image of `plan` and `tiling`, where the device
    /// enables `maintenance4`.
    pub(crate) fn keep(
        &mut self,
        plan: &ImagePlan,
        tiling: vk::ImageTiling,
        requirements: vk::MemoryRequirements,
    ) {
        if !self.keeps {
            return;
        }
        if self.answers.len() == KEPT {
            self.answers.pop_back();
        }
        self.answers
            .push_front((plan.clone(), tiling, requirements));
    }
}
