//! The second validation-layer gap a [`Messages`](super::Messages) sink lets pass: the layer
//! does not model an image made of an imported Metal texture, which Vulkan treats as bound, and
//! reports each use of it as a use with no memory bound.
//!
//! `VK_EXT_metal_objects` makes an image of an `MTLTexture` given at its creation
//! (`VkImportMetalTextureInfoEXT`), and "due to `id<MTLTexture>` already being backed by memory,
//! images created with `VkImportMetalTextureInfoEXT` in the `pNext` of the `VkImageCreateInfo`
//! will be treated as bound to a `VkDeviceMemory`" (the extension's reference page); binding one
//! again is forbidden (`VUID-vkBindImageMemory-image-01044`). The layer's image state reads no
//! such import: `vvl::Image::Image` picks the image's memory tracker from its create flags alone,
//! a `BindableLinearMemoryTracker` for every image that is neither sparse nor disjoint, which
//! only `vkBindImageMemory` marks bound (Vulkan-ValidationLayers v1.4.357,
//! `layers/state_tracker/image_state.cpp`, the `Image` constructors; its Metal code there handles
//! exports alone, `GetMetalExport`; unchanged in main of 2026-10-07). So every barrier, view,
//! framebuffer and descriptor of such an image, every copy from it and every clear of it, reports
//! it "used with no memory bound".
//!
//! The runtime makes such an image for a texture over a no-copy buffer on a driver that runs on
//! Metal (`runtime/src/metal_texture.rs`) and names it [`NAME`] where its instance reports
//! messages (`runtime/src/host_memory.rs`, `METAL_TEXTURE_IMAGE`). [`shape`] matches exactly the
//! six messages the layer gives such an image, and only when the image they name bears that
//! name; [`read`] keys the layer releases whose source was read with the gap in place. On any
//! other release such a message fails, marked "re-read the image state in layer X.Y.Z", so a
//! fixed or changed layer is checked again rather than trusted.

use ash::vk;

/// The messages the layer gives an image it believes unbound, as the lanes saw them on MoltenVK
/// (EV `ZCI-mvkprobe1-mvk4`): a barrier, a view, a framebuffer attachment, a descriptor; the
/// source of an image copy, as the kept-bytes lanes saw it there (a pass copying the texels of
/// such a texture into the image its texture keeps, `runtime/src/kept_bytes.rs`); and the image
/// of a clear, as the window-surface lanes saw it there (a pass's color that its first instance
/// leaves out, cleared before the pass, `runtime/src/render.rs`'s `Clears`).
const ID_NAMES: [&str; 6] = [
    "VUID-VkImageMemoryBarrier-image-01932",
    "VUID-VkImageViewCreateInfo-image-01020",
    "UNASSIGNED-VkFramebufferCreateInfo-BoundResourceFreedMemoryAccess",
    "UNASSIGNED-VkDescriptorImageInfo-BoundResourceFreedMemoryAccess",
    "VUID-vkCmdCopyImage-srcImage-07966",
    "VUID-vkCmdClearColorImage-image-00003",
];

/// The runtime's name for an image made of a Metal texture (`runtime/src/host_memory.rs`,
/// `METAL_TEXTURE_IMAGE`), which the layer prints after the image's handle.
pub(super) const NAME: &str = "metal2vulkan: a Metal texture over imported host memory";

/// What follows the image in each of those messages.
const UNBOUND: &str = " used with no memory bound";

/// The layer releases (`VkLayerProperties::specVersion` of `VK_LAYER_KHRONOS_validation`) whose
/// image state was read with the gap in place: v1.4.357.
pub(super) const READ: &[u32] = &[vk::make_api_version(0, 1, 4, 357)];

/// Where the gap is, for the line each message let pass prints.
pub(super) const SOURCE: &str = "Vulkan-ValidationLayers v1.4.357 \
    layers/state_tracker/image_state.cpp, vvl::Image::Image: the memory tracker is chosen from \
    the create flags alone; no VkImportMetalTextureInfoEXT handling";

/// Whether a message with this id name and text is the gap's: one of the six messages, about a
/// `VkImage` the runtime named [`NAME`], used with no memory bound: the `VkImage` the text names
/// last before that phrase must bear the name.
pub(super) fn shape(id_name: &str, text: &str) -> bool {
    if !ID_NAMES.contains(&id_name) {
        return false;
    }
    let Some(at) = text.find(UNBOUND) else {
        return false;
    };
    let named = format!("[{NAME}]");
    let before = &text[..at];
    let Some(image_at) = before.rfind("VkImage 0x") else {
        return false;
    };
    let image = &before[image_at + "VkImage 0x".len()..];
    let hex_end = image
        .find(|c: char| !c.is_ascii_hexdigit())
        .unwrap_or(image.len());
    hex_end > 0 && image[hex_end..] == named
}

/// Whether the layer release `version` is one whose source was read with the gap in place.
pub(super) fn read(version: u32) -> bool {
    READ.contains(&version)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layer's messages as the lanes saw them on MoltenVK, with the runtime's name after the
    /// handle (the layer prints an object's debug name there).
    fn seen(id_name: &str) -> String {
        let image = format!("VkImage 0x6c000000006c[{NAME}]");
        match id_name {
            "VUID-VkImageMemoryBarrier-image-01932" => format!(
                "vkCmdPipelineBarrier(): pImageMemoryBarriers[0].image {image} used with no \
                 memory bound. Memory should be bound by calling vkBindImageMemory().\nThe \
                 Vulkan spec states: If image is non-sparse then the image or each specified \
                 disjoint plane must be bound completely and contiguously to a single \
                 VkDeviceMemory object"
            ),
            "VUID-VkImageViewCreateInfo-image-01020" => format!(
                "vkCreateImageView(): pCreateInfo->image {image} used with no memory bound. \
                 Memory should be bound by calling vkBindImageMemory()."
            ),
            "VUID-vkCmdCopyImage-srcImage-07966" => format!(
                "vkCmdCopyImage(): srcImage {image} used with no memory bound. Memory should be \
                 bound by calling vkBindImageMemory().\nThe Vulkan spec states: If srcImage is \
                 non-sparse then the image or each specified disjoint plane must be bound \
                 completely and contiguously to a single VkDeviceMemory object"
            ),
            "VUID-vkCmdClearColorImage-image-00003" => format!(
                "vkCmdClearColorImage(): image {image} used with no memory bound. Memory should \
                 be bound by calling vkBindImageMemory().\nThe Vulkan spec states: If image is \
                 non-sparse then it must be bound completely and contiguously to a single \
                 VkDeviceMemory object"
            ),
            _ => format!(
                "vkCreateFramebuffer(): pCreateInfo->pAttachments[0] {image} used with no memory \
                 bound. Memory should be bound by calling vkBindImageMemory()."
            ),
        }
    }

    #[test]
    fn the_layer_s_messages_about_a_metal_texture_image_are_the_gap() {
        for id_name in ID_NAMES {
            assert!(shape(id_name, &seen(id_name)), "{id_name}");
        }
    }

    #[test]
    fn the_messages_about_another_image_are_not_the_gap() {
        for id_name in ID_NAMES {
            let text = seen(id_name);
            // No name: an image the runtime made otherwise, which must be bound.
            let unnamed = text.replace(&format!("[{NAME}]"), "");
            assert!(!shape(id_name, &unnamed), "{id_name}: unnamed");
            // Another name, or the name inside another one.
            let other = text.replace(NAME, "the caller's image");
            assert!(!shape(id_name, &other), "{id_name}: another name");
            let wider = text.replace(NAME, &format!("{NAME} and more"));
            assert!(!shape(id_name, &wider), "{id_name}: a wider name");
            // The name on another object, the image unnamed.
            let elsewhere = format!(
                "VkImageView 0x10[{NAME}] of {}",
                text.replace(&format!("[{NAME}]"), "")
            );
            assert!(!shape(id_name, &elsewhere), "{id_name}: the name elsewhere");
        }
    }

    #[test]
    fn other_messages_about_the_image_are_not_the_gap() {
        let text = seen("VUID-VkImageMemoryBarrier-image-01932");
        assert!(!shape("VUID-VkImageMemoryBarrier-oldLayout-01197", &text));
        assert!(!shape("VUID-VkImageMemoryBarrier2-image-01932", &text));
        assert!(!shape("", &text));
        let freed = text.replace(UNBOUND, " used after its memory was freed");
        assert!(!shape("VUID-VkImageMemoryBarrier-image-01932", &freed));
    }

    #[test]
    fn only_the_releases_read_pass() {
        assert!(read(vk::make_api_version(0, 1, 4, 357)));
        assert!(!read(vk::make_api_version(0, 1, 4, 356)));
        assert!(!read(vk::make_api_version(0, 1, 4, 358)));
        assert!(!read(vk::make_api_version(0, 1, 5, 0)));
    }
}
