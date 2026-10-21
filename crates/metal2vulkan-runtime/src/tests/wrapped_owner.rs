//! An external image owner survives views and an encoded command after guest handles are gone.
use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, LoadAction, Origin, RenderPassDescriptor, ResourceOptions, Size, StoreAction,
    Texture, TextureDescriptor, VkImageLayouts,
};
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct Owner {
    _texture: Texture,
    drops: Arc<AtomicUsize>,
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn a_wrapped_image_keeps_its_owner_through_views_and_commands() {
    let scratch = Scratch::new("wrapped-owner");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let mut desc = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 4, 4, false);
    desc.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    desc.resource_options = ResourceOptions::private();
    let image = device.new_texture(&desc).unwrap();
    // Finish the original texture's initialization, leaving its raw image in GENERAL.
    let bytes = device.new_buffer(64, ResourceOptions::shared()).unwrap();
    let initialized = queue.command_buffer().unwrap();
    let mut blit = initialized.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &image,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(4, 4, 1),
        &bytes,
        0,
        16,
        64,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    initialized.commit().unwrap();
    initialized.wait_until_completed_timeout(WAIT).unwrap();
    drop(initialized);
    let plan = image.image_plan().unwrap().clone();
    let formats = plan.ash_view_formats();
    let mut list = vk::ImageFormatListCreateInfo::default();
    let info = plan.create_info(device.shared().sharing(), &formats, &mut list);
    let drops = Arc::new(AtomicUsize::new(0));
    let raw = image.raw_image().unwrap();
    let owner = Box::new(Owner {
        _texture: image,
        drops: drops.clone(),
    });
    // SAFETY: the owner's texture owns exactly this image, with this creation and GENERAL
    // layout. Only the wrapped texture uses it from here, with the same queue families.
    let wrapped = unsafe {
        device.new_texture_with_vk_image_keep_alive(
            &desc,
            raw,
            &info,
            VkImageLayouts::general(),
            owner,
        )
    }
    .unwrap();
    let view = wrapped.new_texture_view(PixelFormat::RGBA8Unorm).unwrap();
    drop(wrapped);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        0,
        "a view retains its image owner"
    );
    let command = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    pass.color_attachments[0].texture = Some(view.clone());
    pass.color_attachments[0].load_action = LoadAction::Clear;
    pass.color_attachments[0].store_action = StoreAction::Store;
    command
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    drop(pass);
    drop(view);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        0,
        "encoded work retains its image owner"
    );
    command.commit().unwrap();
    command.wait_until_completed_timeout(WAIT).unwrap();
    drop(command);
    let deadline = std::time::Instant::now() + WAIT;
    while drops.load(Ordering::SeqCst) == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "the completed image owner leaked"
        );
        std::thread::yield_now();
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
