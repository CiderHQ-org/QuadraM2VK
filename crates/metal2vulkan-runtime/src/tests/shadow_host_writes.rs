//! A shadow's bytes over host pages lie in no `VkBuffer` while they are detached
//! (`ShadowBuffer::raw`): the device's stand-in, which holds none of them, is never what a copy
//! or a merge's window takes, and a rebind gives the new pages' `VkBuffer`.
//!
use crate::shadow::ShadowBuffer;
use ash::vk;

/// A shadow's bytes over a no-copy buffer's host pages, detached, lie in no `VkBuffer` (a merge
/// planned then bound a window of the device's 256-byte stand-in past its size,
/// `VUID-VkDescriptorBufferInfo-range-00342`), and in the new pages' once a rebind gives them.
#[test]
fn a_shadow_over_detached_host_pages_takes_no_buffer_until_a_rebind() {
    let scratch = Scratch::new("shadow-host-writes-detached");
    let (host, other) = (Host::new(), Host::new());
    let runs = Arc::new(AtomicUsize::new(0));
    let Some(buffer) = over_host(&device, &host, &runs) else {
        return;
    };
    let shadow = ShadowBuffer::of(&buffer, buffer.at(0).0);
    assert_eq!(shadow.raw(), buffer.at(0).0, "in pages");
    assert_eq!(buffer.detach_backing(), Ok(true), "the detach");
    assert_eq!(shadow.raw(), vk::Buffer::null(), "detached");
    // SAFETY: `other` outlives the buffer (dropped after the device), which alone uses it.
    unsafe { buffer.rebind_bytes_no_copy(other.memory(), None, None) }.unwrap();
    assert_ne!(shadow.raw(), vk::Buffer::null(), "given pages again");
    assert_eq!(shadow.raw(), buffer.at(0).0, "given pages again");
    drop(buffer);
    drop(device);
}
