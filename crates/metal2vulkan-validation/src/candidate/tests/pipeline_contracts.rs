use super::*;

#[test]
fn backend_names_cannot_relabel_the_hosts_vulkan_stack() {
    if cfg!(target_os = "macos") {
        assert!(validate_backend_host(Backend::Moltenvk).is_ok());
        assert!(validate_backend_host(Backend::Vulkan).is_err());
    } else if cfg!(target_os = "linux") {
        assert!(validate_backend_host(Backend::Moltenvk).is_err());
        assert!(validate_backend_host(Backend::Vulkan).is_ok());
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn authored_vulkan_draws_match_metals_default_front_face() {
    assert!(platform::METAL_DEFAULT_FRONT_FACE == ash::vk::FrontFace::CLOCKWISE);
}
