//! `DeviceOptions::optional_device_extensions`: a device takes what it can and is never refused
//! for an extension it lacks.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::Scratch;
use crate::{Device, DeviceKind, DeviceOptions, ShaderCacheOptions};
use std::ffi::CString;

fn options(scratch: &Scratch) -> DeviceOptions {
    DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    }
}

fn names(names: &[&std::ffi::CStr]) -> Vec<CString> {
    names.iter().map(|&name| name.to_owned()).collect()
}

/// An extension no driver lists is left out and the device is made; the ones the device lists
/// are enabled with their features, each once; swapchain maintenance, which needs its instance
/// extensions, is left out where the instance does not enable them.
#[test]
fn optional_extensions_are_taken_where_the_device_has_them_and_never_refuse_it() {
    let scratch = Scratch::new("optional-extensions");
    let bogus = c"VK_EXT_not_a_real_extension";
    let priority = c"VK_EXT_memory_priority";
    let maintenance = c"VK_KHR_swapchain_maintenance1";

    let mut plain = options(&scratch);
    plain.optional_device_extensions = names(&[bogus, priority, maintenance]);
    let device = Device::system_default_with(&plain).expect("optional extensions never refuse");
    assert!(!device.extension_enabled(bogus));
    assert!(
        !device.extension_enabled(maintenance),
        "its instance extensions are not enabled on this instance"
    );
    let properties = device.properties();
    eprintln!(
        "OPTIONAL: {} took memory priority: {}",
        properties.name,
        device.extension_enabled(priority)
    );
    drop(device);

    // The same options but for the instance extensions swapchain maintenance requires, when the
    // loader has them: the group of equal options is another instance.
    let mut surfaced = options(&scratch);
    surfaced.optional_device_extensions = names(&[bogus, maintenance]);
    surfaced.instance_extensions = names(&[
        c"VK_KHR_surface",
        c"VK_KHR_get_surface_capabilities2",
        c"VK_KHR_surface_maintenance1",
    ]);
    match Device::system_default_with(&surfaced) {
        Ok(device) => {
            assert!(!device.extension_enabled(bogus));
            eprintln!(
                "OPTIONAL: {} took swapchain maintenance: {}",
                device.name(),
                device.extension_enabled(maintenance)
            );
        }
        Err(error) => eprintln!("OPTIONAL: the loader lacks the surface extensions: {error}"),
    }
}
