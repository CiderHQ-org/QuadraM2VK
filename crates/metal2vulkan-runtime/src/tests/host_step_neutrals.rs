//! A host step makes no neutral (`neutral.rs`): it runs on the scheduler's thread inside the exit
//! guard's section, which reaches only the guard's table of entry points, and a neutral is an image
//! or a buffer, its memory and a submission of the device's own. In a host step a neutral not made
//! before is refused, `Error::HostStepObject`; one made before, on the program's thread, is found
//! there.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The end-to-end case, a host step's draw
//! of a state made after its commit, is the validation lane's
//! `a_host_step_s_draw_of_a_state_made_after_the_commit_binds_a_neutral_made_before_it`.

use crate::error::Error;
use crate::exit_guard;
use crate::{Device, DeviceKind, DeviceOptions, HostStepObject, ShaderCacheOptions};
use metal2vulkan::vk::Format;
use std::path::{Path, PathBuf};

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path =
            std::env::temp_dir().join(format!("m2v-host-step-neutrals-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// Whether `result` is the host step's refusal of a neutral, no Vulkan object made.
fn refused<T>(result: Result<T, Error>) -> bool {
    matches!(result, Err(Error::HostStepObject(HostStepObject::Neutral)))
}

/// Inside a host step, a neutral sampler and a neutral texel buffer not made yet are refused,
/// each time; made on the program's thread, each is found inside a host step, the same object.
#[test]
fn a_host_step_finds_the_neutrals_made_before_it_and_makes_none() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let texel = || device.neutral_texel_view(Format::R32_SFLOAT, false);
    let sampler = || device.neutral_sampler(true);
    for _ in 0..2 {
        assert!(refused(exit_guard::host_step(texel)), "a texel view made");
        assert!(refused(exit_guard::host_step(sampler)), "a sampler made");
    }
    let (made_texel, made_sampler) = (texel().unwrap(), sampler().unwrap());
    assert_eq!(exit_guard::host_step(texel).unwrap(), made_texel);
    assert_eq!(exit_guard::host_step(sampler).unwrap(), made_sampler);
}
