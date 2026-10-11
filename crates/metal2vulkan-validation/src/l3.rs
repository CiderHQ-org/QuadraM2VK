//! L3 on the product's linked modules: requirements within the profile (`DeviceProfile::check`),
//! then `spirv-val` under the plan the product gives (`DeviceProfile::spirv_validation`: Vulkan 1.2
//! and the profile's environment with the layout relaxations it enables), run by the dev-tools
//! crate (`metal2vulkan_devtools::spirv`), the runner the product's own tests use.

use metal2vulkan::device::{DeviceProfile, Requirements};
use metal2vulkan::library::CompiledFunction;
pub use metal2vulkan_devtools::spirv::L3Failure;

/// L3 on linked `words` with `requirements` on `profile`, the failure typed, so a missing or
/// broken `spirv-val` is not reported as an invalid module.
pub fn check_words_l3(
    words: &[u32],
    requirements: &Requirements,
    profile: &DeviceProfile,
) -> Result<(), L3Failure> {
    check_requirements(requirements, profile)?;
    let plan = profile.spirv_validation();
    spirv_val(
        words,
        &plan.target_envs,
        &plan.relaxations,
        &format!("{requirements:?}"),
    )
}

/// L3's first check: `requirements` within `profile` (`DeviceProfile::check`).
pub fn check_requirements(
    requirements: &Requirements,
    profile: &DeviceProfile,
) -> Result<(), L3Failure> {
    profile
        .check(requirements)
        .map_err(|c| L3Failure::Requirements(format!("requirements exceed the profile: {c:?}")))
}

/// L3's second check: `spirv-val` on `words` under each of `envs` with `relaxations`, a
/// rejection reported with `requirements` (the module's, described) and the disassembly.
pub fn spirv_val(
    words: &[u32],
    envs: &[&str],
    relaxations: &[&str],
    requirements: &str,
) -> Result<(), L3Failure> {
    metal2vulkan_devtools::spirv::check_l3(
        words,
        envs,
        relaxations,
        &relaxations.join(" "),
        requirements,
    )
}

/// Link `function` alone, relocate it (`DeviceProfile::link_standalone`), and run L3 on the
/// words: the words, or why not.
pub fn link_and_validate(
    function: &CompiledFunction,
    profile: &DeviceProfile,
) -> Result<Vec<u32>, String> {
    let module = profile
        .link_standalone(function)
        .map_err(|e| format!("link: {e}"))?;
    check_words_l3(&module.words, function.requirements(), profile).map_err(|f| f.to_string())?;
    Ok(module.words)
}
