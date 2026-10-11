//! Shader module creation for the executor's pipelines.

use super::device::VulkanContext;
use ash::vk;

pub(super) fn create_shader_module(
    context: &VulkanContext,
    spv: &[u8],
    label: &str,
) -> Result<vk::ShaderModule, String> {
    if !spv.len().is_multiple_of(4) {
        return Err(format!("{label} SPIR-V byte stream is not word-aligned"));
    }
    let words = spv
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect::<Vec<_>>();
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&words);
    unsafe { context.device.create_shader_module(&shader_info, None) }
        .map_err(|error| format!("create {label} shader module: {error}"))
}
