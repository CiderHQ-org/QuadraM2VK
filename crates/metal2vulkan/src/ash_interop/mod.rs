//! `ash` interop (feature `ash`): conversions between the plain-data [`vk`](crate::vk) values and
//! `ash::vk`, and [`DeviceDescription::query`](crate::device::DeviceDescription::query), which
//! reads one physical device into a description.
//!
//! The registry enums and bitmasks of `vk` convert to and from their `ash::vk` types with `From`,
//! as do the hand-written structs `ash` has a counterpart for, and [`sampler_create_info`] builds
//! a `VkSamplerCreateInfo` from a [`SamplerDesc`](crate::vk::SamplerDesc). The feature stops
//! there: it never creates layouts, pipelines, descriptor sets, or devices. `ash` is pinned to the
//! consumer's minor version (0.38) and stays out of the default dependency graph.

mod convert;
mod query;

pub use convert::sampler_create_info;

#[cfg(test)]
mod tests;
