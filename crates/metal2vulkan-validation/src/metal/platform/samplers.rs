//! Native sampler descriptor construction.

use crate::case::{
    ArgumentBufferSamplerResource, AuthoredCase, SamplerAddressMode, SamplerCompareFunction,
    SamplerFilter, SamplerMipFilter,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCompareFunction, MTLDevice, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState,
};

pub(super) type Sampler = Retained<ProtocolObject<dyn MTLSamplerState>>;

pub(super) fn make_samplers(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
) -> Result<Vec<(u32, Sampler)>, String> {
    case.samplers
        .iter()
        .map(|resource| {
            let descriptor = MTLSamplerDescriptor::new();
            descriptor.setSAddressMode(metal_address_mode(resource.address_mode));
            descriptor.setTAddressMode(metal_address_mode(resource.address_mode));
            descriptor.setRAddressMode(metal_address_mode(resource.address_mode));
            descriptor.setMinFilter(metal_filter(resource.min_filter));
            descriptor.setMagFilter(metal_filter(resource.mag_filter));
            descriptor.setMipFilter(metal_mip_filter(resource.mip_filter));
            descriptor.setNormalizedCoordinates(resource.normalized_coordinates);
            let sampler = device
                .newSamplerStateWithDescriptor(&descriptor)
                .ok_or_else(|| format!("create Metal sampler {}", resource.binding))?;
            Ok((resource.binding, sampler))
        })
        .collect()
}

fn metal_address_mode(mode: SamplerAddressMode) -> MTLSamplerAddressMode {
    match mode {
        SamplerAddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        SamplerAddressMode::ClampToZero => MTLSamplerAddressMode::ClampToZero,
        SamplerAddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        SamplerAddressMode::MirroredRepeat => MTLSamplerAddressMode::MirrorRepeat,
    }
}

fn metal_filter(filter: SamplerFilter) -> MTLSamplerMinMagFilter {
    match filter {
        SamplerFilter::Nearest => MTLSamplerMinMagFilter::Nearest,
        SamplerFilter::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

fn metal_mip_filter(filter: SamplerMipFilter) -> MTLSamplerMipFilter {
    match filter {
        SamplerMipFilter::NotMipmapped => MTLSamplerMipFilter::NotMipmapped,
        SamplerMipFilter::Nearest => MTLSamplerMipFilter::Nearest,
        SamplerMipFilter::Linear => MTLSamplerMipFilter::Linear,
    }
}

/// The sampler state of the argument-buffer sampler `resource`: its descriptor, made with
/// `supportArgumentBuffers` so an argument encoder can write its `gpuResourceID`.
pub(super) fn make_argument_buffer_sampler(
    device: &ProtocolObject<dyn MTLDevice>,
    resource: &ArgumentBufferSamplerResource,
) -> Result<Sampler, String> {
    let descriptor = MTLSamplerDescriptor::new();
    descriptor.setSAddressMode(metal_address_mode(resource.address_mode));
    descriptor.setTAddressMode(metal_address_mode(resource.address_mode));
    descriptor.setRAddressMode(metal_address_mode(resource.address_mode));
    descriptor.setMinFilter(metal_filter(resource.min_filter));
    descriptor.setMagFilter(metal_filter(resource.mag_filter));
    descriptor.setMipFilter(metal_mip_filter(resource.mip_filter));
    descriptor.setNormalizedCoordinates(resource.normalized_coordinates);
    if let Some(compare) = resource.compare_function {
        descriptor.setCompareFunction(metal_compare_function(compare));
    }
    descriptor.setSupportArgumentBuffers(true);
    device
        .newSamplerStateWithDescriptor(&descriptor)
        .ok_or_else(|| {
            format!(
                "create Metal argument-buffer sampler {}+{}",
                resource.buffer_binding, resource.field_offset
            )
        })
}

fn metal_compare_function(function: SamplerCompareFunction) -> MTLCompareFunction {
    match function {
        SamplerCompareFunction::Never => MTLCompareFunction::Never,
        SamplerCompareFunction::Less => MTLCompareFunction::Less,
        SamplerCompareFunction::Equal => MTLCompareFunction::Equal,
        SamplerCompareFunction::LessEqual => MTLCompareFunction::LessEqual,
        SamplerCompareFunction::Greater => MTLCompareFunction::Greater,
        SamplerCompareFunction::NotEqual => MTLCompareFunction::NotEqual,
        SamplerCompareFunction::GreaterEqual => MTLCompareFunction::GreaterEqual,
        SamplerCompareFunction::Always => MTLCompareFunction::Always,
    }
}
