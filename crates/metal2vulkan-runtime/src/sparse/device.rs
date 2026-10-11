//! What a device has of sparse binding and residency.
//!
//! The compile layer's vocabulary has no sparse binding: a sparse heap is the runtime's own need,
//! like `timelineSemaphore`. A device the runtime creates enables `sparseBinding`,
//! `sparseResidencyImage2D`, `sparseResidencyImage3D` and `sparseResidency2Samples` to
//! `16Samples` where it has them and its queue family binds sparse memory
//! (`VK_QUEUE_SPARSE_BINDING_BIT`); an adopted device states what it enabled
//! (`RawDevice::sparse_features`). A sparse heap needs `sparseBinding`, the queue family's bit
//! and residency of some image type; each texture, its type's and sample count's.

use ash::vk;
use metal2vulkan::error::SparseNeed;

/// The sparse members of `available`, every other member false: what a device the runtime
/// creates enables of sparse binding and residency, where its queue family binds sparse memory.
pub(crate) fn wanted(available: &vk::PhysicalDeviceFeatures) -> vk::PhysicalDeviceFeatures {
    vk::PhysicalDeviceFeatures {
        sparse_binding: available.sparse_binding,
        sparse_residency_image2_d: available.sparse_residency_image2_d,
        sparse_residency_image3_d: available.sparse_residency_image3_d,
        sparse_residency2_samples: available.sparse_residency2_samples,
        sparse_residency4_samples: available.sparse_residency4_samples,
        sparse_residency8_samples: available.sparse_residency8_samples,
        sparse_residency16_samples: available.sparse_residency16_samples,
        ..vk::PhysicalDeviceFeatures::default()
    }
}

/// No Metal counterpart: the sparse members of the core features an adopted device enabled
/// ([`RawDevice::sparse_features`](crate::RawDevice::sparse_features)), outside the compile
/// layer's vocabulary. None by default.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SparseFeatures {
    /// `sparseBinding`.
    pub sparse_binding: bool,
    /// `sparseResidencyImage2D`.
    pub sparse_residency_image_2d: bool,
    /// `sparseResidencyImage3D`.
    pub sparse_residency_image_3d: bool,
    /// `sparseResidency2Samples`, `4Samples`, `8Samples` and `16Samples`.
    pub sparse_residency_samples: [bool; 4],
}

impl SparseFeatures {
    /// The sparse members of `features`.
    pub fn of(features: &vk::PhysicalDeviceFeatures) -> SparseFeatures {
        let on = |member: vk::Bool32| member == vk::TRUE;
        SparseFeatures {
            sparse_binding: on(features.sparse_binding),
            sparse_residency_image_2d: on(features.sparse_residency_image2_d),
            sparse_residency_image_3d: on(features.sparse_residency_image3_d),
            sparse_residency_samples: [
                on(features.sparse_residency2_samples),
                on(features.sparse_residency4_samples),
                on(features.sparse_residency8_samples),
                on(features.sparse_residency16_samples),
            ],
        }
    }

    /// The core features of these members, every other member false.
    pub fn features(&self) -> vk::PhysicalDeviceFeatures {
        let bool32 = |on: bool| if on { vk::TRUE } else { vk::FALSE };
        let [two, four, eight, sixteen] = self.sparse_residency_samples.map(bool32);
        vk::PhysicalDeviceFeatures {
            sparse_binding: bool32(self.sparse_binding),
            sparse_residency_image2_d: bool32(self.sparse_residency_image_2d),
            sparse_residency_image3_d: bool32(self.sparse_residency_image_3d),
            sparse_residency2_samples: two,
            sparse_residency4_samples: four,
            sparse_residency8_samples: eight,
            sparse_residency16_samples: sixteen,
            ..vk::PhysicalDeviceFeatures::default()
        }
    }
}

/// What a device has of sparse binding and residency, as the runtime uses it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SparseDevice {
    /// `sparseBinding` is enabled.
    binding: bool,
    /// The queue family the runtime submits to has `VK_QUEUE_SPARSE_BINDING_BIT`.
    queue: bool,
    /// `sparseResidencyImage2D` is enabled.
    image_2d: bool,
    /// `sparseResidencyImage3D` is enabled.
    image_3d: bool,
    /// The sample counts above 1 whose `sparseResidencyNSamples` is enabled, as
    /// `VkSampleCountFlags` bits.
    samples: u32,
    /// `residencyNonResidentStrict`: an unbound texel reads zero and a write to it is dropped.
    pub(crate) strict: bool,
}

impl SparseDevice {
    /// The support of a device that enabled `enabled` (its sparse members read), whose queue
    /// family has `queue_flags`, and whose sparse properties are `properties`.
    pub(crate) fn of(
        enabled: &vk::PhysicalDeviceFeatures,
        queue_flags: vk::QueueFlags,
        properties: &vk::PhysicalDeviceSparseProperties,
    ) -> SparseDevice {
        let on = |member: vk::Bool32| member == vk::TRUE;
        let mut samples = vk::SampleCountFlags::empty();
        for (member, count) in [
            (
                enabled.sparse_residency2_samples,
                vk::SampleCountFlags::TYPE_2,
            ),
            (
                enabled.sparse_residency4_samples,
                vk::SampleCountFlags::TYPE_4,
            ),
            (
                enabled.sparse_residency8_samples,
                vk::SampleCountFlags::TYPE_8,
            ),
            (
                enabled.sparse_residency16_samples,
                vk::SampleCountFlags::TYPE_16,
            ),
        ] {
            if on(member) {
                samples |= count;
            }
        }
        SparseDevice {
            binding: on(enabled.sparse_binding),
            queue: queue_flags.contains(vk::QueueFlags::SPARSE_BINDING),
            image_2d: on(enabled.sparse_residency_image2_d),
            image_3d: on(enabled.sparse_residency_image3_d),
            samples: samples.as_raw(),
            strict: on(properties.residency_non_resident_strict),
        }
    }

    /// Whether the device binds sparse memory on its queues (and so primes and calls
    /// `vkQueueBindSparse`).
    pub(crate) fn binds(&self) -> bool {
        self.binding && self.queue
    }

    /// What a sparse heap lacks on this device: `sparseBinding`, the queue family's bit, or
    /// residency of any image type.
    pub(crate) fn heaps(&self) -> Result<(), SparseNeed> {
        if !self.binding {
            return Err(SparseNeed::SparseBinding);
        }
        if !self.queue {
            return Err(SparseNeed::SparseBindingQueue);
        }
        if !self.image_2d && !self.image_3d {
            return Err(SparseNeed::SparseResidencyImage2D);
        }
        Ok(())
    }

    /// What a sparse resident image of `image_type` at `samples` lacks on this device, beyond a
    /// heap's needs.
    pub(crate) fn image(
        &self,
        image_type: vk::ImageType,
        samples: vk::SampleCountFlags,
    ) -> Result<(), SparseNeed> {
        self.heaps()?;
        if image_type == vk::ImageType::TYPE_3D {
            if !self.image_3d {
                return Err(SparseNeed::SparseResidencyImage3D);
            }
        } else if !self.image_2d {
            return Err(SparseNeed::SparseResidencyImage2D);
        }
        let counted = vk::SampleCountFlags::from_raw(self.samples);
        if samples != vk::SampleCountFlags::TYPE_1 && !counted.contains(samples) {
            return Err(SparseNeed::SparseResidencySamples(samples.as_raw()));
        }
        Ok(())
    }
}
