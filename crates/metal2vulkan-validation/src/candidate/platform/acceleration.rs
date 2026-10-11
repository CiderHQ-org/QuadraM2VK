//! The acceleration structures behind a plan's `WriteSource::AccelerationStructure` writes, shared
//! by the engine compute and render executors.
//!
//! A case's structure is built as the Metal oracle builds it ([`CaseStructure`]) and as a
//! consumer builds Metal's (`metal2vulkan::bind::acceleration`): a bottom-level structure of one
//! triangle geometry with Metal's default geometry settings (`geometry_flags`) over the case's
//! vertices or the instanced triangle, and the top-level structure the write binds: a primitive
//! structure's one-instance wrap (`primitive_wrap`) or an instance structure's instance records
//! (`instance_record`, whose typed refusals the run reports as the product's). A slot the case
//! leaves unbound binds a top-level structure of no instances, which every ray misses. A
//! structure of the other level than the argument's is not implemented (Metal leaves it
//! undefined).
//!
//! Every structure and its build inputs live in host-visible memory the host wrote before the
//! submission. The builds are recorded before the work ([`Structures::record`]): the bottom
//! levels, a build-to-build dependency, the top levels, and a build-to-shader dependency. Each
//! build has its own scratch buffer, so no two builds touch the same memory.

use super::buffers::{create_host_buffer, HostBuffer};
use super::device::VulkanContext;
use crate::candidate::engine::{bound_refusal, CaseStructure};
use crate::case::AccelerationStructureKind;
use crate::check::admission::{ExecutorError, NotSupported};
use ash::khr::acceleration_structure;
use ash::vk;
use metal2vulkan::bind::{geometry_flags, instance_record, primitive_wrap, InstanceRecord};
use metal2vulkan::device::VkFeature;
use metal2vulkan::error::{Error, Unsupported};
use metal2vulkan::pipeline::{AccelerationStructureLevel, SlotRef};
use std::collections::BTreeMap;

/// Bytes of one `float3` vertex.
const VERTEX_BYTES: u32 = 12;
/// `VkAccelerationStructureGeometryInstancesDataKHR::data` alignment.
const INSTANCE_ALIGNMENT: u64 = 16;

/// One build, recorded before the work.
struct Build {
    ty: vk::AccelerationStructureTypeKHR,
    structure: vk::AccelerationStructureKHR,
    geometry: vk::AccelerationStructureGeometryKHR<'static>,
    primitives: u32,
    scratch: vk::DeviceAddress,
}

/// Every acceleration structure of one run, destroyed before the buffers that back them.
pub(super) struct Structures {
    device: ash::Device,
    /// The extension's device functions, loaded with the first structure.
    loader: Option<acceleration_structure::Device>,
    /// Every structure created.
    created: Vec<vk::AccelerationStructureKHR>,
    /// Storage, scratch, and build-input buffers.
    buffers: Vec<HostBuffer>,
    bottom: Vec<Build>,
    top: Vec<Build>,
    /// The top-level structure each slot's write binds.
    bound: BTreeMap<SlotRef, vk::AccelerationStructureKHR>,
}

impl Drop for Structures {
    fn drop(&mut self) {
        if let Some(loader) = &self.loader {
            for structure in &self.created {
                // SAFETY: each structure was created by this run on `loader`'s device and is
                // destroyed once, after the submission that used it completed (or before any).
                unsafe { loader.destroy_acceleration_structure(*structure, None) };
            }
        }
    }
}

fn not_implemented(what: impl Into<String>) -> ExecutorError {
    NotSupported::NotImplemented(format!("engine candidate: {}", what.into())).into()
}

/// A binder refusal of a structure the case builds, reported as the product's, at bind.
fn refusal(unsupported: Unsupported) -> ExecutorError {
    bound_refusal(&Error::Unsupported(unsupported))
}

impl Structures {
    pub(super) fn new(context: &VulkanContext) -> Structures {
        Structures {
            device: context.device.clone(),
            loader: None,
            created: Vec::new(),
            buffers: Vec::new(),
            bottom: Vec::new(),
            top: Vec::new(),
            bound: BTreeMap::new(),
        }
    }

    /// The top-level structure a write of `level` at `slot` binds: built (once per slot) from
    /// `structure`, the case's structure at the slot, if any.
    pub(super) fn top_level(
        &mut self,
        context: &VulkanContext,
        slot: SlotRef,
        level: AccelerationStructureLevel,
        structure: Option<&CaseStructure>,
    ) -> Result<vk::AccelerationStructureKHR, ExecutorError> {
        if let Some(&built) = self.bound.get(&slot) {
            return Ok(built);
        }
        let records = match (level, structure) {
            (_, None) => Vec::new(),
            (
                AccelerationStructureLevel::Primitive,
                Some(CaseStructure::Primitive { vertices }),
            ) => {
                let reference = self.bottom_level(context, vertices)?;
                vec![primitive_wrap(reference, 0).map_err(refusal)?]
            }
            (
                AccelerationStructureLevel::Instance,
                Some(CaseStructure::Instances { instances }),
            ) => {
                let instance = CaseStructure::instance();
                // What Vulkan's instance fields cannot hold is refused before anything is built.
                instance_record(&instance, 0, 0).map_err(refusal)?;
                let vertices: Vec<u8> = AccelerationStructureKind::INSTANCED_TRIANGLE
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let reference = self.bottom_level(context, &vertices)?;
                (0..*instances)
                    .map(|_| instance_record(&instance, reference, 0).map_err(refusal))
                    .collect::<Result<_, _>>()?
            }
            (level, Some(structure)) => {
                let kind = match structure {
                    CaseStructure::Primitive { .. } => "a primitive",
                    CaseStructure::Instances { .. } => "an instance",
                };
                return Err(not_implemented(format!(
                    "{kind} structure bound to a {level:?} acceleration-structure argument"
                )));
            }
        };
        let top = self.top_level_of(context, &records)?;
        self.bound.insert(slot, top);
        Ok(top)
    }

    /// Record every build, bottom levels first, and the dependencies of the top levels on them
    /// and of every shader on the top levels.
    ///
    /// # Safety
    ///
    /// `command` is in the recording state on this run's device, and every structure and buffer
    /// stays alive until its submission completes.
    pub(super) unsafe fn record(&self, command: vk::CommandBuffer) {
        let Some(loader) = &self.loader else {
            return;
        };
        let build = |builds: &[Build]| {
            for b in builds {
                let geometries = [b.geometry];
                let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .ty(b.ty)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .dst_acceleration_structure(b.structure)
                    .geometries(&geometries)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: b.scratch,
                    });
                let range = [vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(b.primitives)];
                // SAFETY: the caller's contract; every address is a live buffer of this run.
                unsafe { loader.cmd_build_acceleration_structures(command, &[info], &[&range]) };
            }
        };
        let built = |dst_stage, dst_access| {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(dst_access);
            // SAFETY: the caller's contract.
            unsafe {
                self.device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                )
            };
        };
        build(&self.bottom);
        built(
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
        );
        build(&self.top);
        built(
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ,
        );
    }

    /// Build a bottom-level structure of one triangle geometry over `vertices`: its device
    /// address.
    fn bottom_level(
        &mut self,
        context: &VulkanContext,
        vertices: &[u8],
    ) -> Result<vk::DeviceAddress, ExecutorError> {
        let count = u32::try_from(vertices.len())
            .map_err(|_| "a primitive structure's vertices overflow")?
            / VERTEX_BYTES;
        let input = self.input(context, "structure vertices", vertices)?;
        let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: input,
            })
            .vertex_stride(u64::from(VERTEX_BYTES))
            .max_vertex(count.saturating_sub(1))
            .index_type(vk::IndexType::NONE_KHR);
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
            .flags(vk::GeometryFlagsKHR::from_raw(geometry_flags(true, true)));
        let (structure, build) = self.structure(
            context,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            geometry,
            count / 3,
        )?;
        self.bottom.push(build);
        let loader = self
            .loader
            .as_ref()
            .ok_or("no acceleration-structure loader")?;
        let info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
            .acceleration_structure(structure);
        // SAFETY: `structure` is live on this device.
        Ok(unsafe { loader.get_acceleration_structure_device_address(&info) })
    }

    /// Build a top-level structure of `records`.
    fn top_level_of(
        &mut self,
        context: &VulkanContext,
        records: &[InstanceRecord],
    ) -> Result<vk::AccelerationStructureKHR, ExecutorError> {
        let mut bytes: Vec<u8> = records.iter().flat_map(|r| *r.bytes()).collect();
        // A structure of no instances still names an instance buffer, which it never reads.
        bytes.resize(bytes.len().max(InstanceRecord::BYTES), 0);
        let input = self.input(context, "structure instances", &bytes)?;
        if input % INSTANCE_ALIGNMENT != 0 {
            return Err(
                format!("the instance buffer's address {input:#x} is not 16-aligned").into(),
            );
        }
        let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: input,
            });
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { instances });
        let primitives = u32::try_from(records.len())
            .map_err(|_| "a top-level structure's instances overflow")?;
        let (structure, build) = self.structure(
            context,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            geometry,
            primitives,
        )?;
        self.top.push(build);
        Ok(structure)
    }

    /// A read-only build input holding `bytes`: its device address.
    fn input(
        &mut self,
        context: &VulkanContext,
        label: &str,
        bytes: &[u8],
    ) -> Result<vk::DeviceAddress, ExecutorError> {
        let usage = vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let buffer = create_host_buffer(context, label, bytes, usage)?;
        let address = address(context, &buffer);
        self.buffers.push(buffer);
        Ok(address)
    }

    /// Create a structure of `ty` sized for `geometry` of `primitives` primitives, and the build
    /// that fills it.
    fn structure(
        &mut self,
        context: &VulkanContext,
        ty: vk::AccelerationStructureTypeKHR,
        geometry: vk::AccelerationStructureGeometryKHR<'static>,
        primitives: u32,
    ) -> Result<(vk::AccelerationStructureKHR, Build), ExecutorError> {
        if !context
            .enabled
            .features
            .contains(&VkFeature::AccelerationStructure)
        {
            return Err("an acceleration structure on a device without them enabled".into());
        }
        let loader = self
            .loader
            .get_or_insert_with(|| context.acceleration_structure())
            .clone();
        let geometries = [geometry];
        let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(ty)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: `info` and its geometry are live for the call; the extension is enabled.
        unsafe {
            loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &info,
                &[primitives],
                &mut sizes,
            )
        };
        let storage = create_host_buffer(
            context,
            "acceleration structure",
            &vec![0; to_usize(sizes.acceleration_structure_size)?.max(1)],
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let create = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(storage.buffer)
            .size(sizes.acceleration_structure_size)
            .ty(ty);
        self.buffers.push(storage);
        // SAFETY: the storage buffer is live and bound to memory.
        let structure = unsafe { loader.create_acceleration_structure(&create, None) }
            .map_err(|error| format!("create acceleration structure: {error}"))?;
        self.created.push(structure);
        let alignment = u64::from(scratch_alignment(context)).max(1);
        let scratch = create_host_buffer(
            context,
            "acceleration structure scratch",
            &vec![0; to_usize(sizes.build_scratch_size + alignment)?],
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        )?;
        let scratch_address = address(context, &scratch).next_multiple_of(alignment);
        self.buffers.push(scratch);
        Ok((
            structure,
            Build {
                ty,
                structure,
                geometry,
                primitives,
                scratch: scratch_address,
            },
        ))
    }
}

fn to_usize(size: vk::DeviceSize) -> Result<usize, ExecutorError> {
    usize::try_from(size).map_err(|_| format!("a structure size {size} overflows").into())
}

/// The device address of `buffer`, created with `SHADER_DEVICE_ADDRESS`.
fn address(context: &VulkanContext, buffer: &HostBuffer) -> vk::DeviceAddress {
    let info = vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer);
    // SAFETY: the buffer was created with SHADER_DEVICE_ADDRESS and its memory is bound.
    unsafe { context.device.get_buffer_device_address(&info) }
}

/// `minAccelerationStructureScratchOffsetAlignment`.
fn scratch_alignment(context: &VulkanContext) -> u32 {
    let mut structures = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut structures);
    // SAFETY: the physical device is the context's; `properties` is live for the call.
    unsafe {
        context
            .instance
            .get_physical_device_properties2(context.physical, &mut properties)
    };
    structures.min_acceleration_structure_scratch_offset_alignment
}
