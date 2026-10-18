//! A heap's two-plane extended-range texture (`RGB10A8_2P_XR10`, `RGB10A8_2P_XR10_sRGB`) whose
//! texels lie in the heap's bytes as Metal lays them (the consumer's report). The moves serve such
//! a texture over a surface's two planes too, at the planes' rows
//! (`texture/surface_two_planes.rs`).
//!
//! **Metal's layout** (an Apple M4 under macOS 27.0.1, plain; a buffer over
//! the heap's bytes from 0 in a placement heap, and one placed after `makeAliasable` in an
//! automatic heap; `Shared` and `Private` storage alike, so a `Private` texture's bytes are not
//! tiled or compressed; 2D textures of 1 x 1, 16 x 12, 37 x 5, 64 x 64, 300 x 70 and 333 x 61 at
//! offsets 0, 16, 48 and 4096; four fills of the buffer naming each byte's offset read back
//! through a kernel's `read()`, and a kernel's `write()` of every texel read through the buffer):
//! a one-level 2D texture's plane 0, each texel's color word, lies from the texture's first byte
//! in rows `align(4 w, 16)` bytes apart; its plane 1, each texel's alpha byte, from
//! `align(h · align(4 w, 16), 16384)` bytes past it, in rows `align(w, 16)` bytes apart
//! ([`TwoPlaneHeapLayout`]): `heapTextureSizeAndAlignWithDescriptor:`'s answer is the span to
//! plane 1's last texel. A write lands on those bytes alone (the bytes between the rows stay),
//! with bits 30 and 31 of each color word 0. The sRGB twin lays its planes out alike. A 2D
//! array's plane 0 slices lie `align(h · align(4 w, 16), 128)` bytes apart, but its alpha reads
//! 0 and its alpha writes land in none of the 1 MiB measured; a mipmapped texture's levels are
//! not laid out in rows. Both stay held apart from the heap's bytes (`heap.rs`,
//! `heap-two-plane-texture`).
//!
//! **Held** as the texture's one `R32G32_UINT` image (each texel's color word, then its alpha
//! byte), in memory of its own, a shadow of the heap's bytes (`shadow.rs`,
//! `ShadowTarget::Planes`): into the image, the texture's bytes copied into a staging buffer,
//! one dispatch of `bind::two_plane_texels_module` moving them into a tight buffer of the image's
//! texels, and that buffer copied into the image; back, the image into the tight buffer, the
//! kernel into the staging buffer, and each row of each plane on into the heap's bytes, so the
//! bytes between the rows stay. Each step is ordered after the one before by a full barrier, and
//! the buffers are the texture's for its life. A buffer or texture over the texture's bytes, the
//! CPU's writes through the heap's pages and the texture's own CPU copies (plane 0's color words
//! at Metal's rows, `heap_plane_layout`) then meet the same bytes as on Metal.

use super::super::buffer::Buffer;
use super::super::error::{internal, Error};
use super::super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::super::row_copies::RowCopies;
use super::super::shadow::{self, ShadowStaging};
use super::super::tracker::Barrier;
use super::super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{PlaneLayout, TwoPlaneTexelsModule};

/// The multiple of bytes past plane 0's start that plane 1 starts at.
const ALPHA_PAGE: u64 = 16 << 10;
/// The multiple of bytes both planes' rows lie at.
const ROW_ALIGNMENT: u64 = 16;
/// Where a shape lies in the shape buffer, per direction: a multiple of every
/// `minStorageBufferOffsetAlignment`.
const SHAPE_SLOT: u64 = 256;

/// Metal's layout of a heap's one-level 2D two-plane texture of `width x height` texels, from
/// its first byte (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TwoPlaneHeapLayout {
    pub(crate) width: u64,
    pub(crate) height: u64,
    /// Plane 0's rows apart: `align(4 w, 16)`.
    pub(crate) color_pitch: u64,
    /// Plane 1's first byte: `align(h · color_pitch, 16384)`.
    pub(crate) alpha_at: u64,
    /// Plane 1's rows apart: `align(w, 16)`.
    pub(crate) alpha_pitch: u64,
}

impl TwoPlaneHeapLayout {
    /// The layout of a `[width, height]` texture.
    pub(crate) fn of([width, height]: [u64; 2]) -> TwoPlaneHeapLayout {
        let color_pitch = (4 * width).next_multiple_of(ROW_ALIGNMENT);
        TwoPlaneHeapLayout {
            width,
            height,
            color_pitch,
            alpha_at: (height * color_pitch).next_multiple_of(ALPHA_PAGE),
            alpha_pitch: width.next_multiple_of(ROW_ALIGNMENT),
        }
    }

    /// The layout of a `[width, height]` texture whose plane 0's rows lie `color_pitch` bytes
    /// apart from its first byte, and plane 1's `alpha_pitch` apart from `alpha_at` past it: a
    /// surface's two planes as Metal's GPU reads them (`texture/surface_two_planes.rs`).
    pub(crate) fn at_rows(
        [width, height]: [u64; 2],
        color_pitch: u64,
        (alpha_at, alpha_pitch): (u64, u64),
    ) -> TwoPlaneHeapLayout {
        TwoPlaneHeapLayout {
            width,
            height,
            color_pitch,
            alpha_at,
            alpha_pitch,
        }
    }

    /// The bytes the planes span: to plane 1's last texel, Metal's answer.
    pub(crate) fn span(&self) -> u64 {
        self.alpha_at + (self.height - 1) * self.alpha_pitch + self.width
    }

    /// Each row of each plane, as (its first byte, its bytes), plane 0's first.
    fn rows(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        let color = (0..self.height).map(|y| (y * self.color_pitch, 4 * self.width));
        let alpha = (0..self.height).map(|y| (self.alpha_at + y * self.alpha_pitch, self.width));
        color.chain(alpha)
    }
}

/// One direction of the kernel's moves: its shape, where it lies in the shape buffer, and the
/// dispatch's workgroups.
struct Move {
    shape: [u8; 52],
    at: u64,
    groups: [u32; 2],
}

/// A heap two-plane texture's moves between its image and its planes in the heap's bytes (the
/// module docs): the tight buffer of the image's texels, the shapes, and the kernel's two sets
/// (into the image, then back).
pub(crate) struct HeapPlaneCopies {
    /// The image's texels, 8 bytes each; held for the texture's life. No command's uses name it:
    /// every copy writes the image or the heap's bytes, so the tracker orders it after every
    /// other copy of the texture, and each records a full barrier first.
    _tight: Buffer,
    tight_raw: vk::Buffer,
    /// The tight buffer's copies to and from the image.
    regions: [vk::BufferImageCopy; 1],
    /// The two shapes; held for the texture's life.
    _shapes: Buffer,
    shapes_raw: vk::Buffer,
    moves: [Move; 2],
    kernel: OwnedKernel,
    sets: OwnedSets,
}

impl HeapPlaneCopies {
    /// Whether the moves of `layout` fit `device`: each buffer within `maxStorageBufferRange`
    /// and each dispatch within `maxComputeWorkGroupCount`.
    pub(crate) fn fit(device: &Device, layout: &TwoPlaneHeapLayout) -> bool {
        let limits = &device.properties().description.limits;
        let range = u64::from(limits.max_storage_buffer_range);
        let texels = layout.width * layout.height;
        let invocations = texels + layout.width.div_ceil(4) * layout.height;
        let groups = invocations.div_ceil(u64::from(TwoPlaneTexelsModule::LOCAL_SIZE));
        let across = groups.min(u64::from(limits.max_compute_work_group_count[0]));
        8 * texels <= range
            && layout.span().next_multiple_of(4) <= range
            && groups.div_ceil(across.max(1)) <= u64::from(limits.max_compute_work_group_count[1])
    }

    /// The staging buffer of a texture of `layout` whose first byte lies at `at` of the heap's
    /// `VkBuffer`: the texture's bytes, from its first to plane 1's last, at its start, of the
    /// texture's `hazard` tracking, in whole 4-byte words; each row of each plane copied back.
    pub(crate) fn staging(
        device: &Device,
        layout: &TwoPlaneHeapLayout,
        at: u64,
        hazard: HazardTrackingMode,
    ) -> Result<ShadowStaging, Error> {
        let span = layout.span();
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
        let buffer = device.own_buffer(span.next_multiple_of(4), options)?;
        let (_, start) = buffer.at(0);
        let back = layout
            .rows()
            .map(|(row, size)| vk::BufferCopy {
                src_offset: start + row,
                dst_offset: at + row,
                size,
            })
            .collect();
        Ok(ShadowStaging {
            into: vk::BufferCopy {
                src_offset: at,
                dst_offset: start,
                size: span,
            },
            back,
            buffer,
        })
    }

    /// The moves of a texture of `layout` through `staging` (its [`staging`](Self::staging)), of
    /// the texture's `hazard` tracking. Refused as `new_buffer` refuses the tight and shape
    /// buffers; the caller checked that they [`fit`](Self::fit).
    pub(crate) fn new(
        device: &Device,
        staging: &ShadowStaging,
        layout: &TwoPlaneHeapLayout,
        hazard: HazardTrackingMode,
    ) -> Result<HeapPlaneCopies, Error> {
        let narrow =
            |v: u64| u32::try_from(v).map_err(|_| internal("a two-plane texture past u32"));
        let limits = &device.properties().description.limits;
        let [width, height] = [narrow(layout.width)?, narrow(layout.height)?];
        let tight_bytes = 8 * layout.width * layout.height;
        let staged_bytes = layout.span().next_multiple_of(4);
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
        let tight = device.own_buffer(tight_bytes, options)?;
        let shapes = device.own_buffer(2 * SHAPE_SLOT, options)?;
        let (tight_raw, tight_at) = tight.at(0);
        let (shapes_raw, shapes_at) = shapes.at(0);
        let (staging_raw, staging_at) = (staging.raw(), staging.into.dst_offset);
        let local = TwoPlaneTexelsModule::LOCAL_SIZE;
        // Within the device's counts: the caller checked that the moves fit.
        let grid = |invocations: u32| -> [u32; 2] {
            let groups = invocations.div_ceil(local).max(1);
            let across = groups.min(limits.max_compute_work_group_count[0]);
            [across, groups.div_ceil(across)]
        };
        let texels = width * height;
        let into_groups = grid(texels);
        let back_groups = grid(texels + TwoPlaneTexelsModule::alpha_words([width, height], 1));
        let color = PlaneLayout::new(0, narrow(layout.color_pitch / 4)?, 0);
        let alpha = PlaneLayout::new(
            narrow(layout.alpha_at / 4)?,
            narrow(layout.alpha_pitch / 4)?,
            0,
        );
        let moves = [true, false].map(|into| {
            let groups = if into { into_groups } else { back_groups };
            Move {
                shape: TwoPlaneTexelsModule::shape(
                    [width, height],
                    1,
                    color,
                    alpha,
                    into,
                    groups[0] * local,
                ),
                at: shapes_at + if into { 0 } else { SHAPE_SLOT },
                groups,
            }
        });
        let shape_bytes = 4 * u64::from(TwoPlaneTexelsModule::SHAPE_WORDS);
        let kernel = device.owned_kernel(OwnedModule::TwoPlaneTexels)?;
        // Each set binds the texels, the planes and the shape, in binding order.
        let sets = OwnedSets::write(
            device,
            &kernel,
            &moves
                .iter()
                .map(|step| {
                    vec![
                        (tight_raw, tight_at, tight_bytes),
                        (staging_raw, staging_at, staged_bytes),
                        (shapes_raw, step.at, shape_bytes),
                    ]
                })
                .collect::<Vec<_>>(),
        )?;
        let regions = [vk::BufferImageCopy {
            buffer_offset: tight_at,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        }];
        Ok(HeapPlaneCopies {
            _tight: tight,
            tight_raw,
            regions,
            _shapes: shapes,
            shapes_raw,
            moves,
            kernel,
            sets,
        })
    }

    /// Record the move of the texture's `image` from `bytes` (the heap's `VkBuffer`) through
    /// `staging` (`into`), or back, its rows back by `back` (`shadow::record_back`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the image, the buffers and the
    /// sets are live until it completes, and the image rests in `GENERAL`; `back` is the copy's
    /// `ShadowCopy::staged_back`, over `bytes`.
    pub(crate) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        (bytes, staging): (vk::Buffer, &ShadowStaging),
        into: bool,
        back: Option<&RowCopies>,
    ) {
        let (step, set) = match into {
            true => (&self.moves[0], self.sets.sets[0]),
            false => (&self.moves[1], self.sets.sets[1]),
        };
        let general = vk::ImageLayout::GENERAL;
        // SAFETY: the caller's contract; the copies lie in the buffers and the image (planned
        // when the texture was made), the shape's 52 bytes fit its slot, and the sets bind the
        // texture's own buffers.
        unsafe {
            Barrier::everything().record(device, command_buffer);
            if into {
                device.cmd_copy_buffer(command_buffer, bytes, staging.raw(), &[staging.into]);
            } else {
                device.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    general,
                    self.tight_raw,
                    &self.regions,
                );
            }
            device.cmd_update_buffer(command_buffer, self.shapes_raw, step.at, &step.shape);
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(command_buffer, step.groups[0], step.groups[1], 1);
            Barrier::everything().record(device, command_buffer);
            if into {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.tight_raw,
                    image,
                    general,
                    &self.regions,
                );
            } else {
                shadow::record_back(device, command_buffer, (staging, bytes), back);
            }
        }
    }
}
