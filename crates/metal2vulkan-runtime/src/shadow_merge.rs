//! A shadowed texture's writes merged into its buffer's bytes: one command that writes through a
//! shadowed texture (`shadow.rs`) and also writes the same buffer's bytes another way (through the
//! buffer, a texture buffer, or another texture over them). A copy back of the shadow would replace
//! the other write's bytes with the shadow's, where Metal keeps both writes when they touch
//! different bytes (Metal probe cases `k.two_ways_disjoint` and `k.two_ways_gap`, M3 Max and M4),
//! so the bytes the command changed through the texture are merged into the buffer's instead:
//!
//! - **Kept.** Before the command, after its copies in, the shadow's texels are copied into the
//!   *kept* part of a scratch buffer, laid out as in the buffer from the texture's first byte.
//! - **Merged.** After the command (before the encoder's next command, or as it ends), the
//!   shadow's texels are copied into the *changed* part, and the buffer's bytes from the
//!   texture's first to its last into the *current* part. One dispatch of the merge kernel then
//!   takes, in each byte of the current part, the changed part's byte where it differs from the
//!   kept one, and the current part is copied back into the buffer. A byte the command wrote
//!   both ways is the texture's where its value changed: Metal leaves such a race unordered.
//! - **The scratch** is the command's, retained until its command buffer completes: three parts
//!   of the texture's bytes rounded up to 256, in one buffer where the device's buffers take it
//!   (else one each), all zero when made, so the bytes between the texture's rows, which no copy
//!   of the shadow writes, are the same in the kept and changed parts.
//! - **The kernel** (`bind::shadow_merge_module`, one per device, `owned_kernel.rs`): one
//!   invocation per 4-byte word, the three parts bound as its kept, changed and current storage
//!   buffers, in windows of at most `maxStorageBufferRange` bytes and
//!   `maxComputeWorkGroupCount[0]` groups, one dispatch each.
//!
//! It costs the scratch, three copies and the dispatches in place of one copy back, which the
//! runtime announces once per device (`shadow-merge`, performance).

use super::buffer::Buffer;
use super::degradation::Degradation;
use super::error::{internal, Error};
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::resource::ResourceKey;
use super::shadow::{ShadowCopies, ShadowSource};
use super::tracker::{Access, Barrier, Use, UseRange};
use super::values::ResourceOptions;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::ShadowMergeModule;
use std::ops::Range;

/// Bytes one workgroup merges: a word per invocation.
const GROUP_BYTES: u64 = 4 * ShadowMergeModule::LOCAL_SIZE as u64;
/// What each part of the scratch is rounded up to: a multiple of every
/// `minStorageBufferOffsetAlignment` and of every texel's size.
const PART_ALIGNMENT: u64 = 256;

/// One part of the scratch: its `VkBuffer` and where it starts there.
#[derive(Clone, Copy)]
struct Part {
    raw: vk::Buffer,
    at: u64,
}

/// The bytes a merge writes: the texture's, in its buffer.
struct Merged {
    buffer: vk::Buffer,
    key: ResourceKey,
    bytes: Range<u64>,
    /// Where the texture's first byte lies in `buffer`.
    first: u64,
}

/// The scratch and the dispatches of one command's merge (the module docs).
pub(super) struct ShadowMerge {
    /// The texture's bytes.
    merged: Merged,
    /// The kept, changed and current parts.
    parts: [Part; 3],
    /// The kept part's identity and range, for the tracker: kept before the command, read by the
    /// merge after it.
    kept: (ResourceKey, bool, Range<u64>),
    /// The buffers the parts lie in, held until the command buffer completes.
    _held: Vec<Buffer>,
    kernel: OwnedKernel,
    /// One set per window, each binding the window of the three parts.
    sets: OwnedSets,
    /// Each window's workgroups and words.
    groups: Vec<(u32, u32)>,
}

impl ShadowMerge {
    /// The scratch and the kernel's sets for a merge of the texture of `copies` on `device`,
    /// announced once per device. Refused as `new_buffer` refuses the scratch.
    pub(super) fn new(device: &Device, copies: &ShadowCopies) -> Result<ShadowMerge, Error> {
        let (
            ShadowSource::Bytes {
                buffer, key, bytes, ..
            },
            Some(first),
        ) = (&copies.source, copies.first_byte())
        else {
            // A command writing an image two ways is refused before (`ShadowSync::before`).
            return Err(internal("a merge of an image's shadow"));
        };
        let merged = Merged {
            buffer: buffer.raw(),
            key: *key,
            bytes: bytes.clone(),
            first,
        };
        let span = merged.bytes.end - merged.bytes.start;
        let part = span.next_multiple_of(PART_ALIGNMENT);
        let options = ResourceOptions::private();
        let place = |buffer: &Buffer, at: u64| {
            let (raw, at) = buffer.at(at);
            Part { raw, at }
        };
        // The kept part is the first `part` bytes of the first buffer either way.
        let (held, parts) = if 3 * part <= device.properties().max_buffer_length {
            let one = device.own_buffer(3 * part, options)?;
            let parts = [0, 1, 2].map(|role| place(&one, role * part));
            (vec![one], parts)
        } else {
            let [kept, changed, current] = [(); 3].map(|()| device.own_buffer(part, options));
            let [kept, changed, current] = [kept?, changed?, current?];
            let parts = [place(&kept, 0), place(&changed, 0), place(&current, 0)];
            (vec![kept, changed, current], parts)
        };
        let kept = {
            let first = held
                .first()
                .ok_or_else(|| internal("a merge without its scratch"))?;
            let (identity, range) = first.tracked(0..part);
            (identity.key, identity.tracked, range)
        };
        let limits = &device.properties().description.limits;
        let window = (u64::from(limits.max_storage_buffer_range) / PART_ALIGNMENT * PART_ALIGNMENT)
            .min(u64::from(limits.max_compute_work_group_count[0]) * GROUP_BYTES);
        if window == 0 {
            return Err(internal(
                "a device whose storage buffers hold no merge window",
            ));
        }
        let mut ranges = Vec::new();
        let mut groups = Vec::new();
        let mut start = 0;
        while start < part {
            let size = window.min(part - start);
            ranges.push(
                parts
                    .iter()
                    .map(|p| (p.raw, p.at + start, size))
                    .collect::<Vec<_>>(),
            );
            let words =
                u32::try_from(size / 4).map_err(|_| internal("a merge window past u32 words"))?;
            groups.push((words.div_ceil(ShadowMergeModule::LOCAL_SIZE), words));
            start += size;
        }
        let kernel = device.owned_kernel(OwnedModule::ShadowMerge)?;
        let sets = OwnedSets::write(device, &kernel, &ranges)?;
        let bytes = merged.bytes.clone();
        device.degrade(Degradation::ShadowMerge, || {
            let object = format!("a command writing bytes {bytes:?} of a buffer two ways");
            let reason = "it writes them through a shadowed texture over the buffer and also \
                          another way, and a copy back of the shadow would replace the other \
                          write's bytes, which Metal keeps"
                .to_owned();
            (object, reason)
        });
        Ok(ShadowMerge {
            merged,
            parts,
            kept,
            _held: held,
            kernel,
            sets,
            groups,
        })
    }

    /// The uses of keeping the shadow of `copies`: a read of the shadow and a write of the kept
    /// part.
    pub(super) fn keep_uses(&self, copies: &ShadowCopies) -> Vec<Use<ResourceKey>> {
        let (key, tracked, range) = self.kept.clone();
        vec![
            Use::reads(
                copies.shadow_key,
                copies.tracked,
                copies.shadow_range.clone(),
                transfer(vk::AccessFlags2::TRANSFER_READ),
            ),
            Use::writes(
                key,
                tracked,
                UseRange::Bytes(range),
                transfer(vk::AccessFlags2::TRANSFER_WRITE),
            ),
        ]
    }

    /// The uses of merging the shadow of `copies`: a read of the shadow and of the kept part, and
    /// a read and a write of the texture's bytes.
    pub(super) fn merge_uses(&self, copies: &ShadowCopies) -> Vec<Use<ResourceKey>> {
        let (key, tracked, range) = self.kept.clone();
        vec![
            Use::reads(
                copies.shadow_key,
                copies.tracked,
                copies.shadow_range.clone(),
                transfer(vk::AccessFlags2::TRANSFER_READ),
            ),
            Use::reads(
                key,
                tracked,
                UseRange::Bytes(range),
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
            ),
            Use {
                key: self.merged.key,
                tracked: copies.tracked,
                range: UseRange::Bytes(self.merged.bytes.clone()),
                read: transfer(vk::AccessFlags2::TRANSFER_READ),
                write: transfer(vk::AccessFlags2::TRANSFER_WRITE),
            },
        ]
    }

    /// Record the keep of the shadow of `copies` into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the buffer, the shadow and the
    /// scratch are live until it completes, and an image is in the `GENERAL` layout.
    pub(super) unsafe fn record_keep(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        copies: &ShadowCopies,
    ) {
        let [kept, _, _] = self.parts;
        // SAFETY: the caller's contract; the kept part holds the texture's bytes.
        unsafe { copies.record_out(device, command_buffer, kept.raw, kept.at) };
    }

    /// Record the merge of the shadow of `copies` into its buffer's bytes (the module docs).
    ///
    /// # Safety
    ///
    /// As `record_keep`, after the keep and the command it surrounds.
    pub(super) unsafe fn record_merge(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        copies: &ShadowCopies,
    ) {
        let [_, changed, current] = self.parts;
        let Merged {
            buffer,
            ref bytes,
            first,
            ..
        } = self.merged;
        let span = bytes.end - bytes.start;
        // SAFETY: the caller's contract; each part holds the texture's bytes, and the sets bind
        // windows inside the parts.
        unsafe {
            copies.record_out(device, command_buffer, changed.raw, changed.at);
            device.cmd_copy_buffer(
                command_buffer,
                buffer,
                current.raw,
                &[vk::BufferCopy {
                    src_offset: first,
                    dst_offset: current.at,
                    size: span,
                }],
            );
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            for (set, &(groups, words)) in self.sets.sets.iter().zip(&self.groups) {
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.kernel.layout,
                    0,
                    std::slice::from_ref(set),
                    &[],
                );
                // The window's words, one row of them.
                let shape = ShadowMergeModule::shape(words, words, 0, 0);
                device.cmd_push_constants(
                    command_buffer,
                    self.kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &shape,
                );
                device.cmd_dispatch(command_buffer, groups, 1, 1);
            }
            Barrier::everything().record(device, command_buffer);
            device.cmd_copy_buffer(
                command_buffer,
                current.raw,
                buffer,
                &[vk::BufferCopy {
                    src_offset: current.at,
                    dst_offset: first,
                    size: span,
                }],
            );
        }
    }
}

fn transfer(access: vk::AccessFlags2) -> Access {
    Access::new(vk::PipelineStageFlags2::COPY, access)
}
