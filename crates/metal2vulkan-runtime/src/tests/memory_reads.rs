//! A global memory barrier's destination names `MEMORY_READ` as the reads it stands for beside
//! its destination stages on the recording family of a device with its extensions enabled
//! (`image_barrier.rs`'s `MemoryReads`), and as nothing narrower or wider: each read the registry
//! makes valid there, or `MEMORY_READ` itself where one of them has no `VkAccessFlags` bit.

mod registry;

use crate::image_barrier::{ImageDst, MemoryReads, EXTENSION_READS};
use ash::vk;
use metal2vulkan::vk::{AccessFlags2, PipelineStageFlags, PipelineStageFlags2};
use registry::{DEPENDS, GRANTS, STAGES};
use std::collections::BTreeSet;
use std::ffi::CStr;

/// The reads `VK_ACCESS_2_SHADER_READ_BIT` "is equivalent to the logical OR of" (the Vulkan
/// specification's `VkAccessFlagBits2`), which `VkAccessFlags`'s `SHADER_READ` names: valid
/// beside the stages `SHADER_READ` is.
const SHADER_READS: [&str; 4] = [
    "SHADER_SAMPLED_READ_BIT",
    "SHADER_STORAGE_READ_BIT",
    "SHADER_BINDING_TABLE_READ_BIT_KHR",
    "SHADER_TILE_ATTACHMENT_READ_BIT_QCOM",
];

/// `VK_QUEUE_DATA_GRAPH_BIT_ARM`, which `ash 0.38` predates.
const DATA_GRAPH: vk::QueueFlags = vk::QueueFlags::from_raw(0x400);

/// The kinds of queue family: transfers alone (with sparse binding, as NVIDIA's), compute,
/// graphics alone, graphics with compute, and the families of video, optical flow and data
/// graph work, alone and with transfers.
const FAMILIES: [vk::QueueFlags; 13] = [
    vk::QueueFlags::TRANSFER,
    vk::QueueFlags::from_raw(
        vk::QueueFlags::TRANSFER.as_raw() | vk::QueueFlags::SPARSE_BINDING.as_raw(),
    ),
    vk::QueueFlags::COMPUTE,
    vk::QueueFlags::from_raw(vk::QueueFlags::COMPUTE.as_raw() | vk::QueueFlags::TRANSFER.as_raw()),
    vk::QueueFlags::GRAPHICS,
    vk::QueueFlags::from_raw(vk::QueueFlags::GRAPHICS.as_raw() | vk::QueueFlags::COMPUTE.as_raw()),
    vk::QueueFlags::from_raw(
        vk::QueueFlags::GRAPHICS.as_raw()
            | vk::QueueFlags::COMPUTE.as_raw()
            | vk::QueueFlags::TRANSFER.as_raw()
            | vk::QueueFlags::SPARSE_BINDING.as_raw(),
    ),
    vk::QueueFlags::VIDEO_DECODE_KHR,
    vk::QueueFlags::from_raw(
        vk::QueueFlags::VIDEO_DECODE_KHR.as_raw() | vk::QueueFlags::TRANSFER.as_raw(),
    ),
    vk::QueueFlags::VIDEO_ENCODE_KHR,
    vk::QueueFlags::OPTICAL_FLOW_NV,
    DATA_GRAPH,
    vk::QueueFlags::from_raw(DATA_GRAPH.as_raw() | vk::QueueFlags::TRANSFER.as_raw()),
];

/// A row of [`STAGES`]: its name, queue flags, extensions and equivalents.
type StageRow = (
    &'static str,
    u32,
    &'static [&'static [&'static CStr]],
    &'static [&'static str],
);

/// The bits of a registry access name (`VK_ACCESS_2_` left off).
fn access_bits(name: &str) -> u64 {
    AccessFlags2::from_name(&format!("VK_ACCESS_2_{name}"))
        .unwrap_or_else(|| panic!("{name} is no access the registry tables name"))
        .as_raw()
}

/// A stage's row of [`STAGES`] by its name (`VK_PIPELINE_STAGE_2_` left off).
fn stage(name: &str) -> &'static StageRow {
    STAGES
        .iter()
        .find(|row| row.0 == name)
        .unwrap_or_else(|| panic!("{name} is no stage of the registry table"))
}

/// The registry name of a stage bit (`VK_PIPELINE_STAGE_2_` left off).
fn stage_name(bit: u64) -> &'static str {
    let (_, name) = (PipelineStageFlags2::KNOWN.iter())
        .find(|(stage, _)| stage.as_raw() == bit)
        .unwrap_or_else(|| panic!("{bit:#x} is no stage the registry tables name"));
    name.trim_start_matches("VK_PIPELINE_STAGE_2_")
}

/// `enabled` with every extension each requires.
fn closed(enabled: &[&'static CStr]) -> BTreeSet<&'static CStr> {
    let mut all: BTreeSet<&'static CStr> = enabled.iter().copied().collect();
    for &extension in enabled {
        if let Some((_, requires)) = DEPENDS.iter().find(|row| row.0 == extension) {
            all.extend(requires.iter().copied());
        }
    }
    all
}

/// Whether a device with `enabled` has the stage `name`.
fn has(name: &str, enabled: &BTreeSet<&CStr>) -> bool {
    let needs = stage(name).2;
    needs
        .iter()
        .any(|all| all.iter().all(|e| enabled.contains(e)))
}

/// Every destination stage mask of one stage `vkCmdPipelineBarrier` may take on a family of
/// `family`'s queue flags with `enabled`: each `VkPipelineStageFlags` bit the device has and the
/// family runs (or no queue limits), `ALL_COMMANDS` among them.
fn single_stages(family: vk::QueueFlags, enabled: &BTreeSet<&CStr>) -> Vec<vk::PipelineStageFlags> {
    (PipelineStageFlags::KNOWN.iter())
        .filter(|(bit, _)| bit.as_raw().count_ones() == 1)
        .filter(|(bit, _)| {
            let name = stage_name(u64::from(bit.as_raw()));
            let queues = vk::QueueFlags::from_raw(stage(name).1);
            has(name, enabled) && (queues.is_empty() || family.intersects(queues))
        })
        .map(|(bit, _)| vk::PipelineStageFlags::from_raw(bit.as_raw()))
        .collect()
}

/// What `MEMORY_READ` in a global barrier's destination beside the stages `dst`, recorded for a
/// family of `family`'s queue flags on a device with `enabled` (with what each requires), must
/// be named as ([`GRANTS`]): every read a stage of the mask (or one it stands for) supports
/// there, `ALL_COMMANDS` holding every stage the family runs; or `MEMORY_READ` itself where one
/// of them has no `VkAccessFlags` bit, or where the family runs none of graphics, compute and
/// transfers.
fn expected(
    family: vk::QueueFlags,
    enabled: &BTreeSet<&CStr>,
    dst: vk::PipelineStageFlags,
) -> vk::AccessFlags {
    let shaders_or_transfers =
        vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
    if !family.intersects(shaders_or_transfers) {
        return vk::AccessFlags::MEMORY_READ;
    }
    let every = dst.contains(vk::PipelineStageFlags::ALL_COMMANDS);
    let mut named: BTreeSet<&str> = BTreeSet::new();
    for bit in (0..32)
        .map(|i| 1u32 << i)
        .filter(|bit| dst.as_raw() & bit != 0)
    {
        let name = stage_name(u64::from(bit));
        named.insert(name);
        named.extend(stage(name).3.iter().copied());
    }
    let mut reads = vk::AccessFlags::empty();
    for (needs, access, at) in GRANTS {
        if !needs.iter().all(|e| enabled.contains(e)) || SHADER_READS.contains(&access) {
            continue;
        }
        let held = match every {
            true => family.intersects(vk::QueueFlags::from_raw(stage(at).1)),
            false => named.contains(at),
        };
        if !held {
            continue;
        }
        match u32::try_from(access_bits(access)) {
            Ok(bit) => reads |= vk::AccessFlags::from_raw(bit),
            Err(_) => return vk::AccessFlags::MEMORY_READ,
        }
    }
    reads
}

/// The sets of extensions to check with: none, all those the registry rows or the runtime's
/// table name, and each of them, each set with what its extensions require.
fn extension_sets() -> Vec<BTreeSet<&'static CStr>> {
    let mut names: BTreeSet<&'static CStr> = BTreeSet::new();
    names.extend(GRANTS.iter().flat_map(|row| row.0.iter().copied()));
    names.extend(
        STAGES
            .iter()
            .flat_map(|row| row.2.iter().flat_map(|all| all.iter().copied())),
    );
    names.extend(EXTENSION_READS.iter().map(|row| row.0));
    let all: Vec<&'static CStr> = names.iter().copied().collect();
    let mut sets = vec![BTreeSet::new(), closed(&all)];
    sets.extend(names.iter().map(|&name| closed(&[name])));
    sets
}

/// `MemoryReads` against [`expected`] beside `dst`.
fn check(family: vk::QueueFlags, enabled: &BTreeSet<&'static CStr>, dst: vk::PipelineStageFlags) {
    let named =
        MemoryReads::of(family, enabled.iter().copied()).named(vk::AccessFlags::MEMORY_READ, dst);
    let expected = expected(family, enabled, dst);
    assert_eq!(
        named.as_raw(),
        expected.as_raw(),
        "{:#x} beside {:#x} with {enabled:?}",
        family.as_raw(),
        dst.as_raw()
    );
}

/// On every kind of family, with no extension, all of them, and each with what it requires,
/// beside every destination stage the device has and the family runs, `ALL_COMMANDS` among them
/// (and, with none and all, every pair of them): `MEMORY_READ` is named as the reads the registry
/// makes valid there, and nothing else. The runtime's destination masks are unions of these.
#[test]
fn memory_read_is_every_read_valid_beside_the_stages() {
    let sets = extension_sets();
    for family in FAMILIES {
        for (i, enabled) in sets.iter().enumerate() {
            let singles = single_stages(family, enabled);
            for &dst in &singles {
                check(family, enabled, dst);
            }
            if i < 2 {
                for (j, &a) in singles.iter().enumerate() {
                    for &b in &singles[j + 1..] {
                        check(family, enabled, a | b);
                    }
                }
            }
        }
    }
}

/// Beside `ALL_TRANSFER` alone, a family of graphics and compute names a transfer's read alone,
/// never an indirect command's, a shader's or a command preprocess's.
#[test]
fn beside_transfers_alone_a_transfer_read_alone() {
    let family = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
    let reads = MemoryReads::of(family, [c"VK_EXT_device_generated_commands"]);
    let named = reads.named(
        vk::AccessFlags::MEMORY_READ,
        vk::PipelineStageFlags::TRANSFER,
    );
    assert_eq!(named.as_raw(), vk::AccessFlags::TRANSFER_READ.as_raw());
}

/// Beside `ALL_COMMANDS`, a family of transfers alone names a transfer's read alone; a family of
/// compute, graphics, or both names more; and every read of an image [`ImageDst::Every`] names
/// is among them.
#[test]
fn transfers_alone_name_the_transfer_read_alone() {
    let transfer_read = vk::AccessFlags::TRANSFER_READ;
    for family in FAMILIES {
        let named = MemoryReads::of(family, std::iter::empty()).named(
            vk::AccessFlags::MEMORY_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
        let transfers_alone = family.contains(vk::QueueFlags::TRANSFER)
            && !family.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE);
        assert_eq!(
            named == transfer_read,
            transfers_alone,
            "{:#x}: {:#x}",
            family.as_raw(),
            named.as_raw()
        );
        if !named.contains(vk::AccessFlags::MEMORY_READ) {
            let image = ImageDst::Every.access(family) & !vk::AccessFlags::MEMORY_WRITE;
            assert!(
                named.contains(image),
                "{:#x}: {:#x} lacks {:#x}",
                family.as_raw(),
                named.as_raw(),
                image.as_raw()
            );
        }
    }
}

/// Only `MEMORY_READ` is replaced: the rest of a destination stays, and a destination without it
/// is unchanged.
#[test]
fn named_replaces_memory_read_alone() {
    let family = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
    let reads = MemoryReads::of(family, std::iter::empty());
    let every = vk::PipelineStageFlags::ALL_COMMANDS;
    let write = vk::AccessFlags::MEMORY_WRITE;
    assert_eq!(reads.named(write, every).as_raw(), write.as_raw());
    assert_eq!(reads.named(vk::AccessFlags::empty(), every).as_raw(), 0);
    let both = reads.named(vk::AccessFlags::MEMORY_READ | write, every);
    assert!(both.contains(write) && !both.contains(vk::AccessFlags::MEMORY_READ));
    let alone = reads.named(vk::AccessFlags::MEMORY_READ, every);
    assert_eq!((both & !write).as_raw(), alone.as_raw());
}

/// Every read and every stage the registry tables (`metal2vulkan::vk`, generated from the pinned
/// registry) name is one [`GRANTS`] and [`STAGES`] hold: a regenerated registry with a new read
/// or stage fails here until both tables, and the runtime's, take it.
#[test]
fn every_registry_read_and_stage_is_reviewed() {
    for (_, name) in AccessFlags2::KNOWN {
        let short = name.trim_start_matches("VK_ACCESS_2_");
        if !short.contains("_READ") || ["MEMORY_READ_BIT", "HOST_READ_BIT"].contains(&short) {
            continue;
        }
        let listed = GRANTS.iter().any(|row| row.1 == short) || SHADER_READS.contains(&short);
        assert!(listed, "{name} is a read no row names");
    }
    for (_, name) in PipelineStageFlags2::KNOWN {
        let short = name.trim_start_matches("VK_PIPELINE_STAGE_2_");
        assert!(
            STAGES.iter().any(|row| row.0 == short),
            "{name} is a stage no row names"
        );
    }
}
