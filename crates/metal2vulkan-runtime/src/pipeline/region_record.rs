//! The keys of the `VkPipeline`s a compute variant's dispatches made, kept on disk, so that a later
//! process's state makes them at its creation, where Metal compiles (`dispatch_making.rs`).
//!
//! - **What it holds.** One record per variant content: the region keys (a region's
//!   specialization, its stage flags and its required subgroup size; one per threadgroup size and
//!   edge region a dispatch ran) its makings took, the most recently made first, at most
//!   [`RECORDED_REGIONS`]. A key a dispatch made that the record does not hold joins it at the
//!   front, and the record is written again; the oldest past the bound leave it. A key a state's
//!   creation made from the record is not written again.
//! - **Named by content, in its build.** A record lies in the shader cache's build directory
//!   beside the driver's data, named by the build, the variant's content (its layout, stages and
//!   SPIR-V, `VariantObjects::making`), the driver's identity, the layer settings and what the
//!   device enabled, not by the profile (`ShaderCache::region_record_name`,
//!   `shader_cache/driver.rs`): a process of the build whose profile leaves a variant's SPIR-V as
//!   it was finds its record; a new build or a driver update starts without one. A record is
//!   written as a temporary file renamed over it, so a reader reads a whole record, the old or
//!   the new. Two processes writing one record at once may each drop the other's newest key,
//!   which costs that key a making at its first dispatch, as without a record.
//! - **Checked.** A record is read only where its frame, version and checksum hold and every key
//!   names constants of the variant's stage within its bytes; anything else is refused and
//!   removed. A key taken from it decides only which `VkPipeline`s are made, never which one a
//!   dispatch takes: a dispatch takes the key its own plan computes, made or not.
//! - **Read once per variant.** A variant reads its record at most once, at the first creation
//!   that asks or its first making that writes, and keeps it while it lives.
//!
//! ```text
//! magic        8 bytes   "M2VRGN1\0"
//! keys         u32
//! each key     stage flags u32, required subgroup size u32 (0: none), entries u32,
//!              each entry (constant id, offset, size: u32 each), data length u32, data
//! checksum     32 bytes  SHA-256 of every byte before it
//! ```
//!
//! Integers are little-endian (`record_frame.rs`).

use super::record_frame;
use super::{RegionKey, Variant};
use crate::device::lock;
use metal2vulkan::vk::PipelineShaderStageCreateFlags;
use std::sync::Mutex;

/// The most region keys a variant's record holds: a kernel dispatched at two threadgroup sizes
/// over grids that are not whole threadgroups takes up to eight (a size's whole region and its
/// three edge regions each).
pub(crate) const RECORDED_REGIONS: usize = 16;

/// The magic of a region record, its version the last byte before the NUL.
const MAGIC: [u8; 8] = *b"M2VRGN1\0";

/// A variant's record as its creations and makings see it: its keys, read once (the module
/// docs).
#[derive(Default)]
pub(super) struct RegionRecord(Mutex<Option<Vec<RegionKey>>>);

impl RegionRecord {
    /// Its keys, `load`ed first where nothing read them yet.
    fn keys(&self, load: impl FnOnce() -> Vec<RegionKey>) -> Vec<RegionKey> {
        lock(&self.0).get_or_insert_with(load).clone()
    }

    /// Where it does not hold `key` (its keys `load`ed first where nothing read them yet),
    /// `key` joins it at the front, the oldest past [`RECORDED_REGIONS`] leaving, and its bytes
    /// go to `write` while it is held: a second key noted meanwhile waits for the write, so of
    /// two makings' writes the last holds both keys.
    fn note(
        &self,
        key: &RegionKey,
        load: impl FnOnce() -> Vec<RegionKey>,
        write: impl FnOnce(&[u8]),
    ) {
        let mut held = lock(&self.0);
        let keys = held.get_or_insert_with(load);
        if keys.contains(key) {
            return;
        }
        keys.insert(0, key.clone());
        keys.truncate(RECORDED_REGIONS);
        write(&encode(keys));
    }
}

impl Variant {
    /// The name of its record (the module docs).
    fn record_name(&self) -> String {
        let shared = self.objects.device.shared();
        shared
            .shader_cache
            .region_record_name(self.objects.making("region keys"))
    }

    /// The keys its record on disk holds; none where it holds no record that reads.
    fn read_record(&self) -> Vec<RegionKey> {
        let constants = &self.pipeline.stage.specialization.constants;
        self.objects
            .device
            .shared()
            .shader_cache
            .load_record(&self.record_name(), |bytes| {
                decode(&bytes, |id| constants.iter().any(|c| c.id == id))
            })
            .unwrap_or_default()
    }

    /// The keys its record holds, for its state's creation to make (`dispatch_making.rs`): read
    /// now where no creation or making of the variant read them yet.
    pub(super) fn recorded_regions(&self) -> Vec<RegionKey> {
        self.record.keys(|| self.read_record())
    }

    /// `key`'s `VkPipeline` was made for a dispatch: where the record does not hold `key`, it
    /// joins it and the record is written ([`RegionRecord::note`]).
    pub(super) fn note_region(&self, key: &RegionKey) {
        self.record.note(
            key,
            || self.read_record(),
            |bytes| {
                self.objects
                    .device
                    .shared()
                    .shader_cache
                    .write_record(&self.record_name(), bytes)
            },
        );
    }
}

/// The keys of a record's `bytes` (the module docs), each entry naming a constant `known` holds
/// (the variant's stage's); `None` where anything does not hold.
fn decode(bytes: &[u8], known: impl Fn(u32) -> bool) -> Option<Vec<RegionKey>> {
    let mut read = record_frame::opened(&MAGIC, bytes)?;
    let count = read.count(RECORDED_REGIONS)?;
    let mut keys = Vec::with_capacity(count);
    for _ in 0..count {
        let stage_flags = PipelineShaderStageCreateFlags::from_raw(read.word()?);
        let required_subgroup_size = Some(read.word()?).filter(|&size| size != 0);
        keys.push(RegionKey {
            specialization: read.specialization(&known)?,
            stage_flags,
            required_subgroup_size,
        });
    }
    read.done().then_some(keys)
}

/// The record of `keys` (the module docs).
fn encode(keys: &[RegionKey]) -> Vec<u8> {
    record_frame::sealed(&MAGIC, |out| {
        out.word(keys.len() as u32);
        for key in keys {
            out.word(key.stage_flags.as_raw());
            out.word(key.required_subgroup_size.unwrap_or(0));
            out.specialization(&key.specialization);
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::cache::SpecializationKey;
    use metal2vulkan::vk::SpecializationMapEntry;

    fn key(constant_id: u32, value: u32, subgroup: Option<u32>) -> RegionKey {
        let mut specialization = SpecializationKey::new();
        specialization.entries.push(SpecializationMapEntry {
            constant_id,
            offset: 0,
            size: 4,
        });
        specialization.data = value.to_le_bytes().to_vec();
        RegionKey {
            specialization,
            stage_flags: PipelineShaderStageCreateFlags::from_raw(2),
            required_subgroup_size: subgroup,
        }
    }

    /// A record reads back the keys written, in order; a flipped byte, a cut record, a key naming
    /// a constant the variant's stage lacks or an entry past its bytes refuses the whole record.
    #[test]
    fn a_record_reads_back_its_keys_and_nothing_else() {
        let keys = vec![key(0, 8, None), key(1, 64, Some(32))];
        let bytes = encode(&keys);
        let known = |id: u32| id < 2;
        assert_eq!(decode(&bytes, known), Some(keys));
        assert_eq!(decode(&encode(&[]), known), Some(Vec::new()));
        let mut flipped = bytes.clone();
        flipped[12] ^= 1;
        assert_eq!(decode(&flipped, known), None);
        assert_eq!(decode(&bytes[..bytes.len() - 1], known), None);
        assert_eq!(decode(&bytes[..4], known), None);
        assert_eq!(decode(&bytes, |id| id == 0), None);
        let mut past = key(0, 8, None);
        past.specialization.entries[0].offset = 2;
        assert_eq!(decode(&encode(&[past]), known), None);
        let many: Vec<RegionKey> = (0..=RECORDED_REGIONS as u32)
            .map(|v| key(0, v, None))
            .collect();
        assert_eq!(decode(&encode(&many), known), None);
    }

    /// A record is held while its bytes are written, so a key noted while another's write is
    /// under way cannot take a snapshot that reaches the disk before the other's: the record
    /// written last holds every key noted, the newest first. A key it holds writes nothing.
    #[test]
    fn the_record_written_last_holds_every_noted_key() {
        let record = RegionRecord::default();
        let written = Mutex::new(Vec::new());
        let write = |bytes: &[u8]| {
            assert!(
                record.0.try_lock().is_err(),
                "the record is held while it is written"
            );
            written.lock().unwrap().push(bytes.to_vec());
        };
        let (older, newer) = (key(0, 8, None), key(0, 64, None));
        record.note(&older, Vec::new, write);
        record.note(&newer, || unreachable!("read once"), write);
        record.note(&older, || unreachable!("read once"), write);
        let written = written.into_inner().unwrap();
        assert_eq!(written.len(), 2, "a held key writes nothing");
        assert_eq!(
            decode(written.last().unwrap(), |id| id == 0),
            Some(vec![newer, older])
        );
    }
}
