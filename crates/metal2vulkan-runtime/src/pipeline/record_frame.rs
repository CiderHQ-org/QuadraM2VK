//! The frame of a record of a variant's keys in the shader cache (`region_record.rs`): a magic
//! naming the record's kind and version, the body, and the SHA-256 of both. Integers are
//! little-endian `u32`s.

use metal2vulkan::cache::{Hash256, SpecializationKey};
use metal2vulkan::vk::SpecializationMapEntry;
use sha2::{Digest, Sha256};

/// The bytes of a record's checksum.
const CHECKSUM: usize = 32;
/// The most entries a key's specialization holds that a record takes, and the most bytes.
const MAX_ENTRIES: usize = 1024;
const MAX_DATA: usize = 8 * MAX_ENTRIES;

/// The record `body` writes, framed with `magic`.
pub(super) fn sealed(magic: &[u8; 8], body: impl FnOnce(&mut Writer)) -> Vec<u8> {
    let mut out = Writer(magic.to_vec());
    body(&mut out);
    let mut bytes = out.0;
    let checksum: Hash256 = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// The body of `bytes`, a record framed with `magic`; `None` where the frame does not hold.
pub(super) fn opened<'a>(magic: &[u8; 8], bytes: &'a [u8]) -> Option<Reader<'a>> {
    let body = bytes.len().checked_sub(CHECKSUM)?;
    let checksum: Hash256 = Sha256::digest(&bytes[..body]).into();
    if bytes[body..] != checksum || bytes.get(..magic.len())? != magic {
        return None;
    }
    Some(Reader(bytes.get(magic.len()..body)?))
}

/// A record's body as it is written.
pub(super) struct Writer(Vec<u8>);

impl Writer {
    pub(super) fn word(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    /// A count, then the bytes.
    pub(super) fn bytes(&mut self, bytes: &[u8]) {
        self.word(bytes.len() as u32);
        self.0.extend_from_slice(bytes);
    }

    /// Its entries (a count, then each one's constant id, offset and size), then its data.
    pub(super) fn specialization(&mut self, specialization: &SpecializationKey) {
        self.word(specialization.entries.len() as u32);
        for entry in &specialization.entries {
            self.word(entry.constant_id);
            self.word(entry.offset);
            self.word(entry.size);
        }
        self.bytes(&specialization.data);
    }
}

/// What is left of a record's body to read.
pub(super) struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    /// The next `length` bytes.
    fn take(&mut self, length: usize) -> Option<&'a [u8]> {
        if self.0.len() < length {
            return None;
        }
        let (head, rest) = self.0.split_at(length);
        self.0 = rest;
        Some(head)
    }

    /// The next word.
    pub(super) fn word(&mut self) -> Option<u32> {
        let bytes = self.take(4)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }

    /// The next word as a count, `None` past `most`.
    pub(super) fn count(&mut self, most: usize) -> Option<usize> {
        usize::try_from(self.word()?)
            .ok()
            .filter(|&count| count <= most)
    }

    /// Bytes as [`Writer::bytes`] wrote them, at most `most`.
    pub(super) fn bytes(&mut self, most: usize) -> Option<&'a [u8]> {
        let length = self.count(most)?;
        self.take(length)
    }

    /// A specialization as [`Writer::specialization`] wrote it, each entry a scalar within its
    /// data naming a constant `known` holds.
    pub(super) fn specialization(
        &mut self,
        known: impl Fn(u32) -> bool,
    ) -> Option<SpecializationKey> {
        let mut specialization = SpecializationKey::new();
        for _ in 0..self.count(MAX_ENTRIES)? {
            specialization.entries.push(SpecializationMapEntry {
                constant_id: self.word()?,
                offset: self.word()?,
                size: self.word()?,
            });
        }
        specialization.data = self.bytes(MAX_DATA)?.to_vec();
        let fits = specialization.entries.iter().all(|entry| {
            matches!(entry.size, 1 | 2 | 4 | 8)
                && (entry.offset as usize)
                    .checked_add(entry.size as usize)
                    .is_some_and(|end| end <= specialization.data.len())
                && known(entry.constant_id)
        });
        fits.then_some(specialization)
    }

    /// Whether the whole body was read.
    pub(super) fn done(&self) -> bool {
        self.0.is_empty()
    }
}
