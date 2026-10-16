//! No-copy buffers over the same host memory: hazard tracking keys each buffer's uses on the buffer
//! (`tracker.rs`), so two no-copy buffers over one range of the caller's memory were two resources
//! to it, and nothing ordered a write through one before a read through the other: the consumer
//! stands each texture of an IOSurface in with a texture of its plane over a new no-copy buffer of
//! the surface's pages (`Buffer::new_texture_with_iosurface_plane`), so two textures of one surface
//! lay over two buffers. Metal orders two textures over one IOSurface as it orders one resource's
//! accesses: in one encoder, in one command buffer and across one queue's command buffers (EV
//! `IOS-alias3-m4`, an M4, macOS 27.0.1, plain and under API validation; across queues only as
//! the program orders them).
//!
//! - **One memory, one key.** A no-copy buffer made over exactly the bytes a live one covers
//!   (the same first byte and length; not a heap's) joins it: the device notes the first such
//!   buffer's key as the memory's ([`Alias`]), and each command's uses of a buffer that joined
//!   carry a use of the memory's key too, of the same bytes and access ([`HostAliases::augment`],
//!   `EncoderCore`). The tracker orders those as one resource's, inside an encoder, between
//!   encoders and command buffers, and across queues as it orders any. The first buffer's own
//!   uses name the memory's key already, and its work from before the second buffer was made is
//!   ordered against the second's.
//! - **One memory, one import.** A buffer that joins a memory, of the memory class of a live
//!   import of it, lies in that import's `VkBuffer` (`HostImport`, `Device::buffer_in_import`):
//!   one resource to the driver, where the tracker's barriers are not all the ordering there is.
//!   MoltenVK records no barrier between blits and leaves their order to Metal's hazard
//!   tracking, which is by `MTLBuffer`: two imports were two, and a blit's write through one
//!   raced a read through the other in one encoder and in one command buffer (EV
//!   `IOS-alane2-mvk4`, an M4: mixed texels); one import is one `MTLBuffer`, which Metal orders.
//!   Each buffer stays its own Metal object, with its own Metal addresses, object ID, key, pages
//!   hold and release: a command buffer lists and holds, and marks reached unnamed
//!   (`live_buffers.rs`), the buffers its commands use, not their memory, as Metal marks them
//!   (EV `IOS-pairreach1-m4`); its deallocator runs once nothing holds its pages, as an import
//!   of its own would let it, while the import lives on in the buffers that still lie in it, and
//!   its keep-alive goes then too, since those buffers keep the same bytes mapped (`buffer.rs`'s
//!   `HostPages`). Copies between two such buffers are copies within one `VkBuffer`, moved as
//!   within one buffer (`blit.rs`, `blit_texture.rs`).
//! - **Ownership stays the buffer's.** The added uses are hazard tracking's alone: what a command
//!   holds and lists (`EncoderCore::touch`, rebind holds, `used_resources`) is named by the
//!   command's own uses, each buffer by its own key.
//! - **Shadows.** A shadowed texture over a buffer that joined compares its truth both ways
//!   (`ShadowCopies::meets`), so a write through the other buffer, or a texture over it, makes
//!   the shadow copy in again, as a write through its own buffer does.
//! - **Not joined.** Buffers over memory that overlaps without being the same range, a heap's
//!   bytes, and the memory a rebound buffer takes, keep their own keys: Metal orders no two
//!   no-copy buffers over one allocation (the same EV's `host` cases read stale or mixed
//!   texels), so the runtime's joining of the same range is the IOSurface's ordering, which a
//!   race over plain no-copy buffers may also give. A rebound buffer leaves the memory it was
//!   made over, keeping the key it joined (`Device::rebind_host_pages`); its pages stay in the
//!   import while commands recorded before the rebind hold them. Two buffers made at once over
//!   the same memory may each import it (each joins after its import): they share the key, not
//!   the `VkBuffer`.
//! - **Ordered by the bytes they lie over now** (the consumer's boot whose keys 113 and 115, over
//!   the same 1080p surface's pages, were never ordered: one of them was rebound there, or made
//!   after a remap of them, so neither joined the other). Joining keeps to the memory a buffer was
//!   made over, for its import; hazard tracking instead orders the uses of every buffer over host
//!   pages, made or rebound there, by the bytes it lies over now ([`Placements`]): the keys over
//!   one range of bytes share a key (the first placed there while any is, as a memory's), and each
//!   use of a key's bytes carries a use of its range's key and of the key of each other range those
//!   bytes overlap, clipped to them ([`HostAliases::augment`]). So two keys over overlapping bytes
//!   are ordered where they overlap, whatever made them, at a cost of one lookup a use while no two
//!   ranges meet, and of one use added per range met otherwise. A remap reported keeps a placement:
//!   it orders more than the pages need, never less. A heap's own buffer is placed as a buffer is.
//! - **Same address, same pages.** Sharing an import assumes the pages at the bytes' addresses
//!   are the ones the import holds: Metal's `newBufferWithBytesNoCopy:` wires the pages at the
//!   address when it makes each buffer, so a buffer made after the caller replaced the pages
//!   under an address (an unmap and a map there) is over the new pages. The caller tells the
//!   device so (`Device::host_range_remapped`): every live buffer made over memory with a byte
//!   in the range then leaves it, as a rebound buffer does ([`HostAliases::remapped`]), so a
//!   later buffer imports the pages there anew and joins no earlier key (the consumer's
//!   `[blocks]` setup keeps one host address for a guest address across the guest's unmap and
//!   remap). Without the call, a buffer over the same bytes as a live one lies in its import, and
//!   so over the pages that were there when that one was made.
//! - **Cheap, from anywhere.** The call sits on the caller's unmap path, at each release of a
//!   buffer's pages and over whole reservations of several GiB: with no memory noted it returns
//!   without the lock, and otherwise it finds the memories with a byte in the range by address
//!   ([`Memories`]), never reading them all, and ends them without reading their buffers' notes
//!   ([`Over`]). The lock is a leaf: nothing else is locked, and no buffer, pages hold or
//!   deallocator is dropped, while it is held (the imports noted are weak), so the call may be
//!   made from any thread, a deallocator's among them.

use super::buffer::HostImport;
use super::device::lock;
use super::key_hash::KeyMap;
use super::memory::MemoryClass;
use super::resource::ResourceKey;
use super::tracker::{Use, UseRange};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The key a buffer that joined another's memory adds to its uses: the memory's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Alias {
    pub(super) key: ResourceKey,
}

/// The device's no-copy buffers by the host memory they were made over (the module docs), and
/// the imports of it they lie in (`I`: a `HostImport`, but in unit tests).
pub(super) struct HostAliases<I = HostImport> {
    /// Whether any buffer joined another's memory: else no use takes another.
    any: AtomicBool,
    /// Whether a range of placed bytes has a second key or meets another range
    /// ([`Placements::crowded`]): else no use takes another for its bytes.
    crowded: AtomicBool,
    /// How many memories are noted: with none, a reported remap has none to end.
    noted: AtomicUsize,
    inner: Mutex<Inner<I>>,
}

impl<I> Default for HostAliases<I> {
    fn default() -> Self {
        HostAliases {
            any: AtomicBool::new(false),
            crowded: AtomicBool::new(false),
            noted: AtomicUsize::new(0),
            inner: Mutex::new(Inner {
                made_over: KeyMap::default(),
                memories: Memories::default(),
                aliases: Arc::default(),
                serial: 0,
                placements: Placements::default(),
            }),
        }
    }
}

struct Inner<I> {
    /// The memory each live buffer was made over, by its key, until it leaves it.
    made_over: KeyMap<ResourceKey, Over>,
    /// Each memory with a live buffer made over it, by its first byte's host address and its
    /// length.
    memories: Memories<Memory<I>>,
    /// The buffers that joined a memory, by their key: a snapshot each command reads.
    aliases: Arc<KeyMap<ResourceKey, Alias>>,
    /// The serial of the memory noted last.
    serial: u64,
    /// The bytes each buffer over host pages lies over now.
    placements: Placements,
}

/// The memory a live buffer was made over: its first byte's host address and its length, and
/// the serial it was noted with. A remap ends a memory without reading its buffers' notes; a
/// memory noted again over the same bytes takes a new serial, which those notes do not name.
#[derive(Clone, Copy)]
struct Over {
    memory: (u64, u64),
    serial: u64,
}

/// A memory live buffers were made over.
struct Memory<I> {
    /// The first such buffer's key: the memory's.
    key: ResourceKey,
    /// How many live buffers were made over it.
    buffers: usize,
    /// Its imports a buffer made, by memory class: a later buffer of the class lies in it while
    /// it lives.
    imports: Vec<(MemoryClass, Weak<I>)>,
    /// The serial it was noted with ([`Over`]).
    serial: u64,
}

impl<I> HostAliases<I> {
    /// The live import of `class` of `memory`, the bytes a live buffer was made over (its first
    /// byte's host address and its length): a buffer made over them lies in it.
    pub(super) fn import(&self, memory: (u64, u64), class: MemoryClass) -> Option<Arc<I>> {
        let inner = lock(&self.inner);
        let memory = inner.memories.get(memory)?;
        memory
            .imports
            .iter()
            .filter(|(made, _)| *made == class)
            .find_map(|(_, import)| import.upgrade())
    }

    /// Note the buffer of `key`, made over `memory` (its first byte's host address and its
    /// length) and lying in `import`, of memory class `class`: it joins the memory a live buffer
    /// was made over, or its key becomes the memory's; later buffers of `class` over the memory
    /// lie in `import` unless a live import of it is noted already.
    pub(super) fn join(
        &self,
        key: ResourceKey,
        memory: (u64, u64),
        class: MemoryClass,
        import: &Arc<I>,
    ) -> Option<Alias> {
        let mut inner = lock(&self.inner);
        let Inner {
            made_over,
            memories,
            aliases,
            serial,
            placements: _,
        } = &mut *inner;
        let noted = memories.get_or_insert_with(memory, || {
            *serial += 1;
            Memory {
                key,
                buffers: 0,
                imports: Vec::new(),
                serial: *serial,
            }
        });
        noted.buffers += 1;
        noted
            .imports
            .retain(|(_, import)| import.strong_count() > 0);
        if !noted.imports.iter().any(|(made, _)| *made == class) {
            noted.imports.push((class, Arc::downgrade(import)));
        }
        let owner = noted.key;
        let over = Over {
            memory,
            serial: noted.serial,
        };
        made_over.insert(key, over);
        self.noted.store(memories.len(), Ordering::Relaxed);
        if owner == key {
            return None;
        }
        let alias = Alias { key: owner };
        Arc::make_mut(aliases).insert(key, alias);
        self.any.store(true, Ordering::Relaxed);
        Some(alias)
    }

    /// The memory the buffer of `key` joined, if it joined one.
    pub(super) fn of(&self, key: ResourceKey) -> Option<Alias> {
        if !self.any.load(Ordering::Relaxed) {
            return None;
        }
        lock(&self.inner).aliases.get(&key).copied()
    }

    /// The buffer of `key` no longer lies in the memory it was made over (a rebind): later
    /// buffers over that memory do not join it. It keeps the key it joined.
    pub(super) fn leave(&self, key: ResourceKey) {
        let mut inner = lock(&self.inner);
        if let Some(over) = inner.made_over.remove(&key) {
            release(&mut inner.memories, over);
            self.noted.store(inner.memories.len(), Ordering::Relaxed);
        }
    }

    /// The caller replaced the pages under the host addresses `start..end`
    /// (`Device::host_range_remapped`): every live buffer made over memory with a byte among
    /// them leaves it, as a rebound buffer does ([`HostAliases::leave`]), keeping its pages, its
    /// import and the key it joined, so that a later buffer over any of those addresses neither
    /// lies in an import made before nor joins their key. With no memory noted it returns at
    /// once; else it ends the memories the lookup by address finds
    /// ([`Memories::overlapping`]).
    pub(super) fn remapped(&self, start: u64, end: u64) {
        if start >= end || self.noted.load(Ordering::Relaxed) == 0 {
            return;
        }
        let mut inner = lock(&self.inner);
        let ended: Vec<(u64, u64)> = inner.memories.overlapping(start, end).collect();
        for memory in ended {
            inner.memories.remove(memory);
        }
        self.noted.store(inner.memories.len(), Ordering::Relaxed);
    }

    /// The buffer of `key` is gone.
    pub(super) fn forget(&self, key: ResourceKey) {
        let mut inner = lock(&self.inner);
        if let Some(over) = inner.made_over.remove(&key) {
            release(&mut inner.memories, over);
            self.noted.store(inner.memories.len(), Ordering::Relaxed);
        }
        if inner.aliases.contains_key(&key) {
            Arc::make_mut(&mut inner.aliases).remove(&key);
            self.any.store(!inner.aliases.is_empty(), Ordering::Relaxed);
        }
        inner.placements.unplace(key);
        self.crowded
            .store(inner.placements.crowded(), Ordering::Relaxed);
    }

    /// The buffer of `key` (a no-copy buffer, or a no-copy heap's own buffer, whose uses are of
    /// bytes counted from its first) lies over the host bytes `at` (its first byte's address and
    /// its length) from now on, made or rebound there, and no longer where it lay before
    /// ([`Placements`]).
    pub(super) fn place(&self, key: ResourceKey, at: (u64, u64)) {
        let mut inner = lock(&self.inner);
        inner.placements.unplace(key);
        inner.placements.place(key, at);
        self.crowded
            .store(inner.placements.crowded(), Ordering::Relaxed);
    }

    /// The buffer of `key` lies over no host bytes from now on (its bytes detached).
    pub(super) fn unplace(&self, key: ResourceKey) {
        let mut inner = lock(&self.inner);
        inner.placements.unplace(key);
        self.crowded
            .store(inner.placements.crowded(), Ordering::Relaxed);
    }

    /// `uses` and, for each use of a buffer that joined a memory, a use of the memory's key of
    /// the same bytes and access, and for each use of bytes of a placed key, the uses of the
    /// keys its bytes share ([`Placements::sharing`]); `None` where no use takes another (`uses`
    /// as they are).
    pub(super) fn augment(&self, uses: &[Use<ResourceKey>]) -> Option<Vec<Use<ResourceKey>>> {
        let joined = self.any.load(Ordering::Relaxed);
        let crowded = self.crowded.load(Ordering::Relaxed);
        if !joined && !crowded {
            return None;
        }
        let inner = lock(&self.inner);
        let placements = crowded.then_some(&inner.placements);
        augmented(&inner.aliases, placements, uses)
    }
}

/// One live buffer less over the memory `over` names: the memory is forgotten with its last,
/// unless a remap ended it already.
fn release<I>(memories: &mut Memories<Memory<I>>, over: Over) {
    let Some(noted) = memories.get_mut(over.memory) else {
        return;
    };
    if noted.serial != over.serial {
        return;
    }
    noted.buffers -= 1;
    if noted.buffers == 0 {
        memories.remove(over.memory);
    }
}

/// [`HostAliases::augment`] over `aliases` and, where given, `placements`.
pub(super) fn augmented(
    aliases: &KeyMap<ResourceKey, Alias>,
    placements: Option<&Placements>,
    uses: &[Use<ResourceKey>],
) -> Option<Vec<Use<ResourceKey>>> {
    let mut added: Vec<Use<ResourceKey>> = Vec::new();
    for used in uses {
        let UseRange::Bytes(bytes) = &used.range else {
            continue;
        };
        let joined = aliases.get(&used.key).map(|alias| alias.key);
        if let Some(key) = joined {
            added.push(Use {
                key,
                ..used.clone()
            });
        }
        let Some(placements) = placements else {
            continue;
        };
        placements.sharing(used.key, bytes, |key, range| {
            // The memory's key carries the same bytes already.
            if Some(key) != joined || range != *bytes {
                added.push(Use {
                    key,
                    range: UseRange::Bytes(range),
                    ..used.clone()
                });
            }
        });
    }
    if added.is_empty() {
        return None;
    }
    Some(uses.iter().cloned().chain(added).collect())
}

/// The bytes each buffer over host pages lies over now, by key, and the ranges of bytes some key
/// lies over, each with the key its keys' uses share: the first placed there, while any key lies
/// there (the module docs).
#[derive(Default)]
pub(super) struct Placements {
    /// Each placed key's bytes: their first byte's host address and their length.
    at: KeyMap<ResourceKey, (u64, u64)>,
    /// Each range of bytes some key lies over, by its first byte's address and its length.
    ranges: Memories<Placed>,
    /// How many ranges are crowded ([`Placed::crowded`]).
    crowded: usize,
}

/// A range of host bytes keys lie over.
struct Placed {
    /// The key its keys' uses share: the first placed there while any key lies there.
    key: ResourceKey,
    /// How many keys lie over it.
    keys: usize,
    /// Whether its key lies over it still.
    first: bool,
    /// How many other ranges meet it (share a byte with it).
    met: usize,
}

impl Placed {
    /// Whether a use of a key over it takes another: a second key lies there, its key went, or
    /// another range meets it.
    fn crowded(&self) -> bool {
        self.keys > 1 || !self.first || self.met > 0
    }
}

impl Placements {
    /// Whether any range is crowded: else no use takes another.
    pub(super) fn crowded(&self) -> bool {
        self.crowded > 0
    }

    /// Place `key`, lying over no bytes, over `at`.
    pub(super) fn place(&mut self, key: ResourceKey, at: (u64, u64)) {
        if at.1 == 0 {
            return;
        }
        self.at.insert(key, at);
        if let Some(range) = self.ranges.get_mut(at) {
            let was = range.crowded();
            range.keys += 1;
            self.crowded += usize::from(!was && range.crowded());
            return;
        }
        let end = at.0.saturating_add(at.1);
        let met: Vec<(u64, u64)> = self.ranges.overlapping(at.0, end).collect();
        for &other in &met {
            if let Some(range) = self.ranges.get_mut(other) {
                let was = range.crowded();
                range.met += 1;
                self.crowded += usize::from(!was && range.crowded());
            }
        }
        let range = Placed {
            key,
            keys: 1,
            first: true,
            met: met.len(),
        };
        self.crowded += usize::from(range.crowded());
        self.ranges.get_or_insert_with(at, || range);
    }

    /// `key` lies over no bytes from now on.
    pub(super) fn unplace(&mut self, key: ResourceKey) {
        let Some(at) = self.at.remove(&key) else {
            return;
        };
        let Some(range) = self.ranges.get_mut(at) else {
            return;
        };
        let was = range.crowded();
        range.keys -= 1;
        if range.key == key {
            range.first = false;
        }
        if range.keys > 0 {
            let now = range.crowded();
            self.crowded = self.crowded + usize::from(now) - usize::from(was);
            return;
        }
        self.crowded -= usize::from(was);
        self.ranges.remove(at);
        let end = at.0.saturating_add(at.1);
        let met: Vec<(u64, u64)> = self.ranges.overlapping(at.0, end).collect();
        for other in met {
            if let Some(range) = self.ranges.get_mut(other) {
                let was = range.crowded();
                range.met -= 1;
                self.crowded -= usize::from(was && !range.crowded());
            }
        }
    }

    /// For a use of `bytes` of `key` (counted from its first byte), each other key those bytes
    /// share, and the bytes of it they are (counted from its range's first byte), to `each`: its
    /// own range's key where another, and each other range's the bytes meet, clipped to it.
    pub(super) fn sharing(
        &self,
        key: ResourceKey,
        bytes: &std::ops::Range<u64>,
        mut each: impl FnMut(ResourceKey, std::ops::Range<u64>),
    ) {
        let Some(&at) = self.at.get(&key) else {
            return;
        };
        let Some(own) = self.ranges.get(at) else {
            return;
        };
        if !own.crowded() {
            return;
        }
        if own.key != key {
            each(own.key, bytes.clone());
        }
        if own.met == 0 {
            return;
        }
        // The host bytes the use covers, within the key's own.
        let start = at.0.saturating_add(bytes.start.min(at.1));
        let end = at.0.saturating_add(bytes.end.min(at.1));
        for other in self.ranges.overlapping(start, end) {
            if other == at {
                continue;
            }
            let Some(range) = self.ranges.get(other) else {
                continue;
            };
            let from = start.max(other.0);
            let to = end.min(other.0.saturating_add(other.1));
            if from < to {
                each(range.key, from - other.0..to - other.0);
            }
        }
    }
}

/// Values by the host bytes they are over (their first byte's address and their length), found
/// by address: the memories a remap ends. Each class of length (`class`) is ordered by first
/// byte, so the lookup of those with a byte in a range reads, in each class, the values whose
/// first byte lies from the class's longest length before the range to its end
/// ([`Memories::candidates`]): the ones it finds, and of the class only those over the one
/// address half its longest length before the range, which end before it.
pub(super) struct Memories<T> {
    /// The values by class of length, each by first byte's address and length.
    classes: Vec<BTreeMap<(u64, u64), T>>,
    /// How many values there are.
    len: usize,
}

/// The classes of length: one for each power of two from 2 a `u64` length may reach, and one
/// for the lengths of 0 and 1.
const CLASSES: usize = u64::BITS as usize + 1;

/// The class of `length`: the least `b` with `length <= 2^b` (so its lengths are past the class
/// below's longest).
fn class(length: u64) -> usize {
    (u64::BITS - length.saturating_sub(1).leading_zeros()) as usize
}

/// The longest length of the class `class`.
fn longest(class: usize) -> u64 {
    u32::try_from(class)
        .ok()
        .and_then(|class| 1u64.checked_shl(class))
        .unwrap_or(u64::MAX)
}

impl<T> Default for Memories<T> {
    fn default() -> Self {
        Memories {
            classes: (0..CLASSES).map(|_| BTreeMap::new()).collect(),
            len: 0,
        }
    }
}

impl<T> Memories<T> {
    /// How many values there are.
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// The value over `memory` (its first byte's address and its length).
    pub(super) fn get(&self, memory: (u64, u64)) -> Option<&T> {
        self.classes[class(memory.1)].get(&memory)
    }

    /// The value over `memory`, to change.
    pub(super) fn get_mut(&mut self, memory: (u64, u64)) -> Option<&mut T> {
        self.classes[class(memory.1)].get_mut(&memory)
    }

    /// The value over `memory`, made by `make` where there is none.
    pub(super) fn get_or_insert_with(
        &mut self,
        memory: (u64, u64),
        make: impl FnOnce() -> T,
    ) -> &mut T {
        let by = &mut self.classes[class(memory.1)];
        if !by.contains_key(&memory) {
            self.len += 1;
        }
        by.entry(memory).or_insert_with(make)
    }

    /// Remove the value over `memory`.
    pub(super) fn remove(&mut self, memory: (u64, u64)) -> Option<T> {
        let removed = self.classes[class(memory.1)].remove(&memory);
        self.len -= usize::from(removed.is_some());
        removed
    }

    /// The memories the lookup of those with a byte in `start..end` reads (the type's docs): of
    /// each class, those whose first byte lies from its longest length before `start` to `end`.
    pub(super) fn candidates(&self, start: u64, end: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.classes
            .iter()
            .enumerate()
            .filter(|(_, by)| !by.is_empty())
            .flat_map(move |(class, by)| {
                let from = start.saturating_sub(longest(class) - 1);
                by.range((from, 0)..(end.max(from), 0))
                    .map(|(&memory, _)| memory)
            })
    }

    /// The memories with a byte in `start..end`.
    pub(super) fn overlapping(
        &self,
        start: u64,
        end: u64,
    ) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.candidates(start, end)
            .filter(move |&(at, length)| at < end && start < at.saturating_add(length))
    }
}
