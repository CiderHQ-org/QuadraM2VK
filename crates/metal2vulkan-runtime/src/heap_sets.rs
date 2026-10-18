//! Set 1, the heap, and the texture table, device-wide (residency part C): every live texture is at
//! its resource ID for every command, as Metal's textures are (the M4 probe: a texture
//! read through its ID in an argument buffer is reached with no encoder naming it, and through a
//! later encoder's `use_resource`).
//!
//! - **The texture table.** One buffer per device of [`TABLE_ENTRIES`] handle-table entries
//!   (`bind::handle_entry`), entry `i` for resource ID `i`: `{i, i, flags}` while a live texture
//!   holds the ID's slot (`resource_ids.rs`), its flags the texture's (`bind::handle_entry_flags`),
//!   else `{i, 0, 0}`, the neutral element. So the table is sorted, starts with the null entry, and
//!   maps every ID a heap array can reach, the handle search's reach (`HANDLE_TABLE_CAPACITY`,
//!   the most any pipeline's heap arrays have; 1 MiB): a texture Metal numbers past its
//!   placement limit after a reservation (`resource_ids.rs`) is reached too. Each command binds
//!   it at every texture handle table of its layout (`Synthetic(HandleTable)`) over the entries
//!   its heap arrays' count reaches ([`texture_table`]): the shader's search reads its length
//!   from the range, and an ID past the arrays misses, announced
//!   ([`Degradation::UnnamedTexturesPastCapacity`]).
//! - **Heap sets.** One per set-1 layout ([`HeapSet`], which each variant of that layout holds),
//!   shared by every command buffer. Element 0 of each array is its class's neutral (an image's,
//!   the neutral sampler, the null function table: zeros in the table's buffer). Element `i` of
//!   each image array holds the texture at ID `i` as the array's class takes it
//!   (`bind::heap_element`: its view where it fits the class, else the neutral), written as the
//!   texture takes the slot ([`place`]), for each set that lives then, and by a set made later
//!   for each texture that lives then. A batch holds those writes from its first work's take
//!   until its submission was made ([`hold`]): one asked meanwhile is made then, before the next
//!   submission takes its first work, so no texture registered after the take, whose move is
//!   not the batch's, is in its sets or table in `UNDEFINED` (`texture/first_move.rs`), and its
//!   making waits for no submission. A rebind or a detach of host pages writes again the
//!   elements of the textures whose images or texel views lie in those bytes ([`renew_over`]),
//!   and of no other texture (`WorkCounts::heap_slots_rewritten`). A texture the class fits that it cannot view (a format
//!   the device cannot sample or store formatless, or that needs its function's decode) is the
//!   neutral, announced ([`Degradation::UnnamedTextureUnviewable`]); one whose usage the class
//!   lacks is the neutral, as Metal's contract leaves the read undefined. A shadowed texture's
//!   element is its shadow image's view, which holds the texels an encoder that named it last
//!   copied in ([`Degradation::UnnamedShadowedTexture`], announced at the first command that
//!   binds a set holding it while neither naming it nor making it resident, [`bind`]: one that
//!   does has its shadow copied in). An element whose ID no texture holds
//!   is never read: the table maps the ID to element 0. The argument samplers' and function
//!   tables' elements are the commands' writes ([`bind`]), each written where the set held
//!   another there.
//! - **Who reads an element while it changes.** The arrays are `UPDATE_AFTER_BIND` and, where
//!   the device has `descriptorBindingUpdateUnusedWhilePending`, `UPDATE_UNUSED_WHILE_PENDING`
//!   (`pipeline::link`): every command binds the set's one version ([`SetVersion`]), and a
//!   texture made while command buffers run writes its elements while they are pending, before
//!   the table names them, so a command reads them only once they hold it. Where the device lacks
//!   the feature, nothing writes a set a committed command buffer binds: each command buffer binds
//!   a version of its own, made as it first binds the set, every element a texture's ID does not
//!   name the neutral and every other the set's ([`SetEntry::written`]), announced
//!   ([`Degradation::HeapSetCopied`]). Writes go to every version whose command buffer has not
//!   committed, and a committed one keeps its elements (a texture made after is the neutral
//!   there). A gone texture's elements turn neutral in every version no submission of its command
//!   buffer took yet, committed or not: its set is `UPDATE_AFTER_BIND`, and nothing pending binds
//!   it.
//! - **Retirement.** A texture going clears its table entry. A command buffer submitted before
//!   then may still read the element, so the texture's image (which holds the view) is kept
//!   until every such command buffer completed: each command buffer that binds a heap set, or
//!   whose host steps may, takes a [`Ticket`] as its first batch is submitted, and an image
//!   retired while tickets are out waits for those taken before it. (The ticket was taken at the
//!   commit.) One committed but not yet submitted (held at its gate) never reads the gone texture:
//!   the table its submission's commands read maps the ID to the neutral, and its own versions'
//!   elements turned neutral. So the image goes as Metal's does where no submitted command buffer
//!   may read it, with the buffer or heap it lies in, whose no-copy pages' deallocator then runs
//!   with their import freed (`buffer.rs`). What a rebind or a detach replaced ([`retire`]) waits
//!   instead for the command buffers that committed before it, each holding a second ticket from
//!   its commit: one committed before it keeps the elements that name it until it completes (a
//!   placement's or a rebind's writes go only to versions not committed).
//! - **Samplers.** A sampler state is at its element of the sampler arrays (`resource_ids.rs`),
//!   which its command's sampler table maps its ID to: the device's live states as the command was
//!   encoded. Until it completes, a command buffer whose commands list them is counted in the
//!   device's `sampler_tables` (`sampler.rs` announces a state made meanwhile).
//! - **Release.** A heap set goes with the last variant of its layout and the last command buffer
//!   that bound it; the table with the device.

use super::binding::{write_sets, Info, Write};
use super::degradation::Degradation;
use super::descriptor_arena::SetNeeds;
use super::device::{lock, Device};
use super::error::{internal, vulkan, Error};
use super::exit_guard;
use super::memory::{Memory, MemoryClass, Placement};
use super::pipeline::{set_needs, VariantObjects};
use super::release;
use super::resource::ResourceKey;
use super::sampler::SamplerState;
use super::texture::{Texture, TextureShared};
use super::tracker::{Access, Use};
use ash::vk;
use metal2vulkan::bind::{
    handle_entry, handle_entry_flags, heap_element, ArgumentSampler, BindState, BindingPlan,
    FunctionTableWords, HeapElement, HeapElements, NeutralResource, HANDLE_ENTRY_BYTES,
    HANDLE_TABLE_CAPACITY,
};
use metal2vulkan::error::BindError;
use metal2vulkan::pipeline::{
    BindingSource, DescriptorSetLayout, HandleClass, ImageHandleClass, PipelineLayout,
    SyntheticBinding,
};
use std::any::Any;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

/// The heap's set number.
pub(super) const HEAP_SET: u32 = 1;

/// The entries of a device's texture table: one per texture resource ID a heap array can reach,
/// the handle search's reach (the module docs).
pub(super) const TABLE_ENTRIES: u32 = HANDLE_TABLE_CAPACITY;

/// The bytes of the texture table: past them, the null function table's words.
const TABLE_BYTES: u64 = TABLE_ENTRIES as u64 * HANDLE_ENTRY_BYTES as u64;

/// What keeps a resource apart from it: a texture's image, a sampler, a table's words.
pub(super) type Keeper = Box<dyn Any + Send + Sync>;

/// A device's heap sets and texture table (the module docs), under its `heaps` lock: taken
/// before its resource IDs' lock, never under it.
#[derive(Default)]
pub(super) struct DeviceHeaps {
    /// By resource ID, the textures placed: each live slot's holder.
    textures: BTreeMap<u64, Placed>,
    /// Made at the first command that binds a texture handle table or a heap set.
    table: Option<Table>,
    /// The live heap sets, by key ([`HeapSet`]).
    sets: BTreeMap<u64, SetEntry>,
    /// The live heap sets by layout, so the variants of one layout share one.
    by_layout: HashMap<DescriptorSetLayout, (u64, Weak<HeapSet>)>,
    next_set: u64,
    /// Gone textures' images, which wait for the command buffers submitted before they went.
    retirement: Retirement<Keeper>,
    /// What a rebind or a detach replaced ([`retire`]), which waits for the command buffers
    /// committed before it.
    replaced: Retirement<Keeper>,
    /// The batches holding the texture slots' writes ([`hold`]): while one is, a write waits in
    /// `deferred`.
    holds: usize,
    /// The slot writes asked while a batch held them, in order, made as the last hold ends.
    deferred: VecDeque<Slots>,
}

/// A texture the table and the heap sets hold: its slot's generation and hold serial when it
/// was written, its table entry's flags, and the texture (not held: its ID stays the program's
/// to release; upgraded only where the caller drops it once the lock is free).
struct Placed {
    generation: u64,
    serial: u64,
    flags: u32,
    /// Whether a heap write stores the device's conversion of a NaN in it ([`NanResidue::of`]).
    residue: [bool; 2],
    /// The key of the buffer whose bytes its images and texel views lie in
    /// (`Texture::bytes_key`), whose rebind or detach writes its elements again ([`renew_over`]).
    bytes: Option<ResourceKey>,
    texture: Weak<TextureShared>,
}

/// The live textures the device's heap holds in which a `float` or `half` write through a heap
/// handle of a function that states `RuntimeFact::HeapHalfTexelWrites` stores the device's
/// conversion of a NaN, not Metal's (`bind::heap_write_nans_converted`): those of every such
/// function (a half-float texture its unsigned integer twin does not hold), and those of one
/// whose `half` writes reach packed small floats too. Counted as they are placed and cleared, so
/// a draw or dispatch reads them without the `heaps` lock (`Device::degrade_half_texel_writes`).
#[derive(Default)]
pub(super) struct NanResidue {
    /// By `packed`: `false` counts the textures every such write converts, `true` those a `half`
    /// write converts.
    counts: [AtomicUsize; 2],
}

impl NanResidue {
    /// Of a texture of `facts`: whether a write under `HeapHalfTexelWrites { packed }` converts
    /// its NaN, by `packed`.
    fn of(facts: &metal2vulkan::bind::TextureFacts) -> [bool; 2] {
        [false, true].map(|packed| metal2vulkan::bind::heap_write_nans_converted(facts, packed))
    }

    fn add(&self, residue: [bool; 2]) {
        for (count, held) in self.counts.iter().zip(residue) {
            if held {
                count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn remove(&self, residue: [bool; 2]) {
        for (count, held) in self.counts.iter().zip(residue) {
            if held {
                count.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Whether the heap holds a texture whose NaN a write under
    /// `HeapHalfTexelWrites { packed }` converts.
    pub(super) fn holds(&self, packed: bool) -> bool {
        self.counts[usize::from(packed)].load(Ordering::Relaxed) > 0
    }
}

/// Images retired while command buffers that may read them run (the module docs): each waits
/// for the tickets taken before it.
#[derive(Debug)]
pub(super) struct Retirement<T> {
    /// The epoch a ticket taken now has; each retirement while tickets are out ends one.
    epoch: u64,
    /// The tickets out, by epoch.
    out: BTreeMap<u64, usize>,
    /// In epoch order.
    retired: VecDeque<(u64, T)>,
}

impl<T> Default for Retirement<T> {
    fn default() -> Retirement<T> {
        Retirement {
            epoch: 0,
            out: BTreeMap::new(),
            retired: VecDeque::new(),
        }
    }
}

impl<T> Retirement<T> {
    /// Retire `kept`: returned, to drop now, where no ticket is out.
    pub(super) fn retire(&mut self, kept: T) -> Option<T> {
        if self.out.is_empty() {
            return Some(kept);
        }
        self.retired.push_back((self.epoch, kept));
        self.epoch += 1;
        None
    }

    /// Take a ticket: its epoch.
    pub(super) fn take(&mut self) -> u64 {
        *self.out.entry(self.epoch).or_default() += 1;
        self.epoch
    }

    /// Give back a ticket of `epoch`: what no ticket out waits for any more, to drop.
    pub(super) fn give(&mut self, epoch: u64) -> Vec<T> {
        if let Some(count) = self.out.get_mut(&epoch) {
            *count -= 1;
            if *count == 0 {
                self.out.remove(&epoch);
            }
        }
        let least = self.out.keys().next().copied().unwrap_or(u64::MAX);
        let mut freed = Vec::new();
        while self
            .retired
            .front()
            .is_some_and(|(retired, _)| *retired < least)
        {
            freed.extend(self.retired.pop_front().map(|(_, kept)| kept));
        }
        freed
    }
}

/// The texture table's buffer, with the null function table's words past its entries.
struct Table {
    buffer: vk::Buffer,
    placement: Placement,
}

impl Table {
    /// A new table of `device`, every entry the neutral's, and the null table's words zero.
    fn make(device: &Device) -> Result<Table, Error> {
        let shared = device.shared();
        let raw = shared.vk();
        let null = FunctionTableWords::byte_len(0).next_multiple_of(16);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(TABLE_BYTES + null)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
        );
        // SAFETY: the device is live and `info` valid.
        let buffer = unsafe { raw.create_buffer(&info, None) }.map_err(vulkan)?;
        // SAFETY: `buffer` is a live buffer of this device.
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let placement = match crate::memory::allocate(
            &shared.memory,
            raw,
            &requirements,
            MemoryClass::WriteCombined,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw, buffer) };
                return Err(error);
            }
        };
        let table = Table { buffer, placement };
        // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
        let bound = unsafe {
            raw.bind_buffer_memory(buffer, table.placement.memory, table.placement.offset)
        };
        if let Err(error) = bound.map_err(vulkan).and_then(|()| table.start()) {
            table.destroy(raw, &mut lock(&shared.memory));
            return Err(error);
        }
        for id in 0..u64::from(TABLE_ENTRIES) {
            table.put(id, 0, 0);
        }
        let start = table.start()?;
        // SAFETY: the null table's words lie inside the placement, which nothing reads yet.
        unsafe { std::ptr::write_bytes(start.add(TABLE_BYTES as usize), 0, null as usize) };
        Ok(table)
    }

    /// The host address of the table's first byte.
    fn start(&self) -> Result<*mut u8, Error> {
        self.placement
            .mapped
            .map(|mapped| mapped.as_ptr())
            .ok_or_else(|| internal("a texture table in memory the host does not see"))
    }

    /// Write entry `id`: element `element`, flags `flags`.
    fn put(&self, id: u64, element: u32, flags: u32) {
        let Ok(start) = self.start() else {
            return;
        };
        let entry = handle_entry(id, element, flags);
        let at = id as usize * entry.len();
        // SAFETY: `id` is below `TABLE_ENTRIES`, so the entry lies inside the mapped placement;
        // a command reading it meanwhile reads the old entry or the new one (the module docs).
        unsafe { std::ptr::copy_nonoverlapping(entry.as_ptr(), start.add(at), entry.len()) };
    }

    /// The descriptor of the table's first `entries` entries.
    fn entries(&self, entries: u32) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(0)
            .range(u64::from(entries) * u64::from(HANDLE_ENTRY_BYTES))
    }

    /// The descriptor of the null function table.
    fn null_words(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(TABLE_BYTES)
            .range(FunctionTableWords::byte_len(0))
    }

    /// Destroy the buffer, freeing its memory into `memory`. Nothing uses it.
    fn destroy(&self, raw: &ash::Device, memory: &mut Memory) {
        // SAFETY: the caller's.
        unsafe { release::object(raw, self.buffer) };
        memory.free(raw, &self.placement);
    }
}

impl DeviceHeaps {
    /// Destroy the texture table, freeing its memory into `memory`: the device is idle.
    pub(super) fn destroy(&self, raw: &ash::Device, memory: &mut Memory) {
        if let Some(table) = &self.table {
            table.destroy(raw, memory);
        }
    }

    /// The texture table of `device`, made at its first use.
    fn table(&mut self, device: &Device) -> Result<&Table, Error> {
        if self.table.is_none() {
            let table = Table::make(device)?;
            for (&id, placed) in &self.textures {
                table.put(id, id as u32, placed.flags);
            }
            self.table = Some(table);
        }
        self.table
            .as_ref()
            .ok_or_else(|| internal("a texture table not made"))
    }

    /// Write `slots` ([`sync`]): the slots that failed, each with its error, their entries the
    /// neutral's. Their holders go to `held`, to drop once the lock is free.
    fn write_slots(
        &mut self,
        device: &Device,
        slots: Slots,
        held: &mut Vec<Texture>,
        notes: &mut Notes,
    ) -> Vec<(u64, Error)> {
        let (ids, force): (Vec<u64>, bool) = match slots {
            Slots::Changed(id) => (vec![id], false),
            Slots::All => (self.textures.keys().copied().collect(), true),
            Slots::Over(key) => {
                let over = self.textures.iter();
                let over = over.filter(|(_, placed)| placed.bytes == Some(key));
                (over.map(|(&id, _)| id).collect(), true)
            }
        };
        if force {
            device.shared().counts.heap_slots_rewritten(ids.len());
        }
        let mut failed = Vec::new();
        for id in ids {
            if let Err(error) = self.sync(device, id, force, held, notes) {
                let cleared = self.clear(device, id);
                failed.push((id, cleared.err().unwrap_or(error)));
            }
        }
        failed
    }

    /// Write texture slot `id` into the table and every heap set where its holder changed since
    /// (whatever changed where `force`, a rebind's): its holder's elements, or the neutral's entry where
    /// it has none. The holder goes to `held`, to drop once the lock is free.
    fn sync(
        &mut self,
        device: &Device,
        id: u64,
        force: bool,
        held: &mut Vec<Texture>,
        notes: &mut Notes,
    ) -> Result<(), Error> {
        if id == 0 || id >= u64::from(TABLE_ENTRIES) {
            return Ok(());
        }
        let (generation, holder) = lock(&device.shared().resource_ids).texture(id);
        let unchanged = self
            .textures
            .get(&id)
            .is_some_and(|placed| placed.generation == generation);
        if unchanged && !force {
            return Ok(());
        }
        let Some((serial, weak, texture)) = holder.and_then(|(serial, weak)| {
            let texture = Texture::upgrade(&weak)?;
            Some((serial, weak, texture))
        }) else {
            return self.clear(device, id);
        };
        // The holder may be a texture's last handle: it goes once the lock is free.
        held.push(texture.clone());
        let facts = texture.facts();
        let flags = handle_entry_flags(&facts);
        let residue = NanResidue::of(&facts);
        let bytes = texture.bytes_key();
        // The neutral's entry until every set holds the texture.
        if let Some(table) = &self.table {
            table.put(id, 0, 0);
        }
        self.unplace(device, id);
        for entry in self.sets.values_mut() {
            let (writes, shadow) = texture_writes(device, &entry.images, id, &texture, notes)?;
            entry.write(device, writes)?;
            entry.shadow(id, &texture, shadow);
        }
        if let Some(table) = &self.table {
            table.put(id, id as u32, flags);
        }
        device.shared().heap_nan_residue.add(residue);
        self.textures.insert(
            id,
            Placed {
                generation,
                serial,
                flags,
                residue,
                bytes,
                texture: weak,
            },
        );
        notes.placed(&texture, !self.sets.is_empty());
        Ok(())
    }

    /// Forget the texture placed at ID `id`, and its count in the device's [`NanResidue`].
    fn unplace(&mut self, device: &Device, id: u64) {
        if let Some(placed) = self.textures.remove(&id) {
            device.shared().heap_nan_residue.remove(placed.residue);
        }
    }

    /// Clear ID `id`: the neutral's entry, no element of it kept for a version made later, and
    /// the neutral at it in each version a command buffer has yet to commit (the module docs).
    fn clear(&mut self, device: &Device, id: u64) -> Result<(), Error> {
        self.unplace(device, id);
        for entry in self.sets.values_mut() {
            entry.shadows.remove(&id);
        }
        if let Some(table) = &self.table {
            table.put(id, 0, 0);
        }
        let Ok(element) = u32::try_from(id) else {
            return Ok(());
        };
        for entry in self.sets.values_mut() {
            for array in &entry.images {
                entry.written.remove(&(array.binding, element));
            }
            let open = entry.unsubmitted();
            if open.is_empty() {
                continue;
            }
            let neutrals: Vec<Write> = entry
                .images
                .iter()
                .filter(|array| element < array.count)
                .map(|array| raw_write(array.binding, element, array.ty, image_info(array.neutral)))
                .collect();
            for version in &open {
                // SAFETY: no submission of the version's command buffer took it (its flag is set
                // under the `heaps` lock held here, before the submission), so nothing pending
                // binds it, and its set is `UPDATE_AFTER_BIND`; the neutral views live with the
                // device.
                unsafe { write_sets(device, &[version.set], &neutrals) }?;
            }
        }
        Ok(())
    }

    /// A new heap set of `layout`, holding every live texture.
    fn made_set(
        &mut self,
        device: &Device,
        layout: &DescriptorSetLayout,
        held: &mut Vec<Texture>,
        notes: &mut Notes,
    ) -> Result<Arc<HeapSet>, Error> {
        let needs = set_needs(layout)?;
        let raw_layout = Arc::new(RawLayout::create(device, layout)?);
        let key = self.next_set;
        // The one version every command binds, where the device lets a pending one's set be
        // written; else each command buffer makes its own ([`bind`]).
        let shared = match unused_while_pending(device) {
            true => Some(Arc::new(SetVersion::allocate(
                device,
                &raw_layout,
                &needs,
                key,
            )?)),
            false => None,
        };
        let null_words = self.table(device)?.null_words();
        let mut images = Vec::new();
        let mut neutrals = Vec::new();
        for binding in &layout.bindings {
            let ty: vk::DescriptorType = binding.ty.into();
            let info = match binding.source {
                BindingSource::Heap(HandleClass::Image(class)) => {
                    let neutral = NeutralResource::for_image(&class)
                        .ok_or_else(|| internal("a heap array of a class without a neutral"))?;
                    let neutral = device.neutral_view(neutral)?;
                    images.push(ImageArray {
                        binding: binding.binding,
                        class,
                        ty,
                        count: binding.count,
                        neutral,
                    });
                    image_info(neutral)
                }
                BindingSource::Heap(HandleClass::Sampler { compare, .. }) => Info::Image(
                    vk::DescriptorImageInfo::default().sampler(device.neutral_sampler(compare)?),
                ),
                BindingSource::Heap(HandleClass::FunctionTable) => Info::Buffer(null_words),
                _ => return Err(internal("a heap set binding of no heap array")),
            };
            neutrals.push(HeapWrite {
                binding: binding.binding,
                element: 0,
                ty,
                info,
                keeper: None,
            });
        }
        let mut entry = SetEntry {
            key,
            layout: layout.clone(),
            raw_layout,
            needs,
            images,
            shared,
            open: Vec::new(),
            written: BTreeMap::new(),
            shadows: BTreeMap::new(),
        };
        entry.write(device, neutrals)?;
        let mut writes = Vec::new();
        for (&id, placed) in &self.textures {
            if let Some(texture) = Texture::upgrade(&placed.texture) {
                held.push(texture.clone());
                let (placed, shadow) = texture_writes(device, &entry.images, id, &texture, notes)?;
                writes.extend(placed);
                entry.shadow(id, &texture, shadow);
                notes.placed(&texture, true);
            }
        }
        entry.write(device, writes)?;
        self.next_set += 1;
        let set = Arc::new(HeapSet {
            device: device.clone(),
            key,
        });
        self.sets.insert(key, entry);
        self.by_layout
            .insert(layout.clone(), (key, Arc::downgrade(&set)));
        Ok(set)
    }
}

/// Whether `device`'s heap arrays can be written while a command buffer binding them is pending
/// (the module docs).
fn unused_while_pending(device: &Device) -> bool {
    metal2vulkan::pipeline::heap_unused_while_pending(device.compile_device().profile())
}

/// One image array of a heap set: its binding, class, descriptor type, element count, and
/// neutral's view (the device's).
struct ImageArray {
    binding: u32,
    class: ImageHandleClass,
    ty: vk::DescriptorType,
    count: u32,
    neutral: vk::ImageView,
}

/// One element a heap set holds: the descriptor and what keeps what it names (a sampler, a
/// table's words; a texture's image is retired as the texture goes instead).
struct HeapWrite {
    binding: u32,
    element: u32,
    ty: vk::DescriptorType,
    info: Info,
    keeper: Option<Keeper>,
}

/// One live heap set's state (the module docs).
struct SetEntry {
    /// Its key ([`HeapSet`]).
    key: u64,
    layout: DescriptorSetLayout,
    raw_layout: Arc<RawLayout>,
    needs: SetNeeds,
    images: Vec<ImageArray>,
    /// The one version every command binds, where the device has
    /// `descriptorBindingUpdateUnusedWhilePending`.
    shared: Option<Arc<SetVersion>>,
    /// Where it lacks it, the command buffers' own versions ([`bind`]), while they live and no
    /// submission took them.
    open: Vec<Weak<SetVersion>>,
    /// Every element written, by `(binding, element)`, but those of textures gone: what a
    /// command buffer's own version holds as it is made.
    written: BTreeMap<(u32, u32), Written>,
    /// By resource ID, the shadowed textures whose shadow image's view an array of the set
    /// holds, with the key a command's uses name each by (`Texture::bound_use`): a command that
    /// binds the set and neither names one nor makes it resident is announced
    /// (`Degradation::UnnamedShadowedTexture`).
    shadows: BTreeMap<u64, ResourceKey>,
}

/// An element a heap set holds ([`HeapWrite`]).
struct Written {
    ty: vk::DescriptorType,
    info: Info,
    /// Held while the set holds the element; dropped once another takes its place, the
    /// command buffers that bound it holding their own.
    _keeper: Option<Keeper>,
}

impl SetEntry {
    /// Note texture `texture` at ID `id` among the shadowed textures the set holds a view of
    /// where `shadow` (its arrays' writes hold its shadow image's view), else not.
    fn shadow(&mut self, id: u64, texture: &Texture, shadow: bool) {
        match shadow {
            true => {
                let key = texture.bound_use(Access::NONE, Access::NONE).key;
                self.shadows.insert(id, key)
            }
            false => self.shadows.remove(&id),
        };
    }

    /// The command buffers' own versions whose command buffer has not committed.
    fn open(&mut self) -> Vec<Arc<SetVersion>> {
        self.unsubmitted_where(|version| !version.committed.load(Ordering::Acquire))
    }

    /// The command buffers' own versions no submission took yet, committed or not.
    fn unsubmitted(&mut self) -> Vec<Arc<SetVersion>> {
        self.unsubmitted_where(|_| true)
    }

    /// The own versions no submission took yet that `pick` picks (those gone or submitted leave
    /// `open`).
    fn unsubmitted_where(&mut self, pick: impl Fn(&SetVersion) -> bool) -> Vec<Arc<SetVersion>> {
        let mut picked = Vec::new();
        self.open.retain(|version| match version.upgrade() {
            Some(version) if !version.submitted.load(Ordering::Acquire) => {
                if pick(&version) {
                    picked.push(version);
                }
                true
            }
            _ => false,
        });
        picked
    }

    /// The versions writes go to: the shared one, and the [`open`](SetEntry::open) ones.
    fn targets(&mut self) -> Vec<Arc<SetVersion>> {
        let mut targets: Vec<Arc<SetVersion>> = self.shared.iter().cloned().collect();
        targets.extend(self.open());
        targets
    }

    /// Write `writes` into every version writes go to ([`SetEntry::targets`]), and keep them
    /// for a version made later.
    fn write(&mut self, device: &Device, writes: Vec<HeapWrite>) -> Result<(), Error> {
        if writes.is_empty() {
            return Ok(());
        }
        let raw: Vec<Write> = writes
            .iter()
            .map(|write| raw_write(write.binding, write.element, write.ty, write.info))
            .collect();
        for version in self.targets() {
            // SAFETY: the shared version's arrays are `UPDATE_UNUSED_WHILE_PENDING`, and no
            // pending command buffer reads the elements written (the module docs); a command
            // buffer's own version is written only until it commits. Every view, sampler and
            // buffer written is held while the set holds it.
            unsafe { write_sets(device, &[version.set], &raw) }?;
        }
        for write in writes {
            self.written.insert(
                (write.binding, write.element),
                Written {
                    ty: write.ty,
                    info: write.info,
                    _keeper: write.keeper,
                },
            );
        }
        Ok(())
    }

    /// A command buffer's own version of the set (the module docs): every image array's
    /// elements a texture's ID may name the neutral, then every element `written`. `notes`
    /// takes the count written.
    fn own_version(
        &mut self,
        device: &Device,
        notes: &mut Notes,
    ) -> Result<Arc<SetVersion>, Error> {
        let version = SetVersion::allocate(device, &self.raw_layout, &self.needs, self.key)?;
        let mut all: Vec<Write> = Vec::new();
        for array in &self.images {
            let reach = array.count.min(TABLE_ENTRIES);
            all.extend((0..reach).map(|element| {
                raw_write(array.binding, element, array.ty, image_info(array.neutral))
            }));
        }
        all.extend(self.written.iter().map(|(&(binding, element), written)| {
            raw_write(binding, element, written.ty, written.info)
        }));
        // SAFETY: the set was just allocated; every view, sampler and buffer `written` names is
        // held (a texture's until its image retires), and the neutral views live with the device.
        unsafe { write_sets(device, &[version.set], &all) }?;
        notes.copied = Some(all.len());
        let version = Arc::new(version);
        self.open.push(Arc::downgrade(&version));
        Ok(version)
    }
}

/// The write of `info` at element `element` of binding `binding` of a heap set.
fn raw_write(binding: u32, element: u32, ty: vk::DescriptorType, info: Info) -> Write {
    Write {
        set: 0,
        binding,
        element,
        ty,
        info,
        held: None,
    }
}

/// An image view's descriptor info, in `GENERAL` (`binding.rs`).
fn image_info(view: vk::ImageView) -> Info {
    Info::Image(
        vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL),
    )
}

/// The writes of texture `texture`, at ID `id`, into `arrays`: element `id` of each, which it
/// holds as the array's class takes it (the module docs). An ID past an array's count is left
/// out, noted. With whether an element is its shadow image's view.
fn texture_writes(
    device: &Device,
    arrays: &[ImageArray],
    id: u64,
    texture: &Texture,
    notes: &mut Notes,
) -> Result<(Vec<HeapWrite>, bool), Error> {
    let mut writes = Vec::with_capacity(arrays.len());
    let mut shadow = false;
    for array in arrays {
        let element = match u32::try_from(id) {
            Ok(element) if element < array.count => element,
            _ => {
                notes.past = Some((id, array.count));
                continue;
            }
        };
        let (info, shadowed) = texture_info(device, array, texture, notes)?;
        shadow |= shadowed;
        writes.push(HeapWrite {
            binding: array.binding,
            element,
            ty: array.ty,
            info,
            keeper: None,
        });
    }
    Ok((writes, shadow))
}

/// What an element of `array` holds for `texture` (the module docs), and whether it is the
/// texture's shadow image's view.
fn texture_info(
    device: &Device,
    array: &ImageArray,
    texture: &Texture,
    notes: &mut Notes,
) -> Result<(Info, bool), Error> {
    let neutral = || Ok((image_info(array.neutral), false));
    let profile = device.compile_device().profile();
    let class = HandleClass::Image(array.class);
    let desc = match heap_element(
        profile,
        &class,
        &texture.facts(),
        metal2vulkan::vk::ImageLayout::GENERAL,
    ) {
        Ok(HeapElement::View(desc)) => desc,
        Ok(_) => return neutral(),
        Err(BindError::Unsupported(_)) => {
            notes.unviewable(texture);
            return neutral();
        }
        Err(error) => return Err(Error::Bind(error)),
    };
    let storage = array.ty == vk::DescriptorType::STORAGE_IMAGE;
    match texture.bound_view("new_texture", &desc, storage) {
        Ok(view) => Ok((image_info(view), texture.shadow().is_some())),
        // The class needs what the texture's usage or levels lack: Metal's read is undefined.
        Err(Error::Invalid(_)) => neutral(),
        Err(Error::Unsupported(_)) => {
            notes.unviewable(texture);
            neutral()
        }
        Err(error) => Err(error),
    }
}

/// What placing textures and writing sets announces, once no lock is held.
#[derive(Default)]
struct Notes {
    /// A texture a class fits that it cannot view.
    unviewable: Option<Texture>,
    /// The resource ID of a shadowed texture a set holds that the command binding the set
    /// neither names nor makes resident.
    shadowed: Option<u64>,
    /// An ID past an array's count, and the count.
    past: Option<(u64, u32)>,
    /// The elements a command buffer's own version of a set was made with.
    copied: Option<usize>,
    /// A texture a set holds whose blocks a carrier holds for copies (`pvrtc.rs`).
    unreadable: Option<Texture>,
}

impl Notes {
    fn unviewable(&mut self, texture: &Texture) {
        if self.unviewable.is_none() {
            self.unviewable = Some(texture.clone());
        }
    }

    /// `texture` placed, into a set where `in_set`.
    fn placed(&mut self, texture: &Texture, in_set: bool) {
        let held = || super::pvrtc::held_for_copies(texture).is_some();
        if in_set && self.unreadable.is_none() && held() {
            self.unreadable = Some(texture.clone());
        }
    }

    /// Announce what was noted.
    fn announce(self, device: &Device) {
        if let Some(texture) = &self.unviewable {
            device.degrade(Degradation::UnnamedTextureUnviewable, || {
                (
                    format!(
                        "{:?} texture at resource ID {}",
                        texture.pixel_format(),
                        texture.gpu_resource_id().0
                    ),
                    "a heap array whose class the texture fits cannot view its format (one the \
                     device cannot sample or store without a format, or that needs its \
                     function's decode)"
                        .to_owned(),
                )
            });
        }
        if let Some(id) = self.shadowed {
            device.degrade(Degradation::UnnamedShadowedTexture, || {
                (
                    format!("texture at resource ID {id}"),
                    "the texture's texels are a buffer's, which a shadow image holds only as \
                     the encoders that name it copy them in"
                        .to_owned(),
                )
            });
        }
        if let Some((id, count)) = self.past {
            device.degrade(Degradation::UnnamedTexturesPastCapacity, || {
                (
                    format!("texture at resource ID {id}"),
                    format!(
                        "a pipeline's set-1 heap arrays hold {count} elements, which the ID is past"
                    ),
                )
            });
        }
        if let Some(elements) = self.copied {
            device.degrade(Degradation::HeapSetCopied, || {
                (
                    "a set-1 heap set".to_owned(),
                    format!(
                        "the device lacks descriptorBindingUpdateUnusedWhilePending, so each \
                         command buffer binds a copy of its own, written as it first binds the \
                         set ({elements} elements here) and not after it commits"
                    ),
                )
            });
        }
        if let Some(texture) = &self.unreadable {
            super::pvrtc::degrade_unreadable(device, [texture], "a heap set");
        }
    }
}

/// A heap set's `VkDescriptorSetLayout`, which its versions hold.
struct RawLayout {
    device: Device,
    raw: vk::DescriptorSetLayout,
}

impl RawLayout {
    /// The layout of `layout`, a set-1 layout (no immutable sampler).
    fn create(device: &Device, layout: &DescriptorSetLayout) -> Result<RawLayout, Error> {
        if layout
            .bindings
            .iter()
            .any(|b| b.immutable_sampler.is_some())
        {
            return Err(internal("a heap set with an immutable sampler"));
        }
        let raw = super::pipeline::create_set_layout(device, layout, &[], false)?;
        Ok(RawLayout {
            device: device.clone(),
            raw,
        })
    }
}

impl Drop for RawLayout {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: every set of the layout went (each version holds it).
        unsafe { release::object(self.device.shared().vk(), self.raw) };
    }
}

/// One version of a heap set: a pool and the one set allocated from it (the module docs).
pub(crate) struct SetVersion {
    device: Device,
    _layout: Arc<RawLayout>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// Its heap set's key.
    key: u64,
    /// A command buffer that binds it committed (set under the device's `heaps` lock): nothing
    /// writes a command buffer's own version any more (the shared one's writers ignore it), but
    /// a gone texture's neutrals until it is submitted.
    committed: AtomicBool,
    /// A batch of a command buffer that binds it was submitted (set under the device's `heaps`
    /// lock, before the `vkQueueSubmit`): nothing writes it any more.
    submitted: AtomicBool,
}

impl SetVersion {
    /// A new pool of one set of `layout`, which takes `needs`, and the set, of heap set `key`.
    fn allocate(
        device: &Device,
        layout: &Arc<RawLayout>,
        needs: &SetNeeds,
        key: u64,
    ) -> Result<SetVersion, Error> {
        let raw = device.shared().vk();
        let sizes = needs.sizes(1);
        let info = vk::DescriptorPoolCreateInfo::default()
            .flags(needs.flags)
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: `info` is live for the call.
        let pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        let layouts = [layout.raw];
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: the pool holds one set of the layout.
        let allocated = unsafe { raw.allocate_descriptor_sets(&allocate) }.map_err(vulkan);
        let set = match allocated.and_then(|sets| {
            sets.first()
                .copied()
                .ok_or_else(|| internal("the driver allocated no heap set"))
        }) {
            Ok(set) => set,
            Err(error) => {
                // SAFETY: nothing uses the pool.
                unsafe { release::object(raw, pool) };
                return Err(error);
            }
        };
        Ok(SetVersion {
            device: device.clone(),
            _layout: Arc::clone(layout),
            pool,
            set,
            key,
            committed: AtomicBool::new(false),
            submitted: AtomicBool::new(false),
        })
    }
}

impl Drop for SetVersion {
    fn drop(&mut self) {
        // As for `RawLayout`.
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: no pending command buffer binds the set (each holds the version until it
        // completes); the set goes with its pool.
        unsafe { release::object(self.device.shared().vk(), self.pool) };
    }
}

/// A variant's hold on the heap set of its set-1 layout (`VariantObjects::heap_set`): the set
/// lives while a variant of the layout does, and its versions while a command buffer binds them.
pub(crate) struct HeapSet {
    device: Device,
    key: u64,
}

impl Drop for HeapSet {
    fn drop(&mut self) {
        let entry = {
            let mut heaps = lock(&self.device.shared().heaps);
            let entry = heaps.sets.remove(&self.key);
            if let Some(entry) = &entry {
                let ours = heaps
                    .by_layout
                    .get(&entry.layout)
                    .is_some_and(|(key, _)| *key == self.key);
                if ours {
                    heaps.by_layout.remove(&entry.layout);
                }
            }
            entry
        };
        // What the set held goes with no lock held.
        drop(entry);
    }
}

/// The heap set of set-1 layout `layout` on `device`: the live one, else a new one holding every
/// live texture (the module docs).
pub(crate) fn heap_set(
    device: &Device,
    layout: &DescriptorSetLayout,
) -> Result<Arc<HeapSet>, Error> {
    let mut held = Vec::new();
    let mut notes = Notes::default();
    let set = {
        let mut heaps = lock(&device.shared().heaps);
        match heaps
            .by_layout
            .get(layout)
            .and_then(|(_, set)| set.upgrade())
        {
            Some(set) => Ok(set),
            None => heaps.made_set(device, layout, &mut held, &mut notes),
        }
    };
    drop(held);
    notes.announce(device);
    set
}

/// Place `texture`, a program's texture that just took its ID's slot, into the table and every
/// heap set (the module docs). A failure to write it leaves the ID's entry the neutral's,
/// announced.
pub(super) fn place(device: &Device, texture: &Texture) {
    sync(device, Slots::Changed(texture.gpu_resource_id().0));
}

/// The texture slots a [`sync`] writes.
enum Slots {
    /// One, where its holder changed since it was written.
    Changed(u64),
    /// Every live texture's.
    All,
    /// Those of the live textures whose images and texel views lie in the bytes of the buffer of
    /// a key (`Texture::bytes_key`).
    Over(ResourceKey),
}

/// Write `slots`: a changed one where its holder changed, the others whatever changed. While a
/// batch holds the slots' writes ([`hold`]), they wait for it.
fn sync(device: &Device, slots: Slots) {
    let mut held = Vec::new();
    let mut notes = Notes::default();
    let failed = {
        let mut heaps = lock(&device.shared().heaps);
        match heaps.holds {
            0 => heaps.write_slots(device, slots, &mut held, &mut notes),
            _ => {
                heaps.deferred.push_back(slots);
                device.shared().counts.heap_slot_writes_deferred();
                Vec::new()
            }
        }
    };
    drop(held);
    notes.announce(device);
    for (id, error) in failed {
        failed_write(device, id, &error);
    }
}

/// A batch's hold of the device's texture slots' writes ([`hold`]), from before its first work's
/// take until its submission was made or the work given back (`texture/first_move.rs`'s
/// `Taken`). The writes asked meanwhile ([`place`], [`renew_over`], [`renew`]) are made as the
/// last hold ends, before the batch lets go of its place in the first work's order: so no
/// texture's elements reach the table or a heap set between the take and the `vkQueueSubmit`
/// (a texture registered after the take, whose move the batch did not take, would be in them in
/// `UNDEFINED`), and every later batch, which takes its place after, finds them written.
pub(super) struct HeldSlots {
    device: Device,
}

/// Hold `device`'s texture slots' writes for a batch taking its first work (the module docs): only
/// on a device whose heap sets take writes while a command buffer that binds them is pending, as
/// the writes made as it ends reach the sets the batch's commands bound, which it is by then
/// (`texture/first_move.rs` takes no hold elsewhere).
pub(super) fn hold(device: &Device) -> HeldSlots {
    lock(&device.shared().heaps).holds += 1;
    HeldSlots {
        device: device.clone(),
    }
}

impl Drop for HeldSlots {
    fn drop(&mut self) {
        let device = &self.device;
        let mut held = Vec::new();
        let mut notes = Notes::default();
        let failed = {
            let mut heaps = lock(&device.shared().heaps);
            heaps.holds -= 1;
            let mut failed = Vec::new();
            while heaps.holds == 0 {
                let Some(slots) = heaps.deferred.pop_front() else {
                    break;
                };
                failed.extend(heaps.write_slots(device, slots, &mut held, &mut notes));
            }
            failed
        };
        drop(held);
        notes.announce(device);
        for (id, error) in failed {
            failed_write(device, id, &error);
        }
    }
}

/// Announce that writing the heap elements of texture slot `id` failed with `error`: its entry
/// is the neutral's.
fn failed_write(device: &Device, id: u64, error: &Error) {
    device.degrade(Degradation::UnnamedTextureUnviewable, || {
        (
            format!("texture at resource ID {id}"),
            format!("writing its heap elements failed: {error}"),
        )
    });
}

/// Whether a command buffer that may read the heap sets is not yet complete: a ticket is out.
pub(super) fn tickets_out(device: &Device) -> bool {
    let heaps = lock(&device.shared().heaps);
    !heaps.retirement.out.is_empty() || !heaps.replaced.out.is_empty()
}

/// Write every live texture's elements again: a change of storage form swapped images in
/// (`texture/storage_form.rs`) while no command buffer that may read the elements is pending.
pub(super) fn renew(device: &Device) {
    sync(device, Slots::All);
}

/// Write again the elements of the live textures whose images and texel views lie in the bytes
/// of the buffer of `key` (`Texture::bytes_key`; a heap's own buffer for its textures and those
/// over its buffers), and no other's: a rebind or a detach of those bytes remade their images or
/// views, or made them present stand-ins or their own images again (`buffer_descriptor.rs`,
/// `detach.rs`); its caller commits no command that uses the bytes meanwhile, so no pending
/// command buffer reads the elements rewritten. Every other texture's images and views stay
/// what its elements were written with.
pub(super) fn renew_over(device: &Device, key: ResourceKey) {
    sync(device, Slots::Over(key));
}

/// A program's texture at `id`, of hold serial `serial`, went: its entry is the neutral's where
/// it held the slot, and `keeper` (what keeps its image) waits for the command buffers that may
/// read it (the module docs).
pub(super) fn gone(device: &Device, id: u64, serial: u64, keeper: Option<Keeper>) {
    let (freed, cleared) = {
        let mut heaps = lock(&device.shared().heaps);
        let held = heaps
            .textures
            .get(&id)
            .is_some_and(|placed| placed.serial == serial);
        let cleared = match held {
            true => heaps.clear(device, id),
            false => Ok(()),
        };
        (
            keeper.and_then(|keeper| heaps.retirement.retire(keeper)),
            cleared,
        )
    };
    drop(freed);
    if let Err(error) = cleared {
        failed_write(device, id, &error);
    }
}

/// Keep `kept`, what the sets' elements named until they were written again without it
/// (`renew_over`): the views and images of textures a detach or a reattach of their bytes replaced
/// (`detach.rs`, `buffer_descriptor.rs`). It waits for the command buffers committed before it,
/// which may read it (the module docs).
pub(super) fn retire(device: &Device, kept: Keeper) {
    let freed = lock(&device.shared().heaps).replaced.retire(kept);
    // It goes with no lock held.
    drop(freed);
}

/// A command buffer's heap: the versions its commands bound and its sampler listing (the module
/// docs), until it completes.
#[derive(Default)]
pub(super) struct HeapUse {
    versions: Vec<Arc<SetVersion>>,
    listed: Option<Listed>,
    /// Some command lists the argument samplers.
    lists_samplers: bool,
    /// The command buffer committed.
    committed: bool,
    /// Its first submission takes a ticket: its commands, or its host steps', may read the heap.
    wants_ticket: bool,
    /// Its ticket was taken.
    ticketed: bool,
}

impl HeapUse {
    /// Whether a command bound a heap set.
    pub(super) fn bound(&self) -> bool {
        !self.versions.is_empty()
    }

    /// As the command buffer commits, before it is submitted, once its commands listed the
    /// argument samplers again (`sampler_listing.rs`), where its commands or its host steps may
    /// read the heap: its ticket for what a rebind or a detach replaces, which it holds until it
    /// completes; its first submission takes its ticket for gone textures ([`submit`]). Nothing
    /// writes its own versions from now on but a gone texture's neutrals (the module docs). A
    /// command buffer that lists the argument samplers is counted in the device's
    /// `sampler_tables` until it completes: a state made meanwhile is not among them
    /// (`sampler.rs`).
    ///
    /// [`submit`]: HeapUse::submit
    pub(super) fn commit(&mut self, device: &Device) -> Ticket {
        self.committed = true;
        self.wants_ticket = true;
        if self.lists_samplers && self.listed.is_none() {
            self.listed = Some(Listed::new(&device.shared().sampler_tables));
        }
        let mut heaps = lock(&device.shared().heaps);
        for version in &self.versions {
            version.committed.store(true, Ordering::Release);
        }
        let epoch = heaps.replaced.take();
        drop(heaps);
        Ticket {
            device: device.clone(),
            epoch,
            of: Retired::Replaced,
        }
    }

    /// As a batch of the command buffer is submitted, before its `vkQueueSubmit`: its versions
    /// are submitted (nothing writes them any more), and the first such batch of one whose
    /// commit wanted it takes its ticket for gone textures, which it holds until it completes.
    pub(super) fn submit(&mut self, device: &Device) -> Option<Ticket> {
        let ticket = self.wants_ticket && !self.ticketed;
        if self.versions.is_empty() && !ticket {
            return None;
        }
        let mut heaps = lock(&device.shared().heaps);
        for version in &self.versions {
            version.submitted.store(true, Ordering::Release);
        }
        if !ticket {
            return None;
        }
        self.ticketed = true;
        let epoch = heaps.retirement.take();
        drop(heaps);
        Some(Ticket {
            device: device.clone(),
            epoch,
            of: Retired::Gone,
        })
    }

    /// Release what the command buffer held: it completed.
    pub(super) fn clear(&mut self) {
        self.versions.clear();
        self.listed = None;
        self.lists_samplers = false;
        self.committed = false;
        self.wants_ticket = false;
        self.ticketed = false;
    }
}

/// A command buffer's hold on what it may read through the heap (the module docs'
/// **Retirement**), given back as it completes.
pub(super) struct Ticket {
    device: Device,
    epoch: u64,
    of: Retired,
}

/// Which retirement a [`Ticket`] holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Retired {
    /// Gone textures' images: taken as the command buffer's first batch is submitted.
    Gone,
    /// What a rebind or a detach replaced: taken as it commits.
    Replaced,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let freed = {
            let mut heaps = lock(&self.device.shared().heaps);
            match self.of {
                Retired::Gone => heaps.retirement.give(self.epoch),
                Retired::Replaced => heaps.replaced.give(self.epoch),
            }
        };
        // The images go with no lock held.
        drop(freed);
    }
}

/// Where a command binds the device's texture table: its layout's texture handle tables, and
/// the table's entries its heap arrays reach ([`texture_table`]).
pub(super) struct TextureTable {
    pub(super) info: vk::DescriptorBufferInfo,
    /// The `(set, binding)` of each texture handle table.
    pub(super) at: Vec<(u32, u32)>,
}

/// The texture table a command of `layout` binds at its texture handle tables, over the entries
/// its heap arrays' count reaches: `None` for a layout without one.
pub(super) fn texture_table(
    device: &Device,
    layout: &PipelineLayout,
) -> Result<Option<TextureTable>, Error> {
    let at: Vec<(u32, u32)> = layout
        .sets
        .iter()
        .flat_map(|set| set.bindings.iter().map(move |b| (set.set, b)))
        .filter(|(_, b)| {
            matches!(
                b.source,
                BindingSource::Synthetic {
                    binding: SyntheticBinding::HandleTable,
                    ..
                }
            )
        })
        .map(|(set, b)| (set, b.binding))
        .collect();
    if at.is_empty() {
        return Ok(None);
    }
    // A table without heap arrays (a function that only tests handles for null) reaches every
    // entry.
    let capacity = layout
        .sets
        .get(HEAP_SET as usize)
        .and_then(|set| set.bindings.iter().map(|b| b.count).min())
        .unwrap_or(TABLE_ENTRIES);
    let entries = capacity.min(TABLE_ENTRIES);
    let mut heaps = lock(&device.shared().heaps);
    let past = heaps
        .textures
        .keys()
        .next_back()
        .copied()
        .filter(|&id| id >= u64::from(entries));
    let info = heaps.table(device)?.entries(entries);
    drop(heaps);
    if let Some(id) = past {
        Notes {
            past: Some((id, entries)),
            ..Notes::default()
        }
        .announce(device);
    }
    Ok(Some(TextureTable { info, at }))
}

/// The heap set a command of `objects` (a layout of two sets) binds, with `writes` (its set-1
/// writes) at their elements: the device's set of its layout, where every live texture is at its
/// ID (the module docs). Of `writes`, the argument samplers' and function tables' are written
/// where the set held another there; the textures' and the neutrals' the set holds already. The
/// command buffer's `heap` holds the set's version until it completes. `uses` are the command's
/// (`binding::command_uses`): a shadowed texture the set holds a view of that they neither name nor
/// make resident is announced (`Degradation::UnnamedShadowedTexture`): the command's function may
/// read its shadow through its ID, which the command's copies do not bring up to date, where a
/// texture they name or make resident has its shadow copied in. `None` for a command buffer listing
/// its argument samplers again as it commits, which reads nothing new.
pub(super) fn bind(
    device: &Device,
    objects: &VariantObjects,
    heap: &mut HeapUse,
    writes: Vec<Write>,
    uses: Option<&[Use<ResourceKey>]>,
) -> Result<vk::DescriptorSet, Error> {
    if objects.set_count() != HEAP_SET as usize + 1 {
        return Err(internal("a heap in a layout of other than two sets"));
    }
    if writes.iter().any(|write| write.set != HEAP_SET) {
        return Err(internal("a heap write to another set"));
    }
    let set = objects.heap_set()?;
    // A command buffer whose commands list the argument samplers lists them again as it commits
    // (`sampler_listing.rs`), and is counted from then on; a command a host step records after
    // the commit lists the states as they stand then, counted at once.
    if writes.iter().any(|w| w.ty == vk::DescriptorType::SAMPLER) {
        heap.lists_samplers = true;
        if heap.committed && heap.listed.is_none() {
            heap.listed = Some(Listed::new(&device.shared().sampler_tables));
        }
    }
    let mut notes = Notes::default();
    let version = {
        let mut heaps = lock(&device.shared().heaps);
        let entry = heaps
            .sets
            .get_mut(&set.key)
            .ok_or_else(|| internal("a variant's heap set gone"))?;
        // A shadowed texture the command neither names nor makes resident (`useResource:`, a
        // texture of a heap it uses, the command buffer's residency): its function may read the
        // shadow through the ID, which the command's copies leave as the last encoder that named
        // it copied it (Metal's own non-resident read is undefined). One its uses name has its
        // shadow copied in (`shadow::shadow_uses`, compute's and render's alike).
        notes.shadowed = uses.and_then(|uses| {
            entry
                .shadows
                .iter()
                .find(|(_, key)| !uses.iter().any(|used| used.key == **key))
                .map(|(&id, _)| id)
        });
        // The shared version, or the command buffer's own, made as it first binds the set.
        let own = heap.versions.iter().find(|v| v.key == set.key).cloned();
        let version = match (&entry.shared, own) {
            (Some(shared), _) => Arc::clone(shared),
            (None, Some(own)) => own,
            (None, None) => entry.own_version(device, &mut notes)?,
        };
        let staged: Vec<HeapWrite> = writes
            .into_iter()
            .filter(|write| {
                write.element != 0
                    && !matches!(
                        write.ty,
                        vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::STORAGE_IMAGE
                    )
                    && !entry
                        .written
                        .get(&(write.binding, write.element))
                        .is_some_and(|written| same(&written.info, &write.info))
            })
            .map(|write| HeapWrite {
                binding: write.binding,
                element: write.element,
                ty: write.ty,
                info: write.info,
                keeper: write.held,
            })
            .collect();
        entry.write(device, staged)?;
        version
    };
    notes.announce(device);
    let raw = version.set;
    if !heap.versions.iter().any(|v| Arc::ptr_eq(v, &version)) {
        heap.versions.push(version);
    }
    Ok(raw)
}

/// Whether two descriptors name the same objects (each held while a set holds it, so a handle
/// is never another object's meanwhile).
fn same(a: &Info, b: &Info) -> bool {
    match (a, b) {
        (Info::Buffer(a), Info::Buffer(b)) => {
            (a.buffer, a.offset, a.range) == (b.buffer, b.offset, b.range)
        }
        (Info::Image(a), Info::Image(b)) => {
            (a.sampler, a.image_view, a.image_layout) == (b.sampler, b.image_view, b.image_layout)
        }
        (Info::Texel(a), Info::Texel(b)) => a == b,
        _ => false,
    }
}

/// `samplers` as the bind state's argument samplers, each at its element of the sampler arrays
/// (`resource_ids.rs`; one past every array's capacity at `u32::MAX`, which [`fit_samplers`]
/// leaves out).
pub(super) fn listed(samplers: &[SamplerState]) -> Vec<ArgumentSampler> {
    samplers.iter().map(argument_sampler).collect()
}

/// `sampler` as one of the bind state's argument samplers ([`listed`]).
fn argument_sampler(sampler: &SamplerState) -> ArgumentSampler {
    let element = u32::try_from(sampler.heap_element()).unwrap_or(u32::MAX);
    ArgumentSampler::new(*sampler.descriptor(), sampler.gpu_resource_id().0).at_element(element)
}

/// Whether `kept` is `samplers` [`listed`], bit for bit (a float of the other sign, which
/// compares equal, is another listing): what a kept bind state need not take again.
pub(super) fn lists(kept: &[ArgumentSampler], samplers: &[SamplerState]) -> bool {
    let bits = |s: &ArgumentSampler| [s.desc.lod_min_clamp, s.desc.lod_max_clamp].map(f32::to_bits);
    kept.len() == samplers.len()
        && kept.iter().zip(samplers).all(|(kept, sampler)| {
            let listed = argument_sampler(sampler);
            *kept == listed && bits(kept) == bits(&listed)
        })
}

/// Where `plan`'s heap writes go for `state`, each heap array holding the capacity of set 1 of
/// `layout`: `None` for a layout without a heap.
pub(super) fn heap_elements(
    layout: &PipelineLayout,
    plan: &BindingPlan,
    state: &BindState,
) -> Result<Option<HeapElements>, Error> {
    let Some(set) = layout.sets.get(1) else {
        return Ok(None);
    };
    let capacity = set.bindings.iter().map(|b| b.count).min().unwrap_or(0);
    plan.heap_elements(state, capacity)
        .map(Some)
        .map_err(Error::Bind)
}

/// Whether every one of `samplers` (the argument samplers a command lists) has an element in
/// the heap set `set` of the command's variant (`None`: a layout without one, which reads none):
/// each heap array has the variant's heap capacity of elements, and a state's element is the
/// lowest no other live state held as it was made (`resource_ids.rs`), so one is past them only
/// where more states live than they hold. Where some have none, `samplers` keeps those that do
/// and the caller plans again with them; `unlisted` then takes the first left out's resource ID
/// and the capacity, where the set has sampler arrays (nothing reads them where it has none),
/// for [`degrade_unlisted`].
pub(super) fn fit_samplers(
    set: Option<&DescriptorSetLayout>,
    samplers: &mut Vec<SamplerState>,
    unlisted: &mut Option<(u64, u64)>,
) -> bool {
    let Some(set) = set else {
        return true;
    };
    let capacity = u64::from(set.bindings.iter().map(|b| b.count).min().unwrap_or(0));
    let fits = |sampler: &SamplerState| sampler.heap_element() < capacity;
    if samplers.iter().all(fits) {
        return true;
    }
    if set
        .bindings
        .iter()
        .any(|b| b.ty == metal2vulkan::vk::DescriptorType::SAMPLER)
    {
        let past = samplers.iter().find(|sampler| !fits(sampler));
        *unlisted = past.map(|sampler| (sampler.gpu_resource_id().0, capacity));
    }
    samplers.retain(fits);
    false
}

/// Announce that `commands` (the commands of a function, named) leave the sampler state at
/// resource ID `id` out of their argument samplers, their variant's set-1 sampler arrays holding
/// `capacity` elements ([`fit_samplers`]): their command buffer fails as it completes, where the
/// state still lives as it commits (`sampler_listing.rs`), or where a host step recorded them after
/// the commit.
pub(super) fn degrade_unlisted(device: &Device, (id, capacity): (u64, u64), commands: &str) {
    device.degrade(Degradation::UnlistedArgumentSampler, || {
        (
            format!("sampler state at resource ID {id}"),
            format!(
                "{commands}' set-1 sampler arrays hold {capacity} elements, fewer than the live \
                 sampler states, so they leave the state out of their argument samplers and \
                 their command buffer fails (CommandBufferError::UnlistedArgumentSampler)"
            ),
        )
    });
}

/// A command buffer whose commands list the argument samplers, counted in its device's
/// `sampler_tables` until it goes.
struct Listed(Arc<AtomicUsize>);

impl Listed {
    fn new(count: &Arc<AtomicUsize>) -> Listed {
        count.fetch_add(1, Ordering::AcqRel);
        Listed(Arc::clone(count))
    }
}

impl Drop for Listed {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
