//! The resources an encoder makes resident: the compute and render encoders' `use_resource(s)` and
//! `use_heap(s)`.
//!
//! - **Metal's contract.** A function reaches a resource the encoder does not bind (a buffer
//!   whose address an argument buffer or another buffer holds, a texture whose resource ID one
//!   holds) only once the encoder made it resident. The runtime lists each such resource among
//!   the bind state's used resources, so every later dispatch of the encoder reaches it:
//!   - a buffer's range is in the resident set, so a pointer loaded from memory into it is
//!     translated, and one into no resident range misses (it reads zero and drops writes, where
//!     Metal leaves the access undefined). The live buffers Metal reaches unnamed are in the set
//!     its command buffer's root tables search too (`live_buffers.rs`): every buffer of its own,
//!     and a no-copy buffer or a heap's buffer once a command buffer's commands used it; what
//!     misses is such a buffer no command buffer's commands used yet;
//!   - a texture is a member of the command's plan, under its resource ID (`resource_ids.rs`), so
//!     its uses are ordered as a bound texture's are; a handle loaded from memory finds every live
//!     texture, named or not, through the device's texture table (`heap_sets.rs`), as Metal's does;
//!   - a heap's whole range is in the resident set as one used resource (the `Heap` entry), so a
//!     pointer loaded from memory into any buffer placed in it is translated, and each texture made
//!     in it that lives when the encoder uses the heap is made resident as a texture is, and so is
//!     each texture over a buffer placed in it, which Metal's heap holds as it holds the buffer. A
//!     sparse heap holds no buffer and no range of its own: its live sparse textures alone are made
//!     resident. Metal's `useHeap:` makes a heap's resources resident for reading only, so its
//!     usage is `READ`;
//!   - a texture made in a heap after the encoder used it is made resident too, as the encoder
//!     encodes its next command (`sync_heaps`): Metal's `useHeap:` reaches every texture the heap
//!     holds when the command buffer is committed, those made after the call included (SPH heap
//!     probe 1, an M3 Max and an M4, compute and render, every heap type, with shader validation
//!     reporting nothing). One made after a command was encoded is reached by that command too,
//!     through the device's texture table, which holds every live texture as it is made
//!     (`heap_sets.rs`), though its uses are not ordered against the command's;
//!   - an indirect command buffer's records are in the resident set as its buffer's range, so a
//!     function's command encoders reach them through its resource ID, which is their address,
//!     while the buffer holds its ID's slot (a buffer made later with the ID takes it, and then a
//!     function reaches that one's records, or none once it is gone). Made resident for writing,
//!     the buffer is written by the GPU from then on (`IndirectCommandBuffer::written_by_gpu`): its
//!     executions are replayed on the host.
//! - **The command buffer's** (the coordinator's ruling of 2026-10-07). Metal's residency belongs
//!   to the command buffer: on an M3 Max, an indirect command buffer's compute command executed by
//!   one compute encoder wrote a buffer that only a later encoder's `useResource:` named, and one a
//!   later blit filled. So each encoder, as it ends, joins what it made resident into its command
//!   buffer's residency ([`Residency::join_encoder`]), and the commit makes resident the textures
//!   made since in the heaps it holds (`sync_heaps`). The work recorded at the commit or by a host
//!   step (an indirect command buffer's executions, compute and render, and an indirect dispatch
//!   the host reads) resolves and binds against that residency, every encoder's, made resident
//!   before the call or after it. A compute encoder's resources reach every render stage there, as
//!   it names none. A later compute or render encoder starts with the shadowed textures of that
//!   residency: every live texture is reached through its ID anyway (`heap_sets.rs`), but a shadow
//!   holds a texture's texels only as an encoder that makes it resident copies them in
//!   (`shadow.rs`), so each such encoder copies them in for its commands and orders them against
//!   its other uses, and a function that reads the texture through its ID reads its bytes as they
//!   stand. Its usage is the joined one, and so are its render stages in a render encoder.
//! - **Ordering.** A plan lists a used resource's use where its function reads loaded pointers
//!   (a buffer or a heap) or resolves handles (a texture or a heap), over the whole resource,
//!   reading or writing as the usage says (no usage does both), and the tracker orders it as it
//!   orders a bound one (`binding::command_uses`): a heap as the one resource its tracked buffers
//!   are (`heap.rs`). A shadowed texture is copied in and back as at a slot.
//! - **Tokens.** A resource has one token in the encoder, its position in the order the encoder
//!   first used it, and the union of every usage it was used with; so a plan's shape changes
//!   only when the encoder uses another resource or widens a usage.
//! - **Stages.** A render encoder makes a resource resident for the render stages it names
//!   (`useResource:usage:stages:`, `useHeap:stages:`), joined as the usages are: a draw's
//!   functions reach it in those of their stages alone (the binder's `UsedResource::stages`), and
//!   a tile dispatch only where they hold the tile stage. A compute encoder names none, and its
//!   dispatches reach every resource it made resident.

use super::buffer::{Buffer, ReadApart};
use super::heap::Heap;
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::live_buffers::LiveTable;
use super::resource::ResourceRef;
use super::texture::Texture;
use metal2vulkan::bind::{ResidentRange, ResourceClass, UsedResource};
use metal2vulkan::metal::{RenderStages, ResourceUsage};
use metal2vulkan::pipeline::{BindingSource, PipelineLayout, Stage, SyntheticBinding};
use std::sync::Arc;

/// A resource an encoder made resident.
#[derive(Clone)]
pub(super) enum Used {
    Buffer(Buffer),
    Texture(Texture),
    /// A heap, with its own buffer (a sparse heap, which has none, is not listed: only its
    /// textures are).
    Heap {
        heap: Heap,
        bytes: Buffer,
    },
    /// Its records' buffer.
    IndirectCommandBuffer(IndirectCommandBuffer),
}

/// The resources an encoder made resident, by token, each with its usage and render stages.
#[derive(Clone, Default)]
pub(super) struct Residency {
    used: Vec<(Used, ResourceUsage, RenderStages)>,
    /// The heaps the encoder used, each with the render stages it used it in and the number of
    /// the first of its textures not made resident yet (`Heap::textures_from`).
    heaps: Vec<HeapUse>,
    /// A command buffer's: the device's live buffers at its commit, which an address in no
    /// resource made resident resolves against (`live_buffers.rs`).
    live: Option<Arc<LiveTable>>,
}

/// A heap an encoder used (`Residency::heaps`).
#[derive(Clone)]
struct HeapUse {
    heap: Heap,
    stages: RenderStages,
    next: u64,
}

impl Residency {
    /// Make `resource` resident with `usage` in render `stages` (none for a compute encoder),
    /// joined to the usage and stages it had: the resource, when the encoder first uses it here
    /// (the caller then holds it until the command buffer completes).
    pub(super) fn use_resource(
        &mut self,
        resource: ResourceRef<'_>,
        usage: ResourceUsage,
        stages: RenderStages,
    ) -> Option<Used> {
        let used = match resource {
            ResourceRef::Buffer(buffer) => Used::Buffer(buffer.clone()),
            ResourceRef::Texture(texture) => Used::Texture(texture.clone()),
            ResourceRef::IndirectCommandBuffer(icb) => {
                if usage.contains(ResourceUsage::WRITE) {
                    icb.written_by_gpu();
                }
                Used::IndirectCommandBuffer(icb.clone())
            }
        };
        self.join(used, usage, stages)
    }

    /// Make `heap` (with its own buffer; a sparse heap has none) and its live textures resident
    /// for reading in render `stages` (none for a compute encoder), as Metal's `useHeap:` and
    /// `useHeap:stages:` do, joined to the stages the encoder used the heap in before, and
    /// remember the heap for [`sync_heaps`](Self::sync_heaps): those the encoder first uses here
    /// (the caller then holds them until the command buffer completes).
    pub(super) fn use_heap(&mut self, heap: &Heap, stages: RenderStages) -> Vec<Used> {
        let (textures, next) = heap.textures_from(0);
        let stages = match self.heaps.iter_mut().find(|used| used.heap == *heap) {
            Some(used) => {
                used.stages = used.stages.union(stages);
                used.next = next;
                used.stages
            }
            None => {
                self.heaps.push(HeapUse {
                    heap: heap.clone(),
                    stages,
                    next,
                });
                stages
            }
        };
        let used = heap.bytes().map(|bytes| Used::Heap {
            heap: heap.clone(),
            bytes: bytes.clone(),
        });
        let textures = textures.into_iter().map(Used::Texture);
        used.into_iter()
            .chain(textures)
            .filter_map(|used| self.join(used, ResourceUsage::READ, stages))
            .collect()
    }

    /// The heaps of what the encoder made resident: those it used, and those its resident
    /// buffers, textures and indirect command buffers lie in.
    pub(super) fn heaps(&self) -> impl Iterator<Item = Heap> + '_ {
        self.used.iter().filter_map(|(used, ..)| match used {
            Used::Buffer(buffer) => buffer.heap().cloned(),
            Used::Texture(texture) => texture.heap().cloned(),
            Used::Heap { heap, .. } => Some(heap.clone()),
            Used::IndirectCommandBuffer(icb) => icb.heap().cloned(),
        })
    }

    /// Make resident each texture made in a heap the encoder used since the encoder last looked,
    /// for reading in the stages it used the heap in, as `use_heap` made the heap's textures
    /// (the module docs): an encoder calls it as it encodes each command that reaches its
    /// resident resources. Those it first uses here (the caller then holds them until the
    /// command buffer completes). A heap that made no texture since costs one lock.
    pub(super) fn sync_heaps(&mut self) -> Vec<Used> {
        let mut made = Vec::new();
        for used in &mut self.heaps {
            if used.heap.textures_made() == used.next {
                continue;
            }
            let (textures, next) = used.heap.textures_from(used.next);
            used.next = next;
            made.extend(textures.into_iter().map(|texture| (texture, used.stages)));
        }
        made.into_iter()
            .filter_map(|(texture, stages)| {
                self.join(Used::Texture(texture), ResourceUsage::READ, stages)
            })
            .collect()
    }

    /// Join what an ended encoder made resident (`ended`) into its command buffer's residency
    /// (the module docs): each resource with its usage, and each heap it used, in the render
    /// stages it named, or every render stage for a compute encoder's (`compute`), which names
    /// none; a heap keeps the least number of a texture not made resident yet, so the commit's
    /// [`sync_heaps`](Self::sync_heaps) makes resident every texture made in it since.
    pub(super) fn join_encoder(&mut self, ended: &Residency, compute: bool) {
        let reach = |stages: RenderStages| match compute {
            true => every_render_stage(),
            false => stages,
        };
        for (used, usage, stages) in &ended.used {
            self.join(used.clone(), *usage, reach(*stages));
        }
        for heap in &ended.heaps {
            let stages = reach(heap.stages);
            match self.heaps.iter_mut().find(|had| had.heap == heap.heap) {
                Some(had) => {
                    had.stages = had.stages.union(stages);
                    had.next = had.next.min(heap.next);
                }
                None => self.heaps.push(HeapUse {
                    heap: heap.heap.clone(),
                    stages,
                    next: heap.next,
                }),
            }
        }
    }

    /// The residency a compute command reaches, stated as a compute encoder's (no render
    /// stages), so its plan is a kernel's of the same shape as the encoder's own dispatch's
    /// (`tile_resources` alike).
    pub(super) fn kernel(mut self) -> Residency {
        for (_, _, stages) in &mut self.used {
            *stages = RenderStages::empty();
        }
        for heap in &mut self.heaps {
            heap.stages = RenderStages::empty();
        }
        self
    }

    /// The indirect command buffers made resident for writing, each once.
    pub(super) fn writable_indirect_command_buffers(&self) -> Vec<IndirectCommandBuffer> {
        let mut writable: Vec<IndirectCommandBuffer> = Vec::new();
        for (used, usage, _) in &self.used {
            if let Used::IndirectCommandBuffer(icb) = used {
                if usage.contains(ResourceUsage::WRITE) && !writable.contains(icb) {
                    writable.push(icb.clone());
                }
            }
        }
        writable
    }

    /// The resources made resident.
    pub(super) fn resources(&self) -> impl Iterator<Item = &Used> {
        self.used.iter().map(|(used, ..)| used)
    }

    /// The heaps used (`use_heap`), a sparse heap among them.
    pub(super) fn used_heaps(&self) -> impl Iterator<Item = &Heap> {
        self.heaps.iter().map(|used| &used.heap)
    }

    /// Whether it holds a resource `other` does not.
    pub(super) fn beyond(&self, other: &Residency) -> bool {
        self.used
            .iter()
            .any(|(used, ..)| !other.used.iter().any(|(had, ..)| same(had, used)))
    }

    /// `used` resident with `usage` in `stages`, joined to the usage and stages it had: `used`,
    /// when it is new here.
    fn join(&mut self, used: Used, usage: ResourceUsage, stages: RenderStages) -> Option<Used> {
        let had = self.used.iter_mut().find(|(had, ..)| same(had, &used));
        if let Some((_, had_usage, had_stages)) = had {
            *had_usage = had_usage.union(usage);
            *had_stages = had_stages.union(stages);
            return None;
        }
        self.used.push((used.clone(), usage, stages));
        Some(used)
    }

    /// The bind state's used resources, each with its render stages. A buffer is stated in its
    /// `VkBuffer`'s space, as the argument table states a bound one (its length and Metal address
    /// widened by its base offset), so the resident range covers every byte the `VkBuffer` maps; a
    /// heap as its own buffer is (`Heap::bytes`), whose range holds every buffer placed in it; a
    /// texture by its facts and resource ID.
    pub(super) fn used_resources(&self) -> Vec<UsedResource> {
        self.used
            .iter()
            .zip(0u32..)
            .map(|((resource, usage, stages), token)| {
                let mut used = match resource {
                    Used::Buffer(buffer) => stated(buffer, ResourceClass::Buffer, token),
                    Used::Heap { bytes, .. } => stated(bytes, ResourceClass::Heap, token),
                    // Reached through its ID only while it holds the ID's slot: a buffer made
                    // later with the ID takes it (`address.rs`).
                    Used::IndirectCommandBuffer(icb) => {
                        let mut used = stated(icb.storage(), ResourceClass::Buffer, token);
                        if !icb.holds_id_slot() {
                            used.gpu_address = None;
                        }
                        used
                    }
                    Used::Texture(texture) => {
                        let class = ResourceClass::Texture(texture.facts());
                        let mut used = UsedResource::new(class, token);
                        used.resource_id = Some(texture.gpu_resource_id().0);
                        used
                    }
                };
                used.usage = *usage;
                used.stages = *stages;
                used
            })
            .collect()
    }

    /// The used resources a tile dispatch reaches: those made resident in the tile stage, under
    /// their tokens, stated as a compute dispatch's are (no stages; its plan is a kernel's).
    pub(super) fn tile_resources(&self) -> Vec<UsedResource> {
        let mut used = self.used_resources();
        used.retain(|resource| resource.stages.contains(RenderStages::TILE));
        for resource in &mut used {
            resource.stages = RenderStages::empty();
        }
        used
    }

    /// The resource of `token`.
    pub(super) fn get(&self, token: u32) -> Option<&Used> {
        self.used.get(token as usize).map(|(used, ..)| used)
    }

    /// The command buffer's residency with `live`, the device's live buffers at its commit.
    pub(super) fn with_live(self, live: Arc<LiveTable>) -> Residency {
        Residency {
            live: Some(live),
            ..self
        }
    }

    /// It without the device's live buffers: what keeps it past its command buffer's commit
    /// holds no table of them, whose pages a release or a rebind lets go until the command
    /// buffer is first submitted.
    pub(super) fn without_live(self) -> Residency {
        Residency { live: None, ..self }
    }

    /// The ranges of its buffers, heaps (their own buffers) and indirect command buffers'
    /// records for a resident set, in their `VkBuffer`s' space, where they have device
    /// addresses: as the bind state's used resources state them.
    pub(super) fn resident_ranges(&self) -> Vec<ResidentRange> {
        self.used_resources()
            .into_iter()
            .filter_map(
                |used| match (used.class, used.gpu_address, used.device_address) {
                    (ResourceClass::Buffer | ResourceClass::Heap, Some(guest), Some(host)) => {
                        Some(ResidentRange::new(host, guest, used.length))
                    }
                    _ => None,
                },
            )
            .collect()
    }

    /// The resident buffer that Metal address `address` lies in, and its offset there: a buffer
    /// made resident, else a heap's own buffer (`Heap::bytes`), else, for a command buffer's,
    /// one of the device's live buffers of their own. An address at a resource's end is in it too
    /// (an empty binding), where no resource holds it. `None`: in none of them.
    pub(super) fn resolve(&self, address: u64) -> Option<(Buffer, u64)> {
        let ranges = || {
            self.used.iter().filter_map(|(used, ..)| {
                let buffer = match used {
                    Used::Buffer(buffer) => buffer,
                    Used::Heap { bytes, .. } => bytes,
                    Used::Texture(_) | Used::IndirectCommandBuffer(_) => return None,
                };
                let offset = address.checked_sub(buffer.metal_address().0)?;
                (offset <= buffer.length()).then_some((buffer, offset))
            })
        };
        let inside = ranges().find(|(buffer, offset)| *offset < buffer.length());
        inside
            .or_else(|| ranges().next())
            .map(|(buffer, offset)| (buffer.clone(), offset))
            .or_else(|| self.live.as_ref()?.resolve(address))
    }

    /// What a command buffer's next compute or render (`render`) encoder starts with, of the
    /// command buffer's residency (the module docs): the shadowed textures (`shadow.rs`) its
    /// ended encoders made resident, each with its usage and, for a render encoder, its render
    /// stages (a compute encoder names none).
    pub(super) fn shadows(&self, render: bool) -> Residency {
        let used = (self.used.iter())
            .filter(|(used, ..)| matches!(used, Used::Texture(t) if t.shadow().is_some()))
            .map(|(used, usage, stages)| {
                let stages = if render {
                    *stages
                } else {
                    RenderStages::empty()
                };
                (used.clone(), *usage, stages)
            })
            .collect();
        Residency {
            used,
            heaps: Vec::new(),
            live: None,
        }
    }

    /// The textures made resident.
    pub(super) fn textures(&self) -> impl Iterator<Item = &Texture> {
        self.used.iter().filter_map(|(used, ..)| match used {
            Used::Texture(texture) => Some(texture),
            Used::Buffer(_) | Used::Heap { .. } | Used::IndirectCommandBuffer(_) => None,
        })
    }

    /// The textures made resident that a render command whose functions read texture handles
    /// from memory in render `stages` reaches: those made resident in one of those stages.
    pub(super) fn textures_reached(&self, stages: RenderStages) -> impl Iterator<Item = &Texture> {
        self.used
            .iter()
            .filter_map(move |(used, _, made)| match used {
                Used::Texture(texture) if made.intersects(stages) => Some(texture),
                _ => None,
            })
    }

    /// The textures made resident, each with whether a function's use of it may write: its
    /// usage writes, or neither reads nor writes, which a plan takes as both (the plan's used
    /// resources, `metal2vulkan::bind`).
    pub(super) fn textures_written(&self) -> impl Iterator<Item = (&Texture, bool)> {
        self.used.iter().filter_map(|(used, usage, _)| match used {
            Used::Texture(texture) => {
                let reads = usage.intersects(ResourceUsage::READ | ResourceUsage::SAMPLE);
                Some((texture, usage.contains(ResourceUsage::WRITE) || !reads))
            }
            Used::Buffer(_) | Used::Heap { .. } | Used::IndirectCommandBuffer(_) => None,
        })
    }

    /// The buffers made resident, a heap as its own buffer, each with whether a function's use
    /// of it may write, as [`textures_written`](Self::textures_written) says it of a texture.
    pub(super) fn buffers_written(&self) -> impl Iterator<Item = (&Buffer, bool)> {
        self.used.iter().filter_map(|(used, usage, _)| {
            let buffer = match used {
                Used::Buffer(buffer) => buffer,
                Used::Heap { bytes, .. } => bytes,
                Used::Texture(_) | Used::IndirectCommandBuffer(_) => return None,
            };
            let reads = usage.intersects(ResourceUsage::READ | ResourceUsage::SAMPLE);
            Some((buffer, usage.contains(ResourceUsage::WRITE) || !reads))
        })
    }
}

/// Whether `a` and `b` are one resource.
fn same(a: &Used, b: &Used) -> bool {
    match (a, b) {
        (Used::Buffer(a), Used::Buffer(b)) => a == b,
        (Used::Texture(a), Used::Texture(b)) => a == b,
        (Used::Heap { heap: a, .. }, Used::Heap { heap: b, .. }) => a == b,
        (Used::IndirectCommandBuffer(a), Used::IndirectCommandBuffer(b)) => a == b,
        _ => false,
    }
}

/// `buffer` as a used resource of `class` with `token`, in its `VkBuffer`'s space.
fn stated(buffer: &Buffer, class: ResourceClass, token: u32) -> UsedResource {
    let base = buffer.base();
    let mut used = UsedResource::new(class, token);
    used.gpu_address = Some(buffer.exposed_address(ReadApart::Binding).0 - base);
    used.length = base + buffer.length();
    // A detached buffer's bytes are in no pages: an address reaching them misses and reads 0, as
    // Metal's unnamed reach of a detached buffer does (`detach.rs`), never the stand-in's.
    used.device_address = match buffer.detached() {
        true => None,
        false => buffer.device_address().map(|address| address - base),
    };
    used
}

/// The render stages whose functions read texture handles from memory in `layout`: those with a
/// handle table (`SyntheticBinding::HandleTable`), a kernel's as `kernel` (the tile stage for a
/// tile function). A stage the runtime does not know reaches every render stage.
pub(super) fn handle_stages(layout: &PipelineLayout, kernel: RenderStages) -> RenderStages {
    layout
        .sets
        .iter()
        .flat_map(|set| &set.bindings)
        .filter_map(|binding| match binding.source {
            BindingSource::Synthetic {
                stage,
                binding: SyntheticBinding::HandleTable,
            } => Some(stage),
            _ => None,
        })
        .fold(RenderStages::empty(), |stages, stage| {
            stages.union(match stage {
                Stage::Vertex => RenderStages::VERTEX,
                Stage::Fragment => RenderStages::FRAGMENT,
                Stage::Kernel => kernel,
                Stage::Object => RenderStages::OBJECT,
                Stage::Mesh => RenderStages::MESH,
                _ => every_render_stage(),
            })
        })
}

/// Every render stage Metal names.
pub(super) fn every_render_stage() -> RenderStages {
    RenderStages::VERTEX
        .union(RenderStages::FRAGMENT)
        .union(RenderStages::TILE)
        .union(RenderStages::OBJECT)
        .union(RenderStages::MESH)
}
