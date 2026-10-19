//! A render encoder whose calls the device's resolver encodes (`resolver.rs`), each answered at
//! the call.
//!
//! - **The answer at the call.** A call returns what its encoding at the call would return. A call
//!   whose refusals depend on its arguments and the device alone (the argument tables' setters,
//!   the viewport, scissor and depth/stencil state setters, and those that refuse nothing) checks
//!   them at the call, in the encoding's order. The open, a draw, `set_render_pipeline_state` and
//!   `end_encoding` are answered `Ok` where an earlier call of the device was answered `Ok` at the
//!   call with the same state: the same pass descriptor, pipeline state, render state, objects in
//!   the same slots, buffers at offsets of the same class, and kind of draw (the device's
//!   [`AnswerMemo`], keyed by these exactly, objects by identities no other object takes while the
//!   memo names them).
//! - **A buffer's offset by its class.** A buffer's bind offset is keyed by its alignment up to 256
//!   bytes, whether it is the buffer's end, and whether more than 4 GiB of the buffer lie past it
//!   ([`offset_class`]), so a ring buffer bound at moving offsets keys one draw. A keyed draw's
//!   answer reads a bind offset through those alone: its binding plan reads it only through its
//!   alignment class under the device's view base alignment, at most 256 by Vulkan's limits on
//!   offset alignments (`BindState::shape`; `minUniformBufferOffsetAlignment` and
//!   `minStorageBufferOffsetAlignment` are at most 256); the binding takes the range past it as a
//!   value (`binding.rs`); the vertex fetch past 4 GiB refuses only past a buffer of more than
//!   that past the offset (`render/vertex_reach.rs`), which the memo keys no draw of; and the
//!   setter's own refusal (an offset past the buffer's length) is checked at the call.
//! - **What the memo keys.** Passes with no color groups, rate map, custom sample positions,
//!   layers or extended-range colors, no attachment in a buffer's or a heap's bytes and none over
//!   the caller's image, in a command buffer whose earlier encoders made no shadowed texture
//!   resident, keyed with whether a write switched their depth on (`texture/signed_depth.rs`);
//!   pipeline states that draw primitives with no tessellation, mesh stages, tile memory,
//!   imageblock, attachment reads or writes, memory writes, argument buffers or function tables,
//!   and at most one amplification; direct draws, an indexed one where it reads its indices in
//!   place (no copy, `draw_source.rs`, which it would announce); the calls above and no other, no
//!   vertex buffer bound with more than 4 GiB past its offset, and no texture bound over the
//!   caller's image, whose commit announces its layout moves.
//! - **Notices at the call alone.** The memo learns a call only from its encoding at the call where
//!   that delivered no message on the calling thread (`device::heard_here`) and left none to a
//!   later draw: a draw that recorded a pipeline being made or its residual form, whose key's own
//!   announces its notices at the first draw that finds it made (`device::unsettled`). An encoder
//!   whose call at the call announced anything teaches nothing more. So a call whose encoding
//!   announces (a kind the device announces at every occurrence, a kind not announced yet, an open
//!   whose attachments a notice names) is encoded at the call each time it would announce, and its
//!   notice comes inside the program's call, as without a resolver; a captured call repeats an
//!   encoding that announced nothing. A message the resolver's thread delivers anyway is counted
//!   (`WorkCounts::resolver_notices`), and the memo forgets every key (`resolver.rs`).
//! - **Otherwise the encoder is encoded at the call** from that call on: everything handed to the
//!   resolver is encoded first (one drain), the encoder's encoding is taken back, and the call and
//!   the encoder's later calls run at the call, as without a resolver. Calls answered `Ok` at the
//!   call teach the memo, while every call of the encoder so far is one it keys.
//! - **Handed in pieces.** Answered calls are logged with their arguments (handles held, bytes
//!   copied, `render/replay.rs`) and handed to the resolver at a draw where it waits for work or
//!   the piece grew long ([`PIECE_CALLS`]), and at the end, so the resolver encodes the pass while
//!   the program makes its next calls; a piece handed before the end wakes no sleeping resolver
//!   (`ResolverShared::hand`).
//! - **The command buffer's state at the call.** The open refuses as the encoding's open does
//!   (committed, an encoder open, one abandoned) and marks an encoder open; `end_encoding` marks
//!   it free for the next encoder; a drop before the end abandons the command buffer at the drop,
//!   the encoding taken back and let go of there. The resolver's open and end of the encoder leave
//!   that state alone (`EncoderCore::captured`).
//! - **Untaught answers** (`DeviceOptions::answer_untaught`, for validation): every call the
//!   memo keys is answered `Ok` without an earlier answer, so a key that leaves out what decides
//!   an answer shows as a refusal on the resolver (`WorkCounts::answered_calls_refused`).
//! - **A failure on the resolver** (out of memory, a lost device, a runtime defect) fails the
//!   command buffer as it completes (`resolver.rs`).

use super::replay::{RenderCall, RenderRoom, Table};
use super::{OpenAs, RenderCommandEncoder, RenderEncoding};
use crate::buffer::Buffer;
use crate::command_buffer::{CommandBuffer, ResolverHold};
use crate::depth_stencil::DepthStencilState;
use crate::device::{lock, Device};
use crate::encoder::OpenGuard;
use crate::error::{invalid, Error, Problem};
use crate::fetch_reach::REACHED;
use crate::pipeline::{RenderPipelineState, RenderStateKind};
use crate::render_pass::RenderPassDescriptor;
use crate::render_state::RenderStateProblem;
use crate::render_values::{ScissorRect, Viewport};
use crate::resolver::Work;
use crate::sampler::SamplerState;
use crate::texture::Texture;
use metal2vulkan::metal::{CullMode, IndexType, PrimitiveType, Winding};
use metal2vulkan::pipeline::{BufferKind, SlotInterface};
use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};

/// How a render encoder encodes its calls ([`RenderCommandEncoder`]).
pub(super) enum Form {
    /// Each call at the call, from the open on.
    AtTheCall(Box<RenderEncoding>),
    /// Answered at the call and encoded by the device's resolver.
    Captured(Box<Captured>),
    /// Encoded at the call, its calls teaching the memo (the module docs).
    Demoted(Box<Demoted>),
    /// The resolver could not open the pass (its command buffer fails as it completes): each
    /// call refuses nothing, and nothing is encoded.
    Failed(Box<Front>),
    /// Ended.
    Ended,
}

/// What a call answers.
pub(crate) trait Answer {
    /// The answer of a call that refuses nothing.
    fn ok() -> Self;
    /// The answer of a call refused with `error`.
    fn refused(error: Error) -> Self;
    /// Whether it refuses nothing.
    fn is_ok(&self) -> bool;
}

impl Answer for () {
    fn ok() {}
    fn refused(_: Error) {}
    fn is_ok(&self) -> bool {
        true
    }
}

impl Answer for Result<(), Error> {
    fn ok() -> Self {
        Ok(())
    }
    fn refused(error: Error) -> Self {
        Err(error)
    }
    fn is_ok(&self) -> bool {
        Result::is_ok(self)
    }
}

/// A piece of a captured encoder the resolver encodes (`Work::Render`): the pass's open first
/// where `open` holds its descriptor, then `calls`, then its end where `end`.
pub(crate) struct RenderWork {
    slot: Arc<RenderSlot>,
    open: Option<Box<RenderPassDescriptor>>,
    calls: RenderRoom,
    end: bool,
}

/// Where a captured encoder's encoding is kept between its pieces.
pub(crate) struct RenderSlot {
    command_buffer: ResolverHold,
    encoding: Mutex<Option<Box<RenderEncoding>>>,
}

impl RenderWork {
    /// The device whose encoder it is.
    pub(crate) fn device(&self) -> &Device {
        self.slot.command_buffer.device()
    }

    /// Encode it, on the resolver's thread (`resolver.rs`).
    pub(crate) fn resolve(self) {
        let RenderWork {
            slot,
            open,
            calls,
            end,
        } = self;
        let command_buffer = &slot.command_buffer;
        let mut held = lock(&slot.encoding);
        if let Some(descriptor) = open {
            match command_buffer.open_captured(&descriptor) {
                Ok(encoding) => *held = Some(Box::new(encoding)),
                Err(error) => command_buffer.resolver_failed(error),
            }
        }
        if let Some(encoding) = held.as_mut() {
            calls.run_on(encoding, |error| command_buffer.resolver_failed(error));
        }
        if end {
            if let Some(encoding) = held.take() {
                if let Err(error) = encoding.end_encoding() {
                    command_buffer.resolver_failed(error);
                }
            }
        }
        drop(held);
        let shared = command_buffer.device().shared();
        shared.scheduler.spare_logs().give_room(calls);
    }
}

/// A captured encoder: what it hands the resolver, and what it knows at the call.
pub(super) struct Captured {
    front: Front,
    slot: Arc<RenderSlot>,
    /// The pass's descriptor, until the first piece hands it with the open.
    open: Option<Box<RenderPassDescriptor>>,
    /// The calls logged since the last piece.
    calls: RenderRoom,
}

/// An encoder encoded at the call, its calls teaching the memo.
pub(super) struct Demoted {
    front: Front,
    encoding: Box<RenderEncoding>,
}

/// What an encoder the memo follows knows of its state at the call: its answers' and the
/// memo's keys.
pub(super) struct Front {
    /// The encoder's hold on its command buffer, where its encoding's is not: dropped before the
    /// end, it abandons it.
    guard: OpenGuard,
    pass: Arc<PassFacts>,
    /// The pipeline state set, where the memo keys its draws.
    pipeline: Option<usize>,
    vertex: TableKey,
    fragment: TableKey,
    render: RenderKey,
    /// Every call so far was one the memo keys: its keys stand for the encoder's state.
    keyed: bool,
    /// A depth/stencil state with depth writes was set in the pass.
    depth_writes: bool,
    /// The end's key so far: each draw's memo entry and whether it drew nothing.
    ends: Vec<u64>,
    /// The words of the key being looked up.
    scratch: Vec<u64>,
    /// Its calls answered at the call, not yet counted (`WorkCounts::calls_answered`).
    answered: u64,
}

/// The lists a [`Front`] keys its state in, kept between encoders by the thread that lets go of
/// them, so an open allocates none ([`FrontRoom::take`]).
#[derive(Default)]
struct FrontRoom {
    vertex: Vec<[u64; 3]>,
    fragment: Vec<[u64; 3]>,
    viewports: Vec<u64>,
    scissors: Vec<u64>,
    cull: Vec<u64>,
    winding: Vec<u64>,
    ends: Vec<u64>,
    scratch: Vec<u64>,
}

/// The most [`FrontRoom`]s a thread keeps.
const SPARE_FRONT_ROOMS: usize = 4;

/// The most words a kept [`FrontRoom`] holds room for; one with more goes.
const FRONT_ROOM_WORDS: usize = 4096;

thread_local! {
    /// This thread's kept [`FrontRoom`]s.
    static FRONT_ROOMS: RefCell<Vec<FrontRoom>> = const { RefCell::new(Vec::new()) };
    /// The words of a pass descriptor being looked up ([`pass_words`]).
    static PASS_WORDS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

impl FrontRoom {
    /// A kept room of this thread's, else a new one, which allocates nothing until used.
    fn take() -> FrontRoom {
        FRONT_ROOMS
            .try_with(|rooms| rooms.borrow_mut().pop())
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    /// Keep this room, emptied, for this thread's next encoder, where the thread keeps fewer than
    /// [`SPARE_FRONT_ROOMS`] and it holds room for at most [`FRONT_ROOM_WORDS`] words.
    fn keep(mut self) {
        let words = 3 * (self.vertex.capacity() + self.fragment.capacity())
            + self.viewports.capacity()
            + self.scissors.capacity()
            + self.cull.capacity()
            + self.winding.capacity()
            + self.ends.capacity()
            + self.scratch.capacity();
        if words > FRONT_ROOM_WORDS {
            return;
        }
        self.vertex.clear();
        self.fragment.clear();
        self.viewports.clear();
        self.scissors.clear();
        self.cull.clear();
        self.winding.clear();
        self.ends.clear();
        self.scratch.clear();
        let _ = FRONT_ROOMS.try_with(|rooms| {
            let mut rooms = rooms.borrow_mut();
            if rooms.len() < SPARE_FRONT_ROOMS {
                rooms.push(self);
            }
        });
    }
}

impl Drop for Front {
    /// Its lists go to this thread's kept rooms ([`FrontRoom::keep`]).
    fn drop(&mut self) {
        FrontRoom {
            vertex: std::mem::take(&mut self.vertex.entries),
            fragment: std::mem::take(&mut self.fragment.entries),
            viewports: std::mem::take(&mut self.render.viewports),
            scissors: std::mem::take(&mut self.render.scissors),
            cull: std::mem::take(&mut self.render.cull),
            winding: std::mem::take(&mut self.render.winding),
            ends: std::mem::take(&mut self.ends),
            scratch: std::mem::take(&mut self.scratch),
        }
        .keep();
    }
}

/// What took a captured encoder back to the call (`RenderCommandEncoder::demote`), counted apart
/// (`WorkCounts::taken_back_plural_buffers` and kin). A draw the memo does not answer counts its
/// first difference in this order, of every difference its key has from those the memo knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TakenBack {
    /// `set_*_buffers`.
    PluralBuffers,
    /// `set_*_textures`.
    PluralTextures,
    /// `set_*_sampler_states`.
    PluralSamplers,
    /// `set_*_buffer_offset`.
    BufferOffset,
    /// A deferred store action or its options (`set_*_store_action*`).
    StoreAction,
    /// Render state the memo does not key: depth bias, depth clip mode, triangle fill mode,
    /// visibility result mode, vertex amplification, tessellation factors.
    RenderState,
    /// Any other call the memo does not key: an indirect draw, patches, mesh or tile work,
    /// resource uses, fences, executions of indirect commands.
    OtherCall,
    /// `set_render_pipeline_state` of a state whose draws the memo does not key.
    PipelineUnkeyed,
    /// `set_render_pipeline_state` of a keyed state not answered `Ok` in the pass before.
    PipelineUnseen,
    /// A draw the memo keys none of: no keyed pipeline set, or a vertex buffer bound with more
    /// than 4 GiB past its offset.
    DrawUnkeyed,
    /// An indexed draw that reads its indices through a copy.
    IndexCopy,
    /// A draw whose key the memo saw at the call not settled: its own pipeline being made, a
    /// residual form, or a notice heard.
    DrawUnsettled,
    /// A draw whose key names an object (a buffer, texture, sampler or depth/stencil state) no key
    /// names.
    DrawNewObject,
    /// A draw whose key binds an object, or bytes, at a slot of a stage where no key binds it.
    DrawNewBinding,
    /// A draw whose key binds a buffer where keys bind it, at an offset class (or past its end, or
    /// far) no key binds it at there.
    DrawNewOffsetClass,
    /// A draw whose key binds bytes where keys bind bytes, of a length no key binds there.
    DrawNewBytesLength,
    /// A draw whose bindings keys bind, in render state, a pipeline, a pass or a kind of draw no
    /// key draws them in.
    DrawNewState,
    /// `end_encoding` of draws the memo has not seen end.
    EndUnseen,
}

impl TakenBack {
    /// How many there are.
    pub(crate) const COUNT: usize = 18;
}

/// Why a render encoder of a device with a resolver opened at the call, not captured
/// (`CommandBuffer::followed_render`), counted apart (`WorkCounts::render_opens_unkeyed_descriptors`
/// and kin).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpenedAtTheCall {
    /// A descriptor the memo keys no pass of: a rate map, custom sample positions or layers.
    UnkeyedDescriptor,
    /// Its command buffer's earlier encoders made a shadowed texture resident.
    InheritedShadows,
    /// Its open announced.
    Announced,
    /// A pass the memo keys none of: color groups, extended-range colors, an attachment in a
    /// buffer's or a heap's bytes, or one over the caller's image.
    UnkeyedPass,
    /// A pass the memo keys but had not seen: its encoder teaches it.
    UnseenPass,
}

impl OpenedAtTheCall {
    /// How many there are.
    pub(crate) const COUNT: usize = 5;
}

/// What the memo knows of a pass whose open was answered `Ok`.
pub(crate) struct PassFacts {
    /// Its number among the memo's passes: never another's.
    id: u64,
    /// Its tile size, as `tile_width` and `tile_height` answer it.
    tile: Result<[u32; 2], Error>,
}

/// The device's answers of earlier calls (the module docs), each keyed by exactly what decides
/// it. Cleared whole once it holds [`MEMO_ENTRIES`] keys.
#[derive(Default)]
pub(crate) struct AnswerMemo {
    passes: HashMap<Box<[u64]>, Arc<PassFacts>, WordHash>,
    /// A pipeline state, by its address, set in a pass.
    pipelines: HashSet<(usize, u64), WordHash>,
    /// A draw's key, and its entry's number.
    draws: HashMap<Box<[u64]>, u64, WordHash>,
    /// An encoder's end's key.
    ends: HashSet<Box<[u64]>, WordHash>,
    /// The allocations of the objects whose addresses its keys name, held weakly: the objects go
    /// as the program lets go of them, and no other object takes their addresses while the keys
    /// stand.
    held: Vec<Weak<dyn Any + Send + Sync>>,
    held_addresses: HashSet<usize, WordHash>,
    /// What its draws' keys bind, to tell why a draw it does not answer missed ([`TakenBack`],
    /// `Front::missed`): the objects they name ([`Front::objects`]); each stage's slots with what
    /// they bind there ([`binding`]); and those with their offset classes or lengths
    /// ([`Front::entries`]).
    named: HashSet<[u64; 2], WordHash>,
    bindings: HashSet<[u64; 2], WordHash>,
    bound: HashSet<[u64; 3], WordHash>,
    /// The keys of draws seen not settled at the call (`device::unsettled`), not learned.
    unsettled: HashSet<Box<[u64]>, WordHash>,
    /// The last number given to a pass or a draw entry: never given again, clears included.
    next: u64,
}

/// The most keys the memo holds before it is cleared.
const MEMO_ENTRIES: usize = 16 * 1024;

impl AnswerMemo {
    /// Make room for a key: forget everything once full.
    fn room(&mut self) {
        let entries = self.passes.len()
            + self.pipelines.len()
            + self.draws.len()
            + self.ends.len()
            + self.unsettled.len();
        if entries >= MEMO_ENTRIES {
            self.forget();
        }
    }

    /// Forget every key: their calls are encoded at the call again until they teach it anew.
    /// An encoder that holds a forgotten number misses at its next call and is taken back.
    pub(crate) fn forget(&mut self) {
        self.passes.clear();
        self.pipelines.clear();
        self.draws.clear();
        self.ends.clear();
        self.held.clear();
        self.held_addresses.clear();
        self.named.clear();
        self.bindings.clear();
        self.bound.clear();
        self.unsettled.clear();
    }

    fn number(&mut self) -> u64 {
        self.next += 1;
        self.next
    }

    /// Hold `object`'s allocation, whose address a key names.
    fn hold(&mut self, object: &Arc<dyn Any + Send + Sync>) {
        if self.held_addresses.insert(address(object)) {
            self.held.push(Arc::downgrade(object));
        }
    }
}

/// An object's address, as a key names it.
fn address(object: &Arc<dyn Any + Send + Sync>) -> usize {
    Arc::as_ptr(object).cast::<()>() as usize
}

/// A [`BuildHasher`] for keys of words: one multiply per word.
#[derive(Clone, Copy, Default)]
pub(crate) struct WordHash;

/// [`WordHash`]'s hasher.
pub(crate) struct WordHasher(u64);

impl BuildHasher for WordHash {
    type Hasher = WordHasher;
    fn build_hasher(&self) -> WordHasher {
        WordHasher(0)
    }
}

impl WordHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(26) ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

impl Hasher for WordHasher {
    fn finish(&self) -> u64 {
        let h = self.0 ^ (self.0 >> 33);
        let h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^ (h >> 33)
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }
    fn write_u64(&mut self, word: u64) {
        self.add(word);
    }
    fn write_usize(&mut self, word: usize) {
        self.add(word as u64);
    }
}

/// A [`Hasher`] that keeps every word written to it: the exact words of a value's `Hash`, which
/// tell values of one type apart.
struct Words<'a>(&'a mut Vec<u64>);

impl Hasher for Words<'_> {
    fn finish(&self) -> u64 {
        0
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.push(bytes.len() as u64);
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0.push(u64::from_le_bytes(word));
        }
    }
    fn write_u8(&mut self, value: u8) {
        self.0.push(u64::from(value));
    }
    fn write_u32(&mut self, value: u32) {
        self.0.push(u64::from(value));
    }
    fn write_u64(&mut self, value: u64) {
        self.0.push(value);
    }
    fn write_usize(&mut self, value: usize) {
        self.0.push(value as u64);
    }
    fn write_isize(&mut self, value: isize) {
        self.0.push(value as u64);
    }
}

/// The words of `value`'s `Hash` appended to `into`.
fn words_of(value: &impl Hash, into: &mut Vec<u64>) {
    value.hash(&mut Words(into));
}

/// A texture's identity in a key: its number, never another's (`ObjectId`); 0 for none.
/// The textures `descriptor` attaches, their resolve textures included.
fn attachments(descriptor: &RenderPassDescriptor) -> impl Iterator<Item = &Texture> {
    let depth = &descriptor.depth_attachment;
    let stencil = &descriptor.stencil_attachment;
    (descriptor.color_attachments.iter())
        .flat_map(|color| [color.texture.as_ref(), color.resolve_texture.as_ref()])
        .chain([depth.texture.as_ref(), depth.resolve_texture.as_ref()])
        .chain([stencil.texture.as_ref(), stencil.resolve_texture.as_ref()])
        .flatten()
}

fn object(texture: Option<&Texture>) -> u64 {
    texture.map_or(0, |texture| texture.object_id().get())
}

/// Whether the memo keys a pass of `descriptor` (the module docs), its descriptor's part.
fn keyed_descriptor(descriptor: &RenderPassDescriptor) -> bool {
    descriptor.rasterization_rate_map.is_none()
        && descriptor.sample_positions.is_empty()
        && descriptor.render_target_array_length <= 1
}

/// The exact words of everything `descriptor` says.
fn pass_words(descriptor: &RenderPassDescriptor, words: &mut Vec<u64>) {
    words.clear();
    let w = words;
    for color in &descriptor.color_attachments {
        w.extend_from_slice(&[object(color.texture.as_ref()), color.level, color.slice]);
        w.extend_from_slice(&[color.depth_plane, object(color.resolve_texture.as_ref())]);
        w.extend_from_slice(&[color.resolve_level, color.resolve_slice]);
        w.push(color.resolve_depth_plane);
        words_of(&color.load_action, w);
        words_of(&color.store_action, w);
        words_of(&color.store_action_options, w);
        let clear = color.clear_color;
        for value in [clear.red, clear.green, clear.blue, clear.alpha] {
            w.push(value.to_bits());
        }
    }
    let depth = &descriptor.depth_attachment;
    w.extend_from_slice(&[object(depth.texture.as_ref()), depth.level, depth.slice]);
    w.extend_from_slice(&[depth.depth_plane, object(depth.resolve_texture.as_ref())]);
    w.extend_from_slice(&[depth.resolve_level, depth.resolve_slice]);
    w.push(depth.resolve_depth_plane);
    words_of(&depth.load_action, w);
    words_of(&depth.store_action, w);
    words_of(&depth.store_action_options, w);
    w.push(depth.clear_depth.to_bits());
    words_of(&depth.depth_resolve_filter, w);
    // Whether a write switched the depth on (`texture/signed_depth.rs`): from then on a pass
    // that loads it renders over its magnitude stand-in and announces what that does not keep.
    // Set once, at a call that drains the resolver first or at a pass's open, which a pass of
    // this key opened at the call, so the resolver reads what the call read.
    w.push(u64::from(
        depth.texture.as_ref().is_some_and(Texture::switched_on),
    ));
    let stencil = &descriptor.stencil_attachment;
    w.extend_from_slice(&[
        object(stencil.texture.as_ref()),
        stencil.level,
        stencil.slice,
    ]);
    w.extend_from_slice(&[
        stencil.depth_plane,
        object(stencil.resolve_texture.as_ref()),
    ]);
    w.extend_from_slice(&[stencil.resolve_level, stencil.resolve_slice]);
    w.push(stencil.resolve_depth_plane);
    words_of(&stencil.load_action, w);
    words_of(&stencil.store_action, w);
    words_of(&stencil.store_action_options, w);
    w.push(u64::from(stencil.clear_stencil));
    words_of(&stencil.stencil_resolve_filter, w);
    let visibility = descriptor.visibility_result_buffer.as_ref();
    w.push(visibility.map_or(0, |buffer| buffer.object_id().get()));
    w.extend_from_slice(&[
        descriptor.render_target_array_length,
        descriptor.render_target_width,
        descriptor.render_target_height,
        descriptor.default_raster_sample_count,
        descriptor.tile_width,
        descriptor.tile_height,
        descriptor.imageblock_sample_length,
        descriptor.threadgroup_memory_length,
    ]);
}

/// A slot of a stage table in a key: its buffer index (buffers and bytes share it), a texture's
/// index past [`TEXTURE_SLOT`], a sampler's past [`SAMPLER_SLOT`].
const TEXTURE_SLOT: u64 = 1 << 8;
const SAMPLER_SLOT: u64 = 1 << 16;
/// What a buffer slot holds, in its entry's first word's top bits.
const BUFFER: u64 = 1 << 62;
const BYTES: u64 = 2 << 62;
/// A buffer bound with more than 4 GiB past its offset (`render/vertex_reach.rs`).
const FAR: u64 = 1 << 61;
/// A buffer bound at its length, nothing past its offset.
const AT_END: u64 = 1 << 60;

/// The largest alignment a bind offset's class tells apart: Vulkan's bound on
/// `minUniformBufferOffsetAlignment`, `minStorageBufferOffsetAlignment` and
/// `minTexelBufferOffsetAlignment`, and so on the view base alignment a binding plan classes
/// offsets by (the module docs).
const OFFSET_CLASSES: u64 = 256;

/// The class of bind offset `offset` in a key (the module docs): the exponent of the largest
/// power of two up to [`OFFSET_CLASSES`] that divides it. Every alignment predicate of a power of
/// two up to that is a function of it.
pub(crate) fn offset_class(offset: u64) -> u64 {
    u64::from((offset | OFFSET_CLASSES).trailing_zeros())
}

/// The object a stage table's entry (tagged, `Front::entries`) names, with its kind: a buffer
/// (0), a texture (1) or a sampler (2); none for bytes.
fn entry_object([slot, a, _]: [u64; 3]) -> Option<[u64; 2]> {
    match slot & u64::from(u32::MAX) {
        slot if slot < TEXTURE_SLOT => {
            (a & BUFFER != 0).then_some([0, a & !(BUFFER | FAR | AT_END)])
        }
        slot if slot < SAMPLER_SLOT => Some([1, a]),
        _ => Some([2, a]),
    }
}

/// A stage table's entry (tagged, `Front::entries`) less its offset class or length: the slot
/// and what it binds there.
fn binding([slot, a, _]: [u64; 3]) -> [u64; 2] {
    [slot, a & !(FAR | AT_END)]
}

/// A stage table as its calls set it: per slot, sorted by slot, what it holds.
#[derive(Default)]
struct TableKey {
    entries: Vec<[u64; 3]>,
}

impl TableKey {
    fn set(&mut self, slot: u64, entry: Option<[u64; 2]>) {
        let at = self.entries.binary_search_by_key(&slot, |entry| entry[0]);
        match (at, entry) {
            (Ok(at), Some([a, b])) => self.entries[at] = [slot, a, b],
            (Ok(at), None) => {
                self.entries.remove(at);
            }
            (Err(at), Some([a, b])) => self.entries.insert(at, [slot, a, b]),
            (Err(_), None) => {}
        }
    }

    fn far(&self) -> bool {
        self.entries.iter().any(|entry| entry[1] & FAR != 0)
    }

    fn words(&self, into: &mut Vec<u64>) {
        into.push(self.entries.len() as u64);
        for entry in &self.entries {
            into.extend_from_slice(entry);
        }
    }
}

/// The encoder's render state as its calls set it, as words.
#[derive(Default)]
struct RenderKey {
    viewports: Vec<u64>,
    scissors: Vec<u64>,
    /// The cull mode's and the winding's words, where set.
    cull: Vec<u64>,
    winding: Vec<u64>,
    depth_stencil: u64,
    references: [u32; 2],
    blend: [u32; 4],
}

impl RenderKey {
    fn words(&self, into: &mut Vec<u64>) {
        for list in [&self.viewports, &self.scissors, &self.cull, &self.winding] {
            into.push(list.len() as u64);
            into.extend_from_slice(list);
        }
        into.push(self.depth_stencil);
        into.push(u64::from(self.references[0]) | (u64::from(self.references[1]) << 32));
        into.extend(self.blend.map(u64::from));
    }
}

/// The address of `state` as the memo keys it, where the memo keys its draws (the module docs).
fn keyed_pipeline(state: &RenderPipelineState) -> Option<usize> {
    let RenderStateKind::Graphics(drawing) = state.kind() else {
        return None;
    };
    let pipeline = drawing.base();
    let graphics = &pipeline.graphics;
    let interface = &pipeline.interface;
    let plain_slots = interface.slots.values().all(|slot| match slot {
        SlotInterface::Buffer(buffer) => matches!(buffer.kind, BufferKind::Buffer),
        SlotInterface::Texture(_) | SlotInterface::Sampler(_) => true,
        // A slot of a kind added later is one the memo does not know.
        _ => false,
    });
    let plain = drawing.descriptor().vertex().is_some()
        && graphics.emulated_tessellation.is_none()
        && graphics.mesh_records.is_none()
        && !drawing.reads_tile_memory()
        && graphics.fragment_imageblock.is_none()
        && graphics.input_attachments.is_empty()
        && graphics.storage_attachments.is_empty()
        && graphics.shadow_stencil.is_none()
        && !graphics.snapshot_reads
        && !graphics.fragment_writes_memory
        && !graphics.vertex_writes_memory
        && graphics.max_vertex_amplification_count <= 1
        && !interface.needs_pointer_translation
        && !interface.needs_handle_translation
        && !interface.reaches_command_buffers
        && plain_slots;
    plain.then(|| drawing.address())
}

/// A draw's kind in a key.
#[derive(Clone, Copy)]
pub(super) enum DrawKind {
    Primitives(PrimitiveType),
    Indexed(PrimitiveType, IndexType),
}

impl DrawKind {
    fn words(self, empty: bool, into: &mut Vec<u64>) {
        into.push(u64::from(empty));
        match self {
            DrawKind::Primitives(primitive) => {
                into.push(1);
                words_of(&primitive, into);
            }
            DrawKind::Indexed(primitive, index) => {
                into.push(2);
                words_of(&primitive, into);
                words_of(&index, into);
            }
        }
    }
}

impl Front {
    fn new(guard: OpenGuard, pass: Arc<PassFacts>) -> Front {
        let room = FrontRoom::take();
        Front {
            guard,
            ends: room.ends,
            pass,
            pipeline: None,
            vertex: TableKey {
                entries: room.vertex,
            },
            fragment: TableKey {
                entries: room.fragment,
            },
            render: RenderKey {
                viewports: room.viewports,
                scissors: room.scissors,
                cull: room.cull,
                winding: room.winding,
                ..RenderKey::default()
            },
            keyed: true,
            depth_writes: false,
            scratch: room.scratch,
            answered: 0,
        }
    }

    /// Count the calls answered at the call so far.
    fn count_answered(&mut self) {
        if self.answered > 0 {
            self.device().shared().counts.calls_answered(self.answered);
            self.answered = 0;
        }
    }

    fn device(&self) -> &Device {
        self.guard.command_buffer.device()
    }

    fn check_device(&self, call: &'static str, device: &Device) -> Result<(), Error> {
        match self.device().same(device) {
            true => Ok(()),
            false => Err(invalid(call, Problem::ForeignObject)),
        }
    }

    fn table(&mut self, table: Table) -> &mut TableKey {
        match table {
            Table::Fragment => &mut self.fragment,
            _ => &mut self.vertex,
        }
    }

    /// `set_*_buffer` (`ArgumentTable::set_buffer`): checked, then set.
    fn buffer(
        &mut self,
        (call, table): (&'static str, Table),
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        crate::argument_table::check_index(call, index, crate::argument_table::BUFFER_SLOTS)?;
        let entry = match buffer {
            Some(buffer) => {
                self.check_device(call, buffer.device())?;
                let length = buffer.length();
                if offset > length {
                    return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
                }
                let far = match length - offset > REACHED {
                    true => FAR,
                    false => 0,
                };
                let end = match offset == length {
                    true => AT_END,
                    false => 0,
                };
                let id = buffer.object_id().get();
                Some([BUFFER | far | end | id, offset_class(offset)])
            }
            None => None,
        };
        self.table(table).set(u64::from(index), entry);
        Ok(())
    }

    /// `set_*_bytes` (`ArgumentTable::set_bytes`): checked, then set.
    fn bytes(
        &mut self,
        (call, table): (&'static str, Table),
        bytes: &[u8],
        index: u32,
    ) -> Result<(), Error> {
        crate::argument_table::check_index(call, index, crate::argument_table::BUFFER_SLOTS)?;
        let length = bytes.len() as u64;
        let limit = crate::argument_table::INLINE_BYTES_LIMIT;
        if length > limit {
            return Err(invalid(call, Problem::InlineBytesLength { length, limit }));
        }
        self.table(table)
            .set(u64::from(index), Some([BYTES, length]));
        Ok(())
    }

    /// `set_*_texture` (`ArgumentTable::set_textures`): checked, then set.
    fn texture(
        &mut self,
        (call, table): (&'static str, Table),
        texture: Option<&Texture>,
        index: u32,
    ) -> Result<(), Error> {
        crate::argument_table::check_range(call, index, 1, crate::argument_table::TEXTURE_SLOTS)?;
        if let Some(texture) = texture {
            self.check_device(call, texture.device())?;
            texture.check_not_memoryless(call)?;
        }
        let entry = texture.map(|texture| [texture.object_id().get(), 0]);
        self.table(table)
            .set(TEXTURE_SLOT + u64::from(index), entry);
        Ok(())
    }

    /// `set_*_sampler_state` (`ArgumentTable::set_samplers`): checked, then set.
    fn sampler(
        &mut self,
        (call, table): (&'static str, Table),
        sampler: Option<&SamplerState>,
        index: u32,
    ) -> Result<(), Error> {
        crate::argument_table::check_range(call, index, 1, crate::argument_table::SAMPLER_SLOTS)?;
        if let Some(sampler) = sampler {
            self.check_device(call, sampler.device())?;
        }
        let entry = sampler.map(|sampler| [sampler.address() as u64, 0]);
        self.table(table)
            .set(SAMPLER_SLOT + u64::from(index), entry);
        Ok(())
    }

    /// `set_viewport(s)` (`EncoderState::set_viewports`): checked, then set.
    fn viewports(&mut self, call: &'static str, viewports: &[Viewport]) -> Result<(), Error> {
        let refused = crate::render_state::depth_ranges_refused(self.device());
        let taken = crate::render_state::checked_viewports(call, viewports, refused)?;
        self.render.viewports.clear();
        for v in taken.iter() {
            let fields = [v.origin_x, v.origin_y, v.width, v.height, v.znear, v.zfar];
            self.render.viewports.extend(fields.map(f64::to_bits));
        }
        Ok(())
    }

    /// `set_scissor_rect(s)` (`EncoderState::set_scissors`): checked, then set.
    fn scissors(&mut self, call: &'static str, rects: &[ScissorRect]) -> Result<(), Error> {
        if rects.is_empty() {
            let problem = RenderStateProblem::Count { count: 0 };
            return Err(invalid(call, Problem::RenderState(problem)));
        }
        self.render.scissors.clear();
        for r in rects {
            self.render
                .scissors
                .extend_from_slice(&[r.x, r.y, r.width, r.height]);
        }
        Ok(())
    }

    /// `set_depth_stencil_state`: checked, then set.
    fn depth_stencil(&mut self, state: Option<&DepthStencilState>) -> Result<(), Error> {
        if let Some(state) = state {
            self.check_device("set_depth_stencil_state", state.device())?;
        }
        self.depth_writes |= state.is_some_and(|s| s.descriptor().depth_write_enabled);
        self.render.depth_stencil = state.map_or(0, |s| s.address() as u64);
        Ok(())
    }

    /// The memo key of the draw of `kind` at the call in `scratch`, where the memo keys it.
    fn draw_key(&mut self, kind: DrawKind, empty: bool) -> bool {
        let Some(pipeline) = self.pipeline else {
            return false;
        };
        if !self.keyed || self.vertex.far() {
            return false;
        }
        let words = &mut self.scratch;
        words.clear();
        words.extend_from_slice(&[pipeline as u64, self.pass.id]);
        kind.words(empty, words);
        self.render.words(words);
        self.vertex.words(words);
        self.fragment.words(words);
        true
    }

    /// Its stage tables' entries, each slot tagged with its stage.
    fn entries(&self) -> impl Iterator<Item = [u64; 3]> + '_ {
        let tables = [&self.vertex, &self.fragment].into_iter().zip(0u64..);
        tables.flat_map(|(table, stage)| {
            (table.entries.iter()).map(move |&[slot, a, b]| [slot | (stage << 32), a, b])
        })
    }

    /// The objects its stage tables and render state name, each with its kind: buffers and
    /// textures by identity, samplers and the depth/stencil state by address.
    fn objects(&self) -> impl Iterator<Item = [u64; 2]> + '_ {
        let depth_stencil = self.render.depth_stencil;
        let depth_stencil = (depth_stencil != 0).then_some([3, depth_stencil]);
        self.entries().filter_map(entry_object).chain(depth_stencil)
    }

    /// Why the memo does not answer the draw whose key `scratch` holds ([`TakenBack`]): its first
    /// difference from what the memo's keys bind.
    fn missed(&self, memo: &AnswerMemo) -> TakenBack {
        if memo.unsettled.contains(&self.scratch[..]) {
            return TakenBack::DrawUnsettled;
        }
        let tables = self.entries().filter(|entry| !memo.bound.contains(entry));
        let tables = tables.map(|entry| match entry_object(entry) {
            Some(object) if !memo.named.contains(&object) => TakenBack::DrawNewObject,
            _ if !memo.bindings.contains(&binding(entry)) => TakenBack::DrawNewBinding,
            _ if entry[1] == BYTES => TakenBack::DrawNewBytesLength,
            _ => TakenBack::DrawNewOffsetClass,
        });
        let depth_stencil = self.render.depth_stencil;
        let depth_stencil = (depth_stencil != 0 && !memo.named.contains(&[3, depth_stencil]))
            .then_some(TakenBack::DrawNewObject);
        let first = tables.chain(depth_stencil).min_by_key(|why| *why as usize);
        first.unwrap_or(TakenBack::DrawNewState)
    }

    /// Note the draw of `kind`, not settled at the call, so a captured encoder's draw of its key
    /// counts why it missed ([`TakenBack::DrawUnsettled`]).
    fn unsettled_draw(&mut self, kind: DrawKind, empty: bool) {
        if !self.draw_key(kind, empty) {
            return;
        }
        let device = self.device().clone();
        let mut memo = lock(&device.shared().answers);
        if !memo.unsettled.contains(&self.scratch[..]) {
            memo.room();
            memo.unsettled.insert(self.scratch[..].into());
        }
    }

    /// Answer the draw of `kind` at the call from the memo: `Ok` where it is answered `Ok`, else
    /// why the memo does not answer it.
    fn answer_draw(&mut self, kind: DrawKind, empty: bool) -> Result<(), TakenBack> {
        if !self.draw_key(kind, empty) {
            return Err(TakenBack::DrawUnkeyed);
        }
        let shared = self.device().shared();
        let memo = lock(&shared.answers);
        let entry = match memo.draws.get(&self.scratch[..]) {
            Some(&entry) => entry,
            None if shared.answer_untaught => 0,
            None => return Err(self.missed(&memo)),
        };
        drop(memo);
        self.ends.push((entry << 1) | u64::from(empty));
        Ok(())
    }

    /// Learn the draw of `kind`, answered `Ok` at the call, its key naming `objects`.
    fn learn_draw(&mut self, kind: DrawKind, empty: bool, objects: &[Arc<dyn Any + Send + Sync>]) {
        if !self.draw_key(kind, empty) {
            self.keyed = false;
            return;
        }
        let device = self.device().clone();
        let mut memo = lock(&device.shared().answers);
        let entry = match memo.draws.get(&self.scratch[..]) {
            Some(&entry) => entry,
            None => {
                memo.room();
                objects.iter().for_each(|object| memo.hold(object));
                let entry = memo.number();
                memo.draws.insert(self.scratch[..].into(), entry);
                memo.named.extend(self.objects());
                memo.bindings.extend(self.entries().map(binding));
                memo.bound.extend(self.entries());
                entry
            }
        };
        drop(memo);
        self.ends.push((entry << 1) | u64::from(empty));
    }

    /// The end's key in `scratch`.
    fn end_key(&mut self) {
        let words = &mut self.scratch;
        words.clear();
        words.push(self.pass.id);
        words.push(self.ends.len() as u64);
        words.extend_from_slice(&self.ends);
        words.push(u64::from(self.depth_writes));
        words.push(self.pipeline.map_or(0, |pipeline| pipeline as u64));
        self.render.words(words);
        self.vertex.words(words);
        self.fragment.words(words);
    }

    /// Whether the memo answers the encoder's end `Ok`.
    fn answer_end(&mut self) -> bool {
        if !self.keyed {
            return false;
        }
        self.end_key();
        let shared = self.device().shared();
        shared.answer_untaught || lock(&shared.answers).ends.contains(&self.scratch[..])
    }

    /// Learn the encoder's end, answered `Ok` at the call, its key naming `objects`.
    fn learn_end(&mut self, objects: &[Arc<dyn Any + Send + Sync>]) {
        if !self.keyed {
            return;
        }
        self.end_key();
        let device = self.device().clone();
        let mut memo = lock(&device.shared().answers);
        if !memo.ends.contains(&self.scratch[..]) {
            memo.room();
            objects.iter().for_each(|object| memo.hold(object));
            memo.ends.insert(self.scratch[..].into());
        }
    }

    /// Whether the memo answers setting the state of `address` in the pass `Ok`.
    fn answer_pipeline(&self, address: usize) -> bool {
        let shared = self.device().shared();
        shared.answer_untaught
            || (lock(&shared.answers).pipelines).contains(&(address, self.pass.id))
    }

    /// Learn setting `state`, of `address`, answered `Ok` at the call.
    fn learn_pipeline(&self, state: &RenderPipelineState, address: usize) {
        let mut memo = lock(&self.device().shared().answers);
        if !memo.pipelines.contains(&(address, self.pass.id)) {
            memo.room();
            if let RenderStateKind::Graphics(drawing) = state.kind() {
                memo.hold(&drawing.held());
            }
            memo.pipelines.insert((address, self.pass.id));
        }
    }
}

impl CommandBuffer {
    /// `render_command_encoder`'s encoder where the device has a resolver: captured where the memo
    /// answers its open, else encoded at the call, after everything handed to the resolver, and
    /// followed by the memo where it keys the pass (the module docs).
    pub(super) fn captured_render(
        &self,
        descriptor: &RenderPassDescriptor,
    ) -> Result<RenderCommandEncoder, Error> {
        const CALL: &str = "render_command_encoder";
        let device = self.device().clone();
        let keyed = keyed_descriptor(descriptor);
        let known = keyed
            .then(|| {
                PASS_WORDS.with(|words| {
                    let mut words = words.borrow_mut();
                    pass_words(descriptor, &mut words);
                    lock(&device.shared().answers)
                        .passes
                        .get(&words[..])
                        .cloned()
                })
            })
            .flatten();
        let Some(pass) = known else {
            return self.followed_render(descriptor, keyed);
        };
        let mut state = self.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid(CALL, problem));
        }
        if state.inherits_shadows() {
            drop(state);
            return self.followed_render(descriptor, false);
        }
        state.encoder_open = true;
        state.captured_encoders += 1;
        drop(state);
        let shared = device.shared();
        shared.identifiers.encoder_opened();
        shared.counts.render_encoder_captured();
        let guard = OpenGuard {
            command_buffer: self.clone(),
            ended: false,
        };
        let mut front = Front::new(guard, pass);
        front.answered = 1;
        Ok(RenderCommandEncoder::of(Form::Captured(Box::new(
            Captured {
                front,
                slot: Arc::new(RenderSlot {
                    command_buffer: self.resolver_hold(),
                    encoding: Mutex::new(None),
                }),
                open: Some(Box::new(descriptor.clone())),
                calls: shared.scheduler.spare_logs().room(),
            },
        ))))
    }

    /// The encoder of `descriptor`, encoded at the call after everything handed to the resolver;
    /// followed by the memo where it keys its pass (`keyed`, as far as the descriptor tells) and
    /// the open announced nothing (the module docs).
    fn followed_render(
        &self,
        descriptor: &RenderPassDescriptor,
        keyed: bool,
    ) -> Result<RenderCommandEncoder, Error> {
        let device = self.device().clone();
        if let Some(resolver) = device.shared().resolver() {
            resolver.drain(crate::resolver::Drain::RenderOpen);
        }
        let inherits = self.state().inherits_shadows();
        let heard = crate::device::settled_mark();
        let encoding = self.open_render(descriptor, OpenAs::Program)?;
        self.state().at_call_work = true;
        let quiet = crate::device::settled_mark() == heard;
        let unkeyed = if inherits {
            Some(OpenedAtTheCall::InheritedShadows)
        } else if !keyed {
            Some(OpenedAtTheCall::UnkeyedDescriptor)
        } else if !quiet {
            Some(OpenedAtTheCall::Announced)
        } else if !encoding.keyed_pass() || attachments(descriptor).any(Texture::callers_image) {
            Some(OpenedAtTheCall::UnkeyedPass)
        } else {
            None
        };
        if let Some(why) = unkeyed {
            device.shared().counts.render_opened_at_the_call(why);
            return Ok(RenderCommandEncoder::of(Form::AtTheCall(Box::new(
                encoding,
            ))));
        }
        device
            .shared()
            .counts
            .render_opened_at_the_call(OpenedAtTheCall::UnseenPass);
        let tile = encoding.tile_size();
        let mut words = Vec::new();
        pass_words(descriptor, &mut words);
        let pass = {
            let mut memo = lock(&device.shared().answers);
            match memo.passes.get(&words[..]) {
                Some(pass) => Arc::clone(pass),
                None => {
                    memo.room();
                    let id = memo.number();
                    let pass = Arc::new(PassFacts { id, tile });
                    memo.passes.insert(words.into(), Arc::clone(&pass));
                    pass
                }
            }
        };
        // The encoding's core holds the command buffer.
        let guard = OpenGuard {
            command_buffer: self.clone(),
            ended: true,
        };
        Ok(RenderCommandEncoder::of(Form::Demoted(Box::new(Demoted {
            front: Front::new(guard, pass),
            encoding: Box::new(encoding),
        }))))
    }
}

/// The calls a piece logs before it is handed at a draw whether or not the resolver waits for
/// work ([`Captured::hand_at_draw`]).
const PIECE_CALLS: usize = 64;

impl Captured {
    /// Hand what is logged at a draw where the resolver encoded everything handed to it, so it
    /// waits for work, or where the piece logged [`PIECE_CALLS`] calls; else the piece goes on
    /// logging, handed at a later draw or at the end. So a resolver that keeps up encodes the pass
    /// while the program makes its next calls, and one that does not takes the pass in a few
    /// pieces.
    fn hand_at_draw(&mut self) {
        let waits = match self.front.device().shared().resolver() {
            Some(resolver) => resolver.waits(),
            None => return,
        };
        if waits || self.calls.len() >= PIECE_CALLS {
            self.hand(false);
        }
    }

    /// Hand the calls logged since the last piece (and the open, with the first), and the end
    /// where `end`.
    fn hand(&mut self, end: bool) {
        let device = self.front.device().clone();
        let Some(resolver) = device.shared().resolver() else {
            return;
        };
        if self.open.is_none() && self.calls.is_empty() && !end {
            return;
        }
        let spare = device.shared().scheduler.spare_logs().room();
        let calls = std::mem::replace(&mut self.calls, spare);
        let work = RenderWork {
            slot: Arc::clone(&self.slot),
            open: self.open.take(),
            calls,
            end,
        };
        resolver.hand(Work::Render(Box::new(work)), end);
    }

    /// Hand what is logged, wait for the resolver to encode everything handed to it, and take the
    /// encoding back: `None` where its open failed.
    fn take_back(&mut self) -> Option<Box<RenderEncoding>> {
        self.hand(false);
        let device = self.front.device().clone();
        if let Some(resolver) = device.shared().resolver() {
            resolver.drain(crate::resolver::Drain::TakeBack);
        }
        lock(&self.slot.encoding).take()
    }
}

impl RenderCommandEncoder {
    /// The encoder in `form`.
    pub(super) fn of(form: Form) -> RenderCommandEncoder {
        RenderCommandEncoder { form }
    }

    /// Encode the encoder at the call from now on (the module docs), taken back for `why`.
    fn demote(&mut self, why: TakenBack) {
        let Form::Captured(captured) = &mut self.form else {
            return;
        };
        let encoding = captured.take_back();
        let Form::Captured(captured) = std::mem::replace(&mut self.form, Form::Ended) else {
            return;
        };
        let Captured { mut front, .. } = *captured;
        front.count_answered();
        let command_buffer = &front.guard.command_buffer;
        command_buffer.state().at_call_work = true;
        command_buffer
            .device()
            .shared()
            .counts
            .render_encoder_demoted(why);
        self.form = match encoding {
            Some(encoding) => Form::Demoted(Box::new(Demoted { front, encoding })),
            None => Form::Failed(Box::new(front)),
        };
    }

    /// Run a call the memo does not key (the module docs) at the call.
    pub(crate) fn on<R: Answer>(&mut self, run: impl FnOnce(&mut RenderEncoding) -> R) -> R {
        self.on_as(TakenBack::OtherCall, run)
    }

    /// [`on`](Self::on), a call of the kind `why` counts.
    pub(crate) fn on_as<R: Answer>(
        &mut self,
        why: TakenBack,
        run: impl FnOnce(&mut RenderEncoding) -> R,
    ) -> R {
        self.demote(why);
        match &mut self.form {
            Form::AtTheCall(encoding) => run(encoding),
            Form::Demoted(demoted) => {
                demoted.front.keyed = false;
                run(&mut demoted.encoding)
            }
            Form::Captured(_) | Form::Failed(_) | Form::Ended => R::ok(),
        }
    }

    /// A call the memo keys that refuses by its arguments alone: `check` refuses at the call what
    /// the call refuses there and sets the front's state; `log` logs the call for the resolver;
    /// `run` runs it at the call.
    fn follow<R: Answer>(
        &mut self,
        check: impl FnOnce(&mut Front) -> Result<(), Error>,
        log: impl FnOnce(&mut RenderRoom),
        run: impl FnOnce(&mut RenderEncoding) -> R,
    ) -> R {
        match &mut self.form {
            Form::AtTheCall(encoding) => run(encoding),
            Form::Captured(captured) => match check(&mut captured.front) {
                Err(error) => R::refused(error),
                Ok(()) => {
                    log(&mut captured.calls);
                    captured.front.answered += 1;
                    R::ok()
                }
            },
            Form::Demoted(demoted) => {
                let heard = crate::device::settled_mark();
                let answer = run(&mut demoted.encoding);
                // A refused call may leave state the memo's keys do not name; one that announced
                // anything is not one a captured call may repeat (the module docs).
                if !answer.is_ok()
                    || crate::device::settled_mark() != heard
                    || check(&mut demoted.front).is_err()
                {
                    demoted.front.keyed = false;
                }
                answer
            }
            Form::Failed(_) | Form::Ended => R::ok(),
        }
    }

    /// `set_*_buffer` of `call`'s table.
    pub(super) fn follow_buffer(
        &mut self,
        call: (&'static str, Table),
        (buffer, offset, index): (Option<&Buffer>, u64, u32),
        run: impl FnOnce(&mut RenderEncoding) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.follow(
            |front| front.buffer(call, buffer, offset, index),
            |calls| {
                calls.push(|log| {
                    let buffer = buffer.map(|buffer| log.hold_buffer(buffer));
                    RenderCall::Buffer(call.1, buffer, offset, index)
                });
            },
            run,
        )
    }

    /// `set_*_bytes` of `call`'s table.
    pub(super) fn follow_bytes(
        &mut self,
        call: (&'static str, Table),
        (bytes, index): (&[u8], u32),
        run: impl FnOnce(&mut RenderEncoding) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.follow(
            |front| front.bytes(call, bytes, index),
            |calls| calls.push(|log| RenderCall::Bytes(call.1, log.byte_span(bytes), index)),
            run,
        )
    }

    /// `set_*_texture` of `call`'s table.
    pub(super) fn follow_texture(
        &mut self,
        call: (&'static str, Table),
        (texture, index): (Option<&Texture>, u32),
        run: impl FnOnce(&mut RenderEncoding) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.follow(
            |front| front.texture(call, texture, index),
            |calls| {
                calls.push(|log| {
                    let texture = texture.map(|texture| log.hold_texture(texture));
                    RenderCall::Texture(call.1, texture, index)
                });
            },
            run,
        )
    }

    /// `set_*_sampler_state` of `call`'s table.
    pub(super) fn follow_sampler(
        &mut self,
        call: (&'static str, Table),
        (sampler, index): (Option<&SamplerState>, u32),
        run: impl FnOnce(&mut RenderEncoding) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.follow(
            |front| front.sampler(call, sampler, index),
            |calls| {
                calls.push(|log| {
                    let sampler = sampler.map(|sampler| log.hold_sampler(sampler));
                    RenderCall::Sampler(call.1, sampler, index)
                });
            },
            run,
        )
    }

    /// `set_viewport(s)`.
    pub(super) fn follow_viewports(
        &mut self,
        call: &'static str,
        viewports: &[Viewport],
        run: impl FnOnce(&mut RenderEncoding) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.follow(
            |front| front.viewports(call, viewports),
            |calls| {
                calls.push(|_| match (call, viewports) {
                    ("set_viewport", [one]) => RenderCall::Viewport(Box::new(*one)),
                    (_, all) => RenderCall::Viewports(all.into()),
                });
            },
            run,
        )
    }

    /// `set_scissor_rect(s)`.
    pub(super) fn follow_scissors(
        &mut self,
        call: &'static str,
        rects: &[ScissorRect],
        run: impl FnOnce(&mut RenderEncoding) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.follow(
            |front| front.scissors(call, rects),
            |calls| {
                calls.push(|_| match (call, rects) {
                    ("set_scissor_rect", [one]) => RenderCall::ScissorRect(*one),
                    (_, all) => RenderCall::ScissorRects(all.into()),
                });
            },
            run,
        )
    }

    /// `set_cull_mode`.
    pub(super) fn follow_cull_mode(&mut self, mode: CullMode) {
        self.follow(
            |front| {
                front.render.cull.clear();
                words_of(&mode, &mut front.render.cull);
                Ok(())
            },
            |calls| calls.push(|_| RenderCall::CullMode(mode)),
            |encoding| encoding.set_cull_mode(mode),
        );
    }

    /// `set_front_facing_winding`.
    pub(super) fn follow_winding(&mut self, winding: Winding) {
        self.follow(
            |front| {
                front.render.winding.clear();
                words_of(&winding, &mut front.render.winding);
                Ok(())
            },
            |calls| calls.push(|_| RenderCall::FrontFacingWinding(winding)),
            |encoding| encoding.set_front_facing_winding(winding),
        );
    }

    /// `set_depth_stencil_state`.
    pub(super) fn follow_depth_stencil(
        &mut self,
        state: Option<&DepthStencilState>,
    ) -> Result<(), Error> {
        self.follow(
            |front| front.depth_stencil(state),
            |calls| {
                calls.push(|log| {
                    RenderCall::DepthStencilState(state.map(|state| log.hold_depth_stencil(state)))
                });
            },
            |encoding| encoding.set_depth_stencil_state(state),
        )
    }

    /// `set_stencil_reference_value` (`one`) or `set_stencil_reference_values`.
    pub(super) fn follow_references(&mut self, [front_value, back_value]: [u32; 2], one: bool) {
        self.follow(
            |front| {
                front.render.references = [front_value, back_value];
                Ok(())
            },
            |calls| {
                calls.push(|_| match one {
                    true => RenderCall::StencilReferenceValue(front_value),
                    false => RenderCall::StencilReferenceValues(front_value, back_value),
                });
            },
            |encoding| match one {
                true => encoding.set_stencil_reference_value(front_value),
                false => encoding.set_stencil_reference_values(front_value, back_value),
            },
        );
    }

    /// `set_blend_color`.
    pub(super) fn follow_blend_color(&mut self, color: [f32; 4]) {
        self.follow(
            |front| {
                front.render.blend = color.map(f32::to_bits);
                Ok(())
            },
            |calls| calls.push(|_| RenderCall::BlendColor(color)),
            |encoding| {
                let [red, green, blue, alpha] = color;
                encoding.set_blend_color(red, green, blue, alpha);
            },
        );
    }

    /// `set_render_pipeline_state`: answered where the memo knows the state in the pass.
    pub(super) fn follow_pipeline(&mut self, state: &RenderPipelineState) -> Result<(), Error> {
        const CALL: &str = "set_render_pipeline_state";
        let keyed = keyed_pipeline(state);
        if let Form::Captured(captured) = &mut self.form {
            let front = &mut captured.front;
            front.check_device(CALL, state.device())?;
            let why = match keyed {
                None => TakenBack::PipelineUnkeyed,
                Some(address) if front.answer_pipeline(address) => {
                    front.pipeline = Some(address);
                    captured
                        .calls
                        .push(|log| RenderCall::PipelineState(log.hold_render_state(state)));
                    front.answered += 1;
                    return Ok(());
                }
                Some(_) => TakenBack::PipelineUnseen,
            };
            self.demote(why);
        }
        match &mut self.form {
            Form::AtTheCall(encoding) => encoding.set_render_pipeline_state(state),
            Form::Demoted(demoted) => {
                let heard = crate::device::settled_mark();
                let answer = demoted.encoding.set_render_pipeline_state(state);
                let quiet = crate::device::settled_mark() == heard;
                let front = &mut demoted.front;
                match (&answer, keyed) {
                    (Ok(()), Some(address)) if quiet => {
                        front.pipeline = Some(address);
                        if front.keyed {
                            front.learn_pipeline(state, address);
                        }
                    }
                    // A refused call may leave state the memo's keys do not name (`set_state`
                    // routes the state before it refuses it).
                    _ => front.keyed = false,
                }
                answer
            }
            Form::Captured(_) | Form::Failed(_) | Form::Ended => Ok(()),
        }
    }

    /// A draw of `kind`, `empty` where it draws nothing, of the index buffer `index_buffer` where
    /// indexed, with whether the draw reads its indices through a copy: answered where the memo
    /// knows its key and it reads them in place; `log` logs it, `run` runs it at the call.
    pub(super) fn follow_draw(
        &mut self,
        (kind, empty): (DrawKind, bool),
        index_buffer: Option<(&'static str, &Buffer, &dyn Fn() -> bool)>,
        log: impl FnOnce(&mut RenderRoom),
        run: impl FnOnce(&mut RenderEncoding) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if let Form::Captured(captured) = &mut self.form {
            let mut copied = false;
            if let Some((call, buffer, copies)) = index_buffer {
                captured.front.check_device(call, buffer.device())?;
                copied = copies();
            }
            let answer = match copied {
                true => Err(TakenBack::IndexCopy),
                false => captured.front.answer_draw(kind, empty),
            };
            match answer {
                Ok(()) => {
                    log(&mut captured.calls);
                    captured.front.answered += 1;
                    captured.hand_at_draw();
                    return Ok(());
                }
                Err(why) => self.demote(why),
            }
        }
        match &mut self.form {
            Form::AtTheCall(encoding) => run(encoding),
            Form::Demoted(demoted) => {
                let named = demoted.encoding.named_objects();
                let heard = crate::device::settled_mark();
                let answer = run(&mut demoted.encoding);
                let quiet = crate::device::settled_mark() == heard;
                match &answer {
                    Ok(()) if quiet && !demoted.encoding.binds_callers_image() => {
                        demoted.front.learn_draw(kind, empty, &named);
                    }
                    Ok(()) if !quiet => {
                        demoted.front.unsettled_draw(kind, empty);
                        demoted.front.keyed = false;
                    }
                    _ => demoted.front.keyed = false,
                }
                answer
            }
            Form::Captured(_) | Form::Failed(_) | Form::Ended => Ok(()),
        }
    }

    /// `end_encoding`.
    pub(super) fn follow_end(mut self) -> Result<(), Error> {
        if let Form::Captured(captured) = &mut self.form {
            if captured.front.answer_end() {
                captured.hand(true);
                captured.front.answered += 1;
                if let Form::Captured(captured) = std::mem::replace(&mut self.form, Form::Ended) {
                    let Captured { mut front, .. } = *captured;
                    front.count_answered();
                    front.guard.command_buffer.state().encoder_open = false;
                    front.guard.ended = true;
                }
                return Ok(());
            }
            self.demote(TakenBack::EndUnseen);
        }
        match std::mem::replace(&mut self.form, Form::Ended) {
            Form::AtTheCall(encoding) => encoding.end_encoding(),
            Form::Demoted(demoted) => {
                let Demoted {
                    mut front,
                    encoding,
                } = *demoted;
                let named = encoding.named_objects();
                let callers = encoding.binds_callers_image();
                let heard = crate::device::settled_mark();
                let answer = encoding.end_encoding();
                if answer.is_ok() {
                    if crate::device::settled_mark() == heard && !callers {
                        front.learn_end(&named);
                    }
                    if !front.guard.ended {
                        front.guard.command_buffer.state().encoder_open = false;
                        front.guard.ended = true;
                    }
                }
                answer
            }
            Form::Failed(mut front) => {
                front.guard.command_buffer.state().encoder_open = false;
                front.guard.ended = true;
                Ok(())
            }
            Form::Captured(_) | Form::Ended => Ok(()),
        }
    }

    /// The pass's tile size, as `tile_width` and `tile_height` answer it.
    pub(super) fn follow_tile(&self) -> Result<[u32; 2], Error> {
        match &self.form {
            Form::AtTheCall(encoding) => encoding.tile_size(),
            Form::Demoted(demoted) => demoted.encoding.tile_size(),
            Form::Captured(captured) => captured.front.pass.tile.clone(),
            Form::Failed(front) => front.pass.tile.clone(),
            Form::Ended => Ok([0, 0]),
        }
    }
}

impl Drop for RenderCommandEncoder {
    /// An encoder dropped before its end abandons its command buffer at the drop (its front's
    /// guard), its encoding taken back and let go of on this thread, as one encoded at the call is.
    fn drop(&mut self) {
        if let Form::Captured(captured) = &mut self.form {
            captured.front.count_answered();
            drop(captured.take_back());
        }
    }
}

impl std::fmt::Debug for RenderCommandEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.form {
            Form::AtTheCall(encoding) => encoding.fmt(f),
            Form::Demoted(demoted) => demoted.encoding.fmt(f),
            Form::Captured(_) | Form::Failed(_) | Form::Ended => f
                .debug_struct("RenderCommandEncoder")
                .finish_non_exhaustive(),
        }
    }
}

impl RenderEncoding {
    /// The objects the memo's keys of its state name by address, which the memo holds weakly
    /// with the keys (`AnswerMemo::hold`): its pipeline state, depth/stencil state and samplers.
    fn named_objects(&self) -> Vec<Arc<dyn Any + Send + Sync>> {
        let mut named = Vec::new();
        if let Some(RenderStateKind::Graphics(drawing)) = self.pipeline.as_ref().map(|p| p.kind()) {
            named.push(drawing.held());
        }
        if let Some(state) = self.state.depth_stencil() {
            named.push(state.held());
        }
        for table in [&self.vertex, &self.fragment] {
            named.extend(table.samplers.values().map(|(sampler, _)| sampler.held()));
        }
        named
    }

    /// Whether the memo keys the encoder's pass (the module docs): no color groups, rate map,
    /// layers or extended-range shadows, and no attachment in a buffer's or a heap's bytes
    /// (`render_target::attachment_bytes`), whose encoding reads which resources the encoder
    /// names share its bytes or its host pages, which bind offsets and rebinds change
    /// (`linear_attachment::choose`, `Device::degrade_attachment_reads`).
    pub(super) fn keyed_pass(&self) -> bool {
        self.grouped.is_none()
            && self.rate_map.is_none()
            && self.render_target_array_length <= 1
            && !self.extended.keeps_draws()
            && crate::render_target::attachment_bytes(&self.layout)
                .next()
                .is_none()
    }

    /// Whether a texture its tables bind lies over the caller's image (`texture/wrapped.rs`):
    /// its command buffer's commit announces the image's layout moves, so the memo keys no draw
    /// or end that binds one.
    fn binds_callers_image(&self) -> bool {
        [&self.vertex, &self.fragment]
            .iter()
            .any(|table| table.textures.values().any(Texture::callers_image))
    }
}
