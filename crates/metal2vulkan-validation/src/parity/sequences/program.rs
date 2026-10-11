//! A sequence as plain data: its resources and their first bytes, its queues, and its command
//! buffers in commit order, each a list of steps (an event wait or signal, or an encoder with its
//! fence waits and updates and its commands). The generator makes one, the model runs it on the
//! CPU, the runner encodes it on Metal or on the runtime, and the minimizer shrinks it.
//!
//! A buffer's bytes are its memory's: its own, host pages it wraps without a copy, or a window of
//! a placement heap at its offset, which other buffers of the heap may overlap (aliasing). A
//! texture's bytes are its slices', each [`TEXTURE_BYTES`] long; a view presents one slice of a
//! texture, in a pixel format of its own and with a swizzle of its own, over the same bytes.

use super::SEEDS;
use std::fmt;
use std::sync::OnceLock;

/// The length of every buffer.
pub(crate) const BUFFER_BYTES: u64 = 512;
/// The bytes of each record of a buffer's or a heap's bytes read back: the harness writes the
/// bytes of a record this long in hex, which a difference then names byte by byte.
pub(crate) const CHUNK_BYTES: u64 = 256;
/// The alignment of a buffer placed in a heap (Metal's), and the step between a heap's places.
pub(crate) const HEAP_STEP: u64 = 256;
/// The places of a heap's buffers: offsets `0`, [`HEAP_STEP`], ... below `HEAP_PLACES` steps. A
/// buffer is [`BUFFER_BYTES`] long, two steps, so buffers at neighbouring places overlap by one.
pub(crate) const HEAP_PLACES: u64 = 4;
/// The bytes of a heap its buffers can reach, from its start: the window the runner sets up
/// and reads back through a buffer of its own.
pub(crate) const HEAP_WINDOW: u64 = HEAP_STEP * (HEAP_PLACES - 1) + BUFFER_BYTES;
/// The most slices a texture has (a 2D array of two).
pub(crate) const MAX_SLICES: u32 = 2;
/// The width and height of every texture, an `RGBA8Unorm` 2D texture.
pub(crate) const TARGET: u32 = 8;
/// The bytes of one texel.
pub(crate) const TEXEL_BYTES: u64 = 4;
/// The bytes of one texture row, and of a row in a buffer a copy reads or writes.
pub(crate) const ROW_BYTES: u64 = TARGET as u64 * TEXEL_BYTES;
/// The bytes of a texture.
pub(crate) const TEXTURE_BYTES: u64 = ROW_BYTES * TARGET as u64;
/// The stride of an argument record (a draw's four words, a dispatch's three and a pad word)
/// and of a rectangle (four words).
pub(crate) const RECORD_BYTES: u64 = 16;
/// The most instances a draw draws: the rectangles and colours it may read.
pub(crate) const MAX_INSTANCES: u32 = 3;
/// The vertices of one rectangle, two triangles.
pub(crate) const RECT_VERTICES: u32 = 6;

/// The case name of `seed`'s sequence, `seed-0000`, which its records' names begin with. A seed
/// run is below [`SEEDS`].
pub(crate) fn case_name(seed: u64) -> &'static str {
    static NAMES: OnceLock<Vec<String>> = OnceLock::new();
    let names = NAMES.get_or_init(|| (0..SEEDS).map(|seed| format!("seed-{seed:04}")).collect());
    &names[usize::try_from(seed).expect("a seed below SEEDS")]
}

/// What a buffer's words hold, which bounds what may be written into it: any word (`Data`), a
/// coordinate from 0 to [`TARGET`] (`Rects`), or a field of an argument record (`DrawArguments`:
/// a vertex count of 0 or [`RECT_VERTICES`], an instance count up to [`MAX_INSTANCES`], a vertex
/// start and a base instance of 0; `DispatchArguments`: threadgroup counts of 0 or 1 and a pad
/// word). Copies keep a field in its place, and a fill writes zeros, so every argument a command
/// reads is one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Data,
    Rects,
    DrawArguments,
    DispatchArguments,
}

/// Where a buffer's bytes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// Its own (`newBufferWithLength:options:`).
    Own,
    /// A host page of the runner's, wrapped without a copy
    /// (`newBufferWithBytesNoCopy:length:options:deallocator:`, no deallocator): `Shared`.
    NoCopy,
    /// Heap `heap`'s bytes from `offset`, a multiple of [`HEAP_STEP`] below [`HEAP_PLACES`]
    /// steps (`newBufferWithLength:options:offset:`), with the heap's storage and tracking.
    Heap { heap: usize, offset: u64 },
}

/// A buffer: its role, whether it is `Private` (else `Shared`), whether Metal tracks its hazards,
/// where its bytes are, and its bytes before the sequence (none for a heap's buffer: they are the
/// heap's).
#[derive(Clone, Debug)]
pub(crate) struct BufferSpec {
    pub(crate) role: Role,
    pub(crate) private: bool,
    pub(crate) tracked: bool,
    pub(crate) place: Place,
    pub(crate) initial: Vec<u8>,
}

/// A placement heap of buffers (the runner's `HEAP_BYTES`): whether it is `Private` (else `Shared`), whether
/// Metal tracks its hazards (as one resource, Apple's "at the granularity of the heap"), and the
/// bytes of its window ([`HEAP_WINDOW`]) before the sequence.
#[derive(Clone, Debug)]
pub(crate) struct HeapSpec {
    pub(crate) private: bool,
    pub(crate) tracked: bool,
    pub(crate) initial: Vec<u8>,
}

/// A texture: a `Private` [`TARGET`]-square `RGBA8Unorm` 2D texture, or 2D array of
/// [`MAX_SLICES`] slices, that is a render target, read and written by kernels and viewed in other
/// formats; whether Metal tracks its hazards; whether it lies at offset 0 of a placement heap of
/// its own (the runner's `TEXTURE_HEAP_BYTES`); and its bytes before the sequence, slice after slice.
#[derive(Clone, Debug)]
pub(crate) struct TextureSpec {
    pub(crate) tracked: bool,
    pub(crate) slices: u32,
    pub(crate) heap: bool,
    pub(crate) initial: Vec<u8>,
}

/// The pixel format a texture is seen through: its own, or a view's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Rgba8Unorm,
    Bgra8Unorm,
}

impl Format {
    /// The bytes a texel of channels `rgba` (red first) is stored as.
    pub(crate) fn store(self, rgba: [u8; 4]) -> [u8; 4] {
        match self {
            Format::Rgba8Unorm => rgba,
            Format::Bgra8Unorm => [rgba[2], rgba[1], rgba[0], rgba[3]],
        }
    }

    /// The channels (red first) of a texel stored as `bytes`.
    pub(crate) fn load(self, bytes: [u8; 4]) -> [u8; 4] {
        // Swapping red and blue is its own inverse.
        self.store(bytes)
    }
}

/// The source of a channel a view's reads return (`MTLTextureSwizzle`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Channel {
    Zero,
    One,
    Red,
    Green,
    Blue,
    Alpha,
}

impl Channel {
    /// The channel's byte, of the texel's channels `rgba`.
    pub(crate) fn of(self, rgba: [u8; 4]) -> u8 {
        match self {
            Channel::Zero => 0,
            Channel::One => 0xFF,
            Channel::Red => rgba[0],
            Channel::Green => rgba[1],
            Channel::Blue => rgba[2],
            Channel::Alpha => rgba[3],
        }
    }
}

/// How a view is made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ViewForm {
    /// `newTextureViewWithPixelFormat:` of a texture of one slice.
    Whole,
    /// `newTextureViewWithPixelFormat:textureType:levels:slices:`, a 2D view of one slice.
    Slices,
    /// `newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:`, a 2D view of one
    /// slice whose reads take the swizzle.
    Swizzled([Channel; 4]),
}

/// A view of slice `slice` of texture `texture`, in `format`. Only kernels read a swizzled one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ViewSpec {
    pub(crate) texture: usize,
    pub(crate) slice: u32,
    pub(crate) format: Format,
    pub(crate) form: ViewForm,
}

impl ViewSpec {
    /// The swizzle of its reads, if it has one.
    pub(crate) fn swizzle(&self) -> Option<[Channel; 4]> {
        match self.form {
            ViewForm::Swizzled(channels) => Some(channels),
            _ => None,
        }
    }
}

/// What a command reaches a texture through: the texture itself or a view of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Surface {
    Texture(usize),
    View(usize),
}

/// The memory a sequence's bytes are in: a buffer's own (or its host page), a heap's, or a
/// texture's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Resource {
    Buffer(usize),
    Heap(usize),
    Texture(usize),
}

/// A Metal object a command names, whose hazards Metal tracks when it is tracked: a buffer of
/// its own, a heap (its buffers are tracked as the heap), a texture, or a view (taken as an
/// object of its own: the order relies on no tracking between a view and its texture).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Object {
    Buffer(usize),
    Heap(usize),
    Texture(usize),
    View(usize),
}

/// A texture's slice as a surface presents it: the texture, the slice, the format, the swizzle,
/// and the object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Seen {
    pub(crate) texture: usize,
    pub(crate) slice: u32,
    pub(crate) format: Format,
    pub(crate) swizzle: Option<[Channel; 4]>,
    pub(crate) object: Object,
}

/// A place in a buffer: the buffer, and the offset a command binds or reads it at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct At {
    pub(crate) buffer: usize,
    pub(crate) offset: u64,
}

/// A kernel of the fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kernel {
    /// `seq_copy`: `b`'s word becomes `a`'s.
    Copy,
    /// `seq_mix`: `a`'s word is mixed with `b`'s, in place.
    Mix,
}

impl Kernel {
    pub(crate) fn entry(self) -> &'static str {
        match self {
            Kernel::Copy => "seq_copy",
            Kernel::Mix => "seq_mix",
        }
    }
}

/// The fixture's kernel of [`Command::TexelRead`].
pub(crate) const TEXEL_READ: &str = "seq_texel_read";
/// The fixture's kernel of [`Command::TexelWrite`].
pub(crate) const TEXEL_WRITE: &str = "seq_texel_write";

/// One command of an encoder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Command {
    /// A blit: `fillBuffer:range:value:` over `start..end`.
    Fill {
        buffer: usize,
        start: u64,
        end: u64,
        value: u8,
    },
    /// A blit: `size` bytes copied from `from` to `to`.
    Copy { from: At, to: At, size: u64 },
    /// A blit: rows `rows.0..rows.1` of slice `slice` of a surface (whole rows) from a buffer's
    /// bytes at `from`, [`ROW_BYTES`] a row.
    ToTexture {
        from: At,
        to: Surface,
        slice: u32,
        rows: (u32, u32),
    },
    /// A blit: rows of a surface's slice into a buffer's bytes at `to`, [`ROW_BYTES`] a row.
    FromTexture {
        from: Surface,
        slice: u32,
        rows: (u32, u32),
        to: At,
    },
    /// A compute dispatch of one thread of `kernel`, buffer 0 at `a` and buffer 1 at `b`; with
    /// `indirect`, `dispatchThreadgroups` of one-thread groups from the arguments there.
    Dispatch {
        kernel: Kernel,
        a: At,
        b: At,
        indirect: Option<At>,
    },
    /// A compute dispatch of `seq_texel_read` over rows `0..rows` of a 2D surface: each texel,
    /// as the surface reads it, packed into a word at `to`, [`ROW_BYTES`] a row.
    TexelRead { from: Surface, rows: u32, to: At },
    /// A compute dispatch of `seq_texel_write` over rows `0..rows` of a 2D surface: each word at
    /// `from`, [`ROW_BYTES`] a row, written as a texel through the surface.
    TexelWrite { from: At, to: Surface, rows: u32 },
    /// A compute encoder's `memoryBarrierWithScope:` of buffers, and of textures with
    /// `textures` (in a concurrent encoder).
    Barrier { textures: bool },
    /// A draw of `seq_rect` and `seq_color`: `instances` rectangles from `rects` in the colours
    /// at `colors`; with `indirect`, `drawPrimitives:indirectBuffer:` from the arguments there.
    Draw {
        rects: At,
        colors: At,
        instances: u32,
        indirect: Option<At>,
    },
}

/// A render pass's load action of its one colour attachment (its store action is `Store`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Load {
    /// `Clear`, to a colour whose bytes these are.
    Clear([u8; 4]),
    Load,
}

/// What an encoder is. A render pass's target is slice `slice` of a surface (0 for a view).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Blit,
    Compute {
        concurrent: bool,
    },
    Render {
        target: Surface,
        slice: u32,
        load: Load,
    },
}

/// An encoder: its kind, the fences it waits for as it begins, the fences it updates as it
/// ends, and its commands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Encoder {
    pub(crate) kind: Kind,
    pub(crate) waits: Vec<usize>,
    pub(crate) updates: Vec<usize>,
    pub(crate) commands: Vec<Command>,
}

/// One step of a command buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Step {
    /// `encodeWaitForEvent:value:` of the event, for value 1.
    Wait(usize),
    /// `encodeSignalEvent:value:` of the event, value 1.
    Signal(usize),
    Encoder(Encoder),
}

/// A command buffer: its queue and its steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CommandBufferSpec {
    pub(crate) queue: usize,
    pub(crate) steps: Vec<Step>,
}

/// A sequence. Its command buffers are committed in order, each to its queue; every event is
/// signalled once, to 1, and waited for at that value.
#[derive(Clone, Debug)]
pub(crate) struct Program {
    pub(crate) heaps: Vec<HeapSpec>,
    pub(crate) buffers: Vec<BufferSpec>,
    pub(crate) textures: Vec<TextureSpec>,
    pub(crate) views: Vec<ViewSpec>,
    pub(crate) queues: usize,
    pub(crate) events: usize,
    pub(crate) fences: usize,
    pub(crate) command_buffers: Vec<CommandBufferSpec>,
}

impl Program {
    /// Every object the sequence's commands may name, buffers first.
    pub(crate) fn objects(&self) -> impl Iterator<Item = Object> {
        (0..self.buffers.len())
            .map(Object::Buffer)
            .chain((0..self.heaps.len()).map(Object::Heap))
            .chain((0..self.textures.len()).map(Object::Texture))
            .chain((0..self.views.len()).map(Object::View))
    }

    /// Whether Metal tracks `object`'s hazards: a view's are its texture's mode.
    pub(crate) fn tracked(&self, object: Object) -> bool {
        match object {
            Object::Buffer(index) => self.buffers[index].tracked,
            Object::Heap(index) => self.heaps[index].tracked,
            Object::Texture(index) => self.textures[index].tracked,
            Object::View(index) => self.textures[self.views[index].texture].tracked,
        }
    }

    /// The memory of `at` and the offset of its bytes there, with the object a command that
    /// binds the buffer names.
    pub(crate) fn memory(&self, at: At) -> (Resource, u64, Object) {
        match self.buffers[at.buffer].place {
            Place::Own | Place::NoCopy => (
                Resource::Buffer(at.buffer),
                at.offset,
                Object::Buffer(at.buffer),
            ),
            Place::Heap { heap, offset } => {
                (Resource::Heap(heap), offset + at.offset, Object::Heap(heap))
            }
        }
    }

    /// Whether buffers `one` and `other` are two buffers over bytes in common: heap buffers at
    /// overlapping places of one heap.
    pub(crate) fn share_memory(&self, one: usize, other: usize) -> bool {
        match (&self.buffers[one].place, &self.buffers[other].place) {
            (Place::Heap { heap: a, offset: x }, Place::Heap { heap: b, offset: y }) => {
                one != other && a == b && *x < *y + BUFFER_BYTES && *y < *x + BUFFER_BYTES
            }
            _ => false,
        }
    }

    /// What `surface` presents of slice `slice` (0 for a view, whose slice is its own).
    pub(crate) fn seen(&self, surface: Surface, slice: u32) -> Seen {
        match surface {
            Surface::Texture(texture) => Seen {
                texture,
                slice,
                format: Format::Rgba8Unorm,
                swizzle: None,
                object: Object::Texture(texture),
            },
            Surface::View(index) => {
                let view = &self.views[index];
                Seen {
                    texture: view.texture,
                    slice: view.slice,
                    format: view.format,
                    swizzle: view.swizzle(),
                    object: Object::View(index),
                }
            }
        }
    }

    /// The commands of the sequence, counted.
    pub(crate) fn command_count(&self) -> usize {
        self.encoders()
            .map(|(_, _, encoder)| encoder.commands.len())
            .sum()
    }

    /// Every encoder with its command buffer and step.
    pub(crate) fn encoders(&self) -> impl Iterator<Item = (usize, usize, &Encoder)> {
        self.command_buffers
            .iter()
            .enumerate()
            .flat_map(|(cb, spec)| {
                spec.steps
                    .iter()
                    .enumerate()
                    .filter_map(move |(step, s)| match s {
                        Step::Encoder(encoder) => Some((cb, step, encoder)),
                        _ => None,
                    })
            })
    }
}

impl fmt::Display for Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Surface::Texture(index) => write!(f, "texture{index}"),
            Surface::View(index) => write!(f, "view{index}"),
        }
    }
}

impl fmt::Display for At {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "buffer{}+{}", self.buffer, self.offset)
    }
}

impl fmt::Display for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let indirect = |at: &Option<At>| match at {
            Some(at) => format!(" indirect {at}"),
            None => String::new(),
        };
        match self {
            Command::Fill {
                buffer,
                start,
                end,
                value,
            } => write!(f, "fill buffer{buffer}[{start}..{end}] with {value:#04x}"),
            Command::Copy { from, to, size } => write!(f, "copy {size} bytes {from} to {to}"),
            Command::ToTexture {
                from,
                to,
                slice,
                rows,
            } => write!(
                f,
                "copy {from} to {to} slice {slice} rows {}..{}",
                rows.0, rows.1
            ),
            Command::FromTexture {
                from,
                slice,
                rows,
                to,
            } => {
                write!(
                    f,
                    "copy {from} slice {slice} rows {}..{} to {to}",
                    rows.0, rows.1
                )
            }
            Command::TexelRead { from, rows, to } => {
                write!(f, "dispatch {TEXEL_READ} of {from} rows 0..{rows} to {to}")
            }
            Command::TexelWrite { from, to, rows } => {
                write!(f, "dispatch {TEXEL_WRITE} of {from} to {to} rows 0..{rows}")
            }
            Command::Dispatch {
                kernel,
                a,
                b,
                indirect: at,
            } => write!(f, "dispatch {} a {a} b {b}{}", kernel.entry(), indirect(at)),
            Command::Barrier { textures: false } => write!(f, "memory barrier of buffers"),
            Command::Barrier { textures: true } => {
                write!(f, "memory barrier of buffers and textures")
            }
            Command::Draw {
                rects,
                colors,
                instances,
                indirect: at,
            } => match at {
                Some(_) => write!(f, "draw rects {rects} colors {colors}{}", indirect(at)),
                None => write!(f, "draw {instances} rects {rects} colors {colors}"),
            },
        }
    }
}

fn modes(private: bool, tracked: bool) -> String {
    format!(
        "{}, {}",
        if private { "Private" } else { "Shared" },
        if tracked { "tracked" } else { "untracked" }
    )
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, heap) in self.heaps.iter().enumerate() {
            writeln!(
                f,
                "heap{index}: placement, {}",
                modes(heap.private, heap.tracked)
            )?;
        }
        for (index, buffer) in self.buffers.iter().enumerate() {
            let place = match buffer.place {
                Place::Own => String::new(),
                Place::NoCopy => ", over a host page".to_owned(),
                Place::Heap { heap, offset } => format!(", heap{heap}+{offset}"),
            };
            writeln!(
                f,
                "buffer{index}: {:?}, {}{place}",
                buffer.role,
                modes(buffer.private, buffer.tracked)
            )?;
        }
        for (index, texture) in self.textures.iter().enumerate() {
            writeln!(
                f,
                "texture{index}: {} slice(s), {}{}",
                texture.slices,
                modes(true, texture.tracked),
                if texture.heap { ", in a heap" } else { "" }
            )?;
        }
        for (index, view) in self.views.iter().enumerate() {
            writeln!(
                f,
                "view{index}: texture{} slice {} as {:?}, {:?}",
                view.texture, view.slice, view.format, view.form
            )?;
        }
        for (cb, spec) in self.command_buffers.iter().enumerate() {
            writeln!(f, "cb{cb} on queue{}:", spec.queue)?;
            for (step, s) in spec.steps.iter().enumerate() {
                match s {
                    Step::Wait(event) => writeln!(f, "  s{step} wait event{event}")?,
                    Step::Signal(event) => writeln!(f, "  s{step} signal event{event}")?,
                    Step::Encoder(encoder) => {
                        let kind = match encoder.kind {
                            Kind::Blit => "blit".to_owned(),
                            Kind::Compute { concurrent } => {
                                if concurrent {
                                    "concurrent compute".to_owned()
                                } else {
                                    "serial compute".to_owned()
                                }
                            }
                            Kind::Render {
                                target,
                                slice,
                                load,
                            } => match load {
                                Load::Clear(color) => {
                                    format!("render {target} slice {slice} clear {color:02x?}")
                                }
                                Load::Load => format!("render {target} slice {slice} load"),
                            },
                        };
                        writeln!(
                            f,
                            "  s{step} {kind}, waits fences {:?}, updates fences {:?}",
                            encoder.waits, encoder.updates
                        )?;
                        for (index, command) in encoder.commands.iter().enumerate() {
                            writeln!(f, "    c{index} {command}")?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
