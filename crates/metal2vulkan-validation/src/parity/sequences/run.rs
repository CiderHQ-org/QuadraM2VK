//! The runner: a sequence made, encoded and run through [`SeqApi`], each call recorded, and
//! every memory read back at its end.
//!
//! 1. The kernels' and the draws' pipeline states, the queues, and the resources: each heap with
//!    a buffer over its window; each buffer of its own, over a host page of the runner's (its
//!    first bytes written there before it is made), or placed in its heap; each texture alone or
//!    at offset 0 of a heap of its own; each view. A `Shared` buffer or heap window is written by
//!    the host; a `Private` one and every texture slice are copied from a `Shared` staging buffer
//!    by a setup command buffer, committed and waited for first.
//! 2. The events and fences, then each command buffer in order: encoded, then committed.
//! 3. Each command buffer's completion, in commit order, once every one is committed.
//! 4. A readback command buffer copies each `Private` buffer or heap window and each texture
//!    slice into a `Shared` buffer of its own; then every memory's bytes are recorded
//!    ([`readback_records`]): the buffers' (a host page's read from the page itself), the heaps'
//!    windows, the textures' slices.
//!
//! A refused call ends the run there, as Metal's API validation ends its process.

use super::super::api::Observation;
use super::api::SeqApi;
use super::model::{readback_records, State};
use super::program::{
    Channel, Command, Format, Kernel, Kind, Place, Program, Step, Surface, BUFFER_BYTES,
    HEAP_WINDOW, RECT_VERTICES, ROW_BYTES, TARGET, TEXEL_READ, TEXEL_WRITE, TEXTURE_BYTES,
};
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode, TextureDescriptor,
    TextureSwizzleChannels,
};
use std::alloc::Layout;
use std::ptr::NonNull;

/// The size of a heap of buffers (Metal's 16-KiB granule).
const HEAP_BYTES: u64 = 16 << 10;
/// The size of the heap a texture placed in a heap has to itself, at offset 0: room for the
/// image of any device's answer for a two-slice 8x8 `RGBA8Unorm` texture.
const TEXTURE_HEAP_BYTES: u64 = 1 << 20;
/// The host pages a buffer over host memory wraps: one 16-KiB page, the buffer its first
/// [`BUFFER_BYTES`].
const HOST_PAGE: u64 = 16 << 10;
/// Make `$call` as `$op`; a refusal ends the run.
macro_rules! call {
    ($obs:expr, $op:expr, $call:expr) => {
        match $obs.call(&$op, || $call) {
            Some(value) => value,
            None => return Ok(()),
        }
    };
}

/// The host pages a sequence's buffers over host memory wrap: one [`HOST_PAGE`] each, zeroed and
/// aligned to it. The caller keeps them until the device that used them is gone.
pub(crate) struct HostPages {
    pages: Option<NonNull<u8>>,
    layout: Layout,
}

impl HostPages {
    /// The pages of `program`'s buffers over host memory.
    pub(crate) fn of(program: &Program) -> HostPages {
        let count = program
            .buffers
            .iter()
            .filter(|spec| spec.place == Place::NoCopy)
            .count();
        let page = usize::try_from(HOST_PAGE).expect("a page");
        let layout = Layout::from_size_align(page * count.max(1), page).expect("a layout");
        let pages = (count > 0).then(|| {
            // SAFETY: the layout's size is not zero.
            let pages = unsafe { std::alloc::alloc_zeroed(layout) };
            NonNull::new(pages).expect("host pages")
        });
        HostPages { pages, layout }
    }

    /// Page `index`.
    fn page(&self, index: usize) -> NonNull<u8> {
        let pages = self.pages.expect("a host page");
        assert!((index + 1) * HOST_PAGE as usize <= self.layout.size());
        // SAFETY: inside the allocation (above).
        unsafe { pages.add(index * HOST_PAGE as usize) }
    }

    /// Write `bytes` at the start of page `index`.
    fn write(&self, index: usize, bytes: &[u8]) {
        assert!(bytes.len() as u64 <= HOST_PAGE);
        // SAFETY: the page holds `bytes`; no command buffer uses it while the runner writes it.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.page(index).as_ptr(), bytes.len())
        };
    }

    /// The first `len` bytes of page `index`.
    fn read(&self, index: usize, len: u64) -> Vec<u8> {
        assert!(len <= HOST_PAGE);
        // SAFETY: the page holds `len` bytes; the command buffers that wrote it completed.
        unsafe { std::slice::from_raw_parts(self.page(index).as_ptr(), len as usize) }.to_vec()
    }
}

impl Drop for HostPages {
    fn drop(&mut self) {
        if let Some(pages) = self.pages {
            // SAFETY: allocated with this layout in `of`.
            unsafe { std::alloc::dealloc(pages.as_ptr(), self.layout) };
        }
    }
}

/// The pixel format of `format`.
pub(crate) fn pixel_format(format: Format) -> PixelFormat {
    match format {
        Format::Rgba8Unorm => PixelFormat::RGBA8Unorm,
        Format::Bgra8Unorm => PixelFormat::BGRA8Unorm,
    }
}

/// The swizzle of `channels`.
pub(crate) fn swizzle(channels: [Channel; 4]) -> TextureSwizzleChannels {
    let of = |channel| match channel {
        Channel::Zero => TextureSwizzle::Zero,
        Channel::One => TextureSwizzle::One,
        Channel::Red => TextureSwizzle::Red,
        Channel::Green => TextureSwizzle::Green,
        Channel::Blue => TextureSwizzle::Blue,
        Channel::Alpha => TextureSwizzle::Alpha,
    };
    TextureSwizzleChannels::new(
        of(channels[0]),
        of(channels[1]),
        of(channels[2]),
        of(channels[3]),
    )
}

fn options(private: bool, tracked: bool) -> ResourceOptions {
    ResourceOptions::new(
        if private {
            StorageMode::Private
        } else {
            StorageMode::Shared
        },
        CpuCacheMode::DefaultCache,
        if tracked {
            HazardTrackingMode::Tracked
        } else {
            HazardTrackingMode::Untracked
        },
    )
}

fn texture_descriptor(tracked: bool, slices: u32) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(
        PixelFormat::RGBA8Unorm,
        u64::from(TARGET),
        u64::from(TARGET),
        false,
    );
    if slices > 1 {
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = u64::from(slices);
    }
    descriptor.usage = TextureUsage::RENDER_TARGET
        | TextureUsage::SHADER_READ
        | TextureUsage::SHADER_WRITE
        | TextureUsage::PIXEL_FORMAT_VIEW;
    descriptor.resource_options = options(true, tracked);
    descriptor
}

/// The command buffers committed and not yet waited for, waited for when dropped: a run a
/// refusal ends leaves no work in flight on the memory the caller frees after it (the host
/// pages).
struct InFlight<'a, A: SeqApi> {
    api: &'a A,
    committed: Vec<A::CommandBuffer>,
}

impl<A: SeqApi> Drop for InFlight<'_, A> {
    fn drop(&mut self) {
        for cb in self.committed.drain(..) {
            // The run is over; only the wait matters.
            let _ = self.api.completion(cb);
        }
    }
}

/// What a setup copy fills: a buffer, or a texture's slice.
enum Staged {
    Buffer(usize),
    Heap(usize),
    Slice(usize, u32),
}

/// Run `program` on `api`, recording into `obs`, its buffers over host memory on `pages`
/// ([`HostPages::of`] the program). An error is the harness's own (a wait that failed), not a
/// record.
pub(crate) fn run<A: SeqApi>(
    api: &A,
    program: &Program,
    pages: &HostPages,
    obs: &mut Observation,
) -> Result<(), String> {
    let copy = call!(obs, "kernel seq_copy", api.kernel(Kernel::Copy.entry()));
    let mix = call!(obs, "kernel seq_mix", api.kernel(Kernel::Mix.entry()));
    let texel_read = call!(obs, format!("kernel {TEXEL_READ}"), api.kernel(TEXEL_READ));
    let texel_write = call!(
        obs,
        format!("kernel {TEXEL_WRITE}"),
        api.kernel(TEXEL_WRITE)
    );
    let draw_rgba = call!(
        obs,
        "draw state",
        api.draw_state(pixel_format(Format::Rgba8Unorm))
    );
    let draw_bgra = call!(
        obs,
        "draw state bgra",
        api.draw_state(pixel_format(Format::Bgra8Unorm))
    );
    let mut queues = Vec::new();
    for queue in 0..program.queues {
        queues.push(call!(obs, format!("queue{queue}"), api.queue()));
    }
    let shared = ResourceOptions::shared();
    let mut staged = Vec::new();
    let mut heaps = Vec::new();
    let mut windows = Vec::new();
    for (index, spec) in program.heaps.iter().enumerate() {
        let heap_options = options(spec.private, spec.tracked);
        let heap = call!(
            obs,
            format!("heap{index}"),
            api.heap(HEAP_BYTES, heap_options)
        );
        let window = call!(
            obs,
            format!("heap{index} window"),
            api.heap_buffer(&heap, HEAP_WINDOW, heap_options, 0)
        );
        if spec.private {
            let staging = call!(
                obs,
                format!("heap{index} staging"),
                api.buffer(HEAP_WINDOW, shared)
            );
            api.write(&staging, &spec.initial);
            staged.push((Staged::Heap(index), staging));
        } else {
            api.write(&window, &spec.initial);
        }
        heaps.push(heap);
        windows.push(window);
    }
    let mut buffers = Vec::new();
    let mut page_of = vec![None; program.buffers.len()];
    let mut next_page = 0;
    for (index, spec) in program.buffers.iter().enumerate() {
        let buffer_options = options(spec.private, spec.tracked);
        let name = format!("buffer{index}");
        let buffer = match spec.place {
            Place::Own => call!(obs, name, api.buffer(BUFFER_BYTES, buffer_options)),
            Place::NoCopy => {
                pages.write(next_page, &spec.initial);
                page_of[index] = Some(next_page);
                let page = pages.page(next_page);
                next_page += 1;
                // SAFETY: the page is the caller's, kept until the device is gone.
                call!(obs, name, unsafe {
                    api.no_copy_buffer(page, BUFFER_BYTES, buffer_options)
                })
            }
            Place::Heap { heap, offset } => call!(
                obs,
                name,
                api.heap_buffer(&heaps[heap], BUFFER_BYTES, buffer_options, offset)
            ),
        };
        if spec.place == Place::Own {
            if spec.private {
                let staging = call!(
                    obs,
                    format!("buffer{index} staging"),
                    api.buffer(BUFFER_BYTES, shared)
                );
                api.write(&staging, &spec.initial);
                staged.push((Staged::Buffer(index), staging));
            } else {
                api.write(&buffer, &spec.initial);
            }
        }
        buffers.push(buffer);
    }
    let mut textures = Vec::new();
    let mut texture_heaps = Vec::new();
    for (index, spec) in program.textures.iter().enumerate() {
        let descriptor = texture_descriptor(spec.tracked, spec.slices);
        let texture = if spec.heap {
            let heap = call!(
                obs,
                format!("texture{index} heap"),
                api.heap(TEXTURE_HEAP_BYTES, descriptor.resource_options)
            );
            let texture = call!(
                obs,
                format!("texture{index}"),
                api.heap_texture(&heap, &descriptor, 0)
            );
            texture_heaps.push(heap);
            texture
        } else {
            call!(obs, format!("texture{index}"), api.texture(&descriptor))
        };
        for slice in 0..spec.slices {
            let staging = call!(
                obs,
                format!("texture{index} slice{slice} staging"),
                api.buffer(TEXTURE_BYTES, shared)
            );
            let start = (u64::from(slice) * TEXTURE_BYTES) as usize;
            api.write(
                &staging,
                &spec.initial[start..start + TEXTURE_BYTES as usize],
            );
            staged.push((Staged::Slice(index, slice), staging));
        }
        textures.push(texture);
    }
    let mut views = Vec::new();
    for (index, view) in program.views.iter().enumerate() {
        views.push(call!(
            obs,
            format!("view{index}"),
            api.view(&textures[view.texture], view)
        ));
    }
    // The setup command buffer: each Private buffer's and heap's and each texture slice's first
    // bytes.
    let setup = call!(obs, "setup", api.command_buffer(&queues[0]));
    let mut blit = call!(obs, "setup blit", api.blit(&setup));
    for (what, staging) in &staged {
        match *what {
            Staged::Buffer(index) => call!(
                obs,
                format!("setup buffer{index}"),
                api.copy(&mut blit, (staging, 0), (&buffers[index], 0), BUFFER_BYTES)
            ),
            Staged::Heap(index) => call!(
                obs,
                format!("setup heap{index}"),
                api.copy(&mut blit, (staging, 0), (&windows[index], 0), HEAP_WINDOW)
            ),
            Staged::Slice(index, slice) => call!(
                obs,
                format!("setup texture{index} slice{slice}"),
                api.buffer_to_texture(
                    &mut blit,
                    (staging, 0),
                    (&textures[index], slice),
                    0..TARGET,
                    (TARGET, ROW_BYTES)
                )
            ),
        }
    }
    call!(obs, "setup end", api.end_blit(blit));
    call!(obs, "setup commit", api.commit(&setup));
    obs.completed("setup: completion", api.completion(setup)?);

    let mut events = Vec::new();
    for event in 0..program.events {
        events.push(call!(obs, format!("event{event}"), api.event()));
    }
    let mut fences = Vec::new();
    for fence in 0..program.fences {
        fences.push(call!(obs, format!("fence{fence}"), api.fence()));
    }
    let at = |at: super::program::At| (&buffers[at.buffer], at.offset);
    let surface = |surface: Surface| match surface {
        Surface::Texture(index) => &textures[index],
        Surface::View(index) => &views[index],
    };
    let draw_state = |target: Surface| match program.seen(target, 0).format {
        Format::Rgba8Unorm => &draw_rgba,
        Format::Bgra8Unorm => &draw_bgra,
    };
    let mut in_flight = InFlight {
        api,
        committed: Vec::new(),
    };
    for (index, spec) in program.command_buffers.iter().enumerate() {
        let cb = call!(
            obs,
            format!("cb{index}"),
            api.command_buffer(&queues[spec.queue])
        );
        for (step, s) in spec.steps.iter().enumerate() {
            let name = format!("cb{index} s{step}");
            match s {
                Step::Wait(event) => call!(
                    obs,
                    format!("{name} wait event{event}"),
                    api.wait(&cb, &events[*event], 1)
                ),
                Step::Signal(event) => call!(
                    obs,
                    format!("{name} signal event{event}"),
                    api.signal(&cb, &events[*event], 1)
                ),
                Step::Encoder(encoder) => match encoder.kind {
                    Kind::Blit => {
                        let mut blit = call!(obs, format!("{name} blit"), api.blit(&cb));
                        for &fence in &encoder.waits {
                            call!(
                                obs,
                                format!("{name} wait fence{fence}"),
                                api.blit_wait_fence(&mut blit, &fences[fence])
                            );
                        }
                        for (c, command) in encoder.commands.iter().enumerate() {
                            let op = format!("{name} c{c}");
                            match *command {
                                Command::Fill {
                                    buffer,
                                    start,
                                    end,
                                    value,
                                } => call!(
                                    obs,
                                    op,
                                    api.fill(&mut blit, &buffers[buffer], start..end, value)
                                ),
                                Command::Copy { from, to, size } => {
                                    call!(obs, op, api.copy(&mut blit, at(from), at(to), size))
                                }
                                Command::ToTexture {
                                    from,
                                    to,
                                    slice,
                                    rows,
                                } => call!(
                                    obs,
                                    op,
                                    api.buffer_to_texture(
                                        &mut blit,
                                        at(from),
                                        (surface(to), slice),
                                        rows.0..rows.1,
                                        (TARGET, ROW_BYTES)
                                    )
                                ),
                                Command::FromTexture {
                                    from,
                                    slice,
                                    rows,
                                    to,
                                } => call!(
                                    obs,
                                    op,
                                    api.texture_to_buffer(
                                        &mut blit,
                                        (surface(from), slice),
                                        rows.0..rows.1,
                                        (TARGET, ROW_BYTES),
                                        at(to)
                                    )
                                ),
                                ref other => {
                                    return Err(format!("{op}: {other} in a blit encoder"))
                                }
                            }
                        }
                        for &fence in &encoder.updates {
                            call!(
                                obs,
                                format!("{name} update fence{fence}"),
                                api.blit_update_fence(&mut blit, &fences[fence])
                            );
                        }
                        call!(obs, format!("{name} end"), api.end_blit(blit));
                    }
                    Kind::Compute { concurrent } => {
                        let mut compute =
                            call!(obs, format!("{name} compute"), api.compute(&cb, concurrent));
                        for &fence in &encoder.waits {
                            call!(
                                obs,
                                format!("{name} wait fence{fence}"),
                                api.compute_wait_fence(&mut compute, &fences[fence])
                            );
                        }
                        for (c, command) in encoder.commands.iter().enumerate() {
                            let op = format!("{name} c{c}");
                            match *command {
                                Command::Dispatch {
                                    kernel,
                                    a,
                                    b,
                                    indirect,
                                } => {
                                    let pipeline = match kernel {
                                        Kernel::Copy => &copy,
                                        Kernel::Mix => &mix,
                                    };
                                    call!(
                                        obs,
                                        op,
                                        api.dispatch(
                                            &mut compute,
                                            pipeline,
                                            at(a),
                                            at(b),
                                            indirect.map(at)
                                        )
                                    )
                                }
                                Command::TexelRead { from, rows, to } => call!(
                                    obs,
                                    op,
                                    api.texel_dispatch(
                                        &mut compute,
                                        &texel_read,
                                        surface(from),
                                        at(to),
                                        (TARGET, rows)
                                    )
                                ),
                                Command::TexelWrite { from, to, rows } => call!(
                                    obs,
                                    op,
                                    api.texel_dispatch(
                                        &mut compute,
                                        &texel_write,
                                        surface(to),
                                        at(from),
                                        (TARGET, rows)
                                    )
                                ),
                                Command::Barrier { textures } => {
                                    call!(obs, op, api.barrier(&mut compute, textures))
                                }
                                ref other => {
                                    return Err(format!("{op}: {other} in a compute encoder"))
                                }
                            }
                        }
                        for &fence in &encoder.updates {
                            call!(
                                obs,
                                format!("{name} update fence{fence}"),
                                api.compute_update_fence(&mut compute, &fences[fence])
                            );
                        }
                        call!(obs, format!("{name} end"), api.end_compute(compute));
                    }
                    Kind::Render {
                        target,
                        slice,
                        load,
                    } => {
                        let mut render = call!(
                            obs,
                            format!("{name} render"),
                            api.render(&cb, (surface(target), slice), load)
                        );
                        for &fence in &encoder.waits {
                            call!(
                                obs,
                                format!("{name} wait fence{fence}"),
                                api.render_wait_fence(&mut render, &fences[fence])
                            );
                        }
                        for (c, command) in encoder.commands.iter().enumerate() {
                            let op = format!("{name} c{c}");
                            match *command {
                                Command::Draw {
                                    rects,
                                    colors,
                                    instances,
                                    indirect,
                                } => call!(
                                    obs,
                                    op,
                                    api.draw(
                                        &mut render,
                                        draw_state(target),
                                        at(rects),
                                        at(colors),
                                        (RECT_VERTICES, instances),
                                        indirect.map(at)
                                    )
                                ),
                                ref other => {
                                    return Err(format!("{op}: {other} in a render encoder"))
                                }
                            }
                        }
                        for &fence in &encoder.updates {
                            call!(
                                obs,
                                format!("{name} update fence{fence}"),
                                api.render_update_fence(&mut render, &fences[fence])
                            );
                        }
                        call!(obs, format!("{name} end"), api.end_render(render));
                    }
                },
            }
        }
        call!(obs, format!("cb{index} commit"), api.commit(&cb));
        in_flight.committed.push(cb);
    }
    for (index, cb) in in_flight.committed.drain(..).enumerate() {
        obs.completed(&format!("cb{index}: completion"), api.completion(cb)?);
    }

    // The readback: each Private buffer and heap window, and each texture slice, into a Shared
    // buffer of its own.
    let readback = call!(obs, "readback", api.command_buffer(&queues[0]));
    let mut blit = call!(obs, "readback blit", api.blit(&readback));
    let mut copies = Vec::new();
    for (index, spec) in program.buffers.iter().enumerate() {
        if spec.place == Place::Own && spec.private {
            let into = call!(
                obs,
                format!("readback buffer{index}"),
                api.buffer(BUFFER_BYTES, shared)
            );
            call!(
                obs,
                format!("readback buffer{index} copy"),
                api.copy(&mut blit, (&buffers[index], 0), (&into, 0), BUFFER_BYTES)
            );
            copies.push(into);
        }
    }
    let mut heap_copies = Vec::new();
    for (index, spec) in program.heaps.iter().enumerate() {
        if spec.private {
            let into = call!(
                obs,
                format!("readback heap{index}"),
                api.buffer(HEAP_WINDOW, shared)
            );
            call!(
                obs,
                format!("readback heap{index} copy"),
                api.copy(&mut blit, (&windows[index], 0), (&into, 0), HEAP_WINDOW)
            );
            heap_copies.push(into);
        }
    }
    let mut texture_copies = Vec::new();
    for (index, texture) in textures.iter().enumerate() {
        for slice in 0..program.textures[index].slices {
            let into = call!(
                obs,
                format!("readback texture{index} slice{slice}"),
                api.buffer(TEXTURE_BYTES, shared)
            );
            call!(
                obs,
                format!("readback texture{index} slice{slice} copy"),
                api.texture_to_buffer(
                    &mut blit,
                    (texture, slice),
                    0..TARGET,
                    (TARGET, ROW_BYTES),
                    (&into, 0)
                )
            );
            texture_copies.push(into);
        }
    }
    call!(obs, "readback end", api.end_blit(blit));
    call!(obs, "readback commit", api.commit(&readback));
    obs.completed("readback: completion", api.completion(readback)?);
    let mut copies = copies.into_iter();
    let mut state = State {
        buffers: Vec::new(),
        heaps: Vec::new(),
        textures: Vec::new(),
    };
    for (index, spec) in program.buffers.iter().enumerate() {
        state.buffers.push(match spec.place {
            Place::Own if spec.private => api.read(&copies.next().expect("a readback copy")),
            Place::Own => api.read(&buffers[index]),
            Place::NoCopy => pages.read(page_of[index].expect("a page"), BUFFER_BYTES),
            Place::Heap { .. } => Vec::new(),
        });
    }
    let mut heap_copies = heap_copies.into_iter();
    for (index, spec) in program.heaps.iter().enumerate() {
        state.heaps.push(if spec.private {
            api.read(&heap_copies.next().expect("a readback copy"))
        } else {
            api.read(&windows[index])
        });
    }
    let mut texture_copies = texture_copies.iter();
    for spec in &program.textures {
        let mut bytes = Vec::new();
        for _ in 0..spec.slices {
            bytes.extend(api.read(texture_copies.next().expect("a readback copy")));
        }
        state.textures.push(bytes);
    }
    for (name, bytes) in readback_records(program, &state) {
        obs.bytes(&name, &bytes);
    }
    Ok(())
}
