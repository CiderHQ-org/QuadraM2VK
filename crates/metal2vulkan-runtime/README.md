# metal2vulkan-runtime

The Metal API on Vulkan. A program calls it the way it calls Apple's Metal (a device, command
queues, command buffers, the encoders, buffers, textures, heaps, pipeline states, events), and the
runtime runs the work on a Vulkan device through [`ash`](https://crates.io/crates/ash). It
translates each shader from Metal AIR to SPIR-V, owns the command buffers, submission and
barriers, and tracks hazards as Metal does.

> **Unstable.** Names and shapes may change before the runtime's API is frozen. Where the runtime
> cannot give Metal's answer exactly it refuses with a typed error, or announces a recorded
> near-equivalent; it never silently runs something else.

- Objects keep Metal's names without the `MTL` prefix (`Device`, `CommandQueue`, `Buffer`, ...).
  Methods are Metal's selectors in snake case: `newBufferWithLength:options:` is
  `Device::new_buffer`. Each item's rustdoc names its Metal counterpart and what differs or is
  refused.
- Rust ownership stands for retain and release. Handles are cheap to clone, `Send + Sync`, and
  equal when they are the same object; encoders are owned values that `end_encoding` consumes.
- Where Metal raises, ends the process, answers nil or fills an `NSError`, a call returns a typed
  `Error` ([Errors](#errors)).

## The crates

| Crate | Directory | What it is |
|---|---|---|
| `metal2vulkan-runtime` | `crates/metal2vulkan-runtime` (this crate) | The Metal API on Vulkan, built only on the translator's public API |
| `metal2vulkan` | [`crates/metal2vulkan`](../metal2vulkan) | The translator: AIR to SPIR-V for a described Vulkan device, and its compile-and-plan API ([docs/API.md](../../docs/API.md)) |
| `metal2vulkan-air` | [`crates/metal2vulkan-air`](../metal2vulkan-air) | The AIR front end |

The three are released together at one version. A program using the runtime depends on the
runtime and on `metal2vulkan`: the runtime's calls take the translator's descriptors and Metal's
value types, which it does not re-export (`metal2vulkan::pipeline::RenderPipelineDescriptor`,
`metal2vulkan::metal::PixelFormat`, ...).

## Depending on it

Not on crates.io yet. Depend on both crates from the same git revision (or checkout):

```toml
[dependencies]
metal2vulkan-runtime = { git = "https://github.com/steelbrain/metal2vulkan", rev = "<sha>" }
metal2vulkan = { git = "https://github.com/steelbrain/metal2vulkan", rev = "<sha>" }
# Only for the Vulkan interop calls (raw handles, adopting a VkBuffer or VkImage):
ash = { version = "0.38", default-features = false, features = ["std"] }
```

- Rust 1.98 or newer.
- Nothing links against Vulkan at build time. The runtime opens the Vulkan loader when it makes a
  device, or takes the consumer's `vkGetInstanceProcAddr` when it adopts one. A device needs
  Vulkan 1.2 and `timelineSemaphore`.
- Shaders come in as a metallib (`Device::new_library_with_data`); there is no MSL compiler. Like
  Metal, `new_library_with_data` refuses bare AIR bitcode. For AIR bitcode, parse it with
  `metal2vulkan::library::Library::from_air` and pass the result to
  `Device::new_library_with_compile_library`.
- Every shader the runtime translates is kept on disk, by default in the platform's per-user cache
  directory, and in memory ([HOWTO](../../docs/HOWTO.md#the-runtimes-shader-cache)).
  `ShaderCacheOptions` names another directory; give each test its own.

## A first dispatch

From [`examples/compute_dispatch.rs`](examples/compute_dispatch.rs):

```rust,ignore
let device = Device::system_default()?;
let library = device.new_library_with_data(&metallib_bytes)?;
let function = library.new_function_with_name("scale_and_offset")?;
let state =
    device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?;

let input = device.new_buffer_with_bytes(&input_bytes, ResourceOptions::shared())?;
let output = device.new_buffer((COUNT * 4) as u64, ResourceOptions::shared())?;

let queue = device.new_command_queue()?;
let command_buffer = queue.command_buffer()?;
let mut encoder = command_buffer.compute_command_encoder()?;
encoder.set_compute_pipeline_state(&state)?;
encoder.set_buffer(Some(&input), 0, 0)?;
encoder.set_buffer(Some(&output), 0, 1)?;
encoder.dispatch_threads(Size::new(COUNT as u64, 1, 1), Size::new(64, 1, 1))?;
encoder.end_encoding()?;
command_buffer.commit()?;
command_buffer.wait_until_completed()?;
// `output.contents()` now holds the kernel's results.
```

Creating a pipeline state translates its functions, each under the 20-second translation budget.
A function the device cannot run exactly is refused there, typed, naming what is missing.

Creating a render pipeline state also makes the `VkPipeline` its first draw is predicted to take,
before it returns, where Metal compiles. The prediction comes from the descriptor and the device
alone: the pass of its attachments, its input primitive class, each vertex-descriptor layout's
buffer bound at an aligned offset (a layout of `MTLBufferLayoutStrideDynamic` at the stride that
packs its attributes), Metal's default sampler at each runtime sampler slot, no texture, and
Metal's default encoder state. A draw whose key differs makes its own `VkPipeline` at that draw.
Tessellated and mesh states, separate depth and stencil formats and fragment functions that write
depth make nothing at creation.

## Examples

Each example runs on the system default device and uses only public API. Without an argument it
runs the repository's own fixture
[`runtime_examples.metal`](../metal2vulkan-validation/fixtures/public/examples/runtime_examples.metal),
assembled from its committed LLVM text by `llvm-as` (LLVM 21 or newer) on `PATH`. With an argument
it runs that metallib instead, for example one built with
`xcrun -sdk macosx metal -o runtime_examples.metallib runtime_examples.metal`.

| Example | What it shows |
|---|---|
| [`compute_dispatch`](examples/compute_dispatch.rs) | A library from bytes, a compute pipeline state, buffers, one dispatch, the result read on the CPU |
| [`render_to_texture`](examples/render_to_texture.rs) | A render pipeline state, a render pass that clears and draws into a `Private` texture, and a blit that reads it back |
| [`indirect_command_buffer`](examples/indirect_command_buffer.rs) | Compute commands encoded on the CPU, read back as typed records, and executed after `use_resource` |
| [`heap_placement`](examples/heap_placement.rs) | A placement heap, textures at offsets `heap_texture_size_and_align` answers, and fences between the encoders that use them |
| [`adopted_buffer`](examples/adopted_buffer.rs) | A `Shared` buffer over a `VkBuffer` and `VkDeviceMemory` the caller made, the shape of a dma-buf import |

```sh
cargo run -p metal2vulkan-runtime --example compute_dispatch
cargo run -p metal2vulkan-runtime --example render_to_texture -- runtime_examples.metallib
```

## Opening a device

**Made by the runtime.** `Device::system_default()` is `MTLCreateSystemDefaultDevice()`: the first
physical device the runtime can open, in the order discrete, integrated, virtual, CPU.
`Device::all()` is `MTLCopyAllDevices()`. Their `_with` forms take `DeviceOptions`, the
Vulkan-side choices Metal has no counterpart for:

| `DeviceOptions` field | What it chooses |
|---|---|
| `policy` | The translator's `DevicePolicy` ([docs/API.md](../../docs/API.md#the-policy)): containment, and the fast or exact form of each accuracy choice ([FAST-PATHS](../../docs/FAST-PATHS.md)). |
| `kinds` | Which `DeviceKind`s qualify; empty admits every kind. |
| `loader` | A Vulkan loader library to open instead of the default ones (the platform's loader by name, then on macOS `$VULKAN_SDK/lib`, `/opt/homebrew/lib` and `/usr/local/lib`). If none opens, `Error::LoaderNotFound` lists each path tried. |
| `layers`, `layer_settings` | Instance layers, in order, and their settings. |
| `instance_extensions`, `device_extensions` | Extensions beyond those the runtime enables (a swapchain, for example). |
| `instance_api_version` | The instance's `apiVersion` (1.2 when unset); a version outside 1.2 to the loader's is refused, never clamped. |
| `withheld_features`, `withheld_extensions`, `withhold_generated_commands`, `withhold_view_reinterpretation` | Parts of the recommended enables to leave off; the device then compiles for what is enabled. |
| `messages` | A `MessageSink` for the validation layer's messages and the runtime's notices; without one, notices go to stderr. |
| `shader_cache` | Where translated shaders are kept, and how much. |
| `gpu_times_from_host` | Answer command buffers' GPU times from the host clock, sparing two timestamps per command buffer. |
| `announce_each` | Degradation kinds to announce at every occurrence instead of once. |
| `metal_level` | The macOS release whose Metal the device answers as (the host's by default). |
| `record_executions` | Keep, per command buffer, which pipeline each dispatch and draw ran (`CommandBuffer::executions`). Off by default. |

While a handle lives, equal options give the same device for a physical device, as Metal has one
`MTLDevice` per GPU. Nothing is read from the environment but the platform's default cache
directory.

**Adopted from the consumer.** `unsafe Device::from_raw(RawDevice, DevicePolicy)` works on an
instance and device the consumer made and keeps. `RawDevice` takes the `vkGetInstanceProcAddr`,
the instance and its version, the physical device, the device, the queue family and index, and
exactly what the device enabled (an `EnableSet`), plus what the runtime may use outside that
vocabulary (calibrated timestamps, sparse features, view format reinterpretation, host image copy)
and the same per-device choices as `DeviceOptions`. Dropping the runtime device destroys only what
the runtime made.

**The Vulkan objects underneath** (no Metal counterpart; each call is `unsafe`):

- `Device::raw()` gives the entry, instance, physical device, `ash::Device` and queue family.
  `CommandQueue::with_vk_queue` lends a queue's `VkQueue` under the runtime's lock.
- `Texture::vk_image`, `Buffer::vk_buffer` and `SharedEvent::vk_semaphore` give an object's handle.
  A texture over a buffer's bytes that aliases them gives a linear image (the same one for textures
  of one image over the same bytes of one import of host pages, which share it), which over a
  no-copy buffer's or dma-buf host pages may be of `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT`
  (`DRM_FORMAT_MOD_LINEAR`, its one memory plane at the texture's `bytes_per_row`): ask its layout
  of memory plane 0.
- `CommandBuffer::encode_wait_for_vk_semaphore` and `encode_signal_vk_semaphore` order a command
  buffer against the consumer's own work.
- `Device::new_texture_with_vk_image` makes a `Private` texture over the consumer's `VkImage` (a
  swapchain image, for presenting).
- `Device::new_buffer_with_vk_buffer` makes a buffer over the consumer's `VkBuffer` and memory,
  such as an imported dma-buf, made with `Device::vk_buffer_usage()`. The runtime never frees it.

## The memory model

- **Storage modes** keep Metal's meaning. `DeviceProperties::has_unified_memory` answers from the
  device type: yes for an integrated GPU or a CPU implementation, no for a discrete GPU. Without
  unified memory a `Managed` buffer has two copies, as on a Mac with a discrete GPU
  (`did_modify_range` and a blit's `synchronize_resource` move the bytes), and a `Shared` resource
  lies in host-cached system memory (announced once, `shared-in-system-memory`).
- **New resources** read what plain Metal's measured answer is: a new buffer reads zero.
- **The Metal address space.** Every buffer and heap has a `gpu_address`, and textures, samplers,
  pipeline states and indirect command buffers a `gpu_resource_id`. Translated shaders read both
  from argument buffers, as on Metal 3. A consumer may pin either (`AddressPlacement`,
  `IdPlacement`, `Device::reserve_resource_ids`); a pin the space cannot honour is
  `Error::PlacementConflict`.
- **Host memory without a copy.** `unsafe Device::new_buffer_with_bytes_no_copy` imports host
  memory, given as a `HostMemory`, through `VK_EXT_external_memory_host`; memory the device cannot
  import exactly is refused. `unsafe Device::new_heap_with_bytes_no_copy` is its heap twin.
  `Buffer::rebind_bytes_no_copy` and `Heap::rebind_bytes_no_copy` move either to another range,
  keeping the objects, addresses and IDs. A rebind is refused (`RebindProblem::EncodedUses`) while
  a command buffer not yet committed, or an open encoder, used the bytes; a render encoder whose
  attachment lies in them uses them from its open. Where a driver refuses the pointer (RADV
  refuses shared mappings, a `memfd`'s among them), memory made with `HostMemory::with_dma_buf` is
  imported from dma-bufs of its pages instead
  ([HOWTO](../../docs/HOWTO.md#6-host-memory-a-driver-refuses)). A no-copy buffer's import of such
  dma-bufs that only blits have read may be released while idle and imported again at its next
  blit, which then costs time: RADV lists every import in every submission. Nothing of Metal's
  observes it.
  `CommandBuffer::used_resources` lists what a command buffer uses until it completes, so a
  program knows when it may take memory away.
- **Hazards** are tracked as Metal tracks them. A heap's resources are untracked by default, so
  encoders that share them order themselves with fences. Commands that reach a resource through an
  address or ID in memory need it resident (`use_resource`, `use_heap`), as on Metal. The runtime
  orders a tracked buffer by the bytes a command reaches: those its translation proves a shader
  reads or writes, else from the bound offset to the buffer's end. Vulkan's synchronization
  validation takes a shader's buffer as accessed over its whole descriptor, which spans those
  bytes widened to 16-byte granules from the device's offset alignment, so it reports hazards
  between commands whose bytes are apart, as it does between untracked uses Metal leaves to the
  program; no order of either changes a result.
- **Textures over buffers** share the buffer's bytes. The runtime keeps a copy and announces it
  (`shadowed-texture`) over a no-copy buffer's host pages (an image bound to them would start
  `UNDEFINED`, and its first move may rewrite any byte bound to it), where the device's linear
  images cannot hold Metal's layout, and where the rows lie with bytes between them.

## Runtime extensions

Calls Metal does not have, for a consumer that emulates a Metal device:

- `CommandBuffer::commit_failed(CommitFailure)` commits a command buffer as failed, running the
  encoders Metal would still run and keeping its signals, as a consumer measured on an M3 Max.
- `CommandBuffer::set_submission_check` runs a check as the command buffer is submitted, and fails
  it as Metal fails a submission.
- `CommandBuffer::add_submission_handler` is called when the runtime takes the command buffer's
  last batch for its queue (submits it, or holds it for one kernel submission with later ones);
  `add_executed_commands_handler` each time an execution's indirect records are read.
- `RenderCommandEncoder::set_color_resolve_target`, `set_depth_resolve_target` and
  `set_stencil_resolve_target` change where a pass resolves until it ends (the last target set
  wins), as Metal's private calls do.
- `TextureDescriptor::address_mode_map` presents a texture as sampling by other address modes than
  its sampler's.
- Resource ID reservations, placements and counts (`Device::reserve_resource_ids`,
  `IdPlacement`, `Device::sampler_ids_live`), and the private getters of indirect commands.
- `HostMemory::with_dma_buf` and its `DmaBufSource`: dma-bufs of a no-copy object's pages, for a
  device that refuses their pointer.

[docs/API.md section 10](../../docs/API.md#10-accepted-differences) lists where the runtime
answers otherwise than Metal on purpose.

## Errors

Every failure is a value of `metal2vulkan_runtime::Error`. Match its variants; never read the
message text. The ones a consumer meets most:

| Variant | Means |
|---|---|
| `Invalid(Invalid)` | What Metal's validation stops, ending the process: `call` names the method, `problem` the typed `Problem`. A program that runs on Metal never makes it. |
| `Nil(Nil)` | What plain Metal answers nil for, the process going on. |
| `MetalError(MetalError)` | What plain Metal answers nil for with an `NSError`: Metal's `domain`, `code` and description as measured, for a consumer to pass on. |
| `Unproven(Unproven)` | What the runtime cannot prove plain Metal's answer for. A consumer that answers as Metal stops its guest, naming the problem. |
| `Unsupported(Unsupported)` | The device lacks what an exact answer needs: `feature` names the Metal feature, `needs` what would change the answer. |
| `NotImplemented(RuntimeGap)` | Plain Metal the runtime does not build yet. Metal takes the input, so a later build may too. |
| `AppleOnly(AppleOnly)` | An Apple-private call off Metal's default with no Vulkan form, or whose meaning on Apple's GPU is unmeasured. Calls that state Metal's default are accepted. |
| `Compile(...)` | A library or pipeline the translator refused, over budget or cancelled ([docs/API.md section 8](../../docs/API.md#8-refusals-requested-inputs-and-precision)). |
| `OutOfMemory`, `DeviceLost`, `Vulkan`, `NoDevice` | The device's own failures. `NoDevice`: the loader opened but found no usable device. |
| `LoaderNotFound(LoaderNotFound)` | The Vulkan loader did not open; `tried` lists each path. |
| `DriverSetting(DriverSetting)` | A driver setting the runtime cannot run under, such as MoltenVK with `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` off. |
| `WaitWouldDeadlock(DeadlockCause)` | A command buffer wait that would deadlock, refused at once instead of hanging. |
| `ShaderCache(...)` | The shader cache directory cannot be made or written. |

```rust,ignore
match device.new_render_pipeline_state(&descriptor, &PipelineOptions::default()) {
    Ok(state) => use_it(state),
    Err(Error::Unsupported(gap)) => report_device_gap(gap.feature, gap.needs),
    Err(Error::NotImplemented(gap)) => report_runtime_gap(gap),
    Err(Error::AppleOnly(call)) => report_apple_only(call),
    Err(other) => return Err(other.into()),
}
```

A refused call records nothing. A command buffer that fails while it runs ends with status
`Error` and an `error()`; `metal_code()` gives Metal's code, and `origin()` whether Metal or the
runtime failed it. What a translated shader reports (a resident-set miss, for one) is a fact of
the run, not an error: `CommandBuffer::execution_report` reads it once the command buffer
completed.

## Degrade notices

Where the runtime gives Metal's answer only at a cost Metal does not pay (a copy, a scratch
resource, an extra pass), or gives a recorded near-equivalent, the device announces it once per
kind: through `DeviceOptions::messages` as a `DebugMessage` whose `degradation` holds a
`DegradeNotice`, else on stderr. A notice is never fatal. `Degradation::code()` is a stable
`metal2vulkan-degrade-<kind>` string and `category()` says whether it costs time and memory or
exactness; decode a notice by its `Degradation`, never its text.
[docs/API.md section 9](../../docs/API.md#9-degrade-notices) lists every kind.

```rust,ignore
let mut options = DeviceOptions::default();
options.messages = Some(MessageSink::new(|message: &DebugMessage| {
    if let Some(notice) = &message.degradation {
        log_degradation(notice.degradation, &notice.object, &notice.reason);
    }
}));
let device = Device::system_default_with(&options)?;
```

## GPU families

`DeviceProperties` answers Metal's device queries from the Vulkan device.
`Device::supports_family` answers yes for a GPU family only where every row of Apple's feature set
tables for it (features, limits and pixel-format capabilities) is exact on the device;
`Device::family_blockers` names each blocking row and why: not built yet (`RowState::Gap`), not
possible on this device (a refusal), or Metal's own answer is no. Run it on the device for its
list.

## Out of scope

What an ordinary Metal program on Linux needs belongs here; what exists only because the work comes
from a paravirtualized guest belongs to that consumer. So the runtime has none of these, and a
consumer builds them on public calls:

- IOSurface and Mach VM: use host memory without a copy, textures over buffers, and adopted
  `VkBuffer`s and `VkImage`s. Importing a file descriptor or dma-buf is plain Vulkan on the
  runtime's device.
- Scattered backing (`replaceBackingWithRanges:`, `detachBacking`): remap memory to be contiguous.
- Apple's serialized pipeline descriptors and guest-visible identifiers (`resourceIndex`,
  `uniqueIdentifier`, private device answers): keep them beside the runtime's objects.
- Presentation (`CAMetalLayer`): render into a texture over a swapchain image and present with
  plain Vulkan.
- Threads, clocks, preferences and other OS services.

Plain Metal the runtime does not build yet is refused typed where a program reaches it
(`RuntimeGap` lists every gap; [docs/METAL-COVERAGE.md](../../docs/METAL-COVERAGE.md) maps the
whole Metal API).

## Further reading

- [docs/RUNTIME-PORTING.md](../../docs/RUNTIME-PORTING.md): Metal's calls, as Objective-C or
  metal-rs makes them, mapped to the runtime's.
- [docs/API.md](../../docs/API.md): the translator's API, the degrade notices, and the accepted
  differences.
- [docs/HOWTO.md](../../docs/HOWTO.md): integration, the shader cache, failures.
- [docs/REFUSALS.md](../../docs/REFUSALS.md): every refusal of something Metal accepts.
- The rustdoc (`cargo doc -p metal2vulkan-runtime --open`): every item with its Metal counterpart.

## License

Dual-licensed, as the rest of the repository: the GNU General Public License, version 2 or (at
your option) any later version, or the GNU Lesser General Public License, version 3 or (at your
option) any later version, at your choice (`GPL-2.0-or-later OR LGPL-3.0-or-later`).
[LICENSE](LICENSE) is the notice; [LICENSE-GPL-2.0](LICENSE-GPL-2.0),
[LICENSE-LGPL-3.0](LICENSE-LGPL-3.0) and [LICENSE-GPL-3.0](LICENSE-GPL-3.0) (which the LGPL v3
incorporates) are the texts.
