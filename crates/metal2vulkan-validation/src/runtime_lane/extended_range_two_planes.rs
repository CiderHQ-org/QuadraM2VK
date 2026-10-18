//! Kernel reads and writes of Apple's two-plane extended-range format on the device the cfg
//! selects, under the validation layer: a texture of `RGB10A8_2P_XR10` (or its sRGB twin), and a
//! consumer's two plane views of one over a buffer.
//!
//! The owned fixture `xr_two_planes` writes a buffer's `half4` or `float4` values to a texture
//! (`texture2d<half>` or `<float>`, `access::write`) and reads the texture back into a buffer
//! (`access::read`), one thread per texel of a grid whose rows are `w` texels apart, under 32 x 32
//! threadgroups:
//!
//! - a 256 x 256 texture of each two-plane format, one thread per texel: every `half` pattern in
//!   every channel (lane `c` of texel `n` holds pattern `(n + 16384 c) mod 65536`) and a dense
//!   `float` grid (`((n + 16384 c) mod 65536 - 32768) / 16384`), each read back as `half` and
//!   `float`;
//! - the consumer's shape: a 64 x 4 texture under 2 x 1 threadgroups (a 64 x 32 grid, rows 4 to
//!   31 outside the texture), lane `c` of grid thread `n` writing pattern `0x3000 + n + 2048 c`;
//! - the consumer's plane views: one shared buffer of 4096 bytes (byte `i` holding
//!   `(89 i + 7) mod 256`) with plane 0 at offset 0 as `BGR10_XR` (or its sRGB twin) and plane 1
//!   at offset 2048 as `ARGB8Unorm`, each a 2D texture made with `new_texture` over the buffer,
//!   64 x 4 with rows 256 bytes apart and 40 x 3 with rows 192 bytes apart, under the consumer's
//!   threadgroups: both views read, then written with `half` patterns
//!   (`(32 n + 16384 c + 7 c) mod 65536`) and read, then written with `float`s
//!   (`((32 n + 16384 c) mod 65536 - 32768) / 16384`) and read, with the buffer's bytes after each
//!   write.
//!
//! Metal's answer: the SHA-256 of each output, the same on an Apple M3 Max (macOS 26.7) and an
//! Apple M4 (macOS 27.0.1), plain and under Metal's validation layer alike: the textures' and the
//! plane views' from our probes, which keep the bytes. A differing output is
//! printed whole, on one `XR-TWO-PLANES-DIFFERS` line.

use super::degrade_notices::{counts, on_device};
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::error::Feature;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, ClearColor, CommandBufferStatus, CommandQueue,
    ComputePipelineState, CpuCacheMode, Degradation, Device, Error, HazardTrackingMode, Heap,
    HeapDescriptor, HeapType, Invalid, IoSurfacePlane, IoSurfaceYCbCr, LoadAction, Nil, Origin,
    PipelineOptions, Problem, Region, RenderPassDescriptor, ResourceOptions, Size, StorageMode,
    StoreAction, Texture, TextureDescriptor, TextureProblem,
};
use sha2::{Digest, Sha256};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Metal's outputs of a texture of each two-plane format: its format, the output (`HR`: reads
/// after the `half` writes, `FR`: after the `float` writes, `OR`: the consumer's shape; `H` and
/// `F`: the reads' result type) and the SHA-256 of its bytes.
const TEXTURES: [(PixelFormat, &str, &str); 12] = [
    (
        PixelFormat::RGB10A82PXR10,
        "HRH",
        "22cdc45b67da6527333067df72530e09e9dbcc84264693e68dbb931a7e71f390",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "HRF",
        "a35846dbd312d807c50310d22849a4a5ff789cd274a27ff926c59e562f2c00fa",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "FRH",
        "3d5a0a1f6040bc8fa020a49dc7fec306bab6a86722a6c324f347272db3c70ba6",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "FRF",
        "3ead12fa542258ee768c9cf388f96acca8cf94a761fbe6a82013f3146c13e33a",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "ORH",
        "957ef0211e3032b942c71651570ff625670198bfebe0f1c2978fdd21129d71d9",
    ),
    (
        PixelFormat::RGB10A82PXR10,
        "ORF",
        "eac8dabf8163dacdcf619a65681c3c980548e97c39d8ac35304b44208b8a73f3",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "HRH",
        "84b306e7871b19413062634947bfbdb8a0149c2aacf74f26ca3b9b6faccedc3d",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "HRF",
        "bbf1d6107835195d3daef224de56d24aa6b8b0221d3be8e36ca7de88dd09c0a3",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "FRH",
        "ea11a73fe55459ded35d32d872daf417b683fc9e3fdceb3416c74b61747e420f",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "FRF",
        "df70ed07347b993b43fdae512e9e9cc3e145a8fe2771e89772489a60eee88a19",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "ORH",
        "9ab95cab828e6aa443e91e0ed960269c33732dcf3e9364dd3944d0467fcb0780",
    ),
    (
        PixelFormat::RGB10A82PXR10SRGB,
        "ORF",
        "6760e2782c211b2b70bee680a6e9b59bd9c110e81955cf30d5697bfb38b37630",
    ),
];

/// Metal's outputs of the plane views: plane 0's format, the output (`G<g>`: geometry `g` of
/// [`GEOMETRIES`]; `R<s>P<p><H|F>`: the reads of plane `p` after step `s`, 0 before any write, 1
/// after the `half` writes, 2 after the `float` writes; `W<s>BUF`: the buffer's bytes after step
/// `s`'s writes) and the SHA-256 of its bytes.
const VIEWS: [(PixelFormat, &str, &str); 56] = [
    (
        PixelFormat::BGR10XR,
        "G0R0P0H",
        "b483b1d7b3b05b4d74a6e2327cf236c417db1eb2e2b81c8a531db95080afaf73",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R0P0F",
        "6ff881356a5ca03fd54fca5ee6731e745a21bc69322b1b186db358479d29c355",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R0P1H",
        "f761d8eb4502fd5c6f0e1c503ab49460daca7668c481afd9ff788d174e76431e",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R0P1F",
        "15497052576b0dab3f2c9c0d62c35dbd4d5d6cca8050928cb89b8b13cee52a43",
    ),
    (
        PixelFormat::BGR10XR,
        "G0W1BUF",
        "19843f1b277fe8f5959d6972d0339c2aff91364b79c0ad161c190e37713f9dab",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R1P0H",
        "1327e94fecd16989672c75481fd2002805fac2a4607c809e57134d6e35f015a9",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R1P0F",
        "f59b9721689d9de08aa301336521a9ce2700167d5c139b41e5a4d65c807dd1ba",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R1P1H",
        "86ac42b4c82d0a96aa8c98f82e937bedba235a00650dd392aa4d3847fba5f9ef",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R1P1F",
        "4a14a7766350e5c557aaba4a0f3d87a56e336d8f97c17aff8c08abb3770a641b",
    ),
    (
        PixelFormat::BGR10XR,
        "G0W2BUF",
        "980d76185f2053779ab82ff89f72e136803dec9fbae6009b3243d32b72352956",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R2P0H",
        "c57a627dc6d1385e29eaa3e2bd449b3279aa01d2d269d981e03f8e0c0da98647",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R2P0F",
        "479fa0d8d1eb3cf3cf631008bc44319cbcd4722a98056eb8741eaeb42687e853",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R2P1H",
        "ffa8a58c3b205876f527f34c43b995df9ca39830e12b118cc7c2943a796a9f27",
    ),
    (
        PixelFormat::BGR10XR,
        "G0R2P1F",
        "df728d6477a22251d213f356cf2cf34965e8b9e38cd41a371f37fe977ec5166f",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R0P0H",
        "95ddef7435e9473baf8de155074df8785a3d8268ae88e9310c6b5e3a0cccf0d0",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R0P0F",
        "c263b4f41df1b523b837482476b7e04971065b447ed842480f35b76759ab7262",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R0P1H",
        "2d5828773550df7e6945fe80dcf5ab08f1e20308cb484fbf5b0ee1968e971bf8",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R0P1F",
        "7c1b8e689df20bb0b2456482e340e8e093eee1f1a76c414e400cbb2ffeb825d7",
    ),
    (
        PixelFormat::BGR10XR,
        "G1W1BUF",
        "4050578c92bdd1c63170a3968bd318941322c06248d6896362654f9ded87a1d6",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R1P0H",
        "985e576ba856b4c4745465c3e4057ad1c8cb75e03f4c4a00a96a1e35cc74f018",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R1P0F",
        "6d0848522dfea8b6083689718ef76c64c14d36de097a7fda82e2436a6c020452",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R1P1H",
        "36bd67d77aada00c5b064987f9c3adc9fcbd85f215b6bb170eb9ceaa75659b37",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R1P1F",
        "18d70d1c735a4b56f9a769aae7d64539e46553bdb28858471e438fcbb0c8ec6e",
    ),
    (
        PixelFormat::BGR10XR,
        "G1W2BUF",
        "3954d3f5b58b05d9362952ee6aba2f74b215e34c2238c016ef1822746272dfea",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R2P0H",
        "7d5eb36c1b5ffc414ae5d0473027e8174b4bb80fe434db33ff1de1b25cd493ca",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R2P0F",
        "e756ae67309b05930f10e75ec06327b92540f9bc0e0596282e8c56a46700d0f7",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R2P1H",
        "ed4e0e0fe4ee0b63d434722dd61fa4c79d180e76b6fe55af05c3c065df1de6b0",
    ),
    (
        PixelFormat::BGR10XR,
        "G1R2P1F",
        "0662049a1954905e032b58b76a78d3e55d76641af9f510912af40c9fc238d26d",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R0P0H",
        "99286e79493d32759423220db6b7cec6253699d86d7155085c76d71f9305deef",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R0P0F",
        "73a9165d9d5644f33d8b01cbe446361d38e402eee21836f0a4898f54f8b095a9",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R0P1H",
        "f761d8eb4502fd5c6f0e1c503ab49460daca7668c481afd9ff788d174e76431e",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R0P1F",
        "15497052576b0dab3f2c9c0d62c35dbd4d5d6cca8050928cb89b8b13cee52a43",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0W1BUF",
        "2282821c0cea3cf718ff030629dd403af227c8b7ae6d910c355a37cda912ce20",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R1P0H",
        "090f5ac305e09d16af247f98ba3dd0bcf820608b52d3524ac677745d86d6d9fe",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R1P0F",
        "eef6dd65dad347d096f2c1aedef847e28b72eb13922c9c8b58c3e5ba06c16a31",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R1P1H",
        "86ac42b4c82d0a96aa8c98f82e937bedba235a00650dd392aa4d3847fba5f9ef",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R1P1F",
        "4a14a7766350e5c557aaba4a0f3d87a56e336d8f97c17aff8c08abb3770a641b",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0W2BUF",
        "ad267fce676137bd9407833a970f47b4321d3d19a988a1433695d5ab36e8acb7",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R2P0H",
        "40ea900c659d89ebf8d81fe59e1fcf8900aeb3c3417abf3871e88b5d158ee2f2",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R2P0F",
        "a72c1e024c4d90eb62913677a024cbfc048a65a43fc08bb0b2d75f5659c63da9",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R2P1H",
        "ffa8a58c3b205876f527f34c43b995df9ca39830e12b118cc7c2943a796a9f27",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G0R2P1F",
        "df728d6477a22251d213f356cf2cf34965e8b9e38cd41a371f37fe977ec5166f",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R0P0H",
        "2b7053f9b3a609596f3f7e43468277bd293e29596895506721b58694f895f8c1",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R0P0F",
        "05a34979f30eff134b266d0e884c5d1f64bf80f8a54238e0dcfd84f9f867285b",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R0P1H",
        "2d5828773550df7e6945fe80dcf5ab08f1e20308cb484fbf5b0ee1968e971bf8",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R0P1F",
        "7c1b8e689df20bb0b2456482e340e8e093eee1f1a76c414e400cbb2ffeb825d7",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1W1BUF",
        "8a00511dc824517be1f3056c646e572b8ecebea55233e3dd37018016e0b696eb",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R1P0H",
        "8b097d68c10282432a09df84f3efc6a8d43fdbac03204bddba62241e73da7fad",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R1P0F",
        "d5cc080aa785dcb83058eec1e2ca10bded0647836bb119257dd8f36f5f3bb48f",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R1P1H",
        "36bd67d77aada00c5b064987f9c3adc9fcbd85f215b6bb170eb9ceaa75659b37",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R1P1F",
        "18d70d1c735a4b56f9a769aae7d64539e46553bdb28858471e438fcbb0c8ec6e",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1W2BUF",
        "44bdcba10e9ce1c7b826c1b2243bf96d9cf7507e9215b9cce8b885c83d0cb514",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R2P0H",
        "98906c5cbe8d7e5c2029cf25254f492a255e56e9b4d889596cbd9ac43dc13671",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R2P0F",
        "80741b2f62576719a80eacb03a7c8a6aafe4284e3392f0711c3e4de03a594ef9",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R2P1H",
        "ed4e0e0fe4ee0b63d434722dd61fa4c79d180e76b6fe55af05c3c065df1de6b0",
    ),
    (
        PixelFormat::BGR10XRSRGB,
        "G1R2P1F",
        "0662049a1954905e032b58b76a78d3e55d76641af9f510912af40c9fc238d26d",
    ),
];

/// The plane views' geometries: width, height and bytes per row.
const GEOMETRIES: [(u64, u64, u64); 2] = [(64, 4, 256), (40, 3, 192)];

/// The plane views' buffer length, and plane 1's offset in it.
const VIEW_BUFFER: u64 = 4096;
const PLANE_1: u64 = 2048;

/// The consumer's grid: 2 x 1 threadgroups of 32 x 32, rows 64 threads wide.
const GRID_WIDTH: u64 = 64;
const GRID: usize = 64 * 32;

/// The fixture's kernel `name`.
fn state(device: &Device, name: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("xr_two_planes.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(name).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// The fixture's four kernels.
pub(super) struct Kernels {
    write_half: ComputePipelineState,
    write_float: ComputePipelineState,
    read_half: ComputePipelineState,
    read_float: ComputePipelineState,
}

impl Kernels {
    pub(super) fn new(device: &Device) -> Kernels {
        Kernels {
            write_half: state(device, "xr2p_write_half"),
            write_float: state(device, "xr2p_write_float"),
            read_half: state(device, "xr2p_read_half"),
            read_float: state(device, "xr2p_read_float"),
        }
    }
}

/// How a kernel is dispatched: one thread per texel of a `width` x `height` texture, or the
/// consumer's grid; 32 x 32 threads per threadgroup either way.
#[derive(Clone, Copy)]
pub(super) enum Dispatch {
    Texels(u64, u64),
    Consumer,
}

impl Dispatch {
    /// The grid's row width, `w`.
    fn width(self) -> u64 {
        match self {
            Dispatch::Texels(width, _) => width,
            Dispatch::Consumer => GRID_WIDTH,
        }
    }

    /// The grid's threads.
    fn threads(self) -> usize {
        match self {
            Dispatch::Texels(width, height) => (width * height) as usize,
            Dispatch::Consumer => GRID,
        }
    }
}

/// Runs `state` over `texture` with `buffer` at 0 and the grid's row width at 1.
fn run(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
    (texture, buffer): (&Texture, &Buffer),
    dispatch: Dispatch,
) {
    let width = buffer_with(device, &(dispatch.width() as u32).to_le_bytes()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(buffer), 0, 0).unwrap();
    encoder.set_buffer(Some(&width), 0, 1).unwrap();
    let group = Size::new(32, 32, 1);
    match dispatch {
        Dispatch::Texels(width, height) => {
            encoder.dispatch_threads(Size::new(width, height, 1), group)
        }
        Dispatch::Consumer => encoder.dispatch_threadgroups(Size::new(2, 1, 1), group),
    }
    .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
}

/// Writes `values` (`half4`s where `half`, else `float4`s, one per grid thread) to `texture`.
pub(super) fn write(
    device: &Device,
    queue: &CommandQueue,
    kernels: &Kernels,
    texture: &Texture,
    (values, half): (&[u8], bool),
    dispatch: Dispatch,
) {
    let input = buffer_with(device, values).unwrap();
    let state = if half {
        &kernels.write_half
    } else {
        &kernels.write_float
    };
    run(device, queue, state, (texture, &input), dispatch);
}

/// What `texture` reads, as `half4`s and as `float4`s, one per grid thread, the output buffers
/// filled with `0xab` first.
pub(super) fn reads(
    device: &Device,
    queue: &CommandQueue,
    kernels: &Kernels,
    texture: &Texture,
    dispatch: Dispatch,
) -> [Vec<u8>; 2] {
    [(&kernels.read_half, 8u64), (&kernels.read_float, 16)].map(|(state, size)| {
        let length = dispatch.threads() as u64 * size;
        let out = buffer_with(device, &vec![0xab; length as usize]).unwrap();
        run(device, queue, state, (texture, &out), dispatch);
        read_buffer(&out, 0, length).unwrap()
    })
}

/// `n` grid threads' `half4`s: lane `c` of thread `n` holds pattern `pattern(n, c)`.
fn halves(n: usize, pattern: impl Fn(usize, usize) -> usize) -> Vec<u8> {
    (0..n)
        .flat_map(|t| (0..4).map(move |c| (t, c)))
        .flat_map(|(t, c)| ((pattern(t, c) & 0xffff) as u16).to_le_bytes())
        .collect()
}

/// `n` grid threads' `float4`s: lane `c` of thread `n` holds `(k mod 65536 - 32768) / 16384`,
/// `k = step n + 16384 c`.
fn floats(n: usize, step: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|t| (0..4).map(move |c| (t, c)))
        .flat_map(|(t, c)| {
            let k = (step * t + 16384 * c) & 0xffff;
            ((k as f32 - 32768.0) / 16384.0).to_le_bytes()
        })
        .collect()
}

/// The SHA-256 of `bytes`, in hex.
pub(super) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Checks `bytes` against Metal's digest `want` of output `what`, keeping a difference in `wrong`.
fn check(wrong: &mut Vec<String>, what: String, want: &str, bytes: &[u8]) {
    let got = digest(bytes);
    if got != want {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        wrong.push(format!("{what}: {got}, Metal {want}; bytes {hex}"));
    }
}

/// Prints `wrong`'s lines and fails when there is one.
fn report(device: &Device, wrong: &[String]) {
    for line in wrong {
        println!("XR-TWO-PLANES-DIFFERS {}: {line}", device.name());
    }
    assert!(
        wrong.is_empty(),
        "{} outputs differ from Metal",
        wrong.len()
    );
}

/// A `width` x `height` texture of `format` that kernels read and write.
pub(super) fn texture(device: &Device, format: PixelFormat, width: u64, height: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    descriptor.resource_options = ResourceOptions::shared();
    device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {format:?} texture: {e}"))
}

/// A texture of each two-plane format reads back Metal's values after kernel writes of every
/// `half` pattern and of a dense `float` grid, and in the consumer's shape.
#[test]
fn two_plane_textures_read_and_write_as_metal_does() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let mut wrong = Vec::new();
        let side = 256;
        let texels = Dispatch::Texels(side, side);
        let count = texels.threads();
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let metal = |tag: &str| {
                TEXTURES
                    .iter()
                    .find(|m| m.0 == format && m.1 == tag)
                    .map(|m| m.2)
                    .unwrap()
            };
            let t = texture(device, format, side, side);
            let steps = [
                ("HR", halves(count, |n, c| n + 16384 * c), true),
                ("FR", floats(count, 1), false),
            ];
            for (tag, values, half) in steps {
                write(device, &queue, &kernels, &t, (&values, half), texels);
                let [h, f] = reads(device, &queue, &kernels, &t, texels);
                check(
                    &mut wrong,
                    format!("{format:?} {tag}H"),
                    metal(&format!("{tag}H")),
                    &h,
                );
                check(
                    &mut wrong,
                    format!("{format:?} {tag}F"),
                    metal(&format!("{tag}F")),
                    &f,
                );
            }
            let t = texture(device, format, 64, 4);
            let values = halves(GRID, |n, c| 0x3000 + n + 2048 * c);
            write(
                device,
                &queue,
                &kernels,
                &t,
                (&values, true),
                Dispatch::Consumer,
            );
            let [h, f] = reads(device, &queue, &kernels, &t, Dispatch::Consumer);
            check(&mut wrong, format!("{format:?} ORH"), metal("ORH"), &h);
            check(&mut wrong, format!("{format:?} ORF"), metal("ORF"), &f);
        }
        report(device, &wrong);
    });
}

/// The kind a heap's two-plane texture announces where it lies apart from the heap's bytes: a
/// mipmapped or array one.
pub(super) const HEAP_PLANES: [Degradation; 1] = [Degradation::HeapTwoPlaneTexture];

/// A texture of each two-plane format in an automatic and a placement heap of each storage mode,
/// at the size and alignment the device answers, reads back Metal's values in the consumer's
/// shape as a device texture does: a one-level 2D one, a shadow of its planes in the heap's
/// bytes, announces `shadowed-texture`, and a mipmapped one, apart from them,
/// `heap-two-plane-texture` once.
#[test]
fn two_plane_heap_textures_read_and_write_as_metal_does() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let mut wrong = Vec::new();
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let metal = |tag: &str| {
                TEXTURES
                    .iter()
                    .find(|m| m.0 == format && m.1 == tag)
                    .map(|m| m.2)
                    .unwrap()
            };
            for options in [ResourceOptions::private(), ResourceOptions::shared()] {
                let mut descriptor = TextureDescriptor::texture_2d(format, 64, 4, false);
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                descriptor.resource_options = options;
                let answer = device
                    .heap_texture_size_and_align(&descriptor)
                    .unwrap_or_else(|e| panic!("{format:?} {options:?} size and align: {e}"));
                assert!(answer.size >= 64 * 4 * 5 && answer.align.is_power_of_two());
                let mut mipmapped = descriptor;
                mipmapped.mipmap_level_count = 2;
                let kinds = [
                    (HeapType::Automatic, descriptor),
                    (HeapType::Placement, descriptor),
                    (HeapType::Placement, mipmapped),
                ];
                for (heap_type, descriptor) in kinds {
                    let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
                    let heap = device
                        .new_heap(&HeapDescriptor::new(heap_type, answer.size, options))
                        .unwrap();
                    let t = match heap_type {
                        HeapType::Placement => heap.new_texture_at_offset(&descriptor, 0),
                        _ => heap.new_texture(&descriptor),
                    }
                    .unwrap_or_else(|e| panic!("{format:?} {options:?} {heap_type:?}: {e}"));
                    assert_eq!(t.pixel_format(), format);
                    let values = halves(GRID, |n, c| 0x3000 + n + 2048 * c);
                    write(
                        device,
                        &queue,
                        &kernels,
                        &t,
                        (&values, true),
                        Dispatch::Consumer,
                    );
                    let [h, f] = reads(device, &queue, &kernels, &t, Dispatch::Consumer);
                    let what = format!("{format:?} {options:?} {heap_type:?}");
                    check(&mut wrong, format!("{what} ORH"), metal("ORH"), &h);
                    check(&mut wrong, format!("{what} ORF"), metal("ORF"), &f);
                }
            }
        }
        report(device, &wrong);
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&HEAP_PLANES[0]).copied(),
            Some(1),
            "{notices:?}"
        );
        assert!(
            notices.contains_key(&Degradation::ShadowedTexture),
            "{notices:?}"
        );
    });
}

/// The size of the heaps of [`two_plane_heap_textures_keep_their_texels_in_the_heap_s_bytes`],
/// the texture's offset in them, and its extent.
const PLANES_HEAP: u64 = 64 << 10;
const PLANES_AT: u64 = 48;
const PLANES_EXTENT: [u64; 2] = [37, 5];

/// Where a heap's one-level 2D two-plane texture of `[w, h]` texels keeps its planes, from its
/// first byte, as an Apple M4 lays them (`Shared` and `Private` alike): plane
/// 0's rows `align(4 w, 16)` bytes apart, and plane 1 from `align(h · align(4 w, 16), 16384)`,
/// its rows `align(w, 16)` apart.
fn heap_planes([w, h]: [u64; 2]) -> (u64, u64, u64) {
    let color = (4 * w).next_multiple_of(16);
    (
        color,
        (h * color).next_multiple_of(16 << 10),
        w.next_multiple_of(16),
    )
}

/// `bytes` into `buffer` from its start: through a blit from a shared buffer where it is
/// `Private`.
pub(super) fn put(device: &Device, queue: &CommandQueue, buffer: &Buffer, bytes: &[u8]) {
    let staging = buffer_with(device, bytes).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&staging, 0, buffer, 0, bytes.len() as u64)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
}

/// `buffer`'s first `length` bytes, through a blit into a shared buffer.
pub(super) fn get(device: &Device, queue: &CommandQueue, buffer: &Buffer, length: u64) -> Vec<u8> {
    let staging = buffer_with(device, &vec![0; length as usize]).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(buffer, 0, &staging, 0, length)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    read_buffer(&staging, 0, length).unwrap()
}

/// A one-level 2D texture of each two-plane format in a placement heap of each storage mode keeps
/// its texels in the heap's bytes where Metal does, as a buffer over the same
/// bytes sees them while the texture lives: a kernel's writes land on its two planes' rows, each
/// color word with bits 30 and 31 clear, and on no other byte; bytes written into the planes
/// reach the texture's reads; and its CPU copies move plane 0's words there. A texture of the
/// device's own, written alike, holds the same color words and alpha bytes, and, given the same
/// color words, reads the same colors. The device announces `shadowed-texture`, and not
/// `heap-two-plane-texture`.
#[test]
fn two_plane_heap_textures_keep_their_texels_in_the_heap_s_bytes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let [w, h] = PLANES_EXTENT;
        let (color, alpha, alpha_row) = heap_planes(PLANES_EXTENT);
        let texels = (w * h) as usize;
        let dispatch = Dispatch::Texels(w, h);
        let at = |x: u64, y: u64| (PLANES_AT + y * color + 4 * x) as usize;
        let alpha_at = |x: u64, y: u64| (PLANES_AT + alpha + y * alpha_row + x) as usize;
        let alpha_of = |floats: &[u8], t: usize| {
            let a = f32::from_le_bytes(floats[16 * t + 12..16 * t + 16].try_into().unwrap());
            (a * 255.0).round() as u8
        };
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let reference = texture(device, format, w, h);
            for options in [ResourceOptions::shared(), ResourceOptions::private()] {
                let what = format!("{format:?} {:?}", options.storage_mode().unwrap());
                let heap: Heap = device
                    .new_heap(&HeapDescriptor::new(
                        HeapType::Placement,
                        PLANES_HEAP,
                        options,
                    ))
                    .unwrap();
                let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                descriptor.resource_options = options;
                let t = heap
                    .new_texture_at_offset(&descriptor, PLANES_AT)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let bytes = heap
                    .new_buffer_at_offset(PLANES_HEAP, options, 0)
                    .unwrap_or_else(|e| panic!("{what}: a buffer over the texture: {e}"));
                // A kernel's writes land on the planes' rows alone.
                put(device, &queue, &bytes, &vec![0xee; PLANES_HEAP as usize]);
                let values = halves(texels, |n, c| 0x3100 + 37 * n + 2311 * c);
                write(device, &queue, &kernels, &t, (&values, true), dispatch);
                write(
                    device,
                    &queue,
                    &kernels,
                    &reference,
                    (&values, true),
                    dispatch,
                );
                let mut words = vec![0; 4 * texels];
                reference
                    .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                    .unwrap();
                let [_, floats] = reads(device, &queue, &kernels, &reference, dispatch);
                let mut want = vec![0xee; PLANES_HEAP as usize];
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) as usize;
                        want[at(x, y)..at(x, y) + 4].copy_from_slice(&words[4 * i..4 * i + 4]);
                        want[alpha_at(x, y)] = alpha_of(&floats, i);
                        assert_eq!(words[4 * i + 3] >> 6, 0, "{what}: bits 30 and 31 of {i}");
                    }
                }
                let got = get(device, &queue, &bytes, PLANES_HEAP);
                let differ = (0..got.len()).filter(|&k| got[k] != want[k]).count();
                assert_eq!(
                    differ, 0,
                    "{what}: {differ} bytes of the heap differ after writes"
                );
                // Bytes written into the planes reach the texture's reads.
                let pattern: Vec<u8> = (0..PLANES_HEAP).map(|k| (k * 131 + 7) as u8).collect();
                put(device, &queue, &bytes, &pattern);
                let [_, got] = reads(device, &queue, &kernels, &t, dispatch);
                let mut words = vec![0; 4 * texels];
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * w + x) as usize;
                        words[4 * i..4 * i + 4].copy_from_slice(&pattern[at(x, y)..at(x, y) + 4]);
                        assert_eq!(
                            alpha_of(&got, i),
                            pattern[alpha_at(x, y)],
                            "{what}: alpha {i}"
                        );
                    }
                }
                reference
                    .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &words, 4 * w, 0)
                    .unwrap();
                let [_, colors] = reads(device, &queue, &kernels, &reference, dispatch);
                for i in 0..texels {
                    assert_eq!(
                        got[16 * i..16 * i + 12],
                        colors[16 * i..16 * i + 12],
                        "{what}: colors of texel {i}"
                    );
                }
                // The CPU copies plane 0's words where they lie in the heap.
                if options.storage_mode().unwrap() == StorageMode::Shared {
                    let mut read = vec![0; 4 * texels];
                    t.get_bytes(&mut read, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    assert_eq!(read, words, "{what}: get_bytes");
                    let replaced: Vec<u8> = (0..4 * texels).map(|k| (k * 7 + 1) as u8).collect();
                    t.replace_region(Region::new_2d(0, 0, w, h), 0, 0, &replaced, 4 * w, 0)
                        .unwrap();
                    let got = read_buffer(&bytes, 0, PLANES_HEAP).unwrap();
                    let mut want = pattern.clone();
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            want[at(x, y)..at(x, y) + 4]
                                .copy_from_slice(&replaced[4 * i..4 * i + 4]);
                        }
                    }
                    assert!(got == want, "{what}: replace_region");
                }
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(
            notices.contains_key(&Degradation::ShadowedTexture),
            "{notices:?}"
        );
        assert!(
            !notices.contains_key(&Degradation::HeapTwoPlaneTexture),
            "{notices:?}"
        );
    });
}

/// A surface a two-plane texture is made over: the texture's extent, plane 0's offset and
/// `bytesPerRow`, plane 1's, and the surface's bytes.
type SurfaceShape = ([u64; 2], [u64; 2], [u64; 2], u64);

/// The surfaces a two-plane texture is made over: its extent, plane 0's offset and
/// `bytesPerRow`, plane 1's, and the surface's bytes. The consumer's `'b3a8'` surface of 384 x
/// 256 (plane 0 at 1536 bytes a row, plane 1 right after it at 512), and one Metal was measured
/// over at rows off 16 bytes (37 x 5 at 260 and 68 bytes a row, plane 1 4 bytes past plane 0;
/// its GPU reads them at 0 by 256 and 1296 by 64).
const SURFACES: [SurfaceShape; 2] = [
    ([384, 256], [0, 1536], [393_216, 512], 524_288),
    ([37, 5], [0, 260], [1304, 68], 1644),
];

/// `'b3a8'`.
const B3A8: u32 = u32::from_be_bytes(*b"b3a8");

/// A texture of `format` over plane 0 of `surface`, a two-plane `'b3a8'` surface laid out as
/// `shape` (an entry of [`SURFACES`]), of `options`.
fn surface_texture(
    surface: &Buffer,
    format: PixelFormat,
    shape: SurfaceShape,
    options: ResourceOptions,
) -> Result<Texture, Error> {
    let ([w, h], [offset, row], [alpha, alpha_row], _) = shape;
    let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
    descriptor.usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = options;
    let facts = IoSurfaceYCbCr::new(B3A8, 0, [w, h], alpha, alpha_row);
    surface.new_texture_with_iosurface_plane(
        &descriptor,
        IoSurfacePlane::new(offset, row, true).with_ycbcr(facts),
    )
}

/// `Managed` storage with the other modes at their defaults.
fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A `width` x `height` texture of `format` that kernels read and write and passes render into.
fn target(device: &Device, format: PixelFormat, width: u64, height: u64) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, width, height, false);
    descriptor.usage =
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    device
        .new_texture(&descriptor)
        .unwrap_or_else(|e| panic!("a {format:?} texture: {e}"))
}

/// Records one blit command through `record` and waits for it.
fn blit_once(
    queue: &CommandQueue,
    record: impl FnOnce(&mut BlitCommandEncoder) -> Result<(), Error>,
) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    record(&mut blit).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A two-plane texture's planes in bytes as Metal's blit addresses them: plane 0's first byte,
/// the alpha plane's, and the rows both are addressed at.
type BlitPlanes = (usize, usize, usize);

/// Metal's blit of a copy of `w` x `h` texels from `source`'s planes into `destination`'s, each
/// at its own blit rows from its first bytes: each row's color words and alpha bytes in turn, a
/// byte read past the source's end 0 and one written past the destination's end dropped.
fn blit_planes(
    source: &[u8],
    from: BlitPlanes,
    destination: &mut [u8],
    to: BlitPlanes,
    [w, h]: [usize; 2],
) {
    let read = |at: usize| source.get(at).copied().unwrap_or(0);
    for r in 0..h {
        for c in 0..w {
            for k in 0..4 {
                let value = read(from.0 + r * from.2 + 4 * c + k);
                if let Some(byte) = destination.get_mut(to.0 + r * to.2 + 4 * c + k) {
                    *byte = value;
                }
            }
            let value = read(from.1 + r * from.2 + c);
            if let Some(byte) = destination.get_mut(to.1 + r * to.2 + c) {
                *byte = value;
            }
        }
    }
}

/// The planes of a `w` x `h` texture of the device's memory as Metal lays them out: `words` in rows
/// of `align(4 w, 16)` bytes, `alphas` from `align(h` rows`, 16384)` in rows of `align(w, 16)`,
/// every other byte 0; and their blit planes (rows `⌊4 w / 16⌋ · 16`).
fn own_planes(words: &[u8], alphas: &[u8], [w, h]: [usize; 2]) -> (Vec<u8>, BlitPlanes) {
    let (color, alpha) = ((4 * w).next_multiple_of(16), w.next_multiple_of(16));
    let alpha_at = (h * color).next_multiple_of(16384);
    let mut memory = vec![0; alpha_at + h * alpha];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            memory[y * color + 4 * x..y * color + 4 * x + 4]
                .copy_from_slice(&words[4 * i..4 * i + 4]);
            memory[alpha_at + y * alpha + x] = alphas[i];
        }
    }
    (memory, (0, alpha_at, 4 * w / 16 * 16))
}

/// The color words and alpha bytes of a `w` x `h` texture's planes laid out as [`own_planes`].
fn own_texels(memory: &[u8], [w, h]: [usize; 2]) -> (Vec<u8>, Vec<u8>) {
    let (color, alpha) = ((4 * w).next_multiple_of(16), w.next_multiple_of(16));
    let alpha_at = (h * color).next_multiple_of(16384);
    let mut words = Vec::with_capacity(4 * w * h);
    let mut alphas = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            words.extend_from_slice(&memory[y * color + 4 * x..y * color + 4 * x + 4]);
            alphas.push(memory[alpha_at + y * alpha + x]);
        }
    }
    (words, alphas)
}

/// Clears `texture` through a render pass to `color`.
fn clear(queue: &CommandQueue, texture: &Texture, color: ClearColor) {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(texture.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.store_action = StoreAction::Store;
    attachment.clear_color = color;
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&pass)
        .unwrap()
        .end_encoding()
        .unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A texture of each two-plane format over plane 0 of a two-plane surface, of `Shared` and
/// `Managed` storage, keeps its texels in the surface's two planes where Metal does: a kernel's
/// writes and a clear land on plane 0's and plane 1's GPU rows (their offsets and rows rounded down
/// to 16 bytes) and on no other byte, each color word with bits 30 and 31 clear; bytes written into
/// the planes reach the texture's reads; its CPU copies move plane 0's words at the plane's own
/// offset and rows, and its blits with a buffer at the GPU's, plane 1 untouched. A texture of the
/// device's own, written alike, holds the same color words and alpha bytes. A fill writes plane 0's
/// words at the GPU's rows, and a copy out of the surface into a texture of the device's memory,
/// into the surface from one, and into another surface's planes moves both planes' bytes at Metal's
/// blit rows (plane 0's GPU rows for a surface's, `⌊4 w / 16⌋ · 16` for the device's), past the
/// planes too but not past the surface.
#[test]
fn two_plane_surface_textures_keep_their_texels_in_the_surface_s_planes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let alpha_of = |floats: &[u8], t: usize| {
            let a = f32::from_le_bytes(floats[16 * t + 12..16 * t + 16].try_into().unwrap());
            (a * 255.0).round() as u8
        };
        for shape in SURFACES {
            let ([w, h], [offset, row], [alpha, alpha_row], length) = shape;
            let texels = (w * h) as usize;
            let dispatch = Dispatch::Texels(w, h);
            let down = |v: u64| v - v % 16;
            let gpu = |x: u64, y: u64| (down(offset) + y * down(row) + 4 * x) as usize;
            let cpu = |x: u64, y: u64| (offset + y * row + 4 * x) as usize;
            let alpha_at = |x: u64, y: u64| (down(alpha) + y * down(alpha_row) + x) as usize;
            for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
                let reference = target(device, format, w, h);
                for options in [ResourceOptions::shared(), managed()] {
                    let what = format!("{format:?} {w}x{h} {:?}", options.storage_mode().unwrap());
                    let surface = buffer_with(device, &vec![0xee; length as usize]).unwrap();
                    let t = surface_texture(&surface, format, shape, options)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    // A kernel's writes land on the planes' GPU rows alone.
                    let values = halves(texels, |n, c| 0x3100 + 37 * n + 2311 * c);
                    write(device, &queue, &kernels, &t, (&values, true), dispatch);
                    write(
                        device,
                        &queue,
                        &kernels,
                        &reference,
                        (&values, true),
                        dispatch,
                    );
                    let mut words = vec![0; 4 * texels];
                    reference
                        .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    let [_, floats] = reads(device, &queue, &kernels, &reference, dispatch);
                    let mut want = vec![0xee; length as usize];
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            want[gpu(x, y)..gpu(x, y) + 4]
                                .copy_from_slice(&words[4 * i..4 * i + 4]);
                            want[alpha_at(x, y)] = alpha_of(&floats, i);
                            assert_eq!(words[4 * i + 3] >> 6, 0, "{what}: bits 30 and 31 of {i}");
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    let differ = (0..got.len()).filter(|&k| got[k] != want[k]).count();
                    assert_eq!(differ, 0, "{what}: {differ} bytes differ after writes");
                    // A clear lands on the same bytes as the reference's clear.
                    let color = ClearColor::new(1.25, -0.25, 0.5, 0.75);
                    clear(&queue, &t, color);
                    clear(&queue, &reference, color);
                    reference
                        .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    let [_, floats] = reads(device, &queue, &kernels, &reference, dispatch);
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            want[gpu(x, y)..gpu(x, y) + 4]
                                .copy_from_slice(&words[4 * i..4 * i + 4]);
                            want[alpha_at(x, y)] = alpha_of(&floats, i);
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    let differ = (0..got.len()).filter(|&k| got[k] != want[k]).count();
                    assert_eq!(differ, 0, "{what}: {differ} bytes differ after a clear");
                    // Bytes written into the planes reach the texture's reads.
                    let pattern: Vec<u8> = (0..length).map(|k| (k * 131 + 7) as u8).collect();
                    put(device, &queue, &surface, &pattern);
                    let [_, got] = reads(device, &queue, &kernels, &t, dispatch);
                    let mut words = vec![0; 4 * texels];
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            words[4 * i..4 * i + 4]
                                .copy_from_slice(&pattern[gpu(x, y)..gpu(x, y) + 4]);
                            assert_eq!(
                                alpha_of(&got, i),
                                pattern[alpha_at(x, y)],
                                "{what}: alpha {i}"
                            );
                        }
                    }
                    reference
                        .replace_region(Region::new_2d(0, 0, w, h), 0, 0, &words, 4 * w, 0)
                        .unwrap();
                    let [_, colors] = reads(device, &queue, &kernels, &reference, dispatch);
                    for i in 0..texels {
                        assert_eq!(
                            got[16 * i..16 * i + 12],
                            colors[16 * i..16 * i + 12],
                            "{what}: colors of texel {i}"
                        );
                    }
                    // The CPU copies plane 0's words at the plane's own rows.
                    let mut read = vec![0; 4 * texels];
                    t.get_bytes(&mut read, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                        .unwrap();
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            assert_eq!(
                                read[4 * i..4 * i + 4],
                                pattern[cpu(x, y)..cpu(x, y) + 4],
                                "{what}: get_bytes {i}"
                            );
                        }
                    }
                    let replaced: Vec<u8> = (0..4 * texels).map(|k| (k * 7 + 1) as u8).collect();
                    t.replace_region(Region::new_2d(0, 0, w, h), 0, 0, &replaced, 4 * w, 0)
                        .unwrap();
                    let mut want = pattern.clone();
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            want[cpu(x, y)..cpu(x, y) + 4]
                                .copy_from_slice(&replaced[4 * i..4 * i + 4]);
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    assert!(got == want, "{what}: replace_region");
                    // Blits with a buffer move plane 0's words at the GPU's rows.
                    let out = buffer_with(device, &vec![0xcd; 4 * texels]).unwrap();
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.copy_from_texture_to_buffer(
                        &t,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        Size::new(w, h, 1),
                        &out,
                        0,
                        4 * w,
                        4 * w * h,
                        BlitOption::None,
                    )
                    .unwrap();
                    let into: Vec<u8> = (0..4 * texels).map(|k| (k * 13 + 5) as u8).collect();
                    let source = buffer_with(device, &into).unwrap();
                    blit.copy_from_buffer_to_texture(
                        &source,
                        0,
                        4 * w,
                        4 * w * h,
                        Size::new(w, h, 1),
                        &t,
                        0,
                        0,
                        Origin::new(0, 0, 0),
                        BlitOption::None,
                    )
                    .unwrap();
                    blit.end_encoding().unwrap();
                    command_buffer.commit().unwrap();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    let moved = read_buffer(&out, 0, 4 * texels as u64).unwrap();
                    let before = want.clone();
                    for y in 0..h {
                        for x in 0..w {
                            let i = (y * w + x) as usize;
                            assert_eq!(
                                moved[4 * i..4 * i + 4],
                                before[gpu(x, y)..gpu(x, y) + 4],
                                "{what}: blit out {i}"
                            );
                            want[gpu(x, y)..gpu(x, y) + 4].copy_from_slice(&into[4 * i..4 * i + 4]);
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    assert!(got == want, "{what}: blit in");
                    // A fill writes plane 0's words at the GPU's rows, cut at the level.
                    let mut want = read_buffer(&surface, 0, length).unwrap();
                    blit_once(&queue, |blit| {
                        blit.fill_texture(&t, 0, 0, Region::new_2d(1, 1, w, h), &[1, 2, 3])
                    });
                    for y in 1..h {
                        for x in 1..w {
                            want[gpu(x, y)..gpu(x, y) + 4].copy_from_slice(&[1, 2, 3, 1]);
                        }
                    }
                    let got = read_buffer(&surface, 0, length).unwrap();
                    assert!(got == want, "{what}: fill");
                    // Copies with a texture of the device's memory and with another surface's
                    // move both planes' bytes at the blit rows, as Metal's blit does.
                    let extent = [w as usize, h as usize];
                    let texels_of = |texture: &Texture| -> (Vec<u8>, Vec<u8>) {
                        let mut words = vec![0; 4 * texels];
                        texture
                            .get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
                            .unwrap();
                        let [_, floats] = reads(device, &queue, &kernels, texture, dispatch);
                        (words, (0..texels).map(|i| alpha_of(&floats, i)).collect())
                    };
                    let planes = (
                        down(offset) as usize,
                        down(alpha) as usize,
                        down(row) as usize,
                    );
                    // Out of the surface into a texture of the device's memory.
                    let own = target(device, format, w, h);
                    write(device, &queue, &kernels, &own, (&values, true), dispatch);
                    let (words, alphas) = texels_of(&own);
                    let (mut memory, own_at) = own_planes(&words, &alphas, extent);
                    let bytes = read_buffer(&surface, 0, length).unwrap();
                    blit_once(&queue, |blit| blit.copy_texture(&t, &own));
                    blit_planes(&bytes, planes, &mut memory, own_at, extent);
                    assert!(
                        texels_of(&own) == own_texels(&memory, extent),
                        "{what}: copy out of the surface"
                    );
                    // Into the surface from a texture of the device's memory.
                    let fresh = target(device, format, w, h);
                    let others = halves(texels, |n, c| 0x2900 + 53 * n + 977 * c);
                    write(device, &queue, &kernels, &fresh, (&others, true), dispatch);
                    let (words, alphas) = texels_of(&fresh);
                    let (memory, fresh_at) = own_planes(&words, &alphas, extent);
                    let mut want = read_buffer(&surface, 0, length).unwrap();
                    blit_once(&queue, |blit| blit.copy_texture(&fresh, &t));
                    blit_planes(&memory, fresh_at, &mut want, planes, extent);
                    let got = read_buffer(&surface, 0, length).unwrap();
                    assert!(got == want, "{what}: copy into the surface");
                    // Into another surface's planes.
                    let pattern: Vec<u8> = (0..length).map(|k| (k * 29 + 7) as u8).collect();
                    let other = buffer_with(device, &pattern).unwrap();
                    let t2 = surface_texture(&other, format, shape, options)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    let mut want = pattern;
                    blit_once(&queue, |blit| blit.copy_texture(&t, &t2));
                    blit_planes(&got, planes, &mut want, planes, extent);
                    let into = read_buffer(&other, 0, length).unwrap();
                    assert!(into == want, "{what}: copy between surfaces");
                }
            }
        }
        let notices = counts(&messages.take_notices());
        assert!(
            notices.contains_key(&Degradation::ShadowedTexture),
            "{notices:?}"
        );
    });
}

/// A consumer's two plane views over one buffer, plane 0 in `BGR10_XR` (or its sRGB twin) and
/// plane 1 in `ARGB8Unorm`, read Metal's values and leave Metal's bytes in the buffer after
/// kernel writes, in both geometries.
#[test]
fn plane_views_over_a_buffer_read_and_write_as_metal_does() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let kernels = Kernels::new(device);
        let mut wrong = Vec::new();
        let steps = [
            (halves(GRID, |n, c| 32 * n + 16384 * c + 7 * c), true),
            (floats(GRID, 32), false),
        ];
        for plane0 in [PixelFormat::BGR10XR, PixelFormat::BGR10XRSRGB] {
            for (g, (width, height, row)) in GEOMETRIES.into_iter().enumerate() {
                let metal = |tag: &str| {
                    VIEWS
                        .iter()
                        .find(|m| m.0 == plane0 && m.1 == format!("G{g}{tag}"))
                        .map(|m| m.2)
                        .unwrap()
                };
                let initial: Vec<u8> = (0..VIEW_BUFFER).map(|i| (i * 89 + 7) as u8).collect();
                let buffer = buffer_with(device, &initial).unwrap();
                let view = |format: PixelFormat, offset: u64| {
                    let mut descriptor =
                        TextureDescriptor::texture_2d(format, width, height, false);
                    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                    descriptor.resource_options = ResourceOptions::shared();
                    buffer
                        .new_texture(&descriptor, offset, row)
                        .unwrap_or_else(|e| panic!("a {format:?} view: {e}"))
                };
                let planes = [view(plane0, 0), view(PixelFormat::ARGB8Unorm, PLANE_1)];
                let mut outputs = Vec::new();
                for (p, plane) in planes.iter().enumerate() {
                    let [h, f] = reads(device, &queue, &kernels, plane, Dispatch::Consumer);
                    outputs.push((format!("R0P{p}H"), h));
                    outputs.push((format!("R0P{p}F"), f));
                }
                for (s, (values, half)) in steps.iter().enumerate() {
                    let s = s + 1;
                    for plane in &planes {
                        write(
                            device,
                            &queue,
                            &kernels,
                            plane,
                            (values, *half),
                            Dispatch::Consumer,
                        );
                    }
                    outputs.push((
                        format!("W{s}BUF"),
                        read_buffer(&buffer, 0, VIEW_BUFFER).unwrap(),
                    ));
                    for (p, plane) in planes.iter().enumerate() {
                        let [h, f] = reads(device, &queue, &kernels, plane, Dispatch::Consumer);
                        outputs.push((format!("R{s}P{p}H"), h));
                        outputs.push((format!("R{s}P{p}F"), f));
                    }
                }
                for (tag, bytes) in outputs {
                    check(
                        &mut wrong,
                        format!("{plane0:?} G{g}{tag}"),
                        metal(&tag),
                        &bytes,
                    );
                }
            }
        }
        report(device, &wrong);
    });
}

/// Each two-plane format makes what Metal makes and refuses what Metal stops (one case per
/// process): a 2D texture of each storage mode and usage, of 1 x 1 and 63 x 5, mipmapped, a 2D
/// array, 1D, 3D and `new_shared_texture`'s are made; a cube (Metal's
/// nil, `Error::Nil`) and a multisample texture (Metal asserts, `Error::Invalid`) are refused
/// `TypeForFormat`, and `ShaderAtomic` usage (Metal asserts) `UsageForFormat`.
#[test]
fn two_plane_textures_are_made_as_metal_makes_them() {
    on_device(|device, _| {
        let rw = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        for format in [PixelFormat::RGB10A82PXR10, PixelFormat::RGB10A82PXR10SRGB] {
            let of = |ty: TextureType, [width, height, depth]: [u64; 3], usage: TextureUsage| {
                let mut d = TextureDescriptor::texture_2d(format, width, height, false);
                d.texture_type = ty;
                d.depth = depth;
                d.usage = usage;
                d.resource_options = ResourceOptions::private();
                d
            };
            let mut made = vec![
                of(TextureType::Type2D, [64, 4, 1], rw),
                of(TextureType::Type2D, [64, 4, 1], TextureUsage::SHADER_READ),
                of(TextureType::Type2D, [64, 4, 1], TextureUsage::SHADER_WRITE),
                of(TextureType::Type2D, [64, 4, 1], TextureUsage::RENDER_TARGET),
                of(
                    TextureType::Type2D,
                    [64, 4, 1],
                    TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
                ),
                of(TextureType::Type2D, [1, 1, 1], rw),
                of(TextureType::Type2D, [63, 5, 1], rw),
                of(TextureType::Type1D, [64, 1, 1], rw),
                of(TextureType::Type3D, [16, 16, 2], rw),
            ];
            let mut shared = of(TextureType::Type2D, [64, 4, 1], rw);
            shared.resource_options = ResourceOptions::shared();
            made.push(shared);
            let mut mipmapped = of(TextureType::Type2D, [64, 4, 1], rw);
            mipmapped.mipmap_level_count = 2;
            made.push(mipmapped);
            let mut array = of(TextureType::Type2DArray, [64, 4, 1], rw);
            array.array_length = 2;
            made.push(array);
            for d in &made {
                let t = device
                    .new_texture(d)
                    .unwrap_or_else(|e| panic!("{format:?} {d:?}: {e}"));
                assert_eq!(t.pixel_format(), format);
            }
            let t = device
                .new_shared_texture(&of(TextureType::Type2D, [64, 4, 1], rw))
                .unwrap_or_else(|e| panic!("{format:?} new_shared_texture: {e}"));
            assert_eq!(t.pixel_format(), format);
            let refused = |d: &TextureDescriptor, want: TextureProblem| match device
                .new_texture(d)
                .map(|_| ())
            {
                Err(Error::Invalid(Invalid {
                    problem: Problem::Texture(problem),
                    ..
                })) => assert_eq!(problem, want, "{format:?} {d:?}"),
                other => panic!("{format:?} {d:?}: {other:?}"),
            };
            let mut cube = of(TextureType::Cube, [16, 16, 1], rw);
            cube.height = 16;
            assert_eq!(
                device.new_texture(&cube).map(|_| ()).err(),
                Some(Error::Nil(Nil::new(
                    "new_texture",
                    Problem::Texture(TextureProblem::TypeForFormat)
                ))),
                "{format:?}"
            );
            let mut multisample = of(
                TextureType::Type2DMultisample,
                [64, 4, 1],
                TextureUsage::RENDER_TARGET,
            );
            multisample.sample_count = 2;
            refused(&multisample, TextureProblem::TypeForFormat);
            refused(
                &of(
                    TextureType::Type2D,
                    [64, 4, 1],
                    rw | TextureUsage::SHADER_ATOMIC,
                ),
                TextureProblem::UsageForFormat,
            );
            // A cube array of more than 341 cubes is stopped before the nil, and a memoryless
            // cube or cube array, which Metal makes, is refused not implemented.
            let mut cubes = of(TextureType::CubeArray, [16, 16, 1], rw);
            cubes.array_length = 342;
            refused(&cubes, TextureProblem::ArrayLength);
            for (ty, length) in [(TextureType::Cube, 1), (TextureType::CubeArray, 341)] {
                let mut d = of(ty, [16, 16, 1], TextureUsage::RENDER_TARGET);
                d.array_length = length;
                d.resource_options = ResourceOptions::new(
                    StorageMode::Memoryless,
                    CpuCacheMode::DefaultCache,
                    HazardTrackingMode::Default,
                );
                match device.new_texture(&d).map(|_| ()) {
                    Err(Error::Unsupported(gap)) => {
                        assert_eq!(gap.feature, Feature::Format, "{format:?} {ty:?}")
                    }
                    other => panic!("{format:?} memoryless {ty:?}: {other:?}"),
                }
            }
        }
        // Every format's cube array stops past 341 cubes.
        let mut cubes = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
        cubes.texture_type = TextureType::CubeArray;
        cubes.resource_options = ResourceOptions::private();
        cubes.array_length = 341;
        device.new_texture(&cubes).unwrap();
        cubes.array_length = 342;
        match device.new_texture(&cubes).map(|_| ()) {
            Err(Error::Invalid(Invalid {
                problem: Problem::Texture(TextureProblem::ArrayLength),
                ..
            })) => {}
            other => panic!("RGBA8Unorm, 342 cubes: {other:?}"),
        }
    });
}
