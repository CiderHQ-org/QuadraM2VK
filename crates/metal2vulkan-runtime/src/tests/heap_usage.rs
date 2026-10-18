//! Metal's heap answers by usage: which usages Metal keeps uncompressed, and the model's plain and
//! compressed answers, against our probe (an Apple M4 under macOS 27.0.1; the
//! consumer's M3 Max rows under 26.7 alike).

use crate::metal_level::MetalLevel;
use crate::texture::heap_tiled::{metal_heap_layout, metal_keeps_uncompressed};
use crate::texture_descriptor::{effective_usage, TextureLayout};
use metal2vulkan::metal::{PixelFormat, PixelFormat as P, TextureType as T, TextureUsage};
use metal2vulkan::vk::Format;

/// The usage bits whose every set the probe asked.
const BITS: [u32; 8] = [0x1, 0x2, 0x4, 0x8, 0x10, 0x4000, 0x8000, 0x10000];

/// The sets ShaderAtomic usage was asked in.
const ATOMIC: [u32; 9] = [0x20, 0x21, 0x23, 0x25, 0x27, 0x31, 0x22, 0x10020, 0x4021];

/// The sets of [`BITS`] Metal compressed a 64 x 64 `R8Unorm` texture of, `allowGPUOptimizedContents`
/// on (every colour format and shape alike); a depth or stencil texture of every set.
#[rustfmt::skip]
const COMPRESSED: [u32; 63] = [
    0x1, 0x4, 0x5, 0x8, 0x9, 0xc, 0xd, 0x4000, 0x4001, 0x4004, 0x4005, 0x4008, 0x4009, 0x400c, 0x400d, 0x8000, 0x8001, 0x8004, 0x8005, 0x8008, 0x8009, 0x800c, 0x800d, 0xc000, 0xc001, 0xc004, 0xc005, 0xc008, 0xc009, 0xc00c, 0xc00d, 0x10000, 0x10001, 0x10004, 0x10005, 0x10008, 0x10009, 0x1000c, 0x1000d, 0x14000, 0x14001, 0x14004, 0x14005, 0x14008, 0x14009, 0x1400c, 0x1400d, 0x18000, 0x18001, 0x18004, 0x18005, 0x18008, 0x18009, 0x1800c, 0x1800d, 0x1c000, 0x1c001, 0x1c004, 0x1c005, 0x1c008, 0x1c009, 0x1c00c, 0x1c00d
];

/// Each probe shape of a format: its type, extent, levels, layers, samples, and Metal's plain
/// (`ShaderRead`, the property off) and compressed (`ShaderRead`, on) size and alignment.
#[rustfmt::skip]
const SHAPES: [(PixelFormat, T, [u32; 2], u32, u32, u32, (u64, u64), (u64, u64)); 180] = [
    (P::R8Unorm, T::Type2D, [64, 64], 1, 1, 1, (0x1000, 0x80), (0x1080, 0x80)),
    (P::R8Unorm, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x2000, 0x80), (0x2100, 0x80)),
    (P::R8Unorm, T::Type2D, [16, 16], 1, 1, 1, (0x100, 0x80), (0x180, 0x80)),
    (P::R8Unorm, T::Type2D, [16, 8], 1, 1, 1, (0x80, 0x80), (0x80, 0x80)),
    (P::R8Unorm, T::Type2D, [8, 8], 1, 1, 1, (0x80, 0x80), (0x80, 0x80)),
    (P::R8Unorm, T::Type2D, [100, 37], 1, 1, 1, (0x2000, 0x80), (0x2100, 0x80)),
    (P::R8Unorm, T::Type2D, [1440, 1080], 1, 1, 1, (0x1b0000, 0x80), (0x1d0000, 0x80)),
    (P::R8Unorm, T::Type2D, [64, 64], 7, 1, 1, (0x1700, 0x80), (0x1880, 0x80)),
    (P::R8Unorm, T::Type2DArray, [64, 64], 1, 3, 1, (0x3000, 0x80), (0x3180, 0x80)),
    (P::R8Unorm, T::Cube, [64, 64], 1, 1, 1, (0x6000, 0x80), (0x6300, 0x80)),
    (P::R8Unorm, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x8000, 0x80), (0x8400, 0x80)),
    (P::R8Unorm, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x200, 0x80), (0x280, 0x80)),
    (P::R8Unorm, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x100, 0x80), (0x180, 0x80)),
    (P::R8Unorm, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x6000, 0x80), (0x6300, 0x80)),
    (P::R8Unorm, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0x330000, 0x80), (0x370000, 0x80)),
    (P::R16Unorm, T::Type2D, [64, 64], 1, 1, 1, (0x2000, 0x80), (0x2080, 0x80)),
    (P::R16Unorm, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x4000, 0x80), (0x4100, 0x80)),
    (P::R16Unorm, T::Type2D, [16, 16], 1, 1, 1, (0x200, 0x80), (0x280, 0x80)),
    (P::R16Unorm, T::Type2D, [16, 8], 1, 1, 1, (0x100, 0x80), (0x100, 0x80)),
    (P::R16Unorm, T::Type2D, [8, 8], 1, 1, 1, (0x80, 0x80), (0x80, 0x80)),
    (P::R16Unorm, T::Type2D, [100, 37], 1, 1, 1, (0x4000, 0x80), (0x4100, 0x80)),
    (P::R16Unorm, T::Type2D, [1440, 1080], 1, 1, 1, (0x330000, 0x80), (0x350000, 0x80)),
    (P::R16Unorm, T::Type2D, [64, 64], 7, 1, 1, (0x2c00, 0x80), (0x2d80, 0x80)),
    (P::R16Unorm, T::Type2DArray, [64, 64], 1, 3, 1, (0x6000, 0x80), (0x6180, 0x80)),
    (P::R16Unorm, T::Cube, [64, 64], 1, 1, 1, (0xc000, 0x80), (0xc300, 0x80)),
    (P::R16Unorm, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x10000, 0x80), (0x10400, 0x80)),
    (P::R16Unorm, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x400, 0x80), (0x480, 0x80)),
    (P::R16Unorm, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x200, 0x80), (0x280, 0x80)),
    (P::R16Unorm, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0xc000, 0x4000), (0xc300, 0x4000)),
    (P::R16Unorm, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0x61c000, 0x80), (0x65c000, 0x80)),
    (P::R32Float, T::Type2D, [64, 64], 1, 1, 1, (0x4000, 0x80), (0x4080, 0x80)),
    (P::R32Float, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x8000, 0x80), (0x8100, 0x80)),
    (P::R32Float, T::Type2D, [16, 16], 1, 1, 1, (0x400, 0x80), (0x480, 0x80)),
    (P::R32Float, T::Type2D, [16, 8], 1, 1, 1, (0x200, 0x80), (0x200, 0x80)),
    (P::R32Float, T::Type2D, [8, 8], 1, 1, 1, (0x100, 0x80), (0x100, 0x80)),
    (P::R32Float, T::Type2D, [100, 37], 1, 1, 1, (0x8000, 0x80), (0x8100, 0x80)),
    (P::R32Float, T::Type2D, [1440, 1080], 1, 1, 1, (0x61c000, 0x80), (0x63c000, 0x80)),
    (P::R32Float, T::Type2D, [64, 64], 7, 1, 1, (0x5680, 0x80), (0x5800, 0x80)),
    (P::R32Float, T::Type2DArray, [64, 64], 1, 3, 1, (0xc000, 0x4000), (0xc180, 0x4000)),
    (P::R32Float, T::Cube, [64, 64], 1, 1, 1, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::R32Float, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x20000, 0x80), (0x20400, 0x80)),
    (P::R32Float, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x800, 0x80), (0x880, 0x80)),
    (P::R32Float, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x400, 0x80), (0x480, 0x80)),
    (P::R32Float, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::R32Float, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0xc38000, 0x80), (0xc78000, 0x80)),
    (P::RGBA8Unorm, T::Type2D, [64, 64], 1, 1, 1, (0x4000, 0x80), (0x4080, 0x80)),
    (P::RGBA8Unorm, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x8000, 0x80), (0x8100, 0x80)),
    (P::RGBA8Unorm, T::Type2D, [16, 16], 1, 1, 1, (0x400, 0x80), (0x480, 0x80)),
    (P::RGBA8Unorm, T::Type2D, [16, 8], 1, 1, 1, (0x200, 0x80), (0x200, 0x80)),
    (P::RGBA8Unorm, T::Type2D, [8, 8], 1, 1, 1, (0x100, 0x80), (0x100, 0x80)),
    (P::RGBA8Unorm, T::Type2D, [100, 37], 1, 1, 1, (0x8000, 0x80), (0x8100, 0x80)),
    (P::RGBA8Unorm, T::Type2D, [1440, 1080], 1, 1, 1, (0x61c000, 0x80), (0x63c000, 0x80)),
    (P::RGBA8Unorm, T::Type2D, [64, 64], 7, 1, 1, (0x5680, 0x80), (0x5800, 0x80)),
    (P::RGBA8Unorm, T::Type2DArray, [64, 64], 1, 3, 1, (0xc000, 0x4000), (0xc180, 0x4000)),
    (P::RGBA8Unorm, T::Cube, [64, 64], 1, 1, 1, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::RGBA8Unorm, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x20000, 0x80), (0x20400, 0x80)),
    (P::RGBA8Unorm, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x800, 0x80), (0x880, 0x80)),
    (P::RGBA8Unorm, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x400, 0x80), (0x480, 0x80)),
    (P::RGBA8Unorm, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::RGBA8Unorm, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0xc38000, 0x80), (0xc78000, 0x80)),
    (P::BGRA8Unorm, T::Type2D, [64, 64], 1, 1, 1, (0x4000, 0x80), (0x4080, 0x80)),
    (P::BGRA8Unorm, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x8000, 0x80), (0x8100, 0x80)),
    (P::BGRA8Unorm, T::Type2D, [16, 16], 1, 1, 1, (0x400, 0x80), (0x480, 0x80)),
    (P::BGRA8Unorm, T::Type2D, [16, 8], 1, 1, 1, (0x200, 0x80), (0x200, 0x80)),
    (P::BGRA8Unorm, T::Type2D, [8, 8], 1, 1, 1, (0x100, 0x80), (0x100, 0x80)),
    (P::BGRA8Unorm, T::Type2D, [100, 37], 1, 1, 1, (0x8000, 0x80), (0x8100, 0x80)),
    (P::BGRA8Unorm, T::Type2D, [1440, 1080], 1, 1, 1, (0x61c000, 0x80), (0x63c000, 0x80)),
    (P::BGRA8Unorm, T::Type2D, [64, 64], 7, 1, 1, (0x5680, 0x80), (0x5800, 0x80)),
    (P::BGRA8Unorm, T::Type2DArray, [64, 64], 1, 3, 1, (0xc000, 0x4000), (0xc180, 0x4000)),
    (P::BGRA8Unorm, T::Cube, [64, 64], 1, 1, 1, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::BGRA8Unorm, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x20000, 0x80), (0x20400, 0x80)),
    (P::BGRA8Unorm, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x800, 0x80), (0x880, 0x80)),
    (P::BGRA8Unorm, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x400, 0x80), (0x480, 0x80)),
    (P::BGRA8Unorm, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::BGRA8Unorm, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0xc38000, 0x80), (0xc78000, 0x80)),
    (P::RGBA16Float, T::Type2D, [64, 64], 1, 1, 1, (0x8000, 0x80), (0x8080, 0x80)),
    (P::RGBA16Float, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x10000, 0x80), (0x10100, 0x80)),
    (P::RGBA16Float, T::Type2D, [16, 16], 1, 1, 1, (0x800, 0x80), (0x880, 0x80)),
    (P::RGBA16Float, T::Type2D, [16, 8], 1, 1, 1, (0x400, 0x80), (0x400, 0x80)),
    (P::RGBA16Float, T::Type2D, [8, 8], 1, 1, 1, (0x200, 0x80), (0x200, 0x80)),
    (P::RGBA16Float, T::Type2D, [100, 37], 1, 1, 1, (0x10000, 0x80), (0x10100, 0x80)),
    (P::RGBA16Float, T::Type2D, [1440, 1080], 1, 1, 1, (0xc38000, 0x80), (0xc58000, 0x80)),
    (P::RGBA16Float, T::Type2D, [64, 64], 7, 1, 1, (0xab80, 0x80), (0xad00, 0x80)),
    (P::RGBA16Float, T::Type2DArray, [64, 64], 1, 3, 1, (0x18000, 0x4000), (0x18180, 0x4000)),
    (P::RGBA16Float, T::Cube, [64, 64], 1, 1, 1, (0x30000, 0x4000), (0x30300, 0x4000)),
    (P::RGBA16Float, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x30000, 0x80), (0x30400, 0x80)),
    (P::RGBA16Float, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x1000, 0x80), (0x1080, 0x80)),
    (P::RGBA16Float, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x800, 0x80), (0x880, 0x80)),
    (P::RGBA16Float, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x30000, 0x4000), (0x30300, 0x4000)),
    (P::RGBA16Float, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0x17e8000, 0x80), (0x1828000, 0x80)),
    (P::RGBA32Float, T::Type2D, [64, 64], 1, 1, 1, (0x10000, 0x80), (0x10080, 0x80)),
    (P::RGBA32Float, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x20000, 0x80), (0x20100, 0x80)),
    (P::RGBA32Float, T::Type2D, [16, 16], 1, 1, 1, (0x1000, 0x80), (0x1080, 0x80)),
    (P::RGBA32Float, T::Type2D, [16, 8], 1, 1, 1, (0x800, 0x80), (0x800, 0x80)),
    (P::RGBA32Float, T::Type2D, [8, 8], 1, 1, 1, (0x400, 0x80), (0x400, 0x80)),
    (P::RGBA32Float, T::Type2D, [100, 37], 1, 1, 1, (0x20000, 0x80), (0x20100, 0x80)),
    (P::RGBA32Float, T::Type2D, [1440, 1080], 1, 1, 1, (0x17e8000, 0x80), (0x1808000, 0x80)),
    (P::RGBA32Float, T::Type2D, [64, 64], 7, 1, 1, (0x15600, 0x80), (0x15780, 0x80)),
    (P::RGBA32Float, T::Type2DArray, [64, 64], 1, 3, 1, (0x30000, 0x4000), (0x30180, 0x4000)),
    (P::RGBA32Float, T::Cube, [64, 64], 1, 1, 1, (0x60000, 0x4000), (0x60300, 0x4000)),
    (P::RGBA32Float, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x54000, 0x80), (0x54400, 0x80)),
    (P::RGBA32Float, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x2000, 0x80), (0x2080, 0x80)),
    (P::RGBA32Float, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x1000, 0x80), (0x1080, 0x80)),
    (P::RGBA32Float, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x60000, 0x4000), (0x60300, 0x4000)),
    (P::RGBA32Float, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0x2fd0000, 0x80), (0x3010000, 0x80)),
    (P::R32Uint, T::Type2D, [64, 64], 1, 1, 1, (0x4000, 0x80), (0x4080, 0x80)),
    (P::R32Uint, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x8000, 0x80), (0x8100, 0x80)),
    (P::R32Uint, T::Type2D, [16, 16], 1, 1, 1, (0x400, 0x80), (0x480, 0x80)),
    (P::R32Uint, T::Type2D, [16, 8], 1, 1, 1, (0x200, 0x80), (0x200, 0x80)),
    (P::R32Uint, T::Type2D, [8, 8], 1, 1, 1, (0x100, 0x80), (0x100, 0x80)),
    (P::R32Uint, T::Type2D, [100, 37], 1, 1, 1, (0x8000, 0x80), (0x8100, 0x80)),
    (P::R32Uint, T::Type2D, [1440, 1080], 1, 1, 1, (0x61c000, 0x80), (0x63c000, 0x80)),
    (P::R32Uint, T::Type2D, [64, 64], 7, 1, 1, (0x5680, 0x80), (0x5800, 0x80)),
    (P::R32Uint, T::Type2DArray, [64, 64], 1, 3, 1, (0xc000, 0x4000), (0xc180, 0x4000)),
    (P::R32Uint, T::Cube, [64, 64], 1, 1, 1, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::R32Uint, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x20000, 0x80), (0x20400, 0x80)),
    (P::R32Uint, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x800, 0x80), (0x880, 0x80)),
    (P::R32Uint, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x400, 0x80), (0x480, 0x80)),
    (P::R32Uint, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::R32Uint, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0xc38000, 0x80), (0xc78000, 0x80)),
    (P::Depth16Unorm, T::Type2D, [64, 64], 1, 1, 1, (0x2000, 0x80), (0x2080, 0x80)),
    (P::Depth16Unorm, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x4000, 0x80), (0x4100, 0x80)),
    (P::Depth16Unorm, T::Type2D, [16, 16], 1, 1, 1, (0x200, 0x80), (0x280, 0x80)),
    (P::Depth16Unorm, T::Type2D, [16, 8], 1, 1, 1, (0x100, 0x80), (0x100, 0x80)),
    (P::Depth16Unorm, T::Type2D, [8, 8], 1, 1, 1, (0x80, 0x80), (0x80, 0x80)),
    (P::Depth16Unorm, T::Type2D, [100, 37], 1, 1, 1, (0x4000, 0x80), (0x4100, 0x80)),
    (P::Depth16Unorm, T::Type2D, [1440, 1080], 1, 1, 1, (0x330000, 0x80), (0x350000, 0x80)),
    (P::Depth16Unorm, T::Type2D, [64, 64], 7, 1, 1, (0x2c00, 0x80), (0x2d80, 0x80)),
    (P::Depth16Unorm, T::Type2DArray, [64, 64], 1, 3, 1, (0x6000, 0x80), (0x6180, 0x80)),
    (P::Depth16Unorm, T::Cube, [64, 64], 1, 1, 1, (0xc000, 0x80), (0xc300, 0x80)),
    (P::Depth16Unorm, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x10000, 0x80), (0x10400, 0x80)),
    (P::Depth16Unorm, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x400, 0x80), (0x480, 0x80)),
    (P::Depth16Unorm, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x200, 0x80), (0x280, 0x80)),
    (P::Depth16Unorm, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0xc000, 0x4000), (0xc300, 0x4000)),
    (P::Depth16Unorm, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0x61c000, 0x80), (0x65c000, 0x80)),
    (P::Depth32Float, T::Type2D, [64, 64], 1, 1, 1, (0x4000, 0x80), (0x4080, 0x80)),
    (P::Depth32Float, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x8000, 0x80), (0x8100, 0x80)),
    (P::Depth32Float, T::Type2D, [16, 16], 1, 1, 1, (0x400, 0x80), (0x480, 0x80)),
    (P::Depth32Float, T::Type2D, [16, 8], 1, 1, 1, (0x200, 0x80), (0x200, 0x80)),
    (P::Depth32Float, T::Type2D, [8, 8], 1, 1, 1, (0x100, 0x80), (0x100, 0x80)),
    (P::Depth32Float, T::Type2D, [100, 37], 1, 1, 1, (0x8000, 0x80), (0x8100, 0x80)),
    (P::Depth32Float, T::Type2D, [1440, 1080], 1, 1, 1, (0x61c000, 0x80), (0x63c000, 0x80)),
    (P::Depth32Float, T::Type2D, [64, 64], 7, 1, 1, (0x5680, 0x80), (0x5800, 0x80)),
    (P::Depth32Float, T::Type2DArray, [64, 64], 1, 3, 1, (0xc000, 0x4000), (0xc180, 0x4000)),
    (P::Depth32Float, T::Cube, [64, 64], 1, 1, 1, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::Depth32Float, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x20000, 0x80), (0x20400, 0x80)),
    (P::Depth32Float, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x800, 0x80), (0x880, 0x80)),
    (P::Depth32Float, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x400, 0x80), (0x480, 0x80)),
    (P::Depth32Float, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x18000, 0x4000), (0x18300, 0x4000)),
    (P::Depth32Float, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0xc38000, 0x80), (0xc78000, 0x80)),
    (P::Stencil8, T::Type2D, [64, 64], 1, 1, 1, (0x1000, 0x80), (0x1080, 0x80)),
    (P::Stencil8, T::Type2DMultisample, [64, 64], 1, 1, 2, (0x2000, 0x80), (0x2100, 0x80)),
    (P::Stencil8, T::Type2D, [16, 16], 1, 1, 1, (0x100, 0x80), (0x180, 0x80)),
    (P::Stencil8, T::Type2D, [16, 8], 1, 1, 1, (0x80, 0x80), (0x80, 0x80)),
    (P::Stencil8, T::Type2D, [8, 8], 1, 1, 1, (0x80, 0x80), (0x80, 0x80)),
    (P::Stencil8, T::Type2D, [100, 37], 1, 1, 1, (0x2000, 0x80), (0x2100, 0x80)),
    (P::Stencil8, T::Type2D, [1440, 1080], 1, 1, 1, (0x1b0000, 0x80), (0x1d0000, 0x80)),
    (P::Stencil8, T::Type2D, [64, 64], 7, 1, 1, (0x1700, 0x80), (0x1880, 0x80)),
    (P::Stencil8, T::Type2DArray, [64, 64], 1, 3, 1, (0x3000, 0x80), (0x3180, 0x80)),
    (P::Stencil8, T::Cube, [64, 64], 1, 1, 1, (0x6000, 0x80), (0x6300, 0x80)),
    (P::Stencil8, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x8000, 0x80), (0x8400, 0x80)),
    (P::Stencil8, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x200, 0x80), (0x280, 0x80)),
    (P::Stencil8, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x100, 0x80), (0x180, 0x80)),
    (P::Stencil8, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x6000, 0x80), (0x6300, 0x80)),
    (P::Stencil8, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0x330000, 0x80), (0x370000, 0x80)),
    (P::Depth32FloatStencil8, T::Type2D, [64, 64], 1, 1, 1, (0x5000, 0x80), (0x9080, 0x80)),
    (P::Depth32FloatStencil8, T::Type2DMultisample, [64, 64], 1, 1, 2, (0xa000, 0x80), (0xe100, 0x80)),
    (P::Depth32FloatStencil8, T::Type2D, [16, 16], 1, 1, 1, (0x4100, 0x80), (0x4180, 0x80)),
    (P::Depth32FloatStencil8, T::Type2D, [16, 8], 1, 1, 1, (0x4080, 0x80), (0x4080, 0x80)),
    (P::Depth32FloatStencil8, T::Type2D, [8, 8], 1, 1, 1, (0x4080, 0x80), (0x4080, 0x80)),
    (P::Depth32FloatStencil8, T::Type2D, [100, 37], 1, 1, 1, (0xa000, 0x80), (0xe100, 0x80)),
    (P::Depth32FloatStencil8, T::Type2D, [1440, 1080], 1, 1, 1, (0x7cc000, 0x80), (0x80c000, 0x80)),
    (P::Depth32FloatStencil8, T::Type2D, [64, 64], 7, 1, 1, (0x9700, 0x80), (0x9880, 0x80)),
    (P::Depth32FloatStencil8, T::Type2DArray, [64, 64], 1, 3, 1, (0xf000, 0x4000), (0x13180, 0x4000)),
    (P::Depth32FloatStencil8, T::Cube, [64, 64], 1, 1, 1, (0x1e000, 0x4000), (0x22300, 0x4000)),
    (P::Depth32FloatStencil8, T::Type2DMultisample, [100, 37], 1, 1, 4, (0x28000, 0x80), (0x2c400, 0x80)),
    (P::Depth32FloatStencil8, T::Type2DMultisample, [16, 16], 1, 1, 2, (0x4200, 0x80), (0x4280, 0x80)),
    (P::Depth32FloatStencil8, T::Type2DMultisample, [8, 8], 1, 1, 4, (0x4100, 0x80), (0x4180, 0x80)),
    (P::Depth32FloatStencil8, T::Type2DMultisampleArray, [64, 64], 1, 3, 2, (0x1e000, 0x4000), (0x22300, 0x4000)),
    (P::Depth32FloatStencil8, T::Type2DMultisample, [1440, 1080], 1, 1, 2, (0xf68000, 0x80), (0xfe8000, 0x80)),
];

/// The image format the runtime plans for `pf`.
fn image_format(pf: PixelFormat) -> Format {
    match pf {
        P::R8Unorm => Format::R8_UNORM,
        P::R16Unorm => Format::R16_UNORM,
        P::R32Float => Format::R32_SFLOAT,
        P::RGBA8Unorm => Format::R8G8B8A8_UNORM,
        P::BGRA8Unorm => Format::B8G8R8A8_UNORM,
        P::RGBA16Float => Format::R16G16B16A16_SFLOAT,
        P::RGBA32Float => Format::R32G32B32A32_SFLOAT,
        P::R32Uint => Format::R32_UINT,
        P::Depth16Unorm => Format::D16_UNORM,
        P::Depth32Float => Format::D32_SFLOAT,
        P::Stencil8 => Format::S8_UINT,
        P::Depth32FloatStencil8 => Format::D32_SFLOAT_S8_UINT,
        other => panic!("not probed: {other:?}"),
    }
}

/// Every set of [`BITS`], and the [`ATOMIC`] sets.
fn usages() -> impl Iterator<Item = u32> {
    (0..1u32 << BITS.len())
        .map(|m| {
            (0..BITS.len())
                .filter(|b| m & (1 << b) != 0)
                .fold(0, |u, b| u | BITS[b])
        })
        .chain(ATOMIC)
}

/// Metal compresses a colour texture of a usage set exactly where it holds none of
/// `ShaderWrite`, `PixelFormatView` and `ShaderAtomic`, `Unknown` (every usage) not included, and
/// a depth or stencil texture of every set; bit 0x8 alone too, which the runtime keeps in a
/// texture's usage.
#[test]
fn metal_keeps_uncompressed_the_usages_metal_does() {
    for raw in usages() {
        let usage = effective_usage(TextureUsage::try_from(raw).unwrap());
        assert!(!metal_keeps_uncompressed(false, usage), "{raw:#x}");
        assert_eq!(
            !metal_keeps_uncompressed(true, usage),
            COMPRESSED.contains(&raw),
            "{raw:#x}"
        );
    }
}

/// Where the model answers, it answers Metal's plain size for a texture Metal keeps
/// uncompressed and its compressed size for one it compresses, at every probe shape.
#[test]
fn the_model_answers_metal_s_sizes_by_usage() {
    let mut unanswered = Vec::new();
    for (pf, ty, [w, h], levels, length, samples, plain, compressed) in SHAPES {
        let faces = if ty == T::Cube { 6 } else { 1 };
        let layout = TextureLayout {
            texture_type: ty,
            pixel_format: pf,
            extent: [w, h, 1],
            levels,
            layers: length * faces,
            samples,
        };
        let colour = !metal2vulkan::metal::format::is_depth_stencil(pf);
        let format = image_format(pf);
        for raw in [
            0x0, 0x1, 0x2, 0x5, 0x10, 0x4000, 0x8000, 0x10000, 0x10002, 0x20, 0x21,
        ] {
            let usage = effective_usage(TextureUsage::try_from(raw).unwrap());
            for optimized in [false, true] {
                let what = format!(
                    "{pf:?} {ty:?} {w}x{h} {levels} {length} x{samples} {raw:#x} {optimized}"
                );
                let metal =
                    metal_heap_layout(&layout, usage, optimized, format, MetalLevel::MacOS27_0_1);
                let Some(answer) = metal.answer else {
                    unanswered.push(what);
                    continue;
                };
                let squeezed = optimized && !metal_keeps_uncompressed(colour, usage);
                let want = if squeezed { compressed } else { plain };
                assert_eq!((answer.size, answer.align), want, "{what}");
            }
        }
    }
    println!("{unanswered:#?}");
}
