//! Textures over buffers (`linear_plan`): which pixel formats a buffer can back and their
//! alignment, where Metal lays a texture out in its buffer and which textures it refuses, as
//! probes 9 to 9f measured them on an Apple M3 Max (macOS 26.7) and an M4 (27.0.1).

use super::super::image_plan::plan_image;
use super::super::linear_plan::{
    aliases, check_linear_limits, holding, linear_alignment, linear_lack, plan_buffer_texture,
    BufferFacts, BufferLayout, BufferRefusal, BufferTexture, BufferTextureCall,
    BufferTextureProblem, LinearImage, LinearLack, Placed, LINEAR_ALIGNMENT,
};
use super::super::texture_descriptor::{effective_usage, no_texture_alone, TextureDescriptor};
use super::super::values::StorageMode;
use super::image_plan::{committed, facts};
use super::texture_descriptor::APPLE;
use crate::MetalLevel;
use metal2vulkan::metal::format::direct_format;
use metal2vulkan::metal::PixelFormat as P;
use metal2vulkan::metal::TextureType;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan::vk::{Format, FormatFeatureFlags2, ImageCreateFlags, ImageUsageFlags};

/// The formats for which Metal answered 16 (the linear and the texture-buffer alignment alike);
/// `ARGB8Unorm` and the two-plane extended-range formats, which are not in Metal's public header,
/// were asked on the M3 Max.
const ANSWERED: [P; 59] = [
    P::A8Unorm,
    P::R8Unorm,
    P::R8UnormSRGB,
    P::R8Snorm,
    P::R8Uint,
    P::R8Sint,
    P::R16Unorm,
    P::R16Snorm,
    P::R16Uint,
    P::R16Sint,
    P::R16Float,
    P::RG8Unorm,
    P::RG8UnormSRGB,
    P::RG8Snorm,
    P::RG8Uint,
    P::RG8Sint,
    P::B5G6R5Unorm,
    P::A1BGR5Unorm,
    P::ABGR4Unorm,
    P::BGR5A1Unorm,
    P::R32Uint,
    P::R32Sint,
    P::R32Float,
    P::RG16Unorm,
    P::RG16Snorm,
    P::RG16Uint,
    P::RG16Sint,
    P::RG16Float,
    P::RGBA8Unorm,
    P::RGBA8UnormSRGB,
    P::RGBA8Snorm,
    P::RGBA8Uint,
    P::RGBA8Sint,
    P::BGRA8Unorm,
    P::BGRA8UnormSRGB,
    P::RGB10A2Unorm,
    P::RGB10A2Uint,
    P::RG11B10Float,
    P::RGB9E5Float,
    P::BGR10A2Unorm,
    P::BGRA10XR,
    P::BGRA10XRSRGB,
    P::BGR10XR,
    P::BGR10XRSRGB,
    P::ARGB8Unorm,
    P::RGB10A82PXR10,
    P::RGB10A82PXR10SRGB,
    P::RG32Uint,
    P::RG32Sint,
    P::RG32Float,
    P::RGBA16Unorm,
    P::RGBA16Snorm,
    P::RGBA16Uint,
    P::RGBA16Sint,
    P::RGBA16Float,
    P::RGBA32Uint,
    P::RGBA32Sint,
    P::RGBA32Float,
    P::Unspecialized,
];

/// The added linear formats for which an Apple M4 under macOS 27.0.1 answered 16, the linear and
/// the texture-buffer alignment alike; it asserted on the other 18, the formats
/// macOS 27.0.1 describes none of (`MetalLevel::describes`).
const ANSWERED_M4: [P; 62] = [
    P::R16Bfloat,
    P::BGRA4Unorm,
    P::RG16Bfloat,
    P::ARGB8UnormSRGB,
    P::RGBA16Bfloat,
    P::RGBX8Unorm,
    P::RGBX8UnormSRGB,
    P::RGBX8Snorm,
    P::RGBX8Uint,
    P::RGBX8Sint,
    P::RGBX16Uint,
    P::RGBX16Sint,
    P::RGBX16Float,
    P::RGBX32Uint,
    P::RGBX32Sint,
    P::RGBX32Float,
    P::YCBCR84221P,
    P::YCBCR104441P,
    P::YCBCR84221PSRGB,
    P::YCBCR104441PSRGB,
    P::YCBCRA84441P,
    P::YCBCR84221PREV,
    P::YCBCR104441PPQ,
    P::R10UnormX6,
    P::R10UnormX6SRGB,
    P::RG10UnormX12,
    P::RG10UnormX12SRGB,
    P::RGB10A2UnormSRGB,
    P::RGB10A2UnormPQ,
    P::R10UnormPACKED,
    P::RG10UnormPACKED,
    P::YCBCR104441PXR,
    P::R12UnormX4,
    P::R12UnormX4PQ,
    P::RG12UnormX8,
    P::R10UnormX6PQ,
    P::YCBCR104441PHLG,
    P::R10UnormX6HLG,
    P::R12UnormX4HLG,
    P::RGB10A2UnormHLG,
    P::RGB10A82P,
    P::RGB10A82PSRGB,
    P::RGB10A82PPQ,
    P::RGB10A82PHLG,
    P::BGR10A2UnormSRGB,
    P::BGR10A2UnormPQ,
    P::BGRA10UintPACKED,
    P::BGRA10XRPACKED,
    P::BGRA10XRPACKEDSRGB,
    P::BGRA10UnormPACKED,
    P::BGRA10UnormPACKEDSRGB,
    P::BGRA10UnormPACKEDHLG,
    P::BGRA10UnormPACKEDPQ,
    P::BGRA10Uint,
    P::BGRA10Unorm,
    P::BGRA10UnormSRGB,
    P::BGRA10UnormHLG,
    P::BGRA10UnormPQ,
    P::R10UintX6,
    P::R10UintPACKED,
    P::RG10UintX12,
    P::RG10UintPACKED,
];

/// The formats of [`ANSWERED_M4`] an Apple M3 Max asserted on, under macOS 26.7 and 27.0.1 alike:
/// the 16 from 630 to 645, which it describes none of, nor does either Metal level.
const ASSERTED_M3: [P; 16] = [
    P::BGRA10UintPACKED,
    P::BGRA10XRPACKED,
    P::BGRA10XRPACKEDSRGB,
    P::BGRA10UnormPACKED,
    P::BGRA10UnormPACKEDSRGB,
    P::BGRA10UnormPACKEDHLG,
    P::BGRA10UnormPACKEDPQ,
    P::BGRA10Uint,
    P::BGRA10Unorm,
    P::BGRA10UnormSRGB,
    P::BGRA10UnormHLG,
    P::BGRA10UnormPQ,
    P::R10UintX6,
    P::R10UintPACKED,
    P::RG10UintX12,
    P::RG10UintPACKED,
];

/// Metal answers 16 for exactly the formats it answered for; it asserts on every other one
/// (compressed formats, the 4:2:2 ones among them, and depth and stencil formats; on the M3 and
/// M4, `Depth24Unorm_Stencil8` and `X24_Stencil8` as unsupported formats), which the runtime
/// refuses, at each Metal level the M3 Max's (at macOS 27.0.1: of the raw values
/// from 0 to 700, 175 answered and every other one asserted, each as the runtime answers).
/// `Invalid` was not asked.
#[test]
fn buffers_back_exactly_the_formats_metal_answers_for() {
    // Probe 9 took none of the formats Metal makes no texture of alone (the 69 two-plane YCbCr
    // formats and three more): a buffer's texture of one is refused before this rule
    // (`texture_over`).
    let unprobed = P::ALL.iter().filter(|&&pf| no_texture_alone(pf)).count();
    assert_eq!(unprobed, 72);
    for level in MetalLevel::ALL {
        for &pf in P::ALL {
            if pf == P::Invalid || no_texture_alone(pf) {
                continue;
            }
            let asserted = ASSERTED_M3.contains(&pf);
            let want = if ANSWERED.contains(&pf) || (ANSWERED_M4.contains(&pf) && !asserted) {
                Ok(LINEAR_ALIGNMENT)
            } else {
                Err(BufferTextureProblem::NotLinear { pixel_format: pf })
            };
            assert_eq!(linear_alignment(pf, level), want, "{level:?} {pf:?}");
        }
    }
    // Nor ASTC's 4x2 and 8x4 formats, which the runtime refuses as the compressed formats they
    // are (their type rules are measured as `ASTC_4x4_LDR`'s).
    let astc = [
        P::ASTC4x2LDR,
        P::ASTC4x2SRGB,
        P::ASTC4x2HDR,
        P::ASTC8x4LDR,
        P::ASTC8x4SRGB,
        P::ASTC8x4HDR,
    ];
    // + the 18 added linear formats Metal asserted on.
    assert_eq!(
        P::ALL.len() - 1 - unprobed - astc.len() - ANSWERED.len() - ANSWERED_M4.len(),
        83 + 18,
        "one probe saw 83 formats refused, another 18 more"
    );
    for level in MetalLevel::ALL {
        assert_eq!(
            linear_alignment(P::Invalid, level),
            Err(BufferTextureProblem::NotLinear {
                pixel_format: P::Invalid
            })
        );
        // Nor `AnyDepth` and `AnyStencil`, of which Metal makes no texture alone: it asserts as on
        // every depth and stencil format ("Linear textures do not support
        // depth/stencil pixel formats").
        for pf in [P::AnyDepth, P::AnyStencil] {
            assert_eq!(
                linear_alignment(pf, level),
                Err(BufferTextureProblem::NotLinear { pixel_format: pf }),
                "{level:?} {pf:?}"
            );
        }
    }
}

/// One texture over a buffer: the call, the descriptor's type, format, width, height, slices,
/// levels and storage mode (`Shared` by default), the offset, `bytesPerRow`, and the buffer's
/// length.
#[derive(Clone, Copy, Debug)]
struct Case {
    call: BufferTextureCall,
    texture_type: TextureType,
    pixel_format: P,
    width: u64,
    height: u64,
    slices: u64,
    levels: u64,
    offset: u64,
    bytes_per_row: u64,
    length: u64,
    storage: StorageMode,
}

/// A 2D RGBA8Unorm texture through the public call.
fn texture(width: u64, height: u64, offset: u64, bytes_per_row: u64, length: u64) -> Case {
    Case {
        call: BufferTextureCall::Texture,
        texture_type: TextureType::Type2D,
        pixel_format: P::RGBA8Unorm,
        width,
        height,
        slices: 1,
        levels: 1,
        offset,
        bytes_per_row,
        length,
        storage: StorageMode::Shared,
    }
}

/// A 2D array of RGBA8Unorm through the SPI.
fn array(
    (width, height, slices): (u64, u64, u64),
    offset: u64,
    bytes_per_row: u64,
    length: u64,
) -> Case {
    Case {
        call: BufferTextureCall::Linear,
        texture_type: TextureType::Type2DArray,
        slices,
        ..texture(width, height, offset, bytes_per_row, length)
    }
}

/// A texture buffer of RGBA8Unorm through the public call.
fn texture_buffer(width: u64, offset: u64, bytes_per_row: u64, length: u64) -> Case {
    Case {
        texture_type: TextureType::TextureBuffer,
        ..texture(width, 1, offset, bytes_per_row, length)
    }
}

fn plan(case: Case) -> Result<BufferLayout, BufferRefusal> {
    placed(case).map(|placed| {
        // Only an IOSurface plane's CPU layout differs from its GPU one.
        assert_eq!(placed.gpu, placed.cpu, "{case:?}");
        placed.gpu
    })
}

/// Where `case` lies for the GPU and the CPU.
fn placed(case: Case) -> Result<Placed, BufferRefusal> {
    let layout = TextureDescriptor {
        texture_type: case.texture_type,
        pixel_format: case.pixel_format,
        width: case.width,
        height: case.height,
        depth: if case.texture_type == TextureType::Type3D {
            2
        } else {
            1
        },
        array_length: case.slices,
        mipmap_level_count: case.levels,
        sample_count: if case.texture_type == TextureType::Type2DMultisample {
            4
        } else {
            1
        },
        ..TextureDescriptor::default()
    }
    .check(APPLE)
    .unwrap_or_else(|problem| panic!("{case:?}: {problem:?}"));
    let texel_bytes = direct_format(case.pixel_format)
        .and_then(|f| f.block())
        .map_or(0, |block| u64::from(block.size));
    plan_buffer_texture(
        &BufferTexture {
            layout,
            texel_bytes,
            storage: case.storage,
            offset: case.offset,
            bytes_per_row: case.bytes_per_row,
            call: case.call,
        },
        &BufferFacts {
            length: case.length,
        },
    )
}

fn refused(problem: BufferTextureProblem) -> Result<BufferLayout, BufferRefusal> {
    Err(BufferRefusal::Metal(problem))
}

/// The probes' 2D textures and texture buffers through the public call: what Metal made, and
/// what it asserted on or stopped under validation.
#[test]
fn two_dimensional_textures_and_texture_buffers_follow_metal() {
    use BufferTextureProblem as Pr;
    let made = |offset, row_pitch, rows, row_bytes| {
        Ok(BufferLayout {
            offset,
            row_pitch,
            slice_pitch: row_pitch * rows,
            row_bytes,
            rows,
            slices: 1,
        })
    };
    let cases = [
        // Probe 9's base, and bytesPerRow below the packed row of 8 RGBA8 texels.
        (texture(8, 4, 0, 32, 4096), made(0, 32, 4, 32)),
        (
            texture(8, 4, 0, 16, 4096),
            refused(Pr::RowPitch {
                bytes_per_row: 16,
                row_bytes: 32,
            }),
        ),
        (
            texture(8, 4, 0, 0, 4096),
            refused(Pr::RowPitch {
                bytes_per_row: 0,
                row_bytes: 32,
            }),
        ),
        // The length: the last row's packed bytes may end on the buffer's end.
        (texture(8, 4, 3872, 64, 4096), made(3872, 64, 4, 32)),
        (
            texture(8, 4, 3888, 64, 4096),
            refused(Pr::PastBuffer {
                counted: 4112,
                length: 4096,
            }),
        ),
        (texture(8, 4, 3856, 64, 4096), made(3856, 64, 4, 32)),
        (texture(8, 4, 3920, 48, 4096), made(3920, 48, 4, 32)),
        (
            texture(4, 1, 4096, 16, 4096),
            refused(Pr::PastBuffer {
                counted: 4112,
                length: 4096,
            }),
        ),
        // Offsets and pitches are multiples of 16.
        (
            texture(4, 4, 8, 16, 4096),
            refused(Pr::OffsetAlignment {
                offset: 8,
                alignment: 16,
            }),
        ),
        (
            texture(5, 4, 0, 24, 4096),
            refused(Pr::RowPitchAlignment {
                bytes_per_row: 24,
                alignment: 16,
            }),
        ),
        // One level, and no other type.
        (
            Case {
                levels: 2,
                ..texture(8, 4, 0, 32, 4096)
            },
            refused(Pr::MipmapLevelCount { count: 2 }),
        ),
        (
            array((4, 4, 1), 0, 16, 4096),
            Ok(BufferLayout {
                slice_pitch: 128,
                ..made(0, 16, 4, 16).unwrap()
            }),
        ),
        (
            Case {
                call: BufferTextureCall::Texture,
                ..array((4, 4, 1), 0, 16, 4096)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Type2DArray,
            }),
        ),
        (
            Case {
                texture_type: TextureType::Type2DMultisample,
                ..texture(4, 4, 0, 16, 4096)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Type2DMultisample,
            }),
        ),
        (
            Case {
                texture_type: TextureType::Cube,
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 0, 16, 4096)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Cube,
            }),
        ),
        (
            Case {
                texture_type: TextureType::Type3D,
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 0, 16, 4096)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Type3D,
            }),
        ),
        // A compressed format.
        (
            Case {
                pixel_format: P::BC1RGBA,
                ..texture(4, 4, 0, 16, 4096)
            },
            refused(Pr::NotLinear {
                pixel_format: P::BC1RGBA,
            }),
        ),
        // Texture buffers: any offset (one off 16, which Metal's validation alone stops, its
        // functions read from the offset rounded down), any bytesPerRow that covers the row, and a
        // length bound of the packed row. The SPI's keeps an offset of 16.
        (texture_buffer(64, 16, 256, 4096), made(16, 256, 1, 256)),
        (texture_buffer(64, 16, 260, 4096), made(16, 260, 1, 256)),
        (texture_buffer(64, 3840, 512, 4096), made(3840, 512, 1, 256)),
        (texture_buffer(64, 4, 256, 4096), made(4, 256, 1, 256)),
        (texture_buffer(64, 1, 256, 4096), made(1, 256, 1, 256)),
        (texture_buffer(64, 20, 256, 4096), made(20, 256, 1, 256)),
        (
            Case {
                call: BufferTextureCall::Linear,
                ..texture_buffer(64, 4, 256, 4096)
            },
            refused(Pr::OffsetAlignment {
                offset: 4,
                alignment: 16,
            }),
        ),
        (
            texture_buffer(64, 16, 128, 4096),
            refused(Pr::RowPitch {
                bytes_per_row: 128,
                row_bytes: 256,
            }),
        ),
        (
            texture_buffer(64, 3856, 256, 4096),
            refused(Pr::PastBuffer {
                counted: 4112,
                length: 4096,
            }),
        ),
        (
            texture_buffer(1024, 16, 4096, 4096),
            refused(Pr::PastBuffer {
                counted: 4112,
                length: 4096,
            }),
        ),
        // Through the SPI, a 2D texture keeps the 2D rules.
        (
            Case {
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 16, 16, 16384)
            },
            made(16, 16, 4, 16),
        ),
        (
            Case {
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 0, 32, 111)
            },
            refused(Pr::PastBuffer {
                counted: 112,
                length: 111,
            }),
        ),
        (
            Case {
                call: BufferTextureCall::Linear,
                ..texture(4, 4, 0, 32, 112)
            },
            made(0, 32, 4, 16),
        ),
    ];
    for (case, want) in cases {
        assert_eq!(plan(case), want, "{case:?}");
    }
}

/// Metal's resource-option rules: a texture over a buffer takes the buffer's options whatever
/// its descriptor names, through the public call and the SPI (probe 9d; `Buffer::texture_over`),
/// and an IOSurface plane its descriptor's, so the plan refuses no storage mode but
/// a plane's `Memoryless`, whose use is unmeasured.
#[test]
fn only_a_memoryless_plane_is_refused_its_storage_mode() {
    use StorageMode::{Managed, Memoryless, Private, Shared};
    let calls = [
        BufferTextureCall::Texture,
        BufferTextureCall::Linear,
        BufferTextureCall::IoSurfacePlane { planar: true },
        BufferTextureCall::IoSurfacePlane { planar: false },
    ];
    for call in calls {
        for storage in [Shared, Managed, Private, Memoryless] {
            let case = Case {
                call,
                storage,
                ..texture(8, 4, 0, 32, 4096)
            };
            let surface = matches!(call, BufferTextureCall::IoSurfacePlane { .. });
            let want = if surface && storage == Memoryless {
                Err(BufferRefusal::Metal(BufferTextureProblem::Options))
            } else {
                Ok(())
            };
            assert_eq!(placed(case).map(|_| ()), want, "{call:?}: {storage:?}");
        }
    }
}

/// A 2D array's layout through the SPI (probes 9b, 9c and 9e): offsets at multiples of 128,
/// `bytesPerRow` at multiples of 16, and each slice `bytesPerRow × height` rounded up to 128
/// after the one before, whatever `bytesPerImage` is (the plan takes none).
#[test]
fn an_arrays_slices_start_at_metals_own_stride() {
    use BufferTextureProblem as Pr;
    // (width, height, bytesPerRow) and the slice stride Metal used.
    let strides = [
        ((4, 4, 16), 128),
        ((4, 3, 16), 128),
        ((4, 5, 16), 128),
        ((4, 4, 32), 128),
        ((4, 4, 48), 256),
        ((4, 4, 64), 256),
        ((4, 3, 64), 256),
        ((4, 4, 128), 512),
        ((4, 4, 256), 1024),
        ((5, 4, 32), 128),
        ((64, 2, 256), 512),
        ((64, 2, 272), 640),
    ];
    for ((w, h, bpr), stride) in strides {
        for offset in [0, 128, 256] {
            let case = array((w, h, 2), offset, bpr, 16384);
            let layout = plan(case).unwrap_or_else(|e| panic!("{case:?}: {e:?}"));
            assert_eq!(layout.slice_pitch, stride, "{case:?}");
            assert_eq!(layout.offset, offset);
        }
    }
    // RGBA32Float's 16-byte texels at bytesPerRow 64 and height 5: 320 rounds up to 384.
    let wide = Case {
        pixel_format: P::RGBA32Float,
        ..array((4, 5, 3), 128, 64, 16384)
    };
    assert_eq!(plan(wide).unwrap().slice_pitch, 384);
    for offset in [16, 64, 4] {
        assert_eq!(
            plan(array((4, 4, 2), offset, 16, 16384)),
            refused(Pr::OffsetAlignment {
                offset,
                alignment: 128,
            })
        );
    }
    for bytes_per_row in [20, 24, 28] {
        assert_eq!(
            plan(array((5, 4, 2), 0, bytes_per_row, 16384)),
            refused(Pr::RowPitchAlignment {
                bytes_per_row,
                alignment: 16,
            })
        );
    }
    assert_eq!(
        plan(Case {
            levels: 2,
            ..array((4, 4, 2), 0, 16, 16384)
        }),
        refused(Pr::MipmapLevelCount { count: 2 })
    );
}

/// Where an array may end (probes 9b, 9c, 9e and 9f): validation counts its rows as if its
/// slices were not rounded; Metal answers nil where a buffer of at least 16 KiB has the rounded
/// slices end past its `allocatedSize`; elsewhere it makes an array whose last slice runs past
/// the length, which the runtime refuses as not implemented.
#[test]
fn an_array_ends_where_metal_lets_it() {
    use BufferTextureProblem as Pr;
    let past =
        |counted, length| Err::<(), _>(BufferRefusal::Metal(Pr::PastBuffer { counted, length }));
    let nil = |end, allocation| {
        Err::<(), _>(BufferRefusal::Metal(Pr::PastAllocation { end, allocation }))
    };
    let gap = Err::<(), _>(BufferRefusal::TexelsPastLength);
    let ok = |case: Case| plan(case).map(|_| ());
    let cases = [
        // Two 4x4 slices at bytesPerRow 16 from 16128 (a 64-byte image, a 128-byte stride).
        (array((4, 4, 2), 16128, 16, 16255), past(16256, 16255)),
        (array((4, 4, 2), 16128, 16, 16256), gap),
        (array((4, 4, 2), 16128, 16, 16319), gap),
        (array((4, 4, 2), 16128, 16, 16320), Ok(())),
        (array((4, 4, 2), 16128, 16, 16384), Ok(())),
        // From 16256 the slices end at 16512: nil in 16384 bytes, made in 16385 or more.
        (array((4, 4, 2), 16256, 16, 16384), nil(16512, 16384)),
        (array((4, 4, 2), 16256, 16, 16385), gap),
        (array((4, 4, 2), 16256, 16, 16512), Ok(())),
        (array((4, 4, 2), 16384, 16, 16385), past(16512, 16385)),
        (array((4, 4, 2), 32640, 16, 32768), nil(32896, 32768)),
        // Small buffers: made past the length whatever the rounded slices' end.
        (array((4, 4, 2), 0, 16, 64), past(128, 64)),
        (array((4, 4, 2), 0, 16, 127), past(128, 127)),
        (array((4, 4, 2), 0, 16, 128), gap),
        (array((4, 4, 2), 0, 16, 191), gap),
        (array((4, 4, 2), 0, 16, 192), Ok(())),
        (array((4, 4, 3), 0, 16, 300), gap),
        (array((4, 4, 64), 0, 16, 4096), gap),
        (array((4, 4, 128), 0, 16, 4096), past(8192, 4096)),
        (array((4, 4, 129), 0, 16, 4096), past(8256, 4096)),
        // bytesPerRow above the packed row: validation counts the last row's packed bytes.
        (array((4, 4, 2), 0, 32, 239), past(240, 239)),
        (array((4, 4, 2), 0, 32, 240), Ok(())),
        (array((4, 3, 3), 0, 48, 399), past(400, 399)),
        (array((4, 3, 3), 0, 48, 400), gap),
        (array((4, 3, 3), 0, 48, 432), gap),
    ];
    for (case, want) in cases {
        assert_eq!(ok(case), want, "{case:?}");
    }
}

/// A linear image aliases a texture's place in its buffer only where every one of its reports
/// is Metal's layout's, and its subresources hold no byte but the texture's texels.
#[test]
fn a_linear_image_aliases_only_metals_layout() {
    let single = BufferLayout {
        offset: 256,
        row_pitch: 80,
        slice_pitch: 320,
        row_bytes: 80,
        rows: 4,
        slices: 1,
    };
    let image = LinearImage {
        alignment: 16,
        size: 320,
        offset: 0,
        subresource: 320,
        row_pitch: 80,
        array_pitch: 320,
    };
    assert!(aliases(&single, 4096 + 256, 320, &image));
    // The binding off the image's alignment (RADV's 256 at a 16-byte offset).
    assert!(!aliases(
        &single,
        4096 + 272,
        320,
        &LinearImage {
            alignment: 256,
            ..image
        }
    ));
    // The driver's own pitch where another was asked for (NVIDIA's explicit layouts).
    assert!(!aliases(
        &single,
        4096,
        320,
        &LinearImage {
            row_pitch: 128,
            ..image
        }
    ));
    // Texels not at the binding, or memory past the buffer's.
    assert!(!aliases(
        &single,
        4096,
        320,
        &LinearImage {
            offset: 64,
            ..image
        }
    ));
    assert!(!aliases(&single, 4096, 319, &image));
    // Memory past the subresource, within the buffer's: no write through the image reaches it.
    assert!(aliases(
        &single,
        4096,
        4096,
        &LinearImage {
            size: 4096,
            ..image
        }
    ));
    // A subresource past the texels (lavapipe's rows rounded up to 4, of a texture of 3).
    let three = BufferLayout { rows: 3, ..single };
    assert!(!aliases(&three, 4096, 320, &image));
    assert!(aliases(
        &three,
        4096,
        320,
        &LinearImage {
            subresource: 240,
            ..image
        }
    ));
    // Rows with the buffer's bytes between them, at Metal's pitch and the image's alike.
    let padded = BufferLayout {
        row_bytes: 68,
        ..single
    };
    assert!(!aliases(&padded, 4096, 320, &image));
    // Whatever its usage: a texture that only reads takes a first move over them too.
    for usage in [
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ,
    ] {
        assert!(holding(&padded, usage).is_err(), "{usage:?}");
        assert_eq!(holding(&single, usage), Ok(()), "{usage:?}");
    }
    // One row has nothing between rows, but its pitch's padding past the texels.
    let row = BufferLayout {
        rows: 1,
        slice_pitch: 80,
        ..padded
    };
    assert_eq!(holding(&row, TextureUsage::SHADER_READ), Ok(()));
    assert!(!aliases(
        &row,
        4096,
        80,
        &LinearImage {
            size: 80,
            subresource: 80,
            ..image
        }
    ));
    assert!(aliases(
        &row,
        4096,
        80,
        &LinearImage {
            size: 80,
            subresource: 68,
            ..image
        }
    ));
    // An array's slices at Metal's stride, and at no other.
    let slices = BufferLayout {
        slices: 3,
        ..single
    };
    let stacked = LinearImage { size: 960, ..image };
    assert!(aliases(&slices, 4096, 960, &stacked));
    assert!(!aliases(
        &slices,
        4096,
        960,
        &LinearImage {
            array_pitch: 384,
            ..stacked
        }
    ));
    // Slices with the buffer's bytes between them (an array's slices rounded up to 128).
    let apart = BufferLayout {
        slice_pitch: 384,
        ..slices
    };
    for usage in [
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ,
    ] {
        assert!(holding(&apart, usage).is_err(), "{usage:?}");
    }
    assert!(!aliases(
        &apart,
        4096,
        1088,
        &LinearImage {
            size: 1088,
            array_pitch: 384,
            ..image
        }
    ));
    // A driver that reports the array's whole bytes as its first subresource's.
    assert!(!aliases(
        &slices,
        4096,
        960,
        &LinearImage {
            subresource: 960,
            ..stacked
        }
    ));
    // A single slice ignores the array pitch a driver reports.
    assert!(aliases(
        &single,
        4096,
        320,
        &LinearImage {
            array_pitch: 0,
            ..image
        }
    ));
}

/// A texture over padded rows is a shadow, whatever its usage, naming the operations Vulkan lets
/// write the padding: a render target its image's layout transitions (its first move among them)
/// and a render pass's load and store operations; any other (one that only reads, or that a
/// shader writes) its layout transitions alone. Its usage as Metal reports it decides, `Unknown`
/// a render target's.
#[test]
fn a_padded_texture_s_shadow_names_what_may_write_its_padding() {
    let padded = BufferLayout {
        offset: 0,
        row_pitch: 80,
        slice_pitch: 320,
        row_bytes: 68,
        rows: 4,
        slices: 1,
    };
    let transitions = "its rows lie 80 bytes apart, 68 bytes of texels each: an image over them \
                       holds the bytes between them, which Vulkan lets its layout transitions \
                       read and write, its first move from the layout it is made in among them";
    let passes = "its rows lie 80 bytes apart, 68 bytes of texels each: an image over them holds \
                  the bytes between them, which Vulkan lets its layout transitions read and write, \
                  and a render pass's load and store operations write back";
    for usage in [
        TextureUsage::SHADER_READ,
        TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW,
        TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        TextureUsage::SHADER_WRITE,
        TextureUsage::SHADER_ATOMIC,
        TextureUsage::BLOCK_WRITES_ONLY,
    ] {
        assert_eq!(
            holding(&padded, usage),
            Err(transitions.to_owned()),
            "{usage:?}"
        );
    }
    for usage in [
        TextureUsage::RENDER_TARGET,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        effective_usage(TextureUsage::UNKNOWN),
    ] {
        assert_eq!(holding(&padded, usage), Err(passes.to_owned()), "{usage:?}");
    }
}

/// A linear image serves a texture where its tiling has every feature the optimal one has for
/// the texture's uses, in every format it is viewed as; the formats a device's linear tiling
/// lacks send the texture to a shadow image.
#[test]
fn a_linear_image_serves_only_what_its_tiling_holds() {
    let mut served = 0;
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        for usage in [
            TextureUsage::SHADER_READ,
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        ] {
            let layout = TextureDescriptor {
                pixel_format: P::RGBA8Unorm,
                width: 8,
                height: 4,
                usage,
                ..TextureDescriptor::default()
            }
            .check(APPLE)
            .unwrap();
            let Ok(plan) = plan_image(&layout, usage, false, &facts) else {
                continue;
            };
            let serves = linear_lack(&plan, &facts).is_none();
            served += usize::from(serves);
            // With the linear tiling of the image's format stripped, nothing is served.
            let mut bare = desc.formats.clone();
            if let Some(p) = bare.formats.get_mut(&plan.format) {
                p.linear_tiling = FormatFeatureFlags2::empty();
            }
            let stripped = super::super::image_plan::ImageFacts {
                formats: &bare,
                ..facts
            };
            assert!(linear_lack(&plan, &stripped).is_some(), "{name} {usage:?}");
            // A plan of extended usage is checked as any plan is: served exactly where the plan
            // without it is, and not with the image's format's tiling stripped.
            let mutable = super::super::image_plan::ImagePlan {
                flags: plan.flags | ImageCreateFlags::MUTABLE_FORMAT,
                ..plan.clone()
            };
            let extended = super::super::image_plan::ImagePlan {
                flags: mutable.flags | ImageCreateFlags::EXTENDED_USAGE,
                ..plan.clone()
            };
            assert_eq!(
                linear_lack(&extended, &facts).is_none(),
                linear_lack(&mutable, &facts).is_none(),
                "{name} {usage:?}"
            );
            assert!(
                !linear_lack(&extended, &stripped).is_none(),
                "{name} {usage:?}"
            );
            // The devices measured sample and store RGBA8Unorm through linear images.
            if MEASURED.contains(&name.as_str()) {
                assert!(serves, "{name}: {:?}", facts.linear(Format::R8G8B8A8_UNORM));
            }
        }
    }
    assert!(served > 0);
}

/// What a linear image lacks is named exactly (a heap texture's linear image): with a format's
/// linear tiling stripped, the features its optimal tiling has for the texture's usage; a texture
/// of several levels also relies on the blit and linear-filter features its mipmaps need, and one
/// level does not; views of any format are checked in every format a view may take; and extended
/// usage format by format, each format asked only what its optimal tiling takes.
#[test]
fn a_linear_image_names_what_its_tiling_lacks() {
    use FormatFeatureFlags2 as F;
    let (mut named, mut any_checked, mut extended_checked) = (0, 0, 0);
    for (name, desc, device) in committed() {
        let facts = facts(&desc, &device);
        let plan_of = |levels: u64| {
            let layout = TextureDescriptor {
                pixel_format: P::RGBA8Unorm,
                width: 64,
                height: 64,
                mipmap_level_count: levels,
                usage: TextureUsage::SHADER_READ,
                ..TextureDescriptor::default()
            }
            .check(APPLE)
            .unwrap();
            plan_image(&layout, TextureUsage::SHADER_READ, false, &facts).ok()
        };
        let (Some(one), Some(seven)) = (plan_of(1), plan_of(7)) else {
            continue;
        };
        let optimal = facts.optimal(one.format);
        let sampled = F::SAMPLED_IMAGE
            | F::SAMPLED_IMAGE_FILTER_LINEAR
            | F::SAMPLED_IMAGE_FILTER_MINMAX
            | F::TRANSFER_SRC
            | F::TRANSFER_DST;
        // Stripped: what the optimal tiling has of the usage's features.
        let mut bare = desc.formats.clone();
        let mut blitless = desc.formats.clone();
        if let Some(p) = bare.formats.get_mut(&one.format) {
            p.linear_tiling = F::empty();
        }
        if let Some(p) = blitless.formats.get_mut(&one.format) {
            p.linear_tiling = p.optimal_tiling.difference(F::BLIT_DST);
        }
        let stripped = super::super::image_plan::ImageFacts {
            formats: &bare,
            ..facts
        };
        let expected = optimal.intersection(sampled);
        if !expected.is_empty() {
            named += 1;
            assert_eq!(
                linear_lack(&one, &stripped),
                Some(LinearLack {
                    format: one.format,
                    features: expected
                }),
                "{name}"
            );
        }
        // Without `BLIT_DST`, where the device's linear images serve both: only the texture of
        // several levels lacks it.
        let blitless = super::super::image_plan::ImageFacts {
            formats: &blitless,
            ..facts
        };
        let served = linear_lack(&one, &facts).is_none() && linear_lack(&seven, &facts).is_none();
        if served {
            assert_eq!(linear_lack(&one, &blitless), None, "{name}");
        }
        if served && optimal.contains(F::BLIT_DST) {
            assert_eq!(
                linear_lack(&seven, &blitless),
                Some(LinearLack {
                    format: one.format,
                    features: F::BLIT_DST
                }),
                "{name}"
            );
        }
        // Views of any format (`pixelFormatView`) are checked in each format a view may take:
        // `R32Float`'s among `RGBA8Unorm`'s, which the plan's own list does not hold.
        let any = super::super::image_plan::ImagePlan {
            flags: one.flags | ImageCreateFlags::MUTABLE_FORMAT,
            view_formats: Vec::new(),
            ..one.clone()
        };
        let mut r32 = desc.formats.clone();
        if let Some(p) = r32.formats.get_mut(&Format::R32_SFLOAT) {
            p.linear_tiling = F::empty();
        }
        let r32 = super::super::image_plan::ImageFacts {
            formats: &r32,
            ..facts
        };
        let r32_needed = facts.optimal(Format::R32_SFLOAT).intersection(sampled);
        if linear_lack(&any, &facts).is_none() && !r32_needed.is_empty() {
            any_checked += 1;
            assert_eq!(
                linear_lack(&any, &r32),
                Some(LinearLack {
                    format: Format::R32_SFLOAT,
                    features: r32_needed
                }),
                "{name}"
            );
            assert_eq!(linear_lack(&one, &r32), None, "{name}");
        }
        // Extended usage: storage where the image's own format takes none in either tiling, through
        // views of any format. Nothing is asked of the image's format for it, and each view format
        // is asked what its optimal tiling takes: `R32Float`'s storage, named once its linear
        // tiling lacks it.
        let stored = super::super::image_plan::ImagePlan {
            flags: any.flags | ImageCreateFlags::EXTENDED_USAGE,
            usage: any.usage | ImageUsageFlags::STORAGE,
            ..any.clone()
        };
        let mut storeless = desc.formats.clone();
        if let Some(p) = storeless.formats.get_mut(&one.format) {
            p.optimal_tiling = p.optimal_tiling.difference(F::STORAGE_IMAGE);
            p.linear_tiling = p.linear_tiling.difference(F::STORAGE_IMAGE);
        }
        let mut r32_storeless = storeless.clone();
        if let Some(p) = r32_storeless.formats.get_mut(&Format::R32_SFLOAT) {
            p.linear_tiling = p.linear_tiling.difference(F::STORAGE_IMAGE);
        }
        let storeless = super::super::image_plan::ImageFacts {
            formats: &storeless,
            ..facts
        };
        let r32_storeless = super::super::image_plan::ImageFacts {
            formats: &r32_storeless,
            ..facts
        };
        let r32_stores = facts.optimal(Format::R32_SFLOAT).contains(F::STORAGE_IMAGE);
        if linear_lack(&stored, &storeless).is_none() && r32_stores {
            extended_checked += 1;
            assert_eq!(
                linear_lack(&stored, &r32_storeless),
                Some(LinearLack {
                    format: Format::R32_SFLOAT,
                    features: F::STORAGE_IMAGE
                }),
                "{name}"
            );
        }
    }
    assert!(named > 0 && any_checked > 0 && extended_checked > 0);
}

/// A linear image's limits name what it lacks: no image at all where the device reports none (every
/// limit 0, as the RTX 3090 reports for a linear cube, 3D or depth image), else the first extent,
/// the levels, the layers or the sample count past the device's.
#[test]
fn a_linear_image_s_limits_name_what_it_lacks() {
    use ash::vk;
    use metal2vulkan::error::LinearNeed;
    let (_, desc, device) = committed().into_iter().next().unwrap();
    let facts = facts(&desc, &device);
    let plan = |width: u64, levels: u64, slices: u64, samples: u64| {
        let layout = TextureDescriptor {
            pixel_format: P::RGBA8Unorm,
            texture_type: if slices > 1 {
                TextureType::Type2DArray
            } else if samples > 1 {
                TextureType::Type2DMultisample
            } else {
                TextureType::Type2D
            },
            width,
            height: 64,
            mipmap_level_count: levels,
            array_length: slices,
            sample_count: samples,
            usage: TextureUsage::SHADER_READ,
            ..TextureDescriptor::default()
        }
        .check(APPLE)
        .unwrap();
        plan_image(&layout, TextureUsage::SHADER_READ, false, &facts).unwrap()
    };
    // The RTX 3090's limits for a linear 2D RGBA8Unorm image (NVIDIA 615.71.09).
    let nvidia = vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: 32768,
            height: 32768,
            depth: 1,
        },
        max_mip_levels: 1,
        max_array_layers: 1,
        sample_counts: vk::SampleCountFlags::TYPE_1,
        max_resource_size: 1 << 32,
    };
    assert_eq!(check_linear_limits(&plan(64, 1, 1, 1), &nvidia), Ok(()));
    assert_eq!(
        check_linear_limits(&plan(64, 1, 1, 1), &vk::ImageFormatProperties::default()),
        Err(LinearNeed::Image)
    );
    assert_eq!(
        check_linear_limits(&plan(64, 7, 1, 1), &nvidia),
        Err(LinearNeed::MipLevels {
            required: 7,
            available: 1
        })
    );
    assert_eq!(
        check_linear_limits(&plan(64, 1, 2, 1), &nvidia),
        Err(LinearNeed::ArrayLayers {
            required: 2,
            available: 1
        })
    );
    assert_eq!(
        check_linear_limits(&plan(64, 1, 1, 4), &nvidia),
        Err(LinearNeed::SampleCount(4))
    );
    let narrow = vk::ImageFormatProperties {
        max_extent: vk::Extent3D {
            width: 32,
            height: 32768,
            depth: 1,
        },
        ..nvidia
    };
    assert_eq!(
        check_linear_limits(&plan(64, 1, 1, 1), &narrow),
        Err(LinearNeed::Extent {
            required: 64,
            available: 32
        })
    );
}

/// A texture over a buffer that takes no linear image says which need failed: the features the
/// format's linear tiling lacks, the format query's refusal with the format, usage and flags,
/// a limit with what it needs and what the device holds, an import that is not importable.
#[test]
fn a_refused_linear_image_names_the_need_that_failed() {
    use super::super::texture::LinearRefusal;
    use ash::vk;
    use metal2vulkan::error::LinearNeed;
    let (_, desc, device) = committed().into_iter().next().unwrap();
    let facts = facts(&desc, &device);
    let layout = TextureDescriptor {
        pixel_format: P::BGRA8Unorm,
        texture_type: TextureType::Type2DArray,
        width: 64,
        height: 64,
        array_length: 2,
        usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        ..TextureDescriptor::default()
    }
    .check(APPLE)
    .unwrap();
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let plan = plan_image(&layout, usage, false, &facts).unwrap();
    let describe = |refusal: LinearRefusal| refusal.describe(&plan);
    let lacks = describe(LinearRefusal::Need(LinearNeed::Features {
        format: Format::B8G8R8A8_UNORM,
        features: FormatFeatureFlags2::COLOR_ATTACHMENT,
    }));
    assert!(
        lacks.contains("B8G8R8A8_UNORM") && lacks.contains("COLOR_ATTACHMENT"),
        "{lacks}"
    );
    let refused = describe(LinearRefusal::Need(LinearNeed::Image));
    assert!(
        refused.contains("B8G8R8A8_UNORM") && refused.contains("COLOR_ATTACHMENT"),
        "{refused}"
    );
    let layers = describe(LinearRefusal::Need(LinearNeed::ArrayLayers {
        required: 2,
        available: 1,
    }));
    assert!(
        layers.starts_with("2 array layers, where its linear images of")
            && layers.ends_with("hold 1"),
        "{layers}"
    );
    let import = describe(LinearRefusal::NotImportable(
        vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY,
    ));
    assert!(
        import.contains("importable false, dedicated only true"),
        "{import}"
    );
}

/// The committed profiles of the devices linear images were measured on.
const MEASURED: [&str; 5] = [
    "moltenvk-m3",
    "amd-780m-radv",
    "intel-irisxe-rplp-anv",
    "nvidia-rtx5080-laptop",
    "gate-lavapipe-x86_64",
];

/// The IOSurface stand-in's planes (`BufferTextureCall::IoSurfacePlane`) where plain Metal put
/// them on the M3 Max and an M4: the CPU's layout is the plane's; the GPU's starts at the
/// plane's offset rounded down to 16, rows the pitch rounded down to 16 apart, or 16 apart under
/// 16 bytes a row (`narrow`, where Metal's reads past the first row are memory-dependent); rows
/// that would overlap are refused; a surface without planes takes only a pitch and an offset of
/// 16 bytes.
#[test]
fn iosurface_planes_lie_where_metal_puts_them() {
    use BufferTextureProblem as Pr;
    let plane = |pf: P, (width, height), (offset, bpr), planar, length| Case {
        call: BufferTextureCall::IoSurfacePlane { planar },
        pixel_format: pf,
        ..texture(width, height, offset, bpr, length)
    };
    let at = |offset, pitch, row_bytes, rows| BufferLayout {
        offset,
        row_pitch: pitch,
        slice_pitch: pitch * rows,
        row_bytes,
        rows,
        slices: 1,
    };
    // (format, texel bytes, extent, (offset, bytesPerRow), the GPU's (offset, pitch)), each a
    // case Metal made.
    let made = [
        // The consumer's 4x4 planes: R8Unorm at 4 and RG8Unorm at 8 bytes a row (narrow), and
        // the ones Buffer::new_texture already took.
        (P::R8Unorm, 1, (4, 4), (0, 4), (0, 16)),
        (P::RG8Unorm, 2, (4, 4), (0, 8), (0, 16)),
        (P::RGBA8Unorm, 4, (4, 4), (0, 16), (0, 16)),
        (P::RGBA16Uint, 8, (4, 4), (0, 32), (0, 32)),
        (P::RGBA32Uint, 16, (4, 4), (0, 64), (0, 64)),
        // Other pitches under 16: rows 16 apart.
        (P::R8Unorm, 1, (4, 4), (0, 5), (0, 16)),
        (P::R8Unorm, 1, (4, 4), (0, 15), (0, 16)),
        (P::R8Unorm, 1, (4, 8), (0, 4), (0, 16)),
        (P::R8Unorm, 1, (1, 4), (0, 1), (0, 16)),
        (P::R8Unorm, 1, (12, 4), (0, 13), (0, 16)),
        (P::RG8Unorm, 2, (4, 4), (0, 9), (0, 16)),
        (P::RG8Unorm, 2, (6, 4), (0, 14), (0, 16)),
        // From 16 up: the pitch rounded down to 16, at least the row.
        (P::R8Unorm, 1, (4, 4), (0, 16), (0, 16)),
        (P::R8Unorm, 1, (4, 4), (0, 40), (0, 32)),
        (P::R8Unorm, 1, (16, 4), (0, 17), (0, 16)),
        (P::RG8Unorm, 2, (4, 4), (0, 40), (0, 32)),
        (P::RGBA8Unorm, 4, (4, 4), (0, 17), (0, 16)),
        (P::RGBA8Unorm, 4, (4, 4), (0, 20), (0, 16)),
        (P::RGBA8Unorm, 4, (4, 4), (0, 40), (0, 32)),
        (P::RGBA8Unorm, 4, (5, 4), (0, 32), (0, 32)),
        (P::RGBA16Uint, 8, (4, 4), (0, 33), (0, 32)),
        (P::RGBA16Uint, 8, (4, 4), (0, 40), (0, 32)),
        (P::RGBA16Uint, 8, (4, 4), (0, 56), (0, 48)),
        (P::RGBA32Uint, 16, (4, 4), (0, 65), (0, 64)),
        (P::RGBA32Uint, 16, (4, 4), (0, 88), (0, 80)),
        // Plane offsets: 4 from 0, 16 from 16, NV12 chroma planes' 36 from 32, 48 from 48 and 96
        // from 96, the consumer's '420v' planes at 256 and 512.
        (P::R8Unorm, 1, (4, 4), (4, 4), (0, 16)),
        (P::R8Unorm, 1, (4, 4), (16, 4), (16, 16)),
        (P::RG8Unorm, 2, (4, 2), (36, 10), (32, 16)),
        (P::RG8Unorm, 2, (4, 2), (48, 12), (48, 16)),
        (P::RG8Unorm, 2, (4, 2), (96, 24), (96, 16)),
        (P::RG8Unorm, 2, (2, 2), (256, 16), (256, 16)),
        (P::RG8Unorm, 2, (2, 2), (512, 16), (512, 16)),
    ];
    for (pf, texel, (w, h), (offset, bpr), (gpu_offset, gpu_pitch)) in made {
        let case = plane(pf, (w, h), (offset, bpr), true, 16384);
        assert_eq!(
            placed(case),
            Ok(Placed {
                gpu: at(gpu_offset, gpu_pitch, w * texel, h),
                cpu: at(offset, bpr, w * texel, h),
                narrow: bpr < 16,
            }),
            "{case:?}"
        );
    }
    let refused = |problem| Err(BufferRefusal::Metal(problem));
    let r8 = P::R8Unorm;
    let cases = [
        // A surface without planes: a pitch and an offset of 16 bytes only.
        (
            plane(r8, (4, 4), (0, 4), false, 16384),
            refused(Pr::RowPitchAlignment {
                bytes_per_row: 4,
                alignment: LINEAR_ALIGNMENT,
            }),
        ),
        (
            plane(r8, (4, 4), (8, 16), false, 16384),
            refused(Pr::OffsetAlignment {
                offset: 8,
                alignment: LINEAR_ALIGNMENT,
            }),
        ),
        (
            plane(r8, (4, 4), (0, 16), false, 16384),
            Ok(Placed {
                gpu: at(0, 16, 4, 4),
                cpu: at(0, 16, 4, 4),
                narrow: false,
            }),
        ),
        // Rows that overlap on the GPU: 20 bytes 16 apart (RGBA8Unorm 5 wide at 20
        // to 28, R8Unorm 20 and 17 wide, RGBA16Uint 3 wide at 24 and 28).
        (
            plane(P::RGBA8Unorm, (5, 4), (0, 20), true, 16384),
            Err(BufferRefusal::OverlappingRows),
        ),
        (
            plane(P::RGBA8Unorm, (5, 4), (0, 28), true, 16384),
            Err(BufferRefusal::OverlappingRows),
        ),
        (
            plane(r8, (17, 4), (0, 31), true, 16384),
            Err(BufferRefusal::OverlappingRows),
        ),
        (
            plane(P::RGBA16Uint, (3, 4), (0, 24), true, 16384),
            Err(BufferRefusal::OverlappingRows),
        ),
        // One such row overlaps nothing; nor does one row under 16 bytes read past itself.
        (
            plane(r8, (20, 1), (0, 20), true, 16384),
            Ok(Placed {
                gpu: at(0, 32, 20, 1),
                cpu: at(0, 20, 20, 1),
                narrow: false,
            }),
        ),
        (
            plane(r8, (4, 1), (4, 4), true, 16384),
            Ok(Placed {
                gpu: at(0, 16, 4, 1),
                cpu: at(4, 4, 4, 1),
                narrow: false,
            }),
        ),
        // No plane below its row; a plane past the buffer; GPU rows past it.
        (
            plane(r8, (4, 4), (0, 3), true, 16384),
            refused(Pr::RowPitch {
                bytes_per_row: 3,
                row_bytes: 4,
            }),
        ),
        (
            plane(r8, (4, 4), (0, 4), true, 15),
            refused(Pr::PastBuffer {
                counted: 16,
                length: 15,
            }),
        ),
        (
            plane(r8, (4, 4), (0, 4), true, 16),
            Err(BufferRefusal::TexelsPastLength),
        ),
        // A 2D texture alone, of one level.
        (
            Case {
                texture_type: TextureType::Type2DArray,
                slices: 2,
                ..plane(r8, (4, 4), (0, 16), true, 16384)
            },
            refused(Pr::TextureType {
                texture_type: TextureType::Type2DArray,
            }),
        ),
        (
            Case {
                levels: 2,
                ..plane(r8, (4, 4), (0, 16), true, 16384)
            },
            refused(Pr::MipmapLevelCount { count: 2 }),
        ),
    ];
    for (case, want) in cases {
        assert_eq!(placed(case), want, "{case:?}");
    }
}

/// The depth and stencil formats Metal makes over a plane of a surface with planes (an M4 under
/// macOS 27.0.1, TTG2's format scout), sampled or read and drawn.
const PLANAR_DEPTH: [P; 5] = [
    P::Depth16Unorm,
    P::Depth32Float,
    P::Depth32FloatStencil8,
    P::Stencil8,
    P::X32Stencil8,
];

/// A depth or stencil texture over a plane of a surface with planes, which Metal makes, is laid
/// out as a colour texture over the plane is, once Metal's own stops passed; over a surface
/// without planes, over a buffer and through the linear SPI Metal stops it (`NotLinear`), as it
/// stops every buffer's texture of those formats.
#[test]
fn a_planar_surface_s_depth_or_stencil_texture_is_laid_out_as_a_colour_one() {
    use BufferTextureCall::{IoSurfacePlane, Linear, Texture};
    for pf in PLANAR_DEPTH {
        let case = |call| Case {
            call,
            pixel_format: pf,
            ..texture(4, 4, 0, 64, 4096)
        };
        let planar = IoSurfacePlane { planar: true };
        let made = placed(case(planar)).unwrap_or_else(|refusal| panic!("{pf:?}: {refusal:?}"));
        assert_eq!(
            (made.gpu.offset, made.gpu.row_pitch, made.gpu.rows),
            (0, 64, 4),
            "{pf:?}"
        );
        for call in [Texture, Linear, IoSurfacePlane { planar: false }] {
            assert_eq!(
                placed(case(call)).map(|_| ()),
                Err(BufferRefusal::Metal(BufferTextureProblem::NotLinear {
                    pixel_format: pf
                })),
                "{pf:?} {call:?}"
            );
        }
        // Metal's own stops of the descriptor come first.
        let levels = Case {
            levels: 2,
            ..case(planar)
        };
        assert_eq!(
            placed(levels).map(|_| ()),
            Err(BufferRefusal::Metal(
                BufferTextureProblem::MipmapLevelCount { count: 2 }
            )),
            "{pf:?}"
        );
        let memoryless = Case {
            storage: StorageMode::Memoryless,
            ..case(planar)
        };
        assert_eq!(
            placed(memoryless).map(|_| ()),
            Err(BufferRefusal::Metal(BufferTextureProblem::Options)),
            "{pf:?}"
        );
    }
}

/// A render target's linear image serves it where the device enabled `linearColorAttachment` and
/// its formats' linear tiling has `LINEAR_COLOR_ATTACHMENT_NV` (the NVIDIA profiles), which the
/// facts read as the color attachment feature under the extension's rule; without the feature
/// enabled it does not. A tiling with the color attachment feature itself renders beside any
/// attachment, so no format is under the rule there.
#[test]
fn a_render_target_s_linear_image_serves_it_by_the_linear_attachment_rule() {
    let mut ruled = 0;
    for (name, desc, device) in committed() {
        let off = facts(&desc, &device);
        let on = super::super::image_plan::ImageFacts {
            linear_attachments: true,
            ..off
        };
        let reported = desc
            .formats
            .formats
            .get(&Format::B8G8R8A8_UNORM)
            .map(|p| p.linear_tiling)
            .unwrap_or_default();
        let bit = reported.contains(FormatFeatureFlags2::LINEAR_COLOR_ATTACHMENT_NV);
        let core = reported.contains(FormatFeatureFlags2::COLOR_ATTACHMENT);
        assert!(
            !off.linear_attachment_rule(Format::B8G8R8A8_UNORM),
            "{name}"
        );
        assert_eq!(
            on.linear_attachment_rule(Format::B8G8R8A8_UNORM),
            bit && !core,
            "{name}"
        );
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let layout = TextureDescriptor {
            pixel_format: P::BGRA8Unorm,
            width: 1920,
            height: 1080,
            usage,
            ..TextureDescriptor::default()
        }
        .check(APPLE)
        .unwrap();
        let Ok(plan) = plan_image(&layout, usage, false, &on) else {
            continue;
        };
        if !(bit && !core) {
            assert_eq!(
                linear_lack(&plan, &on).is_none(),
                linear_lack(&plan, &off).is_none(),
                "{name}"
            );
            continue;
        }
        ruled += 1;
        assert!(
            plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT),
            "{name}"
        );
        assert!(
            linear_lack(&plan, &off).is_some(),
            "{name}: served without the feature"
        );
        assert!(
            linear_lack(&plan, &on).is_none(),
            "{name}: {:?}",
            linear_lack(&plan, &on)
        );
    }
    assert!(
        ruled >= 2,
        "the NVIDIA profiles render linear images by the rule"
    );
}
