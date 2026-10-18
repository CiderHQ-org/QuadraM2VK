use std::collections::HashSet;

use super::*;

/// Metal's raw `textureWriteRoundingMode` values: 0 and 1 round toward zero, 2 to nearest even,
/// and 3 and above are refused as Metal's descriptor validation stops them.
#[test]
fn texture_write_rounding_modes_take_metals_raw_values() {
    use crate::error::{DescriptorError, Error};
    for (raw, want) in [
        (0, TextureWriteRoundingMode::TowardZero),
        (1, TextureWriteRoundingMode::TowardZero),
        (2, TextureWriteRoundingMode::NearestEven),
    ] {
        assert_eq!(TextureWriteRoundingMode::from_raw(raw), Ok(want), "{raw}");
    }
    for (raw, value) in [(3, 3), (4, 4), (u64::MAX, u32::MAX)] {
        assert_eq!(
            TextureWriteRoundingMode::from_raw(raw),
            Err(Error::InvalidDescriptor(
                DescriptorError::UnknownEnumValue {
                    ty: "MTLTextureWriteRoundingMode",
                    value,
                }
            )),
            "{raw}"
        );
    }
    assert_eq!(
        TextureWriteRoundingMode::default(),
        TextureWriteRoundingMode::TowardZero
    );
}
use crate::error::{DescriptorError, Error};

/// `result` is the refusal of `value` as an unknown ordinal of `ty`. The front end's enums
/// (`DataType`) refuse with its error, which the translator's maps to the same refusal.
fn assert_unknown<T: std::fmt::Debug, E: Into<Error>>(
    result: Result<T, E>,
    ty: &'static str,
    value: u32,
) {
    match result.map_err(Into::into) {
        Err(Error::InvalidDescriptor(DescriptorError::UnknownEnumValue { ty: t, value: v })) => {
            assert_eq!((t, v), (ty, value));
        }
        other => panic!("{ty} {value}: expected UnknownEnumValue, got {other:?}"),
    }
}

/// `TryFrom<u32>` round trips over every value, and rejection of every other ordinal up to a
/// bound past the largest value.
macro_rules! check_enum {
    ($($ty:ident),* $(,)?) => {$({
        let all = $ty::ALL;
        assert!(!all.is_empty(), "{} has no values", $ty::METAL_NAME);
        let mut raws = HashSet::new();
        let mut names = HashSet::new();
        for &v in all {
            let raw = u32::from(v);
            assert!(raws.insert(raw), "{} repeats {raw}", $ty::METAL_NAME);
            assert!(names.insert(v.name()), "{} repeats a name", $ty::METAL_NAME);
            assert_eq!($ty::try_from(raw).ok(), Some(v));
            assert_eq!($ty::from_name(v.name()), Some(v));
            assert!(v.name().starts_with("MTL"));
        }
        let max = raws.iter().copied().max().unwrap_or(0);
        for raw in 0..=max.saturating_add(64) {
            if !raws.contains(&raw) {
                assert_unknown($ty::try_from(raw), $ty::METAL_NAME, raw);
            }
        }
        assert_unknown($ty::try_from(u32::MAX), $ty::METAL_NAME, u32::MAX);
    })*};
}

/// `TryFrom<u32>` accepts exactly the subsets of the named bits and the dropped ones, keeping
/// them.
macro_rules! check_flags {
    ($($ty:ident),* $(,)?) => {$({
        let all = $ty::ALL_KNOWN.as_raw();
        for &(v, name) in $ty::KNOWN {
            assert_eq!($ty::try_from(v.as_raw()).ok(), Some(v), "{name}");
            assert_eq!($ty::from_name(name), Some(v), "{name}");
            assert!($ty::ALL_KNOWN.contains(v));
        }
        assert_eq!($ty::try_from(all).ok(), Some($ty::ALL_KNOWN));
        for bit in 0..32 {
            let raw = 1u32 << bit;
            if ($ty::DROPPED >> bit) & 1 == 1 {
                assert_eq!(all & raw, 0, "a dropped bit is named");
                let kept = $ty::try_from(raw | all).unwrap();
                assert_eq!(kept.as_raw(), raw | all);
                assert_eq!(kept.without_dropped(), $ty::ALL_KNOWN);
                assert_eq!($ty::try_from(raw).unwrap().without_dropped(), $ty::empty());
            } else if all & raw == 0 {
                assert_unknown($ty::try_from(raw), $ty::METAL_NAME, raw);
                assert_unknown($ty::try_from(raw | all), $ty::METAL_NAME, raw | all);
            }
        }
    })*};
}

#[test]
fn enums_round_trip_and_reject_unknown_ordinals() {
    check_enum!(
        ArgumentBuffersTier,
        AttributeFormat,
        BindingAccess,
        BlendFactor,
        BlendOperation,
        CompareFunction,
        CullMode,
        DataType,
        DepthClipMode,
        DispatchType,
        FunctionType,
        IndexType,
        Mutability,
        PatchType,
        PixelFormat,
        PrimitiveTopologyClass,
        PrimitiveType,
        ReadWriteTextureTier,
        SamplerAddressMode,
        SamplerBorderColor,
        SamplerMinMagFilter,
        SamplerMipFilter,
        SamplerReductionMode,
        StencilOperation,
        StepFunction,
        TessellationControlPointIndexType,
        TessellationFactorFormat,
        TessellationFactorStepFunction,
        TessellationPartitionMode,
        TextureSwizzle,
        TextureType,
        TriangleFillMode,
        VertexFormat,
        VertexStepFunction,
        Winding,
    );
}

#[test]
fn flags_round_trip_and_reject_unknown_bits() {
    check_flags!(
        BarrierScope,
        ColorWriteMask,
        FunctionOptions,
        IndirectCommandType,
        RenderStages,
        ResourceUsage,
        TextureUsage
    );
}

/// Ordinals fixed by the Metal API (and relied on by guest command streams), checked
/// independently of the generator.
#[test]
fn metal_ordinals_match_the_api() {
    assert_eq!(PixelFormat::Invalid.as_raw(), 0);
    assert_eq!(PixelFormat::R8Unorm.as_raw(), 10);
    assert_eq!(PixelFormat::RGBA8Unorm.as_raw(), 70);
    assert_eq!(PixelFormat::RGBA8UnormSRGB.as_raw(), 71);
    assert_eq!(PixelFormat::BGRA8Unorm.as_raw(), 80);
    assert_eq!(PixelFormat::BGRA8UnormSRGB.as_raw(), 81);
    assert_eq!(PixelFormat::RGBA32Float.as_raw(), 125);
    assert_eq!(PixelFormat::Depth32Float.as_raw(), 252);
    assert_eq!(PixelFormat::Depth24UnormStencil8.as_raw(), 255);
    assert_eq!(PixelFormat::Depth32FloatStencil8.as_raw(), 260);
    assert_eq!(PixelFormat::BGRA10XR.as_raw(), 552);
    assert_eq!(VertexFormat::Float4.as_raw(), 31);
    assert_eq!(VertexFormat::UChar4NormalizedBGRA.as_raw(), 42);
    assert_eq!(VertexFormat::Half.as_raw(), 53);
    assert_eq!(AttributeFormat::Float4.as_raw(), 31);
    assert_eq!(TextureType::Type2DArray.as_raw(), 3);
    assert_eq!(TextureType::TextureBuffer.as_raw(), 9);
    assert_eq!(DataType::Float4.as_raw(), 6);
    assert_eq!(DataType::Bool.as_raw(), 53);
    assert_eq!(PrimitiveType::TriangleStrip.as_raw(), 4);
    assert_eq!(IndexType::UInt32.as_raw(), 1);
    assert_eq!(CompareFunction::Always.as_raw(), 7);
    assert_eq!(BlendFactor::OneMinusSource1Alpha.as_raw(), 18);
    assert_eq!(SamplerAddressMode::ClampToBorderColor.as_raw(), 5);
    // Metal's own cubic filter: 2 is a filter, and 3 stops Metal.
    assert_eq!(
        SamplerMinMagFilter::try_from(2).ok(),
        Some(SamplerMinMagFilter::Cubic)
    );
    assert_eq!(
        SamplerMinMagFilter::Cubic.name(),
        "MTLSamplerMinMagFilterCubic"
    );
    assert!(SamplerMinMagFilter::try_from(3).is_err());
    assert_eq!(VertexStepFunction::PerInstance.as_raw(), 2);
    assert_eq!(ColorWriteMask::RED.as_raw(), 0x8);
    assert_eq!(ColorWriteMask::ALL.as_raw(), 0xf);
    assert_eq!(TextureUsage::SHADER_WRITE.as_raw(), 0x2);
    // Apple's private bits: kept, and 0x8 dropped.
    assert_eq!(TextureUsage::NO_CONCURRENT_ACCESS.as_raw(), 0x4000);
    assert_eq!(TextureUsage::VIDEO.as_raw(), 0x8000);
    assert_eq!(TextureUsage::BLOCK_WRITES_ONLY.as_raw(), 0x10000);
    assert_eq!(
        TextureUsage::try_from(0x1_0005).ok(),
        Some(
            TextureUsage::SHADER_READ
                | TextureUsage::RENDER_TARGET
                | TextureUsage::BLOCK_WRITES_ONLY
        )
    );
    let kept = TextureUsage::try_from(0x9).unwrap();
    assert_eq!(kept.as_raw(), 0x9);
    assert_eq!(kept.without_dropped(), TextureUsage::SHADER_READ);
    assert_eq!(ResourceUsage::SAMPLE.as_raw(), 0x4);
    assert_eq!(RenderStages::FRAGMENT.as_raw(), 0x2);
}

#[test]
fn deprecated_spellings_resolve() {
    assert_eq!(
        BindingAccess::from_name("MTLArgumentAccessReadWrite"),
        Some(BindingAccess::ReadWrite)
    );
    assert_eq!(PixelFormat::from_name("MTLPixelFormatNotAFormat"), None);
    assert_eq!(
        PixelFormat::BGRA8UnormSRGB.name(),
        "MTLPixelFormatBGRA8Unorm_sRGB"
    );
}

#[test]
fn flags_debug() {
    let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    assert_eq!(
        format!("{usage:?}"),
        "MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite"
    );
    assert_eq!(
        format!("{:?}", TextureUsage::empty()),
        "MTLTextureUsageUnknown"
    );
    assert_eq!(format!("{:?}", ColorWriteMask::ALL), "MTLColorWriteMaskAll");
    assert!(usage.contains(TextureUsage::SHADER_READ));
    assert!(!usage.intersects(TextureUsage::RENDER_TARGET));
}

#[test]
fn values_are_send_sync() {
    fn send_sync<T: Send + Sync + Copy>() {}
    send_sync::<PixelFormat>();
    send_sync::<VertexFormat>();
    send_sync::<TextureUsage>();
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_and_rejects_unknown_bits() {
    let json = serde_json::to_string(&PixelFormat::BGRA8Unorm).expect("serialize");
    let back: PixelFormat = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, PixelFormat::BGRA8Unorm);
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let json = serde_json::to_string(&usage).expect("serialize");
    assert_eq!(json, "5");
    let back: TextureUsage = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, usage);
    assert!(serde_json::from_str::<TextureUsage>("1024").is_err());
}

/// A swizzle's word holds each channel's `MTLTextureSwizzle` raw value in three bits, red's
/// lowest (`SpecKey::TextureSwizzle`).
#[test]
fn a_swizzle_word_holds_each_channels_source() {
    use super::{TextureSwizzle as S, TextureSwizzleChannels as C};
    assert_eq!(C::DEFAULT_WORD, C::DEFAULT.word());
    assert_eq!(C::DEFAULT_WORD, 2 | 3 << 3 | 4 << 6 | 5 << 9);
    let bgr1 = C::new(S::Blue, S::Green, S::Red, S::One);
    assert_eq!(bgr1.word(), 4 | 3 << 3 | 2 << 6 | 1 << 9);
    assert_eq!(C::new(S::Zero, S::Zero, S::Zero, S::Zero).word(), 0);
}
