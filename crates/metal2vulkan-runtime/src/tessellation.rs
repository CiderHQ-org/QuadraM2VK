//! A render encoder's tessellation factors: the buffer of
//! `setTessellationFactorBuffer:offset:instanceStride:` and the scale of
//! `setTessellationFactorScale:`, which a patch draw of a tessellation pipeline reads.
//!
//! The binder takes them as `BindState::tessellation_factors`, a tessellation pipeline's draws
//! only: the buffer's length, offset and instance stride in the buffer's space (as a buffer
//! slot's, `ArgumentTable::stage_table`), and the scale. A draw binds the buffer where the plan
//! writes `WriteSource::TessellationFactors`, over the range the binder gives, and uses it in the
//! control stage (`binding.rs`). The encoder refuses a patch draw without a buffer, which
//! Metal's validation stops and plain Metal's process ends at (`render.rs`). The binder refuses
//! factors past what one descriptor binds. Metal's validation stops an offset or a stride that is
//! not a multiple of 4, but plain Metal reads an offset of 2 at that byte (probe 10
//! `factor_offset_2` drew the factors 2 bytes past a 2-byte pad), and an odd one from the byte
//! below (1 from 0, 3 from 2, 7 from 6), so the binder reads from the offset rounded down to a
//! multiple of 2, and the control stage from any even byte. Plain Metal steps the instances by a
//! stride rounded down to a multiple of 4 (Metal probes, two instances: strides 2 and 14 drew what
//! 0 and 12 draw, 6 and 10 what 4 and 8 draw: 13 and 15 what 12, 17 and 19 what 16; probe 10's
//! `factor_stride_2` drew one instance, which reads no stride), as the binder steps them.
//!
//! Metal multiplies the factors by the scale rounded to `half` (the M3 Max and the M4, with and
//! without API validation, which stops none of these): 2.0005 and 1.3
//! drew what factors of 2 draw, a scale that is subnormal as a `half` (2^-15, 2^-24, 10^-6) what
//! its product draws, and one that is infinite as a `half` (70000) what the maximum factor draws,
//! as the binder takes them. One that is 0 as a `half` (2^-30) culls every patch.
//!
//! Until `setTessellationFactorScale:` is called, a pipeline that enables the scale culls every
//! patch on Metal (a Metal probe on an M4: factors of 1 marked no domain point, with and without
//! API validation), as a scale of 0 would; one that does not enable it never reads the scale. So an
//! encoder holds no scale until one is set, and such a draw records nothing (`render.rs`), as does
//! one with a scale that is 0 or negative as a `half`, which the binder refuses.
//!
//! Metal's validation stops a scale of 0, a negative one, NaN, an infinity or a subnormal `float`
//! at the setter, but plain Metal takes each (Metal probes on an M4): 0, -1, NaN, `-∞` and both
//! subnormals (`±2^-139`, 0 as a `half`) culled every patch of a pipeline that enables the scale,
//! `+∞` drew what the maximum factor draws (as 16 does under a maximum of 16), and each drew as 1
//! does on a pipeline that does not enable it. So the setter takes every scale, and one that is 0,
//! negative or NaN as a `half` culls.

use super::buffer::Buffer;
use metal2vulkan::bind::{TessellationFactorBuffer, TessellationFactors};
use metal2vulkan::metal::TessellationControlPointIndexType;
use metal2vulkan::pipeline::GraphicsState;

/// What a pipeline's patch draws take of it, whether its tessellation is the device's
/// (`GraphicsState::tessellation`) or emulated on a device without `tessellationShader`
/// (`GraphicsState::emulated_tessellation`, `render/tessellation_scan.rs`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Patches {
    /// The control points its post-tessellation vertex function states (0 for none).
    pub(crate) control_points: u32,
    /// Its `tessellationControlPointIndexType`.
    pub(crate) control_point_index_type: TessellationControlPointIndexType,
    /// Whether its tessellation is emulated.
    pub(crate) emulated: bool,
}

/// The patch draws of the pipeline whose state is `graphics`: `None` for one that draws no
/// patches.
pub(crate) fn patches(graphics: &GraphicsState) -> Option<Patches> {
    match (graphics.tessellation, graphics.emulated_tessellation) {
        (Some(t), _) => Some(Patches {
            control_points: t.control_points,
            control_point_index_type: t.control_point_index_type,
            emulated: false,
        }),
        (None, Some(e)) => Some(Patches {
            control_points: e.control_points,
            control_point_index_type: e.descriptor.control_point_index_type,
            emulated: true,
        }),
        (None, None) => None,
    }
}

/// The encoder's factor buffer, its offset and its instance stride.
#[derive(Clone, Debug)]
pub(super) struct FactorBuffer {
    pub(super) buffer: Buffer,
    pub(super) offset: u64,
    pub(super) instance_stride: u64,
}

/// The encoder's tessellation factor state: no buffer and no scale until set (the module docs).
#[derive(Clone, Debug, Default)]
pub(super) struct Factors {
    pub(super) buffer: Option<FactorBuffer>,
    pub(super) scale: Option<f32>,
}

impl Factors {
    /// Whether a pipeline that enables the scale culls every patch: no scale is set, or one that
    /// is 0, negative or NaN as a `half` (the module docs).
    pub(super) fn culls(&self) -> bool {
        self.scale.is_none_or(|scale| {
            let half = crate::clear_value::f32_to_f16(scale);
            half & 0x7fff == 0 || half & 0x8000 != 0 || half & 0x7fff > 0x7c00
        })
    }

    /// The factors as the binder takes them. Without a scale set, or with one that culls, 1: the
    /// pipeline the draw binds either does not read the scale or draws nothing (the module docs).
    pub(super) fn bind_state(&self) -> TessellationFactors {
        let buffer = self.buffer.as_ref().map(|f| {
            // The buffer's space starts `base` bytes before its byte 0 (`Buffer::base`).
            let base = f.buffer.base();
            TessellationFactorBuffer::new(
                base + f.buffer.length(),
                base + f.offset,
                f.instance_stride,
            )
        });
        let scale = self.scale.filter(|_| !self.culls()).unwrap_or(1.0);
        TessellationFactors::new(buffer, scale)
    }
}
