//! The Metal values only render passes and render encoders take: load and store actions, store
//! action options, sample positions, resolve filters, clear colors, viewports and scissor
//! rectangles, with Metal's raw values. A raw value Metal does not define is a typed error, never a
//! cast.
//!
//! Like `values.rs`, these live here rather than in the generated `metal` module: the
//! compile-and-plan layer never reads them.

use super::error::{invalid, Error, Problem};

/// `MTLLoadAction`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum LoadAction {
    /// `MTLLoadActionDontCare`: the pass starts from texels it does not read.
    #[default]
    DontCare,
    /// `MTLLoadActionLoad`: the pass starts from the attachment's texels.
    Load,
    /// `MTLLoadActionClear`: the pass starts from the attachment's clear value.
    Clear,
}

impl LoadAction {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            LoadAction::DontCare => 0,
            LoadAction::Load => 1,
            LoadAction::Clear => 2,
        }
    }

    /// The action of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<LoadAction, Error> {
        Ok(match raw {
            0 => LoadAction::DontCare,
            1 => LoadAction::Load,
            2 => LoadAction::Clear,
            raw => return Err(undefined("MTLLoadAction", raw)),
        })
    }
}

/// `MTLStoreAction`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StoreAction {
    /// `MTLStoreActionDontCare`: the pass's texels are not kept. Apple's GPU leaves the
    /// attachment's earlier texels.
    DontCare,
    /// `MTLStoreActionStore`: the pass's texels are kept.
    #[default]
    Store,
    /// `MTLStoreActionMultisampleResolve`: the samples are resolved into the resolve texture
    /// and not kept.
    MultisampleResolve,
    /// `MTLStoreActionStoreAndMultisampleResolve`: kept and resolved.
    StoreAndMultisampleResolve,
    /// `MTLStoreActionUnknown`: decided later, by the encoder's `set_*_store_action` calls,
    /// before `end_encoding`.
    Unknown,
    /// `MTLStoreActionCustomSampleDepthStore`: depth kept for the custom sample positions.
    CustomSampleDepthStore,
}

impl StoreAction {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            StoreAction::DontCare => 0,
            StoreAction::Store => 1,
            StoreAction::MultisampleResolve => 2,
            StoreAction::StoreAndMultisampleResolve => 3,
            StoreAction::Unknown => 4,
            StoreAction::CustomSampleDepthStore => 5,
        }
    }

    /// The action of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<StoreAction, Error> {
        Ok(match raw {
            0 => StoreAction::DontCare,
            1 => StoreAction::Store,
            2 => StoreAction::MultisampleResolve,
            3 => StoreAction::StoreAndMultisampleResolve,
            4 => StoreAction::Unknown,
            5 => StoreAction::CustomSampleDepthStore,
            raw => return Err(undefined("MTLStoreAction", raw)),
        })
    }

    /// Whether the action resolves the samples into a resolve texture.
    pub fn resolves(self) -> bool {
        matches!(
            self,
            StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
        )
    }
}

/// `MTLStoreActionOptions`: bit 0 is `MTLStoreActionOptionCustomSamplePositions`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct StoreActionOptions(u64);

impl StoreActionOptions {
    /// `MTLStoreActionOptionNone`.
    pub const NONE: StoreActionOptions = StoreActionOptions(0);
    /// `MTLStoreActionOptionCustomSamplePositions`.
    pub const CUSTOM_SAMPLE_POSITIONS: StoreActionOptions = StoreActionOptions(1);

    /// The options of Metal's raw value `raw`: refused when it sets a bit Metal does not define.
    pub fn from_raw(raw: u64) -> Result<StoreActionOptions, Error> {
        if raw & !1 != 0 {
            return Err(undefined("MTLStoreActionOptions", raw));
        }
        Ok(StoreActionOptions(raw))
    }

    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// Whether every option of `other` is set.
    pub fn contains(self, other: StoreActionOptions) -> bool {
        self.0 & other.0 == other.0
    }
}

/// `MTLSamplePosition`: where a sample lies in its pixel, from its top-left corner, each
/// coordinate in `[0, 1)` (Metal asserts on another, measured).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplePosition {
    /// `x`: from the pixel's left edge.
    pub x: f32,
    /// `y`: from the pixel's top edge.
    pub y: f32,
}

impl SamplePosition {
    /// `MTLSamplePositionMake`.
    pub fn new(x: f32, y: f32) -> SamplePosition {
        SamplePosition { x, y }
    }
}

/// `MTLMultisampleDepthResolveFilter`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MultisampleDepthResolveFilter {
    /// `MTLMultisampleDepthResolveFilterSample0`.
    #[default]
    Sample0,
    /// `MTLMultisampleDepthResolveFilterMin`.
    Min,
    /// `MTLMultisampleDepthResolveFilterMax`.
    Max,
}

impl MultisampleDepthResolveFilter {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            MultisampleDepthResolveFilter::Sample0 => 0,
            MultisampleDepthResolveFilter::Min => 1,
            MultisampleDepthResolveFilter::Max => 2,
        }
    }

    /// The filter of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<MultisampleDepthResolveFilter, Error> {
        Ok(match raw {
            0 => MultisampleDepthResolveFilter::Sample0,
            1 => MultisampleDepthResolveFilter::Min,
            2 => MultisampleDepthResolveFilter::Max,
            raw => return Err(undefined("MTLMultisampleDepthResolveFilter", raw)),
        })
    }
}

/// `MTLMultisampleStencilResolveFilter`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MultisampleStencilResolveFilter {
    /// `MTLMultisampleStencilResolveFilterSample0`.
    #[default]
    Sample0,
    /// `MTLMultisampleStencilResolveFilterDepthResolvedSample`.
    DepthResolvedSample,
}

impl MultisampleStencilResolveFilter {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            MultisampleStencilResolveFilter::Sample0 => 0,
            MultisampleStencilResolveFilter::DepthResolvedSample => 1,
        }
    }

    /// The filter of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<MultisampleStencilResolveFilter, Error> {
        Ok(match raw {
            0 => MultisampleStencilResolveFilter::Sample0,
            1 => MultisampleStencilResolveFilter::DepthResolvedSample,
            raw => return Err(undefined("MTLMultisampleStencilResolveFilter", raw)),
        })
    }
}

/// `MTLClearColor`: the color a `Clear` load action writes, in the attachment format's channels
/// (red, green, blue, alpha), converted to the format as Metal converts it (`clear_value.rs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClearColor {
    /// `red`.
    pub red: f64,
    /// `green`.
    pub green: f64,
    /// `blue`.
    pub blue: f64,
    /// `alpha`.
    pub alpha: f64,
}

impl Default for ClearColor {
    /// Metal's default: opaque black.
    fn default() -> ClearColor {
        ClearColor::new(0.0, 0.0, 0.0, 1.0)
    }
}

impl ClearColor {
    /// `MTLClearColorMake(red, green, blue, alpha)`.
    pub fn new(red: f64, green: f64, blue: f64, alpha: f64) -> ClearColor {
        ClearColor {
            red,
            green,
            blue,
            alpha,
        }
    }

    /// The four channels in order.
    pub(super) fn channels(self) -> [f64; 4] {
        [self.red, self.green, self.blue, self.alpha]
    }
}

/// `MTLViewport`: the region of the render target a draw's normalized device coordinates map
/// to, in pixels with the origin at the top left, and its depth range.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Viewport {
    /// `originX`, in pixels.
    pub origin_x: f64,
    /// `originY`, in pixels.
    pub origin_y: f64,
    /// `width`, in pixels; a negative width mirrors the draw, as Metal's does.
    pub width: f64,
    /// `height`, in pixels.
    pub height: f64,
    /// `znear`: the depth an NDC depth of 0 maps to. A range past `[0, 1]` is refused where the
    /// device cannot draw it ([`RenderCommandEncoder::set_viewport`](crate::RenderCommandEncoder::set_viewport)).
    pub znear: f64,
    /// `zfar`: the depth an NDC depth of 1 maps to.
    pub zfar: f64,
}

impl Viewport {
    /// A viewport of these fields.
    pub fn new(
        origin_x: f64,
        origin_y: f64,
        width: f64,
        height: f64,
        znear: f64,
        zfar: f64,
    ) -> Viewport {
        Viewport {
            origin_x,
            origin_y,
            width,
            height,
            znear,
            zfar,
        }
    }
}

/// `MTLScissorRect`: the pixels a draw may write, origin at the top left.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ScissorRect {
    /// `x`, in pixels.
    pub x: u64,
    /// `y`, in pixels.
    pub y: u64,
    /// `width`, in pixels.
    pub width: u64,
    /// `height`, in pixels.
    pub height: u64,
}

impl ScissorRect {
    /// A scissor rectangle of these fields.
    pub fn new(x: u64, y: u64, width: u64, height: u64) -> ScissorRect {
        ScissorRect {
            x,
            y,
            width,
            height,
        }
    }
}

fn undefined(ty: &'static str, raw: u64) -> Error {
    invalid("from_raw", Problem::UndefinedValue { ty, raw })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value round-trips through Metal's raw value, and an undefined one is refused.
    #[test]
    fn raw_values_round_trip() {
        for raw in 0..3 {
            assert_eq!(LoadAction::from_raw(raw).unwrap().raw(), raw);
            assert_eq!(
                MultisampleDepthResolveFilter::from_raw(raw).unwrap().raw(),
                raw
            );
        }
        for raw in 0..6 {
            assert_eq!(StoreAction::from_raw(raw).unwrap().raw(), raw);
        }
        for raw in 0..2 {
            assert_eq!(
                MultisampleStencilResolveFilter::from_raw(raw)
                    .unwrap()
                    .raw(),
                raw
            );
            assert_eq!(StoreActionOptions::from_raw(raw).unwrap().raw(), raw);
        }
        assert!(LoadAction::from_raw(3).is_err());
        assert!(StoreAction::from_raw(6).is_err());
        assert!(StoreActionOptions::from_raw(2).is_err());
        assert!(MultisampleDepthResolveFilter::from_raw(3).is_err());
        assert!(MultisampleStencilResolveFilter::from_raw(2).is_err());
    }

    /// Metal's defaults (probe 1 on an M4): opaque black, and the store action of a color
    /// attachment.
    #[test]
    fn defaults_are_metals() {
        assert_eq!(ClearColor::default().channels(), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(LoadAction::default(), LoadAction::DontCare);
        assert_eq!(StoreAction::default(), StoreAction::Store);
    }
}
