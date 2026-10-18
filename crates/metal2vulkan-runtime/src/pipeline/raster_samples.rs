//! Metal's refusal of a pipeline state whose `rasterSampleCount` an Apple GPU does not take.
//!
//! - **What Metal answers** (an Apple M4 under macOS 27.0.1, plain and under API validation alike).
//!   `newRenderPipelineStateWithDescriptor:` answers nil with an `NSError` for a
//!   `rasterSampleCount` other than 1, 2 and 4 (0, 3, 8, 16 and 32 measured, with a color
//!   attachment, a depth attachment alone, no attachment, and rasterization disabled alike), and
//!   `newRenderPipelineStateWithMeshDescriptor:` alike (0, 3, 8 and 16): code 2,
//!   `localizedDescription` "rasterSampleCount (8) is not supported by
//!   device.", the domain the GPU driver's class. `newRenderPipelineStateWithTileDescriptor:`
//!   answers so for 0 alone, and makes a tile state of 3, 8, 16, 32 and 64 samples (which the
//!   compile layer refuses above 1 on every device, `Unsupported(TileShading)`).
//! - **The answer, on every device.** [`Error::MetalError`] with [`Problem::RasterSampleCount`],
//!   before anything is compiled, whatever sample counts the Vulkan device rasterizes: an NVIDIA
//!   device that rasterizes 8 samples makes no state Metal refuses, as no device makes a texture
//!   of more samples than an Apple GPU takes. The domain is the M3 Max's driver class, which the
//!   runtime answers for every creation error Metal answers so ([`DOMAIN`]); the M3 Max's answer to
//!   these counts is not measured here.

use super::indirect_arguments::DOMAIN;
use crate::error::{Error, MetalError, Problem};

/// The `rasterSampleCount`s a render or mesh state of an Apple GPU takes.
const APPLE_SAMPLE_COUNTS: [u32; 3] = [1, 2, 4];

/// The code of Metal's error.
const CODE: i64 = 2;

/// Metal's refusal in `call` of `count`.
fn refused(call: &'static str, count: u32) -> Error {
    Error::MetalError(MetalError::with_description(
        call,
        Problem::RasterSampleCount { count },
        DOMAIN,
        CODE,
        format!("rasterSampleCount ({count}) is not supported by device."),
    ))
}

/// Refused in `call` (the module docs): a render or mesh state's `rasterSampleCount` of `count`,
/// unless an Apple GPU takes it.
pub(super) fn check(call: &'static str, count: u32) -> Result<(), Error> {
    match APPLE_SAMPLE_COUNTS.contains(&count) {
        true => Ok(()),
        false => Err(refused(call, count)),
    }
}

/// Refused in `call` (the module docs): a tile state's `rasterSampleCount` of 0.
pub(super) fn check_tile(call: &'static str, count: u32) -> Result<(), Error> {
    match count {
        0 => Err(refused(call, count)),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal's error for every count but 1, 2 and 4 of a render or mesh state, and for 0 alone of
    /// a tile state.
    #[test]
    fn metal_refuses_a_raster_sample_count_an_apple_gpu_lacks() {
        let refused = |call: &'static str, count: u32| {
            Err(Error::MetalError(MetalError::with_description(
                call,
                Problem::RasterSampleCount { count },
                "AGXMetalG15X_M1",
                2,
                format!("rasterSampleCount ({count}) is not supported by device."),
            )))
        };
        for count in [1, 2, 4] {
            assert_eq!(check("r", count), Ok(()), "{count}");
        }
        for count in [0, 3, 5, 8, 16, 32, 64, u32::MAX] {
            assert_eq!(check("r", count), refused("r", count), "{count}");
        }
        assert_eq!(check_tile("t", 0), refused("t", 0));
        for count in [1, 2, 3, 4, 8, 16, 32, 64] {
            assert_eq!(check_tile("t", count), Ok(()), "{count}");
        }
    }
}
