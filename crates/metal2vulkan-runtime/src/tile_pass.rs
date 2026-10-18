//! A render pass's tile memory (`MTLRenderPassDescriptor`'s `tileWidth`, `tileHeight`,
//! `imageblockSampleLength` and `threadgroupMemoryLength`): the checks Metal makes of them when the
//! encoder opens, and the tile size a tile dispatch runs over.
//!
//! Measured by our probes (an M3 Max and an M4, which agree, and an M4 alone):
//!
//! - **Tile sizes.** A pass's tile is 16x16, 32x16 or 32x32 pixels, or unset (both 0); any other
//!   pair is refused ("Invalid tile dimensions (8, 8)"), a pair with one side 0 included. Unset,
//!   the encoder's tile is 32x32, and 32x16 at 4 samples, where a 32x32 tile is refused ("4X MSAA
//!   does not support 32 x 32 tiles"); 2 samples take every size.
//! - **Per-sample storage** is the pass's `imageblockSampleLength` rounded up to 8 bytes, at
//!   least 8, whatever the attachments pack to: more than 64 is refused ("Per sample storage (72)
//!   cannot be greater than (64)"). Eight `RGBA32Float` attachments, 128 bytes, open a pass whose
//!   sample length is unset.
//! - **Tile memory** is that storage for every sample of every pixel of the tile, a 16x16 one
//!   when the tile is unset, plus the pass's `threadgroupMemoryLength`: more than 32768 bytes is
//!   refused ("Total allocated tile memory (40960) cannot be greater than (32768)" for 40 bytes
//!   a sample of 16x16 pixels at 4 samples), and so is a threadgroup memory length past 32768 by
//!   itself ("Requested threadgroup memory length exceeds amount supported by device (32768)").
//!   A threadgroup memory length need not be a multiple of 16 here.
//!
//! The checks run in that order, the first failing one refusing the encoder
//! ([`RenderPassProblem`]); Metal's validation states every failing one, and its encoder is nil
//! without validation too, so the runtime answers nil ([`TileRefusal::Nil`]); a 32x32 tile at 4
//! samples, which plain Metal faults on, is the validation's stop ([`TileRefusal::Stops`]). Apple's
//! GPUs have passes of 1, 2 and 4 samples only (8 is "not supported by device"): a pass of another
//! count, which a Vulkan device may have, has no Metal tile size, and one that sets any of the four
//! is not measured ([`TileRefusal::NotMeasured`], `NotImplemented(TileAtSampleCount)`).

use super::render_pass::RenderPassProblem;

/// The tile sizes a pass takes, as (width, height).
const TILE_SIZES: [[u32; 2]; 3] = [[16, 16], [32, 16], [32, 32]];

/// The sample counts of the passes Apple's GPUs make.
const APPLE_SAMPLE_COUNTS: [u32; 3] = [1, 2, 4];

/// The most bytes of per-sample storage a pass takes.
const MAX_PER_SAMPLE_STORAGE: u64 = 64;

/// The most bytes of tile memory a pass takes, and of threadgroup memory: Apple's GPUs'
/// `maxThreadgroupMemoryLength`.
const MAX_TILE_MEMORY: u64 = 32768;

/// A pass's tile memory, checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TilePass {
    /// The encoder's tile size (`tileWidth`, `tileHeight`): the pass's, or the default for its
    /// samples; `None` at a sample count Apple's GPUs lack.
    pub(super) tile: Option<[u32; 2]>,
    /// The pass's `imageblockSampleLength`, unrounded (0 unset).
    pub(super) sample_length: u64,
    /// The pass's `threadgroupMemoryLength`: the bytes of each tile's threadgroup memory, where
    /// its tile functions' threadgroup arguments sit (`render/tile_memory.rs`).
    pub(super) threadgroup_memory_length: u64,
}

/// Why a pass's tile memory is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum TileRefusal {
    /// As Metal refuses it, its encoder nil without validation (an [`Error::Nil`]).
    ///
    /// [`Error::Nil`]: crate::Error::Nil
    Nil(RenderPassProblem),
    /// A 32x32 tile at 4 samples, which Metal's validation stops and plain Metal faults on, as
    /// measured (an [`Error::Invalid`]).
    ///
    /// [`Error::Invalid`]: crate::Error::Invalid
    Stops(RenderPassProblem),
    /// A pass at a sample count Apple's GPUs lack sets its tile memory.
    NotMeasured,
}

/// The tile memory of a pass of `samples` whose descriptor sets `tile` (width, height),
/// `sample_length` and `threadgroup_memory_length`, checked as the module docs say.
pub(super) fn check(
    tile: [u64; 2],
    sample_length: u64,
    threadgroup_memory_length: u64,
    samples: u32,
) -> Result<TilePass, TileRefusal> {
    let nil = TileRefusal::Nil;
    if !APPLE_SAMPLE_COUNTS.contains(&samples) {
        if tile != [0, 0] || sample_length != 0 || threadgroup_memory_length != 0 {
            return Err(TileRefusal::NotMeasured);
        }
        return Ok(TilePass {
            tile: None,
            sample_length,
            threadgroup_memory_length,
        });
    }
    let [width, height] = tile;
    let set = match tile {
        [0, 0] => None,
        _ => Some(
            TILE_SIZES
                .iter()
                .copied()
                .find(|&[w, h]| u64::from(w) == width && u64::from(h) == height)
                .ok_or(nil(RenderPassProblem::TileSize { width, height }))?,
        ),
    };
    if samples == 4 && set == Some([32, 32]) {
        return Err(TileRefusal::Stops(RenderPassProblem::TileSizeAtSamples {
            samples,
        }));
    }
    let per_sample = sample_length.div_ceil(8).saturating_mul(8).max(8);
    if per_sample > MAX_PER_SAMPLE_STORAGE {
        return Err(nil(RenderPassProblem::PerSampleStorage {
            bytes: per_sample,
        }));
    }
    if threadgroup_memory_length > MAX_TILE_MEMORY {
        return Err(nil(RenderPassProblem::ThreadgroupMemoryLength {
            length: threadgroup_memory_length,
        }));
    }
    let [w, h] = set.unwrap_or([16, 16]);
    let bytes = per_sample * u64::from(w * h) * u64::from(samples) + threadgroup_memory_length;
    if bytes > MAX_TILE_MEMORY {
        return Err(nil(RenderPassProblem::TileMemory { bytes }));
    }
    let tile = match (set, samples) {
        (Some(tile), _) => tile,
        (None, 4) => [32, 16],
        (None, _) => [32, 32],
    };
    Ok(TilePass {
        tile: Some(tile),
        sample_length,
        threadgroup_memory_length,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a pass of `samples` with `tile`, `sample_length` and `threadgroup` answers: its
    /// tile, or its refusal.
    fn pass(
        tile: [u64; 2],
        sample_length: u64,
        threadgroup: u64,
        samples: u32,
    ) -> Result<Option<[u32; 2]>, TileRefusal> {
        check(tile, sample_length, threadgroup, samples).map(|pass| pass.tile)
    }

    /// [`RenderPassProblem`]'s refusal.
    fn refused(problem: RenderPassProblem) -> Result<Option<[u32; 2]>, TileRefusal> {
        Err(TileRefusal::Nil(problem))
    }

    #[test]
    fn a_pass_takes_metals_tile_sizes() {
        for samples in [1, 2] {
            for tile in [[16, 16], [32, 16], [32, 32]] {
                let answer = Some([tile[0] as u32, tile[1] as u32]);
                assert_eq!(pass(tile, 0, 0, samples), Ok(answer));
            }
            assert_eq!(pass([0, 0], 0, 0, samples), Ok(Some([32, 32])));
        }
        assert_eq!(pass([0, 0], 0, 0, 4), Ok(Some([32, 16])));
        assert_eq!(pass([16, 16], 0, 0, 4), Ok(Some([16, 16])));
        assert_eq!(pass([32, 16], 0, 0, 4), Ok(Some([32, 16])));
        for [width, height] in [
            [1, 1],
            [4, 4],
            [8, 8],
            [24, 24],
            [48, 48],
            [64, 64],
            [8, 16],
            [16, 8],
            [16, 32],
            [0, 16],
            [16, 0],
        ] {
            assert_eq!(
                pass([width, height], 0, 0, 1),
                refused(RenderPassProblem::TileSize { width, height })
            );
        }
        assert_eq!(
            pass([32, 32], 0, 0, 4),
            Err(TileRefusal::Stops(RenderPassProblem::TileSizeAtSamples {
                samples: 4
            }))
        );
        // No Apple GPU makes a pass of 8 samples.
        assert_eq!(pass([0, 0], 0, 0, 8), Ok(None));
        assert_eq!(pass([16, 16], 0, 0, 8), Err(TileRefusal::NotMeasured));
        assert_eq!(pass([0, 0], 16, 0, 8), Err(TileRefusal::NotMeasured));
        assert_eq!(pass([0, 0], 0, 16, 8), Err(TileRefusal::NotMeasured));
    }

    #[test]
    fn a_pass_takes_metals_tile_memory() {
        let memory = |bytes| refused(RenderPassProblem::TileMemory { bytes });
        let threadgroup = |length| refused(RenderPassProblem::ThreadgroupMemoryLength { length });
        let per_sample = |bytes| refused(RenderPassProblem::PerSampleStorage { bytes });
        // The measured passes: tile, sample length, threadgroup memory
        // length, samples.
        for (tile, sample_length, tg, samples, answer) in [
            ([32, 32], 32, 0, 1, Ok(Some([32, 32]))),
            ([32, 32], 33, 0, 1, memory(40960)),
            ([32, 32], 64, 0, 1, memory(65536)),
            ([16, 16], 64, 0, 1, Ok(Some([16, 16]))),
            ([16, 16], 65, 0, 1, per_sample(72)),
            ([32, 16], 64, 0, 1, Ok(Some([32, 16]))),
            ([32, 16], 40, 0, 1, Ok(Some([32, 16]))),
            ([0, 0], 33, 0, 1, Ok(Some([32, 32]))),
            ([0, 0], 64, 0, 1, Ok(Some([32, 32]))),
            ([0, 0], 65, 0, 1, per_sample(72)),
            ([32, 32], 0, 32768, 1, memory(40960)),
            ([32, 32], 0, 32784, 1, threadgroup(32784)),
            ([32, 32], 16, 16384, 1, Ok(Some([32, 32]))),
            ([32, 32], 16, 16400, 1, memory(32784)),
            ([32, 16], 32, 16384, 1, Ok(Some([32, 16]))),
            ([32, 16], 32, 16400, 1, memory(32784)),
            ([0, 0], 64, 16384, 1, Ok(Some([32, 32]))),
            ([0, 0], 64, 16400, 1, memory(32784)),
            ([0, 0], 0, 32784, 1, threadgroup(32784)),
            ([0, 0], 0, 8, 1, Ok(Some([32, 32]))),
            ([0, 0], 0, 24, 1, Ok(Some([32, 32]))),
            ([16, 16], 1, 30720, 1, Ok(Some([16, 16]))),
            ([16, 16], 1, 30736, 1, memory(32784)),
            ([16, 16], 256, 0, 1, per_sample(256)),
            ([32, 32], 24, 0, 2, memory(49152)),
            ([32, 32], 32, 0, 2, memory(65536)),
            ([0, 0], 40, 0, 2, Ok(Some([32, 32]))),
            ([16, 16], 32, 0, 4, Ok(Some([16, 16]))),
            ([16, 16], 40, 0, 4, memory(40960)),
            ([16, 16], 64, 16384, 4, memory(81920)),
            ([32, 16], 40, 0, 4, memory(81920)),
            ([32, 16], 16, 0, 4, Ok(Some([32, 16]))),
            ([0, 0], 32, 0, 4, Ok(Some([32, 16]))),
            ([0, 0], 40, 0, 4, memory(40960)),
        ] {
            assert_eq!(
                pass(tile, sample_length, tg, samples),
                answer,
                "{tile:?} {sample_length} {tg} {samples}"
            );
        }
    }
}
