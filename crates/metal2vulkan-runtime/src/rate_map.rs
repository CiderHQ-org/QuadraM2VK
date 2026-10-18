//! `RasterizationRateMap` (`MTLRasterizationRateMap`) and `Device::new_rasterization_rate_map`
//! (Metal's bytes and arithmetic).
//!
//! - **The compile layer's map.** A map is the compile layer's compile of its descriptor
//!   ([`metal2vulkan::bind::RasterizationRateMap::new`], Apple's rule), with the descriptor's
//!   label. Its parameter data is Metal's: 24,624 bytes in Apple's layout, which a function's
//!   `rasterization_rate_map_decoder` reads from a buffer bound like any other, with Apple's GPU
//!   arithmetic; and its coordinate maps are Apple's CPU arithmetic over the same bytes. Measured
//!   on an Apple M4 and an M3 Max: the bytes of every map probed and every answer, bit for bit. A
//!   render pass over a map that is not the identity draws by regions whose lines are the tiles'
//!   exact lines, a near-equivalent of Apple's fixed-point ones, so the map's creation announces
//!   the `rasterization-rate-map` degradation.
//! - **What Metal's API validation stops**, measured by our probes on an M3
//!   Max and an M4 alike, is refused at creation, `Invalid` with [`RateMapProblem`]: a screen width
//!   or height of 0 or past 32768, and a layer count past 2. The count is Metal's `layerCount`:
//!   the layers from index 0 up to the first without samples on an axis, which Metal's descriptor
//!   holds as nil; the map is made of those, and the layers after are ignored, as Metal ignores
//!   them. Metal's validation also stops a count of 0, but Metal makes such a map without it, and
//!   so does the runtime: its parameter data is Metal's (no horizontal tables, a vertical axis of
//!   the least quality), and a render pass over it runs its load, clear, and store actions and
//!   draws nothing.
//! - **Where the compile is this crate's bound.** Metal takes a screen of up to 32768 pixels a
//!   side and any samples; the compile layer compiles only where Apple's compile is measured,
//!   and refuses the rest, `Error::RateMap` with its
//!   [`UnpinnedCompile`](metal2vulkan::bind::UnpinnedCompile): a screen past 16384 a side (where Apple's
//!   compile crashes or corrupts the map) and sample densities whose compile is not measured.
//! - **Metal's answers.** `screenSize` is the descriptor's, depth 1; `physicalGranularity` 32 by
//!   32, depth 1; `parameterBufferSizeAndAlign` 24,624 bytes, align 4; the coordinate maps past
//!   the layers read the last layer (both Macs, probes 7 and 9). `physicalSizeForLayer:` past the
//!   layers is the layers' shared vertical extent on Metal too, but its width Metal reads past
//!   its own layer table (probe 9: 0 just past a map of one layer, else a screen side, the
//!   vertical extent or a small number, by index; a map of no layers answers 0 by 16 at layers 0
//!   and 1 and 16 by 16 at layer 2): the runtime answers width 0 there, and
//!   its vertical extent. A map of no layers' coordinate maps end the process on Metal (all 12
//!   probed queries); the runtime answers the decode of its bytes, 0 across.
//! - **A map of no layers made after a released one**. The device notes when it releases a map of
//!   one or more layers (its last handle dropped); a map of no layers made after that is one whose
//!   passes Metal draws (an Apple M4 and an Apple M3 Max drew a 64-pixel map's rows 0 to 15 as with
//!   no map), where the runtime draws nothing, so a draw in such a pass announces
//!   `zero-layer-map-after-release` (`rate_mapped_pass.rs`).
//! - **The copy.** `copy_parameter_data_to_buffer` writes the parameter data through the buffer's
//!   mapping, at any offset, as Metal takes a misaligned one. It refuses a buffer of another
//!   device, a `Private` buffer (Metal's copy ends the process there), and a write past the
//!   buffer's length (Metal's writes past it). The bytes are Metal's.

use super::buffer::Buffer;
use super::capabilities::RATE_MAP_LAYERS;
use super::degradation::Degradation;
use super::device::Device;
use super::error::{internal, invalid, Error, Problem};
use super::heap::SizeAndAlign;
use super::values::{Coordinate2D, Size};
use metal2vulkan::bind::{RateMapDescriptor, RateMapError};
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// The largest screen width or height Metal's API validation takes ("screenSize.width (0)
/// must be >0 and <=32768").
const MAX_SCREEN_EXTENT: u32 = 32768;

/// What a rasterization-rate map's descriptor did wrong (`Problem::RateMap`): what Metal's API
/// validation stops at `newRasterizationRateMapWithDescriptor:` (an M3 Max and an M4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateMapProblem {
    /// A screen width or height of 0 or past 32768.
    ScreenSize {
        /// The screen width given.
        width: u32,
        /// The screen height given.
        height: u32,
    },
    /// A layer count past 2 ("layerCount (3) must be >0 and <=2"): the layers from index 0 up to
    /// the first without samples on an axis, the count Metal's descriptor answers. A count of 0,
    /// which Metal's validation also stops, makes a map of no layers, as plain Metal does (the
    /// module docs).
    LayerCount {
        /// The descriptor's layer count.
        count: usize,
    },
}

/// A rasterization-rate map (`MTLRasterizationRateMap`). Cheap to clone (a retain); `Send +
/// Sync`. Two handles are equal when they are the same map.
#[derive(Clone)]
pub struct RasterizationRateMap(Arc<MapShared>);

struct MapShared {
    device: Device,
    label: Option<String>,
    map: metal2vulkan::bind::RasterizationRateMap,
    /// A map of no layers made after the device released a map of layers (the module docs).
    after_release: bool,
}

impl Drop for MapShared {
    fn drop(&mut self) {
        if self.map.layer_count() > 0 {
            self.device
                .shared()
                .layered_rate_map_released
                .store(true, Ordering::Relaxed);
        }
    }
}

impl std::fmt::Debug for RasterizationRateMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RasterizationRateMap")
            .field("label", &self.0.label)
            .field("screen_size", &self.0.map.screen_size())
            .field("layer_count", &self.0.map.layer_count())
            .finish()
    }
}

impl PartialEq for RasterizationRateMap {
    fn eq(&self, other: &RasterizationRateMap) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for RasterizationRateMap {}

impl Device {
    /// `newRasterizationRateMapWithDescriptor:` (the module docs): refused `Invalid` where Metal's
    /// API validation stops the descriptor, and `Error::RateMap` where the compile layer does not
    /// compile it.
    pub fn new_rasterization_rate_map(
        &self,
        descriptor: &RateMapDescriptor,
    ) -> Result<RasterizationRateMap, Error> {
        let map = compile(descriptor)?;
        if !map.is_identity() {
            self.degrade(Degradation::RasterizationRateMap, || {
                let [width, height] = map.screen_size();
                (
                    format!("a rasterization rate map of {width} by {height} screen pixels"),
                    "a render pass over it draws by regions whose lines are its tiles' exact \
                     lines, where Apple's are its fixed-point ones; its \
                     parameter data and coordinate maps are Metal's"
                        .to_owned(),
                )
            });
        }
        let after_release = map.layer_count() == 0
            && self
                .shared()
                .layered_rate_map_released
                .load(Ordering::Relaxed);
        Ok(RasterizationRateMap(Arc::new(MapShared {
            device: self.clone(),
            label: descriptor.label.clone(),
            map,
            after_release,
        })))
    }
}

/// The compile layer's map of `descriptor`, or why Metal's API validation (`Invalid`) or the
/// compile layer (`Error::RateMap`) stops it (the module docs).
pub(super) fn compile(
    descriptor: &RateMapDescriptor,
) -> Result<metal2vulkan::bind::RasterizationRateMap, Error> {
    const CALL: &str = "new_rasterization_rate_map";
    let refused = |problem| invalid(CALL, Problem::RateMap(problem));
    let [width, height] = descriptor.screen_size;
    let extent = 1..=MAX_SCREEN_EXTENT;
    if !extent.contains(&width) || !extent.contains(&height) {
        return Err(refused(RateMapProblem::ScreenSize { width, height }));
    }
    // Metal's `layerCount`: a layer without samples on an axis is nil, and the count stops there.
    let count = descriptor
        .layers
        .iter()
        .take_while(|layer| !(layer.horizontal.is_empty() || layer.vertical.is_empty()))
        .count();
    if count as u64 > RATE_MAP_LAYERS {
        return Err(refused(RateMapProblem::LayerCount { count }));
    }
    let counted;
    let descriptor = if count == descriptor.layers.len() {
        descriptor
    } else {
        let mut kept = descriptor.clone();
        kept.layers.truncate(count);
        counted = kept;
        &counted
    };
    metal2vulkan::bind::RasterizationRateMap::new(descriptor).map_err(|error| match error {
        RateMapError::Unpinned { reason } => Error::RateMap(reason),
        other => internal(&format!(
            "the compile layer refused a rate map Metal's validation takes: {other}"
        )),
    })
}

/// `physicalSizeForLayer:` of `map`: the layer's physical width and height, depth 1; past the
/// layers (every layer of a map of none), width 0 and the vertical extent the layers share (the
/// module docs).
pub(super) fn physical_size(map: &metal2vulkan::bind::RasterizationRateMap, layer: u64) -> Size {
    match u32::try_from(layer).ok().and_then(|l| map.physical_size(l)) {
        Some([width, height]) => Size::new(width.into(), height.into(), 1),
        None => Size::new(0, map.vertical().physical_extent().into(), 1),
    }
}

/// A layer index as the compile layer's coordinate maps take it: one past `u32` reads the last
/// layer, as every index past the layers does.
fn layer_index(layer: u64) -> u32 {
    u32::try_from(layer).unwrap_or(u32::MAX)
}

impl RasterizationRateMap {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// Whether the map has no layers and was made after its device released a map of layers:
    /// one whose passes Metal may draw from the released map's memory (the module docs).
    pub(super) fn made_after_release(&self) -> bool {
        self.0.after_release
    }

    /// The compile layer's map, which a rate-mapped pass plans its draws with
    /// (`rate_mapped_pass.rs`).
    pub(super) fn compiled(&self) -> &metal2vulkan::bind::RasterizationRateMap {
        &self.0.map
    }

    /// `label`: the descriptor's.
    pub fn label(&self) -> Option<&str> {
        self.0.label.as_deref()
    }

    /// `screenSize`: the descriptor's width and height, depth 1 (as Metal answers on an M3 Max
    /// and an M4, where its header says 0).
    pub fn screen_size(&self) -> Size {
        let [width, height] = self.0.map.screen_size();
        Size::new(width.into(), height.into(), 1)
    }

    /// `physicalGranularity`: the physical width and height of every tile but each axis's last,
    /// 32 by 32, depth 1.
    pub fn physical_granularity(&self) -> Size {
        let [width, height] = self.0.map.physical_granularity();
        Size::new(width.into(), height.into(), 1)
    }

    /// `layerCount`.
    pub fn layer_count(&self) -> u64 {
        self.0.map.layer_count().into()
    }

    /// `parameterBufferSizeAndAlign`: 24,624 bytes, aligned to 4, Metal's for every map.
    pub fn parameter_buffer_size_and_align(&self) -> SizeAndAlign {
        SizeAndAlign {
            size: self.0.map.parameter_data_size(),
            align: metal2vulkan::bind::RasterizationRateMap::PARAMETER_DATA_ALIGNMENT,
        }
    }

    /// `physicalSizeForLayer:` (`physical_size`).
    pub fn physical_size_for_layer(&self, layer: u64) -> Size {
        physical_size(&self.0.map, layer)
    }

    /// `mapScreenToPhysicalCoordinates:forLayer:`: where `screen` lands in the layer's physical
    /// space, as Apple's CPU map answers it, bit for bit. A layer past the map's reads the last;
    /// over a map of no layers, 0 across (the module docs).
    pub fn map_screen_to_physical_coordinates(
        &self,
        screen: Coordinate2D,
        layer: u64,
    ) -> Coordinate2D {
        let [x, y] = self
            .0
            .map
            .screen_to_physical([screen.x, screen.y], layer_index(layer));
        Coordinate2D::new(x, y)
    }

    /// `mapPhysicalToScreenCoordinates:forLayer:`: the screen coordinate of `physical` in the
    /// layer, as Apple's CPU map answers it, bit for bit. A layer past the map's reads the last;
    /// over a map of no layers, 0 across (the module docs).
    pub fn map_physical_to_screen_coordinates(
        &self,
        physical: Coordinate2D,
        layer: u64,
    ) -> Coordinate2D {
        let [x, y] = self
            .0
            .map
            .physical_to_screen([physical.x, physical.y], layer_index(layer));
        Coordinate2D::new(x, y)
    }

    /// `copyParameterDataToBuffer:offset:`: Metal's parameter data, written at `offset` of
    /// `buffer` through its mapping (the module docs). Refused for a buffer of another device
    /// (`ForeignObject`), a `Private` one (`StorageMode`), and a write past its length
    /// (`RangeOutOfBounds`).
    pub fn copy_parameter_data_to_buffer(&self, buffer: &Buffer, offset: u64) -> Result<(), Error> {
        const CALL: &str = "copy_parameter_data_to_buffer";
        if !buffer.device().same(&self.0.device) {
            return Err(invalid(CALL, Problem::ForeignObject));
        }
        let Some(contents) = buffer.contents() else {
            return Err(invalid(CALL, Problem::StorageMode));
        };
        let bytes = self.0.map.to_bytes();
        let size = bytes.len() as u64;
        let length = buffer.length();
        if offset.checked_add(size).is_none_or(|end| end > length) {
            return Err(invalid(
                CALL,
                Problem::RangeOutOfBounds {
                    offset,
                    size,
                    length,
                },
            ));
        }
        // SAFETY: the buffer's mapping holds its `length` bytes from `contents`, and
        // `offset + size` is at most that. A command that reads or writes those bytes meanwhile
        // races, as on Metal.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                contents.as_ptr().add(offset as usize),
                bytes.len(),
            )
        };
        Ok(())
    }
}
