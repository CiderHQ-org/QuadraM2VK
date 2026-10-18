//! The rasterization-rate map's creation rules and answers that need no device
//! (`rate_map::compile`, `rate_map::physical_size`).

use super::super::error::{Error, Invalid, Problem};
use super::super::rate_map::{compile, physical_size, RateMapProblem};
use super::super::values::Size;
use metal2vulkan::bind::{
    RasterizationRateMap, RateMapDescriptor, RateMapLayerDescriptor, UnpinnedCompile,
};

/// A descriptor of `width` by `height` screen pixels and these layers' samples.
fn descriptor(width: u32, height: u32, layers: &[(&[f32], &[f32])]) -> RateMapDescriptor {
    let mut descriptor = RateMapDescriptor::new([width, height]);
    descriptor.layers = layers
        .iter()
        .map(|&(h, v)| RateMapLayerDescriptor::new(h.to_vec(), v.to_vec()))
        .collect();
    descriptor
}

/// `problem` at `new_rasterization_rate_map`.
fn refused(problem: RateMapProblem) -> Result<RasterizationRateMap, Error> {
    Err(Error::Invalid(Invalid::new(
        "new_rasterization_rate_map",
        Problem::RateMap(problem),
    )))
}

const HALF: (&[f32], &[f32]) = (&[0.5], &[0.5]);

/// A screen of 1 to 32768 pixels a side passes Metal's validation, and 0 or more does not; the
/// screen's check comes first.
#[test]
fn a_screen_side_is_1_to_32768_pixels() {
    for (width, height) in [
        (0, 64),
        (64, 0),
        (0, 0),
        (32769, 1),
        (1, 32769),
        (u32::MAX, 64),
    ] {
        assert_eq!(
            compile(&descriptor(width, height, &[HALF])),
            refused(RateMapProblem::ScreenSize { width, height }),
            "{width}x{height}"
        );
    }
    assert_eq!(
        compile(&descriptor(0, 64, &[])),
        refused(RateMapProblem::ScreenSize {
            width: 0,
            height: 64
        })
    );
    for (width, height) in [(1, 1), (64, 16384), (16384, 1)] {
        assert!(
            compile(&descriptor(width, height, &[HALF])).is_ok(),
            "{width}x{height}"
        );
    }
}

/// A map takes 0 to 2 layers, counted as Metal counts them: from index 0 up to the first layer
/// without samples on an axis (nil in Metal's descriptor). The layers after it are ignored. A count
/// of 0 makes a map of no layers, as plain Metal does.
#[test]
fn a_map_takes_zero_to_two_layers() {
    let empty = compile(&descriptor(64, 64, &[])).unwrap();
    assert_eq!(empty.layer_count(), 0);
    assert_eq!(
        compile(&descriptor(64, 64, &[HALF; 3])),
        refused(RateMapProblem::LayerCount { count: 3 })
    );
    let none: &[f32] = &[];
    let quarter: (&[f32], &[f32]) = (&[0.25], &[0.25]);
    for layers in [&[(none, &[0.5][..]), quarter][..], &[(&[0.5][..], none)]] {
        assert_eq!(compile(&descriptor(64, 64, layers)), Ok(empty.clone()));
    }
    let first = compile(&descriptor(64, 64, &[HALF])).unwrap();
    for layers in [
        &[HALF, (none, &[0.5][..])][..],
        &[HALF, (&[0.5][..], none), quarter],
    ] {
        assert_eq!(compile(&descriptor(64, 64, layers)), Ok(first.clone()));
    }
    let two = compile(&descriptor(64, 64, &[HALF, quarter])).unwrap();
    assert_eq!(
        compile(&descriptor(64, 64, &[HALF, quarter, (none, none), HALF])),
        Ok(two)
    );
}

/// Past Metal's validation, the compile layer's bounds are `Error::RateMap`: a screen past 16384 a
/// side, and samples whose compile is not measured.
#[test]
fn the_compile_layers_bounds_are_its_own_error() {
    for (width, height, extent) in [(16385, 64, 16385), (64, 32768, 32768)] {
        assert_eq!(
            compile(&descriptor(width, height, &[HALF])),
            Err(Error::RateMap(UnpinnedCompile::ScreenExtent { extent })),
            "{width}x{height}"
        );
    }
    let dense: &[f32] = &[0.5, 0.25, 0.5, 0.25];
    assert!(matches!(
        compile(&descriptor(2, 64, &[(dense, &[0.5])])),
        Err(Error::RateMap(UnpinnedCompile::DenseSamples { .. }))
    ));
}

/// `physicalSizeForLayer:` is the layer's physical size, depth 1, and past the layers width 0
/// and the layers' shared vertical extent: Metal's height there, and its width just past a map
/// of one layer.
#[test]
fn a_layer_past_the_map_has_no_width_and_the_shared_height() {
    let map = compile(&descriptor(64, 64, &[HALF, (&[1.0], &[0.25])])).unwrap();
    assert_eq!(physical_size(&map, 0), Size::new(33, 33, 1));
    assert_eq!(physical_size(&map, 1), Size::new(64, 33, 1));
    for past in [2, u64::from(u32::MAX), u64::MAX] {
        assert_eq!(physical_size(&map, past), Size::new(0, 33, 1), "{past}");
    }
    // A map of no layers: width 0 and its vertical axis's 16 rows at layers 0 and 1, as Metal
    // answers (an M4); at layer 2 Metal reads 16 past its table, the runtime 0.
    let empty = compile(&descriptor(256, 256, &[])).unwrap();
    for layer in [0, 1, 2] {
        assert_eq!(physical_size(&empty, layer), Size::new(0, 16, 1), "{layer}");
    }
}
