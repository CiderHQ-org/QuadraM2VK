//! Rasterization-rate maps through the runtime on the device the cfg selects, under the validation
//! layer:
//!
//! - Every committed public kernel case whose buffers or argument buffers hold a map's parameter
//!   data runs through the runtime ([`census`]): each map is made with
//!   `new_rasterization_rate_map` and copied into a buffer of the size it answers with
//!   `copy_parameter_data_to_buffer`, as the Metal oracle makes its own.
//! - A map answers Metal's queries as an M3 Max and an M4 do, a map of no layers included, its
//!   creation refuses what Metal's API validation stops (but a count of 0, which plain Metal makes)
//!   and what the compile layer does not compile, and its copy writes Metal's parameter data at any
//!   offset of a buffer the CPU reaches, refusing the rest.

use super::degrade_notices::{assert_once, counts};
use super::kernel_textures::census;
use super::{public_cases, read_buffer, validated_device, Messages, ScratchCache};
use crate::case::Stage;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::{
    RasterizationRateMap as CompiledMap, RateMapDescriptor, RateMapLayerDescriptor, UnpinnedCompile,
};
use metal2vulkan_runtime::{
    Coordinate2D, CpuCacheMode, Degradation, Device, Error, HazardTrackingMode, Invalid, Problem,
    RateMapProblem, ResourceOptions, Size, StorageMode,
};

/// Every committed public kernel case whose buffers or argument buffers hold a rasterization-rate
/// map's parameter data runs through the runtime as `census` judges it.
#[test]
fn kernels_reading_rate_maps_match_metal() {
    let cases = public_cases(|case| {
        case.stage == Stage::Kernel
            && case.public_source.is_some()
            && (case
                .buffers
                .iter()
                .any(|buffer| buffer.rasterization_rate_map.is_some())
                || case
                    .argument_buffer_buffers
                    .iter()
                    .any(|buffer| buffer.rasterization_rate_map.is_some()))
    })
    .unwrap();
    assert!(
        !cases.is_empty(),
        "no committed kernel case reads a rate map"
    );
    census("runtime rate-map kernels", &cases);
}

/// Run `test` on a validated device of the lane reporting to its messages, then fail on any
/// validation message.
fn on_device(test: impl FnOnce(&Device, &Messages)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        test(&device, &messages);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// A descriptor of `width` by `height` screen pixels and these layers, each one sample per axis.
pub(super) fn descriptor(width: u32, height: u32, layers: &[(f32, f32)]) -> RateMapDescriptor {
    let mut descriptor = RateMapDescriptor::new([width, height]);
    descriptor.layers = layers
        .iter()
        .map(|&(h, v)| RateMapLayerDescriptor::new(vec![h], vec![v]))
        .collect();
    descriptor
}

/// `problem` at `new_rasterization_rate_map`.
fn refused(problem: RateMapProblem) -> Error {
    Error::Invalid(Invalid::new(
        "new_rasterization_rate_map",
        Problem::RateMap(problem),
    ))
}

/// A map of half quality over 64 by 64 screen pixels answers what an M3 Max and an M4 answer: its
/// screen and granularity of depth 1, one layer of 33 by 33 physical pixels, width 0 and the shared
/// height past it, the label it was made with, and Metal's coordinate maps, a layer past the map's
/// reading the last. Its parameter data is Metal's 24,624 bytes, aligned to 4. A map of no layers
/// has no width and its vertical axis's 16 rows. The first map that is not the identity announces
/// the `rasterization-rate-map` degradation, once; an identity map announces nothing.
#[test]
fn a_rate_map_answers_as_metal_does() {
    on_device(|device, messages| {
        // What the device announced before its first map.
        messages.take_notices();
        let identity = device
            .new_rasterization_rate_map(&descriptor(64, 64, &[(1.0, 1.0)]))
            .unwrap();
        assert_eq!(identity.physical_size_for_layer(0), Size::new(64, 64, 1));
        assert_eq!(messages.take_notices(), Vec::new(), "an identity map");
        let mut half = descriptor(64, 64, &[(0.5, 0.5)]);
        half.label = Some("rm".to_owned());
        let map = device.new_rasterization_rate_map(&half).unwrap();
        assert_eq!(map.device().name(), device.name());
        assert_eq!(map.label(), Some("rm"));
        assert_eq!(map.screen_size(), Size::new(64, 64, 1));
        assert_eq!(map.physical_granularity(), Size::new(32, 32, 1));
        assert_eq!(map.layer_count(), 1);
        assert_eq!(map.physical_size_for_layer(0), Size::new(33, 33, 1));
        assert_eq!(map.physical_size_for_layer(1), Size::new(0, 33, 1));
        assert_eq!(map.physical_size_for_layer(u64::MAX), Size::new(0, 33, 1));
        let at = Coordinate2D::new(10.0, 10.0);
        for layer in [0, 1, u64::MAX] {
            let physical = map.map_screen_to_physical_coordinates(at, layer);
            let screen = map.map_physical_to_screen_coordinates(at, layer);
            for value in [physical.x, physical.y] {
                assert!(
                    (value - 5.079_365_3).abs() <= 1.0e-5,
                    "{layer}: {physical:?}"
                );
            }
            for value in [screen.x, screen.y] {
                assert!((value - 19.687_498).abs() <= 1.0e-5, "{layer}: {screen:?}");
            }
        }
        let compiled = CompiledMap::new(&half).unwrap().to_bytes();
        let sized = map.parameter_buffer_size_and_align();
        assert_eq!((sized.size, sized.align), (24_624, 4));
        assert_eq!(compiled.len(), 24_624);
        let empty = device
            .new_rasterization_rate_map(&descriptor(256, 256, &[]))
            .unwrap();
        assert_eq!(empty.layer_count(), 0);
        assert_eq!(empty.physical_granularity(), Size::new(32, 32, 1));
        assert_eq!(empty.physical_size_for_layer(0), Size::new(0, 16, 1));
        let two = device
            .new_rasterization_rate_map(&descriptor(64, 64, &[(0.5, 0.5), (1.0, 0.25)]))
            .unwrap();
        assert_eq!(two.label(), None);
        assert_eq!(two.layer_count(), 2);
        assert_eq!(two.physical_size_for_layer(0), Size::new(33, 33, 1));
        assert_eq!(two.physical_size_for_layer(1), Size::new(64, 33, 1));
        assert_once(
            &messages.take_notices(),
            &[Degradation::RasterizationRateMap],
        );
    });
}

/// Creation refuses what Metal's API validation stops, typed, and, `Error::RateMap`, what the
/// compile layer does not compile.
#[test]
fn a_rate_map_is_refused_where_metal_or_the_compile_stops_it() {
    on_device(|device, _| {
        let make = |d: &RateMapDescriptor| device.new_rasterization_rate_map(d).map(|_| ());
        let half = [(0.5, 0.5)];
        for (width, height) in [(0, 64), (64, 0), (0, 0), (32769, 64), (64, 40000)] {
            assert_eq!(
                make(&descriptor(width, height, &half)),
                Err(refused(RateMapProblem::ScreenSize { width, height })),
                "{width}x{height}"
            );
        }
        assert_eq!(
            make(&descriptor(64, 64, &[(0.5, 0.5); 3])),
            Err(refused(RateMapProblem::LayerCount { count: 3 })),
        );
        // A count of 0, which Metal's validation stops and plain Metal makes, is a map of none.
        let mut nil_first = descriptor(64, 64, &[(0.5, 0.5)]);
        nil_first.layers[0].horizontal.clear();
        assert_eq!(make(&nil_first), Ok(()));
        assert_eq!(make(&descriptor(64, 64, &[])), Ok(()));
        assert_eq!(
            make(&descriptor(16385, 64, &half)),
            Err(Error::RateMap(UnpinnedCompile::ScreenExtent {
                extent: 16385
            }))
        );
    });
}

/// `copy_parameter_data_to_buffer` writes the map's parameter data, Metal's bytes, at the
/// offset, misaligned included, into a `Shared` or `Managed` buffer, leaving the other bytes; it
/// refuses a `Private` buffer, a write past the buffer's length, and a buffer of another device,
/// changing nothing. No copy announces anything (the bytes are Metal's).
#[test]
fn parameter_data_copies_where_the_cpu_reaches() {
    on_device(|device, messages| {
        let half = descriptor(64, 64, &[(0.5, 0.5)]);
        let map = device.new_rasterization_rate_map(&half).unwrap();
        // What the device announced before its first copy (the map's own kind).
        messages.take_notices();
        let data = CompiledMap::new(&half).unwrap().to_bytes();
        let size = data.len() as u64;
        let options = |storage| {
            ResourceOptions::new(
                storage,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Tracked,
            )
        };
        for (storage, offset) in [
            (StorageMode::Shared, 0),
            (StorageMode::Shared, 16),
            (StorageMode::Shared, 2),
            (StorageMode::Managed, 0),
        ] {
            let buffer = device.new_buffer(size + 64, options(storage)).unwrap();
            map.copy_parameter_data_to_buffer(&buffer, offset).unwrap();
            let bytes = read_buffer(&buffer, 0, size + 64).unwrap();
            let at = offset as usize;
            assert_eq!(
                &bytes[at..at + data.len()],
                &data[..],
                "{storage:?} at {offset}"
            );
            assert!(
                bytes[..at]
                    .iter()
                    .chain(&bytes[at + data.len()..])
                    .all(|&b| b == 0),
                "{storage:?} at {offset}: bytes outside the copy changed"
            );
        }
        let copy = |buffer, offset| map.copy_parameter_data_to_buffer(buffer, offset);
        let invalid = |problem| {
            Err(Error::Invalid(Invalid::new(
                "copy_parameter_data_to_buffer",
                problem,
            )))
        };
        // Other kinds the copies' buffers made announce (NVIDIA's `shared-in-system-memory`)
        // are the device's; no copy announces a rate-map kind.
        let copies = counts(&messages.take_notices());
        assert!(
            !copies.contains_key(&Degradation::RasterizationRateMap),
            "{copies:?}"
        );
        let private = device
            .new_buffer(size + 64, options(StorageMode::Private))
            .unwrap();
        assert_eq!(copy(&private, 0), invalid(Problem::StorageMode));
        let short = device
            .new_buffer(size, options(StorageMode::Shared))
            .unwrap();
        assert_eq!(
            copy(&short, 16),
            invalid(Problem::RangeOutOfBounds {
                offset: 16,
                size,
                length: size,
            })
        );
        assert_eq!(
            copy(&short, u64::MAX),
            invalid(Problem::RangeOutOfBounds {
                offset: u64::MAX,
                size,
                length: size,
            })
        );
        assert!(read_buffer(&short, 0, size)
            .unwrap()
            .iter()
            .all(|&b| b == 0));
        // The refused copies announce no rate-map kind either.
        let later = counts(&messages.take_notices());
        assert!(
            !later.contains_key(&Degradation::RasterizationRateMap),
            "{later:?}"
        );
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        let other = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let foreign = other
            .new_buffer(size, options(StorageMode::Shared))
            .unwrap();
        assert_eq!(copy(&foreign, 0), invalid(Problem::ForeignObject));
        drop((foreign, other));
        let reported = messages.take();
        assert!(reported.is_empty(), "the second device: {reported:#?}");
    });
}
