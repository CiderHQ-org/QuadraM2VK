//! A render pass's sample count: the schema, its identity, the checks, the literal bytes and
//! selections of a multisampled attachment, and the requirement it carries.

use super::*;
use crate::executor_contract::case_requirements;
use crate::executor_contract::multisample::{attachment_transfer, AttachmentTransfer};
use crate::literal::LiteralResources;

/// A fragment case drawing into one `r8_unorm` target of `dimensions` holding `samples` samples
/// of each pixel (sample `s` of pixel `i` is `16 * i + s`), with a depth aspect, selecting the
/// whole target. The pass's count is left as authored by the caller.
fn fragment_case(dimensions: [u32; 2], samples: u32) -> AuthoredCase {
    let mut case = example();
    case.stage = Stage::Fragment;
    case.dispatch = None;
    case.buffers = vec![];
    case.draw = Some(Draw {
        primitive: Primitive::Triangle,
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        index_buffer: None,
        base_vertex: 0,
        base_instance: 0,
    });
    let pixels = dimensions[0] * dimensions[1];
    let bytes = (0..pixels)
        .flat_map(|pixel| (0..samples).map(move |sample| (16 * pixel + sample) as u8))
        .collect::<Vec<_>>();
    case.render_targets = vec![RenderTargetResource {
        index: 0,
        format: TextureFormat::R8Unorm,
        dimensions,
        initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
    }];
    case.depth_stencil = Some(DepthStencilResource {
        dimensions,
        initial_depth_b64: Some(base64::engine::general_purpose::STANDARD.encode(vec![
            0;
            (pixels * samples * 4)
                as usize
        ])),
        initial_stencil_b64: None,
        state: None,
    });
    case.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [0, 0],
        dimensions,
    };
    case.case_id = case.computed_case_id().unwrap();
    case
}

fn errors(case: &AuthoredCase) -> Vec<String> {
    let mut case = case.clone();
    case.case_id = case.computed_case_id().unwrap();
    case.validate_literal_resources().err().unwrap_or_default()
}

/// The field is part of the identity only when present: a case of one sample omits it and keeps
/// the identity it had before the field existed.
#[test]
fn a_render_pass_carries_its_sample_count_only_above_one() {
    let plain = fragment_case([2, 1], 1);
    let text = serde_json::to_string(&plain).unwrap();
    assert!(!text.contains("raster_sample_count"), "{text}");
    assert_eq!(plain.raster_samples(), 1);
    assert_eq!(plain.validate_literal_resources(), Ok(()));
    let mut multisampled = fragment_case([2, 1], 4);
    multisampled.raster_sample_count = Some(4);
    multisampled.case_id = multisampled.computed_case_id().unwrap();
    assert_eq!(multisampled.raster_samples(), 4);
    assert_eq!(multisampled.validate_literal_resources(), Ok(()));
    let text = serde_json::to_string(&multisampled).unwrap();
    assert!(text.contains("\"raster_sample_count\":4"), "{text}");
    assert_eq!(
        serde_json::from_str::<AuthoredCase>(&text).unwrap(),
        multisampled
    );
    // The same bytes under another count are another case.
    let mut two = multisampled.clone();
    two.raster_sample_count = Some(2);
    assert_ne!(two.computed_case_id().unwrap(), multisampled.case_id);
    assert_ne!(
        two.computed_input_sha256().unwrap(),
        multisampled.computed_input_sha256().unwrap()
    );
}

/// One sample has one spelling, and only the counts the Metal reference device rasterizes are
/// authorable: 8 has no Metal answer on an Apple M3 Max.
#[test]
fn a_sample_count_is_two_or_four() {
    for (count, accepted) in [(1, false), (2, true), (3, false), (4, true), (8, false)] {
        let mut case = fragment_case([2, 1], count);
        case.raster_sample_count = Some(count);
        let errors = errors(&case);
        assert_eq!(errors.is_empty(), accepted, "{count}: {errors:?}");
        if count == 1 {
            assert!(errors.iter().any(|e| e.contains("omitting")), "{errors:?}");
        }
        if count == 8 {
            assert!(
                errors
                    .iter()
                    .any(|e| e.contains("no Metal answer exists for another count")),
                "{errors:?}"
            );
        }
    }
    assert_eq!(RASTER_SAMPLE_COUNTS, [2, 4]);
}

/// Every attachment holds the pass's samples: bytes for one sample per pixel are the wrong length
/// under a count of four, for a color target and for the depth aspect alike.
#[test]
fn attachment_bytes_hold_every_sample() {
    let mut case = fragment_case([2, 1], 1);
    case.raster_sample_count = Some(4);
    let errors = errors(&case);
    assert!(
        errors
            .iter()
            .any(|e| e.contains("render target 0 initial_bytes_b64")),
        "{errors:?}"
    );
    assert!(
        errors.iter().any(|e| e.contains("initial_depth_b64")),
        "{errors:?}"
    );
}

/// A multisampled pass is a fragment case's draw, and the families no executor fills or runs
/// sample by sample are refused: a stencil aspect whose samples differ (one of a single value is
/// cleared to it), a fragment imageblock, and a rate map.
#[test]
fn a_multisampled_pass_refuses_what_no_executor_runs_per_sample() {
    let base = || {
        let mut case = fragment_case([2, 1], 2);
        case.raster_sample_count = Some(2);
        case
    };
    assert_eq!(errors(&base()), Vec::<String>::new());
    let mut stencil = base();
    stencil.depth_stencil.as_mut().unwrap().initial_stencil_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode([0x5au8; 4]));
    assert_eq!(errors(&stencil), Vec::<String>::new());
    stencil.depth_stencil.as_mut().unwrap().initial_stencil_b64 =
        Some(base64::engine::general_purpose::STANDARD.encode([0x5a, 0x5a, 0x5b, 0x5a]));
    assert!(
        errors(&stencil)
            .iter()
            .any(|e| e.contains("stencil aspect whose samples differ")),
        "{:?}",
        errors(&stencil)
    );
    let mut mapped = base();
    mapped.render_pass_rate_map = Some(RateMapLiteral {
        screen_size: [2, 1],
        layers: vec![RateMapLayerLiteral {
            horizontal: vec![RateSample(1.0)],
            vertical: vec![RateSample(1.0)],
        }],
    });
    assert!(
        errors(&mapped)
            .iter()
            .any(|e| e.contains("rasterization-rate map")),
        "{:?}",
        errors(&mapped)
    );
    let mut imageblock = base();
    imageblock.fragment_imageblock = Some(FragmentImageblockResource {
        dimensions: [2, 1],
        members: vec![],
    });
    assert!(
        errors(&imageblock)
            .iter()
            .any(|e| e.contains("fragment imageblock: a multisampled")),
        "{:?}",
        errors(&imageblock)
    );
    // A kernel's dispatch has no raster.
    let mut kernel = example();
    kernel.raster_sample_count = Some(2);
    assert!(
        errors(&kernel)
            .iter()
            .any(|e| e.contains("only for a fragment case's draw")),
        "{:?}",
        errors(&kernel)
    );
}

/// A multisampled attachment's literal holds each pixel's samples in order, and a region selects
/// every sample of each of its pixels, never a resolved value.
#[test]
fn a_selection_reads_every_sample_of_its_pixels() {
    let mut case = fragment_case([2, 2], 2);
    case.raster_sample_count = Some(2);
    case.output = OutputSelection::RenderTarget {
        index: 0,
        origin: [1, 0],
        dimensions: [1, 2],
    };
    let resources = LiteralResources::prepare(&case).unwrap();
    let target = &resources.render_targets[0];
    assert_eq!(target.samples, 2);
    assert_eq!(target.bytes, [0, 1, 16, 17, 32, 33, 48, 49]);
    // Pixels (1, 0) and (1, 1), both samples of each.
    assert_eq!(target.select([1, 0], [1, 2]).unwrap(), [16, 17, 48, 49]);
    let depth = resources.depth_stencil.as_ref().unwrap();
    assert_eq!(depth.samples, 2);
    assert_eq!(
        depth.select(true, [0, 0], [2, 2]).unwrap().len(),
        2 * 2 * 2 * 4
    );
    assert!(depth.select(false, [0, 0], [1, 1]).is_err());
}

/// A multisampled pass is no executor's gap: both executors render it. A case of one sample moves
/// its attachments by copies, as every case did before the field.
#[test]
fn a_multisampled_pass_stops_no_executor() {
    let mut case = fragment_case([2, 1], 2);
    assert!(case_requirements(&case).is_empty());
    assert_eq!(attachment_transfer(&case), AttachmentTransfer::Copy);
    case.raster_sample_count = Some(2);
    assert!(case_requirements(&case).is_empty());
    assert_eq!(
        attachment_transfer(&case),
        AttachmentTransfer::PerSample { samples: 2 }
    );
}
