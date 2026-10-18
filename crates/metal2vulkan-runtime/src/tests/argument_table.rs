//! The debug builds' check that a stage table brought in place is the table read whole
//! (`ArgumentTable::check_reading`, `argument_table.rs`), which compares a sampler as it prints
//! without printing it (`same_as_printed`).

use super::super::argument_table::same_as_printed;
use metal2vulkan::bind::{BoundSampler, SamplerDescriptor, StageTable};

/// The field names of a value's `{:?}` print, `Name { a: .., b: .. }`, at its top level.
fn field_names(printed: &str) -> Vec<String> {
    let inner = &printed[printed.find('{').unwrap() + 1..printed.rfind('}').unwrap()];
    let mut names = Vec::new();
    let (mut depth, mut start) = (0, 0);
    for (at, c) in inner.char_indices() {
        match c {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                names.push(inner[start..at].trim().to_owned());
                start = at + 1;
            }
            _ => {}
        }
    }
    names.push(inner[start..].trim().to_owned());
    names
        .iter()
        .filter(|field| !field.is_empty())
        .map(|field| field.split(':').next().unwrap().trim().to_owned())
        .collect()
}

/// Two bound samplers compare alike exactly where they print alike: LOD clamps of opposite zero
/// signs differ, NaN clamps of any sign and payload are alike, and every other field differs
/// where its value does.
#[test]
fn samplers_compare_as_they_print() {
    let base = BoundSampler::new(SamplerDescriptor::new());
    let clamped = |min: f32, max: f32| {
        let mut sampler = base;
        sampler.desc.lod_min_clamp = min;
        sampler.desc.lod_max_clamp = max;
        sampler
    };
    let other = |change: fn(&mut BoundSampler)| {
        let mut sampler = base;
        change(&mut sampler);
        sampler
    };
    let pairs = [
        (base, base),
        (clamped(0.0, 1.0), clamped(-0.0, 1.0)),
        (clamped(0.0, 0.0), clamped(0.0, -0.0)),
        (clamped(f32::NAN, 1.0), clamped(-f32::NAN, 1.0)),
        (
            clamped(0.5, f32::from_bits(0x7fc0_0001)),
            clamped(0.5, f32::NAN),
        ),
        (clamped(f32::NAN, 1.0), clamped(0.0, 1.0)),
        (clamped(0.0, f32::MAX), clamped(0.0, f32::INFINITY)),
        (base, other(|s| s.desc.max_anisotropy = 4)),
        (base, other(|s| s.desc.normalized_coordinates = false)),
        (base, other(|s| s.resource_id = Some(7))),
    ];
    for (a, b) in pairs {
        assert_eq!(
            same_as_printed(&a, &b),
            format!("{a:?}") == format!("{b:?}"),
            "{a:?} against {b:?}"
        );
    }
}

/// The fields the check compares: a field added to a stage table, a bound sampler or a sampler
/// descriptor fails this until `check_reading` and `same_as_printed` compare it as it prints
/// (`at_bind_views` as the state's views leave it, which another encoder may narrow since).
#[test]
fn the_check_compares_every_field() {
    assert_eq!(
        field_names(&format!("{:?}", StageTable::default())),
        [
            "buffers",
            "textures",
            "samplers",
            "threadgroup_lengths",
            "acceleration_structures",
            "at_bind_views"
        ]
    );
    let sampler = BoundSampler::new(SamplerDescriptor::new());
    assert_eq!(
        field_names(&format!("{sampler:?}")),
        ["desc", "resource_id"]
    );
    assert_eq!(
        field_names(&format!("{:?}", sampler.desc)),
        [
            "min_filter",
            "mag_filter",
            "mip_filter",
            "s_address_mode",
            "t_address_mode",
            "r_address_mode",
            "border_color",
            "normalized_coordinates",
            "lod_min_clamp",
            "lod_max_clamp",
            "lod_average",
            "max_anisotropy",
            "compare_function",
            "reduction_mode"
        ]
    );
}
