use super::linear_images::{pitch_candidates, FORMATS, PITCH_ALIGNMENTS, WIDTHS};
use super::*;

#[test]
fn align_up_rounds_to_the_next_multiple() {
    assert_eq!(align_up(0, 16), 0);
    assert_eq!(align_up(1, 16), 16);
    assert_eq!(align_up(16, 16), 16);
    assert_eq!(align_up(17, 16), 32);
    assert_eq!(align_up(17, 1), 17);
    assert_eq!(align_up(17, 0), 17);
    assert_eq!(align_up(100, 48), 144);
}

#[test]
fn flag_names_name_known_bits_in_table_order_then_the_rest_in_hex() {
    let table = [(0x4, "C"), (0x1, "A"), (0x2, "B")];
    assert_eq!(flag_names(0, &table), Vec::<String>::new());
    assert_eq!(flag_names(0x7, &table), ["C", "A", "B"]);
    assert_eq!(flag_names(0x8 | 0x1 | 0x100, &table), ["A", "0x8", "0x100"]);
}

#[test]
fn pitch_candidates_are_the_packed_row_and_its_round_ups_without_repeats() {
    assert_eq!(pitch_candidates(17, 1), [17, 20, 32, 64, 128, 256]);
    assert_eq!(pitch_candidates(3, 2), [6, 8, 16, 64, 128, 256]);
    assert_eq!(pitch_candidates(64, 16), [1024]);
    assert_eq!(pitch_candidates(100, 4), [400, 448, 512]);
    for format in FORMATS {
        for &width in WIDTHS {
            let packed = u64::from(width) * u64::from(format.bytes);
            let pitches = pitch_candidates(width, format.bytes);
            assert_eq!(pitches[0], packed);
            assert!(pitches.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(pitches.len() <= PITCH_ALIGNMENTS.len());
            // Every pitch holds whole texels, as Metal's bytesPerRow must.
            assert!(pitches
                .iter()
                .all(|pitch| pitch % u64::from(format.bytes) == 0));
        }
    }
}

/// Each probe format is the Metal pixel format it names, mapped the way the product maps it
/// (`metal::format::direct_format`), with the registry's name and texel size.
#[test]
fn probe_formats_are_the_products_direct_mappings() {
    use metal2vulkan::metal::format::direct_format;
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::vk as mvk;
    let mut seen = std::collections::BTreeSet::new();
    for format in FORMATS {
        assert!(seen.insert(format.metal), "{} twice", format.metal);
        let pixel_format = PixelFormat::from_name(format.metal)
            .unwrap_or_else(|| panic!("{} is no Metal pixel format", format.metal));
        let vulkan = mvk::Format::from_raw(format.vulkan.as_raw());
        assert_eq!(
            direct_format(pixel_format),
            Some(vulkan),
            "{}",
            format.metal
        );
        assert_eq!(vulkan.name(), Some(format.name));
        let block = vulkan.block().unwrap();
        assert_eq!(
            (block.size, block.texels),
            (format.bytes, 1),
            "{}",
            format.name
        );
    }
}

#[test]
fn the_first_memory_type_holding_the_wanted_flags_among_the_bits_is_chosen() {
    use vk::MemoryPropertyFlags as M;
    let coherent = M::HOST_VISIBLE | M::HOST_COHERENT;
    let types = [
        M::DEVICE_LOCAL,
        M::HOST_VISIBLE,
        coherent | M::HOST_CACHED,
        coherent,
    ];
    assert_eq!(memory_type_with(&types, 0b1111, coherent), Some(2));
    assert_eq!(memory_type_with(&types, 0b1011, coherent), Some(3));
    assert_eq!(memory_type_with(&types, 0b0011, coherent), None);
    assert_eq!(memory_type_with(&types, 0, M::empty()), None);
    assert_eq!(memory_type_with(&types, 0b0010, M::empty()), Some(1));
}

/// The alias round trips compare two patterns: they must differ almost everywhere, or a copy
/// that moved nothing (or the wrong rows) could pass.
#[test]
fn the_two_alias_patterns_differ_and_vary() {
    let packed = 17 * 16;
    let mut same = 0;
    let mut values = std::collections::BTreeSet::new();
    for row in 0..4 {
        for column in 0..packed {
            let first = pattern_byte(1, row, column);
            same += u32::from(first == pattern_byte(2, row, column));
            values.insert(first);
        }
        // Rows differ from one another too.
        let row_bytes = |row| (0..packed).map(move |c| pattern_byte(1, row, c));
        assert!(row_bytes(row).ne(row_bytes(row + 1)));
    }
    assert!(same * 50 < 4 * packed as u32, "{same} equal bytes");
    assert!(values.len() > 200, "{} distinct bytes", values.len());
}

#[test]
fn the_dispatch_words_are_distinct_and_their_results_are_three_x_plus_one() {
    let mut inputs = std::collections::BTreeSet::new();
    for index in 0..1024 {
        let (input, output) = dispatch_word(index);
        assert!(inputs.insert(input));
        assert_eq!(output, input.wrapping_mul(3).wrapping_add(1));
        assert_ne!(input, output);
    }
}

/// On the test device the probe runs to the end without a validation message, and what it measured
/// holds together: where the context imported host memory, both dispatches wrote every word into
/// the host's own allocation; every linear row holds its packed texels; every alias round trip that
/// ran matched both ways; and every explicit layout the driver accepted has the pitch asked for.
#[cfg(any(m2v_gpu, m2v_cpu_icd))]
#[test]
fn the_runtime_probe_measures_the_test_device_consistently() {
    let report = probe_runtime(crate::gpu::TEST_DEVICE).unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&report).unwrap());
    assert_eq!(report["validation_messages"], 0, "{report:#}");
    let host = &report["host_memory"];
    assert_eq!(host["error"], serde_json::Value::Null);
    if host["enabled"] == true {
        let import = &host["import"];
        let words = &import["descriptor_dispatch"]["words"];
        assert_eq!(import["descriptor_dispatch"]["matched"], *words);
        if import["device_address"] == true {
            assert_eq!(import["address_dispatch"]["matched"], *words);
        }
    }
    let textures = &report["linear_textures"];
    assert_eq!(textures["error"], serde_json::Value::Null);
    for format in textures["formats"].as_array().unwrap() {
        assert_eq!(format["error"], serde_json::Value::Null, "{format:#}");
        for pitch in format["pitches"].as_array().unwrap() {
            assert!(pitch["row_pitch"].as_u64() >= pitch["packed"].as_u64());
        }
        let aliases = [&format["alias"], &format["drm_linear"]["alias"]];
        for alias in aliases.into_iter().filter(|alias| !alias.is_null()) {
            assert_eq!(alias["error"], serde_json::Value::Null, "{alias:#}");
            for check in [
                "image_reads_buffer",
                "buffer_reads_image",
                "host_reads_image",
            ] {
                assert_eq!(alias[check]["mismatched"], 0, "{alias:#}");
                assert_ne!(alias[check]["bytes"], 0);
            }
        }
        if let Some(explicit) = format["drm_linear"]["explicit"].as_array() {
            for attempt in explicit.iter().filter(|a| a["outcome"] == "accepted") {
                assert_eq!(attempt["row_pitch"], attempt["requested"], "{attempt:#}");
            }
        }
    }
}
