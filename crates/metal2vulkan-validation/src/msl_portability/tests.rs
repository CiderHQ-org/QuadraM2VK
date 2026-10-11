//! MoltenVK's configuration of hand-built layouts: the MSL options, resource bindings and
//! discrete sets the helper converts a module with, and how the Metal compile follows what the
//! helper reports.

use super::moltenvk::{self, Binding, Entry, Layout, Set0, SetLayout, Stage};
use metal2vulkan::vk::{DescriptorType, ShaderStageFlags};

fn binding(binding: u32, ty: DescriptorType, count: u32, stages: ShaderStageFlags) -> Binding {
    Binding {
        binding,
        ty,
        count,
        stages,
    }
}

/// A kernel's layout shaped like the product's handle domain: two buffers, a texture, and the
/// handle table in set 0, and one sampled heap array of `heap` elements in set 1.
fn heap_layout(heap: u32, push_descriptor_eligible: bool) -> Layout {
    let c = ShaderStageFlags::COMPUTE;
    Layout {
        sets: vec![
            SetLayout {
                set: 0,
                bindings: vec![
                    binding(0, DescriptorType::STORAGE_BUFFER, 1, c),
                    binding(1, DescriptorType::STORAGE_BUFFER, 1, c),
                    binding(2, DescriptorType::SAMPLED_IMAGE, 1, c),
                    binding(3, DescriptorType::STORAGE_BUFFER, 1, c),
                ],
            },
            SetLayout {
                set: 1,
                bindings: vec![binding(0, DescriptorType::SAMPLED_IMAGE, heap, c)],
            },
        ],
        push_constant_stages: c,
        push_descriptor_eligible,
    }
}

fn kernel() -> Entry {
    Entry {
        name: "k".to_owned(),
        stage: Stage::Compute,
    }
}

/// `(basetype, set, binding, count, msl_buffer, msl_texture, msl_sampler)` of each binding.
fn rows(conversion: &moltenvk::Conversion) -> Vec<String> {
    conversion
        .bindings
        .iter()
        .map(|b| {
            format!(
                "{:?} {} {} {} {} {} {}",
                b.basetype, b.set, b.binding, b.count, b.msl_buffer, b.msl_texture, b.msl_sampler
            )
        })
        .collect()
}

#[test]
fn pooled_sets_are_argument_buffers_with_every_binding_count_and_padded_positions() {
    let conversions = moltenvk::conversions(&kernel(), &heap_layout(65536, false)).expect("config");
    let [pooled] = conversions.as_slice() else {
        panic!("{conversions:?}");
    };
    assert_eq!(pooled.set0, Set0::Pooled);
    assert!(pooled.discrete_sets.is_empty());
    assert!(pooled.dynamic_buffers.is_empty());
    assert_eq!(
        rows(pooled),
        [
            // Push constants after the eight argument-buffer indices.
            "Void 4294967295 0 1 8 0 0",
            // Set 0: the buffer-size buffer first, then each binding at its argument-buffer
            // position (a texture's buffer and sampler positions follow its textures).
            "Void 0 4294967293 1 0 0 0",
            "Void 0 0 1 1 1 1",
            "Void 0 1 1 2 2 2",
            "Image 0 2 1 4 3 4",
            "Void 0 3 1 4 4 4",
            // Set 1: the heap array with the layout's count (no buffer, so no size buffer).
            "Image 1 0 65536 65536 0 65536",
        ]
    );
    let text = pooled.config_text();
    assert!(text.starts_with("entry 5 k\ncommon flip_vert_y 1\ncommon fixup_clipspace 0\n"));
    assert!(
        text.contains("\nbinding 5 image 1 0 65536 65536 0 65536\n"),
        "{text}"
    );
    for option in [
        "option argument_buffers 1",
        "option pad_argument_buffer_resources 1",
        "option argument_buffers_tier 1",
        "option fixed_subgroup_size 32",
        "option indirect_params_buffer_index 28",
        "option buffer_size_buffer_index 29",
        "option dynamic_offsets_buffer_index 30",
    ] {
        assert!(text.contains(&format!("{option}\n")), "{option}: {text}");
    }
}

#[test]
fn a_push_descriptor_set_0_is_discrete_after_the_reserved_buffer_indices() {
    let conversions = moltenvk::conversions(&kernel(), &heap_layout(65536, true)).expect("config");
    let [pooled, push] = conversions.as_slice() else {
        panic!("{conversions:?}");
    };
    assert_eq!((pooled.set0, push.set0), (Set0::Pooled, Set0::Push));
    assert_eq!(push.discrete_sets, [0]);
    assert_eq!(
        rows(push),
        [
            "Void 4294967295 0 1 8 0 0",
            // Metal binding indices after the argument buffers (set 1 still has one) and the
            // push constants, counted per resource kind, only for used bindings.
            "Void 0 0 1 9 0 0",
            "Void 0 1 1 10 0 0",
            "Image 0 2 1 11 0 0",
            "Void 0 3 1 11 1 0",
            "Image 1 0 65536 65536 0 65536",
        ]
    );
    // Without the heap set no argument buffer reserves indices, and with no push constants the
    // discrete set starts at 0.
    let mut alone = heap_layout(0, true);
    alone.sets.truncate(1);
    alone.push_constant_stages = ShaderStageFlags::empty();
    let conversions = moltenvk::conversions(&kernel(), &alone).expect("config");
    assert_eq!(
        rows(&conversions[1]),
        [
            "Void 0 0 1 0 0 0",
            "Void 0 1 1 1 0 0",
            "Image 0 2 1 2 0 0",
            "Void 0 3 1 2 1 0"
        ]
    );
}

#[test]
fn a_push_variant_that_changes_nothing_is_not_checked_twice() {
    // No binding in set 0: MoltenVK binds it discretely whether pooled or pushed.
    let layout = Layout {
        sets: vec![SetLayout {
            set: 0,
            bindings: vec![],
        }],
        push_constant_stages: ShaderStageFlags::COMPUTE,
        push_descriptor_eligible: true,
    };
    let conversions = moltenvk::conversions(&kernel(), &layout).expect("config");
    assert_eq!(conversions.len(), 1);
    assert_eq!(conversions[0].discrete_sets, [0]);
    assert_eq!(rows(&conversions[0]), ["Void 4294967295 0 1 0 0 0"]);
}

#[test]
fn samplers_texel_buffers_and_dynamic_buffers_follow_moltenvk_tables() {
    let f = ShaderStageFlags::FRAGMENT;
    let layout = Layout {
        sets: vec![SetLayout {
            set: 0,
            bindings: vec![
                binding(0, DescriptorType::COMBINED_IMAGE_SAMPLER, 2, f),
                binding(1, DescriptorType::SAMPLER, 1, f),
                binding(2, DescriptorType::STORAGE_TEXEL_BUFFER, 1, f),
                binding(3, DescriptorType::UNIFORM_BUFFER_DYNAMIC, 2, f),
                binding(4, DescriptorType::INPUT_ATTACHMENT, 1, f),
                // Another stage's binding: an argument buffer lays it out all the same.
                binding(
                    5,
                    DescriptorType::STORAGE_IMAGE,
                    1,
                    ShaderStageFlags::VERTEX,
                ),
            ],
        }],
        push_constant_stages: ShaderStageFlags::empty(),
        push_descriptor_eligible: true,
    };
    let entry = Entry {
        name: "f".to_owned(),
        stage: Stage::Fragment,
    };
    let conversions = moltenvk::conversions(&entry, &layout).expect("config");
    let [pooled, push] = conversions.as_slice() else {
        panic!("{conversions:?}");
    };
    assert_eq!(
        rows(pooled),
        [
            "Void 0 4294967293 1 0 0 0",
            "SampledImage 0 0 2 3 1 3",
            "Sampler 0 1 1 5 5 5",
            "Image 0 2 1 7 6 7",
            "Void 0 3 2 7 7 7",
            "Image 0 4 1 10 9 10",
            "Image 0 5 1 11 10 11",
        ]
    );
    assert_eq!(pooled.dynamic_buffers, [(0, 3, 0)]);
    assert_eq!(
        rows(push),
        [
            "SampledImage 0 0 2 0 0 0",
            "Sampler 0 1 1 0 2 2",
            "Image 0 2 1 0 2 3",
            "Void 0 3 2 0 3 3",
            "Image 0 4 1 2 3 3",
        ]
    );
    assert_eq!(push.dynamic_buffers, [(0, 3, 0)]);
    let text = push.config_text();
    for option in [
        "option fixed_subgroup_size 32",
        "option check_discarded_frag_stores 1",
        "option agx_manual_cube_grad_fixup 1",
        "option view_mask_buffer_index 28",
    ] {
        assert!(text.contains(&format!("{option}\n")), "{option}: {text}");
    }
    assert!(
        text.contains("\ndiscrete 0\n") && text.contains("\ndynamic 4 0 3 0\n"),
        "{text}"
    );
    // A vertex stage leaves the subgroup size variable, as MoltenVK does.
    let vertex = Entry {
        name: "v".to_owned(),
        stage: Stage::Vertex,
    };
    let text = moltenvk::conversions(&vertex, &layout).expect("config")[0].config_text();
    assert!(!text.contains("fixed_subgroup_size"), "{text}");
    assert!(
        text.contains("option shader_output_buffer_index 27\n"),
        "{text}"
    );
}

#[test]
fn layouts_the_tables_were_not_read_for_are_refused() {
    let c = ShaderStageFlags::COMPUTE;
    let inline = Layout {
        sets: vec![SetLayout {
            set: 0,
            bindings: vec![binding(0, DescriptorType::INLINE_UNIFORM_BLOCK, 16, c)],
        }],
        ..Layout::default()
    };
    assert!(moltenvk::conversions(&kernel(), &inline)
        .expect_err("inline uniform block")
        .contains("descriptor type"));
    let gap = Layout {
        sets: vec![SetLayout {
            set: 1,
            bindings: vec![],
        }],
        ..Layout::default()
    };
    assert!(moltenvk::conversions(&kernel(), &gap)
        .expect_err("set 1 first")
        .contains("layout position 0"));
}

fn module_words(model: u32, name: &str) -> Vec<u32> {
    let mut literal = name.as_bytes().to_vec();
    literal.resize(name.len() / 4 * 4 + 4, 0);
    let literal = literal
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<_>>();
    let mut words = vec![0x0723_0203, 0x0001_0500, 0, 10, 0];
    words.push(((3 + literal.len() as u32) << 16) | 15);
    words.extend([model, 1]);
    words.extend(literal);
    words
}

#[test]
fn the_entry_point_names_the_stage() {
    assert_eq!(
        moltenvk::entry_point(&module_words(5, "kernel_main")).expect("entry"),
        Entry {
            name: "kernel_main".to_owned(),
            stage: Stage::Compute,
        }
    );
    assert_eq!(
        moltenvk::entry_point(&module_words(4, "frag"))
            .expect("entry")
            .stage,
        Stage::Fragment
    );
    assert!(moltenvk::entry_point(&module_words(1, "tesc")).is_err());
    assert!(moltenvk::entry_point(&[0x0723_0203, 0x0001_0500, 0, 1, 0]).is_err());
    let mut two = module_words(5, "a");
    two.extend_from_slice(&module_words(5, "b")[5..]);
    assert!(moltenvk::entry_point(&two).is_err());
}

#[test]
fn the_metal_compile_follows_the_reported_fast_math_flags_and_invariance() {
    let reported = moltenvk::Reported::parse("fp_fast_math_flags=196623\nposition_invariant=0\n")
        .expect("parse");
    assert_eq!(reported.fp_fast_math_flags, 0x3_000f);
    let args = |flags, invariant| {
        moltenvk::Reported {
            fp_fast_math_flags: flags,
            position_invariant: invariant,
        }
        .metal_arguments()
        .join(" ")
    };
    assert_eq!(
        args(0x3_000f, false),
        "-sdk macosx metal -std=metal4.0 -fmetal-math-mode=fast -fmetal-math-fp32-functions=fast -c"
    );
    // NSZ, AllowRecip, AllowReassoc, and AllowContract without NotNaN: relaxed.
    assert!(args(0x3_000e, false)
        .contains("-fmetal-math-mode=relaxed -fmetal-math-fp32-functions=precise"));
    // Without AllowContract: safe.
    assert!(args(0x2_000f, true).contains(
        "-fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -fpreserve-invariance -c"
    ));
    for bad in [
        "",
        "fp_fast_math_flags=1\n",
        "fp_fast_math_flags=x\nposition_invariant=0\n",
        "other=1\n",
    ] {
        assert!(moltenvk::Reported::parse(bad).is_err(), "{bad:?}");
    }
}
