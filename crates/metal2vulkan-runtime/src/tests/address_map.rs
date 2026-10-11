//! `address_map.rs`: the address-mode map and how a command's runtime samplers take it, without a
//! device.

use super::super::address_map::{resolve, AddressModeMap, MappedSampling, Pair, SamplerMaps};
use super::super::error::RuntimeGap;
use metal2vulkan::bind::{BindState, BoundSampler, SamplerDescriptor, StageTable};
use metal2vulkan::metal::SamplerAddressMode as Mode;
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage};

/// The consumer's first measured rule (mirror clamp, repeat and mirrored repeat sample as clamp
/// to edge; the border modes kept).
fn edges() -> AddressModeMap {
    AddressModeMap::IDENTITY
        .with(Mode::MirrorClampToEdge, Mode::ClampToEdge)
        .with(Mode::Repeat, Mode::ClampToEdge)
        .with(Mode::MirrorRepeat, Mode::ClampToEdge)
}

/// The consumer's second (every mode but clamp to edge sampled as it).
fn all_edges() -> AddressModeMap {
    edges()
        .with(Mode::ClampToZero, Mode::ClampToEdge)
        .with(Mode::ClampToBorderColor, Mode::ClampToEdge)
}

fn slot(stage: Stage, class: SlotClass, index: u32) -> SlotRef {
    SlotRef {
        stage,
        class,
        index,
    }
}

fn texture(index: u32) -> SlotRef {
    slot(Stage::Fragment, SlotClass::Texture, index)
}

fn sampler(index: u32) -> SlotRef {
    slot(Stage::Fragment, SlotClass::Sampler, index)
}

fn addressing(mode: Mode) -> SamplerDescriptor {
    let mut desc = SamplerDescriptor::new();
    desc.s_address_mode = mode;
    desc.t_address_mode = mode;
    desc.r_address_mode = mode;
    desc
}

fn modes(desc: &SamplerDescriptor) -> [Mode; 3] {
    [
        desc.s_address_mode,
        desc.t_address_mode,
        desc.r_address_mode,
    ]
}

/// The maps `pairs` take with each texture slot's map from `maps` (identity otherwise) and
/// every sampler slot bound with `bound`, no handles sampled.
fn maps_of(
    pairs: &[Pair],
    maps: &[(SlotRef, AddressModeMap)],
    bound: SamplerDescriptor,
) -> Result<SamplerMaps, RuntimeGap> {
    let map_of = |slot: SlotRef| {
        maps.iter()
            .find(|(s, _)| *s == slot)
            .map_or(AddressModeMap::IDENTITY, |(_, map)| *map)
    };
    resolve(pairs, false, map_of, |_| Some(bound), [])
}

#[test]
fn a_map_reads_each_mode_by_its_own() {
    let map = edges();
    assert!(AddressModeMap::IDENTITY.is_identity());
    assert_eq!(AddressModeMap::default(), AddressModeMap::IDENTITY);
    assert!(!map.is_identity());
    assert_eq!(map.mode(Mode::Repeat), Mode::ClampToEdge);
    assert_eq!(map.mode(Mode::MirrorRepeat), Mode::ClampToEdge);
    assert_eq!(map.mode(Mode::MirrorClampToEdge), Mode::ClampToEdge);
    assert_eq!(map.mode(Mode::ClampToEdge), Mode::ClampToEdge);
    assert_eq!(map.mode(Mode::ClampToZero), Mode::ClampToZero);
    assert_eq!(map.mode(Mode::ClampToBorderColor), Mode::ClampToBorderColor);
    for mode in Mode::ALL {
        assert_eq!(all_edges().mode(*mode), Mode::ClampToEdge, "{mode:?}");
        assert_eq!(AddressModeMap::IDENTITY.mode(*mode), *mode, "{mode:?}");
    }
    // A map back to the identity is the identity.
    assert!(map
        .with(Mode::Repeat, Mode::Repeat)
        .with(Mode::MirrorRepeat, Mode::MirrorRepeat)
        .with(Mode::MirrorClampToEdge, Mode::MirrorClampToEdge)
        .is_identity());
}

/// The check that a command binding no mapped texture changes nothing: its maps are empty, and
/// empty maps leave a bind state, and so its shape and the variant it plans, as they were.
#[test]
fn without_a_mapped_texture_nothing_changes() {
    let pairs = [
        Pair::Slot {
            texture: texture(0),
            sampler: sampler(0),
        },
        Pair::Constexpr {
            texture: texture(1),
            modes: Some([Mode::Repeat; 3]),
        },
    ];
    for mode in Mode::ALL {
        let maps = maps_of(&pairs, &[], addressing(*mode)).unwrap();
        assert_eq!(maps, SamplerMaps::default(), "{mode:?}");
    }
    let mut state = BindState::default();
    let mut table = StageTable::default();
    table
        .samplers
        .insert(0, BoundSampler::new(addressing(Mode::Repeat)));
    state.stages.insert(Stage::Fragment, table);
    let before = state.clone();
    SamplerMaps::default().apply(&mut state);
    assert_eq!(state, before);
    assert_eq!(state.shape(), before.shape());
}

/// A runtime sampler slot sampling a mapped texture is bound with its modes mapped, in the bind
/// state (its specialization's word) and in its descriptor write alike; a slot the map leaves as
/// it is takes no map.
#[test]
fn a_runtime_sampler_takes_its_textures_map() {
    let pairs = [Pair::Slot {
        texture: texture(0),
        sampler: sampler(0),
    }];
    let maps = maps_of(&pairs, &[(texture(0), edges())], addressing(Mode::Repeat)).unwrap();
    let mapped = maps.sampler(sampler(0), addressing(Mode::Repeat));
    assert_eq!(modes(&mapped), [Mode::ClampToEdge; 3]);
    // Another slot, or the same index in another stage, is not mapped.
    let other = slot(Stage::Vertex, SlotClass::Sampler, 0);
    assert_eq!(
        modes(&maps.sampler(other, addressing(Mode::Repeat))),
        [Mode::Repeat; 3]
    );
    let mut state = BindState::default();
    let mut table = StageTable::default();
    table
        .samplers
        .insert(0, BoundSampler::new(addressing(Mode::Repeat)));
    table
        .samplers
        .insert(1, BoundSampler::new(addressing(Mode::Repeat)));
    state.stages.insert(Stage::Fragment, table);
    maps.apply(&mut state);
    let samplers = &state.stages[&Stage::Fragment].samplers;
    assert_eq!(modes(&samplers[&0].desc), [Mode::ClampToEdge; 3]);
    assert_eq!(modes(&samplers[&1].desc), [Mode::Repeat; 3]);
    // The mapped state is exactly a sampler bound with the mapped modes.
    let mut direct = BindState::default();
    let mut table = StageTable::default();
    table
        .samplers
        .insert(0, BoundSampler::new(addressing(Mode::ClampToEdge)));
    table
        .samplers
        .insert(1, BoundSampler::new(addressing(Mode::Repeat)));
    direct.stages.insert(Stage::Fragment, table);
    assert_eq!(state.shape(), direct.shape());
    // Mixed modes map axis by axis.
    let mut mixed = addressing(Mode::ClampToBorderColor);
    mixed.s_address_mode = Mode::MirrorRepeat;
    let maps = maps_of(&pairs, &[(texture(0), edges())], mixed).unwrap();
    assert_eq!(
        modes(&maps.sampler(sampler(0), mixed)),
        [
            Mode::ClampToEdge,
            Mode::ClampToBorderColor,
            Mode::ClampToBorderColor
        ]
    );
    // A map that leaves the bound modes as they are maps nothing.
    let maps = maps_of(
        &pairs,
        &[(texture(0), edges())],
        addressing(Mode::ClampToZero),
    )
    .unwrap();
    assert_eq!(maps, SamplerMaps::default());
    // An unbound slot binds the neutral sampler, which no map changes.
    let none = resolve(&pairs, false, |_| edges(), |_| None, []).unwrap();
    assert_eq!(none, SamplerMaps::default());
}

/// One sampler slot sampling two textures in one stage takes one map: refused where their maps
/// give it different modes, bound where they agree on its modes.
#[test]
fn one_slot_takes_one_map() {
    let pairs = [
        Pair::Slot {
            texture: texture(0),
            sampler: sampler(0),
        },
        Pair::Slot {
            texture: texture(1),
            sampler: sampler(0),
        },
    ];
    let shared = |map: AddressModeMap| RuntimeGap::MappedTextureSampling {
        map,
        through: MappedSampling::SharedSamplerSlot,
    };
    // A mapped and an unmapped texture, either first.
    for maps in [vec![(texture(0), edges())], vec![(texture(1), edges())]] {
        assert_eq!(
            maps_of(&pairs, &maps, addressing(Mode::Repeat)),
            Err(shared(edges())),
        );
    }
    // Two maps that differ on the bound mode.
    let maps = [(texture(0), edges()), (texture(1), all_edges())];
    assert_eq!(
        maps_of(&pairs, &maps, addressing(Mode::ClampToZero)),
        Err(shared(all_edges())),
    );
    // Two maps that agree on it.
    let maps = maps_of(&pairs, &maps, addressing(Mode::Repeat)).unwrap();
    assert_eq!(
        modes(&maps.sampler(sampler(0), addressing(Mode::Repeat))),
        [Mode::ClampToEdge; 3]
    );
    // A map that leaves the bound modes as they are beside an unmapped texture.
    let maps = [(texture(0), edges())];
    assert_eq!(
        maps_of(&pairs, &maps, addressing(Mode::ClampToEdge)),
        Ok(SamplerMaps::default())
    );
    // The same slot index in two stages is two slots.
    let staged = [
        Pair::Slot {
            texture: texture(0),
            sampler: sampler(0),
        },
        Pair::Slot {
            texture: slot(Stage::Vertex, SlotClass::Texture, 1),
            sampler: slot(Stage::Vertex, SlotClass::Sampler, 0),
        },
    ];
    let maps = maps_of(&staged, &[(texture(0), edges())], addressing(Mode::Repeat)).unwrap();
    assert_eq!(
        modes(&maps.sampler(sampler(0), addressing(Mode::Repeat))),
        [Mode::ClampToEdge; 3]
    );
    let vertex = slot(Stage::Vertex, SlotClass::Sampler, 0);
    assert_eq!(
        modes(&maps.sampler(vertex, addressing(Mode::Repeat))),
        [Mode::Repeat; 3]
    );
}

/// A constexpr sampler whose modes a mapped texture's map sends where a constexpr sampler
/// cannot address, and a mapped texture a function sampling handles can reach, are refused,
/// naming the map; an unmapped texture is not.
#[test]
fn what_cannot_follow_the_map_is_refused() {
    let pairs = [Pair::Constexpr {
        texture: texture(0),
        modes: Some([Mode::Repeat; 3]),
    }];
    let mirror_clamps = AddressModeMap::IDENTITY.with(Mode::Repeat, Mode::MirrorClampToEdge);
    assert_eq!(
        maps_of(
            &pairs,
            &[(texture(0), mirror_clamps)],
            addressing(Mode::Repeat)
        ),
        Err(RuntimeGap::MappedTextureSampling {
            map: mirror_clamps,
            through: MappedSampling::ConstexprSampler,
        })
    );
    assert_eq!(
        maps_of(
            &pairs,
            &[(texture(1), mirror_clamps)],
            addressing(Mode::Repeat)
        ),
        Ok(SamplerMaps::default())
    );
    let handles = |reached: Vec<AddressModeMap>| {
        resolve(&[], true, |_| AddressModeMap::IDENTITY, |_| None, reached)
    };
    assert_eq!(
        handles(vec![AddressModeMap::IDENTITY, all_edges()]),
        Err(RuntimeGap::MappedTextureSampling {
            map: all_edges(),
            through: MappedSampling::Handles,
        })
    );
    assert_eq!(
        handles(vec![AddressModeMap::IDENTITY]),
        Ok(SamplerMaps::default())
    );
    // A function that samples no handles reaches no texture through them.
    assert_eq!(
        resolve(
            &[],
            false,
            |_| AddressModeMap::IDENTITY,
            |_| None,
            [all_edges()]
        ),
        Ok(SamplerMaps::default())
    );
}

/// A constexpr sampler sampling a mapped texture is not refused where the map sends each of its
/// modes where a constexpr sampler can address (clamp to edge, repeat, mirrored repeat, or the
/// mode itself): its slot's pipeline variant samples with the modes mapped
/// (`VariantKey::address_maps`), and no runtime slot takes a map for it. A mode sent to
/// mirror-clamp-to-edge, or to a clamp to zero or the border from another mode, on any axis, or
/// modes the layout does not state, refuse it, naming the map.
#[test]
fn a_constexpr_sampler_takes_the_map_its_variant_can_state() {
    let constexpr = |modes: Option<[Mode; 3]>| {
        [Pair::Constexpr {
            texture: texture(0),
            modes,
        }]
    };
    let refused = |map| {
        Err(RuntimeGap::MappedTextureSampling {
            map,
            through: MappedSampling::ConstexprSampler,
        })
    };
    let bound = addressing(Mode::Repeat);
    for map in [edges(), all_edges()] {
        for modes in [
            [Mode::ClampToEdge; 3],
            [Mode::Repeat, Mode::ClampToEdge, Mode::ClampToEdge],
            [Mode::ClampToEdge, Mode::MirrorRepeat, Mode::ClampToEdge],
            [
                Mode::ClampToZero,
                Mode::ClampToEdge,
                Mode::ClampToBorderColor,
            ],
        ] {
            assert_eq!(
                maps_of(&constexpr(Some(modes)), &[(texture(0), map)], bound),
                Ok(SamplerMaps::default()),
                "{modes:?}"
            );
        }
        assert_eq!(
            maps_of(&constexpr(None), &[(texture(0), map)], bound),
            refused(map)
        );
    }
    for to in [
        Mode::MirrorClampToEdge,
        Mode::ClampToZero,
        Mode::ClampToBorderColor,
    ] {
        let map = AddressModeMap::IDENTITY.with(Mode::MirrorRepeat, to);
        let pairs = constexpr(Some([
            Mode::ClampToEdge,
            Mode::ClampToEdge,
            Mode::MirrorRepeat,
        ]));
        assert_eq!(
            maps_of(&pairs, &[(texture(0), map)], bound),
            refused(map),
            "{to:?}"
        );
    }
    // Modes the layout does not state, over an unmapped texture.
    assert_eq!(
        maps_of(&constexpr(None), &[], bound),
        Ok(SamplerMaps::default())
    );
}
