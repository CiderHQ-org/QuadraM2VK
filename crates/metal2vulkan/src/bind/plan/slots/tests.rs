//! Length classes, the `BufferRangeLimit` variant, and the uses of used resources.

use super::*;
use crate::bind::state::{
    length_class, BindState, BoundBuffer, ResourceClass, StageShape, UsedResource,
};
use crate::device::profile::BufferAccess;
use crate::pipeline::facts::{Mechanism, UniformViewChoice};
use crate::pipeline::interface::{BufferKind, DescriptorView};
use crate::pipeline::sym::Stage;
use crate::spirv::testing::committed_profiles;

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

/// The class bounds the padded length from above and is the smallest such power of two.
#[test]
fn a_length_class_bounds_the_padded_length() {
    for length in [
        0u64,
        1,
        15,
        16,
        17,
        255,
        256,
        1 << 27,
        (1 << 27) + 1,
        u64::MAX,
    ] {
        let c = length_class(length);
        let padded = u128::from(length).next_multiple_of(16);
        assert!(padded < 1u128 << c, "{length}: class {c}");
        if padded > 0 {
            assert!(
                padded >= 1u128 << (c - 1),
                "{length}: class {c} is not the smallest"
            );
        }
    }
    assert_eq!(length_class(0), 0);
    // A class proves a range fits exactly when every padded length below `2^class` does.
    let limit = (1u64 << 27) - 1;
    assert!(class_fits(length_class((1 << 27) - 16), limit));
    assert!(!class_fits(length_class(1 << 27), limit));
    assert!(class_fits(
        length_class(u64::from(u32::MAX) - 15),
        u64::from(u32::MAX)
    ));
}

fn planned(index: u32, bound: BoundKind, form: BufferForm) -> PlannedBuffer {
    PlannedBuffer {
        slot: slot(index),
        window: Window::Whole,
        form,
        access: AccessSet::READ,
        binding: None,
        alignment: 16,
        max_range: 0,
        bound,
        disjoint: false,
    }
}

/// On Profile A a large bound buffer under a `BufferRangeLimit` fact needs the variant, in either
/// form (a fixed point); a small one, an unbound slot, or a slot without the fact does not.
/// Profile F honors the fact at instantiation and never asks for a variant.
#[test]
fn a_large_buffer_needs_the_device_address_variant_on_profile_a_only() {
    let deps = [0, 1, 2, 3].map(|n| RuntimeDependency {
        fact: RuntimeFact::BufferRangeLimit { slot: slot(n) },
        mechanism: Mechanism::Variant,
    });
    let storage = BufferForm::Descriptor(DescriptorView::Storage);
    let mut seen = [false; 2];
    for (stem, p) in committed_profiles().expect("profiles") {
        let max = u64::from(p.fields().limits.max_storage_buffer_range);
        let big = length_class(max.saturating_add(1));
        let small = length_class(max / 4);
        let buffers = [
            planned(
                0,
                BoundKind::Buffer {
                    device_address: true,
                    length_class: big,
                },
                storage,
            ),
            planned(
                1,
                BoundKind::Buffer {
                    device_address: true,
                    length_class: big,
                },
                BufferForm::Psb,
            ),
            planned(
                2,
                BoundKind::Buffer {
                    device_address: true,
                    length_class: small,
                },
                storage,
            ),
            planned(3, BoundKind::Unbound, storage),
            planned(
                4,
                BoundKind::Buffer {
                    device_address: true,
                    length_class: big,
                },
                storage,
            ),
        ];
        let got = psb_variant(&deps, &buffers, &p);
        match p.fields().buffer_access {
            BufferAccess::Addressed => {
                seen[0] = true;
                assert_eq!(got, [slot(0), slot(1)].into(), "{stem}");
            }
            BufferAccess::Floor => {
                seen[1] = true;
                assert!(got.is_empty(), "{stem}");
            }
        }
    }
    assert_eq!(seen, [true, true], "both profile kinds are committed");
}

/// A shape asks for the storage-view variant exactly when it binds, at a slot the pipeline views as
/// a uniform buffer, a buffer whose length class does not prove the range fits
/// `maxUniformBufferRange`; the key then names every such candidate (viewed, spare, or kept)
/// and the slots the pipeline was compiled keeping, so the variant's plan restates it.
#[test]
fn a_large_viewed_slot_asks_for_every_large_candidate_to_stay_storage() {
    let p = crate::pipeline::testing::profile("vk12-floor", Default::default()).expect("profile");
    let max = u64::from(p.fields().limits.max_uniform_buffer_range);
    let big = BoundKind::Buffer {
        device_address: false,
        length_class: length_class(max.saturating_add(1)),
    };
    let fits = BoundKind::Buffer {
        device_address: false,
        length_class: length_class(max - 16),
    };
    let dep = |n, choice| RuntimeDependency {
        fact: RuntimeFact::UniformViewRange {
            slot: slot(n),
            choice,
        },
        mechanism: Mechanism::Variant,
    };
    let uniform = BufferForm::Descriptor(DescriptorView::UniformWords);
    let storage = BufferForm::Descriptor(DescriptorView::Storage);
    use UniformViewChoice::{Kept, Spare, Viewed};
    // Slot 0 viewed, slot 1 spare, slot 2 viewed, slot 3 spare.
    let base = [dep(0, Viewed), dep(1, Spare), dep(2, Viewed), dep(3, Spare)];
    let bound = |b: [BoundKind; 4]| {
        [
            planned(0, b[0], uniform),
            planned(1, b[1], storage),
            planned(2, b[2], uniform),
            planned(3, b[3], storage),
        ]
    };
    // Small everywhere, or large only at spare slots, or at an unbound or inline slot: no variant.
    let none = [BoundKind::Unbound, big, BoundKind::Bytes, big];
    assert!(storage_view_variant(&base, &bound([fits, big, fits, big]), &p).is_empty());
    assert!(storage_view_variant(&base, &bound(none), &p).is_empty());
    // A large viewed slot: every large candidate.
    let key = storage_view_variant(&base, &bound([big, big, fits, fits]), &p);
    assert_eq!(key, [slot(0), slot(1)].into());
    // The variant's own facts for the same shape: slots 0 and 1 kept, 2 still viewed, 3 now
    // viewed; the plan restates the key.
    let variant = [dep(0, Kept), dep(1, Kept), dep(2, Viewed), dep(3, Viewed)];
    let buffers = [
        planned(0, big, storage),
        planned(1, big, storage),
        planned(2, fits, uniform),
        planned(3, fits, uniform),
    ];
    assert_eq!(storage_view_variant(&variant, &buffers, &p), key);
    // A kept slot the shape binds small is still restated.
    let buffers = [
        planned(0, fits, storage),
        planned(1, fits, storage),
        planned(2, fits, uniform),
        planned(3, fits, uniform),
    ];
    assert_eq!(storage_view_variant(&variant, &buffers, &p), key);
}

/// Used resources reach a plan's `uses` exactly when a function can reach them, with the guest's
/// declared usage, and the shape unions a token listed twice.
#[test]
fn used_resources_are_listed_when_the_pipeline_reaches_them() {
    let mut state = BindState::default();
    let mut buffer = UsedResource::new(ResourceClass::Buffer, 1);
    buffer.usage = metal::ResourceUsage::READ;
    let mut again = buffer;
    again.usage = metal::ResourceUsage::WRITE;
    again.stages = metal::RenderStages::FRAGMENT;
    let mut heap = UsedResource::new(ResourceClass::Heap, 2);
    heap.stages = metal::RenderStages::VERTEX;
    let mut texture = UsedResource::new(
        ResourceClass::Texture(crate::bind::state::TextureFacts::new(
            metal::TextureType::Type2D,
            metal::PixelFormat::RGBA8Unorm,
        )),
        3,
    );
    texture.usage = metal::ResourceUsage::SAMPLE;
    state.used_resources = vec![buffer, heap, texture, again];
    let shape = state.shape();
    assert_eq!(
        shape.used[&1].usage,
        metal::ResourceUsage::READ | metal::ResourceUsage::WRITE
    );
    let compute = |_| vk::PipelineStageFlags2::COMPUTE_SHADER;
    let tokens = |uses: &[ResourceUse]| -> Vec<(UseTarget, AccessSet, AccessClass)> {
        uses.iter()
            .map(|u| (u.resource, u.access, u.class))
            .collect()
    };
    assert!(used_uses(&shape.used, false, false, compute).is_empty());
    assert_eq!(
        tokens(&used_uses(&shape.used, true, false, compute)),
        vec![
            (
                UseTarget::Used(1),
                AccessSet::READ_WRITE,
                AccessClass::Storage
            ),
            (
                UseTarget::Used(2),
                AccessSet::READ_WRITE,
                AccessClass::Storage
            ),
        ]
    );
    assert_eq!(
        tokens(&used_uses(&shape.used, false, true, compute)),
        vec![
            (
                UseTarget::Used(2),
                AccessSet::READ_WRITE,
                AccessClass::Storage
            ),
            (UseTarget::Used(3), AccessSet::READ, AccessClass::Sampled),
        ]
    );
    // A render plan lists a resource only in the stages it is used in.
    let vertex_only = |s: metal::RenderStages| {
        if s.contains(metal::RenderStages::VERTEX) {
            vk::PipelineStageFlags2::VERTEX_SHADER
        } else {
            vk::PipelineStageFlags2::empty()
        }
    };
    assert_eq!(
        tokens(&used_uses(&shape.used, true, true, vertex_only)),
        vec![(
            UseTarget::Used(2),
            AccessSet::READ_WRITE,
            AccessClass::Storage
        )]
    );
}

/// A buffer bound by shape and state agree on the bound kind, including the length class.
#[test]
fn the_bound_kind_carries_the_length_class() {
    let b = BoundBuffer::Buffer {
        length: 100,
        offset: 4,
        gpu_address: 0,
        device_address: Some(8),
        attribute_stride: None,
    };
    assert_eq!(
        BoundKind::of(Some(&b.shape(0))),
        BoundKind::Buffer {
            device_address: true,
            length_class: 7
        }
    );
    assert_eq!(BoundKind::of(None), BoundKind::Unbound);
}

#[test]
fn slots_beyond_the_metal_tables_are_refused() {
    let mut shape = BindShape::default();
    let mut s = StageShape::default();
    s.buffers.insert(
        30,
        BufferShape::Bytes {
            attribute_stride: None,
        },
    );
    shape.stages.insert(Stage::Kernel, s.clone());
    assert_eq!(check_slots(&shape), Ok(()));
    s.buffers.insert(
        31,
        BufferShape::Bytes {
            attribute_stride: None,
        },
    );
    shape.stages.insert(Stage::Kernel, s);
    assert_eq!(
        check_slots(&shape),
        Err(BindError::SlotOutOfRange {
            slot: SlotRef {
                stage: Stage::Kernel,
                class: SlotClass::Buffer,
                index: 31
            }
        })
    );
    let mut s = StageShape::default();
    s.threadgroup_lengths.insert(31, 16);
    shape.stages.insert(Stage::Kernel, s);
    assert!(check_slots(&shape).is_err());
}

/// On every committed profile a storage or uniform view's descriptor base is aligned to
/// `max(min*BufferOffsetAlignment, 16)`, so whatever the bind offset and window, every byte of the
/// range sits at a descriptor offset congruent to its allocation offset modulo 16: a 16-byte leaf
/// whose Metal address is a multiple of 16 is one `uvec4` element of the view.
#[test]
fn descriptor_bases_are_sixteen_aligned_on_every_profile() {
    let buffer = |index| BindingSource::Buffer(slot(index));
    let bindings: BTreeMap<_, _> = [(buffer(0), (0, 0)), (buffer(1), (0, 1))].into();
    let shape = BufferShape::Buffer {
        device_address: false,
        length_class: length_class(4096),
        attribute_stride: None,
        offset_alignment: 4,
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let limits = &p.fields().limits;
        for (index, view, min) in [
            (
                0,
                DescriptorView::Storage,
                limits.min_storage_buffer_offset_alignment,
            ),
            (
                1,
                DescriptorView::UniformWords,
                limits.min_uniform_buffer_offset_alignment,
            ),
        ] {
            let b = BufferSlot {
                access: AccessSet::READ,
                window: Window::Whole,
                form: BufferForm::Descriptor(view),
                kind: BufferKind::Buffer,
            };
            let pb = plan_buffer(slot(index), &b, Some(&shape), limits, &bindings)
                .unwrap_or_else(|e| panic!("{stem} {view:?}: {e:?}"));
            assert_eq!(pb.alignment, min.max(16), "{stem} {view:?}");
            for bind_offset in (0..64).step_by(4) {
                for window in [
                    Window::Whole,
                    Window::Bounded { start: 0, len: 64 },
                    Window::Bounded { start: 4, len: 28 },
                    Window::Bounded { start: 52, len: 16 },
                    Window::Bounded { start: -8, len: 40 },
                ] {
                    let (range, residual) = crate::bind::instantiate::descriptor_range(
                        bind_offset,
                        4096,
                        window,
                        pb.alignment,
                        u64::MAX,
                    )
                    .unwrap_or_else(|e| panic!("{stem} {view:?}: {e:?}"));
                    assert_eq!(range.offset % 16, 0, "{stem} {view:?} {bind_offset}");
                    // Byte `m` from the bind offset is byte `residual + m` of the range.
                    assert_eq!(
                        u64::from(residual) % 16,
                        bind_offset % 16,
                        "{stem} {view:?} {bind_offset} {window:?}"
                    );
                }
            }
        }
    }
}

/// A uniform view's `RangeAtBindOffset` holds exactly where instantiation starts its range at the
/// bind offset, its residual 0, on every committed profile, for a buffer bound anywhere before the
/// end of its allocation under a window starting at or after the bind offset and inside the
/// allocation (elsewhere it never holds where the residual is not 0; false reads correctly at any
/// residual); and for inline bytes, at offset 0 of their upload. An unbound slot, whose residual
/// is 0, takes false: the key's residual form, which a draw's key matches where it leaves the slot
/// unbound.
#[test]
fn a_uniform_range_starts_at_the_bind_offset_where_the_specialization_says() {
    const LENGTH: u64 = 4096;
    for (stem, p) in committed_profiles().expect("profiles") {
        let limits = &p.fields().limits;
        let alignment =
            crate::memory::views::base_alignment(limits.min_uniform_buffer_offset_alignment);
        for window in [
            Window::Whole,
            Window::Bounded { start: 0, len: 64 },
            Window::Bounded { start: 4, len: 28 },
            Window::Bounded { start: 52, len: 16 },
            Window::Bounded {
                start: 192,
                len: 16,
            },
            Window::Bounded { start: 255, len: 1 },
            Window::Bounded { start: -8, len: 40 },
        ] {
            let b = BufferSlot {
                access: AccessSet::READ,
                window,
                form: BufferForm::Descriptor(DescriptorView::UniformWords),
                kind: BufferKind::Buffer,
            };
            let zero = |bind_offset: u64| {
                let (_, residual) = crate::bind::instantiate::descriptor_range(
                    bind_offset,
                    LENGTH,
                    window,
                    alignment,
                    u64::MAX,
                )
                .unwrap_or_else(|e| panic!("{stem} {window:?} {bind_offset}: {e:?}"));
                residual == 0
            };
            for bind_offset in (0..LENGTH).step_by(4) {
                let shape = BoundBuffer::Buffer {
                    gpu_address: 0x1_0000,
                    length: LENGTH,
                    offset: bind_offset,
                    device_address: None,
                    attribute_stride: None,
                }
                .shape(alignment as u32);
                let at = range_at_bind_offset(&b, Some(&shape), limits);
                let what = format!("{stem} {window:?} {bind_offset}");
                // A window starting below the bind offset, or past the allocation's end (whose
                // range is widened down into it), may still start at the bind offset.
                match window {
                    Window::Bounded { start, .. }
                        if start < 0 || bind_offset as i64 + start >= LENGTH as i64 =>
                    {
                        assert!(!at || zero(bind_offset), "{what}")
                    }
                    _ => assert_eq!(at, zero(bind_offset), "{what}"),
                }
            }
            let bytes = BufferShape::Bytes {
                attribute_stride: None,
            };
            let at = range_at_bind_offset(&b, Some(&bytes), limits);
            assert_eq!(
                at,
                zero(0) && !matches!(window, Window::Bounded { start, .. } if start < 0),
                "{stem} {window:?} bytes"
            );
            assert!(
                !range_at_bind_offset(&b, None, limits),
                "{stem} {window:?} unbound"
            );
        }
    }
}

/// A slot whose function observes residency (`RuntimeFact::SparseTexture`) binds a texture that is
/// not sparse (or nothing) on every profile, and a sparse one where the module has the sparse arm
/// or the tile-map arm; without either a sparse texture is refused, naming
/// `shaderResourceResidency` where the profile lacks it and `NotImplemented` where it has it (an
/// access without a sparse form). A slot the function writes binds a sparse texture (its writes
/// dropped under the specialization), and one it updates atomically refuses it as
/// `NotRepresentable` on every profile.
#[test]
fn a_sparse_texture_needs_the_slots_sparse_arm() {
    use crate::bind::state::TextureFacts;
    let at = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 2,
    };
    let shape = |sparse: Option<bool>| {
        let mut s = BindShape::default();
        let mut stage = StageShape::default();
        if let Some(sparse) = sparse {
            let mut facts =
                TextureFacts::new(metal::TextureType::Type2D, metal::PixelFormat::RGBA8Unorm);
            facts.sparse = sparse;
            stage.textures.insert(2, facts.shape());
        }
        s.stages.insert(Stage::Kernel, stage);
        s
    };
    let arms = |sparse_arm: bool, tile_map: bool| {
        vec![RuntimeDependency {
            fact: RuntimeFact::SparseTexture {
                slot: at,
                sparse_arm,
                tile_map,
            },
            mechanism: if sparse_arm || tile_map {
                Mechanism::Specialization
            } else {
                Mechanism::PlanTimeCheck
            },
        }]
    };
    let deps = |sparse_arm: bool| arms(sparse_arm, false);
    let writes = |atomic: bool| {
        vec![RuntimeDependency {
            fact: RuntimeFact::SparseTextureWrites { slot: at, atomic },
            mechanism: if atomic {
                Mechanism::PlanTimeCheck
            } else {
                Mechanism::Specialization
            },
        }]
    };
    assert_eq!(sparse_texture(&shape(Some(true)), at), 1);
    assert_eq!(sparse_texture(&shape(Some(false)), at), 0);
    assert_eq!(sparse_texture(&shape(None), at), 0);
    let mut seen = [0; 2];
    for (stem, p) in committed_profiles().expect("profiles") {
        for bound in [None, Some(false)] {
            for arm in [false, true] {
                check_sparse_textures(&deps(arm), &p, &shape(bound))
                    .unwrap_or_else(|e| panic!("{stem}: {bound:?} {arm}: {e}"));
            }
        }
        check_sparse_textures(&deps(true), &p, &shape(Some(true)))
            .unwrap_or_else(|e| panic!("{stem}: sparse arm: {e}"));
        // A tile-map arm answers a sparse texture too.
        check_sparse_textures(&arms(false, true), &p, &shape(Some(true)))
            .unwrap_or_else(|e| panic!("{stem}: tile-map arm: {e}"));
        let residency = p
            .fields()
            .enabled
            .features
            .contains(&VkFeature::ShaderResourceResidency);
        let feature = Feature::SparseTexture { slot: at };
        let want = if residency {
            Unsupported::not_implemented(feature)
        } else {
            Unsupported::needs(
                feature,
                Capability::Feature(VkFeature::ShaderResourceResidency),
            )
        };
        match check_sparse_textures(&deps(false), &p, &shape(Some(true))) {
            Err(BindError::Unsupported(u)) => assert_eq!(u, want, "{stem}"),
            other => panic!("{stem}: {other:?}"),
        }
        // A written slot binds a sparse texture, its writes dropped; one updated atomically refuses
        // it, since Metal's answer there is not deterministic.
        for bound in [None, Some(false), Some(true)] {
            check_sparse_textures(&writes(false), &p, &shape(bound))
                .unwrap_or_else(|e| panic!("{stem}: written, {bound:?}: {e}"));
        }
        for bound in [None, Some(false)] {
            check_sparse_textures(&writes(true), &p, &shape(bound))
                .unwrap_or_else(|e| panic!("{stem}: atomic, {bound:?}: {e}"));
        }
        let mut atomic = deps(false);
        atomic.extend(writes(true));
        for deps in [writes(true), atomic] {
            match check_sparse_textures(&deps, &p, &shape(Some(true))) {
                Err(BindError::Unsupported(u)) => {
                    let feature = Feature::SparseTexture { slot: at };
                    assert_eq!(u, Unsupported::not_representable(feature), "{stem}")
                }
                other => panic!("{stem}: atomic: {other:?}"),
            }
        }
        seen[usize::from(residency)] += 1;
    }
    assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
}

/// A constexpr-sampled slot enters `VariantKey::address_maps` only where the bound texture's
/// address-mode map changes a mode the slot's constexpr samplers state, with the map restricted
/// to those modes: a draw of an ordinary texture, or of a mapped one whose map keeps those modes,
/// keeps its key, and textures whose maps agree on them share a variant. A slot no constexpr
/// sampler samples takes no map, whatever its texture's.
#[test]
fn only_a_map_that_moves_a_constexpr_mode_enters_the_variant() {
    use crate::bind::state::TextureFacts;
    use crate::metal::sampler::AddressModeMap;
    use crate::metal::SamplerAddressMode as M;
    let texture = |index| SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    };
    let identity = AddressModeMap::IDENTITY;
    let edges = identity.with(M::Repeat, M::ClampToEdge);
    // Every mode to the edge, as a two-plane 4:2:0 texture reads.
    let all_edges = [
        M::MirrorClampToEdge,
        M::Repeat,
        M::MirrorRepeat,
        M::ClampToZero,
        M::ClampToBorderColor,
    ]
    .into_iter()
    .fold(identity, |map, mode| map.with(mode, M::ClampToEdge));
    let shape = |maps: [AddressModeMap; 2]| {
        let mut s = BindShape::default();
        let mut stage = StageShape::default();
        for (index, map) in (0..).zip(maps) {
            let mut facts =
                TextureFacts::new(metal::TextureType::Type2D, metal::PixelFormat::RGBA8Unorm);
            facts.address_modes = map;
            stage.textures.insert(index, facts.shape());
        }
        s.stages.insert(Stage::Kernel, stage);
        s
    };
    // Slot 0 is constexpr-sampled, by a repeat sampler and by one of `other` modes; slot 1 is
    // not.
    let deps = |other: Option<M>| {
        let sampled = |mode| RuntimeDependency {
            fact: RuntimeFact::ConstexprSampled {
                slot: texture(0),
                modes: [mode; 3],
            },
            mechanism: Mechanism::Variant,
        };
        let mut deps = vec![sampled(M::Repeat)];
        deps.extend(other.map(sampled));
        deps
    };
    let repeat = deps(None);
    assert!(address_map_variant(&repeat, &shape([identity, identity])).is_empty());
    assert!(address_map_variant(&repeat, &shape([identity, edges])).is_empty());
    assert!(address_map_variant(&[], &shape([edges, edges])).is_empty());
    assert_eq!(
        address_map_variant(&repeat, &shape([edges, identity])),
        BTreeMap::from([(texture(0), edges)])
    );
    // The map restricted to the stated modes: one that sends every mode to the edge states the
    // same variant as one that sends repeat there alone.
    assert_eq!(
        address_map_variant(&repeat, &shape([all_edges, identity])),
        BTreeMap::from([(texture(0), edges)])
    );
    // A map that keeps every stated mode adds nothing: a clamp-to-edge sampler of a texture whose
    // every mode reads the edge samples it as before.
    let edge = vec![RuntimeDependency {
        fact: RuntimeFact::ConstexprSampled {
            slot: texture(0),
            modes: [M::ClampToEdge; 3],
        },
        mechanism: Mechanism::Variant,
    }];
    assert!(address_map_variant(&edge, &shape([all_edges, identity])).is_empty());
    // Two samplers of the slot: the map on the modes of both.
    assert_eq!(
        address_map_variant(&deps(Some(M::ClampToZero)), &shape([all_edges, identity])),
        BTreeMap::from([(texture(0), edges.with(M::ClampToZero, M::ClampToEdge))])
    );
    // A slot with no texture bound takes no map.
    assert!(address_map_variant(&repeat, &BindShape::default()).is_empty());
}
