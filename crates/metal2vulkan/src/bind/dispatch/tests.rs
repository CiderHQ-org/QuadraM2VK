//! Dispatch regions, splits, local-size checks, and instantiation on synthetic templates, including
//! the required test: a `ThreadgroupsIndirect` plan has no CPU-argument dependency; and the
//! emulated-simdgroups variant a partial simdgroup asks for.

use super::*;
use crate::bind::plan::{BoundKind, PlannedBuffer};
use crate::bind::state::{BoundBuffer, ResourceClass, StageTable, UsedResource};
use crate::pipeline::artifact::SpecConstant;
use crate::pipeline::interface::{AccessSet, BufferForm, DescriptorView, PlacedField, Window};
use crate::pipeline::sym::{SlotClass, SlotRef, Stage};

/// `BindingPlan::instantiate_dispatch` of a compute plan of template `t` with no variant.
fn instantiate(
    t: &ComputeTemplate,
    s: &BindState,
    d: &Dispatch,
) -> Result<DispatchCommands, BindError> {
    compute_plan(t, VariantKey::new()).instantiate_dispatch(s, d)
}

fn compute_plan(t: &ComputeTemplate, variant: VariantKey) -> BindingPlan {
    BindingPlan {
        variant,
        compute: Some(t.clone()),
        ..BindingPlan::default()
    }
}

fn rules() -> LocalRules {
    LocalRules {
        flat: false,
        multiples: Vec::new(),
        max_size: [1024, 1024, 64],
        max_invocations: 1024,
        max_subgroups: 32,
        max_count: [65535, 65535, 65535],
        required_subgroup_size: None,
        stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
        scratch: None,
        tile: false,
        tile_threadgroup: None,
    }
}

fn kslot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Buffer,
        index,
    }
}

fn spec(id: u32, key: SpecKey) -> SpecConstant {
    SpecConstant {
        id,
        key,
        size: 4,
        default: 1,
        unread_bits: 0,
    }
}

/// A kernel reading grid builtins with one storage slot, the region field at 16, and one
/// residual at 48 (after the root and status addresses at 0 and 8).
fn template() -> ComputeTemplate {
    let f = |field, offset, size| PlacedField {
        field,
        loc: PushLoc::Push { offset },
        size,
    };
    ComputeTemplate {
        buffers: vec![PlannedBuffer {
            slot: kslot(0),
            window: Window::Bounded { start: 0, len: 64 },
            form: BufferForm::Descriptor(DescriptorView::Storage),
            access: AccessSet::READ_WRITE,
            binding: Some((0, 0)),
            alignment: 64,
            max_range: 1 << 27,
            bound: BoundKind::Buffer {
                device_address: false,
                length_class: crate::bind::state::length_class(4096),
            },
            disjoint: false,
        }],
        spec: vec![
            spec(0, SpecKey::LocalSize(Axis::X)),
            spec(1, SpecKey::LocalSize(Axis::Y)),
            spec(2, SpecKey::LocalSize(Axis::Z)),
            spec(3, SpecKey::MultiRegion),
        ],
        fixed: Vec::new(),
        local: rules(),
        fields: vec![
            f(PushField::DispatchRegion, 0, 32),
            f(PushField::SlotBindOffset(kslot(0)), 32, 4),
        ],
        push_range: Some((0, 36)),
        table: None,
        root_first: None,
        storage_alignment: 64,
        resources: Default::default(),
        stage_in_region: false,
    }
}

fn state(offset: u64) -> BindState {
    let mut t = StageTable::default();
    t.buffers.insert(
        0,
        BoundBuffer::Buffer {
            length: 4096,
            offset,
            gpu_address: 0x1_0000,
            device_address: None,
            attribute_stride: None,
        },
    );
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    s
}

fn spec_value(key: &SpecializationKey, id: u32) -> Option<u32> {
    let e = key.entries.iter().find(|e| e.constant_id == id)?;
    let at = e.offset as usize;
    Some(u32::from_le_bytes(key.data[at..at + 4].try_into().ok()?))
}

fn push_words(r: &DispatchRegion) -> Vec<u32> {
    r.push_constants[0]
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

#[test]
fn a_threadgroups_indirect_plan_has_no_cpu_argument_dependency() {
    let t = template();
    let d = |token, offset| Dispatch::ThreadgroupsIndirect {
        indirect: IndirectRef::new(token, offset),
        threads_per_group: [8, 4, 2],
    };
    let a = instantiate(&t, &state(0), &d(1, 0)).expect("a");
    let b = instantiate(&t, &state(0), &d(9, 4096)).expect("b");
    assert_eq!(a.regions.len(), 1);
    let (ra, rb) = (&a.regions[0], &b.regions[0]);
    // One region, `MultiRegion = 0`, the grid from `NumWorkgroups`: nothing but the argument
    // reference differs, so the commands cannot depend on the GPU-written counts.
    assert_eq!(spec_value(&ra.specialization, 3), Some(0));
    assert_eq!(ra.specialization, rb.specialization);
    assert_eq!(ra.push_constants, rb.push_constants);
    assert_eq!(ra.group_count, [0; 3]);
    assert_eq!(
        (ra.indirect, rb.indirect),
        (
            Some(IndirectArgs::ClampScratch),
            Some(IndirectArgs::ClampScratch)
        )
    );
    // The region field is written anyway (VUID-vkCmdDispatch-maintenance4-08602): base 0,
    // the nominal size packed, the unknown grid 0.
    let w = push_words(ra);
    assert_eq!(&w[..8], &[0, 0, 0, 7 | 3 << 10 | 1 << 20, 0, 0, 0, 0]);
    let (ca, cb) = (
        a.indirect_clamp.expect("clamp"),
        b.indirect_clamp.expect("clamp"),
    );
    assert_eq!((ca.source.token, cb.source.token), (1, 9));
    assert_eq!(ca.limits, [65535; 3]);
    // The arguments are a use of the dispatch: the clamp prepass reads them as a storage buffer.
    let args = |class, stages| ResourceUse {
        resource: UseTarget::Indirect,
        access: AccessSet::READ,
        class,
        stages,
        range: Some(ByteRange { offset: 0, len: 12 }),
    };
    assert_eq!(
        a.uses,
        vec![args(
            AccessClass::Storage,
            vk::PipelineStageFlags2::COMPUTE_SHADER
        )]
    );
    // A device whose limit is the full `u32` range dispatches from Metal's buffer directly.
    let mut t = template();
    t.local.max_count = [u32::MAX; 3];
    let c = instantiate(&t, &state(0), &d(5, 64)).expect("unclamped");
    assert_eq!(c.indirect_clamp, None);
    assert_eq!(
        c.regions[0].indirect,
        Some(IndirectArgs::Buffer(IndirectRef::new(5, 64)))
    );
    assert_eq!(
        c.uses,
        vec![args(
            AccessClass::Indirect,
            vk::PipelineStageFlags2::DRAW_INDIRECT
        )]
    );
    // A direct dispatch has no uses of its own.
    let direct = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [8, 4, 2],
    };
    assert!(instantiate(&t, &state(0), &direct)
        .expect("direct")
        .uses
        .is_empty());
}

#[test]
fn dispatch_threads_edges_become_multi_region_dispatches() {
    let t = template();
    let d = Dispatch::Threads {
        threads: [40, 3, 1],
        threads_per_group: [16, 2, 1],
    };
    let c = instantiate(&t, &state(0), &d).expect("threads");
    let got: Vec<([u32; 3], [u32; 3], Option<u32>, Vec<u32>)> = c
        .regions
        .iter()
        .map(|r| {
            let local = [0, 1, 2].map(|id| spec_value(&r.specialization, id).unwrap_or(0));
            (
                local,
                r.group_count,
                spec_value(&r.specialization, 3),
                push_words(r)[..8].to_vec(),
            )
        })
        .collect();
    let nominal = 15 | 1 << 10;
    let words = |base: [u32; 3]| vec![base[0], base[1], base[2], nominal, 40, 3, 1, 0];
    assert_eq!(
        got,
        vec![
            ([16, 2, 1], [2, 1, 1], Some(1), words([0, 0, 0])),
            ([16, 1, 1], [2, 1, 1], Some(1), words([0, 1, 0])),
            ([8, 2, 1], [1, 1, 1], Some(1), words([2, 0, 0])),
            ([8, 1, 1], [1, 1, 1], Some(1), words([2, 1, 0])),
        ]
    );
    // A multiple of the threadgroup size is one `MultiRegion = 0` region.
    let d = Dispatch::Threads {
        threads: [64, 4, 1],
        threads_per_group: [16, 2, 1],
    };
    let c = instantiate(&t, &state(0), &d).expect("uniform");
    assert_eq!(c.regions.len(), 1);
    assert_eq!(c.regions[0].group_count, [4, 2, 1]);
    assert_eq!(spec_value(&c.regions[0].specialization, 3), Some(0));
    // An empty grid dispatches nothing.
    let d = Dispatch::Threads {
        threads: [0, 4, 1],
        threads_per_group: [16, 2, 1],
    };
    assert!(instantiate(&t, &state(0), &d)
        .expect("empty")
        .regions
        .is_empty());
}

#[test]
fn group_counts_above_the_limit_are_split_with_their_base() {
    let t = template();
    let d = Dispatch::Threadgroups {
        groups: [70000, 1, 2],
        threads_per_group: [1, 1, 1],
    };
    let c = instantiate(&t, &state(0), &d).expect("split");
    let got: Vec<([u32; 3], [u32; 3])> = c
        .regions
        .iter()
        .map(|r| {
            let w = push_words(r);
            (r.group_count, [w[0], w[1], w[2]])
        })
        .collect();
    assert_eq!(
        got,
        vec![([65535, 1, 2], [0, 0, 0]), ([4465, 1, 2], [65535, 0, 0])]
    );
    for r in &c.regions {
        assert_eq!(spec_value(&r.specialization, 3), Some(1));
        assert_eq!(&push_words(r)[4..7], &[70000, 1, 2], "the whole grid");
    }
    // Within the limits: one region without `MultiRegion`, the grid still written.
    let d = Dispatch::Threadgroups {
        groups: [3, 1, 1],
        threads_per_group: [32, 1, 1],
    };
    let c = instantiate(&t, &state(0), &d).expect("one");
    assert_eq!(c.regions.len(), 1);
    assert_eq!(spec_value(&c.regions[0].specialization, 3), Some(0));
    assert_eq!(&push_words(&c.regions[0])[..8], &[0, 0, 0, 31, 96, 1, 1, 0]);
    // Past `MAX_REGIONS`: a typed refusal naming the axis.
    let mut t = template();
    t.local.max_count = [1, 1, 1];
    let d = Dispatch::Threadgroups {
        groups: [1, 5000, 1],
        threads_per_group: [1, 1, 1],
    };
    let e = instantiate(&t, &state(0), &d).expect_err("too many");
    assert!(
        matches!(&e, BindError::Unsupported(u) if u.feature == Feature::GroupCount { axis: Some(Axis::Y) }),
        "{e:?}"
    );
    // A grid past 2^32 - 1 threads on an axis is not Metal's.
    let d = Dispatch::Threadgroups {
        groups: [1 << 22, 1, 1],
        threads_per_group: [1024, 1, 1],
    };
    assert!(instantiate(&template(), &state(0), &d).is_err());
}

#[test]
fn local_sizes_are_checked_against_every_d4_limit() {
    let mut r = rules();
    assert!(check_local(&r, [1024, 1, 1]).is_ok());
    let e = check_local(&r, [1, 1, 65]).expect_err("z");
    assert!(
        matches!(&e, BindError::Unsupported(u) if u.feature == Feature::LocalSize { axis: Some(Axis::Z) }),
        "{e:?}"
    );
    assert!(check_local(&r, [64, 32, 1]).is_err(), "invocations");
    // Flattened: `sn` against size[0], invocations, and subgroups · S; the multiple.
    r.flat = true;
    r.max_size = [512, 1024, 64];
    r.required_subgroup_size = Some(32);
    r.max_subgroups = 8;
    r.multiples = vec![32];
    assert!(check_local(&r, [16, 16, 1]).is_ok());
    assert!(check_local(&r, [32, 32, 1]).is_err(), "sn > size[0]");
    r.max_size = [1024, 1024, 64];
    let e = check_local(&r, [32, 16, 1]).expect_err("subgroups");
    assert!(
        matches!(&e, BindError::Unsupported(u) if matches!(u.needs, crate::error::Needs::AnyOf(ref a) if a[0][0] == Capability::Limit { limit: VkLimit::MaxComputeWorkgroupSubgroups, required: 16, available: 8 })),
        "{e:?}"
    );
    // A partial simdgroup runs on the emulated-simdgroups variant, whose local size is not
    // flattened: each axis and the product are checked, not `sn` and the subgroups.
    assert_eq!(
        fit(&r, &[[33, 1, 1]]).expect("partial"),
        Fit::EmulatedSimdgroups
    );
    assert_eq!(fit(&r, &[[32, 1, 1]]).expect("full"), Fit::Planned);
    assert_eq!(
        fit(&r, &[[1, 1, 1]]).expect("one thread"),
        Fit::EmulatedSimdgroups
    );
    r.max_subgroups = 1;
    assert_eq!(
        fit(&r, &[[60, 1, 1]]).expect("the subgroup limit is the flattened tier's"),
        Fit::EmulatedSimdgroups
    );
    let e = fit(&r, &[[1, 1, 65]]).expect_err("z of a partial region");
    assert!(
        matches!(&e, BindError::Unsupported(u) if u.feature == Feature::LocalSize { axis: Some(Axis::Z) }),
        "{e:?}"
    );
    let e = fit(&r, &[[33, 33, 1]]).expect_err("invocations of a partial region");
    assert!(
        matches!(&e, BindError::Unsupported(u) if u.feature == Feature::LocalSize { axis: None }),
        "{e:?}"
    );
    // The whole dispatch runs on the variant, so every region of a dispatch with a partial one
    // is checked against the variant's limits, its whole-subgroup regions included: the z of
    // [1, 1, 96] is past them, and [1, 1, 64], past the flattened tier's subgroups, is not.
    let e = fit(&r, &[[1, 1, 96], [1, 1, 4]]).expect_err("z of a whole region");
    assert!(
        matches!(&e, BindError::Unsupported(u) if u.feature == Feature::LocalSize { axis: Some(Axis::Z) }),
        "{e:?}"
    );
    assert!(
        fit(&r, &[[1, 1, 64]]).is_err(),
        "two subgroups on the flattened tier"
    );
    assert_eq!(
        fit(&r, &[[1, 1, 64], [1, 1, 36]]).expect("mixed"),
        Fit::EmulatedSimdgroups
    );
    // Outside the flattened tiers a multiple is a derivative group's, and stays refused.
    let mut d4 = rules();
    d4.multiples = vec![4];
    assert_eq!(fit(&d4, &[[8, 1, 1]]).expect("quads"), Fit::Planned);
    let e = fit(&d4, &[[6, 1, 1]]).expect_err("a partial quad");
    assert!(
        matches!(&e, BindError::Unsupported(u) if u.feature == Feature::ComputeDerivatives),
        "{e:?}"
    );
    // Metal's nominal size is 1..=1024 per axis.
    let d = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [0, 1, 1],
    };
    assert!(regions(&rules(), &d).is_err());
    let d = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [1025, 1, 1],
    };
    assert!(regions(&rules(), &d).is_err());
    // An edge region is checked too: a flattened kernel's partial edge runs on the
    // emulated-simdgroups variant, and a full grid on the flattened pipeline.
    let mut r = rules();
    r.multiples = vec![32];
    r.flat = true;
    let d = Dispatch::Threads {
        threads: [40, 1, 1],
        threads_per_group: [32, 1, 1],
    };
    assert!(regions(&r, &d).expect("edge").emulated);
    let d = Dispatch::Threads {
        threads: [64, 1, 1],
        threads_per_group: [32, 1, 1],
    };
    assert!(!regions(&r, &d).expect("full").emulated);
}

/// A flattened (Direct) kernel's template: `sn` in one spec constant, whole 32-lane subgroups.
fn flat_template() -> ComputeTemplate {
    let mut t = template();
    t.spec.push(spec(4, SpecKey::FlatLocalSize));
    t.local.flat = true;
    t.local.multiples = vec![32];
    t.local.required_subgroup_size = Some(32);
    t.local.stage_flags = vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS;
    t
}

fn groups(threads_per_group: [u32; 3]) -> Dispatch {
    Dispatch::Threadgroups {
        groups: [2, 1, 1],
        threads_per_group,
    }
}

/// A dispatch of a flattened kernel whose threadgroup has a partial simdgroup (1, 4, or 16 threads,
/// the authored cases' sizes; a `dispatchThreads` edge; an indirect dispatch) runs on the kernel's
/// emulated-simdgroups variant: `dispatch_variant` adds it to the plan's key, and
/// `instantiate_dispatch` names that key instead of writing anything. Whole simdgroups keep the
/// planned pipeline, and so does an empty grid.
#[test]
fn a_partial_simdgroup_asks_for_the_emulated_simdgroups_variant() {
    let t = flat_template();
    let mut base = VariantKey::new();
    base.psb_slots.insert(kslot(3));
    let plan = compute_plan(&t, base.clone());
    let mut emulated = base.clone();
    emulated.emulated_simdgroups = true;
    let partial = [
        groups([1, 1, 1]),
        groups([4, 1, 1]),
        groups([16, 1, 1]),
        groups([8, 2, 1]),
        groups([33, 1, 1]),
        Dispatch::Threads {
            threads: [100, 1, 1],
            threads_per_group: [32, 1, 1],
        },
        Dispatch::ThreadgroupsIndirect {
            indirect: IndirectRef::new(0, 0),
            threads_per_group: [8, 1, 1],
        },
    ];
    for d in &partial {
        assert_eq!(plan.dispatch_variant(d), Ok(emulated.clone()), "{d:?}");
        assert_eq!(
            plan.instantiate_dispatch(&state(0), d),
            Err(BindError::NeedsVariant {
                variant: Box::new(emulated.clone())
            }),
            "{d:?}"
        );
    }
    let whole = [
        groups([32, 1, 1]),
        groups([8, 4, 2]),
        Dispatch::Threads {
            threads: [96, 2, 1],
            threads_per_group: [32, 1, 1],
        },
        Dispatch::Threadgroups {
            groups: [0, 1, 1],
            threads_per_group: [16, 1, 1],
        },
    ];
    for d in &whole {
        assert_eq!(plan.dispatch_variant(d), Ok(base.clone()), "{d:?}");
        assert!(plan.instantiate_dispatch(&state(0), d).is_ok(), "{d:?}");
    }
    // A limit every tier has is reported, not answered with a variant that cannot run it.
    let too_many = groups([33, 33, 1]);
    assert!(matches!(
        plan.dispatch_variant(&too_many),
        Err(BindError::Unsupported(_))
    ));
    assert!(matches!(
        plan.instantiate_dispatch(&state(0), &too_many),
        Err(BindError::Unsupported(_))
    ));
    // A `dispatchThreads` grid whose edge is partial runs whole on the variant, so its interior
    // is checked against the variant's per-axis limits (z at most 64 here), not the flattened
    // `sn`: the answer is a refusal, not a variant whose plan refuses the interior.
    let mixed = |tpg_z| Dispatch::Threads {
        threads: [1, 1, 100],
        threads_per_group: [1, 1, tpg_z],
    };
    let variant = compute_plan(&template(), emulated.clone());
    for r in [
        plan.dispatch_variant(&mixed(96)).map(|_| ()),
        plan.instantiate_dispatch(&state(0), &mixed(96)).map(|_| ()),
        variant
            .instantiate_dispatch(&state(0), &mixed(96))
            .map(|_| ()),
    ] {
        let e = r.expect_err("z of the interior");
        assert!(
            matches!(&e, BindError::Unsupported(u) if u.feature == Feature::LocalSize { axis: Some(Axis::Z) }),
            "{e:?}"
        );
    }
    assert_eq!(plan.dispatch_variant(&mixed(64)), Ok(emulated.clone()));
    let c = variant
        .instantiate_dispatch(&state(0), &mixed(64))
        .expect("the variant runs both regions");
    assert_eq!(c.regions.len(), 2);
    // `Cluster32` runs whole 64-lane subgroups: one simdgroup of 32 threads is a partial
    // subgroup there.
    let mut cluster = flat_template();
    cluster.local.multiples = vec![64];
    let cluster = compute_plan(&cluster, VariantKey::new());
    let mut only = VariantKey::new();
    only.emulated_simdgroups = true;
    assert_eq!(cluster.dispatch_variant(&groups([32, 1, 1])), Ok(only));
    assert_eq!(
        cluster.dispatch_variant(&groups([64, 1, 1])),
        Ok(VariantKey::new())
    );
}

/// The plan of the emulated-simdgroups variant (its local size not flattened, no multiple) states
/// the key and runs every threadgroup size with the local size as Metal gave it.
#[test]
fn the_emulated_variant_runs_every_threadgroup_size() {
    let mut emulated = VariantKey::new();
    emulated.emulated_simdgroups = true;
    let plan = compute_plan(&template(), emulated.clone());
    for n in [1, 4, 16, 32, 33] {
        let d = groups([n, 1, 1]);
        assert_eq!(plan.dispatch_variant(&d), Ok(emulated.clone()));
        let c = plan.instantiate_dispatch(&state(0), &d).expect("runs");
        assert_eq!(c.regions.len(), 1);
        let r = &c.regions[0];
        assert_eq!(spec_value(&r.specialization, 0), Some(n));
        assert_eq!(r.stage_flags, vk::PipelineShaderStageCreateFlags::empty());
        assert_eq!(r.required_subgroup_size, None);
    }
}

#[test]
fn residuals_ranges_extents_and_holes_follow_the_state() {
    let mut t = template();
    t.fields.insert(
        0,
        PlacedField {
            field: PushField::RootTableAddress,
            loc: PushLoc::Push { offset: 40 },
            size: 8,
        },
    );
    t.fields.push(PlacedField {
        field: PushField::ExecutionStatusAddress,
        loc: PushLoc::Push { offset: 48 },
        size: 8,
    });
    t.push_range = Some((0, 56));
    t.root_first = Some(2);
    let mut s = state(100);
    let mut heap = UsedResource::new(ResourceClass::Heap, 7);
    heap.gpu_address = Some(0x8000);
    heap.length = 0x100;
    heap.device_address = Some(0x9_0000);
    s.used_resources.push(heap);
    let d = Dispatch::Threadgroups {
        groups: [1, 1, 1],
        threads_per_group: [1, 1, 1],
    };
    let c = instantiate(&t, &s, &d).expect("instantiate");
    assert_eq!(c.descriptor_ranges.len(), 1);
    let r = c.descriptor_ranges[0].range;
    assert_eq!((r.offset, r.range), (64, 112));
    assert_eq!(push_words(&c.regions[0])[8], 36, "residual 100 - 64");
    assert_eq!(c.extents.len(), 1);
    assert_eq!(
        c.push_holes,
        vec![
            PushHole::RootTableAddress {
                stages: vk::ShaderStageFlags::COMPUTE,
                offset: 40
            },
            PushHole::ExecutionStatusAddress {
                stages: vk::ShaderStageFlags::COMPUTE,
                offset: 48
            },
        ]
    );
    // The resident set holds the heap only: the bound buffer has no device address.
    let table = c.root_table.expect("root table");
    assert_eq!(table.bytes.len(), 3 * 32);
    let word = |at: usize| u64::from_le_bytes(table.bytes[at..at + 8].try_into().expect("u64"));
    assert_eq!((word(40), word(48)), (2, 1), "R0 = 2, N = 1");
    assert_eq!((word(64), word(72), word(80)), (0x9_0000, 0x8000, 0x100));
    // A state of another shape is the consumer's bug, reported, never guessed.
    let e = instantiate(&t, &BindState::default(), &d).expect_err("shape");
    assert!(matches!(e, BindError::Internal(_)), "{e:?}");
}

/// A kernel with emulated simdgroups takes its scratch per simdgroup of each region beside its
/// fixed threadgroup memory; a region whose total exceeds the limit is refused, and an edge region
/// is checked with its own (smaller) size.
#[test]
fn emulated_simdgroup_scratch_is_checked_per_region() {
    use crate::bind::plan::ScratchRule;
    let mut r = rules();
    r.scratch = Some(ScratchRule {
        per_simdgroup: 512,
        fixed: 28 * 1024,
        available: 32 * 1024,
    });
    // 8 simdgroups: 28 KiB + 4 KiB fits; 257 threads are 9 simdgroups.
    assert!(check_local(&r, [16, 16, 1]).is_ok());
    let e = check_local(&r, [257, 1, 1]).expect_err("scratch");
    assert!(
        matches!(&e, BindError::Unsupported(u) if u.feature == Feature::ThreadgroupMemorySize && matches!(u.needs, crate::error::Needs::AnyOf(ref a) if a[0][0] == Capability::Limit { limit: VkLimit::MaxComputeSharedMemorySize, required: 28 * 1024 + 9 * 512, available: 32 * 1024 })),
        "{e:?}"
    );
    // `dispatchThreads`: a region of 300 threads does not fit; one of 256 with an edge region
    // of 44 does.
    let d = Dispatch::Threads {
        threads: [300, 1, 1],
        threads_per_group: [300, 1, 1],
    };
    assert!(regions(&r, &d).is_err());
    let d = Dispatch::Threads {
        threads: [300, 1, 1],
        threads_per_group: [256, 1, 1],
    };
    assert!(regions(&r, &d).is_ok(), "256 + an edge of 44");
}

/// A range of a dispatch's threadgroups (a region tile dispatch) runs those threadgroups alone,
/// each region taking its first threadgroup and the whole dispatch's grid from the push field
/// (`MultiRegion` 1), split at the group-count limit as any dispatch; the range is cut to the
/// grid, and one that holds no threadgroup of it runs nothing.
#[test]
fn a_threadgroup_range_runs_its_threadgroups_of_the_whole_grid() {
    let t = template();
    let range = |first, count| Dispatch::ThreadgroupRange {
        groups: [4, 3, 1],
        first,
        count,
        threads_per_group: [8, 8, 1],
    };
    let c = instantiate(&t, &state(0), &range([1, 1, 0], [2, 1, 1])).expect("range");
    assert_eq!(c.regions.len(), 1);
    let r = &c.regions[0];
    assert_eq!(r.group_count, [2, 1, 1]);
    assert_eq!(spec_value(&r.specialization, 3), Some(1));
    let w = push_words(r);
    assert_eq!(&w[..3], &[1, 1, 0], "the first threadgroup");
    assert_eq!(&w[4..7], &[32, 24, 1], "the whole grid");
    // Cut to the grid.
    let c = instantiate(&t, &state(0), &range([3, 2, 0], [5, 5, 1])).expect("cut");
    assert_eq!(
        c.regions.iter().map(|r| r.group_count).collect::<Vec<_>>(),
        vec![[1, 1, 1]]
    );
    // Nothing of the grid.
    for (first, count) in [([4, 0, 0], [1, 1, 1]), ([0, 0, 0], [0, 1, 1])] {
        let c = instantiate(&t, &state(0), &range(first, count)).expect("empty");
        assert!(c.regions.is_empty(), "{first:?} {count:?}");
    }
    // Split at the group-count limit, each chunk from its own first threadgroup.
    let mut t = template();
    t.local.max_count = [2, 65535, 65535];
    let c = instantiate(&t, &state(0), &range([1, 0, 0], [3, 1, 1])).expect("split");
    let got: Vec<([u32; 3], [u32; 3])> = c
        .regions
        .iter()
        .map(|r| {
            let w = push_words(r);
            (r.group_count, [w[0], w[1], w[2]])
        })
        .collect();
    assert_eq!(got, vec![([2, 1, 1], [1, 0, 0]), ([1, 1, 1], [3, 0, 0])]);
}

/// A tile function whose threadgroup is the tile (`threadgroupSizeMatchesTileSize`) runs its
/// threads per tile, 8x8 here, in workgroups of that size, but every region carries the 16x16
/// tile as Metal's threadgroup and the tiles times the tile as the grid (`MultiRegion` 1), as
/// plain Metal answers them (on an M3 Max and an M4); a region of the tiles alike.
#[test]
fn a_tile_sized_threadgroup_carries_the_tile_in_every_region() {
    let mut t = template();
    t.local.tile = true;
    t.local.tile_threadgroup = Some([16, 16, 1]);
    let tile_nominal = 15 | 15 << 10;
    let d = Dispatch::Threadgroups {
        groups: [4, 2, 1],
        threads_per_group: [8, 8, 1],
    };
    let c = instantiate(&t, &state(0), &d).expect("tiles");
    assert_eq!(c.regions.len(), 1);
    let r = &c.regions[0];
    let local = [0, 1, 2].map(|id| spec_value(&r.specialization, id).unwrap_or(0));
    assert_eq!(local, [8, 8, 1], "the threads per tile run");
    assert_eq!(r.group_count, [4, 2, 1]);
    assert_eq!(spec_value(&r.specialization, 3), Some(1));
    assert_eq!(push_words(r)[..8], [0, 0, 0, tile_nominal, 64, 32, 1, 0]);
    let range = Dispatch::ThreadgroupRange {
        groups: [4, 2, 1],
        first: [1, 1, 0],
        count: [2, 1, 1],
        threads_per_group: [8, 8, 1],
    };
    let c = instantiate(&t, &state(0), &range).expect("range");
    let r = &c.regions[0];
    assert_eq!(r.group_count, [2, 1, 1]);
    assert_eq!(push_words(r)[..8], [1, 1, 0, tile_nominal, 64, 32, 1, 0]);
    // Without it the nominal threadgroup is the dispatch's, one region needing no push field.
    t.local.tile_threadgroup = None;
    let c = instantiate(&t, &state(0), &d).expect("plain");
    assert_eq!(spec_value(&c.regions[0].specialization, 3), Some(0));
    assert_eq!(push_words(&c.regions[0])[3..7], [7 | 7 << 10, 32, 16, 1]);
}
