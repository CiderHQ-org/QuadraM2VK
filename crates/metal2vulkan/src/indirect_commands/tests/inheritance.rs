use super::*;

#[test]
fn pipeline_and_buffer_inheritance_select_independent_sources() {
    let mut record = CommandRecord {
        action: Action::DispatchThreadgroups {
            grid: [1; 3],
            local_size: [1; 3],
        },
        bindings: Bindings {
            pipeline: NonZeroU64::new(20),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut parent = Bindings {
        pipeline: NonZeroU64::new(10),
        ..Default::default()
    };
    record.bindings.buffers_mut(BufferStage::Kernel)[0] = Some(BufferBinding {
        address: 2 << 32,
        attribute_stride: None,
    });
    record.threadgroup_memory_lengths[0] = 16;
    record.threadgroup_memory_lengths[THREADGROUP_COUNT - 1] = 64;
    parent.buffers_mut(BufferStage::Kernel)[0] = Some(BufferBinding {
        address: 1 << 32,
        attribute_stride: None,
    });
    for (pipeline, expected_pipeline) in [(StateSource::Command, 20), (StateSource::Parent, 10)] {
        for (buffers, expected_address) in [
            (StateSource::Command, 2 << 32),
            (StateSource::Parent, 1 << 32),
        ] {
            let resolved = record
                .resolve_bindings(Inheritance { pipeline, buffers }, &parent)
                .unwrap();
            assert_eq!(resolved.pipeline.unwrap().get(), expected_pipeline);
            assert_eq!(resolved.threadgroup_memory_lengths[0], 16);
            assert_eq!(
                resolved.threadgroup_memory_lengths[THREADGROUP_COUNT - 1],
                64
            );
            assert!(std::ptr::eq(
                resolved.threadgroup_memory_lengths,
                &record.threadgroup_memory_lengths
            ));
            assert_eq!(
                resolved.buffers(BufferStage::Kernel)[0].map(|b| b.address),
                Some(expected_address)
            );
            let owner = match buffers {
                StateSource::Command => &record.bindings,
                StateSource::Parent => &parent,
            };
            assert!(std::ptr::eq(
                resolved.buffers(BufferStage::Kernel),
                owner.buffers(BufferStage::Kernel)
            ));
        }
    }
    let before = record.clone();
    parent.pipeline = NonZeroU64::new(30);
    let resolved = record
        .resolve_bindings(
            Inheritance {
                pipeline: StateSource::Parent,
                buffers: StateSource::Command,
            },
            &parent,
        )
        .unwrap();
    assert_eq!(resolved.pipeline.unwrap().get(), 30);
    assert_eq!(record, before);
}

#[test]
fn a_missing_selected_pipeline_does_not_fall_back_to_the_other_source() {
    for source in [StateSource::Command, StateSource::Parent] {
        let mut record = CommandRecord {
            action: Action::DispatchThreads {
                grid: [1; 3],
                local_size: [1; 3],
            },
            ..Default::default()
        };
        let mut parent = Bindings::default();
        match source {
            StateSource::Command => parent.pipeline = NonZeroU64::new(1),
            StateSource::Parent => record.bindings.pipeline = NonZeroU64::new(2),
        };
        let policy = Inheritance {
            pipeline: source,
            buffers: source,
        };
        assert_eq!(
            record.resolve_bindings(policy, &parent).unwrap_err(),
            BindingError::MissingPipeline { source }
        );
        record.action = Action::Empty;
        assert!(record
            .resolve_bindings(policy, &parent)
            .unwrap()
            .pipeline
            .is_none());
    }
}
