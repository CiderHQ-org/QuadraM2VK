use super::*;

/// The base of the test's allocations: the lowest address a binding names a buffer at.
const BASE: u64 = BufferBinding::MIN_BUFFER_ADDRESS;

/// A binding of `address`, at the pipeline's stride.
fn at(address: u64) -> BufferBinding {
    BufferBinding {
        address,
        attribute_stride: None,
    }
}

#[test]
fn buffer_resolution_rejects_ambiguous_registrations_in_either_order() {
    let first = 1;
    let second = 2;
    for reversed in [false, true] {
        let mut buffers = [
            RegisteredBuffer::new(&first, BASE + 0x1000, 64).unwrap(),
            RegisteredBuffer::new(&second, BASE + 0x1020, 32).unwrap(),
        ];
        if reversed {
            buffers.reverse();
        }
        assert!(matches!(
            at(BASE + 0x1024).resolve_buffer(&buffers, 4),
            Err(AddressError::Ambiguous {
                address,
                required_bytes: 4
            }) if address == BASE + 0x1024
        ));
        let unique = at(BASE + 0x1004)
            .resolve_buffer(&buffers, 4)
            .unwrap()
            .unwrap();
        assert!(std::ptr::eq(unique.resource, &first));
        assert_eq!(unique.offset, 4);
    }
}

#[test]
fn buffer_resolution_preserves_interior_offsets_and_aliases() {
    let allocation = [0u8; 64];
    let buffers = [RegisteredBuffer::new(&allocation, BASE + 0x1000, 64).unwrap()];
    for (address, required, offset, available) in
        [(0x1004, 4, 4, 60), (0x1008, 56, 8, 56), (0x1040, 0, 64, 0)]
    {
        let view = at(BASE + address)
            .resolve_buffer(&buffers, required)
            .unwrap()
            .unwrap();
        assert!(std::ptr::eq(view.resource, &allocation));
        assert_eq!((view.offset, view.available_bytes), (offset, available));
    }
    // Below 2^32 an address is a nil buffer's offset: it names no buffer, an allocation there
    // included.
    let low = [RegisteredBuffer::new(&allocation, BASE, 64).unwrap()];
    for address in [0, 0x30, BASE - 1] {
        assert!(at(address).resolve_buffer(&low, 4).unwrap().is_none());
    }
    for (address, required_bytes) in [(0xfff, 1), (0x1040, 1), (0x1041, 0), (0x1004, u64::MAX)] {
        assert!(matches!(
            at(BASE + address).resolve_buffer(&buffers, required_bytes),
            Err(AddressError::Unavailable { .. })
        ));
    }
    let last_bytes = [RegisteredBuffer::new(&allocation, u64::MAX - 3, 4).unwrap()];
    let last = at(u64::MAX)
        .resolve_buffer(&last_bytes, 1)
        .unwrap()
        .unwrap();
    assert_eq!((last.offset, last.available_bytes), (3, 1));
    for (address, length) in [
        (0, 4),
        (0x1000, 4),
        (BASE - 1, 4),
        (BASE + 0x1000, 0),
        (u64::MAX - 3, 8),
    ] {
        assert!(matches!(
            RegisteredBuffer::new(&allocation, address, length),
            Err(AddressError::InvalidAllocation { .. })
        ));
    }
}

#[test]
fn a_malformed_later_record_prevents_exposing_earlier_commands() {
    let mut bytes = vec![0; COMMAND_BUFFER_HEADER_BYTES + 2 * COMMAND_BYTES];
    CommandBuffer::initialize(&mut bytes, 2, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
    let first = CommandRecord {
        action: Action::DispatchThreadgroups {
            grid: [1; 3],
            local_size: [1; 3],
        },
        ..Default::default()
    };
    first
        .encode_into(
            &mut bytes[COMMAND_BUFFER_HEADER_BYTES..COMMAND_BUFFER_HEADER_BYTES + COMMAND_BYTES],
        )
        .unwrap();
    let second = (COMMAND_BUFFER_HEADER_BYTES + COMMAND_BYTES) / 4;
    for (word, value, error) in [
        (0, 99, RecordError::Action { value: 99 }),
        (1, 2, RecordError::Barrier { value: 2 }),
    ] {
        put_word(&mut bytes, second + word, value);
        assert!(
            matches!(CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE), Err(CommandBufferError::Record { index: 1, error: actual }) if actual == error)
        );
        put_word(&mut bytes, second + word, 0);
    }
    let view = CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
    assert_eq!(
        view.records(0..2).unwrap().collect::<Vec<_>>(),
        [first, CommandRecord::default()]
    );
}

#[test]
fn command_execution_ranges_select_exact_records_and_reject_invalid_bounds() {
    let mut bytes = vec![0; COMMAND_BUFFER_HEADER_BYTES + 3 * COMMAND_BYTES];
    CommandBuffer::initialize(&mut bytes, 3, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
    for index in 0..3 {
        let record = CommandRecord {
            action: Action::DispatchThreads {
                grid: [index as u32 + 1, 1, 1],
                local_size: [1; 3],
            },
            ..CommandRecord::default()
        };
        let start = COMMAND_BUFFER_HEADER_BYTES + index * COMMAND_BYTES;
        record
            .encode_into(&mut bytes[start..start + COMMAND_BYTES])
            .unwrap();
    }
    let view = CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
    for start in 0..=3 {
        for end in start..=3 {
            let records = view.records(start..end).unwrap();
            assert_eq!(records.len(), (end - start) as usize);
            for (index, record) in (start..end).zip(records) {
                assert_eq!(
                    record.action,
                    Action::DispatchThreads {
                        grid: [index + 1, 1, 1],
                        local_size: [1; 3],
                    }
                );
            }
        }
    }
    for (start, end) in [(2, 1), (0, 4), (4, 4), (0, u32::MAX)] {
        assert!(matches!(
            view.records(start..end),
            Err(CommandBufferError::Range { capacity: 3, .. })
        ));
    }
}

#[test]
fn detached_policy_headers_validate_without_allocating_command_records() {
    for capacity in [0, 1, u32::MAX] {
        let mut header = TEST_DESCRIPTOR
            .encode_header(capacity, OWNED_STATE)
            .unwrap();
        assert_eq!(header.len(), COMMAND_BUFFER_HEADER_BYTES);
        TEST_DESCRIPTOR
            .validate_header(&header, capacity, OWNED_STATE)
            .unwrap();
        assert!(matches!(
            TEST_DESCRIPTOR.validate_header(&header, capacity ^ 1, OWNED_STATE),
            Err(CommandBufferError::HeaderPolicyMismatch)
        ));
        let inherited = Inheritance {
            pipeline: StateSource::Parent,
            ..OWNED_STATE
        };
        assert!(matches!(
            TEST_DESCRIPTOR.validate_header(&header, capacity, inherited),
            Err(CommandBufferError::HeaderPolicyMismatch)
        ));
        put_word(&mut header, 1, 7);
        assert_eq!(
            TEST_DESCRIPTOR.validate_header(&header, capacity, OWNED_STATE),
            Err(CommandBufferError::EncodingFault(7))
        );
        assert_eq!(
            TEST_DESCRIPTOR.validate_header(&header[..15], capacity, OWNED_STATE),
            Err(CommandBufferError::Length)
        );
    }
}

#[test]
fn command_buffer_initialization_and_fault_checks_bound_the_record_view() {
    for capacity in [0, 1, 3] {
        let mut bytes = vec![0xa5; COMMAND_BUFFER_HEADER_BYTES + capacity as usize * COMMAND_BYTES];
        CommandBuffer::initialize(&mut bytes, capacity, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
        let view = CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
        assert_eq!(view.records(0..capacity).unwrap().len(), capacity as usize);
        assert!(view
            .records(0..capacity)
            .unwrap()
            .all(|record| record == CommandRecord::default()));
        put_word(&mut bytes, 1, 1);
        assert!(matches!(
            CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE),
            Err(CommandBufferError::EncodingFault(1))
        ));
        CommandBuffer::initialize(&mut bytes, capacity, &TEST_DESCRIPTOR, OWNED_STATE).unwrap();
        put_word(&mut bytes, 2, 1);
        assert!(matches!(
            CommandBuffer::decode(&bytes, &TEST_DESCRIPTOR, OWNED_STATE),
            Err(CommandBufferError::HeaderPolicyMismatch)
        ));
    }
    let mut truncated = vec![0xa5; COMMAND_BUFFER_HEADER_BYTES + COMMAND_BYTES - 1];
    assert_eq!(
        CommandBuffer::initialize(&mut truncated, 1, &TEST_DESCRIPTOR, OWNED_STATE),
        Err(CommandBufferError::Length)
    );
    assert!(truncated.iter().all(|&byte| byte == 0xa5));
    assert!(matches!(
        CommandBuffer::decode(&[], &TEST_DESCRIPTOR, OWNED_STATE),
        Err(CommandBufferError::Length)
    ));
    let mut header = [0; COMMAND_BUFFER_HEADER_BYTES];
    put_word(&mut header, 0, u32::MAX);
    assert!(matches!(
        CommandBuffer::decode(&header, &TEST_DESCRIPTOR, OWNED_STATE),
        Err(CommandBufferError::Length)
    ));
}
