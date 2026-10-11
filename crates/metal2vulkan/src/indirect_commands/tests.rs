use super::*;

const OWNED_STATE: Inheritance = Inheritance {
    pipeline: StateSource::Command,
    buffers: StateSource::Command,
};

const TEST_DESCRIPTOR: CommandDescriptor = CommandDescriptor::Compute {
    dispatch: ComputeDispatch::Both,
    max_buffer_bind_count: 32,
    max_threadgroup_memory_bind_count: 31,
    support_dynamic_attribute_stride: true,
};

mod buffer_views;
mod inheritance;
mod records;

/// Each resource ID's records lie at `2^63 + id * 2^40`, a range of 2^40 bytes of their own: ID 1's
/// starts at `2^63 + 2^40`, the last ID's range ends at 2^64, the null ID's lies below ID 1's, and
/// every one lies above every Metal address (below 2^63).
#[test]
fn each_records_id_has_a_range_of_its_own_above_the_metal_addresses() {
    let span = 1u64 << RECORDS_ADDRESS_SHIFT;
    assert_eq!(records_address(0), 1 << 63);
    assert_eq!(records_address(1), (1 << 63) + span);
    assert_eq!(records_address(2) - records_address(1), span);
    let last = RECORDS_ID_LIMIT - 1;
    assert_eq!(
        u128::from(records_address(last)) + u128::from(span),
        1u128 << 64
    );
    for id in [1, 2, 3, 65_536, 70_000, last] {
        assert!(records_address(id) > records_address(id - 1), "{id}");
        assert_eq!(
            (records_address(id) - RECORDS_ADDRESS_BASE) >> RECORDS_ADDRESS_SHIFT,
            id
        );
    }
}
