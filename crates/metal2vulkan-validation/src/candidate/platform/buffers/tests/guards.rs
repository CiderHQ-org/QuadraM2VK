use super::*;

/// A guarded buffer is bound past a front guard of at least its reach before it, its own bytes map
/// where the buffer is, every guard byte holds the guard fill and every byte of the memory bound to
/// it past its length the padding fill, and each changed guard and padding byte is counted apart;
/// an unguarded buffer reports no guard.
#[test]
fn a_guarded_buffer_counts_each_changed_guard_byte() {
    with_validated_context(crate::gpu::TEST_DEVICE, |context| {
        // Not a multiple of the buffer alignment, so the memory requirement may pad it.
        let bytes: Vec<u8> = (0..20).collect();
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let reach = GuardReach {
            before: 512,
            after: 1024,
        };
        let buffer = create_guarded_host_buffer(context, "guarded", &bytes, usage, reach)?;
        assert_eq!(unsafe { buffer.map()? }.bytes, &bytes[..]);
        let count = |escaped, past_end| Some(GuardCount { escaped, past_end });
        assert_eq!(unsafe { buffer.escaped()? }, count(0, 0));
        let guard = buffer.guard.expect("a guarded buffer has a guard");
        assert!(buffer.offset >= 512);
        assert!(guard.bound_end >= buffer.offset + 20);
        let end = (buffer.offset + buffer.len) as usize;
        assert!(guard.memory_len - guard.bound_end >= 1024);
        {
            let memory = unsafe { buffer.map_range(0, guard.memory_len)? };
            assert!(memory.bytes[..buffer.offset as usize]
                .iter()
                .chain(&memory.bytes[guard.bound_end as usize..])
                .all(|byte| *byte == GUARD_FILL));
            assert!(memory.bytes[end..guard.bound_end as usize]
                .iter()
                .all(|byte| *byte == PAD_FILL));
            memory.bytes[0] = 0;
            memory.bytes[guard.bound_end as usize + 3] = 1;
            // The buffer's own bytes are neither guard nor padding.
            memory.bytes[buffer.offset as usize] = 0xff;
            if end < guard.bound_end as usize {
                memory.bytes[end] = 0;
            }
        }
        let padded = u64::from(end < guard.bound_end as usize);
        assert_eq!(unsafe { buffer.escaped()? }, count(2, padded));
        let plain = create_host_buffer(context, "plain", &bytes, usage)?;
        assert_eq!(unsafe { plain.escaped()? }, None);
        Ok(())
    })
    .unwrap();
}
