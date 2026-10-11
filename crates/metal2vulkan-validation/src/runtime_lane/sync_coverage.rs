//! What the validation layer's synchronization validation sees of the runtime's commands: positive
//! controls. Each test runs a program whose two accesses of one untracked buffer nothing orders (no
//! fence, event or barrier: Metal leaves their order to the program, a data race whose bytes these
//! tests never read), on a device whose layer checks synchronization as the lane's `sync` switch
//! has it ([`LaneChecks`]), and states whether the layer reported the hazard. A clean lane run
//! under `METAL2VULKAN_LANE_VALIDATION=sync` is evidence only for the accesses the layer sees;
//! these say which those are.
//!
//! The races are harmless to the device: both accesses write or read the same few bytes.

use super::{
    buffer_with, buffer_with_options, encode_case, pipeline_state, public_case, read_buffer,
    validated_device_with, CaseResources, LaneChecks, Messages, PublicCase, ScratchCache,
};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::DispatchType;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CpuCacheMode, DebugMessage, Device, HazardTrackingMode, ResourceOptions,
    Size, StorageMode,
};
use std::collections::BTreeMap;
use std::time::Duration;

/// The longest a command buffer of these tests may take.
const WAIT: Duration = Duration::from_secs(60);

/// The committed case the controls dispatch: `output[0] = input[0]`, `input` at buffer 0 and
/// `output` at buffer 1.
const COPY_WORD: &str = "single-thread-copy-word";

fn untracked() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Shared,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Untracked,
    )
}

/// The messages the layer reported for `test`, run on a device whose layer checks
/// synchronization, once the device is gone.
fn reported(test: impl FnOnce(&Device)) -> Vec<DebugMessage> {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            LaneChecks {
                synchronization: true,
                ..LaneChecks::default()
            }
            .apply(options)
        })
        .unwrap();
        test(&device);
    }
    messages.take()
}

/// The hazards among `messages`, printed with the test's name for the lane's log.
fn hazards(test: &str, messages: &[DebugMessage]) -> Vec<String> {
    for message in messages {
        eprintln!(
            "SYNC-COVERAGE {test}: {} {}",
            message.id_name,
            message.text.replace('\n', " ")
        );
    }
    let found: Vec<String> = messages
        .iter()
        .filter(|message| message.id_name.starts_with("SYNC-HAZARD"))
        .map(|message| message.id_name.clone())
        .collect();
    eprintln!("SYNC-COVERAGE {test}: hazards {found:?}");
    found
}

fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// The copy-word case's buffers, untracked.
fn case_buffers(device: &Device, case: &PublicCase) -> BTreeMap<u32, Buffer> {
    case.compute
        .buffers
        .iter()
        .map(|(&index, bytes)| {
            (
                index,
                buffer_with_options(device, bytes, untracked()).unwrap(),
            )
        })
        .collect()
}

/// Two blit encoders of one command buffer fill one untracked buffer: the layer sees transfer
/// writes, and reports the write after write.
#[test]
fn fills_of_one_untracked_buffer_in_two_encoders() {
    let messages = reported(|device| {
        let queue = device.new_command_queue().unwrap();
        let buffer = device.new_buffer(256, untracked()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        for value in [1, 2] {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..256, value).unwrap();
            blit.end_encoding().unwrap();
        }
        run(&command_buffer);
    });
    let found = hazards("fills_of_one_untracked_buffer_in_two_encoders", &messages);
    assert!(!found.is_empty(), "no hazard was reported: {messages:#?}");
}

/// Two compute encoders of one command buffer dispatch a kernel that writes one untracked buffer
/// at a bound slot: whether the layer sees a shader's write of a bound buffer.
#[test]
fn shader_writes_of_one_untracked_buffer_in_two_encoders() {
    let messages = reported(|device| {
        let case = public_case(COPY_WORD).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let buffers = case_buffers(device, &case);
        let command_buffer = queue.command_buffer().unwrap();
        for _ in 0..2 {
            let mut compute = command_buffer.compute_command_encoder().unwrap();
            encode_case(
                &mut compute,
                &pipeline,
                &case.compute,
                &buffers,
                &CaseResources::default(),
            )
            .unwrap();
            compute.end_encoding().unwrap();
        }
        run(&command_buffer);
    });
    let found = hazards(
        "shader_writes_of_one_untracked_buffer_in_two_encoders",
        &messages,
    );
    assert!(!found.is_empty(), "no hazard was reported: {messages:#?}");
}

/// A blit encoder fills an untracked buffer a later compute encoder's kernel reads at a bound
/// slot: whether the layer sees a shader's read of a bound buffer after a transfer write.
#[test]
fn a_fill_then_a_shader_read_of_one_untracked_buffer() {
    let messages = reported(|device| {
        let case = public_case(COPY_WORD).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let buffers = case_buffers(device, &case);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let input = &buffers[&0];
        blit.fill_buffer(input, 0..input.length(), 7).unwrap();
        blit.end_encoding().unwrap();
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        encode_case(
            &mut compute,
            &pipeline,
            &case.compute,
            &buffers,
            &CaseResources::default(),
        )
        .unwrap();
        compute.end_encoding().unwrap();
        run(&command_buffer);
    });
    let found = hazards(
        "a_fill_then_a_shader_read_of_one_untracked_buffer",
        &messages,
    );
    assert!(!found.is_empty(), "no hazard was reported: {messages:#?}");
}

/// A compute encoder's kernel writes an untracked buffer at a bound slot that a later blit
/// encoder copies: whether the layer sees a transfer read after a shader's write.
#[test]
fn a_shader_write_then_a_copy_of_one_untracked_buffer() {
    let messages = reported(|device| {
        let case = public_case(COPY_WORD).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let buffers = case_buffers(device, &case);
        let copy = device
            .new_buffer(buffers[&1].length(), untracked())
            .unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        encode_case(
            &mut compute,
            &pipeline,
            &case.compute,
            &buffers,
            &CaseResources::default(),
        )
        .unwrap();
        compute.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&buffers[&1], 0, &copy, 0, copy.length())
            .unwrap();
        blit.end_encoding().unwrap();
        run(&command_buffer);
    });
    let found = hazards(
        "a_shader_write_then_a_copy_of_one_untracked_buffer",
        &messages,
    );
    assert!(!found.is_empty(), "no hazard was reported: {messages:#?}");
}

/// One concurrent compute encoder dispatches the kernel twice with no memory barrier between:
/// the first writes an untracked buffer at a bound slot that the second reads at another. Metal
/// orders neither dispatch after the other, and the runtime records no barrier between them (the
/// encoder leaves that order to the program): whether the layer sees, inside one command buffer,
/// a shader's read of a bound buffer after another shader's write.
///
/// A read after the write, not two writes: the layer of 1.4.341 (Ubuntu 26.04's, the x86
/// lavapipe venue's) suppresses every write-after-write hazard of a bound descriptor found while
/// recording (`SuppressedBoundDescriptorWAW`), which later versions report; across command
/// buffers, at submission, it reports them.
#[test]
fn a_shader_write_then_a_shader_read_of_one_untracked_buffer_in_one_concurrent_encoder() {
    let messages = reported(|device| {
        let case = public_case(COPY_WORD).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let first = case_buffers(device, &case);
        // The second dispatch copies the word the first wrote into a buffer of its own.
        let written = &first[&1];
        let second = BTreeMap::from([
            (0, written.clone()),
            (1, device.new_buffer(written.length(), untracked()).unwrap()),
        ]);
        let command_buffer = queue.command_buffer().unwrap();
        let mut compute = command_buffer
            .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)
            .unwrap();
        for buffers in [&first, &second] {
            encode_case(
                &mut compute,
                &pipeline,
                &case.compute,
                buffers,
                &CaseResources::default(),
            )
            .unwrap();
        }
        compute.end_encoding().unwrap();
        run(&command_buffer);
    });
    let found = hazards(
        "a_shader_write_then_a_shader_read_of_one_untracked_buffer_in_one_concurrent_encoder",
        &messages,
    );
    assert!(!found.is_empty(), "no hazard was reported: {messages:#?}");
}

/// A layer false positive, byte by byte: a descriptor's range broader than its command's access
/// (the lanes' `argument_buffers` and `render_executions` reports). One serial compute encoder
/// of tracked buffers dispatches the copy-word kernel three times over one 16-byte `block`: the
/// first writes its word 0, the second its word 1, the third reads its word 2. The runtime's
/// hazard tracking orders a dispatch by the plan's windows, here 4 bytes each and disjoint, so it
/// records no barrier between them; the descriptor each binds covers the whole block (a bounded
/// window rounds out to 16 bytes, and no range is shorter: `bind::descriptor_range`), which the
/// layer takes as accessed. So the layer reports a hazard (the read after the writes; on layers
/// after 1.4.341 the write after write too) that the bytes show is none: the block ends with the
/// two copied words and its other words as written, and the third dispatch read word 2 as
/// written.
#[test]
fn disjoint_words_of_one_block_are_a_hazard_to_the_layer_alone() {
    const VALUES: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    const BLOCK: [u8; 16] = [
        0xa0, 0xa1, 0xa2, 0xa3, 0xb0, 0xb1, 0xb2, 0xb3, 0xc0, 0xc1, 0xc2, 0xc3, 0xd0, 0xd1, 0xd2,
        0xd3,
    ];
    let mut bytes = None;
    let messages = reported(|device| {
        let case = public_case(COPY_WORD).unwrap();
        let pipeline = pipeline_state(device, &case).unwrap();
        let queue = device.new_command_queue().unwrap();
        let values = buffer_with(device, &VALUES).unwrap();
        let block = buffer_with(device, &BLOCK).unwrap();
        let read = buffer_with(device, &[0; 4]).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut compute = command_buffer.compute_command_encoder().unwrap();
        compute.set_compute_pipeline_state(&pipeline).unwrap();
        let one = Size::new(1, 1, 1);
        // Words 0 and 1 of `values` into words 0 and 1 of `block`, then word 2 of `block` out.
        for (input, input_offset, output, output_offset) in [
            (&values, 0, &block, 0),
            (&values, 4, &block, 4),
            (&block, 8, &read, 0),
        ] {
            compute.set_buffer(Some(input), input_offset, 0).unwrap();
            compute.set_buffer(Some(output), output_offset, 1).unwrap();
            compute.dispatch_threads(one, one).unwrap();
        }
        compute.end_encoding().unwrap();
        run(&command_buffer);
        bytes = Some((
            read_buffer(&block, 0, 16).unwrap(),
            read_buffer(&read, 0, 4).unwrap(),
        ));
    });
    let found = hazards(
        "disjoint_words_of_one_block_are_a_hazard_to_the_layer_alone",
        &messages,
    );
    let (block, read) = bytes.unwrap();
    let mut expected = BLOCK;
    expected[..8].copy_from_slice(&VALUES);
    assert_eq!(block, expected, "each write reached its own word alone");
    assert_eq!(read, BLOCK[8..12], "the read found word 2 as written");
    assert!(
        found.iter().any(|id| id == "SYNC-HAZARD-READ-AFTER-WRITE"),
        "the layer reported no read after write: {messages:#?}"
    );
}
