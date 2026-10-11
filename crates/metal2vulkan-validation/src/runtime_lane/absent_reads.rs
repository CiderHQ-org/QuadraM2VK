//! Reads of an argument a function constant removes, on the device the cfg selects, under the
//! validation layer.
//!
//! The owned fixture `engine/rfs_absent_reads`, with `has_source` false: `rfs_absent_query`
//! writes `7 + i`, or the width of the removed texture `source` where its run-time flag is
//! nonzero; `rfs_absent_read` writes `(i, 0.5, 0.25, 1)`, or texel `(i, 0)` of `source`;
//! `rfs_absent_sampler` writes texel `(i, 0)` of the present `tex`, or a sample of it through the
//! removed sampler `smp`. Metal gives each such read no value. Making the three pipeline states
//! announces [`Degradation::AbsentArgumentRead`] once; a command buffer whose flag is 0 completes
//! with Metal's answer (its authored cases' Metal rows), and one whose flag is 1 runs the read,
//! which takes zero (Metal's answer under its debug layer), and completes too, its execution
//! report stating the read (it failed before). A device that does not address buffers refuses the
//! states, typed (`FunctionConstantArgument`, needing the addresses).

use super::degrade_notices::{assert_once, counts, on_device};
use super::shared_kinds;
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::error::{Feature, Needs};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{DataType, PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferError, CommandBufferStatus, ComputePipelineState, Degradation, Device, Error,
    PipelineOptions, Region, ResourceOptions, Size, Texture, TextureDescriptor,
};

/// What one run of a kernel left: the command buffer's status and error, whether its execution
/// report states a removed argument's read, and the words of `out`.
type Ran = (
    CommandBufferStatus,
    Option<CommandBufferError>,
    Option<bool>,
    Vec<u32>,
);
use std::time::Duration;

/// The kind making the states takes.
pub(super) const ABSENT_READS: [Degradation; 1] = [Degradation::AbsentArgumentRead];

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// Threads per dispatch, one threadgroup.
const THREADS: u64 = 4;

/// The state of `name` in the fixture with `has_source` false, or the device's typed refusal.
fn state(device: &Device, name: &str) -> Result<ComputePipelineState, Error> {
    let bitcode = llvm_assemble(&engine_dir().join("rfs_absent_reads.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let constants = FunctionConstantValues::defaults()
        .with_index(0, FunctionConstantValue::new(DataType::Bool, [0]));
    let function = library
        .new_function_with_name_constant_values(name, &constants)
        .unwrap();
    device.new_compute_pipeline_state(
        &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
        &PipelineOptions::default(),
    )
}

/// `tex`: 4x1 `RGBA32Float`, texel `i` = `(i + 1) * (1, 2, 3, 4)`.
fn texels() -> Vec<f32> {
    (0..THREADS)
        .flat_map(|i| (1..=4).map(move |c| ((i + 1) * c) as f32))
        .collect()
}

fn texture(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, THREADS, 1, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let bytes: Vec<u8> = texels().iter().flat_map(|f| f.to_le_bytes()).collect();
    texture
        .replace_region(
            Region::new_2d(0, 0, THREADS, 1),
            0,
            0,
            &bytes,
            16 * THREADS,
            0,
        )
        .unwrap();
    texture
}

/// One dispatch of `state` over [`THREADS`] threads with `flag` at buffer 1 and `tex` at texture
/// 0: the command buffer's status and error, whether its execution report states a removed
/// argument's read, and the words of the 16-byte-per-thread `out` at buffer 0, which starts
/// `0xab` in every byte.
fn run(device: &Device, state: &ComputePipelineState, flag: u32, tex: Option<&Texture>) -> Ran {
    let bytes = 16 * THREADS;
    let out = buffer_with(device, &vec![0xab; bytes as usize]).unwrap();
    let flag = buffer_with(device, &flag.to_le_bytes()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    if let Some(tex) = tex {
        encoder.set_texture(Some(tex), 0).unwrap();
    }
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&flag), 0, 1).unwrap();
    let grid = Size::new(THREADS, 1, 1);
    encoder.dispatch_threads(grid, grid).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let words = read_buffer(&out, 0, bytes)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    let absent = command_buffer
        .execution_report()
        .map(|report| report.absent_argument_read);
    (
        command_buffer.status(),
        command_buffer.error(),
        absent,
        words,
    )
}

/// The three kernels' states are announced once; flag 0 completes with what each writes without
/// the read, its report stating no such read, and flag 1 completes too, each read zero (the
/// width 0, the texel and the sample `(0, 0, 0, 0)`), its report stating the read: the path ran.
/// Over the whole test every kind is announced at most once: the runs may announce a kind of
/// their own for the first time (on a device without unified memory, the `Shared` buffers and
/// texture they make in system memory, `SharedInSystemMemory`), never one announced before.
#[test]
fn a_read_of_a_removed_argument_reads_zero_and_its_command_buffer_completes() {
    on_device(|device, messages| {
        let names = ["rfs_absent_query", "rfs_absent_read", "rfs_absent_sampler"];
        let mut states = Vec::new();
        for name in names {
            match state(device, name) {
                Ok(state) => states.push(state),
                Err(Error::Unsupported(u))
                    if u.feature == Feature::FunctionConstantArgument
                        && matches!(u.needs, Needs::AnyOf(_)) =>
                {
                    println!(
                        "NO-DEVICE-ADDRESSES {}: {name} is refused, typed: {u:?}",
                        device.name()
                    );
                    return;
                }
                Err(error) => panic!("{name}: {error}"),
            }
        }
        let mut notices = messages.take_notices();
        assert_once(&notices, &ABSENT_READS);
        let tex = texture(device);
        let as_words = |floats: &[f32]| floats.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let query: Vec<u32> = (0..THREADS as u32).map(|i| 7 + i).collect();
        let read: Vec<f32> = (0..THREADS)
            .flat_map(|i| [i as f32, 0.5, 0.25, 1.0])
            .collect();
        let expected = [
            (None, query),
            (None, as_words(&read)),
            (Some(&tex), as_words(&texels())),
        ];
        for ((name, state), (tex, want)) in names.iter().zip(&states).zip(expected) {
            let (status, error, absent, words) = run(device, state, 0, tex);
            assert_eq!(status, CommandBufferStatus::Completed, "{name}: {error:?}");
            assert_eq!(error, None, "{name}");
            assert_eq!(absent, Some(false), "{name}: flag 0");
            assert_eq!(&words[..want.len()], &want[..], "{name}: flag 0");
            let (status, error, absent, words) = run(device, state, 1, tex);
            assert_eq!(
                status,
                CommandBufferStatus::Completed,
                "{name}: flag 1: {error:?}"
            );
            assert_eq!(error, None, "{name}: flag 1");
            assert_eq!(absent, Some(true), "{name}: flag 1");
            assert_eq!(
                &words[..want.len()],
                &vec![0; want.len()][..],
                "{name}: flag 1"
            );
            println!("ABSENT-READ-COMPLETED {}: {name}", device.name());
        }
        notices.extend(messages.take_notices());
        println!("ANNOUNCED {}: {:?}", device.name(), counts(&notices));
        let mut kinds = ABSENT_READS.to_vec();
        kinds.extend(shared_kinds(device));
        assert_once(&notices, &kinds);
    });
}
