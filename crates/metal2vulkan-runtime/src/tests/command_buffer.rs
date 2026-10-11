//! A command buffer committed as failed (`command_buffer/mod.rs`, `CommandBuffer::commit_failed`):
//! which encoders a failure fails, and which it refuses.

use crate::command_buffer::{CommitFailure, FailedEncoders, FailedSignals, Failure};
use crate::error::{CommandBufferError, Problem};
use std::collections::BTreeSet;

/// The encoders among `count` that `failure` fails.
fn failed(failure: &FailedEncoders, count: usize) -> Vec<usize> {
    (0..count).filter(|&at| failure.fails(at)).collect()
}

#[test]
fn a_new_failure_fails_every_encoder_and_keeps_the_signals() {
    let failure = CommitFailure::new(8);
    assert_eq!(failure.code, 8);
    assert!(failure.signals);
    assert_eq!(failure.encoders, FailedEncoders::From(0));
    assert_eq!(failed(&failure.encoders, 4), [0, 1, 2, 3]);
    assert_eq!(failure.problem(0), None);
    assert_eq!(failure.problem(4), None);
}

#[test]
fn a_suffix_fails_its_first_encoder_and_every_one_after() {
    assert_eq!(failed(&FailedEncoders::From(2), 5), [2, 3, 4]);
    // One past the ended encoders fails none, and is no refusal.
    let mut failure = CommitFailure::new(2);
    failure.encoders = FailedEncoders::From(5);
    assert_eq!(failed(&failure.encoders, 5), Vec::<usize>::new());
    assert_eq!(failure.problem(5), None);
}

/// The probe shapes' sets: the encoders between and around them run.
#[test]
fn a_set_fails_its_encoders_alone() {
    let set = |encoders: &[usize]| FailedEncoders::Set(encoders.iter().copied().collect());
    assert_eq!(failed(&set(&[1]), 3), [1]);
    assert_eq!(failed(&set(&[0, 1]), 3), [0, 1]);
    assert_eq!(failed(&set(&[0, 2, 4]), 5), [0, 2, 4]);
    assert_eq!(failed(&set(&[]), 3), Vec::<usize>::new());
}

/// A set naming an encoder past the ended ones is refused, naming the first such and the count.
#[test]
fn a_set_past_the_ended_encoders_is_refused() {
    let mut failure = CommitFailure::new(8);
    failure.encoders = FailedEncoders::Set(BTreeSet::from([1, 3, 7]));
    assert_eq!(failure.problem(8), None);
    assert_eq!(
        failure.problem(3),
        Some(Problem::IndexOutOfRange { index: 3, limit: 3 })
    );
    assert_eq!(
        failure.problem(0),
        Some(Problem::IndexOutOfRange { index: 1, limit: 0 })
    );
    failure.encoders = FailedEncoders::Set(BTreeSet::from([usize::MAX]));
    assert_eq!(
        failure.problem(2),
        Some(Problem::IndexOutOfRange {
            index: u32::MAX,
            limit: 2
        })
    );
}

/// A command buffer Metal fails from an encoder on (a render pass without attachments past 16384
/// a side) fails that encoder and every one after it, signals only what was encoded before it,
/// and completes with `RenderTargetSize`; with `commit_failed` too, the encoders either fails,
/// with the program's code.
#[test]
fn metal_fails_a_command_buffer_from_an_encoder_on() {
    assert_eq!(Failure::of(None, None, 3), None);
    assert_eq!(
        Failure::of(None, Some((1, CommandBufferError::RenderTargetSize)), 3),
        Some(Failure {
            encoders: FailedEncoders::From(1),
            signals: FailedSignals::From(1),
            error: Some(CommandBufferError::RenderTargetSize),
        })
    );
    // A pass at a default sample count Metal fails the command buffer at fails it alike, with
    // its own error.
    assert_eq!(
        Failure::of(
            None,
            Some((1, CommandBufferError::DefaultRasterSampleCount)),
            3
        )
        .and_then(|f| f.error),
        Some(CommandBufferError::DefaultRasterSampleCount)
    );
    assert_eq!(
        Failure::of(Some(CommitFailure::new(8)), None, 3),
        Some(Failure {
            encoders: FailedEncoders::From(0),
            signals: FailedSignals::Every,
            error: Some(CommandBufferError::Injected(8)),
        })
    );
    let mut later = CommitFailure::new(2);
    later.encoders = FailedEncoders::From(2);
    assert_eq!(
        Failure::of(
            Some(later),
            Some((1, CommandBufferError::RenderTargetSize)),
            3
        )
        .map(|f| (f.encoders, f.signals)),
        Some((FailedEncoders::From(1), FailedSignals::From(1)))
    );
    let mut set = CommitFailure::new(16);
    set.encoders = FailedEncoders::Set(BTreeSet::from([0]));
    set.signals = false;
    assert_eq!(
        Failure::of(
            Some(set),
            Some((2, CommandBufferError::RenderTargetSize)),
            4
        ),
        Some(Failure {
            encoders: FailedEncoders::Set(BTreeSet::from([0, 2, 3])),
            signals: FailedSignals::Dropped,
            error: Some(CommandBufferError::Injected(16)),
        })
    );
}

/// Metal's submissions of encoders of the probe shapes' kinds with these event counts before
/// each (`checks::metal_submissions`): each render pass alone, runs of one kind of blits or
/// computes, split where an event wait or signal lies between two encoders.
/// Encoders Metal's GPU faults run nothing: alone, the command buffer completes with `PageFault`
/// and every signal fires; beside another failure, that failure's error and signals stand, and
/// the encoders either fails run nothing.
#[test]
fn encoders_metal_s_gpu_faults_run_nothing() {
    let faulted = BTreeSet::from([1, 3]);
    assert_eq!(Failure::with_faults(None, &BTreeSet::new(), 4), None);
    assert_eq!(
        Failure::with_faults(None, &faulted, 4),
        Some(Failure {
            encoders: FailedEncoders::Set(BTreeSet::from([1, 3])),
            signals: FailedSignals::Every,
            error: Some(CommandBufferError::PageFault),
        })
    );
    assert_eq!(
        Failure::with_faults(
            Failure::of(None, Some((2, CommandBufferError::RenderTargetSize)), 4),
            &faulted,
            4
        ),
        Some(Failure {
            encoders: FailedEncoders::Set(BTreeSet::from([1, 2, 3])),
            signals: FailedSignals::From(2),
            error: Some(CommandBufferError::RenderTargetSize),
        })
    );
    assert_eq!(
        Failure::with_faults(
            Failure::of(None, Some((1, CommandBufferError::RenderTargetSize)), 4),
            &faulted,
            4
        )
        .map(|f| f.encoders),
        Some(FailedEncoders::From(1))
    );
    let mut set = CommitFailure::new(16);
    set.encoders = FailedEncoders::Set(BTreeSet::from([0]));
    assert_eq!(
        Failure::with_faults(Failure::of(Some(set), None, 4), &faulted, 4)
            .map(|f| (f.encoders, f.error)),
        Some((
            FailedEncoders::Set(BTreeSet::from([0, 1, 3])),
            Some(CommandBufferError::Injected(16))
        ))
    );
    assert_eq!(CommandBufferError::PageFault.metal_code(), 3);
    // A code of 0 is no error; a faulted encoder it lets run faults the command buffer, and
    // Metal's own failure from a render pass keeps its error.
    assert_eq!(
        Failure::of(Some(CommitFailure::new(0)), None, 3).map(|f| (f.encoders, f.error)),
        Some((FailedEncoders::From(0), None))
    );
    let mut first = CommitFailure::new(0);
    first.encoders = FailedEncoders::Set(BTreeSet::from([0]));
    assert_eq!(
        Failure::with_faults(Failure::of(Some(first), None, 4), &faulted, 4).and_then(|f| f.error),
        Some(CommandBufferError::PageFault)
    );
    let mut covering = CommitFailure::new(0);
    covering.encoders = FailedEncoders::Set(BTreeSet::from([1, 3]));
    assert_eq!(
        Failure::with_faults(Failure::of(Some(covering), None, 4), &faulted, 4).map(|f| f.error),
        Some(None)
    );
    assert_eq!(
        Failure::of(
            Some(CommitFailure::new(0)),
            Some((2, CommandBufferError::RenderTargetSize)),
            3
        )
        .and_then(|f| f.error),
        Some(CommandBufferError::RenderTargetSize)
    );
}

/// Every command buffer error says whose failure it is: Metal's, whose `metal_code` is the code
/// Metal's command buffer carries for such work, or the runtime's, whose `metal_code` is the
/// nearest of Metal's codes where Metal fails no command buffer (the codes are unchanged).
#[test]
fn a_command_buffer_error_names_whose_failure_it_is() {
    use crate::error::{
        FailureOrigin::{Metal, Runtime},
        HostStepObject, MemoryKind, MeshRecordsOverflow, PipelineCreationCause, VkResultCode,
    };
    use crate::sparse::SparseProblem;
    use CommandBufferError as E;
    let overflow = MeshRecordsOverflow {
        draw: 0,
        object_threadgroups: 1,
        mesh_threadgroups: 1,
        budget: 0,
    };
    for (error, origin, code) in [
        (E::Injected(16), Metal, 16),
        (E::DeviceLost, Metal, 11),
        (E::OutOfMemory, Metal, 8),
        (E::PageFault, Metal, 3),
        (E::Execution(1), Metal, 1),
        (E::Execution(12), Metal, 12),
        (E::RenderTargetSize, Metal, 1),
        (E::DefaultRasterSampleCount, Metal, 1),
        (E::DetachedBacking, Metal, 8),
        (E::Internal(VkResultCode(-13)), Runtime, 1),
        (E::IndexPoolOverflow, Runtime, 8),
        (E::PredicatedLoopPastCopies, Runtime, 1),
        (E::SparseHeapSplit, Runtime, 1),
        (
            E::SparseMappingArguments(SparseProblem::Lengths),
            Runtime,
            1,
        ),
        (E::SparseMappingSplit, Runtime, 1),
        (E::CubicArgumentSampler, Runtime, 1),
        (E::UnlistedArgumentSampler, Runtime, 1),
        (E::MeshRecordsOverflow(overflow), Runtime, 8),
        (E::Refused, Runtime, 1),
        (
            E::PipelineCreation(PipelineCreationCause::OutOfMemory(MemoryKind::Device)),
            Runtime,
            8,
        ),
        (
            E::PipelineCreation(PipelineCreationCause::DeviceLost),
            Runtime,
            11,
        ),
        (
            E::PipelineCreation(PipelineCreationCause::Internal),
            Runtime,
            1,
        ),
        (E::HostStepObject(HostStepObject::Image), Runtime, 1),
        (E::EncodedUseRebound, Runtime, 1),
    ] {
        assert_eq!(
            (error.origin(), error.metal_code()),
            (origin, code),
            "{error:?}"
        );
    }
}

#[test]
fn metal_submissions_follow_the_consumer_s_measurements() {
    use crate::command_buffer::checks::metal_submissions;
    use crate::EncoderKind::{Blit, Compute, Render, ResourceState};
    let split = |encoders: &[(crate::EncoderKind, usize, usize)]| {
        metal_submissions(encoders.iter().copied())
            .into_iter()
            .map(|(_, range)| (range.start, range.end))
            .collect::<Vec<_>>()
    };
    // oCo, cdc, rRr, ocdco: every encoder its own.
    assert_eq!(
        split(&[(Blit, 0, 0), (Compute, 0, 0), (Blit, 0, 0)]),
        [(0, 1), (1, 2), (2, 3)]
    );
    assert_eq!(
        split(&[(Render, 0, 0), (Render, 0, 0), (Render, 0, 0)]),
        [(0, 1), (1, 2), (2, 3)]
    );
    assert_eq!(
        split(&[
            (Blit, 0, 0),
            (Compute, 0, 0),
            (Blit, 0, 0),
            (Compute, 0, 0),
            (Blit, 0, 0)
        ]),
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    );
    // odo: one run of blits; odso and odwo: split at the signal and at the wait.
    assert_eq!(split(&[(Blit, 0, 0), (Blit, 0, 0), (Blit, 0, 0)]), [(0, 3)]);
    assert_eq!(
        split(&[(Blit, 0, 0), (Blit, 0, 0), (Blit, 0, 1)]),
        [(0, 2), (2, 3)]
    );
    assert_eq!(
        split(&[(Blit, 1, 0), (Blit, 1, 0), (Blit, 2, 0)]),
        [(0, 2), (2, 3)]
    );
    // Consecutive computes share one; resource state encoders join their own runs.
    assert_eq!(
        split(&[
            (Compute, 0, 0),
            (Compute, 0, 0),
            (ResourceState, 0, 0),
            (ResourceState, 0, 0)
        ]),
        [(0, 2), (2, 4)]
    );
    let kinds: Vec<_> = metal_submissions([(Render, 0, 0), (Blit, 0, 0)])
        .into_iter()
        .map(|(kind, _)| kind)
        .collect();
    assert_eq!(kinds, [Render, Blit]);
    assert!(metal_submissions([]).is_empty());
}
