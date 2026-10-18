//! `recorder/replays.rs`: a new recorder's queue has room for a frame's commits, and the recorder's
//! thread, taking the first replays queued, leaves the queue room for twice as many as waited, so
//! the commits after it queue theirs without growing it; a command buffer's replays, made new,
//! have room to end each deferred command buffer made for them.

use crate::loader_word::LoaderWord;
use crate::recorder::{queue_room, take_leaving_room, RecorderShared};
use crate::scheduler::inbox::ROOM;
use ash::vk;
use std::collections::VecDeque;

/// A new recorder's queue holds as many replays as the scheduler's inbox holds commits ([`ROOM`])
/// without growing: so a frame's commits, which wait at most that many, queue theirs with no
/// allocation on their own thread, in `commit`, however far the recorder's thread falls behind
/// before it first took one.
#[test]
fn a_new_recorder_queue_holds_room_replays_without_growing() {
    let room = queue_room();
    assert!(room >= ROOM, "a new recorder's queue has room for {room}");
}

/// Taking the first of 5 queued leaves the queue room for 10, made on the taking thread: so the
/// commits after it push theirs with no allocation on their own thread, in `commit`, until more
/// than twice as many wait. Grown by the pushes alone, it had room for 8 after 5 (Rust 1.98.1's
/// growth), and a commit's push past 8 allocated in `commit`.
#[test]
fn taking_the_first_queued_leaves_room_for_twice_as_many_as_waited() {
    let mut queue: VecDeque<u32> = (0..5).collect();
    assert_eq!(take_leaving_room(&mut queue), Some(0));
    let room = queue.capacity();
    assert!(
        room >= 10,
        "taking the first of 5 left the queue room for {room}"
    );
    queue.extend(5..11);
    assert_eq!(queue.len(), 10);
    assert_eq!(
        queue.capacity(),
        room,
        "the pushes up to twice as many as waited grew the queue"
    );
    assert_eq!(take_leaving_room(&mut VecDeque::<u32>::new()), None);
}

/// A command buffer's replays made new (none kept to take, as where the command buffers before it
/// have yet to complete) have room to end every deferred command buffer made for them, as each is
/// made: so the end of the last, which the commit makes (`CommandBufferShared::close_tail`),
/// allocates nothing in `commit`. Grown by the ends alone, the first end grew it, and a commit
/// whose command buffer ended nothing before it allocated once more than one taking kept replays
/// (`runtime_lane::frame_costs`, `a_frame_s_commits_allocate_within_their_bound`, on lavapipe
/// under load).
#[test]
fn new_replays_have_room_to_end_every_command_buffer_made_for_them() {
    let word = Box::leak(Box::new(0usize));
    // SAFETY: a word that lives as long as the process; no device's, and nothing restores it.
    let loader_word =
        unsafe { LoaderWord::of_handle(std::ptr::from_mut(word).expose_provenance() as u64) };
    let recorder = RecorderShared::new(0, loader_word);
    let replays = recorder.replays(0, None);
    for level in [
        vk::CommandBufferLevel::PRIMARY,
        vk::CommandBufferLevel::SECONDARY,
    ]
    .repeat(5)
    {
        replays.make(level).unwrap();
        let (made, room) = replays.room_to_end();
        assert!(
            room >= made,
            "room to end {room} of the {made} deferred command buffers made"
        );
    }
    assert_eq!(replays.release(false).map(|(_, settled)| settled), None);
}
