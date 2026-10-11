//! The queues' serial contexts (`contexts/`): the lines as a structure, and the pool of handler
//! threads with the waits it counts, on plain items without a device.

use crate::command_buffer::own::Context;
use crate::contexts::lines::Lines;
use crate::contexts::{block, Contexts, Run, THREADS};
use crate::error::DeadlockCause;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest any step here may take.
const WAIT: Duration = Duration::from_secs(20);

#[test]
fn a_line_runs_one_item_at_a_time_in_order_and_others_go_on() {
    let mut lines = Lines::default();
    lines.push(1, "a1");
    lines.push(1, "a2");
    lines.push(2, "b1");
    assert_eq!(lines.ready(), 2);
    assert_eq!(lines.take(), Some((1, "a1")));
    // Line 1 runs: its next item waits, line 2 does not.
    assert_eq!(lines.take(), Some((2, "b1")));
    assert_eq!(lines.take(), None);
    lines.push(2, "b2");
    lines.done(1);
    // Line 1 is ready again, behind nothing; line 2 still runs.
    assert_eq!(lines.take(), Some((1, "a2")));
    lines.done(2);
    assert_eq!(lines.take(), Some((2, "b2")));
    lines.done(1);
    lines.done(2);
    assert!(lines.is_empty());
    assert!(!lines.is_ready());
}

#[test]
fn ready_lines_are_taken_in_the_order_they_became_ready() {
    let mut lines = Lines::default();
    for queue in [3, 1, 2] {
        lines.push(queue, queue);
    }
    let taken: Vec<u64> = std::iter::from_fn(|| lines.take().map(|(queue, _)| queue)).collect();
    assert_eq!(taken, [3, 1, 2]);
}

/// A plain item.
struct Job(Box<dyn FnOnce() + Send>);

impl Run for Job {
    fn run(self) {
        (self.0)();
    }
}

fn job(f: impl FnOnce() + Send + 'static) -> Job {
    Job(Box::new(f))
}

fn context(device: usize, queue: u64) -> Context {
    Context { device, queue }
}

/// While one of a queue's items blocks, the queue's next item waits and another queue's runs,
/// on another thread (Metal's measured order).
///
/// The items after the blocking one are posted once it runs. The pool starts the items of
/// different queues in no order, so another queue's item posted beside it may run before it
/// starts: under load one did, and this test failed, reading `["c"]`. The blocking item starts
/// late here (20 ms), as a thread the host schedules late does, so the test holds by its waits.
#[test]
fn a_blocked_item_holds_back_its_queue_and_no_other() {
    let contexts: Contexts<Job> = Contexts::start(THREADS).unwrap();
    let log = Arc::new(Mutex::new(Vec::new()));
    let (release, released) = mpsc::channel::<()>();
    let (started, starts) = mpsc::channel::<()>();
    let (sender, done) = mpsc::channel();
    {
        let log = Arc::clone(&log);
        contexts.post(
            1,
            job(move || {
                std::thread::sleep(Duration::from_millis(20));
                log.lock().unwrap().push("a starts");
                started.send(()).unwrap();
                released.recv_timeout(WAIT).unwrap();
                log.lock().unwrap().push("a ends");
            }),
        );
    }
    starts.recv_timeout(WAIT).unwrap();
    for (queue, name) in [(1, "b"), (2, "c")] {
        let (log, sender) = (Arc::clone(&log), sender.clone());
        contexts.post(
            queue,
            job(move || {
                log.lock().unwrap().push(name);
                sender.send(name).unwrap();
            }),
        );
    }
    assert_eq!(done.recv_timeout(WAIT).unwrap(), "c");
    assert_eq!(*log.lock().unwrap(), ["a starts", "c"]);
    release.send(()).unwrap();
    assert_eq!(done.recv_timeout(WAIT).unwrap(), "b");
    assert_eq!(*log.lock().unwrap(), ["a starts", "c", "a ends", "b"]);
    contexts.stop();
}

/// Each of `count` queues' items counts a wait for another context and holds it until told: the
/// waits each item's `block` answered (`Ok`, or the refusal's cause), in the order answered.
fn blocked_waits(
    contexts: &Contexts<Job>,
    device: usize,
    count: u64,
) -> Vec<Result<(), DeadlockCause>> {
    let (release, released) = mpsc::channel::<()>();
    let released = Arc::new(Mutex::new(released));
    let (sender, answers) = mpsc::channel();
    for queue in 0..count {
        let (sender, released) = (sender.clone(), Arc::clone(&released));
        contexts.post(
            queue,
            job(move || {
                let blocked = block(context(device, queue), context(device, count));
                sender
                    .send(blocked.as_ref().map(|_| ()).map_err(|&cause| cause))
                    .unwrap();
                if blocked.is_ok() {
                    released.lock().unwrap().recv_timeout(WAIT).unwrap();
                }
            }),
        );
    }
    let answered: Vec<Result<(), DeadlockCause>> = (0..count)
        .map(|_| answers.recv_timeout(WAIT).unwrap())
        .collect();
    for _ in answered.iter().filter(|answer| answer.is_ok()) {
        release.send(()).unwrap();
    }
    answered
}

/// Metal ran 256 handlers blocked at once: the pool counts as many waits, and refuses none.
#[test]
fn as_many_waits_as_metal_ran_at_once_are_all_counted() {
    let contexts: Contexts<Job> = Contexts::start(THREADS).unwrap();
    let answered = blocked_waits(&contexts, 0x100, 256);
    assert!(answered.iter().all(Result::is_ok));
    contexts.stop();
}

/// A pool of four threads counts three blocked waits; the fourth would leave it no free thread,
/// and is refused for that cause.
#[test]
fn the_wait_that_would_leave_no_free_thread_is_refused() {
    let contexts: Contexts<Job> = Contexts::start(4).unwrap();
    let answered = blocked_waits(&contexts, 0x200, 4);
    assert_eq!(answered.iter().filter(|answer| answer.is_ok()).count(), 3);
    let refused: Vec<_> = answered.iter().filter(|answer| answer.is_err()).collect();
    assert_eq!(refused, [&Err(DeadlockCause::HandlerThreadCap)]);
    // Every wait let go: the count is back to none.
    let deadline = Instant::now() + WAIT;
    while contexts.census().1 != 0 {
        assert!(Instant::now() < deadline, "blocked waits left counted");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(contexts.census().0 <= 4);
    contexts.stop();
}

/// Queue 1's item waits for queue 2; queue 2's item's wait for queue 1 would close the cycle.
#[test]
fn a_wait_closing_a_cycle_is_refused() {
    let contexts: Contexts<Job> = Contexts::start(THREADS).unwrap();
    let device = 0x300;
    let (release, released) = mpsc::channel::<()>();
    let (sender, answers) = mpsc::channel();
    {
        let sender = sender.clone();
        contexts.post(
            1,
            job(move || {
                let blocked = block(context(device, 1), context(device, 2));
                sender
                    .send(("1 waits for 2", blocked.as_ref().err().copied()))
                    .unwrap();
                released.recv_timeout(WAIT).unwrap();
            }),
        );
    }
    assert_eq!(answers.recv_timeout(WAIT).unwrap(), ("1 waits for 2", None));
    contexts.post(
        2,
        job(move || {
            let blocked = block(context(device, 2), context(device, 1));
            sender
                .send(("2 waits for 1", blocked.as_ref().err().copied()))
                .unwrap();
        }),
    );
    assert_eq!(
        answers.recv_timeout(WAIT).unwrap(),
        ("2 waits for 1", Some(DeadlockCause::Cycle))
    );
    release.send(()).unwrap();
    contexts.stop();
}

/// A thread that is not one of the pool's has no wait to count: refused, as no free handler
/// thread is left for it.
#[test]
fn a_wait_off_the_handler_threads_is_not_counted() {
    assert_eq!(
        block(context(0x400, 1), context(0x400, 2)).err(),
        Some(DeadlockCause::HandlerThreadCap)
    );
}

/// After a burst, the threads beyond the first end once idle; a stop from one of the threads
/// does not wait for itself.
#[test]
fn idle_threads_end_and_a_thread_s_own_stop_returns() {
    let contexts: Arc<Contexts<Job>> = Arc::new(Contexts::start(THREADS).unwrap());
    let answered = blocked_waits(&contexts, 0x500, 16);
    assert!(answered.iter().all(Result::is_ok));
    let deadline = Instant::now() + WAIT;
    while contexts.census().0 != 1 {
        assert!(Instant::now() < deadline, "idle threads stayed");
        std::thread::sleep(Duration::from_millis(50));
    }
    let (sender, stopped) = mpsc::channel();
    let inner = Arc::clone(&contexts);
    contexts.post(
        1,
        job(move || {
            inner.stop();
            sender.send(()).unwrap();
        }),
    );
    stopped.recv_timeout(WAIT).unwrap();
    let deadline = Instant::now() + WAIT;
    while contexts.census().0 != 0 {
        assert!(Instant::now() < deadline, "the last thread stayed");
        std::thread::sleep(Duration::from_millis(10));
    }
}
