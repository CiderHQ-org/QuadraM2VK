//! `exit_guard.rs`: sections end before the hook returns, none starts after it closed, the hook
//! never waits for the exiting thread's own sections, its wait is bounded, and it acts only in
//! the process it was registered in. Each test uses a guard of its own; only the fork test uses
//! the process's, which nothing here closes.

use crate::exit_guard::{self, hook, Busy, Call, Guard, InProcess, InProcessDevices};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex, Weak};
use std::time::{Duration, Instant};

/// A guard of the test's own, for the life of the test process.
fn guard() -> &'static Guard {
    Box::leak(Box::new(Guard::new()))
}

/// No device whose queues run in this process.
fn none_in_process() -> InProcessDevices {
    InProcessDevices::new()
}

#[test]
fn sections_start_until_the_guard_closes_and_never_after() {
    let guard = guard();
    let before = exit_guard::own();
    {
        let first = guard.enter();
        assert!(first.is_some());
        let nested = guard.enter();
        assert!(nested.is_some());
        assert_eq!(exit_guard::own(), before + 2);
    }
    assert_eq!(exit_guard::own(), before);
    assert!(!guard.is_closed());
    assert!(guard.close(0, Duration::ZERO));
    assert!(guard.is_closed());
    assert!(guard.enter().is_none());
    assert_eq!(exit_guard::own(), before);
    // On every thread.
    assert!(std::thread::spawn(move || guard.enter().is_none())
        .join()
        .unwrap());
}

/// The hook waits for a section in progress on another thread (a submission, a poll, a device's
/// teardown) and returns once it ended. That section runs to its end, the sections inside it
/// included (a drop's releases), while no section starts afresh after the close.
#[test]
fn the_hook_waits_for_the_sections_in_progress_on_other_threads() {
    let guard = guard();
    let (entered, in_section) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
    let worker = {
        let (entered, in_section) = (Arc::clone(&entered), Arc::clone(&in_section));
        std::thread::spawn(move || {
            let busy = guard.enter().expect("open");
            entered.wait();
            // The guard closes meanwhile.
            in_section.wait();
            std::thread::sleep(Duration::from_millis(200));
            let nested_after_close = guard.enter().is_some();
            drop(busy);
            let fresh_after_close = guard.enter().is_some();
            (nested_after_close, fresh_after_close)
        })
    };
    entered.wait();
    let start = Instant::now();
    let closer = std::thread::spawn(move || guard.close(0, Duration::from_secs(10)));
    while !guard.is_closed() {
        std::thread::yield_now();
    }
    in_section.wait();
    assert!(
        closer.join().unwrap(),
        "the hook gave up on a section that ended"
    );
    let waited = start.elapsed();
    assert!(
        waited >= Duration::from_millis(200),
        "the hook returned after {waited:?}, before the section ended"
    );
    assert!(waited < Duration::from_secs(5), "{waited:?}");
    let (nested, fresh) = worker.join().unwrap();
    assert!(nested, "a section inside one in progress did not start");
    assert!(!fresh, "a section started afresh after the close");
}

/// A section that never ends (a driver or a message sink that blocks) holds the hook back at most
/// its bound, and the hook says so.
#[test]
fn the_hook_s_wait_is_bounded() {
    let guard = guard();
    let (held, release) = (mpsc::channel(), mpsc::channel::<()>());
    let (entered, stop) = (held.0, release.1);
    let worker = std::thread::spawn(move || {
        let _busy: Busy<'static> = guard.enter().expect("open");
        entered.send(()).unwrap();
        let _ = stop.recv();
    });
    held.1.recv().unwrap();
    let bound = Duration::from_millis(300);
    let start = Instant::now();
    assert!(!guard.close(0, bound), "a stuck section ended");
    let waited = start.elapsed();
    assert!(
        waited >= bound && waited < bound + Duration::from_secs(2),
        "the hook waited {waited:?} for a bound of {bound:?}"
    );
    release.0.send(()).unwrap();
    worker.join().unwrap();
}

/// A section around a driver's making of a `VkPipeline` holds the hook past its bound until it
/// ends, within `MAKING_BOUND`: a making outlasts the bound, and the program need not wait for
/// it. One run only on a thread in a section takes such a section (`driver_making`).
#[test]
fn the_hook_waits_past_its_bound_for_a_making_in_progress() {
    let guard = guard();
    let (held, release) = (mpsc::channel(), mpsc::channel::<()>());
    let (entered, stop) = (held.0, release.1);
    let worker = std::thread::spawn(move || {
        let _busy: Busy<'static> = guard.enter_making().expect("open");
        entered.send(()).unwrap();
        let _ = stop.recv();
        std::thread::sleep(Duration::from_millis(300));
    });
    held.1.recv().unwrap();
    release.0.send(()).unwrap();
    let bound = Duration::from_millis(50);
    let start = Instant::now();
    assert!(guard.close(0, bound), "the making's section did not end");
    let waited = start.elapsed();
    assert!(
        waited >= Duration::from_millis(250) && waited < exit_guard::MAKING_BOUND,
        "the hook waited {waited:?}"
    );
    assert!(guard.enter_making().is_none());
    worker.join().unwrap();

    // `driver_making` takes such a section of the process's guard only on a thread in a section.
    let before = exit_guard::own();
    assert_eq!(exit_guard::driver_making(exit_guard::own), before);
    let busy = exit_guard::enter().expect("open");
    assert_eq!(exit_guard::driver_making(exit_guard::own), before + 2);
    drop(busy);
}

/// An exit from inside a section (a message sink that calls `exit`) does not wait for the exiting
/// thread's own section, which can never end first; it still waits for the others.
#[test]
fn the_hook_does_not_wait_for_the_exiting_thread_s_own_sections() {
    let guard = guard();
    let own = guard.enter().expect("open");
    let start = Instant::now();
    assert!(guard.close(exit_guard::own(), Duration::from_secs(10)));
    assert!(start.elapsed() < Duration::from_secs(1));
    drop(own);

    let guard = self::guard();
    let own = guard.enter().expect("open");
    let (other, done) = (mpsc::channel(), mpsc::channel::<()>());
    let worker = std::thread::spawn(move || {
        let _busy = guard.enter().expect("open");
        other.0.send(()).unwrap();
        let _ = done.1.recv();
    });
    other.1.recv().unwrap();
    // One section is this thread's, the other a stuck one's: the hook waits for that one only.
    assert!(!guard.close(exit_guard::own(), Duration::from_millis(100)));
    done.0.send(()).unwrap();
    worker.join().unwrap();
    drop(own);
}

/// Many threads entering and leaving sections while the guard closes: once the hook returned, no
/// thread is in a section, and none enters one again.
#[test]
fn no_section_runs_once_the_hook_returned() {
    let guard = guard();
    let inside = Arc::new(AtomicUsize::new(0));
    let after = Arc::new(AtomicBool::new(false));
    let late = Arc::new(AtomicUsize::new(0));
    let workers: Vec<_> = (0..8)
        .map(|_| {
            let (inside, after, late) =
                (Arc::clone(&inside), Arc::clone(&after), Arc::clone(&late));
            std::thread::spawn(move || {
                for _ in 0..200_000 {
                    let Some(busy) = guard.enter() else {
                        return;
                    };
                    inside.fetch_add(1, Ordering::SeqCst);
                    if after.load(Ordering::SeqCst) {
                        late.fetch_add(1, Ordering::SeqCst);
                    }
                    inside.fetch_sub(1, Ordering::SeqCst);
                    drop(busy);
                }
            })
        })
        .collect();
    std::thread::sleep(Duration::from_millis(5));
    assert!(guard.close(0, Duration::from_secs(10)));
    after.store(true, Ordering::SeqCst);
    assert_eq!(
        inside.load(Ordering::SeqCst),
        0,
        "a section ran past the hook"
    );
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(
        late.load(Ordering::SeqCst),
        0,
        "a section started after the hook"
    );
}

/// The hook acts only in the process it was registered in.
#[test]
fn the_hook_does_nothing_in_another_process() {
    let guard = guard();
    let busy = guard.enter().expect("open");
    let start = Instant::now();
    let devices = none_in_process();
    assert_eq!(
        hook(guard, &devices, 41, 42, 0, Duration::from_secs(10)),
        None
    );
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(!guard.is_closed());
    drop(busy);
    assert_eq!(
        hook(guard, &devices, 42, 42, 0, Duration::from_secs(10)),
        Some(true)
    );
    assert!(guard.is_closed());
}

/// A device whose queues run in this process, as the hook sees it: its work completes when the
/// test says so, and it records when the hook began to wait for it.
struct Running {
    done: Mutex<Option<mpsc::Receiver<()>>>,
    asked: Mutex<Option<Instant>>,
}

impl Running {
    fn new(done: mpsc::Receiver<()>) -> Arc<Running> {
        Arc::new(Running {
            done: Mutex::new(Some(done)),
            asked: Mutex::new(None),
        })
    }

    fn asked(&self) -> Option<Instant> {
        *self.asked.lock().unwrap()
    }
}

impl InProcess for Running {
    fn drain(&self, deadline: Instant) -> bool {
        self.asked.lock().unwrap().get_or_insert_with(Instant::now);
        let done = self.done.lock().unwrap().take();
        let left = deadline.saturating_duration_since(Instant::now());
        done.is_some_and(|done| done.recv_timeout(left).is_ok())
    }
}

/// Add `device` to `devices`, as a device whose queues run in this process.
fn add(devices: &InProcessDevices, device: &Arc<Running>) {
    let weak: Weak<dyn InProcess> = Arc::<Running>::downgrade(device);
    devices.add(weak);
}

/// The interleaving of an exit while a CPU device runs submitted work (the in-flight dispatch
/// a host step's exit leaves on lavapipe), made deterministic: a section is in progress as the
/// hook starts, and the device's work completes only after it ended. The hook waits for the
/// section first, then for the work, and returns once the work completed, in the bound.
#[test]
fn the_hook_waits_for_a_cpu_device_s_work_once_the_sections_ended() {
    let guard = guard();
    let devices = none_in_process();
    let (finish, done) = mpsc::channel();
    let running = Running::new(done);
    add(&devices, &running);
    let (entered, end_section) = (mpsc::channel(), mpsc::channel::<()>());
    let section = std::thread::spawn(move || {
        let busy = guard.enter().expect("open");
        entered.0.send(()).unwrap();
        let _ = end_section.1.recv();
        // Before the count falls: the hook cannot see the section end earlier.
        let ending = Instant::now();
        drop(busy);
        ending
    });
    entered.1.recv().unwrap();
    let pid = std::process::id();
    let start = Instant::now();
    let hooked = std::thread::scope(|scope| {
        let hooking = scope.spawn(|| hook(guard, &devices, pid, pid, 0, Duration::from_secs(10)));
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            running.asked().is_none(),
            "the hook drained before the section ended"
        );
        end_section.0.send(()).unwrap();
        let ended = section.join().unwrap();
        while running.asked().is_none() {
            std::thread::yield_now();
        }
        assert!(
            running.asked().unwrap() >= ended,
            "drained before the section ended"
        );
        std::thread::sleep(Duration::from_millis(100));
        assert!(
            !hooking.is_finished(),
            "the hook returned before the work completed"
        );
        finish.send(()).unwrap();
        hooking.join().unwrap()
    });
    assert_eq!(hooked, Some(true));
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
}

/// Work that never completes holds the hook back at most its bound, sections and work together,
/// and the hook says so; a device gone, or made in another process (a forked child's parent's),
/// is not waited for.
#[test]
fn the_hook_s_wait_for_a_cpu_device_s_work_shares_its_bound() {
    let pid = std::process::id();
    let guard = guard();
    let devices = none_in_process();
    let (_never, done) = mpsc::channel();
    let stuck = Running::new(done);
    add(&devices, &stuck);
    let bound = Duration::from_millis(300);
    let start = Instant::now();
    assert_eq!(hook(guard, &devices, pid, pid, 0, bound), Some(false));
    let waited = start.elapsed();
    assert!(
        waited >= bound && waited < bound + Duration::from_secs(2),
        "the hook waited {waited:?} for a bound of {bound:?}"
    );

    let guard = self::guard();
    let devices = none_in_process();
    let (_never, done) = mpsc::channel();
    let gone = Running::new(done);
    add(&devices, &gone);
    drop(gone);
    let start = Instant::now();
    assert_eq!(
        hook(guard, &devices, pid, pid, 0, Duration::from_secs(10)),
        Some(true)
    );
    assert!(start.elapsed() < Duration::from_secs(1));

    // A device of another process: as a child forked after the parent made it sees it.
    let devices = none_in_process();
    let (_never, done) = mpsc::channel();
    let parents = Running::new(done);
    add(&devices, &parents);
    let start = Instant::now();
    assert!(devices.drain(pid + 1, Instant::now() + Duration::from_secs(10)));
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(parents.asked().is_none());
}

/// A child forked (without exec) from a process whose hook is registered, while another of the
/// parent's threads is in a section of the process's guard, exits at once: the hook it inherits
/// checks the process, and the section's thread is not the child's.
///
/// The child runs the guard's exit handler alone and ends with `_exit`, not `exit`: the handlers
/// a child inherits are every library's the parent loaded, not only the hook and the C library's.
/// A Vulkan driver's may wait for threads only the parent has (Mesa's thread queues join their
/// threads at exit), so a child calling `exit` while another test's lavapipe device lives in this
/// process never ends.
#[cfg(unix)]
#[test]
fn a_forked_child_s_exit_does_not_wait_for_its_parent_s_sections() {
    exit_guard::register();
    let (entered, done) = (mpsc::channel(), mpsc::channel::<()>());
    let worker = std::thread::spawn(move || {
        let _busy = exit_guard::enter().expect("the process is not exiting");
        entered.0.send(()).unwrap();
        let _ = done.1.recv();
    });
    entered.1.recv().unwrap();
    let start = Instant::now();
    // SAFETY: the child runs only the guard's exit handler (a process id and atomics) and
    // `_exit`, which runs no handler.
    let child = unsafe { libc::fork() };
    if child == 0 {
        exit_guard::run_on_exit();
        // SAFETY: as above.
        unsafe { libc::_exit(0) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    let took = start.elapsed();
    done.0.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(reaped, child);
    assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
    assert!(
        took < exit_guard::BOUND / 2,
        "the child's exit took {took:?}: it waited for its parent's section"
    );
}

/// Each entry point has a bit of its own.
#[test]
fn every_entry_point_has_a_bit_of_its_own() {
    let mut bits = 0_u128;
    for &call in Call::ALL {
        let bit = call.bit();
        assert_eq!(bit.count_ones(), 1, "{call:?} has no bit of the word");
        assert_eq!(bits & bit, 0, "{call:?} shares a bit");
        bits |= bit;
    }
}

/// The exit status of a child that registers its hook (its first device), then a destructor (a
/// driver's or a layer's static, made at its first use), then runs `sections`, and exits. The
/// destructor ends the child with 42 when it sees the guard closed (the hook ran first), and 43
/// when it ran first.
#[cfg(unix)]
fn late_destructor_exit(sections: fn()) -> libc::c_int {
    late_destructor_exit_after(|| {}, sections)
}

/// [`late_destructor_exit`], the child running `before` between its hook's registration and the
/// destructor's.
#[cfg(unix)]
fn late_destructor_exit_after(before: fn(), sections: fn()) -> libc::c_int {
    extern "C" fn late_destructor() {
        let code = if exit_guard::closed() { 42 } else { 43 };
        // SAFETY: ends the child at once, from its exit handlers.
        unsafe { libc::_exit(code) };
    }
    // SAFETY: the child registers its hook and the destructor, runs sections (atomics and the
    // C library's `atexit`), and exits.
    let child = unsafe { libc::fork() };
    if child == 0 {
        exit_guard::register();
        before();
        // SAFETY: `late_destructor` is a function with no arguments that never unwinds.
        unsafe { libc::atexit(late_destructor) };
        sections();
        // SAFETY: as above.
        unsafe { libc::exit(0) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    assert_eq!(reaped, child);
    assert!(libc::WIFEXITED(status), "the child did not exit: {status}");
    libc::WEXITSTATUS(status)
}

/// With no section after the destructor's registration, the destructor runs first: the case the
/// registrations below close.
#[cfg(unix)]
#[test]
fn a_destructor_registered_after_the_hook_runs_before_it() {
    assert_eq!(late_destructor_exit(|| {}), 43);
}

/// A section that ends after the first call of an entry point registers the hook again, so a
/// destructor registered before it (by that call) runs after the hook. Its count is not a power
/// of two, so that trigger is the first call's.
#[cfg(unix)]
#[test]
fn a_section_after_an_entry_point_s_first_call_registers_the_hook_again() {
    assert_eq!(
        late_destructor_exit(|| {
            let call = Call::ALL
                .iter()
                .copied()
                .find(|&call| !exit_guard::was_called(call))
                .expect("an entry point no section called");
            // Sections before the destructor, until the next one's count is no power of two.
            while (exit_guard::sections() + 1).is_power_of_two() {
                drop(exit_guard::enter());
            }
            let busy = exit_guard::enter();
            exit_guard::called(call);
            drop(busy);
        }),
        42,
        "the destructor ran before the hook"
    );
}

/// A section whose count reaches a power of two registers the hook again, for statics made at a
/// later call of an entry point already called.
#[cfg(unix)]
#[test]
fn a_section_at_a_power_of_two_registers_the_hook_again() {
    assert_eq!(
        late_destructor_exit(|| loop {
            drop(exit_guard::enter());
            if exit_guard::sections().is_power_of_two() {
                break;
            }
        }),
        42,
        "the destructor ran before the hook"
    );
}

/// The runtime's own threads may reach the device only inside a section or while they run the
/// program's code; the program's threads always may.
#[test]
fn runtime_threads_reach_the_device_only_inside_sections() {
    assert!(exit_guard::may_call(), "the program's thread");
    let answers = exit_guard::spawn(std::thread::Builder::new(), || {
        let outside = exit_guard::may_call();
        let busy = exit_guard::enter().expect("the process is not exiting");
        let inside = exit_guard::may_call();
        drop(busy);
        let mut in_program = false;
        exit_guard::program(|| in_program = exit_guard::may_call());
        let after_program = exit_guard::may_call();
        let mut after_panic = true;
        exit_guard::program(|| panic!("the program's code panics"));
        after_panic &= !exit_guard::may_call();
        (outside, inside, in_program, after_program, after_panic)
    })
    .unwrap()
    .join()
    .unwrap();
    assert_eq!(answers, (false, true, true, false, true));
}

/// A runtime thread started from a thread at the background or utility quality of service runs at
/// the default class; from a thread at any other class it runs as a plain thread started there
/// does. (On macOS 27.0.1, an Apple M4 mini, a plain thread started from a thread that set its own
/// class to background or utility already started at the default class.)
#[cfg(target_os = "macos")]
#[test]
fn a_runtime_thread_runs_at_no_lower_quality_of_service_than_the_default() {
    use libc::qos_class_t as Q;
    fn class() -> u32 {
        let mut class = Q::QOS_CLASS_UNSPECIFIED;
        let mut relative = 0;
        // SAFETY: the calling thread's own handle, and two out-parameters that live for the call.
        let read = unsafe {
            libc::pthread_get_qos_class_np(libc::pthread_self(), &mut class, &mut relative)
        };
        assert_eq!(read, 0, "pthread_get_qos_class_np");
        class as u32
    }
    // A plain thread's class and a runtime thread's, each started from a thread at `start`.
    let from = |start: Q| {
        std::thread::spawn(move || {
            // SAFETY: sets the calling thread's own class; no pointer.
            assert_eq!(unsafe { libc::pthread_set_qos_class_self_np(start, 0) }, 0);
            let plain = std::thread::spawn(class).join().unwrap();
            let runtime = exit_guard::spawn(std::thread::Builder::new(), class)
                .unwrap()
                .join()
                .unwrap();
            (plain, runtime)
        })
        .join()
        .unwrap()
    };
    for start in [Q::QOS_CLASS_BACKGROUND, Q::QOS_CLASS_UTILITY] {
        let (plain, runtime) = from(start);
        assert_eq!(
            runtime,
            if plain == start as u32 {
                Q::QOS_CLASS_DEFAULT as u32
            } else {
                plain
            },
            "from {start:?}: a plain thread {plain:#x}"
        );
    }
    for start in [
        Q::QOS_CLASS_DEFAULT,
        Q::QOS_CLASS_USER_INITIATED,
        Q::QOS_CLASS_USER_INTERACTIVE,
    ] {
        let (plain, runtime) = from(start);
        assert_eq!(runtime, plain, "from {start:?}");
    }
}

/// The threads that take a command buffer from its commit to its handlers run at the
/// user-interactive quality of service from their start, whatever the class of the thread that
/// started them, as read from inside each: the scheduler's, the completion thread, each
/// `VkQueue`'s waiter, and the handler pool's first thread and a later one, which also run their
/// items there. Another of the runtime's threads, the reaper, runs as [`exit_guard::spawn`]
/// starts a thread, below the user-interactive class unless started from it.
#[cfg(target_os = "macos")]
#[test]
fn the_threads_that_take_command_buffers_to_their_handlers_run_at_the_user_interactive_class() {
    use crate::command_buffer::completion::Completion;
    use crate::contexts::{Contexts, Run, THREADS};
    use crate::pipeline::reaper::Reaper;
    use crate::scheduler::Scheduler;
    use libc::qos_class_t as Q;
    use std::thread::ThreadId;

    // The calling thread's requested class (`<sys/qos.h>`), which `libc` does not declare.
    unsafe extern "C" {
        safe fn qos_class_self() -> u32;
    }
    /// A pool's item.
    struct Item(Box<dyn FnOnce() + Send>);
    impl Run for Item {
        fn run(self) {
            (self.0)();
        }
    }
    /// Sends the class of the thread that drops it: the reaper's, which lets go of it.
    struct Reaped(mpsc::Sender<u32>);
    impl Drop for Reaped {
        fn drop(&mut self) {
            let _ = self.0.send(qos_class_self());
        }
    }
    const WAIT: Duration = Duration::from_secs(20);
    let interactive = Q::QOS_CLASS_USER_INTERACTIVE as u32;

    for start in [
        Q::QOS_CLASS_BACKGROUND,
        Q::QOS_CLASS_UTILITY,
        Q::QOS_CLASS_DEFAULT,
        Q::QOS_CLASS_USER_INITIATED,
        Q::QOS_CLASS_USER_INTERACTIVE,
    ] {
        // On a thread at `start`, standing for the program's that makes a device.
        let (started, items, reaper, plain) = std::thread::spawn(move || {
            // SAFETY: sets the calling thread's own class; no pointer.
            assert_eq!(unsafe { libc::pthread_set_qos_class_self_np(start, 0) }, 0);
            let (note, notes) = mpsc::channel();
            exit_guard::ON_START.with(|on_start| {
                *on_start.borrow_mut() = Some(Arc::new(move || {
                    let thread = std::thread::current();
                    let name = thread.name().unwrap_or_default().to_owned();
                    let _ = note.send((name, qos_class_self()));
                }));
            });
            let scheduler = Scheduler::start(1).unwrap();
            let completion = Completion::start(2).unwrap();
            // The pool's first thread runs an item that blocks; another queue's item then starts
            // a thread of its own.
            let contexts: Contexts<Item> = Contexts::start(THREADS).unwrap();
            let (ran, runs) = mpsc::channel::<(ThreadId, u32)>();
            let (release, released) = mpsc::channel::<()>();
            let blocking = ran.clone();
            contexts.post(
                1,
                Item(Box::new(move || {
                    blocking
                        .send((std::thread::current().id(), qos_class_self()))
                        .unwrap();
                    released.recv_timeout(WAIT).unwrap();
                })),
            );
            let first = runs.recv_timeout(WAIT).unwrap();
            contexts.post(
                2,
                Item(Box::new(move || {
                    ran.send((std::thread::current().id(), qos_class_self()))
                        .unwrap();
                })),
            );
            let later = runs.recv_timeout(WAIT).unwrap();
            release.send(()).unwrap();
            contexts.stop();
            scheduler.stop();
            completion.stop();
            exit_guard::ON_START.with(|on_start| on_start.borrow_mut().take());
            let started: Vec<(String, u32)> = notes.try_iter().collect();

            let reaper = Reaper::default();
            let (dropped, drops) = mpsc::channel();
            reaper.start().unwrap().reap(Box::new(Reaped(dropped)));
            let reaped = drops.recv_timeout(WAIT).unwrap();
            reaper.stop();
            let plain = exit_guard::spawn(std::thread::Builder::new(), || qos_class_self())
                .unwrap()
                .join()
                .unwrap();
            (started, [first, later], reaped, plain)
        })
        .join()
        .unwrap();

        let mut names: Vec<&str> = started.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            [
                "metal2vulkan-completion",
                "metal2vulkan-handlers",
                "metal2vulkan-handlers",
                "metal2vulkan-queue-wait-0",
                "metal2vulkan-queue-wait-1",
                "metal2vulkan-scheduler",
            ],
            "from {start:?}"
        );
        for (name, class) in &started {
            assert_eq!(*class, interactive, "{name} from {start:?}");
        }
        let [(first, first_class), (later, later_class)] = items;
        assert_ne!(first, later, "the later item ran on a thread of its own");
        assert_eq!(
            (first_class, later_class),
            (interactive, interactive),
            "the pool's items from {start:?}"
        );
        assert_eq!(reaper, plain, "the reaper from {start:?}");
        if start as u32 != interactive {
            assert_ne!(reaper, interactive, "the reaper from {start:?}");
        }
    }
}

/// In builds with debug assertions, a runtime thread that reaches the device outside a section
/// ends the process.
#[cfg(all(unix, debug_assertions))]
#[test]
fn a_runtime_thread_reaching_the_device_outside_a_section_aborts() {
    // SAFETY: the child starts one thread, which checks and aborts, and otherwise exits.
    let child = unsafe { libc::fork() };
    if child == 0 {
        let checked = exit_guard::spawn(std::thread::Builder::new(), exit_guard::check)
            .map(|thread| thread.join());
        let code = if checked.is_ok() { 43 } else { 44 };
        // SAFETY: ends the child at once.
        unsafe { libc::_exit(code) };
    }
    assert!(child > 0, "fork");
    let mut status = 0;
    // SAFETY: `child` is this process's child.
    let reaped = unsafe { libc::waitpid(child, &mut status, 0) };
    assert_eq!(reaped, child);
    assert!(
        libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGABRT,
        "the child was not aborted: {status}"
    );
}

/// The runtime's sources, without comments, by path relative to `src`; its tests left out.
fn runtime_sources() -> Vec<(String, String)> {
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<(String, String)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            if path.is_dir() {
                if relative != "tests" {
                    walk(&path, root, out);
                }
            } else if relative.ends_with(".rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                let code: Vec<&str> = text
                    .lines()
                    .map(|line| line.find("//").map_or(line, |at| &line[..at]))
                    .collect();
                out.push((relative, code.join("\n")));
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut sources = Vec::new();
    walk(&root, &root, &mut sources);
    assert!(sources.len() > 50, "the runtime's sources were not found");
    sources
}

/// The places in the runtime's sources, outside `allowed`, where `needle` is followed (after
/// spaces and line breaks) by one of `then`.
fn uses(allowed: &[&str], needle: &str, then: &[&str]) -> Vec<String> {
    let mut found = Vec::new();
    for (path, code) in runtime_sources() {
        if allowed.contains(&path.as_str()) {
            continue;
        }
        let compact: String = code.split_whitespace().collect();
        let mut rest = compact.as_str();
        while let Some(at) = rest.find(needle) {
            let after = &rest[at + needle.len()..];
            if then.is_empty() || then.iter().any(|next| after.starts_with(next)) {
                found.push(format!("{path}: {}", &rest[at..(at + 60).min(rest.len())]));
            }
            rest = after;
        }
    }
    found
}

/// Every release of a Vulkan object goes through `release.rs`, which runs it inside a section
/// and marks its entry point: no other file calls a `vkDestroy*` or `vkFree*` entry point.
#[test]
fn every_release_goes_through_the_release_door() {
    let found: Vec<String> = [".destroy_", ".free_"]
        .iter()
        .flat_map(|needle| uses(&["release.rs"], needle, &[]))
        .collect();
    assert!(
        found.is_empty(),
        "Vulkan releases outside release.rs: {found:#?}"
    );
}

/// Every image, render pass and framebuffer the runtime makes goes through `creation.rs`, which
/// refuses it in a host step before any Vulkan call: no other file calls `vkCreateImage`,
/// `vkCreateRenderPass`, `vkCreateRenderPass2` or `vkCreateFramebuffer`, but the device's
/// priming, on the thread that makes the device.
#[test]
fn every_image_render_pass_and_framebuffer_goes_through_the_creation_door() {
    let found: Vec<String> = [
        ".create_image(",
        ".create_render_pass",
        ".create_framebuffer(",
    ]
    .iter()
    .flat_map(|needle| uses(&["creation.rs", "device/prime.rs"], needle, &[]))
    .collect();
    assert!(
        found.is_empty(),
        "images, render passes or framebuffers made outside creation.rs: {found:#?}"
    );
}

/// The creation door refuses each kind of object in a host step, typed, and lets it through
/// outside one.
#[test]
fn the_creation_door_refuses_each_kind_in_a_host_step() {
    use crate::{creation, Error, HostStepObject};
    for object in [
        HostStepObject::Neutral,
        HostStepObject::Image,
        HostStepObject::RenderPass,
        HostStepObject::Framebuffer,
    ] {
        let refused = exit_guard::host_step(|| creation::outside_host_step(object));
        assert_eq!(refused, Err(Error::HostStepObject(object)));
        assert_eq!(creation::outside_host_step(object), Ok(()));
    }
}

/// Every thread the runtime starts is started by `exit_guard::spawn`, so the check knows it, and
/// every call of the program's code goes through `exit_guard::program`, which catches its panics.
#[test]
fn threads_and_the_program_s_code_go_through_the_guard() {
    let spawns = uses(&["exit_guard.rs"], ".spawn(", &[]);
    assert!(spawns.is_empty(), "threads started elsewhere: {spawns:#?}");
    let spawns = uses(&["exit_guard.rs"], "thread::spawn(", &[]);
    assert!(spawns.is_empty(), "threads started elsewhere: {spawns:#?}");
    let calls = uses(&["exit_guard.rs"], "catch_unwind(", &[]);
    assert!(
        calls.is_empty(),
        "the program's code called elsewhere: {calls:#?}"
    );
}

/// The device's functions are kept only in `device/functions.rs`, whose door
/// (`Functions::get`, behind `DeviceShared::vk`) checks the caller and
/// hands a section the exit guard's table alone, and in the parts a device is assembled from.
#[test]
fn the_device_s_functions_are_reached_through_checked_accessors() {
    let kept = uses(
        &["device/functions.rs", "device/mod.rs"],
        ":ash::Device,",
        &[],
    );
    assert!(
        kept.is_empty(),
        "the device's functions kept elsewhere: {kept:#?}"
    );
}

/// The modules that may keep a copy of the device's functions by value, each with why the copy
/// cannot be called after the device's teardown. Any other module that keeps one (a field, a
/// tuple field, a type argument or a by-value parameter) fails
/// `only_the_listed_modules_keep_the_device_s_functions`; so does a listed module that keeps none
/// any more.
const KEEPERS: [(&str, &str); 17] = [
    (
        "device/adopt.rs",
        "the consumer's loader entry, read while the device is adopted",
    ),
    (
        "device/compiles.rs",
        "the driver's pipeline makings, found by the device's handle until the runtime's last \
         hold of the device lets go in its teardown (`unregister`); a making for a device \
         without a record fails before any driver",
    ),
    ("device/functions.rs", "the device's functions themselves"),
    (
        "device/mod.rs",
        "the device's shared state, which the teardown ends",
    ),
    (
        "device/prime.rs",
        "a priming call's parameter, while the device is made",
    ),
    (
        "device/relay.rs",
        "the queues' flights, taken back from the instance's relay first in the teardown",
    ),
    (
        "draw_source.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
    (
        "dynamic_state.rs",
        "the device's extended dynamic state commands (DeviceShared)",
    ),
    (
        "generated_commands.rs",
        "the device's generated-commands loader (DeviceShared)",
    ),
    ("gpu_time.rs", "the device's timing (DeviceShared)"),
    (
        "host_memory.rs",
        "the device's host-memory importer (DeviceShared)",
    ),
    (
        "index_bounds.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
    (
        "index_rewrite.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
    (
        "pipeline/creation_log.rs",
        "MoltenVK's statistics call, inside a call that borrows the device",
    ),
    (
        "render.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
    (
        "render/pending.rs",
        "a pending draw's recording, held by its command buffer",
    ),
    (
        "render_state.rs",
        "a draw's recording (vkCmd*), held by what holds its command buffer",
    ),
];

/// The device itself: a holder of it keeps the device alive, so it is no copy of its functions.
const THE_DEVICE: [&str; 3] = ["Device", "DeviceShared", "InstanceShared"];

/// The runtime's sources as `uses` reads them (no comments, no whitespace), `use` declarations
/// removed: they name types without keeping them.
fn compact_sources() -> Vec<(String, String)> {
    runtime_sources()
        .into_iter()
        .map(|(path, code)| {
            let compact: String = code.split_whitespace().collect();
            (path, without_uses(&compact))
        })
        .collect()
}

/// `code` without its `use` declarations: `use` where an item starts (after `;`, `{`, `}`, `]`,
/// a `pub` or `pub(..)`, or at the start), through its `;`.
fn without_uses(code: &str) -> String {
    let b = code.as_bytes();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    let mut kept = 0;
    while i < b.len() {
        let item_start = i == 0
            || matches!(b[i - 1], b';' | b'{' | b'}' | b']' | b')')
            || (i >= 3 && b[i - 3..i] == *b"pub");
        if item_start && b[i..].starts_with(b"use") {
            if let Some(end) = code[i..].find(';') {
                // A path list only (`a::b::{C,D as E}`), not a field or an expression.
                let path = &code[i + 3..i + end];
                let listed = path.bytes().all(|c| {
                    c.is_ascii_alphanumeric()
                        || matches!(c, b'_' | b':' | b'{' | b'}' | b',' | b'*')
                }) && !path.replace("::", "").contains(':');
                if listed {
                    out.push_str(&code[kept..i]);
                    i += end + 1;
                    kept = i;
                    continue;
                }
            }
        }
        i += 1;
    }
    out.push_str(&code[kept..]);
    out
}

/// The start and end of each path (`a::b::C`) in `code`.
fn paths(code: &str) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let first = |c: u8| c.is_ascii_alphabetic() || c == b'_';
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !first(b[i]) {
            i += 1;
            continue;
        }
        let start = i;
        loop {
            while i < b.len() && ident(b[i]) {
                i += 1;
            }
            if i + 2 < b.len() && b[i] == b':' && b[i + 1] == b':' && first(b[i + 2]) {
                i += 2;
            } else {
                break;
            }
        }
        out.push((start, i));
    }
    out
}

/// Whether `path` names a function table: a Vulkan function pointer, ash's device functions or a
/// device extension's, or one of the runtime's types in `own` (bare, or behind modules only).
fn is_table(path: &str, own: &[String]) -> bool {
    let segments: Vec<&str> = path.split("::").collect();
    let last = segments.last().copied().unwrap_or_default();
    if last.starts_with("PFN_") {
        return true;
    }
    if segments.first() == Some(&"ash")
        && segments.get(1) != Some(&"vk")
        && ["Device", "DeviceFn", "InstanceFn"].contains(&last)
    {
        return true;
    }
    own.iter().any(|name| name == last)
        && segments[..segments.len() - 1]
            .iter()
            .all(|segment| segment.starts_with(|c: char| c.is_ascii_lowercase()))
}

/// Each place in `code` that keeps a function table by value: a field's or a by-value
/// parameter's type, a type argument, a tuple's member, or a function-pointer type.
fn stored(code: &str, own: &[String]) -> Vec<String> {
    let b = code.as_bytes();
    let around = |start: usize, end: usize| {
        let mut from = start.saturating_sub(30);
        while !code.is_char_boundary(from) {
            from -= 1;
        }
        let mut to = (end + 5).min(code.len());
        while !code.is_char_boundary(to) {
            to += 1;
        }
        code[from..to].to_string()
    };
    let mut found = Vec::new();
    for (start, end) in paths(code) {
        if !is_table(&code[start..end], own) {
            continue;
        }
        let before = if start > 0 { b[start - 1] } else { 0 };
        let after = b.get(end).copied().unwrap_or(0);
        if after == b'{' {
            // A value's construction, not a place that keeps one.
            continue;
        }
        let typed = before == b':' && (start < 2 || b[start - 2] != b':');
        if typed
            || before == b'<'
            || before == b'('
            || (before == b',' && (after == b')' || after == b'>'))
        {
            found.push(around(start, end));
        }
    }
    let pointer = "extern\"system\"fn(";
    let mut from = 0;
    while let Some(at) = code[from..].find(pointer) {
        found.push(around(from + at, from + at + pointer.len()));
        from += at + pointer.len();
    }
    found
}

/// The structs, enums and type aliases whose definitions keep a function table, by name, and
/// whether each is visible outside its module (`pub`, `pub(..)`).
fn wrappers(sources: &[(String, String)], own: &[String]) -> Vec<(String, bool)> {
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut found = Vec::new();
    for (_, code) in sources {
        let b = code.as_bytes();
        for keyword in ["struct", "enum", "type"] {
            let mut from = 0;
            while let Some(at) = code[from..].find(keyword) {
                let k = from + at;
                from = k + keyword.len();
                let visible = (k >= 3 && b[k - 3..k] == *b"pub")
                    || (k >= 1
                        && b[k - 1] == b')'
                        && code[..k].rfind("pub(").is_some_and(|p| {
                            code[p + 4..k - 1]
                                .bytes()
                                .all(|c| c.is_ascii_lowercase() || c == b':')
                        }));
                if !visible && k > 0 && ident(b[k - 1]) {
                    continue;
                }
                let name_start = from;
                if !b.get(name_start).is_some_and(u8::is_ascii_uppercase) {
                    continue;
                }
                let mut i = name_start;
                while i < b.len() && ident(b[i]) {
                    i += 1;
                }
                let name = code[name_start..i].to_string();
                // Skip generics, then take the body: a type alias's right side, or a struct's
                // or enum's braces or parentheses.
                let mut depth = 0i32;
                while i < b.len() {
                    match b[i] {
                        b'<' => depth += 1,
                        b'>' => depth -= 1,
                        _ if depth == 0 => break,
                        _ => {}
                    }
                    i += 1;
                }
                let body = if keyword == "type" {
                    if b.get(i) != Some(&b'=') {
                        continue;
                    }
                    let end = code[i..].find(';').map_or(code.len(), |e| i + e);
                    &code[i + 1..end]
                } else {
                    if !matches!(b.get(i), Some(b'{' | b'(')) {
                        continue;
                    }
                    let open = i;
                    let mut depth = 0i32;
                    while i < b.len() {
                        match b[i] {
                            b'{' | b'(' => depth += 1,
                            b'}' | b')' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    &code[open..(i + 1).min(code.len())]
                };
                let keeps = if keyword == "type" {
                    paths(body).iter().any(|&(s, e)| is_table(&body[s..e], own))
                        || body.contains("extern\"system\"fn(")
                } else {
                    !stored(body, own).is_empty()
                };
                if keeps {
                    found.push((name, visible));
                }
            }
        }
    }
    found
}

/// Only the listed modules keep a copy of the device's functions (`KEEPERS`). A copy is any by
/// value: a Vulkan function pointer (`PFN_*`, an `extern "system" fn` type), ash's device
/// functions or a device extension's (`ash::Device`, `ash::*::Device`, `*::DeviceFn`), the
/// runtime's `Functions` and generated-commands `Commands`, the visible types of the runtime's
/// that keep one of those, and the visible types that keep such a type. Elsewhere a copy could be
/// called after the device's teardown wherever its holder outlived the device: the SIGSEGV, a
/// shader module's own `ash::Device`. An object that releases Vulkan objects holds the device
/// (`Device`, reaching its functions through `DeviceShared::vk`), or is the device's and destroyed
/// by it (`pipeline/content_keyed.rs`).
#[test]
fn only_the_listed_modules_keep_the_device_s_functions() {
    let sources = compact_sources();
    let mut own: Vec<String> = vec!["Functions".into(), "Commands".into()];
    // Two levels: the types that keep a table, and the visible types that keep one of those.
    // Deeper holders (an encoder, a command buffer) hold the device themselves.
    for _ in 0..2 {
        let more: Vec<String> = wrappers(&sources, &own)
            .into_iter()
            .filter(|(name, visible)| {
                *visible && !own.contains(name) && !THE_DEVICE.contains(&name.as_str())
            })
            .map(|(name, _)| name)
            .collect();
        if more.is_empty() {
            break;
        }
        for name in more {
            if !own.contains(&name) {
                own.push(name);
            }
        }
    }
    let mut outside = Vec::new();
    let mut keeping = std::collections::BTreeSet::new();
    for (path, code) in &sources {
        let found = stored(code, &own);
        if found.is_empty() {
            continue;
        }
        keeping.insert(path.clone());
        if !KEEPERS.iter().any(|(file, _)| file == path) {
            outside.extend(found.into_iter().map(|at| format!("{path}: {at}")));
        }
    }
    assert!(
        outside.is_empty(),
        "copies of the device's functions outside the listed modules (types {own:?}): {outside:#?}"
    );
    let stale: Vec<&str> = KEEPERS
        .iter()
        .map(|(file, _)| *file)
        .filter(|file| !keeping.contains(*file))
        .collect();
    assert!(
        stale.is_empty(),
        "listed modules that keep no copy any more: {stale:?}"
    );
    // The scan sees the copies it must: the device's own functions and an extension's.
    assert!(
        own.iter().any(|name| name == "ExtendedDynamicState"),
        "{own:?}"
    );
    assert!(keeping.contains("host_memory.rs"));
    // Each kind of copy is seen wherever it is written, a holder of the device is not, and a
    // `use` names a type without keeping it.
    let scan = |code: &str| {
        stored(
            &without_uses(&code.split_whitespace().collect::<String>()),
            &own,
        )
    };
    for copy in [
        "struct A { destroy: vk::PFN_vkDestroyShaderModule }",
        "struct A { loader: ash::khr::swapchain::DeviceFn }",
        "struct A(ash::Device);",
        "struct A { table: Arc<Functions> }",
        "struct A { dynamic: Option<ExtendedDynamicState> }",
        "fn a(raw: extern \"system\" fn()) {}",
    ] {
        assert!(!scan(copy).is_empty(), "not seen: {copy}");
    }
    for no_copy in [
        "struct A { device: Device, raw: vk::ShaderModule }",
        "use ash::vk::PFN_vkDestroyShaderModule; struct A { shared: Arc<DeviceShared> }",
    ] {
        assert!(scan(no_copy).is_empty(), "seen: {no_copy}");
    }
}

/// The exit guard's table names each entry point once, by its Vulkan name.
#[test]
fn the_table_names_each_entry_point_once() {
    let names: Vec<&std::ffi::CStr> = Call::ALL.iter().map(|call| call.name()).collect();
    for (at, name) in names.iter().enumerate() {
        assert!(name.to_bytes().starts_with(b"vk"), "{name:?}");
        assert!(!names[..at].contains(name), "{name:?} twice");
    }
}

/// Priming's check misses nothing but the entry points it cannot make (and, on a queue family
/// that writes no timestamps, the timestamp commands, on a device without device addresses, a
/// buffer's address, and on one without generated commands, the release of their objects), and
/// names an entry point of the table it did not make.
#[test]
fn the_priming_check_names_what_was_not_made() {
    use crate::device::prime::{
        missing, Has, DEVICE_ADDRESS_CALLS, GENERATED_COMMANDS_CALLS, SPARSE_CALLS,
        TIMESTAMP_CALLS, UNPRIMED,
    };
    let has = |timestamps, device_addresses| Has {
        timestamps,
        device_addresses,
        generated_commands: true,
        sparse_binding: true,
    };
    let all = Call::ALL.iter().fold(0, |bits, call| bits | call.bit());
    assert!(missing(all, has(true, true)).is_empty());
    let made = Call::ALL
        .iter()
        .filter(|call| !UNPRIMED.contains(call))
        .fold(0, |bits, call| bits | call.bit());
    assert!(missing(made, has(true, true)).is_empty());
    assert_eq!(
        missing(0, has(true, true)).len(),
        Call::ALL.len() - UNPRIMED.len()
    );
    for &call in Call::ALL {
        if !UNPRIMED.contains(&call) {
            assert_eq!(missing(made & !call.bit(), has(true, true)), [call]);
        }
    }
    // Without timestamps, their commands are neither made nor missed.
    let untimed = TIMESTAMP_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    assert!(missing(untimed, has(false, true)).is_empty());
    assert_eq!(missing(untimed, has(true, true)), TIMESTAMP_CALLS);
    // Without device addresses, nor is a buffer's address.
    let unaddressed = DEVICE_ADDRESS_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    assert!(missing(unaddressed, has(true, false)).is_empty());
    assert_eq!(missing(unaddressed, has(true, true)), DEVICE_ADDRESS_CALLS);
    // Without generated commands, nor is the release of their objects.
    let ungenerated = GENERATED_COMMANDS_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    let without = Has {
        generated_commands: false,
        ..has(true, true)
    };
    assert!(missing(ungenerated, without).is_empty());
    assert_eq!(
        missing(ungenerated, has(true, true)),
        GENERATED_COMMANDS_CALLS
    );
    // Without sparse binding, nor is a batch's bind.
    let unbound = SPARSE_CALLS
        .iter()
        .fold(made, |bits, call| bits & !call.bit());
    let without = Has {
        sparse_binding: false,
        ..has(true, true)
    };
    assert!(missing(unbound, without).is_empty());
    assert_eq!(missing(unbound, has(true, true)), SPARSE_CALLS);
}

/// A thread is in a section, so that the device's functions are the table's alone, while its
/// innermost frame is one of the runtime's sections: not while it runs the program's code a
/// section called (a message sink), and again inside a section that code starts.
#[test]
fn the_table_applies_inside_the_runtime_s_sections_only() {
    use std::cell::Cell;
    let guard = guard();
    assert!(!exit_guard::in_section());
    let outer = guard.enter();
    assert!(exit_guard::in_section());
    let seen = Cell::new((true, false, true));
    exit_guard::program(|| {
        let program = exit_guard::in_section();
        let inner = guard.enter();
        let nested = exit_guard::in_section();
        drop(inner);
        seen.set((program, nested, exit_guard::in_section()));
    });
    assert_eq!(seen.get(), (false, true, false));
    assert!(exit_guard::in_section());
    drop(outer);
    assert!(!exit_guard::in_section());
}

/// A device registers the hook when it can have made statics since the hook's last registration:
/// the process's first (a forked child's included), one of an identity new since the last load
/// of a shared object, and the first after such a load. Made again on what stays loaded, devices
/// register nothing, however many, and the identities kept stay as many as the distinct ones.
#[test]
fn devices_register_the_hook_only_when_they_can_have_made_statics() {
    use crate::exit_guard::{Admitted, Devices};
    let mut devices = Devices::new();
    assert_eq!(
        devices.admit(false, Some(7), vec![1]),
        Some(Admitted::Load),
        "the process's first"
    );
    assert_eq!(devices.admit(true, Some(7), vec![1]), None);
    assert_eq!(
        devices.admit(true, Some(7), vec![2]),
        Some(Admitted::Identity),
        "a new identity"
    );
    for _ in 0..1000 {
        assert_eq!(devices.admit(true, Some(7), vec![1]), None);
        assert_eq!(devices.admit(true, Some(7), vec![2]), None);
    }
    assert_eq!(devices.kept(), 2);
    assert_eq!(
        devices.admit(true, Some(8), vec![1]),
        Some(Admitted::Load),
        "the first after a load"
    );
    assert_eq!(devices.kept(), 1);
    assert_eq!(devices.admit(true, Some(8), vec![1]), None);
    assert_eq!(
        devices.admit(true, Some(8), vec![2]),
        Some(Admitted::Identity),
        "known before the load, new since"
    );
    assert_eq!(
        devices.admit(false, Some(8), vec![1]),
        Some(Admitted::Load),
        "a forked child's first"
    );
    // Where the host keeps no count, every device registers as at a load, and one identity is
    // kept.
    for _ in 0..10 {
        assert_eq!(devices.admit(true, None, vec![1]), Some(Admitted::Load));
    }
    assert_eq!(devices.kept(), 1);
}

/// A section calling the table's first entry point.
#[cfg(unix)]
fn call_the_first_entry_point() {
    let busy = exit_guard::enter();
    exit_guard::called(Call::ALL[0]);
    drop(busy);
}

/// The call before the layer's load, then empty sections until the next section's count is no
/// power of two. Those sections run before the destructor's registration: one ending at a power
/// of two registers the hook again, and after it that would pass the tests below for that
/// trigger's reason, not the first call's.
#[cfg(unix)]
fn call_the_first_entry_point_then_skip_powers_of_two() {
    call_the_first_entry_point();
    while (exit_guard::sections() + 1).is_power_of_two() {
        drop(exit_guard::enter());
    }
}

/// The gap this closes: a layer loaded again registers a destructor at an entry point's first call
/// since (after the hook's registration at the device), and a section calling that entry point,
/// called before the load, registers nothing, so the destructor runs before the hook.
#[cfg(unix)]
#[test]
fn a_call_made_before_a_load_registers_nothing_again_unforgotten() {
    assert_eq!(
        late_destructor_exit_after(
            call_the_first_entry_point_then_skip_powers_of_two,
            call_the_first_entry_point
        ),
        43
    );
}

/// Once the device made after the load forgot the entry points called
/// (`exit_guard::forget_calls`), the section calling one again registers the hook again as it
/// ends, so the layer's destructor registered before runs after the hook.
#[cfg(unix)]
#[test]
fn a_section_after_an_entry_point_s_first_call_since_a_load_registers_the_hook_again() {
    assert_eq!(
        late_destructor_exit_after(call_the_first_entry_point_then_skip_powers_of_two, || {
            exit_guard::forget_calls();
            call_the_first_entry_point();
        }),
        42,
        "the destructor ran before the hook"
    );
}

/// The process's count of loaded shared objects never falls, and rises when one is loaded (where
/// a library not loaded yet can be found here).
#[cfg(any(target_os = "linux", target_vendor = "apple"))]
#[test]
fn the_count_of_loaded_objects_rises_with_a_load() {
    use std::ffi::CString;
    let before = crate::loaded::count().expect("the host keeps a count");
    let candidates: &[&str] = if cfg!(target_vendor = "apple") {
        &[
            "/usr/lib/libz.1.dylib",
            "/usr/lib/libbz2.1.0.dylib",
            "/usr/lib/libxml2.2.dylib",
        ]
    } else {
        &[
            "libz.so.1",
            "libbz2.so.1.0",
            "libexpat.so.1",
            "libxml2.so.2",
        ]
    };
    let fresh = candidates.iter().find_map(|name| {
        let name = CString::new(*name).unwrap();
        // SAFETY: a library name; `RTLD_NOLOAD` loads nothing, so only one not loaded yet is
        // loaded below.
        let loaded = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD) };
        if !loaded.is_null() {
            // SAFETY: the handle `dlopen` just answered.
            unsafe { libc::dlclose(loaded) };
            return None;
        }
        // SAFETY: as above; a system library, kept loaded.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        (!handle.is_null()).then_some(name)
    });
    let after = crate::loaded::count().unwrap();
    match fresh {
        Some(name) => assert!(after > before, "{name:?} loaded: {before} then {after}"),
        None => {
            eprintln!("no library left to load among {candidates:?}");
            assert!(after >= before);
        }
    }
}

/// The public query (`process_exiting`) reads the process's guard: open while the process runs. The
/// exits lane reads it closed in an exit handler that runs after the hook (`runtime_lane::exits`).
#[test]
fn the_process_is_not_exiting_while_its_tests_run() {
    assert!(!crate::process_exiting());
    assert_eq!(crate::process_exiting(), exit_guard::closed());
}
