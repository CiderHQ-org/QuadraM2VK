//! What a Linux process-group member holds resident on its own.
//!
//! The watchdog sums the resident memory of every member of a worker's process group. On Linux a
//! process started through glibc's `posix_spawn` (which `std::process::Command` uses) or `vfork`
//! runs in its parent's memory from its creation until its `exec` installs its own image, and the
//! kernel lets the parent go on before that switch. `/proc` then reports the parent's resident
//! memory as the child's: a worker sampled just after its spawn would be charged the supervisor's
//! memory, and a tool the worker starts would charge the worker's memory a second time. So a
//! member is charged only memory that is its own:
//!
//! - One read of `/proc/<pid>/stat` states its process group, its parent, and its flags together:
//!   the kernel takes the process's exec lock for that read, the lock an exec holds across the
//!   switch to its new image.
//! - A member whose flags still carry `PF_FORKNOEXEC` has not completed an `exec` since it was
//!   created: the kernel clears the flag only after the new image is installed.
//! - The group's leader is the supervisor's own spawn, which `std` always execs. A leader that has
//!   not completed its exec is still in the supervisor's memory (or, after a fork-based spawn, in a
//!   copy-on-write copy of it) and is charged nothing.
//! - Any other member that has not completed an exec either still shares its parent's memory or
//!   is a copy-on-write copy of it, and only `kcmp(KCMP_VM)` tells which. One that shares its
//!   parent's memory is charged nothing: those pages are counted with the parent when the parent
//!   is a member. Where the kernel refuses the comparison (a container whose seccomp filter refuses
//!   `kcmp` without `CAP_SYS_PTRACE`, a kernel built without it), the member is counted as before,
//!   so the measure can briefly overcount and never undercounts.
//! - Every other member is charged its own `VmRSS`, read after its flags, so it is the resident
//!   size of its own image.

/// `PF_FORKNOEXEC` (`include/linux/sched.h`): the process has not completed an `exec` since its
/// creation.
const PF_FORKNOEXEC: i64 = 0x40;

/// `KCMP_VM` (`include/uapi/linux/kcmp.h`): compare two processes' address spaces.
const KCMP_VM: libc::c_int = 1;

/// A group member as one read of its `/proc/<pid>/stat` states it.
#[derive(Debug, PartialEq, Eq)]
struct Stat {
    parent: u32,
    group: u32,
    /// `PF_FORKNOEXEC`: no exec completed since the process was created.
    before_exec: bool,
}

/// The resident bytes `pid`, a member of `group` (whose leader is the supervisor's spawn), holds
/// on its own; `None` once it has left the group or is gone.
pub(super) fn member_resident_bytes(group: u32, pid: u32) -> Result<Option<u64>, String> {
    let Some(stat) = stat(pid)? else {
        return Ok(None);
    };
    if stat.group != group {
        return Ok(None);
    }
    if stat.before_exec && (pid == group || shares_address_space(pid, stat.parent)) {
        return Ok(Some(0));
    }
    super::process_resident_bytes(pid)
}

/// One read of `/proc/<pid>/stat`, or `None` when the process is gone. The read waits while the
/// process is inside the part of an exec that switches its image.
fn stat(pid: u32) -> Result<Option<Stat>, String> {
    let path = format!("/proc/{pid}/stat");
    match std::fs::read_to_string(&path) {
        Ok(text) => parse_stat(&text).map_err(|error| format!("read {path}: {error}")),
        Err(error) if super::process_gone(&error) => Ok(None),
        Err(error) => Err(format!("read {path}: {error}")),
    }
}

/// What one `/proc/<pid>/stat` text states, or `None` for a process that has finished exiting.
/// Once an exiting process has released its signal state (state `X`, dead), the kernel prints
/// `-1` for its group and `0` for its parent. By then it has released its memory too, so it is
/// gone, as a process whose `/proc` directory has vanished is.
fn parse_stat(text: &str) -> Result<Option<Stat>, String> {
    // Field 2, the command name, is parenthesized and may hold spaces and parentheses itself:
    // the fields after it start past its last `)`. `field(n)` is proc(5)'s field `n`.
    let rest = text
        .rsplit_once(')')
        .map(|(_, rest)| rest.split_whitespace().collect::<Vec<_>>())
        .unwrap_or_default();
    let field = |n: usize| {
        rest.get(n - 3)
            .copied()
            .ok_or_else(|| format!("no field {n} in {text:?}"))
    };
    let number = |n: usize| {
        field(n)?
            .parse::<i64>()
            .map_err(|error| format!("field {n} of {text:?}: {error}"))
    };
    let (state, parent, group) = (field(3)?, number(4)?, number(5)?);
    if state == "X" || group < 0 {
        return Ok(None);
    }
    let id = |value: i64| u32::try_from(value).map_err(|error| format!("{text:?}: {error}"));
    Ok(Some(Stat {
        parent: id(parent)?,
        group: id(group)?,
        before_exec: number(9)? & PF_FORKNOEXEC != 0,
    }))
}

/// Whether `pid` and `other` use one address space, as `kcmp(2)` compares them; `false` when the
/// kernel refuses or cannot compare (the member is then counted) or either process is gone.
fn shares_address_space(pid: u32, other: u32) -> bool {
    // `KCMP_VM` compares no file descriptors, so both index arguments are unused.
    let unused: libc::c_ulong = 0;
    // SAFETY: a system call on two process ids; it reads and writes no memory of this process.
    let order = unsafe {
        libc::syscall(
            libc::SYS_kcmp,
            pid as libc::pid_t,
            other as libc::pid_t,
            KCMP_VM,
            unused,
            unused,
        )
    };
    order == 0
}

#[cfg(test)]
mod tests {
    use super::super::{
        process_resident_bytes, worker_resident_bytes, MonitoredChild, ProcessControl,
        ProcessFailure, ProcessStop,
    };
    use super::*;
    use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    const MIB: u64 = 1024 * 1024;
    /// The per-attempt ceiling every validation worker runs under.
    const CEILING: u64 = crate::worker::MEMORY_LIMIT_BYTES;

    /// `bytes` of memory this process holds resident until the result is dropped.
    fn hold(bytes: u64) -> Vec<u8> {
        let held = vec![1u8; bytes as usize];
        std::hint::black_box(&held);
        held
    }

    /// What [`shared_memory_child`] reads; it lives in the creating process's memory, which the
    /// child shares.
    #[repr(C)]
    struct ChildArgs {
        /// The process group to join (0: its own), or -1 to stay in the creator's.
        group: libc::pid_t,
        ready: libc::c_int,
        release_read: libc::c_int,
        release_write: libc::c_int,
    }

    /// The body of a process created with `clone(CLONE_VM)` that never execs: it runs on its own
    /// stack in its creator's memory with no runtime of its own, so it makes only raw system
    /// calls. It drops its copy of the release pipe's write end (so its creator's exit releases
    /// it), joins its group, says it is ready, and waits to be released.
    extern "C" fn shared_memory_child(arg: *mut libc::c_void) -> libc::c_int {
        // SAFETY: `arg` is the `ChildArgs` its creator keeps alive until this process is gone, in
        // the memory this process shares; the calls touch only the bytes passed to them.
        unsafe {
            let args = &*(arg as *const ChildArgs);
            libc::syscall(libc::SYS_close, args.release_write);
            if args.group >= 0 {
                libc::syscall(libc::SYS_setpgid, 0, args.group);
            }
            let byte = 1u8;
            libc::syscall(libc::SYS_write, args.ready, &byte as *const u8, 1usize);
            let mut byte = 0u8;
            libc::syscall(
                libc::SYS_read,
                args.release_read,
                &mut byte as *mut u8,
                1usize,
            );
        }
        0
    }

    /// A process in this process's memory that never execs: the state of a spawn between its
    /// creation and its exec, held until dropped.
    struct SharedMemoryChild {
        pid: libc::pid_t,
        release: Option<OwnedFd>,
        _args: Box<ChildArgs>,
        _stack: Vec<u8>,
    }

    impl SharedMemoryChild {
        /// Create one in process group `group` (0: its own; `None`: this process's) and wait until
        /// it is there.
        fn start(group: Option<libc::pid_t>) -> SharedMemoryChild {
            let pipe = || {
                let mut fds = [0; 2];
                // SAFETY: `fds` has room for the two descriptors `pipe2` writes.
                assert_eq!(unsafe { libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC) }, 0);
                // SAFETY: both descriptors were just created and are owned here alone.
                unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) }
            };
            let (ready_read, ready_write) = pipe();
            let (release_read, release_write) = pipe();
            let args = Box::new(ChildArgs {
                group: group.unwrap_or(-1),
                ready: ready_write.as_raw_fd(),
                release_read: release_read.as_raw_fd(),
                release_write: release_write.as_raw_fd(),
            });
            let mut stack = vec![0u8; 256 * 1024];
            let top = (stack.as_mut_ptr() as usize + stack.len()) & !15;
            // SAFETY: the child runs `shared_memory_child` on `stack` (kept alive, as `args` is,
            // until it is reaped) and makes only raw system calls.
            let pid = unsafe {
                libc::clone(
                    shared_memory_child,
                    top as *mut libc::c_void,
                    libc::CLONE_VM | libc::SIGCHLD,
                    &*args as *const ChildArgs as *mut libc::c_void,
                )
            };
            assert!(
                pid > 0,
                "clone(CLONE_VM): {}",
                std::io::Error::last_os_error()
            );
            drop(ready_write);
            let mut poll = libc::pollfd {
                fd: ready_read.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: one valid `pollfd`.
            assert_eq!(unsafe { libc::poll(&mut poll, 1, 5000) }, 1, "never ready");
            SharedMemoryChild {
                pid,
                release: Some(release_write),
                _args: args,
                _stack: stack,
            }
        }
    }

    impl Drop for SharedMemoryChild {
        fn drop(&mut self) {
            drop(self.release.take());
            let mut status = 0;
            // SAFETY: reaps this process's own child.
            unsafe { libc::waitpid(self.pid, &mut status, 0) };
        }
    }

    /// Whether this kernel lets this process compare address spaces (`kcmp`); a container's
    /// seccomp filter refuses it without `CAP_SYS_PTRACE`.
    fn kcmp_compares() -> bool {
        let own = std::process::id();
        shares_address_space(own, own)
    }

    /// A member's stat as the kernel prints it at each stage. The dead `spirv-val` is the line
    /// that stopped five attempts as tooling failures in our probe: a member that finished
    /// exiting between the group's enumeration and its read is gone, not an unreadable member.
    #[test]
    fn a_members_stat_states_its_group_parent_and_exec() {
        let dead = "1462190 (spirv-val) X 0 -1 -1 0 -1 4227084 302 0 0 0 0 0 0 0 36 16 0 0 \
                    2577247 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 17 0 0 5 0 0 0 0 0 0 0 0 0 0 0";
        assert_eq!(parse_stat(dead), Ok(None));
        // A command name holding spaces and parentheses; PF_FORKNOEXEC (0x40) set and clear.
        let spawned = "4242 (a) b (c) S 4000 4100 4100 0 -1 4194368 0 0";
        assert_eq!(
            parse_stat(spawned),
            Ok(Some(Stat {
                parent: 4000,
                group: 4100,
                before_exec: true,
            }))
        );
        let execed = "4242 (spirv-val) R 4000 4100 4100 0 -1 4194304 0 0";
        assert_eq!(
            parse_stat(execed),
            Ok(Some(Stat {
                parent: 4000,
                group: 4100,
                before_exec: false,
            }))
        );
        assert!(parse_stat("4242 (cut) S 4000").is_err());
    }

    /// The defect, made deterministic: a worker still in the supervisor's memory (here one that
    /// never execs, as a spawn is until its exec switches its image) read as the supervisor's
    /// resident size, so a supervisor past the 500-MiB ceiling stopped every attempt sampled in
    /// that window. The sampler charges such a leader nothing, kcmp or not.
    #[test]
    fn a_worker_still_in_the_supervisors_memory_is_charged_nothing() {
        let held = hold(CEILING + 100 * MIB);
        let leader = SharedMemoryChild::start(Some(0));
        let group = leader.pid as u32;
        let reported = process_resident_bytes(group).unwrap().unwrap();
        assert!(
            reported > CEILING,
            "the shared leader reads as the supervisor's {} MiB, not {} MiB",
            held.len() as u64 / MIB,
            reported / MIB
        );
        assert_eq!(worker_resident_bytes(group), Ok(Some(0)));
        drop(leader);
        drop(held);
    }

    /// The real race: right after `posix_spawn` returns, a busy supervisor's child can still be in
    /// the supervisor's memory (its exec waits on the supervisor's memory map, which these threads
    /// keep busy). No trivial worker of a supervisor past the ceiling is stopped or charged it.
    #[test]
    fn a_trivial_worker_of_an_inflated_supervisor_is_never_stopped() {
        let length = (CEILING + 100 * MIB) as usize;
        // SAFETY: a fresh private anonymous mapping, unmapped below.
        let region = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(region, libc::MAP_FAILED);
        // SAFETY: the mapping is `length` writable bytes.
        unsafe { std::ptr::write_bytes(region.cast::<u8>(), 1, length) };
        let stop = std::sync::atomic::AtomicBool::new(false);
        let address = region as usize;
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    let mut writable = false;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let protection = if writable {
                            libc::PROT_READ | libc::PROT_WRITE
                        } else {
                            libc::PROT_READ
                        };
                        // SAFETY: changes the protection of the mapping this test owns, which
                        // nothing writes while the threads run.
                        unsafe { libc::mprotect(address as *mut libc::c_void, length, protection) };
                        writable = !writable;
                    }
                });
            }
            let result = (|| -> Result<(), String> {
                for attempt in 0..40 {
                    let mut command = Command::new("sleep");
                    command.arg("0.02");
                    let exit = MonitoredChild::spawn(&mut command)
                        .map_err(|error| error.to_string())?
                        .wait(&ProcessControl {
                            deadline: Some(Instant::now() + Duration::from_secs(20)),
                            memory_limit_bytes: Some(CEILING),
                            cancelled: None,
                        })
                        .map_err(|stop| format!("attempt {attempt}: {stop}"))?;
                    let peak = exit.peak_resident_bytes.unwrap_or(0);
                    if peak > 64 * MIB {
                        return Err(format!("attempt {attempt} charged {} MiB", peak / MIB));
                    }
                }
                Ok(())
            })();
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            result.unwrap();
        });
        // SAFETY: unmaps the mapping created above, which nothing uses any more.
        unsafe { libc::munmap(region, length) };
    }

    /// The worker the tool-case tests supervise: it holds `M2V_TEST_SHARING_MIB` MiB, with
    /// `M2V_TEST_SHARING_CHILD` starts a tool-like process in its own memory and group (as a
    /// spawn is until its exec), writes `M2V_TEST_SHARING_MARKER`, and after 300 ms releases the
    /// tool and exits.
    #[test]
    fn sharing_worker_fixture() {
        let Ok(marker) = std::env::var("M2V_TEST_SHARING_MARKER") else {
            return;
        };
        let mib = std::env::var("M2V_TEST_SHARING_MIB")
            .unwrap()
            .parse::<u64>()
            .unwrap();
        let held = hold(mib * MIB);
        let tool = std::env::var_os("M2V_TEST_SHARING_CHILD")
            .is_some()
            .then(|| SharedMemoryChild::start(None));
        std::fs::write(marker, std::process::id().to_string()).unwrap();
        std::thread::sleep(Duration::from_millis(300));
        drop(tool);
        drop(held);
    }

    /// Supervise the fixture worker holding `mib` MiB (with a tool in its memory when
    /// `sharing_tool`) under the 500-MiB ceiling.
    fn supervise_fixture(mib: u64, sharing_tool: bool) -> Result<u64, ProcessStop> {
        let scratch = crate::ScratchDir::new("process-test").unwrap();
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "worker::process::linux::tests::sharing_worker_fixture",
            ])
            .env("M2V_TEST_SHARING_MARKER", scratch.path().join("ready"))
            .env("M2V_TEST_SHARING_MIB", mib.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        if sharing_tool {
            command.env("M2V_TEST_SHARING_CHILD", "1");
        } else {
            command.env_remove("M2V_TEST_SHARING_CHILD");
        }
        let exit = MonitoredChild::spawn(&mut command)
            .unwrap()
            .wait(&ProcessControl {
                deadline: Some(Instant::now() + Duration::from_secs(20)),
                memory_limit_bytes: Some(CEILING),
                cancelled: None,
            })?;
        assert!(exit.status.success(), "{:?}", exit.status);
        assert!(
            scratch.path().join("ready").exists(),
            "the fixture never got ready"
        );
        Ok(exit.peak_resident_bytes.unwrap_or(0))
    }

    /// The bytes a memory stop measured; any other stop fails the test.
    fn measured_at_stop(stop: &ProcessStop) -> u64 {
        match stop.failure {
            ProcessFailure::MemoryLimitExceeded { measured, .. } => measured,
            _ => panic!("not a memory stop: {stop}"),
        }
    }

    /// One level down: a tool the worker starts runs in the worker's memory until its exec, and
    /// was counted as a second copy of it (a 300-MiB worker read as 600 MiB and stopped). Where
    /// the kernel compares address spaces the tool is charged nothing and the worker once; where
    /// it refuses (a container without `CAP_SYS_PTRACE`), the documented fallback counts the
    /// tool, an overcount, and the test says so.
    #[test]
    fn a_tool_in_its_workers_memory_counts_once() {
        let measured = supervise_fixture(300, true);
        if kcmp_compares() {
            let peak = measured.unwrap();
            assert!(
                (300 * MIB..CEILING).contains(&peak),
                "the worker's 300 MiB, once: {} MiB",
                peak / MIB
            );
        } else {
            eprintln!(
                "kcmp refused here: a not-yet-exec'd tool is counted beside its worker \
                 (the documented fallback)"
            );
            let stop = measured.unwrap_err();
            assert!(measured_at_stop(&stop) >= 600 * MIB, "{stop}");
        }
    }

    /// The ceiling still catches real memory: a worker that holds more than it is stopped,
    /// measured at least at what it holds.
    #[test]
    fn a_worker_over_the_ceiling_is_still_stopped() {
        let stop = supervise_fixture(CEILING / MIB + 100, false).unwrap_err();
        assert!(measured_at_stop(&stop) > CEILING, "{stop}");
    }
}
