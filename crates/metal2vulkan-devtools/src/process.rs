//! Bounded child waits without a process-wide signal handler or socket wakeups.
use std::io;
use std::process::{Child, ExitStatus};
use std::time::{Duration, Instant};

/// Wait up to `timeout` for `child`, without changing its signal handlers or requiring sockets.
/// Returns `None` at the deadline, leaving the caller to kill and reap the child and its process
/// group as its worker contract requires. A completed child is reaped, including at a zero timeout.
pub fn wait_for_child(child: &mut Child, timeout: Duration) -> io::Result<Option<ExitStatus>> {
    let deadline = Instant::now().checked_add(timeout).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "child wait deadline overflow")
    })?;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Ok(None);
        }
        std::thread::sleep(left.min(Duration::from_millis(1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    struct Reaped(Child);
    impl Drop for Reaped {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[test]
    #[ignore = "child process for bounded_wait_leaves_a_running_child_for_its_owner"]
    fn parked_child() {
        std::thread::sleep(Duration::from_secs(30));
    }

    #[test]
    fn bounded_wait_leaves_a_running_child_for_its_owner() {
        let mut child = Reaped(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "process::tests::parked_child", "--ignored"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        assert!(wait_for_child(&mut child.0, Duration::ZERO)
            .unwrap()
            .is_none());
        assert!(wait_for_child(&mut child.0, Duration::from_millis(5))
            .unwrap()
            .is_none());
        assert!(child.0.try_wait().unwrap().is_none());
        child.0.kill().unwrap();
        assert!(wait_for_child(&mut child.0, Duration::from_secs(10))
            .unwrap()
            .is_some());
    }

    #[test]
    fn a_finished_child_is_reaped_and_stays_finished() {
        let mut child = Reaped(
            Command::new(std::env::current_exe().unwrap())
                .arg("--list")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let status = wait_for_child(&mut child.0, Duration::from_secs(10))
            .unwrap()
            .unwrap();
        assert!(status.success());
        assert_eq!(
            wait_for_child(&mut child.0, Duration::ZERO).unwrap(),
            Some(status)
        );
    }
}
