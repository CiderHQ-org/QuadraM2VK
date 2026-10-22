// GPL-3.0-only. Exercise the actual bounded wait on owned LLVM fixtures, in release mode.
#[path = "../metal2vulkan/crates/metal2vulkan-devtools/src/process.rs"]
mod process;

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(command: &mut Command) {
    let mut child = command.stdin(Stdio::null()).spawn().unwrap();
    match process::wait_for_child(&mut child, Duration::from_secs(20)).unwrap() {
        Some(status) => assert!(status.success(), "tool failed: {command:?}"),
        None => {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("tool exceeded the unchanged 20 second limit: {command:?}");
        }
    }
}

fn main() {
    let directory = std::env::temp_dir().join(format!("quadram2vk-wait-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let scratch = Scratch(directory);
    let bitcode = scratch.0.join("owned.bc");
    for source in std::env::args_os().skip(1).map(PathBuf::from) {
        let started = Instant::now();
        run(Command::new("llvm-as").arg(&source).arg("-o").arg(&bitcode));
        run(Command::new("llvm-dis").arg(&bitcode).arg("-o").arg("/dev/null"));
        println!(
            "{}\t{}\t{:.6}",
            source.display(),
            std::fs::metadata(&source).unwrap().len(),
            started.elapsed().as_secs_f64() * 1000.0
        );
        std::fs::remove_file(&bitcode).unwrap();
    }
}
