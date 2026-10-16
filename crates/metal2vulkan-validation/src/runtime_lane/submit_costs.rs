//! What a compositor's command buffers cost the driver's submission path on the device the cfg
//! selects, without the validation layer: report only, in release builds, to weigh the runtime's
//! submissions before and after a change to them.
//!
//! The workload is `commit_costs`'s guest window server's (its working set, its blits, passes and
//! pointer walk), with the caller's host memory beside it: [`IMPORTS`] no-copy buffers over pages
//! of the process's own, which uploads copy into textures of the working set, and of which
//! [`IMPORTS_A_FRAME`] are released and made again over fresh pages before each frame. Printed as
//! `SUBMIT-COST` lines:
//!
//! - **One command buffer at a time** (`serial`), each shape [`SERIAL`] times after [`WARM`]
//!   untimed: an empty command buffer, a blit encoder, a render pass, a pointer walk, an upload
//!   out of a host import, and a render pass committed after a texture of the working set was made
//!   again (its new resource's first work rides with the commit, `texture/first_move.rs`).
//! - **Frames** (`frames`): [`FRAMES`] frames of [`PER_FRAME`] command buffers committed in a row
//!   and waited for at the frame's end: passes and blits alternating, a pointer walk, two uploads
//!   and an empty command buffer, after `commit_costs`'s churn of the working set and the imports'.
//! - **Frames beside a busy scheduler thread** (`busy scheduler`): the frames again, while a
//!   thread of the test's keeps the device's scheduler thread submitting, as a consumer's events
//!   do: a command buffer of a queue of its own that waits for an event, committed, then released
//!   by the host's signal of the event, which wakes the scheduler's thread to submit it, every
//!   [`SIDE_PACE`]. The committing thread's time off the CPU in `commit` (its wall time less its
//!   CPU time: waits for a lock, and for the CPU), per commit and per frame, the commits a frame
//!   that were off it over [`WAITED_US`], and the commits a frame of either thread handed to the
//!   scheduler's thread (`WorkCounts::commits_handed_off`: every commit).
//!
//! For the first two: the runtime's `vkQueueSubmit` and `vkQueueBindSparse` calls and the `VkCommandBuffer`s
//! they submitted per command buffer (`WorkCounts::queue_submissions`,
//! `command_buffers_submitted`), the committing thread's CPU time in `commit` (which hands the
//! batches to the scheduler's thread, which submits them, `scheduler`), and the process's CPU time
//! per command buffer. Where the i915
//! counter (`scripts/i915-execbuffers/counter.c`) is preloaded: the kernel's execbuffers per
//! command buffer, the buffer objects each lists, by origin (host imports, the runtime's 64-MiB
//! blocks, the rest: ANV's own and the runtime's other blocks), and the microseconds each spent
//! in the ioctl. And the runtime's live `VkDeviceMemory` blocks by kind once the frames ran.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.

use super::commit_costs::{
    percentiles, process_cpu_us, since, texture, Shape, Workload, TEXTURE_SIDE,
};
use super::host_memory::{no_copy, Mapping};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, Device, EventRef, Origin,
    ResourceOptions, Size, WorkCounts,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// The host imports live at once.
const IMPORTS: usize = 32;
/// The bytes of each: a 256-KiB guest surface.
const IMPORT_BYTES: usize = 256 << 10;
/// The imports released and made again before each frame.
const IMPORTS_A_FRAME: usize = 2;
/// Command buffers of each serial shape run untimed first.
const WARM: usize = 20;
/// Command buffers of each serial shape measured.
const SERIAL: usize = 200;
/// Frames measured.
const FRAMES: usize = 60;
/// Command buffers a frame.
const PER_FRAME: usize = 24;
/// The pause between the batches the side thread has the scheduler's thread submit.
const SIDE_PACE: Duration = Duration::from_micros(100);
/// A commit off the CPU for longer than this, in microseconds, waited.
const WAITED_US: f64 = 20.0;

/// What one command buffer of the lane does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// One of `commit_costs`'s shapes.
    Shape(Shape),
    /// A blit of a 64x64 texel square out of a host import into a texture of the working set.
    Upload,
    /// A render pass, committed after a texture of the working set was made again.
    Fresh,
}

/// The counts of the kernel-submission counter (`scripts/i915-execbuffers/counter.c`), as it
/// exports them.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Execbuffers {
    /// The kernel submissions: i915's execbuffers, amdgpu's command submissions.
    pub(super) execbuffers: u64,
    /// The buffer objects they listed.
    pub(super) objects: u64,
    /// The nanoseconds they spent in the ioctl.
    pub(super) nanos: u64,
    /// The listed objects that were host imports (userptrs).
    pub(super) userptr: u64,
    /// The listed objects that were dma-buf imports.
    pub(super) prime: u64,
    block64: u64,
    created_other: u64,
    unknown: u64,
    max_objects: u64,
    min_objects: u64,
}

impl Execbuffers {
    /// The counter's counts so far, where it is loaded.
    pub(super) fn read() -> Option<Execbuffers> {
        // SAFETY: a lookup of a symbol by its name among the process's loaded objects.
        let found = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c"m2v_i915_shim_counts".as_ptr()) };
        if found.is_null() {
            return None;
        }
        // SAFETY: the counter exports the symbol as a function of this signature.
        let read = unsafe {
            std::mem::transmute::<*mut libc::c_void, extern "C" fn(*mut Execbuffers)>(found)
        };
        let mut counts = Execbuffers::default();
        read(&mut counts);
        Some(counts)
    }

    /// What was counted from `before` to `self`, per `commits` command buffers, printed.
    fn since(&self, before: &Execbuffers, commits: usize) -> String {
        let execbuffers = self.execbuffers - before.execbuffers;
        let each = |count: u64| count as f64 / execbuffers.max(1) as f64;
        format!(
            "; {:.2} execbuffers per command buffer, {:.1} objects each (host imports {:.1}, \
             64-MiB blocks {:.1}, dma-bufs {:.1}, other {:.1}), {:.1} us each in the ioctl",
            execbuffers as f64 / commits as f64,
            each(self.objects - before.objects),
            each(self.userptr - before.userptr),
            each(self.block64 - before.block64),
            each(self.prime - before.prime),
            each((self.created_other - before.created_other) + (self.unknown - before.unknown)),
            (self.nanos - before.nanos) as f64 / 1e3 / execbuffers.max(1) as f64,
        )
    }
}

/// The calling thread's CPU time so far, in microseconds.
pub(super) fn thread_cpu_us() -> f64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    let read = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    assert_eq!(read, 0, "the thread's CPU clock");
    now.tv_sec as f64 * 1e6 + now.tv_nsec as f64 / 1e3
}

/// A no-copy buffer of `device` over `bytes` of pages of its own, which its deallocator unmaps.
pub(super) fn import(device: &Device, bytes: usize) -> Buffer {
    let pages = Mapping::anonymous(bytes);
    pages.fill(0x5a);
    let pointer = pages.pointer();
    // SAFETY: the pages are this process's, mapped read and write, and unmapped only by the
    // deallocator, once the runtime let go of them.
    unsafe {
        no_copy(
            device,
            pointer,
            bytes,
            ResourceOptions::shared(),
            Some(Box::new(move |_, _| drop(pages))),
        )
    }
    .unwrap()
}

/// The workload and its host imports.
struct Lane {
    work: Workload,
    imports: Vec<Buffer>,
    next: usize,
}

impl Lane {
    /// Encode `kind` into a new command buffer: the command buffer.
    fn encode(&mut self, kind: Kind) -> CommandBuffer {
        match kind {
            Kind::Shape(shape) => self.work.encode(shape),
            Kind::Upload => {
                self.next = (self.next + 5) % IMPORTS;
                let to = self.work.take(self.work.textures.len());
                let command_buffer = self.work.queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_from_buffer_to_texture(
                    &self.imports[self.next],
                    0,
                    TEXTURE_SIDE * 4,
                    TEXTURE_SIDE * TEXTURE_SIDE * 4,
                    Size::new(TEXTURE_SIDE, TEXTURE_SIDE, 1),
                    &self.work.textures[to],
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    BlitOption::None,
                )
                .unwrap();
                blit.end_encoding().unwrap();
                command_buffer
            }
            Kind::Fresh => {
                let at = self.work.take(self.work.textures.len());
                let rt = TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ);
                self.work.textures[at] = texture(&self.work.device, TEXTURE_SIDE, rt);
                self.work.encode(Shape::Draw)
            }
        }
    }

    /// Release [`IMPORTS_A_FRAME`] imports and make them again over fresh pages.
    fn churn_imports(&mut self, frame: usize) {
        for at in 0..IMPORTS_A_FRAME {
            let at = (frame * IMPORTS_A_FRAME + at) % IMPORTS;
            self.imports[at] = import(&self.work.device, IMPORT_BYTES);
        }
    }
}

/// What a run of `commits` command buffers cost, from the counts before it.
struct Run {
    counts: WorkCounts,
    execbuffers: Option<Execbuffers>,
    process: f64,
}

impl Run {
    fn start(lane: &Lane) -> Run {
        Run {
            counts: lane.work.device.work_counts(),
            execbuffers: Execbuffers::read(),
            process: process_cpu_us(),
        }
    }

    /// Print what the run cost as `what`: `commits` command buffers, whose `commit` calls took
    /// `commit_cpu` microseconds of the committing thread's time each.
    fn print(&self, lane: &Lane, what: &str, commits: usize, commit_cpu: Vec<f64>) {
        let process = (process_cpu_us() - self.process) / commits as f64;
        let after = lane.work.device.work_counts();
        let per = |after: u64, before: u64| (after - before) as f64 / commits as f64;
        let [_, cpu, cpu_90] = percentiles(commit_cpu);
        let execbuffers = Execbuffers::read()
            .zip(self.execbuffers)
            .map(|(after, before)| after.since(&before, commits))
            .unwrap_or_default();
        eprintln!(
            "SUBMIT-COST {}: {what}: {:.2} vkQueueSubmit calls and {:.2} VkCommandBuffers per \
             command buffer; commit cpu {cpu:.1} (p90 {cpu_90:.1}) us; process cpu {process:.1} us \
             per command buffer{execbuffers}",
            lane.work.device.name(),
            per(after.queue_submissions, self.counts.queue_submissions),
            per(
                after.command_buffers_submitted,
                self.counts.command_buffers_submitted
            ),
        );
    }
}

/// Commit `command_buffer`: the committing thread's CPU time in the call, in microseconds.
fn commit(command_buffer: &CommandBuffer) -> f64 {
    timed_commit(command_buffer).0
}

/// Commit `command_buffer`: the committing thread's CPU time in the call and the call's wall
/// time, in microseconds.
fn timed_commit(command_buffer: &CommandBuffer) -> (f64, f64) {
    let (wall, start) = (Instant::now(), thread_cpu_us());
    command_buffer.commit().unwrap();
    (thread_cpu_us() - start, since(wall))
}

/// One frame: each commit's CPU and wall time in `commit`, and the frame's wall time, in
/// microseconds.
struct Frame {
    commits: Vec<(f64, f64)>,
    wall: f64,
}

/// [`FRAMES`] frames of the lane (the module docs).
fn frames(lane: &mut Lane) -> Vec<Frame> {
    (0..FRAMES)
        .map(|frame| {
            lane.work.churn(frame);
            lane.churn_imports(frame);
            let begin = Instant::now();
            let mut last = None;
            let mut commits = Vec::new();
            for at in 0..PER_FRAME {
                let kind = match at {
                    0 => Kind::Shape(Shape::DrawShadowedView),
                    1 => Kind::Shape(Shape::Walk),
                    2 => Kind::Shape(Shape::DrawOwnFormatView),
                    3 | 9 => Kind::Upload,
                    5 => Kind::Shape(Shape::Empty),
                    _ if at % 2 == 0 => Kind::Shape(Shape::Draw),
                    _ => Kind::Shape(Shape::Blit),
                };
                let command_buffer = lane.encode(kind);
                commits.push(timed_commit(&command_buffer));
                last = Some(command_buffer);
            }
            let last = last.expect("a frame's command buffers");
            last.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(last.status(), CommandBufferStatus::Completed);
            Frame {
                commits,
                wall: since(begin),
            }
        })
        .collect()
}

/// The committing thread's time off the CPU in `commit` over `frames`, printed as `what`, with
/// the commits of any thread `handed_off` to the scheduler's thread meanwhile.
fn print_off_cpu(lane: &Lane, what: &str, frames: &[Frame], handed_off: u64) {
    let commits = frames.iter().flat_map(|frame| &frame.commits);
    let off = |&(cpu, wall): &(f64, f64)| (wall - cpu).max(0.0);
    let [_, wall, wall_90] = percentiles(commits.clone().map(|commit| commit.1).collect());
    let [_, each, each_90] = percentiles(commits.clone().map(off).collect());
    let per_frame = frames
        .iter()
        .map(|frame| frame.commits.iter().map(off).sum::<f64>());
    let [_, frame, frame_90] = percentiles(per_frame.collect());
    let waited = commits.filter(|&commit| off(commit) > WAITED_US).count();
    eprintln!(
        "SUBMIT-COST {}: {what}: commit wall {wall:.1} (p90 {wall_90:.1}) us, off cpu {each:.1} \
         (p90 {each_90:.1}) us; off cpu a frame {frame:.0} (p90 {frame_90:.0}) us, {:.2} commits \
         a frame off it over {WAITED_US} us; {:.2} commits a frame handed off",
        lane.work.device.name(),
        waited as f64 / frames.len() as f64,
        handed_off as f64 / frames.len() as f64,
    );
}

/// The commits handed to the scheduler's thread so far (`WorkCounts::commits_handed_off`).
fn handed_off(lane: &Lane) -> u64 {
    lane.work.device.work_counts().commits_handed_off
}

/// Keep the device's scheduler thread busy until `stop` (the module docs): the batches it had
/// it submit.
pub(super) fn side_load(device: &Device, stop: &AtomicBool) -> u64 {
    let queue = device.new_command_queue().unwrap();
    let event = device.new_shared_event().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let mut flight = VecDeque::new();
    let mut value = 0;
    while !stop.load(Ordering::Relaxed) {
        value += 1;
        let command_buffer = queue.command_buffer().unwrap();
        (command_buffer.encode_wait_for_event(EventRef::from(&event), value)).unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..4096, value as u8).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        // Held for the event: the host's signal wakes the scheduler's thread to submit it.
        event.set_signaled_value(value);
        command_buffer.wait_until_scheduled().unwrap();
        flight.push_back(command_buffer);
        if flight.len() > 4 {
            let oldest = flight.pop_front().unwrap();
            oldest.wait_until_completed_for(WAIT).unwrap();
        }
        std::thread::sleep(SIDE_PACE);
    }
    for command_buffer in flight {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
    value
}

/// The submission path's costs, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn submit_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let work = Workload::new(device);
    let imports = (0..IMPORTS)
        .map(|_| import(&work.device, IMPORT_BYTES))
        .collect();
    let mut lane = Lane {
        work,
        imports,
        next: 0,
    };
    let kinds = [
        Kind::Shape(Shape::Empty),
        Kind::Shape(Shape::Blit),
        Kind::Shape(Shape::Draw),
        Kind::Shape(Shape::Walk),
        Kind::Upload,
        Kind::Fresh,
    ];
    for kind in kinds {
        let run_one = |lane: &mut Lane| {
            let command_buffer = lane.encode(kind);
            let cpu = commit(&command_buffer);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            cpu
        };
        for _ in 0..WARM {
            run_one(&mut lane);
        }
        let run = Run::start(&lane);
        let cpu: Vec<f64> = (0..SERIAL).map(|_| run_one(&mut lane)).collect();
        run.print(&lane, &format!("serial {kind:?}"), SERIAL, cpu);
    }
    let run = Run::start(&lane);
    let before = handed_off(&lane);
    let plain = frames(&mut lane);
    let cpu = plain.iter().flat_map(|frame| &frame.commits);
    run.print(
        &lane,
        "frames",
        FRAMES * PER_FRAME,
        cpu.map(|c| c.0).collect(),
    );
    print_off_cpu(&lane, "frames", &plain, handed_off(&lane) - before);
    let [low, wall, high] = percentiles(plain.iter().map(|frame| frame.wall).collect());
    let counts = lane.work.device.work_counts();
    eprintln!(
        "SUBMIT-COST {}: frames: frame {wall:.0} (p10 {low:.0}, p90 {high:.0}) us; live memory \
         blocks: {} pooled, {} pooled empty, {} dedicated, {} host imports",
        lane.work.device.name(),
        counts.memory_blocks_live,
        counts.empty_memory_blocks_live,
        counts.dedicated_memory_blocks_live,
        counts.host_imports_live,
    );
    let before = handed_off(&lane);
    let stop = AtomicBool::new(false);
    let device = lane.work.device.clone();
    let (busy, released) = std::thread::scope(|scope| {
        let side = scope.spawn(|| side_load(&device, &stop));
        let busy = frames(&mut lane);
        stop.store(true, Ordering::Relaxed);
        (busy, side.join().unwrap())
    });
    let what = format!(
        "busy scheduler, {:.1} batches a frame released to its thread",
        released as f64 / FRAMES as f64
    );
    print_off_cpu(&lane, &what, &busy, handed_off(&lane) - before);
}
