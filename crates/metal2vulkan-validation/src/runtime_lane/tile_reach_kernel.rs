//! The tile reach kernel (`bind::tile_reach_module`) on the device the cfg selects, under the
//! validation layer. Each pass of the product's fixture
//! (`fixtures/public/tile_reach/device_passes.txt`: every measured case drawn, each in a slice of
//! its own, and seeded random passes, with the rule's answers; the product's
//! `bind::tile_reach::tests::device` checks it is the rule's, and the kernel's through the CPU
//! interpreter) is dispatched on the device's own `VkQueue` (`CommandQueue::with_vk_queue`), over
//! three of the runtime's `Shared` buffers: the primitives, the header and states, and the mask,
//! cleared. Its mask's set bits and the header's two counts are the rule's, bit for bit. A device
//! without `shaderInt64` is refused `Feature::TileReach`, printed.
//!
//! The kernel's cost on a large pass is measured and printed (`TILEREACH-TIME`), not judged: two
//! seeded passes over a 3840 by 2160 target, a mesh of 18,432 triangles tiling it and a scatter of
//! 16,384 triangles, lines and points of every size. Each pass first runs 64 times back to back in
//! one submission, so the GPU leaves its idle clocks, and then 21 times alone and 32 times back to
//! back, timed by GPU timestamps about the dispatches where the queue writes them and by the
//! host's time from the submission to its fence; a pass of no primitives gives the dispatch's own
//! cost.

use super::degrade_notices::on_device;
use super::{buffer_with, read_buffer};
use crate::candidate::LoaderWord;
use crate::gpu::wait::{self, SubmissionBackend};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::bind::{tile_reach_module, TileReachModule as M};
use metal2vulkan::error::{Error as TranslateError, Feature};
use metal2vulkan_runtime::{Buffer, CommandQueue, Device};
use std::time::{Duration, Instant};

/// The product's fixture (the module docs).
const FIXTURE: &str = include_str!("../../fixtures/public/tile_reach/device_passes.txt");

/// One pass of the fixture: the header's fields, the states' and primitives' words, and the
/// rule's answer.
#[derive(Debug, Default)]
struct Case {
    area: [u32; 2],
    samples: u32,
    layers: u32,
    states: Vec<u32>,
    primitives: Vec<u32>,
    near_far: u32,
    unmeasured: u32,
    bits: Vec<u32>,
}

impl Case {
    fn count(&self) -> u32 {
        (self.primitives.len() / M::PRIMITIVE_WORDS as usize) as u32
    }

    /// The pass buffer's bytes: the header, then the states.
    fn pass_bytes(&self) -> Vec<u8> {
        let header = M::header(self.area, self.samples, self.layers, self.count());
        let states = self.states.iter().flat_map(|w| w.to_le_bytes());
        header.into_iter().chain(states).collect()
    }
}

/// The fixture's passes, in order.
fn cases() -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    let number = |s: &str| s.parse::<u32>().expect("a number");
    let word = |s: &str| u32::from_str_radix(s, 16).expect("a hex word");
    for line in FIXTURE.lines().filter(|l| !l.starts_with('#')) {
        let mut fields = line.split_whitespace();
        let tag = fields.next().expect("a tag");
        let rest: Vec<&str> = fields.collect();
        if tag == "pass" {
            let [w, h, samples, layers] = [0, 1, 2, 3].map(|i| number(rest[i]));
            cases.push(Case {
                area: [w, h],
                samples,
                layers,
                ..Case::default()
            });
            continue;
        }
        let case = cases.last_mut().expect("a pass first");
        match tag {
            "state" => case.states.extend(rest.iter().map(|s| word(s))),
            "primitive" => case.primitives.extend(rest.iter().map(|s| word(s))),
            "expect" => [case.near_far, case.unmeasured] = [0, 1].map(|i| number(rest[i])),
            "bits" => case.bits.extend(rest.iter().map(|s| number(s))),
            other => panic!("an unknown line {other:?}"),
        }
    }
    cases
}

/// The test's Vulkan objects on the runtime's device: the kernel's pipeline, a descriptor set of
/// its three storage buffers, a command buffer, a fence and two timestamp queries. Destroyed on
/// drop, after its last submission was waited for.
struct Kernel {
    driver: ash::Device,
    word: LoaderWord,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptors: vk::DescriptorPool,
    set: vk::DescriptorSet,
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
    queries: vk::QueryPool,
    /// The queue family's `timestampValidBits` (0: it writes none) and the device's
    /// `timestampPeriod`, nanoseconds a tick.
    stamp_bits: u32,
    period: f64,
}

/// What one dispatch took: the host's time from its submission to its fence, and the GPU's
/// between the timestamps about it, where the queue writes them.
#[derive(Clone, Copy, Debug)]
struct Took {
    wall: Duration,
    gpu: Option<Duration>,
}

impl Kernel {
    /// Made on `device` before it submitted anything, so the loader word is the device's own.
    fn new(device: &Device, spirv: &[u32]) -> Kernel {
        // SAFETY: read while the device lives; the objects are destroyed in `drop`, before it.
        let raw = unsafe { device.raw() };
        let driver = raw.device.clone();
        // SAFETY: the handle is the live device's.
        let word = unsafe { LoaderWord::of_device(driver.handle()) };
        let info = vk::ShaderModuleCreateInfo::default().code(spirv);
        // SAFETY: the device is live; `info` points at the module's words for the call.
        let module = unsafe { driver.create_shader_module(&info, None) }.unwrap();
        let bindings = [M::PRIMITIVES_BINDING, M::PASS_BINDING, M::TILES_BINDING].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: the device is live.
        let set_layout = unsafe { driver.create_descriptor_set_layout(&info, None) }.unwrap();
        let set_layouts = [set_layout];
        let info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: the device is live.
        let layout = unsafe { driver.create_pipeline_layout(&info, None) }.unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: the device is live; every pointer in `info` outlives the call.
        let pipeline =
            unsafe { driver.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) }
                .map_err(|(_, error)| error)
                .unwrap()[0];
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: the device is live.
        let descriptors = unsafe { driver.create_descriptor_pool(&info, None) }.unwrap();
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptors)
            .set_layouts(&set_layouts);
        // SAFETY: the device is live; the pool holds one set of the layout's.
        let set = unsafe { driver.allocate_descriptor_sets(&info) }.unwrap()[0];
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(raw.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: the device is live; the pool is the test's own.
        let pool = unsafe { driver.create_command_pool(&info, None) }.unwrap();
        let allocation = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: the device is live; the loader serves the call after the word is put back.
        let buffer = unsafe {
            word.restore();
            #[expect(
                clippy::disallowed_methods,
                reason = "the runtime's device, after the loader word is put back"
            )]
            let buffer = driver.allocate_command_buffers(&allocation);
            buffer
        }
        .unwrap()[0];
        // SAFETY: the device is live.
        let fence = unsafe { driver.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2);
        // SAFETY: the device is live.
        let queries = unsafe { driver.create_query_pool(&info, None) }.unwrap();
        // SAFETY: instance-level queries of the live device's physical device.
        let (stamp_bits, period) = unsafe {
            let families = raw
                .instance
                .get_physical_device_queue_family_properties(raw.physical_device);
            let limits = raw
                .instance
                .get_physical_device_properties(raw.physical_device)
                .limits;
            (
                families[raw.queue_family as usize].timestamp_valid_bits,
                f64::from(limits.timestamp_period),
            )
        };
        Kernel {
            driver,
            word,
            module,
            set_layout,
            layout,
            pipeline,
            descriptors,
            set,
            pool,
            buffer,
            fence,
            queries,
            stamp_bits,
            period,
        }
    }

    /// Bind `buffers` (primitives, pass, mask), dispatch `groups` workgroups `times` times back
    /// to back (a barrier between) on `queue`'s `VkQueue` and wait for them: the mask and the
    /// counts are then the host's to read. The times taken are the whole submission's.
    fn run(&self, queue: &CommandQueue, buffers: [&Buffer; 3], groups: u32, times: u32) -> Took {
        let driver = &self.driver;
        let infos = buffers.map(|b| {
            // SAFETY: the buffer lives past the submission, which is waited for below.
            let (handle, offset) = unsafe { b.vk_buffer() };
            [vk::DescriptorBufferInfo::default()
                .buffer(handle)
                .offset(offset)
                .range(b.length())]
        });
        let bindings = [M::PRIMITIVES_BINDING, M::PASS_BINDING, M::TILES_BINDING];
        let writes: Vec<vk::WriteDescriptorSet<'_>> = (0..3)
            .map(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(bindings[i])
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i])
            })
            .collect();
        let after = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the set and the command buffer are the test's own and not in use (the last
        // submission was waited for); the buffers are live and no work of the runtime's touches
        // them (the host wrote them, which the submission makes visible).
        unsafe {
            driver.update_descriptor_sets(&writes, &[]);
            driver.begin_command_buffer(self.buffer, &begin).unwrap();
            driver.cmd_reset_query_pool(self.buffer, self.queries, 0, 2);
            driver.cmd_bind_pipeline(self.buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            driver.cmd_bind_descriptor_sets(
                self.buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                M::SET,
                &[self.set],
                &[],
            );
            let top = vk::PipelineStageFlags::TOP_OF_PIPE;
            driver.cmd_write_timestamp(self.buffer, top, self.queries, 0);
            for at in 0..times {
                if at > 0 {
                    let between = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        );
                    let compute = vk::PipelineStageFlags::COMPUTE_SHADER;
                    driver.cmd_pipeline_barrier(
                        self.buffer,
                        compute,
                        compute,
                        vk::DependencyFlags::empty(),
                        &[between],
                        &[],
                        &[],
                    );
                }
                driver.cmd_dispatch(self.buffer, groups, 1, 1);
            }
            let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
            driver.cmd_write_timestamp(self.buffer, bottom, self.queries, 1);
            driver.cmd_pipeline_barrier(
                self.buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
            driver.end_command_buffer(self.buffer).unwrap();
        }
        let buffers = [self.buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&buffers);
        let submitted = Instant::now();
        // SAFETY: the command buffer is executable and the fence unsignaled; the submission is
        // made under the runtime's lock of its `VkQueue`, and the call waits for nothing.
        unsafe {
            queue.with_vk_queue(|raw, _| driver.queue_submit(raw, &[submit], self.fence).unwrap());
        }
        let backend = SubmissionBackend::vulkan(TEST_DEVICE);
        let operation = "the tile reach kernel's dispatch";
        wait::vulkan::wait_for_fence(driver, self.fence, backend, operation).unwrap();
        let wall = submitted.elapsed();
        let gpu = (self.stamp_bits > 0).then(|| {
            let mut stamps = [0u64; 2];
            // SAFETY: the submission that wrote both queries completed.
            unsafe {
                driver.get_query_pool_results(
                    self.queries,
                    0,
                    &mut stamps,
                    vk::QueryResultFlags::TYPE_64,
                )
            }
            .unwrap();
            let mask = match self.stamp_bits {
                64.. => u64::MAX,
                bits => (1 << bits) - 1,
            };
            let ticks = stamps[1].wrapping_sub(stamps[0]) & mask;
            Duration::from_nanos((ticks as f64 * self.period).round() as u64)
        });
        // SAFETY: the submission completed: the fence and the command buffer are free.
        unsafe {
            driver.reset_fences(&[self.fence]).unwrap();
            driver
                .reset_command_buffer(self.buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        }
        Took { wall, gpu }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: nothing of the test's is pending (each submission was waited for).
        unsafe {
            self.word.restore();
            self.driver.destroy_query_pool(self.queries, None);
            self.driver.destroy_fence(self.fence, None);
            self.driver.destroy_command_pool(self.pool, None);
            self.driver.destroy_descriptor_pool(self.descriptors, None);
            self.driver.destroy_pipeline(self.pipeline, None);
            self.driver.destroy_pipeline_layout(self.layout, None);
            self.driver
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.driver.destroy_shader_module(self.module, None);
        }
    }
}

/// The set bits of little-endian mask words.
fn set_bits(bytes: &[u8]) -> Vec<u32> {
    let mut bits = Vec::new();
    for (i, w) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let w = u32::from_le_bytes(*w);
        bits.extend(
            (0..32)
                .filter(|b| w >> b & 1 == 1)
                .map(|b| 32 * i as u32 + b),
        );
    }
    bits
}

/// Every pass of the fixture binned on the device: its mask's set bits and the header's near-far
/// and unmeasured counts are the rule's (the module docs).
#[test]
fn the_tile_reach_kernel_bins_the_fixture_as_the_rule_does() {
    on_device(|device, _messages| {
        let module = match tile_reach_module(device.compile_device()) {
            Ok(module) => module,
            Err(TranslateError::Unsupported(u)) if u.feature == Feature::TileReach => {
                eprintln!("TILEREACH: refused on a device without shaderInt64: {u:?}");
                return;
            }
            Err(error) => panic!("tile_reach_module: {error}"),
        };
        let kernel = Kernel::new(device, &module.spirv);
        let queue = device.new_command_queue().unwrap();
        let cases = cases();
        let mut wrong = Vec::new();
        let mut primitives = 0;
        for (at, case) in cases.iter().enumerate() {
            let prims: Vec<u8> = case
                .primitives
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let words = M::mask_words(case.area, case.samples, case.layers).max(1) as usize;
            let prims = buffer_with(device, &prims).unwrap();
            let pass = buffer_with(device, &case.pass_bytes()).unwrap();
            let tiles = buffer_with(device, &vec![0; 4 * words]).unwrap();
            let groups = case.count().div_ceil(M::LOCAL_SIZE).max(1);
            kernel.run(&queue, [&prims, &pass, &tiles], groups, 1);
            let header = read_buffer(&pass, 0, 4 * u64::from(M::HEADER_WORDS)).unwrap();
            let count = |w: u32| {
                let at = 4 * w as usize;
                u32::from_le_bytes(header[at..at + 4].try_into().expect("a word"))
            };
            let mask = read_buffer(&tiles, 0, 4 * words as u64).unwrap();
            let got = (
                count(M::NEAR_FAR_WORD),
                count(M::UNMEASURED_WORD),
                set_bits(&mask),
            );
            let want = (case.near_far, case.unmeasured, case.bits.clone());
            if got != want {
                let differ: Vec<_> = got
                    .2
                    .iter()
                    .filter(|b| !want.2.contains(b))
                    .chain(want.2.iter().filter(|b| !got.2.contains(b)))
                    .take(16)
                    .collect();
                wrong.push(format!(
                    "pass {at} ({:?}, {} samples): counts {:?}, the rule's {:?}; bits differing \
                     (first 16) {differ:?}",
                    case.area,
                    case.samples,
                    (got.0, got.1),
                    (want.0, want.1)
                ));
            }
            primitives += case.count();
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
        eprintln!(
            "TILEREACH: {} passes, {primitives} primitives: the rule's on {}",
            cases.len(),
            device.name()
        );
    });
}

/// The timed passes' target (the module docs).
const TARGET: [u32; 2] = [3840, 2160];

/// A seeded generator (splitmix64) of the timed passes.
struct Seeded(u64);

impl Seeded {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `[lo, hi)`.
    fn within(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The clip-space position of the target's pixel `(x, y)` at `w` and depth `z / w` (the whole
/// target's viewport).
fn clip_at(x: f64, y: f64, z: f64, w: f64) -> [f32; 4] {
    let [tw, th] = TARGET.map(f64::from);
    let xn = 2.0 * x / tw - 1.0;
    let yn = 1.0 - 2.0 * y / th;
    [(xn * w) as f32, (yn * w) as f32, (z * w) as f32, w as f32]
}

/// A timed pass: its primitives' and its pass buffer's bytes, and its count.
struct Large {
    what: &'static str,
    prims: Vec<u8>,
    count: u32,
}

impl Large {
    /// The pass buffer: the header and the one state (the whole target, no culling, depth
    /// clipped), of `count` primitives.
    fn pass_bytes(&self, count: u32) -> Vec<u8> {
        let [w, h] = TARGET;
        let viewport = [0.0, 0.0, w as f32, h as f32];
        let state = M::state(viewport, [0, 0, w, h], 0, 0, 0);
        let header = M::header(TARGET, 1, 1, count);
        header.into_iter().chain(state).collect()
    }
}

/// A mesh of two triangles a cell of 30 by 30 pixels tiling the target, its depth `w` rising
/// from 1 to 4 across it.
fn mesh() -> Large {
    let (cols, rows) = (TARGET[0] / 30, TARGET[1] / 30);
    let corner = |c: u32, r: u32| {
        let w = 1.0 + 3.0 * f64::from(c) / f64::from(cols);
        clip_at(f64::from(30 * c), f64::from(30 * r), 0.5, w)
    };
    let mut prims = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let [a, b, d, e] = [
                corner(c, r),
                corner(c + 1, r),
                corner(c, r + 1),
                corner(c + 1, r + 1),
            ];
            prims.extend(M::primitive(M::TRIANGLE, 0, 0, 0.0, [a, b, d]));
            prims.extend(M::primitive(M::TRIANGLE, 0, 0, 0.0, [b, e, d]));
        }
    }
    Large {
        what: "mesh",
        count: 2 * cols * rows,
        prims,
    }
}

/// 16,384 primitives about the target and past its edges: nine in ten triangles of sides from
/// one pixel to 400 (3% across the near plane), the rest lines and points, at depths `w` from
/// 0.5 to 8.
fn scatter() -> Large {
    let mut rng = Seeded(0x7111_e7e5);
    let count = 16_384;
    let [tw, th] = TARGET.map(f64::from);
    let mut prims = Vec::new();
    for _ in 0..count {
        let (cx, cy) = (rng.within(-64.0, tw + 64.0), rng.within(-64.0, th + 64.0));
        let size = rng.within(0.0, 400f64.ln()).exp();
        let w = rng.within(0.5, 8.0);
        let kind = rng.within(0.0, 1.0);
        let crossing = rng.within(0.0, 1.0) < 0.03;
        let mut vertex = |z: f64| {
            let x = cx + size * rng.within(-1.0, 1.0);
            let y = cy + size * rng.within(-1.0, 1.0);
            clip_at(x, y, z, w)
        };
        let prim = if kind < 0.9 {
            let first = vertex(if crossing { -0.3 } else { 0.5 });
            M::primitive(M::TRIANGLE, 0, 0, 0.0, [first, vertex(0.5), vertex(0.5)])
        } else if kind < 0.95 {
            let ends = [vertex(0.5), vertex(0.5)];
            M::primitive(M::LINE, 0, 0, 0.0, [ends[0], ends[1], [0.0; 4]])
        } else {
            let at = vertex(0.5);
            let size = rng.within(1.0, 64.0) as f32;
            M::primitive(M::POINT, 0, 0, size, [at, [0.0; 4], [0.0; 4]])
        };
        prims.extend(prim);
    }
    Large {
        what: "scatter",
        count,
        prims,
    }
}

/// The median and the least of `durations`, in microseconds.
fn spread(mut durations: Vec<Duration>) -> (f64, f64) {
    durations.sort();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    (us(durations[durations.len() / 2]), us(durations[0]))
}

/// The kernel's cost on the large passes (the module docs): printed, `TILEREACH-TIME`.
#[test]
fn the_tile_reach_kernel_times_a_large_pass() {
    const WARM: u32 = 64;
    const RUNS: usize = 21;
    const BATCH: u32 = 32;
    on_device(|device, _messages| {
        let module = match tile_reach_module(device.compile_device()) {
            Ok(module) => module,
            Err(TranslateError::Unsupported(u)) if u.feature == Feature::TileReach => {
                eprintln!("TILEREACH-TIME: refused on a device without shaderInt64: {u:?}");
                return;
            }
            Err(error) => panic!("tile_reach_module: {error}"),
        };
        let kernel = Kernel::new(device, &module.spirv);
        let queue = device.new_command_queue().unwrap();
        let words = M::mask_words(TARGET, 1, 1) as usize;
        let tiles = TARGET[0].div_ceil(32) * TARGET[1].div_ceil(32);
        for large in [mesh(), scatter()] {
            let prims = buffer_with(device, &large.prims).unwrap();
            for (count, what) in [(0, "none"), (large.count, large.what)] {
                let mut walls = Vec::with_capacity(RUNS);
                let mut gpus = Vec::with_capacity(RUNS);
                let mut reached = (0, 0, 0);
                let groups = count.div_ceil(M::LOCAL_SIZE).max(1);
                let mask = buffer_with(device, &vec![0; 4 * words]).unwrap();
                let pass = buffer_with(device, &large.pass_bytes(count)).unwrap();
                kernel.run(&queue, [&prims, &pass, &mask], groups, WARM);
                let batch = kernel.run(&queue, [&prims, &pass, &mask], groups, BATCH);
                for _ in 0..RUNS {
                    let pass = buffer_with(device, &large.pass_bytes(count)).unwrap();
                    let mask = buffer_with(device, &vec![0; 4 * words]).unwrap();
                    let took = kernel.run(&queue, [&prims, &pass, &mask], groups, 1);
                    walls.push(took.wall);
                    gpus.extend(took.gpu);
                    let header = read_buffer(&pass, 0, 4 * u64::from(M::HEADER_WORDS)).unwrap();
                    let word = |w: u32| {
                        let at = 4 * w as usize;
                        u32::from_le_bytes(header[at..at + 4].try_into().expect("a word"))
                    };
                    let bits = read_buffer(&mask, 0, 4 * words as u64).unwrap();
                    let set = set_bits(&bits).len();
                    reached = (set, word(M::NEAR_FAR_WORD), word(M::UNMEASURED_WORD));
                }
                if what == "mesh" {
                    assert_eq!(
                        reached.0, tiles as usize,
                        "a mesh tiling the target reaches it all"
                    );
                }
                let (wall, wall_least) = spread(walls);
                let per = |d: Duration| d.as_secs_f64() * 1e6 / f64::from(BATCH);
                let gpu = match (gpus.len(), batch.gpu) {
                    (0, _) | (_, None) => "no timestamps".to_string(),
                    (_, Some(back_to_back)) => {
                        let (median, least) = spread(gpus);
                        format!(
                            "GPU median {median:.1} us, least {least:.1} us, {:.1} us a \
                             dispatch back to back",
                            per(back_to_back)
                        )
                    }
                };
                eprintln!(
                    "TILEREACH-TIME: {what}: {count} primitives over {}x{} on {}: {gpu}; \
                     submission to fence median {wall:.1} us, least {wall_least:.1} us, \
                     {:.1} us a dispatch back to back ({RUNS} runs alone, {BATCH} back to \
                     back, after {WARM}); {} of {tiles} tiles, near-far {}, unmeasured {}",
                    TARGET[0],
                    TARGET[1],
                    device.name(),
                    per(batch.wall),
                    reached.0,
                    reached.1,
                    reached.2
                );
            }
        }
    });
}
