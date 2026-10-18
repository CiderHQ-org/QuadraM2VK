//! Each pipeline making's own `VkPipelineCache` and its record (the runtime's
//! `shader_cache/driver.rs`), on the device the cfg selects, under the validation layer. It
//! replaced the device's one driver cache entry and the variant caches of a driver that compiles
//! saved data at load, on every driver:
//!
//! - **Seed and hit.** A making with no record writes one, named by what it reads: the state's
//!   creation writes its first draw's, a draw of points another (another key of the same
//!   stages). A later device over the same directory makes each from its record, the first at
//!   the state's creation and the points' at their draw (nothing is kept from earlier draws): both
//!   loaded, both the driver's cache hits where it reports them
//!   (`VK_EXT_pipeline_creation_feedback`: the whole pipeline's hit, or every stage's, as the
//!   runtime counts them; MoltenVK 1.4.1 says it for each stage alone), and each draw's target
//!   byte for byte the first device's. A making from a record writes it again exactly where the driver's data after the
//!   making is no longer the record's bytes (RADV's after its first hit from a compile's record is
//!   not, and settles), and counts every other read unchanged (`driver_data_unchanged`).
//! - **Overwrite.** A record holding another pipeline's data (a dispatch's compute pipeline's
//!   record copied over each of a render state's) is loaded: the driver's miss, and where the
//!   driver's data holds its pipelines the making adds its own, so the record is written again,
//!   holding this pipeline now. A later device then hits it, writing it again only where the data
//!   changed. The other pipeline is a compute pipeline, which no making of a render state is on
//!   any driver: another render state's would not do, since a driver may key that state's
//!   libraries as this one's (NVIDIA does for `RGBA8Unorm` and `BGRA8Unorm` targets) and its
//!   data for one pipeline need not be the same bytes from one making to the next (NVIDIA
//!   615.71.09's on an RTX 3090 and an RTX 5080 Laptop are not), so no comparison of two
//!   makings' bytes tells whether one's data holds the other's pipeline.
//! - **Makings at once.** Two processes making the same pipeline on one directory at the same
//!   time leave one whole record of it and no temporary file, which a later device hits.
//!
//! **Makings where the device links libraries**: a pipeline there is linked from libraries of its
//! parts, and each library of shaders is a making of its own with its record ([`Makings`]): a
//! state's creation makes two, its pre-rasterization and fragment shader libraries, and the points'
//! draw one, its pre-rasterization library (Metal biases depth for triangles only, so the points'
//! pre-rasterization state is another, and their fragment's is the state's). Every count above is
//! of those makings there; a library's record written over with the compute pipeline's data holds
//! another pipeline's data, whichever library it is. The driver reports no hit or miss of a
//! library, which [`assert_reported`] takes.
//!
//! **The driver's own cache is fresh for each device**: Mesa's and NVIDIA's shader disk caches are
//! pointed into a new scratch directory before each device or maker process is made
//! ([`DriverCache`]), so a making with no record is the driver's miss and a hit is a hit from
//! the runtime's record, whatever this host's drivers compiled before.
//!
//! What the driver does with its data is read from the driver, not from the runtime's records:
//! [`DriverData`] makes caches and two compute pipelines through raw Vulkan on the test's device
//! and reads `vkGetPipelineCacheData` of each. Where that data ends inside the header it states
//! (lavapipe of Mesa 26.2.4 native), the runtime keeps no record: each making counts that reason
//! (`driver_data_truncated`), no record is written, and the targets still match. Where the two
//! pipelines' data differ, the driver's data holds its pipelines: every record holds data other
//! than an empty cache's, two makings' records differ, and a later making from one is the driver's
//! hit. Where they are the same bytes (Mesa 26.0.8's lavapipe), every record holds those bytes, a
//! making from one is a miss, and a record copied over another's is that one's own bytes, which the
//! making loads and leaves as they were. Every record's frame (magic, the driver's identity,
//! length, SHA-256) is checked. The pipelines are the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll`. The tests print the driver's data and
//! each device's counts (`PIPELINE-RECORDS`).

use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    BlitOption, ClearColor, CommandBufferStatus, Device, Library, LoadAction, Origin,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    ShaderCacheStatistics, Size, StoreAction, TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The render target's side.
const SIDE: u64 = 8;
/// The magic a making's record begins with (`shader_cache/driver.rs`).
const RECORD_MAGIC: &[u8] = b"M2VPPLC\0";
/// The environment variable that names the cache directory to [`record_maker_process`].
const MAKER_DIR: &str = "M2V_LANE_PIPELINE_RECORDS_DIR";
/// The line [`record_maker_process`] reports on.
const MAKER_REPORT: &str = "PIPELINE-RECORDS-MAKER";

/// The owned fixture's library on `device`.
fn library(device: &Device) -> Library {
    let ll =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public/examples/runtime_examples.ll");
    let bitcode = llvm_assemble(&ll).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The render pipeline state of the fixture's `colored_vertex` and `colored_fragment`, one
/// target of `format`, of an unspecified topology class.
fn state(device: &Device, library: &Library, format: PixelFormat) -> RenderPipelineState {
    let vertex = library.new_function_with_name("colored_vertex").unwrap();
    let fragment = library.new_function_with_name("colored_fragment").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The driver's makings that keep a record (the module docs): of a state's creation, and of the
/// points' draw after it.
#[derive(Clone, Copy, Debug)]
struct Makings {
    creation: u64,
    points: u64,
}

impl Makings {
    /// Those of `device`, which has made a state: one whole pipeline each where it makes its
    /// pipelines whole; where it links them from libraries (it linked the state's), the state's
    /// pre-rasterization and fragment shader libraries, and the points' pre-rasterization one.
    fn of(device: &Device) -> Makings {
        match device.work_counts().pipelines_linked {
            0 => Makings {
                creation: 1,
                points: 1,
            },
            _ => Makings {
                creation: 2,
                points: 1,
            },
        }
    }
}

/// The records (`M2VPPLC\0`) under `cache`'s layout directory, by their paths under it, with
/// their bytes.
fn records(cache: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut found = BTreeMap::new();
    let Ok(dirs) = std::fs::read_dir(cache.join("v1")) else {
        return found;
    };
    for dir in dirs {
        let dir = dir.unwrap().path();
        if !dir.is_dir() {
            continue;
        }
        for group in std::fs::read_dir(dir).unwrap() {
            let group = group.unwrap().path();
            if !group.is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(group).unwrap() {
                let path = entry.unwrap().path();
                let bytes = std::fs::read(&path).unwrap();
                if bytes.starts_with(RECORD_MAGIC) {
                    found.insert(path.strip_prefix(cache).unwrap().to_path_buf(), bytes);
                }
            }
        }
    }
    found
}

/// The temporary files a writer leaves beside an entry while it writes (`.<name>.<..>.tmp`)
/// under `cache`'s layout directory.
fn temporaries(cache: &Path) -> usize {
    let Ok(dirs) = std::fs::read_dir(cache.join("v1")) else {
        return 0;
    };
    let mut found = 0;
    for dir in dirs.flatten().filter(|dir| dir.path().is_dir()) {
        for group in std::fs::read_dir(dir.path()).unwrap().flatten() {
            if !group.path().is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(group.path()).unwrap().flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                found += usize::from(name.starts_with('.') && name.ends_with(".tmp"));
            }
        }
    }
    found
}

/// Draw six vertices over a new target of `format` as `primitive` with `state` and wait: the
/// target's bytes.
fn draw(
    device: &Device,
    state: &RenderPipelineState,
    (format, primitive): (PixelFormat, PrimitiveType),
) -> Vec<u8> {
    let mut descriptor = TextureDescriptor::texture_2d(format, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::private();
    let target = device.new_texture(&descriptor).unwrap();
    // A list's two triangles over the whole target; points at their corners.
    let corners: [[f32; 2]; 6] = [
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ];
    let positions: Vec<u8> = corners
        .iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let colors: Vec<u8> = (0..corners.len())
        .flat_map(|i| [1.0, 0.2, i as f32 / 8.0, 1.0f32])
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let positions = device
        .new_buffer_with_bytes(&positions, ResourceOptions::shared())
        .unwrap();
    let colors = device
        .new_buffer_with_bytes(&colors, ResourceOptions::shared())
        .unwrap();
    let row = 4 * SIDE;
    let readback = device
        .new_buffer(row * SIDE, ResourceOptions::shared())
        .unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
    encoder.set_vertex_buffer(Some(&colors), 0, 1).unwrap();
    encoder.draw_primitives(primitive, 0, 6, 1, 0).unwrap();
    encoder.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &target,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(SIDE, SIDE, 1),
        &readback,
        0,
        row,
        row * SIDE,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let contents = readback.contents().unwrap();
    // SAFETY: the buffer's contents hold its `row * SIDE` bytes, and the command buffer that
    // wrote them completed.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), (row * SIDE) as usize) }.to_vec()
}

/// Dispatch the fixture's `scale_and_offset` over four values and wait: the making of its
/// compute pipeline, which no making of a render state is.
fn dispatch(device: &Device, library: &Library) {
    let function = library.new_function_with_name("scale_and_offset").unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let input = device
        .new_buffer_with_bytes(&[0u8; 16], ResourceOptions::shared())
        .unwrap();
    let output = device.new_buffer(16, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&output), 0, 1).unwrap();
    encoder.set_bytes(&1.0f32.to_le_bytes(), 2).unwrap();
    encoder
        .dispatch_threads(Size::new(4, 1, 1), Size::new(4, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The driver cache counts of `statistics`, printed for `which`.
fn report(which: &str, s: &ShaderCacheStatistics) {
    println!(
        "PIPELINE-RECORDS {which}: loads={} saves={} pipeline_hits={} pipeline_misses={} \
         truncated={} mismatched={} rejected={} write_failures={}",
        s.driver_loads,
        s.driver_saves,
        s.driver_pipeline_hits,
        s.driver_pipeline_misses,
        s.driver_data_truncated,
        s.driver_data_mismatched,
        s.rejected,
        s.write_failures,
    );
}

/// Whether the device of `statistics`, after its first making, keeps the driver's data: no read
/// of it was refused. Every read refused is a driver that keeps nothing.
fn kept(statistics: &ShaderCacheStatistics) -> bool {
    assert_eq!(statistics.driver_data_mismatched, 0, "{statistics:?}");
    assert_eq!(statistics.write_failures, 0, "{statistics:?}");
    statistics.driver_data_truncated == 0
}

/// A compute shader that does nothing, in threadgroups of `width` threads: SPIR-V 1.0, written out
/// here so the probe needs no tool.
#[rustfmt::skip]
fn empty_kernel(width: u32) -> [u32; 35] {
    [
        0x0723_0203, 0x0001_0000, 0, 5, 0, // the header: SPIR-V 1.0, ids below 5
        0x0002_0011, 1, // OpCapability Shader
        0x0003_000e, 0, 1, // OpMemoryModel Logical GLSL450
        0x0005_000f, 5, 1, 0x6e69_616d, 0, // OpEntryPoint GLCompute %1 "main"
        0x0006_0010, 1, 17, width, 1, 1, // OpExecutionMode %1 LocalSize width 1 1
        0x0002_0013, 2, // %2 = OpTypeVoid
        0x0003_0021, 3, 2, // %3 = OpTypeFunction %2
        0x0005_0036, 2, 1, 0, 3, // %1 = OpFunction %2 None %3
        0x0002_00f8, 4, // %4 = OpLabel
        0x0001_00fd, // OpReturn
        0x0001_0038, // OpFunctionEnd
    ]
}

/// `vkGetPipelineCacheData` of a new cache on `raw` after it made a pipeline of [`empty_kernel`]
/// of `width` threads, or nothing: the size the driver's size query stated, and the bytes it then
/// wrote.
fn driver_data(raw: &ash::Device, width: Option<u32>) -> (usize, Vec<u8>) {
    // SAFETY: every object is made here on the live device, used only while it lives, and
    // destroyed after its last use; no queue is used.
    unsafe {
        let cache = raw
            .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
            .unwrap();
        if let Some(width) = width {
            let words = empty_kernel(width);
            let module = raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .unwrap();
            let layout = raw
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
                .unwrap();
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let pipelines = raw
                .create_compute_pipelines(cache, &[info], None)
                .map_err(|(_, error)| error)
                .unwrap();
            for pipeline in pipelines {
                raw.destroy_pipeline(pipeline, None);
            }
            raw.destroy_pipeline_layout(layout, None);
            raw.destroy_shader_module(module, None);
        }
        let get = raw.fp_v1_0().get_pipeline_cache_data;
        let mut stated = 0usize;
        get(raw.handle(), cache, &mut stated, std::ptr::null_mut())
            .result()
            .unwrap();
        let mut data = vec![0u8; stated];
        let mut written = stated;
        let read = get(raw.handle(), cache, &mut written, data.as_mut_ptr().cast());
        assert_eq!(read, vk::Result::SUCCESS, "vkGetPipelineCacheData");
        data.truncate(written);
        raw.destroy_pipeline_cache(cache, None);
        (stated, data)
    }
}

/// What the test device's driver does with pipeline cache data, read through raw Vulkan on the
/// device itself, apart from the runtime and its records (the runtime audit found the lanes
/// deciding it from the runtime's own records): `vkGetPipelineCacheData` of a cache that made
/// nothing, and of two caches that each made one of two compute pipelines (threadgroups of 1 and of
/// 64 threads), each with the size its size query stated.
#[derive(Debug)]
struct DriverData {
    /// What a record names the driver by: vendor, device, driver version, `pipelineCacheUUID`.
    identity: Vec<u8>,
    /// The empty cache's read, then each making's: the size stated, and the bytes written.
    read: [(usize, Vec<u8>); 3],
}

impl DriverData {
    /// The driver's data on `device`.
    fn of(device: &Device) -> DriverData {
        // SAFETY: read while the device lives; the probe's objects are its own (`driver_data`).
        let raw = unsafe { device.raw() };
        // SAFETY: a physical device of the instance.
        let properties = unsafe {
            raw.instance
                .get_physical_device_properties(raw.physical_device)
        };
        let mut identity = Vec::new();
        for word in [
            properties.vendor_id,
            properties.device_id,
            properties.driver_version,
        ] {
            identity.extend_from_slice(&word.to_le_bytes());
        }
        identity.extend_from_slice(&properties.pipeline_cache_uuid);
        let data = DriverData {
            identity,
            read: [None, Some(1), Some(64)].map(|width| driver_data(raw.device, width)),
        };
        let [empty, one, other] = &data.read;
        println!(
            "PIPELINE-RECORDS driver data: whole={} holds={}; stated and written: empty {} {}, \
             made {} {} and {} {}",
            data.whole(),
            data.holds(),
            empty.0,
            empty.1.len(),
            one.0,
            one.1.len(),
            other.0,
            other.1.len()
        );
        data
    }

    /// Whether every read is whole: at least the version-one header it states, of 32 bytes or
    /// more, as Vulkan requires. The runtime keeps no record of a driver whose data is not.
    fn whole(&self) -> bool {
        self.read.iter().all(|(_, data)| {
            data.get(..4)
                .map(|size| u32::from_le_bytes(size.try_into().unwrap()) as usize)
                .is_some_and(|size| size >= 32 && data.len() >= size)
        })
    }

    /// Whether the driver's data holds the pipelines its caches made: the two makings' data
    /// differ.
    fn holds(&self) -> bool {
        self.read[1].1 != self.read[2].1
    }

    /// The data a record frames in `bytes`, its frame checked as `shader_cache/driver.rs` frames
    /// one: the magic, this driver's identity, the data's length, and the SHA-256 of every byte
    /// before the checksum.
    fn framed<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        assert!(bytes.starts_with(RECORD_MAGIC), "a record's magic");
        let header = RECORD_MAGIC.len() + self.identity.len() + 8;
        assert_eq!(
            &bytes[RECORD_MAGIC.len()..header - 8],
            &self.identity[..],
            "a record's identity"
        );
        let length = u64::from_le_bytes(bytes[header - 8..header].try_into().unwrap()) as usize;
        let end = header + length;
        assert_eq!(bytes.len(), end + 32, "a record's length");
        let checksum: [u8; 32] = Sha256::digest(&bytes[..end]).into();
        assert_eq!(&bytes[end..], &checksum[..], "a record's checksum");
        &bytes[header..end]
    }

    /// Each of `records` holds this driver's data for a making: where it holds its pipelines,
    /// data other than an empty cache's (two makings' may be alike: a driver that keeps shader
    /// binaries keeps one for two pipelines whose state changes no shader, as the RTX 3090's and
    /// ANV's do for `RGBA8Unorm` and `BGRA8Unorm` targets); where it holds none, a
    /// making's data, which every making's is.
    fn assert_records(&self, records: &[&[u8]]) {
        for record in records {
            let data = self.framed(record);
            match self.holds() {
                true => assert_ne!(
                    data,
                    &self.read[0].1[..],
                    "a record of an empty cache's data"
                ),
                false => assert_eq!(
                    data,
                    &self.read[1].1[..],
                    "a record of data not the driver's"
                ),
            }
        }
    }

    /// The makings the driver reports of `n` makings from records: hits where its data holds its
    /// pipelines, else misses.
    fn reported_of_records(&self, n: u64) -> (u64, u64) {
        match self.holds() {
            true => (n, 0),
            false => (0, n),
        }
    }
}

/// Of `statistics`, the makings the driver reported (`driver_pipeline_hits`, `misses`): where it
/// reports them at all, `hits` hits and `misses` misses.
fn assert_reported(which: &str, statistics: &ShaderCacheStatistics, (hits, misses): (u64, u64)) {
    let reported = (
        statistics.driver_pipeline_hits,
        statistics.driver_pipeline_misses,
    );
    assert!(
        reported == (0, 0) || reported == (hits, misses),
        "{which}: the driver reported {reported:?}, not {:?}",
        (hits, misses)
    );
}

/// The environment variables that name a driver's own shader disk cache: Mesa's, and NVIDIA's.
const DRIVER_CACHE_VARIABLES: [&str; 2] = ["MESA_SHADER_CACHE_DIR", "__GL_SHADER_DISK_CACHE_PATH"];

/// A fresh, empty directory for the driver's own shader disk cache, named to the drivers while it
/// lives. A device made meanwhile finds in it nothing an earlier device or process compiled, so a
/// making the driver reports as its hit is a hit from the runtime's record, and a making with no
/// record is its miss, whatever this host compiled before. Test setup alone: the runtime reads
/// neither variable. It is made while the process has no live device, since a driver reads the
/// variables as the device's instance is made, and is declared before the device so it goes after
/// it.
struct DriverCache(ScratchCache);

impl DriverCache {
    /// A new directory, named to this process's drivers.
    fn fresh() -> DriverCache {
        let dir = ScratchCache::new().unwrap();
        std::fs::create_dir_all(dir.path()).unwrap();
        for variable in DRIVER_CACHE_VARIABLES {
            std::env::set_var(variable, dir.path());
        }
        DriverCache(dir)
    }

    /// A new directory, named to the drivers of the process `command` starts.
    fn fresh_for(command: &mut Command) -> ScratchCache {
        let dir = ScratchCache::new().unwrap();
        std::fs::create_dir_all(dir.path()).unwrap();
        for variable in DRIVER_CACHE_VARIABLES {
            command.env(variable, dir.path());
        }
        dir
    }
}

impl Drop for DriverCache {
    /// The variables are removed where they still name this directory; the directory goes with
    /// its `ScratchCache`.
    fn drop(&mut self) {
        for variable in DRIVER_CACHE_VARIABLES {
            if std::env::var_os(variable).as_deref() == Some(self.0.path().as_os_str()) {
                std::env::remove_var(variable);
            }
        }
    }
}

/// The records of `now` that are not `before`'s byte for byte: new, or written again.
fn changed(before: &BTreeMap<PathBuf, Vec<u8>>, now: &BTreeMap<PathBuf, Vec<u8>>) -> u64 {
    now.iter()
        .filter(|(path, bytes)| before.get(*path) != Some(*bytes))
        .count() as u64
}

/// Of `statistics` since `since`, the makings that loaded a record: each wrote it again exactly
/// where the driver's data was no longer the record's, so `changed` records are written, and
/// every other one read its data unchanged.
fn assert_loaded(
    which: &str,
    (since, statistics): (&ShaderCacheStatistics, &ShaderCacheStatistics),
    loads: u64,
    changed: u64,
) {
    let saves = statistics.driver_saves - since.driver_saves;
    let unchanged = statistics.driver_data_unchanged - since.driver_data_unchanged;
    assert_eq!(
        (
            statistics.driver_loads - since.driver_loads,
            saves,
            unchanged
        ),
        (loads, changed, loads - changed),
        "{which}: the makings from records, written again exactly where a record changed"
    );
}

#[test]
fn a_later_device_makes_every_pipeline_an_earlier_one_made_from_its_records() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let format = PixelFormat::RGBA8Unorm;
    // The first device: an empty directory.
    let (driver, targets, written, makings) = {
        let _driver_cache = DriverCache::fresh();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let driver = DriverData::of(&device);
        let library = library(&device);
        let before = records(cache.path());
        let state = state(&device, &library, format);
        let makings = Makings::of(&device);
        let created = device.shader_cache_statistics();
        report("first device, created", &created);
        let keeps = driver.whole();
        assert_eq!(kept(&created), keeps, "the driver's data: {driver:?}");
        let made: Vec<_> = records(cache.path())
            .into_keys()
            .filter(|path| !before.contains_key(path))
            .collect();
        if keeps {
            assert_eq!(created.driver_loads, 0, "a load from an empty directory");
            assert_eq!(
                made.len() as u64,
                makings.creation,
                "the creation's makings' records ({makings:?})"
            );
            // The driver's own cache is fresh: its miss.
            assert_reported("first device, created", &created, (0, makings.creation));
        } else {
            assert!(made.is_empty(), "a record of a driver that keeps nothing");
        }
        let triangles = draw(&device, &state, (format, PrimitiveType::Triangle));
        // Points: another key of the same stages, made for the draw.
        let points = draw(&device, &state, (format, PrimitiveType::Point));
        assert_eq!(
            draw(&device, &state, (format, PrimitiveType::Triangle)),
            triangles
        );
        let drawn = device.shader_cache_statistics();
        report("first device, drawn", &drawn);
        let written: Vec<_> = records(cache.path())
            .into_iter()
            .filter(|(path, _)| !before.contains_key(path))
            .collect();
        if keeps {
            assert_eq!(
                drawn.driver_saves - created.driver_saves,
                makings.points,
                "the points' making wrote its record"
            );
            assert_eq!(
                written.len() as u64,
                makings.creation + makings.points,
                "one record per making"
            );
            let bytes: Vec<&[u8]> = written.iter().map(|(_, bytes)| &bytes[..]).collect();
            driver.assert_records(&bytes);
            assert_reported(
                "first device, drawn",
                &drawn,
                (0, makings.creation + makings.points),
            );
        } else {
            assert!(written.is_empty());
        }
        (driver, (triangles, points), written, makings)
    };
    // The second device: the first one's directory, the driver's own cache fresh again.
    {
        let _driver_cache = DriverCache::fresh();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let library = library(&device);
        let none = device.shader_cache_statistics();
        let state = state(&device, &library, format);
        let created = device.shader_cache_statistics();
        report("second device, created", &created);
        let at_creation = records(cache.path());
        assert_eq!(
            draw(&device, &state, (format, PrimitiveType::Triangle)),
            targets.0
        );
        assert_eq!(
            draw(&device, &state, (format, PrimitiveType::Point)),
            targets.1
        );
        let drawn = device.shader_cache_statistics();
        report("second device, drawn", &drawn);
        let now = records(cache.path());
        let written: BTreeMap<PathBuf, Vec<u8>> = written.into_iter().collect();
        if driver.whole() {
            assert_loaded(
                "second device, created: the triangles' makings from their records",
                (&none, &created),
                makings.creation,
                changed(&written, &at_creation),
            );
            assert_loaded(
                "second device, drawn: the points' making from its record",
                (&created, &drawn),
                makings.points,
                changed(&at_creation, &now),
            );
            // Hits where the driver's data holds its pipelines, misses where it holds none; its
            // own cache is fresh, so a hit is the record's.
            assert_reported(
                "second device, created",
                &created,
                driver.reported_of_records(makings.creation),
            );
            assert_reported(
                "second device, drawn",
                &drawn,
                driver.reported_of_records(makings.creation + makings.points),
            );
            assert_eq!(
                now.len() as u64,
                makings.creation + makings.points,
                "no record but the makings'"
            );
            let bytes: Vec<&[u8]> = now.values().map(|bytes| &bytes[..]).collect();
            driver.assert_records(&bytes);
        } else {
            assert_eq!((drawn.driver_loads, drawn.driver_saves), (0, 0));
            assert!(now.is_empty(), "a record of a driver that keeps nothing");
        }
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

#[test]
fn a_record_holding_another_pipeline_s_data_is_written_again() {
    let messages = Messages::default();
    // The record of a dispatch's compute pipeline, in a directory of its own.
    let other = ScratchCache::new().unwrap();
    let (driver, foreign) = {
        let _driver_cache = DriverCache::fresh();
        let device = validated_device_with(TEST_DEVICE, &messages, other.path(), |_| {}).unwrap();
        let driver = DriverData::of(&device);
        dispatch(&device, &library(&device));
        let made = device.shader_cache_statistics();
        report("the compute pipeline", &made);
        assert_eq!(kept(&made), driver.whole(), "the driver's data: {driver:?}");
        (driver, records(other.path()))
    };
    // Data that ends inside its header is never kept (the module docs): no record to copy.
    if !driver.whole() {
        assert!(
            foreign.is_empty(),
            "a record of a driver that keeps nothing"
        );
        println!("PIPELINE-RECORDS: the driver's data is not whole; nothing to write again");
        assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
        return;
    }
    let foreign: Vec<Vec<u8>> = foreign.into_values().collect();
    assert_eq!(foreign.len(), 1, "the compute pipeline's record");
    let foreign = &foreign[0];
    // A `BGRA8Unorm` state's makings: their records, then the compute pipeline's bytes put in
    // each one's place: another pipeline's data.
    let cache = ScratchCache::new().unwrap();
    let format = PixelFormat::BGRA8Unorm;
    let (target, makings) = {
        let _driver_cache = DriverCache::fresh();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let state = state(&device, &library(&device), format);
        let makings = Makings::of(&device);
        let made = device.shader_cache_statistics();
        assert_reported("its own state", &made, (0, makings.creation));
        (
            draw(&device, &state, (format, PrimitiveType::Triangle)),
            makings,
        )
    };
    let own: Vec<(PathBuf, Vec<u8>)> = records(cache.path()).into_iter().collect();
    assert_eq!(
        own.len() as u64,
        makings.creation,
        "its own state's records"
    );
    // Each record holds the driver's data; where that holds no pipeline, the compute pipeline's
    // record is each one's own bytes, which the making loads and finds its data, writing nothing.
    let all: Vec<&[u8]> = std::iter::once(&foreign[..])
        .chain(own.iter().map(|(_, bytes)| &bytes[..]))
        .collect();
    driver.assert_records(&all);
    for (path, _) in &own {
        std::fs::write(cache.path().join(path), foreign).unwrap();
    }
    // Loaded, then the driver's miss: the compute pipeline's data holds no making of a render
    // state. Where the driver's data holds its pipelines, the miss adds this one, so each record
    // is written again; where it holds none, the data stays the record's bytes.
    let rewritten = {
        let _driver_cache = DriverCache::fresh();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let none = device.shader_cache_statistics();
        let state = state(&device, &library(&device), format);
        let made = device.shader_cache_statistics();
        report("the compute pipeline's data in its place", &made);
        assert_reported(
            "the compute pipeline's data in its place",
            &made,
            (0, makings.creation),
        );
        let rewritten: Vec<Vec<u8>> = own
            .iter()
            .map(|(path, _)| std::fs::read(cache.path().join(path)).unwrap())
            .collect();
        let changed = rewritten
            .iter()
            .filter(|rewritten| *rewritten != foreign)
            .count() as u64;
        assert_loaded(
            "the compute pipeline's data in its place",
            (&none, &made),
            makings.creation,
            changed,
        );
        assert_eq!(
            changed,
            match driver.holds() {
                true => makings.creation,
                false => 0,
            },
            "the driver's misses added this pipeline to each record"
        );
        assert_eq!(
            draw(&device, &state, (format, PrimitiveType::Triangle)),
            target
        );
        rewritten
    };
    let bytes: Vec<&[u8]> = rewritten.iter().map(|bytes| &bytes[..]).collect();
    driver.assert_records(&bytes);
    // A later device hits them, writing each again exactly where the driver's data changed.
    {
        let _driver_cache = DriverCache::fresh();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
        let none = device.shader_cache_statistics();
        let state = state(&device, &library(&device), format);
        let made = device.shader_cache_statistics();
        report("written again, a later device", &made);
        assert_reported(
            "written again, a later device",
            &made,
            driver.reported_of_records(makings.creation),
        );
        let now: Vec<Vec<u8>> = own
            .iter()
            .map(|(path, _)| std::fs::read(cache.path().join(path)).unwrap())
            .collect();
        let changed = now
            .iter()
            .zip(&rewritten)
            .filter(|(now, rewritten)| now != rewritten)
            .count() as u64;
        assert_loaded(
            "written again, a later device",
            (&none, &made),
            makings.creation,
            changed,
        );
        let bytes: Vec<&[u8]> = now.iter().map(|bytes| &bytes[..]).collect();
        driver.assert_records(&bytes);
        assert_eq!(
            draw(&device, &state, (format, PrimitiveType::Triangle)),
            target
        );
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// The processes [`two_processes_making_one_pipeline_leave_one_whole_record`] runs at once.
const MAKERS: usize = 2;

#[test]
fn two_processes_making_one_pipeline_leave_one_whole_record() {
    let cache = ScratchCache::new().unwrap();
    let mut driver_caches = Vec::new();
    let makers: Vec<_> = (0..MAKERS)
        .map(|_| {
            let mut maker = Command::new(std::env::current_exe().unwrap());
            maker
                .args([
                    "--exact",
                    "runtime_lane::pipeline_records::record_maker_process",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(MAKER_DIR, cache.path());
            driver_caches.push(DriverCache::fresh_for(&mut maker));
            maker.spawn().unwrap()
        })
        .collect();
    for maker in makers {
        let made = maker.wait_with_output().unwrap();
        assert!(made.status.success(), "a maker process failed");
    }
    drop(driver_caches);
    assert_eq!(temporaries(cache.path()), 0, "a temporary file left");
    let written = records(cache.path());
    let messages = Messages::default();
    let driver_cache = DriverCache::fresh();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
    let driver = DriverData::of(&device);
    let format = PixelFormat::RGBA8Unorm;
    let none = device.shader_cache_statistics();
    let state = state(&device, &library(&device), format);
    let makings = Makings::of(&device);
    let made = device.shader_cache_statistics();
    report("after two makers", &made);
    assert_eq!(kept(&made), driver.whole(), "the driver's data: {driver:?}");
    if driver.whole() {
        assert_eq!(
            written.len() as u64,
            makings.creation,
            "one record of each of the one creation's makings"
        );
        let record: Vec<&[u8]> = written.values().map(|bytes| &bytes[..]).collect();
        driver.assert_records(&record);
        assert_reported(
            "after two makers",
            &made,
            driver.reported_of_records(makings.creation),
        );
        let now = records(cache.path());
        assert_loaded(
            "after two makers: the records loaded whole",
            (&none, &made),
            makings.creation,
            changed(&written, &now),
        );
        let bytes: Vec<&[u8]> = now.values().map(|bytes| &bytes[..]).collect();
        driver.assert_records(&bytes);
    } else {
        assert!(
            written.is_empty(),
            "a record of a driver that keeps nothing"
        );
    }
    let before_draw = records(cache.path());
    draw(&device, &state, (format, PrimitiveType::Triangle));
    let drawn = device.shader_cache_statistics();
    drop(state);
    drop(device);
    drop(driver_cache);
    assert_eq!(
        (drawn.driver_loads, drawn.driver_saves),
        (made.driver_loads, made.driver_saves),
        "the draw took the creation's pipeline"
    );
    assert_eq!(records(cache.path()), before_draw, "the draw wrote nothing");
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// One of [`two_processes_making_one_pipeline_leave_one_whole_record`]'s processes: the
/// `RGBA8Unorm` state's creation on the directory [`MAKER_DIR`] names.
#[test]
#[ignore = "run by two_processes_making_one_pipeline_leave_one_whole_record"]
fn record_maker_process() {
    let Some(dir) = std::env::var_os(MAKER_DIR) else {
        return;
    };
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, Path::new(&dir), |_| {}).unwrap();
    let _state = state(&device, &library(&device), PixelFormat::RGBA8Unorm);
    let made = device.shader_cache_statistics();
    println!(
        "\n{MAKER_REPORT} loads={} saves={} truncated={}",
        made.driver_loads, made.driver_saves, made.driver_data_truncated
    );
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}
