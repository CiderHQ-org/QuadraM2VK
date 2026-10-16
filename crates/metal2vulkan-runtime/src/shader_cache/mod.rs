//! The translated-shader cache on disk: always on, so a second process does not translate again
//! what the first already did, as Metal programs get from the OS.
//!
//! - **The directory.** [`ShaderCacheOptions::directory`], or the platform's per-user cache
//!   directory: `~/Library/Caches/metal2vulkan` on macOS, `$XDG_CACHE_HOME/metal2vulkan`
//!   elsewhere, or `~/.cache/metal2vulkan` when that is unset or not absolute. Never the OS temp
//!   directory. Creating or adopting a device makes it (owner-only) and checks that it takes a
//!   write; if not, the call fails with [`Error::ShaderCache`], naming the path and the I/O error.
//!   After that, a failed read or write is a miss the [statistics](ShaderCacheStatistics) count,
//!   never an error.
//! - **What it holds.** The compile layer's artifacts in their self-checking envelope: translated
//!   functions under their [`FunctionCompileKey`] and compute and render pipelines under their
//!   [`PipelineKey`]; and the driver's `VkPipelineCache` data, one record per pipeline making,
//!   named by what the making reads (`driver.rs`); and what render creations work out from their
//!   descriptors before their compiles, under the descriptor's key (`creation.rs`).
//! - **A new build never reads an older build's entries.** They live under `v1/<build>/`, where
//!   `<build>` is the SHA-256 of the crate's version, both artifact versions, and the digest of
//!   the crate's sources that `build.rs` computes (the product files).
//! - **Records of content, every build's**. Beside the builds', one directory holds records named
//!   by the content they describe, which hold no translation: the driver's data of each pipeline
//!   making (above).
//! - **A wrong entry is never used.** An entry's name is the SHA-256 of its build, kind, key and
//!   profile fingerprint, and a read is a hit only if the envelope's versions, kind, key and
//!   profile equal the request's and its checksum holds (`from_bytes_for_key`). Anything else is
//!   a miss and the file is removed.
//! - **Bounded.** [`ShaderCacheOptions::capacity`] caps the directory (25 GiB by default; 1 GiB
//!   before, which the consumer's boots filled, removing records they needed); the least
//!   recently used entries go first, and at device creation other builds' directories unused for
//!   30 days go (`store.rs`, on the device's cache thread, `worker.rs`). A device's first removal
//!   past the cap and its first failed write are each said once as a warning
//!   (`metal2vulkan-shader-cache-evicted`, `metal2vulkan-shader-cache-write-failed`) naming the
//!   directory and its cap.
//! - **In memory first.** Before the directory, every lookup asks the device's memo
//!   (`pipeline/memo.rs`), which [`ShaderCacheOptions::memory_capacity`] bounds (256 MiB by
//!   default). An entry read or written here is kept there, weighed at its encoded length.

mod creation;
pub(crate) mod driver;
pub(super) mod store;
mod worker;

use super::error::{invalid, Error, Problem, ShaderCacheError};
use super::pipeline::memo::PipelineMemo;
use super::Device;
use crate::MessageSink;
use ash::vk;
use creation::CreationMemo;
use driver::{Making, Records};
use metal2vulkan::cache::{Hash256, PipelineArtifact, PipelineKey};
use metal2vulkan::library::{CompiledFunction, FunctionCompileKey};
use metal2vulkan::pipeline::{ComputePipeline, RenderPipeline};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use store::{count, entry_name, Directory, Lookup, Store};
use worker::Worker;

/// The default cap of a cache directory, in bytes: 25 GiB.
pub const DEFAULT_CAPACITY: u64 = 25 << 30;

/// The default cap of a device's memo of compiled pipelines, in bytes: 256 MiB.
pub const DEFAULT_MEMORY_CAPACITY: u64 = 256 << 20;

/// Where a device keeps its translated shaders, and how much it keeps. No Metal counterpart:
/// Metal's shader cache is the OS's.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderCacheOptions {
    /// The cache directory; `None` is the platform's per-user cache directory (the module's
    /// documentation names it). Devices of different processes and builds may share one.
    pub directory: Option<PathBuf>,
    /// The most bytes the directory holds, every build's entries together; past it the least
    /// recently used entries are removed, the device's first removal said once as a warning.
    /// [`DEFAULT_CAPACITY`] (25 GiB) by default. Not zero.
    pub capacity: u64,
    /// The most bytes of translated functions and compiled pipelines the device keeps in
    /// memory, each counted at its encoded length; past it, or past 4096 of them, the least
    /// recently used are dropped. A live pipeline state keeps what it uses whatever this says.
    /// Not zero.
    pub memory_capacity: u64,
}

impl Default for ShaderCacheOptions {
    fn default() -> ShaderCacheOptions {
        ShaderCacheOptions {
            directory: None,
            capacity: DEFAULT_CAPACITY,
            memory_capacity: DEFAULT_MEMORY_CAPACITY,
        }
    }
}

impl ShaderCacheOptions {
    /// The cache in `directory`, with the default caps.
    pub fn in_directory(directory: impl Into<PathBuf>) -> ShaderCacheOptions {
        ShaderCacheOptions {
            directory: Some(directory.into()),
            ..ShaderCacheOptions::default()
        }
    }
}

/// The most stages a graphics pipeline has (vertex, two tessellation stages, geometry,
/// fragment; or task, mesh and fragment): a making reports each one's feedback where its
/// pipeline has no more ([`found_in_cache`]).
pub(crate) const REPORTED_STAGES: usize = 5;

/// What a making's `VK_EXT_pipeline_creation_feedback` says of its pipeline: found in the cache
/// the making handed the driver, `Some(true)`; compiled, `Some(false)`; `None` where the whole
/// pipeline's feedback is not valid, so the driver reported nothing.
///
/// Found where the whole pipeline's feedback says `APPLICATION_PIPELINE_CACHE_HIT`, or where
/// `stages` (each stage's feedback, empty where the making chained none) is not empty and every
/// one is valid and says it. The specification lets a driver say it either way:
/// `APPLICATION_PIPELINE_CACHE_HIT` "specifies that a readily usable pipeline or pipeline stage
/// was found in the `pipelineCache` specified by the application in the pipeline creation
/// command", and "when an implementation sets this bit for the entire pipeline, it may leave it
/// unset for any stage" (`VkPipelineCreationFeedbackFlagBits`). A driver that sets it for the
/// whole pipeline counts as before; MoltenVK 1.4.1 sets it for each stage alone, where it found
/// the stage's converted Metal function in the cache (`MVKPipelineCache::getShaderLibraryImpl`),
/// and leaves the whole pipeline's feedback valid alone (`MVKGraphicsPipeline`,
/// `MVKComputePipeline`).
pub(crate) fn found_in_cache(
    whole: &vk::PipelineCreationFeedback,
    stages: &[vk::PipelineCreationFeedback],
) -> Option<bool> {
    let valid = vk::PipelineCreationFeedbackFlags::VALID;
    let hit = vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT;
    if !whole.flags.contains(valid) {
        return None;
    }
    let every_stage =
        !stages.is_empty() && stages.iter().all(|stage| stage.flags.contains(valid | hit));
    Some(whole.flags.contains(hit) || every_stage)
}

/// A device's shader cache counts since it was made ([`Device::shader_cache_statistics`]). No
/// Metal counterpart.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderCacheStatistics {
    /// Artifacts read and accepted: a translation or pipeline compile not run.
    pub hits: u64,
    /// Artifact lookups that found nothing usable, for any reason; each was compiled.
    pub misses: u64,
    /// Entries found under the right name whose contents failed the checks, each removed.
    pub rejected: u64,
    /// Entries that could not be read.
    pub read_failures: u64,
    /// Entries written.
    pub writes: u64,
    /// Entries that could not be written.
    pub write_failures: u64,
    /// Entries removed to keep the directory under its cap.
    pub evictions: u64,
    /// Pipeline makings whose `VkPipelineCache` started from the making's saved record (it was
    /// whether the device's one driver cache did, 0 or 1).
    pub driver_loads: u64,
    /// Pipelines of the program's states that the driver reports it found in their making's
    /// cache, seeded from the making's record (`VK_EXT_pipeline_creation_feedback`, where the
    /// device lists the extension): the whole pipeline's feedback is valid and says
    /// `APPLICATION_PIPELINE_CACHE_HIT`, or every stage's feedback is valid and says it. Drivers
    /// differ in where they say it: most for the whole pipeline; MoltenVK 1.4.1 for each stage
    /// alone, whose Metal function it found converted in the cache (its whole pipeline's
    /// feedback is only ever valid).
    pub driver_pipeline_hits: u64,
    /// Such pipelines whose valid feedback says no hit, for the whole pipeline or for every
    /// stage: compiled, or a stage converted again.
    pub driver_pipeline_misses: u64,
    /// Records of the driver's pipeline cache data written: one per making that added to its
    /// cache's data (no record, or its pipeline not in it). (It was the saves of the device's one
    /// driver cache.)
    pub driver_saves: u64,
    /// Reads of the driver's pipeline cache data refused because the data ends inside the
    /// header it states, at a making's save. Such data is never kept, so the making's cache
    /// starts empty in every process. A driver defect.
    pub driver_data_truncated: u64,
    /// Reads of the driver's pipeline cache data refused because its header is not this
    /// device's version-one header (version, vendor, device or UUID), counted as
    /// [`driver_data_truncated`](Self::driver_data_truncated) is.
    pub driver_data_mismatched: u64,
    /// Reads of the driver's pipeline cache data that were the record the making loaded, byte
    /// for byte: nothing written. With [`driver_saves`](Self::driver_saves) and the two refusals,
    /// one read per making the runtime finished while the process was not exiting.
    pub driver_data_unchanged: u64,
}

/// What an entry holds, in its name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum EntryKind {
    Function = 1,
    ComputePipeline = 2,
    // 3 named the device's one driver cache entry, retired.
    RenderPipeline = 4,
    // 5 named a pipeline's kept graphics keys, removed. 6 named a variant's own pipeline cache,
    // retired.
    /// A pipeline making's driver data (`driver.rs`), in the content directory.
    PipelineData = 7,
    /// A render descriptor's linked-use answers (`creation.rs`).
    LinkedAnswers = 8,
    /// A render descriptor's reflection (`creation.rs`).
    RenderReflection = 9,
    /// The pipeline key of a render descriptor under a variant key (`creation.rs`).
    VariantPipeline = 10,
}

/// The build this crate is: the SHA-256 of its version, both artifact versions, the translator's
/// source digest, and its own sources' digest (`build.rs`).
pub(crate) fn this_build() -> Hash256 {
    let mut hash = Sha256::new();
    hash.update(b"metal2vulkan shader cache build\0");
    hash.update(env!("CARGO_PKG_VERSION").as_bytes());
    hash.update([0]);
    hash.update(metal2vulkan::cache::ARTIFACT_FORMAT_VERSION.to_le_bytes());
    hash.update(metal2vulkan::cache::SEMANTICS_VERSION.to_le_bytes());
    hash.update(metal2vulkan::cache::SOURCE_DIGEST.as_bytes());
    hash.update(env!("M2V_RUNTIME_SOURCE_ID").as_bytes());
    hash.finalize().into()
}

/// The platform's per-user cache directory for this crate.
fn default_directory() -> Result<PathBuf, ShaderCacheError> {
    let absolute = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    let home = || {
        absolute(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).ok_or_else(|| {
            ShaderCacheError::new(
                None,
                std::io::ErrorKind::NotFound,
                "no home directory for the default shader cache directory",
            )
        })
    };
    if cfg!(target_os = "macos") {
        return Ok(home()?.join("Library/Caches/metal2vulkan"));
    }
    match absolute("XDG_CACHE_HOME") {
        Some(base) => Ok(base.join("metal2vulkan")),
        None => Ok(home()?.join(".cache/metal2vulkan")),
    }
}

/// A device's shader cache as its options set it up, before the device is made: the directory
/// opened for this build, and the memo's cap.
#[derive(Clone, Debug)]
pub(super) struct CacheSetup {
    directory: Directory,
    memory_capacity: u64,
}

impl CacheSetup {
    /// The cache `options` describe: its directory opened for this build. `call` names the
    /// creating call.
    pub(super) fn of(
        options: &ShaderCacheOptions,
        call: &'static str,
    ) -> Result<CacheSetup, Error> {
        if options.capacity == 0 || options.memory_capacity == 0 {
            return Err(invalid(call, Problem::ShaderCacheCapacity));
        }
        let root = match &options.directory {
            Some(directory) => directory.clone(),
            None => default_directory().map_err(Error::ShaderCache)?,
        };
        let directory =
            Directory::open(root, this_build(), options.capacity).map_err(|failed| {
                Error::ShaderCache(ShaderCacheError::new(
                    Some(failed.path),
                    failed.error.kind(),
                    &failed.error.to_string(),
                ))
            })?;
        Ok(CacheSetup {
            directory,
            memory_capacity: options.memory_capacity,
        })
    }
}

/// One device's shader cache: its memo, its view of the directory, what names its makings'
/// records, and the cache thread.
pub(super) struct ShaderCache {
    /// The compiled pipelines kept in memory.
    pub(super) memo: PipelineMemo,
    /// What render creations worked out from their descriptors before their compiles, in
    /// memory (`creation.rs`).
    creation: CreationMemo,
    store: Arc<Store>,
    worker: Worker,
    /// What names the makings' records (`driver.rs`).
    records: Records,
    /// Whether a making reports whether the driver found its pipeline in its cache
    /// (`VK_EXT_pipeline_creation_feedback`).
    driver_reports: bool,
}

impl ShaderCache {
    /// The cache `setup` describes, of the device of `properties` and profile fingerprint
    /// `profile`, with the cache thread started (which scans the directory). Its makings' records
    /// are named with `settings` (the digest of its instance's layer settings), and its makings
    /// report whether the driver found their pipelines where `feedback`
    /// (`VK_EXT_pipeline_creation_feedback`). (No driver cache is made or loaded here; each making
    /// makes its own.) Its warnings go to `sink` (the instance's message sink), else on stderr.
    pub(super) fn start(
        setup: CacheSetup,
        properties: &vk::PhysicalDeviceProperties,
        profile: Hash256,
        (settings, feedback): (Hash256, bool),
        sink: Option<MessageSink>,
    ) -> Result<ShaderCache, Error> {
        let CacheSetup {
            directory,
            memory_capacity,
        } = setup;
        let store = Arc::new(Store::new(directory, sink));
        let worker = Worker::start(Arc::clone(&store))?;
        Ok(ShaderCache {
            memo: PipelineMemo::new(memory_capacity),
            creation: CreationMemo::new(),
            store,
            worker,
            records: Records {
                identity: driver::Identity::of(properties),
                settings,
                profile,
            },
            driver_reports: feedback,
        })
    }

    /// Stop the cache thread: the device is going.
    pub(super) fn close(&self) {
        self.worker.stop();
    }

    /// What `create` makes with the `VkPipelineCache` it is handed, its `vkCreate*Pipelines`
    /// call on the device of `raw` (`DeviceShared::vk`): a cache of the making's own, seeded from
    /// the record `making` names where one loads, whose data is written as that record where the
    /// making added to it, and destroyed after (`driver.rs`); null where the driver makes none.
    /// The call holds a turn of the process's budget of makings in flight (`pipeline/budget.rs`):
    /// its own, waited for, or its pool piece's.
    pub(super) fn make_pipelines<T, E>(
        &self,
        raw: &ash::Device,
        making: Making,
        create: impl FnOnce(vk::PipelineCache) -> Result<T, E>,
    ) -> Result<T, E> {
        let _turn = crate::pipeline::budget::program_turn();
        let made = driver::make(
            &self.store,
            &self.records,
            raw,
            making,
            crate::exit_guard::closed,
            create,
        );
        if self.store.over_capacity() {
            self.worker.evict();
        }
        made
    }

    /// Whether a making chains a `VkPipelineCreationFeedbackCreateInfo`, its whole pipeline's
    /// feedback and its stages', which [`driver_reported`](Self::driver_reported) counts.
    pub(crate) fn driver_reports(&self) -> bool {
        self.driver_reports
    }

    /// Count what a making reported of its pipeline, `whole`, and of its stages, `stages` (empty
    /// where it chained none): found in its cache, or compiled ([`found_in_cache`]); nothing
    /// where the driver left it unreported.
    pub(crate) fn driver_reported(
        &self,
        whole: &vk::PipelineCreationFeedback,
        stages: &[vk::PipelineCreationFeedback],
    ) {
        let counters = &self.store.counters;
        if let Some(found) = found_in_cache(whole, stages) {
            count(match found {
                true => &counters.driver_pipeline_hits,
                false => &counters.driver_pipeline_misses,
            });
        }
    }

    /// The function `compile` compiled under `key`, from the cache, and its encoded length.
    pub(super) fn load_function(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &FunctionCompileKey,
    ) -> Option<(CompiledFunction, u64)> {
        self.load(EntryKind::Function, key.digest(), compile, |bytes| {
            CompiledFunction::from_bytes_for_key(bytes, key, compile)
        })
    }

    /// Keep `function`, which `compile` compiled; its encoded length.
    pub(super) fn store_function(
        &self,
        compile: &metal2vulkan::device::Device,
        function: &CompiledFunction,
    ) -> u64 {
        let bytes = function.to_bytes();
        self.store(
            EntryKind::Function,
            function.key().digest(),
            compile,
            &bytes,
        )
    }

    /// The compute pipeline `compile` compiled under `key`, from the cache, and its encoded
    /// length.
    pub(super) fn load_compute_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &PipelineKey,
    ) -> Option<(ComputePipeline, u64)> {
        self.load(EntryKind::ComputePipeline, key.digest(), compile, |bytes| {
            ComputePipeline::from_bytes_for_key(bytes, key, compile)
        })
    }

    /// Keep `pipeline`, which `compile` compiled; its encoded length.
    pub(super) fn store_compute_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        pipeline: &ComputePipeline,
    ) -> u64 {
        let bytes = pipeline.to_bytes();
        self.store(
            EntryKind::ComputePipeline,
            pipeline.key().digest(),
            compile,
            &bytes,
        )
    }

    /// The render pipeline `compile` compiled under `key`, from the cache, and its encoded
    /// length.
    pub(super) fn load_render_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        key: &PipelineKey,
    ) -> Option<(RenderPipeline, u64)> {
        self.load(EntryKind::RenderPipeline, key.digest(), compile, |bytes| {
            RenderPipeline::from_bytes_for_key(bytes, key, compile)
        })
    }

    /// Keep `pipeline`, which `compile` compiled; its encoded length.
    pub(super) fn store_render_pipeline(
        &self,
        compile: &metal2vulkan::device::Device,
        pipeline: &RenderPipeline,
    ) -> u64 {
        let bytes = pipeline.to_bytes();
        self.store(
            EntryKind::RenderPipeline,
            pipeline.key().digest(),
            compile,
            &bytes,
        )
    }

    /// The entry of `kind` under `key` for `compile`'s profile, as `decode` accepts it, and its
    /// length. A miss when absent, unreadable, or refused (then removed).
    fn load<T, E>(
        &self,
        kind: EntryKind,
        key: &Hash256,
        compile: &metal2vulkan::device::Device,
        decode: impl FnOnce(&[u8]) -> Result<T, E>,
    ) -> Option<(T, u64)> {
        let counters = &self.store.counters;
        let name = entry_name(
            self.store.directory().build(),
            kind,
            key,
            &compile.profile_fingerprint(),
        );
        let found = match self.store.load(&name, |bytes| {
            decode(bytes).map(|value| (value, bytes.len() as u64))
        }) {
            Lookup::Hit(found) => Some(found),
            Lookup::Absent | Lookup::Unreadable | Lookup::Rejected => None,
        };
        count(if found.is_some() {
            &counters.hits
        } else {
            &counters.misses
        });
        found
    }

    /// Write `bytes` as the entry of `kind` under `key` for `compile`'s profile; their length.
    fn store(
        &self,
        kind: EntryKind,
        key: &Hash256,
        compile: &metal2vulkan::device::Device,
        bytes: &[u8],
    ) -> u64 {
        let counters = &self.store.counters;
        let name = entry_name(
            self.store.directory().build(),
            kind,
            key,
            &compile.profile_fingerprint(),
        );
        match self.store.write(&name, |out| out.write_all(bytes)) {
            Ok(_) => count(&counters.writes),
            Err(_) => self.store.write_failed(),
        }
        if self.store.over_capacity() {
            self.worker.evict();
        }
        bytes.len() as u64
    }

    /// The counts so far.
    fn statistics(&self) -> ShaderCacheStatistics {
        let counters = &self.store.counters;
        let read = |counter: &std::sync::atomic::AtomicU64| counter.load(Ordering::Relaxed);
        ShaderCacheStatistics {
            hits: read(&counters.hits),
            misses: read(&counters.misses),
            rejected: read(&counters.rejected),
            read_failures: read(&counters.read_failures),
            writes: read(&counters.writes),
            write_failures: read(&counters.write_failures),
            evictions: read(&counters.evictions),
            driver_loads: read(&counters.driver_loads),
            driver_pipeline_hits: read(&counters.driver_pipeline_hits),
            driver_pipeline_misses: read(&counters.driver_pipeline_misses),
            driver_saves: read(&counters.driver_saves),
            driver_data_truncated: read(&counters.driver_data_truncated),
            driver_data_mismatched: read(&counters.driver_data_mismatched),
            driver_data_unchanged: read(&counters.driver_data_unchanged),
        }
    }
}

impl Device {
    /// The device's shader cache counts since it was made. No Metal counterpart.
    pub fn shader_cache_statistics(&self) -> ShaderCacheStatistics {
        self.shared().shader_cache.statistics()
    }

    /// The device's shader cache directory. No Metal counterpart.
    pub fn shader_cache_directory(&self) -> &Path {
        self.shared().shader_cache.store.directory().root()
    }
}
