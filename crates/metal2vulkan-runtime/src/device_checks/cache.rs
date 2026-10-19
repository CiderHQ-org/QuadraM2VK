//! The device-checks cache: what the checks found, one file per setup.
//!
//! - **The key** ([`Key`]): a SHA-256 over the setup's `driverID`, `vendorID`, `deviceID`,
//!   `driverVersion`, `pipelineCacheUUID`, the OS build (MoltenVK's answers are macOS's Metal's:
//!   `kern.osversion` on macOS, the kernel's release elsewhere), the suite's version
//!   ([`CHECK_SUITE_VERSION`](super::CHECK_SUITE_VERSION)), the runtime's version, the translator's
//!   and the runtime's source digests, and what the device enables and its policy (which choose the
//!   checks' modules and the paths they check). A setup that changes any of them is checked again.
//!   The source digests (`metal2vulkan::cache::SOURCE_DIGEST`, the runtime's `build.rs`) cover
//!   everything a check is made of, which no version need name: its modules' words, which the
//!   translator's lowering builds, its inputs, its expected answer and how it is read; a build of
//!   other sources checks again, at no cost to a warm making, which hashes nothing.
//! - **The file**: `<directory>/v1/<key>.checks`, text: a header line, the key and its fields,
//!   when the checks ran, one line per answer with what it states (how the other answer differs,
//!   why a check does not run; an unanswered check's is never kept), and last a SHA-256 of
//!   everything before it. A file that does not read, whose digest or key differs, or
//!   that names a fact the suite does not check is not read: the setup is checked again and the
//!   file replaced.
//! - **Writes** merge into the entry: each reads the entry, keeps its answers but those of the
//!   facts the write answers, whose answers are the write's, and goes to a new file of the
//!   process's own beside it, renamed over it, so a reader in another process sees the old entry
//!   or the new one whole. The making writes what it found, and the checks thread each answer as
//!   its check settles (`background.rs`), with every answer it holds: two processes that check
//!   one setup at once keep each other's answers entry by entry, and where one's rename lands over
//!   the other's between its read and its rename, the other's next write brings its answers back.
//!   Nothing locks.
//! - `<directory>/v1/<key>.learned/` beside it is the learned records' (device loss).

use super::{Check, DeviceChecksOptions, Target, CHECK_SUITE_VERSION, SUITE};
use metal2vulkan::device::{CheckAnswer, DeviceChecks, DeviceFact, DevicePolicy};
use sha2::{Digest, Sha256};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// The first line of every entry.
const HEADER: &str = "metal2vulkan device checks v1";

/// The per-user cache directory's subdirectory the answers are kept in by default.
const DEFAULT_NAME: &str = "metal2vulkan-device-checks";

/// A setup's key: its fields, and their digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Key {
    fields: Vec<(&'static str, String)>,
    digest: [u8; 32],
}

impl Key {
    /// The key of `fields`, each a name and its value (neither holding a line break).
    pub(crate) fn new(fields: Vec<(&'static str, String)>) -> Key {
        let mut hash = Sha256::new();
        hash.update(b"metal2vulkan/device-checks/v1\0");
        for (name, value) in &fields {
            hash.update(format!("{name}={value}\n").as_bytes());
        }
        Key {
            fields,
            digest: hash.finalize().into(),
        }
    }

    /// The key of `target`'s setup under `policy` (the module docs), its physical device's from
    /// `instance`.
    ///
    /// # Safety
    ///
    /// `target.physical` was enumerated from `instance`.
    pub(crate) unsafe fn of(
        instance: &ash::Instance,
        target: &Target<'_>,
        policy: DevicePolicy,
    ) -> Key {
        // SAFETY: the caller's.
        let properties = unsafe { instance.get_physical_device_properties(target.physical) };
        let description = target.description;
        let digest = |text: String| hex(&Sha256::digest(text.as_bytes()));
        let mut fields = vec![
            ("driver", format!("{:?}", description.driver_id)),
            ("vendor", format!("{:#06x}", description.vendor_id)),
            ("device", format!("{:#06x}", description.device_id)),
            (
                "driver-version",
                format!("{:#010x}", description.driver_version),
            ),
            ("pipeline-cache-uuid", hex(&properties.pipeline_cache_uuid)),
            ("os-build", os_build()),
        ];
        fields.extend(build_fields());
        fields.push(("enabled", digest(format!("{:?}", target.enabled))));
        fields.push(("policy", digest(format!("{policy:?}"))));
        Key::new(fields)
    }

    /// The digest in hex: the setup's name in the cache and in the feature status.
    pub(crate) fn hex(&self) -> String {
        hex(&self.digest)
    }
}

/// The key's fields of this build (the module docs): the suite's and the runtime's versions, and
/// the translator's and the runtime's source digests, which cover what each check is made of.
pub(crate) fn build_fields() -> [(&'static str, String); 4] {
    [
        ("suite", CHECK_SUITE_VERSION.to_string()),
        ("runtime", env!("CARGO_PKG_VERSION").to_owned()),
        (
            "translator-source",
            metal2vulkan::cache::SOURCE_DIGEST.to_owned(),
        ),
        ("runtime-source", env!("M2V_RUNTIME_SOURCE_ID").to_owned()),
    ]
}

/// `bytes` in lowercase hex.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The OS build the device's driver runs on: macOS's `kern.osversion` (`26A434`), elsewhere the
/// kernel's release (`uname -r`); empty where it does not read.
pub(crate) fn os_build() -> String {
    #[cfg(target_os = "macos")]
    {
        let mut text = [0u8; 64];
        let mut len = text.len();
        // SAFETY: the name is NUL-terminated and `len` is the buffer's size; the call writes at
        // most `len` bytes and sets `len` to the length written.
        let read = unsafe {
            libc::sysctlbyname(
                c"kern.osversion".as_ptr(),
                text.as_mut_ptr().cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if read != 0 {
            return String::new();
        }
        let text = &text[..len.min(text.len())];
        let text = text.split(|&b| b == 0).next().unwrap_or_default();
        String::from_utf8_lossy(text).into_owned()
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        // SAFETY: `utsname` is plain data, which `uname` fills.
        let mut name: libc::utsname = unsafe { std::mem::zeroed() };
        // SAFETY: `name` is a live `utsname`.
        if unsafe { libc::uname(&mut name) } != 0 {
            return String::new();
        }
        // SAFETY: `uname` NUL-terminates each field.
        let release = unsafe { std::ffi::CStr::from_ptr(name.release.as_ptr()) };
        release.to_string_lossy().into_owned()
    }
    #[cfg(not(unix))]
    {
        String::new()
    }
}

/// The platform's per-user cache directory's `metal2vulkan-device-checks`, as the shader cache
/// finds its own (`shader_cache::default_directory`): `~/Library/Caches` on macOS, else
/// `$XDG_CACHE_HOME` or `~/.cache`. `None` with no absolute home.
fn default_directory() -> Option<PathBuf> {
    let absolute = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    let home = || absolute(if cfg!(windows) { "USERPROFILE" } else { "HOME" });
    if cfg!(target_os = "macos") {
        return Some(home()?.join("Library/Caches").join(DEFAULT_NAME));
    }
    match absolute("XDG_CACHE_HOME") {
        Some(base) => Some(base.join(DEFAULT_NAME)),
        None => Some(home()?.join(".cache").join(DEFAULT_NAME)),
    }
}

/// Where the entries lie: the options' directory's `v1`.
pub(crate) struct Place {
    dir: PathBuf,
}

/// One setup's kept answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    /// When the checks ran, in seconds past the Unix epoch.
    pub(crate) checked_at: u64,
    answers: DeviceChecks,
}

impl Entry {
    /// The answer kept for `fact`.
    pub(crate) fn answer(&self, fact: DeviceFact) -> Option<&CheckAnswer> {
        self.answers.get(fact)
    }

    /// The answers kept for every fact of `check`, in its order, where all are: a making takes
    /// them and runs no check.
    pub(crate) fn answers_of(&self, check: &Check) -> Option<Vec<CheckAnswer>> {
        (check.facts.iter())
            .map(|&fact| self.answer(fact).cloned())
            .collect()
    }
}

impl Place {
    /// The place `options` name, or why there is none.
    pub(crate) fn of(options: &DeviceChecksOptions) -> Result<Place, String> {
        let root = match &options.directory {
            Some(directory) => directory.clone(),
            None => default_directory()
                .ok_or("no home directory for the default device-checks directory")?,
        };
        Ok(Place {
            dir: root.join("v1"),
        })
    }

    /// Its directory: the options' directory's `v1`.
    pub(crate) fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, key: &Key) -> PathBuf {
        self.dir.join(format!("{}.checks", key.hex()))
    }

    /// The entry of `key`, where one reads whole (the module docs).
    pub(crate) fn read(&self, key: &Key) -> Option<Entry> {
        let text = std::fs::read_to_string(self.path(key)).ok()?;
        parse(&text, key)
    }

    /// Keep every answer of `checks` but the unanswered in `key`'s entry, over what the entry
    /// holds (the module docs: each fact `checks` answers takes its answer, the entry's others
    /// stay), checked now; nothing where none is kept.
    pub(crate) fn write(&self, key: &Key, checks: &DeviceChecks) -> Result<(), String> {
        let checked_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        let mut kept = self
            .read(key)
            .map_or_else(DeviceChecks::new, |entry| entry.answers);
        for (fact, answer) in checks.iter() {
            if word(answer).is_some() {
                kept.insert(fact, answer.clone());
            }
        }
        let Some(text) = render(key, checked_at, &kept) else {
            return Ok(());
        };
        let error = |what: &str, e: std::io::Error| format!("{what} {}: {e}", self.dir.display());
        create_private_dir_all(&self.dir).map_err(|e| error("creating", e))?;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.subsec_nanos());
        let temporary = self.dir.join(format!(
            "{}.checks.{}.{nanos}.tmp",
            key.hex(),
            std::process::id()
        ));
        let written = new_private_file(&temporary)
            .and_then(|mut file| file.write_all(text.as_bytes()))
            .and_then(|()| std::fs::rename(&temporary, self.path(key)));
        if let Err(e) = written {
            // The temporary file is the process's own; a leftover one is the failure's.
            let _ = std::fs::remove_file(&temporary);
            return Err(error("writing an entry in", e));
        }
        Ok(())
    }
}

/// `dir` and its parents, made readable by the user alone where they are made.
pub(crate) fn create_private_dir_all(dir: &std::path::Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)
}

/// A new file at `path`, readable by the user alone; an existing one is an error.
pub(crate) fn new_private_file(path: &std::path::Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options.open(path)
}

/// An answer's word in an entry, and its reason; `None` for an unanswered check's, never kept.
fn word(answer: &CheckAnswer) -> Option<(&'static str, String)> {
    match answer {
        CheckAnswer::Metals => Some(("metals", String::new())),
        CheckAnswer::NotMetals(how) => Some(("not-metals", how.replace(['\n', '\r'], " "))),
        CheckAnswer::NotRun(why) => Some(("not-run", why.replace(['\n', '\r'], " "))),
        _ => None,
    }
}

/// The entry of `key` keeping `checks`' answers, checked at `checked_at`; `None` where it would
/// keep none.
pub(crate) fn render(key: &Key, checked_at: u64, checks: &DeviceChecks) -> Option<String> {
    let answers: Vec<String> = checks
        .iter()
        .filter_map(|(fact, answer)| {
            let (word, why) = word(answer)?;
            let line = format!("answer {} {} {word}", fact.kind(), fact.name());
            Some(match why.is_empty() {
                true => line,
                false => format!("{line} {why}"),
            })
        })
        .collect();
    if answers.is_empty() {
        return None;
    }
    let mut text = format!("{HEADER}\nkey {}\n", key.hex());
    for (name, value) in &key.fields {
        text.push_str(&format!("field {name} {value}\n"));
    }
    text.push_str(&format!("checked-at {checked_at}\n"));
    for line in answers {
        text.push_str(&line);
        text.push('\n');
    }
    let digest = hex(&Sha256::digest(text.as_bytes()));
    text.push_str(&format!("sha256 {digest}\n"));
    Some(text)
}

/// The entry `text` holds for `key`, where it reads whole (the module docs).
pub(crate) fn parse(text: &str, key: &Key) -> Option<Entry> {
    let body = text.strip_suffix('\n')?;
    let (body, digest) = body.rsplit_once('\n')?;
    let body = format!("{body}\n");
    if digest.strip_prefix("sha256 ")? != hex(&Sha256::digest(body.as_bytes())) {
        return None;
    }
    let mut lines = body.lines();
    if lines.next()? != HEADER {
        return None;
    }
    if lines.next()?.strip_prefix("key ")? != key.hex() {
        return None;
    }
    let mut checked_at = None;
    let mut answers = DeviceChecks::new();
    for line in lines {
        if line.starts_with("field ") {
            continue;
        }
        if let Some(at) = line.strip_prefix("checked-at ") {
            checked_at = Some(at.parse().ok()?);
            continue;
        }
        let mut parts = line.strip_prefix("answer ")?.splitn(4, ' ');
        let (kind, name, word) = (parts.next()?, parts.next()?, parts.next()?);
        let why = parts.next().unwrap_or_default().to_owned();
        let fact = *SUITE
            .iter()
            .flat_map(|check| check.facts)
            .find(|fact| fact.kind() == kind && fact.name() == name)?;
        let answer = match word {
            "metals" => CheckAnswer::Metals,
            "not-metals" => CheckAnswer::NotMetals(why),
            "not-run" => CheckAnswer::NotRun(why),
            _ => return None,
        };
        answers.insert(fact, answer);
    }
    Some(Entry {
        checked_at: checked_at?,
        answers,
    })
}
