//! `format-scout`: the empirical side of the format coverage matrix. It drives every Metal pixel
//! format through the runtime's public API, one use at a time, and records what each use did: made,
//! refused (a typed error), died (the process ended on a signal, a panic, a GPU wait or a timeout),
//! vvl (Vulkan's validation layer reported), or skip (the scout drove nothing). A made use that
//! reads texels back records a digest of them, for comparison with other venues and with Metal's
//! answer to the same writes.
//!
//! - `format-scout run --venue NAME --device-type cpu|gpu --out FILE [--phases LIST]
//!   [--first LIST] [--then LIST] [--formats LIST] [--cell-timeout SECONDS]`
//!   The parent: for each phase, each format in order, it runs a child process over the format's
//!   cells of the phase that `--out` does not hold yet, and appends a row per cell to `--out`
//!   (a TSV of venue, format, use, variant, cell, detail, evidence) as each cell ends. A child
//!   prints `BEGIN <index>` before a cell and `CELL <index> <cell> <detail>` after it, flushed, so
//!   a cell the child dies in (a signal, an exit, no `CELL` within `--cell-timeout`, default 90 s)
//!   is recorded as `died` with the signal and the end of the child's stderr, printed as a
//!   `CRASH` line with its repro command, and the parent starts a new child past it.
//!   Phases, in order (default all): `crash` (every format numbered 500 or more: surface
//!   creation, sampling, render targets, views and aliases), `first` (the `--first` formats,
//!   every cell), `attachments` (formats 26, 501, 556, 624, 651 and 652: pipeline attachments,
//!   render targets and clears), `fill` (compressed, 4:2:2, stencil and combined depth/stencil
//!   formats: fills), and `rest` (every format, the `--then` formats first, every cell).
//!   Formats are named as Metal names them, with or without `MTLPixelFormat`, or by number.
//!   `--side metal` (macOS) drives the same cells through plain Metal (`format_scout/metal.rs`),
//!   the reference the runtime's rows are compared with; `--shard K/N` keeps the formats at
//!   positions K, K+N, ... of `PixelFormat::ALL`, so one venue's run can be split across hosts.
//! - `format-scout child --format F --cells I,J,... --device-type cpu|gpu --cache DIR`: one
//!   child; runs the format's cells of those indices on one device.
//! - `format-scout list --format F`: prints the format's cells, one `index use variant` a line.
//! - `--dump DIR` (a child's option): every byte read back whose digest a cell records is also
//!   written to `DIR/<cell>.<k>.bin` (`k` counting the cell's digests from 0), to compare venues
//!   byte by byte where their digests differ.
//!
//! The last cells are the guest-reached surface shapes (the coordinator, 2026-10-09): a texture
//! of the format over plane 0, and over plane 1 (the chroma plane, at its offset, half the size),
//! of a two-plane 10-bit 4:2:0 surface (`x420`), read as sampled 2D; and a draw blended source
//! over or additively onto a target loaded with the pattern: plane 0 of the format's surface with
//! planes, a placement heap's texture, or the device's own.
//!
//! The functions are the fixture `format_scout` (validation/fixtures/public/engine), our own
//! MSL. Every texture is 16 x 16; texels are written as a byte pattern seeded by the format's
//! number, through a buffer blit (a host copy for the host-copy cells), and read back through a
//! blit into a buffer (or the kernels' outputs). The device carries the validation layer.

use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::format::{aspects, direct_format, sampled_type, ycbcr_format};
use metal2vulkan::metal::format::{ChromaSubsampling, YCbCrFormat};
use metal2vulkan::metal::{
    BlendFactor, PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureSwizzle,
    TextureSwizzleChannels, TextureType, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, SampledType,
    StageSource,
};
use metal2vulkan::vk::ImageAspectFlags;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, ClearColor, CommandBuffer, CommandBufferStatus, ComputePipelineState,
    CpuCacheMode, Device, Error, HazardTrackingMode, HeapDescriptor, HeapType, IoSurfacePlane,
    IoSurfaceYCbCr, Library, LoadAction, Origin, PipelineOptions, Region, RenderPassDescriptor,
    RenderPipelineState, ResourceOptions, SamplerState, Size, StorageMode, StoreAction, Texture,
    TextureDescriptor,
};
use metal2vulkan_validation::air_library::AirLibrary;
use metal2vulkan_validation::candidate::runtime_case::{
    open_device, read_buffer, validated_options, Messages,
};
use metal2vulkan_validation::gpu::DeviceType;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_os = "macos")]
#[path = "format_scout/metal.rs"]
mod metal;
#[cfg(target_os = "macos")]
#[path = "format_scout/surface.rs"]
mod surface;

const USAGE: &str = "usage: format-scout run --venue NAME --device-type cpu|gpu --out FILE \
                     [--phases LIST] [--first LIST] [--then LIST] [--formats LIST] \
                     [--cell-timeout SECONDS] [--side runtime|metal] [--shard K/N]\n\
                     \x20      format-scout child --format F --cells I,J,... --device-type cpu|gpu \
                     --cache DIR [--side runtime|metal] [--dump DIR]\n\
                     \x20      format-scout list --format F";

/// The side of every texture the scout makes.
const SIDE: u64 = 16;
/// How long a command buffer may take before the cell is recorded as died.
const GPU_WAIT: Duration = Duration::from_secs(20);
/// The phases, in the order they run.
const PHASES: [&str; 5] = ["crash", "first", "attachments", "fill", "rest"];
/// The uses the `crash` phase drives: where the formats numbered 500 and up crashed a consumer.
const CRASH_USES: [&str; 8] = [
    "create.iosurface",
    "sample.nearest",
    "sample.linear",
    "render",
    "clear",
    "view.same",
    "view.swizzle",
    "alias",
];
/// The formats the `attachments` phase asks pipeline attachments of.
const ATTACHMENT_FORMATS: [u32; 6] = [26, 501, 556, 624, 651, 652];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("run") => run(&options(&args[1..])),
        Some("child") => child(&options(&args[1..])),
        Some("list") => list(&options(&args[1..])),
        _ => Err(USAGE.to_owned()),
    };
    if let Err(error) = result {
        eprintln!("format-scout: {error}");
        std::process::exit(2);
    }
}

/// `--name value` pairs.
fn options(args: &[String]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut it = args.iter();
    while let Some(name) = it.next() {
        if let Some(name) = name.strip_prefix("--") {
            map.insert(name.to_owned(), it.next().cloned().unwrap_or_default());
        }
    }
    map
}

fn required<'a>(options: &'a BTreeMap<String, String>, name: &str) -> Result<&'a str, String> {
    options
        .get(name)
        .map(String::as_str)
        .ok_or_else(|| format!("--{name} is required\n{USAGE}"))
}

/// Metal's name of `pf` without `MTLPixelFormat`, as the matrix's files name formats.
fn short(pf: PixelFormat) -> &'static str {
    pf.name()
        .strip_prefix("MTLPixelFormat")
        .unwrap_or(pf.name())
}

/// The format `text` names: a Metal name with or without its prefix, or a number.
fn format_named(text: &str) -> Result<PixelFormat, String> {
    let text = text.trim();
    if let Ok(raw) = text.parse::<u32>() {
        return PixelFormat::try_from(raw).map_err(|e| format!("format {raw}: {e}"));
    }
    PixelFormat::from_name(text)
        .or_else(|| PixelFormat::from_name(&format!("MTLPixelFormat{text}")))
        .ok_or_else(|| format!("no format named {text}"))
}

fn formats_named(list: Option<&String>) -> Result<Vec<PixelFormat>, String> {
    list.map_or(Ok(Vec::new()), |list| {
        list.split(',')
            .filter(|s| !s.trim().is_empty())
            .map(format_named)
            .collect()
    })
}

// ---------------------------------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------------------------------

/// What one cell does.
#[derive(Clone, Copy, Debug)]
enum Op {
    Own(StorageMode, TextureType),
    Heap(bool),
    OverBuffer,
    OverSurface {
        planar: bool,
        render: bool,
    },
    ViewSame,
    ViewSwizzle,
    HostCopy(StorageMode),
    BlitBuffer,
    BlitCopy,
    Mipmaps,
    FillBytes,
    FillColor,
    Read,
    Write,
    ReadWrite,
    Sample(bool),
    Render {
        blend: bool,
    },
    Clear,
    Msaa {
        resolve: bool,
    },
    Attachment {
        blend: bool,
        depth_or_stencil: bool,
    },
    Alias {
        texture: bool,
    },
    // ---- The guest-reached surface shapes (coordinator 2026-10-09) ----
    /// A texture of the format over plane `plane` of a two-plane 10-bit 4:2:0 surface (OSType
    /// `x420`, plane 1 its chroma plane at its offset, half the size), read as sampled 2D.
    OverX420 {
        plane: u64,
    },
    /// A draw with `blend` over a target of `form` loaded with the pattern.
    BlendLoaded {
        form: Form,
        blend: Blend,
    },
}

/// A blend a drawing cell's color attachment takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Blend {
    /// None.
    Off,
    /// Source over: color `src·αs + dst·(1 − αs)`, alpha `αs + αd·(1 − αs)`.
    SourceOver,
    /// Additive: color and alpha `src + dst`.
    Additive,
}

impl Blend {
    /// Source over where `on`, else none.
    fn source_over(on: bool) -> Blend {
        if on {
            Blend::SourceOver
        } else {
            Blend::Off
        }
    }

    /// The color and alpha factors, source then destination: none for [`Blend::Off`].
    fn factors(self) -> Option<[BlendFactor; 4]> {
        use BlendFactor::{One, OneMinusSourceAlpha, SourceAlpha};
        match self {
            Blend::Off => None,
            Blend::SourceOver => Some([SourceAlpha, OneMinusSourceAlpha, One, OneMinusSourceAlpha]),
            Blend::Additive => Some([One, One, One, One]),
        }
    }
}

/// Where a loaded blend cell's target lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// Plane 0 of the format's surface with planes (as `create.iosurface` `planar=true`).
    Surface,
    /// Placed in a placement heap at offset 0.
    Heap,
    /// The device's own private texture.
    Own,
}

/// The OSType of a two-plane 10-bit 4:2:0 surface, as CoreVideo names it.
const X420: [u8; 4] = *b"x420";
/// Plane 1's first byte in every two-plane surface the scout makes, and the bytes between rows.
const CHROMA_OFFSET: u64 = 4096;
const SURFACE_ROW: u64 = 256;

/// One cell: its use and variant as the matrix's files name them, and what it does.
#[derive(Clone, Debug)]
struct Cell {
    use_: &'static str,
    variant: String,
    op: Op,
}

fn cell(use_: &'static str, variant: &str, op: Op) -> Cell {
    Cell {
        use_,
        variant: variant.to_owned(),
        op,
    }
}

/// Every cell of a format, in index order (the same for every format).
fn cells() -> Vec<Cell> {
    use StorageMode::{Managed, Memoryless, Private, Shared};
    let mut cells = vec![
        cell(
            "create.own",
            "storage=private type=2D",
            Op::Own(Private, TextureType::Type2D),
        ),
        cell(
            "create.own",
            "storage=shared type=2D",
            Op::Own(Shared, TextureType::Type2D),
        ),
        cell(
            "create.own",
            "storage=managed type=2D",
            Op::Own(Managed, TextureType::Type2D),
        ),
        cell(
            "create.own",
            "storage=memoryless type=2D",
            Op::Own(Memoryless, TextureType::Type2D),
        ),
    ];
    for (name, ty) in [
        ("1D", TextureType::Type1D),
        ("1DArray", TextureType::Type1DArray),
        ("2DArray", TextureType::Type2DArray),
        ("Cube", TextureType::Cube),
        ("CubeArray", TextureType::CubeArray),
        ("3D", TextureType::Type3D),
        ("2DMultisample", TextureType::Type2DMultisample),
    ] {
        cells.push(Cell {
            use_: "create.own",
            variant: format!("storage=private type={name}"),
            op: Op::Own(Private, ty),
        });
    }
    cells.extend([
        cell(
            "create.heap",
            "storage=private type=2D heap=automatic",
            Op::Heap(false),
        ),
        cell(
            "create.heap",
            "storage=private type=2D heap=placement",
            Op::Heap(true),
        ),
        cell("create.buffer", "storage=shared type=2D", Op::OverBuffer),
    ]);
    for planar in [false, true] {
        for render in [false, true] {
            cells.push(Cell {
                use_: "create.iosurface",
                variant: format!(
                    "storage=shared type=2D planar={planar} usage={}",
                    if render { "render" } else { "read" }
                ),
                op: Op::OverSurface { planar, render },
            });
        }
    }
    cells.extend([
        cell("view.same", "", Op::ViewSame),
        cell("view.swizzle", "", Op::ViewSwizzle),
        cell("replace_get", "storage=shared", Op::HostCopy(Shared)),
        cell("replace_get", "storage=managed", Op::HostCopy(Managed)),
        cell("blit.buffer", "", Op::BlitBuffer),
        cell("blit.copy", "", Op::BlitCopy),
        cell("blit.mipmaps", "", Op::Mipmaps),
        cell("fill", "bytes", Op::FillBytes),
        cell("fill", "color", Op::FillColor),
        cell("read", "", Op::Read),
        cell("write", "", Op::Write),
        cell("read_write", "", Op::ReadWrite),
        cell("sample.nearest", "", Op::Sample(false)),
        cell("sample.linear", "", Op::Sample(true)),
        cell("render", "", Op::Render { blend: false }),
        cell("render", "blend", Op::Render { blend: true }),
        cell("clear", "", Op::Clear),
        cell("msaa", "samples=4", Op::Msaa { resolve: false }),
        cell("resolve", "samples=4", Op::Msaa { resolve: true }),
        cell(
            "pipeline.attachment",
            "color",
            Op::Attachment {
                blend: false,
                depth_or_stencil: false,
            },
        ),
        cell(
            "pipeline.attachment",
            "color blend",
            Op::Attachment {
                blend: true,
                depth_or_stencil: false,
            },
        ),
        cell(
            "pipeline.attachment",
            "depth_or_stencil",
            Op::Attachment {
                blend: false,
                depth_or_stencil: true,
            },
        ),
        cell("alias", "buffer", Op::Alias { texture: false }),
        cell("alias", "texture", Op::Alias { texture: true }),
    ]);
    // ---- The guest-reached surface shapes (coordinator 2026-10-09), after every other cell so
    // the other cells keep their indices ----
    for plane in [0, 1] {
        cells.push(Cell {
            use_: "create.iosurface",
            variant: format!("storage=shared type=2D surface=x420 plane={plane} usage=read"),
            op: Op::OverX420 { plane },
        });
    }
    for (form, where_) in [
        (Form::Surface, "surface planar=true"),
        (Form::Heap, "heap=placement"),
        (Form::Own, "storage=private"),
    ] {
        for (blend, name) in [
            (Blend::SourceOver, "source-over"),
            (Blend::Additive, "additive"),
        ] {
            cells.push(Cell {
                use_: "render.loaded",
                variant: format!("blend={name} {where_}"),
                op: Op::BlendLoaded { form, blend },
            });
        }
    }
    cells
}

/// Whether `pf` is a block-compressed format.
fn compressed(pf: PixelFormat) -> bool {
    let name = short(pf);
    ["BC", "ASTC_", "ETC2_", "EAC_", "PVRTC_"]
        .iter()
        .any(|p| name.starts_with(p))
}

/// Whether phase `phase` drives cell `cell` of `pf`.
fn in_phase(phase: &str, pf: PixelFormat, cell: &Cell, first: &[PixelFormat]) -> bool {
    match phase {
        "crash" => pf.as_raw() >= 500 && CRASH_USES.contains(&cell.use_),
        "first" => first.contains(&pf),
        "attachments" => {
            ATTACHMENT_FORMATS.contains(&pf.as_raw())
                && matches!(cell.use_, "pipeline.attachment" | "render" | "clear")
        }
        "fill" => {
            let a = aspects(pf).unwrap_or(ImageAspectFlags::COLOR);
            let wanted = compressed(pf)
                || matches!(pf, PixelFormat::GBGR422 | PixelFormat::BGRG422)
                || a.contains(ImageAspectFlags::STENCIL);
            wanted && cell.use_ == "fill"
        }
        _ => true,
    }
}

fn list(options: &BTreeMap<String, String>) -> Result<(), String> {
    let pf = format_named(required(options, "format")?)?;
    for (i, c) in cells().iter().enumerate() {
        println!("{i}\t{}\t{}\t{}", short(pf), c.use_, c.variant);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The parent
// ---------------------------------------------------------------------------------------------

/// A line a child printed, or the end of its output.
enum Line {
    Text(String),
    End,
}

/// The rows the output file holds already: (format, use, variant).
fn done_keys(path: &Path) -> BTreeSet<(String, String, String)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return BTreeSet::new();
    };
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split('\t').collect();
            (f.len() >= 4).then(|| (f[1].to_owned(), f[2].to_owned(), f[3].to_owned()))
        })
        .collect()
}

/// `text` on one line of at most `max` characters, without tabs.
fn one_line(text: &str, max: usize) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let flat = flat.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > max {
        let cut: String = flat.chars().take(max).collect();
        format!("{cut}...")
    } else {
        flat
    }
}

struct Parent {
    venue: String,
    device_type: String,
    /// `runtime` or `metal`: which side the children drive.
    side: String,
    out: std::fs::File,
    run_id: String,
    cache: PathBuf,
    cell_timeout: Duration,
    exe: PathBuf,
    crashes: usize,
    rows: usize,
}

impl Parent {
    fn row(&mut self, pf: PixelFormat, index: usize, verdict: &str, detail: &str) {
        let all = cells();
        let c = &all[index];
        let evidence = format!(
            "run={} repro=format-scout child --format {} --cells {index} --device-type {} --side {}",
            self.run_id,
            short(pf),
            self.device_type,
            self.side
        );
        let line = format!(
            "{}\t{}\t{}\t{}\t{verdict}\t{}\t{evidence}\n",
            self.venue,
            short(pf),
            c.use_,
            c.variant,
            one_line(detail, 400)
        );
        if self.out.write_all(line.as_bytes()).is_err() || self.out.flush().is_err() {
            eprintln!("format-scout: could not write a row");
        }
        self.rows += 1;
        if verdict == "died" {
            self.crashes += 1;
            println!(
                "CRASH\t{}\t{}\t{}\t{}\t{}\trepro: {} child --format {} --cells {index} \
                 --device-type {} --side {} --cache DIR",
                self.venue,
                short(pf),
                c.use_,
                c.variant,
                one_line(detail, 400),
                self.exe.display(),
                short(pf),
                self.device_type,
                self.side
            );
            let _ = std::io::stdout().flush();
        }
    }

    /// Runs the cells `indices` of `pf` in children until each has a row.
    fn format(&mut self, pf: PixelFormat, indices: Vec<usize>) {
        let mut pending: VecDeque<usize> = indices.into();
        while !pending.is_empty() {
            let list: Vec<String> = pending.iter().map(usize::to_string).collect();
            let mut command = Command::new(&self.exe);
            if self.side == "metal" {
                // Plain Metal, the reference: none of Metal's debug and validation layers, which
                // turn answers plain Metal makes into aborts and the other way round.
                for name in [
                    "MTL_DEBUG_LAYER",
                    "MTL_DEBUG_LAYER_ERROR_MODE",
                    "MTL_SHADER_VALIDATION",
                    "METAL_DEVICE_WRAPPER_TYPE",
                ] {
                    command.env_remove(name);
                }
            }
            let spawned = command
                .args(["child", "--format", short(pf), "--cells", &list.join(",")])
                .args([
                    "--device-type",
                    &self.device_type,
                    "--side",
                    &self.side,
                    "--cache",
                ])
                .arg(&self.cache)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .process_group(0)
                .spawn();
            let mut child = match spawned {
                Ok(child) => child,
                Err(e) => {
                    for i in pending.drain(..) {
                        self.row(pf, i, "skip", &format!("the child did not start: {e}"));
                    }
                    return;
                }
            };
            let (tx, rx) = mpsc::channel();
            let stdout = child.stdout.take().expect("piped");
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if tx.send(Line::Text(line)).is_err() {
                        return;
                    }
                }
                let _ = tx.send(Line::End);
            });
            let tail: Arc<Mutex<VecDeque<String>>> = Arc::default();
            let stderr = child.stderr.take().expect("piped");
            let kept = Arc::clone(&tail);
            let stderr_thread = std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let mut reader = BufReader::new(stderr);
                let mut buf = [0u8; 8192];
                while let Ok(n) = reader.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&buf[..n]);
                    while let Some(at) = bytes.iter().position(|&b| b == b'\n') {
                        let line: Vec<u8> = bytes.drain(..=at).collect();
                        let mut kept = kept.lock().unwrap_or_else(|p| p.into_inner());
                        kept.push_back(String::from_utf8_lossy(&line).trim_end().to_owned());
                        while kept.len() > 12 {
                            kept.pop_front();
                        }
                    }
                }
            });
            let mut in_flight: Option<usize> = None;
            let mut timed_out = false;
            let mut last = Instant::now();
            loop {
                // A cell has `cell_timeout`; the child's start (the device opening) and the time
                // between two cells have twice that.
                let bound = if in_flight.is_some() {
                    self.cell_timeout
                } else {
                    self.cell_timeout * 2
                };
                let wait = bound.saturating_sub(last.elapsed());
                match rx.recv_timeout(wait.max(Duration::from_millis(1))) {
                    Ok(Line::Text(line)) => {
                        last = Instant::now();
                        let f: Vec<&str> = line.splitn(4, '\t').collect();
                        match f.as_slice() {
                            ["BEGIN", i] => in_flight = i.parse().ok(),
                            ["CELL", i, verdict, detail] => {
                                if let Ok(i) = i.parse::<usize>() {
                                    self.row(pf, i, verdict, detail);
                                    pending.retain(|&p| p != i);
                                }
                                in_flight = None;
                            }
                            ["DEVICE", "fail", detail, ..] => {
                                for i in pending.drain(..) {
                                    self.row(pf, i, "skip", &format!("no device: {detail}"));
                                }
                            }
                            _ => {}
                        }
                    }
                    Ok(Line::End) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        // SAFETY: kill(2) of the child's own process group.
                        unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL) };
                        timed_out = true;
                        break;
                    }
                }
            }
            let status = child.wait();
            let _ = stderr_thread.join();
            let tail = {
                let kept = tail.lock().unwrap_or_else(|p| p.into_inner());
                kept.iter()
                    .rev()
                    .filter(|l| !l.trim().is_empty())
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join(" | ")
            };
            let how = match (&status, timed_out) {
                (_, true) => format!("timeout after {} s", self.cell_timeout.as_secs()),
                (Ok(s), _) => match (s.signal(), s.code()) {
                    (Some(sig), _) => format!("signal {sig}"),
                    (None, Some(code)) => format!("exit {code}"),
                    _ => "ended".to_owned(),
                },
                (Err(e), _) => format!("no status: {e}"),
            };
            match in_flight {
                Some(i) => {
                    self.row(pf, i, "died", &format!("{how}; stderr: {tail}"));
                    pending.retain(|&p| p != i);
                }
                None if !pending.is_empty() => {
                    // The child ended between cells (or before its first): the next cell is
                    // where it stopped, so the scout moves past it rather than loop.
                    if let Some(i) = pending.pop_front() {
                        self.row(
                            pf,
                            i,
                            "died",
                            &format!("{how} before the cell began; stderr: {tail}"),
                        );
                    }
                }
                None => {}
            }
        }
    }
}

fn run(options: &BTreeMap<String, String>) -> Result<(), String> {
    let venue = required(options, "venue")?.to_owned();
    let device_type = required(options, "device-type")?.to_owned();
    DeviceType::parse(&device_type)?;
    let side = options
        .get("side")
        .cloned()
        .unwrap_or_else(|| "runtime".into());
    if side != "runtime" && side != "metal" {
        return Err(format!("--side {side}: runtime or metal"));
    }
    let out_path = PathBuf::from(required(options, "out")?);
    let phases: Vec<String> = options.get("phases").map_or_else(
        || PHASES.iter().map(|s| (*s).to_owned()).collect(),
        |p| p.split(',').map(str::to_owned).collect(),
    );
    for p in &phases {
        if !PHASES.contains(&p.as_str()) {
            return Err(format!("no phase {p}; the phases are {}", PHASES.join(",")));
        }
    }
    let first = formats_named(options.get("first"))?;
    let then = formats_named(options.get("then"))?;
    let only = formats_named(options.get("formats"))?;
    // `--shard K/N`: the formats at positions K, K+N, ... of `PixelFormat::ALL` (a venue's run
    // split across hosts).
    let shard = match options.get("shard") {
        None => None,
        Some(text) => {
            let (k, n) = text
                .split_once('/')
                .and_then(|(k, n)| Some((k.parse::<usize>().ok()?, n.parse::<usize>().ok()?)))
                .filter(|&(k, n)| n > 0 && k < n)
                .ok_or_else(|| format!("--shard {text}: K/N with K < N"))?;
            Some((k, n))
        }
    };
    let in_shard = |pf: &PixelFormat| {
        shard.is_none_or(|(k, n)| {
            PixelFormat::ALL
                .iter()
                .position(|f| f == pf)
                .is_some_and(|at| at % n == k)
        })
    };
    let cell_timeout = Duration::from_secs(
        options
            .get("cell-timeout")
            .map_or(Ok(90), |s| s.parse())
            .map_err(|e| format!("--cell-timeout: {e}"))?,
    );
    let fresh = !out_path.exists();
    let mut done = done_keys(&out_path);
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .map_err(|e| format!("{}: {e}", out_path.display()))?;
    if fresh {
        out.write_all(b"venue\tformat\tuse\tvariant\tcell\tdetail\tevidence\n")
            .map_err(|e| e.to_string())?;
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let cache = std::env::temp_dir().join(format!("format-scout-{}-{stamp}", std::process::id()));
    std::fs::create_dir_all(&cache).map_err(|e| format!("{}: {e}", cache.display()))?;
    let mut parent = Parent {
        venue,
        device_type,
        side,
        out,
        run_id: format!("{}-{stamp}", required(options, "venue")?),
        cache: cache.clone(),
        cell_timeout,
        exe: std::env::current_exe().map_err(|e| e.to_string())?,
        crashes: 0,
        rows: 0,
    };
    let all = cells();
    let started = Instant::now();
    for phase in &phases {
        let mut order: Vec<PixelFormat> = Vec::new();
        let mut seen = BTreeSet::new();
        let lead = if phase == "rest" { &then } else { &first };
        for pf in lead.iter().chain(PixelFormat::ALL) {
            if seen.insert(*pf) && (only.is_empty() || only.contains(pf)) && in_shard(pf) {
                order.push(*pf);
            }
        }
        let phase_start = Instant::now();
        let rows_before = parent.rows;
        for pf in order {
            let indices: Vec<usize> = all
                .iter()
                .enumerate()
                .filter(|(_, c)| in_phase(phase, pf, c, &first))
                .filter(|(_, c)| {
                    !done.contains(&(short(pf).to_owned(), c.use_.to_owned(), c.variant.clone()))
                })
                .map(|(i, _)| i)
                .collect();
            if indices.is_empty() {
                continue;
            }
            for &i in &indices {
                done.insert((
                    short(pf).to_owned(),
                    all[i].use_.to_owned(),
                    all[i].variant.clone(),
                ));
            }
            parent.format(pf, indices);
        }
        println!(
            "PHASE\t{phase}\trows {}\tin {} s\tcrashes so far {}",
            parent.rows - rows_before,
            phase_start.elapsed().as_secs(),
            parent.crashes
        );
        let _ = std::io::stdout().flush();
    }
    let _ = std::fs::remove_dir_all(&cache);
    println!(
        "DONE\trows {}\tcrashes {}\tin {} s",
        parent.rows,
        parent.crashes,
        started.elapsed().as_secs()
    );
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The child
// ---------------------------------------------------------------------------------------------

/// Why a cell made nothing.
enum Fail {
    /// A typed refusal: the runtime's error.
    Refused(String),
    /// The GPU's work failed or did not end.
    Died(String),
    /// The scout drove nothing.
    Skip(String),
}

impl From<Error> for Fail {
    fn from(e: Error) -> Fail {
        Fail::Refused(format!("{e:?}"))
    }
}

type Done = Result<String, Fail>;

/// How a function reads a format's texels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Float,
    Int,
    Uint,
    Depth,
}

impl Class {
    fn suffix(self) -> &'static str {
        match self {
            Class::Float => "f",
            Class::Int => "i",
            Class::Uint => "u",
            Class::Depth => "d",
        }
    }

    /// The bytes a kernel writes for each texel it reads.
    fn out_bytes(self) -> u64 {
        if self == Class::Depth {
            4
        } else {
            16
        }
    }
}

fn aspects_of(pf: PixelFormat) -> ImageAspectFlags {
    aspects(pf).unwrap_or(ImageAspectFlags::COLOR)
}

fn has_depth(pf: PixelFormat) -> bool {
    pf == PixelFormat::AnyDepth || aspects_of(pf).contains(ImageAspectFlags::DEPTH)
}

fn has_stencil(pf: PixelFormat) -> bool {
    pf == PixelFormat::AnyStencil || aspects_of(pf).contains(ImageAspectFlags::STENCIL)
}

fn class(pf: PixelFormat) -> Option<Class> {
    if has_depth(pf) {
        return Some(Class::Depth);
    }
    if has_stencil(pf) {
        return Some(Class::Uint);
    }
    match sampled_type(pf)? {
        SampledType::Float => Some(Class::Float),
        SampledType::Sint => Some(Class::Int),
        SampledType::Uint => Some(Class::Uint),
        _ => None,
    }
}

/// The bytes of a 16 x 16 image of `pf` a copy moves: (bytes per row, rows, bytes of a unit,
/// units across, units down), where a unit is a texel or a compressed block.
#[derive(Clone, Copy, Debug)]
struct Layout {
    row: u64,
    rows: u64,
    unit: u64,
    across: u64,
    down: u64,
}

fn layout(pf: PixelFormat) -> Option<Layout> {
    let (unit, w, h) = if has_depth(pf) && has_stencil(pf) {
        // The depth of a combined format, as `DepthFromDepthStencil` moves it.
        (4, 1, 1)
    } else if has_depth(pf) {
        (
            if pf == PixelFormat::Depth16Unorm {
                2
            } else {
                4
            },
            1,
            1,
        )
    } else if has_stencil(pf) {
        (1, 1, 1)
    } else {
        let block = direct_format(pf)?.block()?;
        let name = short(pf);
        let (w, h) = if block.texels == 1 {
            (1, 1)
        } else if let Some(dims) = name.strip_prefix("ASTC_") {
            let dims = dims.split('_').next().unwrap_or("4x4");
            let mut it = dims.split('x').map(|d| d.parse::<u64>().unwrap_or(4));
            (it.next().unwrap_or(4), it.next().unwrap_or(4))
        } else if name.starts_with("PVRTC") && name.contains("2BPP") {
            (8, 4)
        } else if block.texels == 2 {
            (2, 1)
        } else {
            (4, 4)
        };
        (u64::from(block.size), w, h)
    };
    let across = SIDE.div_ceil(w);
    let down = SIDE.div_ceil(h);
    Some(Layout {
        row: across * unit,
        rows: down,
        unit,
        across,
        down,
    })
}

fn blit_option(pf: PixelFormat) -> BlitOption {
    if has_depth(pf) && has_stencil(pf) {
        BlitOption::DepthFromDepthStencil
    } else if short(pf).starts_with("PVRTC") {
        BlitOption::RowLinearPVRTC
    } else {
        BlitOption::None
    }
}

/// `n` bytes of a pattern seeded by `seed` (xorshift).
fn pattern(seed: u32, n: u64) -> Vec<u8> {
    let mut x = u64::from(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn digest(bytes: &[u8]) -> String {
    if let Some(dir) = DUMP.get() {
        let cell = CURRENT.with(|c| c.get());
        let k = DUMPED.with(|d| {
            let k = d.get();
            d.set(k + 1);
            k
        });
        let _ = std::fs::write(dir.join(format!("{cell}.{k}.bin")), bytes);
    }
    Sha256::digest(bytes)
        .iter()
        .take(6)
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn options_of(storage: StorageMode) -> ResourceOptions {
    ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A 16 x 16 two-dimensional descriptor of `pf`.
fn desc(pf: PixelFormat, usage: TextureUsage, storage: StorageMode) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    d.usage = usage;
    d.resource_options = options_of(storage);
    d
}

/// The usage a format's creation cells ask: a depth, stencil or color target where it can be
/// one, read everywhere.
fn read_usage() -> TextureUsage {
    TextureUsage::SHADER_READ
}

struct Child {
    pf: PixelFormat,
    device: Device,
    queue: metal2vulkan_runtime::CommandQueue,
    messages: Messages,
    library: Library,
    kernels: HashMap<String, ComputePipelineState>,
    samplers: HashMap<bool, SamplerState>,
}

impl Child {
    fn commit(&self, cb: &CommandBuffer) -> Result<(), Fail> {
        cb.commit()?;
        if cb.wait_until_completed_for(GPU_WAIT).is_err() {
            println!(
                "CELL\t{}\tdied\tthe command buffer did not complete within {} s",
                CURRENT.with(|c| c.get()),
                GPU_WAIT.as_secs()
            );
            let _ = std::io::stdout().flush();
            std::process::exit(4);
        }
        if cb.status() != CommandBufferStatus::Completed || cb.error().is_some() {
            return Err(Fail::Died(format!(
                "command buffer {:?}: {:?}",
                cb.status(),
                cb.error()
            )));
        }
        Ok(())
    }

    fn buffer(&self, bytes: &[u8]) -> Result<Buffer, Fail> {
        let b = self
            .device
            .new_buffer(bytes.len().max(16) as u64, ResourceOptions::shared())?;
        let contents = b
            .contents()
            .ok_or_else(|| Fail::Skip("a shared buffer without contents".into()))?;
        // SAFETY: the buffer is shared, at least `bytes.len()` long, and no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len());
        }
        Ok(b)
    }

    fn read(&self, buffer: &Buffer, length: u64) -> Result<Vec<u8>, Fail> {
        read_buffer(buffer, 0, length).map_err(|e| Fail::Skip(format!("read back: {e:?}")))
    }

    fn kernel(&mut self, name: &str) -> Result<ComputePipelineState, Fail> {
        if let Some(k) = self.kernels.get(name) {
            return Ok(k.clone());
        }
        let function = self.library.new_function_with_name(name)?;
        let state = self.device.new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )?;
        self.kernels.insert(name.to_owned(), state.clone());
        Ok(state)
    }

    fn sampler(&mut self, linear: bool) -> Result<SamplerState, Fail> {
        if let Some(s) = self.samplers.get(&linear) {
            return Ok(s.clone());
        }
        let mut d = SamplerDescriptor::default();
        let filter = if linear {
            SamplerMinMagFilter::Linear
        } else {
            SamplerMinMagFilter::Nearest
        };
        d.min_filter = filter;
        d.mag_filter = filter;
        let s = self.device.new_sampler_state(&d)?;
        self.samplers.insert(linear, s.clone());
        Ok(s)
    }

    fn texture(&self, d: &TextureDescriptor) -> Result<Texture, Fail> {
        Ok(self.device.new_texture(d)?)
    }

    /// Writes the format's pattern into level 0 of `t` through a buffer blit.
    fn upload(&self, t: &Texture) -> Result<(), Fail> {
        let l = layout(self.pf).ok_or_else(|| Fail::Skip("no copy layout".into()))?;
        let bytes = pattern(self.pf.as_raw(), l.row * l.rows);
        let src = self.buffer(&bytes)?;
        let cb = self.queue.command_buffer()?;
        let mut blit = cb.blit_command_encoder()?;
        blit.copy_from_buffer_to_texture(
            &src,
            0,
            l.row,
            l.row * l.rows,
            Size::new(SIDE, SIDE, 1),
            t,
            0,
            0,
            Origin::new(0, 0, 0),
            blit_option(self.pf),
        )?;
        blit.end_encoding()?;
        self.commit(&cb)
    }

    /// Level `level` of `t` (`side` square), read back through a blit into a buffer.
    fn read_back(&self, t: &Texture, level: u64, side: u64) -> Result<Vec<u8>, Fail> {
        let l = layout(self.pf).ok_or_else(|| Fail::Skip("no copy layout".into()))?;
        let row = side.div_ceil(SIDE.div_ceil(l.across).max(1)).max(1) * l.unit;
        let rows = side.div_ceil(SIDE.div_ceil(l.down).max(1)).max(1);
        let dst = self.buffer(&vec![0xcd; (row * rows) as usize])?;
        let cb = self.queue.command_buffer()?;
        let mut blit = cb.blit_command_encoder()?;
        blit.copy_from_texture_to_buffer(
            t,
            0,
            level,
            Origin::new(0, 0, 0),
            Size::new(side, side, 1),
            &dst,
            0,
            row,
            row * rows,
            blit_option(self.pf),
        )?;
        blit.end_encoding()?;
        self.commit(&cb)?;
        self.read(&dst, row * rows)
    }

    /// The digest of `t` read back, or of a kernel's reads of it where the blit is refused.
    fn digest_of(&mut self, t: &Texture) -> Done {
        match self.read_back(t, 0, SIDE) {
            Ok(bytes) => Ok(format!("sha={}", digest(&bytes))),
            Err(Fail::Refused(why)) => match self.kernel_read(t, None) {
                Ok(bytes) => Ok(format!("via=kernel sha={}", digest(&bytes))),
                Err(_) => Ok(format!("unread: {}", one_line(&why, 160))),
            },
            Err(other) => Err(other),
        }
    }

    /// The kernel `scout_read_*` (or `scout_sample_*` with `sampler`) of every texel of `t`.
    fn kernel_read(&mut self, t: &Texture, sampler: Option<bool>) -> Result<Vec<u8>, Fail> {
        let class = class(self.pf).ok_or_else(|| Fail::Skip("no read class".into()))?;
        let name = match sampler {
            Some(_) if class == Class::Int || class == Class::Uint => {
                return Err(Fail::Skip("an integer format is not sampled".into()));
            }
            Some(_) => format!("scout_sample_{}", class.suffix()),
            None => format!("scout_read_{}", class.suffix()),
        };
        let state = self.kernel(&name)?;
        let length = SIDE * SIDE * class.out_bytes();
        let out = self.buffer(&vec![0xab; length as usize])?;
        let sampler = sampler.map(|linear| self.sampler(linear)).transpose()?;
        let cb = self.queue.command_buffer()?;
        let mut e = cb.compute_command_encoder()?;
        e.set_compute_pipeline_state(&state)?;
        e.set_texture(Some(t), 0)?;
        e.set_buffer(Some(&out), 0, 0)?;
        if let Some(s) = &sampler {
            e.set_sampler_state(Some(s), 0)?;
        }
        e.dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, SIDE, 1))?;
        e.end_encoding()?;
        self.commit(&cb)?;
        self.read(&out, length)
    }

    /// Dispatches `scout_<what>_<class>` over `t` alone.
    fn kernel_store(&mut self, t: &Texture, what: &str) -> Result<(), Fail> {
        let class = class(self.pf).ok_or_else(|| Fail::Skip("no write class".into()))?;
        if class == Class::Depth || has_stencil(self.pf) {
            return Err(Fail::Skip(
                "depth and stencil formats take no function writes".into(),
            ));
        }
        let state = self.kernel(&format!("scout_{what}_{}", class.suffix()))?;
        let cb = self.queue.command_buffer()?;
        let mut e = cb.compute_command_encoder()?;
        e.set_compute_pipeline_state(&state)?;
        e.set_texture(Some(t), 0)?;
        e.dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(SIDE, SIDE, 1))?;
        e.end_encoding()?;
        self.commit(&cb)
    }

    /// The pipeline that draws a full-target triangle into an attachment of the format.
    fn pipeline(
        &mut self,
        blend: Blend,
        samples: u32,
        depth_or_stencil: bool,
    ) -> Result<RenderPipelineState, Fail> {
        let pf = self.pf;
        let function = |library: &Library, name: &str| -> Result<StageSource, Fail> {
            Ok(StageSource::Function(
                library.new_function_with_name(name)?.function_descriptor(),
            ))
        };
        let mut d = RenderPipelineDescriptor::new(function(&self.library, "scout_vertex")?);
        d.raster_sample_count = samples;
        let as_target = depth_or_stencil || has_depth(pf) || has_stencil(pf);
        if as_target {
            if has_depth(pf) || !has_stencil(pf) {
                d.depth_format = Some(pf);
            }
            if has_stencil(pf) {
                d.stencil_format = Some(pf);
            }
        } else {
            let fragment = match class(pf) {
                Some(Class::Int) => "scout_fragment_i",
                Some(Class::Uint) => "scout_fragment_u",
                _ => "scout_fragment_f",
            };
            d.fragment = Some(function(&self.library, fragment)?);
            let mut color = ColorAttachmentDescriptor::new(pf);
            if let Some([rgb_source, rgb_destination, alpha_source, alpha_destination]) =
                blend.factors()
            {
                color.blending_enabled = true;
                color.source_rgb_blend_factor = rgb_source;
                color.destination_rgb_blend_factor = rgb_destination;
                color.source_alpha_blend_factor = alpha_source;
                color.destination_alpha_blend_factor = alpha_destination;
            }
            d.color_attachments[0] = Some(color);
        }
        Ok(self
            .device
            .new_render_pipeline_state(&d, &PipelineOptions::default())?)
    }

    /// A pass over `target` (resolving into `resolve`), cleared, drawn with `pipeline` if any.
    fn pass(
        &self,
        target: &Texture,
        resolve: Option<&Texture>,
        pipeline: Option<&RenderPipelineState>,
    ) -> Result<(), Fail> {
        self.pass_with(target, resolve, pipeline, false)
    }

    /// A pass over `target` (resolving into `resolve`), loaded where `load` and cleared
    /// elsewhere, drawn with `pipeline` if any.
    fn pass_with(
        &self,
        target: &Texture,
        resolve: Option<&Texture>,
        pipeline: Option<&RenderPipelineState>,
        load: bool,
    ) -> Result<(), Fail> {
        let pf = self.pf;
        let load_action = if load {
            LoadAction::Load
        } else {
            LoadAction::Clear
        };
        let mut pass = RenderPassDescriptor::new();
        let store = if resolve.is_some() {
            StoreAction::MultisampleResolve
        } else {
            StoreAction::Store
        };
        if has_depth(pf) || has_stencil(pf) {
            if has_depth(pf) {
                let a = &mut pass.depth_attachment;
                a.texture = Some(target.clone());
                a.resolve_texture = resolve.cloned();
                a.load_action = LoadAction::Clear;
                a.store_action = store;
                a.clear_depth = 0.375;
            }
            if has_stencil(pf) {
                let a = &mut pass.stencil_attachment;
                a.texture = Some(target.clone());
                a.resolve_texture = resolve.cloned();
                a.load_action = LoadAction::Clear;
                a.store_action = store;
                a.clear_stencil = 0x5a;
            }
        } else {
            let a = &mut pass.color_attachments[0];
            a.texture = Some(target.clone());
            a.resolve_texture = resolve.cloned();
            a.load_action = load_action;
            a.store_action = store;
            a.clear_color = ClearColor::new(0.125, 0.25, 0.5, 0.75);
        }
        let cb = self.queue.command_buffer()?;
        let mut e = cb.render_command_encoder(&pass)?;
        if let Some(p) = pipeline {
            e.set_render_pipeline_state(p)?;
            e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        }
        e.end_encoding()?;
        self.commit(&cb)
    }

    fn target_usage(&self) -> TextureUsage {
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ
    }

    /// A surface's buffer and the plane a texture of the format is made over.
    fn surface(&self, planar: bool) -> Result<(Buffer, IoSurfacePlane), Fail> {
        let buffer = self.buffer(&pattern(self.pf.as_raw() ^ 0x5u32, 1 << 16))?;
        let row = 256;
        let plane = IoSurfacePlane::new(0, row, planar);
        let facts = surface_facts(self.pf);
        Ok((
            buffer,
            match facts {
                Some(f) if planar => plane.with_ycbcr(f),
                _ => plane,
            },
        ))
    }

    fn cell(&mut self, op: Op) -> Done {
        let pf = self.pf;
        match op {
            Op::Own(storage, ty) => {
                let usage = if storage == StorageMode::Memoryless {
                    TextureUsage::RENDER_TARGET
                } else {
                    read_usage()
                };
                let mut d = desc(pf, usage, storage);
                d.texture_type = ty;
                match ty {
                    TextureType::Type1D | TextureType::Type1DArray => d.height = 1,
                    TextureType::Type3D => d.depth = 4,
                    TextureType::Type2DMultisample => {
                        d.sample_count = 4;
                        d.usage = TextureUsage::RENDER_TARGET;
                    }
                    _ => {}
                }
                if matches!(
                    ty,
                    TextureType::Type1DArray | TextureType::Type2DArray | TextureType::CubeArray
                ) {
                    d.array_length = 2;
                }
                self.texture(&d)?;
                Ok("made".into())
            }
            Op::Heap(placement) => {
                let heap = self.device.new_heap(&HeapDescriptor::new(
                    if placement {
                        HeapType::Placement
                    } else {
                        HeapType::Automatic
                    },
                    4 << 20,
                    ResourceOptions::private(),
                ))?;
                let d = desc(pf, read_usage(), StorageMode::Private);
                let t = if placement {
                    heap.new_texture_at_offset(&d, 0)?
                } else {
                    heap.new_texture(&d)?
                };
                self.upload(&t)?;
                self.digest_of(&t)
            }
            Op::OverBuffer => {
                let l = layout(pf);
                let bytes = pattern(pf.as_raw(), 1 << 16);
                let buffer = self.buffer(&bytes)?;
                let row = l.map_or(256, |l| l.row.max(256).next_multiple_of(256));
                let t = buffer.new_texture(&desc(pf, read_usage(), StorageMode::Shared), 0, row)?;
                self.digest_of(&t)
            }
            Op::OverSurface { planar, render } => {
                let (buffer, plane) = self.surface(planar)?;
                let usage = if render {
                    self.target_usage()
                } else {
                    read_usage()
                };
                let t = buffer.new_texture_with_iosurface_plane(
                    &desc(pf, usage, StorageMode::Shared),
                    plane,
                )?;
                if render {
                    let p = self.pipeline(Blend::Off, 1, false)?;
                    self.pass(&t, None, Some(&p))?;
                    let bytes = self.read(&buffer, 1 << 16)?;
                    Ok(format!("drawn surface sha={}", digest(&bytes)))
                } else {
                    match self.kernel_read(&t, Some(false)) {
                        Ok(bytes) => Ok(format!("sampled sha={}", digest(&bytes))),
                        Err(Fail::Skip(_)) => {
                            let bytes = self.kernel_read(&t, None)?;
                            Ok(format!("read sha={}", digest(&bytes)))
                        }
                        Err(other) => Err(other),
                    }
                }
            }
            Op::ViewSame | Op::ViewSwizzle => {
                let t = self.texture(&desc(
                    pf,
                    read_usage() | TextureUsage::PIXEL_FORMAT_VIEW,
                    StorageMode::Private,
                ))?;
                self.upload(&t)?;
                let view = if matches!(op, Op::ViewSame) {
                    t.new_texture_view(pf)?
                } else {
                    t.new_texture_view_with_swizzle(
                        pf,
                        TextureType::Type2D,
                        0..1,
                        0..1,
                        TextureSwizzleChannels {
                            red: TextureSwizzle::Blue,
                            green: TextureSwizzle::Green,
                            blue: TextureSwizzle::Red,
                            alpha: TextureSwizzle::One,
                        },
                    )?
                };
                let bytes = self.kernel_read(&view, None)?;
                Ok(format!("read sha={}", digest(&bytes)))
            }
            Op::HostCopy(storage) => {
                let t = self.texture(&desc(pf, read_usage(), storage))?;
                let l = layout(pf).ok_or_else(|| Fail::Skip("no copy layout".into()))?;
                let bytes = pattern(pf.as_raw(), l.row * l.rows);
                let region = Region::new_2d(0, 0, SIDE, SIDE);
                t.replace_region(region, 0, 0, &bytes, l.row, 0)?;
                let mut back = vec![0xcd; bytes.len()];
                t.get_bytes(&mut back, l.row, 0, region, 0, 0)?;
                let same = if back == bytes { "same" } else { "differs" };
                Ok(format!("roundtrip={same} sha={}", digest(&back)))
            }
            Op::BlitBuffer => {
                let t = self.texture(&desc(pf, read_usage(), StorageMode::Private))?;
                self.upload(&t)?;
                let l = layout(pf).ok_or_else(|| Fail::Skip("no copy layout".into()))?;
                let back = self.read_back(&t, 0, SIDE)?;
                let same = if back == pattern(pf.as_raw(), l.row * l.rows) {
                    "same"
                } else {
                    "differs"
                };
                Ok(format!("roundtrip={same} sha={}", digest(&back)))
            }
            Op::BlitCopy => {
                let d = desc(pf, read_usage(), StorageMode::Private);
                let (src, dst) = (self.texture(&d)?, self.texture(&d)?);
                self.upload(&src)?;
                let cb = self.queue.command_buffer()?;
                let mut blit = cb.blit_command_encoder()?;
                blit.copy_from_texture_to_texture(
                    &src,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                    Size::new(SIDE, SIDE, 1),
                    &dst,
                    0,
                    0,
                    Origin::new(0, 0, 0),
                )?;
                blit.end_encoding()?;
                self.commit(&cb)?;
                self.digest_of(&dst)
            }
            Op::Mipmaps => {
                let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, true);
                d.usage = read_usage() | TextureUsage::RENDER_TARGET;
                d.resource_options = ResourceOptions::private();
                let t = self.texture(&d)?;
                self.upload(&t)?;
                let cb = self.queue.command_buffer()?;
                let mut blit = cb.blit_command_encoder()?;
                blit.generate_mipmaps(&t)?;
                blit.end_encoding()?;
                self.commit(&cb)?;
                let back = self.read_back(&t, 1, SIDE / 2)?;
                Ok(format!("level1 sha={}", digest(&back)))
            }
            Op::FillBytes | Op::FillColor => {
                let l = layout(pf);
                let t = self.texture(&desc(pf, read_usage(), StorageMode::Private))?;
                let (across, down) = l.map_or((SIDE, SIDE), |l| (l.across, l.down));
                let region = if compressed(pf) {
                    Region::new_2d(0, 0, across, down)
                } else {
                    Region::new_2d(0, 0, SIDE, SIDE)
                };
                let cb = self.queue.command_buffer()?;
                let mut blit = cb.blit_command_encoder()?;
                if matches!(op, Op::FillBytes) {
                    blit.fill_texture(&t, 0, 0, region, &pattern(pf.as_raw() ^ 0xf1, 16))?;
                } else {
                    blit.fill_texture_color(
                        &t,
                        0,
                        0,
                        region,
                        ClearColor::new(0.25, 0.5, 0.75, 1.0),
                        None,
                    )?;
                }
                blit.end_encoding()?;
                self.commit(&cb)?;
                self.digest_of(&t)
            }
            Op::Read => {
                let t = self.texture(&desc(pf, read_usage(), StorageMode::Private))?;
                self.upload(&t)?;
                let bytes = self.kernel_read(&t, None)?;
                Ok(format!("read sha={}", digest(&bytes)))
            }
            Op::Write => {
                let t = self.texture(&desc(
                    pf,
                    read_usage() | TextureUsage::SHADER_WRITE,
                    StorageMode::Private,
                ))?;
                self.kernel_store(&t, "write")?;
                self.digest_of(&t)
            }
            Op::ReadWrite => {
                let t = self.texture(&desc(
                    pf,
                    read_usage() | TextureUsage::SHADER_WRITE,
                    StorageMode::Private,
                ))?;
                self.upload(&t)?;
                self.kernel_store(&t, "read_write")?;
                self.digest_of(&t)
            }
            Op::Sample(linear) => {
                let t = self.texture(&desc(pf, read_usage(), StorageMode::Private))?;
                self.upload(&t)?;
                let bytes = self.kernel_read(&t, Some(linear))?;
                Ok(format!("sampled sha={}", digest(&bytes)))
            }
            Op::Render { blend } => {
                let p = self.pipeline(Blend::source_over(blend), 1, false)?;
                let t = self.texture(&desc(pf, self.target_usage(), StorageMode::Private))?;
                self.pass(&t, None, Some(&p))?;
                self.digest_of(&t)
            }
            Op::Clear => {
                let t = self.texture(&desc(pf, self.target_usage(), StorageMode::Private))?;
                self.pass(&t, None, None)?;
                self.digest_of(&t)
            }
            Op::Msaa { resolve } => {
                let p = self.pipeline(Blend::Off, 4, false)?;
                let mut d = desc(pf, TextureUsage::RENDER_TARGET, StorageMode::Private);
                d.texture_type = TextureType::Type2DMultisample;
                d.sample_count = 4;
                let t = self.texture(&d)?;
                if resolve {
                    let r = self.texture(&desc(pf, self.target_usage(), StorageMode::Private))?;
                    self.pass(&t, Some(&r), Some(&p))?;
                    self.digest_of(&r)
                } else {
                    self.pass(&t, None, Some(&p))?;
                    Ok("drawn".into())
                }
            }
            Op::Attachment {
                blend,
                depth_or_stencil,
            } => {
                if depth_or_stencil {
                    self.pipeline(Blend::Off, 1, true)?;
                } else if has_depth(pf) || has_stencil(pf) {
                    return Err(Fail::Skip("a depth or stencil format's color cell".into()));
                } else {
                    self.pipeline(Blend::source_over(blend), 1, false)?;
                }
                Ok("made".into())
            }
            Op::Alias { texture } => {
                let heap = self.device.new_heap(&HeapDescriptor::new(
                    HeapType::Placement,
                    4 << 20,
                    ResourceOptions::private(),
                ))?;
                let live =
                    heap.new_texture_at_offset(&desc(pf, read_usage(), StorageMode::Private), 0)?;
                let alias = if texture {
                    let d = desc(PixelFormat::R8Unorm, read_usage(), StorageMode::Private);
                    let a = heap.new_texture_at_offset(&d, 0)?;
                    format!("alias texture {}x{}", a.width(), a.height())
                } else {
                    let b = heap.new_buffer_at_offset(4096, ResourceOptions::private(), 0)?;
                    format!("alias buffer {}", b.length())
                };
                self.upload(&live)?;
                let read = self.digest_of(&live)?;
                Ok(format!("{alias}; {read}"))
            }
            Op::OverX420 { plane } => {
                let buffer = self.buffer(&pattern(pf.as_raw() ^ 0x5u32, 1 << 16))?;
                let (offset, side) = match plane {
                    0 => (0, SIDE),
                    _ => (CHROMA_OFFSET, SIDE / 2),
                };
                let on = IoSurfacePlane::new(offset, SURFACE_ROW, true).with_ycbcr(x420_facts());
                let mut d = desc(pf, read_usage(), StorageMode::Shared);
                d.width = side;
                d.height = side;
                let t = buffer.new_texture_with_iosurface_plane(&d, on)?;
                let bytes = self.kernel_read(&t, Some(false))?;
                Ok(format!("sampled sha={}", digest(&bytes)))
            }
            Op::BlendLoaded { form, blend } => {
                let p = self.pipeline(blend, 1, false)?;
                let usage = self.target_usage();
                match form {
                    Form::Surface => {
                        let (buffer, plane) = self.surface(true)?;
                        let t = buffer.new_texture_with_iosurface_plane(
                            &desc(pf, usage, StorageMode::Shared),
                            plane,
                        )?;
                        self.pass_with(&t, None, Some(&p), true)?;
                        let bytes = self.read(&buffer, 1 << 16)?;
                        Ok(format!("drawn surface sha={}", digest(&bytes)))
                    }
                    Form::Heap | Form::Own => {
                        let d = desc(pf, usage, StorageMode::Private);
                        let t = if form == Form::Heap {
                            let heap = self.device.new_heap(&HeapDescriptor::new(
                                HeapType::Placement,
                                4 << 20,
                                ResourceOptions::private(),
                            ))?;
                            heap.new_texture_at_offset(&d, 0)?
                        } else {
                            self.texture(&d)?
                        };
                        self.upload(&t)?;
                        self.pass_with(&t, None, Some(&p), true)?;
                        self.digest_of(&t)
                    }
                }
            }
        }
    }
}

/// The surface facts a texture of a two-plane format reads (none for other formats): a
/// CoreVideo OSType whose codes match the format's, 16 x 16, the chroma (or alpha) plane at
/// byte 4096 with rows 256 bytes apart.
fn surface_facts(pf: PixelFormat) -> Option<IoSurfaceYCbCr> {
    let ostype = |code: &[u8; 4]| u32::from_be_bytes(*code);
    let raw = pf.as_raw();
    if matches!(raw, 550 | 551 | 624..=627) {
        return Some(IoSurfaceYCbCr::new(
            ostype(b"b3a8"),
            0,
            [SIDE, SIDE],
            4096,
            256,
        ));
    }
    let YCbCrFormat {
        bits,
        subsampling,
        packed,
        ..
    } = ycbcr_format(pf)?;
    let sub = match subsampling {
        ChromaSubsampling::Chroma420 => 0,
        ChromaSubsampling::Chroma422 => 1,
        _ => 2,
    };
    let code: &[u8; 4] = match (bits, packed, sub) {
        (8, _, 0) => b"420v",
        (8, _, 1) => b"422v",
        (8, _, _) => b"444v",
        (_, false, 0) => b"x420",
        (_, false, 1) => b"x422",
        (_, false, _) => b"x444",
        (_, true, 0) => b"p420",
        (_, true, 1) => b"p422",
        (_, true, _) => b"p444",
    };
    Some(IoSurfaceYCbCr::new(
        ostype(code),
        0,
        [SIDE, SIDE],
        4096,
        256,
    ))
}

/// The facts of the scout's two-plane 10-bit 4:2:0 surface (`x420`), laid out as
/// [`surface_facts`] lays out `YCBCR10_420_2P`'s.
fn x420_facts() -> IoSurfaceYCbCr {
    IoSurfaceYCbCr::new(
        u32::from_be_bytes(X420),
        0,
        [SIDE, SIDE],
        CHROMA_OFFSET,
        SURFACE_ROW,
    )
}

thread_local! {
    /// The index of the cell the child runs.
    static CURRENT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
    /// The digests the cell recorded so far (`--dump`).
    static DUMPED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Where `--dump` writes the bytes each digest is of.
static DUMP: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Starts cell `i`: [`CURRENT`], and its digests counted from 0 ([`DUMPED`]).
fn begin_cell(i: usize) {
    CURRENT.with(|c| c.set(i));
    DUMPED.with(|d| d.set(0));
}

fn child(options: &BTreeMap<String, String>) -> Result<(), String> {
    let pf = format_named(required(options, "format")?)?;
    let device_type = DeviceType::parse(required(options, "device-type")?)?;
    let cache = PathBuf::from(required(options, "cache")?);
    let all = cells();
    let indices: Vec<usize> = required(options, "cells")?
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("--cells {s}: {e}"))
        })
        .collect::<Result<_, _>>()?;
    if let Some(bad) = indices.iter().find(|&&i| i >= all.len()) {
        return Err(format!("no cell {bad}"));
    }
    if let Some(dir) = options.get("dump") {
        std::fs::create_dir_all(dir).map_err(|e| format!("--dump {dir}: {e}"))?;
        let _ = DUMP.set(PathBuf::from(dir));
    }
    match options.get("side").map(String::as_str) {
        None | Some("runtime") => {}
        #[cfg(target_os = "macos")]
        Some("metal") => return metal::child(pf, &indices),
        Some(other) => return Err(format!("--side {other}: no such side on this host")),
    }
    let messages = Messages::default();
    let opened = (|| -> Result<(Device, Library), String> {
        let device = open_device(device_type, validated_options(&messages, &cache))
            .map_err(|e| format!("{e}"))?;
        let bitcode = llvm_assemble(&engine_dir().join("format_scout.ll"))
            .map_err(|e| format!("llvm-as of format_scout.ll: {e:?}"))?;
        let library = device
            .new_library_with_air(&bitcode)
            .map_err(|e| format!("the scout's library: {e:?}"))?;
        Ok((device, library))
    })();
    let (device, library) = match opened {
        Ok(opened) => opened,
        Err(e) => {
            println!("DEVICE\tfail\t{}", one_line(&e, 300));
            return Ok(());
        }
    };
    let queue = device
        .new_command_queue()
        .map_err(|e| format!("the queue: {e:?}"))?;
    let mut child = Child {
        pf,
        device,
        queue,
        messages,
        library,
        kernels: HashMap::new(),
        samplers: HashMap::new(),
    };
    println!("DEVICE\tok");
    let mut out = std::io::stdout();
    for i in indices {
        begin_cell(i);
        let _ = child.messages.take();
        let _ = child.messages.take_notices();
        let _ = writeln!(out, "BEGIN\t{i}");
        let _ = out.flush();
        let op = all[i].op;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| child.cell(op)));
        let failing = child.messages.take();
        let notices = child.messages.take_notices();
        let (mut verdict, mut detail) = match result {
            Ok(Ok(detail)) => ("made", detail),
            Ok(Err(Fail::Refused(why))) => ("refused", why),
            Ok(Err(Fail::Died(why))) => ("died", why),
            Ok(Err(Fail::Skip(why))) => ("skip", why),
            Err(panic) => {
                let text = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default();
                ("died", format!("panic: {text}"))
            }
        };
        if !failing.is_empty() {
            let mut ids: Vec<&str> = failing.iter().map(|m| m.id_name.as_str()).collect();
            ids.dedup();
            let first = failing
                .first()
                .map(|m| one_line(&m.text, 200))
                .unwrap_or_default();
            detail = format!(
                "{} message(s) {}; first: {first}; {verdict}: {detail}",
                failing.len(),
                ids.join(",")
            );
            verdict = "vvl";
        }
        if !notices.is_empty() {
            let kinds: BTreeSet<&'static str> = notices
                .iter()
                .filter_map(|m| m.degradation.as_ref().map(|n| n.degradation.code()))
                .collect();
            detail = format!(
                "{detail}; notices {}",
                kinds.into_iter().collect::<Vec<_>>().join(",")
            );
        }
        let _ = writeln!(out, "CELL\t{i}\t{verdict}\t{}", one_line(&detail, 600));
        let _ = out.flush();
    }
    Ok(())
}
