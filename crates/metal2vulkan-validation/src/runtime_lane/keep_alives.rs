//! Device tests of no-copy buffers' keep-alives (`KeepAlive`, a runtime extension) over one
//! import of host pages, on the device the cfg selects, under the validation layer: the
//! consumer's presents, each a new no-copy buffer over the presented surface's whole memory made
//! before the last one's release, so that some buffer over the surface always lives.
//!
//! Every no-copy buffer made over the same bytes as a live one lies in its import
//! (`host_aliases.rs`), and the import keeps the pages imported while anything holds it. A
//! keep-alive goes at its buffer's release, after the deallocator, where the import's pages stay
//! mapped without it: another buffer lies in the import, or the import keeps another's keep-alive
//! already; else the import keeps it until it is freed. An import freed with only a keep-alive
//! waiting for the free (the deallocator ran at the release) is freed on one of the device's
//! handler threads, and the keep-alive goes there after the free, a moment after the release
//! returned (`HostImport`'s drop): the tests wait for it there. So an import keeps one keep-alive
//! at most, however many buffers come and go over its bytes (`WorkCounts::host_keep_alives_kept`
//! held now, `WorkCounts::host_keep_alives_covered` dropped at the release). A buffer over part of
//! a released buffer's bytes is another import, and covers nothing of it.
//!
//! What holds an import past its buffers' release is a texture's image bound to its pages, while
//! a running command buffer reaches the image through the texture's ID
//! (`resource_ids::Reaching`): a texture over a buffer keeps the buffer, as Metal's retains it, so
//! the buffer goes at the texture's release, its deallocator run then, as Metal's
//! (`resource_ids.rs`), while the image stays until that command buffer completed.
//!
//! The memory is private anonymous pages, which every measured driver imports by their pointer;
//! a driver that refuses them has each test say so and pass without running.

use super::host_memory::{not_importable, on_device, Mapping, GUEST_PAGE};
use super::resource_ids::Sampling;
use super::WAIT;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, BufferContents, BufferDescriptor, CpuCacheMode, Deallocator, Device, Error,
    HazardTrackingMode, HostMemory, KeepAlive, ResourceOptions, StorageMode, Texture,
    TextureDescriptor, WorkCounts,
};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Buffers made and released over one range while another over it lives.
const ROUNDS: usize = 64;

/// The bytes between rows of the textures over the buffers: Metal's linear alignment, and an
/// IOSurface's.
const ROW_BYTES: u64 = 256;

/// The deallocators run and the keep-alives dropped, of the buffers made with one tally.
#[derive(Default)]
struct Tally {
    deallocated: AtomicUsize,
    dropped: AtomicUsize,
}

impl Tally {
    fn deallocated(&self) -> usize {
        self.deallocated.load(Ordering::SeqCst)
    }

    fn dropped(&self) -> usize {
        self.dropped.load(Ordering::SeqCst)
    }
}

/// A keep-alive that counts its drop.
struct Counted(Arc<Tally>);

impl Drop for Counted {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

/// A no-copy buffer over `len` bytes at `at`, of `storage`, with a deallocator and a keep-alive
/// that count into `tally`.
///
/// # Safety
///
/// [`HostMemory::new`]'s contract for the bytes.
unsafe fn counted(
    device: &Device,
    at: NonNull<u8>,
    len: usize,
    storage: StorageMode,
    tally: &Arc<Tally>,
) -> Result<Buffer, Error> {
    let seen = Arc::clone(tally);
    let deallocator: Deallocator = Box::new(move |_, _| {
        seen.deallocated.fetch_add(1, Ordering::SeqCst);
    });
    let keep_alive: KeepAlive = Box::new(Counted(Arc::clone(tally)));
    // SAFETY: the caller's contract.
    let memory = unsafe { HostMemory::new(at) };
    let contents = BufferContents::NoCopy {
        memory,
        deallocator: Some(deallocator),
        keep_alive: Some(keep_alive),
    };
    let options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    device.new_buffer_with_descriptor(BufferDescriptor::new(len as u64, options, contents))
}

/// A texture of two 256-byte rows over `buffer`'s first bytes, and whether its image lies in the
/// buffer's pages (it holds their import then), not a shadow of them.
fn texture_over(buffer: &Buffer) -> (Texture, bool) {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Uint, ROW_BYTES / 4, 2, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    let texture = buffer.new_texture(&descriptor, 0, ROW_BYTES).unwrap();
    // SAFETY: the handle is only compared, while the texture lives.
    let bound = unsafe { texture.vk_image() }.is_some();
    (texture, bound)
}

/// Wait until `done`, at most [`WAIT`]: an image and its import may go on a runtime thread.
fn eventually(done: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + WAIT;
    while !done() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    done()
}

/// The keep-alives the device holds now, and those it dropped at a release, since `base`.
fn since(device: &Device, base: &WorkCounts) -> (u64, u64) {
    let counts = device.work_counts();
    (
        counts.host_keep_alives_kept - base.host_keep_alives_kept,
        counts.host_keep_alives_covered - base.host_keep_alives_covered,
    )
}

/// [`ROUNDS`] buffers made over one range, each before the last one's release, as a consumer's
/// presents make them over a surface: each released one's deallocator runs and its keep-alive is
/// dropped at its release, covered by the next, and the device keeps none. The last goes at the
/// release of a texture over it, which keeps it till then; where the texture's image lies in its
/// pages and a running command buffer reaches the image, the import keeps its keep-alive until
/// that command buffer completed; a texture whose image is a shadow holds no import, and the
/// keep-alive goes at the release.
#[test]
fn keep_alives_over_one_import_stay_at_most_one_as_buffers_come_and_go() {
    let len = 2 * GUEST_PAGE;
    let mapping = Mapping::anonymous(len);
    on_device(|device| {
        let sampling = Sampling::new(device);
        let tally = Arc::new(Tally::default());
        let base = device.work_counts();
        let make = || {
            // SAFETY: the mapping outlives the device, and so every buffer over it.
            unsafe { counted(device, mapping.pointer(), len, StorageMode::Shared, &tally) }
        };
        let mut live = match make() {
            Ok(buffer) => buffer,
            Err(error) if not_importable(&error) => {
                println!("KEEP-ALIVES {}: refused {error:?}", device.name());
                return;
            }
            Err(error) => panic!("a no-copy buffer on {}: {error:?}", device.name()),
        };
        for round in 1..=ROUNDS {
            let next = make().unwrap();
            drop(live);
            live = next;
            assert_eq!(
                tally.deallocated(),
                round,
                "round {round}: deallocators run"
            );
            assert_eq!(tally.dropped(), round, "round {round}: keep-alives dropped");
            assert_eq!(
                since(device, &base),
                (0, round as u64),
                "round {round}: (held, covered)"
            );
        }
        let (texture, bound) = texture_over(&live);
        drop(live);
        println!(
            "KEEP-ALIVES {}: {ROUNDS} covered, an image bound to the bytes {bound}",
            device.name()
        );
        assert_eq!(tally.deallocated(), ROUNDS, "the texture keeps the last");
        assert_eq!(tally.dropped(), ROUNDS, "the texture keeps the last");
        let reaching = sampling.reaching(device, &texture);
        drop(texture);
        assert_eq!(tally.deallocated(), ROUNDS + 1, "the last at the release");
        let held = u64::from(bound);
        // Where nothing holds the import, its free and the keep-alive go on a handler thread.
        assert!(
            eventually(|| tally.dropped() == ROUNDS + 1 - held as usize),
            "the last keep-alive goes with the import unless the image holds it"
        );
        assert_eq!(
            since(device, &base),
            (held, ROUNDS as u64),
            "(held, covered)"
        );
        reaching.complete(&sampling);
        assert!(
            eventually(|| tally.dropped() == ROUNDS + 1),
            "the kept keep-alive goes as the import is freed"
        );
        assert_eq!(since(device, &base), (0, ROUNDS as u64), "(held, covered)");
    });
}

/// The keep-alive an import keeps covers a later buffer's over the same bytes. A buffer of
/// another memory class over them keeps the bytes noted, so a buffer made over them while an
/// image holds the import of a released one lies in that import (`host_aliases.rs`). Released one
/// after another, [`ROUNDS`] such buffers leave the device one keep-alive, the first's, until the
/// command buffer reaching the image completed; each later one's is dropped at its release. Where
/// the texture's image is a shadow (no import held) or the device refuses the memory or the class,
/// nothing is kept and the test says so.
#[test]
fn a_kept_keep_alive_covers_later_buffers_over_its_bytes() {
    let len = 2 * GUEST_PAGE;
    let mapping = Mapping::anonymous(len);
    on_device(|device| {
        let sampling = Sampling::new(device);
        let tally = Arc::new(Tally::default());
        let other = Arc::new(Tally::default());
        let base = device.work_counts();
        let make = |storage, tally: &Arc<Tally>| {
            // SAFETY: the mapping outlives the device, and so every buffer over it.
            unsafe { counted(device, mapping.pointer(), len, storage, tally) }
        };
        let (first, noting) = match (
            make(StorageMode::Shared, &tally),
            make(StorageMode::Managed, &other),
        ) {
            (Ok(first), Ok(noting)) => (first, noting),
            (Err(error), _) | (_, Err(error)) if not_importable(&error) => {
                println!("KEEP-ALIVES-KEPT {}: refused {error:?}", device.name());
                return;
            }
            (Err(error), _) | (_, Err(error)) => {
                panic!("a no-copy buffer on {}: {error:?}", device.name())
            }
        };
        let (texture, bound) = texture_over(&first);
        let reaching = sampling.reaching(device, &texture);
        drop(first);
        drop(texture);
        println!(
            "KEEP-ALIVES-KEPT {}: an image bound to the bytes {bound}",
            device.name()
        );
        assert_eq!(tally.deallocated(), 1, "the first's at the release");
        if !bound {
            assert!(
                eventually(|| tally.dropped() == 1),
                "no import held: the keep-alive goes with it"
            );
            assert_eq!(since(device, &base), (0, 0), "(held, covered)");
            reaching.complete(&sampling);
            return;
        }
        assert_eq!(since(device, &base), (1, 0), "the import keeps the first's");
        for round in 1..=ROUNDS {
            let later = make(StorageMode::Shared, &tally).unwrap();
            drop(later);
            assert_eq!(
                tally.deallocated(),
                1 + round,
                "round {round}: deallocators run"
            );
            assert_eq!(tally.dropped(), round, "round {round}: keep-alives dropped");
            assert_eq!(
                since(device, &base),
                (1, round as u64),
                "round {round}: (held, covered)"
            );
        }
        reaching.complete(&sampling);
        assert!(
            eventually(|| tally.dropped() == ROUNDS + 1),
            "the first's keep-alive goes as its import is freed"
        );
        assert_eq!(since(device, &base), (0, ROUNDS as u64), "(held, covered)");
        drop(noting);
        assert!(
            eventually(|| other.dropped() == 1),
            "the other class's import went with its buffer"
        );
    });
}

/// A narrower live buffer over the first half of a wider released one's bytes is another import
/// (`host_aliases.rs` joins only the same bytes), and covers nothing of the wider one's: while an
/// image over the wider one's pages is reached by a running command buffer, the import keeps its
/// keep-alive though the narrower buffer lives, until that command buffer completed. The narrower
/// one's own keep-alive goes as its own import is freed, on a handler thread. No keep-alive is
/// counted covered.
#[test]
fn a_narrower_live_buffer_covers_no_wider_released_one() {
    let len = 2 * GUEST_PAGE;
    let mapping = Mapping::anonymous(len);
    on_device(|device| {
        let sampling = Sampling::new(device);
        let (wide_tally, narrow_tally) = (Arc::new(Tally::default()), Arc::new(Tally::default()));
        let base = device.work_counts();
        let make = |len, tally: &Arc<Tally>| {
            // SAFETY: the mapping outlives the device, and so every buffer over it.
            unsafe { counted(device, mapping.pointer(), len, StorageMode::Shared, tally) }
        };
        let (wide, narrow) = match (make(len, &wide_tally), make(GUEST_PAGE, &narrow_tally)) {
            (Ok(wide), Ok(narrow)) => (wide, narrow),
            (Err(error), _) | (_, Err(error)) if not_importable(&error) => {
                println!("KEEP-ALIVES-NARROW {}: refused {error:?}", device.name());
                return;
            }
            (Err(error), _) | (_, Err(error)) => {
                panic!("a no-copy buffer on {}: {error:?}", device.name())
            }
        };
        let (texture, bound) = texture_over(&wide);
        let reaching = sampling.reaching(device, &texture);
        drop(wide);
        drop(texture);
        println!(
            "KEEP-ALIVES-NARROW {}: an image bound to the bytes {bound}",
            device.name()
        );
        assert_eq!(
            wide_tally.deallocated(),
            1,
            "the wider one's deallocator at the release"
        );
        let held = u64::from(bound);
        // Where nothing holds the wider one's import, its free and the keep-alive go on a
        // handler thread.
        assert!(
            eventually(|| wide_tally.dropped() == 1 - held as usize),
            "the wider one's keep-alive stays while the image holds its import"
        );
        assert_eq!(since(device, &base), (held, 0), "(held, covered)");
        drop(narrow);
        assert_eq!(
            narrow_tally.deallocated(),
            1,
            "the narrower one's deallocator"
        );
        assert!(
            eventually(|| narrow_tally.dropped() == 1),
            "the narrower one's keep-alive with its import"
        );
        assert_eq!(wide_tally.dropped(), 1 - held as usize, "still the image's");
        reaching.complete(&sampling);
        assert!(
            eventually(|| wide_tally.dropped() == 1),
            "the wider one's keep-alive goes as its import is freed"
        );
        assert_eq!(since(device, &base), (0, 0), "(held, covered)");
    });
}
