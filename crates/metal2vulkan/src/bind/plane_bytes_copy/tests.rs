//! The plane bytes copy kernel on the CPU interpreter, against a byte-by-byte model of Metal's
//! blit into a heap's two-plane texture.

use super::{
    typed_module, PlaneBytes, PlaneBytesCopyModule, PlaneSource, DESTINATION, SHAPE, SOURCE,
};
use crate::bind::PlaneRows;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

#[test]
fn the_plane_bytes_copy_kernel_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(typed_module().expect("module"), &p);
    }
}

/// Bytes of a seeded generator, as the probe made a heap's.
fn random_bytes(count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u8
        })
        .collect()
}

/// A texture's level (width, height) at byte `at` of the heap: its plane 0's first byte, its
/// alpha plane's (from `align(h · align(4 w, 16), 16384)` past it), and its blit rows.
fn planes(level: [u32; 2], at: u32) -> (u32, u32, u32) {
    let rows = PlaneRows::of(level[0]);
    let alpha = (level[1] * rows.color).next_multiple_of(16384);
    (at, at + alpha, rows.blit)
}

/// One copy between two textures whose planes lie in one heap's bytes.
#[derive(Clone, Copy, Debug)]
struct Case {
    source: [u32; 2],
    source_at: u32,
    destination: [u32; 2],
    destination_at: u32,
    from: [u32; 2],
    to: [u32; 2],
    size: [u32; 2],
    /// The heap's bytes.
    heap: u32,
}

impl Case {
    /// The copy's sides: the source's and the destination's planes in the heap's bytes.
    fn sides(&self) -> (PlaneBytes, PlaneBytes) {
        let (sc, sa, sb) = planes(self.source, self.source_at);
        let (dc, da, db) = planes(self.destination, self.destination_at);
        let side = |(c, a, b): (u32, u32, u32), at: [u32; 2]| {
            PlaneBytes::new(
                c + at[1] * b + 4 * at[0],
                a + at[1] * b + at[0],
                b,
                self.heap,
            )
        };
        (side((sc, sa, sb), self.from), side((dc, da, db), self.to))
    }
}

/// The destination's bytes after the kernel copies `size` texels from `source` (its binding's
/// `words`) into `destination` (over `bytes`), in both invocation orders alike and in
/// dispatches of at most `batch` workgroups, each with the shape of its first invocation.
fn copied(
    source: &PlaneSource,
    words: &[u8],
    destination: &PlaneBytes,
    bytes: &[u8],
    size: [u32; 2],
    batch: u32,
) -> Vec<u8> {
    let m = typed_module().expect("module");
    let invocations = PlaneBytesCopyModule::invocations(destination, size);
    let groups = u32::try_from(invocations.div_ceil(u64::from(PlaneBytesCopyModule::LOCAL_SIZE)))
        .expect("groups");
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let mut now = bytes.to_vec();
        let mut done = 0;
        while done < groups {
            let n = batch.min(groups - done);
            let first = done * PlaneBytesCopyModule::LOCAL_SIZE;
            let shape = PlaneBytesCopyModule::shape(source, destination, size, first);
            let part = |slot, bytes: Vec<u8>| Buffer {
                slot: Some(slot),
                address: None,
                bytes,
            };
            let res = Resources {
                buffers: vec![
                    part(SOURCE, words.to_vec()),
                    part(DESTINATION, now.clone()),
                    part(SHAPE, shape.to_vec()),
                ],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [n, 1, 1],
                ..Config::default()
            };
            let buffers = run(&m, &cfg, res).expect("run").buffers;
            assert_eq!(buffers[0], words, "the source is only read");
            assert_eq!(buffers[2], shape.to_vec(), "the shape is only read");
            now = buffers[1].clone();
            done += n;
        }
        out.push(now);
    }
    assert_eq!(out[0], out[1], "order independent");
    out.swap_remove(0)
}

/// The copy by brute force: row after row, each byte from the source's `before` at Metal's blit
/// rows (0 at or past the source's end) onto the destination's, later rows over earlier ones;
/// none at or past the destination's end.
fn modelled(
    read: impl Fn(u32) -> u8,
    source: &PlaneBytes,
    destination: &PlaneBytes,
    bytes: &[u8],
    size: [u32; 2],
) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for r in 0..size[1] {
        for c in 0..size[0] {
            for (unit, from, to) in [
                (4, source.color, destination.color),
                (1, source.alpha, destination.alpha),
            ] {
                for k in 0..unit {
                    let s = from + r * source.blit + unit * c + k;
                    let d = to + r * destination.blit + unit * c + k;
                    if d < destination.end {
                        out[d as usize] = read(s);
                    }
                }
            }
        }
    }
    out
}

/// Copies between two textures of one heap: whole and regions, at widths on 16, on 4 and off 4,
/// from wider and narrower textures, below a width of 4, a source past the destination (its
/// bytes read as they were before the copy), and a destination whose alpha bytes pass the heap's
/// end.
#[test]
fn copies_write_the_heap_s_bytes_at_metal_s_rows() {
    let case = |source: [u32; 2], destination: [u32; 2], from, to, size| Case {
        source,
        source_at: 0,
        destination,
        destination_at: 114_688,
        from,
        to,
        size,
        heap: 229_376,
    };
    let whole = |level: [u32; 2]| case(level, level, [0, 0], [0, 0], level);
    let mut cases = vec![
        whole([48, 20]),
        whole([64, 9]),
        whole([5, 5]),
        whole([9, 7]),
        whole([63, 11]),
        whole([4, 6]),
        whole([3, 4]),
        case([48, 20], [48, 20], [3, 5], [11, 2], [17, 9]),
        case([20, 8], [33, 9], [0, 0], [2, 1], [20, 8]),
        case([48, 20], [9, 7], [3, 2], [0, 0], [9, 7]),
        case([64, 13], [63, 13], [0, 0], [0, 0], [63, 13]),
        case([48, 20], [48, 20], [0, 3], [0, 7], [48, 1]),
        case([2, 9], [3, 5], [0, 1], [1, 0], [2, 3]),
    ];
    // The source past the destination, in the bytes its alpha bytes pass over.
    cases.push(Case {
        source_at: 17_408,
        destination_at: 0,
        ..whole([48, 20])
    });
    // The destination at the heap's end: its alpha bytes pass it.
    cases.push(Case {
        destination_at: 114_688,
        heap: 114_688 + 17_344,
        ..whole([48, 20])
    });
    for (n, copy) in cases.into_iter().enumerate() {
        let heap = random_bytes(copy.heap as usize, 11 + n as u64);
        let (source, destination) = copy.sides();
        let read = |s: u32| heap.get(s as usize).copied().unwrap_or(0);
        let want = modelled(read, &source, &destination, &heap, copy.size);
        for batch in [u32::MAX, 3] {
            let got = copied(
                &PlaneSource::Bytes(source),
                &heap,
                &destination,
                &heap,
                copy.size,
                batch,
            );
            let wrong = (0..heap.len()).filter(|&i| got[i] != want[i]).count();
            assert_eq!(wrong, 0, "{copy:?} in batches of {batch}");
        }
        assert_ne!(want, heap, "{copy:?} writes");
    }
}

/// A source the binding holds fewer bytes of reads 0 past them.
#[test]
fn a_source_past_its_end_reads_zero() {
    let copy = Case {
        source: [48, 20],
        source_at: 0,
        destination: [48, 20],
        destination_at: 114_688,
        from: [0, 0],
        to: [0, 0],
        size: [48, 20],
        heap: 229_376,
    };
    let heap = random_bytes(copy.heap as usize, 5);
    let (source, destination) = copy.sides();
    let short = PlaneBytes::new(source.color, source.alpha, source.blit, 17_000);
    let read = |s: u32| if s < 17_000 { heap[s as usize] } else { 0 };
    let want = modelled(read, &short, &destination, &heap, copy.size);
    let got = copied(
        &PlaneSource::Bytes(short),
        &heap,
        &destination,
        &heap,
        copy.size,
        u32::MAX,
    );
    assert!(got == want, "a short source");
}

/// A source of texels reads, at Metal's blit rows of its level, each texel's color word and alpha
/// byte, and 0 in a row's padding or past a plane.
#[test]
fn a_source_of_texels_reads_its_planes_bytes() {
    let cases = [
        ([48, 20], [48, 20], [0, 0], [0, 0], [48, 20]),
        ([9, 7], [48, 20], [0, 0], [3, 2], [9, 7]),
        ([20, 8], [33, 9], [1, 2], [2, 1], [17, 6]),
        ([3, 4], [5, 5], [0, 1], [0, 0], [3, 3]),
    ];
    for (n, (level, to_level, from, to, size)) in cases.into_iter().enumerate() {
        let texels: Vec<u32> = (0..2 * level[0] * level[1])
            .map(|k| 0x0101_0101_u32.wrapping_mul(k + 3).wrapping_add(n as u32))
            .collect();
        let words: Vec<u8> = texels.iter().flat_map(|w| w.to_le_bytes()).collect();
        let rows = PlaneRows::of(level[0]);
        // Byte `p` of the source's plane `plane` (0: color, 1: alpha), from its first byte.
        let byte = |plane: u32, p: u32| -> u8 {
            let (pitch, unit) = if plane == 0 {
                (rows.color, 4)
            } else {
                (rows.alpha, 1)
            };
            let (y, x, k) = (p / pitch, p % pitch / unit, p % pitch % unit);
            if x < level[0] && y < level[1] {
                let word = texels[(2 * (y * level[0] + x) + plane) as usize];
                if plane == 0 {
                    word.to_le_bytes()[k as usize]
                } else {
                    word as u8
                }
            } else {
                0
            }
        };
        let heap = random_bytes(65_536, 40 + n as u64);
        let (dc, da, db) = planes(to_level, 0);
        let destination = PlaneBytes::new(
            dc + to[1] * db + 4 * to[0],
            da + to[1] * db + to[0],
            db,
            65_536,
        );
        // The model's source: plane 0's bytes from byte 0, the alpha plane's from 2^30.
        let blit = rows.blit;
        let model_source = PlaneBytes::new(
            from[1] * blit + 4 * from[0],
            (1 << 30) + from[1] * blit + from[0],
            blit,
            u32::MAX,
        );
        let read = |s: u32| {
            if s >= 1 << 30 {
                byte(1, s - (1 << 30))
            } else {
                byte(0, s)
            }
        };
        let want = modelled(read, &model_source, &destination, &heap, size);
        let got = copied(
            &PlaneSource::Texels { level, from },
            &words,
            &destination,
            &heap,
            size,
            u32::MAX,
        );
        assert!(got == want, "texels of {level:?} from {from:?}");
    }
}
