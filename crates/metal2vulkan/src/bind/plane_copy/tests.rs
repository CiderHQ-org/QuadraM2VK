//! The plane copy kernel on the CPU interpreter, against Metal's measured copies and a byte-by-byte
//! model.

use super::{typed_module, PlaneCopyModule, PlaneRows, DESTINATION, SHAPE, SOURCE};
use crate::bind::{PlaneBytes, PlaneSource};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};
use std::collections::BTreeMap;

#[test]
fn the_plane_copy_kernel_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(typed_module().expect("module"), &p);
    }
}

fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn bytes_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// One copy: the source and destination levels (width, height), the copy's origins and size.
#[derive(Clone, Copy, Debug)]
struct Case {
    source: [u32; 2],
    destination: [u32; 2],
    from: [u32; 2],
    to: [u32; 2],
    size: [u32; 2],
}

/// The destination's texels after the kernel copies `copy` from `source` (two words a texel)
/// over `destination`, both invocation orders alike.
fn copied(copy: Case, source: &[u32], destination: &[u32]) -> Vec<u32> {
    copied_in(copy, source, destination, u32::MAX)
}

/// [`copied`] in dispatches of at most `batch` workgroups, each with the shape of its first
/// texel.
fn copied_in(copy: Case, source: &[u32], destination: &[u32], batch: u32) -> Vec<u32> {
    let form = PlaneSource::Texels {
        level: copy.source,
        from: copy.from,
    };
    run_copy(copy, &form, source, destination, batch)
}

/// The destination's texels after the kernel copies `copy` from `source` (the binding's words,
/// in `form`) over `destination`, in dispatches of at most `batch` workgroups, both invocation
/// orders alike.
fn run_copy(
    copy: Case,
    form: &PlaneSource,
    source: &[u32],
    destination: &[u32],
    batch: u32,
) -> Vec<u32> {
    let m = typed_module().expect("module");
    let texels = copy.destination[0] * copy.destination[1];
    let groups = texels.div_ceil(PlaneCopyModule::LOCAL_SIZE);
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let mut words = destination.to_vec();
        let mut done = 0;
        while done < groups {
            let n = batch.min(groups - done);
            let first = done * PlaneCopyModule::LOCAL_SIZE;
            let shape = PlaneCopyModule::shape(form, copy.destination, copy.to, copy.size, first);
            let part = |slot, bytes: Vec<u8>| Buffer {
                slot: Some(slot),
                address: None,
                bytes,
            };
            let res = Resources {
                buffers: vec![
                    part(SOURCE, bytes_of(source)),
                    part(DESTINATION, bytes_of(&words)),
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
            assert_eq!(words_of(&buffers[0]), source, "the source is only read");
            assert_eq!(buffers[2], shape.to_vec(), "the shape is only read");
            words = words_of(&buffers[1]);
            done += n;
        }
        out.push(words);
    }
    assert_eq!(out[0], out[1], "order independent");
    out.swap_remove(0)
}

/// The copy by brute force: each row's bytes in turn, later rows over earlier ones, on a map of
/// each plane's bytes; then each destination texel's words from the bytes landing on it.
fn modelled(copy: Case, source: &[u32], destination: &[u32]) -> Vec<u32> {
    let s = PlaneRows::of(copy.source[0]);
    modelled_reads(copy, destination, |unit, r, c| {
        let (s_rows, word) = if unit == 4 {
            (s.color, 0)
        } else {
            (s.alpha, 1)
        };
        let read = (copy.from[1] + r) * s.blit + unit * (copy.from[0] + c);
        let (y, x) = (read / s_rows, read % s_rows / unit);
        if x < copy.source[0] && y < copy.source[1] {
            source[(2 * (y * copy.source[0] + x) + word) as usize]
        } else {
            0
        }
    })
}

/// The copy by brute force from `bytes` laid out as `planes`: the color word and the alpha byte
/// of each row and column at the planes' blit rows, 0 at or past their end.
fn modelled_bytes(copy: Case, planes: PlaneBytes, bytes: &[u8], destination: &[u32]) -> Vec<u32> {
    let byte = |at: u32| {
        if at < planes.end {
            bytes[at as usize]
        } else {
            0
        }
    };
    modelled_reads(copy, destination, |unit, r, c| {
        if unit == 4 {
            let at = planes.color + r * planes.blit + 4 * c;
            u32::from_le_bytes([byte(at), byte(at + 1), byte(at + 2), byte(at + 3)])
        } else {
            u32::from(byte(planes.alpha + r * planes.blit + c))
        }
    })
}

/// The copy by brute force, `read` giving the source's color word (`unit` 4) or alpha byte
/// (`unit` 1) of the copy's row and column.
fn modelled_reads(
    copy: Case,
    destination: &[u32],
    read: impl Fn(u32, u32, u32) -> u32,
) -> Vec<u32> {
    let d = PlaneRows::of(copy.destination[0]);
    let mut out = destination.to_vec();
    for (unit, word, d_rows) in [(4, 0, d.color), (1, 1, d.alpha)] {
        let mut plane = BTreeMap::new();
        for r in 0..copy.size[1] {
            for c in 0..copy.size[0] {
                let write = (copy.to[1] + r) * d.blit + unit * (copy.to[0] + c);
                plane.insert(write, read(unit, r, c));
            }
        }
        for (at, value) in plane {
            let (y, x) = (at / d_rows, at % d_rows / unit);
            if x < copy.destination[0] && y < copy.destination[1] {
                out[(2 * (y * copy.destination[0] + x) + word) as usize] = value;
            }
        }
    }
    out
}

/// Run-length encoded tags, as the probe printed them: `s:t*n` says entries `s .. s + n` hold
/// `t .. t + n` (or `n` zeros where `t` is 0).
fn tags(rle: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for run in rle.split_whitespace() {
        let (at, rest) = run.split_once(':').expect("a run");
        let (first, n) = rest.split_once('*').expect("a count");
        let (at, first, n): (usize, u32, u32) = (
            at.parse().expect("an index"),
            first.parse().expect("a tag"),
            n.parse().expect("a count"),
        );
        assert_eq!(at, out.len(), "runs in order");
        out.extend((0..n).map(|k| if first == 0 { 0 } else { first + k }));
    }
    out
}

/// Metal's copies between two `RGB10A8_2P_XR10` textures of the same size, 5 rows high (an Apple M4
/// under macOS 27.0.1, shared and private textures, plain and under Metal's validation layer
/// alike): the width, the region (`None`: the whole texture), and each
/// destination texel's source texel plus 1 (0: the destination's own, which was 0) of plane 0
/// and of the alpha plane.
type Measured = (
    u32,
    Option<([u32; 2], [u32; 2], [u32; 2])>,
    &'static str,
    &'static str,
);
const METAL: [Measured; 10] = [
    (4, None, "0:1*20", "0:1*20"),
    (
        4,
        Some(([1, 2], [2, 3], [2, 1])),
        "0:0*6 6:10*2 8:0*2 10:14*2 12:0*2 14:18*2 16:0*4",
        "0:0*6 6:10*2 8:0*2 10:14*2 12:0*2 14:18*2 16:0*4",
    ),
    (5, None, "0:1*15 15:0*10", "0:1*25"),
    (
        5,
        Some(([1, 2], [2, 3], [3, 1])),
        "0:0*5 5:8*1 6:0*4 10:13*1 11:0*14",
        "0:0*8 8:12*2 10:0*3 13:17*2 15:0*3 18:22*2 20:0*5",
    ),
    (7, None, "0:1*21 21:0*14", "0:1*35"),
    (
        7,
        Some(([1, 2], [3, 3], [4, 1])),
        "0:0*7 7:9*3 10:0*1 11:13*2 13:0*1 14:16*3 17:0*18",
        "0:0*11 11:16*3 14:0*4 18:23*3 21:0*4 25:30*3 28:0*7",
    ),
    (8, None, "0:1*40", "0:1*8 8:0*8 16:17*8 24:0*8 32:33*8"),
    (
        8,
        Some(([1, 2], [4, 3], [4, 1])),
        "0:0*12 12:18*4 16:0*4 20:26*4 24:0*4 28:34*4 32:0*8",
        "0:0*20 20:34*4 24:0*16",
    ),
    (
        257,
        None,
        "0:1*1269 1269:0*16",
        "0:1*257 257:0*722 979:980*242 1221:0*64",
    ),
    (
        257,
        Some(([1, 2], [128, 3], [129, 1])),
        "0:0*382 382:511*4 386:0*3 389:515*121 510:0*125 635:764*8 643:0*3 646:772*117 \
         763:0*125 888:1017*12 900:0*3 903:1029*113 1016:0*269",
        "0:0*1285",
    ),
];

/// The kernel copies both planes as Metal's blit did, whole and a region, at widths that are and
/// are not multiples of 4: plane 0's texels where its rows are the blit's, its rows lost or moved
/// where they are not, and the alpha bytes moved at the blit's rows.
#[test]
fn copies_move_both_planes_as_metal_does() {
    const HEIGHT: u32 = 5;
    for (width, region, color, alpha) in METAL {
        let n = width * HEIGHT;
        let (from, size, to) = region.unwrap_or(([0, 0], [width, HEIGHT], [0, 0]));
        let copy = Case {
            source: [width, HEIGHT],
            destination: [width, HEIGHT],
            from,
            to,
            size,
        };
        let source: Vec<u32> = (0..n).flat_map(|i| [i + 1, i + 1]).collect();
        let destination = vec![0; 2 * n as usize];
        let got = copied(copy, &source, &destination);
        let planes: [Vec<u32>; 2] =
            [0, 1].map(|w| got.iter().skip(w).step_by(2).copied().collect());
        assert_eq!(planes[0], tags(color), "plane 0 of {copy:?}");
        assert_eq!(planes[1], tags(alpha), "the alpha plane of {copy:?}");
    }
}

/// Between textures of other sizes, below a width of 4, and where the copy's rows overlap in
/// the destination, the kernel keeps the model's bytes: the later row's where two land on one.
#[test]
fn copies_between_sizes_and_overlapping_rows_keep_the_later_row() {
    let copies = [
        // Overlapping rows at the same addresses, and at others.
        Case {
            source: [9, 7],
            destination: [9, 7],
            from: [0, 0],
            to: [0, 0],
            size: [9, 7],
        },
        Case {
            source: [9, 7],
            destination: [9, 7],
            from: [1, 2],
            to: [2, 1],
            size: [7, 3],
        },
        Case {
            source: [13, 6],
            destination: [10, 9],
            from: [0, 1],
            to: [1, 0],
            size: [9, 5],
        },
        // Between sizes.
        Case {
            source: [48, 20],
            destination: [33, 9],
            from: [3, 4],
            to: [10, 2],
            size: [17, 5],
        },
        Case {
            source: [16, 4],
            destination: [16, 30],
            from: [0, 0],
            to: [0, 0],
            size: [16, 4],
        },
        Case {
            source: [100, 7],
            destination: [17, 30],
            from: [50, 0],
            to: [1, 20],
            size: [15, 7],
        },
        Case {
            source: [24, 16],
            destination: [8, 40],
            from: [2, 1],
            to: [0, 25],
            size: [8, 12],
        },
        // Below a width of 4: every row on the first row's bytes.
        Case {
            source: [2, 9],
            destination: [3, 5],
            from: [0, 1],
            to: [1, 0],
            size: [2, 3],
        },
        Case {
            source: [1, 6],
            destination: [1, 6],
            from: [0, 0],
            to: [0, 0],
            size: [1, 6],
        },
    ];
    for copy in copies {
        let (s, d) = (
            copy.source[0] * copy.source[1],
            copy.destination[0] * copy.destination[1],
        );
        let source: Vec<u32> = (0..2 * s).map(|k| 0x1000_0000 + k).collect();
        let destination: Vec<u32> = (0..2 * d).map(|k| 0x2000_0000 + k).collect();
        assert_eq!(
            copied(copy, &source, &destination),
            modelled(copy, &source, &destination),
            "{copy:?}"
        );
    }
}

/// The rows of each plane and of the blit.
#[test]
fn plane_rows_are_metals() {
    let rows = |w| {
        let r = PlaneRows::of(w);
        (r.color, r.alpha, r.blit)
    };
    assert_eq!(rows(1), (16, 16, 0));
    assert_eq!(rows(4), (16, 16, 16));
    assert_eq!(rows(5), (32, 16, 16));
    assert_eq!(rows(48), (192, 48, 192));
    assert_eq!(rows(257), (1040, 272, 1024));
}

/// A destination dispatched in batches, each from its first texel, copies as one dispatch does.
#[test]
fn batches_copy_as_one_dispatch() {
    let copy = Case {
        source: [37, 9],
        destination: [29, 11],
        from: [3, 1],
        to: [2, 3],
        size: [26, 7],
    };
    let source: Vec<u32> = (0..2 * 37 * 9).map(|k| 0x1000_0000 + k).collect();
    let destination: Vec<u32> = (0..2 * 29 * 11).map(|k| 0x2000_0000 + k).collect();
    let whole = copied(copy, &source, &destination);
    assert_eq!(whole, modelled(copy, &source, &destination));
    assert_eq!(copied_in(copy, &source, &destination, 2), whole);
}

/// A source given as its planes' bytes, as a texture over a surface's planes lays them (plane 0
/// and the alpha plane at their own rows, both read at the blit rows of plane 0's, the copy's
/// alpha bytes past the alpha plane among other bytes and past the binding's end), copies the
/// bytes as they lie: each row's color word whole and its alpha byte alone, 0 past the end, the
/// later row's where rows meet in the destination.
#[test]
fn a_bytes_source_copies_the_bytes_as_they_lie() {
    // (surface: plane 0's first byte and rows, the alpha plane's first byte and rows, the
    // binding's end; the copy).
    let cases = [
        (
            (0, 256, 1296, 64, 1644),
            Case {
                source: [37, 5],
                destination: [37, 5],
                from: [0, 0],
                to: [0, 0],
                size: [37, 5],
            },
        ),
        (
            (32, 192, 1088, 64, 1424),
            Case {
                source: [33, 5],
                destination: [9, 7],
                from: [3, 1],
                to: [1, 2],
                size: [7, 4],
            },
        ),
        (
            (0, 64, 1024, 16, 1200),
            Case {
                source: [16, 16],
                destination: [16, 16],
                from: [0, 3],
                to: [0, 0],
                size: [16, 13],
            },
        ),
    ];
    for ((color, rows, alpha, _alpha_rows, end), copy) in cases {
        let bytes: Vec<u8> = (0..end + 64).map(|k| (k * 151 + 29) as u8).collect();
        let planes = PlaneBytes::new(
            color + copy.from[1] * rows + 4 * copy.from[0],
            alpha + copy.from[1] * rows + copy.from[0],
            rows,
            end,
        );
        let d = copy.destination[0] * copy.destination[1];
        let destination: Vec<u32> = (0..2 * d).map(|k| 0x2000_0000 + k).collect();
        let words = words_of(&bytes);
        let got = run_copy(
            copy,
            &PlaneSource::Bytes(planes),
            &words,
            &destination,
            u32::MAX,
        );
        assert_eq!(
            got,
            modelled_bytes(copy, planes, &bytes, &destination),
            "{copy:?}"
        );
        assert_eq!(
            run_copy(copy, &PlaneSource::Bytes(planes), &words, &destination, 1),
            got,
            "{copy:?} in batches"
        );
    }
}

/// A level's texels laid out as Metal lays an own texture's planes (rows of `align(4 w, 16)` and
/// `align(w, 16)` bytes, the padding 0, nothing past the alpha plane) copy from their bytes as
/// from the texels.
#[test]
fn a_level_s_bytes_copy_as_its_texels() {
    let copies = [
        Case {
            source: [9, 7],
            destination: [9, 7],
            from: [1, 2],
            to: [2, 1],
            size: [7, 3],
        },
        Case {
            source: [48, 20],
            destination: [33, 9],
            from: [3, 4],
            to: [10, 2],
            size: [17, 5],
        },
        Case {
            source: [24, 16],
            destination: [8, 40],
            from: [2, 1],
            to: [0, 25],
            size: [8, 12],
        },
    ];
    for copy in copies {
        let [w, h] = copy.source;
        let rows = PlaneRows::of(w);
        // Plane 0, a gap of zeros past every row the copy reaches, then the alpha plane.
        let alpha_at = (h + copy.size[1]) * rows.color.max(rows.blit) + 4096;
        let end = alpha_at + h * rows.alpha;
        let source: Vec<u32> = (0..2 * w * h).map(|k| 0x1000_0000 + k).collect();
        let mut bytes = vec![0u8; end as usize];
        for y in 0..h {
            for x in 0..w {
                let texel = (2 * (y * w + x)) as usize;
                let at = (y * rows.color + 4 * x) as usize;
                bytes[at..at + 4].copy_from_slice(&source[texel].to_le_bytes());
                bytes[(alpha_at + y * rows.alpha + x) as usize] = source[texel + 1] as u8;
            }
        }
        // The texels' second words hold the alpha byte alone.
        let source: Vec<u32> = source
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|t| [t[0], t[1] & 0xff])
            .collect();
        let planes = PlaneBytes::new(
            copy.from[1] * rows.blit + 4 * copy.from[0],
            alpha_at + copy.from[1] * rows.blit + copy.from[0],
            rows.blit,
            end,
        );
        let d = copy.destination[0] * copy.destination[1];
        let destination: Vec<u32> = (0..2 * d).map(|k| 0x2000_0000 + k).collect();
        assert_eq!(
            run_copy(
                copy,
                &PlaneSource::Bytes(planes),
                &words_of(&bytes),
                &destination,
                u32::MAX
            ),
            copied(copy, &source, &destination),
            "{copy:?}"
        );
    }
}
