//! The capture assembly and settle kernels against plain Rust references, through the CPU
//! interpreter in both invocation orders: the records' and entries' bytes; random draws of every
//! topology, instance counts, restarting strips with flagged records, culling on and off; the
//! scan's bound; entries that describe no primitive; the settle's two cases; and both modules
//! built and valid on every committed device. (That the placeholders reach nothing under the tile
//! reach rule is `tile_reach`'s test `assembly`.)

use std::path::PathBuf;

use super::kernel::{
    assembly_typed_module, settle_typed_module, DRAWS, PASS, PRIMITIVES, RECORDS, STATUS, TILES,
};
use super::{
    capture_assembly_module, capture_settle_module, CaptureAssemblyModule as A,
    CaptureSettleModule as S, NOTHING, UNKNOWN,
};
use crate::bind::status::{ExecutionStatus, StatusBits, StatusWord};
use crate::bind::tile_reach::TileReachModule as R;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::metal::PrimitiveType;
use crate::pipeline::sym::ResourceSlot;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::assert_valid;

/// A splitmix64 stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn word(&mut self) -> u32 {
        self.next() as u32
    }

    /// A number in `[0, n)`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// True `k` times in `n`.
    fn chance(&mut self, k: u64, n: u64) -> bool {
        self.below(n) < k
    }

    fn pick<X: Copy>(&mut self, xs: &[X]) -> X {
        xs[self.below(xs.len() as u64) as usize]
    }
}

/// Little-endian bytes as words.
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn buffer(slot: ResourceSlot, words: &[u32]) -> Buffer {
    Buffer {
        slot: Some(slot),
        address: None,
        bytes: bytes(words),
    }
}

/// The topologies, by raw value.
const TOPOLOGIES: [PrimitiveType; 5] = [
    PrimitiveType::Point,
    PrimitiveType::Line,
    PrimitiveType::LineStrip,
    PrimitiveType::Triangle,
    PrimitiveType::TriangleStrip,
];

/// What the assembly reads: the records, the draws' header and entries (words), and the
/// primitives' binding's words before the run.
#[derive(Clone, Debug)]
struct Batch {
    records: Vec<[u32; 8]>,
    draws: Vec<u32>,
    out: Vec<u32>,
}

impl Batch {
    /// A batch of `entries` (each [`A::DRAW_WORDS`] words) after a header naming `named` of
    /// them and `total` primitives, its primitives' binding `out` words of a fill pattern.
    fn of(records: Vec<[u32; 8]>, named: u32, total: u32, entries: &[[u32; 8]], out: u32) -> Batch {
        let mut draws = words(&A::header(named, total));
        draws.extend(entries.iter().flatten());
        Batch {
            records,
            draws,
            out: (0..out).map(|w| 0xa5a5_0000 | w).collect(),
        }
    }

    /// The primitives' binding after the kernel ran, through the interpreter in both orders
    /// (alike).
    fn kernel(&self, m: &TypedModule) -> Vec<u32> {
        let records: Vec<u32> = self.records.iter().flatten().copied().collect();
        let total = self.draws.get(1).copied().unwrap_or(0);
        let mut outs = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![
                    buffer(RECORDS, &records),
                    buffer(DRAWS, &self.draws),
                    buffer(PRIMITIVES, &self.out),
                ],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [total.div_ceil(A::LOCAL_SIZE).max(1), 1, 1],
                step_limit: 1 << 40,
                ..Config::default()
            };
            outs.push(run(m, &cfg, res).expect("run").buffers);
        }
        assert_eq!(outs[0], outs[1], "order independent");
        words(&outs[0][2])
    }

    /// The primitives' binding after the reference assembly: each primitive below the header's
    /// total and the binding's whole primitives written by [`reference`].
    fn rule(&self) -> Vec<u32> {
        let mut out = self.out.clone();
        if self.draws.len() < A::HEADER_WORDS as usize {
            return out;
        }
        let (named, total) = (self.draws[0], self.draws[1]);
        let entries: Vec<[u32; 8]> = self.draws[A::HEADER_WORDS as usize..]
            .as_chunks::<{ A::DRAW_WORDS as usize }>()
            .0
            .iter()
            .copied()
            .take(named as usize)
            .collect();
        let fits = out.len() as u32 / R::PRIMITIVE_WORDS;
        for i in 0..total.min(fits) {
            let at = (i * R::PRIMITIVE_WORDS) as usize;
            out[at..at + 16].copy_from_slice(&reference(&self.records, &entries, i));
        }
        out
    }

    /// Where the kernel and the reference differ, primitive by primitive.
    fn differences(&self, m: &TypedModule) -> Vec<String> {
        let (kernel, rule) = (self.kernel(m), self.rule());
        if kernel == rule {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (i, (k, r)) in kernel.chunks(16).zip(rule.chunks(16)).enumerate() {
            if k != r {
                out.push(format!(
                    "primitive {i}: kernel {k:08x?}, reference {r:08x?}"
                ));
            }
        }
        if out.is_empty() {
            out.push("the binding differs past its whole primitives".to_string());
        }
        out
    }
}

/// A primitive's 16 words: its kind, state, layer 0, size and vertices.
fn primitive(kind: u32, state: u32, size: u32, vertices: [[u32; 4]; 3]) -> [u32; 16] {
    let mut w = [0; 16];
    w[0] = kind;
    w[1] = state;
    w[3] = size;
    for (v, vertex) in vertices.iter().enumerate() {
        w[4 + 4 * v..8 + 4 * v].copy_from_slice(vertex);
    }
    w
}

/// The triangle of three `vertex`es, of size 0, under `state`.
fn placeholder(vertex: [u32; 4], state: u32) -> [u32; 16] {
    primitive(R::TRIANGLE, state, 0, [vertex; 3])
}

/// Metal's assembly of primitive `i` of the pass (the module docs), in plain Rust: from the last
/// entry whose first primitive is at most `i` (entry 0 where none is), its `j`th primitive's
/// records, by its topology; a restarting strip's primitive touching a flagged record reaches
/// nothing; a triangle strip's triangle takes its parity from its piece's first record, found
/// over the whole instance, where the entry restarts and culls, and is unknown where the last
/// flagged record before it lies more than the scan's bound back (or, with none, where `j` does);
/// anything the entries do not describe is unknown.
fn reference(records: &[[u32; 8]], entries: &[[u32; 8]], i: u32) -> [u32; 16] {
    let Some(e) = entries.iter().rev().find(|e| e[0] <= i).or(entries.first()) else {
        return placeholder(UNKNOWN, 0);
    };
    let [first, primitives, first_record, count, topology, restarts, culls, state] = *e;
    let unknown = placeholder(UNKNOWN, state);
    let j = i.wrapping_sub(first);
    let record = |v: u32| -> Option<[u32; 8]> {
        if v >= count {
            return None;
        }
        let slot = first_record.checked_add(v)?;
        records.get(slot as usize).copied()
    };
    let (n, a) = match TOPOLOGIES.iter().find(|t| t.as_raw() == topology) {
        Some(PrimitiveType::Point) => (1, j),
        Some(PrimitiveType::Line) => (2, j.wrapping_mul(2)),
        Some(PrimitiveType::LineStrip) => (2, j),
        Some(PrimitiveType::Triangle) => (3, j.wrapping_mul(3)),
        Some(PrimitiveType::TriangleStrip) => (3, j),
        None => return unknown,
    };
    if j >= primitives {
        return unknown;
    }
    let Some(vs) = (0..n)
        .map(|k| record(a.wrapping_add(k)))
        .collect::<Option<Vec<_>>>()
    else {
        return unknown;
    };
    let strip = topology == PrimitiveType::LineStrip.as_raw()
        || topology == PrimitiveType::TriangleStrip.as_raw();
    let restarting = strip && restarts != 0;
    if restarting && vs.iter().any(|r| r[5] != 0) {
        return placeholder(NOTHING, state);
    }
    let triangle_strip = topology == PrimitiveType::TriangleStrip.as_raw();
    let mut odd = j % 2 == 1;
    if triangle_strip && restarting && culls != 0 {
        let flagged = |q: u32| record(q).is_some_and(|r| r[5] != 0);
        let last = (0..j).rev().find(|&q| flagged(q));
        let known = match last {
            Some(f) => j - f <= A::SCAN_LIMIT,
            None => j <= A::SCAN_LIMIT,
        };
        if !known {
            return unknown;
        }
        odd = (j - last.map_or(0, |f| f + 1)) % 2 == 1;
    }
    let order = match triangle_strip && odd {
        true => [0, 2, 1],
        false => [0, 1, 2],
    };
    let kind = match n {
        1 => R::POINT,
        2 => R::LINE,
        _ => R::TRIANGLE,
    };
    let mut vertices = [[0; 4]; 3];
    for (slot, &k) in vertices.iter_mut().zip(&order).take(n as usize) {
        slot.copy_from_slice(&vs[k][..4]);
    }
    primitive(kind, state, vs[0][4], vertices)
}

/// A record of random position and size words (any bits, or finite values), restarting where
/// `flag` (its restart word 1, or another value not zero).
fn random_record(rng: &mut Rng, flag: bool) -> [u32; 8] {
    let mut position = [0f32; 4].map(|_| (rng.below(4001) as f32 - 2000.0) / 1000.0);
    position[3] = 1.0;
    let mut r = A::record(position, rng.below(64) as f32 / 4.0, flag);
    if rng.chance(1, 8) {
        r[..16].copy_from_slice(&bytes(&[0; 4].map(|_| rng.word())));
    }
    let mut w: [u32; 8] = words(&r).try_into().expect("a record");
    if flag && rng.chance(1, 4) {
        w[5] = rng.pick(&[2, 7, u32::MAX]);
    }
    w
}

/// A seeded batch: up to six draws of random topologies, 1 to 3 instances each, of a few to a
/// few hundred records (some strips longer than the scan's bound), restarting strips with
/// flagged records at several densities, culling or not, under a few states; records of an
/// instance sometimes after a gap; an entry of no primitives sometimes kept.
fn random_batch(rng: &mut Rng) -> Batch {
    let mut records = Vec::new();
    let mut entries = Vec::new();
    let mut total = 0;
    for _ in 0..1 + rng.below(6) {
        let topology = rng.pick(&TOPOLOGIES);
        let strip = matches!(
            topology,
            PrimitiveType::LineStrip | PrimitiveType::TriangleStrip
        );
        let long = match topology {
            PrimitiveType::TriangleStrip => rng.chance(1, 3),
            _ => rng.chance(1, 10),
        };
        let count = match long {
            true => 240 + rng.below(160),
            false => {
                let below = rng.pick(&[4, 16, 16, 64, 64]);
                rng.below(below)
            }
        } as u32;
        let restarts = rng.chance(3, 4);
        let culls = rng.chance(1, 2);
        let density = rng.pick(&[0, 0, 2, 8, 40, 200]);
        let state = rng.below(4) as u32;
        for _ in 0..1 + rng.below(3) {
            if rng.chance(1, 4) {
                for _ in 0..rng.below(5) {
                    records.push(random_record(rng, false));
                }
            }
            let first_record = records.len() as u32;
            for _ in 0..count {
                let flag = density != 0 && rng.chance(density, 1000);
                records.push(random_record(rng, flag));
            }
            let primitives = A::primitives(topology, count);
            if primitives == 0 && rng.chance(1, 2) {
                continue;
            }
            let restarts = restarts && (strip || rng.chance(1, 2));
            let e = A::draw(total, first_record, count, topology, restarts, culls, state);
            entries.push(words(&e).try_into().expect("an entry"));
            total += primitives;
        }
    }
    let out = total * R::PRIMITIVE_WORDS + rng.below(20) as u32;
    Batch::of(records, entries.len() as u32, total, &entries, out)
}

/// A capture record's bytes, the draws' header's and an entry's are the documented words, and an
/// instance's primitives are its topology's.
#[test]
fn the_records_and_entries_are_the_documented_words() {
    let r = A::record([1.5, -2.0, 0.25, 1.0], 3.0, true);
    assert_eq!(
        words(&r),
        [
            1.5f32.to_bits(),
            (-2.0f32).to_bits(),
            0.25f32.to_bits(),
            1.0f32.to_bits(),
            3.0f32.to_bits(),
            1,
            0,
            0
        ]
    );
    assert_eq!(words(&A::record([0.0; 4], 0.0, false))[5], 0);
    assert_eq!(r.len(), 4 * A::RECORD_WORDS as usize);
    assert_eq!(words(&A::header(7, 300)), [7, 300]);
    let e = A::draw(40, 12, 9, PrimitiveType::TriangleStrip, true, false, 3);
    assert_eq!(words(&e), [40, 7, 12, 9, 4, 1, 0, 3]);
    let e = A::draw(0, 0, 9, PrimitiveType::Line, false, true, 0);
    assert_eq!(words(&e), [0, 4, 0, 9, 1, 0, 1, 0]);
    assert_eq!(e.len(), 4 * A::DRAW_WORDS as usize);
    let primitives = |t: PrimitiveType| [0, 1, 2, 3, 7].map(|n| A::primitives(t, n));
    assert_eq!(primitives(PrimitiveType::Point), [0, 1, 2, 3, 7]);
    assert_eq!(primitives(PrimitiveType::Line), [0, 0, 1, 1, 3]);
    assert_eq!(primitives(PrimitiveType::LineStrip), [0, 0, 1, 2, 6]);
    assert_eq!(primitives(PrimitiveType::Triangle), [0, 0, 0, 1, 2]);
    assert_eq!(primitives(PrimitiveType::TriangleStrip), [0, 0, 0, 1, 5]);
    assert_eq!(NOTHING.map(f32::from_bits), [0.0, 0.0, 0.0, 1.0]);
    assert!(UNKNOWN.iter().all(|&w| f32::from_bits(w).is_nan()));
}

/// On 48 seeded batches (every topology; 1 to 3 instances; restarting strips with flagged records
/// at several densities, culling or not; strips past the scan's bound; gaps between instances'
/// records; entries of no primitives), the kernel writes the reference's primitives.
#[test]
fn the_kernel_assembles_what_the_reference_does_on_random_draws() {
    let m = assembly_typed_module().expect("module");
    let mut rng = Rng(0xca97_0a55);
    let mut wrong = Vec::new();
    let (mut unknown, mut nothing) = (0, 0);
    for _ in 0..48 {
        let batch = random_batch(&mut rng);
        let rule = batch.rule();
        for p in rule.as_chunks::<16>().0.iter() {
            unknown += usize::from(p[4..8] == UNKNOWN);
            nothing += usize::from(p[4..8] == NOTHING && p[0] == R::TRIANGLE && p[3] == 0);
        }
        wrong.extend(batch.differences(&m));
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert!(
        unknown > 0 && nothing > 0,
        "{unknown} unknown, {nothing} nothing"
    );
}

/// A restarting triangle strip that culls, of 1000 records flagged at 100 and 700, then an
/// instance flagged nowhere: each triangle touching a flag reaches nothing; a piece's triangles
/// take their parity from its first record (101: even, 102: odd); a triangle is known up to
/// [`A::SCAN_LIMIT`] records past the last flag (356 and 956) and unknown beyond (357, 957),
/// and, with no flag, up to that many past vertex 0. The same strip that does not cull has no
/// unknown triangle and counts parity from vertex 0; one that does not restart ignores the flags.
#[test]
fn a_culling_strip_s_parity_is_its_piece_s_within_the_scan() {
    let m = assembly_typed_module().expect("module");
    let mut rng = Rng(0x5ca7_1e55);
    let count = 1000;
    let mut records: Vec<[u32; 8]> = (0..count)
        .map(|p| random_record(&mut rng, p == 100 || p == 700))
        .collect();
    records.extend((0..count).map(|_| random_record(&mut rng, false)));
    let per = A::primitives(PrimitiveType::TriangleStrip, count);
    for (restarts, culls) in [(true, true), (true, false), (false, true)] {
        let entries: Vec<[u32; 8]> = (0..2)
            .map(|n| {
                let e = A::draw(
                    n * per,
                    n * count,
                    count,
                    PrimitiveType::TriangleStrip,
                    restarts,
                    culls,
                    5,
                );
                words(&e).try_into().expect("an entry")
            })
            .collect();
        let batch = Batch::of(records.clone(), 2, 2 * per, &entries, 2 * per * 16);
        let out = batch.kernel(&m);
        assert_eq!(out, batch.rule(), "restarts {restarts}, culls {culls}");
        let at = |i: u32| &out[16 * i as usize..16 * i as usize + 16];
        let is = |i: u32, v: [u32; 4]| at(i)[4..8] == v;
        let order = |i: u32| -> Vec<u32> {
            (0..3)
                .map(|v| {
                    let w = &at(i)[4 + 4 * v..8 + 4 * v];
                    (0..count)
                        .find(|&q| records[q as usize][..4] == *w)
                        .expect("a record's position")
                })
                .collect()
        };
        let unknown: Vec<u32> = (0..2 * per).filter(|&i| is(i, UNKNOWN)).collect();
        let nothing: Vec<u32> = (0..2 * per).filter(|&i| is(i, NOTHING)).collect();
        match (restarts, culls) {
            (true, true) => {
                assert_eq!(nothing, [98, 99, 100, 698, 699, 700]);
                assert_eq!(order(101), [101, 102, 103]);
                assert_eq!(order(102), [102, 104, 103]);
                assert_eq!(order(97), [97, 99, 98]);
                assert!(!is(356, UNKNOWN) && is(357, UNKNOWN));
                assert!(!is(956, UNKNOWN) && is(957, UNKNOWN));
                let unflagged = (per + 257..2 * per).count();
                let expect = (357..698).count() + (957..per).count() + unflagged;
                assert_eq!(unknown.len(), expect);
                assert!(!is(per + 256, UNKNOWN) && is(per + 257, UNKNOWN));
            }
            (true, false) => {
                assert_eq!(nothing, [98, 99, 100, 698, 699, 700]);
                assert!(unknown.is_empty());
                assert_eq!(order(101), [101, 103, 102]);
            }
            (false, _) => {
                assert!(nothing.is_empty() && unknown.is_empty());
                assert_eq!(order(101), [101, 103, 102]);
            }
        }
    }
}

/// Entries that describe no primitive, and bindings that hold less than the header names: the
/// kernel writes the unknown triangle (no entry, a topology that is none, primitives past the
/// entry's records or the records' binding) and nothing past the primitives' binding or without
/// a header, as the reference does.
#[test]
fn what_the_entries_do_not_describe_is_unknown() {
    let m = assembly_typed_module().expect("module");
    let mut rng = Rng(0x0dd5_e575);
    let records: Vec<[u32; 8]> = (0..12).map(|_| random_record(&mut rng, false)).collect();
    let raw = |first, primitives, first_record, count, topology, state| {
        [
            first,
            primitives,
            first_record,
            count,
            topology,
            0,
            0,
            state,
        ]
    };
    let triangles = PrimitiveType::Triangle.as_raw();
    // Each case, the unknown primitives it writes, and the batch.
    let cases: Vec<(&str, usize, Batch)> = vec![
        ("no entry", 3, Batch::of(records.clone(), 0, 3, &[], 48)),
        (
            "topologies that are none",
            3,
            Batch::of(
                records.clone(),
                3,
                3,
                &[
                    raw(0, 1, 0, 3, 5, 1),
                    raw(1, 1, 0, 3, 7, 2),
                    raw(2, 1, 0, 3, u32::MAX, 3),
                ],
                48,
            ),
        ),
        (
            "primitives past the entry's records",
            2,
            Batch::of(records.clone(), 1, 4, &[raw(0, 4, 0, 6, triangles, 1)], 64),
        ),
        (
            "records past the binding",
            1,
            Batch::of(records.clone(), 1, 3, &[raw(0, 3, 6, 9, triangles, 2)], 48),
        ),
        (
            "a first primitive past the invocation's",
            2,
            Batch::of(records.clone(), 1, 3, &[raw(2, 3, 0, 9, triangles, 4)], 48),
        ),
        (
            "more entries named than held",
            0,
            Batch::of(records.clone(), 9, 3, &[raw(0, 3, 0, 9, triangles, 4)], 48),
        ),
        (
            "a total past the primitives' binding",
            0,
            Batch::of(records.clone(), 1, 4, &[raw(0, 4, 0, 12, triangles, 4)], 40),
        ),
    ];
    let mut wrong = Vec::new();
    for (what, unknown, batch) in cases {
        let rule = batch.rule();
        let written = rule
            .as_chunks::<16>()
            .0
            .iter()
            .filter(|p| p[4..8] == UNKNOWN)
            .count();
        if written != unknown {
            wrong.push(format!(
                "{what}: the reference writes {written} unknown, not {unknown}"
            ));
        }
        wrong.extend(
            batch
                .differences(&m)
                .into_iter()
                .map(|d| format!("{what}: {d}")),
        );
    }
    let mut short = Batch::of(records, 1, 3, &[raw(0, 3, 0, 9, triangles, 4)], 48);
    short.draws.truncate(1);
    assert_eq!(short.kernel(&m), short.out, "no header: nothing written");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The settle on passes of render areas past and within the bound, 1, 2 and 4 samples and
/// several layers: where neither count is set the mask and the status are unchanged; where the
/// near-far or the unmeasured count is, every tile's bit of the pass is set, the last word's to
/// its last tile, the words past the pass's mask are unchanged, and the status reports
/// `captured_reach_unmeasured` beside the bits it held.
#[test]
fn the_settle_sets_every_tile_where_a_count_is_set() {
    let m = settle_typed_module().expect("module");
    let mut rng = Rng(0x5e77_1e00);
    let passes = [
        ([100, 70], 1, 1),
        ([300, 300], 4, 3),
        ([20000, 100], 1, 1),
        ([33, 31], 2, 5),
        ([1, 1], 4, 1),
        ([4096, 2048], 1, 2),
    ];
    for (area, samples, layers) in passes {
        let rows = if samples == 4 { 16 } else { 32 };
        let [w, h] = area.map(|a: u32| a.min(16384));
        let tiles = w.div_ceil(32) * h.div_ceil(rows) * layers;
        let mask = R::mask_words(area, samples, layers);
        assert_eq!(mask, tiles.div_ceil(32));
        for (near_far, unmeasured) in [(0, 0), (3, 0), (0, 1), (2, 5)] {
            let mut pass = words(&R::header(area, samples, layers, 9));
            pass[R::NEAR_FAR_WORD as usize] = near_far;
            pass[R::UNMEASURED_WORD as usize] = unmeasured;
            pass.extend(words(&R::state([0.0; 4], [0; 4], 0, 0, 0)));
            let before: Vec<u32> = (0..mask + 3).map(|_| rng.word() & rng.word()).collect();
            // Another writer's bit, kept.
            let mut status = words(&ExecutionStatus::initial());
            status[StatusWord::Status.index() as usize] = StatusBits::OVERFLOW.0;
            let mut outs = Vec::new();
            for order in [Order::Forward, Order::Reverse] {
                let res = Resources {
                    buffers: vec![
                        buffer(PASS, &pass),
                        buffer(TILES, &before),
                        buffer(STATUS, &status),
                    ],
                    push: Vec::new(),
                    spec: Default::default(),
                };
                let cfg = Config {
                    order,
                    workgroups: [(mask + 3).div_ceil(S::LOCAL_SIZE), 1, 1],
                    step_limit: 1 << 32,
                    ..Config::default()
                };
                outs.push(run(&m, &cfg, res).expect("run").buffers);
            }
            assert_eq!(outs[0], outs[1], "order independent");
            let after = words(&outs[0][1]);
            let fell = near_far != 0 || unmeasured != 0;
            let expect: Vec<u32> = before
                .iter()
                .enumerate()
                .map(|(i, &old)| {
                    let first = 32 * i as u32;
                    let bits = match tiles.saturating_sub(first) {
                        0 => 0,
                        rest if rest >= 32 => u32::MAX,
                        rest => (1 << rest) - 1,
                    };
                    match fell {
                        true => old | bits,
                        false => old,
                    }
                })
                .collect();
            let case = format!("{area:?} {samples} {layers} {near_far} {unmeasured}");
            assert_eq!(after, expect, "{case}");
            let report: [u8; ExecutionStatus::BYTES] = outs[0][2].clone().try_into().expect("32");
            let report = ExecutionStatus::decode(&report);
            assert_eq!(report.captured_reach_unmeasured, fell, "{case}");
            assert!(report.overflow, "{case}");
        }
    }
}

/// Every committed device (`validation/profiles/*.json`), derived with the default policy, by
/// file stem.
fn committed_devices() -> Vec<(String, Device)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("profiles") {
        let path = entry.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("read");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        let device = Device::new(desc, DevicePolicy::default()).expect("device");
        out.push((stem, device));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// On every committed device (those the tile reach module's device tests derive) both modules are
/// built, valid and within the profile: they need nothing past Vulkan 1.2.
#[test]
fn both_modules_are_valid_on_every_committed_device() {
    let devices = committed_devices();
    assert!(!devices.is_empty());
    for (name, device) in devices {
        let profile = device.profile();
        let assembly = capture_assembly_module(&device).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!assembly.spirv.is_empty(), "{name}");
        assert!(profile.check(&assembly.requirements).is_ok(), "{name}");
        assert_valid(assembly_typed_module().expect("module"), profile);
        let settle = capture_settle_module(&device).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!settle.spirv.is_empty(), "{name}");
        assert!(profile.check(&settle.requirements).is_ok(), "{name}");
        assert_valid(settle_typed_module().expect("module"), profile);
    }
}
