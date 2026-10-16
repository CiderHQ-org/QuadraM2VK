//! The tessellation scan: valid on every committed profile (L3, `G-val`), and, on the CPU
//! interpreter, each record's patch factors read at the control stage's addressing, its first
//! triangle, and the chunks' draws; an indirect draw's arguments as the consumer copies them into
//! the header; the records' reservation in the pool; and a draw the pool cannot hold, or whose
//! records pass a `u32`, draws nothing, sets `TESSELLATION_RECORDS_OVERFLOW` and names itself where
//! it is the command buffer's first records fault.
//!
//! The counts asserted here are the tessellator's for factors it draws as a regular grid (a quad
//! of every factor `n` under integer partitioning draws `2 · n²` triangles, by Euler's formula over
//! its `(n + 1)²` points), so they hold for any tessellator that places Metal's points.

use super::*;
use crate::bind::status::{
    ExecutionStatus, MeshRecordsFault, StatusReport, TessellationRecordsFault,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn kernel(domain: PatchDomain, step: Step, scale: bool) -> TessellationScanKernel {
    TessellationScanKernel {
        domain,
        partition_mode: Partition::Integer,
        output_winding_order: Winding::Clockwise,
        max_tessellation_factor: 16,
        factor_step_function: step,
        factor_scale_enabled: scale,
    }
}

/// `half` bits of the small integers the tests draw at.
fn half(n: u32) -> u16 {
    match n {
        0 => 0x0000,
        1 => 0x3c00,
        2 => 0x4000,
        3 => 0x4200,
        4 => 0x4400,
        5 => 0x4500,
        _ => panic!("no half for {n}"),
    }
}

/// A quad's factors, every one `n`.
fn quad(n: u32) -> [u16; 6] {
    [half(n); 6]
}

/// A quad's factors with every inside factor `n` and an edge factor 0, which culls it.
fn culled(n: u32) -> [u16; 6] {
    let mut f = quad(n);
    f[1] = half(0);
    f
}

/// A factor buffer of `len` bytes holding each `(byte, factors)`.
fn factor_bytes_of(len: usize, at: &[(usize, [u16; 6])]) -> Vec<u8> {
    let mut b = vec![0u8; len];
    for (byte, f) in at {
        for (i, h) in f.iter().enumerate() {
            b[byte + 2 * i..byte + 2 * i + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
    b
}

/// A draw's work buffer before the scan: the binder's header, `chunks` empty slots.
#[derive(Clone, Copy)]
struct Draw {
    patch_count: u32,
    instance_count: u32,
    patch_start: u32,
    base_instance: u32,
    pool_words: u32,
    factor_offset: u32,
    factor_stride: u32,
    scale: f32,
    chunks: u32,
}

impl Draw {
    fn of(patch_count: u32, instance_count: u32) -> Draw {
        Draw {
            patch_count,
            instance_count,
            patch_start: 0,
            base_instance: 0,
            pool_words: 0,
            factor_offset: 0,
            factor_stride: 0,
            scale: 1.0,
            chunks: 2,
        }
    }

    fn words(&self) -> Vec<u32> {
        let mut w = vec![0u32; work::words(self.chunks) as usize];
        for (at, v) in [
            (work::PATCH_COUNT, self.patch_count),
            (work::INSTANCE_COUNT, self.instance_count),
            (work::PATCH_START, self.patch_start),
            (work::BASE_INSTANCE, self.base_instance),
            (work::POOL_WORDS, self.pool_words),
            (work::DRAW_ID, 7),
            (work::FACTOR_OFFSET, self.factor_offset),
            (work::FACTOR_STRIDE, self.factor_stride),
            (work::SCALE, self.scale.to_bits()),
            (work::CHUNKS, self.chunks),
        ] {
            w[at as usize] = v;
        }
        w
    }

    /// The words of the draw's records binding: its own records, or the pool.
    fn record_words(&self) -> usize {
        if self.pool_words != 0 {
            self.pool_words as usize
        } else {
            (self.patch_count * self.instance_count * work::RECORD_WORDS).max(1) as usize
        }
    }
}

/// What a scan left: the work words, the records, the status report, and the pool's cursor.
struct Scanned {
    work: Vec<u32>,
    records: Vec<u32>,
    report: StatusReport,
    cursor: u32,
}

impl Scanned {
    /// The first triangle of each of the `n` records from the records' base.
    fn first_triangles(&self, n: usize) -> Vec<u32> {
        let base = self.work[work::RECORDS_BASE as usize] as usize;
        (0..n)
            .map(|r| {
                self.records[base + r * work::RECORD_WORDS as usize + work::FIRST_TRIANGLE as usize]
            })
            .collect()
    }

    /// Chunk `k`'s `VkDrawIndirectCommand`.
    fn chunk_draw(&self, k: u32) -> Vec<u32> {
        let at = (work::draw_offset(k) / 4) as usize;
        self.work[at..at + 4].to_vec()
    }

    fn chunk_first_record(&self, k: u32) -> u32 {
        self.work[(work::HEADER + work::SLOT_WORDS * k + work::SLOT_FIRST_RECORD) as usize]
    }
}

/// The status words before a scan: `bits`, the pool's cursor at `cursor`, and the fault words.
fn status_words(bits: u32, cursor: u32, fault: [u32; 3]) -> [u32; 8] {
    let mut status = [0u32; 8];
    status[StatusWord::Status.index() as usize] = bits;
    status[StatusWord::MeshRecordsCursor.index() as usize] = cursor;
    for (w, v) in [
        StatusWord::MeshFaultDraw,
        StatusWord::MeshFaultObjects,
        StatusWord::MeshFaultGroups,
    ]
    .into_iter()
    .zip(fault)
    {
        status[w.index() as usize] = v;
    }
    status
}

/// Run the scan of `kernel` over `draw` and the factor buffer `factors` (each bound whole) with
/// the status words `status`, in both invocation orders.
fn scan(kernel: &TessellationScanKernel, draw: &Draw, factors: &[u8], status: [u32; 8]) -> Scanned {
    let module = scan_typed_module(kernel).expect("scan");
    let mut ran = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let buffer = |slot, bytes| Buffer {
            slot: Some(slot),
            address: None,
            bytes,
        };
        let res = Resources {
            buffers: vec![
                buffer(WORK, bytes(&draw.words())),
                buffer(RECORDS, vec![0xEE; 4 * draw.record_words()]),
                buffer(STATUS, bytes(&status)),
                buffer(FACTORS, factors.to_vec()),
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            ..Config::default()
        };
        ran.push(run(&module, &cfg, res).expect("scan run").buffers);
    }
    assert_eq!(ran[0], ran[1], "order independent");
    let after = &ran[0];
    let status_bytes: [u8; 32] = after[2].as_slice().try_into().expect("status");
    let status = words(&after[2]);
    Scanned {
        work: words(&after[0]),
        records: words(&after[1]),
        report: ExecutionStatus::decode(&status_bytes),
        cursor: status[StatusWord::MeshRecordsCursor.index() as usize],
    }
}

/// The exclusive prefix of `counts`.
fn prefix(counts: &[u32]) -> Vec<u32> {
    counts
        .iter()
        .scan(0u32, |s, &n| {
            let at = *s;
            *s += n;
            Some(at)
        })
        .collect()
}

/// Kernels of both domains, three step functions, the scale on and off, and two partitions.
fn kernels() -> [TessellationScanKernel; 3] {
    [
        kernel(PatchDomain::Quad, Step::PerPatchAndPerInstance, false),
        kernel(PatchDomain::Triangle, Step::Constant, true),
        TessellationScanKernel {
            partition_mode: Partition::FractionalOdd,
            output_winding_order: Winding::CounterClockwise,
            max_tessellation_factor: 64,
            ..kernel(PatchDomain::Quad, Step::PerInstance, true)
        },
    ]
}

#[test]
fn the_scan_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        for k in &kernels() {
            assert_valid(scan_typed_module(k).expect("scan"), &p);
        }
    }
}

/// Each record `r = instance · patchCount + patch` reads the factors of the position
/// `patchStart + patch` and the instance `instance + baseInstance`, from the residual bind
/// offset; its first triangle is the triangles of the records before it; chunk 0 draws every
/// triangle's vertices from the first record that draws one; the binder's header stays, and the
/// draw's own records reserve nothing.
#[test]
fn each_record_reads_its_patches_factors_and_follows_the_records_before() {
    let k = kernel(PatchDomain::Quad, Step::PerPatchAndPerInstance, false);
    let mut draw = Draw::of(3, 2);
    draw.patch_start = 1;
    draw.base_instance = 3;
    draw.factor_offset = 2;
    // Four positions of 12 bytes per instance.
    draw.factor_stride = 48;
    let at = |position: usize, instance: usize| 2 + 12 * position + 48 * instance;
    // Record `(instance, patch)` reads position `1 + patch`, instance `3 + instance`.
    let factors = factor_bytes_of(
        2 + 48 * 5,
        &[
            (at(1, 3), culled(2)),
            (at(2, 3), quad(1)),
            (at(3, 3), quad(3)),
            (at(1, 4), quad(2)),
            (at(2, 4), quad(4)),
            (at(3, 4), quad(1)),
            // Positions and instances no record reads.
            (at(0, 3), quad(5)),
            (at(1, 2), quad(5)),
        ],
    );
    let s = scan(&k, &draw, &factors, status_words(0, 40, [0; 3]));
    let counts = [0, 2, 18, 8, 32, 2];
    assert_eq!(s.first_triangles(6), prefix(&counts));
    assert_eq!(s.work[work::RECORDS as usize], 6);
    assert_eq!(s.work[work::RECORDS_BASE as usize], 0);
    let total: u32 = counts.iter().sum();
    assert_eq!(s.chunk_draw(0), vec![3 * total, 1, 0, 0]);
    assert_eq!(s.chunk_first_record(0), 1);
    assert_eq!(s.chunk_draw(1), vec![0, 1, 0, 0]);
    assert_eq!(&s.work[..4], &[3, 2, 1, 3]);
    assert_eq!(s.cursor, 40);
    assert_eq!(s.report, StatusReport::default());
}

/// The step function decides which of the position and the instance step the factors' address,
/// as at the control stage.
#[test]
fn the_step_function_picks_the_factors_each_record_reads() {
    let mut draw = Draw::of(2, 2);
    draw.factor_stride = 24;
    let factors = factor_bytes_of(
        48,
        &[(0, quad(1)), (12, quad(2)), (24, quad(3)), (36, quad(4))],
    );
    for (step, counts) in [
        (Step::Constant, [2, 2, 2, 2]),
        (Step::PerPatch, [2, 8, 2, 8]),
        (Step::PerInstance, [2, 2, 18, 18]),
        (Step::PerPatchAndPerInstance, [2, 8, 18, 32]),
    ] {
        let k = kernel(PatchDomain::Quad, step, false);
        let s = scan(&k, &draw, &factors, status_words(0, 0, [0; 3]));
        assert_eq!(s.first_triangles(4), prefix(&counts), "{step:?}");
        let total: u32 = counts.iter().sum();
        assert_eq!(s.chunk_draw(0), vec![3 * total, 1, 0, 0], "{step:?}");
    }
}

/// The factor scale multiplies the factors where the pipeline enables it, and only there.
#[test]
fn the_factor_scale_applies_where_the_pipeline_enables_it() {
    let mut draw = Draw::of(1, 1);
    draw.scale = 2.0;
    let factors = factor_bytes_of(12, &[(0, quad(1))]);
    let k = kernel(PatchDomain::Quad, Step::Constant, true);
    let s = scan(&k, &draw, &factors, status_words(0, 0, [0; 3]));
    assert_eq!(s.chunk_draw(0), vec![3 * 8, 1, 0, 0]);
    let k = kernel(PatchDomain::Quad, Step::Constant, false);
    let s = scan(&k, &draw, &factors, status_words(0, 0, [0; 3]));
    assert_eq!(s.chunk_draw(0), vec![3 * 2, 1, 0, 0]);
}

/// More records than invocations: each run's records follow the runs before.
#[test]
fn runs_of_many_records_follow_the_runs_before() {
    let k = kernel(PatchDomain::Quad, Step::PerPatch, false);
    let n = 150;
    let draw = Draw::of(n, 1);
    let levels: Vec<u32> = (0..n).map(|p| [1, 2, 0, 3][p as usize % 4]).collect();
    let at: Vec<(usize, [u16; 6])> = levels
        .iter()
        .enumerate()
        .map(|(p, &l)| (12 * p, if l == 0 { culled(1) } else { quad(l) }))
        .collect();
    let factors = factor_bytes_of(12 * n as usize, &at);
    let s = scan(&k, &draw, &factors, status_words(0, 0, [0; 3]));
    let counts: Vec<u32> = levels.iter().map(|l| 2 * l * l).collect();
    assert_eq!(s.first_triangles(n as usize), prefix(&counts));
    let total: u32 = counts.iter().sum();
    assert_eq!(s.chunk_draw(0), vec![3 * total, 1, 0, 0]);
    assert_eq!(s.chunk_first_record(0), 0);
}

/// An indirect draw's arguments, which the consumer copies into the header's first four words,
/// draw as a direct draw's of the same words, its records in the pool after the cursor; arguments
/// of no patches draw nothing.
#[test]
fn an_indirect_scan_draws_the_arguments_copied_into_its_header() {
    let k = kernel(PatchDomain::Quad, Step::PerPatchAndPerInstance, false);
    let mut draw = Draw::of(2, 2);
    draw.patch_start = 1;
    draw.base_instance = 2;
    draw.pool_words = 1000;
    draw.factor_stride = 36;
    // Positions 1 and 2 of instances 2 and 3.
    let factors = factor_bytes_of(
        36 * 4,
        &[
            (12 + 72, quad(1)),
            (24 + 72, quad(2)),
            (12 + 108, quad(3)),
            (24 + 108, quad(1)),
        ],
    );
    let s = scan(&k, &draw, &factors, status_words(0, 16, [0; 3]));
    assert_eq!(&s.work[..4], &[2, 2, 1, 2]);
    assert_eq!(s.work[work::RECORDS as usize], 4);
    assert_eq!(s.work[work::RECORDS_BASE as usize], 16);
    assert_eq!(s.cursor, 16 + 4 * work::RECORD_WORDS);
    let counts = [2, 8, 18, 2];
    assert_eq!(s.first_triangles(4), prefix(&counts));
    assert_eq!(s.chunk_draw(0), vec![3 * 30, 1, 0, 0]);
    // No patches.
    let none = Draw {
        pool_words: 1000,
        ..Draw::of(0, 2)
    };
    let s = scan(&k, &none, &factors, status_words(0, 16, [0; 3]));
    assert_eq!(s.work[work::RECORDS as usize], 0);
    assert_eq!(s.cursor, 16);
    assert_eq!(s.chunk_draw(0), vec![0, 1, 0, 0]);
    assert_eq!(s.report, StatusReport::default());
}

/// A draw whose records the rest of the pool cannot hold, or whose records pass a `u32`, draws
/// nothing, keeps the cursor, sets the bit and names itself, with 0 object threadgroups.
#[test]
fn a_draw_the_pool_cannot_hold_draws_nothing_and_says_so() {
    let k = kernel(PatchDomain::Quad, Step::Constant, false);
    let mut draw = Draw::of(2, 1);
    draw.pool_words = 100;
    let factors = factor_bytes_of(12, &[(0, quad(1))]);
    let s = scan(&k, &draw, &factors, status_words(0, 90, [0; 3]));
    assert_eq!(s.work[work::RECORDS as usize], 0);
    assert_eq!(s.chunk_draw(0), vec![0, 1, 0, 0]);
    assert_eq!(s.cursor, 90);
    assert_eq!(
        s.report.tessellation_records_overflow,
        Some(TessellationRecordsFault {
            draw: 7,
            records: 2
        })
    );
    assert_eq!(s.report.mesh_records_overflow, None);
    // Exactly the room left: drawn.
    let s = scan(&k, &draw, &factors, status_words(0, 84, [0; 3]));
    assert_eq!(s.work[work::RECORDS as usize], 2);
    assert_eq!(s.chunk_draw(0), vec![3 * 4, 1, 0, 0]);
    assert_eq!(s.cursor, 100);
    assert_eq!(s.report, StatusReport::default());
    // Records past a `u32`: none, saturated.
    let wide = Draw {
        pool_words: 100,
        ..Draw::of(1 << 16, (1 << 16) + 1)
    };
    let s = scan(&k, &wide, &factors, status_words(0, 0, [0; 3]));
    assert_eq!(s.work[work::RECORDS as usize], 0);
    assert_eq!(s.cursor, 0);
    assert_eq!(
        s.report.tessellation_records_overflow,
        Some(TessellationRecordsFault {
            draw: 7,
            records: u32::MAX
        })
    );
}

/// The command buffer's first records fault keeps the fault words: a mesh draw's before a
/// tessellation draw's, and a tessellation draw's before a mesh draw's, each kind reading the
/// other's as all zeros.
#[test]
fn the_first_records_fault_of_either_kind_keeps_the_words() {
    let k = kernel(PatchDomain::Quad, Step::Constant, false);
    let mut draw = Draw::of(2, 1);
    draw.pool_words = 100;
    let factors = factor_bytes_of(12, &[(0, quad(1))]);
    let mesh_first = status_words(StatusBits::MESH_RECORDS_OVERFLOW.0, 100, [3, 2, 9]);
    let s = scan(&k, &draw, &factors, mesh_first);
    assert_eq!(
        s.report.mesh_records_overflow,
        Some(MeshRecordsFault {
            draw: 3,
            object_threadgroups: 2,
            mesh_threadgroups: 9,
        })
    );
    assert_eq!(
        s.report.tessellation_records_overflow,
        Some(TessellationRecordsFault::default())
    );
    // A tessellation draw's words, then a mesh draw's fault (which the mesh scan leaves).
    let tessellation_first = status_words(StatusBits::RECORDS_FAULTS.0, 100, [5, 0, 4]);
    let s = scan(&k, &draw, &factors, tessellation_first);
    assert_eq!(
        s.report.tessellation_records_overflow,
        Some(TessellationRecordsFault {
            draw: 5,
            records: 4
        })
    );
    assert_eq!(
        s.report.mesh_records_overflow,
        Some(MeshRecordsFault::default())
    );
}
