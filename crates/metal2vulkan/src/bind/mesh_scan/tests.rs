//! The mesh scan: valid on every committed profile (L3, `G-val`), and, on the CPU interpreter,
//! each object threadgroup's first mesh threadgroup, the draw's `T`, its reservation in the
//! records pool, its dispatch folded into rows and its draw; an indirect draw's grid from Metal's
//! arguments, held to Metal's limit; and a draw the pool cannot hold draws nothing, sets
//! `MESH_RECORDS_OVERFLOW` and names itself, the first such draw only.

use super::*;
use crate::bind::status::{ExecutionStatus, MeshRecordsFault, StatusReport};
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

/// A draw's work buffer before the scan: `grids` of the object threadgroups, slots of
/// `slot_words`, the records of `group_words` words and `group_vertices` vertices per mesh
/// threadgroup, a pool of `pool_words`, draw number 7, rows of `row`, and Metal's arguments at
/// word `arguments - 1` where it is not 0.
struct Draw {
    grids: Vec<[u32; 3]>,
    slot_words: u32,
    group_words: u32,
    group_vertices: u32,
    pool_words: u32,
    row: u32,
    arguments: u32,
}

impl Draw {
    fn of(grids: &[[u32; 3]]) -> Draw {
        Draw {
            grids: grids.to_vec(),
            slot_words: 6,
            group_words: 10,
            group_vertices: 3,
            pool_words: 1000,
            row: 4,
            arguments: 0,
        }
    }

    fn words(&self) -> Vec<u32> {
        let o = self.grids.len() as u32;
        let mut w = vec![0u32; work::HEADER as usize];
        w[work::OBJECT_GROUPS as usize] = o;
        w[work::SLOT_WORDS as usize] = self.slot_words;
        w[work::GROUP_WORDS as usize] = self.group_words;
        w[work::GROUP_VERTICES as usize] = self.group_vertices;
        w[work::POOL_WORDS as usize] = self.pool_words;
        w[work::DRAW_ID as usize] = 7;
        w[work::ROW as usize] = self.row;
        w[work::ARGUMENTS as usize] = self.arguments;
        for g in &self.grids {
            let mut slot = vec![0xAAAA_AAAA; self.slot_words as usize];
            slot[..3].copy_from_slice(g);
            slot[3] = 0;
            w.extend(slot);
        }
        // The offsets, which the scan writes.
        w.extend(std::iter::repeat_n(0xEEEE_EEEE, o as usize + 1));
        w
    }
}

/// What a scan left: the work words, the status report, and the pool's cursor.
struct Scanned {
    work: Vec<u32>,
    report: StatusReport,
    cursor: u32,
}

/// Run the scan of `draw` over `arguments` (bound whole) with the pool's cursor at `cursor` and
/// the status bits `bits`, in both invocation orders.
fn scan(draw: &Draw, arguments: &[u32], cursor: u32, bits: u32) -> Scanned {
    let module = scan_typed_module().expect("scan");
    let mut status = [0u32; 8];
    status[StatusWord::Status.index() as usize] = bits;
    status[StatusWord::MeshRecordsCursor.index() as usize] = cursor;
    let mut ran = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(WORK),
                    address: None,
                    bytes: bytes(&draw.words()),
                },
                Buffer {
                    slot: Some(ARGUMENTS),
                    address: None,
                    bytes: bytes(arguments),
                },
                Buffer {
                    slot: Some(STATUS),
                    address: None,
                    bytes: bytes(&status),
                },
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
        report: ExecutionStatus::decode(&status_bytes),
        cursor: status[StatusWord::MeshRecordsCursor.index() as usize],
    }
}

/// The offsets after `draw`'s slots.
fn offsets(draw: &Draw, work: &[u32]) -> Vec<u32> {
    let at = (work::HEADER + draw.grids.len() as u32 * draw.slot_words) as usize;
    work[at..].to_vec()
}

fn header(work: &[u32], at: u32, n: usize) -> Vec<u32> {
    work[at as usize..at as usize + n].to_vec()
}

#[test]
fn the_scan_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        assert_valid(scan_typed_module().expect("scan"), &p);
    }
}

/// Each object threadgroup's first mesh threadgroup in object order, and `T`; the records
/// reserved after the cursor; `T` folded into rows of `ROW`; `T · NP · k` vertices. A grid with
/// an axis of 0 launches none, and one past Metal's limit none (the object function holds it so;
/// the scan holds it again).
#[test]
fn the_scan_places_each_object_threadgroups_mesh_threadgroups() {
    let draw = Draw::of(&[[2, 1, 1], [0, 5, 1], [3, 2, 1], [1, 1, 1], [1 << 20, 2, 1]]);
    let s = scan(&draw, &[0], 40, 0);
    assert_eq!(offsets(&draw, &s.work), vec![0, 2, 2, 8, 9, 9]);
    assert_eq!(s.work[work::MESH_GROUPS as usize], 9);
    assert_eq!(s.work[work::RECORDS_BASE as usize], 40);
    assert_eq!(s.cursor, 40 + 9 * 10);
    assert_eq!(header(&s.work, work::DISPATCH, 3), vec![4, 3, 1]);
    assert_eq!(header(&s.work, work::DRAW, 4), vec![27, 1, 0, 0]);
    assert_eq!(s.report, StatusReport::default());
    // The slots are the object function's: the scan leaves them.
    assert_eq!(&s.work[work::HEADER as usize..][..3], &[2, 1, 1]);
}

/// More object threadgroups than invocations: every run of them is placed after the runs
/// before.
#[test]
fn the_scan_places_runs_of_many_object_threadgroups() {
    let grids: Vec<[u32; 3]> = (0..200u32).map(|i| [i % 3, 1, 1]).collect();
    let mut draw = Draw::of(&grids);
    draw.pool_words = 100_000;
    draw.row = 65535;
    let s = scan(&draw, &[0], 0, 0);
    let mut want = Vec::new();
    let mut running = 0;
    for g in &grids {
        want.push(running);
        running += g[0];
    }
    want.push(running);
    assert_eq!(offsets(&draw, &s.work), want);
    assert_eq!(s.work[work::MESH_GROUPS as usize], running);
    assert_eq!(header(&s.work, work::DISPATCH, 3), vec![running, 1, 1]);
}

/// An indirect draw's grid is Metal's arguments at their word, into slot 0; one past Metal's
/// limit launches nothing; arguments past the bound range read 0.
#[test]
fn an_indirect_scan_reads_metals_arguments() {
    let mut draw = Draw::of(&[[0, 0, 0]]);
    draw.arguments = 1 + 2;
    let s = scan(&draw, &[9, 9, 3, 2, 2], 0, 0);
    assert_eq!(&s.work[work::HEADER as usize..][..3], &[3, 2, 2]);
    assert_eq!(offsets(&draw, &s.work), vec![0, 12]);
    assert_eq!(s.work[work::MESH_GROUPS as usize], 12);
    assert_eq!(header(&s.work, work::DRAW, 4), vec![36, 1, 0, 0]);
    let s = scan(&draw, &[9, 9, 1 << 20, 2, 1], 0, 0);
    assert_eq!(&s.work[work::HEADER as usize..][..3], &[0, 0, 0]);
    assert_eq!(s.work[work::MESH_GROUPS as usize], 0);
    assert_eq!(header(&s.work, work::DISPATCH, 3), vec![0, 0, 1]);
    let s = scan(&draw, &[9, 9, 3, 2], 0, 0);
    assert_eq!(s.work[work::MESH_GROUPS as usize], 0);
}

/// A draw whose records the rest of the pool cannot hold draws nothing, keeps the cursor, sets
/// the bit and names itself; a later fault leaves the first one's words.
#[test]
fn a_draw_the_pool_cannot_hold_draws_nothing_and_says_so() {
    let draw = Draw::of(&[[4, 1, 1], [3, 1, 1]]);
    let s = scan(&draw, &[0], 940, 0);
    assert_eq!(s.work[work::MESH_GROUPS as usize], 0);
    assert_eq!(header(&s.work, work::DISPATCH, 3), vec![0, 0, 1]);
    assert_eq!(header(&s.work, work::DRAW, 4), vec![0, 1, 0, 0]);
    assert_eq!(s.cursor, 940);
    assert_eq!(
        s.report.mesh_records_overflow,
        Some(MeshRecordsFault {
            draw: 7,
            object_threadgroups: 2,
            mesh_threadgroups: 7,
        })
    );
    // Exactly the room left: drawn.
    let s = scan(&draw, &[0], 930, 0);
    assert_eq!(s.work[work::MESH_GROUPS as usize], 7);
    assert_eq!(s.cursor, 1000);
    assert_eq!(s.report.mesh_records_overflow, None);
    // A later fault leaves the first one's words (here all zero).
    let s = scan(&draw, &[0], 940, StatusBits::MESH_RECORDS_OVERFLOW.0);
    assert_eq!(
        s.report.mesh_records_overflow,
        Some(MeshRecordsFault::default())
    );
    // Vertices past a `u32` draw nothing either.
    let mut wide = Draw::of(&[[1 << 10, 1 << 10, 1]]);
    wide.pool_words = u32::MAX;
    wide.group_words = 1;
    wide.group_vertices = 1 << 12;
    let s = scan(&wide, &[0], 0, 0);
    assert_eq!(s.work[work::MESH_GROUPS as usize], 0);
    assert_eq!(
        s.report.mesh_records_overflow.map(|f| f.mesh_threadgroups),
        Some(1 << 20)
    );
}
