//! The index answers what the scans it replaced answered: the first member in member order whose
//! bytes hold an offset, through nested layouts, for disjoint, unsorted, overlapping, empty, and
//! array members alike.

use super::*;
use crate::air::abi::{MemberKind, UserKey};
use crate::air::error::DiagName;
use crate::pipeline::testing::meter;

fn member(offset: u32, size: u32, array_len: u32, layout: Option<u32>) -> Member {
    Member {
        offset,
        size,
        array_len,
        layout: layout.map(StructRef),
        kind: MemberKind::Plain,
        render_target: None,
        raster_order_group: None,
        type_name: DiagName::new("t"),
        name: DiagName::new("m"),
        key: UserKey(b"m".as_slice().into()),
    }
}

/// The scan the index replaced: the first member whose bytes hold `off`.
fn scanned_index(info: &StructInfo, off: u64) -> Option<usize> {
    info.members
        .iter()
        .enumerate()
        .find(|(_, mem)| {
            let (start, end) = span(mem);
            start <= off && off < end
        })
        .map(|(i, _)| i)
}

/// The scan's `member_place_at`, through nested layouts.
fn scanned_place(
    structs: &[StructInfo],
    mut s: StructRef,
    mut off: u64,
) -> Option<((usize, usize), u32)> {
    for _ in 0..LAYOUT_NESTING {
        let info = structs.get(s.0 as usize)?;
        let index = scanned_index(info, off)?;
        let mem = info.members.get(index)?;
        let within = off - u64::from(mem.offset);
        let within = if mem.size > 0 {
            within % u64::from(mem.size)
        } else {
            within
        };
        match mem.layout {
            Some(inner) => {
                s = inner;
                off = within;
            }
            None => {
                let element = u64::from(mem.size);
                return (within == 0 || (element > 0 && within.is_multiple_of(element)))
                    .then_some(((s.0 as usize, index), mem.offset));
            }
        }
    }
    None
}

/// A xorshift generator: the layouts are random but fixed.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n.max(1)
    }
}

/// Random layouts, members in any order, overlapping or not, some empty, some arrays, the
/// first layouts nesting later ones: every offset finds the member the scan found.
#[test]
fn the_member_at_every_offset_is_the_one_the_scan_found() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for round in 0..64 {
        let layouts = 4u32;
        let mut structs = Vec::new();
        for l in 0..layouts {
            let count = 1 + rng.below(12);
            let mut members = Vec::new();
            let mut at = 0u32;
            for _ in 0..count {
                // Disjoint and ascending in even rounds; anywhere, overlapping, in odd ones.
                let offset = if round % 2 == 0 {
                    at + rng.below(3) as u32 * 4
                } else {
                    rng.below(64) as u32
                };
                let size = rng.below(5) as u32 * 4;
                let array_len = if rng.below(3) == 0 {
                    rng.below(4) as u32
                } else {
                    0
                };
                let layout = (l + 1 < layouts && rng.below(4) == 0)
                    .then(|| l + 1 + rng.below(u64::from(layouts - l - 1)) as u32);
                at = offset + size * array_len.max(1);
                members.push(member(offset, size, array_len, layout));
            }
            if round % 4 == 1 {
                members.reverse();
            }
            structs.push(StructInfo { members });
        }
        let mut m = meter();
        let index = AbiIndex::layouts(&structs, &mut m).expect("index");
        for (s, info) in structs.iter().enumerate() {
            let end = info.members.iter().map(|x| span(x).1).max().unwrap_or(0);
            for off in 0..end + 8 {
                let r = StructRef(s as u32);
                assert_eq!(
                    index.member_index(r, off),
                    scanned_index(info, off),
                    "round {round}, layout {s}, offset {off}"
                );
                assert_eq!(
                    index
                        .member_place_at(&structs, r, off)
                        .map(|(place, mem)| (place, mem.offset)),
                    scanned_place(&structs, r, off),
                    "round {round}, layout {s}, offset {off}"
                );
            }
        }
        assert_eq!(index.member_index(StructRef(layouts), 0), None);
    }
}

/// Building the index costs a work unit per member and two per member boundary, and leaves
/// charged only its tables: `n` members cost under `4n` units, whatever their order.
#[test]
fn indexing_a_layout_is_linear_in_its_members() {
    let n = 1u32 << 14;
    let members = (0..n).rev().map(|i| member(8 * i, 8, 0, None)).collect();
    let structs = vec![StructInfo { members }];
    let mut m = meter();
    let index = AbiIndex::layouts(&structs, &mut m).expect("index");
    let units = m.work();
    assert!(
        (u64::from(n)..4 * u64::from(n)).contains(&units),
        "{units} units for {n} members"
    );
    // The layout starts (two) and two pieces per member, as reserved.
    assert_eq!(m.ledger(), (2 * 4 + 2 * n as usize * 16) as u64);
    assert_eq!(
        index.member_index(StructRef(0), 8 * 77 + 3),
        Some((n - 78) as usize)
    );
    assert_eq!(index.member_index(StructRef(0), 8 * u64::from(n)), None);
}
