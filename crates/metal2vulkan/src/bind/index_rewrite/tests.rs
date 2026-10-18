//! The index rewrite pass: its modules valid on every committed profile (L3, `G-val`), and, on
//! the CPU interpreter, the list it writes is the strip's pieces between the restart index as
//! Metal draws them: each piece's triangles in Vulkan's strip order with the
//! parity counted from the piece's start, or its segments, in order; positions past the
//! buffer's whole indices read 0, restart included; the draw's arguments carry Metal's
//! instances and bases; a pooled draw reserves whole words from the pool's counter, and a draw
//! its output cannot hold draws nothing and sets `INDEX_POOL_OVERFLOW`.

use super::*;
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// The pool's index bytes in these tests.
const POOL_BYTES: u32 = 4096;
/// The storage buffer offset alignment of these tests.
const ALIGNMENT: u64 = 16;

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

/// Index buffer bytes of `indices`, 4 bytes each where `wide`, else 2.
fn index_bytes(indices: &[u32], wide: bool) -> Vec<u8> {
    indices
        .iter()
        .flat_map(|&i| {
            if wide {
                i.to_le_bytes().to_vec()
            } else {
                (i as u16).to_le_bytes().to_vec()
            }
        })
        .collect()
}

/// Metal's primitives of a strip of `program` over `values` (each position's index, as the
/// fetch reads it) restarting at `restart` alone: each piece's, in order.
fn metal(program: RewriteProgram, values: &[u32], restart: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut piece: Vec<u32> = Vec::new();
    for &x in values {
        if x == restart {
            piece.clear();
            continue;
        }
        piece.push(x);
        let n = piece.len();
        match program {
            RewriteProgram::LineStrip if n >= 2 => out.extend([piece[n - 2], piece[n - 1]]),
            RewriteProgram::TriangleStrip if n >= 3 => {
                let (a, b, c) = (piece[n - 3], piece[n - 2], piece[n - 1]);
                if (n - 3).is_multiple_of(2) {
                    out.extend([a, b, c]);
                } else {
                    out.extend([a, c, b]);
                }
            }
            _ => {}
        }
    }
    out
}

/// Each position's index: position `start + q` at its index where it lies among the buffer's
/// `indices`, else 0.
fn fetched(indices: &[u32], start: u32, count: u32) -> Vec<u32> {
    (0..u64::from(count))
        .map(|q| {
            let p = u64::from(start) + q;
            indices.get(p as usize).copied().unwrap_or(0)
        })
        .collect()
}

/// One draw.
#[derive(Clone)]
struct Draw {
    program: RewriteProgram,
    wide: bool,
    /// The index buffer's indices from its byte `offset`, after `offset` bytes of 0xAB.
    indices: Vec<u32>,
    offset: u64,
    restart: u32,
    /// Metal's count, instances, first index, base vertex and base instance.
    arguments: [u32; 5],
    /// Indirect: the arguments' offset in the indirect buffer; direct: `None`.
    indirect: Option<u64>,
    /// Pooled: the pool's counter before the pass; owned: `None`.
    counter: Option<u32>,
    /// The workgroups of the count and write dispatches, where not the pass's own.
    groups: Option<u32>,
}

/// What a pass left: the words, the output's indices (from the pool's index region where
/// pooled), the pool's counter, and the status report.
struct Ran {
    words: Vec<u32>,
    output: Vec<u32>,
    counter: Option<u32>,
    report: StatusReport,
}

impl Draw {
    fn strip(program: RewriteProgram, wide: bool, indices: &[u32], restart: u32) -> Draw {
        Draw {
            program,
            wide,
            indices: indices.to_vec(),
            offset: 0,
            restart,
            arguments: [indices.len() as u32, 1, 0, 0, 0],
            indirect: None,
            counter: None,
            groups: None,
        }
    }

    fn size(&self) -> u64 {
        if self.wide {
            4
        } else {
            2
        }
    }

    /// The pass's run: the indices bound as its range names, the arguments, and the output.
    fn run(&self) -> Ran {
        let size = self.size();
        let mut buffer = vec![0xAB; self.offset as usize];
        buffer.extend(index_bytes(&self.indices, self.wide));
        let end = buffer.len() as u64;
        // The `VkBuffer` runs on to a multiple of 16 bytes (the padding), of bytes no position may
        // read as an index.
        buffer.resize(buffer.len().next_multiple_of(16).max(16), 0xCC);
        let [count, instances, _, base, base_instance] = self.arguments;
        let source = match self.indirect {
            None => RewriteSource::Direct {
                count,
                instances,
                base_vertex: base as i32,
                base_instance,
            },
            Some(at) => RewriteSource::Indirect {
                arguments: IndirectRef::new(0, at),
            },
        };
        let capacity = self.program.capacity(u64::from(count)) as u32;
        let output = match self.counter {
            None => RewriteOutput::Owned { capacity },
            Some(_) => RewriteOutput::Pooled {
                pool_bytes: POOL_BYTES,
            },
        };
        let pass = IndexRewritePass::new(
            source,
            self.restart,
            (size, self.offset, end),
            output,
            ALIGNMENT,
            u32::MAX,
        )
        .expect("pass");
        let mut pass = pass;
        if let Some(groups) = self.groups {
            pass.groups = groups;
            pass.parameters[Parameter::Groups as usize] = groups;
        }
        let mut rewrite = pass.parameters.to_vec();
        rewrite.resize((IndexRewritePass::BYTES / 4) as usize, 0xEEEE_EEEE);
        // The indirect buffer: the arguments at their offset, bound as the pass names.
        let arguments = match (self.indirect, pass.arguments_range) {
            (Some(at), Some(range)) => {
                let mut b = vec![0xCDu8; (at + 20) as usize];
                b[at as usize..at as usize + 20].copy_from_slice(&bytes(&self.arguments));
                b[range.offset as usize..(range.offset + range.range) as usize].to_vec()
            }
            (None, None) => vec![0xCC; 4],
            _ => panic!("an indirect draw's range"),
        };
        // Exactly the range the pass names, as Vulkan binds it: a shader reads the words that lie
        // wholly inside it.
        let indices = match pass.indices_range {
            Some(range) => {
                buffer[range.offset as usize..(range.offset + range.range) as usize].to_vec()
            }
            None => vec![0xCC; 4],
        };
        let header = (IndexRewritePass::POOL_HEADER_BYTES / 4) as usize;
        let out = match self.counter {
            None => vec![0xDDDD_DDDD; capacity.max(1) as usize],
            Some(counter) => {
                let mut pool = vec![0xDDDD_DDDD; header + (POOL_BYTES / 4) as usize];
                pool[0] = counter;
                pool
            }
        };
        let modules = RewritePhase::ALL.map(|phase| {
            (
                phase,
                rewrite_typed_module(self.program, phase).expect("module"),
            )
        });
        let mut ran = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let mut buffers = vec![
                bytes(&rewrite),
                arguments.clone(),
                indices.clone(),
                bytes(&out),
                ExecutionStatus::initial().to_vec(),
            ];
            // The three dispatches in order, each over the buffers the one before left.
            for (phase, module) in &modules {
                let slots = [WORDS, ARGUMENTS, INDICES, OUTPUT, STATUS];
                let res = Resources {
                    buffers: slots
                        .into_iter()
                        .zip(buffers)
                        .map(|(slot, bytes)| Buffer {
                            slot: Some(slot),
                            address: None,
                            bytes,
                        })
                        .collect(),
                    push: Vec::new(),
                    spec: Default::default(),
                };
                let cfg = Config {
                    order,
                    workgroups: [phase.workgroups(&pass), 1, 1],
                    ..Config::default()
                };
                buffers = run(module, &cfg, res).expect("rewrite run").buffers;
            }
            ran.push(buffers);
        }
        assert_eq!(ran[0], ran[1], "order independent");
        let after = &ran[0];
        let status: [u8; 32] = after[4].as_slice().try_into().expect("status");
        let out = words(&after[3]);
        let (output, counter) = match self.counter {
            None => (out, None),
            Some(_) => (out[header..].to_vec(), Some(out[0])),
        };
        Ran {
            words: words(&after[0]),
            output,
            counter,
            report: ExecutionStatus::decode(&status),
        }
    }

    /// The indices the draw draws: its command's from the output, every one inside it.
    fn drawn(&self) -> (Vec<u32>, [u32; 5], Ran) {
        let r = self.run();
        let at = (IndexRewritePass::DRAW_OFFSET / 4) as usize;
        let command: [u32; 5] = r.words[at..at + 5].try_into().expect("command");
        let first = command[2] as usize;
        let n = command[0] as usize;
        assert!(
            first + n <= r.output.len(),
            "the draw reads past its output"
        );
        let list = r.output[first..first + n].to_vec();
        (list, command, r)
    }
}

#[test]
fn every_program_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        for program in [RewriteProgram::LineStrip, RewriteProgram::TriangleStrip] {
            for phase in RewritePhase::ALL {
                assert_valid(rewrite_typed_module(program, phase).expect("module"), &p);
            }
        }
    }
}

/// The pass binds the arguments and the whole indices from aligned offsets, and states the
/// residuals, the whole indices, the index size, `k'` and the output.
#[test]
fn the_pass_binds_aligned_ranges_and_states_its_parameters() {
    let pass = IndexRewritePass::new(
        RewriteSource::Indirect {
            arguments: IndirectRef::new(0, 68),
        },
        3,
        (4, 8, 30),
        RewriteOutput::Pooled { pool_bytes: 64 },
        64,
        1 << 27,
    )
    .expect("pass");
    assert_eq!(
        pass.arguments_range,
        Some(BufferRange {
            offset: 64,
            range: 24
        })
    );
    assert_eq!(
        pass.indices_range,
        Some(BufferRange {
            offset: 0,
            range: 28
        })
    );
    assert_eq!(pass.parameters[..8], [2, 8, 5, 1, 3, 64, 1, 1]);
    // The indices' range ends at a whole word: nine 16-bit indices end at byte 18, and ten from
    // byte 2 at byte 22.
    let indices_range = |offset, end| {
        IndexRewritePass::new(
            RewriteSource::Indirect {
                arguments: IndirectRef::new(0, 0),
            },
            3,
            (2, offset, end),
            RewriteOutput::Pooled { pool_bytes: 64 },
            64,
            1 << 27,
        )
        .expect("pass")
        .indices_range
    };
    for (offset, end, range) in [(0, 18, 20), (2, 22, 24), (2, 20, 20), (66, 82, 20)] {
        assert_eq!(
            indices_range(offset, end),
            Some(BufferRange {
                offset: offset - offset % 64,
                range
            }),
            "from {offset} to {end}"
        );
    }
    assert_eq!(pass.groups, 1);
    assert_eq!(pass.parameters[8..], [0; 8]);
    let direct = IndexRewritePass::new(
        RewriteSource::Direct {
            count: 7,
            instances: 2,
            base_vertex: -1,
            base_instance: 5,
        },
        0xFFFF,
        (2, 30, 30),
        RewriteOutput::Owned { capacity: 15 },
        64,
        1 << 27,
    )
    .expect("direct");
    assert_eq!(direct.arguments_range, None);
    assert_eq!(direct.indices_range, None);
    assert_eq!(direct.parameters[..8], [0, 0, 0, 0, 0xFFFF, 15, 0, 1]);
    assert_eq!(direct.parameters[8..13], [7, 2, 0, u32::MAX, 5]);
    let refused = |source, size, offset, output| {
        IndexRewritePass::new(source, 0, (size, offset, 64), output, 64, 1 << 27).is_none()
    };
    let owned = RewriteOutput::Owned { capacity: 1 };
    let at = |offset| RewriteSource::Indirect {
        arguments: IndirectRef::new(0, offset),
    };
    assert!(refused(at(2), 4, 0, owned), "arguments' offset");
    assert!(refused(at(0), 3, 0, owned), "index size");
    assert!(refused(at(0), 4, 2, owned), "index offset");
    assert!(refused(
        at(0),
        4,
        0,
        RewriteOutput::Pooled { pool_bytes: 6 }
    ));
}

/// Probe 5's strips, drawn directly: each restarts at `k'` alone, the all-ones index drawn as a
/// vertex where it is not `k'`; 16- and 32-bit alike.
#[test]
fn a_strip_restarts_at_its_index_alone() {
    for program in [RewriteProgram::LineStrip, RewriteProgram::TriangleStrip] {
        for (wide, indices, restart) in [
            (false, vec![1, 2, 3, 0, 4, 5, 6, 0xFFFF, 7, 8, 9], 0),
            (false, vec![0, 1, 2, 3, 4, 5, 6, 0xFFFF, 7, 8, 9], 3),
            (false, vec![1, 2, 3, 0, 4, 5, 6, 0xFFFF, 7, 8, 9], 0xFFFF),
            (
                true,
                vec![1, 2, 3, 0, 4, 5, 6, u32::MAX, 7, 8, 0xFFFF, 9, 10],
                0,
            ),
            (
                true,
                vec![0, 1, 2, 3, 4, 5, 6, u32::MAX, 7, 8, 0xFFFF, 9, 10],
                0xFFFF,
            ),
            (true, vec![3, 3, 0, 1, 3, 2, 4, 5, 3], 3),
            (true, vec![3, 3, 3], 3),
            (true, vec![1, 2], 3),
            (true, vec![], 3),
        ] {
            let draw = Draw::strip(program, wide, &indices, restart);
            let (list, command, r) = draw.drawn();
            let expected = metal(program, &indices, restart);
            assert_eq!(
                list, expected,
                "{program:?} {wide} {indices:?} k' {restart}"
            );
            assert_eq!(command[1..], [1, 0, 0, 0], "{indices:?}");
            assert_eq!(
                r.words[Output::Primitives as usize] as usize,
                expected.len() / program.vertices() as usize
            );
            assert!(!r.report.index_pool_overflow);
        }
    }
}

/// A strip longer than a workgroup's invocations, over one workgroup and over several: its runs'
/// pieces cross between runs and between groups, and each run's first triangles take their
/// piece's parity from the runs before, its group's and the groups' before it.
#[test]
fn a_strip_of_many_runs_keeps_each_piece_s_parity() {
    // A deterministic mix: restarts at 0 here and there, long pieces and short ones.
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for (len, rate) in [(600, 7), (1000, 97), (777, 3), (1300, 1000)] {
        let indices: Vec<u32> = (0..len)
            .map(|_| {
                let r = next();
                if r % rate == 0 {
                    0
                } else {
                    1 + r % 50
                }
            })
            .collect();
        for program in [RewriteProgram::LineStrip, RewriteProgram::TriangleStrip] {
            for (wide, groups) in [
                (false, None),
                (true, None),
                (false, Some(3)),
                (true, Some(2)),
            ] {
                let draw = Draw {
                    groups,
                    ..Draw::strip(program, wide, &indices, 0)
                };
                let (list, _, _) = draw.drawn();
                assert_eq!(
                    list,
                    metal(program, &indices, 0),
                    "{program:?} {wide} {len} {rate} {groups:?}"
                );
            }
        }
    }
}

/// The count and write dispatches take a workgroup per 4096 positions the walk may take, from 1
/// to 256: a direct draw's count bounds the walk, and an indirect draw's by the buffer's whole
/// indices and the program's vertices past them.
#[test]
fn the_groups_follow_the_walk() {
    let groups = |source, inside: u64| {
        let output = RewriteOutput::Pooled { pool_bytes: 64 };
        IndexRewritePass::new(source, 0, (4, 0, inside * 4), output, 64, u32::MAX)
            .expect("pass")
            .groups
    };
    let direct = |count| RewriteSource::Direct {
        count,
        instances: 1,
        base_vertex: 0,
        base_instance: 0,
    };
    let indirect = RewriteSource::Indirect {
        arguments: IndirectRef::new(0, 0),
    };
    assert_eq!(groups(direct(10), 10), 1);
    assert_eq!(groups(direct(4096), 1 << 20), 1);
    assert_eq!(groups(direct(4097), 1 << 20), 2);
    assert_eq!(groups(direct(1_000_000), 1_000_000), 245);
    assert_eq!(groups(direct(1_000_000), 0), 1);
    assert_eq!(groups(direct(u32::MAX), 1 << 29), 256);
    assert_eq!(groups(indirect, 4093), 1);
    assert_eq!(groups(indirect, 4094), 2);
    assert_eq!(groups(indirect, 2_000_000), 256);
}

/// An indirect draw reads Metal's arguments (its first index, instances and bases), reserves
/// whole words from the pool's counter, and draws from there; a draw the pool cannot hold draws
/// nothing and sets the overflow bit.
#[test]
fn an_indirect_draw_reserves_its_list_in_the_pool() {
    let indices: Vec<u32> = vec![9, 9, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8];
    for program in [RewriteProgram::LineStrip, RewriteProgram::TriangleStrip] {
        let draw = Draw {
            arguments: [8, 3, 2, 0xFFFF_FFFE, 4],
            indirect: Some(36),
            counter: Some(40),
            ..Draw::strip(program, false, &indices, 0)
        };
        let (list, command, r) = draw.drawn();
        let expected = metal(program, &fetched(&indices, 2, 8), 0);
        assert_eq!(list, expected, "{program:?}");
        assert_eq!(command[1..], [3, 10, 0xFFFF_FFFE, 4], "{program:?}");
        assert_eq!(r.counter, Some(40 + 4 * expected.len() as u32));
        assert!(!r.report.index_pool_overflow);
        // No room: nothing drawn, the counter kept, the bit set.
        let full = Draw {
            counter: Some(POOL_BYTES - 4),
            ..draw.clone()
        };
        let (list, command, r) = full.drawn();
        assert!(list.is_empty());
        assert_eq!(command[0], 0);
        assert_eq!(r.counter, Some(POOL_BYTES - 4));
        assert!(r.report.index_pool_overflow);
    }
}

/// Positions past the buffer's whole indices read 0: they restart where `k'` is 0, and are
/// vertex 0 otherwise, the primitives past the walk written without one; an index offset inside
/// a word, and a partial last index, read as Metal's fetch does.
#[test]
fn positions_past_the_buffer_read_index_0() {
    let indices = vec![5, 6, 7, 8, 9];
    for program in [RewriteProgram::LineStrip, RewriteProgram::TriangleStrip] {
        for (restart, count) in [(0, 9), (3, 9), (3, 40), (0, 40), (3, 2)] {
            for wide in [false, true] {
                let draw = Draw {
                    arguments: [count, 1, 0, 0, 0],
                    offset: if wide { 4 } else { 6 },
                    ..Draw::strip(program, wide, &indices, restart)
                };
                let (list, _, _) = draw.drawn();
                let expected = metal(program, &fetched(&indices, 0, count), restart);
                assert_eq!(list, expected, "{program:?} {wide} k' {restart} {count}");
            }
        }
        // An indirect draw from past the buffer reads 0 throughout.
        let draw = Draw {
            arguments: [6, 1, 9, 0, 0],
            indirect: Some(0),
            counter: Some(0),
            ..Draw::strip(program, true, &indices, 4)
        };
        let (list, _, _) = draw.drawn();
        assert_eq!(
            list,
            metal(program, &[0; 6], 4),
            "{program:?} from past the buffer"
        );
    }
}

/// The last whole index of a word the buffer's indices end inside is read, not 0: an odd count
/// of 16-bit indices from a multiple of 4 bytes, or an even count from 2 past one, direct and
/// indirect (the consumer's nine 16-bit indices restarting at 3, from an 18- or a 20-byte buffer,
/// drew their last triangle to vertex 0 on an M3 Max where ten were exact).
#[test]
fn the_last_index_of_a_partial_word_is_read() {
    let nine: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ten: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let seven: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];
    for program in [RewriteProgram::LineStrip, RewriteProgram::TriangleStrip] {
        for (indices, offset) in [
            (&nine, 0),
            (&nine, 2),
            (&ten, 2),
            (&ten, 0),
            (&seven, 4),
            (&seven, 6),
        ] {
            let count = indices.len() as u32;
            let direct = Draw {
                offset,
                ..Draw::strip(program, false, indices, 3)
            };
            let indirect = Draw {
                arguments: [count, 1, 0, 0, 0],
                indirect: Some(4),
                counter: Some(0),
                ..direct.clone()
            };
            for (draw, how) in [(direct, "direct"), (indirect, "indirect")] {
                let (list, _, _) = draw.drawn();
                assert_eq!(
                    list,
                    metal(program, indices, 3),
                    "{program:?} {how}: {count} from byte {offset}"
                );
            }
        }
    }
}
