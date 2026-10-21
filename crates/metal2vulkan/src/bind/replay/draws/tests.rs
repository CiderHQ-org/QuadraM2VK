//! The draw replay's preprocess, on the CPU interpreter: valid where device addresses are and
//! refused by name elsewhere; each command's draw lands in the argument slot of its primitive type
//! with Metal's vertex start and base instance as Vulkan's first vertex and first instance; its
//! vertex and fragment slot tables hold its buffers' resident allocations (or the encoder's,
//! where the commands inherit buffers); an address no allocation holds leaves the slot unbound and
//! counts a resident miss while the command still draws; a command the replay cannot run draws
//! nothing and says why beside clean commands that draw; the host side refuses what it cannot set
//! up.

use std::num::NonZeroU64;
use std::path::PathBuf;

use super::*;
use crate::bind::state::{ResourceClass, UsedResource};
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::device::{DeviceDescription, DevicePolicy};
use crate::indirect_commands::{Action, BufferBinding, INHERIT_PIPELINE, RENDER};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::assert_valid;

const SCRATCH: u64 = 0x0000_0030_0000_0000;
const RECORDS: u64 = 0x0000_0031_0000_0000;
const STATUS: u64 = 0x0000_0012_0000_0000;
const SINK: u64 = 0x0000_0011_0000_0000;

/// The resident allocations `(guest, host, length)`.
const A: (u64, u64, u64) = (0x0000_0001_0000_1000, 0x0000_0020_0000_0000, 256);
const B: (u64, u64, u64) = (0x0000_0001_0000_8000, 0x0000_0021_0000_0000, 60);
/// An address in no resident allocation.
const AWAY: u64 = 0x0000_0001_0004_0000;

/// Every committed profile as a device, by stem.
fn devices() -> Vec<(String, Device)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut out: Vec<(String, Device)> = std::fs::read_dir(&dir)
        .expect("profiles")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|path| {
            let stem = path
                .file_stem()
                .expect("stem")
                .to_string_lossy()
                .into_owned();
            let text = std::fs::read_to_string(&path).expect("profile");
            let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
            (
                stem,
                Device::new(desc, DevicePolicy::default()).expect("device"),
            )
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn addressed(dev: &Device) -> bool {
    dev.profile().fields().buffer_access == BufferAccess::Addressed
}

fn device() -> Device {
    devices()
        .into_iter()
        .map(|(_, d)| d)
        .find(addressed)
        .expect("an addressed profile")
}

/// The resident set: `A` and `B`.
fn state() -> BindState {
    let mut s = BindState::default();
    for (i, (guest, host, length)) in [A, B].into_iter().enumerate() {
        let mut u = UsedResource::new(ResourceClass::Buffer, i as u32);
        u.gpu_address = Some(guest);
        u.device_address = Some(host);
        u.length = length;
        s.used_resources.push(u);
    }
    s
}

fn bind(address: u64) -> Option<BufferBinding> {
    Some(BufferBinding {
        address,
        attribute_stride: None,
    })
}

fn draw(primitive_type: u32, vertex_start: u32, vertex_count: u32, instances: [u32; 2]) -> Action {
    Action::DrawPrimitives {
        primitive_type,
        vertex_start,
        vertex_count,
        instance_count: instances[0],
        base_instance: instances[1],
    }
}

/// A record of `action` with vertex buffers `vertex` and fragment buffers `fragment` (`(index,
/// address)`), its barrier set as `barrier` says.
fn record(action: Action, vertex: &[(usize, u64)], fragment: &[(usize, u64)]) -> CommandRecord {
    let mut r = CommandRecord {
        action,
        ..CommandRecord::default()
    };
    for &(i, address) in vertex {
        r.bindings.buffers_mut(BufferStage::Vertex)[i] = bind(address);
    }
    for &(i, address) in fragment {
        r.bindings.buffers_mut(BufferStage::Fragment)[i] = bind(address);
    }
    r
}

fn bytes(r: &CommandRecord) -> Vec<u8> {
    let mut out = vec![0u8; COMMAND_BYTES];
    r.encode_into(&mut out).expect("record");
    out
}

fn word(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("word"))
}

/// The storage header: `capacity`, the fault word, the policy bits of a render descriptor of
/// draws whose commands inherit the pipeline (and the buffers where `inherit_buffers`), and 4
/// vertex buffers and 2 fragment buffers.
struct Header {
    capacity: u32,
    fault: u32,
    inherit_buffers: bool,
}

/// What one preprocess left.
struct Ran {
    pre: DrawReplayPreprocess,
    scratch: Vec<u8>,
    status: StatusReport,
}

impl Ran {
    /// The `VkDrawIndirectCommand` of command `command` for type slot `slot`.
    fn args(&self, command: u32, slot: u32) -> [u32; 4] {
        let at = self.pre.layout.argument_offset(command, slot) as usize;
        std::array::from_fn(|i| word(&self.scratch, at + 4 * i))
    }

    /// Entry `root` of a slot table at `table`: `[host, guest, size, bind_offset]`.
    fn entry(&self, table: u64, root: u64) -> [u64; 4] {
        let at = (table + root * roots::ENTRY_BYTES) as usize;
        std::array::from_fn(|i| {
            u64::from_le_bytes(
                self.scratch[at + 8 * i..at + 8 * i + 8]
                    .try_into()
                    .expect("u64"),
            )
        })
    }

    fn vertex(&self, command: u32, slot: u64) -> [u64; 4] {
        self.entry(self.pre.layout.vertex_table_offset(command), 2 + slot)
    }

    fn fragment(&self, command: u32, slot: u64) -> [u64; 4] {
        self.entry(self.pre.layout.fragment_table_offset(command), 2 + slot)
    }
}

/// Run the preprocess of `replay` against `records` under `header`, in both scheduling orders.
fn preprocess(replay: &DrawReplay, records: &[CommandRecord], header: Header) -> Ran {
    let dev = device();
    let pre = draw_replay_preprocess(&dev, &state(), replay).expect("preprocess");
    let mut policy = RENDER | DRAW | INHERIT_PIPELINE;
    if header.inherit_buffers {
        policy |= crate::indirect_commands::INHERIT_BUFFERS;
    }
    let counts: u32 = (4 << 8) | (2 << 16);
    let mut storage = Vec::new();
    for w in [header.capacity, header.fault, policy, counts] {
        storage.extend_from_slice(&w.to_le_bytes());
    }
    for r in records {
        storage.extend_from_slice(&bytes(r));
    }
    let l = pre.layout;
    let mut scratch = vec![0xEE; l.bytes as usize];
    let mut put = |at: u64, bytes: &[u8]| {
        scratch[at as usize..at as usize + bytes.len()].copy_from_slice(bytes);
    };
    put(l.params_offset(), &pre.params);
    let mut resident = pre.resident.bytes.clone();
    let sink = RootTable::SINK_HOST_OFFSET as usize;
    resident[sink..sink + 8].copy_from_slice(&SINK.to_le_bytes());
    put(l.resident_offset, &resident);
    put(l.extents_offset, &pre.extents);
    put(l.parent_offset, &pre.parent);
    let mut push = pre.push_constants.clone();
    for hole in &pre.push_holes {
        let (offset, address) = match *hole {
            PushHole::RootTableAddress { offset, .. } => (offset, SCRATCH + l.resident_offset),
            PushHole::ExecutionStatusAddress { offset, .. } => (offset, STATUS),
            PushHole::CommandRecordsAddress { offset, .. } => (offset, RECORDS),
            _ => panic!("a hole the preprocess does not take"),
        };
        push[offset as usize..offset as usize + 8].copy_from_slice(&address.to_le_bytes());
    }
    let m = typed_module().expect("module");
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: None,
                    address: Some(SCRATCH),
                    bytes: scratch.clone(),
                },
                Buffer {
                    slot: None,
                    address: Some(RECORDS),
                    bytes: storage.clone(),
                },
                Buffer {
                    slot: None,
                    address: Some(STATUS),
                    bytes: ExecutionStatus::initial().to_vec(),
                },
            ],
            push: push.clone(),
            spec: Default::default(),
        };
        let cfg = Config {
            workgroups: pre.group_count,
            order,
            ..Config::default()
        };
        outs.push(run(&m, &cfg, res).expect("run").buffers);
    }
    assert_eq!(outs[0], outs[1], "the two scheduling orders agree");
    let mut out = outs.swap_remove(0);
    assert_eq!(out[1], storage, "the records are only read");
    let status: [u8; ExecutionStatus::BYTES] = out[2].as_slice().try_into().expect("status");
    Ran {
        pre,
        scratch: out.swap_remove(0),
        status: ExecutionStatus::decode(&status),
    }
}

fn clean(capacity: u32) -> Header {
    Header {
        capacity,
        fault: 0,
        inherit_buffers: false,
    }
}

#[test]
fn the_draw_preprocess_module_is_valid_where_device_addresses_are() {
    let mut seen = (0, 0);
    for (stem, dev) in devices() {
        let p = dev.profile();
        if addressed(&dev) {
            seen.0 += 1;
            assert_valid(typed_module().expect("module"), p);
            let m = draw_replay_module(&dev).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert!(p.check(&m.requirements).is_ok(), "{stem}");
            assert!(m.layout.sets.is_empty(), "{stem}");
            let range = m.layout.push_constants[0];
            assert_eq!(
                (range.stages, range.offset, range.size),
                (
                    crate::vk::ShaderStageFlags::COMPUTE,
                    0,
                    DrawReplayModule::PUSH_BYTES
                ),
                "{stem}"
            );
        } else {
            seen.1 += 1;
            let e = Unsupported::needs_any_of(
                Feature::GpuEncodedIndirectCommands,
                vec![p.addressed_needs()],
            );
            assert_eq!(
                draw_replay_module(&dev),
                Err(Error::Unsupported(e.clone())),
                "{stem}"
            );
            assert_eq!(
                draw_replay_preprocess(&dev, &state(), &DrawReplay::new(1, 0, 1, vec![3])),
                Err(BindError::Unsupported(e)),
                "{stem}"
            );
        }
    }
    assert!(seen.0 > 0 && seen.1 > 0, "{seen:?}");
}

/// A command's draw lands in the slot of its type, as Vulkan's draw of Metal's: vertex count,
/// instance count, vertex start as the first vertex, base instance as the first instance; the
/// other slot is zero. Its buffers resolve in the resident set at the address's offset, each
/// stage's in its own table; an unset slot and a nil buffer's offset hold no allocation, the nil
/// buffer's offset as the bind offset. An empty command draws nothing and reports nothing.
#[test]
fn each_command_draws_its_own_type_with_its_own_buffers() {
    let replay = DrawReplay::new(3, 0, 3, vec![3, 4]);
    let records = [
        record(
            draw(3, 6, 9, [2, 5]),
            &[(0, A.0 + 16), (2, 40)],
            &[(1, B.0 + 4)],
        ),
        record(draw(4, 0, 4, [1, 0]), &[(1, B.0)], &[]),
        record(Action::Empty, &[(0, A.0)], &[]),
    ];
    let ran = preprocess(&replay, &records, clean(3));
    assert!(!ran.status.replay_fault, "{:?}", ran.status);
    assert!(!ran.status.resident_miss);
    assert_eq!(ran.args(0, 0), [9, 2, 6, 5]);
    assert_eq!(ran.args(0, 1), [0; 4]);
    assert_eq!(ran.args(1, 0), [0; 4]);
    assert_eq!(ran.args(1, 1), [4, 1, 0, 0]);
    assert_eq!(ran.args(2, 0), [0; 4]);
    assert_eq!(ran.args(2, 1), [0; 4]);
    assert_eq!(ran.vertex(0, 0), [A.1, A.0, padded(A.2), 16]);
    assert_eq!(ran.vertex(0, 1), [0; 4]);
    assert_eq!(ran.vertex(0, 2), [0, 0, 0, 40]);
    assert_eq!(ran.fragment(0, 1), [B.1, B.0, padded(B.2), 4]);
    assert_eq!(ran.fragment(0, 0), [0; 4]);
    assert_eq!(ran.vertex(1, 1), [B.1, B.0, padded(B.2), 0]);
    // The sink entry names the shared resident set from each table.
    let sink = ran.entry(ran.pre.layout.vertex_table_offset(1), roots::SINK_ROOT);
    assert_eq!(sink[0], SINK);
    assert_eq!(sink[2], 2, "two resident allocations");
}

/// The resident root table pads each allocation as the binder does.
fn padded(length: u64) -> u64 {
    crate::bind::instantiate::padded(length)
}

/// A command the replay cannot run draws nothing and says why, and the clean commands beside it
/// draw: a draw of a type the execution does not name (`ACTION`), a set barrier (`BARRIER`), a
/// pipeline set while the commands inherit it (`PIPELINE`), a binding past its stage's count
/// (`RECORD`).
#[test]
fn a_command_the_replay_cannot_run_draws_nothing_and_says_why() {
    let replay = DrawReplay::new(5, 0, 5, vec![3]);
    let mut barrier = record(draw(3, 0, 3, [1, 0]), &[], &[]);
    barrier.barrier = true;
    let mut piped = record(draw(3, 0, 3, [1, 0]), &[], &[]);
    piped.bindings.pipeline = NonZeroU64::new(9);
    let cases = [
        (
            record(draw(0, 0, 3, [1, 0]), &[], &[]),
            ReplayFaults::ACTION,
        ),
        (barrier, ReplayFaults::BARRIER),
        (piped, ReplayFaults::PIPELINE),
        (
            record(draw(3, 0, 3, [1, 0]), &[(4, A.0)], &[]),
            ReplayFaults::RECORD,
        ),
    ];
    for (bad, why) in cases {
        let records = [
            record(draw(3, 0, 3, [1, 0]), &[], &[]),
            bad,
            record(draw(3, 3, 3, [1, 0]), &[], &[]),
            record(Action::Empty, &[], &[]),
            record(Action::Empty, &[], &[]),
        ];
        let ran = preprocess(&replay, &records, clean(5));
        assert!(ran.status.replay_fault, "{why:?}");
        assert_eq!(ran.status.replay_faults, why);
        assert_eq!(ran.args(0, 0), [3, 1, 0, 0], "{why:?}");
        assert_eq!(ran.args(1, 0), [0; 4], "{why:?}");
        assert_eq!(ran.args(2, 0), [3, 1, 3, 0], "{why:?}");
    }
}

/// An address no resident allocation holds leaves its slot unbound and counts one resident miss
/// per such buffer, and the command still draws.
#[test]
fn a_buffer_no_allocation_holds_is_unbound_and_counted() {
    let replay = DrawReplay::new(1, 0, 1, vec![3]);
    let records = [record(
        draw(3, 0, 3, [1, 0]),
        &[(0, AWAY), (1, A.0)],
        &[(0, AWAY)],
    )];
    let ran = preprocess(&replay, &records, clean(1));
    assert!(!ran.status.replay_fault);
    assert!(ran.status.resident_miss);
    assert_eq!(ran.status.resident_misses, 2);
    assert_eq!(ran.args(0, 0), [3, 1, 0, 0]);
    assert_eq!(ran.vertex(0, 0), [0; 4]);
    assert_eq!(ran.vertex(0, 1), [A.1, A.0, padded(A.2), 0]);
    assert_eq!(ran.fragment(0, 0), [0; 4]);
}

/// A buffer of the command's own that no slot table holds (past the slots its stage's function
/// reads, or of a stage that reads none) is resolved too: its miss counts, as the host step's
/// does, though no slot holds it. A nil buffer's offset counts none, nor does a command that draws
/// nothing.
#[test]
fn a_buffer_past_the_slots_read_counts_its_miss() {
    let mut replay = DrawReplay::new(2, 0, 2, vec![3]);
    replay.vertex_slots = 1;
    replay.fragment_slots = 0;
    let buffers = |action| {
        record(
            action,
            &[(0, A.0), (1, AWAY), (2, B.0), (3, 0x40)],
            &[(0, AWAY), (1, AWAY)],
        )
    };
    let records = [buffers(draw(3, 0, 3, [1, 0])), buffers(Action::Empty)];
    let ran = preprocess(&replay, &records, clean(2));
    assert!(!ran.status.replay_fault);
    assert!(ran.status.resident_miss);
    assert_eq!(ran.status.resident_misses, 3);
    assert_eq!(ran.args(0, 0), [3, 1, 0, 0]);
    assert_eq!(ran.vertex(0, 0), [A.1, A.0, padded(A.2), 0]);
    assert_eq!(ran.args(1, 0), [0; 4]);
}

/// Where the commands inherit buffers, every command's tables hold the encoder's, and a command
/// that sets one of its own draws nothing (`BUFFERS`).
#[test]
fn inherited_buffers_are_the_encoder_s() {
    let mut replay = DrawReplay::new(2, 0, 2, vec![3]);
    replay.inheritance.buffers = StateSource::Parent;
    replay.parent.buffers_mut(BufferStage::Vertex)[0] = bind(B.0 + 8);
    replay.parent.buffers_mut(BufferStage::Fragment)[1] = bind(A.0);
    let records = [
        record(draw(3, 0, 3, [1, 0]), &[], &[]),
        record(draw(3, 0, 3, [1, 0]), &[(0, A.0)], &[]),
    ];
    let header = Header {
        inherit_buffers: true,
        ..clean(2)
    };
    let ran = preprocess(&replay, &records, header);
    assert_eq!(ran.status.replay_faults, ReplayFaults::BUFFERS);
    assert_eq!(ran.args(0, 0), [3, 1, 0, 0]);
    assert_eq!(ran.args(1, 0), [0; 4]);
    assert_eq!(ran.vertex(0, 0), [B.1, B.0, padded(B.2), 8]);
    assert_eq!(ran.fragment(0, 1), [A.1, A.0, padded(A.2), 0]);
}

/// The producers' sticky encoding fault refuses every command; a command at or past the
/// capacity the header states is not read (`RANGE`), its range starting past the first command.
#[test]
fn a_faulted_or_short_storage_draws_nothing() {
    let replay = DrawReplay::new(2, 0, 2, vec![3]);
    let records = [
        record(draw(3, 0, 3, [1, 0]), &[], &[]),
        record(draw(3, 0, 3, [1, 0]), &[], &[]),
    ];
    let faulted = Header {
        fault: 1,
        ..clean(2)
    };
    let ran = preprocess(&replay, &records, faulted);
    assert_eq!(ran.status.replay_faults, ReplayFaults::ENCODING);
    assert_eq!(ran.args(0, 0), [0; 4]);
    assert_eq!(ran.args(1, 0), [0; 4]);
    let replay = DrawReplay::new(2, 1, 1, vec![3]);
    let ran = preprocess(&replay, &records, clean(1));
    assert_eq!(ran.status.replay_faults, ReplayFaults::RANGE);
    assert_eq!(ran.args(0, 0), [0; 4]);
}

/// The host side refuses a range past the capacity, commands that name their own pipelines, and
/// primitive types that are not one class's draws; tables hold the slots asked for.
#[test]
fn the_host_side_refuses_what_it_cannot_set_up() {
    let dev = device();
    let refused = |r: ReplayRefusal| Err(BindError::InvalidReplay { reason: r });
    let pre = |r: &DrawReplay| draw_replay_preprocess(&dev, &state(), r);
    assert_eq!(
        pre(&DrawReplay::new(2, 1, 2, vec![3])),
        refused(ReplayRefusal::RangeOutsideCapacity)
    );
    let mut own = DrawReplay::new(1, 0, 1, vec![3]);
    own.inheritance.pipeline = StateSource::Command;
    assert_eq!(pre(&own), refused(ReplayRefusal::OwnPipelines));
    for types in [
        vec![],
        vec![5],
        vec![3, 3],
        vec![0, 3],
        vec![2, 3],
        vec![1, 2, 3],
    ] {
        assert_eq!(
            pre(&DrawReplay::new(1, 0, 1, types.clone())),
            refused(ReplayRefusal::DrawPrimitiveTypes),
            "{types:?}"
        );
    }
    for types in [vec![0], vec![1, 2], vec![4, 3]] {
        assert!(pre(&DrawReplay::new(1, 0, 1, types)).is_ok());
    }
    let mut small = DrawReplay::new(4, 0, 4, vec![3]);
    small.vertex_slots = 3;
    small.fragment_slots = 0;
    let l = pre(&small).expect("small").layout;
    assert!(l.uploaded <= l.tables_offset);
    assert_eq!(l.vertex_table_offset(0), l.tables_offset);
    assert_eq!(
        l.fragment_table_offset(0),
        l.tables_offset + 5 * roots::ENTRY_BYTES
    );
    assert_eq!(
        l.vertex_table_offset(1),
        l.tables_offset + 7 * roots::ENTRY_BYTES
    );
    assert_eq!(
        l.arguments_offset,
        l.tables_offset + 28 * roots::ENTRY_BYTES
    );
}
