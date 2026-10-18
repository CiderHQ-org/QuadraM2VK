//! Command predication (`encode_start_if`, `encode_start_else`, `encode_end_if`, and the loops'
//! starts and ends) on the device the cfg selects, under the validation layer, against what Metal
//! answered on an M4 under macOS 27.0.1. The owned synthetic kernel `prd_mark` adds 1 to word `t`
//! of a log, so the log shows which dispatches ran, each once:
//!
//! - streams of records and marks as the probes wrote them ([`run`]'s notation): blocks nested,
//!   with Elses, spanning a unit's encoders of one dispatch type, and killed by misused records,
//!   run the marks Metal ran and answer what Metal answered;
//! - the condition compares as Metal's codes say, reads the word at the offset rounded down to a
//!   multiple of 4, and reads what a dispatch of the encoder wrote just before, with no barrier,
//!   in a serial and a concurrent encoder;
//! - inside a failing block no dispatch runs, of any form: direct, indirect on the flat and folded
//!   paths, indirect by a host step, and an indirect command buffer's execution of a fixed and of
//!   an indirect range; inside a holding one each runs once; setters inside a skipped block take
//!   effect;
//! - loops run as Metal's ran, each pass with the encoder's state each command was encoded with,
//!   nested, around Ifs and every dispatch form, and a loop that would run past the copies the
//!   runtime lays down fails its command buffer;
//! - what the runtime refuses: an unbalanced record, a word past its buffer, a live block open
//!   where its unit ends (at the commit), a fourth loop open, and a loop spanning encoders.

use super::{buffer_with, read_buffer, validated_device, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{BarrierScope, DispatchType, IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, ComputeCommandEncoder,
    ComputePipelineState, Device, Error, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    PipelineOptions, ResourceOptions, RuntimeGap, Size,
};

/// The kernels: `prd_mark` adds 1 to word `t[0]` of `log`; `prd_put` writes `v[0]` into word 0
/// of `w`; `prd_mark_group` adds its threadgroup's width to word `t[0]` of `log` (it observes
/// its partition, so its thread dispatches from GPU arguments take a host step).
const KERNELS: &str = r#"; Owned synthetic kernels: command predication's marks.
source_filename = "predication.metal"

define void @prd_mark(ptr addrspace(2) %t, ptr addrspace(1) %log) {
entry:
  %i = load i32, ptr addrspace(2) %t, align 4
  %i64 = zext i32 %i to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %log, i64 %i64
  %cur = load i32, ptr addrspace(1) %o, align 4
  %n = add i32 %cur, 1
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}

define void @prd_put(ptr addrspace(2) %t, ptr addrspace(1) %log) {
entry:
  %v = load i32, ptr addrspace(2) %t, align 4
  store i32 %v, ptr addrspace(1) %log, align 4
  ret void
}

define void @prd_mark_group(ptr addrspace(2) %t, ptr addrspace(1) %log, <3 x i32> %tg) {
entry:
  %i = load i32, ptr addrspace(2) %t, align 4
  %i64 = zext i32 %i to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %log, i64 %i64
  %cur = load i32, ptr addrspace(1) %o, align 4
  %w = extractelement <3 x i32> %tg, i64 0
  %n = add i32 %cur, %w
  store i32 %n, ptr addrspace(1) %o, align 4
  ret void
}

!air.kernel = !{!0, !5, !6}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @prd_mark, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"log"}
!5 = !{ptr @prd_put, !1, !2}
!6 = !{ptr @prd_mark_group, !1, !7}
!7 = !{!3, !4, !8}
!8 = !{i32 2, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tg"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The log's words.
const LOG_WORDS: usize = 128;

/// Metal's comparison code `==` (0 `==`, 1 `!=`, 2 `>`, 3 `>=`, 4 `<`, 5 `<=`, and `==` for any
/// other code).
const EQUAL: u64 = 0;

/// Metal's comparison codes `!=` and `<`.
const NOT_EQUAL: u64 = 1;
const LESS: u64 = 4;

/// The state of kernel `name` on `device`, made for indirect command buffers where `indirect`.
fn kernel(device: &Device, name: &str, indirect: bool) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(KERNELS).unwrap())
        .unwrap();
    let function = library.new_function_with_name(name).unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = indirect;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// `words` as bytes.
fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// One thread, or a threadgroup of one.
fn one() -> Size {
    Size::new(1, 1, 1)
}

/// Write `value` into word 0 of `buffer` from the CPU.
fn write_word(buffer: &Buffer, value: u32) {
    let contents = buffer
        .contents()
        .expect("a Shared buffer has contents")
        .as_ptr();
    // SAFETY: the buffer holds at least 4 bytes at `contents`, alive with it, and no command
    // buffer of these tests runs while the CPU writes it.
    unsafe { contents.cast::<u32>().write_unaligned(value) }
}

/// What a test runs with: the marking kernel's state, the log, and the condition words (word 0
/// is 1).
struct Marks {
    mark: ComputePipelineState,
    log: Buffer,
    word: Buffer,
}

impl Marks {
    fn new(device: &Device) -> Marks {
        Marks {
            mark: kernel(device, "prd_mark", false),
            log: buffer_with(device, &[0; LOG_WORDS * 4]).unwrap(),
            word: buffer_with(device, &bytes(&[1, 0, 0, 0])).unwrap(),
        }
    }

    /// Bind the marking kernel to mark `tag` on `encoder`.
    fn bind(&self, encoder: &mut ComputeCommandEncoder, tag: u32) {
        encoder.set_compute_pipeline_state(&self.mark).unwrap();
        encoder.set_bytes(&tag.to_le_bytes(), 0).unwrap();
        encoder.set_buffer(Some(&self.log), 0, 1).unwrap();
    }

    /// Mark `tag` from `encoder`.
    fn mark(&self, encoder: &mut ComputeCommandEncoder, tag: u32) {
        self.bind(encoder, tag);
        encoder.dispatch_threads(one(), one()).unwrap();
    }

    /// Each tag's marks, by tag.
    fn counts(&self) -> Vec<u32> {
        read_buffer(&self.log, 0, self.log.length())
            .unwrap()
            .chunks(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect()
    }

    /// The tags that ran, ascending; a tag run more than once fails here.
    fn ran(&self) -> Vec<u32> {
        let log = self.counts();
        for (tag, &n) in log.iter().enumerate() {
            assert!(n <= 1, "mark {tag} ran {n} times");
        }
        (0..LOG_WORDS as u32)
            .filter(|&tag| log[tag as usize] == 1)
            .collect()
    }
}

/// Commit `command_buffer`, wait for it, and require it completed.
fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// Run `test` on a validated device, then require that the layer reported nothing.
fn on_device(test: impl FnOnce(&Device)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        println!("PRD device {}", device.name());
        test(&device);
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// Run `script` on `device` in one command buffer, in the probes' notation: `Tn` marks `n`; `I1`
/// and `I0` start an If holding and failing (word 0 is 1, compared `==` with 1 and with 0); `In`
/// starts one on a nil buffer; `L` is an Else; `W1` and `W0` start a While holding and failing,
/// `D` a DoWhile; `E`, `V`, `Z0` and `Zn` are an `encodeEndIf`, an `encodeEndWhile`, an
/// `encodeEndDoWhile` failing and one on a nil buffer, whose answers are kept; `|` and `%` go on
/// in a new serial and concurrent encoder (a leading `%` makes the first concurrent); `B` puts a
/// blit encoder between two serial ones. The tags that ran, and the answers.
fn run(device: &Device, script: &str) -> (Vec<u32>, String) {
    let marks = Marks::new(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let open = |token: &str| match token {
        "%" => command_buffer
            .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)
            .unwrap(),
        _ => command_buffer.compute_command_encoder().unwrap(),
    };
    let mut tokens = script.split_whitespace().peekable();
    let mut encoder = open(tokens.next_if_eq(&"%").unwrap_or("|"));
    let mut answers = String::new();
    let answer = |yes: bool| if yes { '1' } else { '0' };
    for token in tokens {
        match token {
            "I1" => encoder
                .encode_start_if(Some(&marks.word), 0, EQUAL, 1)
                .unwrap(),
            "I0" => encoder
                .encode_start_if(Some(&marks.word), 0, EQUAL, 0)
                .unwrap(),
            "In" => encoder.encode_start_if(None, 0, EQUAL, 0).unwrap(),
            "L" => encoder.encode_start_else().unwrap(),
            "W1" => encoder
                .encode_start_while(Some(&marks.word), 0, EQUAL, 1)
                .unwrap(),
            "W0" => encoder
                .encode_start_while(Some(&marks.word), 0, EQUAL, 0)
                .unwrap(),
            "D" => encoder.encode_start_do_while().unwrap(),
            "E" => answers.push(answer(encoder.encode_end_if().unwrap())),
            "V" => answers.push(answer(encoder.encode_end_while().unwrap())),
            "Z0" => answers.push(answer(
                encoder
                    .encode_end_do_while(Some(&marks.word), 0, EQUAL, 0)
                    .unwrap(),
            )),
            "Zn" => answers.push(answer(
                encoder.encode_end_do_while(None, 0, EQUAL, 0).unwrap(),
            )),
            "|" | "%" => {
                encoder.end_encoding().unwrap();
                encoder = open(token);
            }
            "B" => {
                encoder.end_encoding().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.fill_buffer(&marks.word, 8..12, 1).unwrap();
                blit.end_encoding().unwrap();
                encoder = open("|");
            }
            tag => marks.mark(
                &mut encoder,
                tag.strip_prefix('T').unwrap().parse().unwrap(),
            ),
        }
    }
    encoder.end_encoding().unwrap();
    complete(&command_buffer);
    (marks.ran(), answers)
}

/// Streams Metal ran on an M4 under macOS 27.0.1 ([`run`]'s notation), the tags that ran and
/// what the Ends answered: from our probes (`else0t` to `elsenest`, `span_*` to `kill_open`,
/// `while_kill`, `n04`, `k*`, `r*`), each case under its probe's name (`while_kill` without its
/// counter, which its killed body never ran; `while0` and `do0`, of a loop whose condition fails at
/// once, without theirs). Only streams the runtime takes are here: none with an unbalanced record
/// or a live block open where its unit ends. `elsenest` and `else2_skipped` logged no answers;
/// theirs are Metal's rule for Ends where no misuse killed a block (each answers YES; every If of
/// `cmp`'s 63 rows) and inside a killed one.
const SCRIPTS: &[(&str, &str, &[u32], &str)] = &[
    ("else0t", "T9 I1 T1 E T8", &[1, 8, 9], "1"),
    ("else0f", "T9 I0 T1 E T8", &[8, 9], "1"),
    ("else2t", "T9 I1 T1 L T2 L T3 E T8", &[8, 9], "1"),
    ("else2f", "T9 I0 T1 L T2 L T3 E T8", &[8, 9], "1"),
    (
        "elsenest",
        "I0 I1 T1 L T2 E L I1 T3 L T4 E I0 T5 L T6 E E T7",
        &[3, 6, 7],
        "1111",
    ),
    (
        "span_ss_t",
        "T9 I1 T1 | T2 E T3 | T4",
        &[1, 2, 3, 4, 9],
        "1",
    ),
    ("span_ss_f", "T9 I0 T1 | T2 E T3 | T4", &[3, 4, 9], "1"),
    (
        "span_cc_t",
        "% T9 I1 T1 % T2 E T3 % T4",
        &[1, 2, 3, 4, 9],
        "1",
    ),
    ("span_cc_f", "% T9 I0 T1 % T2 E T3 % T4", &[3, 4, 9], "1"),
    ("kill_nested", "I1 T1 I1 T2 V T3 E T4 E T5", &[5], "001"),
    ("kill_else_t", "I1 T1 V T2 L T3 E T4", &[4], "01"),
    ("kill_else_f", "I0 T1 V T2 L T3 E T4", &[4], "01"),
    ("else_kill_t", "I1 T1 L T2 V T3 E T4", &[4], "01"),
    ("else_kill_f", "I0 T1 L T2 V T3 E T4", &[4], "01"),
    (
        "else2_skipped",
        "I0 I1 T1 L T2 L T3 E T4 L T5 E T6",
        &[6],
        "01",
    ),
    (
        "else2_inner",
        "I1 T1 I1 T2 L T3 L T4 E T5 L T6 E T7",
        &[7],
        "01",
    ),
    ("kill_open", "T9 I1 T1 V T2 B T3", &[3, 9], "0"),
    ("n04", "T1 I1 T2 In T3 E T4 B T5", &[1, 4, 5], "1"),
    ("k01", "I1 T1 I1 T2 I1 T3 V T4 E T5 E T6 E T7", &[7], "0001"),
    ("k02", "I1 T1 V T2 I1 T3 E T4 E T5", &[5], "001"),
    ("k03", "I1 T1 E T2 I1 T3 V E T4", &[1, 2, 4], "101"),
    ("k04", "I0 T1 I1 T2 V E T3 L T4 E T5", &[5], "001"),
    ("k05", "I1 T1 V V T2 E T3", &[3], "001"),
    ("k06", "I1 T1 | T2 V T3 E T4", &[4], "01"),
    ("k07", "I1 T1 I1 T2 V E T3 L T4 E T5", &[5], "001"),
    (
        "k08",
        "T0 I1 T1 I1 T2 E T3 I1 T4 V E T5 E T6",
        &[0, 6],
        "1001",
    ),
    ("k09", "I1 T1 I1 T2 L T3 V E T4 E T5", &[5], "001"),
    ("k11", "I1 T1 I1 V E T2 B T3", &[3], "00"),
    ("k12", "I0 T1 I1 V E T2 E T3", &[3], "001"),
    ("k14", "I1 T1 I1 T2 I1 T3 V E E T4 E T5", &[5], "0001"),
    ("k17", "I1 T1 I0 T2 V E T3 E T4", &[4], "001"),
    ("k18", "I1 T1 I1 T2 E T3 V T4 E T5", &[5], "101"),
    ("k19", "I1 T1 V T2 E T3 I1 T4 V T5 E T6", &[3, 6], "0101"),
    ("k20", "T0 I1 T1 I1 T2 V T3 E T4 E T5", &[0, 5], "001"),
    ("k21", "I1 T1 I1 T2 V T3 E T4 B T5", &[5], "00"),
    ("k22", "I1 T1 I1 T2 V T3 E T4 | T5 E T6", &[6], "001"),
    ("k23", "I1 I1 I1 V E E E T1", &[1], "0001"),
    ("k24", "I1 I1 I1 I1 V E E E E T1", &[1], "00001"),
    ("k25", "I1 T1 I1 T2 V T3 L T4 E T5 E T6", &[6], "001"),
    ("k26", "I1 T1 V T2 L T3 L T4 E T5", &[5], "01"),
    ("k27", "I1 I1 V E T1 L T2 E T3", &[3], "001"),
    ("k28", "I0 I1 V E T1 L T2 E T3", &[3], "001"),
    ("while_kill", "W1 T7 E T8 V T9", &[9], "01"),
    ("k15", "D T1 D T2 E T3 Z0 T4 Z0 T5", &[5], "001"),
    ("k16", "I1 T1 D T2 E T3 Z0 T4 E T5", &[5], "001"),
    ("k30", "D T1 I1 T2 Zn T3 E T4 Z0 T5", &[5], "001"),
    ("r08", "I0 D T1 Z0 E T2", &[2], "11"),
    ("r09", "I1 V D T1 Z0 T2 E T3", &[3], "001"),
    ("while0", "T1 W0 T2 V T3", &[1, 3], "1"),
    ("do0", "T1 D T2 Z0 T3", &[1, 2, 3], "1"),
];

/// Every stream of [`SCRIPTS`] runs the marks Metal ran, each once, and its Ends answer what
/// Metal's answered.
#[test]
fn streams_run_what_metal_ran() {
    on_device(|device| {
        let mut wrong = Vec::new();
        for &(name, script, ran, answers) in SCRIPTS {
            let got = run(device, script);
            if got != (ran.to_vec(), answers.to_string()) {
                wrong.push(format!(
                    "{name} `{script}`: ran {:?} answered {:?}, Metal ran {ran:?} answered \
                     {answers:?}",
                    got.0, got.1
                ));
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    });
}

/// Which branch each of `cmp`'s rows took on Metal: `T` the If's, `E` the
/// Else's, for the rows [`cmp_rows`] lists, in order.
const CMP_BRANCHES: &str = "ETETETTEETTEEETETTETEETEETEETEETEETETEETEETEEETEETEETEETEEETEET";

/// `cmp`'s rows over the words `0x80000007, 0xffffffff, 5, 0`: each comparison code (and codes
/// past Metal's, which compare `==`) against references below, at and above word 0; then `==`
/// at each offset from 1 to 9, against each word.
fn cmp_rows() -> Vec<(u64, u64, u32)> {
    let codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 255, 0x1_0000_0002, u64::MAX];
    let codes = codes
        .into_iter()
        .flat_map(|code| [0x8000_0006, 0x8000_0007, 0x8000_0008].map(|r| (code, 0, r)));
    let offsets = (1..=9).flat_map(|offset| [0x8000_0007, u32::MAX, 5].map(|r| (EQUAL, offset, r)));
    codes.chain(offsets).collect()
}

/// The condition compares the word as Metal's codes say, unsigned, the word at the offset
/// rounded down to a multiple of 4: each row runs the branch Metal ran, and its End answers YES.
#[test]
fn the_condition_compares_as_metal_does() {
    on_device(|device| {
        let marks = Marks::new(device);
        let words = buffer_with(device, &bytes(&[0x8000_0007, u32::MAX, 5, 0])).unwrap();
        let rows = cmp_rows();
        assert_eq!(rows.len(), CMP_BRANCHES.len());
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        for (row, &(code, offset, reference)) in rows.iter().enumerate() {
            let row = row as u32;
            encoder
                .encode_start_if(Some(&words), offset, code, reference)
                .unwrap();
            marks.mark(&mut encoder, 2 * row);
            encoder.encode_start_else().unwrap();
            marks.mark(&mut encoder, 2 * row + 1);
            assert_eq!(encoder.encode_end_if(), Ok(true));
        }
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let expected: Vec<u32> = CMP_BRANCHES
            .chars()
            .enumerate()
            .map(|(row, branch)| 2 * row as u32 + u32::from(branch == 'E'))
            .collect();
        assert_eq!(marks.ran(), expected);
    });
}

/// The word is read when the GPU reaches the record, as Metal reads it: after a dispatch just
/// before it wrote it, with no barrier, in a serial and in a concurrent encoder; after a CPU write
/// made between the encoding and the commit; and after a blit earlier in the command buffer filled
/// it.
#[test]
fn the_word_is_read_when_the_gpu_reaches_the_record() {
    on_device(|device| {
        let put = kernel(device, "prd_put", false);
        let queue = device.new_command_queue().unwrap();
        for dispatch_type in [DispatchType::Serial, DispatchType::Concurrent] {
            let marks = Marks::new(device);
            write_word(&marks.word, 3);
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap();
            for i in 0..16u32 {
                let value: u32 = if i % 2 == 1 { 7 } else { 3 };
                encoder.set_compute_pipeline_state(&put).unwrap();
                encoder.set_bytes(&value.to_le_bytes(), 0).unwrap();
                encoder.set_buffer(Some(&marks.word), 0, 1).unwrap();
                encoder.dispatch_threads(one(), one()).unwrap();
                encoder
                    .encode_start_if(Some(&marks.word), 0, EQUAL, 7)
                    .unwrap();
                marks.mark(&mut encoder, 2 * i);
                encoder.encode_start_else().unwrap();
                marks.mark(&mut encoder, 2 * i + 1);
                assert_eq!(encoder.encode_end_if(), Ok(true));
                // Metal's probe orders the next write after this If's marks, as a concurrent
                // encoder needs.
                encoder.memory_barrier_with_scope(BarrierScope::BUFFERS);
            }
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let expected: Vec<u32> = (0..16).map(|i| 2 * i + u32::from(i % 2 == 0)).collect();
            assert_eq!(marks.ran(), expected, "{dispatch_type:?}");
        }

        // A CPU write after the encoding, before the commit.
        let marks = Marks::new(device);
        write_word(&marks.word, 3);
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .encode_start_if(Some(&marks.word), 0, EQUAL, 7)
            .unwrap();
        marks.mark(&mut encoder, 4);
        encoder.encode_start_else().unwrap();
        marks.mark(&mut encoder, 5);
        assert_eq!(encoder.encode_end_if(), Ok(true));
        encoder.end_encoding().unwrap();
        write_word(&marks.word, 7);
        complete(&command_buffer);
        assert_eq!(marks.ran(), [4]);

        // A blit earlier in the command buffer.
        let marks = Marks::new(device);
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&marks.word, 0..4, 7).unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .encode_start_if(Some(&marks.word), 0, EQUAL, 0x0707_0707)
            .unwrap();
        marks.mark(&mut encoder, 1);
        encoder.encode_start_else().unwrap();
        marks.mark(&mut encoder, 2);
        assert_eq!(encoder.encode_end_if(), Ok(true));
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(marks.ran(), [1]);
    });
}

/// How many dispatch forms [`Forms::encode`] encodes.
const FORMS: u32 = 8;

/// One of each dispatch form, of the marking kernel: direct of threads and of threadgroups;
/// indirect of threadgroups and of threads, of `prd_mark` (which may take the flat path) and of
/// `prd_mark_group` (which observes its partition, so takes a host step); and an indirect
/// command buffer's command executed over a fixed range and another over an indirect one.
struct Forms {
    group: ComputePipelineState,
    /// `MTLDispatchThreadsIndirectArguments` of one thread in a group of one, which begin with
    /// `MTLDispatchThreadgroupsIndirectArguments` of one group.
    arguments: Buffer,
    /// Command `c` of the indirect command buffer marks [`Forms::command_tag`]`(c)`, reading its
    /// tag from word `tag` of `tags`.
    icb: IndirectCommandBuffer,
    tags: Buffer,
    /// At `8 c` for an even `c`, the `MTLIndirectCommandBufferExecutionRange` of command `c + 1`.
    ranges: Buffer,
}

impl Forms {
    /// The commands of the indirect command buffer: two per block of the tests below.
    const COMMANDS: u64 = 8;

    /// The tag command `command` marks: the 7th and 8th forms of block `command / 2` of
    /// [`every_dispatch_form_runs_as_its_block_says`].
    fn command_tag(command: u64) -> u32 {
        (command / 2) as u32 * 10 + 6 + (command % 2) as u32
    }

    fn new(device: &Device, marks: &Marks) -> Forms {
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
        descriptor.max_kernel_buffer_bind_count = 2;
        let icb = device
            .new_indirect_command_buffer(&descriptor, Self::COMMANDS, ResourceOptions::shared())
            .unwrap();
        let tags: Vec<u32> = (0..LOG_WORDS as u32).collect();
        let tags = buffer_with(device, &bytes(&tags)).unwrap();
        let mark = kernel(device, "prd_mark", true);
        for command in 0..Self::COMMANDS {
            let at = icb.indirect_compute_command_at(command).unwrap();
            at.set_compute_pipeline_state(&mark).unwrap();
            let tag = u64::from(Self::command_tag(command));
            at.set_kernel_buffer(Some(&tags), 4 * tag, 0).unwrap();
            at.set_kernel_buffer(Some(&marks.log), 0, 1).unwrap();
            at.concurrent_dispatch_threads(one(), one()).unwrap();
        }
        let ranges: Vec<u32> = (0..Self::COMMANDS as u32)
            .flat_map(|c| match c % 2 {
                0 => [c + 1, 1],
                _ => [0, 0],
            })
            .collect();
        Forms {
            group: kernel(device, "prd_mark_group", false),
            arguments: buffer_with(device, &bytes(&[1; 6])).unwrap(),
            icb,
            tags,
            ranges: buffer_with(device, &bytes(&ranges)).unwrap(),
        }
    }

    /// Make the commands' buffers resident on `encoder`, as an indirect command's must be.
    fn resident(&self, encoder: &mut ComputeCommandEncoder, marks: &Marks) {
        encoder
            .use_resources(
                &[self.tags.as_resource(), marks.log.as_resource()],
                ResourceUsage::READ | ResourceUsage::WRITE,
            )
            .unwrap();
    }

    /// Encode each form on `encoder`, marking `base` to `base + FORMS - 1`; the executions run
    /// commands `command` (fixed) and `command + 1` (indirect).
    fn encode(&self, encoder: &mut ComputeCommandEncoder, marks: &Marks, command: u64) {
        let base = Self::command_tag(command) - 6;
        marks.bind(encoder, base);
        encoder.dispatch_threads(one(), one()).unwrap();
        marks.bind(encoder, base + 1);
        encoder.dispatch_threadgroups(one(), one()).unwrap();
        marks.bind(encoder, base + 2);
        encoder
            .dispatch_threadgroups_indirect(&self.arguments, 0, one())
            .unwrap();
        marks.bind(encoder, base + 3);
        encoder
            .dispatch_threads_indirect(&self.arguments, 0)
            .unwrap();
        encoder.set_compute_pipeline_state(&self.group).unwrap();
        encoder.set_bytes(&(base + 4).to_le_bytes(), 0).unwrap();
        encoder
            .dispatch_threadgroups_indirect(&self.arguments, 0, one())
            .unwrap();
        encoder.set_bytes(&(base + 5).to_le_bytes(), 0).unwrap();
        encoder
            .dispatch_threads_indirect(&self.arguments, 0)
            .unwrap();
        encoder
            .execute_commands_in_buffer(&self.icb, command..command + 1)
            .unwrap();
        encoder
            .execute_commands_in_buffer_indirect(&self.icb, &self.ranges, 8 * command)
            .unwrap();
    }
}

/// Inside a failing block no dispatch runs, of any form, and inside a holding one each runs once,
/// as on Metal: a failing If's block and a holding If's Else's run nothing; the failing If's Else's
/// and the holding If's run each form. A setter inside a failing block takes effect: the state and
/// tag it set mark after the block.
#[test]
fn every_dispatch_form_runs_as_its_block_says() {
    on_device(|device| {
        let marks = Marks::new(device);
        let forms = Forms::new(device, &marks);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        forms.resident(&mut encoder, &marks);
        for (holds, blocks) in [(false, [0, 2]), (true, [4, 6])] {
            let reference = u32::from(holds);
            encoder
                .encode_start_if(Some(&marks.word), 0, EQUAL, reference)
                .unwrap();
            forms.encode(&mut encoder, &marks, blocks[0]);
            encoder.encode_start_else().unwrap();
            forms.encode(&mut encoder, &marks, blocks[1]);
            assert_eq!(encoder.encode_end_if(), Ok(true));
        }
        encoder
            .encode_start_if(Some(&marks.word), 0, EQUAL, 0)
            .unwrap();
        marks.bind(&mut encoder, 50);
        assert_eq!(encoder.encode_end_if(), Ok(true));
        encoder.dispatch_threads(one(), one()).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let ran: Vec<u32> = (10..10 + FORMS).chain(20..20 + FORMS).chain([50]).collect();
        assert_eq!(marks.ran(), ran);
    });
}

/// What the runtime refuses, recording nothing, where Metal's answer is not one it can give: an
/// unbalanced record (an End or Else with no block open, a Start or an `encodeEndDoWhile` on a nil
/// buffer with none), a word past its buffer's end, and a fourth loop open (item 7); the work
/// around them runs. A live block open where its unit ends refuses the commit: at a change of
/// dispatch type (whose EndIf then has no block open), at a blit encoder, and at the command
/// buffer's end.
#[test]
fn what_the_runtime_refuses() {
    on_device(|device| {
        let marks = Marks::new(device);
        let queue = device.new_command_queue().unwrap();
        let unbalanced = Err(Error::NotImplemented(RuntimeGap::PredicationUnbalanced));
        let past = Err(Error::NotImplemented(RuntimeGap::PredicationWordPastBuffer));
        let deep = Err(Error::NotImplemented(RuntimeGap::PredicatedLoopDepth));
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        marks.mark(&mut encoder, 1);
        assert_eq!(encoder.encode_end_if().map(drop), unbalanced);
        assert_eq!(encoder.encode_start_else(), unbalanced);
        assert_eq!(encoder.encode_end_while().map(drop), unbalanced);
        assert_eq!(
            encoder.encode_end_do_while(None, 0, EQUAL, 0).map(drop),
            unbalanced
        );
        assert_eq!(encoder.encode_start_if(None, 0, EQUAL, 0), unbalanced);
        let word = &marks.word;
        assert_eq!(encoder.encode_start_if(Some(word), 16, EQUAL, 0), past);
        assert_eq!(
            encoder.encode_start_if(Some(word), 0x1_0000_0000, EQUAL, 0),
            past
        );
        assert_eq!(encoder.encode_start_while(None, 0, EQUAL, 0), unbalanced);
        assert_eq!(
            encoder
                .encode_end_do_while(Some(word), 0, EQUAL, 0)
                .map(drop),
            unbalanced
        );
        // Three loops open, failing at once; a fourth is refused.
        encoder.encode_start_while(Some(word), 0, EQUAL, 0).unwrap();
        encoder.encode_start_do_while().unwrap();
        encoder.encode_start_while(Some(word), 0, EQUAL, 0).unwrap();
        assert_eq!(encoder.encode_start_while(Some(word), 0, EQUAL, 1), deep);
        assert_eq!(encoder.encode_start_do_while(), deep);
        marks.mark(&mut encoder, 4);
        assert_eq!(encoder.encode_end_while(), Ok(true));
        assert_eq!(
            encoder.encode_end_do_while(Some(word), 0, EQUAL, 0),
            Ok(true)
        );
        assert_eq!(encoder.encode_end_while(), Ok(true));
        // The last word of the buffer, at an offset Metal rounds down to it.
        encoder.encode_start_if(Some(word), 15, EQUAL, 0).unwrap();
        marks.mark(&mut encoder, 2);
        assert_eq!(encoder.encode_end_if(), Ok(true));
        marks.mark(&mut encoder, 3);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(marks.ran(), [1, 2, 3]);

        let open = Err(Error::NotImplemented(RuntimeGap::PredicationOpenAtUnitEnd));
        for end in ["concurrent", "blit", "commit"] {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.encode_start_if(Some(word), 0, EQUAL, 1).unwrap();
            marks.mark(&mut encoder, 1);
            encoder.end_encoding().unwrap();
            let next = match end {
                "concurrent" => Some(DispatchType::Concurrent),
                "blit" => {
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.fill_buffer(word, 8..12, 1).unwrap();
                    blit.end_encoding().unwrap();
                    Some(DispatchType::Serial)
                }
                _ => None,
            };
            if let Some(dispatch_type) = next {
                let mut encoder = command_buffer
                    .compute_command_encoder_with_dispatch_type(dispatch_type)
                    .unwrap();
                assert_eq!(encoder.encode_end_if().map(drop), unbalanced, "{end}");
                encoder.end_encoding().unwrap();
            }
            assert_eq!(command_buffer.commit(), open, "{end}");
        }

        // A live loop that starts in one encoder and ends in the next of its unit refuses the
        // commit: the runtime lays a loop down from its own encoder's commands.
        let across = Err(Error::NotImplemented(
            RuntimeGap::PredicatedLoopAcrossEncoders,
        ));
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.encode_start_do_while().unwrap();
        marks.mark(&mut encoder, 1);
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        assert_eq!(
            encoder.encode_end_do_while(Some(word), 0, EQUAL, 0),
            Ok(true)
        );
        encoder.end_encoding().unwrap();
        assert_eq!(command_buffer.commit(), across);
    });
}

/// `loop`'s marks on `device`: a command buffer of one serial encoder (concurrent where
/// `concurrent`) that `encode` fills, run to its end; its status and error, and every tag's
/// marks.
fn looped(
    device: &Device,
    concurrent: bool,
    encode: impl FnOnce(&mut ComputeCommandEncoder, &Marks),
) -> (CommandBufferStatus, Option<CommandBufferError>, Vec<u32>) {
    let marks = Marks::new(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let dispatch_type = match concurrent {
        true => DispatchType::Concurrent,
        false => DispatchType::Serial,
    };
    let mut encoder = command_buffer
        .compute_command_encoder_with_dispatch_type(dispatch_type)
        .unwrap();
    encode(&mut encoder, &marks);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    (
        command_buffer.status(),
        command_buffer.error(),
        marks.counts(),
    )
}

/// Write `value` into log word `at` from `encoder`, by `prd_put`.
fn put(encoder: &mut ComputeCommandEncoder, put: &ComputePipelineState, marks: &Marks, at: u64) {
    encoder.set_compute_pipeline_state(put).unwrap();
    encoder.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
    encoder.set_buffer(Some(&marks.log), 4 * at, 1).unwrap();
    encoder.dispatch_threads(one(), one()).unwrap();
}

/// Loops run as Metal's ran on an M4 under macOS 27.0.1, each counting in the
/// log what the probe counted: `while0` to `while5` (`while (log[0] != n) mark 0`), `do0` to
/// `do5` (`do mark 0 while (log[0] < n)`), at 64 runs, the copies the runtime lays down, too;
/// `whileconc` in a concurrent encoder (its body writes the word, no barrier); `loopnest3`
/// (`3 x 4 x 5` passes of a DoWhile in two Whiles, the inner counts reset by a dispatch in the
/// outer bodies, every End answering YES); `loopbind` (each pass runs with the bindings and bytes
/// each dispatch was encoded with, and the last stand after the loop); and `loopif` (an If and
/// its Else inside a DoWhile, per pass: Metal's log `3 1 2 1 3 1 3 1`). Every one completes.
#[test]
fn loops_run_as_metal_ran() {
    on_device(|device| {
        let ok = (CommandBufferStatus::Completed, None);
        let expect = |got: (CommandBufferStatus, Option<CommandBufferError>, Vec<u32>),
                      want: &[(usize, u32)],
                      case: &str| {
            assert_eq!((got.0, got.1), ok, "{case}");
            let mut counts = vec![0; LOG_WORDS];
            for &(tag, n) in want {
                counts[tag] = n;
            }
            assert_eq!(got.2, counts, "{case}");
        };
        for n in [0, 1, 5, 64] {
            let got = looped(device, false, |encoder, marks| {
                encoder
                    .encode_start_while(Some(&marks.log), 0, NOT_EQUAL, n)
                    .unwrap();
                marks.mark(encoder, 0);
                marks.mark(encoder, 1);
                assert_eq!(encoder.encode_end_while(), Ok(true));
                marks.mark(encoder, 2);
            });
            expect(got, &[(0, n), (1, n), (2, 1)], &format!("while{n}"));
            let got = looped(device, false, |encoder, marks| {
                encoder.encode_start_do_while().unwrap();
                marks.mark(encoder, 0);
                marks.mark(encoder, 1);
                let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, n);
                assert_eq!(end, Ok(true));
                marks.mark(encoder, 2);
            });
            let runs = n.max(1);
            expect(got, &[(0, runs), (1, runs), (2, 1)], &format!("do{n}"));
        }
        let got = looped(device, true, |encoder, marks| {
            encoder
                .encode_start_while(Some(&marks.log), 0, NOT_EQUAL, 7)
                .unwrap();
            marks.mark(encoder, 0);
            assert_eq!(encoder.encode_end_while(), Ok(true));
        });
        expect(got, &[(0, 7)], "whileconc");

        let zero = kernel(device, "prd_put", false);
        let got = looped(device, false, |encoder, marks| {
            let log = Some(&marks.log);
            encoder.encode_start_while(log, 0, NOT_EQUAL, 3).unwrap();
            marks.mark(encoder, 0);
            put(encoder, &zero, marks, 1);
            encoder.encode_start_while(log, 4, NOT_EQUAL, 4).unwrap();
            marks.mark(encoder, 1);
            put(encoder, &zero, marks, 2);
            encoder.encode_start_do_while().unwrap();
            marks.mark(encoder, 2);
            marks.mark(encoder, 3);
            assert_eq!(encoder.encode_end_do_while(log, 8, LESS, 5), Ok(true));
            assert_eq!(encoder.encode_end_while(), Ok(true));
            assert_eq!(encoder.encode_end_while(), Ok(true));
        });
        expect(got, &[(0, 3), (1, 4), (2, 5), (3, 60)], "loopnest3");

        let got = looped(device, false, |encoder, marks| {
            encoder.encode_start_do_while().unwrap();
            marks.mark(encoder, 10);
            encoder.set_bytes(&0u32.to_le_bytes(), 0).unwrap();
            encoder.set_buffer_offset(64, 1).unwrap();
            encoder.dispatch_threads(one(), one()).unwrap();
            marks.mark(encoder, 11);
            marks.mark(encoder, 0);
            let end = encoder.encode_end_do_while(Some(&marks.log), 0, NOT_EQUAL, 3);
            assert_eq!(end, Ok(true));
            encoder.dispatch_threads(one(), one()).unwrap();
        });
        expect(got, &[(0, 4), (10, 3), (11, 3), (16, 3)], "loopbind");

        let got = looped(device, false, |encoder, marks| {
            encoder.encode_start_do_while().unwrap();
            marks.mark(encoder, 0);
            encoder
                .encode_start_if(Some(&marks.log), 0, EQUAL, 2)
                .unwrap();
            marks.mark(encoder, 2);
            encoder.encode_start_else().unwrap();
            marks.mark(encoder, 3);
            assert_eq!(encoder.encode_end_if(), Ok(true));
            marks.mark(encoder, 1);
            let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, 4);
            assert_eq!(end, Ok(true));
        });
        expect(got, &[(0, 4), (1, 4), (2, 1), (3, 3)], "loopif");
    });
}

/// Every dispatch form inside a loop runs in each pass: the forms of
/// [`every_dispatch_form_runs_as_its_block_says`] in a DoWhile of three passes mark three times
/// each.
#[test]
fn every_dispatch_form_runs_in_each_pass() {
    on_device(|device| {
        let marks = Marks::new(device);
        let forms = Forms::new(device, &marks);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        forms.resident(&mut encoder, &marks);
        encoder.encode_start_do_while().unwrap();
        forms.encode(&mut encoder, &marks, 2);
        marks.mark(&mut encoder, 0);
        let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, 3);
        assert_eq!(end, Ok(true));
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let mut counts = vec![0; LOG_WORDS];
        counts[0] = 3;
        let base = Forms::command_tag(2) - 6;
        for tag in base..base + FORMS {
            counts[tag as usize] = 3;
        }
        assert_eq!(marks.counts(), counts);
    });
}

/// A loop that would run past the copies the runtime lays down (64, or 16 inside another loop)
/// runs them and fails its command buffer (`CommandBufferError::PredicatedLoopPastCopies`), where
/// Metal runs it on; one that ends within them completes.
#[test]
fn a_loop_past_its_copies_fails_its_command_buffer() {
    on_device(|device| {
        let past = (
            CommandBufferStatus::Error,
            Some(CommandBufferError::PredicatedLoopPastCopies),
        );
        let got = looped(device, false, |encoder, marks| {
            encoder
                .encode_start_while(Some(&marks.log), 0, NOT_EQUAL, 65)
                .unwrap();
            marks.mark(encoder, 0);
            assert_eq!(encoder.encode_end_while(), Ok(true));
            marks.mark(encoder, 1);
        });
        assert_eq!((got.0, got.1), past, "a While of 65");
        assert_eq!(got.2[..2], [64, 1], "a While of 65");
        let got = looped(device, false, |encoder, marks| {
            encoder.encode_start_do_while().unwrap();
            marks.mark(encoder, 0);
            let end = encoder.encode_end_do_while(Some(&marks.log), 0, LESS, 65);
            assert_eq!(end, Ok(true));
        });
        assert_eq!((got.0, got.1), past, "a DoWhile of 65");
        assert_eq!(got.2[0], 64, "a DoWhile of 65");
        let zero = kernel(device, "prd_put", false);
        for (inner, status) in [(16, (CommandBufferStatus::Completed, None)), (17, past)] {
            let got = looped(device, false, |encoder, marks| {
                let log = Some(&marks.log);
                encoder.encode_start_while(log, 0, NOT_EQUAL, 2).unwrap();
                marks.mark(encoder, 0);
                put(encoder, &zero, marks, 1);
                encoder
                    .encode_start_while(log, 4, NOT_EQUAL, inner)
                    .unwrap();
                marks.mark(encoder, 1);
                marks.mark(encoder, 2);
                assert_eq!(encoder.encode_end_while(), Ok(true));
                assert_eq!(encoder.encode_end_while(), Ok(true));
            });
            assert_eq!((got.0, got.1), status, "an inner While of {inner}");
            assert_eq!(got.2[2], 2 * inner.min(16), "an inner While of {inner}");
        }
    });
}
