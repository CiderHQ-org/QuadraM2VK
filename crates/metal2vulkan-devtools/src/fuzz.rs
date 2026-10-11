//! Generated-program differential fuzzing.
//!
//! Each generator builds a random AIR kernel from a fixed-seed [`Xorshift`], computes what the
//! kernel must produce with a Rust reference model, runs the kernel on an [`Execute`]
//! implementation, and reports every divergence. The reference model is the oracle; nothing is
//! compared against another translator's output.
//!
//! The generators are library code here, in the dev-tools crate both sides reach, so every executor
//! runs exactly the same programs: the product's own tests run them on `CpuInterp`
//! (`src/tests/fuzz_interp.rs`, [`GATE_SEEDS`] per generator, and the ignored soak in
//! `src/tests/fuzz_soak.rs`), the engine's typed module on the CPU SPIR-V interpreter, which
//! traps out-of-bounds accesses, reads of never-written workgroup bytes, non-uniform barriers, and
//! runaway loops (its step bound), and which runs every program at every subgroup width the
//! profile's device may run it with and in both scheduling orders, so a race is a failure.
//! Validation's cfg-gated device tests (`candidate/tests/*_fuzz.rs`) run them on a CPU ICD or a
//! GPU.
//!
//! **A refusal is a failure.** A generated program the engine refuses (`Unsupported`, a trap, an
//! L3 failure) is reported like a wrong answer, never skipped. Where the engine does not lower
//! a family yet, the generator is restricted by a documented, typed switch in that generator (see
//! each module), so the restriction is visible and removable in one place.
//! Where the DEVICE lacks what a form needs (a loaded pointer on a profile without device
//! addresses), the generator builds that form only for a [`Target`] that has it: the translation's
//! refusal there is honest, and the form is the device's to lack, not ours.

pub mod access;
pub mod inline;
pub mod irreducible;
pub mod pointer_select;
pub mod structured;
pub mod workgroup;

/// `(operand pattern, the same operation on a u32)`. Every entry must be TOTAL and WRAPPING: the
/// generated programs run to data-dependent successors, so any input can reach any operation and
/// there is nothing to exclude a value at.
pub type Operation = (&'static str, fn(u32) -> u32);

pub const OPS: [Operation; 8] = [
    ("mul i32 %ACC, 3", |a| a.wrapping_mul(3)),
    ("add i32 %ACC, 7", |a| a.wrapping_add(7)),
    ("xor i32 %ACC, 255", |a| a ^ 255),
    ("shl i32 %ACC, 1", |a| a << 1),
    ("lshr i32 %ACC, 1", |a| a >> 1),
    ("sub i32 %ACC, 13", |a| a.wrapping_sub(13)),
    ("or i32 %ACC, 1024", |a| a | 1024),
    ("and i32 %ACC, 65535", |a| a & 65535),
];

/// Threads per dispatch, all in one threadgroup, unless the executor's [`Target`] says otherwise.
/// Thread `t` is fed `t`, so one dispatch sweeps 32 inputs at once.
pub const THREADS: u32 = 32;

/// The dispatch widths a generator can build for: one threadgroup of 32 threads (one Metal
/// simdgroup) or of 64 (two simdgroups, one wave of a 64-lane device).
pub const THREAD_COUNTS: [u32; 2] = [32, 64];

/// What a generator may build for one executor, which runs the engine's output. A generator
/// restricts itself to the families the engine lowers (each generator documents its
/// restrictions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// Threads in the dispatch's one threadgroup, one of [`THREAD_COUNTS`].
    pub threads: u32,
    /// Whether the device reaches buffers by device address (Profile A): a pointer loaded from a
    /// buffer can be dereferenced. Without it the engine refuses such a program as
    /// `Unsupported{DereferenceLoadedPointer}`, honestly, so no generator builds one.
    pub addressed: bool,
}

impl Default for Target {
    /// [`THREADS`] threads on a device with device addresses.
    fn default() -> Target {
        Target {
            threads: THREADS,
            addressed: true,
        }
    }
}

/// How a generated program spells the index operands of its dynamic `getelementptr`s. Every such
/// index the generators build is a small non-negative `i32`, so the three spellings name the same
/// element; they reach different index forms of the engine (a zero-extended word, a
/// sign-extended word, and an `i32` index the GEP itself extends), and [`IndexForm::Narrow32`]
/// programs compute no 64-bit index or offset, which a profile without `shaderInt64` (`vk12-floor`)
/// runs without two-word emulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexForm {
    /// `zext i32 %v to i64`, the Metal front end's spelling of an unsigned index.
    Zext64,
    /// `sext i32 %v to i64`, its spelling of a signed (`int`) index.
    Sext64,
    /// The `i32` itself.
    Narrow32,
}

impl IndexForm {
    /// The form of seed `seed`'s programs, drawn from a stream of its own, so a seed's program
    /// shape does not depend on it.
    pub fn of(seed: u64) -> IndexForm {
        let mut rng = Xorshift(seed ^ 0x6a09_e667_f3bc_c908 | 1);
        [IndexForm::Zext64, IndexForm::Sext64, IndexForm::Narrow32][rng.below(3)]
    }

    /// The index type.
    pub fn ty(self) -> &'static str {
        match self {
            IndexForm::Zext64 | IndexForm::Sext64 => "i64",
            IndexForm::Narrow32 => "i32",
        }
    }

    /// The line defining index `name` (with its `%`) from the `i32` value `value`.
    pub fn extend(self, name: &str, value: &str) -> String {
        match self {
            IndexForm::Zext64 => format!("  {name} = zext i32 {value} to i64\n"),
            IndexForm::Sext64 => format!("  {name} = sext i32 {value} to i64\n"),
            IndexForm::Narrow32 => format!("  {name} = add i32 {value}, 0\n"),
        }
    }
}

/// The seeds the ordinary test gate runs per generator.
pub const GATE_SEEDS: std::ops::RangeInclusive<u64> = 1..=24;

/// Fixed-seed xorshift: a failure names a seed that reproduces it.
pub struct Xorshift(pub u64);

impl Xorshift {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// One input buffer's initial contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Input {
    /// Little-endian `u32` words.
    Words(Vec<u32>),
    /// A pointer table: the Metal (guest) address of each listed kernel buffer, 8 bytes each, as
    /// an argument encoder writes it. Every listed buffer is an [`Input::Words`] input, and inputs
    /// are read only (`air.read`), so an executor may point the table at copies of their initial
    /// bytes.
    Addresses(Vec<usize>),
}

/// One generated kernel, ready to execute.
///
/// Buffer 0 is the output (`air.buffer` index 0, initially [`Kernel::output_init`]); buffers
/// `1..=inputs.len()` are the inputs. The dispatch is one threadgroup of [`Kernel::threads`]
/// threads.
pub struct Kernel<'a> {
    pub entry: &'a str,
    pub air: &'a str,
    pub inputs: &'a [Input],
    pub output_init: Vec<u8>,
    /// Threads in the dispatch's one threadgroup: the [`Target::threads`] the program was built
    /// for.
    pub threads: u32,
}

impl<'a> Kernel<'a> {
    /// A kernel of `threads` threads whose output buffer is `output_words` words per thread,
    /// filled with `0x9c`.
    pub fn filled(
        entry: &'a str,
        air: &'a str,
        inputs: &'a [Input],
        output_words: usize,
        threads: u32,
    ) -> Kernel<'a> {
        Kernel {
            entry,
            air,
            inputs,
            output_init: vec![0x9c; threads as usize * output_words * 4],
            threads,
        }
    }
}

/// What running one generated kernel produced.
pub struct Run {
    /// The output buffer's final contents as little-endian words.
    pub words: Vec<u32>,
}

/// Something that runs a [`Kernel`]: the product tests' `CpuInterp`, or validation's device
/// executor in its cfg-gated tests.
pub trait Execute {
    /// What the generators may build for this executor.
    fn target(&self) -> Target;

    /// Translate and run `kernel`. Every refusal, trap, or validation failure is an `Err`.
    fn execute(&mut self, kernel: &Kernel<'_>) -> Result<Run, String>;
}

/// The outcome of one generator over a seed range.
#[derive(Debug, Default)]
pub struct Report {
    /// Programs that ran to completion (whatever they computed).
    pub executed: usize,
    /// Programs a generator declined to build (for example a record with no array member), counted
    /// so a generator that stops producing anything is visible.
    pub declined: usize,
    /// Every divergence, refusal, or execution failure, one line each.
    pub failures: Vec<String>,
}

impl Report {
    /// Add `other`'s counts and failures to this report (a soak runs a seed range in chunks).
    pub fn absorb(&mut self, other: Report) {
        self.executed += other.executed;
        self.declined += other.declined;
        self.failures.extend(other.failures);
    }

    /// Panic with the failures (the first `shown` of them) unless the report is clean and at least
    /// one program executed.
    pub fn assert_clean(&self, what: &str, shown: usize) {
        assert!(
            self.failures.is_empty(),
            "{what}: {} failures over {} executed programs:\n{}",
            self.failures.len(),
            self.executed,
            self.failures
                .iter()
                .take(shown)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert!(
            self.executed > 0,
            "{what}: no generated program executed at all ({} declined)",
            self.declined
        );
    }
}

/// The AIR metadata tail: buffer 0 written, buffers `1..=inputs` read, `tid` from the grid.
pub fn kernel_metadata(entry: &str, inputs: usize) -> String {
    let mut nodes = String::from(
        "!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"output\"}\n",
    );
    let mut list = String::from("!3");
    for index in 1..=inputs {
        let node = 3 + index;
        list.push_str(&format!(", !{node}"));
        nodes.push_str(&format!(
            "!{node} = !{{i32 {index}, !\"air.buffer\", !\"air.location_index\", i32 {index}, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"in{index}\"}}\n"
        ));
    }
    let tid = 4 + inputs;
    list.push_str(&format!(", !{tid}"));
    nodes.push_str(&format!(
        "!{tid} = !{{i32 {inputs_plus}, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}}\n",
        inputs_plus = inputs + 1
    ));
    format!(
        "!air.kernel = !{{!0}}\n\
         !0 = !{{ptr @{entry}, !1, !2}}\n\
         !1 = !{{}}\n!2 = !{{{list}}}\n{nodes}"
    )
}

/// As [`kernel_metadata`], but buffer 0 is declared `air.read_write` -- a shader that stores into a
/// laid-out buffer and has its untouched bytes checked has to read them back, and a write-only
/// declaration is not that.
pub fn kernel_metadata_written(entry: &str, inputs: usize) -> String {
    kernel_metadata(entry, inputs).replace("!\"air.write\"", "!\"air.read_write\"")
}

/// The data layout the Metal front end emits, verbatim. The vector entries matter: without them a
/// `<3 x i32>` falls back to `next_power_of_two` for its alignment, and a generated module would be
/// laid out by a rule no real module uses.
pub const DATA_LAYOUT: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-\
    f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-\
    v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// The module header every generated program shares.
pub fn air_header(entry: &str) -> String {
    format!(
        "source_filename = \"{entry}.ll\"\n\
         target datalayout = \"{DATA_LAYOUT}\"\n\
         target triple = \"air64_v28-apple-macosx26.0.0\"\n\n"
    )
}

/// Little-endian bytes of `words`.
pub fn word_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// `bytes` as little-endian words (a trailing partial word is dropped).
pub fn bytes_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An executor that never runs anything: it refuses every kernel, or answers every kernel
    /// with the output buffer's initial words.
    struct Fake {
        refuse: bool,
        seen: usize,
    }

    impl Execute for Fake {
        fn target(&self) -> Target {
            Target::default()
        }

        fn execute(&mut self, kernel: &Kernel<'_>) -> Result<Run, String> {
            self.seen += 1;
            if self.refuse {
                return Err("refused".to_owned());
            }
            Ok(Run {
                words: bytes_words(&kernel.output_init),
            })
        }
    }

    type Check = fn(&mut dyn Execute, std::ops::RangeInclusive<u64>) -> Report;

    const CHECKS: [(&str, Check); 7] = [
        ("structured", structured::check),
        ("irreducible", irreducible::check),
        ("inline", inline::check),
        ("workgroup", workgroup::check),
        ("access load", access::check_load),
        ("access store", access::check_store),
        ("pointer select", pointer_select::check),
    ];

    fn fake(refuse: bool) -> Fake {
        Fake { refuse, seen: 0 }
    }

    #[test]
    fn a_refused_program_is_a_failure_not_a_skip() {
        for (name, check) in CHECKS {
            let mut executor = fake(true);
            let report = check(&mut executor, 1..=2);
            assert_eq!(report.executed, 0, "{name}");
            assert!(executor.seen > 0, "{name}: no program was built");
            assert_eq!(report.failures.len(), executor.seen, "{name}: {report:?}");
        }
        assert_eq!(access::check_race(&mut fake(true)).failures.len(), 1);
    }

    #[test]
    fn an_executor_that_computes_nothing_diverges_from_every_reference_model() {
        for (name, check) in CHECKS {
            let report = check(&mut fake(false), 1..=2);
            assert!(report.executed > 0, "{name}");
            assert!(!report.failures.is_empty(), "{name}: {report:?}");
        }
        assert_eq!(access::check_race(&mut fake(false)).failures.len(), 1);
    }

    #[test]
    fn a_generator_is_deterministic_per_seed() {
        let text = |seed: u64| {
            irreducible::Automaton::generate(seed, 4).air("determinism", IndexForm::Zext64)
        };
        assert_eq!(text(7), text(7));
        assert_ne!(text(7), text(8));
    }

    #[test]
    fn every_index_form_is_drawn_over_the_gate_seeds() {
        let forms: std::collections::BTreeSet<_> = GATE_SEEDS
            .map(|seed| format!("{:?}", IndexForm::of(seed)))
            .collect();
        assert_eq!(forms.len(), 3, "{forms:?}");
    }
}
