//! Differential execution of pointers that are CHOSEN at run time.
//!
//! One question: what a `select` or a `phi` between two pointers means when the two arms are
//! rooted in different buffers, at different offsets, in different address models. Authored cases
//! rarely choose a pointer at run time, and the bugs such a choice can hide -- a placeholder
//! pointer leaking as a value, a pointer table forwarding that loses its rawness, a phi whose arms
//! share a base collapsing to a spilled index -- each still yields a module that validates.
//!
//! So: generate a chain of pointer choices -- `select` between two buffers, a `phi` whose arms are
//! computed in different predecessors, a `getelementptr` applied to an already-selected pointer --
//! load through each one, and compare against the same chain resolved in Rust. Every index is
//! DATA DEPENDENT (derived from the accumulator, which comes from a buffer), so no arm folds away
//! and the choice really is made at run time.
//!
//! The last step is a selected STORE, and the whole output buffer is checked, not just the word
//! the thread meant to write -- a store through the wrong arm of a select writes the right value
//! into the wrong place, and only the word that should NOT have changed says so.
//!
//! **The engine** chooses between static-root fat pointers (the buffers themselves); in the
//! **parked** form, between the same roots stored into and loaded back from a private aggregate
//! (private pointer cells); and, in the **loaded** form, between wide pointers whose roots are
//! LOADED from a pointer table ([`Input::Addresses`]: the guest addresses of the three buffers), so
//! every choice, GEP, and access goes through the resident-set search and the root table, and a
//! lookup that missed would be counted and fail the run. The loaded form is built only for a
//! [`Target`] with device addresses (Profile A): without them the engine refuses a loaded
//! pointer's dereference, honestly. Every form may take the nullable choice.

use super::{
    air_header, kernel_metadata_written, Execute, IndexForm, Input, Kernel, Report, Target,
    Xorshift, OPS,
};

/// The index mask the generated program applies. A power of two, so the mask is exact.
const BUF_WORDS: usize = 64;
/// Words actually allocated per input buffer. The trailing `getelementptr` is applied AFTER the
/// mask -- that is the point of it, a step on an already-chosen pointer -- so the buffer has to
/// carry the extra words or the shader reads out of bounds while the Rust model wraps, and the
/// disagreement is the test's, not the translator's.
const BUF_LEN: usize = BUF_WORDS + 8;
/// Output words per thread: two candidate targets for the selected store, then the accumulator.
const OUT_WORDS: usize = 3;
/// Pointer choices per generated program.
const STEPS: usize = 4;

/// Which input buffer an arm is rooted in. Distinct roots are the point: a phi whose arms share a
/// base collapses to an index long before the pointer machinery sees it.
const ROOTS: [&str; 3] = ["%a", "%b", "%c"];

/// One pointer choice.
#[derive(Clone, Copy)]
struct Step {
    /// Bit of the accumulator that decides the arm.
    bit: u32,
    /// Buffer each arm is rooted in.
    roots: (usize, usize),
    /// Per-arm index offsets, added to the accumulator-derived index. Equal offsets with unequal
    /// roots is the case that separates "same address" from "same index".
    offsets: (usize, usize),
    /// `false` -> `select` between two pointers; `true` -> a `phi` whose arms are COMPUTED in
    /// different predecessors, which is a different lowering entirely.
    phi: bool,
    /// Apply a `getelementptr` to the already-chosen pointer (`select_gep.rs`'s subject).
    trailing_gep: usize,
    /// What to do with the loaded word.
    op: usize,
    /// Build the choice out of a NULLABLE select and a null test rather than one plain select:
    /// `select c, pa, null`, then `icmp eq ptr .., null`, then `select that, pb, ..`. The chosen
    /// pointer is the same either way, and the null test's answer -- true exactly when the false
    /// arm was taken -- is folded into the accumulator, so the nullness machinery is on the
    /// observed path instead of merely on the emitted one. Only for the `select` form: a phi's
    /// arms are defined in predecessors. On the engine the null test of a static root
    /// compares its slot's guest base, of a wide one its Metal address.
    nullable: bool,
}

/// Where the three roots come from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Roots {
    /// The buffer parameters themselves.
    Params,
    /// Parked in a local aggregate and read back, next to a float slot an integer atomic
    /// addresses: the aggregate is a private pointer cell. Each thread atomics its own slot, so the
    /// value it reads back is that slot's initial bits and the answer stays exact under any
    /// interleaving.
    Parked,
    /// Loaded from a pointer table buffer holding the buffers' guest addresses, so
    /// the choices are between wide pointers resolved through the resident set.
    Loaded,
}

struct Program {
    /// How the program spells its `getelementptr` indices.
    form: IndexForm,
    steps: Vec<Step>,
    /// Which of the thread's two candidate output words the final selected store targets, and the
    /// bit that decides it.
    store_bit: u32,
    roots: Roots,
}

impl Program {
    fn generate(seed: u64, target: Target, form: IndexForm) -> Self {
        let mut rng = Xorshift(seed | 1);
        let steps = (0..STEPS)
            .map(|_| {
                let first = rng.below(8);
                Step {
                    bit: rng.below(5) as u32,
                    roots: (rng.below(ROOTS.len()), rng.below(ROOTS.len())),
                    // Half the time both arms take the SAME offset, so the only thing that can
                    // distinguish the two pointers is which buffer they are rooted in.
                    offsets: (
                        first,
                        if rng.below(2) == 0 {
                            first
                        } else {
                            rng.below(8)
                        },
                    ),
                    phi: rng.below(2) == 1,
                    trailing_gep: rng.below(4),
                    op: rng.below(OPS.len()),
                    nullable: rng.below(2) == 1,
                }
            })
            .collect();
        let store_bit = rng.below(5) as u32;
        let indirect = rng.below(2) == 1;
        // The draw between the parked and the loaded form comes last, so the steps of a seed's
        // program are the same whichever form it takes.
        let roots = match indirect {
            false => Roots::Params,
            true if target.addressed && rng.below(2) == 1 => Roots::Loaded,
            true => Roots::Parked,
        };
        Program {
            form,
            steps,
            store_bit,
            roots,
        }
    }

    /// Resolve the chain in Rust. `buffers[r][i]` is input buffer `r` word `i`.
    fn run(&self, buffers: &[Vec<u32>; 3], seed: u32) -> (u32, usize) {
        let mut acc = seed;
        for (site, step) in self.steps.iter().enumerate() {
            let taken = (acc >> step.bit) & 1 == 1;
            let (root, offset) = if taken {
                (step.roots.0, step.offsets.0)
            } else {
                (step.roots.1, step.offsets.1)
            };
            let base = (acc as usize + offset) & (BUF_WORDS - 1);
            let loaded = buffers[root][base + step.trailing_gep];
            acc = (OPS[step.op].1)(acc) ^ loaded ^ (site as u32 + 1);
            if step.nullable && !step.phi {
                // The null test is true exactly when the false arm was taken.
                acc ^= u32::from(!taken) << 8;
            }
        }
        let slot = ((acc >> self.store_bit) & 1) as usize;
        (acc, slot)
    }

    /// The input buffers after the three roots: the float atomic slot (parked) or the pointer
    /// table (loaded).
    fn extra_inputs(&self) -> usize {
        match self.roots {
            Roots::Params => 0,
            Roots::Parked | Roots::Loaded => 1,
        }
    }

    fn air(&self, entry: &str) -> String {
        let ty = self.form.ty();
        let mut out = air_header(entry);
        let extra = match self.roots {
            Roots::Params => "",
            Roots::Parked => "ptr addrspace(1) %d, ",
            Roots::Loaded => "ptr addrspace(1) %t, ",
        };
        out.push_str(&format!(
            "define void @{entry}(ptr addrspace(1) %out, ptr addrspace(1) %a, ptr addrspace(1) %b, \
             ptr addrspace(1) %c, {extra}i32 %tid) {{\nentry:\n"
        ));
        let roots: [String; ROOTS.len()] = match self.roots {
            Roots::Params => std::array::from_fn(|index| ROOTS[index].to_string()),
            Roots::Parked => {
                let fields = ROOTS
                    .iter()
                    .map(|_| "ptr addrspace(1)")
                    .collect::<Vec<_>>()
                    .join(", ");
                let aggregate = format!("{{ {fields} }}");
                out.push_str(&format!("  %cache = alloca {aggregate}, align 8\n"));
                for (index, root) in ROOTS.iter().enumerate() {
                    out.push_str(&format!(
                        "  %fld{index} = getelementptr inbounds {aggregate}, ptr %cache, {ty} 0, i32 {index}\n"
                    ));
                    out.push_str(&format!(
                        "  store ptr addrspace(1) {root}, ptr %fld{index}, align 8\n"
                    ));
                }
                for index in 0..ROOTS.len() {
                    out.push_str(&format!(
                        "  %rt{index} = load ptr addrspace(1), ptr %fld{index}, align 8\n"
                    ));
                }
                std::array::from_fn(|index| format!("%rt{index}"))
            }
            Roots::Loaded => {
                for index in 0..ROOTS.len() {
                    out.push_str(&format!(
                        "  %tp{index} = getelementptr inbounds ptr addrspace(1), ptr addrspace(1) %t, {ty} {index}\n"
                    ));
                    out.push_str(&format!(
                        "  %rt{index} = load ptr addrspace(1), ptr addrspace(1) %tp{index}, align 8\n"
                    ));
                }
                std::array::from_fn(|index| format!("%rt{index}"))
            }
        };
        out.push_str(&self.form.extend("%slot", "%tid"));
        out.push_str(&format!(
            "  %seedp = getelementptr inbounds i32, ptr addrspace(1) {}, {ty} %slot\n",
            roots[0]
        ));
        if self.roots == Roots::Parked {
            out.push_str("  %acc0z = load i32, ptr addrspace(1) %seedp, align 4\n");
            out.push_str(&format!(
                "  %dp = getelementptr inbounds float, ptr addrspace(1) %d, {ty} %slot\n"
            ));
            out.push_str("  %db = bitcast ptr addrspace(1) %dp to ptr addrspace(1)\n");
            out.push_str(
                "  %dold = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %db, \
                 i32 1, i32 0, i32 2, i1 true)\n",
            );
            out.push_str("  %acc0 = xor i32 %acc0z, %dold\n");
        } else {
            out.push_str("  %acc0 = load i32, ptr addrspace(1) %seedp, align 4\n");
        }
        let mut acc = "%acc0".to_string();
        for (site, step) in self.steps.iter().enumerate() {
            let n = site;
            // The arm test and both arm indices, all derived from the accumulator so nothing folds.
            out.push_str(&format!("  %sh{n} = lshr i32 {acc}, {}\n", step.bit));
            out.push_str(&format!("  %ms{n} = and i32 %sh{n}, 1\n"));
            out.push_str(&format!("  %cd{n} = icmp eq i32 %ms{n}, 1\n"));
            for (arm, offset) in [("t", step.offsets.0), ("f", step.offsets.1)] {
                out.push_str(&format!("  %o{arm}{n} = add i32 {acc}, {offset}\n"));
                out.push_str(&format!(
                    "  %w{arm}{n} = and i32 %o{arm}{n}, {}\n",
                    BUF_WORDS - 1
                ));
                out.push_str(
                    &self
                        .form
                        .extend(&format!("%x{arm}{n}"), &format!("%w{arm}{n}")),
                );
            }
            let (rt, rf) = (&roots[step.roots.0], &roots[step.roots.1]);
            if step.phi {
                let (t, f, m) = (format!("pt{n}"), format!("pf{n}"), format!("pm{n}"));
                out.push_str(&format!("  br i1 %cd{n}, label %{t}, label %{f}\n"));
                out.push_str(&format!("{t}:\n"));
                out.push_str(&format!(
                    "  %pa{n} = getelementptr inbounds i32, ptr addrspace(1) {rt}, {ty} %xt{n}\n"
                ));
                out.push_str(&format!("  br label %{m}\n"));
                out.push_str(&format!("{f}:\n"));
                out.push_str(&format!(
                    "  %pb{n} = getelementptr inbounds i32, ptr addrspace(1) {rf}, {ty} %xf{n}\n"
                ));
                out.push_str(&format!("  br label %{m}\n"));
                out.push_str(&format!("{m}:\n"));
                out.push_str(&format!(
                    "  %p{n} = phi ptr addrspace(1) [ %pa{n}, %{t} ], [ %pb{n}, %{f} ]\n"
                ));
            } else {
                out.push_str(&format!(
                    "  %pa{n} = getelementptr inbounds i32, ptr addrspace(1) {rt}, {ty} %xt{n}\n"
                ));
                out.push_str(&format!(
                    "  %pb{n} = getelementptr inbounds i32, ptr addrspace(1) {rf}, {ty} %xf{n}\n"
                ));
                if step.nullable {
                    out.push_str(&format!(
                        "  %pn{n} = select i1 %cd{n}, ptr addrspace(1) %pa{n}, ptr addrspace(1) null\n"
                    ));
                    out.push_str(&format!(
                        "  %zq{n} = icmp eq ptr addrspace(1) %pn{n}, null\n"
                    ));
                    out.push_str(&format!(
                        "  %p{n} = select i1 %zq{n}, ptr addrspace(1) %pb{n}, ptr addrspace(1) %pn{n}\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "  %p{n} = select i1 %cd{n}, ptr addrspace(1) %pa{n}, ptr addrspace(1) %pb{n}\n"
                    ));
                }
            }
            // A `getelementptr` on an ALREADY-CHOSEN pointer: the chain has to be rebuilt through
            // whichever arm was taken, which is `select_gep.rs`'s whole subject. Both arms index
            // the same buffer they were rooted in, so the step is wrap-safe by the mask above.
            out.push_str(&format!(
                "  %q{n} = getelementptr inbounds i32, ptr addrspace(1) %p{n}, {ty} {}\n",
                step.trailing_gep
            ));
            out.push_str(&format!(
                "  %v{n} = load i32, ptr addrspace(1) %q{n}, align 4\n"
            ));
            out.push_str(&format!(
                "  %r{n} = {}\n",
                OPS[step.op].0.replace("%ACC", &acc)
            ));
            out.push_str(&format!("  %m1{n} = xor i32 %r{n}, %v{n}\n"));
            if step.nullable && !step.phi {
                out.push_str(&format!("  %m2{n} = xor i32 %m1{n}, {}\n", site + 1));
                out.push_str(&format!("  %ez{n} = zext i1 %zq{n} to i32\n"));
                out.push_str(&format!("  %es{n} = shl i32 %ez{n}, 8\n"));
                out.push_str(&format!("  %acc{} = xor i32 %m2{n}, %es{n}\n", n + 1));
            } else {
                out.push_str(&format!("  %acc{} = xor i32 %m1{n}, {}\n", n + 1, site + 1));
            }
            acc = format!("%acc{}", n + 1);
        }
        // The selected store: two candidate words in this thread's own slice of the output, one
        // chosen by the accumulator. Every other word must be untouched.
        out.push_str(&format!("  %base = mul i32 %tid, {OUT_WORDS}\n"));
        out.push_str(&format!("  %sb = lshr i32 {acc}, {}\n", self.store_bit));
        out.push_str("  %sm = and i32 %sb, 1\n");
        out.push_str("  %sc = icmp eq i32 %sm, 1\n");
        out.push_str(&self.form.extend("%i0", "%base"));
        out.push_str("  %b1 = add i32 %base, 1\n");
        out.push_str(&self.form.extend("%i1", "%b1"));
        out.push_str(&format!(
            "  %s0 = getelementptr inbounds i32, ptr addrspace(1) %out, {ty} %i0\n"
        ));
        out.push_str(&format!(
            "  %s1 = getelementptr inbounds i32, ptr addrspace(1) %out, {ty} %i1\n"
        ));
        out.push_str("  %sp = select i1 %sc, ptr addrspace(1) %s1, ptr addrspace(1) %s0\n");
        out.push_str(&format!(
            "  store i32 {acc}, ptr addrspace(1) %sp, align 4\n"
        ));
        out.push_str(&format!("  %b2 = add i32 %base, {}\n", OUT_WORDS - 1));
        out.push_str(&self.form.extend("%i2", "%b2"));
        out.push_str(&format!(
            "  %s2 = getelementptr inbounds i32, ptr addrspace(1) %out, {ty} %i2\n"
        ));
        out.push_str(&format!(
            "  store i32 {acc}, ptr addrspace(1) %s2, align 4\n"
        ));
        out.push_str("  ret void\n}\n\n");
        match self.roots {
            Roots::Params => out.push_str(&kernel_metadata_written(entry, 3)),
            Roots::Parked => {
                out.push_str(
                    "declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)\n\n",
                );
                // The fourth buffer is the float slot the integer atomic addresses, and the
                // atomic writes it, so it is read_write rather than read.
                out.push_str(
                    &kernel_metadata_written(entry, 4)
                        .replace(
                            "i32 4, i32 1, !\"air.read\",",
                            "i32 4, i32 1, !\"air.read_write\",",
                        )
                        .replace(
                            "!\"uint\", !\"air.arg_name\", !\"in4\"",
                            "!\"float\", !\"air.arg_name\", !\"in4\"",
                        ),
                );
            }
            Roots::Loaded => out.push_str(&kernel_metadata_written(entry, 4)),
        }
        out
    }
}

/// Distinct contents per buffer, with no word repeated within one of them, so a load from the wrong
/// buffer or the wrong index cannot coincidentally give the right answer.
fn buffer_contents(root: usize) -> Vec<u32> {
    (0..BUF_LEN)
        .map(|index| {
            (index as u32)
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add(root as u32 * 0x0100_0193 + 1)
        })
        .collect()
}

/// The pointer table of the loaded form: the guest addresses of kernel buffers 1, 2, and 3 (the
/// three roots).
fn pointer_table() -> Input {
    Input::Addresses((1..=ROOTS.len()).collect())
}

/// Run every seed's program on `executor` and compare the whole output buffer with the Rust
/// resolution of the chain.
pub fn check(executor: &mut dyn Execute, seeds: std::ops::RangeInclusive<u64>) -> Report {
    let buffers = [buffer_contents(0), buffer_contents(1), buffer_contents(2)];
    // The float slot the parked form atomics. Its initial bits are what each thread reads back,
    // so they are part of that form's seed.
    let atomic_slots = buffer_contents(3);
    let target = executor.target();
    let threads = target.threads as usize;
    let fill: Vec<u8> = (0..threads * OUT_WORDS * 4)
        .map(|index| ((index as u64).wrapping_mul(61).wrapping_add(7) & 0xff) as u8)
        .collect();
    let mut report = Report::default();
    for seed in seeds {
        let form = IndexForm::of(seed);
        let program = Program::generate(seed.wrapping_mul(0x9e3779b97f4a7c15), target, form);
        let mut want: Vec<u32> = fill
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let seed_of = |thread: usize| {
            buffers[0][thread]
                ^ if program.roots == Roots::Parked {
                    atomic_slots[thread]
                } else {
                    0
                }
        };
        for thread in 0..threads {
            let (acc, slot) = program.run(&buffers, seed_of(thread));
            want[thread * OUT_WORDS + slot] = acc;
            want[thread * OUT_WORDS + OUT_WORDS - 1] = acc;
        }
        let entry = format!("ptrsel_s{seed}");
        let air = program.air(&entry);
        let mut inputs: Vec<Input> = buffers.iter().cloned().map(Input::Words).collect();
        match program.roots {
            Roots::Params => {}
            Roots::Parked => inputs.push(Input::Words(atomic_slots.clone())),
            Roots::Loaded => inputs.push(pointer_table()),
        }
        debug_assert_eq!(inputs.len(), ROOTS.len() + program.extra_inputs());
        let kernel = Kernel {
            output_init: fill.clone(),
            ..Kernel::filled(&entry, &air, &inputs, 0, target.threads)
        };
        let run = match executor.execute(&kernel) {
            Ok(run) => run,
            Err(error) => {
                report.failures.push(format!("seed={seed}: {error}"));
                continue;
            }
        };
        report.executed += 1;
        for (index, (got, wanted)) in run.words.iter().zip(&want).enumerate() {
            if got != wanted {
                report.failures.push(format!(
                    "seed={seed} out[{index}] (thread {} word {}): got {got}, wanted {wanted}",
                    index / OUT_WORDS,
                    index % OUT_WORDS
                ));
            }
        }
        if run.words.len() != want.len() {
            report.failures.push(format!(
                "seed={seed}: read back {} words, wanted {}",
                run.words.len(),
                want.len()
            ));
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's programs take every root form over the gate seeds, the loaded form only
    /// on a device with device addresses, and nullable choices in all of them.
    #[test]
    fn the_engine_programs_cover_every_root_form_the_device_can_run() {
        let program = |seed: u64, target: Target| {
            Program::generate(
                seed.wrapping_mul(0x9e3779b97f4a7c15),
                target,
                IndexForm::Zext64,
            )
        };
        let count = |target: Target| {
            let mut counts = [0usize; 3];
            for seed in crate::fuzz::GATE_SEEDS {
                counts[program(seed, target).roots as usize] += 1;
            }
            counts
        };
        let addressed = count(Target::default());
        assert!(addressed.iter().all(|&n| n > 0), "{addressed:?}");
        let nullable = crate::fuzz::GATE_SEEDS
            .filter(|&seed| {
                let p = program(seed, Target::default());
                p.roots != Roots::Params && p.steps.iter().any(|s| s.nullable && !s.phi)
            })
            .count();
        assert!(nullable > 0);
        let floor = count(Target {
            addressed: false,
            ..Target::default()
        });
        assert!(floor[Roots::Loaded as usize] == 0 && floor[Roots::Parked as usize] > 0);
    }
}
