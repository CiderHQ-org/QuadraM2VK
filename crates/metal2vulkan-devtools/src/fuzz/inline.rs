//! Differential execution of the same computation written across helpers and written flat.
//!
//! Authored cases are picked as the smallest host for a symbol, and the smallest host is almost
//! always a single function, so they rarely call a helper. These programs do: every stage hands its
//! helper a pointer chosen by a `select` between two DIFFERENT bindings, which has no standalone
//! Logical SPIR-V pointer value. Canonicalization inlines the helpers into the kernel, and the
//! access family lowers the pointer select; both forms are compiled, validated, and run.
//!
//! Inlining fails by a NAME, not an arithmetic slip: a callee value that collides with a caller
//! value, a label that collides with a caller label, a phi whose incoming label was renamed in one
//! place and not the other -- each produces a module that still parses and still validates. So
//! every helper here is spelled with the SAME `%v2..%v19` and the SAME `odd`/`even`/`join`/
//! `loop`/`body`/`done` labels as every other helper and as the kernel.
//!
//! Each program is emitted TWICE: with the stages called as helpers, and with the same stage
//! bodies written straight into the kernel where there is no call to inline. Both are compared
//! against the computation done directly in Rust, so a divergence says which form is wrong.

use super::{
    air_header, kernel_metadata, Execute, IndexForm, Input, Kernel, Report, Xorshift, OPS,
};

/// The two tables a stage's pointer select chooses between. They are DIFFERENT buffers, which is
/// what makes the select a pointer select rather than an index select -- an index select would be
/// folded upstream and never reach the inliner.
const TABLE_A: [u32; 8] = [3, 5, 7, 11, 13, 17, 19, 23];
const TABLE_B: [u32; 8] = [29, 31, 37, 41, 43, 47, 53, 59];

/// One stage. Every stage does the same four things in the same order; only which operations, and
/// which accumulator bits, differ.
#[derive(Clone, Copy)]
struct Stage {
    first: usize,
    refine: usize,
    bit: u32,
    /// The accumulator bit that picks which table the stage's pointer select names.
    table_bit: u32,
    /// A looping stage makes the inliner splice a multi-block body WITH A BACK EDGE into the
    /// middle of its caller, which is the case its label renaming has the most to do in.
    looping: bool,
}

struct Program {
    stages: Vec<Stage>,
    /// How the kernel and every helper spell their `getelementptr` indices.
    form: IndexForm,
}

impl Program {
    fn generate(seed: u64, count: usize, form: IndexForm) -> Self {
        let mut rng = Xorshift(seed | 1);
        let looping_at = rng.below(count.max(1));
        Program {
            stages: (0..count)
                .map(|index| Stage {
                    first: rng.below(OPS.len()),
                    refine: rng.below(OPS.len()),
                    bit: rng.below(6) as u32,
                    table_bit: rng.below(6) as u32,
                    // Exactly one looping stage, at a seed-chosen position, so across the sweep the
                    // loop lands first, in the middle and last.
                    looping: index == looping_at,
                })
                .collect(),
            form,
        }
    }

    /// The computation, done directly. `acc` starts at the thread's index.
    fn run(&self, thread: u32) -> u32 {
        let mut acc = thread;
        for stage in &self.stages {
            // The table is chosen from the accumulator the stage STARTS with, because that is
            // where the kernel computes the select -- the helper is handed a pointer that was
            // already selected. Getting this backwards here is what the flat control form caught:
            // both GPU forms agreed with each other and disagreed with this function.
            let table = if (acc >> stage.table_bit) & 1 == 1 {
                TABLE_B
            } else {
                TABLE_A
            };
            acc = (OPS[stage.first].1)(acc);
            if (acc >> stage.bit) & 1 == 1 {
                acc = (OPS[stage.refine].1)(acc);
            }
            if stage.looping {
                for _ in 0..(acc & 7) {
                    acc = (OPS[stage.refine].1)(acc);
                }
            }
            acc = acc.wrapping_add(table[(acc & 7) as usize]);
        }
        acc
    }

    /// The body of one stage, with every local name and label given `suffix`.
    ///
    /// The suffix is a PARAMETER rather than a rename applied afterwards, and that is deliberate:
    /// the first version of this file renamed the flat form with `str::replace` and turned `%17`
    /// into `%1_f00` and `%done` into `%d_f0one`. A control that has to solve the same problem the
    /// code under test solves is not a control.
    ///
    /// `acc` is the incoming accumulator and `ptr` the already-selected buffer pointer. Returns the
    /// lines and the name holding the result.
    fn stage_body(&self, index: usize, suffix: &str, acc: &str, ptr: &str) -> (String, String) {
        let stage = self.stages[index];
        let v = |n: usize| format!("%v{n}{suffix}");
        let l = |name: &str| format!("{name}{suffix}");
        let mut out = String::new();
        out.push_str(&format!(
            "  {} = {}\n",
            v(2),
            OPS[stage.first].0.replace("%ACC", acc)
        ));
        out.push_str(&format!("  {} = lshr i32 {}, {}\n", v(3), v(2), stage.bit));
        out.push_str(&format!("  {} = and i32 {}, 1\n", v(4), v(3)));
        out.push_str(&format!("  {} = icmp eq i32 {}, 1\n", v(5), v(4)));
        out.push_str(&format!(
            "  br i1 {}, label %{}, label %{}\n\n",
            v(5),
            l("odd"),
            l("even")
        ));
        out.push_str(&format!("{}:\n", l("odd")));
        out.push_str(&format!(
            "  {} = {}\n",
            v(6),
            OPS[stage.refine].0.replace("%ACC", &v(2))
        ));
        out.push_str(&format!("  br label %{}\n\n", l("join")));
        out.push_str(&format!("{}:\n", l("even")));
        out.push_str(&format!("  br label %{}\n\n", l("join")));
        out.push_str(&format!("{}:\n", l("join")));
        out.push_str(&format!(
            "  {} = phi i32 [ {}, %{} ], [ {}, %{} ]\n",
            v(7),
            v(6),
            l("odd"),
            v(2),
            l("even")
        ));
        let refined = if stage.looping {
            out.push_str(&format!("  {} = and i32 {}, 7\n", v(8), v(7)));
            out.push_str(&format!("  br label %{}\n\n", l("loop")));
            out.push_str(&format!("{}:\n", l("loop")));
            out.push_str(&format!(
                "  {} = phi i32 [ 0, %{} ], [ {}, %{} ]\n",
                v(9),
                l("join"),
                v(11),
                l("body")
            ));
            out.push_str(&format!(
                "  {} = phi i32 [ {}, %{} ], [ {}, %{} ]\n",
                v(10),
                v(7),
                l("join"),
                v(12),
                l("body")
            ));
            out.push_str(&format!("  {} = icmp ult i32 {}, {}\n", v(13), v(9), v(8)));
            out.push_str(&format!(
                "  br i1 {}, label %{}, label %{}\n\n",
                v(13),
                l("body"),
                l("done")
            ));
            out.push_str(&format!("{}:\n", l("body")));
            out.push_str(&format!("  {} = add i32 {}, 1\n", v(11), v(9)));
            out.push_str(&format!(
                "  {} = {}\n",
                v(12),
                OPS[stage.refine].0.replace("%ACC", &v(10))
            ));
            out.push_str(&format!("  br label %{}\n\n", l("loop")));
            out.push_str(&format!("{}:\n", l("done")));
            v(10)
        } else {
            v(7)
        };
        out.push_str(&format!("  {} = and i32 {refined}, 7\n", v(14)));
        out.push_str(&self.form.extend(&v(16), &v(14)));
        out.push_str(&format!(
            "  {} = getelementptr inbounds i32, ptr addrspace(1) {ptr}, {} {}\n",
            v(17),
            self.form.ty(),
            v(16)
        ));
        out.push_str(&format!(
            "  {} = load i32, ptr addrspace(1) {}, align 4\n",
            v(18),
            v(17)
        ));
        out.push_str(&format!("  {} = add i32 {refined}, {}\n", v(19), v(18)));
        (out, v(19))
    }

    /// The lines that pick a stage's table, in the KERNEL, from the accumulator the stage has not
    /// run on yet. Both forms compute the select identically and only differ in whether the code
    /// that consumes the resulting pointer is a call or is written in place -- which is the whole
    /// difference the inliner is responsible for erasing.
    fn select_lines(&self, index: usize, suffix: &str, acc: &str) -> (String, String) {
        let stage = self.stages[index];
        let mut out = String::new();
        out.push_str(&format!(
            "  %s0{suffix} = lshr i32 {acc}, {}\n",
            stage.table_bit
        ));
        out.push_str(&format!("  %s1{suffix} = and i32 %s0{suffix}, 1\n"));
        out.push_str(&format!("  %s2{suffix} = icmp eq i32 %s1{suffix}, 1\n"));
        out.push_str(&format!(
            "  %s3{suffix} = select i1 %s2{suffix}, ptr addrspace(1) %tb, ptr addrspace(1) %ta\n"
        ));
        (out, format!("%s3{suffix}"))
    }

    /// The kernel preamble. Its locals are deliberately named `%v2..%v5`, the SAME names every
    /// helper uses, so inlining a helper collides with the caller and not only with the other
    /// helpers.
    fn kernel_header(&self, entry: &str) -> String {
        format!(
            "define void @{entry}(ptr addrspace(1) %output, ptr addrspace(1) %ta, ptr addrspace(1) %tb, ptr addrspace(1) %seeds, i32 %tid) {{\nentry:\n\
             {}\
             \x20 %v3 = getelementptr inbounds i32, ptr addrspace(1) %seeds, {} %v2\n\
             \x20 %v4 = load i32, ptr addrspace(1) %v3, align 4\n",
            self.form.extend("%v2", "%tid"),
            self.form.ty()
        )
    }

    fn kernel_tail(&self, entry: &str, carried: &str) -> String {
        format!(
            "  %v5 = getelementptr inbounds i32, ptr addrspace(1) %output, {} %v2\n\
             \x20 store i32 {carried}, ptr addrspace(1) %v5, align 4\n\
             \x20 ret void\n}}\n\n{}",
            self.form.ty(),
            kernel_metadata(entry, 3)
        )
    }

    /// The helper form: one `define internal` per stage, all spelled with the same names, each
    /// called with a pointer the kernel just selected between two distinct bindings.
    fn air_with_helpers(&self, entry: &str) -> String {
        let mut out = air_header(entry);
        for index in 0..self.stages.len() {
            let (body, result) = self.stage_body(index, "", "%1", "%0");
            out.push_str(&format!(
                "define internal fastcc i32 @stage{index}(ptr addrspace(1) %0, i32 %1) {{\nentry:\n"
            ));
            out.push_str(&body);
            out.push_str(&format!("  ret i32 {result}\n}}\n\n"));
        }
        out.push_str(&self.kernel_header(entry));
        let mut carried = "%v4".to_string();
        for index in 0..self.stages.len() {
            let (lines, pointer) = self.select_lines(index, &format!("_k{index}"), &carried);
            out.push_str(&lines);
            let result = format!("%r{index}");
            out.push_str(&format!(
                "  {result} = call fastcc i32 @stage{index}(ptr addrspace(1) {pointer}, i32 {carried})\n"
            ));
            carried = result;
        }
        out.push_str(&self.kernel_tail(entry, &carried));
        out
    }

    /// The flat form: the same stage bodies written straight into the kernel, consuming the same
    /// pointer selects. There is no call, so there is nothing for the inliner to do -- this is the
    /// control that says a divergence is the inliner's and not the generator's.
    fn air_flat(&self, entry: &str) -> String {
        let mut out = air_header(entry);
        out.push_str(&self.kernel_header(entry));
        let mut carried = "%v4".to_string();
        for index in 0..self.stages.len() {
            let suffix = format!("_f{index}");
            let (lines, pointer) = self.select_lines(index, &suffix, &carried);
            out.push_str(&lines);
            let (body, result) = self.stage_body(index, &suffix, &carried, &pointer);
            out.push_str(&body);
            carried = result;
        }
        out.push_str(&self.kernel_tail(entry, &carried));
        out
    }
}

/// Bindings 1, 2 and 3: the two tables the selects choose between, and the per-thread seeds of a
/// `threads`-thread dispatch.
fn inputs(threads: u32) -> Vec<Input> {
    vec![
        Input::Words(TABLE_A.to_vec()),
        Input::Words(TABLE_B.to_vec()),
        Input::Words((0..threads).collect()),
    ]
}

/// Run the generator for every seed in `seeds` (one to three stages each) on `executor`, in both
/// the helper form and the flat form.
pub fn check(executor: &mut dyn Execute, seeds: std::ops::RangeInclusive<u64>) -> Report {
    let mut report = Report::default();
    let target = executor.target();
    let inputs = inputs(target.threads);
    for seed in seeds {
        let form = IndexForm::of(seed);
        for stages in 1usize..=3 {
            let program = Program::generate(seed.wrapping_mul(0x9e3779b97f4a7c15), stages, form);
            let entry = format!("inline_s{seed}_n{stages}");
            let mut forms = Vec::with_capacity(2);
            for (form, air) in [
                ("helper", program.air_with_helpers(&entry)),
                ("flat", program.air_flat(&entry)),
            ] {
                match executor.execute(&Kernel::filled(&entry, &air, &inputs, 1, target.threads)) {
                    Ok(run) => forms.push(run.words),
                    Err(error) => report
                        .failures
                        .push(format!("seed={seed} stages={stages} {form} form: {error}")),
                }
            }
            let [helper, flat] = forms.as_slice() else {
                continue;
            };
            report.executed += 1;
            for thread in 0..target.threads as usize {
                let want = program.run(thread as u32);
                if helper[thread] != want || flat[thread] != want {
                    report.failures.push(format!(
                        "seed={seed} stages={stages} thread={thread}: helpers gave {}, flat gave \
                         {}, the computation gives {want}",
                        helper[thread], flat[thread]
                    ));
                }
            }
        }
    }
    report
}
