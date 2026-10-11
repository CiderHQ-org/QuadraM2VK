//! The minimizer: a smaller sequence that still fails as a seed's did, to report beside the seed.
//! It removes what it can, coarse first (a command buffer, an encoder, an event or a fence with
//! every use of it, a command), keeping each removal that leaves the sequence race-free
//! ([`problems`]) and still failing, until no removal keeps the failure or its attempts run out.
//!
//! Every removal keeps the model's arguments valid: it removes writes only, and every write
//! keeps a word of a role valid ([`super::program::Role`]). What fails is the caller's question
//! (on the runtime: bytes that differ from the model's, or the validation layer's report), so the
//! minimizer itself is plain data work.

use super::order::problems;
use super::program::{Kind, Program, Step};

/// What a minimization made: the smallest failing sequence found, and the candidates it asked
/// about.
pub(crate) struct Minimized {
    pub(crate) program: Program,
    pub(crate) attempts: usize,
}

/// A smaller sequence than `program` for which `fails` still holds, asking about at most
/// `attempts` race-free candidates. `program` itself is taken to fail.
pub(crate) fn minimize(
    program: &Program,
    attempts: usize,
    mut fails: impl FnMut(&Program) -> bool,
) -> Minimized {
    let mut current = program.clone();
    let mut asked = 0;
    let mut start = 0;
    let mut shrunk = false;
    loop {
        let candidates = removals(&current);
        let mut kept = None;
        for (at, candidate) in candidates.into_iter().enumerate().skip(start) {
            let candidate = tidy(candidate);
            let removed = candidate.command_buffers != current.command_buffers;
            if !removed || !problems(&candidate).is_empty() {
                continue;
            }
            if asked == attempts {
                return Minimized {
                    program: current,
                    attempts: asked,
                };
            }
            asked += 1;
            if fails(&candidate) {
                kept = Some((at, candidate));
                break;
            }
        }
        match kept {
            // Go on from the same place: the removals before it were refused already.
            Some((at, candidate)) => {
                current = candidate;
                start = at;
                shrunk = true;
            }
            // A pass that removed something earns another from the start.
            None if shrunk && start > 0 => {
                start = 0;
                shrunk = false;
            }
            None => {
                return Minimized {
                    program: current,
                    attempts: asked,
                }
            }
        }
    }
}

/// Every sequence one removal from `program`, coarse first: each command buffer, each encoder,
/// each event (its signal and its waits), each fence (its updates and its waits), each command.
pub(crate) fn removals(program: &Program) -> Vec<Program> {
    let mut candidates = Vec::new();
    for cb in 0..program.command_buffers.len() {
        let mut candidate = program.clone();
        candidate.command_buffers.remove(cb);
        candidates.push(candidate);
    }
    for (cb, step, _) in program.encoders() {
        let mut candidate = program.clone();
        candidate.command_buffers[cb].steps.remove(step);
        candidates.push(candidate);
    }
    for event in 0..program.events {
        let mut candidate = program.clone();
        for spec in &mut candidate.command_buffers {
            spec.steps
                .retain(|step| !matches!(step, Step::Signal(e) | Step::Wait(e) if *e == event));
        }
        candidates.push(candidate);
    }
    for fence in 0..program.fences {
        let mut candidate = program.clone();
        for_each_encoder(&mut candidate, |encoder| {
            encoder.updates.retain(|&f| f != fence);
            encoder.waits.retain(|&f| f != fence);
        });
        candidates.push(candidate);
    }
    for (cb, step, encoder) in program.encoders() {
        for command in 0..encoder.commands.len() {
            let mut candidate = program.clone();
            if let Step::Encoder(encoder) = &mut candidate.command_buffers[cb].steps[step] {
                encoder.commands.remove(command);
            }
            candidates.push(candidate);
        }
    }
    candidates
}

fn for_each_encoder(program: &mut Program, mut each: impl FnMut(&mut super::program::Encoder)) {
    for spec in &mut program.command_buffers {
        for step in &mut spec.steps {
            if let Step::Encoder(encoder) = step {
                each(encoder);
            }
        }
    }
}

/// `program` without what a removal left without purpose: a blit or compute encoder with no
/// command but barriers, a fence wait with no update before it in its command buffer, a fence
/// update no encoder waits for, an event wait with no signal, a signal no command buffer waits
/// for, and a command buffer with no step; the events and fences left numbered from 0.
pub(crate) fn tidy(mut program: Program) -> Program {
    loop {
        let before = program.clone();
        for spec in &mut program.command_buffers {
            spec.steps.retain(|step| match step {
                Step::Encoder(encoder) => {
                    matches!(encoder.kind, Kind::Render { .. })
                        || encoder
                            .commands
                            .iter()
                            .any(|c| !matches!(c, super::program::Command::Barrier { .. }))
                }
                _ => true,
            });
        }
        // Fences: a wait needs an update before it in its command buffer, an update a wait.
        for spec in &mut program.command_buffers {
            let mut updated: Vec<usize> = Vec::new();
            for step in &mut spec.steps {
                if let Step::Encoder(encoder) = step {
                    encoder.waits.retain(|fence| updated.contains(fence));
                    updated.extend(&encoder.updates);
                }
            }
        }
        let mut waited: Vec<usize> = Vec::new();
        for_each_encoder(&mut program, |encoder| waited.extend(&encoder.waits));
        for_each_encoder(&mut program, |encoder| {
            encoder.updates.retain(|fence| waited.contains(fence))
        });
        // Events: a wait needs its signal, a signal a wait.
        let mut signalled = Vec::new();
        let mut awaited = Vec::new();
        for spec in &program.command_buffers {
            for step in &spec.steps {
                match step {
                    Step::Signal(event) => signalled.push(*event),
                    Step::Wait(event) => awaited.push(*event),
                    Step::Encoder(_) => {}
                }
            }
        }
        for spec in &mut program.command_buffers {
            spec.steps.retain(|step| match step {
                Step::Signal(event) => awaited.contains(event),
                Step::Wait(event) => signalled.contains(event),
                Step::Encoder(_) => true,
            });
        }
        program
            .command_buffers
            .retain(|spec| !spec.steps.is_empty());
        if program.command_buffers == before.command_buffers {
            break;
        }
    }
    renumber(&mut program);
    program
}

/// Number the events and the fences still used from 0, in order of first use.
fn renumber(program: &mut Program) {
    let mut events: Vec<usize> = Vec::new();
    let mut fences: Vec<usize> = Vec::new();
    let number = |used: &mut Vec<usize>, old: usize| match used.iter().position(|&u| u == old) {
        Some(new) => new,
        None => {
            used.push(old);
            used.len() - 1
        }
    };
    for spec in &mut program.command_buffers {
        for step in &mut spec.steps {
            match step {
                Step::Signal(event) | Step::Wait(event) => *event = number(&mut events, *event),
                Step::Encoder(encoder) => {
                    for fence in encoder.waits.iter_mut().chain(encoder.updates.iter_mut()) {
                        *fence = number(&mut fences, *fence);
                    }
                }
            }
        }
    }
    program.events = events.len();
    program.fences = fences.len();
}
