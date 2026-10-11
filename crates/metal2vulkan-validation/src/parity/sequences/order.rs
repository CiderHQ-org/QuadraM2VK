//! Metal's order between a sequence's accesses, as far as the generator relies on it, and the
//! check that a sequence has no data race under it: two accesses of overlapping bytes of one
//! resource, at least one a write, in different commands, are ordered, or the sequence races.
//!
//! The order is conservative: it takes only what Metal states, and less where its documentation
//! leaves room.
//!
//! - **Memory**: an access is of bytes of a memory ([`Resource`]): a buffer's own, a heap's (its
//!   buffers' bytes at their offsets, so two buffers over the same bytes conflict), or a
//!   texture's (a slice's bytes, through the texture or a view).
//! - **Inside an encoder**: a blit encoder's commands are never relied on to be ordered with each
//!   other; a serial compute encoder orders its dispatches' accesses through one tracked object;
//!   a concurrent one orders two dispatches only across a `memoryBarrierWithScope:` between them
//!   whose scope covers the accesses (buffers, textures); a render pass's draws only read
//!   buffers, and its target is the pass's own.
//! - **Hazard tracking**: two encoders of one queue (of one command buffer, or of two committed
//!   to it in order) whose accesses through one tracked object conflict are ordered as encoded.
//!   A heap's buffers are tracked as the heap (Apple: "at the granularity of the heap"); a view is
//!   an object of its own, so tracking is not relied on between a view and its texture or
//!   another view.
//! - **Fences**: an encoder that updates a fence is ordered before a later encoder of the same
//!   command buffer that waits for it (a render pass updates after its fragment stage and waits
//!   before its vertex stage). Fences across command buffers are not relied on.
//! - **Events**: every encoder before an event's signal is ordered before every encoder after a
//!   wait for it in a later-committed command buffer.
//!
//! Order is transitive along fences, events and the tracking of the object in question, never
//! along another object's tracking. Argument buffers of indirect commands are tracked (the
//! generator makes them so), since a fence's stages are not relied on to cover an argument read.

use super::program::{
    At, Command, Encoder, Kernel, Kind, Load, Object, Program, Resource, Step, Surface,
    MAX_INSTANCES, RECORD_BYTES, ROW_BYTES, TEXTURE_BYTES,
};
use std::collections::{BTreeMap, VecDeque};

/// Bytes `start..end` of a memory, read or written through `object`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Access {
    pub(crate) resource: Resource,
    pub(crate) object: Object,
    pub(crate) start: u64,
    pub(crate) end: u64,
    pub(crate) write: bool,
}

impl Access {
    /// Whether the two touch a common byte of one memory, one of them writing.
    pub(crate) fn conflicts(&self, other: &Access) -> bool {
        self.resource == other.resource
            && (self.write || other.write)
            && self.start < other.end
            && other.start < self.end
    }

    /// Whether it is of a texture's bytes (a barrier needs the textures scope to cover it).
    pub(crate) fn texture(&self) -> bool {
        matches!(self.resource, Resource::Texture(_))
    }
}

/// `len` bytes of the buffer at `at`, in its memory.
fn buffer(program: &Program, at: At, len: u64, write: bool) -> Access {
    let (resource, start, object) = program.memory(at);
    Access {
        resource,
        object,
        start,
        end: start + len,
        write,
    }
}

/// Rows `rows` of slice `slice` of `surface` (0 for a view), in its texture's bytes.
fn texture(
    program: &Program,
    surface: Surface,
    slice: u32,
    rows: (u32, u32),
    write: bool,
) -> Access {
    let seen = program.seen(surface, slice);
    let base = u64::from(seen.slice) * TEXTURE_BYTES;
    Access {
        resource: Resource::Texture(seen.texture),
        object: seen.object,
        start: base + u64::from(rows.0) * ROW_BYTES,
        end: base + u64::from(rows.1) * ROW_BYTES,
        write,
    }
}

/// The accesses of `command` of `program`. An indirect draw is taken to read as many rectangles
/// and colours as any draw may draw.
pub(crate) fn command_accesses(program: &Program, command: &Command) -> Vec<Access> {
    match *command {
        Command::Fill {
            buffer: index,
            start,
            end,
            ..
        } => {
            let at = At {
                buffer: index,
                offset: start,
            };
            vec![buffer(program, at, end - start, true)]
        }
        Command::Copy { from, to, size } => vec![
            buffer(program, from, size, false),
            buffer(program, to, size, true),
        ],
        Command::ToTexture {
            from,
            to,
            slice,
            rows,
        } => vec![
            buffer(program, from, u64::from(rows.1 - rows.0) * ROW_BYTES, false),
            texture(program, to, slice, rows, true),
        ],
        Command::FromTexture {
            from,
            slice,
            rows,
            to,
        } => vec![
            texture(program, from, slice, rows, false),
            buffer(program, to, u64::from(rows.1 - rows.0) * ROW_BYTES, true),
        ],
        Command::TexelRead { from, rows, to } => vec![
            texture(program, from, 0, (0, rows), false),
            buffer(program, to, u64::from(rows) * ROW_BYTES, true),
        ],
        Command::TexelWrite { from, to, rows } => vec![
            buffer(program, from, u64::from(rows) * ROW_BYTES, false),
            texture(program, to, 0, (0, rows), true),
        ],
        Command::Dispatch {
            kernel,
            a,
            b,
            indirect,
        } => {
            let mut accesses = match kernel {
                Kernel::Copy => vec![buffer(program, a, 4, false), buffer(program, b, 4, true)],
                Kernel::Mix => vec![
                    buffer(program, a, 4, false),
                    buffer(program, a, 4, true),
                    buffer(program, b, 4, false),
                ],
            };
            if let Some(at) = indirect {
                accesses.push(buffer(program, at, 12, false));
            }
            accesses
        }
        Command::Barrier { .. } => Vec::new(),
        Command::Draw {
            rects,
            colors,
            instances,
            indirect,
        } => {
            let count = u64::from(if indirect.is_some() {
                MAX_INSTANCES
            } else {
                instances
            });
            let mut accesses = vec![
                buffer(program, rects, RECORD_BYTES * count, false),
                buffer(program, colors, 4 * count, false),
            ];
            if let Some(at) = indirect {
                accesses.push(buffer(program, at, RECORD_BYTES, false));
            }
            accesses
        }
    }
}

/// The accesses of a render pass itself: its target's slice written whole (cleared or drawn, and
/// stored), and read whole when it loads.
pub(crate) fn pass_accesses(program: &Program, kind: Kind) -> Vec<Access> {
    match kind {
        Kind::Render {
            target,
            slice,
            load,
        } => {
            let rows = (0, super::program::TARGET);
            let whole = |write| texture(program, target, slice, rows, write);
            let mut accesses = vec![whole(true)];
            if load == Load::Load {
                accesses.push(whole(false));
            }
            accesses
        }
        _ => Vec::new(),
    }
}

/// Where an access is: its command buffer, its step (an encoder), and its command (`None` for a
/// render pass itself). Ordered as the sequence encodes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Position {
    pub(crate) cb: usize,
    pub(crate) step: usize,
    pub(crate) command: Option<usize>,
}

/// Every access of the sequence with its position, in encoded order.
pub(crate) fn accesses(program: &Program) -> Vec<(Position, Access)> {
    let mut all = Vec::new();
    for (cb, step, encoder) in program.encoders() {
        for access in pass_accesses(program, encoder.kind) {
            all.push((
                Position {
                    cb,
                    step,
                    command: None,
                },
                access,
            ));
        }
        for (index, command) in encoder.commands.iter().enumerate() {
            for access in command_accesses(program, command) {
                all.push((
                    Position {
                        cb,
                        step,
                        command: Some(index),
                    },
                    access,
                ));
            }
        }
    }
    all
}

/// The sequence's order: its encoders, and the edges fences and events make between them.
pub(crate) struct Order<'a> {
    program: &'a Program,
    /// Each encoder's (command buffer, step), in encoded order.
    nodes: Vec<(usize, usize)>,
    index: BTreeMap<(usize, usize), usize>,
    /// Fence and event edges, from each node.
    edges: Vec<Vec<usize>>,
    /// The edges hazard tracking makes for each tracked object, from each node.
    tracked: BTreeMap<Object, BTreeMap<usize, Vec<usize>>>,
    accesses: Vec<(Position, Access)>,
}

impl<'a> Order<'a> {
    pub(crate) fn new(program: &'a Program) -> Order<'a> {
        let nodes: Vec<(usize, usize)> =
            program.encoders().map(|(cb, step, _)| (cb, step)).collect();
        let index: BTreeMap<(usize, usize), usize> = nodes
            .iter()
            .enumerate()
            .map(|(node, &at)| (at, node))
            .collect();
        let mut edges = vec![Vec::new(); nodes.len()];
        // Fences: an updating encoder before a later waiting one of its command buffer.
        for (cb, spec) in program.command_buffers.iter().enumerate() {
            for (u, s) in spec.steps.iter().enumerate() {
                let Step::Encoder(first) = s else { continue };
                for (v, t) in spec.steps.iter().enumerate().skip(u + 1) {
                    let Step::Encoder(second) = t else { continue };
                    if first.updates.iter().any(|f| second.waits.contains(f)) {
                        edges[index[&(cb, u)]].push(index[&(cb, v)]);
                    }
                }
            }
        }
        // Events: the encoders before a signal before those after a wait of a later buffer.
        let mut signals: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        let mut waits: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for (cb, spec) in program.command_buffers.iter().enumerate() {
            for (step, s) in spec.steps.iter().enumerate() {
                match s {
                    Step::Signal(event) => signals.entry(*event).or_default().push((cb, step)),
                    Step::Wait(event) => waits.entry(*event).or_default().push((cb, step)),
                    Step::Encoder(_) => {}
                }
            }
        }
        for (event, signalled) in &signals {
            let [(scb, sstep)] = signalled[..] else {
                continue;
            };
            for &(wcb, wstep) in waits.get(event).map_or(&[][..], Vec::as_slice) {
                if wcb <= scb {
                    continue;
                }
                for &(ucb, ustep) in &nodes {
                    if ucb != scb || ustep > sstep {
                        continue;
                    }
                    for &(vcb, vstep) in &nodes {
                        if vcb == wcb && vstep > wstep {
                            edges[index[&(ucb, ustep)]].push(index[&(vcb, vstep)]);
                        }
                    }
                }
            }
        }
        let accesses = accesses(program);
        let tracked = program
            .objects()
            .filter(|&object| program.tracked(object))
            .map(|object| {
                let edges = tracked_edges(program, &index, &accesses, object);
                (object, edges)
            })
            .collect();
        Order {
            program,
            nodes,
            index,
            edges,
            tracked,
            accesses,
        }
    }

    fn encoder(&self, cb: usize, step: usize) -> &Encoder {
        match &self.program.command_buffers[cb].steps[step] {
            Step::Encoder(encoder) => encoder,
            other => panic!("cb{cb} s{step} is no encoder: {other:?}"),
        }
    }

    /// Whether `first`, at `earlier`, is ordered before the conflicting `second` at `later`.
    pub(crate) fn ordered(
        &self,
        (earlier, first): (Position, &Access),
        (later, second): (Position, &Access),
    ) -> bool {
        if earlier == later {
            return true;
        }
        // Tracking orders accesses through one tracked object only.
        let object = (first.object == second.object && self.program.tracked(first.object))
            .then_some(first.object);
        if (earlier.cb, earlier.step) == (later.cb, later.step) {
            let encoder = self.encoder(earlier.cb, earlier.step);
            return match encoder.kind {
                Kind::Blit | Kind::Render { .. } => false,
                Kind::Compute { concurrent } => {
                    let (Some(a), Some(b)) = (earlier.command, later.command) else {
                        return false;
                    };
                    let textures = first.texture() || second.texture();
                    let barrier = encoder.commands[a + 1..b].iter().any(|command| {
                        matches!(command, Command::Barrier { textures: covers } if *covers || !textures)
                    });
                    barrier || (!concurrent && object.is_some())
                }
            };
        }
        let from = self.index[&(earlier.cb, earlier.step)];
        let to = self.index[&(later.cb, later.step)];
        let tracked = object.and_then(|object| self.tracked.get(&object));
        let mut seen = vec![false; self.nodes.len()];
        let mut queue = VecDeque::from([from]);
        seen[from] = true;
        while let Some(node) = queue.pop_front() {
            if node == to {
                return true;
            }
            let next = self.edges[node].iter().chain(
                tracked
                    .and_then(|edges| edges.get(&node))
                    .map_or(&[][..], Vec::as_slice),
            );
            for &successor in next {
                if !seen[successor] {
                    seen[successor] = true;
                    queue.push_back(successor);
                }
            }
        }
        false
    }

    /// The first access before `at` that conflicts with `access` there and is not ordered
    /// before it.
    pub(crate) fn unordered_before(
        &self,
        at: Position,
        access: &Access,
    ) -> Option<(Position, Access)> {
        self.accesses
            .iter()
            .filter(|(position, _)| *position < at)
            .find(|(position, earlier)| {
                earlier.conflicts(access) && !self.ordered((*position, earlier), (at, access))
            })
            .copied()
    }
}

/// The edges hazard tracking makes for the tracked `object`: between two encoders of one queue
/// whose accesses through it conflict, from the earlier.
fn tracked_edges(
    program: &Program,
    index: &BTreeMap<(usize, usize), usize>,
    accesses: &[(Position, Access)],
    object: Object,
) -> BTreeMap<usize, Vec<usize>> {
    let mut edges: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let of: Vec<&(Position, Access)> = accesses
        .iter()
        .filter(|(_, access)| access.object == object)
        .collect();
    let queue = |cb: usize| program.command_buffers[cb].queue;
    for (i, (p, a)) in of.iter().enumerate() {
        for (q, b) in of.iter().skip(i + 1) {
            let (u, v) = (index[&(p.cb, p.step)], index[&(q.cb, q.step)]);
            if u != v && queue(p.cb) == queue(q.cb) && a.conflicts(b) {
                edges.entry(u).or_default().push(v);
            }
        }
    }
    edges
}

/// Every race of the sequence, every misuse of its events and fences the order does not take (an
/// event signalled other than once, a wait not after its signal's command buffer, a fence waited
/// for in another command buffer than it is updated in, or before its update), and every copy
/// Metal's API validation raises on (between two buffers over bytes in common).
pub(crate) fn problems(program: &Program) -> Vec<String> {
    let mut problems = Vec::new();
    let mut signals = vec![Vec::new(); program.events];
    let mut waits = vec![Vec::new(); program.events];
    let mut updates = vec![Vec::new(); program.fences];
    let mut fence_waits = vec![Vec::new(); program.fences];
    for (cb, spec) in program.command_buffers.iter().enumerate() {
        if spec.queue >= program.queues {
            problems.push(format!(
                "cb{cb}: queue {} of {}",
                spec.queue, program.queues
            ));
        }
        for (step, s) in spec.steps.iter().enumerate() {
            match s {
                Step::Signal(event) => signals[*event].push(cb),
                Step::Wait(event) => waits[*event].push(cb),
                Step::Encoder(encoder) => {
                    for &fence in &encoder.updates {
                        updates[fence].push((cb, step));
                    }
                    for &fence in &encoder.waits {
                        fence_waits[fence].push((cb, step));
                    }
                    for (c, command) in encoder.commands.iter().enumerate() {
                        if let Command::Copy { from, to, .. } = command {
                            if program.share_memory(from.buffer, to.buffer) {
                                problems.push(format!(
                                    "cb{cb} s{step} c{c}: a copy between buffer{} and buffer{}, \
                                     over bytes in common",
                                    from.buffer, to.buffer
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    for event in 0..program.events {
        match signals[event][..] {
            [signalled] => {
                if let Some(cb) = waits[event].iter().find(|&&cb| cb <= signalled) {
                    problems.push(format!(
                        "event{event}: waited for in cb{cb}, signalled in cb{signalled}"
                    ));
                }
            }
            ref other => problems.push(format!("event{event}: signalled in {other:?}")),
        }
    }
    for fence in 0..program.fences {
        for &(cb, step) in &fence_waits[fence] {
            if !updates[fence]
                .iter()
                .any(|&(ucb, ustep)| ucb == cb && ustep < step)
            {
                problems.push(format!(
                    "fence{fence}: waited for in cb{cb} s{step} with no update before it there"
                ));
            }
        }
    }
    let order = Order::new(program);
    let all = &order.accesses;
    for (i, (p, a)) in all.iter().enumerate() {
        for (q, b) in all.iter().skip(i + 1) {
            if p != q && a.conflicts(b) && !order.ordered((*p, a), (*q, b)) {
                problems.push(format!("race: {p:?} {a:?} and {q:?} {b:?}"));
            }
        }
    }
    problems
}
