//! Numbered samplers (`resource_ids/numbered.rs`): Metal's rows replayed on the runtime's IDs
//! alone. Each row is one process of an M4 probe (the M3 Max answered TTG2's the same): sampler
//! states marked `label=ID` between steps that make and drop libraries, function objects and
//! pipeline states, a command queue first where the row does not start with `nq_`. Each state a
//! mark makes is one no other equals, so plain allocation hands it the lowest free ID, and it is
//! kept to the row's end.

use crate::device::lock;
use crate::resource_ids::{
    number, FunctionHolds, IdHold, IdPlacement, IdSpace, LibraryHolds, Numbered, NumberedStage,
    ResourceId, ResourceIds,
};
use metal2vulkan::pipeline::NumberedSampler;
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

const READ: &[NumberedSampler] = &[NumberedSampler::Read];
const NONE: &[NumberedSampler] = &[];
/// A constexpr sampler of the probes' (its words stand for its state).
const T: NumberedSampler = NumberedSampler::Constexpr([2, 0]);

/// One probe process: its device's IDs, its marks, and the parsed libraries it made.
struct Row {
    ids: Arc<Mutex<ResourceIds>>,
    marks: Vec<IdHold>,
    answers: Vec<String>,
    parsed: usize,
}

/// A library object: its holds, and its parsed library's identity (`function_name`'s bytes).
struct Lib {
    holds: Arc<LibraryHolds>,
    bytes: usize,
}

/// A function object: its holds (held for their drop) and its name.
struct Fun {
    _holds: Arc<FunctionHolds>,
    name: (usize, u32),
}

impl Fun {
    /// The function as a pipeline stage with `samplers`.
    fn stage<'a>(&self, device: bool, samplers: &'a [NumberedSampler]) -> NumberedStage<'a> {
        NumberedStage {
            function: Some(self.name),
            device,
            samplers,
        }
    }
}

type State = Vec<Arc<Numbered>>;

impl Row {
    /// A process with a command queue first, where `queue`.
    fn new(queue: bool) -> Row {
        let ids: Arc<Mutex<ResourceIds>> = Arc::default();
        if queue {
            lock(&ids).take_device_sampler();
        }
        Row {
            ids,
            marks: Vec::new(),
            answers: Vec::new(),
            parsed: 0,
        }
    }

    /// A sampler state, kept: its ID, answered as `label=ID`.
    fn mark(&mut self, label: &str) {
        let hold = IdHold::take(&self.ids, IdSpace::Sampler, IdPlacement::Any);
        self.answers.push(format!("{label}={}", hold.id().0));
        self.marks.push(hold);
    }

    /// A library of `content`, parsed anew.
    fn lib(&mut self, content: u8) -> Lib {
        self.parsed += 1;
        Lib {
            holds: LibraryHolds::of(&self.ids, [content; 32]),
            bytes: self.parsed,
        }
    }

    /// A pipeline state of `stages`, in Metal's numbering order.
    fn state(&self, stages: &[NumberedStage<'_>]) -> State {
        number(&self.ids, stages)
    }

    /// The command queue made now.
    fn queue(&self) {
        lock(&self.ids).take_device_sampler();
    }

    fn answers(&self) -> String {
        self.answers.join(" ")
    }

    /// The device's live sampler-ID count (`Device::sampler_ids_live`).
    fn live(&self) -> u64 {
        lock(&self.ids)
            .sampler_id_reads()
            .live
            .load(Ordering::Acquire)
    }

    /// The device's sampler ID pool's size (`Device::sampler_id_pool_size`).
    fn pool(&self) -> u64 {
        lock(&self.ids)
            .sampler_id_reads()
            .pool_size
            .load(Ordering::Acquire)
    }
}

impl Lib {
    /// A function object of function `index`.
    fn function(&self, index: u32) -> Fun {
        let name = (self.bytes, index);
        Fun {
            _holds: self.holds.function(name),
            name,
        }
    }
}

/// The render pipeline of `v` (vertex samplers `vs`) and `f` (fragment samplers `fs`).
fn render(
    row: &Row,
    (v, vs): (&Fun, &[NumberedSampler]),
    (f, fs): (&Fun, &[NumberedSampler]),
) -> State {
    row.state(&[f.stage(true, fs), v.stage(false, vs)])
}

/// sid8's `own`: a library of a vertex function (`vs`) and a fragment function (`fs`), their
/// objects and pipeline; the library, the pipeline and one object dropped, `kept` the other.
fn own(queue: bool, vs: &[NumberedSampler], fs: &[NumberedSampler], kept: char) -> String {
    let mut row = Row::new(queue);
    row.mark("a");
    let l = row.lib(1);
    let (v, f) = (l.function(0), l.function(1));
    let p = render(&row, (&v, vs), (&f, fs));
    row.mark("b");
    let (keep, gone) = if kept == 'v' { (v, f) } else { (f, v) };
    drop((l, p, gone));
    row.mark("c");
    row.mark("d");
    drop(keep);
    row.mark("e");
    row.mark("f");
    row.answers()
}

#[test]
fn a_render_pipeline_s_own_function_objects_hold_their_function_s_samplers() {
    for (queue, vs, fs, kept, expected) in [
        (false, NONE, NONE, 'v', "a=1 b=3 c=2 d=4 e=5 f=6"),
        (false, NONE, NONE, 'f', "a=1 b=3 c=4 d=5 e=2 f=6"),
        (false, NONE, READ, 'v', "a=1 b=4 c=2 d=3 e=5 f=6"),
        (false, NONE, READ, 'f', "a=1 b=4 c=5 d=6 e=2 f=3"),
        (false, READ, NONE, 'v', "a=1 b=4 c=2 d=5 e=3 f=6"),
        (false, READ, NONE, 'f', "a=1 b=4 c=3 d=5 e=2 f=6"),
        (true, NONE, NONE, 'v', "a=2 b=3 c=4 d=5 e=6 f=7"),
        (true, NONE, READ, 'f', "a=2 b=4 c=5 d=6 e=3 f=7"),
        (true, READ, NONE, 'v', "a=2 b=4 c=5 d=6 e=3 f=7"),
    ] {
        assert_eq!(
            own(queue, vs, fs, kept),
            expected,
            "queue {queue}, vertex {vs:?}, fragment {fs:?}, {kept} kept"
        );
    }
}

/// sid8's `steps`: the library `l`, the pipeline `p` and the function objects `v` and `f`
/// dropped one per step in `order`, a mark after each.
fn steps(queue: bool, vs: &[NumberedSampler], fs: &[NumberedSampler], order: &str) -> String {
    let mut row = Row::new(queue);
    row.mark("a");
    let l = row.lib(1);
    let (v, f) = (l.function(0), l.function(1));
    let p = render(&row, (&v, vs), (&f, fs));
    let mut objects: BTreeMap<char, Box<dyn Any>> = BTreeMap::new();
    objects.insert('l', Box::new(l));
    objects.insert('p', Box::new(p));
    objects.insert('v', Box::new(v));
    objects.insert('f', Box::new(f));
    for c in order.chars() {
        drop(objects.remove(&c));
        row.mark(&c.to_string());
    }
    row.mark("z");
    row.answers()
}

#[test]
fn a_library_holds_its_functions_samplers_and_each_function_object_its_own() {
    for (queue, vs, fs, order, expected) in [
        (false, NONE, NONE, "pflv", "a=1 p=3 f=4 l=2 v=5 z=6"),
        (false, NONE, NONE, "pfvl", "a=1 p=3 f=4 v=5 l=2 z=6"),
        (false, NONE, NONE, "plfv", "a=1 p=3 l=4 f=2 v=5 z=6"),
        (false, NONE, NONE, "pvlf", "a=1 p=3 v=4 l=5 f=2 z=6"),
        (false, NONE, NONE, "fplv", "a=1 f=3 p=4 l=2 v=5 z=6"),
        (false, NONE, NONE, "lpfv", "a=1 l=3 p=4 f=2 v=5 z=6"),
        (false, NONE, NONE, "vfpl", "a=1 v=3 f=4 p=5 l=2 z=6"),
        (false, NONE, READ, "pflv", "a=1 p=4 f=5 l=2 v=3 z=6"),
        (false, READ, NONE, "pvlf", "a=1 p=4 v=5 l=3 f=2 z=6"),
        (false, READ, NONE, "pflv", "a=1 p=4 f=5 l=2 v=3 z=6"),
        (true, READ, NONE, "pflv", "a=2 p=4 f=5 l=6 v=3 z=7"),
    ] {
        assert_eq!(
            steps(queue, vs, fs, order),
            expected,
            "queue {queue}, vertex {vs:?}, fragment {fs:?}, order {order}"
        );
    }
}

/// sid8's `two_kernels`: a library of `k1` (reads) and `k2`; `k1`'s pipeline; `kept` the object
/// kept while the library, the pipeline and the others go: '2' `k2`'s made before the pipeline,
/// 'a' `k2`'s made after it, '1' `k1`'s own.
fn two_kernels(queue: bool, kept: char) -> String {
    let mut row = Row::new(queue);
    row.mark("a");
    let l = row.lib(2);
    let f1 = l.function(0);
    let before = (kept == '2').then(|| l.function(1));
    let p = row.state(&[f1.stage(false, READ)]);
    let after = (kept == 'a').then(|| l.function(1));
    row.mark("b");
    let f1 = (kept == '1').then_some(f1);
    drop((l, p));
    row.mark("c");
    row.mark("d");
    drop((f1, before, after));
    row.mark("e");
    row.mark("f");
    row.answers()
}

#[test]
fn another_function_s_objects_hold_nothing() {
    for (queue, kept, expected) in [
        (true, '2', "a=2 b=4 c=3 d=5 e=6 f=7"),
        (false, '2', "a=1 b=3 c=2 d=4 e=5 f=6"),
        (true, 'a', "a=2 b=4 c=3 d=5 e=6 f=7"),
        (false, 'a', "a=1 b=3 c=2 d=4 e=5 f=6"),
        (true, '1', "a=2 b=4 c=5 d=6 e=3 f=7"),
        (false, '1', "a=1 b=3 c=4 d=5 e=2 f=6"),
    ] {
        assert_eq!(
            two_kernels(queue, kept),
            expected,
            "queue {queue}, {kept} kept"
        );
    }
}

/// sid8's `new_lib_after`: `k1`'s object kept while its library and pipeline go, then a library
/// of the same content made, then the object dropped, then the new library.
fn new_lib_after(queue: bool) -> String {
    let mut row = Row::new(queue);
    row.mark("a");
    let l = row.lib(2);
    let f1 = l.function(0);
    let p = row.state(&[f1.stage(false, READ)]);
    row.mark("b");
    drop((l, p));
    row.mark("c");
    let l2 = row.lib(2);
    row.mark("d");
    drop(f1);
    row.mark("e");
    drop(l2);
    row.mark("f");
    row.mark("g");
    row.answers()
}

/// sid11's `fn_after_lib_holds`: `k1`'s pipeline and object dropped, the library kept; a new
/// object of `k1` made, then the library dropped, then the new object.
fn fn_after_lib_holds(queue: bool) -> String {
    let mut row = Row::new(queue);
    row.mark("a");
    let l = row.lib(2);
    let f1 = l.function(0);
    let p = row.state(&[f1.stage(false, READ)]);
    row.mark("b");
    drop((p, f1));
    row.mark("c");
    let f1b = l.function(0);
    row.mark("d");
    drop(l);
    row.mark("e");
    drop(f1b);
    row.mark("f");
    row.mark("g");
    row.answers()
}

/// sid11's `old_fn_new_lib` (`with_object` false) and `old_fn_new_lib_fn`: `k1`'s object made and
/// its library dropped; a library of the same content made (with a `k1` object of its own, the
/// library then dropped, where `with_object`); `k1`'s pipeline from the old object; the pipeline
/// and the old object dropped; then the new library or its object.
fn old_fn_new_lib(queue: bool, with_object: bool) -> String {
    let mut row = Row::new(queue);
    let l = row.lib(2);
    let f1 = l.function(0);
    drop(l);
    let l2 = row.lib(2);
    let f2 = with_object.then(|| l2.function(0));
    let l2 = (!with_object).then_some(l2);
    row.mark("a");
    let p = row.state(&[f1.stage(false, READ)]);
    row.mark("b");
    drop((p, f1));
    row.mark("c");
    drop((l2, f2));
    row.mark("d");
    row.mark("e");
    row.answers()
}

#[test]
fn a_library_of_the_same_content_holds_only_while_one_lives() {
    assert_eq!(new_lib_after(true), "a=2 b=4 c=5 d=6 e=3 f=7 g=8");
    assert_eq!(new_lib_after(false), "a=1 b=3 c=4 d=5 e=2 f=6 g=7");
    assert_eq!(fn_after_lib_holds(true), "a=2 b=4 c=5 d=6 e=7 f=3 g=8");
    assert_eq!(fn_after_lib_holds(false), "a=1 b=3 c=4 d=5 e=6 f=2 g=7");
    for with_object in [false, true] {
        assert_eq!(old_fn_new_lib(true, with_object), "a=2 b=4 c=3 d=5 e=6");
        assert_eq!(old_fn_new_lib(false, with_object), "a=1 b=3 c=2 d=4 e=5");
    }
}

/// TTG2's `data_same_bytes`: a library kept; another of the same bytes, its kernel's pipeline
/// (the function object gone at once); that library and the pipeline dropped, then the first.
fn same_bytes(queue: bool) -> String {
    let mut row = Row::new(queue);
    let x = row.lib(3);
    row.mark("a");
    let l = row.lib(3);
    let p = {
        let k = l.function(0);
        row.state(&[k.stage(false, READ)])
    };
    row.mark("b");
    drop((l, p));
    row.mark("c");
    row.mark("d");
    drop(x);
    row.mark("e");
    row.mark("f");
    row.answers()
}

/// TTG2's `split`: a vertex function's library and a fragment function's, the pipeline of them
/// (the objects gone at once); the pipeline and one library dropped, then the other.
fn split(vs: &[NumberedSampler], fs: &[NumberedSampler], keep_vertex_lib: bool) -> String {
    let mut row = Row::new(false);
    row.mark("a");
    let (a, b) = (row.lib(4), row.lib(5));
    let p = {
        let (v, f) = (a.function(0), b.function(0));
        render(&row, (&v, vs), (&f, fs))
    };
    row.mark("b");
    let (keep, gone) = if keep_vertex_lib { (a, b) } else { (b, a) };
    drop((p, gone));
    row.mark("c");
    row.mark("d");
    drop(keep);
    row.mark("e");
    row.mark("f");
    row.answers()
}

#[test]
fn libraries_of_equal_bytes_share_and_each_stage_s_library_holds_its_own() {
    assert_eq!(same_bytes(true), "a=2 b=4 c=5 d=6 e=3 f=7");
    assert_eq!(same_bytes(false), "a=1 b=3 c=4 d=5 e=2 f=6");
    for (vs, fs, keep_vertex_lib, expected) in [
        (NONE, NONE, true, "a=1 b=3 c=2 d=4 e=5 f=6"),
        (NONE, NONE, false, "a=1 b=3 c=4 d=5 e=2 f=6"),
        (NONE, READ, true, "a=1 b=4 c=2 d=3 e=5 f=6"),
        (NONE, READ, false, "a=1 b=4 c=5 d=6 e=2 f=3"),
        (READ, NONE, true, "a=1 b=4 c=2 d=5 e=3 f=6"),
        (READ, NONE, false, "a=1 b=4 c=3 d=5 e=2 f=6"),
    ] {
        assert_eq!(
            split(vs, fs, keep_vertex_lib),
            expected,
            "vertex {vs:?}, fragment {fs:?}, vertex library kept {keep_vertex_lib}"
        );
    }
}

/// TTG2's `vertex_only`: a render pipeline of a vertex function (samplers `vs`) alone; the
/// pipeline dropped, then the library.
fn vertex_only(vs: &[NumberedSampler]) -> String {
    let mut row = Row::new(false);
    row.mark("a");
    let l = row.lib(6);
    let p = {
        let v = l.function(0);
        row.state(&[v.stage(false, vs)])
    };
    row.mark("b");
    drop(p);
    row.mark("c");
    row.mark("d");
    drop(l);
    row.mark("e");
    row.mark("f");
    row.answers()
}

/// TTG2's `after_render`: a render pipeline of constant functions, then a command queue, then
/// the pipeline and its library dropped.
fn queue_after_render() -> String {
    let mut row = Row::new(false);
    row.mark("a");
    let l = row.lib(1);
    let p = {
        let (v, f) = (l.function(0), l.function(1));
        render(&row, (&v, NONE), (&f, NONE))
    };
    row.mark("b");
    row.queue();
    row.mark("c");
    drop((l, p));
    row.mark("d");
    row.mark("e");
    row.answers()
}

#[test]
fn only_a_fragment_function_numbers_the_device_s_sampler_which_a_queue_then_keeps() {
    assert_eq!(vertex_only(NONE), "a=1 b=2 c=3 d=4 e=5 f=6");
    assert_eq!(vertex_only(READ), "a=1 b=3 c=4 d=5 e=2 f=6");
    assert_eq!(queue_after_render(), "a=1 b=3 c=4 d=5 e=6");
}

/// sid10's `order`: a mesh pipeline of a mesh function reading (`m_read`) or of a fragment
/// function sampling `T` (the objects gone at once); a kernel kept for good that reads
/// (`keep_read`) or samples `T`; the mesh pipeline and its library dropped.
fn mesh_order(keep_read: bool) -> String {
    let mut row = Row::new(false);
    row.mark("a");
    let l = row.lib(7);
    let p = {
        let (m, f) = (l.function(0), l.function(1));
        row.state(&[f.stage(true, &[T]), m.stage(false, READ)])
    };
    row.mark("b");
    let k = row.lib(8);
    let kernel = k.function(0);
    let kept = row.state(&[kernel.stage(false, if keep_read { READ } else { &[T] })]);
    row.mark("c");
    drop((l, p));
    for label in ["d", "e", "f", "g"] {
        row.mark(label);
    }
    drop((k, kernel, kept));
    row.answers()
}

/// sid11's archive adds: a compute (`k1`, reading) or render (fragment reading) pipeline's state
/// made and dropped at once; the archive dropped (nothing); the function objects and library.
fn archive_add(render_add: bool) -> String {
    let mut row = Row::new(false);
    let l = row.lib(if render_add { 9 } else { 2 });
    let (first, second) = (l.function(0), l.function(1));
    row.mark("a");
    let state = if render_add {
        render(&row, (&first, NONE), (&second, READ))
    } else {
        row.state(&[first.stage(false, READ)])
    };
    drop(state);
    row.mark("b");
    row.mark("c");
    drop((first, second, l));
    row.mark("d");
    row.mark("e");
    if render_add {
        row.mark("f");
    }
    row.answers()
}

#[test]
fn mesh_pipelines_number_in_reverse_stage_order_and_archive_adds_as_states() {
    assert_eq!(mesh_order(true), "a=1 b=5 c=6 d=2 e=3 f=7 g=8");
    assert_eq!(mesh_order(false), "a=1 b=5 c=6 d=2 e=4 f=7 g=8");
    assert_eq!(archive_add(false), "a=1 b=3 c=4 d=2 e=5");
    assert_eq!(archive_add(true), "a=1 b=4 c=5 d=2 e=3 f=6");
}

/// The live count (`Device::sampler_ids_live`) follows every take and give-back of the rows
/// above. A render state numbers the device's sampler and the constexpr sampler `T`; a compute
/// state of another library numbers `T` again, held, so it counts once, and a read. The queue
/// keeps the device's numbered sampler, a new ID for nothing. Dropping the render state's
/// objects gives nothing back (`T` held, the device's kept); dropping the compute state's gives
/// `T` and the read back at once, and the next state takes `T`'s ID.
#[test]
fn the_live_sampler_id_count_follows_every_take_and_give_back() {
    let mut row = Row::new(false);
    assert_eq!(row.live(), 0);
    row.mark("a");
    assert_eq!(row.live(), 1);
    let one = row.lib(1);
    let (v, f) = (one.function(0), one.function(1));
    let render_state = render(&row, (&v, NONE), (&f, &[T]));
    assert_eq!(row.live(), 3);
    let two = row.lib(2);
    let k = two.function(0);
    let compute_state = row.state(&[k.stage(false, &[T, NumberedSampler::Read])]);
    assert_eq!(row.live(), 4);
    row.queue();
    assert_eq!(row.live(), 4);
    drop((render_state, one, v, f));
    assert_eq!(row.live(), 4);
    drop((compute_state, two));
    assert_eq!(row.live(), 4, "the kernel's function object holds them");
    drop(k);
    assert_eq!(row.live(), 2);
    row.mark("b");
    assert_eq!(row.live(), 3);
    assert_eq!(row.answers(), "a=1 b=3");
    row.marks.clear();
    assert_eq!(row.live(), 1, "the device keeps its sampler for good");
}

/// A reservation counts nothing; the device's sampler counts once as a queue takes it plain; a
/// sampler state placed at an ID counts as its own while it lives, beside a plain one at the same
/// ID.
#[test]
fn the_live_sampler_id_count_counts_placed_states_and_not_reservations() {
    let row = Row::new(false);
    assert!(lock(&row.ids).reserve(IdSpace::Sampler, &[ResourceId(9)]));
    assert_eq!(row.live(), 0);
    row.queue();
    assert_eq!(row.live(), 1);
    let plain = IdHold::take(&row.ids, IdSpace::Sampler, IdPlacement::Any);
    let placed = IdHold::take(&row.ids, IdSpace::Sampler, IdPlacement::At(plain.id()));
    assert_eq!(row.live(), 3);
    drop(placed);
    assert_eq!(row.live(), 2);
    drop(plain);
    assert_eq!(row.live(), 1);
}

/// The pool size (`Device::sampler_id_pool_size`) after a queue took the device's sampler (1):
/// the k-th state takes ID k + 1, and the pool doubles at the 15th, 31st, 63rd and 127th, the
/// states the consumer measured Metal's descriptor array grow at; given back, it stays. A
/// pipeline state's numbered sampler grows it as a state does.
#[test]
fn the_sampler_pool_doubles_at_metal_s_growth_points_and_never_shrinks() {
    let mut row = Row::new(true);
    assert_eq!((row.live(), row.pool()), (1, 16));
    for k in 1..=127u64 {
        row.mark("s");
        let size = match k + 1 {
            ..16 => 16,
            16..32 => 32,
            32..64 => 64,
            64..128 => 128,
            _ => 256,
        };
        assert_eq!((row.live(), row.pool()), (k + 1, size), "the {k}th state");
    }
    row.marks.clear();
    assert_eq!((row.live(), row.pool()), (1, 256));

    let mut row = Row::new(true);
    for _ in 0..14 {
        row.mark("s");
    }
    assert_eq!((row.live(), row.pool()), (15, 16));
    let l = row.lib(1);
    let k = l.function(0);
    let state = row.state(&[k.stage(false, &[T])]);
    assert_eq!((row.live(), row.pool()), (16, 32));
    drop((state, k, l));
    assert_eq!((row.live(), row.pool()), (15, 32));
}

/// A reservation grows the pool until it holds the reserved ID (40: 64; 16: 32), before any ID
/// is handed out, and a later one is ignored; a state placed at an ID past the pool leaves it,
/// a placed ID being outside plain allocation.
#[test]
fn the_sampler_pool_grows_for_a_reservation_and_not_for_a_placement() {
    let row = Row::new(false);
    assert!(lock(&row.ids).reserve(IdSpace::Sampler, &[ResourceId(40)]));
    assert_eq!((row.live(), row.pool()), (0, 64));
    row.queue();
    assert!(!lock(&row.ids).reserve(IdSpace::Sampler, &[ResourceId(200)]));
    assert_eq!((row.live(), row.pool()), (1, 64));

    let row = Row::new(false);
    assert!(lock(&row.ids).reserve(IdSpace::Sampler, &[ResourceId(16)]));
    assert_eq!(row.pool(), 32);

    let row = Row::new(true);
    let placed = IdHold::take(
        &row.ids,
        IdSpace::Sampler,
        IdPlacement::At(ResourceId(1000)),
    );
    assert_eq!((row.live(), row.pool()), (2, 16));
    drop(placed);
    assert_eq!((row.live(), row.pool()), (1, 16));
}
