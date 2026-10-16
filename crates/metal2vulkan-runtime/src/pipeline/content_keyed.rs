//! Device objects shared by their exact content: one object per distinct content, held by the
//! device's map and shared by every variant that uses it. Who destroys them, and through what, is
//! the rule.
//!
//! - **Exact.** An object is found by its whole content (the map hashes it, then compares it):
//!   two variants share one only where the content that made it is the same.
//! - **The device's, destroyed by the map alone**. An object keeps no device functions and has no
//!   `Drop` that reaches Vulkan: a holder's `Arc` (a variant's objects, a kept instance's,
//!   `kept.rs`) is a share, a count, never a destructor. Only the map destroys an object
//!   ([`Owned::destroy`], through the device it is handed, `DeviceShared::vk`), on one of three
//!   paths, each returning what it let go ([`Gone`]), destroyed once the map's lock is released:
//!   - [`ContentKeyed::settle`], in a program's call, once the map's is the object's only share;
//!   - [`ContentKeyed::sweep`], on any thread, for the objects of a host step's or a pool piece's
//!     own (below) that nothing else holds;
//!   - [`ContentKeyed::drain`], at the device's teardown (`DeviceShared::drop`), after the kept
//!     variants': everything left.
//!
//!   So no object is destroyed before the device's teardown while anything else shares it (a new
//!   holder takes a share from the map, under its lock, so none appears after the check), and none
//!   after it: the map is the device's, so a destruction always has a live device. The check is
//!   `Arc::get_mut`, whose Acquire orders every use a released holder made (a pool piece's
//!   `vkCreate*Pipelines` from a module, on its thread) before the destruction, as the last
//!   drop's fence did when the object destroyed itself; at the teardown the device's own last
//!   drop does (every holder's use came before it let go of its device, or of what held its
//!   share). A share left at the teardown is one whose holder is dropping its own `Device` right
//!   then (a variant whose last device handle goes before its fields), and a share is used only
//!   through its holder's device, which is gone: dropping it calls nothing. The order of a
//!   holder's fields therefore decides nothing (the earlier rule, a destroying `Drop` of the
//!   object's own, gave).
//! - **On the program's threads, in the program's order** (the owner's HITCH rule 1). Only a
//!   program's call ([`program_call`]: a thread of the program's or one running its code, outside
//!   a host step and a pool piece) inserts ([`ContentKeyed::insert`]) or settles, each a debug
//!   assertion; a host step or a pool piece only looks up ([`ContentKeyed::get`]) and, on a miss,
//!   makes an object of its own that no lookup finds. Settling removes in the order of insertion.
//!   What remains timing's is only whether a command buffer let its variants go before a settle,
//!   as for the kept variants.
//! - **The exception: an object of a host step's own.** No lookup finds it, so it changes nothing
//!   another variant takes; the map keeps it apart (`own`) only to destroy it. For a variant a
//!   command buffer's host step made (a device-generated replay's, `generated_execute.rs`, which
//!   keeps no instance), the handler thread releases the variant's layouts and samplers as the
//!   command buffer releases its segment, and its own objects go at the next sweep or settle (the
//!   next variant made, on any thread) or at the teardown.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

/// Whether this call is the program's own (the module docs): a thread of the program's, or a
/// runtime thread running the program's code (`exit_guard::in_program_call`), and neither a
/// command buffer's host step nor a pool piece. Only such a call changes a device's maps.
pub(crate) fn program_call() -> bool {
    crate::exit_guard::in_program_call()
        && !crate::exit_guard::in_host_step()
        && !super::pool::in_piece()
}

/// An object a device's map holds (the module docs): it keeps no device functions, and nothing
/// but the map destroys it, through its owner `C` (the device's shared state, `DeviceShared`).
pub(crate) trait Owned<C: ?Sized> {
    /// Destroy its Vulkan objects through `owner`: once, by the map, when nothing else shares it
    /// or at the device's teardown.
    fn destroy(&self, owner: &C);
}

/// The device's objects of one kind, by their content `K`.
pub(crate) struct ContentKeyed<K, T> {
    /// Each object, with its place in the order of insertion.
    entries: HashMap<K, (u64, Arc<T>)>,
    /// The objects of a host step's or a pool piece's own, which no lookup finds, in the order
    /// they were made: held only until a sweep finds nothing else sharing them.
    own: Vec<Arc<T>>,
    next: u64,
}

impl<K, T> Default for ContentKeyed<K, T> {
    fn default() -> Self {
        ContentKeyed {
            entries: HashMap::new(),
            own: Vec::new(),
            next: 0,
        }
    }
}

/// What a map let go of, in its order: destroyed through the owner once the map's lock is
/// released ([`Gone::destroy`]).
#[must_use = "the map let these go: destroy them through the device"]
pub(crate) struct Gone<T>(Vec<Arc<T>>);

impl<T> Gone<T> {
    /// Destroy each, in order, through `owner`.
    pub(crate) fn destroy<C: ?Sized>(self, owner: &C)
    where
        T: Owned<C>,
    {
        for object in &self.0 {
            object.destroy(owner);
        }
    }
}

impl<K: Hash + Eq, T> ContentKeyed<K, T> {
    /// The object of `key`, if the map holds one.
    pub(crate) fn get(&self, key: &K) -> Option<Arc<T>> {
        self.entries.get(key).map(|(_, object)| Arc::clone(object))
    }

    /// Hold `object` as the one of `key` (a program's call made it; none was there).
    pub(crate) fn insert(&mut self, key: K, object: &Arc<T>) {
        debug_assert!(
            program_call(),
            "a device's map changed outside a program's call"
        );
        let at = self.next;
        self.next += 1;
        self.entries.insert(key, (at, Arc::clone(object)));
    }

    /// The object of `key`, else one `make` makes now (and whether it was made): found by later
    /// lookups where `program` (a program's call made it), else the caller's own (a host step's
    /// or a pool piece's), which the map holds apart until a sweep.
    pub(crate) fn get_or_make<E>(
        &mut self,
        key: K,
        program: bool,
        make: impl FnOnce() -> Result<T, E>,
    ) -> Result<(Arc<T>, bool), E> {
        if let Some(object) = self.get(&key) {
            return Ok((object, false));
        }
        let object = Arc::new(make()?);
        match program {
            true => self.insert(key, &object),
            false => self.own.push(Arc::clone(&object)),
        }
        Ok((object, true))
    }

    /// Let go of every object the map alone holds, those found by lookups in the order they were
    /// inserted, then those of a host step's own: in a program's call.
    pub(crate) fn settle(&mut self) -> Gone<T> {
        debug_assert!(
            program_call(),
            "a device's map settled outside a program's call"
        );
        let mut gone: Vec<(u64, Arc<T>)> = Vec::new();
        self.entries.retain(|_, (at, object)| {
            // `Arc::get_mut` answers whether the map's is the only share with an Acquire, so
            // every earlier use by a holder that let its share go (a pool piece's
            // `vkCreate*Pipelines` from a module) happens before the destruction, as the last
            // drop's fence ordered it (the consumer's review).
            if Arc::get_mut(object).is_none() {
                return true;
            }
            gone.push((*at, Arc::clone(object)));
            false
        });
        gone.sort_by_key(|(at, _)| *at);
        let mut gone: Vec<Arc<T>> = gone.into_iter().map(|(_, object)| object).collect();
        gone.extend(self.unheld_own());
        Gone(gone)
    }

    /// Let go of the objects of a host step's or a pool piece's own that nothing else holds: on
    /// any thread, as no lookup finds them.
    pub(crate) fn sweep(&mut self) -> Gone<T> {
        Gone(self.unheld_own())
    }

    /// The own objects nothing else holds, removed, in the order they were made.
    fn unheld_own(&mut self) -> Vec<Arc<T>> {
        let mut gone = Vec::new();
        self.own.retain_mut(|object| {
            // As in `settle`: unique, with an Acquire.
            if Arc::get_mut(object).is_none() {
                return true;
            }
            gone.push(Arc::clone(object));
            false
        });
        gone
    }

    /// Every object, those found by lookups in the order they were inserted, then the own ones:
    /// the device's teardown, whatever still shares them.
    pub(crate) fn drain(&mut self) -> Gone<T> {
        let mut all: Vec<(u64, Arc<T>)> = self.entries.drain().map(|(_, entry)| entry).collect();
        all.sort_by_key(|(at, _)| *at);
        let mut all: Vec<Arc<T>> = all.into_iter().map(|(_, object)| object).collect();
        all.append(&mut self.own);
        Gone(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// An object whose destruction its owner, a log, records; dropping it records nothing.
    struct Recorded(u32);

    /// The objects destroyed, in order.
    type Log = Mutex<Vec<u32>>;

    impl Owned<Log> for Recorded {
        fn destroy(&self, log: &Log) {
            log.lock().unwrap().push(self.0);
        }
    }

    fn destroyed(log: &Log) -> Vec<u32> {
        log.lock().unwrap().clone()
    }

    #[test]
    fn objects_leave_only_when_the_map_alone_holds_them_in_insertion_order() {
        let log = Log::default();
        let mut map: ContentKeyed<Vec<u32>, Recorded> = ContentKeyed::default();
        let a = Arc::new(Recorded(1));
        let b = Arc::new(Recorded(2));
        let c = Arc::new(Recorded(3));
        map.insert(vec![7, 7], &a);
        map.insert(vec![8], &b);
        map.insert(vec![9], &c);
        // The same content finds the same object; other content none.
        assert!(Arc::ptr_eq(&map.get(&vec![7, 7]).unwrap(), &a));
        assert!(map.get(&vec![7]).is_none());
        // Held by a variant: nothing leaves, nothing is destroyed.
        map.settle().destroy(&log);
        assert_eq!(map.entries.len(), 3);
        // Released by their holders: dropping a share destroys nothing; the settle's result,
        // destroyed through the owner, destroys them in the order of insertion, whatever the
        // order of release.
        drop(c);
        drop(a);
        let taken = map.get(&vec![8]).unwrap();
        drop(b);
        assert!(destroyed(&log).is_empty());
        let gone = map.settle();
        assert!(destroyed(&log).is_empty());
        gone.destroy(&log);
        assert_eq!(destroyed(&log), vec![1, 3]);
        // A holder that took the object from the map keeps it there.
        assert_eq!(map.entries.len(), 1);
        drop(taken);
        map.drain().destroy(&log);
        assert_eq!(destroyed(&log), vec![1, 3, 2]);
        assert_eq!(map.entries.len(), 0);
    }

    #[test]
    fn only_the_program_s_objects_enter_the_map() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        let made = |n| move || Ok::<_, ()>(Recorded(n));
        // A host step's miss: made, its own, found by no lookup.
        let (own, fresh) = map.get_or_make(5, false, made(5)).unwrap();
        assert!(fresh);
        assert_eq!(map.entries.len(), 0);
        assert!(map.get(&5).is_none());
        // The program's: made once, then found.
        let (first, fresh) = map.get_or_make(6, true, made(6)).unwrap();
        assert!(fresh);
        let (again, fresh) = map.get_or_make(6, false, made(60)).unwrap();
        assert!(!fresh);
        assert!(Arc::ptr_eq(&first, &again));
        // A failed making leaves nothing.
        assert!(map
            .get_or_make(7, true, || Err::<Recorded, ()>(()))
            .is_err());
        assert!(map
            .get_or_make(8, false, || Err::<Recorded, ()>(()))
            .is_err());
        assert_eq!((map.entries.len(), map.own.len()), (1, 1));
        drop((first, again, own));
        map.settle().destroy(&log);
        assert_eq!(destroyed(&log), vec![6, 5]);
    }

    /// A host step's own object goes at a sweep, on any thread, once nothing else holds it; a
    /// sweep leaves the map's objects alone.
    #[test]
    fn a_sweep_destroys_only_unheld_own_objects() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        let (program, _) = map
            .get_or_make(1, true, || Ok::<_, ()>(Recorded(1)))
            .unwrap();
        let (held, _) = map
            .get_or_make(2, false, || Ok::<_, ()>(Recorded(2)))
            .unwrap();
        let (released, _) = map
            .get_or_make(3, false, || Ok::<_, ()>(Recorded(3)))
            .unwrap();
        drop((program, released));
        map.sweep().destroy(&log);
        assert_eq!(destroyed(&log), vec![3]);
        assert_eq!((map.entries.len(), map.own.len()), (1, 1));
        drop(held);
        map.sweep().destroy(&log);
        assert_eq!(destroyed(&log), vec![3, 2]);
        assert_eq!(
            map.entries.len(),
            1,
            "the program's object waits for a settle"
        );
    }

    /// The teardown destroys every object, the program's and the own ones, whatever still shares
    /// them: a share left then (its holder dropping its device right then) destroys nothing as it
    /// goes, so each object is destroyed once, with the device.
    #[test]
    fn the_teardown_destroys_everything_once_and_a_late_share_destroys_nothing() {
        let log = Log::default();
        let mut map: ContentKeyed<u32, Recorded> = ContentKeyed::default();
        let (late, _) = map
            .get_or_make(1, true, || Ok::<_, ()>(Recorded(1)))
            .unwrap();
        let (unheld, _) = map
            .get_or_make(2, true, || Ok::<_, ()>(Recorded(2)))
            .unwrap();
        let (late_own, _) = map
            .get_or_make(3, false, || Ok::<_, ()>(Recorded(3)))
            .unwrap();
        drop(unheld);
        map.drain().destroy(&log);
        assert_eq!(destroyed(&log), vec![1, 2, 3]);
        assert_eq!((map.entries.len(), map.own.len()), (0, 0));
        drop((late, late_own));
        drop(map);
        assert_eq!(
            destroyed(&log),
            vec![1, 2, 3],
            "a late share destroyed nothing"
        );
    }
}
