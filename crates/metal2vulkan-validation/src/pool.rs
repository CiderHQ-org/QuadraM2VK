//! The corpus tools' one bounded producer/worker pool.
//!
//! A producer thread streams items into a bounded queue, workers check them, and the calling
//! thread consumes the results. Every sweep that fans rows out to workers runs through
//! [`parallel`], so its liveness argument is made once.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

/// Run `check` over the items `produce` sends, on `workers` threads (at least one).
///
/// `produce` runs on its own thread and hands each item to `send`, which blocks while `queued`
/// items wait (0: a direct hand-off to a worker), so at most `queued + workers + 1` items are
/// alive at once whatever the item count. `consume` sees each result on the calling thread, in
/// completion order, and may stop the run by returning an error: the workers then take no new
/// item, and one they already received is dropped unchecked.
///
/// **Liveness.** Only the workers hold the queue's receiver (one `Arc` each), so it is dropped
/// when the last worker exits, however that happens (the consumer's error, a panic in `check`).
/// `send` then fails, also when the producer is already blocked on a full queue, and the
/// producer stops. That is `send`'s only stop path, so a producer that ignores nothing else is
/// always released by it, and a run never waits on a queue that nothing drains.
///
/// Returns the consumer's error first, then a worker's panic, then the producer's panic or error,
/// and otherwise what `produce` returned.
pub fn parallel<T: Send, R: Send, P: Send>(
    workers: usize,
    queued: usize,
    produce: impl FnOnce(&dyn Fn(T) -> Result<(), String>) -> Result<P, String> + Send,
    check: impl Fn(T) -> R + Sync,
    mut consume: impl FnMut(R) -> Result<(), String>,
) -> Result<P, String> {
    let workers = workers.max(1);
    let stop = AtomicBool::new(false);
    let (item_sender, item_receiver) = mpsc::sync_channel::<T>(queued);
    let item_receiver = Arc::new(Mutex::new(item_receiver));
    let (result_sender, result_receiver) = mpsc::sync_channel::<R>(workers);
    let mut consumed = Ok(());
    let (worker_panicked, produced) = std::thread::scope(|scope| {
        let (stop, check) = (&stop, &check);
        let producer = scope.spawn(move || {
            let send = |item: T| {
                item_sender
                    .send(item)
                    .map_err(|_| "the workers stopped".to_owned())
            };
            produce(&send)
        });
        let handles = (0..workers)
            .map(|_| {
                let item_receiver = Arc::clone(&item_receiver);
                let result_sender = result_sender.clone();
                scope.spawn(move || loop {
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let item = match item_receiver.lock() {
                        Ok(receiver) => receiver.recv(),
                        Err(_) => break,
                    };
                    // A closed, drained queue: the producer is done.
                    let Ok(item) = item else {
                        break;
                    };
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    if result_sender.send(check(item)).is_err() {
                        break;
                    }
                })
            })
            .collect::<Vec<_>>();
        // From here on only the workers hold the receiver and the result senders.
        drop(item_receiver);
        drop(result_sender);
        for outcome in result_receiver {
            if consumed.is_ok() {
                consumed = consume(outcome);
                if consumed.is_err() {
                    stop.store(true, Ordering::Release);
                }
            }
        }
        let worker_panicked = handles
            .into_iter()
            .map(|handle| handle.join())
            .fold(false, |panicked, joined| panicked | joined.is_err());
        let produced = producer
            .join()
            .map_err(|_| "the producer panicked".to_owned());
        (worker_panicked, produced)
    });
    consumed?;
    if worker_panicked {
        return Err("a worker panicked".to_owned());
    }
    produced?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    /// Run `f` on its own thread and return its value; a run still going after a minute is a
    /// deadlock and fails the test instead of hanging it (the pass does not depend on the time).
    fn without_deadlock<V: Send + 'static>(f: impl FnOnce() -> V + Send + 'static) -> V {
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || sender.send(f()));
        receiver
            .recv_timeout(Duration::from_secs(60))
            .expect("parallel deadlocked")
    }

    #[test]
    fn every_item_reaches_the_consumer_once() {
        for (workers, queued) in [(7, 7), (1, 0), (3, 1)] {
            let items = (0..100u32).collect::<Vec<_>>();
            let mut seen = vec![0u32; items.len()];
            let produced = parallel(
                workers,
                queued,
                |send| {
                    for at in 0..items.len() {
                        send(at)?;
                    }
                    Ok(items.len())
                },
                |at| (at, items[at] * 2),
                |(at, doubled)| {
                    assert_eq!(doubled, items[at] * 2);
                    seen[at] += 1;
                    Ok(())
                },
            )
            .expect("sweep");
            assert_eq!(produced, 100);
            assert!(seen.iter().all(|&n| n == 1));
        }
    }

    /// The deadlock this pool exists to rule out, reproduced by construction rather than by
    /// timing: the producer never stops on its own, so once the consumer fails and the one worker
    /// stops, the queue fills and the producer blocks in `send` (or finds the receiver already
    /// gone). Nothing drains the queue after that, so only the receiver's drop with the last
    /// worker can release it; a receiver held anywhere else hangs this test on every run.
    #[test]
    fn a_producer_blocked_on_a_full_queue_is_released_when_the_workers_stop() {
        for (workers, queued) in [(1, 1), (1, 0), (4, 4)] {
            let (result, sent) = without_deadlock(move || {
                let sent = AtomicUsize::new(0);
                let result = parallel(
                    workers,
                    queued,
                    |send| -> Result<(), String> {
                        loop {
                            send(())?;
                            sent.fetch_add(1, Ordering::Relaxed);
                        }
                    },
                    |()| (),
                    |()| Err("index write failed".to_owned()),
                );
                (result, sent.into_inner())
            });
            assert_eq!(result, Err("index write failed".to_owned()));
            // Items checked before the stop is seen (the results held back while the consumer
            // runs), one received per worker after it, and the queue's slots.
            assert!(
                sent <= 3 * workers + queued + 2,
                "{workers}/{queued}: {sent}"
            );
        }
    }

    /// Workers that all panic release the producer too, and the panic is the error.
    #[test]
    fn workers_that_panic_release_the_producer_and_are_reported() {
        let result = without_deadlock(|| {
            parallel(
                3,
                1,
                |send| -> Result<(), String> {
                    loop {
                        send(())?;
                    }
                },
                |()| panic!("check failed"),
                |()| Ok(()),
            )
        });
        assert_eq!(result, Err("a worker panicked".to_owned()));
    }

    #[test]
    fn a_consumer_error_stops_the_producer_and_the_workers_and_is_returned() {
        let (result, produced, checked) = without_deadlock(|| {
            let produced = AtomicUsize::new(0);
            let checked = AtomicUsize::new(0);
            let mut consumed = 0usize;
            let result = parallel(
                4,
                4,
                |send| {
                    for n in 0..10_000u32 {
                        send(n)?;
                        produced.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(())
                },
                |_| checked.fetch_add(1, Ordering::Relaxed),
                |_| {
                    consumed += 1;
                    if consumed == 3 {
                        Err("index write failed".into())
                    } else {
                        Ok(())
                    }
                },
            );
            (result, produced.into_inner(), checked.into_inner())
        });
        assert_eq!(result, Err("index write failed".to_owned()));
        // Three consumed, the result queue's four and one per worker held back while the consumer
        // runs, then at most one item per worker received and dropped, and the queue's four.
        // Receiving the third result frees a result slot before the consumer's error sets the stop
        // flag, so one woken worker can send and check one more item in that window.
        assert!(produced <= 3 + 4 + 4 + 4 + 4 + 1, "{produced} produced");
        assert!(checked <= 3 + 4 + 4 + 1, "{checked} checked");
    }

    /// The parent holds a bounded number of rows whatever the batch size: the producer blocks
    /// once `queued` items wait and each worker holds one.
    #[test]
    fn the_items_alive_at_once_stay_within_the_queue_and_the_workers() {
        static ALIVE: AtomicUsize = AtomicUsize::new(0);
        static PEAK: AtomicUsize = AtomicUsize::new(0);
        struct Counted;
        impl Counted {
            fn new() -> Counted {
                let alive = ALIVE.fetch_add(1, Ordering::SeqCst) + 1;
                PEAK.fetch_max(alive, Ordering::SeqCst);
                Counted
            }
        }
        impl Drop for Counted {
            fn drop(&mut self) {
                ALIVE.fetch_sub(1, Ordering::SeqCst);
            }
        }
        for (workers, queued) in [(1usize, 1usize), (3, 3), (8, 8), (4, 1), (2, 0)] {
            PEAK.store(0, Ordering::SeqCst);
            parallel(
                workers,
                queued,
                |send| {
                    for _ in 0..1_500 {
                        send(Counted::new())?;
                    }
                    Ok(())
                },
                |item| {
                    std::thread::yield_now();
                    drop(item);
                },
                |()| Ok(()),
            )
            .expect("sweep");
            // `queued` waiting, one per worker, and one the producer is sending.
            let peak = PEAK.load(Ordering::SeqCst);
            assert!(
                peak <= queued + workers + 1,
                "{workers} workers, {queued} queued: {peak} alive"
            );
            assert_eq!(ALIVE.load(Ordering::SeqCst), 0);
        }
    }
}
