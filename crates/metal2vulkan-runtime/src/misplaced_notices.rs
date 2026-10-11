//! Notices announced against the runtime's rule, in builds without debug assertions.
//!
//! The rule announces no kind inside a pipeline pool's piece, and a kind asked for at every
//! occurrence only in the program's call or a command buffer's host step. A build with debug
//! assertions ends the process at a breach (`degradation::check_announcement`), so every lane and
//! gate fails on one. In another build a breach ends nothing and silences nothing:
//!
//! - **The notice, unchanged.** Its message, of its own kind and fields, reaches the device's sink
//!   as any notice does, but from the runtime's notice thread (`metal2vulkan-notices`), not the
//!   thread that met it. A consumer's handler may end the process at a notice; the notice thread
//!   holds no lock of the runtime's while it calls the sink (it takes none: it only receives what
//!   it delivers), so neither the handler, its logger nor the runtime's exit hook waits for one
//!   the thread that met the notice holds. The thread that met it only queues it, holding the
//!   queue's lock for the push alone.
//! - **The record.** `misplaced-notice` follows it on the same thread, naming its kind and where
//!   it was announced ([`MisplacedNotice`]), once per device (each time, where the device's
//!   options ask for each).
//! - **In order.** The thread delivers what it was handed in the order it was handed, the notice
//!   before its record; another thread's notices may come between two of one thread's.
//! - **Without a sink** both are written on stderr from the thread that met them, as any notice
//!   is.
//!
//! The thread starts at the first misplaced notice of a process (again in a forked child, which
//! has none of its parent's threads) and lives as long as the process; its queue holds what the
//! sink has not taken yet. Where the host cannot start it, the notice and its record are
//! delivered on the thread that met them rather than lost.

use crate::degradation::{Announced, Degradation, DegradeNotice, MisplacedNotice};
use crate::device::lock;
use crate::{DebugMessage, MessageSink};
use std::sync::mpsc::{self, Sender};
use std::sync::Mutex;

/// What the notice thread delivers: the sink, and the message.
type Delivery = (MessageSink, DebugMessage);

/// The notice thread's queue, with the process that started the thread.
static NOTICES: Mutex<Option<(u32, Sender<Delivery>)>> = Mutex::new(None);

/// The name of the notice thread.
pub(crate) const THREAD: &str = "metal2vulkan-notices";

/// Deliver `message`, announced where `misplaced` says, to `sink` from the notice thread, with the
/// device's `misplaced-notice` after it where `announced` announces that kind now (the module
/// docs); without a sink, write both on stderr.
pub(crate) fn deliver(
    sink: Option<&MessageSink>,
    announced: &Announced,
    misplaced: MisplacedNotice,
    message: DebugMessage,
) {
    let record = announced
        .now(Degradation::MisplacedNotice)
        .then(|| record(misplaced));
    let Some(sink) = sink else {
        for message in std::iter::once(message).chain(record) {
            eprintln!("{}", message.text);
        }
        return;
    };
    let deliveries: Vec<Delivery> = std::iter::once(message)
        .chain(record)
        .map(|message| (sink.clone(), message))
        .collect();
    if let Err(deliveries) = queue(deliveries) {
        // No notice thread: here, rather than lost.
        for (sink, message) in deliveries {
            sink.deliver(&message);
        }
    }
}

/// `misplaced`'s `misplaced-notice`.
fn record(misplaced: MisplacedNotice) -> DebugMessage {
    let mut notice = DegradeNotice::new(
        Degradation::MisplacedNotice,
        format!("a {} notice", misplaced.kind.code()),
        format!(
            "it was announced {}, where the runtime announces none, and reached the sink \
             unchanged just before this one",
            misplaced.place.words()
        ),
    );
    notice.misplaced = Some(misplaced);
    notice.message()
}

/// Hand `deliveries` to the notice thread, in order, starting it in this process first where it
/// has not started: back where the host cannot start it.
fn queue(deliveries: Vec<Delivery>) -> Result<(), Vec<Delivery>> {
    let mut notices = lock(&NOTICES);
    let process = std::process::id();
    if !matches!(&*notices, Some((started, _)) if *started == process) {
        let (sender, receiver) = mpsc::channel::<Delivery>();
        let started =
            crate::exit_guard::spawn(std::thread::Builder::new().name(THREAD.into()), move || {
                for (sink, message) in receiver {
                    sink.deliver(&message);
                }
            });
        if started.is_err() {
            return Err(deliveries);
        }
        *notices = Some((process, sender));
    }
    let Some((_, sender)) = notices.as_ref() else {
        return Err(deliveries);
    };
    let mut deliveries = deliveries.into_iter();
    for delivery in deliveries.by_ref() {
        if let Err(mpsc::SendError(delivery)) = sender.send(delivery) {
            // The thread ended (it never does while the process lives): the rest here.
            return Err(std::iter::once(delivery).chain(deliveries).collect());
        }
    }
    Ok(())
}
