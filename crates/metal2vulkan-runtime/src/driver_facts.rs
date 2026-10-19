//! The runtime's informational messages, and its device's feature status message. The facts, a
//! driver's measured answers where Vulkan leaves the result undefined
//! ([`DriverFact`](metal2vulkan::device::DriverFact)), are rows of the table type the description's
//! deviations share (`device::driver_facts`), keyed on the driver's identity and release, and those
//! the device checks itself take their checks' answers (`device_checks`). A device's facts are in
//! its properties ([`DeviceProperties::driver_facts`](super::DeviceProperties::driver_facts)), and
//! how it decided each in its feature status
//! ([`DeviceProperties::feature_status`](super::DeviceProperties::feature_status)), which it says
//! once as it is made in an informational message (`metal2vulkan-feature-status`, no degrade
//! notice: a fact is taken only where it is Metal's answer, and elsewhere the path is exact).
//!
//! A driver diagnostic fact names a message the driver reports inside one call where Vulkan states
//! nothing wrong: inside that call alone, on a release measured with it, the device withholds that
//! message from its sink ([`expecting`], [`withholds`]) and counts it
//! (`WorkCounts::driver_diagnostics_withheld`).

use crate::device_checks::{status_text, FEATURE_STATUS};
use crate::{DebugMessage, MessageKind, MessageSeverity, MessageSink};
use metal2vulkan::device::DriverDiagnostic;

impl super::Device {
    /// Say once, as the device is made, how it decided each driver fact it takes (the module
    /// docs): an informational message, id name `metal2vulkan-feature-status`, to the device's
    /// sink, else on stderr.
    pub(super) fn say_feature_status(&self) {
        let message = feature_status_message(self.properties());
        self.inform(FEATURE_STATUS, message.text);
    }

    /// Hand the informational message `text`, id name `id_name`, to the device's sink, else to
    /// stderr: what the runtime says that degrades nothing (the pipeline creation log).
    pub(crate) fn inform(&self, id_name: &str, text: String) {
        inform_through(self.shared().instance.sink(), id_name, text);
    }
}

/// [`Device::inform`] through `sink`, else on stderr, where the device is ending.
pub(crate) fn inform_through(sink: Option<&MessageSink>, id_name: &str, text: String) {
    match sink {
        Some(sink) => sink.deliver(&informational(id_name, text)),
        None => {
            crate::device::heard();
            eprintln!("{text}");
        }
    }
}

/// The informational message of id name `id_name` and text `text`, as a sink receives it.
fn informational(id_name: &str, text: String) -> DebugMessage {
    DebugMessage {
        severity: MessageSeverity::Info,
        kind: MessageKind {
            general: true,
            ..MessageKind::default()
        },
        id_name: id_name.into(),
        id: 0,
        text,
        degradation: None,
    }
}

/// The informational message a device of `properties` sends as it is made, its feature status
/// (`metal2vulkan-feature-status`), as its sink receives it: what a check of what a device said
/// as it was made expects.
#[doc(hidden)]
pub fn feature_status_message(properties: &super::DeviceProperties) -> DebugMessage {
    let text = status_text(
        FEATURE_STATUS,
        &properties.name,
        &properties.description,
        &properties.feature_status,
    );
    informational(FEATURE_STATUS, text)
}

/// The driver diagnostics ([`DriverDiagnostic`]) the thread's current driver call is measured to
/// report, and how many of them it reported so far: set only inside [`expecting`].
struct Expected {
    diagnostics: Vec<DriverDiagnostic>,
    withheld: u64,
}

thread_local! {
    static EXPECTED: std::cell::RefCell<Option<Expected>> = const { std::cell::RefCell::new(None) };
}

/// Run `call`, a driver call on this thread that the device's driver facts measure to report
/// `diagnostics` (`DriverFact::pipeline_creation_diagnostic`), withholding exactly those
/// messages from the device's message sink while it runs ([`withholds`]); with how many it
/// withheld. The driver reports from the calling thread inside the call (MoltenVK's
/// `MVKBaseObject::reportMessage`), so a message reported on another thread, or after the call
/// returns, relays. A pipeline making the runtime hands to a thread of its own at the
/// user-initiated class (`device/compiles.rs`) takes the expectation with it ([`expected_here`],
/// [`withheld_for_here`]). With no diagnostics, `call` alone.
pub(crate) fn expecting<T>(
    diagnostics: Vec<DriverDiagnostic>,
    call: impl FnOnce() -> T,
) -> (T, u64) {
    if diagnostics.is_empty() {
        return (call(), 0);
    }
    /// Clears the thread's expectation however `call` ends.
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            let _ = EXPECTED.try_with(|expected| expected.borrow_mut().take());
        }
    }
    EXPECTED.with(|expected| {
        *expected.borrow_mut() = Some(Expected {
            diagnostics,
            withheld: 0,
        })
    });
    let clear = Clear;
    let result = call();
    let withheld = EXPECTED.with(|expected| expected.borrow().as_ref().map_or(0, |e| e.withheld));
    drop(clear);
    (result, withheld)
}

/// The diagnostics this thread's current call is measured to report ([`expecting`]); none
/// outside such a call. A thread of the runtime's that makes the driver call for this one, while
/// it waits (`device/compiles.rs`), expects them inside that call alone.
pub(crate) fn expected_here() -> Vec<DriverDiagnostic> {
    EXPECTED
        .try_with(|expected| {
            expected
                .try_borrow()
                .ok()
                .and_then(|expected| expected.as_ref().map(|e| e.diagnostics.clone()))
        })
        .ok()
        .flatten()
        .unwrap_or_default()
}

/// Count `withheld` diagnostics as this thread's current call's ([`expecting`]): those the
/// thread that made the driver call for it withheld ([`expected_here`]).
pub(crate) fn withheld_for_here(withheld: u64) {
    if withheld == 0 {
        return;
    }
    let _ = EXPECTED.try_with(|expected| {
        if let Ok(mut expected) = expected.try_borrow_mut() {
            if let Some(expected) = expected.as_mut() {
                expected.withheld += withheld;
            }
        }
    });
}

/// Whether the driver's `message` is one the thread's current call is measured to report
/// ([`expecting`]): a warning of the validation kind whose name, number and text are a
/// diagnostic's exactly. Such a message is counted, and the caller withholds it.
pub(crate) fn withholds(message: &DebugMessage) -> bool {
    if message.severity != MessageSeverity::Warning
        || !message.kind.validation
        || message.degradation.is_some()
    {
        return false;
    }
    EXPECTED
        .try_with(|expected| {
            let Ok(mut expected) = expected.try_borrow_mut() else {
                return false;
            };
            let Some(expected) = expected.as_mut() else {
                return false;
            };
            let matched = expected.diagnostics.iter().any(|d| {
                d.id_name == message.id_name && d.id == message.id && d.text == message.text
            });
            expected.withheld += u64::from(matched);
            matched
        })
        .unwrap_or(false)
}
