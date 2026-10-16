//! The runtime's message of a device whose driver, GPU and release were not measured
//! (which replaced the announcement of facts taken past their measurement). The facts, a driver's
//! measured answers where Vulkan leaves the result undefined
//! ([`DriverFact`](metal2vulkan::device::DriverFact)), are rows of the table type the description's
//! deviations share (`device::driver_facts`), keyed on the driver's identity and release:
//! `VkPhysicalDeviceDriverProperties`'s `driverID` and `VkPhysicalDeviceProperties`'s
//! `driverVersion`. A device's facts are in its properties
//! ([`DeviceProperties::driver_facts`](super::DeviceProperties::driver_facts)). The runtime
//! relies on a fact only where the driver is measured to have it, and elsewhere takes the exact
//! path that needs none.
//!
//! Each row names a driver and the first and last releases measured with the fact, and the vendor
//! and GPU where it was measured on some alone. The facts are taken on a measured identity alone, a
//! driver at a release on a GPU a venue ran (`DeviceDescription::measured_identity`). On any other,
//! a later or earlier release or another GPU, the device takes no answer fact (the runtime takes
//! the exact path instead) and assumes its driver's measured defects, and says so once as it is
//! made in an informational message (`metal2vulkan-unmeasured-driver`, no degrade notice, as the
//! paths it takes are exact), naming what it withholds and assumes. A lane that measures a fact runs at the master gate on each venue, so a measured
//! identity whose answer changes fails there rather than drawing wrong.
//!
//! A driver diagnostic fact names a message the driver reports inside one call where Vulkan states
//! nothing wrong: inside that call alone, on a release measured with it, the device withholds that
//! message from its sink ([`expecting`], [`withholds`]) and counts it
//! (`WorkCounts::driver_diagnostics_withheld`).

use crate::{DebugMessage, MessageKind, MessageSeverity, MessageSink};
use metal2vulkan::device::DriverDiagnostic;

impl super::Device {
    /// Say once, as the device is made, that its identity is not measured, where its driver has
    /// a measured answer it therefore withholds or a measured defect it assumes (the module
    /// docs): an informational message, id name `metal2vulkan-unmeasured-driver`, to the device's
    /// sink, else on stderr. The paths it takes are exact, so it degrades nothing and is no
    /// degrade notice.
    pub(super) fn say_unmeasured_driver(&self) {
        if let Some(text) = unmeasured_driver(self.properties()) {
            self.inform(UNMEASURED_DRIVER, text);
        }
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
        None => eprintln!("{text}"),
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

/// The informational message a device of `properties` sends as it is made where its identity is
/// not measured (`metal2vulkan-unmeasured-driver`), as its sink receives it; `None` where it
/// sends none. It reads the fact the device's own message reads (`unmeasured_driver`), so a
/// check of what a device said as it was made expects the message exactly where the device
/// sends it.
#[doc(hidden)]
pub fn unmeasured_driver_message(properties: &super::DeviceProperties) -> Option<DebugMessage> {
    unmeasured_driver(properties).map(|text| informational(UNMEASURED_DRIVER, text))
}

/// The id name of the message a device whose identity is not measured sends as it is made
/// ([`unmeasured_driver`]).
pub(crate) const UNMEASURED_DRIVER: &str = "metal2vulkan-unmeasured-driver";

/// The text of the informational message a device of `properties` sends as it is made: where its
/// identity is not measured and its driver has an answer measured elsewhere that it withholds
/// or a defect measured elsewhere that it assumes, naming the identity and each; `None`
/// elsewhere (a measured identity, or a driver with nothing measured).
pub(crate) fn unmeasured_driver(properties: &super::DeviceProperties) -> Option<String> {
    let description = &properties.description;
    if description.measured_identity() {
        return None;
    }
    let withheld: Vec<String> = (properties.driver_facts_withheld.iter())
        .map(|fact| format!("{fact:?}"))
        .collect();
    let assumed: Vec<String> = (properties.driver_facts.iter())
        .map(|fact| format!("{fact:?}"))
        .chain(properties.driver_defects.iter().map(|d| format!("{d:?}")))
        .collect();
    if withheld.is_empty() && assumed.is_empty() {
        return None;
    }
    let list = |names: Vec<String>| match names.is_empty() {
        true => "none".to_owned(),
        false => names.join(", "),
    };
    Some(format!(
        "metal2vulkan: the device's driver {:?} at driverVersion {:#x} on deviceID {:#x} is an \
         identity not measured: the runtime takes the exact path in place of the driver's answers \
         measured elsewhere ({}) and works around its defects measured elsewhere ({}); every \
         answer is Metal's [{UNMEASURED_DRIVER}]",
        description.driver_id,
        description.driver_version,
        description.device_id,
        list(withheld),
        list(assumed),
    ))
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
