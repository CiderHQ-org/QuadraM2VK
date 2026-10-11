use crate::resource::Identity;
use crate::values::HazardTrackingMode;

#[test]
fn default_tracking_resolves_by_heap_and_keys_are_unique() {
    assert!(Identity::new(HazardTrackingMode::Default, false).tracked);
    assert!(!Identity::new(HazardTrackingMode::Default, true).tracked);
    assert!(Identity::new(HazardTrackingMode::Tracked, true).tracked);
    assert!(!Identity::new(HazardTrackingMode::Untracked, false).tracked);
    let keys: std::collections::BTreeSet<_> = (0..64)
        .map(|_| Identity::new(HazardTrackingMode::Default, false).key)
        .collect();
    assert_eq!(keys.len(), 64);
}
