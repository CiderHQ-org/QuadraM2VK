//! Committed profiles carry no host identity.
//!
//! `scripts/regen-device-profiles/regen_device_profiles.py` withholds from every profile it writes
//! the device-property members that identify the reporting host or driver build (its
//! `HOST_IDENTITY`): the opaque UUIDs and LUIDs, and where the device sits on the host (PCI
//! address, DRM node numbers). It names what it withheld in `provenance.withheld`.

use serde_json::Value;

use super::committed_profiles;

/// The members the regeneration withholds: the script's `HOST_IDENTITY`.
const HOST_IDENTITY: [&str; 15] = [
    "deviceLUID",
    "deviceUUID",
    "driverUUID",
    "optimalTilingLayoutUUID",
    "pipelineCacheUUID",
    "shaderBinaryUUID",
    "shaderModuleIdentifierAlgorithmUUID",
    "pciBus",
    "pciDevice",
    "pciDomain",
    "pciFunction",
    "primaryMajor",
    "primaryMinor",
    "renderMajor",
    "renderMinor",
];

/// The path of every object key in `doc` that names a withheld member.
fn identity_keys(doc: &Value) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack = vec![(String::new(), doc)];
    while let Some((path, value)) = stack.pop() {
        match value {
            Value::Object(members) => {
                for (key, item) in members {
                    let at = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    if HOST_IDENTITY.contains(&key.as_str()) {
                        found.push(at.clone());
                    }
                    stack.push((at, item));
                }
            }
            Value::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    stack.push((format!("{path}[{index}]"), item));
                }
            }
            _ => {}
        }
    }
    found.sort();
    found
}

#[test]
fn committed_profiles_carry_no_host_identity() {
    let profiles = committed_profiles();
    assert!(!profiles.is_empty());
    for (stem, text) in &profiles {
        let doc: Value = serde_json::from_str(text).expect("committed profile is JSON");
        assert_eq!(identity_keys(&doc), Vec::<String>::new(), "{stem}");
        for entry in doc["provenance"]["withheld"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let entry = entry.as_str().expect("a `Struct.member` name");
            let member = entry.rsplit('.').next().unwrap_or(entry);
            assert!(
                HOST_IDENTITY.contains(&member),
                "{stem}: provenance.withheld names {entry}, which is not host identity"
            );
        }
    }
}

/// The walk finds a withheld member at any depth, and nothing else.
#[test]
fn identity_keys_walk_the_whole_document() {
    let doc = serde_json::json!({
        "provenance": {"withheld": ["VkPhysicalDeviceProperties.pipelineCacheUUID"]},
        "capabilities": {
            "device": {"properties": {
                "VkPhysicalDeviceProperties": {"deviceName": "toy", "pipelineCacheUUID": [0]},
                "VkPhysicalDeviceVulkan11Properties": {"deviceLUIDValid": false},
            }},
            "queues": {"queueFamiliesProperties": [{"pciBus": 1}]},
        },
    });
    assert_eq!(
        identity_keys(&doc),
        [
            "capabilities.device.properties.VkPhysicalDeviceProperties.pipelineCacheUUID",
            "capabilities.queues.queueFamiliesProperties[0].pciBus",
        ]
    );
}
