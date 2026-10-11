//! Serde for maps keyed by structured values (`SlotRef`, …): encoded as a sequence of
//! `(key, value)` pairs, because self-describing formats such as JSON only allow string keys.
//!
//! Use with `#[cfg_attr(feature = "serde", serde(with = "crate::pipeline::serde_pairs"))]`.

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serializes `map` as a sequence of pairs, in key order.
pub fn serialize<K, V, S>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
where
    K: Serialize,
    V: Serialize,
    S: Serializer,
{
    serializer.collect_seq(map.iter())
}

/// Deserializes a sequence of pairs; a repeated key is an error, so the encoding stays canonical.
pub fn deserialize<'de, K, V, D>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
where
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
    D: Deserializer<'de>,
{
    let pairs = Vec::<(K, V)>::deserialize(deserializer)?;
    let len = pairs.len();
    let map: BTreeMap<K, V> = pairs.into_iter().collect();
    if map.len() != len {
        return Err(serde::de::Error::custom("duplicate map key"));
    }
    Ok(map)
}
