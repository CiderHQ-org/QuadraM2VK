pub fn sort_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<_> = map.keys().cloned().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for key in keys {
                if let Some(value) = map.get(&key) {
                    out.insert(key, sort_json(value.clone()));
                }
            }
            serde_json::Value::Object(out)
        }
        other => other,
    }
}

pub fn to_sorted_json_string(value: impl serde::Serialize) -> serde_json::Result<String> {
    let value = serde_json::to_value(value)?;
    serde_json::to_string(&sort_json(value))
}
