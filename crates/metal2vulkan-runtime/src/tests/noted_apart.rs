//! The last command an unordered encoder noted apart (`noted_apart.rs`).

use super::super::noted_apart::{NotedApart, NotedInto};
use super::super::resource::ResourceKey;
use super::super::tracker::{Access, Use, UseRange};
use ash::vk;
use std::collections::BTreeSet;

/// A fragment shader's read of `bytes` of the buffer `key`.
fn read(key: ResourceKey, bytes: std::ops::Range<u64>) -> Use<ResourceKey> {
    let access = Access::new(
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    Use::reads(key, true, UseRange::Bytes(bytes), access)
}

/// A command reaching a buffer through the resident set alone is the last one noted while its
/// uses, the keys it so reaches, its stages and the state it is noted into are the last's: not
/// with another key reached so, another use, other stages, another core, a part or held image
/// more, a device with buffers over host pages where it had none, or after the host aliases
/// changed (their version).
#[test]
fn a_command_reaching_the_same_resident_keys_is_noted_already() {
    let (argument, reached, other) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let uses = [read(argument, 0..64), read(reached, 0..4096)];
    let resident = BTreeSet::from([reached]);
    let stages = vk::PipelineStageFlags2::FRAGMENT_SHADER;
    let into = NotedInto {
        core: 7,
        parts: 1,
        held: 0,
        rebinds: false,
        aliases: 3,
    };
    let mut noted = NotedApart::default();
    assert!(!noted.holds(stages, (&uses, &resident), into));
    noted.note(stages, (&uses, &resident), into);
    assert!(noted.holds(stages, (&uses, &resident), into));
    // The same uses, reaching no key or another through the resident set alone.
    assert!(!noted.holds(stages, (&uses, &BTreeSet::new()), into));
    assert!(!noted.holds(stages, (&uses, &BTreeSet::from([argument])), into));
    assert!(!noted.holds(stages, (&uses, &BTreeSet::from([reached, other])), into));
    // Another use, other stages.
    let more = [read(argument, 0..64), read(reached, 0..8192)];
    assert!(!noted.holds(stages, (&more, &resident), into));
    let vertex = vk::PipelineStageFlags2::VERTEX_SHADER;
    assert!(!noted.holds(vertex, (&uses, &resident), into));
    // Another state noted into.
    let changed = [
        NotedInto { core: 8, ..into },
        NotedInto { parts: 2, ..into },
        NotedInto { held: 1, ..into },
        NotedInto {
            rebinds: true,
            ..into
        },
        NotedInto { aliases: 4, ..into },
    ];
    for other in changed {
        assert!(!noted.holds(stages, (&uses, &resident), other));
    }
    // Noted again with another resident key: that one holds, the first no longer.
    let others = BTreeSet::from([other]);
    noted.note(stages, (&uses, &others), into);
    assert!(noted.holds(stages, (&uses, &others), into));
    assert!(!noted.holds(stages, (&uses, &resident), into));
}
