//! The content digest's nodes, and the same rules through `serde` for the types the cache does not
//! encode.

use super::*;

fn node(byte: u8) -> Node {
    Node::leaf(|sink| sink(&[byte]))
}

/// Zero nodes are omitted from structs and keep tuples and sequences apart by position and count;
/// roots of different kinds differ.
#[test]
fn nodes_omit_only_struct_fields() {
    assert!(Node::leaf(|sink| sink(&[0, 0, 0])).is_zero());
    assert!(Node::leaf(|_| {}).is_zero());
    assert!(!node(1).is_zero());
    // A leaf handed in pieces is the leaf of the whole.
    assert_eq!(
        Node::leaf(|sink| {
            sink(&[1, 2]);
            sink(&[]);
            sink(&[3]);
        }),
        Node::leaf(|sink| sink(&[1, 2, 3]))
    );

    assert_eq!(
        Node::fields(&mut [("a", node(1)), ("b", Node::ZERO)]),
        Node::fields(&mut [("a", node(1))])
    );
    assert_eq!(
        Node::fields(&mut [("b", node(2)), ("a", node(1))]),
        Node::fields(&mut [("a", node(1)), ("b", node(2))])
    );
    assert_ne!(
        Node::fields(&mut [("a", node(1))]),
        Node::fields(&mut [("b", node(1))])
    );
    assert!(Node::fields(&mut [("a", Node::ZERO)]).is_zero());
    assert!(Node::fields(&mut []).is_zero());

    assert!(!Node::variant("A", &mut []).is_zero());
    assert_ne!(Node::variant("A", &mut []), Node::variant("B", &mut []));
    assert_eq!(
        Node::variant("A", &mut [("x", Node::ZERO)]),
        Node::variant("A", &mut [])
    );

    assert!(Node::sequence([]).is_zero());
    assert!(!Node::sequence([Node::ZERO]).is_zero());
    assert_ne!(
        Node::sequence([Node::ZERO]),
        Node::sequence([Node::ZERO, Node::ZERO])
    );
    assert_ne!(
        Node::sequence([node(1), Node::ZERO]),
        Node::sequence([Node::ZERO, node(1)])
    );

    assert!(Node::tuple([Node::ZERO, Node::ZERO]).is_zero());
    assert_ne!(
        Node::tuple([node(1), Node::ZERO]),
        Node::tuple([Node::ZERO, node(1)])
    );
    assert!(!Node::some(Node::ZERO).is_zero());
    assert_ne!(Node::some(node(1)), node(1));

    assert_ne!(node(1).root("a"), node(1).root("b"));
    assert_ne!(Node::ZERO.root("a"), node(1).root("a"));
}

#[cfg(feature = "serde")]
mod serialized {
    use std::collections::BTreeMap;

    use serde::Serialize;

    use super::super::node_of;
    use super::*;

    #[derive(Serialize)]
    struct Before {
        count: u32,
        items: Vec<Item>,
        mode: Mode,
    }

    #[derive(Serialize)]
    struct After {
        added: Option<u32>,
        mode: Mode,
        count: u32,
        items: Vec<ItemAfter>,
        flag: bool,
        pairs: Vec<(u32, u32)>,
    }

    #[derive(Serialize)]
    struct Item {
        word: u64,
    }

    #[derive(Serialize)]
    struct ItemAfter {
        word: u64,
        view: Option<Item>,
    }

    #[derive(Serialize)]
    enum Mode {
        Off,
        Level(u32),
        Pair(u32, u32),
        Window { start: u32, len: u32 },
    }

    #[derive(Serialize)]
    enum ModeInserted {
        Window { start: u32, len: u32 },
        Auto,
        Pair(u32, u32),
        Level(u32),
        Off,
    }

    fn digest<T: Serialize>(value: &T) -> Node {
        node_of(value).unwrap()
    }

    /// The serde path follows the codec path's rules: an added field at zero moves nothing, and
    /// its non-zero values do.
    #[test]
    fn an_added_field_at_zero_keeps_the_digest() {
        let before = Before {
            count: 2,
            items: vec![Item { word: 5 }, Item { word: 0 }],
            mode: Mode::Level(3),
        };
        let after = After {
            added: None,
            mode: Mode::Level(3),
            count: 2,
            items: vec![
                ItemAfter {
                    word: 5,
                    view: None,
                },
                ItemAfter {
                    word: 0,
                    view: None,
                },
            ],
            flag: false,
            pairs: Vec::new(),
        };
        assert_eq!(digest(&before), digest(&after));
        let moved = After {
            added: Some(0),
            ..after
        };
        assert_ne!(digest(&before), digest(&moved));
        let moved = After {
            added: None,
            flag: true,
            ..moved
        };
        assert_ne!(digest(&before), digest(&moved));
        let moved = After {
            flag: false,
            pairs: vec![(0, 0)],
            ..moved
        };
        assert_ne!(digest(&before), digest(&moved));
    }

    /// Variants are named: reordering or inserting variants moves nothing, and switching a value
    /// to another variant does.
    #[test]
    fn variants_are_named_not_numbered() {
        let pairs = [
            (Mode::Off, ModeInserted::Off),
            (Mode::Level(4), ModeInserted::Level(4)),
            (Mode::Pair(1, 2), ModeInserted::Pair(1, 2)),
            (
                Mode::Window { start: 1, len: 0 },
                ModeInserted::Window { start: 1, len: 0 },
            ),
        ];
        for (mode, inserted) in &pairs {
            assert_eq!(digest(mode), digest(inserted));
        }
        let digests = [
            digest(&Mode::Off),
            digest(&Mode::Level(0)),
            digest(&Mode::Level(4)),
            digest(&Mode::Pair(1, 2)),
            digest(&Mode::Pair(2, 1)),
            digest(&Mode::Window { start: 0, len: 0 }),
            digest(&ModeInserted::Auto),
        ];
        for (i, a) in digests.iter().enumerate() {
            for b in &digests[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    /// A map is the sequence of its entries as pairs, so it digests as a sequence of pairs does
    /// (the plan's `serde_pairs` maps).
    #[test]
    fn a_map_is_its_entries() {
        let map: BTreeMap<u32, u64> = [(1, 5), (2, 0)].into();
        let pairs: Vec<(u32, u64)> = map.clone().into_iter().collect();
        assert_eq!(digest(&map), digest(&pairs));
        assert!(digest(&BTreeMap::<u32, u64>::new()).is_zero());
    }

    /// A plan's digest is its serialization's, covers what the plan says, and is pinned.
    #[test]
    fn a_plan_digest_follows_the_plan() {
        use crate::bind::BindingPlan;
        let plan = BindingPlan::default();
        let digest = plan.content_digest().unwrap();
        assert_eq!(digest, node_of(&plan).unwrap().root("binding plan"));
        let mut viewed = plan.clone();
        viewed.views.view_mask = 3;
        assert_ne!(viewed.content_digest().unwrap(), digest);
        let hex = viewed
            .content_digest()
            .unwrap()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            hex, "a7fb58b1d327f235fb690324f99969bccd238a3488178ac0cacb8cac19e2ebe0",
            "the plan digest's definition changed"
        );
    }
}
