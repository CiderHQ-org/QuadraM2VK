//! Content digests: a SHA-256 of what a pipeline artifact or a binding plan says, which stays put
//! when the encoding grows a field the value does not use.
//!
//! A cache key names an artifact by its inputs and versions, so every [`SEMANTICS_VERSION`] bump
//! moves every key. A content digest names the artifact by its content instead: two builds whose
//! artifacts for an input say the same thing give the same digest, whatever their versions. C3's
//! currency compares the digest a row executed with the one this build compiles.
//!
//! The digest is a tree of nodes. Every value is a node, and a node is either **zero** or a
//! SHA-256:
//!
//! - A **leaf** (an integer, a float by its bits, `bool`, a string, a registry enum by its raw
//!   number, or any sequence or option of leaves) is the hash of its `codec` encoding; it is zero
//!   when the encoding is all zero bytes: `0`, `false`, `None`, an empty sequence.
//! - A **struct** is the hash of its non-zero fields, each by its name and node, in name order; a
//!   struct whose fields are all zero is zero.
//! - An **enum** value is the hash of its variant's name and its non-zero fields, the same way;
//!   it is never zero. Variants are named, never numbered, so reordering or inserting variants
//!   moves nothing, while a value that switches variant moves its digest.
//! - A sequence of non-leaves is the hash of its items' nodes and count (zero when empty), an
//!   option of one is its value's node under a `Some` tag (zero when `None`), and a tuple or array
//!   is the hash of its items' nodes in order (zero when all are).
//!
//! **Why omitting zero fields is sound, and the rule it needs.** A struct that gains a field holds
//! the same digest for every value whose new field is zero, which is what lets the format grow
//! without moving every digest. That is sound only if **a field added to an encoded struct holds
//! its zero value exactly when the artifact means what it meant before the field existed**: then
//! equal digests still mean equal behaviour. A new field whose zero would mean new behaviour for an
//! existing artifact breaks this, and the digest cannot see it: such an addition must also bump
//! the runtime version (`RUNTIME_VERSION`) or the validation executor's ABI, which C3's currency
//! holds beside the digest (`docs/VALIDATION.md`, "Artifact-keyed currency"). An enum field has no
//! zero value, so adding one moves every digest that holds the struct.
//!
//! [`SEMANTICS_VERSION`]: super::SEMANTICS_VERSION

use sha2::{Digest as _, Sha256};

use super::Hash256;

/// Names this digest's definition in every hash it takes.
const DOMAIN: &[u8] = b"metal2vulkan content digest v1\0";

/// One value's node in a content digest: `None` for a zero value, which a struct omits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Node(Option<Hash256>);

/// What a node's hash covers, as its first byte after [`DOMAIN`].
#[derive(Clone, Copy)]
#[repr(u8)]
enum Form {
    Leaf = b'L',
    Struct = b'S',
    Variant = b'E',
    Sequence = b'Q',
    Some = b'O',
    Tuple = b'T',
    Root = b'R',
}

fn hasher(form: Form) -> Sha256 {
    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update([form as u8]);
    h
}

/// A name: its length, then its bytes.
fn put_name(h: &mut Sha256, name: &str) {
    h.update((name.len() as u64).to_le_bytes());
    h.update(name.as_bytes());
}

/// A child node in its parent: `0` for zero, else `1` and its hash.
fn put_child(h: &mut Sha256, node: Node) {
    match node.0 {
        None => h.update([0]),
        Some(digest) => {
            h.update([1]);
            h.update(digest);
        }
    }
}

/// Named fields, without the zero ones, in name order.
fn put_fields(h: &mut Sha256, fields: &mut [(&str, Node)]) {
    fields.sort_unstable_by(|a, b| a.0.cmp(b.0));
    let present = fields.iter().filter(|(_, node)| !node.is_zero()).count();
    h.update((present as u64).to_le_bytes());
    for (name, node) in fields.iter().filter(|(_, node)| !node.is_zero()) {
        put_name(h, name);
        put_child(h, *node);
    }
}

impl Node {
    /// The zero node.
    pub(crate) const ZERO: Node = Node(None);

    /// Whether the value is zero (a struct omits it).
    pub(crate) fn is_zero(self) -> bool {
        self.0.is_none()
    }

    /// A leaf over the bytes `write` hands its sink, which it may hand in pieces: zero when they
    /// are all zero bytes.
    pub(crate) fn leaf(write: impl FnOnce(&mut dyn FnMut(&[u8]))) -> Node {
        let mut h = hasher(Form::Leaf);
        let mut zero = true;
        write(&mut |bytes: &[u8]| {
            zero &= bytes.iter().all(|&b| b == 0);
            h.update(bytes);
        });
        if zero {
            Node::ZERO
        } else {
            Node(Some(h.finalize().into()))
        }
    }

    /// A struct of `fields` (each by its name, unique within the struct), in any order: zero when
    /// every field is.
    pub(crate) fn fields(fields: &mut [(&str, Node)]) -> Node {
        if fields.iter().all(|(_, node)| node.is_zero()) {
            return Node::ZERO;
        }
        let mut h = hasher(Form::Struct);
        put_fields(&mut h, fields);
        Node(Some(h.finalize().into()))
    }

    /// An enum value: its variant's `name` and `fields` (each by its name, unique within the
    /// variant). Never zero.
    pub(crate) fn variant(name: &str, fields: &mut [(&str, Node)]) -> Node {
        let mut h = hasher(Form::Variant);
        put_name(&mut h, name);
        put_fields(&mut h, fields);
        Node(Some(h.finalize().into()))
    }

    /// A sequence of `items`: zero when empty.
    pub(crate) fn sequence(items: impl IntoIterator<Item = Node>) -> Node {
        let mut h = hasher(Form::Sequence);
        let mut count = 0u64;
        for item in items {
            put_child(&mut h, item);
            count += 1;
        }
        if count == 0 {
            return Node::ZERO;
        }
        h.update(count.to_le_bytes());
        Node(Some(h.finalize().into()))
    }

    /// `Some` of a value whose node is `inner`. Never zero.
    pub(crate) fn some(inner: Node) -> Node {
        let mut h = hasher(Form::Some);
        put_child(&mut h, inner);
        Node(Some(h.finalize().into()))
    }

    /// A tuple or array of `items`, in order: zero when every item is.
    pub(crate) fn tuple(items: impl IntoIterator<Item = Node>) -> Node {
        let mut h = hasher(Form::Tuple);
        let mut zero = true;
        let mut count = 0u64;
        for item in items {
            zero &= item.is_zero();
            put_child(&mut h, item);
            count += 1;
        }
        if zero {
            return Node::ZERO;
        }
        h.update(count.to_le_bytes());
        Node(Some(h.finalize().into()))
    }

    /// The digest of a whole value of `kind` whose node this is.
    pub(crate) fn root(self, kind: &str) -> Hash256 {
        let mut h = hasher(Form::Root);
        put_name(&mut h, kind);
        put_child(&mut h, self);
        h.finalize().into()
    }
}

#[cfg(feature = "serde")]
pub(crate) use serialized::node_of;

/// The same nodes for a value through its `serde` serialization, for the types the cache does not
/// encode (a [`BindingPlan`](crate::bind::BindingPlan)). Struct and variant fields are named by
/// their serialized names, tuple fields by their positions; a map is the sequence of its entries
/// as pairs; a unit or unit struct is zero. Primitives are leaves over their little-endian bytes
/// (floats by their bits), strings and byte strings over their length and bytes, as the codec
/// writes them.
#[cfg(feature = "serde")]
mod serialized {
    use serde::ser::{self, Serialize};

    use super::Node;

    /// A serialization that refused: a `Serialize` impl reported an error of its own.
    #[derive(Debug)]
    pub(crate) struct NotDigested(pub(crate) String);

    impl std::fmt::Display for NotDigested {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "the value did not serialize: {}", self.0)
        }
    }

    impl std::error::Error for NotDigested {}

    impl ser::Error for NotDigested {
        fn custom<T: std::fmt::Display>(msg: T) -> Self {
            NotDigested(msg.to_string())
        }
    }

    /// `value`'s node.
    pub(crate) fn node_of<T: Serialize + ?Sized>(value: &T) -> Result<Node, NotDigested> {
        value.serialize(Nodes)
    }

    fn leaf(bytes: &[u8]) -> Node {
        Node::leaf(|sink| sink(bytes))
    }

    fn counted(bytes: &[u8]) -> Node {
        Node::leaf(|sink| {
            sink(&u32::try_from(bytes.len()).unwrap_or(u32::MAX).to_le_bytes());
            sink(bytes);
        })
    }

    /// The serializer whose output is a value's node.
    struct Nodes;

    /// Positional field names, for tuple variants.
    const POSITIONS: [&str; 16] = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
    ];

    fn position(index: usize) -> Result<&'static str, NotDigested> {
        POSITIONS
            .get(index)
            .copied()
            .ok_or_else(|| NotDigested(format!("a tuple variant of more than {index} fields")))
    }

    /// The items of a sequence, a tuple, or a map's entries.
    struct Items {
        nodes: Vec<Node>,
        tuple: bool,
        key: Option<Node>,
    }

    impl Items {
        fn finish(self) -> Node {
            if self.tuple {
                Node::tuple(self.nodes)
            } else {
                Node::sequence(self.nodes)
            }
        }
    }

    /// The named fields of a struct or a variant (`variant` names it).
    struct Fields {
        variant: Option<&'static str>,
        fields: Vec<(&'static str, Node)>,
    }

    impl Fields {
        fn finish(mut self) -> Node {
            match self.variant {
                Some(name) => Node::variant(name, &mut self.fields),
                None => Node::fields(&mut self.fields),
            }
        }
    }

    impl ser::Serializer for Nodes {
        type Ok = Node;
        type Error = NotDigested;
        type SerializeSeq = Items;
        type SerializeTuple = Items;
        type SerializeTupleStruct = Items;
        type SerializeTupleVariant = Fields;
        type SerializeMap = Items;
        type SerializeStruct = Fields;
        type SerializeStructVariant = Fields;

        fn serialize_bool(self, v: bool) -> Result<Node, NotDigested> {
            Ok(leaf(&[u8::from(v)]))
        }
        fn serialize_i8(self, v: i8) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_i16(self, v: i16) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_i32(self, v: i32) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_i64(self, v: i64) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_i128(self, v: i128) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_u8(self, v: u8) -> Result<Node, NotDigested> {
            Ok(leaf(&[v]))
        }
        fn serialize_u16(self, v: u16) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_u32(self, v: u32) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_u64(self, v: u64) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_u128(self, v: u128) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_le_bytes()))
        }
        fn serialize_f32(self, v: f32) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_bits().to_le_bytes()))
        }
        fn serialize_f64(self, v: f64) -> Result<Node, NotDigested> {
            Ok(leaf(&v.to_bits().to_le_bytes()))
        }
        fn serialize_char(self, v: char) -> Result<Node, NotDigested> {
            Ok(leaf(&u32::from(v).to_le_bytes()))
        }
        fn serialize_str(self, v: &str) -> Result<Node, NotDigested> {
            Ok(counted(v.as_bytes()))
        }
        fn serialize_bytes(self, v: &[u8]) -> Result<Node, NotDigested> {
            Ok(counted(v))
        }
        fn serialize_none(self) -> Result<Node, NotDigested> {
            Ok(Node::ZERO)
        }
        fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> Result<Node, NotDigested> {
            Ok(Node::some(node_of(value)?))
        }
        fn serialize_unit(self) -> Result<Node, NotDigested> {
            Ok(Node::ZERO)
        }
        fn serialize_unit_struct(self, _: &'static str) -> Result<Node, NotDigested> {
            Ok(Node::ZERO)
        }
        fn serialize_unit_variant(
            self,
            _: &'static str,
            _: u32,
            variant: &'static str,
        ) -> Result<Node, NotDigested> {
            Ok(Node::variant(variant, &mut []))
        }
        fn serialize_newtype_struct<T: Serialize + ?Sized>(
            self,
            _: &'static str,
            value: &T,
        ) -> Result<Node, NotDigested> {
            node_of(value)
        }
        fn serialize_newtype_variant<T: Serialize + ?Sized>(
            self,
            _: &'static str,
            _: u32,
            variant: &'static str,
            value: &T,
        ) -> Result<Node, NotDigested> {
            Ok(Node::variant(
                variant,
                &mut [(POSITIONS[0], node_of(value)?)],
            ))
        }
        fn serialize_seq(self, len: Option<usize>) -> Result<Items, NotDigested> {
            Ok(Items {
                nodes: Vec::with_capacity(len.unwrap_or(0)),
                tuple: false,
                key: None,
            })
        }
        fn serialize_tuple(self, len: usize) -> Result<Items, NotDigested> {
            Ok(Items {
                nodes: Vec::with_capacity(len),
                tuple: true,
                key: None,
            })
        }
        fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Items, NotDigested> {
            self.serialize_tuple(len)
        }
        fn serialize_tuple_variant(
            self,
            _: &'static str,
            _: u32,
            variant: &'static str,
            len: usize,
        ) -> Result<Fields, NotDigested> {
            Ok(Fields {
                variant: Some(variant),
                fields: Vec::with_capacity(len),
            })
        }
        fn serialize_map(self, len: Option<usize>) -> Result<Items, NotDigested> {
            self.serialize_seq(len)
        }
        fn serialize_struct(self, _: &'static str, len: usize) -> Result<Fields, NotDigested> {
            Ok(Fields {
                variant: None,
                fields: Vec::with_capacity(len),
            })
        }
        fn serialize_struct_variant(
            self,
            _: &'static str,
            _: u32,
            variant: &'static str,
            len: usize,
        ) -> Result<Fields, NotDigested> {
            Ok(Fields {
                variant: Some(variant),
                fields: Vec::with_capacity(len),
            })
        }
    }

    impl ser::SerializeSeq for Items {
        type Ok = Node;
        type Error = NotDigested;
        fn serialize_element<T: Serialize + ?Sized>(
            &mut self,
            value: &T,
        ) -> Result<(), NotDigested> {
            self.nodes.push(node_of(value)?);
            Ok(())
        }
        fn end(self) -> Result<Node, NotDigested> {
            Ok(self.finish())
        }
    }

    impl ser::SerializeTuple for Items {
        type Ok = Node;
        type Error = NotDigested;
        fn serialize_element<T: Serialize + ?Sized>(
            &mut self,
            value: &T,
        ) -> Result<(), NotDigested> {
            self.nodes.push(node_of(value)?);
            Ok(())
        }
        fn end(self) -> Result<Node, NotDigested> {
            Ok(self.finish())
        }
    }

    impl ser::SerializeTupleStruct for Items {
        type Ok = Node;
        type Error = NotDigested;
        fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), NotDigested> {
            self.nodes.push(node_of(value)?);
            Ok(())
        }
        fn end(self) -> Result<Node, NotDigested> {
            Ok(self.finish())
        }
    }

    /// A map is the sequence of its entries, each the pair of its key's and value's nodes.
    impl ser::SerializeMap for Items {
        type Ok = Node;
        type Error = NotDigested;
        fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> Result<(), NotDigested> {
            self.key = Some(node_of(key)?);
            Ok(())
        }
        fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), NotDigested> {
            let key = self
                .key
                .take()
                .ok_or_else(|| NotDigested("a map value without its key".into()))?;
            self.nodes.push(Node::tuple([key, node_of(value)?]));
            Ok(())
        }
        fn end(self) -> Result<Node, NotDigested> {
            Ok(self.finish())
        }
    }

    impl ser::SerializeTupleVariant for Fields {
        type Ok = Node;
        type Error = NotDigested;
        fn serialize_field<T: Serialize + ?Sized>(&mut self, value: &T) -> Result<(), NotDigested> {
            let name = position(self.fields.len())?;
            self.fields.push((name, node_of(value)?));
            Ok(())
        }
        fn end(self) -> Result<Node, NotDigested> {
            Ok(self.finish())
        }
    }

    impl ser::SerializeStruct for Fields {
        type Ok = Node;
        type Error = NotDigested;
        fn serialize_field<T: Serialize + ?Sized>(
            &mut self,
            key: &'static str,
            value: &T,
        ) -> Result<(), NotDigested> {
            self.fields.push((key, node_of(value)?));
            Ok(())
        }
        fn end(self) -> Result<Node, NotDigested> {
            Ok(self.finish())
        }
    }

    impl ser::SerializeStructVariant for Fields {
        type Ok = Node;
        type Error = NotDigested;
        fn serialize_field<T: Serialize + ?Sized>(
            &mut self,
            key: &'static str,
            value: &T,
        ) -> Result<(), NotDigested> {
            self.fields.push((key, node_of(value)?));
            Ok(())
        }
        fn end(self) -> Result<Node, NotDigested> {
            Ok(self.finish())
        }
    }
}

#[cfg(test)]
mod tests;
