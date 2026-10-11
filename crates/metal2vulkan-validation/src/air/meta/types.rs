use std::collections::HashMap;

/// A reconstructed AIR aggregate type built from a buffer's `air.struct_type_info` metadata. It
/// restores the source layout when an emitted buffer parameter is represented as a bare element
/// pointer (homogeneous/nested aggregates → `T*` with physical/multi-level access chains). The
/// access-chain indices are valid struct-navigation indices for this reconstructed type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AirType {
    /// 32-bit scalar leaf.
    Scalar(AirScalar),
    /// `floatN` / `uintN` / `intN` vector.
    Vec { scalar: AirScalar, lanes: u32 },
    /// `packed_floatN` / `packed_uintN` / `packed_intN`, laid out as scalar array stride 4.
    PackedVec { scalar: AirScalar, lanes: u32 },
    /// Fixed-size metadata array, e.g. `uint[4]` represented as `i32 array_len = 4`.
    Array { elem: Box<AirType>, len: u32 },
    /// `floatCxR`/`halfCxR` matrix → `{ [cols x vec(rows)] }` (matches Metal's
    /// `metal::matrix` wrapper struct).
    Matrix {
        scalar: AirScalar,
        cols: u32,
        rows: u32,
    },
    /// Nested struct, members in declaration order with explicit AIR byte offsets.
    Struct(Vec<AirMember>),
    /// A member AIR named with a type it does not describe the interior of -- a user struct or
    /// class (`Espresso::padding_params_t`, `VFX_RE_C_LightSpot`) that `air.struct_type_info`
    /// mentions by name only. The member tuple still carries its declared byte size, and that size
    /// is the whole of what AIR states, so it is the whole of what this variant claims.
    ///
    /// Modelling such a member as a concrete leaf invents an interior: before this variant existed
    /// the decoder answered `float` for every one of them, which sized a 408-byte member at four
    /// bytes and made it fail to match the emitted member it describes. Over 2880 corpus sources
    /// 2357 members across 817 modules are opaque, and the resulting shape mismatch discarded the
    /// declared offsets for the entire buffer, not just the member that provoked it.
    ///
    /// The translator renders the declared bytes as a concrete storage shape wherever it must
    /// hand a real SPIR-V or LLVM type to the emitter.
    Opaque { size: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AirMember {
    pub offset: u32,
    pub ty: AirType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AirScalar {
    Float,
    Half,
    UInt,
    SInt,
    ULong,
    SLong,
    UShort,
    SShort,
    UChar,
    Bool,
}

/// Map a primitive AIR type-name (`float`, `float2`, `packed_float3`, `float3x4`, `half4`, ...)
/// to an `AirType`. Unknown user struct/class names return `None`.
pub(crate) fn primitive_air_type_from_name(name: &str) -> Option<AirType> {
    let raw = name.trim();
    let (n, packed) = match raw.strip_prefix("packed_") {
        Some(n) => (n, true),
        None => (raw, false),
    };
    for (base, scalar) in [
        ("ulong", AirScalar::ULong),
        ("long", AirScalar::SLong),
        ("ushort", AirScalar::UShort),
        ("short", AirScalar::SShort),
        ("uchar", AirScalar::UChar),
        // `char` (signed 8-bit) shares i8 storage with `uchar`; SPIR-V integer types are
        // signedness-agnostic, so the storage type is identical. Without this entry the
        // struct-member fallback in `air_type_from_name` silently mistypes `char` as `float`,
        // widening an i8 member to 32 bits (seen in MTLGPUBVHBuilderQueueEntry).
        ("char", AirScalar::UChar),
        ("float", AirScalar::Float),
        ("half", AirScalar::Half),
        ("uint", AirScalar::UInt),
        ("int", AirScalar::SInt),
        ("bool", AirScalar::Bool),
    ] {
        if let Some(rest) = n.strip_prefix(base) {
            return parse_dims(rest, scalar, packed);
        }
    }
    None
}

/// Decode one `air.struct_type_info` member from the two facts the tuple carries: the type name and
/// the declared byte size. A name with a primitive lowering wins; anything else is
/// [`AirType::Opaque`] at the declared size, because the size is all AIR said about it.
fn member_air_type(name: &str, size: u32) -> AirType {
    primitive_air_type_from_name(name).unwrap_or(AirType::Opaque { size })
}

fn parse_dims(rest: &str, scalar: AirScalar, packed: bool) -> Option<AirType> {
    if rest.is_empty() {
        return Some(AirType::Scalar(scalar));
    }
    if let Some((c, r)) = rest.split_once('x') {
        let cols = c.trim().parse().ok()?;
        let rows = r.trim().parse().ok()?;
        return Some(AirType::Matrix { scalar, cols, rows });
    }
    match rest.trim().parse::<u32>() {
        Ok(n) if n >= 1 && packed => Some(AirType::PackedVec { scalar, lanes: n }),
        Ok(n) if n >= 2 => Some(AirType::Vec { scalar, lanes: n }),
        _ => None,
    }
}

/// One token of a metadata node body, for parsing `air.struct_type_info`.
pub(crate) enum Tok {
    Int(u32),
    Str(String),
    Ref(u32),
}

/// Tokenize a metadata node body into Int(`i32 N`) / Str(`!"..."`) / Ref(`!N`) tokens, dropping
/// anything else (`ptr @f`, addrspace globals, …).
pub(crate) fn tokenize(body: &str) -> Vec<Tok> {
    let mut out = vec![];
    for field in crate::air::text::split_top_level(body, ',') {
        let f = field.trim();
        if let Some(rest) = f.strip_prefix("i32 ") {
            if let Ok(v) = rest.trim().parse::<u32>() {
                out.push(Tok::Int(v));
            }
        } else if let Some(rest) = f.strip_prefix("!\"") {
            out.push(Tok::Str(rest.strip_suffix('"').unwrap_or(rest).to_string()));
        } else if let Some(rest) = f.strip_prefix('!') {
            if let Ok(v) = rest.trim().parse::<u32>() {
                out.push(Tok::Ref(v));
            }
        }
    }
    out
}

/// One member of an `air.struct_type_info` node, exactly as AIR spells it.
///
/// AIR writes a member as a 5-tuple `i32 offset, i32 size, i32 array_len, !"type", !"name"`,
/// optionally PREFIXED by `!"air.struct_type_info", !N` naming the member's OWN member list, and
/// optionally SUFFIXED by `!"air.indirect_argument"` plus one operand: a node ref `!N` for a
/// resource handle, or a bare `i32` for an argument-buffer wrapper carrying its own argument id.
pub(crate) struct MemberTuple {
    pub(crate) offset: u32,
    /// Per-element declared size. For an array member this is the ELEMENT size, not the total.
    pub(crate) size: u32,
    /// Zero for a scalar member; otherwise the number of elements.
    pub(crate) array_len: u32,
    pub(crate) tyname: String,
    /// The `!"air.struct_type_info", !N` prefix: this member's own member list.
    pub(crate) nested: Option<u32>,
    /// The `!"air.indirect_argument", !N` suffix: the argument-buffer entry this member holds.
    pub(crate) argument_node: Option<u32>,
    /// The `!"air.indirect_argument", i32 N` suffix: this wrapper's own argument-id base.
    pub(crate) argument_wrapper_id: Option<u32>,
}

impl MemberTuple {
    /// How many bytes the member occupies by AIR's own account, from its declared per-element size
    /// and array length. This is what AIR states, not a recomputation from the type name, so it is
    /// the quantity two members are compared on to decide whether they overlap.
    pub(super) fn declared_extent(&self) -> u64 {
        u64::from(self.size) * u64::from(self.array_len.max(1))
    }
}

/// Walk one `air.struct_type_info` node body into its member tuples.
///
/// The single decoder for AIR's member-tuple grammar: [`parse_struct_info`] reads it as a layout
/// and `embedded::collect_argument_members` reads it as a resource list, and they used to walk it
/// separately. `struct_member_starts_at` does NOT treat a suffix as a member start, so scanning to
/// the next member start reads the suffix that belongs to THIS member.
pub(crate) fn member_tuples(toks: &[Tok]) -> Vec<MemberTuple> {
    let mut out = vec![];
    let mut i = 0;
    while i < toks.len() {
        let mut nested = None;
        if let (Some(Tok::Str(s)), Some(Tok::Ref(x))) = (toks.get(i), toks.get(i + 1)) {
            if s == "air.struct_type_info" {
                nested = Some(*x);
                i += 2;
            }
        }
        let (offset, size, array_len, tyname) = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
        ) {
            (
                Some(Tok::Int(offset)),
                Some(Tok::Int(size)),
                Some(Tok::Int(array_len)),
                Some(Tok::Str(t)),
            ) => (*offset, *size, *array_len, t.clone()),
            _ => break,
        };
        i += 5; // 3 ints + type + name
        let mut argument_node = None;
        let mut argument_wrapper_id = None;
        while i < toks.len() && !struct_member_starts_at(toks, i) {
            match (toks.get(i), toks.get(i + 1)) {
                (Some(Tok::Str(s)), Some(Tok::Ref(x))) if s == "air.indirect_argument" => {
                    argument_node = Some(*x);
                }
                (Some(Tok::Str(s)), Some(Tok::Int(id))) if s == "air.indirect_argument" => {
                    argument_wrapper_id = Some(*id);
                }
                _ => {}
            }
            i += 1;
        }
        out.push(MemberTuple {
            offset,
            size,
            array_len,
            tyname,
            nested,
            argument_node,
            argument_wrapper_id,
        });
    }
    out
}

/// Parse an `air.struct_type_info` node into an `AirType::Struct`, one member per
/// [`MemberTuple`], or `None` when the node does not describe storage member-by-member.
///
/// `None` is not a parse failure; it is the answer that AIR's member list cannot be read as a
/// layout, and every caller already has the fact that outranks it -- the enclosing member's
/// declared size, or `air.arg_type_size` for a whole argument. See
/// [`members_are_disjoint`] for the shapes that produce it.
pub(super) fn parse_struct_info(
    nodes: &HashMap<u32, String>,
    id: u32,
    depth: u32,
) -> Option<AirType> {
    if depth > 16 {
        return None;
    }
    let body = nodes.get(&id)?;
    let tuples = member_tuples(&tokenize(body));
    if tuples.is_empty() || !members_are_disjoint(&tuples) {
        return None;
    }
    let members = tuples
        .into_iter()
        .map(|tuple| {
            let mut ty = match tuple.nested {
                // A nested node that does not describe storage leaves the member exactly as well
                // described as before: AIR stated its size right here in the tuple.
                Some(x) => parse_struct_info(nodes, x, depth + 1)
                    .unwrap_or_else(|| storage_air_type_for_size(tuple.size)),
                None if member_holds_resource_handle(nodes, tuple.argument_node) => {
                    storage_air_type_for_size(tuple.size)
                }
                None => member_air_type(&tuple.tyname, tuple.size),
            };
            if tuple.array_len > 0 {
                ty = AirType::Array {
                    elem: Box::new(ty),
                    len: tuple.array_len,
                };
            }
            AirMember {
                offset: tuple.offset,
                ty,
            }
        })
        .collect();
    Some(AirType::Struct(members))
}

/// True when a member's `air.indirect_argument` node describes a resource the argument buffer
/// references by handle, rather than a value the buffer stores inline.
///
/// Metal spells such a member with the type it points AT — `!"char"` for a `device char *`,
/// `!"texture2d<half, sample>"` for a texture, an opaque class name for a `device Foo *` — while the
/// member itself stores a handle of the declared size. Reading the name as the member's storage
/// sizes it wrong and shifts every following member's meaning, so the nested node's role decides
/// instead. `air.indirect_constant` is the one role that means "stored inline"; `air.buffer`,
/// `air.texture`, `air.sampler`, `air.indirect_buffer`, and the pipeline-state / command-buffer
/// roles are all references. An unrecognized role is treated as a reference because that is the
/// safe direction: the declared size describes the bytes either way, while a handle mistaken for
/// its pointee corrupts the rest of the layout.
fn member_holds_resource_handle(nodes: &HashMap<u32, String>, argument_node: Option<u32>) -> bool {
    let Some(body) = argument_node.and_then(|node| nodes.get(&node)) else {
        return false;
    };
    super::primary_role(&super::role_strings(body)).is_some_and(|role| role != "indirect_constant")
}

pub(super) fn struct_member_starts_at(toks: &[Tok], mut i: usize) -> bool {
    if let (Some(Tok::Str(s)), Some(Tok::Ref(_))) = (toks.get(i), toks.get(i + 1)) {
        if s == "air.struct_type_info" {
            i += 2;
        }
    }
    matches!(
        (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
            toks.get(i + 4),
        ),
        (
            Some(Tok::Int(_)),
            Some(Tok::Int(_)),
            Some(Tok::Int(_)),
            Some(Tok::Str(_)),
            Some(Tok::Str(_)),
        )
    )
}

/// Whether a node's members each own their bytes outright.
///
/// AIR states two shapes with overlapping members, and neither is usable as a layout:
///
/// - A **union** gives every member the SAME offset -- `half2 maskCoord` and `half vectorOpacity`
///   both at 0 are two readings of one four-byte run.
/// - A **bitfield** gives each field the offset of the byte its run starts in beside the size of
///   the whole underlying storage unit -- `uint pixel_x` at 0, `uint pixel_y` at 1, `uint matid`
///   at 3, each four bytes wide, is three views of one four-byte word, not a struct reaching to
///   byte seven.
///
/// What settles both is who consumes this layout: the interface pass builds a SPIR-V struct type
/// from it, and a `Block` cannot decorate two members onto the same bytes. An overlapping member
/// list therefore does not become an overlapping struct -- it makes the emitter give up on the
/// WHOLE buffer and address it as raw words, losing every typed access chain in it, not just the
/// overlapping member. Refusing the node here costs far less: the caller always has the size AIR
/// declared for this member, or `air.arg_type_size` for a whole argument.
///
/// So overlap is the test, at any offset and by any amount. Accepting unions was measured: it
/// re-typed 379 corpus sources and cost `87c1f8ca` its `PKMetalStrokeVertex` access chains,
/// widening a `writeonly` buffer to `ReadWrite`. Requiring a strictly increasing offset instead --
/// which is what this used to do, for nested nodes only -- misses the bitfield, whose offsets do
/// increase strictly while its members still overlap, on 4 corpus sources.
fn members_are_disjoint(tuples: &[MemberTuple]) -> bool {
    tuples.windows(2).all(|pair| {
        u64::from(pair[0].offset) + pair[0].declared_extent() <= u64::from(pair[1].offset)
    })
}

/// Render `size` opaque bytes as a concrete AIR leaf, for the places that must hand a real
/// SPIR-V/LLVM type to the emitter. Word-sized runs stay words so natural alignment survives; a
/// size that is not a multiple of four falls to bytes.
pub(crate) fn storage_air_type_for_size(size: u32) -> AirType {
    match size {
        0 | 4 => AirType::Scalar(AirScalar::UInt),
        1 => AirType::Scalar(AirScalar::UChar),
        2 => AirType::Scalar(AirScalar::UShort),
        8 => AirType::Scalar(AirScalar::ULong),
        n if n % 4 == 0 => AirType::Array {
            elem: Box::new(AirType::Scalar(AirScalar::UInt)),
            len: n / 4,
        },
        n => AirType::Array {
            elem: Box::new(AirType::Scalar(AirScalar::UChar)),
            len: n,
        },
    }
}

/// The `air.struct_type_info` ref (`!N`) named directly in a buffer-arg node body, if any.
pub(super) fn struct_info_ref(body: &str) -> Option<u32> {
    let p = body.find("air.struct_type_info")?;
    let after = &body[p..];
    let bang = after.find(", !")? + 3;
    let digits: String = after[bang..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn member_type_and_name_commas_do_not_split_metadata_operands() {
        let body = r#"i32 0, i32 8, i32 0, !"Wrapper<A, B>", !"first, field", !"air.indirect_argument", !7, i32 8, i32 4, i32 0, !"uint", !"second""#;
        let tokens = tokenize(body);
        assert_eq!(tokens.len(), 12);
        let members = member_tuples(&tokens);
        assert_eq!(members.len(), 2);
        assert_eq!(members[0].tyname, "Wrapper<A, B>");
        assert_eq!(members[0].argument_node, Some(7));
        assert_eq!(members[1].offset, 8);
        assert_eq!(members[1].tyname, "uint");
    }
}
