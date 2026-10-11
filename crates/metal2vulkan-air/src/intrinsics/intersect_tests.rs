//! `air.intersect`: the tag grammar, the operand roles, and the result struct, checked against
//! declarations exactly as Apple's compiler emits them.

use super::tests::{classify_self, meter, sym};
use super::*;
use crate::ir::{AddrSpace, Type};

/// The IR type of one LLVM type spelling: `i1`..`i64`, `float`, `<N x float>`, `ptr`,
/// `ptr addrspace(N)`, or a literal struct `{ T, ... }`.
fn llvm_ty(t: &mut TypeTable, text: &str) -> TypeId {
    let m = &mut meter();
    let text = text.trim();
    if let Some(body) = text.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        let fields: Vec<TypeId> = split_top(body).iter().map(|f| llvm_ty(t, f)).collect();
        return t
            .intern(
                Type::Struct {
                    fields: fields.into(),
                    packed: false,
                    name: None,
                },
                m,
            )
            .unwrap();
    }
    if let Some(inner) = text.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        let (len, elem) = inner.split_once(" x ").unwrap();
        let elem = llvm_ty(t, elem);
        let len = len.trim().parse().unwrap();
        return t.intern(Type::Vector { elem, len }, m).unwrap();
    }
    if let Some(space) = text.strip_prefix("ptr addrspace(") {
        let space = space.strip_suffix(')').unwrap().parse().unwrap();
        let space = AddrSpace::from_raw(space).unwrap();
        return t.intern(Type::Ptr { space }, m).unwrap();
    }
    let ty = match text {
        "ptr" => Type::Ptr {
            space: AddrSpace::Thread,
        },
        "i1" => Type::Int(1),
        "i8" => Type::Int(8),
        "i32" => Type::Int(32),
        "i64" => Type::Int(64),
        "float" => Type::Float,
        other => panic!("type {other}"),
    };
    t.intern(ty, m).unwrap()
}

/// Split a comma list at its top level (not inside `<...>` or `{...}`).
fn split_top(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (i, c) in text.char_indices() {
        match c {
            '<' | '{' | '(' => depth += 1,
            '>' | '}' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(text[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = text[start..].trim();
    if !last.is_empty() {
        out.push(last);
    }
    out
}

/// Classify an LLVM declaration `declare <ret> @<name>(<params>)` (attributes stripped).
fn declaration(decl: &str) -> (String, DeclClass) {
    let decl = decl.strip_prefix("declare ").unwrap();
    let at = decl.find(" @").unwrap();
    let ret_text = &decl[..at];
    let rest = &decl[at + 2..];
    let open = rest.find('(').unwrap();
    let name = &rest[..open];
    let params_text = rest[open + 1..].strip_suffix(')').unwrap();
    let mut t = TypeTable::new();
    let ret = llvm_ty(&mut t, ret_text);
    let params: Vec<TypeId> = split_top(params_text)
        .iter()
        .map(|p| llvm_ty(&mut t, p))
        .collect();
    let f = t
        .intern(
            Type::Function {
                ret,
                params: params.into(),
                vararg: false,
            },
            &mut meter(),
        )
        .unwrap();
    let class = classify(name.as_bytes(), sym(), f, &t).unwrap();
    (name.to_string(), class)
}

/// Declarations exactly as the local corpus and compiled probes declare them (parameter
/// attributes removed): the tagless intersector, each operand-adding tag, and the widest form.
const OBSERVED: &[&str] = &[
    "declare { i32, float, i32, i32, ptr addrspace(1) } @air.intersect(<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)",
    "declare { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.triangle_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)",
    "declare { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1, float } @air.intersect.triangle_data.curve_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)",
    "declare { i32, float, i32, i32, ptr addrspace(1), i32, i32, <2 x float>, i1 } @air.intersect.instancing.triangle_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)",
    "declare { i32, float, i32, i32, ptr addrspace(1), i32, i32, <2 x float>, i1 } @air.intersect.instancing.triangle_data.extended_limits(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)",
    "declare { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } @air.intersect.instancing.world_space_data.primitive_motion.instance_motion(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, float, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)",
    "declare { i32, float, i32, i32, ptr addrspace(1) } @air.intersect.primitive_motion(<3 x float>, <3 x float>, float, float, ptr addrspace(1), float, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)",
    "declare { i32, float, i32, i32, ptr addrspace(1), i8 } @air.intersect.multi_level_instancing(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i8, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)",
    "declare { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.intersection_function_buffer.triangle_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), i64, i64, ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i32, i32)",
    "declare { i32, float, i32, i32, ptr addrspace(1), i8, <2 x float>, i1, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } @air.intersect.intersection_function_buffer.multi_level_instancing.triangle_data.world_space_data.user_data.primitive_motion.instance_motion(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, float, ptr addrspace(1), i64, i64, ptr addrspace(1), ptr, i64, i8, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, i32)",
];

fn intersect(class: DeclClass) -> Intersect {
    match class {
        DeclClass::Air(AirIntrinsic::Intersect(i)) => i,
        other => panic!("{other:?}"),
    }
}

/// Every observed declaration decodes, and its tags are the ones its name spells.
#[test]
fn observed_intersect_declarations_decode() {
    for decl in OBSERVED {
        let (name, class) = declaration(decl);
        let i = intersect(class);
        let t = i.tags;
        assert_eq!(t.triangle_data, name.contains(".triangle_data"), "{name}");
        assert_eq!(
            t.function_buffer,
            name.contains("function_buffer"),
            "{name}"
        );
        assert_eq!(t.world_space_data, name.contains("world_space"), "{name}");
    }
    let (_, widest) = declaration(OBSERVED.last().unwrap());
    let widest = intersect(widest);
    assert_eq!(widest.signature().params().len(), super::sig::MAX_OPERANDS);
    assert_eq!(widest.tags.result().fields().count(), 16);
}

/// A declaration whose operands or result disagree with its tags is a signature mismatch.
#[test]
fn intersect_declarations_must_match_their_tags() {
    for decl in [
        // `instancing` without its mask.
        "declare { i32, float, i32, i32, ptr addrspace(1), i32, i32 } @air.intersect.instancing(<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)",
        // `triangle_data` without its result fields.
        "declare { i32, float, i32, i32, ptr addrspace(1) } @air.intersect.triangle_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)",
        // The payload in device memory.
        "declare { i32, float, i32, i32, ptr addrspace(1) } @air.intersect(<3 x float>, <3 x float>, float, float, ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1)",
    ] {
        let (name, class) = declaration(decl);
        assert_eq!(class, DeclClass::Mismatch, "{name}");
    }
}

/// Tags are read in ABI bit order; an out-of-order, repeated, or inconsistent spelling is not an
/// intersector's name and stays unknown.
#[test]
fn intersect_tags_follow_the_abi_order() {
    for name in [
        "air.intersect",
        "air.intersect.instancing",
        "air.intersect.multi_level_instancing.world_space_data",
        "air.intersect.intersection_function_buffer.user_data",
        "air.intersect.triangle_data.curve_data.extended_limits",
    ] {
        assert!(
            matches!(
                classify_self(name),
                DeclClass::Air(AirIntrinsic::Intersect(_))
            ),
            "{name}"
        );
    }
    for name in [
        "air.intersect.",
        "air.intersect.triangle_data.instancing",
        "air.intersect.instancing.multi_level_instancing",
        "air.intersect.triangle_data.triangle_data",
        "air.intersect.user_data",
        "air.intersect.world_space_data",
        "air.intersect.instance_motion",
        "air.intersect.triangle_data.mystery",
        "air.intersection_query.triangle_data",
    ] {
        assert_eq!(classify_self(name), DeclClass::Unknown, "{name}");
    }
}

/// Every tag combination the grammar admits has a consistent layout: the base operands, one
/// optional operand per tag in the documented place, the nine settings, and the flags. The
/// counts match an independent formula (the old engine's `meta::intersections` knowledge).
#[test]
fn intersect_layouts_follow_the_tags() {
    let mut seen = 0;
    for bits in 0u32..(1 << 10) {
        let tokens: Vec<&str> = [
            "intersection_function_buffer",
            "instancing",
            "multi_level_instancing",
            "triangle_data",
            "curve_data",
            "world_space_data",
            "user_data",
            "primitive_motion",
            "instance_motion",
            "extended_limits",
        ]
        .iter()
        .enumerate()
        .filter(|(i, _)| bits & (1 << i) != 0)
        .map(|(_, t)| *t)
        .collect();
        let name = std::iter::once("air.intersect")
            .chain(tokens.iter().copied())
            .collect::<Vec<_>>()
            .join(".");
        let Some(AirIntrinsic::Intersect(i)) = AirIntrinsic::parse(&name) else {
            continue;
        };
        seen += 1;
        let t = i.tags;
        let multi = matches!(t.instancing, Instancing::Multi);
        let count =
            18 + match t.instancing {
                Instancing::None => 0,
                Instancing::Single => 2,
                Instancing::Multi => 5,
            } + usize::from(t.function_buffer) * 4
                + usize::from(t.motion())
                + usize::from(t.user_data);
        let sig = i.signature();
        assert_eq!(sig.params().len(), count, "{name}");
        assert!(sig.params().len() <= super::sig::MAX_OPERANDS, "{name}");
        assert_eq!(i.index(IntersectRole::Structure), Some(4), "{name}");
        let table = 5 + usize::from(t.instanced()) + usize::from(t.motion());
        let first_table = if t.function_buffer {
            i.index(IntersectRole::FunctionBuffer)
        } else {
            i.index(IntersectRole::Table)
        };
        assert_eq!(first_table, Some(table), "{name}");
        assert_eq!(i.index(IntersectRole::MaxLevels).is_some(), multi, "{name}");
        assert_eq!(
            i.index(IntersectRole::IdentityTransforms).is_some(),
            t.instanced(),
            "{name}"
        );
        // The handles are the structure and, without a function buffer, the table.
        let handles: Vec<_> = sig.handles().collect();
        let mut want = vec![(4, HandleKind::AccelerationStructure)];
        if !t.function_buffer {
            want.push((table, HandleKind::IntersectionFunctionTable));
        }
        assert_eq!(handles, want, "{name}");
        // The declaration the signature derives classifies as the same intrinsic.
        assert_eq!(
            classify_self(&name),
            DeclClass::Air(AirIntrinsic::Intersect(i)),
            "{name}"
        );
        // Result fields: the index of each field is its position.
        let shape = t.result();
        for (n, f) in shape.fields().enumerate() {
            assert_eq!(shape.index(f), u32::try_from(n).ok(), "{name} {f:?}");
        }
    }
    // Instancing has three states, and the consistency rules remove the rest.
    assert!(seen > 100, "{seen}");
}
