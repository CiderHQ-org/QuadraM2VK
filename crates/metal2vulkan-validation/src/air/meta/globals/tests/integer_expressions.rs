use super::*;

/// Every `icmp` predicate and integer opcode the local corpus's static initializers contain.
///
/// An expression the evaluator cannot fold leaves its global unknown, and an unknown
/// function-constant predicate is reported as "not enabled by default" -- a guess the stage
/// decodes spend as a fact, dropping outputs and system values. Reading only `eq` and `ne` left
/// 648 ordered comparisons unevaluated. The signed cases are the ones worth stating: the
/// evaluator carries every integer as a masked `u64`, so `-1` compares GREATER than `1` unless
/// the sign is recovered from the operand's own width.
#[test]
fn every_static_initializer_integer_expression_folds() {
    for (expression, expected) in [
        ("%a = add i16 %one, 15", 16),
        ("%a = sub i16 %one, 15", 0xfff2),
        ("%a = mul i16 %one, 15", 15),
        ("%a = udiv i16 %seven, 2", 3),
        ("%a = urem i16 %seven, 2", 1),
        ("%a = sdiv i16 %minus_one, 1", 0xffff),
        ("%a = srem i16 %seven, 2", 1),
        ("%a = and i16 %seven, 2", 2),
        ("%a = or i16 %one, 2", 3),
        ("%a = xor i16 %one, 3", 2),
        ("%a = shl i16 %one, 3", 8),
        ("%a = lshr i16 %minus_one, 12", 0xf),
        // `ashr` reads the operand as signed: -1 stays -1 however far it is shifted.
        ("%a = ashr i16 %minus_one, 12", 0xffff),
        ("%a = icmp ugt i16 %seven, 2", 1),
        ("%a = icmp uge i16 %seven, 7", 1),
        ("%a = icmp ult i16 %seven, 2", 0),
        ("%a = icmp ule i16 %seven, 7", 1),
        // -1 is the LARGEST i16 unsigned and the SMALLEST signed.
        ("%a = icmp ugt i16 %minus_one, 7", 1),
        ("%a = icmp sgt i16 %minus_one, 7", 0),
        ("%a = icmp slt i16 %minus_one, 7", 1),
        ("%a = icmp sge i16 %minus_one, %minus_one", 1),
        ("%a = icmp sle i16 %minus_one, 7", 1),
        ("%a = icmp eq i16 %seven, 7", 1),
        ("%a = icmp ne i16 %seven, 7", 0),
        ("%a = tail call i16 @llvm.umax.i16(i16 %seven, i16 2)", 7),
        ("%a = tail call i16 @llvm.umin.i16(i16 %seven, i16 2)", 2),
        (
            "%a = tail call i16 @llvm.umax.i16(i16 %minus_one, i16 2)",
            0xffff,
        ),
        (
            "%a = tail call i16 @llvm.smax.i16(i16 %minus_one, i16 2)",
            2,
        ),
        (
            "%a = tail call i16 @llvm.smin.i16(i16 %minus_one, i16 2)",
            0xffff,
        ),
    ] {
        let ll = format!(
            r#"
@one = internal addrspace(2) global i16 1, align 2
@seven = internal addrspace(2) global i16 7, align 2
@minus_one = internal addrspace(2) global i16 -1, align 2
@folded = internal addrspace(2) global i16 undef, align 2

define internal void @_GLOBAL__sub_I_fold() section "air.static_init" {{
entry:
  %one = load i16, ptr addrspace(2) @one
  %seven = load i16, ptr addrspace(2) @seven
  %minus_one = load i16, ptr addrspace(2) @minus_one
  {expression}
  store i16 %a, ptr addrspace(2) @folded
  ret void
}}
"#
        );
        assert_eq!(
            static_init_int_global_values(&ll).get("@folded"),
            Some(&expected),
            "{expression}"
        );
    }
}

/// A division by zero must stay UNKNOWN rather than fold to a value the shader would never see,
/// the way [`out_of_width_shift_is_not_folded`] already requires of an over-wide shift.
#[test]
fn an_undefined_division_does_not_fold() {
    for expression in [
        "%a = udiv i16 %seven, 0",
        "%a = urem i16 %seven, 0",
        "%a = sdiv i16 %seven, 0",
        "%a = srem i16 %seven, 0",
    ] {
        let ll = format!(
            r#"
@seven = internal addrspace(2) global i16 7, align 2
@folded = internal addrspace(2) global i16 undef, align 2

define internal void @_GLOBAL__sub_I_fold() section "air.static_init" {{
entry:
  %seven = load i16, ptr addrspace(2) @seven
  {expression}
  store i16 %a, ptr addrspace(2) @folded
  ret void
}}
"#
        );
        assert_eq!(
            static_init_int_global_values(&ll).get("@folded"),
            None,
            "{expression}"
        );
    }
}
