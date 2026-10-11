use super::*;

#[test]
fn function_constant_metadata_definedness_distinguishes_zero_from_omission() {
    for (ty, zero) in [
        ("i8", "0"),
        ("float", "0.0"),
        ("<2 x i16>", "zeroinitializer"),
    ] {
        for (initializer, expected) in [("undef", 0), (zero, 1)] {
            let ll = format!(
                r#"@flag = internal addrspace(2) constant {ty} {initializer}, section "air.fc_initializer"
@defined = internal addrspace(2) global i8 undef
 define internal void @initialize() section "air.static_init" {{
entry:
 %value = call i1 @air.is_function_constant_defined(ptr addrspace(2) @flag)
 %byte = zext i1 %value to i8
 store i8 %byte, ptr addrspace(2) @defined
 ret void
}}
"#
            );
            assert_eq!(
                static_init_int_global_values(&ll).get("@defined"),
                Some(&expected),
                "{ty} {initializer}"
            );
        }
    }
}

#[test]
fn omitted_function_constants_are_unknown_but_explicit_zero_is_known() {
    for ty in ["i8", "i16", "i32", "i64", "<2 x i16>"] {
        let initializer = format!("internal addrspace(2) externally_initialized constant {ty} undef, section \"air.fc_initializer\"");
        assert!(integer_initializer(&initializer).is_none(), "{ty}");
    }
    assert!(matches!(
        integer_initializer("internal constant i32 0"),
        Some(StaticValue::Int(0))
    ));
    assert!(
        matches!(integer_initializer("internal constant <2 x i16> zeroinitializer"), Some(StaticValue::Vector(values)) if values == [0, 0])
    );
}

#[test]
fn function_constant_predicate_normalization_preserves_an_enabled_default() {
    let ll = r#"
@fc.MTL_FC_INIT_1_b = internal addrspace(2) externally_initialized constant i8 1, section "air.fc_initializer", align 1
@predicate = internal addrspace(2) global i8 0, align 1

define internal void @_GLOBAL__sub_I_fc() section "air.static_init" {
entry:
  %value = load i8, ptr addrspace(2) @fc.MTL_FC_INIT_1_b
  %normalized = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %value)
  store i8 %normalized, ptr addrspace(2) @predicate
  ret void
}
"#;

    assert_eq!(
        static_init_int_global_values(ll).get("@predicate"),
        Some(&1)
    );
}

/// Explicit 64-bit values seed constructor mirrors without inventing an omitted value.
#[test]
fn a_64_bit_function_constant_initializer_is_seeded() {
    let ll = r#"
@bits.MTL_FC_INIT_1_m = internal addrspace(2) externally_initialized constant i64 0, section "air.fc_initializer", align 8
@mirror = internal addrspace(2) global i64 undef, align 8
@enabled = internal addrspace(2) global i8 undef, align 1

define internal void @_GLOBAL__sub_I_wide() section "air.static_init" {
entry:
  %value = load i64, ptr addrspace(2) @bits.MTL_FC_INIT_1_m
  store i64 %value, ptr addrspace(2) @mirror
  %high = lshr i64 %value, 63
  %narrow = trunc i64 %high to i8
  store i8 %narrow, ptr addrspace(2) @enabled
  ret void
}
"#;
    let values = static_init_int_global_values(ll);
    assert_eq!(values.get("@mirror"), Some(&0));
    assert_eq!(values.get("@enabled"), Some(&0));
}

/// A store carries its OWN operand width. Narrowing every stored integer to 32 bits dropped the
/// high half of a `store i64`, so a predicate reading a bit above 31 out of the stored mirror --
/// `lshr i64 %x, 63`, the shape two corpus kernels use -- read zero whatever the constant said.
#[test]
fn a_stored_64_bit_value_keeps_its_high_half() {
    let ll = r#"
@bits = internal addrspace(2) global i64 1311768467463790320, align 8
@mirror = internal addrspace(2) global i64 undef, align 8
@high = internal addrspace(2) global i32 undef, align 4
@low = internal addrspace(2) global i32 undef, align 4
@residue = internal addrspace(2) global i64 undef, align 8

define internal void @_GLOBAL__sub_I_wide() section "air.static_init" {
entry:
  %value = load i64, ptr addrspace(2) @bits
  store i64 %value, ptr addrspace(2) @mirror
  %stored = load i64, ptr addrspace(2) @mirror
  %shifted = lshr i64 %stored, 32
  %top = trunc i64 %shifted to i32
  store i32 %top, ptr addrspace(2) @high
  %bottom = trunc i64 %stored to i32
  store i32 %bottom, ptr addrspace(2) @low
  %widened = zext i32 %bottom to i64
  %residue = lshr i64 %widened, 32
  store i64 %residue, ptr addrspace(2) @residue
  ret void
}
"#;
    let values = static_init_int_global_values(ll);
    assert_eq!(values.get("@high"), Some(&0x1234_5678));
    assert_eq!(values.get("@low"), Some(&0x9abc_def0));
    // `trunc ... to i32` discards the half it exists to discard. Masking only `i8` and `i16`
    // left the high half on the value, and only the next STORE happened to hide it -- widening
    // the truncated value again brings it back.
    assert_eq!(values.get("@residue"), Some(&0));
}
