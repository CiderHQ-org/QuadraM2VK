//! AIR-level Metal function-constant specialization: the authored values baked into the stable
//! `air.fc_initializer` globals before metadata parsing, so the static-initializer analysis sees
//! only the selected program. Validation's copy of the old engine's specializer, which the
//! reflection of a case's entry runs with the case's constants, as the old engine's `inspect` did.

use std::borrow::Cow;
use std::collections::BTreeMap;

/// Failure to apply caller-supplied AIR function-constant values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionConstantError {
    DuplicateIndex(u32),
    UnknownIndices(Vec<u32>),
    InvalidValue {
        index: u32,
        error: ConstantValueError,
    },
    MissingInitializerSection {
        index: u32,
    },
    InvalidInitializer {
        index: u32,
        type_name: String,
    },
}

/// Invalid or unsupported exact-byte payload for a declared function constant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstantValueError {
    UnsupportedType {
        type_name: String,
    },
    UnsupportedVectorType {
        type_name: String,
    },
    InvalidVectorLanes {
        type_name: String,
        reason: String,
    },
    UnsupportedVectorLanes {
        lanes: usize,
    },
    PayloadSize {
        type_name: String,
        expected: usize,
        actual: usize,
    },
    InvalidBoolean {
        value: u64,
    },
}

impl std::fmt::Display for FunctionConstantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateIndex(index) => write!(f, "AIR function-constant specialization contains duplicate index {index}"),
            Self::UnknownIndices(indices) => write!(f, "AIR function-constant specialization found no air.fc_initializer for index(es) {indices:?}"),
            Self::InvalidValue { index, error } => write!(f, "function constant {index}: {error}"),
            Self::MissingInitializerSection { index } => write!(f, "function constant {index}: MTL_FC_INIT global is not in the air.fc_initializer section"),
            Self::InvalidInitializer { index, type_name } => write!(f, "function constant {index}: air.fc_initializer is not an undef {type_name} global"),
        }
    }
}

impl std::error::Error for FunctionConstantError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidValue { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl std::fmt::Display for ConstantValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedType { type_name } => {
                write!(f, "unsupported function-constant type `{type_name}`")
            }
            Self::UnsupportedVectorType { type_name } => {
                write!(f, "unsupported vector type `{type_name}`")
            }
            Self::InvalidVectorLanes { type_name, reason } => {
                write!(f, "invalid vector lane count in `{type_name}`: {reason}")
            }
            Self::UnsupportedVectorLanes { lanes } => {
                write!(f, "unsupported vector lane count {lanes}")
            }
            Self::PayloadSize {
                type_name,
                expected,
                actual,
            } => write!(
                f,
                "payload has {actual} bytes, {type_name} requires {expected}"
            ),
            Self::InvalidBoolean { value } => {
                write!(f, "boolean payload must be 0 or 1, got {value}")
            }
        }
    }
}

impl std::error::Error for ConstantValueError {}

pub(crate) fn specialize_air_function_constants<'a>(
    air_ll: &'a str,
    values: &[(u32, Vec<u8>)],
) -> Result<Cow<'a, str>, FunctionConstantError> {
    if values.is_empty() {
        return Ok(Cow::Borrowed(air_ll));
    }

    let mut requested = BTreeMap::<u32, &[u8]>::new();
    for (index, bytes) in values {
        if requested.insert(*index, bytes).is_some() {
            return Err(FunctionConstantError::DuplicateIndex(*index));
        }
    }

    let constants = crate::air::meta::parse_function_constants(air_ll)
        .into_iter()
        .map(|constant| (constant.index, constant))
        .collect::<BTreeMap<_, _>>();
    let missing = requested
        .keys()
        .filter(|index| !constants.contains_key(index))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(FunctionConstantError::UnknownIndices(missing));
    }

    let literals = requested
        .iter()
        .map(|(index, bytes)| {
            let constant = &constants[index];
            llvm_typed_literal(&constant.type_name, &constant.abi_type_encoding, bytes)
                .map(|literal| (*index, (constant.type_name.as_str(), literal)))
                .map_err(|error| FunctionConstantError::InvalidValue {
                    index: *index,
                    error,
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;

    let mut output = String::with_capacity(air_ll.len());
    for line in air_ll.split_inclusive('\n') {
        let Some(index) = line
            .trim_start()
            .split_once(" = ")
            .filter(|(name, _)| name.starts_with('@'))
            .and_then(|(name, _)| {
                crate::air::meta::function_constant_symbol(name[1..].trim())
                    .map(|(_, index, _)| index)
            })
        else {
            output.push_str(line);
            continue;
        };
        let Some((type_name, literal)) = literals.get(&index) else {
            output.push_str(line);
            continue;
        };
        if !crate::air::meta::has_function_constant_section(line) {
            return Err(FunctionConstantError::MissingInitializerSection { index });
        }
        let replacement = line
            .split_once(" = ")
            .and_then(|(_, body)| crate::air::meta::global_initializer_body(body))
            .and_then(|typed| {
                let remainder = typed.strip_prefix(*type_name)?;
                if !remainder.starts_with(char::is_whitespace) {
                    return None;
                }
                let initializer = remainder.trim_start();
                let suffix = initializer.strip_prefix("undef")?;
                if !suffix.is_empty()
                    && !suffix.starts_with(char::is_whitespace)
                    && !suffix.starts_with(',')
                {
                    return None;
                }
                Some((typed, suffix))
            });
        let Some((typed, suffix)) = replacement else {
            return Err(FunctionConstantError::InvalidInitializer {
                index,
                type_name: (*type_name).to_owned(),
            });
        };
        output.push_str(&line[..line.len() - typed.len()]);
        output.push_str(literal);
        output.push_str(suffix);
    }

    Ok(Cow::Owned(
        crate::air::meta::specialize_function_constant_metadata(&output),
    ))
}

fn llvm_typed_literal(
    type_name: &str,
    abi_type_encoding: &str,
    bytes: &[u8],
) -> Result<String, ConstantValueError> {
    let bool_abi = abi_type_encoding == "b"
        || abi_type_encoding
            .rsplit_once('_')
            .is_some_and(|(_, scalar)| scalar == "b");
    if let Some(inner) = type_name
        .strip_prefix('<')
        .and_then(|value| value.strip_suffix('>'))
    {
        let (lanes, scalar) =
            inner
                .split_once(" x ")
                .ok_or_else(|| ConstantValueError::UnsupportedVectorType {
                    type_name: type_name.to_owned(),
                })?;
        let lanes =
            lanes
                .parse::<usize>()
                .map_err(|error| ConstantValueError::InvalidVectorLanes {
                    type_name: type_name.to_owned(),
                    reason: error.to_string(),
                })?;
        if !(1..=4).contains(&lanes) {
            return Err(ConstantValueError::UnsupportedVectorLanes { lanes });
        }
        let scalar_size = scalar_byte_size(scalar)?;
        // At most four lanes of at most eight bytes, as checked above.
        let required = scalar_size * lanes;
        if bytes.len() != required {
            return Err(ConstantValueError::PayloadSize {
                type_name: type_name.to_owned(),
                expected: required,
                actual: bytes.len(),
            });
        }
        let elements = bytes
            .chunks_exact(scalar_size)
            .map(|lane| {
                scalar_literal(scalar, lane, bool_abi).map(|literal| format!("{scalar} {literal}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(format!("{type_name} <{}>", elements.join(", ")));
    }

    let required = scalar_byte_size(type_name)?;
    if bytes.len() != required {
        return Err(ConstantValueError::PayloadSize {
            type_name: type_name.to_owned(),
            expected: required,
            actual: bytes.len(),
        });
    }
    scalar_literal(type_name, bytes, bool_abi).map(|literal| format!("{type_name} {literal}"))
}

fn scalar_byte_size(type_name: &str) -> Result<usize, ConstantValueError> {
    match type_name {
        "i1" | "i8" => Ok(1),
        "i16" | "half" | "bfloat" => Ok(2),
        "i32" | "float" => Ok(4),
        "i64" | "double" => Ok(8),
        _ => Err(ConstantValueError::UnsupportedType {
            type_name: type_name.to_owned(),
        }),
    }
}

fn scalar_literal(
    type_name: &str,
    bytes: &[u8],
    bool_abi: bool,
) -> Result<String, ConstantValueError> {
    let unsigned = || {
        let mut word = [0u8; 8];
        word[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(word)
    };
    match type_name {
        "i1" | "i8" | "i16" | "i32" | "i64" if type_name == "i1" || bool_abi => match unsigned() {
            0 => Ok("0".into()),
            1 => Ok("1".into()),
            value => Err(ConstantValueError::InvalidBoolean { value }),
        },
        "i8" | "i16" | "i32" | "i64" => Ok(unsigned().to_string()),
        "half" => Ok(format!("0xH{:04X}", unsigned())),
        "bfloat" => Ok(format!("0xR{:04X}", unsigned())),
        "float" => Ok(format!("f0x{:08X}", unsigned())),
        "double" => Ok(format!("0x{:016X}", unsigned())),
        _ => Err(ConstantValueError::UnsupportedType {
            type_name: type_name.to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specialization_preserves_other_initializers_and_unsupplied_predicates() {
        let ll = r#"@selected.MTL_FC_INIT_0_j = internal addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@other.MTL_FC_INIT_1_j = internal addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@working_copy = internal addrspace(2) global i32 19, align 4
define void @init() {
  %selected = call i1 @air.is_function_constant_defined(ptr addrspace(2) @selected.MTL_FC_INIT_0_j)
  %other = call i1 @air.is_function_constant_defined(ptr addrspace(2) @other.MTL_FC_INIT_1_j)
  ret void
}
"#;
        for value in [0u32, 7] {
            let specialized =
                specialize_air_function_constants(ll, &[(0, value.to_le_bytes().to_vec())])
                    .expect("specialize supplied index only");
            assert!(specialized.contains(&format!("constant i32 {value}, section")));
            assert!(specialized.contains("@working_copy = internal addrspace(2) global i32 19"));
            assert!(specialized.contains("@other.MTL_FC_INIT_1_j = internal addrspace(2) externally_initialized constant i32 undef"));
            assert!(specialized.contains("%selected = call i1 @air.is_function_constant_defined"));
            assert!(specialized.contains("%other = call i1 @air.is_function_constant_defined"));
        }
    }

    #[test]
    fn floating_payloads_preserve_exact_bits() {
        for (bits, literal) in [
            (0u32, "float f0x00000000"),
            (0x8000_0000, "float f0x80000000"),
            (0x3fc0_0000, "float f0x3FC00000"),
            (0x7f80_0000, "float f0x7F800000"),
            (0x7f80_0001, "float f0x7F800001"),
            (0xffc0_0123, "float f0xFFC00123"),
        ] {
            assert_eq!(
                llvm_typed_literal("float", "f", &bits.to_le_bytes()).unwrap(),
                literal
            );
        }
        assert_eq!(
            llvm_typed_literal("half", "Dh", &0x8000u16.to_le_bytes()).unwrap(),
            "half 0xH8000"
        );
        assert_eq!(
            llvm_typed_literal("double", "d", &0x8000_0000_0000_0000u64.to_le_bytes()).unwrap(),
            "double 0x8000000000000000"
        );
    }

    #[test]
    fn specializes_scalar_and_vector_initializer_globals() {
        let ll = r#"@enabled.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@lanes.MTL_FC_INIT_3_Dv2_j = internal addrspace(2) externally_initialized constant <2 x i32> undef, section "air.fc_initializer", align 8
define void @init() {
  %defined = call i1 @air.is_function_constant_defined(ptr addrspace(2) @enabled.MTL_FC_INIT_0_b)
  ret void
}
"#;
        let specialized = specialize_air_function_constants(
            ll,
            &[
                (0, vec![1]),
                (3, [7u32.to_le_bytes(), 11u32.to_le_bytes()].concat()),
            ],
        )
        .expect("specialize");
        assert!(specialized.contains("constant i8 1, section"));
        assert!(specialized.contains("constant <2 x i32> <i32 7, i32 11>, section"));
        assert!(specialized.contains("%defined = call i1 @air.is_function_constant_defined"));

        let supplied_zero = specialize_air_function_constants(ll, &[(0, vec![0])])
            .expect("specialize a defined false value");
        assert!(supplied_zero.contains("constant i8 0, section"));
        assert!(supplied_zero.contains("%defined = call i1 @air.is_function_constant_defined"));
    }

    #[test]
    fn specialized_predicates_select_metadata_roles_in_both_directions() {
        let ll = r#"@state.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@enabled = internal addrspace(2) global i8 0, align 1
@disabled = internal addrspace(2) global i8 0, align 1
define internal void @_GLOBAL__sub_I_metadata() section "air.static_init" {
  %state = load i8, ptr addrspace(2) @state.MTL_FC_INIT_0_b
  %inverse = xor i8 %state, 1
  store i8 %state, ptr addrspace(2) @enabled
  store i8 %inverse, ptr addrspace(2) @disabled
  ret void
}
!0 = !{i32 0, !"air.function_constant", !2, !"air.buffer", !"air.location_index", i32 3}
!1 = !{i32 1, !"air.function_constant", !3, !"air.texture", !"air.location_index", i32 4}
!2 = !{ptr addrspace(2) @enabled, !"bool", !"enabled"}
!3 = !{ptr addrspace(2) @disabled, !"bool", !"disabled"}
"#;

        let specialized = specialize_air_function_constants(ll, &[(0, vec![1])])
            .expect("specialize metadata predicates");
        assert!(
            specialized.contains("!0 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 3}")
        );
        assert!(specialized
            .contains("!1 = !{i32 1, !\"air.function_constant_disabled\", !3, !\"air.texture\""));
        assert!(!specialized.contains("!0 = !{i32 0, !\"air.function_constant\""));
        let vertex = format!(
            "{specialized}!air.vertex = !{{!4}}\n!4 = !{{ptr @main, !5, !6}}\n!5 = !{{!1}}\n!6 = !{{}}\n"
        );
        assert_eq!(
            crate::air::meta::parse_air_vertex_meta(&vertex)
                .0
                .and_then(|meta| meta.output_role_of(0).cloned()),
            Some(crate::air::meta::VertOutRole::FunctionConstantDisabled)
        );
    }

    #[test]
    fn discovery_and_rewrite_share_the_initializer_symbol_boundary() {
        let ll = r#"@MTL_FC_INIT_9_noise.MTL_FC_INIT_0_j = internal addrspace(2) constant i32 undef, section "air.fc_initializer"
@alias.MTL_FC_INIT_0_j = internal addrspace(2) constant i32 undef, section "air.fc_initializer"
@MTL_FC_INIT_0_noise = internal addrspace(2) global i32 19
"#;
        let constants = crate::air::meta::parse_function_constants(ll);
        assert_eq!(constants.len(), 1);
        assert_eq!(constants[0].index, 0);
        let output =
            specialize_air_function_constants(ll, &[(0, 7u32.to_le_bytes().to_vec())]).unwrap();
        assert_eq!(output.matches("constant i32 7, section").count(), 2);
        assert!(output.contains("@MTL_FC_INIT_0_noise = internal addrspace(2) global i32 19"));
    }

    #[test]
    fn section_attributes_are_shared_with_definedness() {
        let prefix = "@value.MTL_FC_INIT_0_j = internal constant i32 undef";
        for suffix in [
            ", section \"air.fc_initializer\"",
            ", section\t\"air.fc_initializer\", align 4",
            ", section  \"air.fc_initializer\" ; trailing comment",
        ] {
            let source = format!("{prefix}{suffix}");
            assert_eq!(
                crate::air::meta::function_constant_definedness(&source),
                Some(false)
            );
            let specialized =
                specialize_air_function_constants(&source, &[(0, 7u32.to_le_bytes().to_vec())])
                    .unwrap();
            assert_eq!(
                crate::air::meta::function_constant_definedness(&specialized),
                Some(true)
            );
        }
        for suffix in [
            " ; section \"air.fc_initializer\"",
            ", section \"other.air.fc_initializer\"",
            ", section \"other\", !note !{!\"air.fc_initializer\"}",
            ", section_suffix \"air.fc_initializer\"",
            ", partition \"air.fc_initializer\"",
        ] {
            let source = format!("{prefix}{suffix}");
            assert_eq!(
                crate::air::meta::function_constant_definedness(&source),
                None
            );
            assert!(matches!(
                specialize_air_function_constants(&source, &[(0, 7u32.to_le_bytes().to_vec())]),
                Err(FunctionConstantError::MissingInitializerSection { index: 0 })
            ));
        }
    }

    #[test]
    fn initializer_rewrite_uses_storage_and_complete_value_tokens() {
        for storage in ["constant", "global"] {
            let declaration = format!(
                "@value.MTL_FC_INIT_0_j = internal\t{storage}\ti32\tundef, section \"air.fc_initializer\"\n"
            );
            let output = specialize_air_function_constants(
                &declaration,
                &[(0, 17u32.to_le_bytes().to_vec())],
            )
            .unwrap();
            assert_eq!(output, declaration.replace("i32\tundef", "i32 17"));
            for invalid in ["undef_suffix", "undefined", "undef0"] {
                assert!(matches!(
                    specialize_air_function_constants(
                        &declaration.replace("undef", invalid),
                        &[(0, 17u32.to_le_bytes().to_vec())],
                    ),
                    Err(FunctionConstantError::InvalidInitializer { index: 0, .. })
                ));
            }
        }
    }

    #[test]
    fn rejects_unknown_and_malformed_values() {
        let ll = r#"@enabled.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
"#;
        for (values, expected) in [
            (
                vec![(9, vec![1])],
                FunctionConstantError::UnknownIndices(vec![9]),
            ),
            (
                vec![(0, vec![2])],
                FunctionConstantError::InvalidValue {
                    index: 0,
                    error: ConstantValueError::InvalidBoolean { value: 2 },
                },
            ),
            (
                vec![(0, vec![1, 0])],
                FunctionConstantError::InvalidValue {
                    index: 0,
                    error: ConstantValueError::PayloadSize {
                        type_name: "i8".into(),
                        expected: 1,
                        actual: 2,
                    },
                },
            ),
            (
                vec![(0, vec![0]), (0, vec![1])],
                FunctionConstantError::DuplicateIndex(0),
            ),
        ] {
            assert_eq!(
                specialize_air_function_constants(ll, &values).unwrap_err(),
                expected
            );
        }
        assert_eq!(
            specialize_air_function_constants(
                &ll.replace("air.fc_initializer", "other"),
                &[(0, vec![1])]
            )
            .unwrap_err(),
            FunctionConstantError::MissingInitializerSection { index: 0 }
        );
        assert_eq!(
            specialize_air_function_constants(&ll.replace("i8 undef", "i8 0"), &[(0, vec![1])])
                .unwrap_err(),
            FunctionConstantError::InvalidInitializer {
                index: 0,
                type_name: "i8".into()
            }
        );
    }
}
