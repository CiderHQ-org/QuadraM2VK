mod comparison;

pub use comparison::{
    allowance, census_names, lane_within, reduction_table, reorder_ulps, value_difference,
    Allowance, ComparisonClass, FloatLane, LaneTolerance, ObservationLanes, SumFormat, UlpTable,
    ValueDifference, REDUCTIONS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationScalar {
    Float,
    Half,
    Int,
    Uint,
    Short,
    Ushort,
    Bool,
}

/// A varying or render-target type name the census classifies as observable
/// (`executor_contract::unsupported_observation_requirements`): one to four lanes of these
/// scalars, parsed from the AIR's declared type name. The generated graphics companions
/// (`crate::companion`) decide from the IR type instead (`companion::VaryingType`), so the two
/// agree only where the declared name states the IR type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationType {
    pub scalar: ObservationScalar,
    pub lanes: usize,
}

impl ObservationType {
    pub fn parse(type_name: &str) -> Option<Self> {
        let split = type_name
            .bytes()
            .position(|byte| byte.is_ascii_digit())
            .unwrap_or(type_name.len());
        let (base, lanes) = type_name.split_at(split);
        let lanes = if lanes.is_empty() {
            1
        } else {
            lanes.parse().ok()?
        };
        if !(1..=4).contains(&lanes) {
            return None;
        }
        let scalar = match base {
            "float" => ObservationScalar::Float,
            "half" => ObservationScalar::Half,
            "int" => ObservationScalar::Int,
            "uint" => ObservationScalar::Uint,
            "short" => ObservationScalar::Short,
            "ushort" => ObservationScalar::Ushort,
            "bool" => ObservationScalar::Bool,
            _ => return None,
        };
        Some(Self { scalar, lanes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_types_are_one_to_four_lanes_of_the_observed_scalars() {
        assert_eq!(
            ObservationType::parse("half3"),
            Some(ObservationType {
                scalar: ObservationScalar::Half,
                lanes: 3
            })
        );
        assert_eq!(
            ObservationType::parse("bool"),
            Some(ObservationType {
                scalar: ObservationScalar::Bool,
                lanes: 1
            })
        );
        assert!(ObservationType::parse("double2").is_none());
        assert!(ObservationType::parse("float8").is_none());
        assert!(ObservationType::parse("uchar2").is_none());
    }
}
