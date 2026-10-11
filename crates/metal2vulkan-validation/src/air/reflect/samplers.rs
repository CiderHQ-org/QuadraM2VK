//! AIR constexpr samplers: the state their two `i64` words encode, and their place in an
//! interface (the old engine's decode and placement).

use super::descriptors::{sampler_band, RESOURCE_DESCRIPTOR_SET, SAMPLER_BAND_START};
use super::{DescriptorLocation, ResourceBinding, ResourceKind, ShaderInterface};

/// Minification or magnification filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerFilter {
    Nearest,
    Linear,
    Bicubic,
}

/// Mipmap filter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerMipFilter {
    None,
    Nearest,
    Linear,
}

/// Coordinate addressing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerAddressMode {
    ClampToZero,
    ClampToEdge,
    Repeat,
    MirroredRepeat,
    ClampToBorder,
}

/// Coordinate convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerCoordinates {
    Normalized,
    Pixel,
}

/// Comparison function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerCompareFunction {
    None,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Always,
    Never,
}

/// Border color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerBorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

/// Reduction mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerReduction {
    WeightedAverage,
    Minimum,
    Maximum,
}

/// The state of one AIR constexpr sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerState {
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub address_mode_s: SamplerAddressMode,
    pub address_mode_t: SamplerAddressMode,
    pub address_mode_r: SamplerAddressMode,
    pub coordinates: SamplerCoordinates,
    pub compare_function: SamplerCompareFunction,
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub border_color: SamplerBorderColor,
    pub reduction: SamplerReduction,
    pub lod_bias: f32,
}

impl SamplerState {
    /// Decode the two `i64` words AIR stores for a `constexpr sampler`.
    ///
    /// | `words[0]` | Field |
    /// |---|---|
    /// | 0-2, 3-5, 6-8 | `address_mode_s` / `_t` / `_r` |
    /// | 9-10, 11-12, 13-14 | `mag_filter` / `min_filter` / `mip_filter` |
    /// | 15 | `coordinates` |
    /// | 16-19 | `compare_function` |
    /// | 20-23 | `max_anisotropy - 1` |
    /// | 24-39 | `lod_min_clamp` half |
    /// | 40-55 | `lod_max_clamp` half |
    /// | 56-57, 58-59 | `border_color` / `reduction` |
    /// | 60-63 | not read |
    ///
    /// `words[1]` bits 0-15 are the `lod_bias` half; the rest are not read. An unrecognized code in
    /// a field that is read is an error rather than an invented state.
    pub fn from_air_words(words: [u64; 2]) -> Result<Self, String> {
        let word = words[0];
        let border_color = match (word >> 56) & 0x3 {
            0 => SamplerBorderColor::TransparentBlack,
            1 => SamplerBorderColor::OpaqueBlack,
            2 => SamplerBorderColor::OpaqueWhite,
            value => return Err(format!("unsupported AIR sampler border-color code {value}")),
        };
        let address = |shift: u32| match (word >> shift) & 0x7_u64 {
            0 if border_color != SamplerBorderColor::TransparentBlack => {
                Ok(SamplerAddressMode::ClampToBorder)
            }
            0 => Ok(SamplerAddressMode::ClampToZero),
            1 => Ok(SamplerAddressMode::ClampToEdge),
            2 => Ok(SamplerAddressMode::Repeat),
            3 => Ok(SamplerAddressMode::MirroredRepeat),
            value => Err(format!("unsupported AIR sampler address code {value}")),
        };
        let filter = |shift: u32| match (word >> shift) & 0x3_u64 {
            0 => Ok(SamplerFilter::Nearest),
            1 => Ok(SamplerFilter::Linear),
            2 => Ok(SamplerFilter::Bicubic),
            value => Err(format!("unsupported AIR sampler filter code {value}")),
        };
        let mip_filter = match (word >> 13) & 0x3 {
            0 => SamplerMipFilter::None,
            1 => SamplerMipFilter::Nearest,
            2 => SamplerMipFilter::Linear,
            value => return Err(format!("unsupported AIR sampler mip-filter code {value}")),
        };
        let coordinates = match (word >> 15) & 0x1 {
            0 => SamplerCoordinates::Normalized,
            _ => SamplerCoordinates::Pixel,
        };
        let compare_function = match (word >> 16) & 0xf {
            0 => SamplerCompareFunction::None,
            1 => SamplerCompareFunction::Less,
            2 => SamplerCompareFunction::LessEqual,
            3 => SamplerCompareFunction::Greater,
            4 => SamplerCompareFunction::GreaterEqual,
            5 => SamplerCompareFunction::Equal,
            6 => SamplerCompareFunction::NotEqual,
            7 => SamplerCompareFunction::Always,
            8 => SamplerCompareFunction::Never,
            value => {
                return Err(format!(
                    "unsupported AIR sampler compare-function code {value}"
                ))
            }
        };
        let reduction = match (word >> 58) & 0x3 {
            0 => SamplerReduction::WeightedAverage,
            1 => SamplerReduction::Minimum,
            2 => SamplerReduction::Maximum,
            value => return Err(format!("unsupported AIR sampler reduction code {value}")),
        };
        let min_half = ((word >> 24) & 0xffff) as u16;
        let max_half = ((word >> 40) & 0xffff) as u16;
        let bias_half = (words[1] & 0xffff) as u16;
        Ok(Self {
            min_filter: filter(11)?,
            mag_filter: filter(9)?,
            mip_filter,
            address_mode_s: address(0)?,
            address_mode_t: address(3)?,
            address_mode_r: address(6)?,
            coordinates,
            compare_function,
            max_anisotropy: (((word >> 20) & 0xf) as u32) + 1,
            lod_min_clamp: f16_bits_to_f32(min_half),
            lod_max_clamp: f16_bits_to_f32(max_half),
            border_color,
            reduction,
            lod_bias: f16_bits_to_f32(bias_half),
        })
    }
}

/// An IEEE half's bits as the `f32` of the same value (an infinity or NaN keeps its kind and
/// payload).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let negative = bits & 0x8000 != 0;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let significand = (bits & 0x03ff) as u32;
    let magnitude = if exponent == 0 {
        (significand as f32) * (2.0f32).powi(-24)
    } else if exponent == 0x1f {
        if significand == 0 {
            f32::INFINITY
        } else {
            f32::from_bits(0x7f80_0000 | (significand << 13))
        }
    } else {
        f32::from_bits(((exponent + 112) << 23) | (significand << 13))
    };
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

/// Place the module's constexpr samplers at the sampler band's free bindings, in order, as the old
/// engine did: a module with more than the band holds is refused.
pub(super) fn add_static_samplers(interface: &mut ShaderInterface, ll: &str) -> Result<(), String> {
    let constants = parse_static_sampler_constants(ll)?;
    if constants.is_empty() {
        return Ok(());
    }
    let mut occupied = interface
        .bindings
        .iter()
        .filter_map(|binding| binding.descriptor.map(|descriptor| descriptor.binding))
        .collect::<std::collections::BTreeSet<_>>();
    let band = sampler_band();
    for words in constants {
        let binding = band
            .clone()
            .find(|binding| !occupied.contains(binding))
            .ok_or_else(|| {
                format!(
                    "AIR constexpr sampler count exceeds descriptor band [{},{})",
                    band.start, band.end
                )
            })?;
        occupied.insert(binding);
        interface.bindings.push(ResourceBinding {
            kind: ResourceKind::StaticSampler,
            metal_index: binding - SAMPLER_BAND_START,
            descriptor: Some(DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding,
                count: 1,
            }),
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access: None,
            static_sampler: Some(SamplerState::from_air_words(words)?),
        });
    }
    Ok(())
}

fn parse_static_sampler_constants(ll: &str) -> Result<Vec<[u64; 2]>, String> {
    let mut globals = std::collections::HashMap::<String, [u64; 2]>::new();
    let mut nodes = std::collections::HashMap::<u32, String>::new();
    let mut root = None;

    for raw in ll.lines() {
        let line = raw.trim();
        if line.starts_with('@') && line.contains(" constant ") {
            let Some((name, _)) = line.split_once(" = ") else {
                continue;
            };
            let mut values = line.split("i64 ").skip(1).filter_map(|tail| {
                tail.split(|ch: char| ch == ',' || ch == ']' || ch.is_whitespace())
                    .find(|token| !token.is_empty())
                    .and_then(|token| token.parse::<i64>().ok())
                    .map(|value| value as u64)
            });
            if let Some(first) = values.next() {
                globals.insert(name.to_string(), [first, values.next().unwrap_or(0)]);
            }
            continue;
        }
        if let Some(body) = line.strip_prefix("!air.sampler_states = !{") {
            root = Some(metadata_refs(body));
            continue;
        }
        let Some(rest) = line.strip_prefix('!') else {
            continue;
        };
        let Some((id, body)) = rest.split_once(" = !{") else {
            continue;
        };
        if let Ok(id) = id.parse::<u32>() {
            nodes.insert(id, body.trim_end_matches('}').to_string());
        }
    }

    let Some(root) = root else {
        return Ok(Vec::new());
    };
    let mut constants = Vec::with_capacity(root.len());
    for node_id in root {
        let body = nodes
            .get(&node_id)
            .ok_or_else(|| format!("AIR sampler-state metadata node !{node_id} is missing"))?;
        if !body.contains("!\"air.sampler_state\"") {
            return Err(format!(
                "AIR sampler-state root references non-sampler node !{node_id}"
            ));
        }
        let name = global_name(body)
            .ok_or_else(|| format!("AIR sampler-state node !{node_id} has no global"))?;
        let words = globals
            .get(name)
            .copied()
            .ok_or_else(|| format!("AIR sampler-state global {name} has no i64 initializer"))?;
        constants.push((name.to_string(), words));
    }
    constants.sort_by_key(|(name, _)| static_sampler_name_order(name));
    Ok(constants.into_iter().map(|(_, words)| words).collect())
}

/// A total order over a module's `__air_sampler_state` globals: the whole dot-separated suffix
/// (LLVM uniques a repeated name by appending `.N`, and an already uniqued one by appending
/// again), then the name. The old engine's interface pass and its reflection paired the globals
/// off by this order.
fn static_sampler_name_order(name: &str) -> (Vec<u64>, String) {
    let name = name.trim_start_matches('@');
    let components = name
        .strip_prefix("__air_sampler_state")
        .unwrap_or("")
        .split('.')
        .filter(|component| !component.is_empty())
        .map(|component| component.parse::<u64>().unwrap_or(u64::MAX))
        .collect();
    (components, name.to_string())
}

fn metadata_refs(body: &str) -> Vec<u32> {
    body.split('!')
        .skip(1)
        .filter_map(|tail| {
            let digits = tail
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect::<String>();
            (!digits.is_empty())
                .then(|| digits.parse::<u32>().ok())
                .flatten()
        })
        .collect()
}

fn global_name(body: &str) -> Option<&str> {
    let start = body.find('@')?;
    let rest = &body[start..];
    let end = rest
        .find(|character: char| {
            character.is_whitespace() || matches!(character, ',' | ')' | ']' | '}')
        })
        .unwrap_or(rest.len());
    Some(&rest[..end])
}
