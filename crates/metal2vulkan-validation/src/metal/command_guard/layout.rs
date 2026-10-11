//! The data layout the command guard holds a module to: an AIR `target datalayout`'s pointer and
//! scalar ABI, and the old engine's "Raw" allocation rule over the native `LlType`s. Copied from
//! the product's layout oracle with the guard, as far as the guard reaches, so that it accepts and
//! refuses exactly what it did.

use super::llvm::LlType;
use std::collections::{BTreeMap, HashMap};

/// LLVM pointer layouts and scalar/vector/aggregate ABI alignment from one AIR `target datalayout`.
///
/// Scalar/vector entries start from LLVM defaults and store bit widths and alignments.
/// Integer widths use the next larger entry, or the largest entry if none is larger.
/// Vectors without an exact entry use ordinary power-of-two alignment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct AirDataLayout {
    abi_align_bits: BTreeMap<(u8, u32), u32>,
    aggregate_align_bytes: u64,
    pointer_layouts: HashMap<u32, (u64, u64, u32)>,
}

impl Default for AirDataLayout {
    fn default() -> Self {
        Self {
            abi_align_bits: [
                ((b'i', 8), 8),
                ((b'i', 16), 16),
                ((b'i', 32), 32),
                ((b'i', 64), 32),
                ((b'f', 16), 16),
                ((b'f', 32), 32),
                ((b'f', 64), 64),
                ((b'f', 128), 128),
                ((b'v', 64), 64),
                ((b'v', 128), 128),
            ]
            .into(),
            aggregate_align_bytes: 1,
            pointer_layouts: HashMap::new(),
        }
    }
}

impl AirDataLayout {
    pub(super) fn parse(layout: &str) -> Result<Self, String> {
        let mut result = Self::default();
        for component in layout.split('-') {
            if let Some(rest) = component.strip_prefix('p') {
                let (space, fields) = rest
                    .split_once(':')
                    .ok_or_else(|| format!("invalid AIR datalayout pointer entry `{component}`"))?;
                let space = if space.is_empty() {
                    0
                } else {
                    space
                        .parse::<u32>()
                        .map_err(|_| format!("invalid AIR pointer address space `{space}`"))?
                };
                let fields = fields
                    .split(':')
                    .map(str::parse::<u32>)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| format!("invalid AIR datalayout pointer entry `{component}`"))?;
                let valid =
                    (2..=4).contains(&fields.len()) && fields.iter().all(|&value| value != 0);
                if !valid {
                    return Err(format!(
                        "invalid AIR datalayout pointer entry `{component}`"
                    ));
                }
                let size = fields[0];
                let alignment = fields[1];
                let preferred = fields.get(2).copied().unwrap_or(alignment);
                let index = fields.get(3).copied().unwrap_or(size);
                if !size.is_multiple_of(8)
                    || alignment < 8
                    || !alignment.is_power_of_two()
                    || preferred < alignment
                    || !preferred.is_power_of_two()
                    || index > size
                {
                    return Err(format!(
                        "unsupported AIR datalayout pointer entry `{component}`"
                    ));
                }
                result.pointer_layouts.insert(
                    space,
                    (u64::from(size / 8), u64::from(alignment / 8), index),
                );
                continue;
            }
            let Some(kind @ (b'i' | b'f' | b'v' | b'a')) = component.bytes().next() else {
                continue;
            };
            let (width, fields) = component[1..]
                .split_once(':')
                .ok_or_else(|| format!("invalid AIR datalayout alignment entry `{component}`"))?;
            let size = if kind == b'a' && width.is_empty() {
                0
            } else {
                width
                    .parse::<u32>()
                    .map_err(|_| format!("invalid AIR datalayout width `{component}`"))?
            };
            let fields = fields
                .split(':')
                .take(3)
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid AIR datalayout alignment entry `{component}`"))?;
            if !(1..=2).contains(&fields.len()) {
                return Err(format!(
                    "invalid AIR datalayout alignment entry `{component}`"
                ));
            }
            let abi = fields[0];
            let preferred = fields.get(1).copied().unwrap_or(abi);
            let aligned = |value: u32| value >= 8 && value.is_power_of_two();
            if (kind != b'a' && (size == 0 || size >= 1 << 24))
                || (kind == b'a' && size != 0)
                || !(aligned(abi) || (kind == b'a' && abi == 0))
                || !(aligned(preferred) || (kind == b'a' && preferred == 0))
                || preferred < abi
                || (kind == b'i' && size == 8 && abi != 8)
            {
                return Err(format!("unsupported AIR datalayout alignment entry `{component}`: ABI alignments must be powers of two in whole bytes"));
            }
            if kind == b'a' {
                result.aggregate_align_bytes = u64::from(abi / 8).max(1);
            } else {
                result.abi_align_bits.insert((kind, size), abi);
            }
        }
        Ok(result)
    }

    pub(super) fn from_ir(ir: &str) -> Result<Option<Self>, String> {
        let Some(line) = ir
            .lines()
            .map(str::trim_start)
            .find(|line| line.starts_with("target datalayout"))
        else {
            return Ok(None);
        };
        let start = line
            .find('"')
            .ok_or_else(|| "invalid target datalayout: missing opening quote".to_string())?;
        let rest = &line[start + 1..];
        let end = rest
            .find('"')
            .ok_or_else(|| "invalid target datalayout: missing closing quote".to_string())?;
        Self::parse(&rest[..end]).map(Some)
    }

    /// Pointer store size and ABI alignment in bytes; unspecified spaces inherit space zero.
    pub(super) fn pointer_storage_size_align(&self, space: u32) -> (u64, u64) {
        self.pointer_layouts
            .get(&space)
            .or_else(|| self.pointer_layouts.get(&0))
            .map(|&(size, align, _)| (size, align))
            .unwrap_or((8, 8))
    }

    fn scalar_alignment(&self, kind: u8, bits: u32) -> u64 {
        let alignment = if kind == b'i' {
            self.abi_align_bits
                .range((kind, bits)..=(kind, u32::MAX))
                .next()
                .or_else(|| {
                    self.abi_align_bits
                        .range((kind, 0)..=(kind, u32::MAX))
                        .next_back()
                })
                .map(|(_, alignment)| *alignment)
        } else {
            self.abi_align_bits.get(&(kind, bits)).copied()
        };
        u64::from(alignment.expect("modeled scalar types have LLVM default alignments") / 8)
    }

    pub(super) fn struct_layout(&self) -> StructLayout {
        StructLayout {
            end: 0,
            alignment: self.aggregate_align_bytes,
        }
    }
}

/// Incremental aggregate layout shared by the allocation rules. Checked arithmetic keeps
/// malformed aggregate sizes from wrapping into apparently valid offsets.
#[derive(Default)]
pub(super) struct StructLayout {
    end: u64,
    alignment: u64,
}

impl StructLayout {
    pub(super) fn push(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let offset = checked_round_up(self.end, alignment)?;
        let end = offset.checked_add(size)?;
        self.end = end;
        self.alignment = self.alignment.max(alignment);
        Some(offset)
    }

    pub(super) fn finish(self) -> Option<(u64, u64)> {
        let alignment = self.alignment.max(1);
        Some((checked_round_up(self.end, alignment)?, alignment))
    }
}

fn checked_round_up(value: u64, alignment: u64) -> Option<u64> {
    let remainder = value.checked_rem(alignment)?;
    value.checked_add(if remainder == 0 {
        0
    } else {
        alignment - remainder
    })
}

fn pointer_allocation(space: u32, data_layout: Option<&AirDataLayout>) -> (u64, u64) {
    let (size, alignment) =
        data_layout.map_or((8, 8), |layout| layout.pointer_storage_size_align(space));
    (round_up_u64(size, alignment), alignment)
}

fn scalar_allocation(kind: u8, bits: u32, data_layout: Option<&AirDataLayout>) -> (u64, u64) {
    let size = u64::from(bits).div_ceil(8).max(1);
    let alignment = data_layout.map_or_else(
        || size.next_power_of_two().min(8),
        |layout| layout.scalar_alignment(kind, bits),
    );
    (round_up_u64(size, alignment), alignment)
}

fn vector_allocation(
    element: &LlType,
    lanes: u32,
    data_layout: Option<&AirDataLayout>,
) -> Option<(u64, u64)> {
    let bits = match element {
        LlType::Int(bits) => u64::from(*bits),
        LlType::Half | LlType::BFloat => 16,
        LlType::Float => 32,
        LlType::Ptr(space) => {
            data_layout.map_or(8, |layout| layout.pointer_storage_size_align(*space).0) * 8
        }
        _ => return None,
    };
    let store_bits = u32::try_from(bits.checked_mul(u64::from(lanes))?).ok()?;
    let store_size = u64::from(store_bits).div_ceil(8);
    let alignment = data_layout
        .and_then(|layout| layout.abi_align_bits.get(&(b'v', store_bits)))
        .map(|&alignment| u64::from(alignment / 8))
        .or_else(|| store_size.checked_next_power_of_two())?;
    Some((checked_round_up(store_size, alignment)?, alignment))
}

/// The "Raw" rule (emitter): LLVM allocation sizes using source pointer layouts and vector ABI alignment (or LLVM's
/// power-of-two default). An array's alignment is its element's, which is what decides where the
/// explicit `[N x i8]` pads Metal writes into a packed struct put every member behind them. It
/// differs from the optional calculators above in two other ways:
///   * `resolve` is FALLIBLE — a `Named` type the module can't expand is an error, not a fallthrough.
///   * uncovered types (odd-width ints, `Void`) are an `Err`, not `None` — the Raw rule only accepts
///     the standard scalar widths it knows how to lay out.
///     Returns `Result` accordingly. Was `Emitter::raw_type_size_align` (`resolve` = `Emitter::resolve_type`).
pub(super) fn raw_size_align(
    ty: &LlType,
    resolve: &impl Fn(&LlType) -> Result<LlType, String>,
    data_layout: Option<&AirDataLayout>,
) -> Result<(u64, u64), String> {
    match resolve(ty)? {
        LlType::Int(bits @ (1 | 8 | 16 | 32 | 64)) => {
            Ok(scalar_allocation(b'i', bits, data_layout))
        }
        LlType::Half | LlType::BFloat => Ok(scalar_allocation(b'f', 16, data_layout)),
        LlType::Float => Ok(scalar_allocation(b'f', 32, data_layout)),
        LlType::Ptr(space) => Ok(pointer_allocation(space, data_layout)),
        LlType::Vector(elem, lanes) => {
            let element = resolve(&elem)?;
            raw_size_align(&element, resolve, data_layout)?;
            vector_allocation(&element, lanes, data_layout)
                .ok_or_else(|| "native emitter: raw vector layout overflow".into())
        }
        LlType::Array(elem, len) => {
            let (elem_size, elem_align) = raw_size_align(&elem, resolve, data_layout)?;
            let stride = round_up_u64(elem_size, elem_align);
            Ok((stride * len as u64, elem_align))
        }
        LlType::Struct(fields) => {
            let mut layout =
                data_layout.map_or_else(StructLayout::default, AirDataLayout::struct_layout);
            for field in fields {
                let (size, align) = raw_size_align(&field, resolve, data_layout)?;
                layout
                    .push(size, align)
                    .ok_or("native emitter: raw struct layout overflow")?;
            }
            layout
                .finish()
                .ok_or_else(|| "native emitter: raw struct layout overflow".into())
        }
        other => Err(format!(
            "native emitter: raw size/align for {other:?} is not covered yet"
        )),
    }
}

fn round_up_u64(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        value.div_ceil(align) * align
    }
}
