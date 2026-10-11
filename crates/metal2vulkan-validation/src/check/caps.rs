//! Case-schema caps the checker enforces before any executor sees a case.
//!
//! Together with mechanical loop-freedom (or, later, fuel), the caps give every GPU submission a
//! computable worst-case operation count. They are checker rules, not store rules: an installed
//! case over a cap still reads, and `corpus-case-check --recheck-all` names it.

use crate::air::reflect::FragmentImageblock;
use crate::case::{AuthoredCase, TessellationDraw};

/// Total invocations of one compute dispatch (`dispatch.grid` is in threads).
pub const MAX_DISPATCH_INVOCATIONS: u64 = 1 << 20;
/// Threads in one threadgroup (and in one tile dispatch).
pub const MAX_THREADGROUP_INVOCATIONS: u64 = 1024;
/// Vertex invocations times instances of one draw. For a tessellated draw the vertex count is the
/// evaluated-vertex upper bound of every patch.
pub const MAX_DRAW_VERTEX_INSTANCES: u64 = 1 << 16;
/// Width and height of every attachment (render targets, depth/stencil, fragment imageblock).
pub const MAX_ATTACHMENT_EXTENT: u32 = 1024;
/// Literal and allocated bytes of every resource the case declares.
pub const MAX_RESOURCE_BYTES: u64 = 256 << 20;
/// Authored indirect command buffers.
pub const MAX_COMMAND_BUFFERS: usize = 8;
/// Per-invocation loop-header visits a case may declare.
pub const MAX_CASE_FUEL: u32 = 1 << 24;

/// Every cap the case exceeds, as checker errors.
pub fn cap_errors(case: &AuthoredCase) -> Vec<String> {
    let mut errors = Vec::new();
    if let Some(dispatch) = &case.dispatch {
        let invocations = product(&dispatch.grid);
        if invocations > MAX_DISPATCH_INVOCATIONS {
            errors.push(format!(
                "dispatch.grid {:?} is {invocations} invocations; the cap is {MAX_DISPATCH_INVOCATIONS}",
                dispatch.grid
            ));
        }
        let group = product(&dispatch.threads_per_threadgroup);
        if group > MAX_THREADGROUP_INVOCATIONS {
            errors.push(format!(
                "dispatch.threads_per_threadgroup {:?} is {group} threads; the cap is {MAX_THREADGROUP_INVOCATIONS}",
                dispatch.threads_per_threadgroup
            ));
        }
    }
    if let Some(draw) = &case.draw {
        let vertices = u64::from(draw.vertex_count) * u64::from(draw.instance_count);
        if vertices > MAX_DRAW_VERTEX_INSTANCES {
            errors.push(format!(
                "draw vertex_count x instance_count is {vertices}; the cap is {MAX_DRAW_VERTEX_INSTANCES}"
            ));
        }
    }
    if let Some(tessellation) = &case.tessellation {
        match tessellated_vertex_instances(tessellation) {
            Some(vertices) if vertices <= MAX_DRAW_VERTEX_INSTANCES => {}
            Some(vertices) => errors.push(format!(
                "tessellated draw evaluates up to {vertices} vertex instances; the cap is {MAX_DRAW_VERTEX_INSTANCES}"
            )),
            None => errors.push("tessellated draw's evaluated vertex count overflows".into()),
        }
    }
    let attachments = case
        .render_targets
        .iter()
        .map(|target| (format!("render target {}", target.index), target.dimensions))
        .chain(
            case.depth_stencil
                .iter()
                .map(|depth| ("depth/stencil attachment".to_string(), depth.dimensions)),
        )
        .chain(
            case.fragment_imageblock
                .iter()
                .map(|block| ("fragment imageblock".to_string(), block.dimensions)),
        );
    for (label, [width, height]) in attachments {
        if width > MAX_ATTACHMENT_EXTENT || height > MAX_ATTACHMENT_EXTENT {
            errors.push(format!(
                "{label} extent {width}x{height} exceeds {MAX_ATTACHMENT_EXTENT}x{MAX_ATTACHMENT_EXTENT}"
            ));
        }
    }
    match resource_bytes(case) {
        Ok(bytes) if bytes <= MAX_RESOURCE_BYTES => {}
        Ok(bytes) => errors.push(format!(
            "resources total {bytes} bytes; the cap is {MAX_RESOURCE_BYTES}"
        )),
        Err(error) => errors.push(error),
    }
    if case.command_buffers.len() > MAX_COMMAND_BUFFERS {
        errors.push(format!(
            "{} indirect command buffers; the cap is {MAX_COMMAND_BUFFERS}",
            case.command_buffers.len()
        ));
    }
    if let Some(fuel) = case.fuel {
        if fuel == 0 || fuel > MAX_CASE_FUEL {
            errors.push(format!(
                "fuel {fuel} must be in 1..={MAX_CASE_FUEL} loop-header visits"
            ));
        }
    }
    errors
}

fn product(dimensions: &[u32; 3]) -> u64 {
    dimensions
        .iter()
        .map(|dimension| u64::from(*dimension))
        .product()
}

/// Upper bound on evaluated vertices times instances: each patch evaluates at most
/// `(S + 1)^2` domain points, where `S` is the most segments any of its factors gives: the factor
/// clamped to the partition's range (`MTLTessellationPartitionMode`: `[1, max]` under pow2 and
/// integer, `[1, max - 1]` under fractional odd, `[2, max]` under fractional even; a NaN at the
/// lower end and `+inf` at the upper, `max` the maximum factor both executors draw with), rounded
/// up to an integer, then to a power of two under pow2, to an odd integer under fractional odd,
/// and to an even one under fractional even (a fractional partition splits a factor into as many
/// segments as the next such integer, two of them shorter). A patch an edge factor of which is
/// zero, negative, or NaN is culled and evaluates nothing.
fn tessellated_vertex_instances(tessellation: &TessellationDraw) -> Option<u64> {
    use crate::case::TessellationPartition as P;
    let max = f64::from(crate::executor_contract::tessellation::MAX_TESSELLATION_FACTOR);
    let (lower, upper) = match tessellation.partition_mode {
        P::Pow2 | P::Integer => (1.0, max),
        P::FractionalOdd => (1.0, max - 1.0),
        P::FractionalEven => (2.0, max),
    };
    let mut vertices = 0u64;
    for patch in &tessellation.factors {
        let culls = |factor: f64| factor.is_nan() || factor <= 0.0;
        if patch.edge_f16.iter().any(|bits| culls(f16_to_f64(*bits))) {
            continue;
        }
        let largest = patch
            .edge_f16
            .iter()
            .chain(&patch.inside_f16)
            .map(|bits| f16_to_f64(*bits))
            .map(|factor| {
                if factor.is_nan() {
                    lower
                } else {
                    factor.clamp(lower, upper)
                }
            })
            .fold(lower, f64::max)
            .ceil() as u64;
        let segments = match tessellation.partition_mode {
            P::Pow2 => largest.next_power_of_two(),
            P::Integer => largest,
            P::FractionalOdd => largest | 1,
            P::FractionalEven => largest + (largest & 1),
        };
        let side = segments + 1;
        vertices = vertices.checked_add(side.checked_mul(side)?)?;
    }
    vertices
        .checked_mul(u64::from(tessellation.instance_count))?
        .checked_mul(u64::from(tessellation.amplification_count.max(1)))
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f64::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f64.powi(-24),
        0x1f if mantissa == 0.0 => sign * f64::INFINITY,
        0x1f => f64::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// The bytes of one cell of a fragment function's explicit imageblock as the executors hold it: the
/// reflected master's `air.imageblock_data_size`, or the end of its last member if that is larger,
/// rounded up to a whole 32-bit word. The data size comes from the AIR, not the manifest, so the
/// cells are counted against [`MAX_RESOURCE_BYTES`] where the reflection is known
/// ([`fragment_imageblock_cap_errors`]).
pub fn fragment_imageblock_cell_bytes(reflected: &FragmentImageblock) -> u64 {
    let end = reflected
        .members
        .iter()
        .map(|member| u64::from(member.offset) + u64::from(member.size))
        .fold(u64::from(reflected.sample_size), u64::max);
    end.div_ceil(4) * 4
}

/// The cap error of a fragment case whose pass's imageblock cells (one
/// [`fragment_imageblock_cell_bytes`] cell per pixel of its imageblock's extent), with the
/// resources [`cap_errors`] counts, exceed [`MAX_RESOURCE_BYTES`]: the executors allocate the
/// cells before any device work. Nothing when the case declares no imageblock or its literal
/// resources alone exceed the cap ([`cap_errors`] names those).
pub fn fragment_imageblock_cap_errors(
    case: &AuthoredCase,
    reflected: &FragmentImageblock,
) -> Vec<String> {
    let Some(imageblock) = &case.fragment_imageblock else {
        return Vec::new();
    };
    let [width, height] = imageblock.dimensions;
    let cell = fragment_imageblock_cell_bytes(reflected);
    let cells = (u64::from(width) * u64::from(height)).saturating_mul(cell);
    match resource_bytes(case) {
        Ok(bytes)
            if bytes <= MAX_RESOURCE_BYTES && bytes.saturating_add(cells) > MAX_RESOURCE_BYTES =>
        {
            vec![format!(
                "resources total {} bytes with the fragment imageblock's {cells} bytes of \
                 {width}x{height} {cell}-byte cells; the cap is {MAX_RESOURCE_BYTES}",
                bytes.saturating_add(cells)
            )]
        }
        _ => Vec::new(),
    }
}

/// Bytes the executors allocate for the case's resources: every literal `*_b64` payload in the
/// manifest (so a new resource kind is counted without a new rule), plus storage that has no
/// literal bytes: threadgroup memory, indirect command records, and device-buffer address tables.
fn resource_bytes(case: &AuthoredCase) -> Result<u64, String> {
    let manifest =
        serde_json::to_value(case).map_err(|error| format!("serialize case: {error}"))?;
    let mut total = literal_bytes(&manifest);
    for memory in &case.threadgroup_memory {
        total = total.saturating_add(u64::from(memory.length));
    }
    for array in &case.device_buffer_arrays {
        total = total.saturating_add(u64::from(array.length).saturating_mul(8));
    }
    for command_buffer in &case.command_buffers {
        let records = command_buffer.record_storage_bytes()?;
        total = total.saturating_add(records as u64);
    }
    Ok(total)
}

fn literal_bytes(value: &serde_json::Value) -> u64 {
    match value {
        serde_json::Value::Object(fields) => fields
            .iter()
            .map(|(key, value)| match value {
                serde_json::Value::String(text) if key.ends_with("_b64") => {
                    base64_decoded_len(text)
                }
                value => literal_bytes(value),
            })
            .fold(0, u64::saturating_add),
        serde_json::Value::Array(items) => {
            items.iter().map(literal_bytes).fold(0, u64::saturating_add)
        }
        _ => 0,
    }
}

fn base64_decoded_len(text: &str) -> u64 {
    let padding = text.bytes().rev().take_while(|byte| *byte == b'=').count() as u64;
    ((text.len() as u64) / 4 * 3).saturating_sub(padding)
}

#[cfg(test)]
mod tests;
