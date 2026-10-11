//! The data and texture types of AIR's `air.arg_type_name`: the MSL spelling of an argument's or a
//! member's type, which is the only place AIR states a texture's type, a texture's component type,
//! or a buffer's vector and matrix shape (an opaque-pointer module erases them). The grammar is
//! the MSL type grammar, not a list of names: a scalar, an optional vector width or `CxR` matrix
//! shape, an optional `packed_` prefix; a texture template with its component type.

use crate::metal::{DataType, TextureType};

/// The MSL scalar types and the `MTLDataType` name each one's family has.
const SCALARS: [(&str, &str); 12] = [
    ("float", "Float"),
    ("half", "Half"),
    ("bfloat", "BFloat"),
    ("int", "Int"),
    ("uint", "UInt"),
    ("short", "Short"),
    ("ushort", "UShort"),
    ("char", "Char"),
    ("uchar", "UChar"),
    ("long", "Long"),
    ("ulong", "ULong"),
    ("bool", "Bool"),
];

/// The `MTLDataType` of an MSL scalar, vector or matrix type (`float`, `uint4`, `half3x3`,
/// `packed_float3`), or `None` for any other spelling.
pub(super) fn data_type(name: &str) -> Option<DataType> {
    let name = name.strip_prefix("packed_").unwrap_or(name);
    let (base, family) = SCALARS
        .iter()
        .find(|(msl, _)| name.starts_with(msl))
        .copied()?;
    let shape = &name[base.len()..];
    let valid = match shape.as_bytes() {
        [] => true,
        [n] => (b'2'..=b'4').contains(n),
        // Matrices exist for float and half only.
        [c, b'x', r] => {
            matches!(family, "Float" | "Half")
                && (b'2'..=b'4').contains(c)
                && (b'2'..=b'4').contains(r)
        }
        _ => false,
    };
    if !valid {
        return None;
    }
    DataType::from_name(&format!("MTLDataType{family}{shape}"))
}

/// A texture argument's or member's type: its texture type, component type, and whether it is a
/// depth texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Texture {
    pub texture_type: TextureType,
    pub data_type: DataType,
    pub depth: bool,
}

/// The MSL texture templates: (name, texture type, depth).
const TEXTURES: [(&str, TextureType, bool); 16] = [
    ("texture1d", TextureType::Type1D, false),
    ("texture1d_array", TextureType::Type1DArray, false),
    ("texture2d", TextureType::Type2D, false),
    ("texture2d_array", TextureType::Type2DArray, false),
    ("texture2d_ms", TextureType::Type2DMultisample, false),
    (
        "texture2d_ms_array",
        TextureType::Type2DMultisampleArray,
        false,
    ),
    ("texture3d", TextureType::Type3D, false),
    ("texturecube", TextureType::Cube, false),
    ("texturecube_array", TextureType::CubeArray, false),
    ("texture_buffer", TextureType::TextureBuffer, false),
    ("depth2d", TextureType::Type2D, true),
    ("depth2d_array", TextureType::Type2DArray, true),
    ("depth2d_ms", TextureType::Type2DMultisample, true),
    (
        "depth2d_ms_array",
        TextureType::Type2DMultisampleArray,
        true,
    ),
    ("depthcube", TextureType::Cube, true),
    ("depthcube_array", TextureType::CubeArray, true),
];

/// The texture an MSL texture type names (`texture2d<float, sample>`, `depth2d<float, read>`),
/// or `None` for any other spelling.
pub(super) fn texture(name: &str) -> Option<Texture> {
    let (template, rest) = name.split_once('<')?;
    let args = rest.strip_suffix('>')?;
    let component = args.split(',').next()?.trim();
    let (_, texture_type, depth) = TEXTURES.iter().find(|(t, _, _)| *t == template)?;
    Some(Texture {
        texture_type: *texture_type,
        data_type: data_type(component)?,
        depth: *depth,
    })
}

/// The element type an MSL `array<T, N>` names, or the name itself when it is no such array.
pub(super) fn array_element(name: &str) -> &str {
    name.strip_prefix("array<")
        .and_then(|rest| rest.strip_suffix('>'))
        .and_then(|inner| inner.rsplit_once(','))
        .map_or(name, |(element, _)| element.trim())
}
