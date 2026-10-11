/// Descriptor capacity for a texture-handle array whose length AIR does not state: a runtime
/// `array_ref<texture...>`, sized by a function constant rather than by the type name. The old
/// engine bound this many descriptors; a case's texture array may hold at most this many elements
/// (`case::validate`).
pub const TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT: u32 = 128;

/// Texture dimensionality decoded from a Metal texture type name, independent of the SPIR-V emit
/// enum (SPIR-V's `Dim`) so it can appear in the serializable reflection ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TextureDimension {
    D1,
    D2,
    D3,
    Cube,
    Buffer,
}

/// Sampled component class shared by AIR metadata, image lowering, and executable reflection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureComponent {
    Float,
    Sint,
    Uint,
}

/// The SPIR-V `OpTypeImage` format a write-capable Metal texture lowers to, decoded from its type
/// name's scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TextureFormat {
    R8,
    Rgba8,
    R16f,
    R16ui,
    Rg16f,
    Rg32f,
    R32f,
    R32i,
    R32ui,
    Rgba32i,
    Rgba32ui,
    Rgba32f,
    Rgba16f,
    Rgba8ui,
    Rgba16ui,
    Rgba8i,
    Rgba16i,
}

/// The full shape a Metal texture type name (`texture2d_array<half, sample>`, `texture_buffer<uint,
/// read>`, `array_ref<texture2d<float, sample>>`, …) implies. This is THE one decoder of the texture
/// type-name grammar — the interface pass (emit-time image types + storage classification), the
/// embedded-argument-buffer scan, and the public reflection facade all derive from it, so the
/// grammar lives in exactly one place. See [`texture_shape_from_name`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureShape {
    pub dimension: TextureDimension,
    pub arrayed: bool,
    pub multisampled: bool,
    pub component: TextureComponent,
    /// The access qualifier declares `write`/`read_write` — a storage image (vs `sample`/`read`).
    pub writable: bool,
    /// The argument is a descriptor array of texture handles rather than a single texture — either
    /// a fixed `array<texture..., N>` (see `array_length`) or a runtime `array_ref<texture...>`.
    pub array_ref: bool,
    /// Fixed texture-handle array length (`array<texture..., N>`), or `None` for a runtime
    /// `array_ref` and for a single texture.
    pub array_length: Option<u32>,
    /// For a `writable` texture, the storage-image texel format the emitter decorates the
    /// `OpTypeImage` with; `None` for a sampled texture.
    pub storage_format: Option<TextureFormat>,
}

impl TextureShape {
    /// The descriptors this texture argument occupies: a fixed `array<texture..., N>`'s `N`,
    /// [`TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT`] for a runtime-length `array_ref`, and `1` for a
    /// single texture.
    pub fn descriptor_count(&self) -> u32 {
        if !self.array_ref {
            return 1;
        }
        self.array_length
            .unwrap_or(TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT)
    }
}

/// Decode a Metal texture argument/type name into its [`TextureShape`]. Matches what the interface
/// pass uses to construct the emitted image type, including its multisample operand.
///
/// "Arrayed" is read ONCE, off the `_array` suffix, rather than from a per-dimension substring per
/// arrayed spelling. The enumerated form had to name every combination and it missed one:
/// `texture2d_ms_array` does not contain the substring `2d_array`, so it reflected as a plain
/// multisample 2D texture while the emitted `OpTypeImage` was arrayed -- two derivations of one
/// fact, disagreeing. `depth2d_ms_array` had the same hole.
pub fn texture_shape_from_name(name: &str) -> TextureShape {
    let (writable, array_ref) = texture_access_from_name(name);
    let array_length = texture_handle_array(name).flatten();
    let shape_name = if array_ref {
        name.find("texture")
            .or_else(|| name.find("depth"))
            .and_then(|start| name.get(start..))
            .unwrap_or(name)
    } else {
        name
    };
    let head = shape_name
        .split_once('<')
        .map(|(h, _)| h)
        .unwrap_or(shape_name);
    let dimension = if head.contains("texture_buffer") {
        TextureDimension::Buffer
    } else if head.contains("1d") {
        TextureDimension::D1
    } else if head.contains("3d") {
        TextureDimension::D3
    } else if head.contains("cube") {
        TextureDimension::Cube
    } else {
        TextureDimension::D2
    };
    // `texture_buffer` is the one name that cannot be arrayed, and it does not end in `_array`
    // either, so the suffix answers every spelling on its own.
    let arrayed = head.ends_with("_array");
    let multisampled = head.contains("_ms");
    let component = texture_component_from_name(shape_name);
    let storage_format = if writable {
        Some(storage_format_from_name(name, component))
    } else {
        None
    };
    TextureShape {
        dimension,
        arrayed,
        multisampled,
        component,
        writable,
        array_ref,
        array_length,
        storage_format,
    }
}

/// Whether a texture type name declares a handle ARRAY rather than a single texture, and — for the
/// fixed `array<texture..., N>` spelling — the declared length `N`.
///
/// One scan answers both questions, so "is this an array" and "how long is it" can never be decided
/// by two different rules. `Some(None)` is a runtime-length `array_ref<texture...>`, whose length is
/// not in the type name; `None` is a single texture.
fn texture_handle_array(name: &str) -> Option<Option<u32>> {
    let compact = name
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if compact.starts_with("array_ref<texture") || compact.starts_with("array_ref<depth") {
        return Some(None);
    }
    if !compact.starts_with("array<texture") && !compact.starts_with("array<depth") {
        return None;
    }
    // `array<texture2d<half, sample>, 32>` — the length is the last comma-separated field before the
    // outermost `>`. A malformed spelling stays an array of unknown (runtime) length rather than
    // silently degrading to a single texture.
    let length = compact
        .rfind('>')
        .and_then(|end| compact[..end].rsplit_once(','))
        .and_then(|(_, tail)| tail.parse().ok());
    Some(length)
}

/// The storage-image texel format a write-capable texture lowers to, from its scalar precision. A
/// `half`/`ushort` scalar narrows the format; the default is the full-width form.
fn storage_format_from_name(name: &str, component: TextureComponent) -> TextureFormat {
    match component {
        TextureComponent::Float => {
            if name.contains("<half") {
                TextureFormat::Rgba16f
            } else if name.contains("<float") {
                TextureFormat::R32f
            } else {
                TextureFormat::Rgba32f
            }
        }
        TextureComponent::Uint => {
            if name.contains("<ushort") {
                TextureFormat::Rgba16ui
            } else {
                TextureFormat::Rgba8ui
            }
        }
        TextureComponent::Sint => {
            // The signed default has to widen for `<short` exactly as the unsigned one does for
            // `<ushort` just above. Without this branch a `texture2d<short, write>` was decorated
            // `Rgba8i`, an 8-bit storage image for a 16-bit texture, so every component a shader
            // wrote was truncated to the signed byte range; 5 of the 14579 local corpus sources
            // reach it.
            if name.contains("<short") {
                TextureFormat::Rgba16i
            } else {
                TextureFormat::Rgba8i
            }
        }
    }
}

fn texture_component_from_name(name: &str) -> TextureComponent {
    let Some((_, rest)) = name.split_once('<') else {
        return TextureComponent::Float;
    };
    let scalar = rest
        .split(|c: char| c == ',' || c == '>' || c.is_whitespace())
        .find(|part| !part.is_empty())
        .unwrap_or("");
    // A nested `array<texture2d<uint, ...` puts the real scalar after the inner `<`.
    let scalar = scalar.rsplit('<').next().unwrap_or(scalar);
    match scalar {
        "uint" | "ushort" | "uchar" => TextureComponent::Uint,
        "int" | "short" | "char" => TextureComponent::Sint,
        _ => TextureComponent::Float,
    }
}

/// `(writable, array_ref)` decoded from the access qualifier — the second template field
/// (`texture2d<float, write>`) and texture-handle array wrappers.
fn texture_access_from_name(name: &str) -> (bool, bool) {
    let array_ref = texture_handle_array(name).is_some();
    let Some((_, rest)) = name.split_once('<') else {
        return (false, array_ref);
    };
    let Some(inner) = rest.split('>').next() else {
        return (false, array_ref);
    };
    let mut fields = inner.split(',').map(str::trim);
    let _scalar = fields.next();
    let writable = matches!(fields.next(), Some("write") | Some("read_write"));
    (writable, array_ref)
}
