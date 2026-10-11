/// A Metal `[[function_constant(N)]]` discovered from its AIR `air.fc_initializer` initializer
/// global. The default value is not recoverable from the IR (the initializer is externally
/// specialized), so only the index, symbol name, LLVM type and ABI type encoding are carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionConstant {
    /// The `[[function_constant(N)]]` index.
    pub index: u32,
    /// The base symbol (the mangled name before the `.MTL_FC_INIT_` marker).
    pub name: String,
    /// The LLVM-IR type of the constant (`i32`, `i1`, `<4 x i32>`, `float`, ...).
    pub type_name: String,
    /// Itanium ABI type encoding carried after `MTL_FC_INIT_<index>_` (`j`, `Dv4_j`, `Dh`, ...).
    /// Unlike LLVM's signless integer type, it keeps the Metal scalar's signedness and lanes.
    pub abi_type_encoding: String,
}

/// The base name, index, and type encoding of the shared initializer symbol ABI.
pub(crate) fn function_constant_symbol(name: &str) -> Option<(&str, u32, &str)> {
    let (base, marker) = name.split_once(".MTL_FC_INIT_")?;
    let end = marker
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(marker.len());
    let index = marker[..end].parse().ok()?;
    let encoding = marker[end..].strip_prefix('_').unwrap_or_default();
    Some((base, index, encoding))
}

/// The module's `[[function_constant]]` initializer globals, `@<base>.MTL_FC_INIT_<N>_<suffix>`
/// (Apple's stable function-constant ABI marker): one entry per distinct index, sorted.
pub(crate) fn parse_function_constants(ll: &str) -> Vec<FunctionConstant> {
    let mut out: Vec<FunctionConstant> = Vec::new();
    for line in ll.lines() {
        let t = line.trim_start();
        if !t.starts_with('@') || !t.contains(".MTL_FC_INIT_") {
            continue;
        }
        let Some(eq) = t.find(" = ") else {
            continue;
        };
        let Some((base, index, abi_type_encoding)) = function_constant_symbol(t[1..eq].trim())
        else {
            continue;
        };
        if out.iter().any(|f| f.index == index) {
            continue;
        }
        let declared = fc_global_decl_type_and_initializer(&t[eq + 3..]);
        out.push(FunctionConstant {
            index,
            name: base.to_string(),
            type_name: declared.map(|(type_name, _)| type_name).unwrap_or_default(),
            abi_type_encoding: abi_type_encoding.to_owned(),
        });
    }
    out.sort_by_key(|f| f.index);
    out
}

/// Whether a global's top-level section attribute names the function-constant initializer ABI.
pub(crate) fn has_function_constant_section(declaration: &str) -> bool {
    crate::air::text::split_top_level(crate::air::text::strip_comment(declaration), ',')
        .into_iter()
        .skip(1)
        .any(|attribute| {
            attribute.strip_prefix("section").is_some_and(|tail| {
                tail.starts_with(char::is_whitespace) && tail.trim() == "\"air.fc_initializer\""
            })
        })
}

/// Definedness declared by an AIR function-constant global, independent of its symbol name.
pub(crate) fn function_constant_definedness(declaration: &str) -> Option<bool> {
    let (name, body) = declaration.trim().split_once(" = ")?;
    if !name.starts_with('@') || !has_function_constant_section(body) {
        return None;
    }
    let (_, initializer) = fc_global_decl_type_and_initializer(body)?;
    Some(initializer.split_whitespace().next()?.trim_end_matches(',') != "undef")
}

/// The typed initializer following a global definition's storage keyword.
pub(crate) fn global_initializer_body(mut declaration: &str) -> Option<&str> {
    loop {
        declaration = declaration.trim_start();
        let end = declaration.find(char::is_whitespace)?;
        let keyword = &declaration[..end];
        declaration = &declaration[end..];
        if matches!(keyword, "constant" | "global") {
            return Some(declaration.trim_start());
        }
    }
}

/// Split the scalar/vector type from its initializer for inspection and definedness.
fn fc_global_decl_type_and_initializer(decl: &str) -> Option<(String, &str)> {
    let s = global_initializer_body(decl)?;
    let (type_name, rest) = if let Some(rest) = s.strip_prefix('<') {
        let end = rest.find('>')?;
        (format!("<{}>", &rest[..end]), &rest[end + 1..])
    } else {
        let token = s.split_whitespace().next()?;
        (token.to_string(), &s[token.len()..])
    };
    Some((type_name, rest.trim_start()))
}
