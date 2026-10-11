//! The old engine's LLVM text front end, as far as the command guard parses with it: function
//! headers, declarations, direct calls and the typed values and types they carry. Copied with the
//! guard so that it accepts and refuses exactly what it did.

use crate::air::text::{
    matching_paren, parse_bfloat_literal, parse_float32_hex_literal, parse_float_literal,
    parse_float_special_literal, parse_half_literal, parse_hex_literal, parse_i64_literal,
    parse_u64_literal, split_top_level, split_top_level_whitespace, strip_comment,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) enum LlType {
    Void,
    Float,
    Half,
    BFloat,
    Int(u32),
    Ptr(u32),
    Vector(Box<LlType>, u32),
    Array(Box<LlType>, u32),
    Struct(Vec<LlType>),
    Named(String),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TypedValue {
    pub(super) ty: LlType,
    pub(super) value: LlValue,
}

impl TypedValue {}

#[derive(Clone, Debug)]
pub(super) enum LlValue {
    Local(String),
    Global(String),
    Bool(bool),
    Int(u64),
    SignedInt(i64),
    Hex(u64),
    Float(f64),
    Float32Bits(u32),
    HalfBits(u16),
    BFloatBits(u16),
    Vector(Vec<TypedValue>),
    Array(Vec<TypedValue>),
    Struct(Vec<TypedValue>),
    Splat(Box<TypedValue>),
    Gep(Box<LlGep>),
    IntToPtr {
        source: Box<TypedValue>,
        destination: LlType,
    },
    Zero,
    Undef,
}

impl PartialEq for LlValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Local(a), Self::Local(b)) | (Self::Global(a), Self::Global(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) | (Self::Hex(a), Self::Hex(b)) => a == b,
            (Self::SignedInt(a), Self::SignedInt(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Float32Bits(a), Self::Float32Bits(b)) => a == b,
            (Self::HalfBits(a), Self::HalfBits(b)) | (Self::BFloatBits(a), Self::BFloatBits(b)) => {
                a == b
            }
            (Self::Vector(a), Self::Vector(b))
            | (Self::Array(a), Self::Array(b))
            | (Self::Struct(a), Self::Struct(b)) => a == b,
            (Self::Splat(a), Self::Splat(b)) => a == b,
            (Self::Gep(a), Self::Gep(b)) => a == b,
            (
                Self::IntToPtr {
                    source: a_source,
                    destination: a_destination,
                },
                Self::IntToPtr {
                    source: b_source,
                    destination: b_destination,
                },
            ) => a_source == b_source && a_destination == b_destination,
            (Self::Zero, Self::Zero) | (Self::Undef, Self::Undef) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LlGep {
    pub(super) inbounds: bool,
    pub(super) source_ty: LlType,
    pub(super) base: TypedValue,
    pub(super) indices: Vec<TypedValue>,
}

#[derive(Clone, Debug)]
pub(super) struct LlFunction {
    pub(super) name: String,
    pub(super) params: Vec<(String, LlType)>,
}

#[derive(Clone, Debug)]
pub(super) struct LlDeclaration {
    pub(super) name: String,
    pub(super) ret: LlType,
    pub(super) params: Vec<LlType>,
}

/// A direct call: its callee. Its return type and arguments are parsed, so a call the old engine
/// refused is refused here too, and not kept.
#[derive(Clone, Debug)]
pub(super) struct LlCall {
    pub(super) callee: String,
}

#[cfg(test)]
pub(super) fn parse_function(
    lines: &[&str],
    start: usize,
) -> Result<(LlFunction, std::ops::Range<usize>, usize), String> {
    let head = strip_comment(lines[start]).trim();
    let function = parse_function_header(head)?;
    let name = function.name.clone();
    let mut i = start + 1;
    while i < lines.len() {
        let line = strip_comment(lines[i]).trim();
        if line == "}" {
            return Ok((function, start + 1..i, i + 1));
        }
        i += 1;
    }
    Err(format!("native emitter: unterminated function {name}"))
}

pub(super) fn parse_function_header(head: &str) -> Result<LlFunction, String> {
    let at = head
        .find('@')
        .ok_or_else(|| format!("native emitter: define without function name: {head}"))?;
    let (name, open) = parse_global_symbol_with_params(head, at, "define")?;
    let close = matching_paren(head, open)
        .ok_or_else(|| format!("native emitter: unmatched params in define: {head}"))?;
    parse_return_type(&head["define".len()..at])?;
    let params = parse_params(&head[open + 1..close])?;
    Ok(LlFunction { name, params })
}

pub(super) fn parse_declaration(line: &str) -> Result<LlDeclaration, String> {
    let head = strip_comment(line).trim();
    let at = head
        .find('@')
        .ok_or_else(|| format!("native emitter: declare without function name: {head}"))?;
    let (name, open) = parse_global_symbol_with_params(head, at, "declare")?;
    let close = matching_paren(head, open)
        .ok_or_else(|| format!("native emitter: unmatched params in declare: {head}"))?;
    let ret = parse_return_type(&head["declare".len()..at])?;
    let params = parse_decl_params(&head[open + 1..close])?;
    Ok(LlDeclaration { name, ret, params })
}

pub(super) fn parse_global_symbol_with_params(
    s: &str,
    at: usize,
    context: &str,
) -> Result<(String, usize), String> {
    let after_at = at + 1;
    if s[after_at..].starts_with('"') {
        let (name, end) = parse_quoted_global_symbol(s, after_at)?;
        let open = s[end..]
            .find('(')
            .map(|p| p + end)
            .ok_or_else(|| format!("native emitter: {context} without params: {s}"))?;
        return Ok((name, open));
    }
    let open = s[at..]
        .find('(')
        .map(|p| p + at)
        .ok_or_else(|| format!("native emitter: {context} without params: {s}"))?;
    Ok((s[after_at..open].trim_end().to_string(), open))
}

pub(super) fn parse_quoted_global_symbol(s: &str, quote: usize) -> Result<(String, usize), String> {
    let mut out = String::new();
    let mut pos = quote + 1;
    while pos < s.len() {
        let ch = s[pos..]
            .chars()
            .next()
            .ok_or_else(|| format!("native emitter: malformed quoted symbol: {s}"))?;
        pos += ch.len_utf8();
        match ch {
            '"' => return Ok((out, pos)),
            '\\' => {
                let hi = s[pos..].chars().next();
                let lo = hi.and_then(|hi| {
                    let next = pos + hi.len_utf8();
                    s[next..].chars().next()
                });
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    if hi.is_ascii_hexdigit() && lo.is_ascii_hexdigit() {
                        pos += hi.len_utf8() + lo.len_utf8();
                        let byte =
                            ((hi.to_digit(16).unwrap() << 4) | lo.to_digit(16).unwrap()) as u8;
                        out.push(byte as char);
                        continue;
                    }
                }
                let escaped = s[pos..].chars().next().ok_or_else(|| {
                    format!("native emitter: unterminated quoted symbol escape: {s}")
                })?;
                pos += escaped.len_utf8();
                out.push(escaped);
            }
            _ => out.push(ch),
        }
    }
    Err(format!("native emitter: unterminated quoted symbol: {s}"))
}

pub(super) fn parse_decl_params(s: &str) -> Result<Vec<LlType>, String> {
    if s.trim().is_empty() {
        return Ok(vec![]);
    }
    split_top_level(s, ',')
        .into_iter()
        .map(|raw| parse_type_prefix(raw).map(|(ty, _)| ty))
        .collect()
}

pub(super) fn parse_return_type(prefix: &str) -> Result<LlType, String> {
    let toks = split_top_level_whitespace(prefix);
    for i in (0..toks.len()).rev() {
        let candidate = toks[i..].join(" ");
        if let Ok(ty) = parse_type(&candidate) {
            return Ok(ty);
        }
    }
    Err(format!(
        "native emitter: could not parse function return type from `{}`",
        prefix.trim()
    ))
}

/// A definition's parameters, named. A `byval(T)` attribute's type is parsed, so a parameter the old
/// engine refused is refused here too, and not kept.
pub(super) fn parse_params(s: &str) -> Result<Vec<(String, LlType)>, String> {
    let mut out = Vec::new();
    if s.trim().is_empty() {
        return Ok(out);
    }
    for raw in split_top_level(s, ',') {
        let raw = raw.trim();
        let pct = raw
            .rfind('%')
            .ok_or_else(|| format!("native emitter: parameter has no SSA name: {raw}"))?;
        let name = raw[pct..]
            .split_whitespace()
            .next()
            .ok_or_else(|| format!("native emitter: malformed parameter name: {raw}"))?
            .to_string();
        let (ty, _) = parse_type_prefix(raw[..pct].trim())?;
        if let Some(byval) = raw.find("byval(") {
            let open = byval + "byval".len();
            let close = matching_paren(raw, open)
                .ok_or_else(|| format!("native emitter: unmatched byval type: {raw}"))?;
            parse_type(raw[open + 1..close].trim())?;
        }
        out.push((name, ty));
    }
    Ok(out)
}

pub(super) fn parse_type_prefix(s: &str) -> Result<(LlType, String), String> {
    let toks = split_top_level_whitespace(s);
    for i in (1..=toks.len()).rev() {
        let candidate = toks[..i].join(" ");
        if let Ok(ty) = parse_type(&candidate) {
            return Ok((ty, candidate));
        }
    }
    Err(format!(
        "native emitter: could not parse type prefix from `{s}`"
    ))
}

pub(super) fn parse_typed_value(s: &str) -> Result<TypedValue, String> {
    let s = s.trim();
    let toks = split_top_level_whitespace(s);
    let mut last_value_err = None;
    for value_start in (1..toks.len()).rev() {
        let value_text = toks[value_start..].join(" ");
        let value = match parse_value(value_text.trim()) {
            Ok(value) => value,
            Err(err) => {
                last_value_err = Some(err);
                continue;
            }
        };
        for ty_start in 0..value_start {
            for ty_end in ((ty_start + 1)..=value_start).rev() {
                let ty_text = toks[ty_start..ty_end].join(" ");
                if let Ok(ty) = parse_type(&ty_text) {
                    return Ok(TypedValue { ty, value });
                }
            }
        }
    }
    if let Some(err) = last_value_err {
        return Err(format!(
            "native emitter: malformed typed value `{s}` ({err})"
        ));
    }
    Err(format!("native emitter: malformed typed value `{s}`"))
}

pub(super) fn parse_value(s: &str) -> Result<LlValue, String> {
    let s = s.trim();
    if s == "undef" || s == "poison" {
        Ok(LlValue::Undef)
    } else if s == "zeroinitializer" || s == "null" {
        Ok(LlValue::Zero)
    } else if s == "true" {
        Ok(LlValue::Bool(true))
    } else if s == "false" {
        Ok(LlValue::Bool(false))
    } else if s.starts_with('<') && s.ends_with('>') {
        let inner = s[1..s.len() - 1].trim();
        if inner.starts_with('{') && inner.ends_with('}') {
            let fields = split_top_level(&inner[1..inner.len() - 1], ',')
                .into_iter()
                .map(parse_typed_value)
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(LlValue::Struct(fields));
        }
        let lanes = split_top_level(inner, ',')
            .into_iter()
            .map(parse_typed_value)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LlValue::Vector(lanes))
    } else if s.starts_with('[') && s.ends_with(']') {
        let elems = split_top_level(&s[1..s.len() - 1], ',')
            .into_iter()
            .map(parse_typed_value)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LlValue::Array(elems))
    } else if s.starts_with('{') && s.ends_with('}') {
        let fields = split_top_level(&s[1..s.len() - 1], ',')
            .into_iter()
            .map(parse_typed_value)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LlValue::Struct(fields))
    } else if s.starts_with("c\"") && s.ends_with('"') {
        parse_byte_string_literal(s).map(|bytes| {
            LlValue::Array(
                bytes
                    .into_iter()
                    .map(|byte| TypedValue {
                        ty: LlType::Int(8),
                        value: LlValue::Int(byte as u64),
                    })
                    .collect(),
            )
        })
    } else if let Some(inner) = s.strip_prefix("splat (").and_then(|s| s.strip_suffix(')')) {
        Ok(LlValue::Splat(Box::new(parse_typed_value(inner)?)))
    } else if let Some(rest) = s.strip_prefix("getelementptr ") {
        Ok(LlValue::Gep(Box::new(parse_gep(rest)?)))
    } else if let Some(inner) = s
        .strip_prefix("inttoptr (")
        .and_then(|value| value.strip_suffix(')'))
    {
        let (source, destination) = inner.rsplit_once(" to ").ok_or_else(|| {
            format!("native emitter: malformed inttoptr constant expression `{s}`")
        })?;
        Ok(LlValue::IntToPtr {
            source: Box::new(parse_typed_value(source)?),
            destination: parse_type(destination.trim())?,
        })
    } else if s.starts_with('%') {
        Ok(LlValue::Local(s.to_string()))
    } else if s.starts_with('@') {
        Ok(LlValue::Global(s.to_string()))
    } else if let Some(bits) = parse_half_literal(s) {
        Ok(LlValue::HalfBits(bits))
    } else if let Some(bits) = parse_bfloat_literal(s) {
        Ok(LlValue::BFloatBits(bits))
    } else if let Some(bits) = parse_float32_hex_literal(s) {
        Ok(LlValue::Float32Bits(bits))
    } else if let Some(bits) = parse_hex_literal(s) {
        Ok(LlValue::Hex(bits))
    } else if let Ok(v) = parse_i64_literal(s) {
        Ok(LlValue::SignedInt(v))
    } else if let Ok(v) = parse_u64_literal(s) {
        Ok(LlValue::Int(v))
    } else if let Some(bits) = parse_float_special_literal(s) {
        // LLVM emits NaN/Inf as `+qnan`, `-qnan`, `+inf`, `-inf`, `qnan`, `snan`, etc.
        // Map to the exact IEEE 754 f32 bit pattern.
        Ok(LlValue::Float32Bits(bits))
    } else if let Ok(v) = parse_float_literal(s) {
        Ok(LlValue::Float(v))
    } else {
        Err(format!("native emitter: unsupported value `{s}`"))
    }
}

fn parse_byte_string_literal(s: &str) -> Result<Vec<u8>, String> {
    let body = s
        .strip_prefix("c\"")
        .and_then(|s| s.strip_suffix('"'))
        .ok_or_else(|| format!("native emitter: malformed byte string `{s}`"))?;
    let bytes = body.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        if i + 1 >= bytes.len() {
            return Err(format!("native emitter: trailing byte string escape `{s}`"));
        }
        if i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        let escaped = match bytes[i + 1] {
            b'\\' => b'\\',
            b'"' => b'"',
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            other => {
                return Err(format!(
                    "native emitter: unsupported byte string escape `\\{}` in `{s}`",
                    other as char
                ));
            }
        };
        out.push(escaped);
        i += 2;
    }
    Ok(out)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Diagnostic for a call with no `@name` callee. An AIR call through a `%reg` value is an INDIRECT
/// call — in Metal these are typically Metal **visible function table** dispatches (`%reg` traces to
/// `air.get_function_pointer_visible_function_table(table, index)`): a runtime-bound function pointer
/// with no clean Logical-SPIR-V equivalent (R1). Name the function-pointer register explicitly and
/// classify it as an indirect call (the frontier bucketer keys on "indirect call"/"function pointer"),
/// rather than the generic "call without callee" which reads like a parse failure.
fn indirect_call_diagnostic(s: &str) -> String {
    // The callee register is the last whitespace-separated token before the argument list `(`.
    let fp = s
        .find('(')
        .map(|open| {
            s[..open]
                .split_whitespace()
                .last()
                .unwrap_or("")
                .to_string()
        })
        .filter(|t| t.starts_with('%'))
        .unwrap_or_default();
    if fp.is_empty() {
        format!("native emitter: call without callee: {s}")
    } else {
        format!(
            "native emitter: unsupported indirect call through function pointer {fp} \
             (Metal visible function table); not expressible in Logical SPIR-V: {s}"
        )
    }
}

pub(super) fn parse_call(s: &str) -> Result<LlCall, String> {
    let at = match s.find('@') {
        Some(at) => at,
        None => return Err(indirect_call_diagnostic(s)),
    };
    let (callee, open) = parse_global_symbol_with_params(s, at, "call")?;
    let close = matching_paren(s, open)
        .ok_or_else(|| format!("native emitter: unmatched call parens: {s}"))?;
    parse_return_type(&s[..at])?;
    let args_text = &s[open + 1..close];
    if !args_text.trim().is_empty() {
        for arg_text in split_top_level(args_text, ',') {
            parse_typed_value(arg_text)?;
        }
    }
    Ok(LlCall { callee })
}

pub(super) fn parse_gep(s: &str) -> Result<LlGep, String> {
    let inbounds = split_top_level_whitespace(s).contains(&"inbounds");
    let s = strip_wrapping_parens(strip_native_gep_flags(s));
    let parts = split_top_level(s, ',');
    if parts.len() < 3 {
        return Err(format!("native emitter: malformed getelementptr: {s}"));
    }
    let source_ty = parse_type(parts[0])?;
    let base = parse_typed_value(parts[1])?;
    let indices = parts[2..]
        .iter()
        .map(|idx| parse_typed_value(idx))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LlGep {
        inbounds,
        source_ty,
        base,
        indices,
    })
}

fn strip_wrapping_parens(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('(') && matching_paren(s, 0) == Some(s.len() - 1) {
        return s[1..s.len() - 1].trim();
    }
    s
}

pub(super) fn strip_native_gep_flags(mut s: &str) -> &str {
    s = s.trim();
    loop {
        let mut matched = false;
        for flag in ["inbounds", "nuw", "nusw"] {
            if let Some(rest) = s.strip_prefix(flag) {
                if rest.is_empty() || rest.starts_with(char::is_whitespace) {
                    s = rest.trim_start();
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            return s;
        }
    }
}

pub(super) fn parse_type(s: &str) -> Result<LlType, String> {
    let s = strip_leading_fast_math_flags(s.trim());
    if s == "void" {
        return Ok(LlType::Void);
    }
    // `metadata` is the operand type of debug/scope marker intrinsics (which `is_ignored_call_line`
    // drops structurally by operand type). Map it to a placeholder so an ignored intrinsic's
    // declaration parses rather than hard-erroring; it is never lowered.
    if s == "metadata" {
        return Ok(LlType::Void);
    }
    if s == "float" {
        return Ok(LlType::Float);
    }
    if s == "half" {
        return Ok(LlType::Half);
    }
    if s == "bfloat" {
        return Ok(LlType::BFloat);
    }
    if let Some(bits) = s.strip_prefix('i') {
        if bits.chars().all(|c| c.is_ascii_digit()) {
            return Ok(LlType::Int(
                bits.parse()
                    .map_err(|e| format!("native emitter: bad int type {s}: {e}"))?,
            ));
        }
    }
    if let Some(name) = s.strip_prefix('%') {
        // LLVM escapes inside a quoted identifier as `\xx`, never `\"`, so the first closing
        // quote ends the name. Requiring it to end the whole string is what stops the name from
        // swallowing a trailing parameter attribute that is quoted too -- a parameter spelled
        // `%"struct.metal::_atomic" addrspace(1)* nocapture "air-buffer-no-alias"` was read as
        // one named type running from the first quote to the last.
        if let Some(inner) = name
            .strip_prefix('"')
            .and_then(|rest| rest.strip_suffix('"'))
            .filter(|inner| !inner.contains('"'))
        {
            return Ok(LlType::Named(format!("%\"{inner}\"")));
        }
        if !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_')
        {
            return Ok(LlType::Named(format!("%{name}")));
        }
    }
    if s == "ptr" {
        return Ok(LlType::Ptr(0));
    }
    if let Some(rest) = s.strip_prefix("ptr addrspace(") {
        let n = rest
            .strip_suffix(')')
            .ok_or_else(|| format!("native emitter: malformed pointer type `{s}`"))?;
        return Ok(LlType::Ptr(n.parse().map_err(|e| {
            format!("native emitter: bad addrspace in `{s}`: {e}")
        })?));
    }
    if s.starts_with('<') && s.ends_with('>') {
        let inner = s[1..s.len() - 1].trim();
        if inner.starts_with('{') && inner.ends_with('}') {
            return parse_struct_fields(&inner[1..inner.len() - 1]).map(LlType::Struct);
        }
        let (lanes, elem) = split_vector_type(inner)?;
        return Ok(LlType::Vector(Box::new(parse_type(elem)?), lanes));
    }
    if s.starts_with('{') && s.ends_with('}') {
        return parse_struct_fields(&s[1..s.len() - 1]).map(LlType::Struct);
    }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = s[1..s.len() - 1].trim();
        let (len, elem) = split_vector_type(inner)?;
        return Ok(LlType::Array(Box::new(parse_type(elem)?), len));
    }
    // LLVM's pre-opaque pointer spelling, which is what `xcrun metal -S -emit-llvm` still emits:
    // `T*` and `T addrspace(N)*`. It carries the same information as `ptr`/`ptr addrspace(N)`
    // plus a pointee our pointer model does not keep, so parse the pointee for validation only.
    // Parsing it is what keeps the trailing `*` from being dropped by a caller that backtracks
    // over type prefixes, which would silently hand back the pointee as if it were the type.
    if let Some(pointee) = s.strip_suffix('*') {
        let (pointee, space) = split_trailing_addrspace(pointee.trim_end())?;
        parse_type(pointee)?;
        return Ok(LlType::Ptr(space));
    }
    Err(format!("native emitter: unsupported type `{s}`"))
}

/// Split a typed pointer's ` addrspace(N)` suffix off its pointee, defaulting to address space 0.
fn split_trailing_addrspace(s: &str) -> Result<(&str, u32), String> {
    let Some(open) = s.strip_suffix(')').and_then(|s| s.rfind(" addrspace(")) else {
        return Ok((s, 0));
    };
    let digits = &s[open + " addrspace(".len()..s.len() - 1];
    let space = digits
        .parse()
        .map_err(|e| format!("native emitter: bad addrspace in `{s}*`: {e}"))?;
    Ok((&s[..open], space))
}

fn strip_leading_fast_math_flags(mut s: &str) -> &str {
    loop {
        let toks = split_top_level_whitespace(s);
        let Some(flag) = toks.first().copied() else {
            return s;
        };
        if !matches!(
            flag,
            "fast" | "nnan" | "ninf" | "nsz" | "arcp" | "contract" | "afn" | "reassoc" | "volatile"
        ) {
            return s;
        }
        s = s[flag.len()..].trim_start();
    }
}

pub(super) fn parse_struct_fields(s: &str) -> Result<Vec<LlType>, String> {
    split_top_level(s, ',')
        .into_iter()
        .map(parse_type)
        .collect::<Result<Vec<_>, _>>()
}

pub(super) fn split_vector_type(s: &str) -> Result<(u32, &str), String> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '<' | '[' | '{' | '(' => depth += 1,
            '>' | ']' | '}' | ')' => depth -= 1,
            'x' if depth == 0 => {
                let before = s[..i].trim();
                let after = s[i + 1..].trim();
                if before.chars().all(|c| c.is_ascii_digit()) {
                    return Ok((
                        before
                            .parse()
                            .map_err(|e| format!("native emitter: bad vector lane count: {e}"))?,
                        after,
                    ));
                }
            }
            _ => {}
        }
    }
    Err(format!("native emitter: malformed vector type `<{s}>`"))
}

#[cfg(test)]
mod indirect_call_tests {
    use super::*;

    // An AIR call through a `%reg` callee (no `@name`) is an indirect call (Metal visible function
    // table). The diagnostic names the function-pointer register and uses the words the frontier
    // bucketer keys on ("indirect call" / "function pointer"), not the generic "call without callee".
    #[test]
    fn indirect_call_through_register_is_named() {
        let s = "fast <4 x float> %86(ptr addrspace(2) %0, ptr %91)";
        let msg = indirect_call_diagnostic(s);
        assert!(msg.contains("indirect call"), "{msg}");
        assert!(msg.contains("function pointer %86"), "{msg}");
        assert!(!msg.contains("call without callee"), "{msg}");
    }

    // A genuinely malformed call with no register callee still gets the generic message.
    #[test]
    fn malformed_call_without_register_falls_back_to_generic() {
        let s = "void ()";
        let msg = indirect_call_diagnostic(s);
        assert!(msg.contains("call without callee"), "{msg}");
    }
}

#[cfg(test)]
mod parse_tests {
    //! Type/value parser edge cases + a deterministic no-panic fuzz (refactor T8). `parse.rs` had
    //! two tests before this; the S7 emitter decomposition re-parses at emit time, so these pin the
    //! shapes it depends on. Pure-static (no external tools), part of the T2 inner-loop gate.
    use super::*;

    #[test]
    fn parse_type_scalars_and_ints() {
        assert_eq!(parse_type("void").unwrap(), LlType::Void);
        assert_eq!(parse_type("metadata").unwrap(), LlType::Void);
        assert_eq!(parse_type("float").unwrap(), LlType::Float);
        assert_eq!(parse_type("half").unwrap(), LlType::Half);
        assert_eq!(parse_type("bfloat").unwrap(), LlType::BFloat);
        assert_eq!(parse_type("i32").unwrap(), LlType::Int(32));
        assert_eq!(parse_type("i1").unwrap(), LlType::Int(1));
        // fast-math flags are stripped before the type
        assert_eq!(parse_type("nnan ninf float").unwrap(), LlType::Float);
    }

    #[test]
    fn parse_type_pointers() {
        assert_eq!(parse_type("ptr").unwrap(), LlType::Ptr(0));
        assert_eq!(parse_type("ptr addrspace(1)").unwrap(), LlType::Ptr(1));
        assert!(parse_type("ptr addrspace(2").is_err()); // missing ')'
        assert!(parse_type("ptr addrspace(x)").is_err()); // non-numeric
    }

    #[test]
    fn parse_type_typed_pointers() {
        // LLVM's pre-opaque spelling carries the address space on the pointee, and the pointee
        // itself is dropped -- both spellings of one pointer have to land on the same type.
        assert_eq!(parse_type("i32*").unwrap(), LlType::Ptr(0));
        assert_eq!(parse_type("i32 addrspace(1)*").unwrap(), LlType::Ptr(1));
        assert_eq!(parse_type("float addrspace(2)*").unwrap(), LlType::Ptr(2));
        assert_eq!(
            parse_type("<4 x float> addrspace(1)*").unwrap(),
            LlType::Ptr(1)
        );
        assert_eq!(
            parse_type("%struct.Foo addrspace(3)*").unwrap(),
            LlType::Ptr(3)
        );
        assert_eq!(parse_type("i32 addrspace(1)**").unwrap(), LlType::Ptr(0));
        // A pointee that is not a type is still a parse failure, so a stray `*` cannot pass.
        assert!(parse_type("notatype*").is_err());
        assert!(parse_type("i32 addrspace(x)*").is_err());
    }

    #[test]
    fn parse_type_prefix_keeps_the_typed_pointer_star() {
        // The prefix scan tries the longest candidate first, so before `parse_type` understood
        // `i32 addrspace(1)*` it fell through to `i32` and handed back the POINTEE as the type.
        assert_eq!(
            parse_type_prefix(r#"i32 addrspace(1)* nocapture readonly "air-buffer-no-alias""#)
                .unwrap(),
            (LlType::Ptr(1), "i32 addrspace(1)*".into())
        );
        let params = parse_params(r#"i32 addrspace(1)* nocapture %0, i32 %1"#).unwrap();
        assert_eq!(
            params,
            vec![
                ("%0".to_string(), LlType::Ptr(1)),
                ("%1".to_string(), LlType::Int(32)),
            ]
        );
    }

    #[test]
    fn parse_quoted_function_and_call_names() {
        let lines = [
            r#"define void @"re::df::pack"(ptr addrspace(1) %out) {"#,
            "  ret void",
            "}",
        ];
        let (function, _, _) = parse_function(&lines, 0).expect("parse function");
        assert_eq!(function.name, "re::df::pack");
        let declaration = parse_declaration(r#"declare void @"helper::quoted"(ptr addrspace(1))"#)
            .expect("parse declaration");
        assert_eq!(declaration.name, "helper::quoted");
        let call =
            parse_call(r#"void @"helper::quoted"(ptr addrspace(1) %out)"#).expect("parse call");
        assert_eq!(call.callee, "helper::quoted");
    }

    #[test]
    fn parse_type_quoted_name_stops_at_its_own_closing_quote() {
        // A parameter attribute is quoted too, so a name that ran to the LAST quote in the string
        // swallowed the whole tail and produced a `Named` nothing could resolve. It has to stop at
        // the first closing quote and leave the rest to the prefix scan.
        assert!(parse_type(r#"%"struct.metal::_atomic" nocapture "air-buffer-no-alias""#).is_err());
        assert_eq!(
            parse_type_prefix(
                r#"%"struct.metal::_atomic" addrspace(1)* nocapture "air-buffer-no-alias""#
            )
            .unwrap(),
            (
                LlType::Ptr(1),
                r#"%"struct.metal::_atomic" addrspace(1)*"#.into()
            )
        );
    }

    #[test]
    fn parse_type_named_and_quoted() {
        assert_eq!(
            parse_type("%struct.Foo").unwrap(),
            LlType::Named("%struct.Foo".to_string())
        );
        assert_eq!(
            parse_type(r#"%"a.b c""#).unwrap(),
            LlType::Named(r#"%"a.b c""#.to_string())
        );
    }

    #[test]
    fn parse_type_vectors_arrays_structs() {
        assert_eq!(
            parse_type("<4 x float>").unwrap(),
            LlType::Vector(Box::new(LlType::Float), 4)
        );
        // nested vector
        assert_eq!(
            parse_type("<2 x <4 x i8>>").unwrap(),
            LlType::Vector(Box::new(LlType::Vector(Box::new(LlType::Int(8)), 4)), 2)
        );
        assert_eq!(
            parse_type("[3 x i16]").unwrap(),
            LlType::Array(Box::new(LlType::Int(16)), 3)
        );
        assert_eq!(
            parse_type("{ float, i32 }").unwrap(),
            LlType::Struct(vec![LlType::Float, LlType::Int(32)])
        );
        // a vector wrapping a literal struct
        assert_eq!(
            parse_type("<{ i8, i8 }>").unwrap(),
            LlType::Struct(vec![LlType::Int(8), LlType::Int(8)])
        );
    }

    #[test]
    fn parse_type_rejects_garbage() {
        assert!(parse_type("banana").is_err());
        assert!(parse_type("iZ").is_err()); // 'i' then non-digits
        assert!(parse_type("").is_err());
    }

    #[test]
    fn parse_typed_value_backtracks_over_multiword_types() {
        // `parse_typed_value` scans splits of "type value"; a multi-token type must be recovered.
        // LlValue has no PartialEq (a production choice); match structurally instead of deriving it.
        let tv = parse_typed_value("i32 7").unwrap();
        assert_eq!(tv.ty, LlType::Int(32));
        assert!(matches!(tv.value, LlValue::Int(7)));

        let tv = parse_typed_value("<4 x float> zeroinitializer").unwrap();
        assert_eq!(tv.ty, LlType::Vector(Box::new(LlType::Float), 4));
        assert!(matches!(tv.value, LlValue::Zero));

        let tv = parse_typed_value("ptr addrspace(1) %0").unwrap();
        assert_eq!(tv.ty, LlType::Ptr(1));
        assert!(matches!(tv.value, LlValue::Local(ref s) if s == "%0"));

        let tv = parse_typed_value(
            "ptr addrspace(3) nonnull captures(none) inttoptr (i64 1024 to ptr addrspace(3))",
        )
        .unwrap();
        assert_eq!(tv.ty, LlType::Ptr(3));
        assert!(matches!(
            tv.value,
            LlValue::IntToPtr {
                source,
                destination: LlType::Ptr(3),
            } if matches!(source.value, LlValue::Int(1024))
        ));

        assert!(parse_typed_value("i32").is_err()); // no value token
        assert!(parse_typed_value("").is_err());
    }

    #[test]
    fn parse_typed_float32_bit_literal_preserves_bits() {
        let tv = parse_typed_value("float f0x358637BD").unwrap();
        assert_eq!(tv.ty, LlType::Float);
        assert!(matches!(tv.value, LlValue::Float32Bits(0x3586_37bd)));
    }

    // Deterministic, dependency-free fuzz: mutate valid seeds and assert the parsers return a
    // Result (never panic). A panic here is a parser bug rather than an input classification.
    #[test]
    fn parsers_never_panic_on_ascii_mutations() {
        let seeds = [
            "i32 7",
            "<4 x float> %0",
            "ptr addrspace(1) %v",
            "{ float, i32 } zeroinitializer",
            "[3 x i16] undef",
            "half 0xH3C00",
            "<2 x <4 x i8>> %x",
        ];
        // a small fixed alphabet of structurally-interesting bytes
        let alphabet = b"<>[]{}(),% x0123456789ifloat\"";
        for seed in seeds {
            let bytes = seed.as_bytes();
            for pos in 0..bytes.len() {
                for &b in alphabet {
                    // substitution
                    let mut m = bytes.to_vec();
                    m[pos] = b;
                    if let Ok(s) = std::str::from_utf8(&m) {
                        let _ = parse_type(s);
                        let _ = parse_typed_value(s);
                    }
                    // deletion
                    let mut d = bytes.to_vec();
                    d.remove(pos);
                    if let Ok(s) = std::str::from_utf8(&d) {
                        let _ = parse_type(s);
                        let _ = parse_typed_value(s);
                    }
                    // insertion
                    let mut ins = bytes.to_vec();
                    ins.insert(pos, b);
                    if let Ok(s) = std::str::from_utf8(&ins) {
                        let _ = parse_type(s);
                        let _ = parse_typed_value(s);
                    }
                }
            }
        }
    }
}

/// The right-hand side of a `%r = <rhs>` instruction (trimmed), or the whole line if it has no `=`.
pub(super) fn rhs_of(line: &str) -> &str {
    line.split_once('=')
        .map(|(_, rhs)| rhs.trim())
        .unwrap_or(line)
}

/// The parsed `LlCall` for a direct `call`/`[must|no]tail call` line — mirrors the `resolve_call_operands`
/// opcode dispatch (drop the `call` keyword, then hand the `<ret> @callee(args)` remainder to
/// `parse_call`). `None` for a non-call line or an indirect call (`parse_call` rejects the missing
/// `@callee`); byte-identical to the emitter's own `parse_call` by construction.
pub(super) fn resolve_call(line: &str) -> Option<LlCall> {
    let rhs = rhs_of(line);
    let opcode = rhs.split_whitespace().next()?;
    let after_opcode = rhs[opcode.len()..].trim_start();
    let after_call = match opcode {
        "call" => after_opcode,
        "tail" | "musttail" | "notail" => after_opcode.strip_prefix("call ")?.trim_start(),
        _ => return None,
    };
    parse_call(after_call).ok()
}
