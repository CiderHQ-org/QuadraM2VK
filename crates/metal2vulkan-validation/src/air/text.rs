//! LLVM IR text helpers of validation's AIR readers (`meta`, the call inventory),
//! moved from the product with them: top-level splitting, comments, delimiters, and what marks an
//! AIR static initializer (the `air.static_init` section, the AIR ABI's own statement of what the
//! function is, never its Itanium name).

pub(crate) fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut out = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            continue;
        }
        match c {
            '<' | '[' | '{' | '(' => depth += 1,
            '>' | ']' | '}' | ')' => depth -= 1,
            _ if c == sep && depth == 0 => {
                out.push(s[start..i].trim());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

pub(crate) fn split_top_level_whitespace(s: &str) -> Vec<&str> {
    let mut depth = 0i32;
    let mut start = None;
    let mut out = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            if start.is_none() {
                start = Some(i);
            }
            in_string = true;
            continue;
        }
        match c {
            '<' | '[' | '{' | '(' => {
                if start.is_none() {
                    start = Some(i);
                }
                depth += 1;
            }
            '>' | ']' | '}' | ')' => depth -= 1,
            c if c.is_whitespace() && depth == 0 => {
                if let Some(st) = start.take() {
                    out.push(&s[st..i]);
                }
            }
            _ => {
                if start.is_none() {
                    start = Some(i);
                }
            }
        }
    }
    if let Some(st) = start {
        out.push(&s[st..]);
    }
    out
}

pub(crate) fn strip_comment(s: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            continue;
        }
        if c == ';' {
            return &s[..i];
        }
    }
    s
}

pub(crate) fn matching_paren(s: &str, open: usize) -> Option<usize> {
    matching_delim(s, open, '(', ')')
}

pub(crate) fn matching_delim(s: &str, open: usize, left: char, right: char) -> Option<usize> {
    if s[open..].chars().next()? != left {
        return None;
    }
    let mut depth = 0i32;
    for (i, c) in s.char_indices().skip_while(|(i, _)| *i < open) {
        if c == left {
            depth += 1;
        } else if c == right {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

pub(crate) fn parse_float_literal(s: &str) -> Result<f64, String> {
    let looks_float = s.contains('.') || s.contains('e') || s.contains('E');
    if !looks_float {
        return Err(format!("native emitter: not a decimal float literal `{s}`"));
    }
    s.parse()
        .map_err(|e| format!("native emitter: expected float `{s}`: {e}"))
}

pub(crate) fn parse_float32_hex_literal(s: &str) -> Option<u32> {
    let hex = s.strip_prefix("f0x")?;
    if hex.len() != 8 {
        return None;
    }
    u32::from_str_radix(hex, 16).ok()
}

pub(crate) fn parse_half_literal(s: &str) -> Option<u16> {
    let hex = s.strip_prefix("0xH")?;
    u16::from_str_radix(hex, 16).ok()
}

pub(crate) fn parse_bfloat_literal(s: &str) -> Option<u16> {
    let hex = s.strip_prefix("0xR")?;
    u16::from_str_radix(hex, 16).ok()
}

pub(crate) fn parse_hex_literal(s: &str) -> Option<u64> {
    let hex = s.strip_prefix("0x")?;
    u64::from_str_radix(hex, 16).ok()
}

pub(crate) fn parse_i64_literal(s: &str) -> Result<i64, String> {
    if !s.starts_with('-') {
        return Err(format!("native emitter: not a signed integer `{s}`"));
    }
    s.parse()
        .map_err(|e| format!("native emitter: expected signed integer `{s}`: {e}"))
}

pub(crate) fn parse_u64_literal(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
            .map_err(|e| format!("native emitter: expected hex integer `{s}`: {e}"))
    } else {
        s.parse()
            .map_err(|e| format!("native emitter: expected integer `{s}`: {e}"))
    }
}

/// Parse LLVM IR special float literals: `+qnan`, `-qnan`, `qnan`, `snan`, `-snan`,
/// `+inf`, `-inf`, `inf`, `nan`. Returns the IEEE 754 f32 bit pattern.
pub(crate) fn parse_float_special_literal(s: &str) -> Option<u32> {
    let s = s.trim();
    // Strip leading + if present
    let s = s.strip_prefix('+').unwrap_or(s);
    match s {
        "qnan" => Some(0x7FC0_0000),  // quiet NaN (positive)
        "-qnan" => Some(0xFFC0_0000), // quiet NaN (negative)
        "snan" => Some(0x7F80_0001),  // signaling NaN
        "-snan" => Some(0xFF80_0001), // signaling NaN (negative)
        "inf" => Some(0x7F80_0000),   // positive infinity
        "-inf" => Some(0xFF80_0000),  // negative infinity
        "nan" => Some(0x7FC0_0000),   // generic NaN
        "-nan" => Some(0xFFC0_0000),  // generic NaN (negative)
        _ => None,
    }
}

/// The section AIR places a translation unit's static initializer in.
pub(crate) const AIR_STATIC_INIT_SECTION: &str = "air.static_init";

/// Whether a `define` header line declares an AIR static initializer.
///
/// For readers that still work on IR text. The section attribute follows the parameter list, so the
/// search starts after it: a parameter default or a quoted symbol name is not an attribute, and
/// scanning the whole line would let one masquerade as the marker.
pub(crate) fn define_line_declares_static_initializer(line: &str) -> bool {
    attribute_tail(line).is_some_and(tail_declares_static_init_section)
}

/// Whether the attribute tail of a `define` header — everything after the parameter list — carries
/// the `air.static_init` section attribute.
pub(crate) fn tail_declares_static_init_section(tail: &str) -> bool {
    let mut rest = tail;
    while let Some(at) = rest.find("section ") {
        let after = rest[at + "section ".len()..].trim_start();
        if let Some(quoted) = after.strip_prefix('"') {
            if let Some(end) = quoted.find('"') {
                if &quoted[..end] == AIR_STATIC_INIT_SECTION {
                    return true;
                }
            }
        }
        rest = &rest[at + "section ".len()..];
    }
    false
}

/// The part of a `define` header line that follows the parameter list, or `None` if the line has no
/// balanced parameter list. Quoted spans are skipped so a `(` or `)` inside a quoted symbol name
/// cannot unbalance the scan.
fn attribute_tail(line: &str) -> Option<&str> {
    let mut depth = 0usize;
    let mut in_quotes = false;
    let mut escaped = false;
    for (offset, ch) in line.char_indices() {
        if in_quotes {
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_quotes = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_quotes = true,
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&line[offset + 1..]);
                }
            }
            _ => {}
        }
    }
    None
}
