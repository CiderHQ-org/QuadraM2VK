use std::collections::{HashMap, HashSet};

/// The Metal slot an argument node declares, resolving a function-constant slot through the
/// module's static initializers.
///
/// The slot is the FIRST operand of the `air.location_index` pair; `fallback` (the parameter
/// ordinal) stands in when it is a function-constant global this module does not initialize.
pub(super) fn location_index_with_static(
    body: &str,
    fallback: u32,
    static_int_globals: &HashMap<String, u32>,
) -> u32 {
    super::declared_slot(body, "air.location_index", fallback, static_int_globals)
}

fn parse_global_name(s: &str) -> Option<String> {
    symbol_references(s)
        .find(|symbol| symbol.starts_with('@'))
        .map(str::to_string)
}

#[derive(Clone, Debug)]
enum StaticValue {
    Bool(bool),
    Int(u64),
    Vector(Vec<u64>),
}

/// Best-effort evaluator for AIR static initializers that materialize default function-constant
/// integer globals. Unknown expressions are ignored so dynamic metadata falls back to its immediate
/// location index rather than guessing.
pub(super) fn static_init_int_global_values(ll: &str) -> HashMap<String, u32> {
    static_init_global_values(ll)
        .into_iter()
        .filter_map(|(global, value)| match value {
            StaticValue::Bool(value) => Some((global, u32::from(value))),
            StaticValue::Int(value) => Some((global, value as u32)),
            StaticValue::Vector(_) => None,
        })
        .collect()
}

fn static_init_global_values(ll: &str) -> HashMap<String, StaticValue> {
    let mut globals = parse_static_global_initializers(ll);
    let definedness = ll
        .lines()
        .filter_map(|line| {
            let defined = super::function_constant_definedness(line)?;
            Some((line.trim().split_once(" = ")?.0, defined))
        })
        .collect::<HashMap<_, _>>();
    let mut unknown_stores = HashSet::new();
    let mut env: HashMap<String, StaticValue> = HashMap::new();
    let mut in_static_init = false;

    for raw in ll.lines() {
        let line = raw.split(';').next().unwrap_or(raw).trim();
        if line.starts_with("define ") {
            in_static_init = crate::air::text::define_line_declares_static_initializer(line);
            env.clear();
            continue;
        }
        if !in_static_init {
            continue;
        }
        if line == "}" {
            in_static_init = false;
            continue;
        }
        if line.is_empty() || line.ends_with(':') || line.starts_with("switch ") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("store ") {
            let mut parts = rest.splitn(2, ',');
            let Some(value) = parts.next() else { continue };
            let Some(ptr) = parts.next() else { continue };
            let Some(global) = parse_global_name(ptr) else {
                continue;
            };
            if let Some(evaluated) = eval_value_token(value_token(value), &env, &globals) {
                unknown_stores.remove(&global);
                // Narrow to the STORE's own operand type. A fixed 32-bit mask clipped every
                // `store i64` to its low half, which is how a 64-bit function constant's high bits
                // -- the bit an `lshr i64 %x, 63` predicate reads -- were lost.
                let mask = integer_result_width_and_mask(value)
                    .map_or(u64::from(u32::MAX), |(_, mask)| mask);
                globals.insert(
                    global,
                    match evaluated {
                        StaticValue::Int(evaluated) => StaticValue::Int(evaluated & mask),
                        evaluated => evaluated,
                    },
                );
            } else {
                globals.remove(&global);
                unknown_stores.insert(global);
            }
            continue;
        }
        let Some((name, rhs)) = line.split_once(" = ") else {
            continue;
        };
        let name = name.trim();
        if !name.starts_with('%') {
            continue;
        }
        if let Some(value) = eval_static_rhs(rhs.trim(), &env, &globals, &definedness) {
            env.insert(name.to_string(), value);
        }
    }

    globals.retain(|global, _| !unknown_stores.contains(global));
    globals
}

/// Borrowed LLVM symbol spellings, scanned once without searching for every candidate.
fn symbol_references(text: &str) -> impl Iterator<Item = &str> {
    let bytes = text.as_bytes();
    let mut cursor = 0;
    std::iter::from_fn(move || {
        while cursor < bytes.len() {
            let start = cursor;
            let symbol = matches!(bytes[cursor], b'@' | b'%');
            if symbol {
                cursor += 1;
            }
            if bytes.get(cursor) == Some(&b'"') {
                cursor += 1;
                while cursor < bytes.len() {
                    let byte = bytes[cursor];
                    cursor += 1;
                    if byte == b'\\' && cursor < bytes.len() {
                        cursor += 1;
                    } else if byte == b'"' {
                        break;
                    }
                }
            } else if symbol {
                while bytes.get(cursor).is_some_and(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'$' | b'-')
                }) {
                    cursor += 1;
                }
            } else {
                cursor += 1;
            }
            if symbol && cursor > start + 1 {
                return Some(&text[start..cursor]);
            }
        }
        None
    })
}

fn parse_static_global_initializers(ll: &str) -> HashMap<String, StaticValue> {
    let mut globals = HashMap::new();
    for raw in ll.lines() {
        let line = raw.split(';').next().unwrap_or(raw).trim();
        if !line.starts_with('@') || !(line.contains(" global ") || line.contains(" constant ")) {
            continue;
        }
        let Some((name, rest)) = line.split_once(" = ") else {
            continue;
        };
        let Some(value) = integer_initializer(rest) else {
            continue;
        };
        globals.insert(name.trim().to_string(), value);
    }
    globals
}

fn integer_initializer(rest: &str) -> Option<StaticValue> {
    let typed_init = rest
        .split_once(" global ")
        .map(|(_, init)| init)
        .or_else(|| rest.split_once(" constant ").map(|(_, init)| init))?;
    if typed_init.starts_with('<') {
        let type_end = typed_init.find('>')?;
        let vector_ty = &typed_init[1..type_end];
        let (lanes, element_ty) = vector_ty.split_once(" x ")?;
        let lanes = lanes.parse::<usize>().ok()?;
        let width = element_ty.strip_prefix('i')?.parse::<u32>().ok()?;
        if width == 0 || width > 32 {
            return None;
        }
        let vector = typed_init[type_end + 1..].trim_start();
        let vector = if vector.starts_with('<') {
            &vector[..=vector.find('>')?]
        } else {
            vector.split(',').next().unwrap_or(vector).trim()
        };
        if vector == "zeroinitializer" {
            return Some(StaticValue::Vector(vec![0; lanes]));
        }
        let values = vector.strip_prefix('<')?.strip_suffix('>')?;
        let mask = if width == 32 {
            u64::from(u32::MAX)
        } else {
            (1_u64 << width) - 1
        };
        let values = values
            .split(',')
            .map(|lane| {
                let (ty, value) = lane.trim().split_once(' ')?;
                (ty == element_ty)
                    .then(|| {
                        value
                            .parse::<u64>()
                            .or_else(|_| value.parse::<i64>().map(|value| value as u64))
                            .ok()
                    })
                    .flatten()
                    .map(|value| value & mask)
            })
            .collect::<Option<Vec<_>>>()?;
        return (values.len() == lanes).then_some(StaticValue::Vector(values));
    }
    let mut tokens = typed_init.split_whitespace();
    let ty = tokens.next()?;
    let width = ty.strip_prefix('i')?.parse::<u32>().ok()?;
    if !matches!(width, 8 | 16 | 32 | 64) {
        return None;
    }
    let (_, mask) = integer_result_width_and_mask(ty)?;
    let value = tokens.next()?.trim_end_matches(',');
    value
        .parse::<u64>()
        .or_else(|_| value.parse::<i64>().map(|value| value as u64))
        .ok()
        .map(|value| StaticValue::Int(value & mask))
}

/// The integer binary opcodes the static-initializer evaluator folds. Every one is matched with the
/// separating space, so no entry can be read out of the start of another and the order here carries
/// no meaning.
const INTEGER_BINARY_OPCODES: &[&str] = &[
    "add", "sub", "mul", "and", "or", "xor", "udiv", "urem", "sdiv", "srem", "shl", "lshr", "ashr",
];

/// `value` read as a SIGNED integer of `width` bits. The evaluator carries every integer as a
/// masked `u64`, so a signed comparison, division or arithmetic shift has to recover the sign bit
/// from the operand's own width rather than from bit 63 of the carrier.
fn signed_of(value: u64, width: u32) -> i64 {
    let shift = 64 - width.min(64);
    ((value << shift) as i64) >> shift
}

fn eval_static_rhs(
    rhs: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
    definedness: &HashMap<&str, bool>,
) -> Option<StaticValue> {
    if let Some((_, arguments)) = rhs.split_once("@air.is_function_constant_defined(") {
        let global = parse_global_name(arguments)?;
        return definedness
            .get(global.as_str())
            .copied()
            .map(StaticValue::Bool);
    }
    // `llvm.umax`/`umin`/`smax`/`smin` are how the front end spells a clamped function-constant
    // expression, and 106 of them appear in the local corpus's static initializers.
    if let Some((name, rest)) = ["umax", "umin", "smax", "smin"]
        .into_iter()
        .find_map(|name| {
            rhs.split_once(&format!("@llvm.{name}."))
                .map(|(_, rest)| (name, rest))
        })
    {
        let arguments = rest.split_once('(')?.1.rsplit_once(')')?.0;
        let (lhs, rhs) = eval_binary_int(arguments, env, globals)?;
        let (width, mask) = integer_result_width_and_mask(arguments)?;
        let (lhs, rhs) = (lhs & mask, rhs & mask);
        let value = match name {
            "umax" => lhs.max(rhs),
            "umin" => lhs.min(rhs),
            "smax" => signed_of(lhs, width).max(signed_of(rhs, width)) as u64,
            "smin" => signed_of(lhs, width).min(signed_of(rhs, width)) as u64,
            _ => unreachable!("matched min/max intrinsic"),
        };
        return Some(StaticValue::Int(value & mask));
    }
    if rhs.contains("@air.normalize_function_constant_predicate.") {
        let arguments = rhs.split_once('(')?.1.rsplit_once(')')?.0;
        let value = eval_int_operand(arguments, env, globals)?;
        return Some(StaticValue::Int(u64::from(value != 0)));
    }
    if rhs.starts_with("load ") {
        let global = parse_global_name(rhs)?;
        return globals.get(&global).cloned();
    }
    if let Some(rest) = rhs.strip_prefix("extractelement ") {
        let parts = rest.split(',').collect::<Vec<_>>();
        let vector = eval_value_token(parts.first()?.split_whitespace().last()?, env, globals)?;
        let idx = eval_int_operand(parts.get(1)?, env, globals)? as usize;
        let StaticValue::Vector(values) = vector else {
            return None;
        };
        return values.get(idx).copied().map(StaticValue::Int);
    }
    if let Some((opcode, rest)) = INTEGER_BINARY_OPCODES.iter().find_map(|opcode| {
        rhs.strip_prefix(opcode)
            .and_then(|rest| rest.strip_prefix(' '))
            .map(|rest| (*opcode, rest))
    }) {
        let (lhs, rhs) = eval_binary_int(rest, env, globals)?;
        let (width, mask) = integer_result_width_and_mask(rest)?;
        let (lhs, rhs) = (lhs & mask, rhs & mask);
        let value = match opcode {
            "add" => lhs.wrapping_add(rhs),
            "sub" => lhs.wrapping_sub(rhs),
            "mul" => lhs.wrapping_mul(rhs),
            "and" => lhs & rhs,
            "or" => lhs | rhs,
            "xor" => lhs ^ rhs,
            "udiv" => lhs.checked_div(rhs)?,
            "urem" => lhs.checked_rem(rhs)?,
            "sdiv" => signed_of(lhs, width).checked_div(signed_of(rhs, width))? as u64,
            "srem" => signed_of(lhs, width).checked_rem(signed_of(rhs, width))? as u64,
            "shl" if rhs < u64::from(width) => lhs.checked_shl(rhs.try_into().ok()?)?,
            "lshr" if rhs < u64::from(width) => lhs.checked_shr(rhs.try_into().ok()?)?,
            // Arithmetic shift right is the one shift that reads the operand as SIGNED, so the
            // sign bit has to be recovered from the operand width before shifting rather than from
            // the 64-bit carrier the value is stored in.
            "ashr" if rhs < u64::from(width) => (signed_of(lhs, width) >> rhs.min(63)) as u64,
            "shl" | "lshr" | "ashr" => return None,
            _ => unreachable!("matched integer opcode"),
        };
        return Some(StaticValue::Int(value & mask));
    }
    if let Some(rest) = rhs.strip_prefix("icmp ") {
        let (predicate, operands) = rest.split_once(' ')?;
        let (lhs, rhs) = eval_binary_int(operands, env, globals)?;
        // The comparison is on the OPERAND type's width: a signed predicate reads the same bits as
        // a negative number, and an unsigned one must not see carrier bits above the operand.
        // Reading only `eq`/`ne` left every ordered comparison unevaluated -- 648 of the 10938
        // `icmp`s in the local corpus's static initializers -- and an unevaluated predicate is
        // reported as "not enabled by default", which is a GUESS the callers spend as a fact.
        let (width, mask) = integer_result_width_and_mask(operands)?;
        let (lhs, rhs) = (lhs & mask, rhs & mask);
        let value = match predicate {
            "eq" => lhs == rhs,
            "ne" => lhs != rhs,
            "ugt" => lhs > rhs,
            "uge" => lhs >= rhs,
            "ult" => lhs < rhs,
            "ule" => lhs <= rhs,
            "sgt" => signed_of(lhs, width) > signed_of(rhs, width),
            "sge" => signed_of(lhs, width) >= signed_of(rhs, width),
            "slt" => signed_of(lhs, width) < signed_of(rhs, width),
            "sle" => signed_of(lhs, width) <= signed_of(rhs, width),
            _ => return None,
        };
        return Some(StaticValue::Bool(value));
    }
    if let Some(rest) = rhs.strip_prefix("select ") {
        let parts = rest.split(',').collect::<Vec<_>>();
        let cond = eval_bool_operand(parts.first()?, env, globals)?;
        let chosen = if cond { parts.get(1)? } else { parts.get(2)? };
        return eval_value_token(value_token(chosen), env, globals);
    }
    if let Some(rest) = rhs
        .strip_prefix("trunc ")
        .or_else(|| rhs.strip_prefix("zext "))
    {
        let (value, to_ty) = rest.split_once(" to ")?;
        let int = eval_int_operand(value, env, globals)?;
        // Narrow to the DESTINATION width, whatever it is. Masking only `i8` and `i16` let a
        // `trunc i64 ... to i32` keep the high half it exists to discard.
        let (_, mask) = integer_result_width_and_mask(to_ty)?;
        return Some(StaticValue::Int(int & mask));
    }
    if rhs.contains("function_constant_predicate") {
        let open = rhs.rfind('(')?;
        let close = rhs.rfind(')')?;
        return Some(StaticValue::Int(eval_int_operand(
            &rhs[open + 1..close],
            env,
            globals,
        )?));
    }
    None
}

fn eval_binary_int(
    rest: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<(u64, u64)> {
    let parts = rest.split(',').collect::<Vec<_>>();
    Some((
        eval_int_operand(parts.first()?, env, globals)?,
        eval_int_operand(parts.get(1)?, env, globals)?,
    ))
}

fn integer_result_width_and_mask(text: &str) -> Option<(u32, u64)> {
    let width = text
        .split(|c: char| c.is_whitespace() || c == ',')
        .find_map(|token| token.strip_prefix('i')?.parse::<u32>().ok())?;
    match width {
        1..=63 => Some((width, (1_u64 << width) - 1)),
        64 => Some((width, u64::MAX)),
        _ => None,
    }
}

fn eval_int_operand(
    text: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<u64> {
    match eval_value_token(value_token(text), env, globals)? {
        StaticValue::Bool(value) => Some(u64::from(value)),
        StaticValue::Int(value) => Some(value),
        StaticValue::Vector(_) => None,
    }
}

fn eval_bool_operand(
    text: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<bool> {
    match eval_value_token(value_token(text), env, globals)? {
        StaticValue::Bool(value) => Some(value),
        StaticValue::Int(value) => Some(value != 0),
        StaticValue::Vector(_) => None,
    }
}

fn eval_value_token(
    token: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<StaticValue> {
    let token = token.trim().trim_end_matches(',');
    if token == "true" {
        return Some(StaticValue::Bool(true));
    }
    if token == "false" {
        return Some(StaticValue::Bool(false));
    }
    if token.starts_with('%') {
        return env.get(token).cloned();
    }
    if token.starts_with('@') {
        return globals.get(token).cloned();
    }
    token
        .parse::<u64>()
        .or_else(|_| token.parse::<i64>().map(|value| value as u64))
        .ok()
        .map(StaticValue::Int)
}

fn value_token(text: &str) -> &str {
    text.split_whitespace()
        .last()
        .unwrap_or_else(|| text.trim())
        .trim_end_matches(',')
}

#[cfg(test)]
mod tests;
