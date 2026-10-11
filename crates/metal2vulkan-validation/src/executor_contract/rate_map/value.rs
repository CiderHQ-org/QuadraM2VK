//! The value an operand's text states: what follows its type and attributes, read by the module
//! text's grammar rather than by its last token, so a type named like a value (`%0 = type { i32 }`)
//! inside a constant expression, a quoted name holding spaces or `;`, and an attribute's argument
//! (`align 8`) are never taken for the value.

use super::layout;

/// What an operand's value is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Value<'a> {
    /// A local value, `%` included (`%name` or `%"name"`).
    Local(&'a str),
    /// A global, without its `@` (quotes removed).
    Global(&'a str),
    /// `null`, `undef`, `poison`, `zeroinitializer`, or `none`: no object the module names.
    Empty,
    /// A constant expression: its opcode, and the text inside its parentheses.
    Expression { opcode: &'a str, operands: &'a str },
    /// Anything else: a number, a boolean, or an aggregate or vector constant.
    Literal,
}

/// The opcodes of LLVM's constant expressions.
const EXPRESSIONS: [&str; 20] = [
    "getelementptr",
    "bitcast",
    "addrspacecast",
    "inttoptr",
    "ptrtoint",
    "trunc",
    "zext",
    "sext",
    "add",
    "sub",
    "mul",
    "shl",
    "xor",
    "select",
    "icmp",
    "extractelement",
    "insertelement",
    "shufflevector",
    "blockaddress",
    "dso_local_equivalent",
];

/// The constants that name no object.
const EMPTY: [&str; 5] = ["null", "undef", "poison", "zeroinitializer", "none"];

/// The value an operand (`[<type>] [attributes] <value> [ordering]`) states, its text: a name,
/// a constant keyword or literal, or a constant expression through its parentheses. Empty when
/// the operand states none.
pub(super) fn value_text(operand: &str) -> &str {
    let operand = operand.trim();
    // A typed operand has text after its type; a bare value (`%v`, which also reads as a named
    // type) has none.
    let mut text = match layout::after_type(operand) {
        Some(rest) if rest.starts_with(char::is_whitespace) && !rest.trim().is_empty() => {
            rest.trim_start()
        }
        _ => operand,
    };
    while !text.is_empty() {
        let (token, after) = next_token(text);
        if EXPRESSIONS.contains(&token) {
            // The expression's flags, then its parenthesized operands.
            let mut rest = after.trim_start();
            while !rest.is_empty() && !rest.starts_with('(') {
                rest = next_token(rest).1.trim_start();
            }
            let end = text.len() - rest.len() + next_token(rest).0.len();
            return &text[..end];
        }
        if starts_value(token) {
            return token;
        }
        // An attribute: `align` takes the integer after it.
        text = after.trim_start();
        if token == "align" {
            text = next_token(text).1.trim_start();
        }
    }
    ""
}

/// What an operand's value is ([`value_text`]).
pub(super) fn value(operand: &str) -> Value<'_> {
    let text = value_text(operand);
    if text.starts_with('%') {
        return Value::Local(text);
    }
    if let Some(name) = text.strip_prefix('@') {
        return Value::Global(name.trim_matches('"'));
    }
    if EMPTY.contains(&text) {
        return Value::Empty;
    }
    let (opcode, mut rest) = next_token(text);
    if EXPRESSIONS.contains(&opcode) {
        // Past its flags (`inbounds`, `inrange(-8, 8)`), its parenthesized operands.
        while !rest.trim().is_empty() {
            let (token, after) = next_token(rest.trim_start());
            if let Some(operands) = token.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
                return Value::Expression { opcode, operands };
            }
            rest = after;
        }
    }
    Value::Literal
}

/// Whether a token starts an operand's value rather than an attribute.
fn starts_value(token: &str) -> bool {
    let first = token.chars().next().unwrap_or(' ');
    matches!(first, '%' | '@' | '{' | '[' | '<')
        || first.is_ascii_digit()
        || (first == '-' && token[1..].starts_with(|ch: char| ch.is_ascii_digit()))
        || token.starts_with("c\"")
        || EMPTY.contains(&token)
        || matches!(token, "true" | "false")
}

/// The token `text` starts with and the text after it: a quoted string, a name (quoted or not),
/// or a word with the brackets that follow it without a space (`addrspace(1)`,
/// `captures(none)`), each bracket and quote kept whole.
pub(super) fn next_token(text: &str) -> (&str, &str) {
    let mut depth = 0usize;
    let mut quoted = false;
    for (index, ch) in text.char_indices() {
        match ch {
            '"' => quoted = !quoted,
            _ if quoted => {}
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth = depth.saturating_sub(1),
            _ if ch.is_whitespace() && depth == 0 && index > 0 => {
                return (&text[..index], &text[index..]);
            }
            _ => {}
        }
    }
    (text, "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_operand_s_value_follows_its_type_and_attributes() {
        for (operand, value) in [
            ("ptr addrspace(1) %p", "%p"),
            ("%p", "%p"),
            ("@g", "@g"),
            ("ptr noundef align 8 %p", "%p"),
            (
                "ptr addrspace(1) readonly captures(none) \"air-buffer-no-alias\" %1",
                "%1",
            ),
            ("ptr addrspace(1) %p unordered", "%p"),
            ("ptr %\"a b;c\"", "%\"a b;c\""),
            ("ptr @\"g h\"", "@\"g h\""),
            ("%struct.T zeroinitializer", "zeroinitializer"),
            ("%0 %1", "%1"),
            ("i64 4", "4"),
            ("i64 -4", "-4"),
            (
                "ptr addrspace(2) getelementptr inbounds (%0, ptr addrspace(2) @g, i64 0, i32 \
                 0)",
                "getelementptr inbounds (%0, ptr addrspace(2) @g, i64 0, i32 0)",
            ),
            (
                "ptr addrspacecast (ptr addrspace(3) @t to ptr)",
                "addrspacecast (ptr addrspace(3) @t to ptr)",
            ),
            ("ptr null", "null"),
        ] {
            assert_eq!(value_text(operand), value, "{operand}");
        }
        assert_eq!(
            value("ptr getelementptr (%0, ptr @g, i64 1)"),
            Value::Expression {
                opcode: "getelementptr",
                operands: "%0, ptr @g, i64 1"
            }
        );
        assert_eq!(value("ptr @\"g h\""), Value::Global("g h"));
        assert_eq!(value("ptr poison"), Value::Empty);
        assert_eq!(value("i1 true"), Value::Literal);
    }
}
