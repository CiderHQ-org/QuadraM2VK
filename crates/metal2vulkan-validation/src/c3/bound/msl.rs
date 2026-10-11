//! Apple's accuracy promises (MSL 4.1 chapter 8) and what they admit for one result.
//!
//! [`row`] is Tables 8.1 (single precision), 8.2 (single precision under fast math), and 8.3
//! (half precision), and Table 6.1's `sign`, as typed rows: per operation an accuracy on a domain
//! of one operand, in the table's order, and the reason Apple states nothing elsewhere. An
//! operation Table 8.2 defines by its "Implemented as" column is a [`Formula`] over other rows.
//!
//! [`admits`] evaluates an [`Expr`] over its operands into what those rows admit ([`Admits`]): the
//! real intervals a conforming result may lie in and whether it may be a NaN, or
//! [`Admits::Unstated`] where Apple states no accuracy. Each operation's exact image is computed in
//! `f64` over its operands' intervals and widened by its row; section 8.1's flush of subnormal
//! operands and results and section 8.2's two rounding modes are applied at every operation, and
//! section 8.1's rule that a NaN or INF operand or result is undefined under fast math makes the
//! result unstated. A formula's operations are widened one by one, each over the interval its
//! operands span, so what a formula admits is a superset of what Apple's implementation can return:
//! wider by the spread of each part's bound over its operands' interval (a few ulps where the
//! operands are points or ulps apart). Section 8.4's reassociation under fast math is not used: an
//! expression is evaluated in the grouping it is written in.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The floating-point format a lane computes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    F32,
    F16,
}

/// `2^e` for `e` in `f64`'s normal range, exactly.
pub fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

impl Format {
    pub fn as_str(self) -> &'static str {
        match self {
            Format::F32 => "f32",
            Format::F16 => "f16",
        }
    }

    fn mantissa_bits(self) -> i32 {
        match self {
            Format::F32 => 23,
            Format::F16 => 10,
        }
    }

    fn min_exponent(self) -> i32 {
        match self {
            Format::F32 => -126,
            Format::F16 => -14,
        }
    }

    fn max_exponent(self) -> i32 {
        match self {
            Format::F32 => 127,
            Format::F16 => 15,
        }
    }

    pub fn min_normal(self) -> f64 {
        pow2(self.min_exponent())
    }

    pub fn max_finite(self) -> f64 {
        (2.0 - pow2(-self.mantissa_bits())) * pow2(self.max_exponent())
    }

    /// Whether `x` lies strictly between zero and the smallest normal magnitude.
    pub fn is_subnormal(self, x: f64) -> bool {
        x != 0.0 && x.abs() < self.min_normal()
    }

    /// The value of the format's bit pattern `bits` (its low 32 or 16 bits).
    pub fn decode(self, bits: u32) -> f64 {
        match self {
            Format::F32 => f64::from(f32::from_bits(bits)),
            Format::F16 => {
                let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
                let exponent = ((bits >> 10) & 0x1f) as i32;
                let mantissa = f64::from(bits & 0x3ff);
                match exponent {
                    0 => sign * mantissa * pow2(-24),
                    31 if mantissa == 0.0 => sign * f64::INFINITY,
                    31 => f64::NAN,
                    _ => sign * (1024.0 + mantissa) * pow2(exponent - 25),
                }
            }
        }
    }

    /// The ulp of `x` as MSL section 8.4 defines it: the distance between the two consecutive
    /// values of the format around `x`, or, where `x` is one of them, between `x` and the nearest
    /// other one (half the spacing above at a power of two). Past the largest finite value, the
    /// spacing below it.
    pub fn ulp(self, x: f64) -> f64 {
        let (exponent, spacing) = self.spacing(x);
        let power_of_two = x.abs() == pow2(exponent);
        if power_of_two && exponent > self.min_exponent() {
            spacing / 2.0
        } else {
            spacing
        }
    }

    /// The exponent of `x`'s binade, clamped to the format's, and the spacing of the format's
    /// values there.
    fn spacing(self, x: f64) -> (i32, f64) {
        let magnitude = x.abs();
        let exponent = if magnitude < self.min_normal() {
            self.min_exponent()
        } else {
            (((magnitude.to_bits() >> 52) & 0x7ff) as i32 - 1023)
                .clamp(self.min_exponent(), self.max_exponent())
        };
        (exponent, pow2(exponent - self.mantissa_bits()))
    }

    /// Whether `x` is one of the format's finite values.
    pub fn represents(self, x: f64) -> bool {
        x.is_finite() && x.abs() <= self.max_finite() && (x / self.spacing(x).1).fract() == 0.0
    }
}

/// Whether a lane's operations compile under fast math (Table 8.2, and section 8.1's rule for NaN
/// and INF) or without it (Table 8.1). Half-precision operations follow Table 8.3 either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Math {
    Fast,
    Precise,
}

impl Math {
    pub fn as_str(self) -> &'static str {
        match self {
            Math::Fast => "fast",
            Math::Precise => "precise",
        }
    }
}

/// An MSL operation, named as an expression spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Prim {
    Add,
    Sub,
    Mul,
    Div,
    Recip,
    Sign,
    Sqrt,
    Rsqrt,
    Exp,
    Exp2,
    Log,
    Log2,
    Sin,
    Cos,
    Tan,
    Asin,
    Atan,
    Atan2,
    Tanh,
    Acosh,
    Asinh,
    Atanh,
    Pow,
    Powr,
    Fmod,
    SinPi,
    CosPi,
    TanPi,
}

impl Prim {
    /// Every operation.
    pub const ALL: [Prim; 28] = [
        Prim::Add,
        Prim::Sub,
        Prim::Mul,
        Prim::Div,
        Prim::Recip,
        Prim::Sign,
        Prim::Sqrt,
        Prim::Rsqrt,
        Prim::Exp,
        Prim::Exp2,
        Prim::Log,
        Prim::Log2,
        Prim::Sin,
        Prim::Cos,
        Prim::Tan,
        Prim::Asin,
        Prim::Atan,
        Prim::Atan2,
        Prim::Tanh,
        Prim::Acosh,
        Prim::Asinh,
        Prim::Atanh,
        Prim::Pow,
        Prim::Powr,
        Prim::Fmod,
        Prim::SinPi,
        Prim::CosPi,
        Prim::TanPi,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Prim::Add => "add",
            Prim::Sub => "sub",
            Prim::Mul => "mul",
            Prim::Div => "div",
            Prim::Recip => "recip",
            Prim::Sign => "sign",
            Prim::Sqrt => "sqrt",
            Prim::Rsqrt => "rsqrt",
            Prim::Exp => "exp",
            Prim::Exp2 => "exp2",
            Prim::Log => "log",
            Prim::Log2 => "log2",
            Prim::Sin => "sin",
            Prim::Cos => "cos",
            Prim::Tan => "tan",
            Prim::Asin => "asin",
            Prim::Atan => "atan",
            Prim::Atan2 => "atan2",
            Prim::Tanh => "tanh",
            Prim::Acosh => "acosh",
            Prim::Asinh => "asinh",
            Prim::Atanh => "atanh",
            Prim::Pow => "pow",
            Prim::Powr => "powr",
            Prim::Fmod => "fmod",
            Prim::SinPi => "sinpi",
            Prim::CosPi => "cospi",
            Prim::TanPi => "tanpi",
        }
    }

    fn named(name: &str) -> Option<Prim> {
        Prim::ALL.into_iter().find(|prim| prim.name() == name)
    }

    pub fn arity(self) -> usize {
        match self {
            Prim::Add
            | Prim::Sub
            | Prim::Mul
            | Prim::Div
            | Prim::Atan2
            | Prim::Pow
            | Prim::Powr
            | Prim::Fmod => 2,
            _ => 1,
        }
    }

    /// The infinitely precise result at a point, in `f64` (`atan2(y, x)` takes `y` first, as
    /// MSL does; `sign` is Table 6.1's).
    fn exact(self, args: &[f64]) -> f64 {
        let a = args[0];
        let b = args.get(1).copied().unwrap_or(f64::NAN);
        match self {
            Prim::Add => a + b,
            Prim::Sub => a - b,
            Prim::Mul => a * b,
            Prim::Div => a / b,
            Prim::Recip => 1.0 / a,
            Prim::Sign if a.is_nan() => 0.0,
            Prim::Sign if a == 0.0 => a,
            Prim::Sign => a.signum(),
            Prim::Sqrt => a.sqrt(),
            Prim::Rsqrt => 1.0 / a.sqrt(),
            Prim::Exp => a.exp(),
            Prim::Exp2 => a.exp2(),
            Prim::Log => a.ln(),
            Prim::Log2 => a.log2(),
            Prim::Sin => a.sin(),
            Prim::Cos => a.cos(),
            Prim::Tan => a.tan(),
            Prim::Asin => a.asin(),
            Prim::Atan => a.atan(),
            Prim::Atan2 => a.atan2(b),
            Prim::Tanh => a.tanh(),
            Prim::Acosh => a.acosh(),
            Prim::Asinh => a.asinh(),
            Prim::Atanh => a.atanh(),
            Prim::Pow => a.powf(b),
            Prim::Powr if a < 0.0 => f64::NAN,
            Prim::Powr => a.powf(b),
            Prim::Fmod => a % b,
            Prim::SinPi | Prim::CosPi | Prim::TanPi => half_turns(self, a),
        }
    }

    /// How the exact result moves with the one operand: `Some(true)` increasing, `Some(false)`
    /// decreasing on each side of a pole, `None` where only points are evaluated.
    fn monotone(self) -> Option<bool> {
        match self {
            Prim::Sqrt
            | Prim::Exp
            | Prim::Exp2
            | Prim::Log
            | Prim::Log2
            | Prim::Asin
            | Prim::Atan
            | Prim::Tanh
            | Prim::Acosh
            | Prim::Asinh
            | Prim::Atanh => Some(true),
            Prim::Rsqrt | Prim::Recip => Some(false),
            _ => None,
        }
    }
}

/// `sinpi`, `cospi`, or `tanpi` (`prim`) of `a` in `f64`, by the exact quarter turn `q` and
/// residue `u` in `[-1/4, 1/4]` of `a = q/2 + u` (mod 2), so that the zeros are exact; at a pole
/// `tanpi(n + 1/2)` is `+inf` for an even `n` and `-inf` for an odd one, as C23's `tanpi` and
/// Metal's answer. Every `float` and `half` is exact in `f64`, and so are the steps.
fn half_turns(prim: Prim, a: f64) -> f64 {
    if !a.is_finite() {
        return f64::NAN;
    }
    let r = a - 2.0 * (a / 2.0).round_ties_even();
    let q = (2.0 * r).round_ties_even();
    let u = r - q / 2.0;
    let (su, cu) = (
        (std::f64::consts::PI * u).sin(),
        (std::f64::consts::PI * u).cos(),
    );
    let (sin, cos) = match q as i64 {
        0 => (su, cu),
        1 => (cu, -su),
        -1 => (-cu, su),
        _ => (-su, -cu),
    };
    match prim {
        Prim::SinPi => sin,
        Prim::CosPi => cos,
        _ if u == 0.0 && q.rem_euclid(2.0) == 1.0 => {
            if (a - 0.5).rem_euclid(2.0) == 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        }
        _ => sin / cos,
    }
}

/// An expression over a lane's operands: `a` is the first, `b` the second, ...; a call is
/// `name(argument, ...)` with an MSL operation's name ([`Prim::name`]); a number is a constant.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Operand(usize),
    Const(f64),
    Call(Prim, Vec<Expr>),
}

impl Expr {
    /// Parse `text` (for example `sub(a, mul(b, add(mul(sign(a), c), d)))`).
    pub fn parse(text: &str) -> Result<Expr, String> {
        let mut parser = Parser { text, at: 0 };
        let expr = parser.expr()?;
        parser.skip_space();
        if parser.at != text.len() {
            return Err(format!("{text:?}: unexpected {:?}", &text[parser.at..]));
        }
        Ok(expr)
    }

    /// One more than the largest operand the expression reads.
    pub fn operands(&self) -> usize {
        match self {
            Expr::Operand(index) => index + 1,
            Expr::Const(_) => 0,
            Expr::Call(_, args) => args.iter().map(Expr::operands).max().unwrap_or(0),
        }
    }

    /// Every operation the expression calls, in order of first appearance.
    pub fn calls(&self, into: &mut Vec<Prim>) {
        if let Expr::Call(prim, args) = self {
            if !into.contains(prim) {
                into.push(*prim);
            }
            for arg in args {
                arg.calls(into);
            }
        }
    }

    /// The infinitely precise value at `operands`, in `f64`: no row's error, no formula.
    pub fn reference(&self, operands: &[f64]) -> f64 {
        self.exact_values(operands, &mut vec![])
    }

    /// [`Expr::reference`], also pushing onto `into` the infinitely precise value of every
    /// operand, constant, and operation the expression reads, its result last.
    pub fn exact_values(&self, operands: &[f64], into: &mut Vec<f64>) -> f64 {
        let value = match self {
            Expr::Operand(index) => operands[*index],
            Expr::Const(value) => *value,
            Expr::Call(prim, args) => prim.exact(
                &args
                    .iter()
                    .map(|arg| arg.exact_values(operands, into))
                    .collect::<Vec<_>>(),
            ),
        };
        into.push(value);
        value
    }
}

struct Parser<'a> {
    text: &'a str,
    at: usize,
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        while self.text[self.at..].starts_with(' ') {
            self.at += 1;
        }
    }

    fn take(&mut self, what: char) -> bool {
        self.skip_space();
        if self.text[self.at..].starts_with(what) {
            self.at += what.len_utf8();
            true
        } else {
            false
        }
    }

    fn token(&mut self) -> &str {
        self.skip_space();
        let rest = &self.text[self.at..];
        let length = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '+')))
            .unwrap_or(rest.len());
        self.at += length;
        &rest[..length]
    }

    fn expr(&mut self) -> Result<Expr, String> {
        let token = self.token().to_string();
        let mut chars = token.chars();
        match (chars.next(), chars.next()) {
            (None, _) => Err(format!(
                "{:?}: expected an operand, a number, or a call",
                self.text
            )),
            (Some(c), None) if c.is_ascii_lowercase() => {
                Ok(Expr::Operand(c as usize - 'a' as usize))
            }
            (Some(c), _) if c.is_ascii_digit() || c == '-' => token
                .parse::<f64>()
                .map(Expr::Const)
                .map_err(|error| format!("{:?}: {token:?}: {error}", self.text)),
            _ => {
                let prim = Prim::named(&token)
                    .ok_or_else(|| format!("{:?}: no MSL operation {token:?}", self.text))?;
                if !self.take('(') {
                    return Err(format!("{:?}: {token} needs its arguments", self.text));
                }
                let mut args = vec![self.expr()?];
                while self.take(',') {
                    args.push(self.expr()?);
                }
                if !self.take(')') {
                    return Err(format!("{:?}: {token}( is not closed", self.text));
                }
                if args.len() != prim.arity() {
                    return Err(format!(
                        "{:?}: {token} takes {} arguments, not {}",
                        self.text,
                        prim.arity(),
                        args.len()
                    ));
                }
                Ok(Expr::Call(prim, args))
            }
        }
    }
}

/// One of Apple's accuracies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Accuracy {
    /// 0 ulp: the exact result.
    Exact,
    /// "Correctly rounded": under either rounding mode section 8.2 allows, round to nearest even
    /// or toward zero.
    CorrectlyRounded,
    /// Within this many ulps of the infinitely precise result.
    Ulp(f64),
    /// `exp` and `exp2` under fast math: within `3 + floor(fabs(2 * x))` ulp.
    UlpGrowing,
    /// Within `2^n` absolute.
    Absolute(i32),
    /// Table 8.2's "Implemented as" column.
    Formula(Formula),
}

impl fmt::Display for Accuracy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Accuracy::Exact => write!(f, "0 ulp"),
            Accuracy::CorrectlyRounded => write!(f, "correctly rounded"),
            Accuracy::Ulp(ulps) => write!(f, "<= {ulps} ulp"),
            Accuracy::UlpGrowing => write!(f, "<= 3 + floor(fabs(2 * x)) ulp"),
            Accuracy::Absolute(n) => write!(f, "<= 2^{n} absolute"),
            Accuracy::Formula(formula) => write!(f, "implemented as {}", formula.text()),
        }
    }
}

/// An operation Table 8.2 implements as a formula over other operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Formula {
    Tan,
    TanPi,
    Tanh,
    Sqrt,
    Pow,
    Acosh,
    Asinh,
    Atanh,
    Atan2,
}

impl Formula {
    pub fn text(self) -> &'static str {
        match self {
            Formula::Tan => "sin(x) * (1.0 / cos(x))",
            Formula::TanPi => "tan(x * pi)",
            Formula::Tanh => "(t - 1.0) / (t + 1.0), where t = exp(2.0 * x)",
            Formula::Sqrt => "x * rsqrt(x) with special cases handled correctly",
            Formula::Pow => "exp2(y * log2(x)), undefined for x = 0 and y = 0",
            Formula::Acosh => "log(x + sqrt(x * x - 1.0))",
            Formula::Asinh => "log(x + sqrt(x * x + 1.0))",
            Formula::Atanh => "0.5 * log((1.0 + x) / (1.0 - x))",
            Formula::Atan2 => {
                "atan(y / x) for x > 0, plus M_PI_F for x < 0 and y > 0, minus M_PI_F for x < 0 \
                 and y < 0, undefined if x = 0 or y = 0"
            }
        }
    }
}

/// A closed range of one operand, `lo..=hi`.
type Span = (f64, f64);

/// Where a piece of a row applies: ranges of one operand.
#[derive(Clone, Debug, PartialEq)]
pub struct Domain {
    pub operand: usize,
    pub ranges: Vec<Span>,
    pub text: &'static str,
}

impl Domain {
    fn everywhere() -> Domain {
        Domain {
            operand: 0,
            ranges: vec![(f64::NEG_INFINITY, f64::INFINITY)],
            text: "",
        }
    }

    fn of(operand: usize, ranges: Vec<Span>, text: &'static str) -> Domain {
        Domain {
            operand,
            ranges,
            text,
        }
    }
}

/// One accuracy on one domain.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub domain: Domain,
    pub accuracy: Accuracy,
}

/// One row of Apple's tables: its pieces in the table's order, and what the table says outside
/// them, where it states no accuracy.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub pieces: Vec<Piece>,
    pub otherwise: &'static str,
    pub source: &'static str,
}

impl Row {
    fn one(accuracy: Accuracy, source: &'static str) -> Row {
        Row {
            pieces: vec![Piece {
                domain: Domain::everywhere(),
                accuracy,
            }],
            otherwise: "",
            source,
        }
    }

    fn on(pieces: Vec<(Domain, Accuracy)>, otherwise: &'static str, source: &'static str) -> Row {
        Row {
            pieces: pieces
                .into_iter()
                .map(|(domain, accuracy)| Piece { domain, accuracy })
                .collect(),
            otherwise,
            source,
        }
    }

    /// The row as the table states it (`MSL Table 8.2: <= 5 ulp`).
    pub fn describe(&self) -> String {
        let pieces = self
            .pieces
            .iter()
            .map(|piece| {
                if piece.domain.text.is_empty() {
                    piece.accuracy.to_string()
                } else {
                    format!("{} {}", piece.accuracy, piece.domain.text)
                }
            })
            .collect::<Vec<_>>()
            .join("; ");
        if self.otherwise.is_empty() {
            format!("{}: {pieces}", self.source)
        } else {
            format!("{}: {pieces}; {}", self.source, self.otherwise)
        }
    }
}

const TABLE_8_1: &str = "MSL Table 8.1";
const TABLE_8_2: &str = "MSL Table 8.2";
const TABLE_8_3: &str = "MSL Table 8.3";
const TABLE_6_1: &str = "MSL Table 6.1";

/// The row of Apple's tables for `prim` computed in `format` under `math`, `None` where no table
/// has one (Table 8.3 states no `pow` or `powr`).
pub fn row(prim: Prim, format: Format, math: Math) -> Option<Row> {
    use Accuracy::*;
    use Prim::*;
    let magnitude = |operand, text| {
        Domain::of(
            operand,
            vec![(-pow2(126), -pow2(-126)), (pow2(-126), pow2(126))],
            text,
        )
    };
    let positive = (f64::from_bits(1), f64::INFINITY);
    if prim == Sign {
        return Some(Row::one(Exact, TABLE_6_1));
    }
    Some(match (format, math) {
        (Format::F16, _) => match prim {
            Add | Sub | Mul | Div | Recip | Sqrt | Rsqrt => Row::one(CorrectlyRounded, TABLE_8_3),
            Fmod => Row::one(Exact, TABLE_8_3),
            Pow | Powr => return None,
            _ => Row::one(Ulp(1.0), TABLE_8_3),
        },
        (Format::F32, Math::Precise) => Row::one(
            match prim {
                Add | Sub | Mul | Div | Recip | Sqrt | Rsqrt => CorrectlyRounded,
                Fmod | Sign => Exact,
                Asin | Acosh | Asinh | Sin | Cos | SinPi | CosPi | Exp | Exp2 | Log | Log2 => {
                    Ulp(4.0)
                }
                Atan | Atanh | Tanh => Ulp(5.0),
                Atan2 | Tan | TanPi => Ulp(6.0),
                Pow | Powr => Ulp(16.0),
            },
            TABLE_8_1,
        ),
        (Format::F32, Math::Fast) => match prim {
            Add | Sub | Mul => Row::one(CorrectlyRounded, TABLE_8_2),
            Recip => Row::on(
                vec![(
                    magnitude(0, "for x in the domain of 2^-126 to 2^126"),
                    Ulp(1.0),
                )],
                "no accuracy stated outside that domain",
                TABLE_8_2,
            ),
            Div => Row::on(
                vec![(
                    magnitude(1, "for y in the domain of 2^-126 to 2^126"),
                    Ulp(2.5),
                )],
                "no accuracy stated outside that domain",
                TABLE_8_2,
            ),
            Asin => Row::on(
                vec![(
                    Domain::of(
                        0,
                        vec![(-1.0, -pow2(-125)), (pow2(-125), 1.0)],
                        "for x in the domain [-1, 1] and |x| >= 2^-125",
                    ),
                    Ulp(5.0),
                )],
                "no accuracy stated outside that domain",
                TABLE_8_2,
            ),
            Atan => Row::one(Ulp(5.0), TABLE_8_2),
            Sin | Cos => Row::on(
                vec![(
                    Domain::of(
                        0,
                        vec![(-std::f64::consts::PI, std::f64::consts::PI)],
                        "for x in the domain [-pi, pi]",
                    ),
                    Absolute(-13),
                )],
                "larger otherwise",
                TABLE_8_2,
            ),
            SinPi | CosPi => Row::on(
                vec![(
                    Domain::of(0, vec![(-1.0, 1.0)], "for x in the domain [-1, 1]"),
                    Absolute(-13),
                )],
                "larger otherwise",
                TABLE_8_2,
            ),
            Exp | Exp2 => Row::one(UlpGrowing, TABLE_8_2),
            Log | Log2 => {
                let (absolute, ulps) = if prim == Log { (-21, 3.0) } else { (-22, 2.0) };
                Row::on(
                    vec![
                        (
                            Domain::of(0, vec![(0.5, 2.0)], "for x in the domain [0.5, 2]"),
                            Absolute(absolute),
                        ),
                        (
                            Domain::of(0, vec![positive], "otherwise if x > 0"),
                            Ulp(ulps),
                        ),
                    ],
                    "otherwise the results are undefined",
                    TABLE_8_2,
                )
            }
            Rsqrt => Row::one(Ulp(2.0), TABLE_8_2),
            Fmod => Row::on(vec![], "Undefined", TABLE_8_2),
            Sqrt => Row::one(Formula(self::Formula::Sqrt), TABLE_8_2),
            Pow | Powr => Row::one(Formula(self::Formula::Pow), TABLE_8_2),
            Tan => Row::one(Formula(self::Formula::Tan), TABLE_8_2),
            TanPi => Row::one(Formula(self::Formula::TanPi), TABLE_8_2),
            Tanh => Row::one(Formula(self::Formula::Tanh), TABLE_8_2),
            Acosh => Row::one(Formula(self::Formula::Acosh), TABLE_8_2),
            Asinh => Row::one(Formula(self::Formula::Asinh), TABLE_8_2),
            Atanh => Row::one(Formula(self::Formula::Atanh), TABLE_8_2),
            Atan2 => Row::one(Formula(self::Formula::Atan2), TABLE_8_2),
            Sign => unreachable!("sign is Table 6.1's"),
        },
    })
}

/// A closed interval of results, `lo..=hi`, in the order `f64::total_cmp` gives them, where -0
/// precedes +0: a zero's sign is a result like any other value. A range ending at +0 holds no -0,
/// a point holds its own zero alone, and a range from -0 to +0 holds both. So an exact or correctly
/// rounded zero admits its own sign only (Table 6.1's `sign(-0)` is -0, and IEEE 754's rounding
/// fixes an exact zero's sign); a bound that widens a zero by an ulp or an absolute amount admits
/// both; a flush admits both (section 8.5), and so does fast math (section 1.6.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    pub lo: f64,
    pub hi: f64,
}

impl Range {
    pub fn point(value: f64) -> Range {
        Range {
            lo: value,
            hi: value,
        }
    }

    fn is_point(self) -> bool {
        self.lo == self.hi && self.lo.is_sign_negative() == self.hi.is_sign_negative()
    }

    fn contains(self, value: f64) -> bool {
        self.lo.total_cmp(&value).is_le() && value.total_cmp(&self.hi).is_le()
    }

    /// Whether the range holds a zero of either sign.
    fn meets_zero(self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }
}

/// The smaller of two results in [`Range`]'s order (-0 before +0).
fn least(a: f64, b: f64) -> f64 {
    if b.total_cmp(&a).is_lt() {
        b
    } else {
        a
    }
}

/// The larger of two results in [`Range`]'s order (+0 after -0).
fn greatest(a: f64, b: f64) -> f64 {
    if b.total_cmp(&a).is_gt() {
        b
    } else {
        a
    }
}

/// What Apple's rows admit for one result.
#[derive(Clone, Debug, PartialEq)]
pub enum Admits {
    /// A conforming result lies in one of `ranges`, or is any NaN when `nan` (MSL states no NaN
    /// encoding).
    Values { ranges: Vec<Range>, nan: bool },
    /// Apple states no accuracy for this result: any answer conforms. The text cites why.
    Unstated(String),
}

impl Admits {
    pub fn point(value: f64) -> Admits {
        if value.is_nan() {
            Admits::Values {
                ranges: vec![],
                nan: true,
            }
        } else {
            Admits::Values {
                ranges: vec![Range::point(value)],
                nan: false,
            }
        }
    }

    /// Whether `value` conforms.
    pub fn admits(&self, value: f64) -> bool {
        match self {
            Admits::Unstated(_) => true,
            Admits::Values { nan, .. } if value.is_nan() => *nan,
            Admits::Values { ranges, .. } => ranges.iter().any(|range| range.contains(value)),
        }
    }
}

/// The operand at the top of a lane: its value, and under section 8.1 a subnormal operand may be
/// flushed to a zero of either sign (section 8.5); under fast math a NaN or INF operand is
/// undefined, and a zero's sign is insignificant (section 1.6.3).
pub fn operand(value: f64, format: Format, math: Math) -> Admits {
    if math == Math::Fast && !value.is_finite() {
        return Admits::Unstated(FAST_NAN_INF.into());
    }
    let mut admits = Admits::point(value);
    if format.is_subnormal(value) || (math == Math::Fast && value == 0.0) {
        flush(&mut admits);
    }
    admits
}

const FAST_NAN_INF: &str =
    "MSL section 8.1: under fast math the behavior of NaN or INF as an input or output is undefined";
const MIXED_CAP: usize = 4096;

/// Add both zeros: a flushed operand or result, whose sign section 8.5 leaves undefined, or under
/// fast math a zero, whose sign section 1.6.3 lets the compiler treat as insignificant ("No Signed
/// Zeroes: Allow optimizations to treat the sign of a zero argument or result as insignificant").
fn flush(admits: &mut Admits) {
    if let Admits::Values { ranges, .. } = admits {
        ranges.push(Range::point(0.0));
        ranges.push(Range::point(-0.0));
    }
}

/// What `expr` admits computed in `format` under `math`, its operand `k` admitting
/// `bindings[k]`. An error says why no bound can be derived (an operand range a derivation cannot
/// follow); it is never read as conforming.
pub fn admits(
    expr: &Expr,
    bindings: &[Admits],
    format: Format,
    math: Math,
) -> Result<Admits, String> {
    match expr {
        Expr::Operand(index) => bindings
            .get(*index)
            .cloned()
            .ok_or_else(|| format!("operand {index} is not bound")),
        Expr::Const(value) => Ok(Admits::point(*value)),
        Expr::Call(prim, args) => {
            let args = args
                .iter()
                .map(|arg| admits(arg, bindings, format, math))
                .collect::<Result<Vec<_>, _>>()?;
            call(*prim, &args, format, math)
        }
    }
}

fn call(prim: Prim, args: &[Admits], format: Format, math: Math) -> Result<Admits, String> {
    if let Some(why) = args.iter().find_map(|arg| match arg {
        Admits::Unstated(why) => Some(why.clone()),
        Admits::Values { .. } => None,
    }) {
        return Ok(Admits::Unstated(why));
    }
    let row = row(prim, format, math).ok_or_else(|| {
        format!(
            "MSL's tables state no {} for {} under {} math",
            prim.name(),
            format.as_str(),
            math.as_str()
        )
    })?;
    if let [Piece {
        accuracy: Accuracy::Formula(formula),
        ..
    }] = row.pieces.as_slice()
    {
        return formula_admits(*formula, args, format, math, row.source);
    }
    let spans = args
        .iter()
        .map(|arg| match arg {
            Admits::Values { ranges, nan } => (ranges.clone(), *nan),
            Admits::Unstated(_) => unreachable!("returned above"),
        })
        .collect::<Vec<_>>();
    let mut result = Admits::Values {
        ranges: vec![],
        nan: false,
    };
    // A NaN operand gives a NaN, but `sign` answers 0.0 (Table 6.1).
    if spans.iter().any(|(_, nan)| *nan) {
        let from_nan = if prim == Prim::Sign {
            Admits::point(0.0)
        } else {
            Admits::point(f64::NAN)
        };
        union(&mut result, from_nan);
    }
    let combinations = spans
        .iter()
        .map(|(ranges, _)| ranges.len())
        .product::<usize>();
    if combinations > MIXED_CAP {
        return Err(format!(
            "{} over {combinations} operand ranges is past the derivation's cap",
            prim.name()
        ));
    }
    for index in 0..combinations {
        let mut rest = index;
        let combo = spans
            .iter()
            .map(|(ranges, _)| {
                let range = ranges[rest % ranges.len()];
                rest /= ranges.len();
                range
            })
            .collect::<Vec<_>>();
        let image = pieces(prim, &row, &combo, format)?;
        if let Admits::Unstated(_) = image {
            return Ok(image);
        }
        union(&mut result, image);
    }
    Ok(settle(result, format, math))
}

/// Split `combo`'s ranges of the row's domain operand over the row's pieces, in order: each part
/// gets its piece's accuracy, and a part no piece covers is unstated.
fn pieces(prim: Prim, row: &Row, combo: &[Range], format: Format) -> Result<Admits, String> {
    let Some(first) = row.pieces.first() else {
        return Ok(Admits::Unstated(format!(
            "{}: {} {}",
            row.source,
            prim.name(),
            row.otherwise
        )));
    };
    let operand = first.domain.operand;
    let mut remaining = vec![combo[operand]];
    let mut result = Admits::Values {
        ranges: vec![],
        nan: false,
    };
    for piece in &row.pieces {
        let mut next = vec![];
        for part in &remaining {
            for &(lo, hi) in &piece.domain.ranges {
                // A bound equal to the part's end keeps the part's zero sign.
                let inside = Range {
                    lo: if part.lo >= lo { part.lo } else { lo },
                    hi: if part.hi <= hi { part.hi } else { hi },
                };
                if inside.lo <= inside.hi {
                    let inside = if part.is_point() { *part } else { inside };
                    let mut args = combo.to_vec();
                    args[operand] = inside;
                    let exact = exact_image(prim, &args)?;
                    union(&mut result, widen(exact, piece.accuracy, &args, format));
                }
            }
            next.extend(subtract(*part, &piece.domain.ranges));
        }
        remaining = next;
    }
    if !remaining.is_empty() {
        return Ok(Admits::Unstated(format!(
            "{}: {} {}",
            row.source,
            prim.name(),
            row.otherwise
        )));
    }
    Ok(result)
}

/// `part` less `ranges`, as closed ranges (a boundary point may stay in both).
fn subtract(part: Range, ranges: &[Span]) -> Vec<Range> {
    let mut left = vec![part];
    for &(lo, hi) in ranges {
        let mut next = vec![];
        for piece in left {
            if piece.hi < lo || piece.lo > hi {
                next.push(piece);
                continue;
            }
            if piece.lo < lo {
                next.push(Range {
                    lo: piece.lo,
                    hi: lo,
                });
            }
            if piece.hi > hi {
                next.push(Range {
                    lo: hi,
                    hi: piece.hi,
                });
            }
        }
        left = next;
    }
    // A boundary point left over is inside the piece that ends there.
    left.retain(|piece| {
        !(piece.lo == piece.hi
            && ranges
                .iter()
                .any(|&(lo, hi)| lo <= piece.lo && piece.lo <= hi))
    });
    left
}

/// The exact results of `prim` over operand ranges `args`: points evaluate in `f64` (a NaN result
/// is admitted as a NaN); ranges follow the operation's monotony or interval arithmetic.
fn exact_image(prim: Prim, args: &[Range]) -> Result<Admits, String> {
    if args.iter().all(|range| range.is_point()) {
        return Ok(Admits::point(
            prim.exact(&args.iter().map(|range| range.lo).collect::<Vec<_>>()),
        ));
    }
    let cannot = || {
        Err(format!(
            "{} over a range of operands is not derived",
            prim.name()
        ))
    };
    let range = |lo: f64, hi: f64| -> Result<Admits, String> {
        if lo.is_nan() || hi.is_nan() || lo > hi {
            return cannot();
        }
        Ok(Admits::Values {
            ranges: vec![Range { lo, hi }],
            nan: false,
        })
    };
    match prim {
        Prim::Add => range(args[0].lo + args[1].lo, args[0].hi + args[1].hi),
        Prim::Sub => range(args[0].lo - args[1].hi, args[0].hi - args[1].lo),
        Prim::Mul | Prim::Div => {
            let (a, b) = (args[0], args[1]);
            if prim == Prim::Div && b.lo <= 0.0 && b.hi >= 0.0 {
                return cannot();
            }
            let corners = [(a.lo, b.lo), (a.lo, b.hi), (a.hi, b.lo), (a.hi, b.hi)].map(|(x, y)| {
                if prim == Prim::Mul {
                    x * y
                } else {
                    x / y
                }
            });
            if corners.iter().any(|value| value.is_nan()) {
                return cannot();
            }
            range(
                corners.iter().copied().fold(f64::INFINITY, least),
                corners.iter().copied().fold(f64::NEG_INFINITY, greatest),
            )
        }
        Prim::Sign => {
            let a = args[0];
            let mut ranges = vec![];
            if a.lo < 0.0 {
                ranges.push(Range::point(-1.0));
            }
            if a.hi > 0.0 {
                ranges.push(Range::point(1.0));
            }
            // Table 6.1: a zero's sign is its own.
            for zero in [0.0, -0.0] {
                if a.contains(zero) {
                    ranges.push(Range::point(zero));
                }
            }
            Ok(Admits::Values { ranges, nan: false })
        }
        // Between the ends' values, and 1 or -1 where the range holds a crest or a trough.
        Prim::Sin | Prim::Cos => {
            let a = args[0];
            if !a.lo.is_finite() || !a.hi.is_finite() {
                return cannot();
            }
            let (ends, crest) = (
                [prim.exact(&[a.lo]), prim.exact(&[a.hi])],
                if prim == Prim::Sin {
                    std::f64::consts::FRAC_PI_2
                } else {
                    0.0
                },
            );
            let holds = |at: f64| {
                let turns = ((a.lo - at) / std::f64::consts::TAU).ceil();
                at + turns * std::f64::consts::TAU <= a.hi
            };
            let lo = if holds(crest + std::f64::consts::PI) {
                -1.0
            } else {
                ends[0].min(ends[1])
            };
            let hi = if holds(crest) {
                1.0
            } else {
                ends[0].max(ends[1])
            };
            range(lo, hi)
        }
        _ => match prim.monotone() {
            Some(increasing) => {
                let (lo, hi) = (prim.exact(&[args[0].lo]), prim.exact(&[args[0].hi]));
                // A pole or the edge of the domain inside the range breaks monotony.
                let crosses = match prim {
                    Prim::Rsqrt | Prim::Recip | Prim::Log | Prim::Log2 => {
                        args[0].lo <= 0.0 && args[0].hi >= 0.0
                    }
                    _ => false,
                };
                if crosses {
                    return cannot();
                }
                if increasing {
                    range(lo, hi)
                } else {
                    range(hi, lo)
                }
            }
            None => cannot(),
        },
    }
}

/// Widen exact results by an accuracy. NaN and infinite exact results stay as they are.
fn widen(exact: Admits, accuracy: Accuracy, args: &[Range], format: Format) -> Admits {
    let Admits::Values { ranges, nan } = exact else {
        return exact;
    };
    let ranges = ranges
        .into_iter()
        .map(|range| {
            if !range.lo.is_finite() || !range.hi.is_finite() {
                return range;
            }
            // The largest magnitude's ulp, so that every exact result inside gets its whole bound.
            let ulp = format.ulp(range.lo.abs().max(range.hi.abs()));
            let (below, above) = match accuracy {
                Accuracy::Exact | Accuracy::Formula(_) => (0.0, 0.0),
                // A representable exact result is its own rounding in either mode.
                Accuracy::CorrectlyRounded
                    if range.lo == range.hi && format.represents(range.lo) =>
                {
                    (0.0, 0.0)
                }
                Accuracy::CorrectlyRounded => (
                    if range.lo > 0.0 { ulp } else { ulp / 2.0 },
                    if range.hi < 0.0 { ulp } else { ulp / 2.0 },
                ),
                Accuracy::Ulp(ulps) => (ulps * ulp, ulps * ulp),
                Accuracy::UlpGrowing => {
                    let x = args[0].lo.abs().max(args[0].hi.abs());
                    let ulps = 3.0 + (2.0 * x).floor();
                    (ulps * ulp, ulps * ulp)
                }
                Accuracy::Absolute(n) => (pow2(n), pow2(n)),
            };
            if below == 0.0 && above == 0.0 {
                range
            } else {
                Range {
                    lo: range.lo - below,
                    hi: range.hi + above,
                }
            }
        })
        .collect();
    Admits::Values { ranges, nan }
}

/// After an operation: a result past the largest finite value rounds to INF (or, rounding toward
/// zero, to the largest finite value), which fast math leaves undefined, as it does a NaN; a
/// subnormal result may be flushed to a zero of either sign (sections 8.1, 8.5); and under fast
/// math a zero result's sign is insignificant (section 1.6.3).
fn settle(admits: Admits, format: Format, math: Math) -> Admits {
    let Admits::Values { mut ranges, nan } = admits else {
        return admits;
    };
    let max = format.max_finite();
    let overflows = ranges.iter().any(|range| range.lo < -max || range.hi > max);
    if math == Math::Fast && (nan || overflows) {
        return Admits::Unstated(FAST_NAN_INF.into());
    }
    let mut extra = vec![];
    for range in &mut ranges {
        // An exact infinity (a division by zero, say) is the result in either rounding mode.
        if range.lo == range.hi && range.lo.is_infinite() {
            continue;
        }
        if range.hi > max {
            extra.push(Range::point(f64::INFINITY));
            extra.push(Range::point(max));
            range.hi = max;
        }
        if range.lo < -max {
            extra.push(Range::point(f64::NEG_INFINITY));
            extra.push(Range::point(-max));
            range.lo = -max;
        }
    }
    ranges.retain(|range| range.lo <= range.hi);
    let subnormal = ranges.iter().any(|range| {
        let min = format.min_normal();
        range.lo < min && range.hi > -min && !(range.lo == 0.0 && range.hi == 0.0)
    });
    let zero = math == Math::Fast && ranges.iter().any(|range| range.meets_zero());
    ranges.extend(extra);
    let mut settled = Admits::Values { ranges, nan };
    if subnormal || zero {
        flush(&mut settled);
    }
    merge(settled)
}

fn union(into: &mut Admits, other: Admits) {
    match (&mut *into, other) {
        (Admits::Unstated(_), _) => {}
        (_, unstated @ Admits::Unstated(_)) => *into = unstated,
        (
            Admits::Values { ranges, nan },
            Admits::Values {
                ranges: more,
                nan: more_nan,
            },
        ) => {
            ranges.extend(more);
            *nan |= more_nan;
        }
    }
}

/// Sort and merge overlapping ranges; zero points of either sign stay apart.
fn merge(admits: Admits) -> Admits {
    let Admits::Values { mut ranges, nan } = admits else {
        return admits;
    };
    ranges.sort_by(|a, b| a.lo.total_cmp(&b.lo).then(a.hi.total_cmp(&b.hi)));
    let mut merged: Vec<Range> = vec![];
    let same =
        |a: &Range, b: &Range| a.lo.to_bits() == b.lo.to_bits() && a.hi.to_bits() == b.hi.to_bits();
    for range in ranges {
        match merged.last_mut() {
            Some(last) if same(last, &range) => {}
            Some(last)
                if range.lo <= last.hi
                    && !(last.is_point() && range.is_point() && last.lo == 0.0) =>
            {
                last.hi = greatest(last.hi, range.hi)
            }
            _ => merged.push(range),
        }
    }
    Admits::Values {
        ranges: merged,
        nan,
    }
}

/// Table 8.2's formula for `formula` over its evaluated operands.
fn formula_admits(
    formula: Formula,
    args: &[Admits],
    format: Format,
    math: Math,
    source: &'static str,
) -> Result<Admits, String> {
    use Expr::{Call, Const, Operand};
    let x = || Operand(0);
    let y = || Operand(1);
    let call = |prim, args: Vec<Expr>| Call(prim, args);
    let spans = |arg: &Admits| match arg {
        Admits::Values { ranges, nan } => (ranges.clone(), *nan),
        Admits::Unstated(_) => (vec![], false),
    };
    let undefined = |what: &str| {
        Ok(Admits::Unstated(format!(
            "{source}: {} {what}",
            match formula {
                Formula::Pow => "pow and powr are",
                Formula::Atan2 => "atan2 is",
                _ => "the formula is",
            }
        )))
    };
    let expr = match formula {
        Formula::Tan => call(
            Prim::Mul,
            vec![
                call(Prim::Sin, vec![x()]),
                call(Prim::Recip, vec![call(Prim::Cos, vec![x()])]),
            ],
        ),
        // `pi` is MSL's `M_PI_F`, the `float` nearest π.
        Formula::TanPi => call(
            Prim::Tan,
            vec![call(
                Prim::Mul,
                vec![x(), Const(f64::from(std::f32::consts::PI))],
            )],
        ),
        Formula::Tanh => {
            let t = || call(Prim::Exp, vec![call(Prim::Mul, vec![Const(2.0), x()])]);
            call(
                Prim::Div,
                vec![
                    call(Prim::Sub, vec![t(), Const(1.0)]),
                    call(Prim::Add, vec![t(), Const(1.0)]),
                ],
            )
        }
        Formula::Sqrt => {
            // "Special cases handled correctly": a zero is its own root.
            let (ranges, nan) = spans(&args[0]);
            if !nan
                && ranges
                    .iter()
                    .all(|range| range.lo == 0.0 && range.hi == 0.0)
            {
                return Ok(args[0].clone());
            }
            call(Prim::Mul, vec![x(), call(Prim::Rsqrt, vec![x()])])
        }
        Formula::Pow => {
            let zero = |arg: &Admits| {
                spans(arg)
                    .0
                    .iter()
                    .any(|range| range.lo <= 0.0 && range.hi >= 0.0)
            };
            if zero(&args[0]) && zero(&args[1]) {
                return undefined("undefined for x = 0 and y = 0");
            }
            call(
                Prim::Exp2,
                vec![call(Prim::Mul, vec![y(), call(Prim::Log2, vec![x()])])],
            )
        }
        Formula::Acosh | Formula::Asinh => {
            let inner = if formula == Formula::Acosh {
                Prim::Sub
            } else {
                Prim::Add
            };
            call(
                Prim::Log,
                vec![call(
                    Prim::Add,
                    vec![
                        x(),
                        call(
                            Prim::Sqrt,
                            vec![call(
                                inner,
                                vec![call(Prim::Mul, vec![x(), x()]), Const(1.0)],
                            )],
                        ),
                    ],
                )],
            )
        }
        Formula::Atanh => call(
            Prim::Mul,
            vec![
                Const(0.5),
                call(
                    Prim::Log,
                    vec![call(
                        Prim::Div,
                        vec![
                            call(Prim::Add, vec![Const(1.0), x()]),
                            call(Prim::Sub, vec![Const(1.0), x()]),
                        ],
                    )],
                ),
            ],
        ),
        Formula::Atan2 => {
            // atan2(y, x): the first operand is y.
            let sign = |arg: &Admits| {
                let (ranges, nan) = spans(arg);
                if nan
                    || ranges
                        .iter()
                        .any(|range| range.lo <= 0.0 && range.hi >= 0.0)
                {
                    None
                } else if ranges.iter().all(|range| range.lo > 0.0) {
                    Some(true)
                } else if ranges.iter().all(|range| range.hi < 0.0) {
                    Some(false)
                } else {
                    None
                }
            };
            let quotient = call(Prim::Atan, vec![call(Prim::Div, vec![x(), y()])]);
            let pi = Const(f64::from(std::f32::consts::PI));
            match (sign(&args[0]), sign(&args[1])) {
                (Some(_), Some(true)) => quotient,
                (Some(true), Some(false)) => call(Prim::Add, vec![quotient, pi]),
                (Some(false), Some(false)) => call(Prim::Sub, vec![quotient, pi]),
                _ => return undefined("undefined if x = 0 or y = 0"),
            }
        }
    };
    admits(&expr, args, format, math)
}
