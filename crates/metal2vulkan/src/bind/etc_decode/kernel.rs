//! The GPU's decode of ETC2 and EAC blocks into a decoded twin's texels: the module
//! [`etc_decode_module`](super::etc_decode_module) serializes, the same decode as the CPU's
//! ([`EtcDecode::decode_into`]) in integer operations alone.
//!
//! The dispatch and shape are the block decode's (`BlockDecodeModule::shape`): one invocation per
//! texel of the region reads its block's 64-bit words, each the byte-swapped pair of the buffer's
//! little-endian words (a block is read most significant byte first), and writes its texel
//! straight-line: an ETC2 RGB block's five modes are each decoded and the block's own selected. An
//! `R11` format's compact texels are 2 bytes, two to a word: the invocation of each even texel
//! (counted through the region row by row, image by image) writes the word of it and the next.

use super::{Channel, EtcDecode, EtcTexels, DISTANCES, EAC_MODIFIERS, MODIFIERS};
use crate::bind::block_decode::{table, BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::bind::ycbcr_convert::{scalar, words_buffer};
use crate::lower::arith::exact_division::div_exact;
use crate::lower::arith::legal::{Form, Held};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule,
};

/// The decode of one texel: the function it builds in, and what every step reads.
struct E<'a, 'b> {
    f: &'a mut FunctionBuilder<'b>,
    u32_: TypeId,
    zero: Value,
    /// The blocks' words.
    blocks: Value,
    /// The region's blocks across and down.
    across: Value,
    down: Value,
    /// The individual and differential modes' modifier pairs, `a` then `b` of each table
    /// (`MODIFIERS`).
    modifiers: GlobalId,
    /// The T and H modes' distances (`DISTANCES`).
    distances: GlobalId,
    /// The EAC modifier tables, eight entries each, two's complement (`EAC_MODIFIERS`).
    eac: GlobalId,
}

/// A texel's place in its region: its column, row and image.
#[derive(Clone, Copy)]
struct At {
    x: Value,
    y: Value,
    z: Value,
}

impl E<'_, '_> {
    fn k(&mut self, v: u32) -> BResult<Value> {
        scalar(self.f, self.u32_, u64::from(v))
    }

    /// The two's complement constant `v`.
    fn signed(&mut self, v: i32) -> BResult<Value> {
        self.k(v as u32)
    }

    /// The `bool` constant `v`.
    fn boolean(&mut self, v: bool) -> BResult<Value> {
        let ty = self.f.module().ty(Type::Bool)?;
        let c = self.f.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(u64::from(v)),
        })?;
        self.f.module().const_value(c)
    }

    fn op(&mut self, op: BinOp, a: Value, b: Value) -> BResult<Value> {
        self.f.binary(op, a, b)
    }

    fn opk(&mut self, op: BinOp, a: Value, b: u32) -> BResult<Value> {
        let b = self.k(b)?;
        self.f.binary(op, a, b)
    }

    fn add(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::IAdd, a, b)
    }

    fn sel(&mut self, c: Value, a: Value, b: Value) -> BResult<Value> {
        self.f.select(c, a, b)
    }

    fn and(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::LogicalAnd, a, b)
    }

    fn not(&mut self, a: Value) -> BResult<Value> {
        self.f.unary(crate::spirv::ir::UnOp::LogicalNot, a)
    }

    /// `0 - v`.
    fn negate(&mut self, v: Value) -> BResult<Value> {
        self.op(BinOp::ISub, self.zero, v)
    }

    /// `(v >> at) & (2^n - 1)` for a constant field.
    fn field(&mut self, v: Value, at: u32, n: u32) -> BResult<Value> {
        let shifted = self.opk(BinOp::ShiftRightLogical, v, at)?;
        self.opk(BinOp::BitwiseAnd, shifted, (1 << n) - 1)
    }

    /// Whether bit `at` of `v` is set.
    fn bit(&mut self, v: Value, at: u32) -> BResult<Value> {
        let field = self.field(v, at, 1)?;
        self.opk(BinOp::INotEqual, field, 0)
    }

    /// `a << n | b`, two fields joined.
    fn join(&mut self, a: Value, n: u32, b: Value) -> BResult<Value> {
        let a = self.opk(BinOp::ShiftLeftLogical, a, n)?;
        self.op(BinOp::BitwiseOr, a, b)
    }

    /// `v`, of `n` bits, widened to 8 by replicating its high bits (`super::widen`).
    fn widen(&mut self, v: Value, n: u32) -> BResult<Value> {
        let high = self.opk(BinOp::ShiftLeftLogical, v, 8 - n)?;
        let low = self.opk(BinOp::ShiftRightLogical, v, 2 * n - 8)?;
        self.op(BinOp::BitwiseOr, high, low)
    }

    /// The `n`-bit two's complement field `v`, sign-extended.
    fn extend(&mut self, v: Value, n: u32) -> BResult<Value> {
        let up = self.opk(BinOp::ShiftLeftLogical, v, 32 - n)?;
        self.opk(BinOp::ShiftRightArithmetic, up, 32 - n)
    }

    /// The signed `v` clamped to `[lo, hi]`.
    fn clamp(&mut self, v: Value, lo: i32, hi: i32) -> BResult<Value> {
        let (lo, hi) = (self.signed(lo)?, self.signed(hi)?);
        let below = self.op(BinOp::SLessThan, v, lo)?;
        let above = self.op(BinOp::SGreaterThan, v, hi)?;
        let v = self.sel(above, hi, v)?;
        self.sel(below, lo, v)
    }

    /// Entry `index` of the `Private` table `table`.
    fn entry(&mut self, table: GlobalId, index: Value) -> BResult<Value> {
        let table = self.f.module().global_value(table)?;
        let p = self.f.access_chain(table, &[index])?;
        self.f.load(p, MemAccess::default())
    }

    /// `w` with its bytes reversed.
    fn swap(&mut self, w: Value) -> BResult<Value> {
        let b0 = self.opk(BinOp::ShiftLeftLogical, w, 24)?;
        let b1 = self.opk(BinOp::BitwiseAnd, w, 0xff00)?;
        let b1 = self.opk(BinOp::ShiftLeftLogical, b1, 8)?;
        let b2 = self.opk(BinOp::ShiftRightLogical, w, 8)?;
        let b2 = self.opk(BinOp::BitwiseAnd, b2, 0xff00)?;
        let b3 = self.opk(BinOp::ShiftRightLogical, w, 24)?;
        let v = self.op(BinOp::BitwiseOr, b0, b1)?;
        let v = self.op(BinOp::BitwiseOr, v, b2)?;
        self.op(BinOp::BitwiseOr, v, b3)
    }

    /// The texel's block's first word, `((z · down + y / 4) · across + x / 4) · words`, and its
    /// index in the block numbered down each column, `4 (x mod 4) + y mod 4`, with its column and
    /// row in the block.
    fn place(&mut self, at: At, words: u32) -> BResult<(Value, [Value; 3])> {
        let row = self.op(BinOp::IMul, at.z, self.down)?;
        let block_y = self.opk(BinOp::ShiftRightLogical, at.y, 2)?;
        let row = self.add(row, block_y)?;
        let block = self.op(BinOp::IMul, row, self.across)?;
        let block_x = self.opk(BinOp::ShiftRightLogical, at.x, 2)?;
        let block = self.add(block, block_x)?;
        let base = self.opk(BinOp::IMul, block, words)?;
        let bx = self.opk(BinOp::BitwiseAnd, at.x, 3)?;
        let by = self.opk(BinOp::BitwiseAnd, at.y, 3)?;
        let p = self.join(bx, 2, by)?;
        Ok((base, [p, bx, by]))
    }

    /// The high and low 32 bits of the 64-bit block half `half` (0 or 1) of the block whose first
    /// word is `base`.
    fn half(&mut self, base: Value, half: u32) -> BResult<[Value; 2]> {
        let mut out = [self.zero; 2];
        for (k, word) in out.iter_mut().enumerate() {
            let index = self.opk(BinOp::IAdd, base, 2 * half + k as u32)?;
            let p = self.f.access_chain(self.blocks, &[self.zero, index])?;
            let raw = self.f.load(p, MemAccess::default())?;
            *word = self.swap(raw)?;
        }
        Ok(out)
    }

    /// The value of texel `p` (numbered down each column) of the EAC block `[hi, lo]` as
    /// `channel` reads it (`super::eac_block`), two's complement.
    fn eac(&mut self, [hi, lo]: [Value; 2], p: Value, channel: Channel) -> BResult<Value> {
        let base = self.field(hi, 24, 8)?;
        let multiplier = self.field(hi, 20, 4)?;
        let table = self.field(hi, 16, 4)?;
        // The texel's 3-bit index from bit `s = 45 - 3 p` of the 64: in the high word past 32, in
        // the low word below 30, across both at 30.
        let three_p = self.opk(BinOp::IMul, p, 3)?;
        let k45 = self.k(45)?;
        let s = self.op(BinOp::ISub, k45, three_p)?;
        let high_shift = self.opk(BinOp::ISub, s, 32)?;
        let high_shift = self.opk(BinOp::BitwiseAnd, high_shift, 31)?;
        let from_high = self.op(BinOp::ShiftRightLogical, hi, high_shift)?;
        let low_shift = self.opk(BinOp::BitwiseAnd, s, 31)?;
        let from_low = self.op(BinOp::ShiftRightLogical, lo, low_shift)?;
        let k32 = self.k(32)?;
        let up = self.op(BinOp::ISub, k32, s)?;
        let up = self.opk(BinOp::BitwiseAnd, up, 31)?;
        let carried = self.op(BinOp::ShiftLeftLogical, hi, up)?;
        let across = self.opk(BinOp::UGreaterThan, s, 29)?;
        let carried = self.sel(across, carried, self.zero)?;
        let from_low = self.op(BinOp::BitwiseOr, from_low, carried)?;
        let in_high = self.opk(BinOp::UGreaterThanEqual, s, 32)?;
        let index = self.sel(in_high, from_high, from_low)?;
        let index = self.opk(BinOp::BitwiseAnd, index, 7)?;
        let entry = self.opk(BinOp::ShiftLeftLogical, table, 3)?;
        let entry = self.add(entry, index)?;
        let modifier = self.entry(self.eac, entry)?;
        let scaled = self.op(BinOp::IMul, modifier, multiplier)?;
        if channel == Channel::Alpha {
            let sum = self.add(base, scaled)?;
            return self.clamp(sum, 0, 255);
        }
        // A multiplier of 0 adds the modifier itself, an eighth of a step.
        let none = self.opk(BinOp::IEqual, multiplier, 0)?;
        let eight = self.opk(BinOp::ShiftLeftLogical, scaled, 3)?;
        let step = self.sel(none, modifier, eight)?;
        match channel {
            Channel::Unsigned => {
                let base = self.opk(BinOp::ShiftLeftLogical, base, 3)?;
                let base = self.opk(BinOp::IAdd, base, 4)?;
                let sum = self.add(base, step)?;
                self.clamp(sum, 0, 2047)
            }
            _ => {
                // A signed base of -128 is -127.
                let base = self.extend(base, 8)?;
                let low = self.signed(-127)?;
                let below = self.op(BinOp::SLessThan, base, low)?;
                let base = self.sel(below, low, base)?;
                let base = self.opk(BinOp::ShiftLeftLogical, base, 3)?;
                let sum = self.add(base, step)?;
                self.clamp(sum, -1023, 1023)
            }
        }
    }

    /// The four 8-bit codes of texel `p` (numbered down each column, at column `bx` and row `by`
    /// of its block) of the ETC2 RGB block `[hi, lo]` (`super::rgb_block`): `punch`, the RGB8A1
    /// form's.
    fn rgb(
        &mut self,
        [hi, lo]: [Value; 2],
        [p, bx, by]: [Value; 3],
        punch: bool,
    ) -> BResult<[Value; 4]> {
        let flag = self.bit(hi, 1)?;
        let yes = self.boolean(true)?;
        let (opaque, differential) = match punch {
            true => (flag, yes),
            false => (yes, flag),
        };
        // The texel's 2-bit index: bit `16 + p` above bit `p`.
        let msb_at = self.opk(BinOp::IAdd, p, 16)?;
        let msb = self.op(BinOp::ShiftRightLogical, lo, msb_at)?;
        let msb = self.opk(BinOp::BitwiseAnd, msb, 1)?;
        let lsb = self.op(BinOp::ShiftRightLogical, lo, p)?;
        let lsb = self.opk(BinOp::BitwiseAnd, lsb, 1)?;
        let index = self.join(msb, 1, lsb)?;
        let odd = self.opk(BinOp::IEqual, lsb, 1)?;
        let index_is = |e: &mut Self, v: u32| e.opk(BinOp::IEqual, index, v);
        // The differential mode's bases and deltas, and whether a sum leaves `[0, 31]` (a
        // negative one wraps past 31).
        let mut diff = [[self.zero; 3]; 2];
        let mut outside = [self.zero; 3];
        for (ch, at) in [27, 19, 11].into_iter().enumerate() {
            let c = self.field(hi, at, 5)?;
            let d = self.field(hi, at - 3, 3)?;
            let d = self.extend(d, 3)?;
            let sum = self.add(c, d)?;
            outside[ch] = self.opk(BinOp::UGreaterThan, sum, 31)?;
            let sum = self.opk(BinOp::BitwiseAnd, sum, 31)?;
            diff[0][ch] = self.widen(c, 5)?;
            diff[1][ch] = self.widen(sum, 5)?;
        }
        // The individual mode's.
        let mut individual = [[self.zero; 3]; 2];
        for (ch, at) in [28, 20, 12].into_iter().enumerate() {
            let first = self.field(hi, at, 4)?;
            let second = self.field(hi, at - 4, 4)?;
            individual[0][ch] = self.widen(first, 4)?;
            individual[1][ch] = self.widen(second, 4)?;
        }
        // The individual and differential modes' texel: its subblock's base plus the modifier
        // its table and index pick (`[a, b, -a, -b]`, or `[0, b, 0, -b]` where not opaque).
        let flip = self.bit(hi, 0)?;
        let low_x = self.opk(BinOp::UGreaterThanEqual, bx, 2)?;
        let low_y = self.opk(BinOp::UGreaterThanEqual, by, 2)?;
        let second = self.sel(flip, low_y, low_x)?;
        let first_table = self.field(hi, 5, 3)?;
        let second_table = self.field(hi, 2, 3)?;
        let table = self.sel(second, second_table, first_table)?;
        let a_at = self.opk(BinOp::ShiftLeftLogical, table, 1)?;
        let b_at = self.opk(BinOp::IAdd, a_at, 1)?;
        let a = self.entry(self.modifiers, a_at)?;
        let b = self.entry(self.modifiers, b_at)?;
        let a = self.sel(opaque, a, self.zero)?;
        let magnitude = self.sel(odd, b, a)?;
        let negative = self.opk(BinOp::UGreaterThanEqual, index, 2)?;
        let negated = self.negate(magnitude)?;
        let modifier = self.sel(negative, negated, magnitude)?;
        let mut normal = [self.zero; 3];
        for (ch, out) in normal.iter_mut().enumerate() {
            let first = self.sel(differential, diff[0][ch], individual[0][ch])?;
            let second_base = self.sel(differential, diff[1][ch], individual[1][ch])?;
            let base = self.sel(second, second_base, first)?;
            let sum = self.add(base, modifier)?;
            *out = self.clamp(sum, 0, 255)?;
        }
        // The T mode: paint colors `c1`, `c2 + d`, `c2` and `c2 - d`.
        let t_red_high = self.field(hi, 27, 2)?;
        let t_red_low = self.field(hi, 24, 2)?;
        let t_red = self.join(t_red_high, 2, t_red_low)?;
        let t1 = [t_red, self.field(hi, 20, 4)?, self.field(hi, 16, 4)?];
        let t2 = [
            self.field(hi, 12, 4)?,
            self.field(hi, 8, 4)?,
            self.field(hi, 4, 4)?,
        ];
        let t_high = self.field(hi, 2, 2)?;
        let t_low = self.field(hi, 0, 1)?;
        let t_distance = self.join(t_high, 1, t_low)?;
        let t_distance = self.entry(self.distances, t_distance)?;
        let first_paint = index_is(self, 0)?;
        let plus = index_is(self, 1)?;
        let minus = index_is(self, 3)?;
        let t_negated = self.negate(t_distance)?;
        let t_shift = self.sel(minus, t_negated, self.zero)?;
        let t_shift = self.sel(plus, t_distance, t_shift)?;
        let mut t_mode = [self.zero; 3];
        for (ch, out) in t_mode.iter_mut().enumerate() {
            let c1 = self.widen(t1[ch], 4)?;
            let c2 = self.widen(t2[ch], 4)?;
            let color = self.sel(first_paint, c1, c2)?;
            let sum = self.add(color, t_shift)?;
            *out = self.clamp(sum, 0, 255)?;
        }
        // The H mode: paint colors `c1 + d`, `c1 - d`, `c2 + d` and `c2 - d`, the distance's low
        // bit whether `c1`'s colors sort at or after `c2`'s.
        let h_green_high = self.field(hi, 24, 3)?;
        let h_green_low = self.field(hi, 20, 1)?;
        let h_blue_high = self.field(hi, 19, 1)?;
        let h_blue_low = self.field(hi, 15, 3)?;
        let h1 = [
            self.field(hi, 27, 4)?,
            self.join(h_green_high, 1, h_green_low)?,
            self.join(h_blue_high, 3, h_blue_low)?,
        ];
        let h2 = [
            self.field(hi, 11, 4)?,
            self.field(hi, 7, 4)?,
            self.field(hi, 3, 4)?,
        ];
        let mut h1_wide = [self.zero; 3];
        let mut h2_wide = [self.zero; 3];
        for ch in 0..3 {
            h1_wide[ch] = self.widen(h1[ch], 4)?;
            h2_wide[ch] = self.widen(h2[ch], 4)?;
        }
        let key = |e: &mut Self, [r, g, b]: [Value; 3]| -> BResult<Value> {
            let rg = e.join(r, 8, g)?;
            e.join(rg, 8, b)
        };
        let key1 = key(self, h1_wide)?;
        let key2 = key(self, h2_wide)?;
        let sorted = self.op(BinOp::UGreaterThanEqual, key1, key2)?;
        let one = self.k(1)?;
        let sorted = self.sel(sorted, one, self.zero)?;
        let h_high = self.field(hi, 2, 1)?;
        let h_mid = self.field(hi, 0, 1)?;
        let h_distance = self.join(h_high, 1, h_mid)?;
        let h_distance = self.join(h_distance, 1, sorted)?;
        let h_distance = self.entry(self.distances, h_distance)?;
        let h_negated = self.negate(h_distance)?;
        let h_shift = self.sel(odd, h_negated, h_distance)?;
        let first_pair = self.opk(BinOp::ULessThan, index, 2)?;
        let mut h_mode = [self.zero; 3];
        for (ch, out) in h_mode.iter_mut().enumerate() {
            let color = self.sel(first_pair, h1_wide[ch], h2_wide[ch])?;
            let sum = self.add(color, h_shift)?;
            *out = self.clamp(sum, 0, 255)?;
        }
        // The planar mode: `(x (H - O) + y (V - O) + 4 O + 2) >> 2`, always opaque.
        let o_green_high = self.field(hi, 24, 1)?;
        let o_green_low = self.field(hi, 17, 6)?;
        let o_blue_high = self.field(hi, 16, 1)?;
        let o_blue_mid = self.field(hi, 11, 2)?;
        let o_blue_low = self.field(hi, 7, 3)?;
        let o_blue = self.join(o_blue_high, 2, o_blue_mid)?;
        let o_blue = self.join(o_blue, 3, o_blue_low)?;
        let h_red_high = self.field(hi, 2, 5)?;
        let h_red_low = self.field(hi, 0, 1)?;
        let origin = [
            (self.field(hi, 25, 6)?, 6),
            (self.join(o_green_high, 6, o_green_low)?, 7),
            (o_blue, 6),
        ];
        let horizontal = [
            (self.join(h_red_high, 1, h_red_low)?, 6),
            (self.field(lo, 25, 7)?, 7),
            (self.field(lo, 19, 6)?, 6),
        ];
        let vertical = [
            (self.field(lo, 13, 6)?, 6),
            (self.field(lo, 6, 7)?, 7),
            (self.field(lo, 0, 6)?, 6),
        ];
        let mut planar = [self.zero; 3];
        for (ch, out) in planar.iter_mut().enumerate() {
            let o = self.widen(origin[ch].0, origin[ch].1)?;
            let h = self.widen(horizontal[ch].0, horizontal[ch].1)?;
            let v = self.widen(vertical[ch].0, vertical[ch].1)?;
            let dh = self.op(BinOp::ISub, h, o)?;
            let dh = self.op(BinOp::IMul, bx, dh)?;
            let dv = self.op(BinOp::ISub, v, o)?;
            let dv = self.op(BinOp::IMul, by, dv)?;
            let four = self.opk(BinOp::ShiftLeftLogical, o, 2)?;
            let sum = self.add(dh, dv)?;
            let sum = self.add(sum, four)?;
            let sum = self.opk(BinOp::IAdd, sum, 2)?;
            let sum = self.opk(BinOp::ShiftRightArithmetic, sum, 2)?;
            *out = self.clamp(sum, 0, 255)?;
        }
        // The block's mode: a differential block whose red sum leaves `[0, 31]` is T, else whose
        // green does H, else whose blue does planar.
        let not_red = self.not(outside[0])?;
        let not_green = self.not(outside[1])?;
        let is_t = self.and(differential, outside[0])?;
        let is_h = self.and(differential, not_red)?;
        let is_planar = self.and(is_h, not_green)?;
        let is_h = self.and(is_h, outside[1])?;
        let is_planar = self.and(is_planar, outside[2])?;
        // Transparent black: index 2 of a block not opaque, but for the planar mode.
        let translucent = self.not(opaque)?;
        let two = index_is(self, 2)?;
        let clear = self.and(translucent, two)?;
        let solid = self.not(is_planar)?;
        let clear = self.and(clear, solid)?;
        let mut codes = [self.zero; 4];
        for ch in 0..3 {
            let color = self.sel(is_planar, planar[ch], normal[ch])?;
            let color = self.sel(is_h, h_mode[ch], color)?;
            let color = self.sel(is_t, t_mode[ch], color)?;
            codes[ch] = self.sel(clear, self.zero, color)?;
        }
        let full = self.k(255)?;
        codes[3] = self.sel(clear, self.zero, full)?;
        Ok(codes)
    }

    /// The `float` bits of an 11-bit `channel`'s value `v`, correctly rounded
    /// (`EtcDecode::metal_value`): `v / 2047`, or `v / 1023` signed.
    fn value(&mut self, v: Value, channel: Channel) -> BResult<Value> {
        let u = Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes: 1,
        };
        if channel != Channel::Signed {
            let q = div_exact(self.f, u, v, channel.divisor() as u32)?;
            return self.f.convert(ConvOp::Bitcast, q, self.u32_);
        }
        let negative = self.op(BinOp::SLessThan, v, self.zero)?;
        let negated = self.negate(v)?;
        let magnitude = self.sel(negative, negated, v)?;
        let q = div_exact(self.f, u, magnitude, channel.divisor() as u32)?;
        let q = self.f.convert(ConvOp::Bitcast, q, self.u32_)?;
        let sign = self.opk(BinOp::BitwiseOr, q, 0x8000_0000)?;
        self.sel(negative, sign, q)
    }

    /// The 16-bit normalized code nearest an 11-bit `channel`'s value `v`, in the low half
    /// (`EtcDecode::twin_texel`): `(2 |v| top + d) / 2d`, signed as `v` is.
    fn code(&mut self, v: Value, channel: Channel) -> BResult<Value> {
        let d = channel.divisor() as u32;
        let top = match channel {
            Channel::Signed => 32767,
            _ => 65535,
        };
        let negative = self.op(BinOp::SLessThan, v, self.zero)?;
        let negated = self.negate(v)?;
        let magnitude = self.sel(negative, negated, v)?;
        let scaled = self.opk(BinOp::IMul, magnitude, 2 * top)?;
        let scaled = self.opk(BinOp::IAdd, scaled, d)?;
        let q = self.opk(BinOp::UDiv, scaled, 2 * d)?;
        let negated = self.negate(q)?;
        let code = self.sel(negative, negated, q)?;
        self.opk(BinOp::BitwiseAnd, code, 0xffff)
    }

    /// The words of the twin's texel at `at` of `decode`'s blocks in `texels` (an `R11` format's
    /// compact texel, the code in the low half alone).
    fn texel(&mut self, decode: EtcDecode, texels: EtcTexels, at: At) -> BResult<Vec<Value>> {
        let (base, place) = self.place(at, decode.block_bytes() / 4)?;
        let pack = |e: &mut Self, codes: [Value; 4]| -> BResult<Value> {
            let mut packed = codes[0];
            for (k, &c) in (1u32..).zip(&codes[1..]) {
                let placed = e.opk(BinOp::ShiftLeftLogical, c, 8 * k)?;
                packed = e.op(BinOp::BitwiseOr, packed, placed)?;
            }
            Ok(packed)
        };
        let channels = match decode {
            EtcDecode::Rgb8 | EtcDecode::Rgb8A1 => {
                let block = self.half(base, 0)?;
                let codes = self.rgb(block, place, decode == EtcDecode::Rgb8A1)?;
                return Ok(vec![pack(self, codes)?]);
            }
            EtcDecode::Rgba8 => {
                let alpha = self.half(base, 0)?;
                let alpha = self.eac(alpha, place[0], Channel::Alpha)?;
                let color = self.half(base, 1)?;
                let [r, g, b, _] = self.rgb(color, place, false)?;
                return Ok(vec![pack(self, [r, g, b, alpha])?]);
            }
            EtcDecode::R11Unorm | EtcDecode::R11Snorm => {
                let red = self.half(base, 0)?;
                vec![self.eac(red, place[0], decode.channel())?]
            }
            EtcDecode::Rg11Unorm | EtcDecode::Rg11Snorm => {
                let red = self.half(base, 0)?;
                let red = self.eac(red, place[0], decode.channel())?;
                let green = self.half(base, 1)?;
                vec![red, self.eac(green, place[0], decode.channel())?]
            }
        };
        let channel = decode.channel();
        match texels {
            EtcTexels::Float => (channels.into_iter())
                .map(|v| self.value(v, channel))
                .collect(),
            EtcTexels::Compact => {
                let mut codes = Vec::with_capacity(channels.len());
                for v in channels {
                    codes.push(self.code(v, channel)?);
                }
                match codes[..] {
                    [r, g] => Ok(vec![self.join(g, 16, r)?]),
                    _ => Ok(codes),
                }
            }
        }
    }
}

/// The module of `decode`'s blocks into `texels` as a typed module, before serialization: the
/// module [`etc_decode_module`](super::etc_decode_module) serializes, which the CPU interpreter
/// runs. Its bindings are the block decode's (`BlockDecodeModule`), its shape
/// `BlockDecodeModule::shape`'s.
pub(super) fn typed_module(decode: EtcDecode, texels: EtcTexels) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let blocks = words_buffer(&mut b, BLOCKS, u32_)?;
    let texel_words = words_buffer(&mut b, TEXELS, u32_)?;
    let pairs: Vec<u32> = (MODIFIERS.iter())
        .flat_map(|&(a, b)| [a as u32, b as u32])
        .collect();
    let modifiers = table(&mut b, u32_, &pairs)?;
    let distances = table(&mut b, u32_, &DISTANCES.map(|d| d as u32))?;
    let eac: Vec<u32> = EAC_MODIFIERS.iter().flatten().map(|&m| m as u32).collect();
    let eac = table(&mut b, u32_, &eac)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = scalar(&mut f, u32_, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let z = f.composite_extract(invocation, &[2])?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = scalar(f, u32_, u64::from(index))?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let width = field(&mut f, 0)?;
    let height = field(&mut f, 1)?;
    let depth = field(&mut f, 2)?;
    let across = field(&mut f, 3)?;
    let down = field(&mut f, 4)?;
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, height)?;
    let in_z = f.binary(BinOp::ULessThan, z, depth)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let inside = f.binary(BinOp::LogicalAnd, inside, in_z)?;
    // The texel's index through the region, `(z · height + y) · width + x`.
    let t = f.binary(BinOp::IMul, z, height)?;
    let t = f.binary(BinOp::IAdd, t, y)?;
    let t = f.binary(BinOp::IMul, t, width)?;
    let t = f.binary(BinOp::IAdd, t, x)?;
    // An `R11` format's compact texels pair up in words: the even texel's invocation writes both.
    let paired =
        texels == EtcTexels::Compact && matches!(decode, EtcDecode::R11Unorm | EtcDecode::R11Snorm);
    let writes = match paired {
        true => {
            let one = scalar(&mut f, u32_, 1)?;
            let parity = f.binary(BinOp::BitwiseAnd, t, one)?;
            let even = f.binary(BinOp::IEqual, parity, zero)?;
            f.binary(BinOp::LogicalAnd, inside, even)?
        }
        false => inside,
    };
    let blocks = f.module().global_value(blocks)?;
    let texel_words = f.module().global_value(texel_words)?;
    let decoded: ArmFn<'_, ()> = Box::new(move |f, _| {
        let mut e = E {
            f,
            u32_,
            zero,
            blocks,
            across,
            down,
            modifiers,
            distances,
            eac,
        };
        let here = At { x, y, z };
        let mut words = e.texel(decode, texels, here)?;
        if paired {
            // The next texel: along the row, else the next row's first, else the next image's.
            let next_x = e.opk(BinOp::IAdd, x, 1)?;
            let wraps_x = e.op(BinOp::IEqual, next_x, width)?;
            let next_x = e.sel(wraps_x, zero, next_x)?;
            let down_one = e.opk(BinOp::IAdd, y, 1)?;
            let next_y = e.sel(wraps_x, down_one, y)?;
            let wraps_y = e.op(BinOp::IEqual, next_y, height)?;
            let next_y = e.sel(wraps_y, zero, next_y)?;
            let deeper = e.opk(BinOp::IAdd, z, 1)?;
            let next_z = e.sel(wraps_y, deeper, z)?;
            let there = At {
                x: next_x,
                y: next_y,
                z: next_z,
            };
            // Past the region's last texel there is none: a block of the first image stands in,
            // and its code is not written.
            let exists = e.op(BinOp::ULessThan, next_z, depth)?;
            let safe_z = e.sel(exists, next_z, zero)?;
            let next = e.texel(decode, texels, At { z: safe_z, ..there })?;
            let next = e.sel(exists, next[0], zero)?;
            words[0] = e.join(next, 16, words[0])?;
        }
        let count = words.len() as u32;
        let at = match paired {
            true => e.opk(BinOp::ShiftRightLogical, t, 1)?,
            false => e.opk(BinOp::IMul, t, count)?,
        };
        for (k, word) in (0u32..).zip(words) {
            let index = e.opk(BinOp::IAdd, at, k)?;
            let ptr = e.f.access_chain(texel_words, &[zero, index])?;
            e.f.store(ptr, word, MemAccess::default())?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), writes, decoded, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: BlockDecodeModule::LOCAL_SIZE,
            y: BlockDecodeModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

#[cfg(test)]
mod tests {
    use super::typed_module;
    use crate::bind::block_decode::{BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
    use crate::bind::{EtcDecode, EtcTexels};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// `n` bytes of the xorshift64* stream from `seed`.
    fn stream(seed: u64, n: usize) -> Vec<u8> {
        let mut x = seed;
        let mut out = Vec::with_capacity(n + 8);
        while out.len() < n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            out.extend_from_slice(&x.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
        }
        out.truncate(n);
        out
    }

    #[test]
    fn the_etc_decode_kernels_are_valid_on_every_committed_profile() {
        let profiles = committed_profiles().expect("profiles");
        for decode in EtcDecode::ALL {
            for texels in [EtcTexels::Compact, EtcTexels::Float] {
                for (_, p) in &profiles {
                    assert_valid(typed_module(decode, texels).expect("module"), p);
                }
            }
        }
    }

    /// The kernel's texels of a region `size` of `decode`'s `blocks` in `texels`; the texel
    /// buffer's bytes past the region's words stay unwritten.
    fn kernel_texels(
        decode: EtcDecode,
        texels: EtcTexels,
        blocks: &[u8],
        size: [u32; 3],
    ) -> Vec<u8> {
        let m = typed_module(decode, texels).expect("module");
        let [w, h, d] = size;
        let length = (w * h * d * decode.texel_bytes(texels)) as usize;
        let words = length.next_multiple_of(4);
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(SHAPE),
                    address: None,
                    bytes: BlockDecodeModule::shape(size).to_vec(),
                },
                Buffer {
                    slot: Some(BLOCKS),
                    address: None,
                    bytes: blocks.to_vec(),
                },
                Buffer {
                    slot: Some(TEXELS),
                    address: None,
                    bytes: vec![0xcd; words + 64],
                },
            ],
            push: Vec::new(),
            spec: Default::default(),
        };
        let groups = |n: u32| n.div_ceil(BlockDecodeModule::LOCAL_SIZE);
        let cfg = Config {
            order: Order::Forward,
            workgroups: [groups(w) + 1, groups(h), d + 1],
            step_limit: 1 << 28,
            ..Config::default()
        };
        let mut out = run(&m, &cfg, res).expect("run").buffers;
        let written = out.swap_remove(2);
        assert!(
            written[words..].iter().all(|&b| b == 0xcd),
            "{decode:?} {texels:?}"
        );
        // An odd count of 2-byte texels leaves the last word's high half zero.
        assert!(
            written[length..words].iter().all(|&b| b == 0),
            "{decode:?} {texels:?}"
        );
        written[..length].to_vec()
    }

    /// The kernel writes the CPU's decode (Metal's, `random_blocks_decode_as_metal_reads_them`)
    /// on every decode in both texels: random blocks of whole and partial blocks, two images, and
    /// regions smaller than a block, an odd count of texels among them.
    #[test]
    fn the_etc_kernel_is_the_decode_on_every_texel() {
        for (k, decode) in EtcDecode::ALL.into_iter().enumerate() {
            for texels in [EtcTexels::Compact, EtcTexels::Float] {
                for size in [[16u32, 12, 2], [13, 9, 1], [7, 5, 2], [3, 3, 1], [1, 1, 1]] {
                    let [w, h, d] = size;
                    let n = (w.div_ceil(4) * h.div_ceil(4) * d * decode.block_bytes()) as usize;
                    let blocks = stream(0x51ED_270B_0000 + k as u64, n);
                    let cpu = decode.decode(texels, &blocks, size);
                    let gpu = kernel_texels(decode, texels, &blocks, size);
                    assert_eq!(gpu, cpu, "{decode:?} {texels:?} {size:?}");
                }
            }
        }
    }
}
