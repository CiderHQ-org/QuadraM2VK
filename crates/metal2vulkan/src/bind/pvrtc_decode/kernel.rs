//! The GPU's decode of PVRTC blocks into a decoded twin's texels: the module
//! [`pvrtc_decode_module`](super::pvrtc_decode_module) serializes, the same decode as the CPU's
//! ([`PvrtcDecode::decode_rows`]) in integer operations alone.
//!
//! One invocation per texel of a band of the level's rows reads the four blocks whose centres
//! surround it, upscales colors A and B, reads its own weight (a 2-bit-a-pixel texel of mode 1
//! off the checkerboard its four neighbours' stored weights, from the blocks they lie in) and
//! writes the blend, straight-line; every block index wraps at the image's blocks.

use super::{PvrtcDecode, PUNCH_WEIGHTS, WEIGHTS};
use crate::bind::block_decode::{table, BlockDecodeModule, BLOCKS, SHAPE, TEXELS};
use crate::bind::ycbcr_convert::{scalar, words_buffer};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ExecMode, ExecModel, IoDir, MemAccess, Type, TypedModule,
};

/// The decode of one texel: the function it builds in, and what every step reads.
struct P<'a, 'b> {
    f: &'a mut FunctionBuilder<'b>,
    u32_: TypeId,
    zero: Value,
    /// The blocks' words.
    blocks: Value,
    /// The weights of B of a 2-bit index, then of a punch-through one (`WEIGHTS`,
    /// `PUNCH_WEIGHTS`).
    weights: GlobalId,
}

impl P<'_, '_> {
    fn k(&mut self, v: u32) -> BResult<Value> {
        scalar(self.f, self.u32_, u64::from(v))
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

    fn mul(&mut self, a: Value, b: Value) -> BResult<Value> {
        self.op(BinOp::IMul, a, b)
    }

    fn sel(&mut self, c: Value, a: Value, b: Value) -> BResult<Value> {
        self.f.select(c, a, b)
    }

    /// Whether bit `at` of `v` is set.
    fn bit(&mut self, v: Value, at: u32) -> BResult<Value> {
        let field = self.field(v, at, 1)?;
        self.opk(BinOp::INotEqual, field, 0)
    }

    /// `(v >> at) & (2^n - 1)` for a constant field.
    fn field(&mut self, v: Value, at: u32, n: u32) -> BResult<Value> {
        let shifted = self.opk(BinOp::ShiftRightLogical, v, at)?;
        self.opk(BinOp::BitwiseAnd, shifted, (1 << n) - 1)
    }

    /// Entry `index` of the `Private` table `table`.
    fn entry(&mut self, table: GlobalId, index: Value) -> BResult<Value> {
        let table = self.f.module().global_value(table)?;
        let p = self.f.access_chain(table, &[index])?;
        self.f.load(p, MemAccess::default())
    }

    /// Word `index` of the blocks.
    fn word(&mut self, index: Value) -> BResult<Value> {
        let p = self.f.access_chain(self.blocks, &[self.zero, index])?;
        self.f.load(p, MemAccess::default())
    }

    /// The low and high words of block `index`.
    fn block(&mut self, index: Value) -> BResult<[Value; 2]> {
        let low = self.opk(BinOp::ShiftLeftLogical, index, 1)?;
        let high = self.opk(BinOp::IAdd, low, 1)?;
        Ok([self.word(low)?, self.word(high)?])
    }

    /// A 4-bit channel widened to 5 bits, `c << 1 | c >> 3`.
    fn four(&mut self, c: Value) -> BResult<Value> {
        let up = self.opk(BinOp::ShiftLeftLogical, c, 1)?;
        let down = self.opk(BinOp::ShiftRightLogical, c, 3)?;
        self.op(BinOp::BitwiseOr, up, down)
    }

    /// Colors A and B of a block whose high word is `high`, each red, green and blue of 5 bits
    /// and alpha of 4 (`super::colors`).
    fn colors(&mut self, high: Value) -> BResult<[[Value; 4]; 2]> {
        let a = self.opk(BinOp::BitwiseAnd, high, 0xffff)?;
        let b = self.opk(BinOp::ShiftRightLogical, high, 16)?;
        let fifteen = self.k(15)?;
        let translucent_alpha = |p: &mut Self, c: Value| -> BResult<Value> {
            let alpha = p.field(c, 12, 3)?;
            p.opk(BinOp::ShiftLeftLogical, alpha, 1)
        };
        // B: RGB555 where opaque, else A3R4G4B4.
        let b_opaque = self.bit(b, 15)?;
        let b_red = self.field(b, 10, 5)?;
        let b_green = self.field(b, 5, 5)?;
        let b_blue = self.field(b, 0, 5)?;
        let tr = self.field(b, 8, 4)?;
        let tr = self.four(tr)?;
        let tg = self.field(b, 4, 4)?;
        let tg = self.four(tg)?;
        let tb = self.field(b, 0, 4)?;
        let tb = self.four(tb)?;
        let ta = translucent_alpha(self, b)?;
        let color_b = [
            self.sel(b_opaque, b_red, tr)?,
            self.sel(b_opaque, b_green, tg)?,
            self.sel(b_opaque, b_blue, tb)?,
            self.sel(b_opaque, fifteen, ta)?,
        ];
        // A: RGB554 where opaque, else A3R4G4B3 (bit 0 is the mode bit).
        let a_opaque = self.bit(a, 15)?;
        let a_red = self.field(a, 10, 5)?;
        let a_green = self.field(a, 5, 5)?;
        let a_blue = self.field(a, 1, 4)?;
        let a_blue = self.four(a_blue)?;
        let tr = self.field(a, 8, 4)?;
        let tr = self.four(tr)?;
        let tg = self.field(a, 4, 4)?;
        let tg = self.four(tg)?;
        let blue3 = self.field(a, 1, 3)?;
        let up = self.opk(BinOp::ShiftLeftLogical, blue3, 2)?;
        let down = self.opk(BinOp::ShiftRightLogical, blue3, 1)?;
        let tb = self.op(BinOp::BitwiseOr, up, down)?;
        let ta = translucent_alpha(self, a)?;
        let color_a = [
            self.sel(a_opaque, a_red, tr)?,
            self.sel(a_opaque, a_green, tg)?,
            self.sel(a_opaque, a_blue, tb)?,
            self.sel(a_opaque, fifteen, ta)?,
        ];
        Ok([color_a, color_b])
    }
}

/// Where a texel's blocks lie: the image's first block, and its blocks across and down.
#[derive(Clone, Copy)]
struct Grid {
    base: Value,
    across: Value,
    down: Value,
}

impl Grid {
    /// The index of block `(bx, by)` of the image (each inside the grid).
    fn index(self, p: &mut P<'_, '_>, bx: Value, by: Value) -> BResult<Value> {
        let row = p.mul(by, self.across)?;
        let at = p.add(row, bx)?;
        p.add(self.base, at)
    }
}

/// The weight of B, in eighths, a 2-bit-a-pixel block stores at texel `(x, y)` of the image's
/// blocks (inside them; `super::Level::stored`).
fn stored(p: &mut P<'_, '_>, grid: Grid, x: Value, y: Value) -> BResult<Value> {
    let bx = p.opk(BinOp::UDiv, x, 8)?;
    let by = p.opk(BinOp::UDiv, y, 4)?;
    let index = grid.index(p, bx, by)?;
    let [low, high] = p.block(index)?;
    let lx = p.opk(BinOp::BitwiseAnd, x, 7)?;
    let ly = p.opk(BinOp::BitwiseAnd, y, 3)?;
    let ly8 = p.opk(BinOp::ShiftLeftLogical, ly, 3)?;
    let at = p.add(ly8, lx)?;
    // Mode 0: one bit a texel, 0 or 8 eighths.
    let one = p.op(BinOp::ShiftRightLogical, low, at)?;
    let one = p.opk(BinOp::BitwiseAnd, one, 1)?;
    let one = p.opk(BinOp::ShiftLeftLogical, one, 3)?;
    // Mode 1: bit 21 into bit 20 where bit 0 is set, then bit 1 into bit 0; the checkerboard's
    // texels hold 2-bit values in turn.
    let flagged = p.bit(low, 0)?;
    let cleared = p.opk(BinOp::BitwiseAnd, low, !(1 << 20))?;
    let b21 = p.field(low, 21, 1)?;
    let b21 = p.opk(BinOp::ShiftLeftLogical, b21, 20)?;
    let moved = p.op(BinOp::BitwiseOr, cleared, b21)?;
    let m = p.sel(flagged, moved, low)?;
    let cleared = p.opk(BinOp::BitwiseAnd, m, !1)?;
    let b1 = p.field(m, 1, 1)?;
    let m = p.op(BinOp::BitwiseOr, cleared, b1)?;
    let k = p.opk(BinOp::ShiftRightLogical, at, 1)?;
    let k2 = p.opk(BinOp::ShiftLeftLogical, k, 1)?;
    let index = p.op(BinOp::ShiftRightLogical, m, k2)?;
    let index = p.opk(BinOp::BitwiseAnd, index, 3)?;
    let two = p.entry(p.weights, index)?;
    let mode = p.bit(high, 0)?;
    p.sel(mode, two, one)
}

/// `(v + span - 1) mod span` and `(v + 1) mod span`: the coordinates before and after `v`,
/// wrapping inside `span`.
fn neighbours(p: &mut P<'_, '_>, v: Value, span: Value) -> BResult<[Value; 2]> {
    let before = p.add(v, span)?;
    let before = p.opk(BinOp::ISub, before, 1)?;
    let before = p.op(BinOp::UMod, before, span)?;
    let after = p.opk(BinOp::IAdd, v, 1)?;
    let after = p.op(BinOp::UMod, after, span)?;
    Ok([before, after])
}

/// The codes of texel `(x, y)` of an image of `decode` on `grid` (`super::Level::texel`).
fn texel(
    p: &mut P<'_, '_>,
    decode: PvrtcDecode,
    grid: Grid,
    x: Value,
    y: Value,
) -> BResult<[Value; 4]> {
    let [bw, bh] = decode.block_extent();
    // The four blocks whose centres surround the texel: `(x + bw / 2) / bw` and the one before,
    // wrapping, with the texel's place `u` past the first's centre.
    let corner = |p: &mut P<'_, '_>, v: Value, b: u32, count: Value| -> BResult<[Value; 3]> {
        let shifted = p.opk(BinOp::IAdd, v, b / 2)?;
        let q = p.opk(BinOp::UDiv, shifted, b)?;
        let u = p.opk(BinOp::UMod, shifted, b)?;
        let first = p.add(q, count)?;
        let first = p.opk(BinOp::ISub, first, 1)?;
        let first = p.op(BinOp::UMod, first, count)?;
        let second = p.op(BinOp::UMod, q, count)?;
        Ok([first, second, u])
    };
    let [x0, x1, u] = corner(p, x, bw, grid.across)?;
    let [y0, y1, v] = corner(p, y, bh, grid.down)?;
    let bwk = p.k(bw)?;
    let bhk = p.k(bh)?;
    let ru = p.op(BinOp::ISub, bwk, u)?;
    let rv = p.op(BinOp::ISub, bhk, v)?;
    let mut sums = [[p.zero; 4]; 2];
    for (bx, by, wx, wy) in [
        (x0, y0, ru, rv),
        (x1, y0, u, rv),
        (x0, y1, ru, v),
        (x1, y1, u, v),
    ] {
        let index = grid.index(p, bx, by)?;
        let [_, high] = p.block(index)?;
        let colors = p.colors(high)?;
        let weight = p.mul(wx, wy)?;
        for (sum, color) in sums.iter_mut().zip(colors) {
            for (s, c) in sum.iter_mut().zip(color) {
                let term = p.mul(c, weight)?;
                *s = p.add(*s, term)?;
            }
        }
    }
    // Each sum widened to 8 bits.
    let mut up = [[p.zero; 4]; 2];
    for (out, sum) in up.iter_mut().zip(sums) {
        for (ch, (o, s)) in out.iter_mut().zip(sum).enumerate() {
            let (a, b) = match (decode.two_bits, ch < 3) {
                (false, true) => (1, 6),
                (false, false) => (0, 4),
                (true, true) => (2, 7),
                (true, false) => (1, 5),
            };
            let first = p.opk(BinOp::ShiftRightLogical, s, a)?;
            let second = p.opk(BinOp::ShiftRightLogical, s, b)?;
            *o = p.add(first, second)?;
        }
    }
    // The texel's own block.
    let bx = p.opk(BinOp::UDiv, x, bw)?;
    let by = p.opk(BinOp::UDiv, y, bh)?;
    let own = grid.index(p, bx, by)?;
    let [low, high] = p.block(own)?;
    let mode = p.bit(high, 0)?;
    let no = p.boolean(false)?;
    let (weight, punch) = match decode.two_bits {
        false => {
            let lx = p.opk(BinOp::BitwiseAnd, x, 3)?;
            let ly = p.opk(BinOp::BitwiseAnd, y, 3)?;
            let ly4 = p.opk(BinOp::ShiftLeftLogical, ly, 2)?;
            let at = p.add(ly4, lx)?;
            let at = p.opk(BinOp::ShiftLeftLogical, at, 1)?;
            let index = p.op(BinOp::ShiftRightLogical, low, at)?;
            let index = p.opk(BinOp::BitwiseAnd, index, 3)?;
            let punch_index = p.opk(BinOp::IAdd, index, 4)?;
            let entry = p.sel(mode, punch_index, index)?;
            let weight = p.entry(p.weights, entry)?;
            let two = p.opk(BinOp::IEqual, index, 2)?;
            (weight, p.op(BinOp::LogicalAnd, mode, two)?)
        }
        true => {
            let width = p.opk(BinOp::IMul, grid.across, bw)?;
            let height = p.opk(BinOp::IMul, grid.down, bh)?;
            let [left, right] = neighbours(p, x, width)?;
            let [above, below] = neighbours(p, y, height)?;
            let centre = stored(p, grid, x, y)?;
            let s_above = stored(p, grid, x, above)?;
            let s_below = stored(p, grid, x, below)?;
            let s_left = stored(p, grid, left, y)?;
            let s_right = stored(p, grid, right, y)?;
            // The four neighbours' average, `(s + 2) / 4`.
            let four = p.add(s_above, s_below)?;
            let four = p.add(four, s_left)?;
            let four = p.add(four, s_right)?;
            let four = p.opk(BinOp::IAdd, four, 2)?;
            let four = p.opk(BinOp::ShiftRightLogical, four, 2)?;
            // The vertical or horizontal pair's, `(a + b + 1) / 2`, by bit 20 as stored.
            let vertical = p.add(s_above, s_below)?;
            let vertical = p.opk(BinOp::IAdd, vertical, 1)?;
            let vertical = p.opk(BinOp::ShiftRightLogical, vertical, 1)?;
            let horizontal = p.add(s_left, s_right)?;
            let horizontal = p.opk(BinOp::IAdd, horizontal, 1)?;
            let horizontal = p.opk(BinOp::ShiftRightLogical, horizontal, 1)?;
            let up_down = p.bit(low, 20)?;
            let pair = p.sel(up_down, vertical, horizontal)?;
            let flagged = p.bit(low, 0)?;
            let between = p.sel(flagged, pair, four)?;
            let parity = p.op(BinOp::BitwiseXor, x, y)?;
            let odd = p.bit(parity, 0)?;
            let interpolated = p.op(BinOp::LogicalAnd, mode, odd)?;
            (p.sel(interpolated, between, centre)?, no)
        }
    };
    let eight = p.k(8)?;
    let rest = p.op(BinOp::ISub, eight, weight)?;
    let mut codes = [p.zero; 4];
    for (ch, code) in codes.iter_mut().enumerate() {
        let a = p.mul(up[0][ch], rest)?;
        let b = p.mul(up[1][ch], weight)?;
        let sum = p.add(a, b)?;
        *code = p.opk(BinOp::ShiftRightLogical, sum, 3)?;
    }
    codes[3] = match decode.alpha {
        true => p.sel(punch, p.zero, codes[3])?,
        false => p.k(255)?,
    };
    Ok(codes)
}

/// The module of `decode` as a typed module, before serialization: the module
/// [`pvrtc_decode_module`](super::pvrtc_decode_module) serializes, which the CPU interpreter
/// runs. Its bindings are the block decode's (`BlockDecodeModule`), its shape
/// [`PvrtcDecode::shape`]'s.
pub(super) fn typed_module(decode: PvrtcDecode) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let blocks = words_buffer(&mut b, BLOCKS, u32_)?;
    let texel_words = words_buffer(&mut b, TEXELS, u32_)?;
    let weights = table(&mut b, u32_, &[WEIGHTS, PUNCH_WEIGHTS].concat())?;
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
    let rows = field(&mut f, 1)?;
    let images = field(&mut f, 2)?;
    let first = field(&mut f, 3)?;
    let height = field(&mut f, 4)?;
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, rows)?;
    let in_z = f.binary(BinOp::ULessThan, z, images)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let inside = f.binary(BinOp::LogicalAnd, inside, in_z)?;
    let blocks = f.module().global_value(blocks)?;
    let texel_words = f.module().global_value(texel_words)?;
    let decoded: ArmFn<'_, ()> = Box::new(move |f, _| {
        let mut p = P {
            f,
            u32_,
            zero,
            blocks,
            weights,
        };
        let [bw, bh] = decode.block_extent();
        let across = p.opk(BinOp::IAdd, width, bw - 1)?;
        let across = p.opk(BinOp::UDiv, across, bw)?;
        let down = p.opk(BinOp::IAdd, height, bh - 1)?;
        let down = p.opk(BinOp::UDiv, down, bh)?;
        let per_image = p.mul(across, down)?;
        let base = p.mul(z, per_image)?;
        let grid = Grid { base, across, down };
        let level_y = p.add(first, y)?;
        let codes = texel(&mut p, decode, grid, x, level_y)?;
        let mut packed = zero;
        for (ch, code) in (0u32..).zip(codes) {
            let placed = p.opk(BinOp::ShiftLeftLogical, code, 8 * ch)?;
            packed = p.op(BinOp::BitwiseOr, packed, placed)?;
        }
        // The texel's word, `(z · rows + y) · width + x`.
        let at = p.mul(z, rows)?;
        let at = p.add(at, y)?;
        let at = p.mul(at, width)?;
        let at = p.add(at, x)?;
        let ptr = p.f.access_chain(texel_words, &[zero, at])?;
        p.f.store(ptr, packed, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, decoded, None)?;
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
