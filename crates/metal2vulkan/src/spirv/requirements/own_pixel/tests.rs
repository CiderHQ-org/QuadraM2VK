//! [`own_pixel_textures`] on hand-built fragment modules: the own-pixel forms, and look-alikes
//! that differ from one of them in one step.

use super::own_pixel_textures;
use crate::budget::{Budget, Meter};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecModel, GlslOp, ImageOperands, ImageQuery,
    ImageType, ImageUsage, IoDir, MemAccess, StorageClass, Type, TypedModule, UnOp,
};
use crate::tests::constant_blob_footprint::held_during;

/// How a module's fetch of texture slot 0 is shaped. Its coordinate is Metal's saturating
/// `uint2(FragCoord.xy)` as the translator lowers it ([`saturating`]) unless the form says
/// otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// The translator's form, its NaN test by the bits: own pixel.
    Saturating,
    /// Its NaN test `OpIsNan`: own pixel.
    SaturatingIsNan,
    /// No NaN select (the lowering's form where `NClamp` gives a NaN Metal's 0): own pixel.
    SaturatingNoNanSelect,
    /// Each lane loaded through an access chain of `FragCoord` and converted alone: own pixel.
    PerLane,
    /// Of `floor(FragCoord.xy) + 0.5`, the pixel-center form: own pixel.
    PixelCenter,
    /// The shader's clamp: `min(coord, size(t, min(0, levels(t) - 1)) - 1)` at that level: own
    /// pixel, clamped into the view.
    Clamped,
    /// A bare `OpConvertFToU`: undefined at a helper invocation's negative coordinate.
    Truncated,
    /// The NaN test's mask `0xffffffff`: a negative float's bits pass it.
    SaturatingMask,
    /// The clamp's upper bound 1024: below a framebuffer's coordinates.
    SaturatingLowClamp,
    /// The clamp's lower bound 1.
    SaturatingClampFloor,
    /// The limit 1024: a pixel's coordinate passes it.
    SaturatingLowLimit,
    /// The NaN select's arms the other way: the zeros wherever no NaN is.
    SaturatingArmsSwapped,
    /// Of `FragCoord.yx`: another pixel.
    Swapped,
    /// Plus `(1, 0)`: the next pixel.
    Shifted,
    /// Of `FragCoord.xy * 0.5`: another pixel.
    Scaled,
    /// Of `floor(FragCoord.xy) + 0.25`: not the pixel-center form (the same pixel, not proven).
    QuarterCenter,
    /// Level 1.
    LevelOne,
    /// A constant offset of `(1, 0)`.
    Offset,
    /// `min(coord, (3, 3))`: a clamp below a constant, not the view's size.
    ClampedConstant,
    /// `min(coord, size(u) - 1)` of another slot's image `u`.
    ClampedOtherSlot,
    /// `min(coord, size(t, 1) - 1)`: the size of level 1.
    ClampedLevelOne,
    /// A 2D-array image's layer 0 at the pixel.
    Arrayed,
}

/// The NaN test of [`saturating`].
#[derive(Clone, Copy, Debug)]
enum NanTest {
    /// `(bits(x) & mask) > 0x7f800000`.
    Bits(u64),
    IsNan,
    /// No NaN select.
    Absent,
}

/// How [`saturating`] differs from the translator's form ([`METAL`]).
#[derive(Clone, Copy, Debug)]
struct Saturation {
    nan: NanTest,
    lo: f32,
    hi: f32,
    limit: f32,
    swapped: bool,
}

const METAL: Saturation = Saturation {
    nan: NanTest::Bits(0x7fff_ffff),
    lo: 0.0,
    hi: 4_294_967_040.0,
    limit: 4_294_967_296.0,
    swapped: false,
};

/// The types [`saturating`] converts between: a float of `lanes` lanes, and an unsigned one.
#[derive(Clone, Copy)]
struct Types {
    f32_: TypeId,
    float: TypeId,
    u32_: TypeId,
    uint: TypeId,
    lanes: usize,
}

fn scalar(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> Value {
    let c = f
        .module()
        .constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })
        .expect("constant");
    f.module().const_value(c).expect("value")
}

/// The constant of type `ty` (of scalar `elem`) holding `bits`: a scalar where `bits` is one.
fn splat(f: &mut FunctionBuilder<'_>, ty: TypeId, elem: TypeId, bits: &[u64]) -> Value {
    if let [one] = bits {
        return scalar(f, elem, *one);
    }
    let parts: Vec<_> = bits
        .iter()
        .map(|&b| {
            f.module()
                .constant(Constant::Scalar {
                    ty: elem,
                    bits: ConstBits(b),
                })
                .expect("part")
        })
        .collect();
    let c = f
        .module()
        .constant(Constant::Composite {
            ty,
            parts: parts.into(),
        })
        .expect("composite");
    f.module().const_value(c).expect("value")
}

/// Metal's saturating conversion of the float `x` to unsigned as `lower::arith::math` lowers it,
/// but as `how` differs: `NClamp` into `[0, largest float below 2^32]`, `OpConvertFToU`, the
/// maximum where `x >= 2^32`, and 0 where `x` is a NaN.
fn saturating(f: &mut FunctionBuilder<'_>, x: Value, t: Types, how: Saturation) -> Value {
    let floats = |f: &mut FunctionBuilder<'_>, v: f32| {
        splat(f, t.float, t.f32_, &vec![u64::from(v.to_bits()); t.lanes])
    };
    let uints = |f: &mut FunctionBuilder<'_>, v: u64| splat(f, t.uint, t.u32_, &vec![v; t.lanes]);
    let lo = floats(f, how.lo);
    let hi = floats(f, how.hi);
    let clamped = f.glsl(GlslOp::NClamp, &[x, lo, hi]).expect("clamp");
    let r = f
        .convert(ConvOp::ConvertFToU, clamped, t.uint)
        .expect("convert");
    let limit = floats(f, how.limit);
    let over = f
        .binary(BinOp::FOrdGreaterThanEqual, x, limit)
        .expect("over");
    let max = uints(f, 0xffff_ffff);
    let r = f.select(over, max, r).expect("saturate");
    let nan = match how.nan {
        NanTest::Absent => return r,
        NanTest::IsNan => f.unary(UnOp::IsNan, x).expect("is nan"),
        NanTest::Bits(mask) => {
            let b = f.convert(ConvOp::Bitcast, x, t.uint).expect("bits");
            let mask = uints(f, mask);
            let infinity = uints(f, 0x7f80_0000);
            let m = f.binary(BinOp::BitwiseAnd, b, mask).expect("magnitude");
            f.binary(BinOp::UGreaterThan, m, infinity).expect("nan")
        }
    };
    let zeros = uints(f, 0);
    match how.swapped {
        false => f.select(nan, zeros, r).expect("nan select"),
        true => f.select(nan, r, zeros).expect("nan select"),
    }
}

/// A fragment that fetches texture slot 0 (and only queries slot 1) as `form` says, and returns.
fn module(form: Form) -> TypedModule {
    module_with(form, 0)
}

/// [`module`], with `fillers` float additions that touch no image before the fetch.
fn module_with(form: Form, fillers: usize) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let u32_ = b
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .expect("uint");
    let i32_ = b
        .ty(Type::Int {
            width: 32,
            signed: true,
        })
        .expect("int");
    let f32_ = b.ty(Type::Float { width: 32 }).expect("float");
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 }).expect("vec2");
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 }).expect("vec4");
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 }).expect("uvec2");
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 }).expect("uvec3");
    let ivec2 = b.ty(Type::Vector { elem: i32_, len: 2 }).expect("ivec2");
    let image = |b: &mut ModuleBuilder, arrayed: bool| {
        b.ty(Type::Image(ImageType {
            sampled_type: f32_,
            dim: Dim::D2,
            depth: false,
            arrayed,
            multisampled: false,
            usage: ImageUsage::Sampled,
            format: crate::spv::ImageFormat::Unknown,
        }))
        .expect("image")
    };
    let t_ty = image(&mut b, form == Form::Arrayed);
    let u_ty = image(&mut b, false);
    let slot = |index| ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Texture,
        index,
    };
    let t = b
        .resource(slot(0), StorageClass::UniformConstant, t_ty)
        .expect("slot 0");
    let u = b
        .resource(slot(1), StorageClass::UniformConstant, u_ty)
        .expect("slot 1");
    let frag_coord = b
        .builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)
        .expect("FragCoord");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("main");
    let tv = f.module().global_value(t).expect("t");
    let uv = f.module().global_value(u).expect("u");
    let fc = f.module().global_value(frag_coord).expect("FragCoord");
    let img = f.load(tv, MemAccess::default()).expect("t image");
    let other = f.load(uv, MemAccess::default()).expect("u image");
    let zero = scalar(&mut f, u32_, 0);
    let one = scalar(&mut f, u32_, 1);
    let ones = splat(&mut f, uvec2, u32_, &[1, 1]);
    let position = f.load(fc, MemAccess::default()).expect("position");
    let xy = f.vector_shuffle(position, position, &[0, 1]).expect("xy");
    let pair = Types {
        f32_,
        float: vec2,
        u32_,
        uint: uvec2,
        lanes: 2,
    };
    let index =
        |f: &mut FunctionBuilder<'_>, v: Value, how: Saturation| saturating(f, v, pair, how);
    // The shader's clamp of a level below the image's level count, and of a coordinate below a
    // size less one.
    let clamped_level = |f: &mut FunctionBuilder<'_>, img: Value, level: Value| {
        let levels = f.query(ImageQuery::Levels, img).expect("levels");
        let top = f.binary(BinOp::ISub, levels, one).expect("top level");
        f.glsl(GlslOp::UMin, &[level, top]).expect("level")
    };
    let below = |f: &mut FunctionBuilder<'_>, img: Value, level: Value, coord: Value| {
        let size = f.query(ImageQuery::SizeLod(level), img).expect("size");
        let top = f.binary(BinOp::ISub, size, ones).expect("top");
        f.glsl(GlslOp::UMin, &[coord, top]).expect("clamp")
    };
    let mut lod = zero;
    let mut ops = ImageOperands::default();
    let coord = match form {
        Form::Saturating | Form::LevelOne | Form::Offset => index(&mut f, xy, METAL),
        Form::SaturatingIsNan => {
            let how = Saturation {
                nan: NanTest::IsNan,
                ..METAL
            };
            index(&mut f, xy, how)
        }
        Form::SaturatingNoNanSelect => {
            let how = Saturation {
                nan: NanTest::Absent,
                ..METAL
            };
            index(&mut f, xy, how)
        }
        Form::SaturatingMask => {
            let how = Saturation {
                nan: NanTest::Bits(0xffff_ffff),
                ..METAL
            };
            index(&mut f, xy, how)
        }
        Form::SaturatingLowClamp => index(
            &mut f,
            xy,
            Saturation {
                hi: 1024.0,
                ..METAL
            },
        ),
        Form::SaturatingClampFloor => index(&mut f, xy, Saturation { lo: 1.0, ..METAL }),
        Form::SaturatingLowLimit => {
            let how = Saturation {
                limit: 1024.0,
                ..METAL
            };
            index(&mut f, xy, how)
        }
        Form::SaturatingArmsSwapped => {
            let how = Saturation {
                swapped: true,
                ..METAL
            };
            index(&mut f, xy, how)
        }
        Form::PerLane => {
            let lane = |f: &mut FunctionBuilder<'_>, i: u32| {
                let at = scalar(f, u32_, u64::from(i));
                let p = f.access_chain(fc, &[at]).expect("lane pointer");
                let x = f.load(p, MemAccess::default()).expect("lane");
                let one_lane = Types {
                    f32_,
                    float: f32_,
                    u32_,
                    uint: u32_,
                    lanes: 1,
                };
                saturating(f, x, one_lane, METAL)
            };
            let x = lane(&mut f, 0);
            let y = lane(&mut f, 1);
            f.composite_construct(uvec2, &[x, y]).expect("coord")
        }
        Form::PixelCenter | Form::QuarterCenter => {
            let floor = f.glsl(GlslOp::Floor, &[xy]).expect("floor");
            let offset = match form {
                Form::PixelCenter => 0.5f32,
                _ => 0.25f32,
            };
            let bits = u64::from(offset.to_bits());
            let half = splat(&mut f, vec2, f32_, &[bits, bits]);
            let center = f.binary(BinOp::FAdd, floor, half).expect("center");
            index(&mut f, center, METAL)
        }
        Form::Clamped => {
            lod = clamped_level(&mut f, img, zero);
            let coord = index(&mut f, xy, METAL);
            below(&mut f, img, lod, coord)
        }
        Form::Truncated => f
            .convert(ConvOp::ConvertFToU, xy, uvec2)
            .expect("truncated"),
        Form::Swapped => {
            let yx = f.vector_shuffle(position, position, &[1, 0]).expect("yx");
            index(&mut f, yx, METAL)
        }
        Form::Shifted => {
            let coord = index(&mut f, xy, METAL);
            let step = splat(&mut f, uvec2, u32_, &[1, 0]);
            f.binary(BinOp::IAdd, coord, step).expect("shifted")
        }
        Form::Scaled => {
            let half = u64::from(0.5f32.to_bits());
            let scale = splat(&mut f, vec2, f32_, &[half, half]);
            let scaled = f.binary(BinOp::FMul, xy, scale).expect("scaled");
            index(&mut f, scaled, METAL)
        }
        Form::ClampedConstant => {
            let coord = index(&mut f, xy, METAL);
            let three = splat(&mut f, uvec2, u32_, &[3, 3]);
            f.glsl(GlslOp::UMin, &[coord, three]).expect("clamp")
        }
        Form::ClampedOtherSlot => {
            let coord = index(&mut f, xy, METAL);
            below(&mut f, other, zero, coord)
        }
        Form::ClampedLevelOne => {
            let coord = index(&mut f, xy, METAL);
            below(&mut f, img, one, coord)
        }
        Form::Arrayed => {
            let coord = index(&mut f, xy, METAL);
            let x = f.composite_extract(coord, &[0]).expect("x");
            let y = f.composite_extract(coord, &[1]).expect("y");
            f.composite_construct(uvec3, &[x, y, zero])
                .expect("layered")
        }
    };
    let mut sum = f.composite_extract(position, &[0]).expect("x");
    for _ in 0..fillers {
        sum = f.binary(BinOp::FAdd, sum, sum).expect("filler");
    }
    if form == Form::LevelOne {
        lod = one;
    }
    if form == Form::Offset {
        ops.const_offset = Some(splat(&mut f, ivec2, i32_, &[1, 0]));
    }
    ops.lod = Some(lod);
    f.fetch(img, coord, &ops).expect("fetch");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, Vec::new())
        .expect("entry");
    b.finalize().expect("finalize")
}

/// A fetch at Metal's saturating `uint2` of `FragCoord`'s x and y, each form of it the translator
/// emits (either NaN test or none, lanes together or apart, the pixel-center form, the shader's
/// clamp into the view), reads the fragment's own pixel; one step off any of them does not: a bare
/// conversion, a saturating conversion whose NaN mask, clamp, limit or select differs, other
/// lanes, arithmetic, another center, another level, an offset, a clamp below anything but the
/// view's first level's size, an arrayed image. Slot 1, only queried, reads no texel.
#[test]
fn only_the_own_pixel_forms_qualify() {
    for (form, own) in [
        (Form::Saturating, true),
        (Form::SaturatingIsNan, true),
        (Form::SaturatingNoNanSelect, true),
        (Form::PerLane, true),
        (Form::PixelCenter, true),
        (Form::Clamped, true),
        (Form::Truncated, false),
        (Form::SaturatingMask, false),
        (Form::SaturatingLowClamp, false),
        (Form::SaturatingClampFloor, false),
        (Form::SaturatingLowLimit, false),
        (Form::SaturatingArmsSwapped, false),
        (Form::Swapped, false),
        (Form::Shifted, false),
        (Form::Scaled, false),
        (Form::QuarterCenter, false),
        (Form::LevelOne, false),
        (Form::Offset, false),
        (Form::ClampedConstant, false),
        (Form::ClampedOtherSlot, false),
        (Form::ClampedLevelOne, false),
        (Form::Arrayed, false),
    ] {
        let mut meter = Meter::new(&Budget::default());
        let slots = own_pixel_textures(&module(form), &mut meter).expect("analysis");
        assert_eq!(
            meter.ledger(),
            0,
            "{form:?}: the analysis releases what it charged"
        );
        let want: &[u32] = match own {
            true => &[0, 1],
            false => &[1],
        };
        assert_eq!(slots, want, "{form:?}");
    }
}

/// What the analysis holds beyond what it charges the byte ledger does not grow with the
/// function: no map over its instructions, which the ledger would not see (the translation's
/// allocator peak must stay within 10/7 of the ledger's). It releases every charge.
#[test]
fn what_the_analysis_holds_uncharged_does_not_grow_with_the_function() {
    let uncharged = |m: &TypedModule| {
        let mut meter = Meter::new(&Budget::default());
        let (slots, held) = held_during(|| own_pixel_textures(m, &mut meter));
        assert_eq!(slots.expect("analysis"), [0, 1]);
        assert_eq!(meter.ledger(), 0, "every charge released");
        held.saturating_sub(meter.ledger_peak())
    };
    let small = uncharged(&module_with(Form::Clamped, 0));
    let large = uncharged(&module_with(Form::Clamped, 20_000));
    assert!(
        large <= small + 1024,
        "{large} bytes uncharged over 20,000 more instructions, {small} without them"
    );
}
