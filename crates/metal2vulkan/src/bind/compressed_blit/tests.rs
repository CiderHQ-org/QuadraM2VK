use super::{typed_module, CompressedBlit, CompressedBlitModule, WORDS};
use crate::metal::PixelFormat;
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

const RULES: [CompressedBlit; 3] = [
    CompressedBlit::ExtendedRangeFields,
    CompressedBlit::PackedFloatNan,
    CompressedBlit::SharedExponent,
];

#[test]
fn the_compressed_blit_kernel_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        for rule in RULES {
            assert_valid(typed_module(rule).expect("module"), &p);
        }
    }
}

/// Exactly `BGRA10_XR`, its `_sRGB` twin, `RG11B10Float` and `RGB9E5Float` take a rule: the
/// census of our probe found every other format's blits keep their bytes.
#[test]
fn four_formats_take_a_rule() {
    let ruled: Vec<PixelFormat> = PixelFormat::ALL
        .iter()
        .copied()
        .filter(|&pf| CompressedBlit::of(pf).is_some())
        .collect();
    assert_eq!(
        ruled,
        [
            PixelFormat::RG11B10Float,
            PixelFormat::RGB9E5Float,
            PixelFormat::BGRA10XR,
            PixelFormat::BGRA10XRSRGB,
        ]
    );
}

/// Words Metal's blits wrote for words they read (an Apple M4 under macOS
/// 27.0.1: a 64 x 64 texture of `ShaderRead` usage, random bytes blitted in and out), each the
/// `odd` word of its texel or not: the CPU rule answers each.
#[test]
fn the_rule_answers_metal_s_measured_words() {
    let measured = [
        // RG11B10Float: blue, green and red NaNs made quiet; an infinity and a number kept.
        (CompressedBlit::PackedFloatNan, 0x84be_acc9, 0x84bf_04c9),
        (CompressedBlit::PackedFloatNan, 0xfd9d_d04e, 0xfc1d_d04e),
        (CompressedBlit::PackedFloatNan, 0x0abf_b7a5, 0x0abf_07a5),
        (CompressedBlit::PackedFloatNan, 0xf838_dcc9, 0xf838_dcc9),
        (CompressedBlit::PackedFloatNan, 0x3384_077e, 0x3384_077e),
        // RGB9E5Float: the exponent lowered by 1 and by 2, and a texel already at its least.
        (CompressedBlit::SharedExponent, 0x6911_8405, 0x6223_080a),
        (CompressedBlit::SharedExponent, 0x3929_ac33, 0x3253_5866),
        (CompressedBlit::SharedExponent, 0x4944_2c69, 0x3d10_b1a4),
        (CompressedBlit::SharedExponent, 0x37e2_073a, 0x37e2_073a),
    ];
    for (rule, read, written) in measured {
        assert_eq!(rule.word(read, false), written, "{rule:?} {read:08x}");
    }
    // BGRA10_XR: (word, odd, written); alpha codes 179 and 1012 clamped to 384 and 894.
    for (read, odd, written) in [
        (0x1430_2089, false, 0x1400_2080),
        (0x5c3b_7bc8, false, 0x5c00_7bc0),
        (0xc911_b109, true, 0xc900_b100),
        (0x2ccd_6de4, true, 0x6000_6dc0),
        (0xfd04_0a1d, true, 0xdf80_0a00),
        (0x3f40_8b00, false, 0x3f40_8b00),
    ] {
        let rule = CompressedBlit::ExtendedRangeFields;
        assert_eq!(rule.word(read, odd), written, "{read:08x} odd {odd}");
    }
}

/// An `RGB9E5Float` texel keeps its value: every mantissa and exponent the rule gives decodes to
/// the value the texel read decodes to, at an exponent no texel of that value has below it.
#[test]
fn a_shared_exponent_texel_keeps_its_value_at_its_least_exponent() {
    let value = |w: u32| -> [f64; 3] {
        let e = f64::from(w >> 27) - 24.0;
        std::array::from_fn(|i| f64::from((w >> (9 * i)) & 0x1ff) * e.exp2())
    };
    let mut rng = Rng(93);
    for _ in 0..100_000 {
        let w = rng.next();
        let made = CompressedBlit::SharedExponent.word(w, false);
        assert_eq!(value(made), value(w), "{w:08x} -> {made:08x}");
        let largest = (0..3).map(|i| (made >> (9 * i)) & 0x1ff).max().unwrap();
        assert!(made >> 27 == 0 || largest >= 256, "{w:08x} -> {made:08x}");
    }
    assert_eq!(CompressedBlit::SharedExponent.word(0xf800_0000, false), 0);
}

/// A splitmix64 stream: every bit of each word random.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) as u32
    }
}

/// Every word the kernel of each rule rewrites is the CPU rule's, both invocation orders, over
/// random words and the edges (zero, all ones, each field's NaNs and infinities, exponent 0 and
/// 31), with invocations past the words writing nothing.
#[test]
fn the_kernel_is_the_rule_on_every_word() {
    let mut rng = Rng(0x0c0b_1170);
    let mut before: Vec<u32> = (0..4096).map(|_| rng.next()).collect();
    before.extend([
        0,
        u32::MAX,
        0x7c0,
        0x7e0,
        0x7c1,
        0x3e_0000,
        0x3f_0000,
        0xf800_0000,
        0xffc0_0000,
    ]);
    before.extend([
        0x07ff_ffff,
        0x0000_0001,
        0xf800_0001,
        0x5ff0_0000,
        0xe010_0000,
    ]);
    if before.len() % 2 == 1 {
        before.push(0x1234_5678);
    }
    let bytes: Vec<u8> = before.iter().flat_map(|w| w.to_le_bytes()).collect();
    let groups = (before.len() as u32).div_ceil(CompressedBlitModule::LOCAL_SIZE) + 2;
    for rule in RULES {
        let m = typed_module(rule).expect("module");
        let mut outs = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(WORDS),
                    address: None,
                    bytes: bytes.clone(),
                }],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [groups, 1, 1],
                ..Config::default()
            };
            outs.push(run(&m, &cfg, res).expect("run").buffers.swap_remove(0));
        }
        assert_eq!(outs[0], outs[1], "{rule:?}: order independent");
        let mut changed = 0;
        for (i, (&word, got)) in before.iter().zip(outs[0].chunks(4)).enumerate() {
            let got = u32::from_le_bytes([got[0], got[1], got[2], got[3]]);
            let odd = rule.texel_bytes() == 8 && i % 2 == 1;
            assert_eq!(got, rule.word(word, odd), "{rule:?} word {i}: {word:08x}");
            changed += usize::from(got != word);
        }
        assert!(changed > 0, "{rule:?}");
    }
}
