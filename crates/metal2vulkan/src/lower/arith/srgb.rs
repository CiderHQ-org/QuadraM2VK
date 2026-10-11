//! Metal's sRGB conversions of `air.pack`/`air.unpack` `unorm4x8_srgb`, exact on every device: both
//! go through Metal's twelve-bit tables, which A1 read in full, rather than the MSL transfer
//! functions (whose `GLSL.std.450 Pow` has the device's precision and misses Metal's table at bytes
//! where the transfer lands near a half step).
//!
//! The two tables are one `Private` array of 256 `u32` with its constant initializer, declared
//! once per module that uses it ([`declare`]): word `b` holds the decode step of byte `b` in its
//! low half and the first encode index of byte `b` in its high half.
//!
//! - Decoding byte `b` loads its step `k` ([`step`]), and `math::unpack` answers `k / 4095`
//!   correctly rounded ([`div_exact`](crate::lower::arith::exact_division::div_exact)), as
//!   Metal does.
//! - Encoding `x` rounds the exact product `x * 4095` to the index `i`
//!   ([`round_exact_product_lanes`](crate::lower::arith::math::round_exact_product_lanes)), and
//!   [`search`] finds the byte in the thresholds: eight dependent loads, branch-free.
//!
//! The products and quotients run once per vector in `math::pack` and `math::unpack`, with a
//! per-lane scale; only the table reads are per channel, since a `Private` array is indexed by a
//! scalar.
//!
//! A fragment function's write of a float to an sRGB color attachment reads its own table instead,
//! the **color table**: 256 `uvec4` entries in a buffer the binder fills with
//! [`color_write_table`]'s bytes, entry `b` holding byte `b`'s first encode index, its linear
//! value's bits, and byte `b - 1`'s. The write estimates the byte from the index's square root to
//! within a byte above Metal's ([`ENCODE_ESTIMATE`]) and reads that byte's entry once
//! ([`color_linear`]): the index against the entry's first index chooses between the two linear
//! values. The buffer is a read-only storage buffer, or a uniform buffer where the driver reads one
//! at different addresses faster ([`ColorTable`]).

use crate::device::profile::{DeviceProfile, DriverWorkaround};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, GlslOp, LayoutRule, MemAccess,
    StorageClass, Type,
};

/// The decode step of every byte `b`: Metal's `unpack_unorm4x8_srgb_to_float` answers `k / 4095`
/// (correctly rounded) for `k = SRGB_DECODE_STEPS[b]`, which is `round(decode(b / 255) * 4095)`
/// for the MSL transfer `decode(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4`
/// (every margin from a half step is at least `1.6e-4` step). Every color word of A1's
/// `srgb-decode-of-every-byte` (bcb3682b, 768 of 768) and the old engine's
/// `kernel_srgb_unpack_all_bytes` read these steps.
pub(crate) const SRGB_DECODE_STEPS: [u16; 256] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 40,
    42, 45, 48, 50, 53, 56, 59, 62, 66, 69, 72, 76, 79, 83, 87, 91, 95, 99, 103, 107, 112, 116,
    121, 126, 131, 136, 141, 146, 151, 156, 162, 168, 173, 179, 185, 191, 197, 204, 210, 216, 223,
    230, 237, 244, 251, 258, 265, 273, 280, 288, 296, 304, 312, 320, 329, 337, 346, 354, 363, 372,
    381, 390, 400, 409, 419, 428, 438, 448, 458, 469, 479, 490, 500, 511, 522, 533, 544, 555, 567,
    578, 590, 602, 614, 626, 639, 651, 664, 676, 689, 702, 715, 728, 742, 755, 769, 783, 797, 811,
    825, 840, 854, 869, 884, 899, 914, 929, 945, 960, 976, 992, 1008, 1024, 1041, 1057, 1074, 1091,
    1108, 1125, 1142, 1159, 1177, 1195, 1213, 1231, 1249, 1267, 1286, 1304, 1323, 1342, 1361, 1381,
    1400, 1420, 1440, 1459, 1480, 1500, 1520, 1541, 1562, 1582, 1603, 1625, 1646, 1668, 1689, 1711,
    1733, 1755, 1778, 1800, 1823, 1846, 1869, 1892, 1916, 1939, 1963, 1987, 2011, 2035, 2059, 2084,
    2109, 2133, 2159, 2184, 2209, 2235, 2260, 2286, 2312, 2339, 2365, 2392, 2419, 2446, 2473, 2500,
    2527, 2555, 2583, 2611, 2639, 2668, 2696, 2725, 2754, 2783, 2812, 2841, 2871, 2901, 2931, 2961,
    2991, 3022, 3052, 3083, 3114, 3146, 3177, 3209, 3240, 3272, 3304, 3337, 3369, 3402, 3435, 3468,
    3501, 3535, 3568, 3602, 3636, 3670, 3705, 3739, 3774, 3809, 3844, 3879, 3915, 3950, 3986, 4022,
    4059, 4095,
];

/// The first encode index of every byte `b >= 1` (`0` for `b = 0`, never read): Metal's
/// `pack_float_to_srgb_unorm4x8` encodes a color channel `x` (clamped to `[0, 1]`, NaN to 0) as
/// the byte `b` of the largest `SRGB_ENCODE_FIRST[b] <= i`, where `i` is the exact product
/// `x * 4095` rounded to the nearest even integer: its twelve-bit encode table. A1's
/// `srgb-encode-of-every-half-and-a-2-14-float-grid` (8f7708ff) observed a byte for every one of
/// the 4096 indices, consistently, and these are its thresholds; the MSL transfer
/// `1.055 c^(1/2.4) - 0.055` misses four of them (indices 804, 1371, 2121, and 2569).
pub(crate) const SRGB_ENCODE_FIRST: [u16; 256] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 23, 24, 26, 28, 30, 32, 34, 37, 39,
    42, 44, 47, 49, 52, 55, 58, 61, 64, 68, 71, 74, 78, 82, 85, 89, 93, 97, 102, 106, 110, 115,
    119, 124, 129, 134, 139, 144, 149, 154, 160, 165, 171, 177, 183, 189, 195, 201, 207, 214, 220,
    227, 234, 241, 248, 255, 262, 270, 277, 285, 293, 300, 308, 317, 325, 333, 342, 350, 359, 368,
    377, 386, 395, 405, 414, 424, 434, 444, 454, 464, 474, 485, 495, 506, 517, 528, 539, 550, 562,
    573, 585, 597, 609, 621, 633, 645, 658, 670, 683, 696, 709, 722, 736, 749, 763, 777, 790, 804,
    819, 833, 848, 862, 877, 892, 907, 922, 938, 953, 969, 985, 1001, 1017, 1033, 1049, 1066, 1083,
    1100, 1117, 1134, 1151, 1169, 1186, 1204, 1222, 1240, 1259, 1277, 1296, 1314, 1333, 1352, 1371,
    1391, 1410, 1430, 1450, 1470, 1490, 1510, 1531, 1552, 1572, 1593, 1615, 1636, 1657, 1679, 1701,
    1723, 1745, 1767, 1790, 1812, 1835, 1858, 1881, 1904, 1928, 1951, 1975, 1999, 2023, 2048, 2072,
    2097, 2121, 2146, 2172, 2197, 2222, 2248, 2274, 2300, 2326, 2352, 2379, 2406, 2433, 2460, 2487,
    2514, 2542, 2569, 2597, 2626, 2654, 2682, 2711, 2740, 2769, 2798, 2827, 2857, 2886, 2916, 2946,
    2977, 3007, 3038, 3068, 3099, 3130, 3162, 3193, 3225, 3257, 3289, 3321, 3354, 3386, 3419, 3452,
    3485, 3518, 3552, 3586, 3620, 3654, 3688, 3722, 3757, 3792, 3827, 3862, 3897, 3933, 3969, 4005,
    4041, 4077,
];

/// The linear value of every sRGB byte `b`: the `f32` nearest `decode(b / 255)` for the sRGB
/// transfer `decode(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4`, as bits. A
/// fragment function that writes a float to an sRGB attachment writes the value of the byte
/// Metal stores (`color_write`): each is within `5e-6` of a step of its byte's center, so a
/// device's encode stores that byte, and Metal's twelve-bit table encodes each to its own byte too.
const SRGB_LINEAR: [u32; 256] = [
    0x0000_0000,
    0x399F_22B4,
    0x3A1F_22B4,
    0x3A6E_B40E,
    0x3A9F_22B4,
    0x3AC6_EB61,
    0x3AEE_B40E,
    0x3B0B_3E5D,
    0x3B1F_22B4,
    0x3B33_070A,
    0x3B46_EB61,
    0x3B5B_518E,
    0x3B70_F18F,
    0x3B83_E1C6,
    0x3B8F_E616,
    0x3B9C_87FD,
    0x3BA9_C9B6,
    0x3BB7_AD6F,
    0x3BC6_354A,
    0x3BD5_6360,
    0x3BE5_39C1,
    0x3BF5_BA71,
    0x3C03_73B6,
    0x3C0C_6153,
    0x3C15_A705,
    0x3C1F_45BE,
    0x3C29_3E6B,
    0x3C33_91F7,
    0x3C3E_4149,
    0x3C49_4D44,
    0x3C54_B6C9,
    0x3C60_7EB4,
    0x3C6C_A5DF,
    0x3C79_2D22,
    0x3C83_0AA9,
    0x3C89_AF9F,
    0x3C90_85DC,
    0x3C97_8DC6,
    0x3C9E_C7C2,
    0x3CA6_3433,
    0x3CAD_D37D,
    0x3CB5_A602,
    0x3CBD_AC21,
    0x3CC5_E63A,
    0x3CCE_54AC,
    0x3CD6_F7D5,
    0x3CDF_D010,
    0x3CE8_DDBA,
    0x3CF2_212D,
    0x3CFB_9AC3,
    0x3D02_A56A,
    0x3D07_98DD,
    0x3D0C_A7E6,
    0x3D11_D2AF,
    0x3D17_1964,
    0x3D1C_7C30,
    0x3D21_FB3C,
    0x3D27_96B2,
    0x3D2D_4EBB,
    0x3D33_2381,
    0x3D39_152B,
    0x3D3F_23E4,
    0x3D45_4FD2,
    0x3D4B_991D,
    0x3D51_FFEC,
    0x3D58_8468,
    0x3D5F_26B6,
    0x3D65_E6FD,
    0x3D6C_C563,
    0x3D73_C20E,
    0x3D7A_DD24,
    0x3D81_0B65,
    0x3D84_B793,
    0x3D88_732E,
    0x3D8C_3E48,
    0x3D90_18F4,
    0x3D94_0344,
    0x3D97_FD49,
    0x3D9C_0715,
    0x3DA0_20BA,
    0x3DA4_4A4A,
    0x3DA8_83D6,
    0x3DAC_CD6F,
    0x3DB1_2727,
    0x3DB5_910F,
    0x3DBA_0B38,
    0x3DBE_95B3,
    0x3DC3_3090,
    0x3DC7_DBE0,
    0x3DCC_97B4,
    0x3DD1_641D,
    0x3DD6_412B,
    0x3DDB_2EEE,
    0x3DE0_2D76,
    0x3DE5_3CD4,
    0x3DEA_5D18,
    0x3DEF_8E51,
    0x3DF4_D090,
    0x3DFA_23E5,
    0x3DFF_885E,
    0x3E02_7F06,
    0x3E05_427F,
    0x3E08_0EA2,
    0x3E0A_E377,
    0x3E0D_C104,
    0x3E10_A753,
    0x3E13_966A,
    0x3E16_8E51,
    0x3E19_8F0F,
    0x3E1C_98AC,
    0x3E1F_AB30,
    0x3E22_C6A1,
    0x3E25_EB07,
    0x3E29_186A,
    0x3E2C_4ED0,
    0x3E2F_8E42,
    0x3E32_D6C5,
    0x3E36_2862,
    0x3E39_831F,
    0x3E3C_E703,
    0x3E40_5417,
    0x3E43_CA60,
    0x3E47_49E6,
    0x3E4A_D2AF,
    0x3E4E_64C3,
    0x3E52_0029,
    0x3E55_A4E7,
    0x3E59_5305,
    0x3E5D_0A89,
    0x3E60_CB7A,
    0x3E64_95DF,
    0x3E68_69BE,
    0x3E6C_471F,
    0x3E70_2E07,
    0x3E74_1E7E,
    0x3E78_188B,
    0x3E7C_1C33,
    0x3E80_14BF,
    0x3E82_2039,
    0x3E84_308B,
    0x3E86_45B8,
    0x3E88_5FC3,
    0x3E8A_7EB0,
    0x3E8C_A281,
    0x3E8E_CB3B,
    0x3E90_F8DF,
    0x3E93_2B72,
    0x3E95_62F6,
    0x3E97_9F6F,
    0x3E99_E0E0,
    0x3E9C_274C,
    0x3E9E_72B6,
    0x3EA0_C321,
    0x3EA3_1890,
    0x3EA5_7307,
    0x3EA7_D288,
    0x3EAA_3716,
    0x3EAC_A0B6,
    0x3EAF_0F68,
    0x3EB1_8332,
    0x3EB3_FC15,
    0x3EB6_7A14,
    0x3EB8_FD34,
    0x3EBB_8576,
    0x3EBE_12DE,
    0x3EC0_A56E,
    0x3EC3_3D2A,
    0x3EC5_DA14,
    0x3EC8_7C30,
    0x3ECB_2380,
    0x3ECD_D008,
    0x3ED0_81CA,
    0x3ED3_38C9,
    0x3ED5_F508,
    0x3ED8_B68A,
    0x3EDB_7D52,
    0x3EDE_4963,
    0x3EE1_1ABF,
    0x3EE3_F169,
    0x3EE6_CD65,
    0x3EE9_AEB5,
    0x3EEC_955B,
    0x3EEF_815C,
    0x3EF2_72B8,
    0x3EF5_6974,
    0x3EF8_6593,
    0x3EFB_6716,
    0x3EFE_6E00,
    0x3F00_BD2B,
    0x3F02_460C,
    0x3F03_D1A5,
    0x3F05_5FF7,
    0x3F06_F104,
    0x3F08_84CD,
    0x3F0A_1B54,
    0x3F0B_B499,
    0x3F0D_509F,
    0x3F0E_EF65,
    0x3F10_90EF,
    0x3F12_353D,
    0x3F13_DC50,
    0x3F15_862A,
    0x3F17_32CC,
    0x3F18_E237,
    0x3F1A_946E,
    0x3F1C_4970,
    0x3F1E_0140,
    0x3F1F_BBDE,
    0x3F21_794D,
    0x3F23_398C,
    0x3F24_FC9F,
    0x3F26_C285,
    0x3F28_8B41,
    0x3F2A_56D2,
    0x3F2C_253C,
    0x3F2D_F67F,
    0x3F2F_CA9C,
    0x3F31_A194,
    0x3F33_7B6A,
    0x3F35_581D,
    0x3F37_37B0,
    0x3F39_1A24,
    0x3F3A_FF7A,
    0x3F3C_E7B2,
    0x3F3E_D2CF,
    0x3F40_C0D2,
    0x3F42_B1BC,
    0x3F44_A58E,
    0x3F46_9C49,
    0x3F48_95EF,
    0x3F4A_9280,
    0x3F4C_91FF,
    0x3F4E_946C,
    0x3F50_99C9,
    0x3F52_A216,
    0x3F54_AD56,
    0x3F56_BB88,
    0x3F58_CCAF,
    0x3F5A_E0CC,
    0x3F5C_F7DF,
    0x3F5F_11EA,
    0x3F61_2EEF,
    0x3F63_4EEE,
    0x3F65_71E9,
    0x3F67_97E0,
    0x3F69_C0D5,
    0x3F6B_ECCA,
    0x3F6E_1BBF,
    0x3F70_4DB5,
    0x3F72_82AE,
    0x3F74_BAAB,
    0x3F76_F5AE,
    0x3F79_33B6,
    0x3F7B_74C6,
    0x3F7D_B8DE,
    0x3F80_0000,
];

/// An upper estimate of the SPIR-V words of the table (its 256 constants, the composite, and the
/// variable), charged to every use.
pub const TABLE_WORDS: u32 = 256 * 4 + 262 + 8;

/// An upper estimate of the SPIR-V words of one encode's search (eight steps of six
/// instructions) besides the index.
pub const SEARCH_WORDS: u32 = 8 * 6 * 6;

/// The estimate of the sRGB byte of an encode index `i >= 13` ([`color_linear`]): a quartic in
/// `t = sqrt(i / 4095)`, its coefficients highest power first, fitted to Metal's bytes and raised
/// by one, so that it lies between `b + 1/3` and `b + 5/3` for Metal's byte `b`, in `f32` with or
/// without fused steps (`srgb_estimates_lie_a_byte_above_metals`).
pub(crate) const ENCODE_ESTIMATE: [f32; 5] =
    [-76.24396, 209.02275, -236.77495, 368.16998, -8.362507];

/// The estimate's slope below index 13, where the transfer is linear (`x < 0.0031308`, index
/// 12.82): `12.92 * 255 / 4095`. The estimate there is `i` times it plus one.
pub(crate) const ENCODE_LINEAR: f32 = 0.8045421;

/// An upper estimate of the SPIR-V words of one decode's step read (an access chain, a load,
/// and a mask) besides the division.
pub const STEP_WORDS: u32 = 3 * 6;

/// Declare the module's sRGB table: a `Private` `u32[256]` initialized to the steps (low halves)
/// and thresholds (high halves).
pub fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(256),
    })?;
    let arr = mb.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(256);
    for (step, first) in SRGB_DECODE_STEPS.iter().zip(SRGB_ENCODE_FIRST) {
        parts.push(mb.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(*step) | (u64::from(first) << 16)),
        })?);
    }
    let init = mb.constant(Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    mb.plain_global(StorageClass::Private, arr, Some(init))
}

/// An upper estimate of the SPIR-V words of the color table's declaration
/// ([`declare_color_table`]): the vector, array, block and pointer types, the length, the
/// variable, and their decorations, charged once per module.
pub const COLOR_TABLE_WORDS: u32 = 64;

/// An upper estimate of the SPIR-V words of one sRGB channel's estimate, read and choice
/// ([`color_linear`]) besides the index: about 30 instructions and 12 constants.
pub const COLOR_WORDS: u32 = 42 * 6;

/// The buffer a fragment function's color table lives in, by the device's cost of its reads: every
/// lane of a subgroup reads the entry of its own channel's byte, so the reads of one instruction go
/// to many addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorTable {
    /// A read-only storage buffer, `StorageBuffer { uvec4[256] }`: the cheaper on NVIDIA's GPUs
    /// (the RTX 3090 and the RTX 5080 Laptop GPU drew such writes in 3.5 to 8.6 times the time
    /// from a uniform buffer) and on lavapipe's 256-bit vectors, and as cheap as a uniform buffer
    /// on RADV.
    Storage,
    /// A uniform buffer, `Uniform { uvec4[256] }` (std140), where the driver reads one at
    /// different addresses faster (`DriverWorkaround::UniformReadsOutpaceStorageReads`).
    Uniform,
}

impl ColorTable {
    /// The color table's buffer on `profile`.
    pub fn of(profile: &DeviceProfile) -> ColorTable {
        if profile
            .fields()
            .workarounds
            .contains(&DriverWorkaround::UniformReadsOutpaceStorageReads)
        {
            ColorTable::Uniform
        } else {
            ColorTable::Storage
        }
    }
}

/// The binding symbol of `stage`'s color table.
pub fn color_table_symbol(stage: Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::ColorWriteTable),
        index: 0,
    }
}

/// The color table's entries, byte `b` from 0: `(first, linear, prev, 0)`, `first` the byte's
/// first encode index (`SRGB_ENCODE_FIRST`), `linear` its linear value's bits (`SRGB_LINEAR`)
/// and `prev` byte `b - 1`'s (0 for byte 0, whose entry no write reads).
fn entries() -> impl Iterator<Item = [u32; 4]> {
    let prev = std::iter::once(0).chain(SRGB_LINEAR);
    SRGB_ENCODE_FIRST
        .iter()
        .zip(SRGB_LINEAR)
        .zip(prev)
        .map(|((first, linear), prev)| [u32::from(*first), linear, prev, 0])
}

/// The bytes of the color table a fragment function that writes a float to an sRGB color
/// attachment reads (`WriteSource::ColorWriteTable`): 4096 bytes, 256 little-endian `uvec4`
/// entries, entry `b` holding sRGB byte `b`'s first index in Metal's twelve-bit encode table, the
/// bits of the `f32` nearest its linear value, and those of byte `b - 1`'s linear value. The binder
/// binds a buffer holding exactly these bytes, whole, as the binding's descriptor type says (a
/// uniform or a storage buffer); they are the same for every pipeline and device.
pub fn color_write_table() -> &'static [u8] {
    static TABLE: std::sync::LazyLock<Vec<u8>> = std::sync::LazyLock::new(|| {
        entries()
            .flat_map(|e| e.into_iter().flat_map(u32::to_le_bytes))
            .collect()
    });
    &TABLE
}

/// Declare `stage`'s color table in the buffer `class` names: a block of one `uvec4[256]`
/// member at the table's binding symbol ([`color_table_symbol`]).
pub fn declare_color_table(
    mb: &mut ModuleBuilder,
    stage: Stage,
    class: ColorTable,
) -> BResult<GlobalId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let v4 = mb.ty(Type::Vector { elem: u32_, len: 4 })?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(256),
    })?;
    let arr = mb.ty(Type::Array {
        elem: v4,
        len,
        stride: Some(16),
    })?;
    let (storage, rule) = match class {
        ColorTable::Uniform => (StorageClass::Uniform, LayoutRule::Std140),
        ColorTable::Storage => (StorageClass::StorageBuffer, LayoutRule::Std430),
    };
    let layout = natural_layout(mb.types(), rule, true, &[arr])?;
    let block = mb.ty(Type::Struct {
        members: [arr].into(),
        layout: Some(layout),
    })?;
    mb.resource(color_table_symbol(stage), storage, block)
}

/// Word `i` (a `u32` scalar below 256) of `table`.
fn word(fb: &mut FunctionBuilder<'_>, table: GlobalId, i: Value) -> BResult<Value> {
    let base = fb.module().global_value(table)?;
    let p = fb.access_chain(base, &[i])?;
    fb.load(p, MemAccess::default())
}

/// The decode step `k` of sRGB byte `b` (a `u32` scalar below 256): the linear value is `k /
/// 4095`, correctly rounded.
pub fn step(fb: &mut FunctionBuilder<'_>, table: GlobalId, b: Value) -> BResult<Value> {
    let w = word(fb, table, b)?;
    let low = fb.u32_const(0xFFFF)?;
    let low = fb.module().const_value(low)?;
    fb.binary(BinOp::BitwiseAnd, w, low)
}

/// The sRGB byte of encode index `i` (a `u32` scalar below 4096: the exact product of the
/// clamped linear channel and 4095, rounded to nearest even): the largest byte whose first
/// index is at most `i`.
pub fn search(fb: &mut FunctionBuilder<'_>, table: GlobalId, i: Value) -> BResult<Value> {
    let k = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<Value> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    };
    let k16 = k(fb, 16)?;
    let mut b = None;
    for step in [128u32, 64, 32, 16, 8, 4, 2, 1] {
        let s = k(fb, step)?;
        // The first step's candidate is the constant 128.
        let (j, below) = match b {
            None => (s, k(fb, 0)?),
            Some(b) => (fb.binary(BinOp::IAdd, b, s)?, b),
        };
        let w = word(fb, table, j)?;
        let first = fb.binary(BinOp::ShiftRightLogical, w, k16)?;
        let at_least = fb.binary(BinOp::UGreaterThanEqual, i, first)?;
        b = Some(fb.select(at_least, j, below)?);
    }
    b.map_or_else(|| k(fb, 0), Ok)
}

/// The linear value (an `f32`) of the sRGB byte Metal's encode gives encode index `i` (a `u32`
/// scalar below 4096: the exact product of the clamped linear channel and 4095, rounded to
/// nearest even), by one read of the color table `table` ([`declare_color_table`]): the estimate
/// `e` ([`ENCODE_ESTIMATE`], [`ENCODE_LINEAR`]) lies between `b + 1/3` and `b + 5/3` for Metal's
/// byte `b`, so `b0 = floor(e)`, clamped to `[1, 255]`, is `b` or `b + 1`, and the byte is `b0`
/// less one exactly where `i` is below `b0`'s first index: entry `b0`'s linear value, else its
/// predecessor's. Its error budget is a third of a byte either side, far above an `f32`
/// evaluation's and a device square root's, so the device's precision cannot move the byte.
pub fn color_linear(fb: &mut FunctionBuilder<'_>, table: GlobalId, i: Value) -> BResult<Value> {
    let f = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let u = f.with(Held::Int {
        bits: 32,
        container: 32,
    });
    let ft = ty(fb, f)?;
    let ut = ty(fb, u)?;
    let x = fb.convert(ConvOp::ConvertUToF, i, ft)?;
    let step = fsplat(fb, f, f64::from(1.0f32 / 4095.0))?;
    let c = fb.binary(BinOp::FMul, x, step)?;
    let t = fb.glsl(GlslOp::Sqrt, &[c])?;
    let [lead, rest @ ..] = ENCODE_ESTIMATE;
    let mut e = fsplat(fb, f, f64::from(lead))?;
    for a in rest {
        let a = fsplat(fb, f, f64::from(a))?;
        let m = fb.binary(BinOp::FMul, e, t)?;
        e = fb.binary(BinOp::FAdd, m, a)?;
    }
    let slope = fsplat(fb, f, f64::from(ENCODE_LINEAR))?;
    let one = fsplat(fb, f, 1.0)?;
    let linear = fb.binary(BinOp::FMul, x, slope)?;
    let linear = fb.binary(BinOp::FAdd, linear, one)?;
    let knee = fsplat(fb, f, 13.0)?;
    let below_knee = fb.binary(BinOp::FOrdLessThan, x, knee)?;
    let e = fb.select(below_knee, linear, e)?;
    let b0 = fb.convert(ConvOp::ConvertFToU, e, ut)?;
    let lo = splat(fb, u, 1)?;
    let hi = splat(fb, u, 255)?;
    let b0 = fb.glsl(GlslOp::UClamp, &[b0, lo, hi])?;
    let base = fb.module().global_value(table)?;
    let member = splat(fb, u, 0)?;
    let p = fb.access_chain(base, &[member, b0])?;
    let entry = fb.load(p, MemAccess::default())?;
    let first = fb.composite_extract(entry, &[0])?;
    let own = fb.composite_extract(entry, &[1])?;
    let prev = fb.composite_extract(entry, &[2])?;
    let below = fb.binary(BinOp::ULessThan, i, first)?;
    let bits = fb.select(below, prev, own)?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}
