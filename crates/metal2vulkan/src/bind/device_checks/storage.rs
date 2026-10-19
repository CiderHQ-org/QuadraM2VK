//! The device's checks of five storage-write deviations Metal's rule decides (`DriverDeviation`'s
//! `MetalHalfStorageWrites`, `MetalHalfStorageNans`, `MetalFloatStorageNans`,
//! `MetalHalfTexelWrites` and `MetalNormStorageWrites`). Each is a set of writes
//! ([`StorageWrite`]): a kernel whose invocation `(x, y)` writes its pixel's input, read as a
//! `float`, to every channel of texel `(x, y)` of a storage image as computed
//! (`StorageImageClass::Identity`, the form translation writes where the profile takes the
//! deviation), through each declaration translation gives the format ([`storage_write_modules`]:
//! for a texture a kernel only writes, and for one it reads and writes). The runtime reads the
//! image back; Metal's texels by its rule ([`StorageWrite::expected`]) are Metal's answer:
//!
//! - a half-float channel: the value rounded toward zero to `half`, a finite value past 65504
//!   stored as 65504, an infinity as itself, and every NaN as `0x7e00`; a `half` widened exactly
//!   (a signalling NaN's included) as its own bits;
//! - `RG11B10Float`'s unsigned floats: the NaN `0x7fc00000` as the format's quiet NaN;
//! - a unorm or snorm channel: the value clamped to its range (a NaN 0) times its largest code
//!   exactly, rounded to nearest with a tie away from zero.
//!
//! The writes of each fact:
//!
//! - `MetalHalfStorageWrites`: `R16Float`, `RG16Float` and `RGBA16Float`, every half and every
//!   tie between neighbouring halves with one `f32` step either side, and the specials, no NaN.
//! - `MetalHalfStorageNans`: the same formats, NaNs of every class (quiet and signalling, both
//!   signs, several payloads).
//! - `MetalFloatStorageNans`: the same formats, the NaN `0x7fc00000`.
//! - `MetalHalfTexelWrites`: the same formats, every `half` NaN widened; and `RG11B10Float`, the
//!   NaN `0x7fc00000`.
//! - `MetalNormStorageWrites`: `R8Unorm`, `R8Snorm`, `R16Unorm`, `R16Snorm` and `RGB10A2Unorm`, every
//!   code's exact value and every tie with two `f32` steps either side, and the specials, NaNs
//!   included.

use std::sync::Arc;

use super::color::{half, half_nans, snorm, specials, texels, unorm, words, Rule};
use super::kernel::finish;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{DriverDeviation, Error};
use crate::float16::f16_bits_to_f32;
use crate::metal::format::{accessed_without_format, spirv_image_format, StorageUse};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, ImageOperands,
    ImageType, ImageUsage, IoDir, LayoutRule, MemAccess, StorageClass, TexelMemory, Type,
    TypedModule,
};
use crate::vk::Format;

/// The storage-write deviations the writes check, in the order [`storage_writes`] takes them.
pub const STORAGE_WRITE_FACTS: [DriverDeviation; 5] = [
    DriverDeviation::MetalHalfStorageWrites,
    DriverDeviation::MetalHalfStorageNans,
    DriverDeviation::MetalFloatStorageNans,
    DriverDeviation::MetalHalfTexelWrites,
    DriverDeviation::MetalNormStorageWrites,
];

/// One write of a storage-write check (the module docs): `format`'s image of
/// [`StorageWrite::WIDTH`] by [`height`](Self::height) texels, written by each of the format's
/// kernels ([`storage_write_modules`]) dispatched as `(WIDTH / LOCAL_SIZE, height, 1)` workgroups,
/// and read back.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageWrite {
    /// The image's format.
    pub format: Format,
    /// Rows of the image.
    pub height: u32,
    /// Each texel's input word, row by row: the storage buffer at
    /// [`StorageWriteModule::VALUES_BINDING`].
    pub values: Vec<u32>,
    /// Bytes of one texel of `format`.
    pub texel_bytes: u32,
    /// Metal's texels, row by row, tightly packed.
    pub expected: Vec<u8>,
}

impl StorageWrite {
    /// Texels of a row of every write's image.
    pub const WIDTH: u32 = 1024;
    /// Invocations of a workgroup along x (one workgroup a row's `LOCAL_SIZE` texels).
    pub const LOCAL_SIZE: u32 = 64;

    /// Whether `texels`, read back from the image row by row, tightly packed, are Metal's.
    pub fn answers_metal(&self, texels: &[u8]) -> bool {
        texels == self.expected.as_slice()
    }

    /// How `texels`, read back from the image row by row, tightly packed, differ from Metal's, in a
    /// few words for the other answer's statement (`CheckAnswer::NotMetals`): how many texels
    /// differ, and the first, its position, the bytes read and Metal's. `None` for Metal's.
    pub fn difference(&self, texels: &[u8]) -> Option<String> {
        super::texel_difference(texels, &self.expected, self.texel_bytes, Self::WIDTH)
    }
}

/// One kernel of a storage write: `GLCompute` entry `main` at local size
/// ([`StorageWrite::LOCAL_SIZE`], 1, 1), the input words a storage buffer at set 0, binding
/// [`StorageWriteModule::VALUES_BINDING`], and the image a storage image at binding
/// [`StorageWriteModule::IMAGE_BINDING`], declared as [`declared`](Self::declared) says.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageWriteModule {
    /// SPIR-V 1.5 words.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within its profile).
    pub requirements: Requirements,
    /// Whether the image is declared with a format; else without one (`Unknown`).
    pub declared: bool,
}

impl StorageWriteModule {
    /// The binding of the input words, in set 0.
    pub const VALUES_BINDING: u32 = 0;
    /// The binding of the storage image, in set 0.
    pub const IMAGE_BINDING: u32 = 1;
}

/// The kernels translation's declarations of `format`'s storage texture give on `device`: a
/// texture a kernel only writes, and one it reads and writes, each without a format where the
/// device writes (and reads) it without one, else with its SPIR-V image format; one where both
/// declare alike. Empty where the format has no SPIR-V image format and needs one.
pub fn storage_write_modules(
    device: &Device,
    format: Format,
) -> Result<Vec<StorageWriteModule>, Error> {
    let profile = device.profile();
    let mut declared = Vec::new();
    for read in [false, true] {
        let without = accessed_without_format(format, StorageUse::texture(read, true), profile);
        let spirv_format = match without {
            true => Some(crate::spv::ImageFormat::Unknown),
            false => spirv_image_format(format),
        };
        if let Some(spirv_format) = spirv_format {
            if !declared.contains(&spirv_format) {
                declared.push(spirv_format);
            }
        }
    }
    declared
        .into_iter()
        .map(|spirv_format| {
            let m = kernel_typed_module(spirv_format)?;
            let (spirv, requirements) = finish(device, m, "storage write check")?;
            Ok(StorageWriteModule {
                spirv,
                requirements,
                declared: spirv_format != crate::spv::ImageFormat::Unknown,
            })
        })
        .collect()
}

/// A `u32` constant as a value.
fn word(f: &mut FunctionBuilder<'_>, u32_: TypeId, bits: u32) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(bits)),
    })?;
    f.module().const_value(c)
}

/// The values buffer: `Block { u32 words[]; }` (std430) at buffer slot
/// [`StorageWriteModule::VALUES_BINDING`].
fn values_buffer(b: &mut ModuleBuilder, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index: StorageWriteModule::VALUES_BINDING,
    };
    b.resource(slot, StorageClass::StorageBuffer, block)
}

/// The kernel of an image declared `spirv_format` as a typed module (the module docs).
fn kernel_typed_module(spirv_format: crate::spv::ImageFormat) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let image_ty = b.ty(Type::Image(ImageType {
        sampled_type: f32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Storage,
        format: spirv_format,
    }))?;
    let values = values_buffer(&mut b, u32_)?;
    let slot = ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Texture,
        index: StorageWriteModule::IMAGE_BINDING,
    };
    let image = b.resource(slot, StorageClass::UniformConstant, image_ty)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let width = word(&mut f, u32_, StorageWrite::WIDTH)?;
    let row = f.binary(BinOp::IMul, y, width)?;
    let at = f.binary(BinOp::IAdd, row, x)?;
    let zero = word(&mut f, u32_, 0)?;
    let values = f.module().global_value(values)?;
    let p = f.access_chain(values, &[zero, at])?;
    let w = f.load(p, MemAccess::default())?;
    let v = f.convert(ConvOp::Bitcast, w, f32_)?;
    let texel = f.composite_construct(vec4, &[v, v, v, v])?;
    let pixel = f.composite_construct(uvec2, &[x, y])?;
    let image = f.module().global_value(image)?;
    let image = f.load(image, MemAccess::default())?;
    f.write(
        image,
        pixel,
        texel,
        &ImageOperands::default(),
        TexelMemory::default(),
    )?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: StorageWrite::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The NaNs of the NaN writes: quiet and signalling, several payloads, each of both signs.
const NANS: [u32; 8] = [
    0x7fc0_0000,
    0x7fc0_0001,
    0x7fd5_5555,
    0x7fff_ffff,
    0x7f80_0001,
    0x7f80_2000,
    0x7fa0_0000,
    0x7fbf_ffff,
];

/// The write of `values` into `format` by `rule`.
fn write(format: Format, rule: Rule, values: Vec<u32>) -> StorageWrite {
    let t = texels(format, false, false, rule, values);
    StorageWrite {
        format,
        height: t.height,
        values: t.values,
        texel_bytes: t.texel_bytes,
        expected: t.expected,
    }
}

/// The half-float formats.
const HALF_FORMATS: [Format; 3] = [
    Format::R16_SFLOAT,
    Format::R16G16_SFLOAT,
    Format::R16G16B16A16_SFLOAT,
];

/// The writes of storage-write deviation `fact` (the module docs), or `None` where `fact` is not
/// one of [`STORAGE_WRITE_FACTS`].
pub fn storage_writes(fact: DriverDeviation) -> Option<Vec<StorageWrite>> {
    let each_half = |values: Vec<u32>, rule: Rule| -> Vec<StorageWrite> {
        HALF_FORMATS
            .iter()
            .map(|&format| write(format, rule, values.clone()))
            .collect()
    };
    let nan = 0x7fc0_0000u32;
    let writes = match fact {
        DriverDeviation::MetalHalfStorageWrites => {
            let mut values = half();
            values.extend(specials(false));
            each_half(words(values), Rule::Metal)
        }
        DriverDeviation::MetalHalfStorageNans => {
            let nans = NANS.iter().flat_map(|&n| [n, n | 0x8000_0000]).collect();
            each_half(nans, Rule::Metal)
        }
        DriverDeviation::MetalFloatStorageNans => {
            each_half(vec![nan; StorageWrite::WIDTH as usize], Rule::Metal)
        }
        DriverDeviation::MetalHalfTexelWrites => {
            let widened = half_nans()
                .into_iter()
                .map(|h| f16_bits_to_f32(h).to_bits())
                .collect();
            let mut writes = each_half(widened, Rule::OwnBits);
            writes.push(write(
                Format::B10G11R11_UFLOAT_PACK32,
                Rule::Metal,
                vec![nan; StorageWrite::WIDTH as usize],
            ));
            writes
        }
        DriverDeviation::MetalNormStorageWrites => [
            (Format::R8_UNORM, unorm(8)),
            (Format::R8_SNORM, snorm(8)),
            (Format::R16_UNORM, unorm(16)),
            (Format::R16_SNORM, snorm(16)),
            (
                Format::A2B10G10R10_UNORM_PACK32,
                [unorm(10), unorm(2)].concat(),
            ),
        ]
        .into_iter()
        .map(|(format, mut values)| {
            values.extend(specials(true));
            write(format, Rule::Metal, words(values))
        })
        .collect(),
        _ => return None,
    };
    Some(writes)
}

#[cfg(test)]
mod tests {
    use super::{kernel_typed_module, storage_writes, StorageWrite, STORAGE_WRITE_FACTS};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    /// The kernel is valid on every committed profile that writes storage images without a
    /// format, declared without one, and declared `Rgba16f` on every one.
    #[test]
    fn the_storage_write_kernels_are_valid_on_the_committed_profiles() {
        let mut without = 0;
        for (_, profile) in committed_profiles().expect("profiles") {
            let m = kernel_typed_module(crate::spv::ImageFormat::Rgba16f).expect("declared");
            assert_valid(m, &profile);
            if profile.fields().images.storage_write_without_format {
                let m = kernel_typed_module(crate::spv::ImageFormat::Unknown).expect("without");
                assert_valid(m, &profile);
                without += 1;
            }
        }
        assert!(without > 0, "no committed profile writes without a format");
    }

    /// Every fact has writes of whole rows; a NaN write stores `0x7e00`, a widened half NaN its
    /// own bits.
    #[test]
    fn every_storage_fact_writes_whole_rows() {
        for fact in STORAGE_WRITE_FACTS {
            let writes = storage_writes(fact).expect("a storage fact");
            assert!(!writes.is_empty(), "{fact:?}");
            for w in &writes {
                let texels = (StorageWrite::WIDTH * w.height) as usize;
                assert_eq!(w.values.len(), texels, "{fact:?} {:?}", w.format);
                assert_eq!(w.expected.len(), texels * w.texel_bytes as usize);
            }
        }
        let nans = storage_writes(crate::error::DriverDeviation::MetalFloatStorageNans).unwrap();
        assert_eq!(&nans[0].expected[..2], &[0x00, 0x7e]);
        let own = storage_writes(crate::error::DriverDeviation::MetalHalfTexelWrites).unwrap();
        assert_eq!(&own[0].expected[..2], &[0x01, 0x7c]);
    }
}
