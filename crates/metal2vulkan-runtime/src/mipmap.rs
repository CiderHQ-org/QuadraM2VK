//! `generateMipmapsForTexture:` on the device: the blit chain [`mipmap_plan`](super::mipmap_plan)
//! plans, and the per-format check that the device's chain gives Metal's answers within the spread
//! Metal's own GPUs show.
//!
//! **Metal has no one answer.** Metal's bytes, measured on an Apple M3 Max (macOS 26.7) and an
//! Apple M4 (macOS 27.0.1) for every format Metal generates mipmaps of
//! (`mipmap_answers.txt`), differ between the two GPUs by at most one unit in the last place
//! per component, except at a box that averages a float format's largest and smallest values
//! and at the levels below it. In 16-bit float the M3 Max answers 0 there and the M4 0.25
//! (`0x0000` against `0x3400`), and a level below they answer as far apart as `0x0040` against
//! `0x2c00` (about 3.8e-6 against 0.0625). In 32-bit float the box agrees and the level below
//! does not: about 7.45e-9 (`0x32000000`) on the M3 Max against 0 on the M4.
//!
//! **The check.** At the first `generate_mipmaps` of a pixel format held in a Vulkan format, of a
//! 2D or a 3D texture, whose chain halves every axis exactly at each level or not, the runtime
//! writes the measured inputs of that kind into textures of that format and usage (a halving
//! chain the 8x8 or 4x4x4 one, any other also the 7x5 one), records the same blit chain on its
//! internal submission, twice, and reads the levels back. A 3D chain that does not halve is
//! refused: Metal's answer for one is not measured. It accepts the format when both runs give
//! the same bytes and every component of every level lies within 1 unit in the last place (2
//! for 32-bit float components) of the M3 Max's answer or of the M4's. Otherwise, or where Metal's answer for the format is not
//! measured, `generate_mipmaps` of it is refused (`Unsupported`, `NotImplemented` for
//! `MipmapGeneration`: a compute path could compute such answers). The verdict lasts for the
//! device's lifetime ([`Device::mipmap_generation`]).
//!
//! An accepted format's mipmaps are therefore within Metal's measured spread, not Metal's exact
//! answer, which no one device gives. Every format Metal's filter is computed for (all but
//! `B5G6R5Unorm` and `BGR5A1Unorm`) runs no blit and no check where the device runs the filter's
//! kernel (`mipmap_filter.rs`). 1D textures are refused: Metal's answer for them is not measured.
//! Each slice of an array or cube texture is filtered as a 2D texture is.

use super::error::Error;
use super::mipmap_plan::{plan_mipmaps, MipBlit, MipmapPlan};
use super::texture::{texture_aspects, transfer_barrier, Texture};
use super::texture_descriptor::TextureDescriptor;
use super::values::{Region, ResourceOptions};
use super::Device;
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::Format;
use std::collections::HashMap;

/// The measured answers, `mipmap_answers.txt`'s format stated at its top.
const ANSWERS: &str = include_str!("mipmap_answers.txt");

/// Whether a device's `generate_mipmaps` of a pixel format gives Metal's answers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MipmapVerdict {
    /// No `generate_mipmaps` of the format has run its check yet.
    NotChecked,
    /// The device's mipmaps of the format lie within Metal's measured spread (each component
    /// within 1 unit in the last place, 2 for 32-bit float, of an Apple M3 Max's answer or of an
    /// Apple M4's), not Metal's exact answer, which no one device gives. The two Macs' answers
    /// lie up to 0.25 apart in 16-bit float and about 7.45e-9 apart in 32-bit float, below a
    /// box of a format's largest and smallest values.
    AcceptedWithinSpread,
    /// The device's mipmaps of the format fall outside that spread, or Metal's answer for the
    /// format is not measured: `generate_mipmaps` of it is refused.
    Refused,
}

/// What one check covers: a pixel format held in one Vulkan format, in a 2D or a 3D texture,
/// whose chain halves every axis exactly at each level or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct MipmapKey {
    pub(super) pixel_format: PixelFormat,
    pub(super) format: Format,
    pub(super) volume: bool,
    pub(super) halving: bool,
}

/// Whether every level of a chain from level-0 `extent` over `levels` levels is exactly half the
/// one above it on each axis (an axis of 1 stays 1): each blit then averages whole 2x2(x2) boxes.
pub(super) fn halves(extent: [u32; 3], levels: u32) -> bool {
    extent.iter().all(|&axis| {
        (1..levels).all(|level| {
            let above = (axis >> (level - 1).min(31)).max(1);
            above == 1 || above.is_multiple_of(2)
        })
    })
}

/// The device's checks: whether each key's mipmaps were accepted.
pub(super) type MipmapChecks = HashMap<MipmapKey, bool>;

/// One field of a texel, from the low bits of its little-endian word up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Field {
    Unorm(u32),
    Snorm(u32),
    /// A float of `exponent` and `mantissa` bits, with a sign bit when `signed`.
    Float {
        exponent: u32,
        mantissa: u32,
        signed: bool,
    },
    /// `RGB9E5`'s whole word: three 9-bit mantissas and a 5-bit shared exponent.
    Rgb9e5,
}

impl Field {
    fn parse(text: &str) -> Option<Field> {
        if text == "rgb9e5" {
            return Some(Field::Rgb9e5);
        }
        let float = |rest: &str, signed: bool| {
            let (e, m) = rest.split_once('.')?;
            Some(Field::Float {
                exponent: e.parse().ok()?,
                mantissa: m.parse().ok()?,
                signed,
            })
        };
        if let Some(rest) = text.strip_prefix("unorm") {
            rest.parse().ok().map(Field::Unorm)
        } else if let Some(rest) = text.strip_prefix("snorm") {
            rest.parse().ok().map(Field::Snorm)
        } else if let Some(rest) = text.strip_prefix("ufloat") {
            float(rest, false)
        } else if let Some(rest) = text.strip_prefix("float") {
            float(rest, true)
        } else {
            None
        }
    }

    fn bits(self) -> u32 {
        match self {
            Field::Unorm(bits) | Field::Snorm(bits) => bits,
            Field::Float {
                exponent,
                mantissa,
                signed,
            } => exponent + mantissa + u32::from(signed),
            Field::Rgb9e5 => 32,
        }
    }

    /// The units in the last place a device's component may lie from a measured answer.
    fn tolerance(self) -> u64 {
        match self {
            Field::Float {
                exponent: 8,
                mantissa: 23,
                ..
            } => 2,
            _ => 1,
        }
    }

    /// Whether every component of the field's bits `device` lies within the tolerance of the
    /// same component of `measured`.
    pub(super) fn near(self, device: u64, measured: u64) -> bool {
        let tolerance = self.tolerance();
        match self {
            Field::Unorm(_) => device.abs_diff(measured) <= tolerance,
            Field::Snorm(bits) => {
                let signed = |raw: u64| {
                    let shift = 64 - bits;
                    ((raw << shift) as i64) >> shift
                };
                signed(device).abs_diff(signed(measured)) <= tolerance
            }
            Field::Float {
                exponent,
                mantissa,
                signed,
            } => {
                // Floats in order of value: the magnitude's bits, negated for a negative value
                // (so -0 and +0 are one value, and adjacent floats one unit apart).
                let ordered = |raw: u64| {
                    let magnitude = (raw & ((1u64 << (exponent + mantissa)) - 1)) as i64;
                    if signed && (raw >> (exponent + mantissa)) & 1 == 1 {
                        -magnitude
                    } else {
                        magnitude
                    }
                };
                ordered(device).abs_diff(ordered(measured)) <= tolerance
            }
            Field::Rgb9e5 => {
                // Each mantissa at the larger exponent's scale: the units in the last place are
                // that exponent's.
                let exponent = |raw: u64| (raw >> 27) & 31;
                let (ea, eb) = (exponent(device), exponent(measured));
                let low = ea.min(eb);
                (0..3).all(|i| {
                    let ma = (device >> (9 * i)) & 511;
                    let mb = (measured >> (9 * i)) & 511;
                    // |ma 2^ea - mb 2^eb| <= tolerance 2^max(ea, eb), all over 2^low.
                    let a = u128::from(ma) << (ea - low);
                    let b = u128::from(mb) << (eb - low);
                    a.abs_diff(b) <= u128::from(tolerance) << (ea.max(eb) - low)
                })
            }
        }
    }
}

/// One measured texture: its level-0 extent and bytes, and each later level's bytes on the M3
/// Max and on the M4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MeasuredCase {
    pub(super) extent: [u32; 3],
    pub(super) input: Vec<u8>,
    pub(super) levels: Vec<[Vec<u8>; 2]>,
}

/// Metal's measured answers for one pixel format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MeasuredFormat {
    pub(super) texel_bytes: usize,
    pub(super) fields: Vec<Field>,
    pub(super) cases: Vec<MeasuredCase>,
}

fn hex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

fn extent(text: &str) -> Option<[u32; 3]> {
    let mut axes = text.split('x').map(|axis| axis.parse().ok());
    let extent = [axes.next()??, axes.next()??, axes.next()??];
    axes.next().is_none().then_some(extent)
}

/// Metal's measured answers for `pixel_format`, or `None` where they are not measured.
pub(super) fn measured(pixel_format: PixelFormat) -> Option<MeasuredFormat> {
    let mut lines = ANSWERS
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .skip_while(|line| line.split_whitespace().nth(1) != Some(pixel_format.name()));
    let mut header = lines.next()?.split_whitespace().skip(2);
    let texel_bytes: usize = header.next()?.parse().ok()?;
    let fields = header.map(Field::parse).collect::<Option<Vec<_>>>()?;
    let bits: u32 = fields.iter().map(|f| f.bits()).sum();
    if fields.is_empty() || bits as usize != 8 * texel_bytes {
        return None;
    }
    let mut cases: Vec<MeasuredCase> = Vec::new();
    for line in lines.take_while(|line| !line.starts_with("format ")) {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [level, m3, m4] if level.starts_with('L') => {
                let case = cases.last_mut()?;
                case.levels.push([hex(m3)?, hex(m4)?]);
            }
            [_, size, input] => cases.push(MeasuredCase {
                extent: extent(size)?,
                input: hex(input)?,
                levels: Vec::new(),
            }),
            _ => return None,
        }
    }
    let format = MeasuredFormat {
        texel_bytes,
        fields,
        cases,
    };
    format.consistent().then_some(format)
}

/// The extent of `level` of a texture of level-0 `extent`.
fn level_extent(extent: [u32; 3], level: u32) -> [u32; 3] {
    extent.map(|axis| (axis >> level).max(1))
}

impl MeasuredFormat {
    /// Every case's bytes have its extent's length at each level.
    fn consistent(&self) -> bool {
        let length = |e: [u32; 3]| e.iter().map(|&a| a as usize).product::<usize>();
        !self.cases.is_empty()
            && self.cases.iter().all(|case| {
                case.input.len() == length(case.extent) * self.texel_bytes
                    && !case.levels.is_empty()
                    && case.levels.iter().enumerate().all(|(i, answers)| {
                        let n = length(level_extent(case.extent, i as u32 + 1)) * self.texel_bytes;
                        answers.iter().all(|bytes| bytes.len() == n)
                    })
            })
    }

    /// Whether every component of the texels `device` lies within the tolerance of the M3 Max's
    /// answer `m3` or of the M4's `m4`, each compared on its own.
    pub(super) fn within_spread(&self, device: &[u8], m3: &[u8], m4: &[u8]) -> bool {
        if device.len() != m3.len() || device.len() != m4.len() {
            return false;
        }
        let word = |texel: &[u8]| {
            texel
                .iter()
                .rev()
                .fold(0u128, |word, &byte| (word << 8) | u128::from(byte))
        };
        let texels = device
            .chunks_exact(self.texel_bytes)
            .zip(m3.chunks_exact(self.texel_bytes))
            .zip(m4.chunks_exact(self.texel_bytes));
        texels.into_iter().all(|((d, a), b)| {
            let (d, a, b) = (word(d), word(a), word(b));
            let mut shift = 0;
            self.fields.iter().all(|field| {
                let bits = field.bits();
                let mask = (1u128 << bits) - 1;
                let take = |w: u128| ((w >> shift) & mask) as u64;
                let near = field.near(take(d), take(a)) || field.near(take(d), take(b));
                shift += bits;
                near
            })
        })
    }
}

/// Record blit `blit` of `image` (of `aspects`), linearly filtered.
pub(super) fn record_blit(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    aspects: vk::ImageAspectFlags,
    blit: &MipBlit,
) {
    let layers = |level| vk::ImageSubresourceLayers {
        aspect_mask: aspects,
        mip_level: level,
        base_array_layer: blit.base_layer,
        layer_count: blit.layers,
    };
    let corner = |[x, y, z]: [u32; 3]| vk::Offset3D {
        x: x as i32,
        y: y as i32,
        z: z as i32,
    };
    let region = vk::ImageBlit {
        src_subresource: layers(blit.src_level),
        src_offsets: [vk::Offset3D::default(), corner(blit.src_extent)],
        dst_subresource: layers(blit.dst_level),
        dst_offsets: [vk::Offset3D::default(), corner(blit.dst_extent)],
    };
    // SAFETY: the command buffer is recording; the image is live, in the `GENERAL` layout, with
    // transfer usages and its format's blit and linear-filter features (`plan_mipmaps` checked
    // them), and the two levels are distinct.
    unsafe {
        device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::GENERAL,
            image,
            vk::ImageLayout::GENERAL,
            &[region],
            vk::Filter::LINEAR,
        );
    }
}

impl Device {
    /// Whether this device's `generate_mipmaps` of `pixel_format` gives Metal's answers: within
    /// the spread measured between an Apple M3 Max and an Apple M4, not Metal's exact answer
    /// (see [`MipmapVerdict`]). A format is checked at its first `generate_mipmaps`; one held in
    /// more than one Vulkan format (by usage), mipmapped in 2D and 3D textures, or in textures
    /// whose levels halve exactly and ones whose levels do not, is checked once for each, and is
    /// `Refused` when any of them was.
    pub fn mipmap_generation(&self, pixel_format: PixelFormat) -> MipmapVerdict {
        let checks = super::device::lock(&self.shared().textures.mipmaps);
        let mut verdicts = checks
            .iter()
            .filter(|(key, _)| key.pixel_format == pixel_format)
            .map(|(_, &accepted)| accepted)
            .peekable();
        if verdicts.peek().is_none() {
            MipmapVerdict::NotChecked
        } else if verdicts.all(|accepted| accepted) {
            MipmapVerdict::AcceptedWithinSpread
        } else {
            MipmapVerdict::Refused
        }
    }

    /// Whether `texture`'s mipmaps are accepted on this device: the cached verdict of its key,
    /// or the check run now and cached.
    pub(super) fn mipmaps_accepted(&self, texture: &Texture) -> Result<bool, Error> {
        let record = texture.record();
        let volume = record.texture_type == TextureType::Type3D;
        let mut extent = record.extent();
        if !volume {
            extent[2] = 1;
        }
        let key = MipmapKey {
            pixel_format: record.pixel_format,
            format: texture.image_plan()?.format,
            volume,
            halving: halves(extent, record.level_count),
        };
        let state = &self.shared().textures;
        if let Some(&accepted) = super::device::lock(&state.mipmaps).get(&key) {
            return Ok(accepted);
        }
        let accepted = self.check_mipmaps(texture, key)?;
        Ok(*super::device::lock(&state.mipmaps)
            .entry(key)
            .or_insert(accepted))
    }

    /// The check of `key` (the module's doc), with textures of `texture`'s usage.
    fn check_mipmaps(&self, texture: &Texture, key: MipmapKey) -> Result<bool, Error> {
        let Some(measured) = measured(key.pixel_format) else {
            return Ok(false);
        };
        // A halving chain is checked on the measured halving textures; any other on every
        // measured texture of its dimensions, at least one of which does not halve.
        let case_halves = |case: &MeasuredCase| halves(case.extent, case.levels.len() as u32 + 1);
        let cases: Vec<&MeasuredCase> = measured
            .cases
            .iter()
            .filter(|case| (case.extent[2] > 1) == key.volume)
            .filter(|case| !key.halving || case_halves(case))
            .collect();
        if !key.halving && cases.iter().all(|case| case_halves(case)) {
            // Metal's answer for such a chain is not measured (3D textures that do not halve).
            return Ok(false);
        }
        let mut checked = false;
        for case in cases {
            let first = self.mipmaps_of(texture, key, case, measured.texel_bytes)?;
            let second = self.mipmaps_of(texture, key, case, measured.texel_bytes)?;
            let (Some(first), Some(second)) = (first, second) else {
                return Ok(false);
            };
            let near = first
                .iter()
                .zip(&case.levels)
                .all(|(device, [m3, m4])| measured.within_spread(device, m3, m4));
            if first != second || first.len() != case.levels.len() || !near {
                return Ok(false);
            }
            checked = true;
        }
        Ok(checked)
    }

    /// The levels after level 0 the device's blit chain gives `case`, in a new texture of
    /// `key`'s formats with `texture`'s usage; `None` when such a texture cannot hold it in
    /// `key`'s Vulkan format or plans no chain.
    fn mipmaps_of(
        &self,
        texture: &Texture,
        key: MipmapKey,
        case: &MeasuredCase,
        texel_bytes: usize,
    ) -> Result<Option<Vec<Vec<u8>>>, Error> {
        let [width, height, depth] = case.extent.map(u64::from);
        let descriptor = TextureDescriptor {
            texture_type: if key.volume {
                TextureType::Type3D
            } else {
                TextureType::Type2D
            },
            pixel_format: key.pixel_format,
            width,
            height,
            depth,
            mipmap_level_count: case.levels.len() as u64 + 1,
            usage: texture.record().usage,
            resource_options: ResourceOptions::shared(),
            ..TextureDescriptor::default()
        };
        let check = self.new_runtime_texture(&descriptor)?;
        if check.image_plan()?.format != key.format {
            return Ok(None);
        }
        let row = width * texel_bytes as u64;
        let region = Region::new_3d(0, 0, 0, width, height, depth);
        check.replace_region(region, 0, 0, &case.input, row, row * height)?;
        let state = &self.shared().textures;
        let facts = state.facts(self.compile_device().profile());
        let Ok(MipmapPlan::Blits(blits)) =
            plan_mipmaps(check.record(), check.image_plan()?, &facts)
        else {
            return Ok(None);
        };
        let (image, aspects) = (check.raw_image()?, texture_aspects(check.image_plan()?));
        self.submit_internal(|device, command_buffer| {
            for blit in &blits {
                // Each blit reads the level the one before it wrote.
                transfer_barrier(device, command_buffer, true);
                record_blit(device, command_buffer, image, aspects, blit);
            }
            transfer_barrier(device, command_buffer, false);
        })?;
        let mut levels = Vec::with_capacity(blits.len());
        for level in 1..=blits.len() as u32 {
            let [w, h, d] = level_extent(case.extent, level).map(u64::from);
            let row = w * texel_bytes as u64;
            let mut bytes = vec![0u8; (row * h * d) as usize];
            let region = Region::new_3d(0, 0, 0, w, h, d);
            check.get_bytes(&mut bytes, row, row * h, region, u64::from(level), 0)?;
            levels.push(bytes);
        }
        Ok(Some(levels))
    }
}
