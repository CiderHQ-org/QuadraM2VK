//! The checks of the block-compressed formats' decodes: the driver facts
//! `ColorBlockDecodeIsMetals` (BC1 to BC3), `ChannelBlockDecodeIsMetals` (BC4 and BC5),
//! `Bc6hBlockDecodeIsMetals`, `Bc7BlockDecodeIsMetals` and `Bc7ReservedBlockIsMetals` (BC7's
//! blocks of the reserved mode, apart from its others). Each case of
//! `metal2vulkan::bind::block_decode_cases` copies its fixed blocks into a 32 x 32 image of its
//! format, and one dispatch of `metal2vulkan::bind::texel_fetch_check_module` per case fetches
//! every texel as `float`s, compared bit for bit with Metal's reads of them, all in one
//! submission. A fact is Metal's where each of its formats reads as Metal's, and the other answer
//! where one does not, the first such format's difference stated. It does not run where the
//! device does not sample or copy into one of its formats.

use super::fetched::{record_all, Fetched};
use super::kit::{CheckError, HostBuffer, Kit};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{
    block_decode_cases, texel_fetch_check_module, BlockDecodeCase, TexelFetch, TexelFetchCheck,
};
use metal2vulkan::device::{CheckAnswer, DriverFact};
use metal2vulkan::vk::FormatFeatureFlags2;

/// The facts the check decides, in the suite's order.
pub(super) const FACTS: [DriverFact; 5] = [
    DriverFact::ColorBlockDecodeIsMetals,
    DriverFact::ChannelBlockDecodeIsMetals,
    DriverFact::Bc6hBlockDecodeIsMetals,
    DriverFact::Bc7BlockDecodeIsMetals,
    DriverFact::Bc7ReservedBlockIsMetals,
];

/// What the fetch's buffer holds before it: a NaN, which no case's texel reads as (each is finite).
const UNWRITTEN: u32 = u32::MAX;

/// The check's answers on `target`, in [`FACTS`]' order.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    let cases = block_decode_cases();
    let fetch = match texel_fetch_check_module(target.compile, TexelFetch::FLOAT) {
        Ok(fetch) => fetch,
        Err(error) => {
            let why = format!("its fetch does not build: {error}");
            return vec![CheckAnswer::Unanswered(why); FACTS.len()];
        }
    };
    let supported: Vec<bool> = cases.iter().map(|c| supports(target, c)).collect();
    let read = match fetched(kit, &fetch, &cases, &supported) {
        Ok(read) => read,
        Err(error) => return vec![CheckAnswer::Unanswered(error.0); FACTS.len()],
    };
    FACTS.map(|fact| answer(fact, &cases, &read)).to_vec()
}

/// Whether the device samples and copies into `case`'s format in optimal tiling.
fn supports(target: &Target<'_>, case: &BlockDecodeCase) -> bool {
    let needed = FormatFeatureFlags2::SAMPLED_IMAGE.union(FormatFeatureFlags2::TRANSFER_DST);
    (target.description.formats.formats.get(&case.format))
        .is_some_and(|p| p.optimal_tiling.contains(needed))
}

/// `fact`'s answer from its cases' words `read` (`None` for a format the device does not take).
fn answer(fact: DriverFact, cases: &[BlockDecodeCase], read: &[Option<Vec<u32>>]) -> CheckAnswer {
    let mut untaken = Vec::new();
    for (case, words) in cases.iter().zip(read).filter(|(c, _)| c.decides(fact)) {
        let Some(words) = words else {
            untaken.push(format!("{:?}", case.format));
            continue;
        };
        if let Some(at) = words.iter().position(|&w| w == UNWRITTEN) {
            return CheckAnswer::Unanswered(format!(
                "{:?}: the fetch left word {at} unwritten",
                case.format
            ));
        }
        if let Some(difference) = case.difference(fact, words) {
            return CheckAnswer::NotMetals(difference);
        }
    }
    match untaken.is_empty() {
        true => CheckAnswer::Metals,
        false => CheckAnswer::NotRun(format!(
            "the device does not sample or copy into {}",
            untaken.join(", ")
        )),
    }
}

/// The fetch's words of each case the device takes (`supported`), in one submission.
fn fetched(
    kit: &mut Kit<'_>,
    fetch: &TexelFetchCheck,
    cases: &[BlockDecodeCase],
    supported: &[bool],
) -> Result<Vec<Option<Vec<u32>>>, CheckError> {
    let module = kit.shader(&fetch.spirv)?;
    let side = TexelFetchCheck::SIDE;
    let mut fetches: Vec<Option<(Fetched, HostBuffer)>> = Vec::with_capacity(cases.len());
    for (case, &taken) in cases.iter().zip(supported) {
        if !taken {
            fetches.push(None);
            continue;
        }
        let unwritten = vec![UNWRITTEN; fetch.words() as usize];
        let texels = kit.host_buffer(&unwritten, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let format = vk::Format::from_raw(case.format.as_raw());
        let buffers = [(TexelFetchCheck::TEXELS_BINDING, texels)];
        let image = TexelFetchCheck::IMAGE_BINDING;
        let fetched = Fetched::new(kit, format, [side, side], &case.blocks, image, &buffers)?;
        fetches.push(Some((fetched, texels)));
    }
    let Some(set_layout) = fetches.iter().flatten().map(|(f, _)| f.set_layout).next() else {
        return Ok(vec![None; cases.len()]);
    };
    // Every set's layout is defined alike, so each is compatible with this one's.
    let layout = kit.layout(&[set_layout], None)?;
    let pipeline = kit.compute_pipeline(module, layout)?;
    let workgroups = fetch.workgroups();
    kit.submit(|device, command| {
        let all = fetches.iter().flatten().map(|(f, _)| (f, workgroups));
        // SAFETY: recording, outside a render pass; the pipeline is of `layout`, whose set 0 is
        // defined as every fetch's.
        unsafe { record_all(device, command, pipeline, layout, all) }
    })?;
    (fetches.iter())
        .map(|f| f.as_ref().map(|(_, texels)| kit.read(texels)).transpose())
        .collect()
}
