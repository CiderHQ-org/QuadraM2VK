//! The check of the driver fact `SampledFetchRecoversCodes`: each case of
//! `metal2vulkan::bind::sampled_fetch_cases`, an image of every code of a format the mipmap filter
//! reads in place, fetched through a sampled view by `metal2vulkan::bind::sampled_fetch_check_module`
//! and each code recovered as the filter's fetch recovers it, all in one submission. Metal's
//! answer: every code read back (`SampledFetchCase::difference`), on each format the device
//! samples and copies into; a format it does not is never read in place, so it is left out. It
//! does not run where the device takes none of them.

use super::fetched::{record_all, Fetched};
use super::kit::{CheckError, HostBuffer, Kit};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{sampled_fetch_cases, sampled_fetch_check_module, SampledFetchCheck};
use metal2vulkan::device::CheckAnswer;
use metal2vulkan::vk::FormatFeatureFlags2;

/// What the module's buffer holds before it. A texel's second word, once written, is 0 (a texel
/// of 32 bits or less) or the codes of its blue and alpha components, which are equal in every
/// case's texels: never this word, whose halves differ.
const UNWRITTEN: u32 = 0xDEAD_BEEF;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![answer(kit, target)]
}

/// The check's one answer.
fn answer(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let module = match sampled_fetch_check_module(target.compile) {
        Ok(module) => module,
        Err(error) => {
            return CheckAnswer::Unanswered(format!("its module does not build: {error}"))
        }
    };
    let needed = FormatFeatureFlags2::SAMPLED_IMAGE.union(FormatFeatureFlags2::TRANSFER_DST);
    let cases: Vec<_> = (sampled_fetch_cases().into_iter())
        .filter(|case| {
            (target.description.formats.formats.get(&case.format))
                .is_some_and(|p| p.optimal_tiling.contains(needed))
        })
        .collect();
    if cases.is_empty() {
        return CheckAnswer::NotRun("the device samples none of the formats".to_owned());
    }
    let read = match fetched(kit, &module, &cases) {
        Ok(read) => read,
        Err(error) => return CheckAnswer::Unanswered(error.0),
    };
    for (case, words) in cases.iter().zip(&read) {
        if let Some(texel) = words.chunks(2).position(|w| w.get(1) == Some(&UNWRITTEN)) {
            return CheckAnswer::Unanswered(format!(
                "{:?}: the module left texel {texel} unwritten",
                case.format
            ));
        }
        if let Some(difference) = case.difference(words) {
            return CheckAnswer::NotMetals(difference);
        }
    }
    CheckAnswer::Metals
}

/// Each case's words after the module, in one submission.
fn fetched(
    kit: &mut Kit<'_>,
    module: &SampledFetchCheck,
    cases: &[metal2vulkan::bind::SampledFetchCase],
) -> Result<Vec<Vec<u32>>, CheckError> {
    let shader = kit.shader(&module.spirv)?;
    let mut fetches: Vec<(Fetched, HostBuffer, u32)> = Vec::with_capacity(cases.len());
    for case in cases {
        let unwritten = vec![UNWRITTEN; case.words() as usize];
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
        let codes = kit.host_buffer(&unwritten, storage)?;
        let parameters = kit.host_buffer(&case.parameters, storage)?;
        let buffers = [
            (SampledFetchCheck::CODES_BINDING, codes),
            (SampledFetchCheck::PARAMETERS_BINDING, parameters),
        ];
        let format = vk::Format::from_raw(case.format.as_raw());
        let image = SampledFetchCheck::IMAGE_BINDING;
        let fetched = Fetched::new(kit, format, case.extent, &case.texels, image, &buffers)?;
        fetches.push((fetched, codes, case.workgroups()));
    }
    // Every set's layout is defined alike, so each is compatible with the first's.
    let layout = kit.layout(&[fetches[0].0.set_layout], None)?;
    let pipeline = kit.compute_pipeline(shader, layout)?;
    kit.submit(|device, command| {
        let all = fetches.iter().map(|(f, _, groups)| (f, *groups));
        // SAFETY: recording, outside a render pass; the pipeline is of `layout`, whose set 0 is
        // defined as every fetch's.
        unsafe { record_all(device, command, pipeline, layout, all) }
    })?;
    (fetches.iter())
        .map(|(_, codes, _)| kit.read(codes))
        .collect()
}
