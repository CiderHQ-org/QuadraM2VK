//! Stand-in storage formats for a pipeline state's creation compile.
//!
//! Metal makes a compute pipeline state before it knows the textures its dispatches bind. On a
//! device without `shaderStorageImageReadWithoutFormat` (Intel's ANV), a function that reads a
//! storage texture declares the texture's format in its module, so its compile needs the bound
//! format (`NeedsInput(StorageTextureFormat)`). The state is still made: its creation compile
//! takes a stand-in for each such slot ([`compile_with_stand_ins`]). That compile only answers
//! Metal's queries about the state, which do not depend on a storage view's format, and is never
//! dispatched: each dispatch's plan names the variant of the bound textures' formats, compiled
//! at its first use. A stand-in compile's key holds its stand-in choices, so the disk cache keeps
//! it under the key of the compile it is.

use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::error::{Error, RuntimeInput};
use metal2vulkan::metal::format::storage_image_choice;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{SlotRef, StorageImageChoice};

/// The stand-in formats, in the order they are tried: Vulkan's mandatory storage formats, one
/// per texel type a storage slot reads (float, unsigned and signed integer).
pub(crate) const STAND_INS: [PixelFormat; 3] = [
    PixelFormat::R32Float,
    PixelFormat::R32Uint,
    PixelFormat::R32Sint,
];

/// The most slots that take stand-ins at once: the search tries every combination of
/// [`STAND_INS`] over them, so it stays bounded.
pub(crate) const MOST_SLOTS: usize = 4;

/// The slot whose format `error` asks for.
fn needs_format(error: &Error) -> Option<SlotRef> {
    match error {
        Error::NeedsInput(RuntimeInput::StorageTextureFormat { slot }) => Some(*slot),
        _ => None,
    }
}

/// Compile with `compile` under the empty key; where that compile needs storage textures'
/// formats, under stand-ins for them: each slot the compile asks for takes the first of
/// [`STAND_INS`] whose choice compiles together with the others' (every combination is tried,
/// over at most [`MOST_SLOTS`] slots). Returns the compiled pipeline and its key, whose
/// `storage_images` are the stand-ins (empty without them). When no combination compiles, the
/// first compile's error.
pub(crate) fn compile_with_stand_ins<P>(
    profile: &DeviceProfile,
    mut compile: impl FnMut(&VariantKey) -> Result<P, Error>,
) -> Result<(P, VariantKey), Error> {
    let empty = VariantKey::new();
    let first = match compile(&empty) {
        Ok(compiled) => return Ok((compiled, empty)),
        Err(error) => error,
    };
    let Some(slot) = needs_format(&first) else {
        return Err(first);
    };
    let choices: Vec<StorageImageChoice> = STAND_INS
        .iter()
        .filter_map(|&pf| storage_image_choice(pf, profile).ok())
        .collect();
    if choices.is_empty() {
        return Err(first);
    }
    // An odometer over the slots in the order the compile asked for them: on a failure the last
    // slot takes its next stand-in, and an exhausted slot is dropped and its predecessor
    // advanced (the compile asks for the dropped slot again).
    let mut slots: Vec<(SlotRef, usize)> = vec![(slot, 0)];
    loop {
        let mut key = VariantKey::new();
        for &(slot, at) in &slots {
            key.storage_images.insert(slot, choices[at]);
        }
        let error = match compile(&key) {
            Ok(compiled) => return Ok((compiled, key)),
            Err(error) => error,
        };
        match needs_format(&error) {
            Some(slot) if !key.storage_images.contains_key(&slot) && slots.len() < MOST_SLOTS => {
                slots.push((slot, 0));
            }
            _ => loop {
                let Some(last) = slots.last_mut() else {
                    return Err(first);
                };
                last.1 += 1;
                if last.1 < choices.len() {
                    break;
                }
                slots.pop();
            },
        }
    }
}
