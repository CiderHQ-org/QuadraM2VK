//! The stand-in search of a pipeline state's creation compile (`pipeline::stand_in`), over a
//! compile that asks for storage formats as a module does.

use super::super::pipeline::stand_in::{compile_with_stand_ins, MOST_SLOTS, STAND_INS};
use super::image_plan::committed;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::error::{Error, Feature, RuntimeInput, Unsupported};
use metal2vulkan::metal::format::storage_image_choice;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{SlotClass, SlotRef, Stage, StorageImageChoice};

fn slot(index: u32) -> SlotRef {
    SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index,
    }
}

fn needs(index: u32) -> Error {
    Error::NeedsInput(RuntimeInput::StorageTextureFormat { slot: slot(index) })
}

/// A compile of a function whose storage slots `wants` read texels of the stand-in at each
/// slot's index into [`STAND_INS`]: it asks for the first slot without a choice, in order,
/// refuses a choice of another texel type as a module does (`NotRepresentable(Format)`), and
/// counts its calls.
fn module<'a>(
    choices: &'a [StorageImageChoice],
    wants: &'a [(u32, usize)],
    calls: &'a mut usize,
) -> impl FnMut(&VariantKey) -> Result<u32, Error> + 'a {
    move |key: &VariantKey| {
        *calls += 1;
        for &(index, want) in wants {
            match key.storage_images.get(&slot(index)) {
                None => return Err(needs(index)),
                Some(choice) if *choice != choices[want] => {
                    return Err(Unsupported::not_representable(Feature::Format).into())
                }
                Some(_) => {}
            }
        }
        Ok(7)
    }
}

#[test]
fn a_compile_that_needs_no_format_takes_no_stand_in() {
    let (_, _, device) = committed().remove(0);
    let mut calls = 0;
    let (compiled, key) = compile_with_stand_ins(device.profile(), |_: &VariantKey| {
        calls += 1;
        Ok::<_, Error>(1)
    })
    .unwrap();
    assert_eq!((compiled, calls), (1, 1));
    assert_eq!(key, VariantKey::new());
}

#[test]
fn each_slot_takes_the_first_stand_in_its_texels_take() {
    // The compile under these stand-ins is the compile of a bind of their formats, and of no
    // other (`cache::tests::a_stand_in_compile_is_the_compile_of_its_formats_bind_and_no_other`).
    assert_eq!(
        STAND_INS,
        [
            PixelFormat::R32Float,
            PixelFormat::R32Uint,
            PixelFormat::R32Sint
        ]
    );
    let mut searched = 0;
    for (name, _, device) in committed() {
        let profile = device.profile();
        let choices: Vec<StorageImageChoice> = STAND_INS
            .iter()
            .map(|&pf| storage_image_choice(pf, profile).unwrap())
            .collect();
        // Distinct choices are what a module tells apart.
        if choices[0] == choices[1] || choices[1] == choices[2] || choices[0] == choices[2] {
            continue;
        }
        searched += 1;
        for wants in [
            vec![(0, 0)],
            vec![(3, 2)],
            vec![(0, 1), (1, 2)],
            vec![(5, 2), (1, 0), (2, 1)],
        ] {
            let mut calls = 0;
            let (compiled, key) =
                compile_with_stand_ins(profile, module(&choices, &wants, &mut calls))
                    .unwrap_or_else(|e| panic!("{name} {wants:?}: {e:?}"));
            assert_eq!(compiled, 7, "{name}");
            assert_eq!(key.storage_images.len(), wants.len(), "{name} {wants:?}");
            for &(index, want) in &wants {
                assert_eq!(key.storage_images[&slot(index)], choices[want], "{name}");
            }
            assert!(
                calls <= 1 + 3usize.pow(wants.len() as u32) * wants.len(),
                "{name} {wants:?}: {calls} compiles"
            );
        }
    }
    assert!(
        searched > 0,
        "no committed profile tells the stand-ins apart"
    );
}

#[test]
fn a_compile_no_stand_in_satisfies_refuses_with_its_first_reason() {
    let (_, _, device) = committed().remove(0);
    let profile = device.profile();
    // A slot whose texels no stand-in holds.
    let mut calls = 0;
    let error = compile_with_stand_ins(profile, |key: &VariantKey| {
        calls += 1;
        match key.storage_images.get(&slot(0)) {
            None => Err::<(), _>(needs(0)),
            Some(_) => Err(Unsupported::not_representable(Feature::Format).into()),
        }
    })
    .unwrap_err();
    assert_eq!(error, needs(0));
    assert_eq!(calls, 1 + STAND_INS.len());
    // Another first refusal is returned as it is, with no stand-in tried.
    let mut calls = 0;
    let refusal: Error = Unsupported::not_representable(Feature::Imageblocks).into();
    let error = compile_with_stand_ins(profile, |_: &VariantKey| {
        calls += 1;
        Err::<(), _>(refusal.clone())
    })
    .unwrap_err();
    assert_eq!((error, calls), (refusal, 1));
    // More slots than the search covers: refused, after a bounded search.
    let mut calls = 0;
    let error = compile_with_stand_ins(profile, |key: &VariantKey| {
        calls += 1;
        let missing = (0..).find(|&i| !key.storage_images.contains_key(&slot(i)));
        Err::<(), _>(needs(missing.unwrap()))
    })
    .unwrap_err();
    assert_eq!(error, needs(0));
    assert!(
        calls <= 2 + 3usize.pow(MOST_SLOTS as u32) * MOST_SLOTS,
        "{calls}"
    );
}
