//! Where an encoder copies shadowed textures from their buffers and back (`shadow.rs`).

use super::super::error::{internal, Error};
use super::super::host_aliases::{self, Alias};
use super::super::resource::ResourceKey;
use super::super::shadow::{
    self, ShadowBuffer, ShadowCopies, ShadowCopy, ShadowSource, ShadowSync, ShadowTarget,
    ShadowUse, Step,
};
use super::super::tracker::{Access, Use, UseRange};
use ash::vk;
use metal2vulkan::error::Feature;
use std::ops::Range;
use std::sync::Arc;

/// A shadowed texture over `bytes` of the buffer `buffer`.
fn shadow(buffer: ResourceKey, bytes: Range<u64>) -> Arc<ShadowCopies> {
    shadow_joined(buffer, None, bytes)
}

/// A shadowed texture over `bytes` of the buffer `buffer`, which joined the host memory of the
/// key `alias` where given (`host_aliases.rs`).
fn shadow_joined(
    buffer: ResourceKey,
    alias: Option<ResourceKey>,
    bytes: Range<u64>,
) -> Arc<ShadowCopies> {
    Arc::new(ShadowCopies {
        source: ShadowSource::Bytes {
            buffer: ShadowBuffer::Fixed(vk::Buffer::null()),
            key: buffer,
            bytes,
            target: ShadowTarget::Image {
                image: vk::Image::null(),
                regions: Vec::new(),
                staging: None,
            },
            alias,
        },
        shadow_key: ResourceKey::new(),
        shadow_range: UseRange::Image {
            aspects: vk::ImageAspectFlags::COLOR,
            levels: 0..1,
            layers: 0..1,
        },
        tracked: true,
        drops_writes: false,
    })
}

fn shader() -> Access {
    Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    )
}

/// A command's use of `texture`, writing through it when `writes`.
fn through(texture: &Arc<ShadowCopies>, writes: bool) -> (ShadowUse, Use<ResourceKey>) {
    let range = texture.shadow_range.clone();
    let used = if writes {
        Use::writes(texture.shadow_key, true, range, shader())
    } else {
        Use::reads(texture.shadow_key, true, range, shader())
    };
    (
        ShadowUse {
            copies: Arc::clone(texture),
            writes,
            overwrites: false,
            kept: false,
        },
        used,
    )
}

/// A use of `bytes` of `buffer`, writing when `writes`.
fn bytes(buffer: ResourceKey, bytes: Range<u64>, writes: bool) -> Use<ResourceKey> {
    if writes {
        Use::writes(buffer, true, UseRange::Bytes(bytes), shader())
    } else {
        Use::reads(buffer, true, UseRange::Bytes(bytes), shader())
    }
}

/// What a copy moves: into the image (`true`) or back (`false`); a keep and a merge are the
/// tests of their own below.
fn named(copies: &[ShadowCopy<u32>]) -> Vec<(ResourceKey, bool)> {
    copies
        .iter()
        .map(|copy| match copy.step {
            Step::Into => (copy.copies.shadow_key, true),
            Step::Back => (copy.copies.shadow_key, false),
            Step::Keep(_) | Step::Merge(_) => panic!("a keep or a merge"),
        })
        .collect()
}

/// A copy's step, its scratch numbered.
#[derive(Debug, PartialEq, Eq)]
enum Moved {
    Into(ResourceKey),
    Back(ResourceKey),
    Keep(ResourceKey, u32),
    Merge(ResourceKey, u32),
}

fn moved(copies: &[ShadowCopy<u32>]) -> Vec<Moved> {
    copies
        .iter()
        .map(|copy| {
            let key = copy.copies.shadow_key;
            match copy.step {
                Step::Into => Moved::Into(key),
                Step::Back => Moved::Back(key),
                Step::Keep(n) => Moved::Keep(key, n),
                Step::Merge(n) => Moved::Merge(key, n),
            }
        })
        .collect()
}

/// The shadows and uses of a command through `textures` with `others`.
fn command_uses(
    textures: &[(&Arc<ShadowCopies>, bool)],
    others: &[Use<ResourceKey>],
) -> (Vec<ShadowUse>, Vec<Use<ResourceKey>>) {
    let (shadows, mut uses): (Vec<ShadowUse>, Vec<Use<ResourceKey>>) = textures
        .iter()
        .map(|&(texture, writes)| through(texture, writes))
        .unzip();
    uses.extend_from_slice(others);
    (shadows, uses)
}

/// One command of an encoder whose merges' scratch is numbered from `made` on: the copies
/// before it, then its record.
fn planned(
    sync: &mut ShadowSync<u32>,
    made: &mut u32,
    textures: &[(&Arc<ShadowCopies>, bool)],
    others: &[Use<ResourceKey>],
) -> Vec<ShadowCopy<u32>> {
    let (shadows, uses) = command_uses(textures, others);
    let copies = sync
        .before(&shadows, &uses, |_| {
            *made += 1;
            Ok(*made)
        })
        .unwrap();
    sync.after(&shadows, &uses);
    copies
}

/// One command of an encoder that writes no bytes two ways: the copies before it, then its
/// record.
fn command(
    sync: &mut ShadowSync<u32>,
    textures: &[(&Arc<ShadowCopies>, bool)],
    others: &[Use<ResourceKey>],
) -> Vec<(ResourceKey, bool)> {
    named(&planned(sync, &mut 0, textures, others))
}

#[test]
fn an_encoder_copies_a_texture_in_before_its_first_use_and_back_after_its_writes() {
    let buffer = ResourceKey::new();
    let t = shadow(buffer, 256..1280);
    let mut sync = ShadowSync::default();
    assert_eq!(
        command(&mut sync, &[(&t, false)], &[]),
        [(t.shadow_key, true)]
    );
    assert_eq!(command(&mut sync, &[(&t, false)], &[]), []);
    assert_eq!(command(&mut sync, &[(&t, true)], &[]), []);
    assert_eq!(command(&mut sync, &[(&t, true)], &[]), []);
    assert_eq!(named(&sync.finish()), [(t.shadow_key, false)]);
    assert_eq!(named(&sync.finish()), []);
    // A texture only read is not copied back.
    let mut sync = ShadowSync::default();
    command(&mut sync, &[(&t, false)], &[]);
    assert_eq!(named(&sync.finish()), []);
}

/// A shadow whose writes Metal drops (a rotated texture's functions' image) is copied in as any
/// shadow, never copied back, and copied in again at its next use after a command that wrote
/// through it; a command writing it and its bytes another way is no two-way write.
#[test]
fn a_shadow_whose_writes_are_dropped_is_copied_in_again_and_never_back() {
    let buffer = ResourceKey::new();
    let mut copies = Arc::into_inner(shadow(buffer, 256..1280)).expect("one holder");
    copies.drops_writes = true;
    let t = Arc::new(copies);
    let mut sync = ShadowSync::default();
    assert_eq!(
        command(&mut sync, &[(&t, false)], &[]),
        [(t.shadow_key, true)]
    );
    assert_eq!(command(&mut sync, &[(&t, false)], &[]), []);
    assert_eq!(command(&mut sync, &[(&t, true)], &[]), []);
    assert_eq!(
        command(&mut sync, &[(&t, false)], &[]),
        [(t.shadow_key, true)]
    );
    assert_eq!(
        command(&mut sync, &[(&t, true)], &[bytes(buffer, 256..512, true)]),
        []
    );
    assert_eq!(named(&sync.finish()), []);
}

/// A use that replaces every texel before it reads any (a pass's attachment that clears the
/// whole of a one-level, one-slice texture, `ShadowUse::overwrites`) takes no copy in, and the
/// image is current after it: storing it, the pass's writes are copied back as the encoder ends;
/// not storing it, nothing is, and the buffer keeps its bytes. Joined with a use that reads, it
/// is copied in (`shadow::join`), and a texture kept to merge a write two ways still copies in.
#[test]
fn a_use_that_overwrites_every_texel_takes_no_copy_in() {
    let buffer = ResourceKey::new();
    let t = shadow(buffer, 256..1280);
    let overwriting = |writes: bool| ShadowUse {
        copies: Arc::clone(&t),
        writes,
        overwrites: true,
        kept: false,
    };
    let (_, written) = through(&t, true);
    for stores in [true, false] {
        let mut sync = ShadowSync::<u32>::default();
        let shadows = [overwriting(stores)];
        let uses = [written.clone()];
        let copies = sync.before(&shadows, &uses, |_| Ok(0)).unwrap();
        assert_eq!(named(&copies), [], "stores {stores}");
        sync.after(&shadows, &uses);
        assert_eq!(
            command(&mut sync, &[(&t, false)], &[]),
            [],
            "stores {stores}"
        );
        let back: &[(ResourceKey, bool)] = if stores {
            &[(t.shadow_key, false)]
        } else {
            &[]
        };
        assert_eq!(named(&sync.finish()), back, "stores {stores}");
    }
    let mut joined = vec![overwriting(true)];
    shadow::join(&mut joined, &through(&t, false).0);
    assert!(!joined[0].overwrites && joined[0].writes);
    let mut sync = ShadowSync::<u32>::default();
    let copies = sync
        .before(&joined, std::slice::from_ref(&written), |_| Ok(0))
        .unwrap();
    assert_eq!(named(&copies), [(t.shadow_key, true)]);
    // Written two ways (the bytes too), the texture is kept and copied in.
    let mut sync = ShadowSync::<u32>::default();
    let uses = [written, bytes(buffer, 256..512, true)];
    let copies = sync.before(&[overwriting(true)], &uses, |_| Ok(7)).unwrap();
    assert_eq!(
        moved(&copies),
        [Moved::Into(t.shadow_key), Moved::Keep(t.shadow_key, 7)]
    );
}

#[test]
fn a_command_touching_the_bytes_another_way_orders_the_copies_around_it() {
    let buffer = ResourceKey::new();
    let t = shadow(buffer, 256..1280);
    let mut sync = ShadowSync::default();
    command(&mut sync, &[(&t, true)], &[]);
    // Bytes elsewhere in the buffer, or another buffer's: nothing to copy.
    assert_eq!(command(&mut sync, &[], &[bytes(buffer, 0..256, true)]), []);
    let other = ResourceKey::new();
    assert_eq!(command(&mut sync, &[], &[bytes(other, 0..4096, true)]), []);
    // A read of the bytes: the texture's writes first; the image stays current.
    assert_eq!(
        command(&mut sync, &[], &[bytes(buffer, 0..4096, false)]),
        [(t.shadow_key, false)]
    );
    assert_eq!(command(&mut sync, &[(&t, false)], &[]), []);
    // A write of the bytes: the next use copies them in again.
    command(&mut sync, &[(&t, true)], &[]);
    assert_eq!(
        command(&mut sync, &[], &[bytes(buffer, 1024..1028, true)]),
        [(t.shadow_key, false)]
    );
    assert_eq!(
        command(&mut sync, &[(&t, false)], &[]),
        [(t.shadow_key, true)]
    );
    assert_eq!(named(&sync.finish()), []);
}

#[test]
fn textures_over_the_same_bytes_see_each_others_writes() {
    let buffer = ResourceKey::new();
    let a = shadow(buffer, 0..1024);
    let b = shadow(buffer, 512..2048);
    let apart = shadow(buffer, 4096..8192);
    let mut sync = ShadowSync::default();
    assert_eq!(
        command(&mut sync, &[(&a, true), (&apart, false)], &[]),
        [(a.shadow_key, true), (apart.shadow_key, true)]
    );
    // `a` written: `b` is copied in after `a` is copied back.
    assert_eq!(
        command(&mut sync, &[(&b, false)], &[]),
        [(a.shadow_key, false), (b.shadow_key, true)]
    );
    // `b` written: `a` is copied in again at its next use, after `b`'s copy back.
    command(&mut sync, &[(&b, true)], &[]);
    assert_eq!(
        command(&mut sync, &[(&a, false), (&apart, false)], &[]),
        [(b.shadow_key, false), (a.shadow_key, true)]
    );
    assert_eq!(named(&sync.finish()), []);
}

/// Two textures over two no-copy buffers made over the same host memory, the second of which
/// joined the first's (`host_aliases.rs`): each sees the other's writes as textures over one
/// buffer do, and a write of the second buffer's bytes, which carries a use of the first's key
/// (`HostAliases::augment`), makes the first copy in again; a third buffer's do not.
#[test]
fn textures_over_two_buffers_of_one_memory_see_each_others_writes() {
    let (first, second, other) = (ResourceKey::new(), ResourceKey::new(), ResourceKey::new());
    let a = shadow(first, 0..1024);
    let b = shadow_joined(second, Some(first), 0..1024);
    let c = shadow(other, 0..1024);
    let aliases = std::collections::HashMap::from([(second, Alias { key: first })]);
    let augmented = |uses: &[Use<ResourceKey>]| {
        host_aliases::augmented(&aliases, None, uses).unwrap_or_else(|| uses.to_vec())
    };
    let mut sync = ShadowSync::default();
    assert_eq!(
        command(&mut sync, &[(&a, true), (&c, false)], &[]),
        [(a.shadow_key, true), (c.shadow_key, true)]
    );
    // `a` written: `b` is copied in after `a` is copied back; `c` lies elsewhere.
    assert_eq!(
        command(&mut sync, &[(&b, false), (&c, false)], &[]),
        [(a.shadow_key, false), (b.shadow_key, true)]
    );
    // `b` written: `a` is copied in again at its next use, after `b`'s copy back.
    command(&mut sync, &[(&b, true)], &[]);
    assert_eq!(
        command(&mut sync, &[(&a, false)], &[]),
        [(b.shadow_key, false), (a.shadow_key, true)]
    );
    // The second buffer's bytes written: `a` copies in again, and so does `b`.
    let written = augmented(&[bytes(second, 0..64, true)]);
    assert_eq!(written.len(), 2);
    assert_eq!(command(&mut sync, &[], &written), []);
    assert_eq!(
        command(&mut sync, &[(&a, false), (&b, false)], &[]),
        [(a.shadow_key, true), (b.shadow_key, true)]
    );
    // The third buffer's bytes written: neither does.
    command(&mut sync, &[], &augmented(&[bytes(other, 0..64, true)]));
    assert_eq!(command(&mut sync, &[(&a, false), (&b, false)], &[]), []);
    assert_eq!(named(&sync.finish()), []);
}

/// One command that writes through a texture and its bytes another way (the buffer, or another
/// texture over them) keeps the texture after its copies in, and the merge goes before the next
/// command, whatever it touches, or at the end, after which the texture is copied in again at
/// its next use. Reading the bytes, writing bytes the texture does not lie in, or reading the
/// other texture is one way.
#[test]
fn one_command_writing_a_texture_and_its_bytes_another_way_keeps_and_merges_it() {
    let buffer = ResourceKey::new();
    let a = shadow(buffer, 0..1024);
    let b = shadow(buffer, 512..2048);
    let elsewhere = ResourceKey::new();
    let (ka, kb) = (a.shadow_key, b.shadow_key);
    let mut sync = ShadowSync::default();
    let mut made = 0;
    let mut run = |sync: &mut ShadowSync<u32>,
                   textures: &[(&Arc<ShadowCopies>, bool)],
                   others: &[Use<ResourceKey>]| {
        moved(&planned(sync, &mut made, textures, others))
    };
    assert_eq!(
        run(&mut sync, &[(&a, true)], &[bytes(buffer, 0..4096, true)]),
        [Moved::Into(ka), Moved::Keep(ka, 1)]
    );
    // The merge before a command of other bytes, which leaves `a` to copy in again.
    assert_eq!(
        run(&mut sync, &[], &[bytes(elsewhere, 0..64, true)]),
        [Moved::Merge(ka, 1)]
    );
    assert_eq!(run(&mut sync, &[(&a, false)], &[]), [Moved::Into(ka)]);
    // Two textures over the same bytes, both written: each kept, each merged at the end.
    assert_eq!(
        run(&mut sync, &[(&a, true), (&b, true)], &[]),
        [Moved::Into(kb), Moved::Keep(ka, 2), Moved::Keep(kb, 3)]
    );
    assert_eq!(
        moved(&sync.finish()),
        [Moved::Merge(ka, 2), Moved::Merge(kb, 3)]
    );
    // Two ways twice in a row: the first merge, `a` copied in again, its second keep.
    let mut sync = ShadowSync::default();
    let written = [bytes(buffer, 1020..1028, true)];
    run(&mut sync, &[(&a, true)], &written);
    assert_eq!(
        run(&mut sync, &[(&a, true)], &written),
        [Moved::Merge(ka, 4), Moved::Into(ka), Moved::Keep(ka, 5)]
    );
    assert_eq!(moved(&sync.finish()), [Moved::Merge(ka, 5)]);
    // One way: no keep, and a copy back at the end.
    for others in [
        vec![bytes(buffer, 0..4096, false)],
        vec![bytes(buffer, 1024..4096, true)],
    ] {
        let mut sync = ShadowSync::default();
        assert_eq!(run(&mut sync, &[(&a, true)], &others), [Moved::Into(ka)]);
        assert_eq!(moved(&sync.finish()), [Moved::Back(ka)]);
    }
    let mut sync = ShadowSync::default();
    assert_eq!(
        run(&mut sync, &[(&a, true), (&b, false)], &[]),
        [Moved::Into(ka), Moved::Into(kb)]
    );
}

/// A command whose scratch is refused changes nothing: the next command copies the texture in.
#[test]
fn a_refused_scratch_changes_nothing() {
    let buffer = ResourceKey::new();
    let a = shadow(buffer, 0..1024);
    let mut sync = ShadowSync::<u32>::default();
    let (shadows, uses) = command_uses(&[(&a, true)], &[bytes(buffer, 0..4096, true)]);
    let refused = sync.before(&shadows, &uses, |_| Err(internal("no scratch")));
    assert!(
        matches!(refused, Err(Error::Compile(_))),
        "{:?}",
        refused.err()
    );
    assert_eq!(
        command(&mut sync, &[(&a, false)], &[]),
        [(a.shadow_key, true)]
    );
}

/// A shadow of `levels` and `layers` of the image `image`: a texture view of another component
/// layout on a device whose Vulkan views reinterpret none (`texture/view_shadow.rs`).
fn view_of(image: ResourceKey, levels: Range<u32>, layers: Range<u32>) -> Arc<ShadowCopies> {
    let range = UseRange::Image {
        aspects: vk::ImageAspectFlags::COLOR,
        levels,
        layers,
    };
    Arc::new(ShadowCopies {
        source: ShadowSource::Image {
            image: vk::Image::null(),
            key: image,
            range: range.clone(),
            shadow: vk::Image::null(),
            regions: Vec::new(),
        },
        shadow_key: ResourceKey::new(),
        shadow_range: range,
        tracked: true,
        drops_writes: false,
    })
}

/// A use of `levels` and `layers` of the image `image`, writing when `writes`.
fn texels(
    image: ResourceKey,
    levels: Range<u32>,
    layers: Range<u32>,
    writes: bool,
) -> Use<ResourceKey> {
    let range = UseRange::Image {
        aspects: vk::ImageAspectFlags::COLOR,
        levels,
        layers,
    };
    if writes {
        Use::writes(image, true, range, shader())
    } else {
        Use::reads(image, true, range, shader())
    }
}

/// A view's shadow of an image follows the rules a texture over a buffer does, the image in the
/// buffer's place: copied in before its first use, back before a command that touches the
/// image's levels and layers it presents another way and at the end of the encoder, and in again
/// after a command writes them another way. Other levels or layers, or another image, touch
/// nothing.
#[test]
fn a_views_shadow_of_an_image_is_copied_as_a_shadow_of_bytes_is() {
    let image = ResourceKey::new();
    let v = view_of(image, 1..2, 0..2);
    let mut sync = ShadowSync::default();
    assert_eq!(
        command(&mut sync, &[(&v, false)], &[]),
        [(v.shadow_key, true)]
    );
    assert_eq!(command(&mut sync, &[(&v, true)], &[]), []);
    // Another level, another layer, another image: nothing to copy.
    let other = ResourceKey::new();
    for elsewhere in [
        texels(image, 0..1, 0..2, true),
        texels(image, 1..2, 2..3, true),
        texels(other, 0..4, 0..4, true),
    ] {
        assert_eq!(command(&mut sync, &[], &[elsewhere]), []);
    }
    // A read of the texels: the view's writes first; the shadow stays current.
    assert_eq!(
        command(&mut sync, &[], &[texels(image, 0..2, 1..2, false)]),
        [(v.shadow_key, false)]
    );
    assert_eq!(command(&mut sync, &[(&v, false)], &[]), []);
    // A write of them: the next use copies them in again.
    assert_eq!(
        command(&mut sync, &[], &[texels(image, 1..2, 0..1, true)]),
        []
    );
    assert_eq!(
        command(&mut sync, &[(&v, true)], &[]),
        [(v.shadow_key, true)]
    );
    assert_eq!(named(&sync.finish()), [(v.shadow_key, false)]);
}

/// One command that writes an image both through a view's shadow and another way (the image,
/// or another view's shadow of the same texels) is refused, typed, before any merge is made
/// (only a buffer's bytes are merged), and changes nothing. Writing through one alias and reading
/// through the other, or writing levels the shadow does not present, is one way: the shadow
/// holds the texels as they were before the command.
#[test]
fn a_command_writing_an_image_through_a_views_shadow_and_another_way_is_refused() {
    let image = ResourceKey::new();
    let v = view_of(image, 0..1, 0..1);
    let w = view_of(image, 0..2, 0..1);
    let refusals: [(&[(&Arc<ShadowCopies>, bool)], Vec<Use<ResourceKey>>); 3] = [
        (&[(&v, true)], vec![texels(image, 0..1, 0..1, true)]),
        (&[(&v, true), (&w, true)], Vec::new()),
        (
            &[(&v, false), (&w, true)],
            vec![texels(image, 0..1, 0..1, true)],
        ),
    ];
    for (textures, others) in refusals {
        let mut sync = ShadowSync::<u32>::default();
        let (shadows, uses) = command_uses(textures, &others);
        let refused = sync.before(&shadows, &uses, |_| panic!("a merge of an image's shadow"));
        match refused {
            Err(Error::Unsupported(gap)) => {
                assert_eq!(gap.feature, Feature::ViewFormatReinterpretation)
            }
            other => panic!("{:?}", other.map(|copies| named(&copies))),
        }
        assert_eq!(
            command(&mut sync, &[(&v, false)], &[]),
            [(v.shadow_key, true)]
        );
    }
    let one_way: [(&[(&Arc<ShadowCopies>, bool)], Vec<Use<ResourceKey>>); 4] = [
        (&[(&v, true)], vec![texels(image, 0..1, 0..1, false)]),
        (&[(&v, false)], vec![texels(image, 0..1, 0..1, true)]),
        (&[(&v, true), (&w, false)], Vec::new()),
        (&[(&v, true)], vec![texels(image, 1..2, 0..1, true)]),
    ];
    for (textures, others) in one_way {
        let mut sync = ShadowSync::<u32>::default();
        let (shadows, uses) = command_uses(textures, &others);
        let copies = sync
            .before(&shadows, &uses, |_| panic!("a merge of an image's shadow"))
            .unwrap();
        let into: Vec<(ResourceKey, bool)> = textures
            .iter()
            .map(|(texture, _)| (texture.shadow_key, true))
            .collect();
        assert_eq!(named(&copies), into);
    }
}
