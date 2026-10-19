use super::{
    typed_module, TiledTexelView, TiledWordsLevel, TiledWordsModule, TiledWordsStep, BYTES, IMAGE,
    KEPT,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// Valid on every committed profile that has what it needs: the `R32G32_UINT` kernel's `Rg32ui`
/// view needs extended storage image formats, which the Vulkan 1.2 floor lacks.
#[test]
fn the_tiled_word_kernels_are_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        for view in TiledTexelView::ALL {
            let m = typed_module(view).expect("module");
            if view == TiledTexelView::R32G32Uint && p.check(m.requirements()).is_err() {
                continue;
            }
            assert_valid(m, &p);
        }
    }
}

/// The Morton index of `(x, y)` in a tile of `2^tx x 2^ty` elements, bit by bit: x's bit lowest,
/// the longer side's extra bits on top.
fn morton(x: u32, y: u32, tx: u32, ty: u32) -> u32 {
    let (mut m, mut k) = (0, 0);
    for i in 0..tx.max(ty) {
        if i < tx {
            m |= ((x >> i) & 1) << k;
            k += 1;
        }
        if i < ty {
            m |= ((y >> i) & 1) << k;
            k += 1;
        }
    }
    m
}

/// The word of element `(x, y)` of layer `z` of `level` from the texture's first, of an element
/// `n` words wide.
fn rel(level: &TiledWordsLevel, n: u32, z: u32, x: u32, y: u32) -> usize {
    let [tw, th] = level.tile;
    let tile = (y / th) * level.across + x / tw;
    let inside = morton(x % tw, y % th, tw.trailing_zeros(), th.trailing_zeros());
    let element = tile * tw * th + inside;
    (level.first / 4 + z * level.chain_stride / 4 + element * n) as usize
}

/// The components an element of `words` takes through `view`.
fn texel(view: TiledTexelView, words: &[u32]) -> [u32; 4] {
    let mut out = Vec::new();
    for &word in words {
        match view {
            TiledTexelView::R8G8B8A8Uint => out.extend(word.to_le_bytes().map(u32::from)),
            TiledTexelView::R16G16B16A16Uint => out.extend([word & 0xffff, word >> 16]),
            _ => out.push(word),
        }
    }
    out.resize(4, 0);
    [out[0], out[1], out[2], out[3]]
}

/// The words of an element whose components through `view` are `texel`.
fn words_of(view: TiledTexelView, texel: [u32; 4]) -> Vec<u32> {
    match view {
        TiledTexelView::R8G8B8A8Uint => vec![u32::from_le_bytes(texel.map(|c| c as u8))],
        TiledTexelView::R16G16B16A16Uint => {
            vec![texel[0] | (texel[1] << 16), texel[2] | (texel[3] << 16)]
        }
        _ => texel[..view.words() as usize].to_vec(),
    }
}

/// An image of `level`'s extent and layers whose texel components through `view` name their
/// place and `salt`, each within the view's component range.
fn named_image(view: TiledTexelView, level: &TiledWordsLevel, salt: u32) -> Image {
    let [w, h] = level.extent;
    let mut image = Image::new(IMAGE, [w, h, 1], level.images, 1, 1);
    let words = view.words();
    for z in 0..level.images {
        for y in 0..h {
            for x in 0..w {
                let place = (z * h + y) * w + x;
                let element: Vec<u32> = (0..words)
                    .map(|k| (salt << 24) ^ (place * 131 + k * 17 + 0x5a00))
                    .collect();
                image.texels[place as usize] = texel(view, &element);
            }
        }
    }
    image
}

/// The bytes, kept words and image after one dispatch of `step` over `level` through `view`,
/// the bytes' binding `skip` words before the texture's first, in both invocation orders.
fn stepped(
    view: TiledTexelView,
    step: TiledWordsStep,
    (level, skip): (&TiledWordsLevel, u32),
    (bytes, kept, image): (&[u32], &[u32], Image),
) -> (Vec<u32>, Vec<u32>, Image) {
    let m = typed_module(view).expect("module");
    let shape = TiledWordsModule::shape(step, level, skip).expect("a shape");
    let to_bytes =
        |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let to_words = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    };
    let mut out = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let part = |slot, words: &[u32]| Buffer {
            slot: Some(slot),
            address: None,
            bytes: to_bytes(words),
        };
        let res = Resources {
            buffers: vec![part(BYTES, bytes), part(KEPT, kept)],
            push: shape.to_vec(),
            spec: Default::default(),
        };
        let cfg = Config {
            order,
            workgroups: TiledWordsModule::groups(level),
            ..Config::default()
        };
        let (outcome, mut images) =
            run_with_images(&m, &cfg, res, vec![image.clone()]).expect("run");
        out.push((
            to_words(&outcome.buffers[0]),
            to_words(&outcome.buffers[1]),
            images.swap_remove(0),
        ));
    }
    let (a, b) = (&out[0], &out[1]);
    assert!(a.0 == b.0 && a.1 == b.1 && a.2 == b.2, "order independent");
    out.swap_remove(0)
}

/// A level of 13 x 9 elements in two layers, tiles of 8 x 4 two a row, so a tile row and a tile
/// column are partly past the extent, from byte 40 of a chain of 1024 bytes a chain; the bytes'
/// binding three words before the texture's first.
fn level_of(view: TiledTexelView) -> (TiledWordsLevel, u32, usize) {
    let n = view.words();
    // Three rows of two tiles of 32 elements each, past the level's first word.
    let region = 3 * 2 * 32 * n * 4;
    let stride = (40 + region).next_multiple_of(256) + 64;
    let level = TiledWordsLevel::new([13, 9], 2, [8, 4], 2, 40, stride);
    let skip = 3;
    let len = (skip * 4 + 2 * stride) as usize / 4;
    (level, skip, len)
}

/// Every element of `level`: its layer, x and y.
fn elements(level: &TiledWordsLevel) -> Vec<(u32, u32, u32)> {
    let [w, h] = level.extent;
    let mut out = Vec::new();
    for z in 0..level.images {
        for y in 0..h {
            for x in 0..w {
                out.push((z, x, y));
            }
        }
    }
    out
}

/// In: each texel of each layer is its element's words in the bytes, through every view; the
/// bytes are only read, and the kept words are written where the step keeps, with the element's
/// words at their place and nothing else.
#[test]
fn a_level_moves_into_the_image_and_the_kept_words() {
    for view in TiledTexelView::ALL {
        let (level, skip, len) = level_of(view);
        let n = view.words();
        let bytes: Vec<u32> = (0..len as u32)
            .map(|i| 0x3c00_0000 ^ i.wrapping_mul(2654435761))
            .collect();
        let kept = vec![0xdead_beef; len];
        let blank = Image::new(IMAGE, [13, 9, 1], 2, 1, 1);
        let mut want_image = blank.clone();
        let mut want_kept = kept.clone();
        for (z, x, y) in elements(&level) {
            let r = rel(&level, n, z, x, y);
            let at = skip as usize + r;
            let words = &bytes[at..at + n as usize];
            want_image.texels[((z * 9 + y) * 13 + x) as usize] = texel(view, words);
            want_kept[r..r + n as usize].copy_from_slice(words);
        }
        for step in [TiledWordsStep::In, TiledWordsStep::InKeep] {
            let (b, k, image) = stepped(view, step, (&level, skip), (&bytes, &kept, blank.clone()));
            assert_eq!(b, bytes, "{view:?} {step:?}: the bytes are only read");
            assert_eq!(image, want_image, "{view:?} {step:?}: the image");
            match step {
                TiledWordsStep::InKeep => assert_eq!(k, want_kept, "{view:?}: the kept words"),
                _ => assert_eq!(k, kept, "{view:?}: nothing kept"),
            }
        }
    }
}

/// Keep and back: each element's words at its place are its texel's, in the kept words or the
/// bytes; every other word is as it was, and the image is only read.
#[test]
fn a_level_moves_out_of_the_image() {
    for view in TiledTexelView::ALL {
        let (level, skip, len) = level_of(view);
        let n = view.words() as usize;
        let image = named_image(view, &level, 1);
        let bytes: Vec<u32> = (0..len as u32).map(|i| 0x7100_0000 | i).collect();
        let kept = vec![0xdead_beef; len];
        let mut want_kept = kept.clone();
        let mut want_bytes = bytes.clone();
        for (z, x, y) in elements(&level) {
            let r = rel(&level, n as u32, z, x, y);
            let words = words_of(view, image.texels[((z * 9 + y) * 13 + x) as usize]);
            want_kept[r..r + n].copy_from_slice(&words);
            let at = skip as usize + r;
            want_bytes[at..at + n].copy_from_slice(&words);
        }
        let (b, k, after) = stepped(
            view,
            TiledWordsStep::Keep,
            (&level, skip),
            (&bytes, &kept, image.clone()),
        );
        assert_eq!(
            (b, after.clone()),
            (bytes.clone(), image.clone()),
            "{view:?}: keep reads"
        );
        assert_eq!(k, want_kept, "{view:?}: the kept words");
        let (b, k, after) = stepped(
            view,
            TiledWordsStep::Back,
            (&level, skip),
            (&bytes, &kept, image.clone()),
        );
        assert_eq!(
            (k, after),
            (kept.clone(), image.clone()),
            "{view:?}: back reads"
        );
        assert_eq!(b, want_bytes, "{view:?}: the bytes");
    }
}

/// Merge: where an element's texel differs from its kept words, each byte that differs becomes
/// the texel's in the bytes, and every other byte keeps what the bytes hold now (another writer's
/// bytes beside it included); a word equal to its kept word is left as the bytes hold it.
#[test]
fn a_merge_writes_only_the_bytes_the_texels_changed() {
    for view in TiledTexelView::ALL {
        let (level, skip, len) = level_of(view);
        let n = view.words() as usize;
        let image = named_image(view, &level, 2);
        // The kept words: the image's, but for a byte of every third element and every word of
        // every seventh.
        let mut kept = vec![0x0bad_f00d; len];
        for (i, (z, x, y)) in elements(&level).into_iter().enumerate() {
            let r = rel(&level, n as u32, z, x, y);
            let mut words = words_of(view, image.texels[((z * 9 + y) * 13 + x) as usize]);
            if i % 3 == 0 {
                words[0] ^= 0x0000_ff00;
            }
            if i % 7 == 0 {
                for word in &mut words {
                    *word = !*word;
                }
            }
            kept[r..r + n].copy_from_slice(&words);
        }
        // The bytes another writer left: none of the texels' or the kept words'.
        let bytes: Vec<u32> = (0..len as u32).map(|i| 0x9e00_0000 ^ (i * 40503)).collect();
        let mut want = bytes.clone();
        for (z, x, y) in elements(&level) {
            let r = rel(&level, n as u32, z, x, y);
            let words = words_of(view, image.texels[((z * 9 + y) * 13 + x) as usize]);
            for k in 0..n {
                let differ = words[k] ^ kept[r + k];
                let mut mask = 0u32;
                for byte in 0..4 {
                    if differ & (0xff << (8 * byte)) != 0 {
                        mask |= 0xff << (8 * byte);
                    }
                }
                let at = skip as usize + r + k;
                want[at] = (want[at] & !mask) | (words[k] & mask);
            }
        }
        let (b, k, after) = stepped(
            view,
            TiledWordsStep::Merge,
            (&level, skip),
            (&bytes, &kept, image.clone()),
        );
        assert_eq!(
            (k, after),
            (kept.clone(), image.clone()),
            "{view:?}: merge reads"
        );
        assert_eq!(b, want, "{view:?}: the merged bytes");
        assert_ne!(b, bytes, "{view:?}: some bytes changed");
    }
}

/// A tile wider than tall, and one taller than wide, lay the longer side's extra bits on top:
/// in and back again leave every element's words where the layout puts them.
#[test]
fn oblong_tiles_take_the_longer_side_on_top() {
    let view = TiledTexelView::R32Uint;
    for tile in [[16, 4], [2, 8]] {
        let across = 20u32.div_ceil(tile[0]);
        let rows = 6u32.div_ceil(tile[1]);
        let stride = across * rows * tile[0] * tile[1] * 4;
        let level = TiledWordsLevel::new([20, 6], 1, tile, across, 0, stride);
        let len = (stride / 4) as usize;
        let bytes: Vec<u32> = (0..len as u32).map(|i| 0x1000 + i).collect();
        let blank = Image::new(IMAGE, [20, 6, 1], 1, 1, 1);
        let (_, _, image) = stepped(view, TiledWordsStep::In, (&level, 0), (&bytes, &[0], blank));
        for (z, x, y) in elements(&level) {
            let [tw, th] = tile;
            let tile_index = (y / th) * across + x / tw;
            let m = morton(x % tw, y % th, tw.trailing_zeros(), th.trailing_zeros());
            assert_eq!(rel(&level, 1, z, x, y), (tile_index * tw * th + m) as usize);
            let want = bytes[rel(&level, 1, z, x, y)];
            assert_eq!(
                image.texels[(y * 20 + x) as usize][0],
                want,
                "{tile:?} ({x}, {y})"
            );
        }
    }
}

/// The shape refuses what its fields cannot hold.
#[test]
fn the_shape_refuses_what_it_cannot_hold() {
    let level =
        |extent, tile, across, first| TiledWordsLevel::new(extent, 1, tile, across, first, 0);
    let step = TiledWordsStep::In;
    assert!(TiledWordsModule::shape(step, &level([16384, 16384], [64, 64], 256, 0), 63).is_some());
    assert!(TiledWordsModule::shape(step, &level([65536, 1], [64, 64], 1, 0), 0).is_none());
    assert!(TiledWordsModule::shape(step, &level([64, 64], [256, 64], 1, 0), 0).is_none());
    assert!(TiledWordsModule::shape(step, &level([64, 64], [48, 64], 1, 0), 0).is_none());
    assert!(TiledWordsModule::shape(step, &level([64, 64], [64, 64], 1, 2), 0).is_none());
    assert!(TiledWordsModule::shape(step, &level([64, 64], [64, 64], 1, 0), 64).is_none());
}

/// The bytes the staged moves leave (`tiled_texels` with copies, and the merge kernel, restated
/// on the CPU), and the moves through the views alike: the copy in and its keep, commands that
/// change some texels (a byte of some, every byte of others), another writer's writes to the
/// bytes between the copy in and the merge (in elements the commands left, in elements they
/// changed, and between the elements), and the merge. The staged keep and changed copies lay the
/// image's elements into the bytes as the copy in read them, the bytes between the elements as
/// they were; the merge takes, in each word where they differ, the changed copy's bytes that
/// differ.
#[test]
fn the_moves_leave_the_bytes_the_staged_moves_leave() {
    for view in TiledTexelView::ALL {
        let (level, skip, len) = level_of(view);
        let n = view.words() as usize;
        let at_in: Vec<u32> = (0..len as u32)
            .map(|i| 0x2200_0000 ^ i.wrapping_mul(0x9e37_79b9))
            .collect();
        // Through the views: the copy in and its keep.
        let blank = Image::new(IMAGE, [13, 9, 1], 2, 1, 1);
        let (b, kept, image) = stepped(
            view,
            TiledWordsStep::InKeep,
            (&level, skip),
            (&at_in, &vec![0xdead_beef; len], blank),
        );
        assert_eq!(b, at_in, "{view:?}: the copy in reads");
        // The commands' texels: a byte of every third element, every word of every fifth.
        let mut after = image.clone();
        for (i, (z, x, y)) in elements(&level).into_iter().enumerate() {
            let at = ((z * 9 + y) * 13 + x) as usize;
            let mut words = words_of(view, after.texels[at]);
            if i % 3 == 0 {
                words[0] ^= 0x00ff_0000;
            }
            if i % 5 == 0 {
                for word in &mut words {
                    *word = !*word;
                }
            }
            after.texels[at] = texel(view, &words);
        }
        // Another writer's bytes meanwhile: every seventh word of the texture's bytes.
        let now: Vec<u32> = (0..len)
            .map(|i| match i >= skip as usize && i % 7 == 0 {
                true => 0x7700_0000 | i as u32,
                false => at_in[i],
            })
            .collect();
        // The staged keep and changed copies, from the texture's first byte.
        let mut kept_staged = at_in[skip as usize..].to_vec();
        let mut changed_staged = kept_staged.clone();
        for (z, x, y) in elements(&level) {
            let r = rel(&level, n as u32, z, x, y);
            let at = ((z * 9 + y) * 13 + x) as usize;
            kept_staged[r..r + n].copy_from_slice(&words_of(view, image.texels[at]));
            changed_staged[r..r + n].copy_from_slice(&words_of(view, after.texels[at]));
        }
        let mut staged = now.clone();
        for (w, (k, c)) in kept_staged.iter().zip(&changed_staged).enumerate() {
            let differ = k ^ c;
            if differ == 0 {
                continue;
            }
            let mut mask = 0u32;
            for byte in 0..4 {
                if differ & (0xff << (8 * byte)) != 0 {
                    mask |= 0xff << (8 * byte);
                }
            }
            let at = skip as usize + w;
            staged[at] = (staged[at] & !mask) | (c & mask);
        }
        let (merged, _, _) = stepped(
            view,
            TiledWordsStep::Merge,
            (&level, skip),
            (&now, &kept, after),
        );
        assert_eq!(merged, staged, "{view:?}: the merged bytes");
        // And the staged copy in's texels are the views'.
        for (z, x, y) in elements(&level) {
            let r = rel(&level, n as u32, z, x, y);
            let at = ((z * 9 + y) * 13 + x) as usize;
            let from = skip as usize + r;
            assert_eq!(
                image.texels[at],
                texel(view, &at_in[from..from + n]),
                "{view:?}: the texel at ({x}, {y}) of layer {z}"
            );
        }
    }
}
