//! Range arithmetic at extremes, the residual, the resident set, and push fields' bytes.

use super::*;

const MAX: u64 = u64::MAX;

/// The contract every range meets: aligned start, at least 16 bytes, inside the padded
/// allocation, and every Metal-defined byte of the window that lies in the allocation is at
/// `residual + m` inside the range.
fn check(bind_offset: u64, length: u64, window: Window, a: u64) {
    let (r, residual) = descriptor_range(bind_offset, length, window, a, MAX)
        .unwrap_or_else(|e| panic!("{bind_offset} {length} {window:?} {a}: {e:?}"));
    let end = padded(length);
    let a4 = a.max(4);
    assert_eq!(r.offset % a4, 0, "aligned start");
    assert!(r.range >= 16, "at least 16 bytes");
    assert!(r.offset + r.range <= end, "inside the padded allocation");
    assert_eq!(
        u64::from(residual),
        bind_offset.wrapping_sub(r.offset) & 0xFFFF_FFFF,
        "residual is bind offset minus base, mod 2^32"
    );
    let (start, len) = match window {
        Window::Bounded { start, len } => (i128::from(start), i128::from(len)),
        Window::Whole => (0, i128::from(end) - i128::from(bind_offset)),
    };
    // Probe the window's first and last byte and its middle.
    for m in [start, start + len / 2, start + len - 1] {
        if len <= 0 {
            break;
        }
        let abs = i128::from(bind_offset) + m;
        if abs < 0 || abs >= i128::from(end) {
            continue;
        }
        let rel = (i128::from(residual) + m).rem_euclid(1 << 32);
        let expect = abs - i128::from(r.offset);
        assert_eq!(rel, expect.rem_euclid(1 << 32), "byte {m} of {window:?}");
        assert!(
            expect >= 0 && expect < i128::from(r.range),
            "byte {m} of {window:?} at {bind_offset} inside [{}, +{})",
            r.offset,
            r.range
        );
    }
}

#[test]
fn bounded_windows_bind_the_aligned_padded_window() {
    // A 6-byte allocation: the last byte is covered by the 16-byte padded range.
    let (r, res) = descriptor_range(0, 6, Window::Bounded { start: 5, len: 1 }, 16, MAX).unwrap();
    assert_eq!((r.offset, r.range, res), (0, 16, 0));
    // A window starting beyond the alignment block: the base moves up, the residual wraps.
    let (r, res) = descriptor_range(
        8,
        4096,
        Window::Bounded {
            start: 256,
            len: 16,
        },
        64,
        MAX,
    )
    .unwrap();
    assert_eq!(r.offset, 256);
    assert_eq!(r.range, 32, "bytes 264..280 round up to 288");
    assert_eq!(res, 8u32.wrapping_sub(256));
    // A negative start clamps to 0: the residual is the bind offset.
    let (r, res) = descriptor_range(
        100,
        4096,
        Window::Bounded {
            start: -200,
            len: 400,
        },
        64,
        MAX,
    )
    .unwrap();
    assert_eq!((r.offset, res), (0, 100));
    assert_eq!(r.range, 304);
    // Whole: from the aligned bind offset to the padded end.
    let (r, res) = descriptor_range(100, 1000, Window::Whole, 64, MAX).unwrap();
    assert_eq!((r.offset, r.range, res), (64, 1008 - 64, 36));
}

#[test]
fn windows_past_the_allocation_still_bind_sixteen_bytes() {
    // Bound past the end, or a window entirely outside: 16 bytes inside the allocation.
    for (b, len, w) in [
        (4096, 4096, Window::Whole),
        (1 << 40, 64, Window::Whole),
        (
            0,
            64,
            Window::Bounded {
                start: 1 << 20,
                len: 4,
            },
        ),
        (0, 64, Window::Bounded { start: -64, len: 8 }),
        (60, 64, Window::Bounded { start: 0, len: 4 }),
    ] {
        let (r, _) = descriptor_range(b, len, w, 16, MAX).unwrap();
        assert!(
            r.range >= 16 && r.offset + r.range <= padded(len),
            "{b} {w:?}"
        );
        check(b, len, w, 16);
    }
    assert_eq!(
        descriptor_range(0, 0, Window::Whole, 16, MAX),
        Err(RangeError::Empty)
    );
}

#[test]
fn ranges_hold_at_extremes_of_every_operand() {
    let offsets = [
        0,
        1,
        3,
        4,
        15,
        16,
        255,
        256,
        (1 << 32) - 4,
        (1 << 32) + 4,
        (1 << 40) + 12,
        MAX - 4095,
    ];
    let lengths = [1, 6, 10, 16, 17, 4096, 1 << 33, MAX - 64];
    let windows = [
        Window::Whole,
        Window::Bounded { start: 0, len: 1 },
        Window::Bounded { start: 5, len: 1 },
        Window::Bounded { start: -1, len: 2 },
        Window::Bounded {
            start: -(1 << 31),
            len: 1 << 32,
        },
        Window::Bounded {
            start: i64::MIN,
            len: MAX,
        },
        Window::Bounded {
            start: i64::MAX,
            len: MAX,
        },
        Window::Bounded {
            start: 1 << 32,
            len: 64,
        },
    ];
    for &b in &offsets {
        for &len in &lengths {
            for &w in &windows {
                for a in [1, 4, 16, 64, 256] {
                    check(b, len, w, a);
                }
            }
        }
    }
}

#[test]
fn ranges_above_the_descriptor_limit_are_refused() {
    let w = Window::Bounded {
        start: 0,
        len: 1 << 20,
    };
    assert_eq!(
        descriptor_range(0, 1 << 30, w, 16, 65536),
        Err(RangeError::TooLarge { needed: 1 << 20 })
    );
    assert!(descriptor_range(0, 1 << 30, w, 16, 1 << 20).is_ok());
    assert_eq!(
        descriptor_range(0, 1 << 30, Window::Whole, 16, u64::from(u32::MAX)),
        Ok((
            BufferRange {
                offset: 0,
                range: 1 << 30
            },
            0
        ))
    );
}

fn entry(host: u64, guest: u64, size: u64) -> RootEntry {
    RootEntry {
        host,
        guest,
        size,
        bind_offset: 0,
    }
}

#[test]
fn the_root_table_follows_d76_layout() {
    let statics = [(0, entry(0x10, 0x20, 0x30)), (3, entry(0x40, 0x50, 0x60))];
    let resident = [entry(0x100, 0x200, 0x300)];
    let mut e = statics[1].1;
    e.bind_offset = 8;
    let statics = [statics[0], (3, e)];
    let t = RootTable::build(&statics, 6, &resident);
    let word = |root: u64, field: u64| {
        let at = (root * ENTRY_BYTES + field) as usize;
        u64::from_le_bytes(t.bytes[at..at + 8].try_into().unwrap())
    };
    assert_eq!(t.bytes.len(), 7 * 32);
    assert!(t.bytes[..32].iter().all(|&b| b == 0), "entry 0 is null");
    assert_eq!(word(1, HOST), 0, "the consumer fills the sink address");
    assert_eq!(RootTable::SINK_HOST_OFFSET, 32);
    assert_eq!((word(1, GUEST), word(1, SIZE)), (6, 1), "R0 and N");
    assert_eq!(
        (word(2, HOST), word(2, GUEST), word(2, SIZE)),
        (0x10, 0x20, 0x30)
    );
    assert_eq!((word(5, HOST), word(5, BIND_OFFSET)), (0x40, 8));
    assert!(t.bytes[3 * 32..5 * 32].iter().all(|&b| b == 0), "unbound");
    assert_eq!(
        (word(6, HOST), word(6, GUEST), word(6, SIZE)),
        (0x100, 0x200, 0x300)
    );
}

/// A texel-buffer view spans the whole blocks of the bound length and its element count
/// (`⌊length / block size⌋ × texels per block`) is checked exactly against
/// `maxTexelBufferElements`; an unaligned offset or a view without a whole block is not
/// representable.
#[test]
fn texel_views_count_whole_blocks_against_the_element_limit() {
    use crate::bind::plan::PlannedTexel;
    use crate::bind::state::StageTable;
    use crate::pipeline::sym::{SlotClass, Stage};
    let slot = SlotRef {
        stage: Stage::Kernel,
        class: SlotClass::Texture,
        index: 3,
    };
    let block = vk::Format::R32G32B32_SFLOAT.block().expect("block");
    let template = ResourceTemplate {
        texel_buffers: vec![PlannedTexel {
            set: 0,
            binding: 5,
            slot,
            alignment: 16,
            max_elements: 100,
            block_size: u64::from(block.size),
            texels_per_block: u64::from(block.texels),
        }],
        ..ResourceTemplate::default()
    };
    let state = |offset: u64, length: u64| {
        let mut t = StageTable::default();
        t.textures.insert(
            3,
            BoundTexture::Buffer {
                pixel_format: crate::metal::PixelFormat::RGBA32Float,
                offset,
                length,
                bytes_per_row: length,
                resource_id: None,
                swizzle: crate::metal::TextureSwizzleChannels::DEFAULT,
            },
        );
        let mut s = BindState::default();
        s.stages.insert(Stage::Kernel, t);
        s
    };
    let range = |offset, length| texel_ranges(&template, &state(offset, length));
    // 12-byte texels: 100 whole texels fit, the ragged tail is not part of the view.
    assert_eq!(range(32, 1200).expect("at the limit")[0].range.range, 1200);
    assert_eq!(range(32, 1210).expect("ragged")[0].range.range, 1200);
    assert_eq!(range(0, 1211).expect("ragged")[0].range.range, 1200);
    assert_eq!(
        range(0, 1212),
        Err(BindError::TexelBufferTooLarge {
            slot,
            elements: 101,
            limit: 100
        })
    );
    let not_representable = Err(BindError::Unsupported(Unsupported::not_representable(
        Feature::TexelBuffer,
    )));
    assert_eq!(range(0, 11), not_representable, "no whole texel");
    assert_eq!(range(8, 120), not_representable, "unaligned offset");
    // A compressed block counts all of its texels.
    let bc1 = vk::Format::BC1_RGB_UNORM_BLOCK.block().expect("block");
    let mut compressed = template.clone();
    compressed.texel_buffers[0].block_size = u64::from(bc1.size);
    compressed.texel_buffers[0].texels_per_block = u64::from(bc1.texels);
    assert!(texel_ranges(&compressed, &state(0, 48)).is_ok());
    assert_eq!(
        texel_ranges(&compressed, &state(0, 56)),
        Err(BindError::TexelBufferTooLarge {
            slot,
            elements: 112,
            limit: 100
        })
    );
}

/// A push field's bytes are the bytes it was made of, however they were made: held in place up
/// to `FieldBytes::INLINE` bytes, past it on the heap, a collect that crosses the bound included.
#[test]
fn field_bytes_hold_what_they_were_made_of() {
    for len in [0, 1, 4, 31, 32, 33, 64, 200] {
        let bytes: Vec<u8> = (0..len).map(|i| (i * 7 + 3) as u8).collect();
        let of = FieldBytes::of(&bytes);
        let collected: FieldBytes = bytes.iter().copied().collect();
        for field in [&of, &collected] {
            assert_eq!(&field[..], &bytes[..], "{len} bytes");
            let inline = matches!(field, FieldBytes::Inline { .. });
            assert_eq!(inline, len <= FieldBytes::INLINE, "{len} bytes");
        }
    }
    let words = FieldBytes::words([1, 0x0403_0201, u32::MAX]);
    assert_eq!(&words[..], &[1, 0, 0, 0, 1, 2, 3, 4, 255, 255, 255, 255]);
}
