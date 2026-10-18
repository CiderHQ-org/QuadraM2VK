//! Blits into Apple's two-plane extended-range textures placed in a heap on the device the cfg
//! selects, under the validation layer, against Metal's bytes: every byte of the heap.
//!
//! Each case replays one of our probe `tph`'s: a placement heap of Metal's size, `Shared` or
//! `Private`, every byte of it a byte of the case's seed ([`random_bytes`]) written through a
//! buffer over the whole heap; two one-level 2D textures of `RGB10A8_2P_XR10` (or its sRGB twin)
//! placed where the probe placed them; one blit of the case's region from one into the other;
//! and the heap's bytes read back through the buffer. Metal's answer: the first 16 hex digits of
//! the SHA-256 of the heap's bytes after the blit, as the probe printed them (an Apple M4 under
//! macOS 27.0.1, each blit three times on the same bytes and every case twice, alike; and an Apple
//! M3 Max under macOS 26.7, the same digests). The copy's alpha bytes land past the destination's
//! alpha plane, on the heap's other bytes, and in `end255` past the heap's end, which Metal's blit
//! writes and the device's announces (`two-plane-texture-copy`). A differing heap's digest is
//! printed on one `XR-TWO-PLANE-HEAP-BLITS-DIFFERS` line.

use super::degrade_notices::{counts, on_device};
use super::extended_range_two_planes::{get, put};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    CommandBufferStatus, Degradation, HeapDescriptor, HeapType, Origin, ResourceOptions, Size,
    TextureDescriptor,
};
use sha2::{Digest, Sha256};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// One of the probe's cases.
struct Case {
    name: &'static str,
    private: bool,
    /// The source's and the destination's formats.
    formats: (PixelFormat, PixelFormat),
    /// The heap's bytes: Metal's size.
    heap: u64,
    /// The source's extent and its place in the heap.
    source: ([u64; 2], u64),
    /// The destination's extent and its place in the heap.
    destination: ([u64; 2], u64),
    /// The copy: its source origin, its destination origin and its size.
    copy: ([u64; 2], [u64; 2], [u64; 2]),
    seed: u64,
    /// Metal's digest of the heap's bytes after the blit.
    metal: &'static str,
}

const P2: PixelFormat = PixelFormat::RGB10A82PXR10;
const P2S: PixelFormat = PixelFormat::RGB10A82PXR10SRGB;

/// A case of one format whose source lies at 0 and whose destination at `at`.
const fn case(
    name: &'static str,
    private: bool,
    (heap, at): (u64, u64),
    (source, destination): ([u64; 2], [u64; 2]),
    copy: ([u64; 2], [u64; 2], [u64; 2]),
    seed: u64,
    metal: &'static str,
) -> Case {
    Case {
        name,
        private,
        formats: (P2, P2),
        heap,
        source: (source, 0),
        destination: (destination, at),
        copy,
        seed,
        metal,
    }
}

const A: [u64; 2] = [48, 20];
const B: [u64; 2] = [63, 65];
const C: [u64; 2] = [257, 9];
const D: [u64; 2] = [255, 64];
const E: [u64; 2] = [100, 30];
const SMALL: (u64, u64) = (1_277_952, 114_688);
const WHOLE_A: ([u64; 2], [u64; 2], [u64; 2]) = ([0, 0], [0, 0], A);
const WHOLE_B: ([u64; 2], [u64; 2], [u64; 2]) = ([0, 0], [0, 0], B);

/// The probe's cases whose answer Metal gives alike in every run: rows that do not overlap in
/// the destination with other bytes than each other's, widths of 4 or more.
const CASES: [Case; 25] = [
    case("a48", false, SMALL, (A, A), WHOLE_A, 11, "cf7a721ed69f675d"),
    case(
        "r48",
        false,
        SMALL,
        (A, A),
        ([3, 5], [11, 2], [17, 9]),
        12,
        "bc30954fd2097534",
    ),
    case(
        "w64",
        false,
        (1_310_720, 131_072),
        ([64, 64], [64, 64]),
        ([0, 0], [0, 0], [64, 64]),
        13,
        "cf53bd41e01d0583",
    ),
    case(
        "w63",
        false,
        (1_507_328, 229_376),
        (B, B),
        WHOLE_B,
        14,
        "979517cd394734e8",
    ),
    case(
        "w5",
        false,
        SMALL,
        ([5, 5], [5, 5]),
        ([0, 0], [0, 0], [5, 5]),
        15,
        "854674c5652d7840",
    ),
    case(
        "w9",
        false,
        SMALL,
        ([9, 7], [9, 7]),
        ([0, 0], [0, 0], [9, 7]),
        16,
        "ae4beef74adc1996",
    ),
    case(
        "g20",
        false,
        SMALL,
        ([20, 8], [33, 9]),
        ([0, 0], [2, 1], [20, 8]),
        17,
        "52c8b953a5dd52c4",
    ),
    case(
        "w257",
        false,
        SMALL,
        (C, C),
        ([0, 0], [0, 0], C),
        18,
        "3498238e64c35fdd",
    ),
    case(
        "w255",
        false,
        (2_031_616, 491_520),
        (D, D),
        ([0, 0], [0, 0], D),
        19,
        "bdc838283ed53d30",
    ),
    case(
        "row48",
        false,
        SMALL,
        (A, A),
        ([0, 3], [0, 7], [48, 1]),
        20,
        "f49d5cc7392c2921",
    ),
    case(
        "row63a",
        false,
        (1_507_328, 229_376),
        (B, B),
        ([0, 3], [0, 0], [63, 1]),
        21,
        "6bb95ed142a8bf06",
    ),
    case(
        "row63b",
        false,
        (1_507_328, 229_376),
        (B, B),
        ([0, 3], [0, 64], [63, 1]),
        22,
        "6b76f907a836cbea",
    ),
    case(
        "w4",
        false,
        SMALL,
        ([4, 6], [4, 6]),
        ([0, 0], [0, 0], [4, 6]),
        23,
        "9d40da5989e9b6ad",
    ),
    case(
        "w16",
        false,
        SMALL,
        ([16, 16], [16, 16]),
        ([0, 0], [0, 0], [16, 16]),
        25,
        "07ccd0b17977890f",
    ),
    case(
        "n100",
        false,
        (1_310_720, 131_072),
        (E, E),
        ([7, 2], [13, 5], [50, 20]),
        26,
        "dc81d76a2bbc3690",
    ),
    Case {
        formats: (P2, P2S),
        ..case(
            "d551",
            false,
            SMALL,
            (A, A),
            WHOLE_A,
            27,
            "7b637b1f83c242ff",
        )
    },
    Case {
        formats: (P2S, P2S),
        ..case(
            "s551",
            false,
            (1_507_328, 229_376),
            (B, B),
            WHOLE_B,
            28,
            "b252db87f05339de",
        )
    },
    case("pa48", true, SMALL, (A, A), WHOLE_A, 31, "0dabf59cbe10ecd5"),
    case(
        "pw63",
        true,
        (1_507_328, 229_376),
        (B, B),
        WHOLE_B,
        32,
        "338f41ba61912259",
    ),
    case(
        "pw257",
        true,
        SMALL,
        (C, C),
        ([0, 0], [0, 0], C),
        33,
        "0d02eb803c5db59f",
    ),
    case(
        "pn100",
        true,
        (1_310_720, 131_072),
        (E, E),
        ([7, 2], [13, 5], [50, 20]),
        35,
        "db6df4fe38d47c8f",
    ),
    case(
        "o257",
        false,
        SMALL,
        ([256, 9], C),
        ([0, 0], [1, 0], [256, 9]),
        84,
        "582e0cde0ebcb3c0",
    ),
    // The source past the destination, where the destination's alpha bytes land.
    Case {
        source: (A, 17_408),
        destination: (A, 0),
        ..case(
            "ovl",
            false,
            (1_179_648, 0),
            (A, A),
            WHOLE_A,
            88,
            "46d3dbcde4772f1f",
        )
    },
    // The destination at the end of a heap only as large as it needs: its alpha bytes pass the
    // heap's end.
    case(
        "end255",
        false,
        (573_440, 491_520),
        (D, D),
        ([0, 0], [0, 0], D),
        90,
        "b7fb61ea05982c35",
    ),
    case(
        "pend255",
        true,
        (573_440, 491_520),
        (D, D),
        ([0, 0], [0, 0], D),
        91,
        "64e1fbb9343ef64f",
    ),
];

/// The probe's bytes of `seed`: each the high bits of a 64-bit linear congruential generator.
fn random_bytes(count: u64, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u8
        })
        .collect()
}

/// Each case's heap holds Metal's bytes after the blit, and a copy whose alpha bytes pass the
/// heap's end is announced.
#[test]
fn blits_into_a_heap_s_two_plane_texture_write_metal_s_bytes() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let mut wrong = Vec::new();
        for case in &CASES {
            let options = if case.private {
                ResourceOptions::private()
            } else {
                ResourceOptions::shared()
            };
            let heap = device
                .new_heap(&HeapDescriptor::new(
                    HeapType::Placement,
                    case.heap,
                    options,
                ))
                .unwrap();
            assert_eq!(heap.size(), case.heap, "{}: Metal's heap size", case.name);
            let texture = |format, [w, h]: [u64; 2], at| {
                let mut descriptor = TextureDescriptor::texture_2d(format, w, h, false);
                descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                descriptor.resource_options = options;
                heap.new_texture_at_offset(&descriptor, at)
                    .unwrap_or_else(|e| panic!("{}: {e}", case.name))
            };
            let source = texture(case.formats.0, case.source.0, case.source.1);
            let destination = texture(case.formats.1, case.destination.0, case.destination.1);
            let bytes = heap.new_buffer_at_offset(case.heap, options, 0).unwrap();
            put(device, &queue, &bytes, &random_bytes(case.heap, case.seed));
            let (from, to, [w, h]) = case.copy;
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_texture_to_texture(
                &source,
                0,
                0,
                Origin::new(from[0], from[1], 0),
                Size::new(w, h, 1),
                &destination,
                0,
                0,
                Origin::new(to[0], to[1], 0),
            )
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(
                command_buffer.status(),
                CommandBufferStatus::Completed,
                "{}",
                case.name
            );
            let got = get(device, &queue, &bytes, case.heap);
            let digest: String = Sha256::digest(&got)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            if digest[..16] != *case.metal {
                println!(
                    "XR-TWO-PLANE-HEAP-BLITS-DIFFERS {} {} (Metal's {})",
                    case.name,
                    &digest[..16],
                    case.metal
                );
                wrong.push(case.name);
            }
        }
        assert!(wrong.is_empty(), "heaps unlike Metal's: {wrong:?}");
        let notices = counts(&messages.take_notices());
        assert!(
            notices.contains_key(&Degradation::TwoPlaneTextureCopy),
            "the copies past the heap's end: {notices:?}"
        );
    });
}
