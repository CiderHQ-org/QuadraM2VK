//! Copies of Apple's two-plane extended-range textures on the device the cfg selects, under the
//! validation layer, against Metal's bytes.
//!
//! Each case replays one of a Metal probe `tx1`'s, on a 48 x 20 texture of `RGB10A8_2P_XR10` (`p2`)
//! or its sRGB twin (`p2s`) holding the probe's random texels: seed 1 of its xorshift generator
//! ([`Rng`]) gives each texel's plane-0 word (bits 30 and 31 random too) and alpha byte, written as
//! the probe writes them, the alpha bytes by a kernel's `float` write of `b / 255` (the owned
//! fixture `xr_two_planes`), then the words by `replace_region`. A texture's planes read back as
//! the probe reads them: plane 0 by `get_bytes`, 4 bytes a texel, and plane 1 as `round(255 a)` of
//! a kernel's `float` read.
//!
//! - `get`: `get_bytes` into 16,384 bytes of `0xab`, and `t2b`: a blit into a buffer of them, in
//!   each variant: rows of 4, 5 or 8 bytes a texel (`r4`, `r5`, `r8`), `r4` with images twice the
//!   plane apart (`r4i`), and a 17 x 9 region at (5, 3) into rows of 80 bytes from offset 256
//!   (`sub`);
//! - `rep`: `replace_region` of seed 2's 16,384 random bytes, and `b2t`: a blit of them from a
//!   buffer, in each variant (`sub`: a 13 x 11 region at (7, 2)), both planes read back;
//! - `t2t`: a blit into a texture of seed 3's texels, whole (`copy_texture`), of a 17 x 9 region at
//!   (3, 5) to (11, 2), and whole into a texture of the other format (`twin`), both planes read
//!   back.
//!
//! Metal's answer: the SHA-256 of each output ([`METAL`]; an Apple M4 under macOS
//! 27.0.1, plain and under Metal's validation layer alike, which stops only `twin`), which the
//! probe keeps. A texture-to-texture copy's planes are also checked against Metal's byte model
//! (`bind::plane_copy`, which Metal's measured copies pin), at 48 x 20 and at widths that are not a
//! multiple of 4; a copy whose rows overlap in the destination is announced once
//! (`two-plane-texture-copy`), and so is a buffer blit at a width off 4 (`two-plane-buffer-blit`).
//! A differing output is printed whole, on one `XR-TWO-PLANE-COPIES-DIFFERS` line.

use super::degrade_notices::{counts, on_device};
use super::extended_range_two_planes::{digest, reads, texture, write, Dispatch, Kernels};
use super::{buffer_with, read_buffer};
use metal2vulkan::bind::PlaneRows;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandBufferStatus, CommandQueue, Degradation, Device, Origin, Region,
    Size, Texture,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The texture's extent.
const W: u64 = 48;
const H: u64 = 20;
/// The bytes of each buffer and host copy.
const BIG: usize = 16 * (W * H) as usize + 1024;

/// The kinds a blit between two such textures, and one between a buffer and such a texture at
/// a width that is not a multiple of 4, announce.
pub(super) const TWO_PLANE_COPIES: [Degradation; 2] = [
    Degradation::TwoPlaneTextureCopy,
    Degradation::TwoPlaneBufferBlit,
];

/// Metal's outputs: the probe's case, the output, and the SHA-256 of its bytes.
const METAL: [(&str, &str, &str); 66] = [
    (
        "get.p2.r4",
        "out",
        "214d8cf2b34e79bee5712a5be1ebf67682b737eee5032760e2410e5b6c580e8c",
    ),
    (
        "get.p2.r5",
        "out",
        "b6139db33b5a7d978edaaefdafb7701ffdda5da49c9b8e991ccd1d7f81302fe6",
    ),
    (
        "get.p2.r8",
        "out",
        "5a074aa7d8aab53a5fffbdece7dd573c28ab3014d11016efcda39d3e7ed75726",
    ),
    (
        "get.p2.r4i",
        "out",
        "214d8cf2b34e79bee5712a5be1ebf67682b737eee5032760e2410e5b6c580e8c",
    ),
    (
        "get.p2.sub",
        "out",
        "30bc6e0ffed19dbc6b6713e8c839924ba33c58f43e21a2c5c1f98403759f5d6e",
    ),
    (
        "t2b.p2.r4",
        "out",
        "214d8cf2b34e79bee5712a5be1ebf67682b737eee5032760e2410e5b6c580e8c",
    ),
    (
        "t2b.p2.r5",
        "out",
        "b6139db33b5a7d978edaaefdafb7701ffdda5da49c9b8e991ccd1d7f81302fe6",
    ),
    (
        "t2b.p2.r8",
        "out",
        "5a074aa7d8aab53a5fffbdece7dd573c28ab3014d11016efcda39d3e7ed75726",
    ),
    (
        "t2b.p2.r4i",
        "out",
        "214d8cf2b34e79bee5712a5be1ebf67682b737eee5032760e2410e5b6c580e8c",
    ),
    (
        "t2b.p2.sub",
        "out",
        "30bc6e0ffed19dbc6b6713e8c839924ba33c58f43e21a2c5c1f98403759f5d6e",
    ),
    (
        "rep.p2.r4",
        "out.p0",
        "9695b6fdf1c2e40ab8125aced617a0565ea07ccbf8535a54eb19cfc5c569238b",
    ),
    (
        "rep.p2.r4",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "rep.p2.r5",
        "out.p0",
        "6148d632f2fd771873f460fb73f196c8f9887ded5623cfc4f2b719c2cbe2ec73",
    ),
    (
        "rep.p2.r5",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "rep.p2.r8",
        "out.p0",
        "bc41bd6aed35ff11c21b475829e83eb36be1a781dc623fab587b5607a44d9125",
    ),
    (
        "rep.p2.r8",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "rep.p2.r4i",
        "out.p0",
        "9695b6fdf1c2e40ab8125aced617a0565ea07ccbf8535a54eb19cfc5c569238b",
    ),
    (
        "rep.p2.r4i",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "rep.p2.sub",
        "out.p0",
        "9b696b573816b7cde2182005a0ef1d4b57a79c766194ced7e2efdc920b011e34",
    ),
    (
        "rep.p2.sub",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2.r4",
        "out.p0",
        "9695b6fdf1c2e40ab8125aced617a0565ea07ccbf8535a54eb19cfc5c569238b",
    ),
    (
        "b2t.p2.r4",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2.r5",
        "out.p0",
        "6148d632f2fd771873f460fb73f196c8f9887ded5623cfc4f2b719c2cbe2ec73",
    ),
    (
        "b2t.p2.r5",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2.r8",
        "out.p0",
        "bc41bd6aed35ff11c21b475829e83eb36be1a781dc623fab587b5607a44d9125",
    ),
    (
        "b2t.p2.r8",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2.r4i",
        "out.p0",
        "9695b6fdf1c2e40ab8125aced617a0565ea07ccbf8535a54eb19cfc5c569238b",
    ),
    (
        "b2t.p2.r4i",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2.sub",
        "out.p0",
        "9b696b573816b7cde2182005a0ef1d4b57a79c766194ced7e2efdc920b011e34",
    ),
    (
        "b2t.p2.sub",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "t2t.p2.whole",
        "out.p0",
        "bc4690d9570ceddf89ff6728f4e8c0288ae8a6141f24eac93edca8b0cf15c1fa",
    ),
    (
        "t2t.p2.sub",
        "out.p0",
        "f3436a44523a2b8cd955d14440cb4f02227cb2be5a97559e8bc52bafc3a4d4d2",
    ),
    (
        "t2t.p2.twin",
        "out.p0",
        "bc4690d9570ceddf89ff6728f4e8c0288ae8a6141f24eac93edca8b0cf15c1fa",
    ),
    (
        "get.p2s.r4",
        "out",
        "214d8cf2b34e79bee5712a5be1ebf67682b737eee5032760e2410e5b6c580e8c",
    ),
    (
        "get.p2s.r5",
        "out",
        "b6139db33b5a7d978edaaefdafb7701ffdda5da49c9b8e991ccd1d7f81302fe6",
    ),
    (
        "get.p2s.r8",
        "out",
        "5a074aa7d8aab53a5fffbdece7dd573c28ab3014d11016efcda39d3e7ed75726",
    ),
    (
        "get.p2s.r4i",
        "out",
        "214d8cf2b34e79bee5712a5be1ebf67682b737eee5032760e2410e5b6c580e8c",
    ),
    (
        "get.p2s.sub",
        "out",
        "30bc6e0ffed19dbc6b6713e8c839924ba33c58f43e21a2c5c1f98403759f5d6e",
    ),
    (
        "t2b.p2s.r4",
        "out",
        "214d8cf2b34e79bee5712a5be1ebf67682b737eee5032760e2410e5b6c580e8c",
    ),
    (
        "t2b.p2s.r5",
        "out",
        "b6139db33b5a7d978edaaefdafb7701ffdda5da49c9b8e991ccd1d7f81302fe6",
    ),
    (
        "t2b.p2s.r8",
        "out",
        "5a074aa7d8aab53a5fffbdece7dd573c28ab3014d11016efcda39d3e7ed75726",
    ),
    (
        "t2b.p2s.r4i",
        "out",
        "214d8cf2b34e79bee5712a5be1ebf67682b737eee5032760e2410e5b6c580e8c",
    ),
    (
        "t2b.p2s.sub",
        "out",
        "30bc6e0ffed19dbc6b6713e8c839924ba33c58f43e21a2c5c1f98403759f5d6e",
    ),
    (
        "rep.p2s.r4",
        "out.p0",
        "9695b6fdf1c2e40ab8125aced617a0565ea07ccbf8535a54eb19cfc5c569238b",
    ),
    (
        "rep.p2s.r4",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "rep.p2s.r5",
        "out.p0",
        "6148d632f2fd771873f460fb73f196c8f9887ded5623cfc4f2b719c2cbe2ec73",
    ),
    (
        "rep.p2s.r5",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "rep.p2s.r8",
        "out.p0",
        "bc41bd6aed35ff11c21b475829e83eb36be1a781dc623fab587b5607a44d9125",
    ),
    (
        "rep.p2s.r8",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "rep.p2s.r4i",
        "out.p0",
        "9695b6fdf1c2e40ab8125aced617a0565ea07ccbf8535a54eb19cfc5c569238b",
    ),
    (
        "rep.p2s.r4i",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "rep.p2s.sub",
        "out.p0",
        "9b696b573816b7cde2182005a0ef1d4b57a79c766194ced7e2efdc920b011e34",
    ),
    (
        "rep.p2s.sub",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2s.r4",
        "out.p0",
        "9695b6fdf1c2e40ab8125aced617a0565ea07ccbf8535a54eb19cfc5c569238b",
    ),
    (
        "b2t.p2s.r4",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2s.r5",
        "out.p0",
        "6148d632f2fd771873f460fb73f196c8f9887ded5623cfc4f2b719c2cbe2ec73",
    ),
    (
        "b2t.p2s.r5",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2s.r8",
        "out.p0",
        "bc41bd6aed35ff11c21b475829e83eb36be1a781dc623fab587b5607a44d9125",
    ),
    (
        "b2t.p2s.r8",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2s.r4i",
        "out.p0",
        "9695b6fdf1c2e40ab8125aced617a0565ea07ccbf8535a54eb19cfc5c569238b",
    ),
    (
        "b2t.p2s.r4i",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "b2t.p2s.sub",
        "out.p0",
        "9b696b573816b7cde2182005a0ef1d4b57a79c766194ced7e2efdc920b011e34",
    ),
    (
        "b2t.p2s.sub",
        "out.pa",
        "120c1f4739647fc263a294ff9132f8546e4faddd27fa3cb1a517213734d7cf16",
    ),
    (
        "t2t.p2s.whole",
        "out.p0",
        "bc4690d9570ceddf89ff6728f4e8c0288ae8a6141f24eac93edca8b0cf15c1fa",
    ),
    (
        "t2t.p2s.sub",
        "out.p0",
        "f3436a44523a2b8cd955d14440cb4f02227cb2be5a97559e8bc52bafc3a4d4d2",
    ),
    (
        "t2t.p2s.twin",
        "out.p0",
        "bc4690d9570ceddf89ff6728f4e8c0288ae8a6141f24eac93edca8b0cf15c1fa",
    ),
];

/// The probe's xorshift generator.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Rng {
        let mut state = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x632B_E59B_D9B4_E019);
        if state == 0 {
            state = 1;
        }
        let mut rng = Rng(state);
        for _ in 0..8 {
            rng.next();
        }
        rng
    }

    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(2_685_821_657_736_338_717)
    }

    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Seed `seed`'s `n` texels: plane 0's words (4 bytes each) and the alpha bytes.
pub(super) fn planes(seed: u64, n: u64) -> (Vec<u8>, Vec<u8>) {
    let mut rng = Rng::new(seed);
    let (mut words, mut alpha) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let high = rng.below(4);
        let red = rng.below(1024);
        let green = rng.below(1024);
        let blue = rng.below(1024);
        let word = (high << 30 | red << 20 | green << 10 | blue) as u32;
        words.extend(word.to_le_bytes());
        alpha.push(rng.below(256) as u8);
    }
    (words, alpha)
}

/// Seed `seed`'s `n` random bytes.
fn random_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.below(256) as u8).collect()
}

/// The device, its queue and the fixture's kernels.
pub(super) struct Lane<'a> {
    pub(super) device: &'a Device,
    pub(super) queue: CommandQueue,
    kernels: Kernels,
}

impl Lane<'_> {
    /// A `w` x `h` texture of `format` holding seed `seed`'s texels, written as the probe writes
    /// them, and the texels.
    pub(super) fn filled(
        &self,
        format: PixelFormat,
        seed: u64,
        (w, h): (u64, u64),
    ) -> (Texture, Vec<u8>, Vec<u8>) {
        let (words, alpha) = planes(seed, w * h);
        let t = self.holding(format, (&words, &alpha), (w, h));
        (t, words, alpha)
    }

    /// A `w` x `h` texture of `format` holding the color words `words` (4 bytes each) and the
    /// alpha bytes `alpha`, written as the probe writes them: the alpha bytes by a kernel's
    /// `float` write of `b / 255`, then the words by `replace_region`.
    pub(super) fn holding(
        &self,
        format: PixelFormat,
        (words, alpha): (&[u8], &[u8]),
        (w, h): (u64, u64),
    ) -> Texture {
        let t = texture(self.device, format, w, h);
        let values: Vec<u8> = alpha
            .iter()
            .flat_map(|&b| [0.0, 0.0, 0.0, f32::from(b) / 255.0])
            .flat_map(f32::to_le_bytes)
            .collect();
        let dispatch = Dispatch::Texels(w, h);
        write(
            self.device,
            &self.queue,
            &self.kernels,
            &t,
            (&values, false),
            dispatch,
        );
        t.replace_region(Region::new_2d(0, 0, w, h), 0, 0, words, 4 * w, 0)
            .unwrap();
        t
    }

    /// The planes of `t`, 48 x 20, read back as the probe reads them.
    fn planes_of(&self, t: &Texture) -> (Vec<u8>, Vec<u8>) {
        self.planes_sized(t, (W, H))
    }

    /// The planes of `t`, `w` x `h`, read back as the probe reads them.
    fn planes_sized(&self, t: &Texture, (w, h): (u64, u64)) -> (Vec<u8>, Vec<u8>) {
        let mut words = vec![0; (4 * w * h) as usize];
        t.get_bytes(&mut words, 4 * w, 0, Region::new_2d(0, 0, w, h), 0, 0)
            .unwrap();
        let [_, floats] = reads(
            self.device,
            &self.queue,
            &self.kernels,
            t,
            Dispatch::Texels(w, h),
        );
        let alpha = floats
            .as_chunks::<16>()
            .0
            .iter()
            .map(|c| {
                let a = f32::from_le_bytes(c[12..16].try_into().unwrap());
                (a * 255.0).round_ties_even() as u8
            })
            .collect();
        (words, alpha)
    }

    /// Runs `body` in a blit encoder, commits and waits.
    fn blit(&self, body: impl FnOnce(&mut metal2vulkan_runtime::BlitCommandEncoder)) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        body(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
    }
}

/// A variant's region and layout: its origin, size, rows, images and buffer offset; `sub` the
/// size and origin of its region.
fn variant(name: &str, sub: [u64; 4]) -> (Origin, Size, u64, u64, u64) {
    let whole = |bpr: u64, bpi: u64| (Origin::new(0, 0, 0), Size::new(W, H, 1), bpr, bpi, 0);
    match name {
        "r4" => whole(4 * W, 4 * W * H),
        "r5" => whole(5 * W, 5 * W * H),
        "r8" => whole(8 * W, 8 * W * H),
        "r4i" => whole(4 * W, 8 * W * H),
        _ => {
            let [w, h, x, y] = sub;
            (Origin::new(x, y, 0), Size::new(w, h, 1), 80, 80 * h, 256)
        }
    }
}

const VARIANTS: [&str; 5] = ["r4", "r5", "r8", "r4i", "sub"];

/// The formats and the probe's names for them.
pub(super) const FORMATS: [(PixelFormat, &str); 2] = [
    (PixelFormat::RGB10A82PXR10, "p2"),
    (PixelFormat::RGB10A82PXR10SRGB, "p2s"),
];

/// Checks output `what` of `case` against Metal's digest, keeping a difference in `wrong`.
fn check(wrong: &mut Vec<String>, case: &str, what: &str, bytes: &[u8]) {
    let want = METAL
        .iter()
        .find(|m| m.0 == case && m.1 == what)
        .map(|m| m.2)
        .unwrap_or_else(|| panic!("no Metal digest of {case} {what}"));
    let got = digest(bytes);
    if got != want {
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        wrong.push(format!("{case} {what}: {got}, Metal {want}; bytes {hex}"));
    }
}

/// Prints `wrong`'s lines and fails when there is one.
fn report(device: &Device, wrong: &[String]) {
    for line in wrong {
        println!("XR-TWO-PLANE-COPIES-DIFFERS {}: {line}", device.name());
    }
    assert!(
        wrong.is_empty(),
        "{} outputs differ from Metal",
        wrong.len()
    );
}

/// The lane's kernels on `device`, and a queue.
pub(super) fn lane(device: &Device) -> Lane<'_> {
    Lane {
        device,
        queue: device.new_command_queue().unwrap(),
        kernels: Kernels::new(device),
    }
}

/// `get_bytes` and `replace_region` move each texel's color word at the caller's rows and leave
/// the alpha plane and every byte between the rows as they are, as Metal's do.
#[test]
fn cpu_copies_move_the_color_words_as_metal_does() {
    on_device(|device, _| {
        let lane = lane(device);
        let mut wrong = Vec::new();
        for (format, name) in FORMATS {
            for v in VARIANTS {
                let (t, _, _) = lane.filled(format, 1, (W, H));
                let (origin, size, bpr, bpi, offset) = variant(v, [17, 9, 5, 3]);
                let mut out = vec![0xab; BIG];
                t.get_bytes(
                    &mut out[offset as usize..],
                    bpr,
                    bpi,
                    Region { origin, size },
                    0,
                    0,
                )
                .unwrap();
                check(&mut wrong, &format!("get.{name}.{v}"), "out", &out);

                let (t, _, _) = lane.filled(format, 1, (W, H));
                let (origin, size, bpr, bpi, offset) = variant(v, [13, 11, 7, 2]);
                let input = random_bytes(2, BIG);
                t.replace_region(
                    Region { origin, size },
                    0,
                    0,
                    &input[offset as usize..],
                    bpr,
                    bpi,
                )
                .unwrap();
                let (words, alpha) = lane.planes_of(&t);
                check(&mut wrong, &format!("rep.{name}.{v}"), "out.p0", &words);
                check(&mut wrong, &format!("rep.{name}.{v}"), "out.pa", &alpha);
            }
        }
        report(device, &wrong);
    });
}

/// Blits between a buffer and the texture move each texel's color word at the buffer's rows and
/// leave the alpha plane and every byte between the rows as they are, as Metal's do.
#[test]
fn buffer_blits_move_the_color_words_as_metal_does() {
    on_device(|device, _| {
        let lane = lane(device);
        let mut wrong = Vec::new();
        for (format, name) in FORMATS {
            for v in VARIANTS {
                let (t, _, _) = lane.filled(format, 1, (W, H));
                let (origin, size, bpr, bpi, offset) = variant(v, [17, 9, 5, 3]);
                let out = buffer_with(device, &[0xab; BIG]).unwrap();
                lane.blit(|e| {
                    e.copy_from_texture_to_buffer(
                        &t,
                        0,
                        0,
                        origin,
                        size,
                        &out,
                        offset,
                        bpr,
                        bpi,
                        BlitOption::None,
                    )
                    .unwrap()
                });
                let bytes = read_buffer(&out, 0, BIG as u64).unwrap();
                check(&mut wrong, &format!("t2b.{name}.{v}"), "out", &bytes);

                let (t, _, _) = lane.filled(format, 1, (W, H));
                let (origin, size, bpr, bpi, offset) = variant(v, [13, 11, 7, 2]);
                let input: Buffer = buffer_with(device, &random_bytes(2, BIG)).unwrap();
                lane.blit(|e| {
                    e.copy_from_buffer_to_texture(
                        &input,
                        offset,
                        bpr,
                        bpi,
                        size,
                        &t,
                        0,
                        0,
                        origin,
                        BlitOption::None,
                    )
                    .unwrap()
                });
                let (words, alpha) = lane.planes_of(&t);
                check(&mut wrong, &format!("b2t.{name}.{v}"), "out.p0", &words);
                check(&mut wrong, &format!("b2t.{name}.{v}"), "out.pa", &alpha);
            }
        }
        report(device, &wrong);
    });
}

/// A texture's planes: its color words, then its alpha bytes.
type Planes = (Vec<u8>, Vec<u8>);

/// A blit of `size` texels from `from` of a texture of `source` texels (width, height) whose
/// planes are `planes.0` (color words, then alpha bytes) to `to` of one of `destination` texels
/// holding `planes.1`, as Metal's blit moves the bytes (`bind::plane_copy` states the model, which
/// Metal's measured copies pin): each row of the copy in turn, the later row's bytes kept where two
/// land on one, a byte in a row's padding or past a plane landing on no texel and read as 0. The
/// destination's planes after it.
fn modelled(
    (source, destination): ((u64, u64), (u64, u64)),
    (from, size, to): ([u64; 2], [u64; 2], [u64; 2]),
    planes: (&Planes, &Planes),
) -> Planes {
    let rows = |w: u64| PlaneRows::of(w as u32);
    let (s, d) = (rows(source.0), rows(destination.0));
    let mut out = planes.1.clone();
    for (unit, s_rows, d_rows) in [(4, s.color, d.color), (1, s.alpha, d.alpha)] {
        let (s_rows, d_rows) = (u64::from(s_rows), u64::from(d_rows));
        let mut plane = std::collections::BTreeMap::new();
        for r in 0..size[1] {
            for c in 0..size[0] {
                let read = (from[1] + r) * u64::from(s.blit) + unit * (from[0] + c);
                let write = (to[1] + r) * u64::from(d.blit) + unit * (to[0] + c);
                let (y, x) = (read / s_rows, read % s_rows / unit);
                let texel = (x < source.0 && y < source.1).then_some(y * source.0 + x);
                plane.insert(write, texel);
            }
        }
        for (at, texel) in plane {
            let (y, x) = (at / d_rows, at % d_rows / unit);
            if x >= destination.0 || y >= destination.1 {
                continue;
            }
            let i = (y * destination.0 + x) as usize;
            if unit == 4 {
                let word = texel.map_or([0; 4], |t| {
                    let t = 4 * t as usize;
                    planes.0 .0[t..t + 4].try_into().unwrap()
                });
                out.0[4 * i..4 * i + 4].copy_from_slice(&word);
            } else {
                out.1[i] = texel.map_or(0, |t| planes.0 .1[t as usize]);
            }
        }
    }
    out
}

/// Blits between two such textures move both planes' bytes as Metal's blit does: at 48 x 20 plane 0
/// against Metal's digests and both planes against the model, whole, a region and
/// into the other format; at widths that are not a multiple of 4 (plane 0's texels lost or moved
/// too) whole copies and regions whose rows do not overlap against the model; none of them
/// announced. A copy whose rows overlap in the destination with other bytes, where Metal's blit
/// writes them in an order that varies, is announced once (`two-plane-texture-copy`).
#[test]
fn texture_blits_move_both_planes_as_metal_does() {
    on_device(|device, messages| {
        let lane = lane(device);
        let mut wrong = Vec::new();
        let mut model_wrong = Vec::new();
        let blit = |format, into, (w, h): (u64, u64), copy: ([u64; 2], [u64; 2], [u64; 2])| {
            let (s, source_words, source_alpha) = lane.filled(format, 1, (w, h));
            let (d, words, alpha) = lane.filled(into, 3, (w, h));
            let (from, size, to) = copy;
            lane.blit(|e| {
                if copy == ([0, 0], [w, h], [0, 0]) {
                    e.copy_texture(&s, &d).unwrap()
                } else {
                    e.copy_from_texture_to_texture(
                        &s,
                        0,
                        0,
                        Origin::new(from[0], from[1], 0),
                        Size::new(size[0], size[1], 1),
                        &d,
                        0,
                        0,
                        Origin::new(to[0], to[1], 0),
                    )
                    .unwrap()
                }
            });
            let got = lane.planes_sized(&d, (w, h));
            let want = modelled(
                ((w, h), (w, h)),
                copy,
                (&(source_words, source_alpha), &(words, alpha)),
            );
            (got, want)
        };
        for (format, name) in FORMATS {
            let other = FORMATS.iter().find(|f| f.0 != format).unwrap().0;
            for v in ["whole", "sub", "twin"] {
                let into = if v == "twin" { other } else { format };
                let copy = match v {
                    "sub" => ([3, 5], [17, 9], [11, 2]),
                    _ => ([0, 0], [W, H], [0, 0]),
                };
                let (got, want) = blit(format, into, (W, H), copy);
                check(&mut wrong, &format!("t2t.{name}.{v}"), "out.p0", &got.0);
                if got != want {
                    model_wrong.push(format!("t2t.{name}.{v}"));
                }
            }
        }
        report(device, &wrong);
        let format = FORMATS[0].0;
        for (size, copy) in [
            ((5, 5), ([0, 0], [5, 5], [0, 0])),
            ((9, 7), ([0, 0], [9, 7], [0, 0])),
            ((257, 3), ([0, 0], [257, 3], [0, 0])),
            ((13, 6), ([1, 2], [6, 3], [5, 1])),
            ((7, 9), ([1, 0], [3, 8], [4, 1])),
        ] {
            let (got, want) = blit(format, format, size, copy);
            if got != want {
                model_wrong.push(format!("{size:?} {copy:?}"));
            }
        }
        assert!(
            model_wrong.is_empty(),
            "not the model's bytes: {model_wrong:?}"
        );
        let notices = counts(&messages.take_notices());
        assert_eq!(notices.get(&TWO_PLANE_COPIES[0]), None, "{notices:?}");
        // Rows overlapping in the destination (9 wide: rows 36 bytes, blit rows 32) from other
        // rows of the source: the model's bytes, announced.
        let copy = ([0, 1], [9, 3], [0, 0]);
        let (got, want) = blit(format, format, (9, 7), copy);
        assert_eq!(got, want, "an overlapping copy keeps the later row's bytes");
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&TWO_PLANE_COPIES[0]).copied(),
            Some(1),
            "{notices:?}"
        );
    });
}

/// Metal's blit keeps the bytes it writes outside a level's texels (a row's padding, past a
/// plane) and reads them back in a later blit; the kernel holds the texels
/// alone, so a copy reading such bytes of a texture an earlier blit wrote them into is announced
/// once (`two-plane-texture-copy`), and the earlier copy, from a fresh texture, is not.
#[test]
fn chained_blits_reading_kept_bytes_are_announced() {
    on_device(|device, messages| {
        let lane = lane(device);
        let format = FORMATS[0].0;
        let (w, h) = (9u64, 7u64);
        let (a, _, _) = lane.filled(format, 1, (w, h));
        let (b, _, _) = lane.filled(format, 3, (w, h));
        let (c, _, _) = lane.filled(format, 5, (w, h));
        lane.blit(|e| e.copy_texture(&a, &b).unwrap());
        let notices = counts(&messages.take_notices());
        assert_eq!(notices.get(&TWO_PLANE_COPIES[0]), None, "{notices:?}");
        lane.blit(|e| e.copy_texture(&b, &c).unwrap());
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&TWO_PLANE_COPIES[0]).copied(),
            Some(1),
            "{notices:?}"
        );
    });
}

/// A blit between a buffer and a level whose width is not a multiple of 4 moves the texels as
/// `get_bytes` and `replace_region` do, announced once (`two-plane-buffer-blit`): Metal's blit
/// addresses the level's rows short of where they lie.
#[test]
fn buffer_blits_at_widths_off_4_are_announced() {
    on_device(|device, messages| {
        let lane = lane(device);
        let (w, h) = (5u64, 4u64);
        let (t, _, alpha) = lane.filled(FORMATS[0].0, 1, (w, h));
        let input = random_bytes(2, (4 * w * h) as usize);
        let buffer = buffer_with(device, &input).unwrap();
        lane.blit(|e| {
            e.copy_from_buffer_to_texture(
                &buffer,
                0,
                4 * w,
                4 * w * h,
                Size::new(w, h, 1),
                &t,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap()
        });
        assert_eq!(lane.planes_sized(&t, (w, h)), (input, alpha));
        let notices = counts(&messages.take_notices());
        assert_eq!(
            notices.get(&TWO_PLANE_COPIES[1]).copied(),
            Some(1),
            "{notices:?}"
        );
    });
}

/// The generator gives the probe's texels: the FNV-1a digests the probe printed of seed 1's
/// planes and of seed 2's bytes (the probe's results).
#[test]
fn the_generator_is_the_probes() {
    let fnv = |bytes: &[u8]| {
        bytes.iter().fold(2_166_136_261u32, |h, &b| {
            (h ^ u32::from(b)).wrapping_mul(16_777_619)
        })
    };
    let (words, alpha) = planes(1, W * H);
    assert_eq!(fnv(&words), 0xa7c5_fb98);
    assert_eq!(fnv(&alpha), 0x5cc7_aa14);
    assert_eq!(fnv(&random_bytes(2, BIG)), 0x778b_70a0);
}
