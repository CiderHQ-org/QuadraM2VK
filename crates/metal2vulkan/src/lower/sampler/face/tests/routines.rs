//! The sampling routines are routines. For a profile whose driver converts every function to one
//! inlined at each call (`DriverWorkaround:: FunctionsInlinedAtEveryCall`, MoltenVK 1.4.1) each
//! states `DontInline`; for any other profile none states a control, and the entry states none
//! anywhere.

use super::super::define_fold;
use super::kernel_on;
use crate::device::profile::DriverWorkaround;
use crate::device::DevicePolicy;
use crate::lower::sampler::bicubic::routines;
use crate::lower::texture::float_texel;
use crate::lower::texture::half::{self, HalfFamilies};
use crate::pipeline::testing::profile;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::serialize::serialize_for;

/// Each `OpFunction`'s function control in `words`, in module order.
fn controls(words: &[u32]) -> Vec<u32> {
    let mut i = 5;
    let mut out = Vec::new();
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Function as u32 {
            out.push(words[i + 3]);
        }
        i += len;
    }
    out
}

/// The texel conversions, the emulated linear filters' and the bicubic filter's sums, and the
/// cube's face fold, beside a kernel entry: on moltenvk-m3 each routine states `DontInline`
/// (0x2) and the entry `None`; on amd-rdna3-radv the six texel routines state `DontInline` (the
/// linear span's, quad's and block's two sums each) and every other function `None`.
#[test]
fn the_sampling_routines_are_declared_dont_inline_where_every_call_is_inlined() {
    let mut b = ModuleBuilder::new();
    half::define(&mut b, HalfFamilies::Every).expect("half texel");
    float_texel::define(&mut b).expect("float texel");
    define_fold(&mut b).expect("fold");
    routines::define_plane(&mut b).expect("plane");
    routines::define_line(&mut b).expect("line");
    routines::define_span(&mut b).expect("span");
    routines::define_quad(&mut b).expect("quad");
    routines::define_block(&mut b).expect("block");
    routines::define_cube(&mut b).expect("cube");
    let m = kernel_on(b, 3, &|_, p, _| Ok(p)).expect("kernel");
    for (stem, inlines_every_call) in [("moltenvk-m3", true), ("amd-rdna3-radv", false)] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        assert_eq!(
            p.fields()
                .workarounds
                .contains(&DriverWorkaround::FunctionsInlinedAtEveryCall),
            inlines_every_call,
            "{stem}"
        );
        let words = serialize_for(m.clone(), &p).expect("serialized").words;
        let mut got = controls(&words);
        got.sort_unstable();
        // 19 routines (the plane's six, two each for the line, span, quad, block and cube, the
        // fold, and the two texel conversions; the cube's fold is a second fold) and the entry.
        assert!(got.len() >= 20, "{stem}: {} functions", got.len());
        // The texel routines state it on every profile.
        let kept = if inlines_every_call { got.len() - 1 } else { 6 };
        let mut want = vec![0x2; kept];
        want.splice(0..0, vec![0; got.len() - kept]);
        assert_eq!(
            got, want,
            "{stem}: each routine's control, and none on the entry"
        );
    }
}
