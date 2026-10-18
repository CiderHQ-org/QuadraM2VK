//! `metal_level`: each level's row of the table is what its release measured, a device's level is
//! chosen at or below its macOS host's release, and the descriptor check and the alignment query
//! answer at the device's level.

use super::capabilities::{committed_descriptions, created_at};
use crate::metal_level::{choose, parse_version, Chosen, Host};
use crate::texture::heap::no_texture_alignment;
use crate::texture_descriptor::{Creator, Refused, TextureDescriptor, TextureProblem};
use crate::texture_descriptor::{TextureRules, APPLE_SAMPLE_COUNTS};
use crate::{BufferTextureProblem, Error, Invalid, MetalLevel, Problem, Unproven};
use metal2vulkan::bind::MipmapRule;
use metal2vulkan::metal::PixelFormat as P;
use metal2vulkan::metal::{TextureType, TextureUsage};

/// The 12 three-channel formats of the macOS 27.0 SDK and `RGB16Bfloat`: values macOS 26.7's
/// descriptor does not know, and formats macOS 27.0.1 describes none of.
const THREE_CHANNEL: [P; 13] = [
    P::RGB8Unorm,
    P::RGB8Snorm,
    P::RGB8Uint,
    P::RGB8Sint,
    P::RGB16Unorm,
    P::RGB16Snorm,
    P::RGB16Uint,
    P::RGB16Sint,
    P::RGB16Float,
    P::RGB16Bfloat,
    P::RGB32Uint,
    P::RGB32Sint,
    P::RGB32Float,
];

/// The formats from 646 to 650, which neither release describes.
const TWELVE_BIT_UINT: [P; 5] = [
    P::R12UintX4,
    P::R12UintPACKED,
    P::RG12UintX8,
    P::RG12UintPACKED,
    P::RGB10A8Uint2P,
];

/// The formats from 630 to 645: made by an Apple M4 under macOS 27.0.1, described by neither
/// release on an Apple M3 Max (at 26.7; at 27.0.1 the consumer's macOS 27 host lane and where the
/// alignment query asserts and the descriptor's validation stops a texture of each, "has invalid
/// pixelFormat").
const FROM_630: [P; 16] = [
    P::BGRA10UintPACKED,
    P::BGRA10XRPACKED,
    P::BGRA10XRPACKEDSRGB,
    P::BGRA10UnormPACKED,
    P::BGRA10UnormPACKEDSRGB,
    P::BGRA10UnormPACKEDHLG,
    P::BGRA10UnormPACKEDPQ,
    P::BGRA10Uint,
    P::BGRA10Unorm,
    P::BGRA10UnormSRGB,
    P::BGRA10UnormHLG,
    P::BGRA10UnormPQ,
    P::R10UintX6,
    P::R10UintPACKED,
    P::RG10UintX12,
    P::RG10UintPACKED,
];

/// The formats a level describes, of every value the runtime names.
fn described(level: MetalLevel) -> Vec<P> {
    P::ALL
        .iter()
        .copied()
        .filter(|&pf| level.describes(pf))
        .collect()
}

/// Each level's row is its release's measurement on an Apple M3 Max: macOS 26.7 and 27.0.1 each
/// describe 264 values (`MTLPixelFormatGetInfoForDevice`, our probe and the consumer's
/// macOS 27 host lane), without `Depth24Unorm_Stencil8` and `X24_Stencil8`, which no Apple GPU
/// has, and without the 16 from 630 to 645, which an Apple M4 under 27.0.1 also describes; 26.7's
/// descriptor does not know the three-channel formats; and a newer release describes every format
/// an older one does.
#[test]
fn each_level_describes_what_its_release_measured() {
    let apple_lacks = [P::Depth24UnormStencil8, P::X24Stencil8];
    let named = P::ALL.iter().filter(|&&pf| pf != P::Invalid).count();
    for level in MetalLevel::ALL {
        assert_eq!(described(level).len(), 264, "{level:?}");
        assert_eq!(named - described(level).len(), 36, "{level:?}");
    }
    for &pf in P::ALL {
        let three = THREE_CHANNEL.contains(&pf);
        assert!(MetalLevel::MacOS27_0_1.knows(pf), "{pf:?}");
        assert_eq!(MetalLevel::MacOS26_7.knows(pf), !three, "{pf:?}");
        let an_m3s = !three
            && !TWELVE_BIT_UINT.contains(&pf)
            && !FROM_630.contains(&pf)
            && !apple_lacks.contains(&pf)
            && pf != P::Invalid;
        for level in MetalLevel::ALL {
            assert_eq!(level.describes(pf), an_m3s, "{level:?} {pf:?}");
        }
    }
    // Ordered by release, and a newer release describes what an older one does.
    assert!(MetalLevel::ALL.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(MetalLevel::NEWEST, *MetalLevel::ALL.last().unwrap());
    for pair in MetalLevel::ALL.windows(2) {
        for &pf in P::ALL {
            assert!(!pair[0].describes(pf) || pair[1].describes(pf), "{pf:?}");
        }
    }
    assert_eq!(MetalLevel::MacOS26_7.to_string(), "macOS 26.7");
    assert_eq!(MetalLevel::MacOS27_0_1.macos(), "27.0.1");
}

/// Each level makes mipmaps by its release's rule on an Apple M3 Max (`bind::MipmapRule`: macOS
/// 26.7's passes, 27.0.1's filter at every level), and answers the heap query of a texture of a
/// format of which Metal makes none alone, with no planes, its release's alignment: 16 KiB for an
/// array, a cube, a cube array, a 3D texture and a multisample array under 26.7, 128 under 27.0.1,
/// and alike for the other types.
#[test]
fn a_level_takes_its_release_s_mipmaps_and_nil_alignments() {
    assert_eq!(
        MetalLevel::MacOS26_7.mipmap_rule(),
        MipmapRule::AppleM3MaxPasses
    );
    assert_eq!(MetalLevel::MacOS27_0_1.mipmap_rule(), MipmapRule::Filter);
    use TextureType as T;
    let types = [
        (T::Type1D, 128, 128),
        (T::Type2D, 128, 128),
        (T::Type2DMultisample, 128, 128),
        (T::TextureBuffer, 16, 16),
        (T::Type1DArray, 16 << 10, 128),
        (T::Type2DArray, 16 << 10, 128),
        (T::Cube, 16 << 10, 128),
        (T::CubeArray, 16 << 10, 128),
        (T::Type3D, 16 << 10, 128),
        (T::Type2DMultisampleArray, 16 << 10, 128),
    ];
    for pf in [
        P::Unspecialized,
        P::RGBA16FloatXR,
        P::AnyDepth,
        P::AnyStencil,
    ] {
        for (ty, older, newer) in types {
            assert_eq!(
                no_texture_alignment(pf, ty, MetalLevel::MacOS26_7),
                older,
                "{pf:?} {ty:?}"
            );
            assert_eq!(
                no_texture_alignment(pf, ty, MetalLevel::MacOS27_0_1),
                newer,
                "{pf:?} {ty:?}"
            );
        }
    }
}

/// A macOS host's release reads as major, minor and patch, the patch 0 where absent.
#[test]
fn a_host_release_parses() {
    assert_eq!(parse_version("26.7"), Some([26, 7, 0]));
    assert_eq!(parse_version("27.0.1"), Some([27, 0, 1]));
    assert_eq!(parse_version(" 27.0.1\n"), Some([27, 0, 1]));
    for refused in ["", "27", "27.", "27.0.1.2", "27.x", "-1.0"] {
        assert_eq!(parse_version(refused), None, "{refused:?}");
    }
}

/// A device's level is the one its options name, on any host; else on macOS the host's release,
/// or the newest measured release below it (announced), and never one above it: a host below
/// every measured release, or whose release cannot be read, refuses the device; off macOS the
/// newest.
#[test]
fn a_device_answers_at_or_below_its_host() {
    let host = |v: [u32; 3]| Host::MacOS(Some(v));
    for level in MetalLevel::ALL {
        for on in [
            host([25, 0, 0]),
            host([28, 0, 0]),
            Host::MacOS(None),
            Host::Other,
        ] {
            assert_eq!(choose(Some(level), on, "t"), Ok(Chosen::Asked(level)));
        }
    }
    assert_eq!(
        choose(None, Host::Other, "t"),
        Ok(Chosen::NotMacOS(MetalLevel::NEWEST))
    );
    let cases = [
        ([26, 7, 0], Chosen::Host(MetalLevel::MacOS26_7)),
        ([27, 0, 1], Chosen::Host(MetalLevel::MacOS27_0_1)),
        (
            [26, 7, 1],
            Chosen::Below {
                host: [26, 7, 1],
                level: MetalLevel::MacOS26_7,
            },
        ),
        (
            [27, 0, 0],
            Chosen::Below {
                host: [27, 0, 0],
                level: MetalLevel::MacOS26_7,
            },
        ),
        (
            [28, 1, 0],
            Chosen::Below {
                host: [28, 1, 0],
                level: MetalLevel::MacOS27_0_1,
            },
        ),
    ];
    for (version, want) in cases {
        let chosen = choose(None, host(version), "t");
        assert_eq!(chosen, Ok(want), "{version:?}");
        assert!(want.level() <= MetalLevel::NEWEST);
    }
    for version in [Some([26, 6, 0]), Some([15, 7, 0]), None] {
        assert_eq!(
            choose(None, Host::MacOS(version), "from_raw"),
            Err(Error::Unproven(Unproven::new(
                "from_raw",
                Problem::HostReleaseUnmeasured { macos: version }
            ))),
            "{version:?}"
        );
    }
    // A level below the host's release is announced, and the default off macOS; a level named or
    // the host's own is not.
    let below = Chosen::Below {
        host: [26, 7, 1],
        level: MetalLevel::MacOS26_7,
    };
    let (kind, (object, reason)) = below.announcement().expect("announced");
    assert_eq!(kind, crate::Degradation::MetalLevelBelowHost);
    assert_eq!(object, "the device, on macOS 26.7.1");
    assert!(reason.ends_with("answers as macOS 26.7, the newest measured release below it"));
    let (kind, (object, reason)) = Chosen::NotMacOS(MetalLevel::NEWEST)
        .announcement()
        .expect("announced");
    assert_eq!(kind, crate::Degradation::MetalLevelNotMacOS);
    assert_eq!(object, "the device, on a host that is not macOS");
    assert!(reason.contains("answers as macOS 27.0.1, the newest measured release"));
    for quiet in [
        Chosen::Asked(MetalLevel::MacOS26_7),
        Chosen::Host(MetalLevel::MacOS26_7),
    ] {
        assert_eq!(quiet.announcement(), None, "{quiet:?}");
    }
}

/// A texture of a format the device's level describes none of is stopped as `Invalid` is, from
/// every creator and the heap size query, before every other rule; one of a format it describes
/// is not (Metal's validation stops it, "has invalid pixelFormat": our probes; at macOS 27.0.1 on
/// the M3 Max). The 16 formats from 630 to 645 are stopped at both levels, as the M3 Max stops them
/// at both releases.
#[test]
fn a_level_stops_textures_of_the_formats_it_describes_none_of() {
    let creators = [
        Creator::Device,
        Creator::Buffer,
        Creator::OverSurface,
        Creator::Shared,
        Creator::SizeQuery,
    ];
    let stop = Err(Refused::Stops(TextureProblem::NotAFormat));
    for level in MetalLevel::ALL {
        let rules = TextureRules {
            sample_counts: APPLE_SAMPLE_COUNTS,
            level,
        };
        for &pf in P::ALL {
            let mut d = TextureDescriptor::texture_2d(pf, 16, 16, false);
            d.usage = TextureUsage::SHADER_READ;
            for creator in creators {
                let checked = d.check_made(creator, rules);
                if level.describes(pf) {
                    assert_ne!(checked, stop, "{level:?} {pf:?} {creator:?}");
                } else {
                    assert_eq!(checked, stop, "{level:?} {pf:?} {creator:?}");
                }
            }
            // Before a rule it breaks too: a zero extent.
            let mut zero = d;
            zero.width = 0;
            if !level.describes(pf) {
                assert_eq!(zero.check(rules), stop, "{level:?} {pf:?}");
            }
        }
    }
    for pf in FROM_630 {
        let d = TextureDescriptor::texture_2d(pf, 16, 16, false);
        let at = |level| TextureRules {
            sample_counts: APPLE_SAMPLE_COUNTS,
            level,
        };
        for level in MetalLevel::ALL {
            assert_eq!(d.check(at(level)), stop, "{level:?} {pf:?}");
        }
    }
}

/// `minimumLinearTextureAlignmentForPixelFormat:` answers at the device's level, an Apple M3 Max's
/// at each: an assertion for the formats from 630 to 645 at macOS 26.7 and at 27.0.1
/// ("MTLPixelFormatInvalid is not supported on this device"; an Apple M4 under 27.0.1 answers 16),
/// which the runtime refuses `Invalid`.
#[test]
fn the_linear_alignment_answers_at_the_devices_level() {
    let (_, desc) = committed_descriptions()
        .into_iter()
        .find(|(s, _)| s == "moltenvk-m3")
        .unwrap();
    let (_, older) = created_at("moltenvk-m3", &desc, MetalLevel::MacOS26_7);
    let (_, newer) = created_at("moltenvk-m3", &desc, MetalLevel::MacOS27_0_1);
    assert_eq!(older.metal_level, MetalLevel::MacOS26_7);
    assert_eq!(newer.metal_level, MetalLevel::MacOS27_0_1);
    let call = "minimum_linear_texture_alignment_for_pixel_format";
    for pf in FROM_630 {
        for p in [&older, &newer] {
            assert_eq!(
                p.minimum_linear_texture_alignment_for_pixel_format(pf),
                Err(Error::Invalid(Invalid::new(
                    call,
                    Problem::BufferTexture(BufferTextureProblem::NotLinear { pixel_format: pf }),
                ))),
                "{:?} {pf:?}",
                p.metal_level
            );
        }
    }
    for pf in [P::RGBA8Unorm, P::R10UnormX6, P::RGB10A82P] {
        for p in [&older, &newer] {
            assert_eq!(
                p.minimum_linear_texture_alignment_for_pixel_format(pf),
                Ok(16)
            );
        }
    }
}

/// No level describes `Depth24Unorm_Stencil8` or `X24_Stencil8`, which an Apple GPU has neither
/// of: the descriptor check stops a texture of each as one of no format, first, from every
/// creator, at every level, as Metal on an Apple M4 ("has invalid pixelFormat
/// (255)") and an M3 Max (the consumer's run) stops them, whatever the Vulkan driver has.
#[test]
fn no_level_describes_either_24_bit_depth_format() {
    let creators = [
        Creator::Device,
        Creator::Buffer,
        Creator::OverSurface,
        Creator::Shared,
        Creator::SizeQuery,
    ];
    let stop = Err(Refused::Stops(TextureProblem::NotAFormat));
    for level in MetalLevel::ALL {
        let rules = TextureRules {
            sample_counts: APPLE_SAMPLE_COUNTS,
            level,
        };
        for pf in [P::Depth24UnormStencil8, P::X24Stencil8] {
            assert!(level.knows(pf), "{level:?} {pf:?}");
            assert!(!level.describes(pf), "{level:?} {pf:?}");
            let mut d = TextureDescriptor::texture_2d(pf, 16, 16, false);
            d.usage = TextureUsage::RENDER_TARGET;
            for creator in creators {
                assert_eq!(
                    d.check_made(creator, rules),
                    stop,
                    "{level:?} {pf:?} {creator:?}"
                );
            }
        }
    }
}
