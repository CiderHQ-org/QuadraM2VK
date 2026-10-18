//! The imageblock size a compute encoder takes (`compute::imageblock_size`), and an indirect
//! command's (`compute::validated_imageblock_size`).

use super::super::compute::{imageblock_size, validated_imageblock_size};
use super::super::error::Problem;
use metal2vulkan::bind::ImageblockSize;

/// Metal's API validation takes a width and a height that are powers of two and at most 1024 cells,
/// and refuses the rest at `setImageblockWidth:height:`, the same on an M3 and an M4 whatever the
/// pipeline (probe 6). An indirect command's size is held to it.
#[test]
fn a_validated_imageblock_size_is_powers_of_two_of_at_most_1024_cells() {
    for (width, height) in [
        (1, 1),
        (8, 8),
        (16, 16),
        (32, 32),
        (1024, 1),
        (1, 1024),
        (64, 16),
    ] {
        assert_eq!(
            validated_imageblock_size(width, height),
            Ok(ImageblockSize::new(width as u32, height as u32))
        );
    }
    for (width, height) in [
        (0, 0),
        (0, 8),
        (8, 0),
        (3, 5),
        (5, 3),
        (48, 40),
        (64, 33),
        (64, 32),
        (2048, 1),
        (65536, 1),
        (4096, 4096),
        (1 << 40, 1),
        (1 << 63, 1 << 63),
        (u64::MAX, 1),
    ] {
        assert_eq!(
            validated_imageblock_size(width, height),
            Err(Problem::ImageblockSize { width, height }),
            "{width}x{height}"
        );
    }
}

/// Plain Metal's answers where its API validation asserts, measured on an M3 Max and an M4 alike,
/// and on an M4, each the size the kernel's `get_width` and `get_height` read. A side of 0 reads 1
/// by 1, a side that is not a power of two reads the power above, and more than 1024 cells in one
/// row or column read one cell. More than 1024 cells over more rows and columns read 1024 by 1 with
/// every cell 0, and a side past 65536 is unmeasured: both stay refused.
#[test]
fn an_imageblock_size_is_what_plain_metal_sets() {
    for ((width, height), read) in [
        ((8, 8), (8, 8)),
        ((0, 0), (1, 1)),
        ((0, 3), (1, 1)),
        ((0, 8), (1, 1)),
        ((5, 0), (1, 1)),
        ((8, 0), (1, 1)),
        ((0, 2048), (1, 1)),
        ((2048, 0), (1, 1)),
        ((3, 5), (4, 8)),
        ((5, 3), (8, 4)),
        ((6, 6), (8, 8)),
        ((17, 17), (32, 32)),
        ((3, 1), (4, 1)),
        ((1, 3), (1, 4)),
        ((1023, 1), (1024, 1)),
        ((2048, 1), (1, 1)),
        ((65536, 1), (1, 1)),
        ((1, 2048), (1, 1)),
    ] {
        assert_eq!(
            imageblock_size(width, height),
            Ok(ImageblockSize::new(read.0, read.1)),
            "{width}x{height}"
        );
    }
    for (width, height) in [
        (2, 1024),
        (33, 31),
        (40, 48),
        (48, 40),
        (64, 32),
        (64, 33),
        (64, 64),
        (65537, 1),
        (1, 1 << 40),
        (u64::MAX, u64::MAX),
    ] {
        assert_eq!(
            imageblock_size(width, height),
            Err(Problem::ImageblockSize { width, height }),
            "{width}x{height}"
        );
    }
}
