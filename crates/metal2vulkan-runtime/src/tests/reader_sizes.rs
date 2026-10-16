//! `reader_sizes.rs`: the draws the snapshot's area rule weighed, counted by texel bytes, side of
//! the rule and share step, and the line written after 64 of them and each power of two past it.

use crate::reader_sizes::ReaderSizes;

/// The pixels of one color in the test.
const TARGET: u64 = 1000 * 1000;

/// Count one draw in `sizes`, keeping the line it writes.
fn add(
    sizes: &mut ReaderSizes,
    lines: &mut Vec<String>,
    (covered, pixels, bytes): (u64, u64, u64),
    within: bool,
) {
    lines.extend(sizes.add(covered, pixels, bytes, within));
}

/// A 4-byte color's reader over 14.5% of its pixels falls in the 14.5% step on the snapshot's
/// side, one over 25% in the 25.0% step on the split's; an 8-byte color's over 30%, the 30.0%
/// step; two colors' reader counts its pixels for each and their bytes over their pixels; a share
/// above the colors' pixels falls in the last step. No line is written before the 64th draw; the
/// 64th and 128th write one, which lists each group's counts in order (by bytes, the split's side
/// first), and no draw between does. A device that ends after a draw past its last line writes one
/// more; one that weighed none, or none since, writes none.
#[test]
fn the_draws_count_by_bytes_side_and_share_and_a_line_follows_each_power_of_two() {
    let mut sizes = ReaderSizes::default();
    let mut lines = Vec::new();
    for _ in 0..40 {
        add(
            &mut sizes,
            &mut lines,
            (TARGET * 145 / 1000, TARGET, TARGET * 4),
            true,
        );
    }
    for _ in 0..20 {
        add(
            &mut sizes,
            &mut lines,
            (TARGET / 4, TARGET, TARGET * 4),
            false,
        );
    }
    add(
        &mut sizes,
        &mut lines,
        (TARGET * 3 / 10, TARGET, TARGET * 8),
        true,
    );
    // Two colors, one of 4 bytes and one of 8: 10% of each, 6 bytes a pixel.
    add(
        &mut sizes,
        &mut lines,
        (2 * TARGET / 10, 2 * TARGET, TARGET * 12),
        true,
    );
    add(
        &mut sizes,
        &mut lines,
        (3 * TARGET, TARGET, TARGET * 16),
        false,
    );
    assert!(lines.is_empty(), "{lines:?}");
    add(&mut sizes, &mut lines, (0, TARGET, TARGET * 4), true);
    assert_eq!(
        lines,
        [
            "64 draws that read their pass's texture weighed by the area rule (texel bytes, side, \
          then share of the colors' pixels the scissors bound, from, in 0.5% steps: count): \
          4 B split 25.0%:20; 4 B snapshot 0.0%:1 14.5%:40; 6 B snapshot 10.0%:1; \
          8 B snapshot 30.0%:1; 16 B split 100%+:1"
        ]
    );
    for _ in 64..127 {
        add(
            &mut sizes,
            &mut lines,
            (TARGET / 2, TARGET, TARGET * 4),
            false,
        );
    }
    assert_eq!(lines.len(), 1, "{lines:?}");
    add(
        &mut sizes,
        &mut lines,
        (TARGET / 2, TARGET, TARGET * 4),
        false,
    );
    assert_eq!(sizes.last_line(), None, "128 draws were written");
    add(&mut sizes, &mut lines, (TARGET, TARGET, TARGET * 4), false);
    let last = sizes.last_line().expect("a line as the device ends");
    assert!(last.starts_with("129 draws "), "{last}");
    assert!(last.contains("50.0%:64 100%+:1;"), "{last}");
    assert_eq!(ReaderSizes::default().last_line(), None, "no draw weighed");
    let second = lines.get(1).expect("a line at 128");
    assert!(second.starts_with("128 draws "), "{second}");
    assert!(
        second.contains("count): 4 B split 25.0%:20 50.0%:64; 4 B snapshot "),
        "{second}"
    );
}
