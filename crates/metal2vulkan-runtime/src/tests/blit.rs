//! The blit encoder's buffer command mapping (`blit.rs`).

use crate::blit::{fill_pieces, fill_word, FillPieces};

fn pieces(
    head: std::ops::Range<u64>,
    words: std::ops::Range<u64>,
    tail: std::ops::Range<u64>,
) -> FillPieces {
    FillPieces { head, words, tail }
}

#[test]
fn an_aligned_fill_is_all_words() {
    assert_eq!(fill_pieces(0..16), pieces(0..0, 0..16, 16..16));
    assert_eq!(fill_pieces(8..12), pieces(8..8, 8..12, 12..12));
}

#[test]
fn an_unaligned_fill_copies_its_ends() {
    assert_eq!(fill_pieces(2..18), pieces(2..4, 4..16, 16..18));
    assert_eq!(fill_pieces(1..8), pieces(1..4, 4..8, 8..8));
    assert_eq!(fill_pieces(4..7), pieces(4..4, 4..4, 4..7));
}

#[test]
fn a_fill_without_a_whole_word_is_all_copies() {
    // Inside one word.
    assert_eq!(fill_pieces(5..7), pieces(5..7, 7..7, 7..7));
    // Across a word boundary.
    assert_eq!(fill_pieces(1..7), pieces(1..4, 4..4, 4..7));
}

#[test]
fn an_empty_fill_has_no_piece() {
    assert_eq!(fill_pieces(6..6), FillPieces::default());
}

#[test]
fn the_pieces_cover_the_range_exactly() {
    for start in 0..24u64 {
        for end in start..32u64 {
            let got = fill_pieces(start..end);
            assert!(
                got.head.end - got.head.start < 4 && got.tail.end - got.tail.start < 4,
                "{start}..{end}: {got:?}"
            );
            if !got.words.is_empty() {
                assert_eq!(got.words.start % 4, 0, "{start}..{end}: {got:?}");
                assert_eq!(got.words.end % 4, 0, "{start}..{end}: {got:?}");
            }
            let covered: Vec<u64> = got
                .head
                .clone()
                .chain(got.words.clone())
                .chain(got.tail.clone())
                .collect();
            assert_eq!(
                covered,
                (start..end).collect::<Vec<_>>(),
                "{start}..{end}: {got:?}"
            );
        }
    }
}

#[test]
fn the_word_repeats_the_byte() {
    assert_eq!(fill_word(0x5a).to_ne_bytes(), [0x5a; 4]);
    assert_eq!(fill_word(0), 0);
    assert_eq!(fill_word(0xff), u32::MAX);
}

#[test]
fn a_fill_at_the_end_of_the_address_range_does_not_overflow() {
    let end = u64::MAX;
    let got = fill_pieces(end - 2..end);
    assert_eq!(got.head.start, end - 2);
    assert_eq!(got.tail.end, end);
}
