use super::writer::{ArrayElt, DefOp, Enc, Field, Writer, XorShift};
use super::*;
use crate::budget::Budget;

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

/// A readable trace of every entry (blocks entered, records with ops and blob).
#[derive(Debug, PartialEq, Eq)]
enum Ev {
    Enter(u32),
    Exit(u32),
    Rec {
        code: u32,
        ops: Vec<u64>,
        blob: Option<Vec<u8>>,
        abbrev: Option<u32>,
    },
}

fn walk_with(bytes: &[u8], m: &mut Meter) -> Result<Vec<Ev>, Error> {
    let mut r = Reader::new(bytes)?;
    let mut out = Vec::new();
    loop {
        match r.advance(m)? {
            Entry::SubBlock { id } => out.push(Ev::Enter(id)),
            Entry::EndBlock { id } => out.push(Ev::Exit(id)),
            Entry::Record(rec) => out.push(Ev::Rec {
                code: rec.code,
                ops: rec.ops.to_vec(),
                blob: rec.blob.map(<[u8]>::to_vec),
                abbrev: rec.abbrev,
            }),
            Entry::End => return Ok(out),
        }
    }
}

fn walk(bytes: &[u8]) -> Result<Vec<Ev>, Error> {
    walk_with(bytes, &mut meter())
}

fn rec(code: u32, ops: &[u64]) -> Ev {
    Ev::Rec {
        code,
        ops: ops.to_vec(),
        blob: None,
        abbrev: None,
    }
}

fn arec(code: u32, ops: &[u64], abbrev: u32) -> Ev {
    Ev::Rec {
        code,
        ops: ops.to_vec(),
        blob: None,
        abbrev: Some(abbrev),
    }
}

fn stream_kind(e: &Error) -> BitstreamError {
    match e {
        Error::InvalidInput(InputError {
            kind: InputErrorKind::Bitstream(k),
            ..
        }) => k.clone(),
        other => panic!("expected a bitstream error, got {other:?}"),
    }
}

fn expect_stream_err(bytes: &[u8], want: BitstreamError) {
    let err = walk(bytes).expect_err("malformed stream must fail");
    assert_eq!(stream_kind(&err), want, "{err:?}");
}

/// One block `id` holding `body`.
fn block(id: u64, width: u32, body: impl FnOnce(&mut Writer)) -> Vec<u8> {
    let mut w = Writer::bitcode();
    w.enter(id, width);
    body(&mut w);
    w.end();
    w.finish()
}

#[test]
fn magic_and_empty_stream() {
    let w = Writer::bitcode();
    let bytes = w.finish();
    let r = Reader::new(&bytes).unwrap();
    assert_eq!(r.magic(), [b'B', b'C', 0xC0, 0xDE]);
    assert_eq!(walk(&bytes).unwrap(), vec![]);
}

#[test]
fn stream_must_be_whole_words() {
    let err = Reader::new(&[b'B', b'C', 0xC0]).unwrap_err();
    assert_eq!(stream_kind(&err), BitstreamError::UnexpectedEnd);
    let err = Reader::new(&[b'B', b'C', 0xC0, 0xDE, 0]).unwrap_err();
    assert_eq!(stream_kind(&err), BitstreamError::UnexpectedEnd);
    assert_eq!(
        stream_kind(&Reader::new(&[]).unwrap_err()),
        BitstreamError::UnexpectedEnd
    );
}

#[test]
fn unabbreviated_records_and_vbr_extremes() {
    let values = [0, 1, 31, 32, 1 << 31, u64::from(u32::MAX), u64::MAX];
    let bytes = block(8, 3, |w| {
        w.unabbrev(7, &values);
        w.unabbrev(u64::from(u32::MAX), &[]);
    });
    assert_eq!(
        walk(&bytes).unwrap(),
        vec![
            Ev::Enter(8),
            rec(7, &values),
            rec(u32::MAX, &[]),
            Ev::Exit(8)
        ]
    );
}

#[test]
fn every_abbrev_operand_kind() {
    // code literal 9; fixed(1); fixed(64); vbr(3); vbr(32); char6; literal 77; zero-width fixed
    // and vbr (literal 0); array of vbr6.
    let def = [
        DefOp::Literal(9),
        DefOp::Fixed(1),
        DefOp::Fixed(64),
        DefOp::Vbr(3),
        DefOp::Vbr(32),
        DefOp::Char6,
        DefOp::Literal(77),
        DefOp::Fixed(0),
        DefOp::Vbr(0),
        DefOp::Array,
        DefOp::Vbr(6),
    ];
    let encs = [
        Enc::Literal,
        Enc::Fixed(1),
        Enc::Fixed(64),
        Enc::Vbr(3),
        Enc::Vbr(32),
        Enc::Char6,
        Enc::Literal,
        Enc::Literal,
        Enc::Literal,
        Enc::Array(ArrayElt::Vbr(6)),
    ];
    let fields = [
        Field::Scalar(9),
        Field::Scalar(1),
        Field::Scalar(u64::MAX),
        Field::Scalar(1000),
        Field::Scalar(u64::MAX),
        Field::Scalar(u64::from(b'Z')),
        Field::Scalar(77),
        Field::Scalar(0),
        Field::Scalar(0),
        Field::Array(vec![0, 63, 64, 1 << 40]),
    ];
    let bytes = block(12, 4, |w| {
        w.define(&def);
        w.abbreviated(4, &encs, &fields);
    });
    assert_eq!(
        walk(&bytes).unwrap(),
        vec![
            Ev::Enter(12),
            arec(
                9,
                &[
                    1,
                    u64::MAX,
                    1000,
                    u64::MAX,
                    u64::from(b'Z'),
                    77,
                    0,
                    0,
                    0,
                    63,
                    64,
                    1 << 40
                ],
                4
            ),
            Ev::Exit(12),
        ]
    );
}

#[test]
fn char6_arrays_fixed_arrays_and_blobs() {
    let chars = b"azAZ09._".map(u64::from).to_vec();
    let bytes = block(3, 5, |w| {
        w.define(&[DefOp::Fixed(4), DefOp::Array, DefOp::Char6]);
        w.define(&[DefOp::Vbr(6), DefOp::Array, DefOp::Fixed(7)]);
        w.define(&[DefOp::Literal(2), DefOp::Fixed(3), DefOp::Blob]);
        w.abbreviated(
            4,
            &[Enc::Fixed(4), Enc::Array(ArrayElt::Char6)],
            &[Field::Scalar(5), Field::Array(chars.clone())],
        );
        w.abbreviated(
            5,
            &[Enc::Vbr(6), Enc::Array(ArrayElt::Fixed(7))],
            &[Field::Scalar(1 << 20), Field::Array(vec![])],
        );
        w.abbreviated(
            6,
            &[Enc::Literal, Enc::Fixed(3), Enc::Blob],
            &[
                Field::Scalar(2),
                Field::Scalar(6),
                Field::Blob(b"hello".to_vec()),
            ],
        );
        w.abbreviated(
            6,
            &[Enc::Literal, Enc::Fixed(3), Enc::Blob],
            &[Field::Scalar(2), Field::Scalar(1), Field::Blob(vec![])],
        );
    });
    assert_eq!(
        walk(&bytes).unwrap(),
        vec![
            Ev::Enter(3),
            arec(5, &chars, 4),
            arec(1 << 20, &[], 5),
            Ev::Rec {
                code: 2,
                ops: vec![6],
                blob: Some(b"hello".to_vec()),
                abbrev: Some(6)
            },
            Ev::Rec {
                code: 2,
                ops: vec![1],
                blob: Some(vec![]),
                abbrev: Some(6)
            },
            Ev::Exit(3),
        ]
    );
}

#[test]
fn blockinfo_abbrevs_apply_to_later_blocks_and_precede_local_ones() {
    // Block 17: abbrev 4 comes from BLOCKINFO, abbrev 5 is local. Block 18: abbrev 4 is BLOCKINFO's.
    let mut w = Writer::bitcode();
    w.enter(BLOCKINFO_BLOCK_ID.into(), 2);
    w.unabbrev(u64::from(BLOCKINFO_CODE_SETBID), &[17]);
    w.define(&[DefOp::Literal(40), DefOp::Fixed(8)]);
    w.unabbrev(2, &[b'n'.into()]); // BLOCKNAME: ignored
    w.unabbrev(3, &[1, b'r'.into()]); // SETRECORDNAME: ignored
    w.unabbrev(u64::from(BLOCKINFO_CODE_SETBID), &[18]);
    w.define(&[DefOp::Literal(50)]);
    w.end();
    w.enter(17, 3);
    w.define(&[DefOp::Literal(41), DefOp::Vbr(4)]);
    w.abbreviated(
        4,
        &[Enc::Literal, Enc::Fixed(8)],
        &[Field::Scalar(40), Field::Scalar(200)],
    );
    w.abbreviated(
        5,
        &[Enc::Literal, Enc::Vbr(4)],
        &[Field::Scalar(41), Field::Scalar(99)],
    );
    w.enter(18, 3);
    w.abbreviated(4, &[Enc::Literal], &[Field::Scalar(50)]);
    w.end();
    w.end();
    let bytes = w.finish();
    assert_eq!(
        walk(&bytes).unwrap(),
        vec![
            Ev::Enter(17),
            arec(40, &[200], 4),
            arec(41, &[99], 5),
            Ev::Enter(18),
            arec(50, &[], 4),
            Ev::Exit(18),
            Ev::Exit(17),
        ]
    );
}

#[test]
fn local_abbrevs_end_with_their_block() {
    let mut w = Writer::bitcode();
    w.enter(1, 3);
    w.define(&[DefOp::Literal(5)]);
    w.enter(2, 3);
    // Abbrev 4 is not visible in the child block.
    w.abbreviated(4, &[Enc::Literal], &[Field::Scalar(5)]);
    w.end();
    w.end();
    expect_stream_err(&w.finish(), BitstreamError::UnknownAbbrevId { id: 4 });

    let mut w = Writer::bitcode();
    w.enter(1, 3);
    w.enter(2, 3);
    w.define(&[DefOp::Literal(5)]);
    w.end();
    w.abbreviated(4, &[Enc::Literal], &[Field::Scalar(5)]);
    w.end();
    expect_stream_err(&w.finish(), BitstreamError::UnknownAbbrevId { id: 4 });
}

#[test]
fn blockinfo_added_after_entering_is_not_visible_in_the_open_block() {
    let mut w = Writer::bitcode();
    w.enter(9, 3);
    w.enter(BLOCKINFO_BLOCK_ID.into(), 2);
    w.unabbrev(u64::from(BLOCKINFO_CODE_SETBID), &[9]);
    w.define(&[DefOp::Literal(1)]);
    w.end();
    w.abbreviated(4, &[Enc::Literal], &[Field::Scalar(1)]);
    w.end();
    expect_stream_err(&w.finish(), BitstreamError::UnknownAbbrevId { id: 4 });
}

#[test]
fn malformed_blockinfo() {
    // A definition before any SETBID.
    let bytes = block(BLOCKINFO_BLOCK_ID.into(), 2, |w| {
        w.define(&[DefOp::Literal(1)]);
    });
    expect_stream_err(&bytes, BitstreamError::BadBlockInfo);
    // SETBID without an operand.
    let bytes = block(BLOCKINFO_BLOCK_ID.into(), 2, |w| {
        w.unabbrev(u64::from(BLOCKINFO_CODE_SETBID), &[]);
    });
    expect_stream_err(&bytes, BitstreamError::BadBlockInfo);
    // SETBID out of range.
    let bytes = block(BLOCKINFO_BLOCK_ID.into(), 2, |w| {
        w.unabbrev(u64::from(BLOCKINFO_CODE_SETBID), &[1 << 32]);
    });
    expect_stream_err(&bytes, BitstreamError::ValueOutOfRange);
    // A nested block.
    let bytes = block(BLOCKINFO_BLOCK_ID.into(), 2, |w| {
        w.enter(5, 2);
        w.end();
    });
    expect_stream_err(&bytes, BitstreamError::BadBlockInfo);
}

#[test]
fn skip_block_uses_the_length_word() {
    let mut w = Writer::bitcode();
    w.enter(1, 3);
    w.unabbrev(1, &[1]);
    w.enter(2, 4);
    w.unabbrev(2, &[2, 2]);
    w.enter(3, 2);
    w.end();
    w.end();
    w.unabbrev(3, &[3]);
    w.end();
    w.enter(4, 2);
    w.end();
    let bytes = w.finish();
    let mut m = meter();
    let mut r = Reader::new(&bytes).unwrap();
    assert_eq!(r.advance(&mut m).unwrap(), Entry::SubBlock { id: 1 });
    assert!(matches!(
        r.advance(&mut m).unwrap(),
        Entry::Record(Record { code: 1, .. })
    ));
    assert_eq!(r.advance(&mut m).unwrap(), Entry::SubBlock { id: 2 });
    assert_eq!(r.depth(), 2);
    r.skip_block(&mut m).unwrap();
    assert_eq!(r.depth(), 1);
    assert_eq!(r.block_id(), Some(1));
    assert!(matches!(
        r.advance(&mut m).unwrap(),
        Entry::Record(Record { code: 3, .. })
    ));
    // Skipping the rest of block 1 from inside it.
    r.skip_block(&mut m).unwrap();
    assert_eq!(r.advance(&mut m).unwrap(), Entry::SubBlock { id: 4 });
    assert_eq!(r.advance(&mut m).unwrap(), Entry::EndBlock { id: 4 });
    assert_eq!(r.advance(&mut m).unwrap(), Entry::End);
    // Skipping at the top level is a caller bug, reported, not a panic.
    assert!(matches!(r.skip_block(&mut m), Err(Error::Internal(_))));
}

#[test]
fn nesting_is_capped() {
    let nest = |depth: u64| {
        let mut w = Writer::bitcode();
        for i in 0..depth {
            w.enter(i + 1, 2);
        }
        for _ in 0..depth {
            w.end();
        }
        w.finish()
    };
    let cap = Limit::BitstreamNesting.fixed_cap().unwrap();
    assert!(walk(&nest(cap)).is_ok());
    assert_eq!(
        walk(&nest(cap + 1)).unwrap_err(),
        Error::LimitExceeded(Limit::BitstreamNesting)
    );
}

#[test]
fn malformed_abbreviation_definitions() {
    let bad = |def: &[DefOp]| {
        let def = def.to_vec();
        expect_stream_err(
            &block(1, 3, move |w| w.define(&def)),
            BitstreamError::BadAbbrev,
        );
    };
    bad(&[]);
    bad(&[DefOp::Encoding(0)]);
    bad(&[DefOp::Encoding(6)]);
    bad(&[DefOp::Encoding(7)]);
    bad(&[DefOp::Array, DefOp::Fixed(3)]);
    bad(&[DefOp::Blob]);
    bad(&[DefOp::Literal(1), DefOp::Blob, DefOp::Fixed(2)]);
    bad(&[
        DefOp::Literal(1),
        DefOp::Array,
        DefOp::Fixed(2),
        DefOp::Fixed(2),
    ]);
    bad(&[DefOp::Literal(1), DefOp::Array, DefOp::Literal(3)]);
    bad(&[DefOp::Literal(1), DefOp::Array, DefOp::Fixed(0)]);
    bad(&[DefOp::Literal(1), DefOp::Array, DefOp::Array]);
    bad(&[DefOp::Literal(1), DefOp::Array, DefOp::Blob]);
    bad(&[DefOp::Literal(1), DefOp::Array]);
    bad(&[DefOp::Vbr(1)]);
    bad(&[DefOp::Vbr(33)]);
    bad(&[DefOp::Fixed(65)]);
    // Definitions claiming more operands than bits remain.
    let bytes = block(1, 3, |w| {
        w.fixed(2, 3);
        w.vbr(1 << 20, 5);
    });
    expect_stream_err(&bytes, BitstreamError::UnexpectedEnd);
}

#[test]
fn malformed_records_and_blocks() {
    // Unknown abbrev id.
    expect_stream_err(
        &block(1, 3, |w| w.fixed(7, 3)),
        BitstreamError::UnknownAbbrevId { id: 7 },
    );
    // Record code out of range for u32.
    expect_stream_err(
        &block(1, 3, |w| w.unabbrev(1 << 32, &[])),
        BitstreamError::ValueOutOfRange,
    );
    let bytes = block(1, 3, |w| {
        w.define(&[DefOp::Fixed(40)]);
        w.abbreviated(4, &[Enc::Fixed(40)], &[Field::Scalar(1 << 33)]);
    });
    expect_stream_err(&bytes, BitstreamError::ValueOutOfRange);
    // VBR wider than 64 bits.
    let bytes = block(1, 3, |w| {
        w.fixed(3, 3);
        w.vbr(1, 6);
        w.vbr(1, 6);
        for _ in 0..13 {
            w.fixed(0b111111, 6);
        }
        w.fixed(1, 6);
    });
    expect_stream_err(&bytes, BitstreamError::BadVbr);
    // More operands than bits remain.
    let bytes = block(1, 3, |w| {
        w.fixed(3, 3);
        w.vbr(1, 6);
        w.vbr(1 << 30, 6);
    });
    expect_stream_err(&bytes, BitstreamError::UnexpectedEnd);
    // Array longer than the block.
    let bytes = block(1, 3, |w| {
        w.define(&[DefOp::Literal(1), DefOp::Array, DefOp::Fixed(8)]);
        w.fixed(4, 3);
        w.vbr(1 << 20, 6);
    });
    expect_stream_err(&bytes, BitstreamError::UnexpectedEnd);
    // Blob longer than the block.
    let bytes = block(1, 3, |w| {
        w.define(&[DefOp::Literal(1), DefOp::Blob]);
        w.fixed(4, 3);
        w.vbr(1000, 6);
    });
    expect_stream_err(&bytes, BitstreamError::UnexpectedEnd);
    // Abbrev width 0 and 33; block id beyond u32.
    for width in [0u64, 33] {
        let mut w = Writer::bitcode();
        w.fixed(1, 2);
        w.vbr(1, 8);
        w.vbr(width, 4);
        w.align32();
        w.fixed(0, 32);
        expect_stream_err(&w.finish(), BitstreamError::ValueOutOfRange);
    }
    let mut w = Writer::bitcode();
    w.fixed(1, 2);
    w.vbr(1 << 33, 8);
    w.vbr(2, 4);
    w.align32();
    w.fixed(0, 32);
    expect_stream_err(&w.finish(), BitstreamError::ValueOutOfRange);
    // Only blocks at the top level.
    let mut w = Writer::bitcode();
    w.fixed(3, 2);
    expect_stream_err(&w.finish(), BitstreamError::UnknownAbbrevId { id: 3 });
    let mut w = Writer::bitcode();
    w.fixed(0, 32);
    expect_stream_err(&w.finish(), BitstreamError::UnknownAbbrevId { id: 0 });
}

#[test]
fn block_lengths_are_exact_and_nested_within_the_parent() {
    let good = block(1, 3, |w| w.unabbrev(1, &[1, 2, 3]));
    assert!(walk(&good).is_ok());
    // The length word is at byte 8 (magic, then the aligned header word).
    let patched = |delta: i64| {
        let mut b = good.clone();
        let words = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let words = (i64::from(words) + delta) as u32;
        b[8..12].copy_from_slice(&words.to_le_bytes());
        b
    };
    // Declared longer than the stream.
    expect_stream_err(&patched(1), BitstreamError::BadBlockLength);
    // Declared shorter than the contents: the END_BLOCK cannot be read inside the block.
    expect_stream_err(&patched(-1), BitstreamError::UnexpectedEnd);
    // Declared longer than the contents (padding words inside): END_BLOCK lands early.
    let mut w = Writer::bitcode();
    w.enter(1, 3);
    w.unabbrev(1, &[]);
    w.end();
    w.fixed(0, 32);
    let mut b = w.finish();
    b[8..12].copy_from_slice(&3u32.to_le_bytes());
    expect_stream_err(&b, BitstreamError::BadBlockLength);
    // A child block whose length runs past its parent's end.
    let mut w = Writer::bitcode();
    w.enter(1, 3);
    w.enter(2, 3);
    w.end();
    w.end();
    w.enter(3, 3);
    w.end();
    let mut b = w.finish();
    // Child length word is at byte 16; the parent's contents end at byte 24.
    b[16..20].copy_from_slice(&3u32.to_le_bytes());
    expect_stream_err(&b, BitstreamError::BadBlockLength);
}

#[test]
fn errors_carry_their_location() {
    let bytes = block(6, 3, |w| {
        w.unabbrev(1, &[]);
        w.enter(7, 3);
        w.unabbrev(1, &[]);
        w.unabbrev(1 << 32, &[]);
        w.end();
    });
    let err = walk(&bytes).unwrap_err();
    let Error::InvalidInput(e) = err else {
        panic!("{err:?}")
    };
    let rendered = e.at.to_string();
    assert!(rendered.contains("blocks [6, 7]"), "{rendered}");
    assert!(rendered.contains("record 1"), "{rendered}");
}

#[test]
fn caller_location_names_the_last_record() {
    let bytes = block(6, 3, |w| {
        w.unabbrev(1, &[]);
        w.unabbrev(2, &[]);
    });
    let mut m = meter();
    let mut r = Reader::new(&bytes).unwrap();
    r.advance(&mut m).unwrap();
    r.advance(&mut m).unwrap();
    let Entry::Record(rec) = r.advance(&mut m).unwrap() else {
        panic!()
    };
    let bit = rec.bit;
    let loc = r.location();
    assert_eq!(loc, Location::bitstream(bit, &[6], Some(1)));
}

#[test]
fn work_units_are_deterministic() {
    let bytes = rich_stream();
    let mut a = meter();
    let mut b = meter();
    let ev_a = walk_with(&bytes, &mut a).unwrap();
    let ev_b = walk_with(&bytes, &mut b).unwrap();
    assert_eq!(ev_a, ev_b);
    assert_eq!(a.work(), b.work());
    assert!(a.work() > 0);
    assert!(a.ledger() > 0, "reservations are charged");
    // A tight work cap trips deterministically.
    let mut tight = Meter::with_caps(
        &Budget::default(),
        a.work() - 1,
        crate::budget::LEDGER_BYTES,
    );
    assert_eq!(
        walk_with(&bytes, &mut tight).unwrap_err(),
        Error::LimitExceeded(Limit::WorkUnits)
    );
    // A tight ledger trips before reserving.
    let mut tight = Meter::with_caps(&Budget::default(), crate::budget::WORK_UNITS, 16);
    assert_eq!(
        walk_with(&bytes, &mut tight).unwrap_err(),
        Error::LimitExceeded(Limit::LedgerBytes)
    );
}

/// A stream using every construct: BLOCKINFO, nested blocks, each operand kind, blobs.
fn rich_stream() -> Vec<u8> {
    let mut w = Writer::bitcode();
    w.enter(BLOCKINFO_BLOCK_ID.into(), 2);
    w.unabbrev(u64::from(BLOCKINFO_CODE_SETBID), &[11]);
    w.define(&[DefOp::Literal(3), DefOp::Vbr(6), DefOp::Array, DefOp::Char6]);
    w.end();
    w.enter(8, 3);
    w.unabbrev(1, &[2]);
    w.define(&[DefOp::Fixed(5), DefOp::Fixed(17), DefOp::Blob]);
    for i in 0..5u64 {
        w.abbreviated(
            4,
            &[Enc::Fixed(5), Enc::Fixed(17), Enc::Blob],
            &[
                Field::Scalar(i),
                Field::Scalar(i * 1000),
                Field::Blob(vec![i as u8; i as usize * 3]),
            ],
        );
    }
    w.enter(11, 4);
    for i in 0..6u64 {
        w.abbreviated(
            4,
            &[Enc::Literal, Enc::Vbr(6), Enc::Array(ArrayElt::Char6)],
            &[
                Field::Scalar(3),
                Field::Scalar(i << 20),
                Field::Array(
                    b"name_1.x"
                        .iter()
                        .take(i as usize)
                        .map(|&c| c.into())
                        .collect(),
                ),
            ],
        );
        w.unabbrev(9, &[i, i * i, u64::MAX >> i]);
    }
    w.end();
    w.enter(12, 3);
    w.define(&[DefOp::Vbr(8), DefOp::Array, DefOp::Fixed(1)]);
    w.abbreviated(
        4,
        &[Enc::Vbr(8), Enc::Array(ArrayElt::Fixed(1))],
        &[Field::Scalar(300), Field::Array(vec![1, 0, 1, 1])],
    );
    w.end();
    w.end();
    w.enter(13, 2);
    w.unabbrev(1, &[]);
    w.end();
    w.finish()
}

/// Bit positions after each entry (entry boundaries) and the depth there, from a clean walk.
fn boundaries(bytes: &[u8]) -> Vec<(u64, usize)> {
    let mut m = meter();
    let mut r = Reader::new(bytes).unwrap();
    let mut out = vec![(r.bit_position(), 0)];
    loop {
        let e = r.advance(&mut m).unwrap();
        let end = matches!(e, Entry::End);
        out.push((r.bit_position(), r.depth()));
        if end {
            return out;
        }
    }
}

#[test]
fn truncation_at_every_entry_boundary_is_an_error() {
    let bytes = rich_stream();
    let full = bytes.len();
    let points = boundaries(&bytes);
    // Between top-level blocks the stream is a complete (shorter) bitstream; the bitcode layer,
    // not this one, requires the module. Everywhere else a cut must fail.
    let top_level: Vec<usize> = points
        .iter()
        .filter(|(_, depth)| *depth == 0)
        .map(|(bit, _)| usize::try_from(bit / 8).unwrap())
        .collect();
    let mut checked = 0;
    for (bit, _) in &points {
        // Cut at the word containing the boundary.
        let cut = usize::try_from(bit / 32 * 4).unwrap();
        if cut == 0 || cut >= full || top_level.contains(&cut) {
            continue;
        }
        let err = walk(&bytes[..cut]).expect_err("truncated stream must fail");
        assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
        checked += 1;
    }
    assert!(checked > 10);
    // A cut at any word decodes only when it ends between top-level blocks (including after the
    // internally consumed BLOCKINFO), and then yields a prefix of the full entries.
    let all = walk(&bytes).unwrap();
    for cut in (4..full).step_by(4) {
        match walk(&bytes[..cut]) {
            Ok(events) => assert!(all.starts_with(&events), "cut at byte {cut}"),
            Err(err) => assert!(matches!(err, Error::InvalidInput(_)), "{err:?}"),
        }
    }
}

#[test]
fn bit_flips_never_panic_and_work_stays_bounded() {
    let bytes = rich_stream();
    let bits = bytes.len() as u64 * 8;
    // Work per entry is one unit plus its operands; operands are bounded by the bits read, except
    // literal operands, which the (bounded) abbreviation definitions multiply.
    let bound = bits * bits;
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let (mut ok, mut failed) = (0, 0);
    for _ in 0..4000 {
        let mut mutated = bytes.clone();
        let flips = 1 + rng.next() % 3;
        for _ in 0..flips {
            let bit = rng.next() % bits;
            mutated[(bit / 8) as usize] ^= 1 << (bit % 8);
        }
        let mut m = meter();
        match walk_with(&mutated, &mut m) {
            Ok(_) => ok += 1,
            Err(Error::InvalidInput(_) | Error::LimitExceeded(_)) => failed += 1,
            Err(other) => panic!("unexpected error class: {other:?}"),
        }
        assert!(m.work() <= bound, "work {} > {bound}", m.work());
    }
    assert!(failed > 0 && ok + failed == 4000);
}

#[test]
fn the_operand_buffer_is_charged_on_growth_not_per_record() {
    let bytes = block(1, 4, |w| {
        w.define(&[DefOp::Literal(2), DefOp::Vbr(6), DefOp::Vbr(6)]);
        for i in 0..10_000u64 {
            w.unabbrev(1, &[i, i, i]);
            w.abbreviated(
                4,
                &[Enc::Literal, Enc::Vbr(6), Enc::Vbr(6)],
                &[Field::Scalar(2), Field::Scalar(i), Field::Scalar(i)],
            );
        }
    });
    let mut m = meter();
    assert_eq!(walk_with(&bytes, &mut m).unwrap().len(), 20_002);
    assert!(m.ledger() < 1024, "ledger {}", m.ledger());
}
