//! Test-only bitstream assembler: hand-built bit sequences for reader tests.

/// An abbreviation operand as written by `DEFINE_ABBREV`.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DefOp {
    Literal(u64),
    Fixed(u64),
    Vbr(u64),
    /// The array marker; the element encoding follows as its own `DefOp`.
    Array,
    Char6,
    Blob,
    /// A raw encoding number (for malformed definitions).
    Encoding(u64),
}

/// A field value for an abbreviated record, matching the abbreviation's operands in order.
#[derive(Clone, Debug)]
pub(crate) enum Field {
    Scalar(u64),
    Array(Vec<u64>),
    Blob(Vec<u8>),
}

/// Scalar encodings used when writing abbreviated records.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Enc {
    Literal,
    Fixed(u32),
    Vbr(u32),
    Char6,
    Array(ArrayElt),
    Blob,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ArrayElt {
    Fixed(u32),
    Vbr(u32),
    Char6,
}

/// LSB-first bit writer with a stack of open blocks.
#[derive(Debug, Default)]
pub(crate) struct Writer {
    bytes: Vec<u8>,
    bits: u64,
    /// Open blocks: (abbrev width of the parent, byte offset of the length word).
    blocks: Vec<(u32, usize)>,
    width: u32,
}

impl Writer {
    /// A writer that starts with a 4-byte magic.
    pub(crate) fn with_magic(magic: [u8; 4]) -> Writer {
        let mut w = Writer {
            width: 2,
            ..Writer::default()
        };
        for b in magic {
            w.fixed(u64::from(b), 8);
        }
        w
    }

    /// The LLVM bitcode magic `BC 0xC0DE`.
    pub(crate) fn bitcode() -> Writer {
        Writer::with_magic([b'B', b'C', 0xC0, 0xDE])
    }

    pub(crate) fn fixed(&mut self, value: u64, width: u32) {
        for i in 0..width {
            let bit = (value >> i) & 1;
            let byte = (self.bits / 8) as usize;
            if byte == self.bytes.len() {
                self.bytes.push(0);
            }
            self.bytes[byte] |= (bit as u8) << (self.bits % 8);
            self.bits += 1;
        }
    }

    pub(crate) fn vbr(&mut self, mut value: u64, width: u32) {
        let payload = width - 1;
        loop {
            let chunk = value & ((1u64 << payload) - 1);
            value >>= payload;
            if value == 0 {
                self.fixed(chunk, width);
                return;
            }
            self.fixed(chunk | (1u64 << payload), width);
        }
    }

    /// The number of bits written so far.
    pub(crate) fn bit(&self) -> u64 {
        self.bits
    }

    pub(crate) fn align32(&mut self) {
        while !self.bits.is_multiple_of(32) {
            self.fixed(0, 1);
        }
    }

    /// `ENTER_SUBBLOCK` with a placeholder length word.
    pub(crate) fn enter(&mut self, id: u64, width: u32) {
        self.fixed(1, self.width);
        self.vbr(id, 8);
        self.vbr(u64::from(width), 4);
        self.align32();
        let at = self.bytes.len();
        self.fixed(0, 32);
        self.blocks.push((self.width, at));
        self.width = width;
    }

    /// `END_BLOCK`, patching the block's length word.
    pub(crate) fn end(&mut self) {
        self.fixed(0, self.width);
        self.align32();
        let (parent, at) = self.blocks.pop().expect("open block");
        let words = ((self.bytes.len() - at - 4) / 4) as u32;
        self.bytes[at..at + 4].copy_from_slice(&words.to_le_bytes());
        self.width = parent;
    }

    pub(crate) fn unabbrev(&mut self, code: u64, ops: &[u64]) {
        self.fixed(3, self.width);
        self.vbr(code, 6);
        self.vbr(ops.len() as u64, 6);
        for &op in ops {
            self.vbr(op, 6);
        }
    }

    /// `DEFINE_ABBREV` with `ops` written verbatim (numops = `ops.len()`).
    pub(crate) fn define(&mut self, ops: &[DefOp]) {
        self.fixed(2, self.width);
        self.vbr(ops.len() as u64, 5);
        for op in ops {
            match *op {
                DefOp::Literal(v) => {
                    self.fixed(1, 1);
                    self.vbr(v, 8);
                }
                DefOp::Fixed(w) => {
                    self.fixed(0, 1);
                    self.fixed(1, 3);
                    self.vbr(w, 5);
                }
                DefOp::Vbr(w) => {
                    self.fixed(0, 1);
                    self.fixed(2, 3);
                    self.vbr(w, 5);
                }
                DefOp::Array => {
                    self.fixed(0, 1);
                    self.fixed(3, 3);
                }
                DefOp::Char6 => {
                    self.fixed(0, 1);
                    self.fixed(4, 3);
                }
                DefOp::Blob => {
                    self.fixed(0, 1);
                    self.fixed(5, 3);
                }
                DefOp::Encoding(e) => {
                    self.fixed(0, 1);
                    self.fixed(e, 3);
                }
            }
        }
    }

    /// An abbreviated record: abbrev id, then each field per `encs` (literals write nothing).
    pub(crate) fn abbreviated(&mut self, id: u64, encs: &[Enc], fields: &[Field]) {
        self.fixed(id, self.width);
        assert_eq!(encs.len(), fields.len());
        for (enc, field) in encs.iter().zip(fields) {
            match (*enc, field) {
                (Enc::Literal, _) => {}
                (Enc::Fixed(w), Field::Scalar(v)) => self.fixed(*v, w),
                (Enc::Vbr(w), Field::Scalar(v)) => self.vbr(*v, w),
                (Enc::Char6, Field::Scalar(v)) => self.fixed(encode_char6(*v), 6),
                (Enc::Array(elt), Field::Array(vs)) => {
                    self.vbr(vs.len() as u64, 6);
                    for &v in vs {
                        match elt {
                            ArrayElt::Fixed(w) => self.fixed(v, w),
                            ArrayElt::Vbr(w) => self.vbr(v, w),
                            ArrayElt::Char6 => self.fixed(encode_char6(v), 6),
                        }
                    }
                }
                (Enc::Blob, Field::Blob(bytes)) => {
                    self.vbr(bytes.len() as u64, 6);
                    self.align32();
                    for &b in bytes {
                        self.fixed(u64::from(b), 8);
                    }
                    self.align32();
                }
                (enc, field) => panic!("field {field:?} does not match {enc:?}"),
            }
        }
    }

    /// The bytes, padded to a whole number of 32-bit words.
    pub(crate) fn finish(mut self) -> Vec<u8> {
        assert!(self.blocks.is_empty(), "unclosed block");
        self.align32();
        self.bytes
    }
}

fn encode_char6(c: u64) -> u64 {
    let c = u8::try_from(c).expect("char6 value");
    match c {
        b'a'..=b'z' => u64::from(c - b'a'),
        b'A'..=b'Z' => u64::from(c - b'A') + 26,
        b'0'..=b'9' => u64::from(c - b'0') + 52,
        b'.' => 62,
        b'_' => 63,
        _ => panic!("not a char6 character: {c}"),
    }
}

/// A fixed-seed xorshift generator for deterministic mutations.
pub(crate) struct XorShift(pub(crate) u64);

impl XorShift {
    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}
