//! Reads SPIR-V's binary form (the specification's "Physical Layout of a SPIR-V Module"): a
//! five-word header, then instructions whose first word holds the word count in its high half and
//! the opcode in its low half, each followed by the operands its grammar form lists.

use super::grammar::{form, Count, Kind};
use super::instruction::{Instruction, Module, Operand};
use crate::spv::{Op, Word, MAGIC_NUMBER};
use std::collections::HashMap;
use std::fmt;

/// Why a module did not parse: the problem, and the index of the word where it was found (the
/// first word of the instruction that has it).
pub(crate) struct Error {
    word: usize,
    problem: Problem,
}

enum Problem {
    /// The bytes are not a whole number of words.
    PartialWord,
    /// Fewer words than the header's five, or a first word that is not the magic number.
    Header,
    /// A word count of zero.
    EmptyInstruction,
    /// A word count past the module's end.
    Truncated,
    /// An opcode the grammar does not define.
    Opcode(u32),
    /// The instruction ends before an operand its form requires.
    MissingOperand,
    /// Words after the instruction's last operand.
    ExtraWords,
    /// A value, or mask bits, the operand's kind does not define.
    Enumerant(Word),
    /// A literal string with no terminating nul, or not UTF-8.
    String,
    /// A literal number whose type's width is unknown or not 32 bits or fewer, or 64.
    Width,
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SPIR-V word {}: ", self.word)?;
        match self.problem {
            Problem::PartialWord => write!(f, "the bytes end inside a word"),
            Problem::Header => write!(f, "no SPIR-V header"),
            Problem::EmptyInstruction => write!(f, "an instruction of zero words"),
            Problem::Truncated => write!(f, "an instruction past the module's end"),
            Problem::Opcode(opcode) => write!(f, "unknown opcode {opcode}"),
            Problem::MissingOperand => write!(f, "an operand is missing"),
            Problem::ExtraWords => write!(f, "words after the last operand"),
            Problem::Enumerant(value) => write!(f, "unknown enumerant or mask {value:#x}"),
            Problem::String => write!(f, "a malformed literal string"),
            Problem::Width => write!(f, "a literal number of unknown width"),
        }
    }
}

/// Parses the module `bytes` holds, little-endian.
pub(crate) fn parse_bytes(bytes: &[u8]) -> Result<Module, Error> {
    let words = bytes.chunks(4).map(|chunk| match *chunk {
        [a, b, c, d] => Ok(Word::from_le_bytes([a, b, c, d])),
        _ => Err(Error {
            word: bytes.len() / 4,
            problem: Problem::PartialWord,
        }),
    });
    parse_words(&words.collect::<Result<Vec<_>, _>>()?)
}

fn parse_words(words: &[Word]) -> Result<Module, Error> {
    let bound = match words {
        [magic, _version, _generator, bound, _schema, ..] if *magic == MAGIC_NUMBER => *bound,
        _ => {
            return Err(Error {
                word: 0,
                problem: Problem::Header,
            })
        }
    };
    let mut types = Types::default();
    let mut instructions = Vec::new();
    let mut at = 5;
    while let Some(&first) = words.get(at) {
        let fail = |problem| Error { word: at, problem };
        let count = (first >> 16) as usize;
        if count == 0 {
            return Err(fail(Problem::EmptyInstruction));
        }
        let body = words
            .get(at + 1..at + count)
            .ok_or_else(|| fail(Problem::Truncated))?;
        let opcode = first & 0xffff;
        let form = form(opcode).ok_or_else(|| fail(Problem::Opcode(opcode)))?;
        let mut reader = Reader {
            words: body,
            next: 0,
            types: &types,
            result_type: None,
            result_id: None,
            operands: Vec::new(),
        };
        reader.read_all(form.operands).map_err(fail)?;
        if reader.next != body.len() {
            return Err(fail(Problem::ExtraWords));
        }
        let instruction = Instruction {
            opcode: form.opcode,
            result_type: reader.result_type,
            result_id: reader.result_id,
            operands: reader.operands,
        };
        types.track(&instruction);
        instructions.push(instruction);
        at += count;
    }
    Ok(Module {
        bound,
        instructions,
    })
}

/// What the module has declared so far that sizes a literal number: the width of each scalar type
/// and the type of each value.
#[derive(Default)]
struct Types {
    widths: HashMap<Word, Word>,
    values: HashMap<Word, Word>,
}

impl Types {
    fn track(&mut self, instruction: &Instruction) {
        let Some(id) = instruction.result_id else {
            return;
        };
        if let (Op::TypeInt | Op::TypeFloat, Some(Operand::Word(width))) =
            (instruction.opcode, instruction.operands.first())
        {
            self.widths.insert(id, *width);
        }
        if let Some(ty) = instruction.result_type {
            self.values.insert(id, ty);
        }
    }

    /// The width of the scalar type `ty`, if it is one.
    fn width(&self, ty: Word) -> Option<Word> {
        self.widths.get(&ty).copied()
    }
}

/// One instruction's operand words, read by its form.
struct Reader<'a> {
    words: &'a [Word],
    next: usize,
    types: &'a Types,
    result_type: Option<Word>,
    result_id: Option<Word>,
    operands: Vec<Operand>,
}

impl Reader<'_> {
    fn done(&self) -> bool {
        self.next == self.words.len()
    }

    fn word(&mut self) -> Result<Word, Problem> {
        let word = *self.words.get(self.next).ok_or(Problem::MissingOperand)?;
        self.next += 1;
        Ok(word)
    }

    fn read_all(&mut self, list: &[(Kind, Count)]) -> Result<(), Problem> {
        for &(kind, count) in list {
            match count {
                Count::One => self.operand(kind)?,
                Count::Optional if !self.done() => self.operand(kind)?,
                Count::Optional => {}
                Count::Any => {
                    while !self.done() {
                        self.operand(kind)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn operand(&mut self, kind: Kind) -> Result<(), Problem> {
        match kind {
            Kind::ResultType => self.result_type = Some(self.word()?),
            Kind::ResultId => self.result_id = Some(self.word()?),
            Kind::Id => self.push(Operand::IdRef)?,
            Kind::Scope => self.push(Operand::IdScope)?,
            Kind::Semantics => self.push(Operand::IdMemorySemantics)?,
            Kind::Word => self.push(Operand::Word)?,
            Kind::ExtInstNumber => self.push(Operand::LiteralExtInstInteger)?,
            Kind::Text => {
                let text = self.text()?;
                self.operands.push(Operand::String(text));
            }
            Kind::Number => {
                let width = self.result_type.and_then(|ty| self.types.width(ty));
                self.number(width)?;
            }
            Kind::SwitchTarget => {
                let width = match self.operands.first() {
                    Some(Operand::IdRef(selector)) => self
                        .types
                        .values
                        .get(selector)
                        .and_then(|ty| self.types.width(*ty)),
                    _ => None,
                };
                self.number(width)?;
                self.push(Operand::IdRef)?;
            }
            Kind::IdWord => {
                self.push(Operand::IdRef)?;
                self.push(Operand::Word)?;
            }
            Kind::IdId => {
                self.push(Operand::IdRef)?;
                self.push(Operand::IdRef)?;
            }
            Kind::SpecOp => {
                let opcode = self.word()?;
                let applied = form(opcode).ok_or(Problem::Opcode(opcode))?;
                self.operands.push(Operand::SpecConstantOp(applied.opcode));
                for &(kind, count) in applied.operands {
                    if !matches!(kind, Kind::ResultType | Kind::ResultId) {
                        self.read_all(&[(kind, count)])?;
                    }
                }
            }
            Kind::Value(enumerants) => {
                let value = self.word()?;
                self.operands.push(Operand::Word(value));
                let at = enumerants
                    .binary_search_by_key(&value, |e| e.value)
                    .map_err(|_| Problem::Enumerant(value))?;
                for &parameter in enumerants[at].parameters {
                    self.operand(parameter)?;
                }
            }
            Kind::Mask(bits) => {
                let mask = self.word()?;
                self.operands.push(Operand::Word(mask));
                let known = bits.iter().fold(0, |known, bit| known | bit.value);
                if mask & !known != 0 {
                    return Err(Problem::Enumerant(mask));
                }
                for bit in bits.iter().filter(|bit| mask & bit.value != 0) {
                    for &parameter in bit.parameters {
                        self.operand(parameter)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn push(&mut self, operand: fn(Word) -> Operand) -> Result<(), Problem> {
        let word = self.word()?;
        self.operands.push(operand(word));
        Ok(())
    }

    /// A literal number `width` bits wide: one word up to 32 bits, two (low word first) at 64.
    fn number(&mut self, width: Option<Word>) -> Result<(), Problem> {
        match width {
            Some(1..=32) => self.push(Operand::Word),
            Some(64) => {
                let low = u64::from(self.word()?);
                let high = u64::from(self.word()?);
                self.operands.push(Operand::Wide(high << 32 | low));
                Ok(())
            }
            _ => Err(Problem::Width),
        }
    }

    /// A literal string: its bytes up to the first nul, in as many words as hold them and the nul.
    fn text(&mut self) -> Result<String, Problem> {
        let rest = self.words.get(self.next..).unwrap_or_default();
        let bytes = rest.iter().flat_map(|word| word.to_le_bytes());
        let text = bytes.take_while(|byte| *byte != 0).collect::<Vec<u8>>();
        if text.len() >= rest.len() * 4 {
            return Err(Problem::String);
        }
        self.next += text.len() / 4 + 1;
        String::from_utf8(text).map_err(|_| Problem::String)
    }
}
