//! A reader of SPIR-V's binary form, for the crate's tests: they read the modules the core
//! serializes back through it, and `pipeline::testing` asks it which instructions have a result.
//! It is driven by the core grammar's instruction table (`table.rs`), which
//! `scripts/regen-spirv-grammar` generates from Khronos's grammar; nothing else here is generated.

mod grammar;
mod instruction;
mod parser;
mod table;

pub(crate) use grammar::instruction_result_shape;
pub(crate) use instruction::{Instruction, Operand};
pub(crate) use parser::parse_bytes;
