//! Visible-function tables: the words a table's buffer holds, which devirtualization reads.
//!
//! A `visible_function_table<T>` is bound at a Metal buffer index (`setVisibleFunctionTable:
//! atBufferIndex:`). Canonicalization reads it as a read-only buffer of little-endian `u32` words
//! at that index:
//!
//! | word | meaning |
//! |---|---|
//! | [`PRESENT_WORD`] (0) | 1 for a bound table; 0 in a zero-filled neutral buffer, so an unbound table is null (`is_null_visible_function_table`) |
//! | [`SIZE_WORD`] (1) | the table's size (`MTLVisibleFunctionTableDescriptor.functionCount`, `size()`) |
//! | [`FIRST_ENTRY_WORD`] + i | entry `i`: 0 for a null entry, else the handle word of the function it holds |
//!
//! A call through entry `i` becomes a switch on that word over the functions the stage links
//! (`LinkedName::handle` of each [`LinkedModule`](super::LinkedModule)'s functions): each case
//! calls one of them, inlined. A null entry, a word no linked function has, or a function of
//! another type than the call's reaches the switch's default, which calls nothing and yields zero.
//! This layout is the translator's own: the translator numbers the handles, and its binder writes
//! a table's words in this layout.

/// Word 0: whether a table is bound.
pub const PRESENT_WORD: u32 = 0;
/// Word 1: the table's size.
pub const SIZE_WORD: u32 = 1;
/// The word of entry 0; entry `i` is word `FIRST_ENTRY_WORD + i`.
pub const FIRST_ENTRY_WORD: u32 = 2;
/// The first word a function can have: a null entry's is 0, and Metal reserves a function
/// pointer's value 1 (the translator's `FunctionHandle` numbers the pipeline's functions from
/// here). A function no table holds whose address is taken has a word of the stage's own past
/// every handle, from here when the pipeline numbers none.
pub const FIRST_FUNCTION_WORD: u32 = 2;
