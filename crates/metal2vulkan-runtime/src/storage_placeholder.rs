//!   as many layers, in memory of its own where binds wait for queued work, else in a pooled
//!   block (`memory.rs`'s `Fit`). It rests in `GENERAL` like every runtime image: its
        let placement = match crate::memory::allocate_image_while_queued(
