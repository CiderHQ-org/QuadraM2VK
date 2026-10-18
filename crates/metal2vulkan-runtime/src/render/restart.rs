//! Primitive restart at a chosen index: Apple's `setPrimitiveRestartEnabled:index:`.
//!
//! Measured on an M4 with macOS 27.0.1, with and without API validation, for a call with index `k`:
//!
//! - An indexed line or triangle strip drawn after the call restarts at `k'` alone, and the index
//!   type's all-ones index draws as a vertex unless it is `k'`. `enabled` changes nothing.
//! - `k'` is `k mod 2^32` for 32-bit indices, and the lesser of `k mod 2^32` and `0xffff` for
//!   16-bit ones.
//! - `k'` is compared with the index as the buffer holds it, before the base vertex.
//! - A triangle's winding parity restarts with each piece, and `primitive_id` counts the drawn
//!   primitives on across a restart, as it does across the all-ones restart.
//! - Lists and points draw every index as a vertex, and a draw of no indices has none to restart
//!   at: the call changes none of them.
//! - An indexed indirect draw follows the call; an execution of indirect render commands does
//!   not, its strips restarting at the all-ones index.
//! - The call applies to the encoder's draws after it, whether before or after its pipeline
//!   state is set; the last call wins; a draw before it keeps the all-ones index; a new encoder
//!   restarts at the all-ones index.
//!
//! Where `k'` is the index type's all-ones index, Vulkan's `primitiveRestartEnable` restarts
//! there as Metal does, and the draw stays as it is. Elsewhere it is drawn as the list the index
//! rewrite pass writes (`index_rewrite.rs`), with the class's list topology.

use super::RenderCall;
use super::{Assembly, RenderCommandEncoder};
use metal2vulkan::bind::{DrawCall, RewriteProgram};
use metal2vulkan::metal::PrimitiveType;

/// A draw the index rewrite pass draws as a list: its program, `k'`, and its index size.
#[derive(Clone, Copy)]
pub(super) struct Restarted {
    pub(super) program: RewriteProgram,
    pub(super) restart: u32,
    pub(super) index_size: u64,
}

impl Restarted {
    /// The list a draw of the program draws: of lines where its primitives have two vertices,
    /// else of triangles.
    pub(super) fn assembly(self) -> Assembly {
        Assembly::Primitives(match self.program.vertices() {
            2 => PrimitiveType::Line,
            _ => PrimitiveType::Triangle,
        })
    }
}

impl RenderCommandEncoder {
    /// `setPrimitiveRestartEnabled:index:` (SPI): the encoder's indexed line and triangle strips
    /// drawn after the call restart at `index` as Metal takes it for their index type (`k mod
    /// 2^32` for 32-bit indices, the lesser of that and `0xffff` for 16-bit ones), and only
    /// there: the all-ones index then draws as a vertex unless it is that index. `_enabled`
    /// changes nothing, as on Metal. Lists, points, draws without indices and executions of
    /// indirect render commands are as before; the last call wins until the encoder ends (an M4). A
    /// strip restarting elsewhere than at the all-ones index is drawn as a list a GPU pass writes
    /// before the pass; a draw it cannot draw so is refused (`NotImplemented(RestartedStrip)`).
    pub fn set_primitive_restart_enabled(&mut self, _enabled: bool, index: u64) {
        self.log(|_| RenderCall::PrimitiveRestartEnabled(_enabled, index));
        self.restart = Some(index);
    }

    /// Where the encoder's call restarts `draw` of `assembly` at another index than Vulkan
    /// would: an indexed strip, with `k'` not its index type's all-ones index. `None` for every
    /// other draw.
    pub(super) fn restarted(&self, assembly: Assembly, draw: &DrawCall) -> Option<Restarted> {
        let k = self.restart?;
        let program = match assembly {
            Assembly::Primitives(PrimitiveType::LineStrip) => RewriteProgram::LineStrip,
            Assembly::Primitives(PrimitiveType::TriangleStrip) => RewriteProgram::TriangleStrip,
            _ => return None,
        };
        let index_type = match *draw {
            DrawCall::Indexed { index_type, .. } => index_type,
            DrawCall::Indirect {
                indexed: Some((index_type, ..)),
                ..
            } => index_type,
            _ => return None,
        };
        let wide = index_type == metal2vulkan::vk::IndexType::UINT32;
        let (restart, all_ones) = restart_index(k, wide);
        (restart != all_ones).then_some(Restarted {
            program,
            restart,
            index_size: if wide { 4 } else { 2 },
        })
    }
}

/// `k'` for a call with `k` over 32-bit indices (`wide`) or 16-bit ones, and the index type's
/// all-ones index.
fn restart_index(k: u64, wide: bool) -> (u32, u32) {
    // `k mod 2^32`.
    let low = k as u32;
    match wide {
        true => (low, u32::MAX),
        false => (low.min(0xffff), 0xffff),
    }
}

#[cfg(test)]
mod tests {
    use super::restart_index;

    #[test]
    fn the_restart_index_is_metal_s_measured_one() {
        // 32-bit indices take `k mod 2^32`, 16-bit ones the lesser of that and 0xffff.
        let wide = [
            (3, 3),
            (0xffff, 0xffff),
            (0x1_0000_0003, 3),
            (u64::MAX, u32::MAX),
        ];
        for (k, expected) in wide {
            assert_eq!(restart_index(k, true), (expected, u32::MAX), "{k:#x}");
        }
        let narrow = [
            (3, 3),
            (0xfffe, 0xfffe),
            (0x1_0000, 0xffff),
            (0xffff_ffff, 0xffff),
            (0x1_0000_0000, 0),
            (0x1_0000_0003, 3),
            (0x1_0000_00ff, 0xff),
            (0x1_0000_ffff, 0xffff),
            (0x2_0001_0003, 0xffff),
            (u64::MAX, 0xffff),
        ];
        for (k, expected) in narrow {
            assert_eq!(restart_index(k, false), (expected, 0xffff), "{k:#x}");
        }
    }
}
