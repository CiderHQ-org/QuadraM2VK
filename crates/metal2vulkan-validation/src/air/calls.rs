//! The `air.*` calls of a module by exact ABI symbol, and the command-encoder operations among
//! them: what classification reads of a module's calls (`executor_contract`, `triage`) and what the
//! Metal oracle's native command guard keys on. Copied from the product's developer inventory
//! with the rest of validation's AIR readers, unchanged.

use std::collections::BTreeMap;

/// Count AIR calls in sanitized LLVM IR by exact ABI symbol.
///
/// Declarations are not uses. The scan accepts ordinary, tail, musttail, and notail calls and counts
/// every `@air.*` callee on the instruction line, matching the translator's call-oriented contract.
pub fn air_call_counts(ll: &str) -> BTreeMap<String, usize> {
    let mut calls = BTreeMap::new();
    for line in ll.lines() {
        for symbol in direct_callees(line)
            .into_iter()
            .filter(|symbol| symbol.starts_with("air."))
        {
            *calls.entry(symbol).or_default() += 1;
        }
    }
    calls
}

fn direct_callees(line: &str) -> Vec<String> {
    if ![
        "call ", "call\t", "invoke ", "invoke\t", "callbr ", "callbr\t",
    ]
    .iter()
    .any(|opcode| line.contains(opcode))
    {
        return Vec::new();
    }
    let line = if line.contains(';') {
        strip_llvm_comment(line)
    } else {
        line
    }
    .trim();
    call_opcode_ends(line)
        .into_iter()
        .filter_map(|offset| direct_callee_after(&line[offset..]))
        .collect()
}

fn call_opcode_ends(line: &str) -> Vec<usize> {
    const OPCODES: [&str; 6] = [
        "call ", "call\t", "invoke ", "invoke\t", "callbr ", "callbr\t",
    ];
    let mut offsets = Vec::new();
    let mut cursor = 0usize;
    while cursor < line.len() {
        let Some((start, opcode)) = OPCODES
            .iter()
            .filter_map(|opcode| line[cursor..].find(opcode).map(|at| (cursor + at, *opcode)))
            .min_by_key(|(at, _)| *at)
        else {
            break;
        };
        let before = line[..start].chars().next_back();
        let opcode_context =
            before.is_none_or(|ch| ch.is_whitespace() || matches!(ch, '=' | '{' | '}' | ';'));
        if opcode_context && !quoted_at(line, start) {
            offsets.push(start + opcode.len());
        }
        cursor = start + 1;
    }
    offsets
}

fn quoted_at(line: &str, offset: usize) -> bool {
    if !line[..offset].contains('"') {
        return false;
    }
    let mut quoted = false;
    let mut escaped = false;
    for ch in line[..offset].chars() {
        if quoted && ch == '\\' && !escaped {
            escaped = true;
            continue;
        }
        if ch == '"' && !escaped {
            quoted = !quoted;
        }
        escaped = false;
    }
    quoted
}

fn direct_callee_after(call: &str) -> Option<String> {
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    let mut at = None;
    for (offset, ch) in call.char_indices() {
        if quoted && ch == '\\' && !escaped {
            escaped = true;
            continue;
        }
        if ch == '"' && !escaped {
            quoted = !quoted;
        } else if !quoted {
            match ch {
                '(' => depth += 1,
                ')' => depth = depth.saturating_sub(1),
                '@' if depth == 0 => {
                    at = Some(offset);
                    break;
                }
                _ => {}
            }
        }
        escaped = false;
    }
    global_symbol_after_at(&call[at?..])
}

fn global_symbol_after_at(text: &str) -> Option<String> {
    let at = text.find('@')?;
    let symbol = &text[at + 1..];
    if let Some(quoted) = symbol.strip_prefix('"') {
        return decode_quoted_symbol(quoted);
    }
    let end = symbol
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '$' | '-')))
        .unwrap_or(symbol.len());
    (end > 0).then(|| symbol[..end].to_string())
}

fn decode_quoted_symbol(quoted: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = quoted.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => return Some(out),
            '\\' => {
                let hi = chars.next()?;
                if hi.is_ascii_hexdigit() && chars.peek().is_some_and(char::is_ascii_hexdigit) {
                    let lo = chars.next().expect("peeked hex digit");
                    let byte = ((hi.to_digit(16)? << 4) | lo.to_digit(16)?) as u8;
                    out.push(byte as char);
                } else {
                    out.push(hi);
                }
            }
            _ => out.push(ch),
        }
    }
    None
}

fn strip_llvm_comment(line: &str) -> &str {
    let mut quoted = false;
    let mut escaped = false;
    for (offset, ch) in line.char_indices() {
        if quoted && ch == '\\' && !escaped {
            escaped = true;
            continue;
        }
        if ch == '"' && !escaped {
            quoted = !quoted;
        } else if ch == ';' && !quoted {
            return &line[..offset];
        }
        escaped = false;
    }
    line
}

/// Buffer namespace of an encoded command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferStage {
    Kernel,
    Vertex,
    Fragment,
    Object,
    Mesh,
}

/// Recognized command-encoder operation; recognition does not establish lowering support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandEncoderOperation {
    Buffer {
        stage: BufferStage,
        address_space: u32,
    },
    Dispatch {
        exact_threads: bool,
    },
    Pipeline {
        render: bool,
    },
    Barrier {
        render: bool,
        enabled: bool,
    },
    ThreadgroupMemory,
    Draw,
    /// `air.draw_indexed_primitives_render_command.p<space>i<16|32>`: an indexed draw whose index
    /// buffer lies in `address_space` (1, device; 2, constant) and holds 32-bit indices where
    /// `wide`, else 16-bit ones. The overload names both; no operand does.
    DrawIndexed {
        address_space: u32,
        wide: bool,
    },
    /// `air.draw_patches_render_command.p<space>i32.p<space>i8`: a patch draw whose patch index
    /// buffer (`uint`s; null for none) lies in `patch_indices` and whose tessellation factor buffer
    /// lies in `factors` (1, device; 2, constant). Its last operand is the factor scale Metal
    /// multiplies the factors by.
    DrawPatches {
        patch_indices: u32,
        factors: u32,
    },
    /// `air.draw_indexed_patches_render_command.p<space>i32.p<space>i8.p<space>i8`: a patch draw
    /// as [`DrawPatches`](Self::DrawPatches) is, with a control-point index buffer (null for none)
    /// in `control_point_indices` between the patch index and factor buffers.
    DrawIndexedPatches {
        patch_indices: u32,
        control_point_indices: u32,
        factors: u32,
    },
    /// `air.draw_mesh_threadgroups_render_command` or `air.draw_mesh_threads_render_command`
    /// (`void (cmd, i32, <3 x i32>, <3 x i32>, <3 x i32>)`): a mesh draw of its grid, in
    /// threadgroups, or in threads where `threads`, then the threads of an object threadgroup and
    /// of a mesh threadgroup.
    DrawMesh {
        threads: bool,
    },
    /// `air.reset_compute_command` or `air.reset_render_command` (`void (cmd, i32)`): the command's
    /// every field reset, as `resetWithRange:` resets it.
    Reset {
        render: bool,
    },
    /// `air.set_stage_in_region_compute_command` (`void (cmd, i32, <3 x i32>, <3 x i32>)`): the
    /// command's stage-in region, its origin then its size.
    StageInRegion,
    /// `air.set_imageblock_size_compute_command` (`void (cmd, i32, <2 x i16>)`): the command's
    /// imageblock width and height.
    ImageblockSize,
    /// `air.copy_compute_command` (`void (cmd, i32, cmd, i32)`): the command copied from a source
    /// command buffer's command, as the product's lowering and Metal agree on it (equal
    /// descriptors, or into a buffer that inherits); any other copy faults.
    Copy,
    /// A known family with an unimplemented pointer encoding: a buffer setter's, an indexed
    /// draw's, a patch draw's or an indexed patch draw's overload whose pointers are not each
    /// `p1` or `p2` of their element types.
    UnsupportedBufferEncoding,
    /// Any other operation of the AIR command-encoder families (`air.*_compute_command`,
    /// `air.*_render_command`): render state setters and an object threadgroup's memory length,
    /// among others. Neither executor models them yet.
    Unmodeled,
}

impl CommandEncoderOperation {
    pub fn from_symbol(symbol: &str) -> Option<Self> {
        match symbol {
            "air.concurrent_dispatch_threadgroups_compute_command" => {
                return Some(Self::Dispatch {
                    exact_threads: false,
                })
            }
            "air.concurrent_dispatch_threads_compute_command" => {
                return Some(Self::Dispatch {
                    exact_threads: true,
                })
            }
            "air.set_pipeline_state_compute_command" => {
                return Some(Self::Pipeline { render: false })
            }
            "air.set_pipeline_state_render_command" => {
                return Some(Self::Pipeline { render: true })
            }
            "air.set_barrier_compute_command" => {
                return Some(Self::Barrier {
                    render: false,
                    enabled: true,
                })
            }
            "air.clear_barrier_compute_command" => {
                return Some(Self::Barrier {
                    render: false,
                    enabled: false,
                })
            }
            "air.set_barrier_render_command" => {
                return Some(Self::Barrier {
                    render: true,
                    enabled: true,
                })
            }
            "air.clear_barrier_render_command" => {
                return Some(Self::Barrier {
                    render: true,
                    enabled: false,
                })
            }
            "air.set_threadgroup_memory_length_compute_command" => {
                return Some(Self::ThreadgroupMemory)
            }
            "air.draw_primitives_render_command" => return Some(Self::Draw),
            "air.draw_mesh_threadgroups_render_command" => {
                return Some(Self::DrawMesh { threads: false })
            }
            "air.draw_mesh_threads_render_command" => {
                return Some(Self::DrawMesh { threads: true })
            }
            "air.reset_compute_command" => return Some(Self::Reset { render: false }),
            "air.reset_render_command" => return Some(Self::Reset { render: true }),
            "air.set_stage_in_region_compute_command" => return Some(Self::StageInRegion),
            "air.set_imageblock_size_compute_command" => return Some(Self::ImageblockSize),
            "air.copy_compute_command" => return Some(Self::Copy),
            _ => {}
        }
        let rest = symbol.strip_prefix("air.")?;
        let (family, suffix) = rest
            .split_once('.')
            .map_or((rest, None), |(family, suffix)| (family, Some(suffix)));
        if family == "draw_indexed_primitives_render_command" {
            let (address_space, wide) = match suffix {
                Some("p1i16") => (1, false),
                Some("p1i32") => (1, true),
                Some("p2i16") => (2, false),
                Some("p2i32") => (2, true),
                _ => return Some(Self::UnsupportedBufferEncoding),
            };
            return Some(Self::DrawIndexed {
                address_space,
                wide,
            });
        }
        if family == "draw_patches_render_command" {
            let space = |suffix: &str, element: &str| match suffix.strip_suffix(element) {
                Some("p1") => Some(1),
                Some("p2") => Some(2),
                _ => None,
            };
            let spaces =
                suffix
                    .and_then(|suffix| suffix.split_once('.'))
                    .and_then(|(indices, factors)| {
                        Some((space(indices, "i32")?, space(factors, "i8")?))
                    });
            return Some(match spaces {
                Some((patch_indices, factors)) => Self::DrawPatches {
                    patch_indices,
                    factors,
                },
                None => Self::UnsupportedBufferEncoding,
            });
        }
        if family == "draw_indexed_patches_render_command" {
            let space = |suffix: &str, element: &str| match suffix.strip_suffix(element) {
                Some("p1") => Some(1),
                Some("p2") => Some(2),
                _ => None,
            };
            let tokens = suffix.map(|suffix| suffix.split('.').collect::<Vec<_>>());
            let spaces = match tokens.as_deref() {
                Some(&[indices, points, factors]) => space(indices, "i32")
                    .zip(space(points, "i8"))
                    .zip(space(factors, "i8"))
                    .map(|((indices, points), factors)| (indices, points, factors)),
                _ => None,
            };
            return Some(match spaces {
                Some((patch_indices, control_point_indices, factors)) => Self::DrawIndexedPatches {
                    patch_indices,
                    control_point_indices,
                    factors,
                },
                None => Self::UnsupportedBufferEncoding,
            });
        }
        let stage = match family {
            "set_kernel_buffer_compute_command" => BufferStage::Kernel,
            "set_vertex_buffer_render_command" => BufferStage::Vertex,
            "set_fragment_buffer_render_command" => BufferStage::Fragment,
            "set_object_buffer_render_command" => BufferStage::Object,
            "set_mesh_buffer_render_command" => BufferStage::Mesh,
            _ if family.ends_with("_compute_command") || family.ends_with("_render_command") => {
                return Some(Self::Unmodeled)
            }
            _ => return None,
        };
        let address_space = match suffix {
            Some("p1i8") => 1,
            Some("p2i8") => 2,
            _ => return Some(Self::UnsupportedBufferEncoding),
        };
        Some(Self::Buffer {
            stage,
            address_space,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reset of either family, a stage-in region, an imageblock size and a copy are their own
    /// operations.
    #[test]
    fn resets_regions_and_imageblock_sizes_decode() {
        assert_eq!(
            CommandEncoderOperation::from_symbol("air.copy_compute_command"),
            Some(CommandEncoderOperation::Copy)
        );
        assert_eq!(
            CommandEncoderOperation::from_symbol("air.set_stage_in_region_compute_command"),
            Some(CommandEncoderOperation::StageInRegion)
        );
        assert_eq!(
            CommandEncoderOperation::from_symbol("air.set_imageblock_size_compute_command"),
            Some(CommandEncoderOperation::ImageblockSize)
        );
        assert_eq!(
            CommandEncoderOperation::from_symbol("air.reset_compute_command"),
            Some(CommandEncoderOperation::Reset { render: false })
        );
        assert_eq!(
            CommandEncoderOperation::from_symbol("air.reset_render_command"),
            Some(CommandEncoderOperation::Reset { render: true })
        );
    }

    /// A patch draw's overload names its patch index buffer's space and its factor buffer's; any
    /// other overload is an encoding no executor implements.
    #[test]
    fn patch_draws_decode_their_buffers_from_the_overload() {
        for (suffix, patch_indices, factors) in [
            ("p1i32.p1i8", 1, 1),
            ("p1i32.p2i8", 1, 2),
            ("p2i32.p1i8", 2, 1),
            ("p2i32.p2i8", 2, 2),
        ] {
            assert_eq!(
                CommandEncoderOperation::from_symbol(&format!(
                    "air.draw_patches_render_command.{suffix}"
                )),
                Some(CommandEncoderOperation::DrawPatches {
                    patch_indices,
                    factors
                })
            );
        }
        for symbol in [
            "air.draw_patches_render_command",
            "air.draw_patches_render_command.p1i32",
            "air.draw_patches_render_command.p3i32.p1i8",
            "air.draw_patches_render_command.p1i16.p1i8",
            "air.draw_patches_render_command.p1i32.p1i32",
        ] {
            assert_eq!(
                CommandEncoderOperation::from_symbol(symbol),
                Some(CommandEncoderOperation::UnsupportedBufferEncoding),
                "{symbol}"
            );
        }
    }

    /// An indexed patch draw's overload names its patch index, control-point index and factor
    /// buffers' spaces, in that order; any other overload is an encoding
    /// no executor implements.
    #[test]
    fn indexed_patch_draws_decode_their_buffers_from_the_overload() {
        for (suffix, patch_indices, control_point_indices, factors) in [
            ("p1i32.p1i8.p1i8", 1, 1, 1),
            ("p2i32.p2i8.p2i8", 2, 2, 2),
            ("p1i32.p2i8.p1i8", 1, 2, 1),
            ("p2i32.p1i8.p2i8", 2, 1, 2),
            ("p1i32.p1i8.p2i8", 1, 1, 2),
        ] {
            assert_eq!(
                CommandEncoderOperation::from_symbol(&format!(
                    "air.draw_indexed_patches_render_command.{suffix}"
                )),
                Some(CommandEncoderOperation::DrawIndexedPatches {
                    patch_indices,
                    control_point_indices,
                    factors
                })
            );
        }
        for symbol in [
            "air.draw_indexed_patches_render_command",
            "air.draw_indexed_patches_render_command.p1i32.p1i8",
            "air.draw_indexed_patches_render_command.p1i32.p1i8.p1i8.p1i8",
            "air.draw_indexed_patches_render_command.p3i32.p1i8.p1i8",
            "air.draw_indexed_patches_render_command.p1i16.p1i8.p1i8",
            "air.draw_indexed_patches_render_command.p1i32.p1i16.p1i8",
            "air.draw_indexed_patches_render_command.p1i32.p1i8.p3i8",
        ] {
            assert_eq!(
                CommandEncoderOperation::from_symbol(symbol),
                Some(CommandEncoderOperation::UnsupportedBufferEncoding),
                "{symbol}"
            );
        }
    }

    /// An indexed draw's overload names its index buffer's space and index type; any other overload
    /// is an encoding no executor implements.
    #[test]
    fn indexed_draws_decode_their_index_buffer_from_the_overload() {
        for (suffix, address_space, wide) in [
            ("p1i16", 1, false),
            ("p1i32", 1, true),
            ("p2i16", 2, false),
            ("p2i32", 2, true),
        ] {
            assert_eq!(
                CommandEncoderOperation::from_symbol(&format!(
                    "air.draw_indexed_primitives_render_command.{suffix}"
                )),
                Some(CommandEncoderOperation::DrawIndexed {
                    address_space,
                    wide
                })
            );
        }
        for symbol in [
            "air.draw_indexed_primitives_render_command",
            "air.draw_indexed_primitives_render_command.p3i16",
            "air.draw_indexed_primitives_render_command.p1i8",
        ] {
            assert_eq!(
                CommandEncoderOperation::from_symbol(symbol),
                Some(CommandEncoderOperation::UnsupportedBufferEncoding)
            );
        }
    }

    /// A mesh draw of either grid form, and an object or a mesh buffer of either space, are their
    /// own operations; an object threadgroup's memory length is one neither executor models.
    #[test]
    fn mesh_draws_and_buffers_decode() {
        for (symbol, threads) in [
            ("air.draw_mesh_threadgroups_render_command", false),
            ("air.draw_mesh_threads_render_command", true),
        ] {
            assert_eq!(
                CommandEncoderOperation::from_symbol(symbol),
                Some(CommandEncoderOperation::DrawMesh { threads })
            );
        }
        for (family, stage) in [
            ("set_object_buffer_render_command", BufferStage::Object),
            ("set_mesh_buffer_render_command", BufferStage::Mesh),
        ] {
            for (suffix, address_space) in [("p1i8", 1), ("p2i8", 2)] {
                assert_eq!(
                    CommandEncoderOperation::from_symbol(&format!("air.{family}.{suffix}")),
                    Some(CommandEncoderOperation::Buffer {
                        stage,
                        address_space
                    })
                );
            }
        }
        assert_eq!(
            CommandEncoderOperation::from_symbol(
                "air.set_object_threadgroup_memory_length_render_command"
            ),
            Some(CommandEncoderOperation::Unmodeled)
        );
    }

    #[test]
    fn call_inventory_ignores_non_callees_and_counts_all_call_forms() {
        let ll = "define void @main() {\nentry:\n %a = call i32 @air.abs.s.i32(i32 -1)\n %b = tail call i32 @air.future_tensor.i32(i32 1)\n %c = musttail\tcall i32 @air.future_tensor.i32(i32 2)\n %d = invoke i32 @\"air.quoted_future.i32\"(i32 3) to label %ok unwind label %err\n callbr void @air.branch_future() to label %ok [label %err]\n call void @ordinary(ptr @air.declaration_only.i32) ; @air.comment_only\n ret void\n}\ndefine void @compact() { %x = call i32 @air.compact_future(i32 0) ret void }\ndeclare i32 @air.declaration_only.i32(i32)";
        assert_eq!(
            air_call_counts(ll),
            BTreeMap::from([
                ("air.abs.s.i32".into(), 1),
                ("air.branch_future".into(), 1),
                ("air.compact_future".into(), 1),
                ("air.future_tensor.i32".into(), 2),
                ("air.quoted_future.i32".into(), 1),
            ])
        );
    }
}
