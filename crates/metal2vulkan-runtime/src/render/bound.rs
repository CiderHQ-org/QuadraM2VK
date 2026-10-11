//! What a draw's secondary command buffer has bound, so a draw leaves out a state command that
//! sets what it already holds.
//!
//! A secondary command buffer begins with nothing bound and inherits nothing from the primary.
//! [`Bound`] follows what the draws recorded into one set there, under Vulkan's rules for what a
//! later command keeps or disturbs:
//!
//! - **The pipeline.** Binding the same `VkPipeline` again is left out. Binding another one
//!   invalidates every dynamic state it does not hold dynamic (and, where it does not make the
//!   vertex strides dynamic, the strides the vertex buffers were bound with), so those are set
//!   again by the next pipeline that holds them. A pipeline holds the dynamic states of the parts
//!   it has: one that discards rasterization has no fragment shader or fragment output state, so
//!   its bind leaves the depth, stencil and blend-constant states unknown too, whatever its
//!   `pDynamicStates` lists, and the next pipeline that holds them sets them again (the draw passes
//!   the states its pipeline holds, `render::fragment_state`).
//! - **Descriptor sets and push constants.** Kept while they are bound or pushed with the same
//!   `VkPipelineLayout` (the same layout is compatible for every set and for the push constants);
//!   a bind or push with another layout forgets what the earlier one held. A draw's push
//!   constants are left out only where they are the same writes, in order, as the last ones.
//! - **Vertex and index buffers.** Each binding is kept until bound again; binding a pipeline
//!   disturbs none.
//! - **Dynamic states.** Each is kept, at the value it was set to, until a pipeline that makes it
//!   static is bound. Values compare as the commands carry them: floats bit for bit, so a zero
//!   of the other sign is set again.
//!
//! Only draws record these commands into the command buffer; its other commands (barriers and
//! queries) disturb none of them. A record follows one command buffer: a draw into another
//! begins it again, empty.

use ash::vk;

use super::{DrawState, IndexBind};
use crate::binding::PushWrite;
use crate::pipeline::FixedState;
use crate::render_state::DepthBiasSetting;

/// The state commands recorded into one secondary command buffer that still hold (the module
/// docs).
#[derive(Default)]
pub(super) struct Bound {
    /// The command buffer: null before its first draw.
    command_buffer: vk::CommandBuffer,
    /// The bound pipeline: null before the first.
    pipeline: vk::Pipeline,
    /// The layout the descriptor sets below were bound with, and each set number's set.
    sets_layout: vk::PipelineLayout,
    sets: Vec<Option<vk::DescriptorSet>>,
    /// The layout the last push constants were pushed with, and their writes: each write's
    /// stages, offset and length, its bytes one after another in `push_bytes`.
    push_layout: vk::PipelineLayout,
    pushes: Vec<(vk::ShaderStageFlags, u32, usize)>,
    push_bytes: Vec<u8>,
    /// Each vertex buffer binding: its buffer, offset and, bound with the strides dynamic, its
    /// stride.
    vertex: Vec<(u32, vk::Buffer, u64, Option<u64>)>,
    /// The index buffer: its buffer, offset, index type and, bound with a size, the size.
    index: Option<(vk::Buffer, u64, vk::IndexType, Option<u64>)>,
    /// The dynamic states set that still hold, and their values.
    set: Vec<vk::DynamicState>,
    values: Option<Values>,
}

/// The values a draw sets its dynamic states to: the encoder's state at the draw and its input
/// assembly.
pub(super) struct Values {
    viewports: Vec<vk::Viewport>,
    scissors: Vec<vk::Rect2D>,
    fixed: FixedState,
    stencil_reference: [u32; 2],
    depth_bias: DepthBiasSetting,
    blend_color: [f32; 4],
    topology: metal2vulkan::vk::PrimitiveTopology,
    primitive_restart: bool,
}

/// A draw's dynamic state values, borrowed: the encoder's state and the input assembly.
#[derive(Clone, Copy)]
pub(super) struct Now<'a> {
    pub(super) state: &'a DrawState,
    pub(super) topology: metal2vulkan::vk::PrimitiveTopology,
    pub(super) primitive_restart: bool,
}

impl Bound {
    /// Follow `command_buffer`: where the record is another's, it begins again, empty.
    pub(super) fn follow(&mut self, command_buffer: vk::CommandBuffer) {
        if self.command_buffer != command_buffer {
            *self = Bound {
                command_buffer,
                ..Bound::default()
            };
        }
    }

    /// Whether binding `pipeline`, which holds the dynamic states `dynamic` (the module docs),
    /// sets anything: noted as bound where it does.
    pub(super) fn pipeline(
        &mut self,
        pipeline: vk::Pipeline,
        dynamic: &[vk::DynamicState],
    ) -> bool {
        if self.pipeline == pipeline {
            return false;
        }
        self.pipeline = pipeline;
        self.set.retain(|state| dynamic.contains(state));
        if !dynamic.contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE) {
            self.vertex.retain(|&(.., stride)| stride.is_none());
        }
        true
    }

    /// Whether binding `sets` from set number `first` with `layout` sets anything: noted as
    /// bound where it does.
    pub(super) fn sets(
        &mut self,
        layout: vk::PipelineLayout,
        first: u32,
        sets: &[vk::DescriptorSet],
    ) -> bool {
        let first = first as usize;
        let held = |at: usize| self.sets.get(at).copied().flatten();
        if self.sets_layout == layout
            && sets
                .iter()
                .enumerate()
                .all(|(i, &set)| held(first + i) == Some(set))
        {
            return false;
        }
        if self.sets_layout != layout {
            self.sets_layout = layout;
            self.sets.clear();
        }
        if self.sets.len() < first + sets.len() {
            self.sets.resize(first + sets.len(), None);
        }
        for (i, &set) in sets.iter().enumerate() {
            self.sets[first + i] = Some(set);
        }
        true
    }

    /// Whether pushing `writes` with `layout` sets anything: noted as pushed where it does.
    pub(super) fn pushes(&mut self, layout: vk::PipelineLayout, writes: &[PushWrite]) -> bool {
        let same = self.push_layout == layout
            && self.pushes.len() == writes.len()
            && self
                .pushes
                .iter()
                .zip(writes)
                .all(|(&(stages, offset, length), write)| {
                    (stages, offset, length) == (write.0, write.1, write.2.len())
                })
            && writes
                .iter()
                .flat_map(|write| write.2.iter())
                .eq(self.push_bytes.iter());
        if same {
            return false;
        }
        self.push_layout = layout;
        self.pushes.clear();
        self.push_bytes.clear();
        for (stages, offset, bytes) in writes {
            self.pushes.push((*stages, *offset, bytes.len()));
            self.push_bytes.extend_from_slice(bytes);
        }
        true
    }

    /// Whether binding `buffer` at `offset` to vertex buffer `binding` (with `stride`, where the
    /// strides are dynamic) sets anything: noted as bound where it does.
    pub(super) fn vertex_buffer(
        &mut self,
        binding: u32,
        buffer: vk::Buffer,
        offset: u64,
        stride: Option<u64>,
    ) -> bool {
        let bound = (binding, buffer, offset, stride);
        match self.vertex.iter_mut().find(|held| held.0 == binding) {
            Some(held) if *held == bound => false,
            Some(held) => {
                *held = bound;
                true
            }
            None => {
                self.vertex.push(bound);
                true
            }
        }
    }

    /// Whether binding `index` sets anything: noted as bound where it does.
    pub(super) fn index_buffer(&mut self, index: &IndexBind) -> bool {
        let bound = (
            index.buffer,
            index.offset,
            index.index_type,
            index.sized.map(|(_, size)| size),
        );
        if self.index == Some(bound) {
            return false;
        }
        self.index = Some(bound);
        true
    }

    /// Whether setting dynamic `state` to its value `now` sets anything: noted as set where it
    /// does. A state no value here holds is always set.
    pub(super) fn dynamic(&mut self, state: vk::DynamicState, now: Now<'_>) -> bool {
        let values = self.values.get_or_insert_with(|| Values::of(now));
        let held = self.set.contains(&state);
        if held && values.holds(state, now) == Some(true) {
            return false;
        }
        if values.take(state, now) && !held {
            self.set.push(state);
        }
        true
    }
}

impl Values {
    fn of(now: Now<'_>) -> Values {
        let s = now.state;
        Values {
            viewports: s.viewports.clone(),
            scissors: s.scissors.clone(),
            fixed: s.fixed,
            stencil_reference: s.stencil_reference,
            depth_bias: s.depth_bias.setting(),
            blend_color: s.blend_color,
            topology: now.topology,
            primitive_restart: now.primitive_restart,
        }
    }

    /// Whether `state` holds the value `now` sets it to; `None` for a state no value here holds.
    fn holds(&self, state: vk::DynamicState, now: Now<'_>) -> Option<bool> {
        let s = now.state;
        let (ds, now_ds) = (&self.fixed.depth_stencil, &s.fixed.depth_stencil);
        let faces = |f: fn(&metal2vulkan::metal::state::StencilOpState) -> u32| {
            [f(&ds.front), f(&ds.back)] == [f(&now_ds.front), f(&now_ds.back)]
        };
        Some(match state {
            vk::DynamicState::VIEWPORT => {
                self.viewports.len() == s.viewports.len()
                    && self
                        .viewports
                        .iter()
                        .zip(&s.viewports)
                        .all(|(a, b)| viewport_bits(a) == viewport_bits(b))
            }
            vk::DynamicState::SCISSOR => {
                self.scissors.len() == s.scissors.len()
                    && self
                        .scissors
                        .iter()
                        .zip(&s.scissors)
                        .all(|(a, b)| rect(a) == rect(b))
            }
            vk::DynamicState::DEPTH_BIAS => self.depth_bias == s.depth_bias.setting(),
            vk::DynamicState::BLEND_CONSTANTS => {
                self.blend_color.map(f32::to_bits) == s.blend_color.map(f32::to_bits)
            }
            vk::DynamicState::STENCIL_COMPARE_MASK => faces(|face| face.compare_mask),
            vk::DynamicState::STENCIL_WRITE_MASK => faces(|face| face.write_mask),
            vk::DynamicState::STENCIL_REFERENCE => self.stencil_reference == s.stencil_reference,
            vk::DynamicState::CULL_MODE => self.fixed.cull_mode == s.fixed.cull_mode,
            vk::DynamicState::FRONT_FACE => self.fixed.front_face == s.fixed.front_face,
            vk::DynamicState::PRIMITIVE_TOPOLOGY => self.topology == now.topology,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE => {
                self.primitive_restart == now.primitive_restart
            }
            vk::DynamicState::DEPTH_TEST_ENABLE => ds.depth_test_enable == now_ds.depth_test_enable,
            vk::DynamicState::DEPTH_WRITE_ENABLE => {
                ds.depth_write_enable == now_ds.depth_write_enable
            }
            vk::DynamicState::DEPTH_COMPARE_OP => ds.depth_compare_op == now_ds.depth_compare_op,
            vk::DynamicState::STENCIL_TEST_ENABLE => {
                ds.stencil_test_enable == now_ds.stencil_test_enable
            }
            vk::DynamicState::STENCIL_OP => {
                let ops = |face: &metal2vulkan::metal::state::StencilOpState| {
                    (
                        face.fail_op,
                        face.pass_op,
                        face.depth_fail_op,
                        face.compare_op,
                    )
                };
                [ops(&ds.front), ops(&ds.back)] == [ops(&now_ds.front), ops(&now_ds.back)]
            }
            _ => return None,
        })
    }

    /// Take `state`'s value from `now`; `false` for a state no value here holds.
    fn take(&mut self, state: vk::DynamicState, now: Now<'_>) -> bool {
        let s = now.state;
        let (ds, now_ds) = (&mut self.fixed.depth_stencil, &s.fixed.depth_stencil);
        match state {
            vk::DynamicState::VIEWPORT => {
                self.viewports.clear();
                self.viewports.extend_from_slice(&s.viewports);
            }
            vk::DynamicState::SCISSOR => {
                self.scissors.clear();
                self.scissors.extend_from_slice(&s.scissors);
            }
            vk::DynamicState::DEPTH_BIAS => self.depth_bias = s.depth_bias.setting(),
            vk::DynamicState::BLEND_CONSTANTS => self.blend_color = s.blend_color,
            vk::DynamicState::STENCIL_COMPARE_MASK => {
                ds.front.compare_mask = now_ds.front.compare_mask;
                ds.back.compare_mask = now_ds.back.compare_mask;
            }
            vk::DynamicState::STENCIL_WRITE_MASK => {
                ds.front.write_mask = now_ds.front.write_mask;
                ds.back.write_mask = now_ds.back.write_mask;
            }
            vk::DynamicState::STENCIL_REFERENCE => self.stencil_reference = s.stencil_reference,
            vk::DynamicState::CULL_MODE => self.fixed.cull_mode = s.fixed.cull_mode,
            vk::DynamicState::FRONT_FACE => self.fixed.front_face = s.fixed.front_face,
            vk::DynamicState::PRIMITIVE_TOPOLOGY => self.topology = now.topology,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE => {
                self.primitive_restart = now.primitive_restart
            }
            vk::DynamicState::DEPTH_TEST_ENABLE => ds.depth_test_enable = now_ds.depth_test_enable,
            vk::DynamicState::DEPTH_WRITE_ENABLE => {
                ds.depth_write_enable = now_ds.depth_write_enable
            }
            vk::DynamicState::DEPTH_COMPARE_OP => ds.depth_compare_op = now_ds.depth_compare_op,
            vk::DynamicState::STENCIL_TEST_ENABLE => {
                ds.stencil_test_enable = now_ds.stencil_test_enable
            }
            vk::DynamicState::STENCIL_OP => {
                for (face, now_face) in
                    [(&mut ds.front, &now_ds.front), (&mut ds.back, &now_ds.back)]
                {
                    face.fail_op = now_face.fail_op;
                    face.pass_op = now_face.pass_op;
                    face.depth_fail_op = now_face.depth_fail_op;
                    face.compare_op = now_face.compare_op;
                }
            }
            _ => return false,
        }
        true
    }
}

/// A viewport's fields as the command carries them, bit for bit.
fn viewport_bits(viewport: &vk::Viewport) -> [u32; 6] {
    [
        viewport.x,
        viewport.y,
        viewport.width,
        viewport.height,
        viewport.min_depth,
        viewport.max_depth,
    ]
    .map(f32::to_bits)
}

/// A scissor's fields.
fn rect(rect: &vk::Rect2D) -> [i64; 4] {
    [
        i64::from(rect.offset.x),
        i64::from(rect.offset.y),
        i64::from(rect.extent.width),
        i64::from(rect.extent.height),
    ]
}
