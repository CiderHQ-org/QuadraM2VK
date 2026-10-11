//! The texture slots of a render pipeline state's functions, and how each function accesses them
//! ([`RenderPipelineState::texture_access`](crate::RenderPipelineState::texture_access)). The
//! answers come from the pipeline's reflection, which the `reflection:` out-parameter of
//! Metal's creation calls gives too. They are read from the descriptor the state was made from
//! the first time someone asks, and kept while the state lives.

use crate::device::device_error;
use crate::error::Error;
use metal2vulkan::metal::{BindingAccess, BindingType, RenderStages};
use metal2vulkan::reflection::{Binding, RenderPipelineReflection};
use metal2vulkan::Budget;
use std::ops::Range;
use std::sync::OnceLock;

/// The texture arguments a render pipeline state's functions reference. Each entry has its
/// stage, the slots it covers (each element's slot, for an array) and its declared access.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TextureAccesses(Vec<(RenderStages, Range<u64>, BindingAccess)>);

/// A state's [`TextureAccesses`], read when first asked. A refused reflection is kept as its
/// error.
pub(crate) type TextureAccessMemo = OnceLock<Result<TextureAccesses, Error>>;

impl TextureAccesses {
    /// The referenced texture arguments of `reflection`'s stages.
    pub(crate) fn of(reflection: &RenderPipelineReflection) -> TextureAccesses {
        let stages: [(RenderStages, &[Binding]); 5] = [
            (RenderStages::VERTEX, reflection.vertex_bindings()),
            (RenderStages::FRAGMENT, reflection.fragment_bindings()),
            (RenderStages::TILE, reflection.tile_bindings()),
            (RenderStages::OBJECT, reflection.object_bindings()),
            (RenderStages::MESH, reflection.mesh_bindings()),
        ];
        let mut accesses = Vec::new();
        for (stage, bindings) in stages {
            for binding in bindings {
                let Some(texture) = binding.texture() else {
                    continue;
                };
                if binding.binding_type() != BindingType::Texture || !binding.is_used() {
                    continue;
                }
                let first = u64::from(binding.index());
                let slots = first..first + u64::from(texture.array_length().max(1));
                accesses.push((stage, slots, binding.access()));
            }
        }
        TextureAccesses(accesses)
    }

    /// The accesses of the referenced texture arguments of `stages` that cover slot `index`,
    /// joined into one; `None` where none covers it.
    pub(crate) fn at(&self, stages: RenderStages, index: u64) -> Option<BindingAccess> {
        self.0
            .iter()
            .filter(|(stage, slots, _)| stages.contains(*stage) && slots.contains(&index))
            .map(|&(_, _, access)| access)
            .reduce(|a, b| if a == b { a } else { BindingAccess::ReadWrite })
    }

    /// `memo`'s accesses, from `reflect`'s reflection (under the standard budget) the first time.
    pub(crate) fn remembered(
        memo: &TextureAccessMemo,
        reflect: impl FnOnce(&Budget) -> Result<RenderPipelineReflection, metal2vulkan::error::Error>,
    ) -> Result<&TextureAccesses, Error> {
        memo.get_or_init(|| {
            reflect(&Budget::standard())
                .map(|reflection| TextureAccesses::of(&reflection))
                .map_err(device_error)
        })
        .as_ref()
        .map_err(Clone::clone)
    }
}

#[cfg(test)]
mod tests {
    use super::TextureAccesses;
    use metal2vulkan::metal::{BindingAccess, RenderStages};

    /// A slot's answer is the access of the referenced texture argument covering it in the
    /// stages asked, an array's at each element's slot; stages that agree keep their access,
    /// stages that differ join to `ReadWrite`; no stage covering the slot, or no stage asked, is
    /// `None`.
    #[test]
    fn a_slot_answers_the_accesses_of_the_stages_asked_joined() {
        use BindingAccess::{ReadOnly, ReadWrite, WriteOnly};
        let (vertex, fragment) = (RenderStages::VERTEX, RenderStages::FRAGMENT);
        let accesses = TextureAccesses(vec![
            (vertex, 0..1, ReadOnly),
            (vertex, 1..2, ReadOnly),
            (fragment, 0..1, ReadOnly),
            (fragment, 1..3, WriteOnly),
        ]);
        let both = vertex.union(fragment);
        for (stages, index, want) in [
            (vertex, 0, Some(ReadOnly)),
            (fragment, 2, Some(WriteOnly)),
            (fragment, 3, None),
            (both, 0, Some(ReadOnly)),
            (both, 1, Some(ReadWrite)),
            (both, 2, Some(WriteOnly)),
            (RenderStages::TILE, 0, None),
            (RenderStages::empty(), 0, None),
        ] {
            assert_eq!(accesses.at(stages, index), want, "{stages:?} {index}");
        }
    }
}
