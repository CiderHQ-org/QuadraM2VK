//! A command's argument samplers, listed again as its command buffer commits.
//!
//! Metal's function reaches a sampler state by its resource ID as the command runs, as it does a
//! texture (`resource_ids.rs`). A command reaches the states its sampler table lists
//! (`BindState::argument_samplers`): the device's live states as it was encoded. So a state made
//! after that, or one that took a listed state's ID, would be missed. Each command that writes a
//! sampler table keeps it here, and the command buffer lists the states live as it commits, before
//! it is submitted:
//!
//! - **Where they fit the plan** (as many as the planned entries hold beside the null entry, twice
//!   the class's, each of a shape the plan was made for or one its checks admit,
//!   `BindingPlan::relisted_sampler_tables`), their table is written over the command's region of
//!   the arena, and their elements into the heap set the command binds
//!   (`binding::argument_sampler_writes`, `heap_sets::bind`): the set's version a pending command
//!   buffer does not use, or the command buffer's own, still written until it commits
//!   (`heap_sets.rs`).
//! - **Where they do not** (a state past the command's set-1 sampler arrays,
//!   `heap_sets::fit_samplers`; more states than its tables hold; or a shape its checks refuse,
//!   such as a cubic state beside a fragment stage), the command cannot reach every state as
//!   Metal's function would, so its command buffer fails as it completes
//!   (`CommandBufferError::UnlistedArgumentSampler`), announced once as `unlisted-argument-sampler`
//!   and counted (`WorkCounts::unlisted_argument_sampler_failures`). The refusal is
//!   over-approximate: it holds whether or not the function reaches the state left out.
//!
//! A state made after the commit, while the command buffer has not completed, is not listed: the
//! device announces it as it is made (`sampler.rs`), Metal's answer then hanging on whether its
//! command ran before the state was made.
//!
//! A command a host step records after the commit is listed again by no commit: it lists the states
//! live as it is recorded, and where more live than its set-1 sampler arrays hold, so that it
//! leaves one out (`heap_sets::fit_samplers`), its command buffer fails as it completes
//! (`CommandBufferError::UnlistedArgumentSampler`, `State::unlisted_argument_sampler`), announced
//! and counted as above, where it completed with the function reading the neutral sampler for the
//! state left out.

use super::binding;
use super::degradation::Degradation;
use super::error::{internal, Error};
use super::heap_sets::{self, HeapUse};
use super::pipeline::VariantObjects;
use super::sampler::{OwnedSampler, SamplerState};
use super::upload::Upload;
use super::Device;
use metal2vulkan::bind::{ArgumentSampler, BindingPlan, HeapPlacement};
use metal2vulkan::pipeline::DescriptorSetLayout;
use std::sync::Arc;

/// One command's sampler tables and what lists them again as its command buffer commits.
pub(super) struct SamplerListing {
    plan: Arc<BindingPlan>,
    objects: Arc<VariantObjects>,
    /// Set 1 of the command's layout, whose sampler arrays' capacity bounds the elements.
    heap: DescriptorSetLayout,
    placement: HeapPlacement,
    /// The argument samplers the command listed.
    listed: Vec<ArgumentSampler>,
    /// Each sampler table's `(set, binding)` and region.
    tables: Vec<((u32, u32), Upload)>,
    /// The commands, named, for the notices.
    commands: String,
}

impl SamplerListing {
    /// The listing of a command of `plan` and `objects` whose layout's set 1 is `heap`, which
    /// listed `listed` under `placement` in `tables`: `None` where it has no sampler table.
    pub(super) fn of(
        (plan, objects): (&Arc<BindingPlan>, &Arc<VariantObjects>),
        heap: Option<&DescriptorSetLayout>,
        (listed, placement): (&[ArgumentSampler], HeapPlacement),
        tables: Vec<((u32, u32), Upload)>,
        commands: impl FnOnce() -> String,
    ) -> Option<SamplerListing> {
        if tables.is_empty() {
            return None;
        }
        Some(SamplerListing {
            plan: Arc::clone(plan),
            objects: Arc::clone(objects),
            heap: heap?.clone(),
            placement,
            listed: listed.to_vec(),
            tables,
            commands: commands(),
        })
    }

    /// List `live` again, the device's live states as the command buffer whose heap use is
    /// `heaps` commits (the module docs): the Vulkan samplers the new elements name, which the
    /// command buffer holds until it completes, or `None` where the command's plan cannot list
    /// them (its command buffer then fails, `CommandBufferError::UnlistedArgumentSampler`).
    pub(super) fn relist(
        &self,
        device: &Device,
        live: &[SamplerState],
        heaps: &mut HeapUse,
    ) -> Result<Option<Vec<Arc<OwnedSampler>>>, Error> {
        let mut live = live.to_vec();
        let mut unlisted = None;
        heap_sets::fit_samplers(Some(&self.heap), &mut live, &mut unlisted);
        // A state past the command's sampler arrays, which no element of them holds.
        if let Some(unlisted) = unlisted {
            heap_sets::degrade_unlisted(device, unlisted, &self.commands);
            return Ok(None);
        }
        let samplers = heap_sets::listed(&live);
        if samplers == self.listed {
            return Ok(Some(Vec::new()));
        }
        let Some(bytes) = self
            .plan
            .relisted_sampler_tables(device.compile_device(), &samplers, self.placement)
            .map_err(Error::Bind)?
        else {
            let commands = &self.commands;
            device.degrade(Degradation::UnlistedArgumentSampler, || {
                (
                    format!("{commands} listing the argument samplers"),
                    "the sampler states live as their command buffer commits are more than their \
                     plan's sampler tables hold, or of a shape its checks refuse, so the command \
                     buffer fails (CommandBufferError::UnlistedArgumentSampler)"
                        .to_owned(),
                )
            });
            return Ok(None);
        };
        for (key, region) in &self.tables {
            let table = bytes
                .iter()
                .find(|table| (table.set, table.binding) == *key)
                .ok_or_else(|| internal("a sampler table its plan does not list"))?;
            region.write(&table.bytes);
        }
        let (writes, samplers) = binding::argument_sampler_writes(device, &self.plan, &live)?;
        if !writes.is_empty() {
            heap_sets::bind(device, &self.objects, heaps, writes, None)?;
        }
        Ok(Some(samplers))
    }
}
