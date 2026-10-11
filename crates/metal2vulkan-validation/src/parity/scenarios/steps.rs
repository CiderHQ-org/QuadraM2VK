//! The steps the scenarios share. Every record's fingerprint covers this file
//! (`parity::HARNESS`): a change here records every scenario again.

use super::super::api::{Api, Fixture, Observation};
use crate::runtime_lane::public_case;
use metal2vulkan_runtime::ResourceOptions;

/// The public fixture of the committed case named `name`.
pub(super) fn fixture(name: &str) -> Result<Fixture, String> {
    let case = public_case(name)?;
    let row = crate::source::public_sources()?
        .into_iter()
        .find(|row| row.air_sha256 == case.case.air_sha256 && row.entry == case.case.entry)
        .ok_or_else(|| format!("{name}: its AIR is no public fixture"))?;
    Ok(Fixture {
        row,
        bitcode: case.bitcode,
    })
}

/// A `Shared` buffer holding `bytes`, made as `op`.
pub(super) fn shared_with<A: Api>(
    api: &A,
    obs: &mut Observation,
    op: &str,
    bytes: &[u8],
) -> Option<A::Buffer> {
    let buffer = obs.call(op, || {
        api.buffer(bytes.len() as u64, ResourceOptions::shared())
    })?;
    api.write(&buffer, bytes);
    Some(buffer)
}

/// Commit `cb` as `op`, wait for it, and record its completion.
pub(super) fn submit<A: Api>(
    api: &A,
    obs: &mut Observation,
    op: &str,
    cb: A::CommandBuffer,
) -> Result<Option<()>, String> {
    if obs.call(op, || api.commit(&cb)).is_none() {
        return Ok(None);
    }
    let completion = api.completion(cb)?;
    obs.completed(&format!("{op}: completion"), completion);
    Ok(Some(()))
}

/// Bytes `0..length`, each `index * 7 + seed`: no two neighbours alike.
pub(super) fn pattern(length: u64, seed: u8) -> Vec<u8> {
    (0..length)
        .map(|index| (index as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}
