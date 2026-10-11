//! What MoltenVK hands SPIRV-Cross for a module : the configuration the pinned
//! MoltenVK MSL helper (`scripts/moltenvk-msl-helper`, SPIRV-Cross at the commit MoltenVK links)
//! converts it with, derived from the pipeline layout the product's link gave the module
//! ([`moltenvk`]). `corpus-pipeline --keep-modules` writes it beside each kept module, so a module
//! can be converted to MSL as MoltenVK 1.4.1 converts it on the `moltenvk-m3` device.

pub mod moltenvk;

#[cfg(test)]
mod tests;
