use super::owned_kernel::{OwnedBinds, OwnedKernel, OwnedModule, OwnedSet, OwnedSets};
    /// Its two buffers' bytes: pushed where the device pushes, else a set of a pool of its own.
    binds: OwnedBinds,
                binds: OwnedBinds::Sets(sets),
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: Vec::new(),
            buffers: buffers.to_vec(),
        let binds = OwnedBinds::new(device, &kernel, &[set]).ok()?;
            binds,
            self.binds.bind(device, command_buffer, &self.kernel, 0);
