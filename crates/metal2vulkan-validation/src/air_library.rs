//! Runtime libraries of the AIR bitcode the validation's cases and lanes hold.
//!
//! The runtime's `Device::new_library_with_data` takes a metallib alone, as Metal's
//! `newLibraryWithData:error:` does. The cases are AIR bitcode, which `llvm-as` assembles from
//! their LLVM text: the translator parses it, and the device takes the parsed library
//! (`Device::new_library_with_compile_library`).

use metal2vulkan_runtime::{Device, Error, Library};

/// A device's library of AIR bitcode.
pub trait AirLibrary {
    /// The library of `bytes`: AIR bitcode, raw or wrapped, as the translator parses it
    /// (`metal2vulkan::library::Library::parse`, which takes a metallib too). A refusal is the
    /// translator's, as the runtime reports it (`Error::Unsupported` or `Error::Compile`).
    fn new_library_with_air(&self, bytes: &[u8]) -> Result<Library, Error>;
}

impl AirLibrary for Device {
    fn new_library_with_air(&self, bytes: &[u8]) -> Result<Library, Error> {
        let library =
            metal2vulkan::library::Library::parse(bytes).map_err(|error| match error {
                metal2vulkan::error::Error::Unsupported(unsupported) => {
                    Error::Unsupported(unsupported)
                }
                other => Error::Compile(other),
            })?;
        Ok(self.new_library_with_compile_library(library))
    }
}
