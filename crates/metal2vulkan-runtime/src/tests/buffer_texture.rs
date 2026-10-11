//! Texture buffers' texel buffer views (`buffer_texture`): the usage a view takes beside its
//! buffer's, by the kinds of texel buffer its format serves on the device.

use super::super::buffer_texture::{view_usage, ViewUsage};
use super::super::error::{Error, RuntimeGap};
use metal2vulkan::device::FormatUsage;

/// Whether `result` is the typed gap of a view the runtime does not make.
fn not_made(result: Result<ViewUsage, Error>) -> bool {
    matches!(result, Err(Error::NotImplemented(RuntimeGap::Textures)))
}

/// A view takes its buffer's usage where its format serves both kinds of texel buffer. Where the
/// format serves only the view's own kind (NVIDIA takes `R64_UINT` for storage texel buffers
/// only), a device with `maintenance5` makes the view naming that kind alone; one without it does
/// not make the view, nor does any device make one in a format that does not serve its kind.
#[test]
fn a_view_names_the_one_kind_its_format_serves_where_maintenance5_allows() {
    let both = |_: FormatUsage| true;
    let storage_only = |usage: FormatUsage| usage == FormatUsage::StorageTexelBuffer;
    let uniform_only = |usage: FormatUsage| usage == FormatUsage::UniformTexelBuffer;
    for storage in [false, true] {
        for maintenance5 in [false, true] {
            assert_eq!(
                view_usage(both, storage, maintenance5).unwrap(),
                ViewUsage::Buffer
            );
        }
    }
    assert_eq!(
        view_usage(storage_only, true, true).unwrap(),
        ViewUsage::Only { storage: true }
    );
    assert_eq!(
        view_usage(uniform_only, false, true).unwrap(),
        ViewUsage::Only { storage: false }
    );
    assert!(not_made(view_usage(storage_only, true, false)));
    assert!(not_made(view_usage(uniform_only, false, false)));
    assert!(not_made(view_usage(storage_only, false, true)));
    assert!(not_made(view_usage(uniform_only, true, true)));
    assert!(not_made(view_usage(|_| false, true, true)));
}
