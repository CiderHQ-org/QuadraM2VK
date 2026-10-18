//! Texture buffers' texel buffer views (`buffer_texture`): the usage a view takes beside its
//! buffer's, by the kinds of texel buffer its format serves on the device; and a staged shadow's
//! regions back into its buffer.

use super::super::buffer_texture::{staged_rows_back, view_usage, ViewUsage};
use super::super::error::{Error, RuntimeGap};
use super::super::linear_plan::BufferLayout;
use ash::vk;
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

/// A staged shadow's texels go back into the buffer as one region where its rows lie one after
/// another (a 1080p surface's packed rows: one region, not 1080), and a region a row where bytes
/// lie between them; each slice of packed rows joins the next where the slices lie one after
/// another too.
#[test]
fn packed_rows_go_back_as_one_region() {
    let layout = |row_pitch: u64, rows: u64, slices: u64| BufferLayout {
        offset: 0,
        row_pitch,
        slice_pitch: row_pitch * rows,
        row_bytes: 7680,
        rows,
        slices,
    };
    let regions = |layout: BufferLayout, start: u64, at: u64| -> Vec<(u64, u64, u64)> {
        (staged_rows_back(&layout, start, at).iter())
            .map(|copy: &vk::BufferCopy| (copy.src_offset, copy.dst_offset, copy.size))
            .collect()
    };
    assert_eq!(
        regions(layout(7680, 1080, 1), 64, 3),
        [(64, 3, 7680 * 1080)]
    );
    assert_eq!(regions(layout(7680, 4, 2), 0, 5), [(0, 5, 7680 * 8)]);
    assert_eq!(
        regions(layout(7744, 3, 1), 0, 5),
        [
            (0, 5, 7680),
            (7744, 7749, 7680),
            (2 * 7744, 2 * 7744 + 5, 7680)
        ]
    );
}
