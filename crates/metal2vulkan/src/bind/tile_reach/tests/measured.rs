//! Every case our probes measured on an Apple M4 under macOS 27.0.1 (each in a process of its own;
//! the kit `share-reach` measured the same 111 answers on an Apple M3 Max under macOS 26.7), as the
//! probes drew it, and the tiles it reached: a `DontCare`-loaded, stored color attachment prefilled
//! with 0x11 bytes, whose reached tiles hold zero or the draw's red and whose others keep the bytes
//! (the module docs). Our own probes.

use super::reference::Cull;
use super::{clip, clip_z, full, lines, points, quad, quad_z, row, tris, Draw, Other, Row};

/// The rows in the probes' order, each with the tiles (`R`) it reached of the 128 by 128
/// attachment, row by row of tiles (32 by 16 at 4 samples), slice by slice.
pub(super) fn rows() -> Vec<Row> {
    vec![
        row("reach1", "nodraw", tris(Vec::new()))
            .draw(Draw::None)
            .reaches(&[".... .... .... ...."]),
        row("reach1", "full", tris(full())).reaches(&["RRRR RRRR RRRR RRRR"]),
        row("reach1", "quad_t11", tris(quad(40.0, 40.0, 50.0, 50.0)))
            .reaches(&[".... .R.. .... ...."]),
        row(
            "reach1",
            "tiny_nocenter_t11",
            tris(vec![clip(40.1, 40.1), clip(40.4, 40.1), clip(40.1, 40.4)]),
        )
        .reaches(&[".... .... .... ...."]),
        row(
            "reach1",
            "tiny_center_t11",
            tris(vec![clip(40.3, 40.3), clip(40.8, 40.3), clip(40.3, 40.8)]),
        )
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach1",
            "corner_tri95",
            tris(vec![clip(0.0, 0.0), clip(95.0, 0.0), clip(0.0, 95.0)]),
        )
        .reaches(&["RRR. RR.. R... ...."]),
        row(
            "reach1",
            "upper_right_half",
            tris(vec![clip(0.0, 0.0), clip(128.0, 0.0), clip(128.0, 128.0)]),
        )
        .reaches(&["RRRR .RRR ..RR ...R"]),
        row(
            "reach1",
            "sliver_diag",
            tris(vec![clip(2.0, 2.0), clip(126.0, 126.0), clip(3.2, 2.0)]),
        )
        .reaches(&["RR.. .RR. ..RR ...R"]),
        row(
            "reach1",
            "sliver_diag_nocenter",
            tris(vec![clip(2.6, 2.6), clip(125.6, 125.6), clip(2.62, 2.58)]),
        )
        .reaches(&["RR.. .RR. ..RR ...R"]),
        row(
            "reach1",
            "straddle_nocenter",
            tris(quad(31.6, 10.0, 32.4, 20.0)),
        )
        .reaches(&[".... .... .... ...."]),
        row(
            "reach1",
            "edge_exact_left",
            tris(quad(20.0, 20.0, 32.0, 30.0)),
        )
        .reaches(&["RR.. .... .... ...."]),
        row(
            "reach1",
            "edge_near_right",
            tris(quad(20.0, 20.0, 32.3, 30.0)),
        )
        .reaches(&["RR.. .... .... ...."]),
        row(
            "reach1",
            "edge_center_right",
            tris(quad(20.0, 20.0, 32.6, 30.0)),
        )
        .reaches(&["RR.. .... .... ...."]),
        row(
            "reach1",
            "backface_culled_t11",
            tris(vec![clip(40.0, 40.0), clip(40.0, 50.0), clip(50.0, 40.0)]),
        )
        .cull(Cull::Back)
        .reaches(&[".... .... .... ...."]),
        row(
            "reach1",
            "frontface_t11",
            tris(vec![clip(40.0, 40.0), clip(50.0, 40.0), clip(40.0, 50.0)]),
        )
        .cull(Cull::Back)
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach1",
            "degenerate_row0",
            tris(vec![clip(2.0, 10.0), clip(64.0, 10.0), clip(126.0, 10.0)]),
        )
        .reaches(&[".... .... .... ...."]),
        row("reach1", "discard_t11", tris(quad(40.0, 40.0, 50.0, 50.0)))
            .discards()
            .reaches(&[".... .R.. .... ...."]),
        row("reach1", "scissor_t11", tris(full()))
            .scissor([40, 40, 10, 10])
            .reaches(&[".... .R.. .... ...."]),
        row(
            "reach1",
            "scissor_nocover",
            tris(quad(2.0, 2.0, 10.0, 10.0)),
        )
        .scissor([40, 40, 10, 10])
        .reaches(&[".... .... .... ...."]),
        row("reach1", "viewport_t11", tris(full()))
            .viewport([32.0, 32.0, 32.0, 32.0])
            .reaches(&[".... .R.. .... ...."]),
        row(
            "reach1",
            "off_partial",
            tris(vec![
                clip(100.0, 100.0),
                clip(300.0, 100.0),
                clip(100.0, 300.0),
            ]),
        )
        .reaches(&[".... .... .... ...R"]),
        row(
            "reach1",
            "off_all",
            tris(vec![
                clip(200.0, 200.0),
                clip(300.0, 200.0),
                clip(200.0, 300.0),
            ]),
        )
        .reaches(&[".... .... .... ...."]),
        row(
            "reach1",
            "behind_near",
            tris(vec![
                clip_z(40.0, 40.0, -0.5),
                clip_z(50.0, 40.0, -0.5),
                clip_z(40.0, 50.0, -0.5),
            ]),
        )
        .reaches(&[".... .... .... ...."]),
        row("reach1", "point1_t11", points(vec![clip(40.5, 40.5)]))
            .size(1.0)
            .reaches(&[".... .R.. .... ...."]),
        row("reach1", "point40_center", points(vec![clip(64.0, 64.0)]))
            .size(40.0)
            .reaches(&[".... .RR. .RR. ...."]),
        row(
            "reach1",
            "line_row1",
            lines(vec![clip(2.0, 40.5), clip(126.0, 40.5)]),
        )
        .reaches(&[".... RRRR .... ...."]),
        row(
            "reach1",
            "line_diag",
            lines(vec![clip(2.0, 2.0), clip(126.0, 126.0)]),
        )
        .reaches(&["RR.. RRR. .RRR ..RR"]),
        // The small-primitive cull: a triangle whose box holds a pixel centre it does not cover.
        row(
            "reach2",
            "box_centre_nocover_t11",
            tris(vec![clip(40.0, 40.0), clip(40.95, 40.0), clip(40.0, 40.95)]),
        )
        .reaches(&[".... .... .... ...."]),
        row(
            "reach2",
            "box_centres_nocover_straddle",
            tris(vec![clip(31.0, 40.0), clip(32.6, 40.0), clip(31.0, 40.6)]),
        )
        .reaches(&[".... .... .... ...."]),
        row(
            "reach2",
            "covers_one_centre_straddle",
            tris(vec![clip(31.0, 40.0), clip(32.6, 40.0), clip(31.0, 41.2)]),
        )
        .reaches(&[".... RR.. .... ...."]),
        // Sub-pixel right edges below a tile boundary: x in [24, 32 - d], y in [8, 16].
        row(
            "reach2",
            "snapR_1_16",
            tris(quad(24.0, 8.0, 32.0 - 1.0 / 16.0, 16.0)),
        )
        .reaches(&["R... .... .... ...."]),
        row(
            "reach2",
            "snapR_1_64",
            tris(quad(24.0, 8.0, 32.0 - 1.0 / 64.0, 16.0)),
        )
        .reaches(&["R... .... .... ...."]),
        row(
            "reach2",
            "snapR_1_256",
            tris(quad(24.0, 8.0, 32.0 - 1.0 / 256.0, 16.0)),
        )
        .reaches(&["R... .... .... ...."]),
        row(
            "reach2",
            "snapR_1_512",
            tris(quad(24.0, 8.0, 32.0 - 1.0 / 512.0, 16.0)),
        )
        .reaches(&["RR.. .... .... ...."]),
        row(
            "reach2",
            "snapR_1_4096",
            tris(quad(24.0, 8.0, 32.0 - 1.0 / 4096.0, 16.0)),
        )
        .reaches(&["RR.. .... .... ...."]),
        row("reach2", "edgeR_32", tris(quad(24.0, 8.0, 32.0, 16.0)))
            .reaches(&["RR.. .... .... ...."]),
        row("reach2", "edgeL_32", tris(quad(32.0, 8.0, 40.0, 16.0)))
            .reaches(&[".R.. .... .... ...."]),
        row(
            "reach2",
            "snapB_1_16",
            tris(quad(8.0, 24.0, 16.0, 32.0 - 1.0 / 16.0)),
        )
        .reaches(&["R... .... .... ...."]),
        row(
            "reach2",
            "snapB_1_512",
            tris(quad(8.0, 24.0, 16.0, 32.0 - 1.0 / 512.0)),
        )
        .reaches(&["R... R... .... ...."]),
        row("reach2", "edgeB_32", tris(quad(8.0, 24.0, 16.0, 32.0)))
            .reaches(&["R... R... .... ...."]),
        // Corner touches.
        row(
            "reach2",
            "corner_tri96",
            tris(vec![clip(0.0, 0.0), clip(96.0, 0.0), clip(0.0, 96.0)]),
        )
        .reaches(&["RRRR RRR. RR.. R..."]),
        row(
            "reach2",
            "corner_tri32",
            tris(vec![clip(0.0, 0.0), clip(32.0, 0.0), clip(0.0, 32.0)]),
        )
        .reaches(&["RR.. R... .... ...."]),
        row(
            "reach2",
            "corner_tri64_from_t11",
            tris(vec![clip(40.0, 40.0), clip(64.0, 40.0), clip(40.0, 64.0)]),
        )
        .reaches(&[".... .RR. .R.. ...."]),
        // Points (square of the point size around the centre).
        row("reach2", "point1_straddle", points(vec![clip(31.75, 10.5)]))
            .size(1.0)
            .reaches(&["RR.. .... .... ...."]),
        row("reach2", "point1_at32", points(vec![clip(32.0, 10.5)]))
            .size(1.0)
            .reaches(&["RR.. .... .... ...."]),
        row("reach2", "point2_at33", points(vec![clip(33.0, 10.5)]))
            .size(2.0)
            .reaches(&[".R.. .... .... ...."]),
        row(
            "reach2",
            "point_half_nocover",
            points(vec![clip(40.0, 40.0)]),
        )
        .size(0.5)
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach2",
            "point40_off_right",
            points(vec![clip(130.0, 64.0)]),
        )
        .size(40.0)
        .reaches(&[".... ...R ...R ...."]),
        // Lines.
        row(
            "reach2",
            "line_y31_9",
            lines(vec![clip(2.0, 31.9), clip(20.0, 31.9)]),
        )
        .reaches(&["R... R... .... ...."]),
        row(
            "reach2",
            "line_y32",
            lines(vec![clip(2.0, 32.0), clip(20.0, 32.0)]),
        )
        .reaches(&["R... R... .... ...."]),
        row(
            "reach2",
            "line_y31_5",
            lines(vec![clip(2.0, 31.5), clip(20.0, 31.5)]),
        )
        .reaches(&["R... R... .... ...."]),
        row(
            "reach2",
            "line_short_nocover",
            lines(vec![clip(40.1, 40.1), clip(40.4, 40.1)]),
        )
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach2",
            "line_x_to_32",
            lines(vec![clip(10.0, 10.5), clip(32.0, 10.5)]),
        )
        .reaches(&["RR.. .... .... ...."]),
        // Depth clipping.
        row(
            "reach2",
            "far_clipped_t11",
            tris(quad_z(40.0, 40.0, 50.0, 50.0, 1.5)),
        )
        .reaches(&[".... .... .... ...."]),
        row(
            "reach2",
            "far_clamped_t11",
            tris(quad_z(40.0, 40.0, 50.0, 50.0, 1.5)),
        )
        .clamp()
        .reaches(&[".... .R.. .... ...."]),
        // Indirect and empty instanced draws.
        row("reach2", "indirect_t11", tris(quad(40.0, 40.0, 50.0, 50.0)))
            .draw(Draw::Indirect)
            .reaches(&[".... .R.. .... ...."]),
        row(
            "reach2",
            "instances0_t11",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .draw(Draw::NoInstances)
        .reaches(&[".... .... .... ...."]),
        // Huge coordinates: an edge across the frame at y = 40 and at y = 32.
        row(
            "reach2",
            "huge_edge_y40",
            tris(vec![clip(-1e5, 40.0), clip(1e5, 40.0), clip(0.0, 1e5)]),
        )
        .reaches(&[".... RRRR RRRR RRRR"]),
        row(
            "reach2",
            "huge_edge_y32",
            tris(vec![clip(-1e5, 32.0), clip(1e5, 32.0), clip(0.0, 1e5)]),
        )
        .reaches(&[".... RRRR RRRR RRRR"]),
        // Fractional and straddling viewports over a full-frame triangle.
        row("reach2", "viewport_frac_t11", tris(full()))
            .viewport([32.25, 32.25, 31.5, 31.5])
            .reaches(&[".... .R.. .... ...."]),
        row("reach2", "viewport_cross", tris(full()))
            .viewport([20.0, 20.0, 20.0, 20.0])
            .reaches(&["RR.. RR.. .... ...."]),
        // Tile size for wider attachments: a quad over pixels 2 to 6.
        row("reach2", "rgba8_t00", tris(quad(2.0, 2.0, 6.0, 6.0)))
            .reaches(&["R... .... .... ...."]),
        row("reach2", "rgba16f_t00", tris(quad(2.0, 2.0, 6.0, 6.0)))
            .reaches(&["R... .... .... ...."]),
        row("reach2", "rgba32f_t00", tris(quad(2.0, 2.0, 6.0, 6.0)))
            .reaches(&["R... .... .... ...."]),
        row("reach2", "rgba32f_x4_t00", tris(quad(2.0, 2.0, 6.0, 6.0)))
            .reaches(&["R... .... .... ...."]),
        row("reach2", "rgba16f_x8_t00", tris(quad(2.0, 2.0, 6.0, 6.0)))
            .reaches(&["R... .... .... ...."]),
        row("reach2", "rgba32f_x8_t00", tris(quad(2.0, 2.0, 6.0, 6.0)))
            .reaches(&["R... .... .... ...."]),
        // The fill rule: one pixel centre (40.5, 40.5) on one edge of a triangle covering no other.
        row(
            "reach3",
            "fill_left_on_centre",
            tris(vec![clip(40.5, 40.0), clip(40.9, 40.5), clip(40.5, 41.0)]),
        )
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach3",
            "fill_right_on_centre",
            tris(vec![clip(40.5, 40.0), clip(40.5, 41.0), clip(40.1, 40.5)]),
        )
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach3",
            "fill_top_on_centre",
            tris(vec![clip(40.0, 40.5), clip(41.0, 40.5), clip(40.5, 40.9)]),
        )
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach3",
            "fill_bottom_on_centre",
            tris(vec![clip(40.0, 40.5), clip(40.5, 40.1), clip(41.0, 40.5)]),
        )
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach3",
            "fill_vertex_on_centre",
            tris(vec![clip(40.5, 40.5), clip(40.9, 40.6), clip(40.6, 40.9)]),
        )
        .reaches(&[".... .R.. .... ...."]),
        // Line ends and widths (horizontal, at a pixel centre row).
        row(
            "reach3",
            "line_end_31_6",
            lines(vec![clip(10.0, 10.5), clip(31.6, 10.5)]),
        )
        .reaches(&["RR.. .... .... ...."]),
        row(
            "reach3",
            "line_end_31_9",
            lines(vec![clip(10.0, 10.5), clip(31.9, 10.5)]),
        )
        .reaches(&["RR.. .... .... ...."]),
        row(
            "reach3",
            "line_y31_4",
            lines(vec![clip(2.0, 31.4), clip(20.0, 31.4)]),
        )
        .reaches(&["R... .... .... ...."]),
        row(
            "reach3",
            "line_y31_45",
            lines(vec![clip(2.0, 31.45), clip(20.0, 31.45)]),
        )
        .reaches(&["R... .... .... ...."]),
        row(
            "reach3",
            "line_steep_x31_6",
            lines(vec![clip(31.6, 2.0), clip(31.6, 20.0)]),
        )
        .reaches(&["RR.. .... .... ...."]),
        row(
            "reach3",
            "line_shallow_cross",
            lines(vec![clip(2.0, 31.0), clip(60.0, 33.0)]),
        )
        .reaches(&["RR.. RR.. .... ...."]),
        // Near-plane clipping: the unclipped triangle reaches tile (2,1), the clipped part (1,1) alone.
        row(
            "reach3",
            "near_cross",
            tris(vec![
                clip(40.0, 40.0),
                clip(40.0, 60.0),
                clip_z(80.0, 50.0, -0.5),
            ]),
        )
        .reaches(&[".... .R.. .... ...."]),
        // The cull against the scissor: samples covered only outside the scissor, area inside it.
        row(
            "reach3",
            "scissor_cull_order",
            tris(vec![clip(2.0, 2.0), clip(4.0, 2.0), clip(52.2, 52.0)]),
        )
        .scissor([50, 50, 4, 4])
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach3",
            "scissor_cull_order_noscissor",
            tris(vec![clip(2.0, 2.0), clip(4.0, 2.0), clip(52.2, 52.0)]),
        )
        .reaches(&["RR.. .R.. .... ...."]),
        // Multisampled tiles and sample positions.
        row("reach3", "ms4_quad_t00", tris(quad(2.0, 2.0, 6.0, 6.0)))
            .samples(4)
            .reaches(&["R... .... .... .... .... .... .... ...."]),
        row("reach3", "ms4_edge_y16", tris(quad(2.0, 8.0, 6.0, 16.0)))
            .samples(4)
            .reaches(&["R... R... .... .... .... .... .... ...."]),
        row("reach3", "ms4_edge_y15_9", tris(quad(2.0, 8.0, 6.0, 15.9)))
            .samples(4)
            .reaches(&["R... .... .... .... .... .... .... ...."]),
        row(
            "reach3",
            "ms4_tiny_sample_not_centre",
            tris(vec![
                clip(40.3, 40.05),
                clip(40.45, 40.05),
                clip(40.3, 40.2),
            ]),
        )
        .samples(4)
        .reaches(&[".... .... .R.. .... .... .... .... ...."]),
        row(
            "reach3",
            "ms1_tiny_sample_not_centre",
            tris(vec![
                clip(40.3, 40.05),
                clip(40.45, 40.05),
                clip(40.3, 40.2),
            ]),
        )
        .reaches(&[".... .... .... ...."]),
        row("reach3", "ms2_quad_t00", tris(quad(2.0, 2.0, 6.0, 6.0)))
            .samples(2)
            .reaches(&["R... .... .... ...."]),
        row("reach3", "ms2_edge_y16", tris(quad(2.0, 8.0, 6.0, 16.0)))
            .samples(2)
            .reaches(&["R... .... .... ...."]),
        // Another attachment's clear.
        row(
            "reach3",
            "clear_c1_store_quad_t11",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .other(Other::ColorClearStore)
        .reaches(&["RRRR RRRR RRRR RRRR"]),
        row(
            "reach3",
            "clear_c1_store_instances0",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .other(Other::ColorClearStore)
        .draw(Draw::NoInstances)
        .reaches(&["RRRR RRRR RRRR RRRR"]),
        row(
            "reach3",
            "clear_c1_store_nodraw",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .other(Other::ColorClearStore)
        .draw(Draw::None)
        .reaches(&[".... .... .... ...."]),
        row(
            "reach3",
            "clear_c1_dontcare_quad_t11",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .other(Other::ColorClearDontCare)
        .reaches(&[".... .R.. .... ...."]),
        row(
            "reach3",
            "clear_depth_store_quad_t11",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .other(Other::DepthClearStore)
        .reaches(&["RRRR RRRR RRRR RRRR"]),
        row(
            "reach3",
            "clear_stencil_store_quad_t11",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .other(Other::StencilClearStore)
        .reaches(&["RRRR RRRR RRRR RRRR"]),
        row(
            "reach3",
            "clear_depth_store_instances0",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .other(Other::DepthClearStore)
        .draw(Draw::NoInstances)
        .reaches(&["RRRR RRRR RRRR RRRR"]),
        // Diagonal lines near a tile edge: a parallelogram of vertical extent 1 reaches y = 32 from
        // a centre line starting at (40, 31.0); a rectangle of width 1 reaches only y = 31.71.
        row(
            "reach4",
            "line45_from_40_31_0",
            lines(vec![clip(40.0, 31.0), clip(60.0, 11.0)]),
        )
        .reaches(&[".R.. .... .... ...."]),
        row(
            "reach4",
            "line45_from_40_31_3",
            lines(vec![clip(40.0, 31.3), clip(60.0, 11.3)]),
        )
        .reaches(&[".R.. .... .... ...."]),
        row(
            "reach4",
            "line45_from_40_30_8",
            lines(vec![clip(40.0, 30.8), clip(60.0, 10.8)]),
        )
        .reaches(&[".R.. .... .... ...."]),
        row(
            "reach4",
            "line_shallow_end_31_0",
            lines(vec![clip(10.0, 20.0), clip(31.0, 21.0)]),
        )
        .reaches(&["R... .... .... ...."]),
        row(
            "reach4",
            "line_shallow_end_31_4",
            lines(vec![clip(10.0, 20.0), clip(31.4, 21.0)]),
        )
        .reaches(&["R... .... .... ...."]),
        // Degenerate (zero-area) triangles through pixel centres, and off them.
        row(
            "reach4",
            "degenerate_on_centres",
            tris(vec![clip(2.0, 10.5), clip(64.0, 10.5), clip(126.0, 10.5)]),
        )
        .reaches(&[".... .... .... ...."]),
        row(
            "reach4",
            "degenerate_vertical_on_centres",
            tris(vec![clip(40.5, 2.0), clip(40.5, 64.0), clip(40.5, 126.0)]),
        )
        .reaches(&[".... .... .... ...."]),
        row(
            "reach4",
            "degenerate_point",
            tris(vec![clip(40.5, 40.5), clip(40.5, 40.5), clip(40.5, 40.5)]),
        )
        .reaches(&[".... .... .... ...."]),
        // The sample cull against the viewport: samples covered only outside the viewport.
        row(
            "reach4",
            "viewport_cull_order",
            tris(vec![clip(2.0, 2.0), clip(4.0, 2.0), clip(52.2, 52.0)]),
        )
        .viewport([48.0, 48.0, 16.0, 16.0])
        .reaches(&[".... .R.. .... ...."]),
        // A render area of 40 over an attachment of 128: a full draw, and a quad past the area.
        row("reach4", "area40_full", tris(full()))
            .area(40)
            .reaches(&["RR.. RR.. .... ...."]),
        row(
            "reach4",
            "area40_quad_past",
            tris(quad(60.0, 60.0, 70.0, 70.0)),
        )
        .area(40)
        .reaches(&["R... .... .... ...."]),
        // Slices: a triangle in tile (1,1) rendered to slice 1 of 2.
        row(
            "reach4",
            "layered_slice1_t11",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .slices(2)
        .layer(1)
        .reaches(&[".... .... .... ....", ".... .R.. .... ...."]),
        row(
            "reach4",
            "layered_slice0_t11",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .slices(2)
        .layer(0)
        .reaches(&[".... .R.. .... ....", ".... .... .... ...."]),
        // No rasterization, and points of size 0 and 0.25.
        row(
            "reach4",
            "no_raster_t11",
            tris(quad(40.0, 40.0, 50.0, 50.0)),
        )
        .unrasterized()
        .reaches(&[".... .... .... ...."]),
        row("reach4", "point_size0_t11", points(vec![clip(40.5, 40.5)]))
            .size(0.0)
            .reaches(&[".... .R.. .... ...."]),
        row(
            "reach4",
            "point_size0_25_straddle",
            points(vec![clip(32.0, 40.5)]),
        )
        .size(0.25)
        .reaches(&[".... RR.. .... ...."]),
    ]
}

/// The triangle with a corner at `(cx + ex, cy + ey)` and legs `leg` along +x and +y (y down), as
/// our probe drew it.
fn corner(cx: f64, cy: f64, ex: f64, ey: f64, leg: f64) -> Vec<[f32; 4]> {
    vec![
        clip(cx + ex, cy + ey),
        clip(cx + ex + leg, cy + ey),
        clip(cx + ex, cy + ey + leg),
    ]
}

/// The sliver from `(x, x + 0.25)` to `(x + l, x + l + 0.25)`, its third vertex `(x, x + 0.35)`:
/// in a band of `y - x` that holds no pixel centre.
fn sliver(x: f64, l: f64) -> Vec<[f32; 4]> {
    vec![
        clip(x, x + 0.25),
        clip(x + l, x + l + 0.25),
        clip(x, x + 0.35),
    ]
}

/// The sliver from `(40, y0)` along `(w, h)`, of slope `p / q`, its third vertex `0.3 / q` below
/// its first, in a band of `y - (p / q) x` midway between the values pixel centres take.
fn band(w: f64, h: f64, p: f64, q: f64) -> Vec<[f32; 4]> {
    let m = p / q;
    let gap = 1.0 / q;
    let t = 0.3 * gap;
    let mid = 0.5 - p / (2.0 * q) + gap / 2.0;
    let mut r = (40.0 * m + mid - t / 2.0 - 40.0) % gap;
    if r < 0.0 {
        r += gap;
    }
    let y0 = 40.0 + r;
    vec![clip(40.0, y0), clip(40.0 + w, y0 + h), clip(40.0, y0 + t)]
}

/// The diagonal sliver from `(40, 40 + b)` to `(40 + l, 40 + l + b)`, its third vertex 0.1 below
/// its first.
fn diag(l: f64, b: f64) -> Vec<[f32; 4]> {
    vec![
        clip(40.0, 40.0 + b),
        clip(40.0 + l, 40.0 + l + b),
        clip(40.0, 40.0 + b + 0.1),
    ]
}

/// Our probes (an Apple M4 under macOS 27.0.1, each case in a process of its own, two rounds
/// alike): the small-primitive cull's precision (1/256), its space (the viewport's), what makes it
/// test a triangle's samples (a box of at most 12 pixel centres at 1 sample, 10 or 20 positions at
/// 2 or 4) and the default sample positions. Our own probes, built in their order and given the
/// answers of `REACH5` to `REACH9`.
pub(super) fn later_rows() -> Vec<Row> {
    let mut drawn: Vec<(&'static str, Vec<[f32; 4]>, u32, Option<[f64; 4]>)> = Vec::new();
    let mut add = |probe: &'static str, v: Vec<[f32; 4]>, samples: u32, vp: Option<[f64; 4]>| {
        drawn.push((probe, v, samples, vp));
    };
    for j in 1..=10 {
        let e = 0.6 * 2f64.powi(-j);
        add("reach5", corner(40.5, 40.5, e, e, 0.4), 1, None);
    }
    for j in [2, 4, 5, 8, 9] {
        let e = 0.6 * 2f64.powi(-j);
        add("reach5", corner(40.5, 40.5, e, 0.0, 0.4), 1, None);
        add("reach5", corner(40.5, 40.5, 0.0, e, 0.4), 1, None);
    }
    add("reach5", corner(40.5, 40.5, -0.01, -0.01, 0.4), 1, None);
    let quarter = vec![clip(2.4, 2.4), clip(3.6, 2.4), clip(2.4, 3.6)];
    add("reach5", quarter.clone(), 1, Some([32.0, 32.0, 32.0, 32.0]));
    add("reach5", quarter, 1, None);
    let shifted = vec![clip(40.1, 40.1), clip(40.4, 40.1), clip(40.1, 40.4)];
    add(
        "reach5",
        shifted.clone(),
        1,
        Some([0.25, 0.25, 128.0, 128.0]),
    );
    add("reach5", shifted, 1, None);
    let half = vec![clip(40.3, 40.3), clip(40.8, 40.3), clip(40.3, 40.8)];
    add("reach5", half, 1, Some([0.0, 0.0, 64.0, 64.0]));
    let positions: [(u32, &[(f64, f64)]); 2] = [
        (2, &[(0.75, 0.75), (0.25, 0.25)]),
        (
            4,
            &[
                (0.375, 0.125),
                (0.875, 0.375),
                (0.125, 0.625),
                (0.625, 0.875),
            ],
        ),
    ];
    for (n, at) in positions {
        for &(px, py) in at {
            let v = corner(40.0 + px - 0.03, 40.0 + py - 0.03, 0.0, 0.0, 0.1);
            add("reach5", v.clone(), n, None);
            if n == 4 {
                add("reach5", v, 1, None);
            }
        }
    }
    add("reach5", corner(40.47, 40.47, 0.0, 0.0, 0.1), 4, None);
    add("reach5", corner(40.22, 40.22, 0.0, 0.0, 0.1), 4, None);
    let lengths = [
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, 24.0, 32.0,
        48.0, 64.0, 80.0,
    ];
    for l in lengths {
        add("reach6", sliver(40.0, l), 1, None);
    }
    for l in [1.5, 2.0, 3.0, 4.0, 6.0] {
        add("reach6", sliver(41.0, l), 1, None);
        add("reach6", sliver(33.0, l), 1, None);
    }
    for l in [2.0, 4.0, 8.0] {
        let s = sliver(40.0, l);
        add("reach6", vec![s[0], s[2], s[1]], 1, None);
    }
    let diagonals = [
        (40.6, 3.2),
        (40.5, 3.0),
        (40.4, 3.0),
        (40.0, 3.1),
        (40.0, 3.2),
        (40.0, 3.3),
        (40.0, 3.4),
        (40.0, 3.49),
        (40.0, 3.0),
        (40.0, 3.5),
    ];
    for (x, l) in diagonals {
        add("reach7", sliver(x, l), 1, None);
    }
    for l in [1.0, 1.4, 1.5, 1.6, 1.75, 2.0, 3.0] {
        let steep = vec![
            clip(40.0, 80.1),
            clip(40.0 + l, 80.1 + 2.0 * l),
            clip(40.0, 80.3),
        ];
        let shallow = vec![
            clip(40.0, 20.35),
            clip(40.0 + 2.0 * l, 20.35 + l),
            clip(40.0, 20.65),
        ];
        add("reach7", steep, 1, None);
        add("reach7", shallow, 1, None);
    }
    let boxes = [
        (8.0, 1.0, 1.0, 8.0),
        (1.0, 8.0, 8.0, 1.0),
        (6.0, 1.2, 1.0, 5.0),
        (5.0, 1.5, 3.0, 10.0),
        (4.5, 2.7, 3.0, 5.0),
        (2.0, 5.2, 13.0, 5.0),
        (2.0, 4.6, 23.0, 10.0),
        (1.8, 4.5, 5.0, 2.0),
        (3.4, 3.4, 1.0, 1.0),
        (3.6, 3.6, 1.0, 1.0),
    ];
    for (w, h, p, q) in boxes {
        add("reach8", band(w, h, p, q), 1, None);
    }
    let counted = [
        (1.3, 9.0, 1.0),
        (1.8, 3.0, 1.0),
        (3.2, 1.0, 1.0),
        (4.3, 1.0, 2.0),
        (5.6, 1.0, 3.0),
        (11.6, 1.0, 9.0),
        (1.4, 9.0, 1.0),
        (12.6, 1.0, 9.0),
        (2.1, 3.0, 1.0),
        (6.6, 1.0, 3.0),
        (1.4, 10.0, 1.0),
        (13.6, 1.0, 10.0),
        (2.6, 2.0, 1.0),
        (4.6, 1.0, 2.0),
        (1.3, 11.0, 1.0),
        (2.4, 3.0, 1.0),
        (7.6, 1.0, 4.0),
        (1.4, 11.0, 1.0),
    ];
    for (w, p, q) in counted {
        add("reach9", band(w, w * (p / q), p, q), 1, None);
    }
    for l in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0] {
        add("reach9", diag(l, 0.05), 4, None);
    }
    for l in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
        add("reach9", diag(l, 0.25), 2, None);
    }
    for &(samples, x0, y0, w, h, t) in SLIVERS10 {
        let v = vec![clip(x0, y0), clip(x0 + w, y0 + h), clip(x0, y0 + t)];
        add("reach10", v, samples, None);
    }
    let answers: Vec<&(&str, &str)> = [REACH5, REACH6, REACH7, REACH8, REACH9, REACH10]
        .iter()
        .flat_map(|a| a.iter())
        .collect();
    assert_eq!(drawn.len(), answers.len(), "the later probes' cases");
    drawn
        .into_iter()
        .zip(answers)
        .map(|((probe, v, samples, vp), &(name, tiles))| {
            let mut r = row(probe, name, tris(v)).samples(samples);
            if let Some(vp) = vp {
                r = r.viewport(vp);
            }
            r.reaches(&[tiles])
        })
        .collect()
}

/// Our probe's answers, in the probe's order.
const REACH5: &[(&str, &str)] = &[
    ("cull_xy_e1", ".... .... .... ...."),
    ("cull_xy_e2", ".... .... .... ...."),
    ("cull_xy_e3", ".... .... .... ...."),
    ("cull_xy_e4", ".... .... .... ...."),
    ("cull_xy_e5", ".... .... .... ...."),
    ("cull_xy_e6", ".... .... .... ...."),
    ("cull_xy_e7", ".... .... .... ...."),
    ("cull_xy_e8", ".... .... .... ...."),
    ("cull_xy_e9", ".... .R.. .... ...."),
    ("cull_xy_e10", ".... .R.. .... ...."),
    ("cull_x_e2", ".... .... .... ...."),
    ("cull_y_e2", ".... .... .... ...."),
    ("cull_x_e4", ".... .... .... ...."),
    ("cull_y_e4", ".... .... .... ...."),
    ("cull_x_e5", ".... .... .... ...."),
    ("cull_y_e5", ".... .... .... ...."),
    ("cull_x_e8", ".... .... .... ...."),
    ("cull_y_e8", ".... .... .... ...."),
    ("cull_x_e9", ".... .R.. .... ...."),
    ("cull_y_e9", ".... .R.. .... ...."),
    ("cull_before", ".... .R.. .... ...."),
    ("vp_quarter_frame_covers", ".... .... .... ...."),
    ("vp_quarter_frame_covers_control", "R... .... .... ...."),
    ("vp_shift_viewport_covers", ".... .R.. .... ...."),
    ("vp_shift_control", ".... .... .... ...."),
    ("vp_half_frame_covers", ".... .... .... ...."),
    ("ms2_pos0", ".... .R.. .... ...."),
    ("ms2_pos1", ".... .R.. .... ...."),
    ("ms4_pos0", ".... .... .R.. .... .... .... .... ...."),
    ("ms1_pos4_0", ".... .... .... ...."),
    ("ms4_pos1", ".... .... .R.. .... .... .... .... ...."),
    ("ms1_pos4_1", ".... .... .... ...."),
    ("ms4_pos2", ".... .... .R.. .... .... .... .... ...."),
    ("ms1_pos4_2", ".... .... .... ...."),
    ("ms4_pos3", ".... .... .R.. .... .... .... .... ...."),
    ("ms1_pos4_3", ".... .... .... ...."),
    ("ms4_centre", ".... .... .... .... .... .... .... ...."),
    ("ms4_quarter", ".... .... .... .... .... .... .... ...."),
];
/// Our probe's answers, in the probe's order.
const REACH6: &[(&str, &str)] = &[
    ("sliver_40_L0.5", ".... .... .... ...."),
    ("sliver_40_L1.0", ".... .... .... ...."),
    ("sliver_40_L1.5", ".... .... .... ...."),
    ("sliver_40_L2.0", ".... .... .... ...."),
    ("sliver_40_L2.5", ".... .... .... ...."),
    ("sliver_40_L3.0", ".... .... .... ...."),
    ("sliver_40_L3.5", ".... .R.. .... ...."),
    ("sliver_40_L4.0", ".... .R.. .... ...."),
    ("sliver_40_L5.0", ".... .R.. .... ...."),
    ("sliver_40_L6.0", ".... .R.. .... ...."),
    ("sliver_40_L7.0", ".... .R.. .... ...."),
    ("sliver_40_L8.0", ".... .R.. .... ...."),
    ("sliver_40_L10.0", ".... .R.. .... ...."),
    ("sliver_40_L12.0", ".... .R.. .... ...."),
    ("sliver_40_L16.0", ".... .R.. .... ...."),
    ("sliver_40_L24.0", ".... .R.. .RR. ...."),
    ("sliver_40_L32.0", ".... .R.. .RR. ...."),
    ("sliver_40_L48.0", ".... .R.. .RR. ...."),
    ("sliver_40_L64.0", ".... .R.. .RR. ..RR"),
    ("sliver_40_L80.0", ".... .R.. .RR. ..RR"),
    ("sliver_41_L1.5", ".... .... .... ...."),
    ("sliver_33_L1.5", ".... .... .... ...."),
    ("sliver_41_L2.0", ".... .... .... ...."),
    ("sliver_33_L2.0", ".... .... .... ...."),
    ("sliver_41_L3.0", ".... .... .... ...."),
    ("sliver_33_L3.0", ".... .... .... ...."),
    ("sliver_41_L4.0", ".... .R.. .... ...."),
    ("sliver_33_L4.0", ".... .R.. .... ...."),
    ("sliver_41_L6.0", ".... .R.. .... ...."),
    ("sliver_33_L6.0", ".... .R.. .... ...."),
    ("sliver_40_L2.0_ccw", ".... .... .... ...."),
    ("sliver_40_L4.0_ccw", ".... .R.. .... ...."),
    ("sliver_40_L8.0_ccw", ".... .R.. .... ...."),
];
/// Our probe's answers, in the probe's order.
const REACH7: &[(&str, &str)] = &[
    ("diag_40.6_L3.2", ".... .... .... ...."),
    ("diag_40.5_L3.0", ".... .... .... ...."),
    ("diag_40.4_L3.0", ".... .... .... ...."),
    ("diag_40.0_L3.1", ".... .... .... ...."),
    ("diag_40.0_L3.2", ".... .... .... ...."),
    ("diag_40.0_L3.3", ".... .... .... ...."),
    ("diag_40.0_L3.4", ".... .... .... ...."),
    ("diag_40.0_L3.49", ".... .... .... ...."),
    ("diag_40.0_L3.0", ".... .... .... ...."),
    ("diag_40.0_L3.5", ".... .R.. .... ...."),
    ("steep_L1.0", ".... .... .... ...."),
    ("shallow_L1.0", ".... .... .... ...."),
    ("steep_L1.4", ".... .... .... ...."),
    ("shallow_L1.4", ".... .... .... ...."),
    ("steep_L1.5", ".... .... .... ...."),
    ("shallow_L1.5", ".... .... .... ...."),
    ("steep_L1.6", ".... .... .... ...."),
    ("shallow_L1.6", ".... .... .... ...."),
    ("steep_L1.75", ".... .... .... ...."),
    ("shallow_L1.75", ".... .... .... ...."),
    ("steep_L2.0", ".... .... .... ...."),
    ("shallow_L2.0", ".... .... .... ...."),
    ("steep_L3.0", ".... .... .R.. ...."),
    ("shallow_L3.0", ".R.. .... .... ...."),
];
/// Our probe's answers, in the probe's order.
const REACH8: &[(&str, &str)] = &[
    ("band_8.0x1.0", ".... .... .... ...."),
    ("band_1.0x8.0", ".... .... .... ...."),
    ("band_6.0x1.2", ".... .... .... ...."),
    ("band_5.0x1.5", ".... .... .... ...."),
    ("band_4.5x2.7", ".... .R.. .... ...."),
    ("band_2.0x5.2", ".... .... .... ...."),
    ("band_2.0x4.6", ".... .... .... ...."),
    ("band_1.8x4.5", ".... .... .... ...."),
    ("band_3.4x3.4", ".... .... .... ...."),
    ("band_3.6x3.6", ".... .R.. .... ...."),
];
/// Our probe's answers, in the probe's order.
const REACH9: &[(&str, &str)] = &[
    ("n12_w1.3_m9_1", ".... .... .... ...."),
    ("n12_w1.8_m3_1", ".... .... .... ...."),
    ("n12_w3.2_m1_1", ".... .... .... ...."),
    ("n12_w4.3_m1_2", ".... .... .... ...."),
    ("n12_w5.6_m1_3", ".... .... .... ...."),
    ("n12_w11.6_m1_9", ".... .... .... ...."),
    ("n13_w1.4_m9_1", ".... .R.. .... ...."),
    ("n13_w12.6_m1_9", ".... .R.. .... ...."),
    ("n14_w2.1_m3_1", ".... .R.. .... ...."),
    ("n14_w6.6_m1_3", ".... .R.. .... ...."),
    ("n14_w1.4_m10_1", ".... .R.. .... ...."),
    ("n14_w13.6_m1_10", ".... .R.. .... ...."),
    ("n15_w2.6_m2_1", ".... .R.. .... ...."),
    ("n15_w4.6_m1_2", ".... .R.. .... ...."),
    ("n15_w1.3_m11_1", ".... .R.. .... ...."),
    ("n16_w2.4_m3_1", ".... .R.. .... ...."),
    ("n16_w7.6_m1_4", ".... .R.. .... ...."),
    ("n16_w1.4_m11_1", ".... .R.. .... ...."),
    ("ms4_diag_L1.0", ".... .... .... .... .... .... .... ...."),
    ("ms4_diag_L1.25", ".... .... .... .... .... .... .... ...."),
    ("ms4_diag_L1.5", ".... .... .... .... .... .... .... ...."),
    ("ms4_diag_L1.75", ".... .... .... .... .... .... .... ...."),
    ("ms4_diag_L2.0", ".... .... .... .... .... .... .... ...."),
    ("ms4_diag_L2.25", ".... .... .... .... .... .... .... ...."),
    ("ms4_diag_L2.5", ".... .... .R.. .... .... .... .... ...."),
    ("ms4_diag_L3.0", ".... .... .R.. .... .... .... .... ...."),
    ("ms4_diag_L3.5", ".... .... .R.. .... .... .... .... ...."),
    ("ms4_diag_L4.0", ".... .... .R.. .... .... .... .... ...."),
    ("ms2_diag_L1.5", ".... .... .... ...."),
    ("ms2_diag_L2.0", ".... .... .... ...."),
    ("ms2_diag_L2.5", ".... .R.. .... ...."),
    ("ms2_diag_L3.0", ".... .R.. .... ...."),
    ("ms2_diag_L3.5", ".... .R.. .... ...."),
    ("ms2_diag_L4.0", ".... .R.. .... ...."),
];

/// Our probe's slivers, in the probe's order: at `samples` samples, the triangle
/// `(x0, y0)`, `(x0 + w, y0 + h)`, `(x0, y0 + t)` (pixels, y down), holding no sample position,
/// its closed box holding 11 to 14 positions at 2 samples and 21 to 24 at 4, in several slopes.
const SLIVERS10: &[(u32, f64, f64, f64, f64, f64)] = &[
    (2, 40.4375, 40.3, 1.75, 3.5, 0.15),
    (2, 40.3125, 40.36875, 3.46875, 1.734375, 0.075),
    (2, 40.0, 40.849999999999994, 1.28125, 3.84375, 0.3),
    (2, 40.8125, 40.22083333366666, 3.46875, 1.15625, 0.1),
    (2, 40.0, 40.35, 2.421875, 2.421875, 0.3),
    (2, 40.1875, 40.8, 1.59375, 3.1875, 0.15),
    (2, 40.0, 40.2125, 3.125, 1.5625, 0.075),
    (2, 40.0, 40.849999999999994, 1.3125, 3.9375, 0.3),
    (
        2,
        40.1875,
        40.01250000033333,
        3.78125,
        1.2604166666666665,
        0.1,
    ),
    (2, 40.3125, 40.6625, 2.46875, 2.46875, 0.3),
    (2, 40.1875, 40.412499999999994, 1.453125, 4.359375, 0.3),
    (2, 40.0, 40.28333333366667, 4.46875, 1.4895833333333333, 0.1),
    (2, 40.0, 40.425, 1.78125, 3.5625, 0.15),
    (2, 40.0, 40.2125, 3.28125, 1.640625, 0.075),
    (2, 40.0, 40.849999999999994, 1.484375, 4.453125, 0.3),
    (
        2,
        40.1875,
        40.01250000033333,
        4.09375,
        1.3645833333333333,
        0.1,
    ),
    (4, 40.6875, 40.4875, 1.390625, 4.171875, 0.15),
    (
        4,
        40.0625,
        40.162499999783336,
        4.09375,
        1.3645833333333333,
        0.0500000001,
    ),
    (4, 40.25, 40.175, 2.40625, 2.40625, 0.15),
    (4, 40.3125, 40.5125, 1.59375, 3.1875, 0.22499999999999998),
    (4, 40.1875, 40.2875, 3.21875, 1.609375, 0.11249999999999999),
    (4, 40.1875, 40.9875, 1.390625, 4.171875, 0.15),
    (4, 40.0, 40.925, 2.40625, 2.40625, 0.15),
    (4, 40.0625, 40.0125, 1.59375, 3.1875, 0.22499999999999998),
    (4, 40.6875, 40.0375, 3.21875, 1.609375, 0.11249999999999999),
    (4, 40.25, 40.175, 2.484375, 2.484375, 0.15),
    (4, 40.0625, 40.0125, 1.703125, 3.40625, 0.22499999999999998),
    (4, 40.0, 40.19375, 3.40625, 1.703125, 0.11249999999999999),
    (
        4,
        40.4375,
        40.287499999783336,
        4.078125,
        1.359375,
        0.0500000001,
    ),
];

/// Our probe's answers, in the probe's order.
const REACH10: &[(&str, &str)] = &[
    ("ms2_c11_m2_1", ".... .... .... ...."),
    ("ms2_c11_m1_2", ".... .... .... ...."),
    ("ms2_c11_m3_1", ".... .... .... ...."),
    ("ms2_c11_m1_3", ".... .... .... ...."),
    ("ms2_c12_m1_1", ".... .... .... ...."),
    ("ms2_c12_m2_1", ".... .... .... ...."),
    ("ms2_c12_m1_2", ".... .... .... ...."),
    ("ms2_c12_m3_1", ".... .... .... ...."),
    ("ms2_c12_m1_3", ".... .... .... ...."),
    ("ms2_c13_m1_1", ".... .R.. .... ...."),
    ("ms2_c13_m3_1", ".... .R.. .... ...."),
    ("ms2_c13_m1_3", ".... .R.. .... ...."),
    ("ms2_c14_m2_1", ".... .R.. .... ...."),
    ("ms2_c14_m1_2", ".... .R.. .... ...."),
    ("ms2_c14_m3_1", ".... .R.. .... ...."),
    ("ms2_c14_m1_3", ".... .R.. .... ...."),
    ("ms4_c21_m3_1", ".... .... .... .... .... .... .... ...."),
    ("ms4_c21_m1_3", ".... .... .... .... .... .... .... ...."),
    ("ms4_c22_m1_1", ".... .... .... .... .... .... .... ...."),
    ("ms4_c22_m2_1", ".... .... .... .... .... .... .... ...."),
    ("ms4_c22_m1_2", ".... .... .... .... .... .... .... ...."),
    ("ms4_c22_m3_1", ".... .... .... .... .... .... .... ...."),
    ("ms4_c23_m1_1", ".... .... .... .... .... .... .... ...."),
    ("ms4_c23_m2_1", ".... .... .... .... .... .... .... ...."),
    ("ms4_c23_m1_2", ".... .... .... .... .... .... .... ...."),
    ("ms4_c24_m1_1", ".... .... .... .... .... .... .... ...."),
    ("ms4_c24_m2_1", ".... .... .... .... .... .... .... ...."),
    ("ms4_c24_m1_2", ".... .... .... .... .... .... .... ...."),
    ("ms4_c24_m1_3", ".... .... .... .... .... .... .... ...."),
];
