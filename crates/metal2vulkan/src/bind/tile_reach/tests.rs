mod assembly;
mod device;
mod kernel;
mod measured;
mod points;
mod reference;

use reference::{every_tile, reach, Cull, Flags, Primitive, Raster, Winding};
use std::collections::BTreeSet;

/// The probes' attachment side, in pixels.
const FRAME: u32 = 128;

/// The clip-space vertex the probes drew at pixel `(x, y)` of the frame (y down) at depth 0.5.
fn clip(x: f64, y: f64) -> [f32; 4] {
    clip_z(x, y, 0.5)
}

/// The clip-space vertex at pixel `(x, y)` and depth `z`, as the probes made it: each
/// coordinate in `f64`, then rounded to `f32`.
fn clip_z(x: f64, y: f64, z: f64) -> [f32; 4] {
    clip_in([FRAME; 2], x, y, z)
}

/// The clip-space vertex at pixel `(x, y)` and depth `z` of a frame of `frame` pixels, as the
/// probes made it.
fn clip_in(frame: [u32; 2], x: f64, y: f64, z: f64) -> [f32; 4] {
    let [w, h] = frame.map(f64::from);
    [
        (2.0 * x / w - 1.0) as f32,
        (1.0 - 2.0 * y / h) as f32,
        z as f32,
        1.0,
    ]
}

/// The two triangles of the rectangle `[x0, x1] x [y0, y1]`, at depth 0.5.
fn quad(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<[f32; 4]> {
    quad_z(x0, y0, x1, y1, 0.5)
}

/// The two triangles of the rectangle `[x0, x1] x [y0, y1]` at depth `z`.
fn quad_z(x0: f64, y0: f64, x1: f64, y1: f64, z: f64) -> Vec<[f32; 4]> {
    [(x0, y0), (x1, y0), (x0, y1), (x1, y0), (x1, y1), (x0, y1)]
        .into_iter()
        .map(|(x, y)| clip_z(x, y, z))
        .collect()
}

/// The probes' triangle over the whole frame.
fn full() -> Vec<[f32; 4]> {
    vec![
        clip(-128.0, -128.0),
        clip(384.0, -128.0),
        clip(-128.0, 384.0),
    ]
}

/// What a probe's draw assembled its vertices into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Topology {
    /// A triangle list.
    Triangles,
    /// A line list.
    Lines,
    /// A point list.
    Points,
}

/// A probe's draw call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Draw {
    /// Drawn directly.
    Direct,
    /// Drawn from an indirect buffer's arguments.
    Indirect,
    /// Drawn with no instances.
    NoInstances,
    /// By a pipeline that does not rasterize.
    Unrasterized,
    /// No draw call.
    None,
}

/// Another attachment of a probe's pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Other {
    /// None.
    None,
    /// A color attachment loaded `Clear` and stored.
    ColorClearStore,
    /// A color attachment loaded `Clear`, its store `DontCare`.
    ColorClearDontCare,
    /// A depth attachment loaded `Clear` and stored.
    DepthClearStore,
    /// A stencil attachment loaded `Clear` and stored.
    StencilClearStore,
}

/// A measured case: its pass and draw, and the tiles the M4 reached.
struct Row {
    probe: &'static str,
    name: &'static str,
    topology: Topology,
    vertices: Vec<[f32; 4]>,
    size: f32,
    raster: Raster,
    draw: Draw,
    other: Other,
    slices: u32,
    layer: u32,
    reached: Vec<String>,
}

/// A case of `probe` drawing `drawn` over the whole frame.
fn row(probe: &'static str, name: &'static str, drawn: (Topology, Vec<[f32; 4]>)) -> Row {
    Row {
        probe,
        name,
        topology: drawn.0,
        vertices: drawn.1,
        size: 1.0,
        raster: Raster::over([FRAME; 2]),
        draw: Draw::Direct,
        other: Other::None,
        slices: 1,
        layer: 0,
        reached: Vec::new(),
    }
}

fn tris(v: Vec<[f32; 4]>) -> (Topology, Vec<[f32; 4]>) {
    (Topology::Triangles, v)
}

fn lines(v: Vec<[f32; 4]>) -> (Topology, Vec<[f32; 4]>) {
    (Topology::Lines, v)
}

fn points(v: Vec<[f32; 4]>) -> (Topology, Vec<[f32; 4]>) {
    (Topology::Points, v)
}

impl Row {
    fn size(mut self, size: f64) -> Row {
        self.size = size as f32;
        self
    }

    fn cull(mut self, cull: Cull) -> Row {
        self.raster.cull = cull;
        self
    }

    fn scissor(mut self, scissor: [u32; 4]) -> Row {
        self.raster.scissor = scissor;
        self
    }

    fn viewport(mut self, viewport: [f64; 4]) -> Row {
        self.raster.viewport = viewport.map(|c| c as f32);
        self
    }

    fn clamp(mut self) -> Row {
        self.raster.clamp = true;
        self
    }

    fn samples(mut self, samples: u32) -> Row {
        self.raster.samples = samples;
        self
    }

    /// A render area (the descriptor's target size) of `side` over the frame: the default
    /// viewport and scissor follow it.
    fn area(mut self, side: u32) -> Row {
        let samples = self.raster.samples;
        self.raster = Raster::over([side; 2]);
        self.raster.samples = samples;
        self
    }

    fn slices(mut self, slices: u32) -> Row {
        self.slices = slices;
        self
    }

    fn layer(mut self, layer: u32) -> Row {
        self.layer = layer;
        self
    }

    fn unrasterized(mut self) -> Row {
        self.draw = Draw::Unrasterized;
        self
    }

    fn draw(mut self, draw: Draw) -> Row {
        self.draw = draw;
        self
    }

    fn other(mut self, other: Other) -> Row {
        self.other = other;
        self
    }

    /// A fragment function that discards: the reach does not follow it.
    fn discards(self) -> Row {
        self
    }

    fn reaches(mut self, reached: &[&str]) -> Row {
        self.reached = reached.iter().map(|s| s.to_string()).collect();
        self
    }

    /// The primitives the draw assembled.
    fn primitives(&self) -> Vec<Primitive> {
        let v = &self.vertices;
        match self.topology {
            Topology::Triangles => v
                .as_chunks::<3>()
                .0
                .iter()
                .map(|t| Primitive::Triangle([t[0], t[1], t[2]]))
                .collect(),
            Topology::Lines => v
                .as_chunks::<2>()
                .0
                .iter()
                .map(|l| Primitive::Line([l[0], l[1]]))
                .collect(),
            Topology::Points => v.iter().map(|&p| Primitive::Point(p, self.size)).collect(),
        }
    }

    /// The tiles the rule reaches, slice by slice: every tile of each where another attachment
    /// is cleared and stored in a pass with a draw call, else the tiles of the slice the draw
    /// renders (a layer past the pass's renders its first) its primitives reach.
    fn answer(&self) -> (Vec<BTreeSet<[u32; 2]>>, Flags) {
        let mut flags = Flags::default();
        let called = self.draw != Draw::None;
        let cleared = matches!(
            self.other,
            Other::ColorClearStore | Other::DepthClearStore | Other::StencilClearStore
        );
        let drawn = matches!(self.draw, Draw::Direct | Draw::Indirect);
        let slice = if self.layer < self.slices {
            self.layer
        } else {
            0
        };
        let slices = (0..self.slices)
            .map(|s| {
                if cleared && called {
                    return every_tile(&self.raster);
                }
                let mut tiles = BTreeSet::new();
                if drawn && s == slice {
                    for prim in self.primitives() {
                        tiles.extend(reach(&self.raster, &prim, &mut flags));
                    }
                }
                tiles
            })
            .collect();
        (slices, flags)
    }

    /// The tiles the M4 reached, slice by slice.
    fn measured(&self) -> Vec<BTreeSet<[u32; 2]>> {
        self.reached
            .iter()
            .map(|slice| {
                let mut tiles = BTreeSet::new();
                for (y, line) in slice.split(' ').enumerate() {
                    for (x, c) in line.chars().enumerate() {
                        if c == 'R' {
                            tiles.insert([x as u32, y as u32]);
                        }
                    }
                }
                tiles
            })
            .collect()
    }
}

/// Every measured case reaches the tiles the M4 reached, and only the case across the near plane is
/// flagged (its reach that of the clipped polygon).
#[test]
fn the_rule_reaches_every_measured_answer() {
    let mut rows = measured::rows();
    assert_eq!(rows.len(), 111, "the probes' cases");
    rows.extend(measured::later_rows());
    rows.extend(points::rows());
    let mut wrong = Vec::new();
    for row in &rows {
        let (answer, flags) = row.answer();
        let near_far = row.name == "near_cross";
        let expected_flags = Flags {
            near_far,
            unmeasured: false,
        };
        if answer != row.measured() || flags != expected_flags {
            wrong.push(format!(
                "{}/{}: reached {:?} ({flags:?}), measured {:?}",
                row.probe,
                row.name,
                answer,
                row.measured()
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Winding and the cull mode: a clockwise triangle (y down) faces front by default, and the
/// other winding turns it.
#[test]
fn the_cull_follows_the_winding() {
    let t = Primitive::Triangle([clip(40.0, 40.0), clip(50.0, 40.0), clip(40.0, 50.0)]);
    let mut raster = Raster::over([FRAME; 2]);
    let reached = |raster: &Raster| !reach(raster, &t, &mut Flags::default()).is_empty();
    for (cull, winding, drawn) in [
        (Cull::Back, Winding::Clockwise, true),
        (Cull::Front, Winding::Clockwise, false),
        (Cull::Back, Winding::CounterClockwise, false),
        (Cull::Front, Winding::CounterClockwise, true),
        (Cull::None, Winding::CounterClockwise, true),
    ] {
        raster.cull = cull;
        raster.winding = winding;
        assert_eq!(reached(&raster), drawn, "{cull:?} {winding:?}");
    }
}

/// A vertex past the coordinates measured, a line of no length, and a point past the depth range
/// are flagged unmeasured; a triangle across the far plane is flagged near-far.
#[test]
fn unmeasured_cases_are_flagged() {
    let raster = Raster::over([FRAME; 2]);
    let flags = |prim: Primitive| {
        let mut flags = Flags::default();
        reach(&raster, &prim, &mut flags);
        flags
    };
    let unmeasured = Flags {
        near_far: false,
        unmeasured: true,
    };
    assert_eq!(
        flags(Primitive::Triangle([
            clip(-1e7, 40.0),
            clip(1e7, 40.0),
            clip(0.0, 1e7)
        ])),
        unmeasured
    );
    assert_eq!(
        flags(Primitive::Line([clip(40.0, 40.0), clip(40.0, 40.0)])),
        unmeasured
    );
    assert_eq!(
        flags(Primitive::Point(clip_z(40.0, 40.0, 1.5), 1.0)),
        unmeasured
    );
    assert_eq!(
        flags(Primitive::Triangle([
            clip(40.0, 40.0),
            clip(40.0, 60.0),
            clip_z(80.0, 50.0, 1.5)
        ])),
        Flags {
            near_far: true,
            unmeasured: false
        }
    );
}
