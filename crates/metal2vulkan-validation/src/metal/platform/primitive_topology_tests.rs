use super::*;

#[test]
fn authored_draw_primitives_map_to_their_pipeline_topology_class() {
    use crate::case::Primitive;

    assert_eq!(
        metal_primitive_topology_class(Primitive::Point),
        MTLPrimitiveTopologyClass::Point
    );
    for primitive in [Primitive::Line, Primitive::LineStrip] {
        assert_eq!(
            metal_primitive_topology_class(primitive),
            MTLPrimitiveTopologyClass::Line
        );
    }
    for primitive in [Primitive::Triangle, Primitive::TriangleStrip] {
        assert_eq!(
            metal_primitive_topology_class(primitive),
            MTLPrimitiveTopologyClass::Triangle
        );
    }
}

#[test]
fn authored_draw_primitives_map_to_metals_primitive_types() {
    use crate::case::Primitive;

    for (primitive, metal) in [
        (Primitive::Point, MTLPrimitiveType::Point),
        (Primitive::Line, MTLPrimitiveType::Line),
        (Primitive::LineStrip, MTLPrimitiveType::LineStrip),
        (Primitive::Triangle, MTLPrimitiveType::Triangle),
        (Primitive::TriangleStrip, MTLPrimitiveType::TriangleStrip),
    ] {
        assert_eq!(metal_primitive(primitive), metal, "{primitive:?}");
    }
}
