use super::*;
use AirScalar::*;

const FRAG_LL: &str = r#"
!air.fragment = !{!15}
!15 = !{ptr @F, !16, !18}
!16 = !{!17}
!17 = !{!"air.render_target", i32 0, i32 0}
!18 = !{!19, !20, !21, !22, !23, !24}
!19 = !{i32 0, !"air.position", !"air.center"}
!20 = !{i32 1, !"air.fragment_input", !"generated", !"air.arg_type_name", !"float2", !"air.arg_name", !"texCoord"}
!21 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"texture2d<float, sample>"}
!22 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 5, i32 1}
!23 = !{i32 4, !"air.point_coord", !"air.arg_type_name", !"float2", !"air.arg_name", !"pointCoord"}
!24 = !{i32 5, !"air.front_facing", !"air.arg_type_name", !"bool", !"air.arg_name", !"front"}
"#;

mod fragment;
mod function_constant_bindings;
mod imageblocks;
mod kernel;
mod role_wrappers;
mod types_layout;
mod vertex;
