use super::*;

fn example() -> AuthoredCase {
    AuthoredCase {
        air_sha256: "11".repeat(32),
        case_id: String::new(),
        name: "writes-one-word".into(),
        entry: "kernel_main".into(),
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        rationale: Some("documentation does not identify semantics".into()),
        authored_by: Some("test".into()),
        render_pass_rate_map: None,
        raster_sample_count: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    }
}

mod buffers;
mod depth_stencil_state;
mod function_tables;
mod identity;
mod indirect_commands;
mod mesh_draws;
mod mips;
mod out_of_bounds;
mod raster_sample_count;
mod render_pass_rate_map;
mod sparse;
mod stages;
mod textures;
