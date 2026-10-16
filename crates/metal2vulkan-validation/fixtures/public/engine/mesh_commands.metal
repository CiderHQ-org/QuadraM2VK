// Owned fixture: kernels that encode mesh draws into an indirect
// command buffer of render commands, for the runtime lane and Metal probe 6, whose kernels these
// are. Each takes its command buffer, and the render pipeline states its commands set, in an
// argument buffer at buffer 0. `enc_own`'s thread 0 sets command 0's object pipeline state, its
// object buffer `ctl` and a draw of 3 object threadgroups of 4 threads, mesh threadgroups of 2;
// thread 1 sets command 1's grid pipeline state and a draw of 12 by 4 threads in mesh
// threadgroups of 4 by 2 (the pipelines `mesh_draws`' functions). `enc_barrier` sets command 0's
// writer state, its mesh buffer `ctl` and a draw of one threadgroup, and command 1's reader
// state, its object buffer `ctl`, a draw of one threadgroup, and its barrier where `barrier` is
// not 0 (the pipelines `mesh_barrier`'s functions). `enc_groups`, `enc_threads` and `enc_draw`
// set command 0's draw of one kind; `enc_reset` sets two commands' draws and resets the second.
// `vtx`, a vertex function for a pipeline with rasterization disabled, marks the vertices it
// runs. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Own {
    command_buffer cmds [[id(0)]];
    render_pipeline_state object_pso [[id(1)]];
    render_pipeline_state grid_pso [[id(2)]];
};

kernel void enc_own(device const Own& e [[buffer(0)]],
                    device uint4* ctl [[buffer(1)]],
                    uint tid [[thread_position_in_grid]]) {
    render_command c(e.cmds, tid);
    if (tid == 0) {
        c.set_render_pipeline_state(e.object_pso);
        c.set_object_buffer(ctl, 0);
        c.draw_mesh_threadgroups(uint3(3, 1, 1), uint3(4, 1, 1), uint3(2, 1, 1));
    } else {
        c.set_render_pipeline_state(e.grid_pso);
        c.draw_mesh_threads(uint3(12, 4, 1), uint3(1, 1, 1), uint3(4, 2, 1));
    }
}

struct Bar {
    command_buffer cmds [[id(0)]];
    render_pipeline_state writer_pso [[id(1)]];
    render_pipeline_state reader_pso [[id(2)]];
};

kernel void enc_barrier(device const Bar& e [[buffer(0)]],
                        device uint4* ctl [[buffer(1)]],
                        constant uint& barrier [[buffer(2)]]) {
    render_command w(e.cmds, 0);
    w.set_render_pipeline_state(e.writer_pso);
    w.set_mesh_buffer(ctl, 0);
    w.draw_mesh_threadgroups(uint3(1, 1, 1), uint3(1, 1, 1), uint3(1, 1, 1));
    render_command r(e.cmds, 1);
    if (barrier != 0) {
        r.set_barrier();
    }
    r.set_render_pipeline_state(e.reader_pso);
    r.set_object_buffer(ctl, 0);
    r.draw_mesh_threadgroups(uint3(1, 1, 1), uint3(1, 1, 1), uint3(1, 1, 1));
}

struct Kind {
    command_buffer cmds [[id(0)]];
};

kernel void enc_groups(device const Kind& e [[buffer(0)]]) {
    render_command c(e.cmds, 0);
    c.draw_mesh_threadgroups(uint3(3, 2, 1), uint3(1, 1, 1), uint3(4, 1, 1));
}

kernel void enc_threads(device const Kind& e [[buffer(0)]]) {
    render_command c(e.cmds, 0);
    c.draw_mesh_threads(uint3(12, 4, 1), uint3(1, 1, 1), uint3(4, 2, 1));
}

kernel void enc_draw(device const Kind& e [[buffer(0)]]) {
    render_command c(e.cmds, 0);
    c.draw_primitives(primitive_type::point, 0, 4, 1, 0);
}

// Two grid commands, then the second reset.
kernel void enc_reset(device const Kind& e [[buffer(0)]]) {
    render_command a(e.cmds, 0);
    a.draw_mesh_threadgroups(uint3(3, 2, 1), uint3(1, 1, 1), uint3(4, 1, 1));
    render_command b(e.cmds, 1);
    b.draw_mesh_threads(uint3(12, 4, 1), uint3(1, 1, 1), uint3(4, 2, 1));
    b.reset();
}

vertex void vtx(uint vid [[vertex_id]], device uint* marks [[buffer(2)]]) {
    marks[vid] = 0xd0000000u | vid;
}
