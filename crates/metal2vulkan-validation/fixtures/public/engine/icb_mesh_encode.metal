// Owned fixture: kernels that encode mesh draws into an indirect command buffer of render
// commands, which the mesh render pipelines of `icb_mesh_template` run. Each takes its command
// buffer, and the render pipeline states its commands set, in an argument buffer at buffer 0, and
// the buffer the draws write at buffer 1.
//
// - `icb_enc_groups` sets command 0's draw of 3 by 2 threadgroups of 4 threads.
// - `icb_enc_threads` sets command 0's mesh buffer 0 and a draw of 12 by 4 threads in
//   threadgroups of 4 by 2.
// - `icb_enc_reset` sets two commands' draws and resets the second.
// - `icb_enc_stages` sets command 0's writer state, its mesh buffer and a draw of one
//   threadgroup, and command 1's barrier, reader state, object buffer and a draw of one thread.
//
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Cmds {
    command_buffer cmds [[id(0)]];
};

kernel void icb_enc_groups(device const Cmds& e [[buffer(0)]],
                           device uint4* marks [[buffer(1)]]) {
    render_command c(e.cmds, 0);
    c.draw_mesh_threadgroups(uint3(3, 2, 1), uint3(1, 1, 1), uint3(4, 1, 1));
}

kernel void icb_enc_threads(device const Cmds& e [[buffer(0)]],
                            device uint4* marks [[buffer(1)]]) {
    render_command c(e.cmds, 0);
    c.set_mesh_buffer(marks, 0);
    c.draw_mesh_threads(uint3(12, 4, 1), uint3(1, 1, 1), uint3(4, 2, 1));
}

kernel void icb_enc_reset(device const Cmds& e [[buffer(0)]],
                          device uint4* marks [[buffer(1)]]) {
    render_command a(e.cmds, 0);
    a.draw_mesh_threadgroups(uint3(3, 2, 1), uint3(1, 1, 1), uint3(4, 1, 1));
    render_command b(e.cmds, 1);
    b.draw_mesh_threads(uint3(12, 4, 1), uint3(1, 1, 1), uint3(4, 2, 1));
    b.reset();
}

struct Stages {
    command_buffer cmds [[id(0)]];
    render_pipeline_state writer_pso [[id(1)]];
    render_pipeline_state reader_pso [[id(2)]];
};

kernel void icb_enc_stages(device const Stages& e [[buffer(0)]],
                           device uint4* ctl [[buffer(1)]]) {
    render_command w(e.cmds, 0);
    w.set_render_pipeline_state(e.writer_pso);
    w.set_mesh_buffer(ctl, 0);
    w.draw_mesh_threadgroups(uint3(1, 1, 1), uint3(1, 1, 1), uint3(1, 1, 1));
    render_command r(e.cmds, 1);
    r.set_barrier();
    r.set_render_pipeline_state(e.reader_pso);
    r.set_object_buffer(ctl, 0);
    r.draw_mesh_threads(uint3(1, 1, 1), uint3(1, 1, 1), uint3(1, 1, 1));
}
