; Owned synthetic fixture: indirect-command buffer setters as AIR 2.5 declares them.
; Not derived from a third-party metallib.
;
; macOS 12's and 13's Metal compiler (AIR 2.5, Metal 2.4) declares an indirect command's kernel
; and vertex buffer setters, `air.set_kernel_buffer_compute_command` and
; `air.set_vertex_buffer_render_command`, without the `i64` attribute stride that today's compiler
; passes before the buffer index. They are MSL's plain `set_kernel_buffer(buffer, index)` and
; `set_vertex_buffer(buffer, index)`, which today's compiler spells with a stride of all ones (the
; pipeline's own stride). `encode_plain_compute_commands` sets a device buffer and a constant one
; on the compute command at its thread's index; `encode_plain_render_commands` sets them as vertex
; buffers on a render command. Each writes an output the command does not depend on.
source_filename = "kernel_icb_encode_plain_setters_air_2_5.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx13.0.0"

%struct.Commands = type { %"struct.metal::command_buffer" }
%"struct.metal::command_buffer" = type { ptr addrspace(1) }

define void @encode_plain_compute_commands(ptr addrspace(1) %encoder, ptr addrspace(1) %data, ptr addrspace(2) %params, ptr addrspace(1) %out, i32 %tid) {
  %slot = getelementptr inbounds %struct.Commands, ptr addrspace(1) %encoder, i64 0, i32 0, i32 0
  %commands = load ptr addrspace(1), ptr addrspace(1) %slot, align 8
  call void @air.set_kernel_buffer_compute_command.p1i8(ptr addrspace(1) %commands, i32 %tid, ptr addrspace(1) %data, i32 0)
  call void @air.set_kernel_buffer_compute_command.p2i8(ptr addrspace(1) %commands, i32 %tid, ptr addrspace(2) %params, i32 1)
  %m = mul i32 %tid, 3
  %v = add i32 %m, 7
  %idx = zext i32 %tid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define void @encode_plain_render_commands(ptr addrspace(1) %encoder, ptr addrspace(1) %vertices, ptr addrspace(2) %params, ptr addrspace(1) %out, i32 %tid) {
  %slot = getelementptr inbounds %struct.Commands, ptr addrspace(1) %encoder, i64 0, i32 0, i32 0
  %commands = load ptr addrspace(1), ptr addrspace(1) %slot, align 8
  call void @air.set_vertex_buffer_render_command.p1i8(ptr addrspace(1) %commands, i32 %tid, ptr addrspace(1) %vertices, i32 0)
  call void @air.set_vertex_buffer_render_command.p2i8(ptr addrspace(1) %commands, i32 %tid, ptr addrspace(2) %params, i32 1)
  %m = mul i32 %tid, 5
  %v = add i32 %m, 2
  %idx = zext i32 %tid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

declare void @air.set_kernel_buffer_compute_command.p1i8(ptr addrspace(1), i32, ptr addrspace(1), i32)

declare void @air.set_kernel_buffer_compute_command.p2i8(ptr addrspace(1), i32, ptr addrspace(2), i32)

declare void @air.set_vertex_buffer_render_command.p1i8(ptr addrspace(1), i32, ptr addrspace(1), i32)

declare void @air.set_vertex_buffer_render_command.p2i8(ptr addrspace(1), i32, ptr addrspace(2), i32)

!air.kernel = !{!0, !10}
!air.version = !{!12}
!air.language_version = !{!13}

!0 = !{ptr @encode_plain_compute_commands, !1, !2}
!1 = !{}
!2 = !{!3, !6, !7, !8, !9}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Commands", !"air.arg_name", !"encoder"}
!4 = !{i32 0, i32 8, i32 0, !"command_buffer", !"commands", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"commands"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"data"}
!7 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"params"}
!8 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!9 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!10 = !{ptr @encode_plain_render_commands, !1, !11}
!11 = !{!3, !14, !7, !8, !9}
!12 = !{i32 2, i32 5, i32 0}
!13 = !{!"Metal", i32 2, i32 4, i32 0}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"vertices"}
