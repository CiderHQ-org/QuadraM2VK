; Owned synthetic fixture for public drift / A/B samples.
; Not derived from a third-party metallib.
;
; `kernel_copy_word` with the destination annotated `air.read` while the kernel still stores
; through it. Real captured AIR does this: `particle_fill_grid` in the corpus annotates the buffer
; it atomically swaps into as `air.read`. `docs/REFLECTION.md` says reflected translation widens a
; buffer's declared classification to cover the stores the finished module performs, while
; `reflect_sanitized` reports the declared classification alone, so this module has no writable
; resource in the declared view and one in the translated view.
source_filename = "kernel_copy_word_declared_read.metal"

define void @copy_word_declared_read(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %value = load i32, ptr addrspace(1) %input, align 4
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @copy_word_declared_read, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
