define void @escaping_pointer_integers(ptr addrspace(1) %data, ptr addrspace(2) %constants, ptr addrspace(1) %output) {
entry:
  %interior1 = getelementptr i8, ptr addrspace(1) %data, i64 12
  %base1 = ptrtoint ptr addrspace(1) %data to i64
  %address1 = ptrtoint ptr addrspace(1) %interior1 to i64
  %difference1 = sub i64 %address1, %base1
  %slot1 = getelementptr i64, ptr addrspace(1) %output, i64 1
  store i64 %difference1, ptr addrspace(1) %slot1, align 8
  %interior2 = getelementptr i8, ptr addrspace(2) %constants, i64 28
  %base2 = ptrtoint ptr addrspace(2) %constants to i64
  %address2 = ptrtoint ptr addrspace(2) %interior2 to i64
  %difference2 = sub i64 %address2, %base2
  %slot2 = getelementptr i64, ptr addrspace(1) %output, i64 2
  store i64 %difference2, ptr addrspace(1) %slot2, align 8
  %interior3 = getelementptr i8, ptr addrspace(1) %data, i64 12
  %base3 = ptrtoint ptr addrspace(1) %data to i32
  %address3 = ptrtoint ptr addrspace(1) %interior3 to i32
  %difference3 = sub i32 %address3, %base3
  %wide3 = zext i32 %difference3 to i64
  %slot3 = getelementptr i64, ptr addrspace(1) %output, i64 3
  store i64 %wide3, ptr addrspace(1) %slot3, align 8
  %interior4 = getelementptr i8, ptr addrspace(2) %constants, i64 28
  %base4 = ptrtoint ptr addrspace(2) %constants to i32
  %address4 = ptrtoint ptr addrspace(2) %interior4 to i32
  %difference4 = sub i32 %address4, %base4
  %wide4 = zext i32 %difference4 to i64
  %slot4 = getelementptr i64, ptr addrspace(1) %output, i64 4
  store i64 %wide4, ptr addrspace(1) %slot4, align 8
  ret void
}
!air.fragment = !{!0}
!0 = !{ptr @escaping_pointer_integers, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"data"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"constants"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"output"}
