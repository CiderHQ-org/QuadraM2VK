; ModuleID = 'kernel_reinterpret_local_views.ll'
source_filename = "validation/fixtures/public/kernel_reinterpret_local_views.metal"
target triple = "spirv-unknown-vulkan1.2"

%struct.HalfLead = type { <4 x half>, i32 }

define void @reinterpret_local_views(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %in, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out, i32 %gid) {
entry:
  %i = zext i32 %gid to i64
  %pin = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %i
  %w = load i32, ptr addrspace(1) %pin, align 4

  ; (1) a <4 x float> stored through a [4 x i32] local view
  %a = alloca [4 x i32], align 16
  %w1 = add i32 %w, 1
  %w2 = add i32 %w, 2
  %w3 = add i32 %w, 3
  %f0 = bitcast i32 %w to float
  %f1 = bitcast i32 %w1 to float
  %f2 = bitcast i32 %w2 to float
  %f3 = bitcast i32 %w3 to float
  %v0 = insertelement <4 x float> zeroinitializer, float %f0, i32 0
  %v1 = insertelement <4 x float> %v0, float %f1, i32 1
  %v2 = insertelement <4 x float> %v1, float %f2, i32 2
  %v3 = insertelement <4 x float> %v2, float %f3, i32 3
  store <4 x float> %v3, ptr %a, align 16
  %a0p = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 0
  %a0 = load i32, ptr %a0p, align 16
  %a1p = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 1
  %a1 = load i32, ptr %a1p, align 4
  %a2p = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 2
  %a2 = load i32, ptr %a2p, align 8
  %a3p = getelementptr inbounds [4 x i32], ptr %a, i64 0, i64 3
  %a3 = load i32, ptr %a3p, align 4
  %a1w = mul i32 %a1, 2
  %a2w = mul i32 %a2, 4
  %a3w = mul i32 %a3, 8
  %as0 = add i32 %a0, %a1w
  %as1 = add i32 %as0, %a2w
  %as2 = add i32 %as1, %a3w
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %as2, ptr addrspace(1) %o0, align 4
  %x03 = add i32 %a2, 0
  %k1 = add nuw nsw i32 %gid, 8
  %ki1 = zext i32 %k1 to i64
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %ki1
  store i32 %x03, ptr addrspace(1) %o1, align 4

  ; (2) a [4 x float] local view read back as <4 x i32>
  %b = alloca [4 x float], align 16
  %w16 = add i32 %w, 16
  %w17 = add i32 %w, 17
  %w18 = add i32 %w, 18
  %w19 = add i32 %w, 19
  %g0 = bitcast i32 %w16 to float
  %g1 = bitcast i32 %w17 to float
  %g2 = bitcast i32 %w18 to float
  %g3 = bitcast i32 %w19 to float
  %b0p = getelementptr inbounds [4 x float], ptr %b, i64 0, i64 0
  store float %g0, ptr %b0p, align 16
  %b1p = getelementptr inbounds [4 x float], ptr %b, i64 0, i64 1
  store float %g1, ptr %b1p, align 4
  %b2p = getelementptr inbounds [4 x float], ptr %b, i64 0, i64 2
  store float %g2, ptr %b2p, align 8
  %b3p = getelementptr inbounds [4 x float], ptr %b, i64 0, i64 3
  store float %g3, ptr %b3p, align 4
  %r = load <4 x i32>, ptr %b0p, align 16
  %r0 = extractelement <4 x i32> %r, i32 0
  %r1 = extractelement <4 x i32> %r, i32 1
  %r2 = extractelement <4 x i32> %r, i32 2
  %r3 = extractelement <4 x i32> %r, i32 3
  %r1w = mul i32 %r1, 2
  %r2w = mul i32 %r2, 4
  %r3w = mul i32 %r3, 8
  %rs0 = add i32 %r0, %r1w
  %rs1 = add i32 %rs0, %r2w
  %rs2 = add i32 %rs1, %r3w
  %k2 = add nuw nsw i32 %gid, 16
  %ki2 = zext i32 %k2 to i64
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %ki2
  store i32 %rs2, ptr addrspace(1) %o2, align 4
  %x03b = add i32 %r2, 0
  %k3 = add nuw nsw i32 %gid, 24
  %ki3 = zext i32 %k3 to i64
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %ki3
  store i32 %x03b, ptr addrspace(1) %o3, align 4

  ; (3) a struct whose leading member is <4 x half>, read as <2 x float>
  %s = alloca %struct.HalfLead, align 16
  %h0i = add i32 %gid, 15360
  %h1i = add i32 %gid, 15616
  %h2i = add i32 %gid, 15872
  %h3i = add i32 %gid, 16128
  %h0t = trunc i32 %h0i to i16
  %h1t = trunc i32 %h1i to i16
  %h2t = trunc i32 %h2i to i16
  %h3t = trunc i32 %h3i to i16
  %h0 = bitcast i16 %h0t to half
  %h1 = bitcast i16 %h1t to half
  %h2 = bitcast i16 %h2t to half
  %h3 = bitcast i16 %h3t to half
  %hv0 = insertelement <4 x half> zeroinitializer, half %h0, i32 0
  %hv1 = insertelement <4 x half> %hv0, half %h1, i32 1
  %hv2 = insertelement <4 x half> %hv1, half %h2, i32 2
  %hv3 = insertelement <4 x half> %hv2, half %h3, i32 3
  %shp = getelementptr inbounds %struct.HalfLead, ptr %s, i64 0, i32 0
  store <4 x half> %hv3, ptr %shp, align 8
  %stp = getelementptr inbounds %struct.HalfLead, ptr %s, i64 0, i32 1
  store i32 0, ptr %stp, align 4
  %fv = load <2 x float>, ptr %s, align 8
  %fx = extractelement <2 x float> %fv, i32 0
  %fy = extractelement <2 x float> %fv, i32 1
  %fxb = bitcast float %fx to i32
  %fyb = bitcast float %fy to i32
  %k4 = add nuw nsw i32 %gid, 32
  %ki4 = zext i32 %k4 to i64
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %ki4
  store i32 %fxb, ptr addrspace(1) %o4, align 4
  %k5 = add nuw nsw i32 %gid, 40
  %ki5 = zext i32 %k5 to i64
  %o5 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %ki5
  store i32 %fyb, ptr addrspace(1) %o5, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @reinterpret_local_views, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
