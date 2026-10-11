; ModuleID = '/tmp/sc/fx.bc'
source_filename = "validation/fixtures/public/kernel_precise_clamp_min_fmod_and_fast_roots.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_precise_clamp_min_fmod_and_fast_roots(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  %4 = load float, ptr addrspace(1) %3, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %5 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1
  %6 = bitcast ptr addrspace(1) %5 to ptr addrspace(1)
  %7 = load float, ptr addrspace(1) %6, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %8 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 2
  %9 = load i32, ptr addrspace(1) %8, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %10 = insertelement <4 x i32> undef, i32 %9, i64 0
  %11 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 3
  %12 = load i32, ptr addrspace(1) %11, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %13 = insertelement <4 x i32> %10, i32 %12, i64 1
  %14 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 4
  %15 = load i32, ptr addrspace(1) %14, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %16 = insertelement <4 x i32> %13, i32 %15, i64 2
  %17 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 5
  %18 = load i32, ptr addrspace(1) %17, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %19 = insertelement <4 x i32> %16, i32 %18, i64 3
  %20 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 6
  %21 = load i32, ptr addrspace(1) %20, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %22 = trunc i32 %21 to i16
  %23 = insertelement <3 x i16> undef, i16 %22, i64 0
  %24 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 7
  %25 = load i32, ptr addrspace(1) %24, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %26 = trunc i32 %25 to i16
  %27 = insertelement <3 x i16> %23, i16 %26, i64 1
  %28 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 8
  %29 = load i32, ptr addrspace(1) %28, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %30 = trunc i32 %29 to i16
  %31 = insertelement <3 x i16> %27, i16 %30, i64 2
  %32 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 9
  %33 = load i32, ptr addrspace(1) %32, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %34 = trunc i32 %33 to i16
  %35 = insertelement <3 x i16> undef, i16 %34, i64 0
  %36 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 10
  %37 = load i32, ptr addrspace(1) %36, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %38 = trunc i32 %37 to i16
  %39 = insertelement <3 x i16> %35, i16 %38, i64 1
  %40 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 11
  %41 = load i32, ptr addrspace(1) %40, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %42 = trunc i32 %41 to i16
  %43 = insertelement <3 x i16> %39, i16 %42, i64 2
  %44 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 12
  %45 = load i32, ptr addrspace(1) %44, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %46 = trunc i32 %45 to i16
  %47 = insertelement <2 x i16> undef, i16 %46, i64 0
  %48 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 13
  %49 = load i32, ptr addrspace(1) %48, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %50 = trunc i32 %49 to i16
  %51 = insertelement <2 x i16> %47, i16 %50, i64 1
  %52 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 14
  %53 = load i32, ptr addrspace(1) %52, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %54 = insertelement <3 x i32> undef, i32 %53, i64 0
  %55 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 15
  %56 = load i32, ptr addrspace(1) %55, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %57 = insertelement <3 x i32> %54, i32 %56, i64 1
  %58 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 16
  %59 = load i32, ptr addrspace(1) %58, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %60 = insertelement <3 x i32> %57, i32 %59, i64 2
  %61 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 17
  %62 = load i32, ptr addrspace(1) %61, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %63 = insertelement <3 x i32> undef, i32 %62, i64 0
  %64 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 18
  %65 = load i32, ptr addrspace(1) %64, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %66 = insertelement <3 x i32> %63, i32 %65, i64 1
  %67 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 19
  %68 = load i32, ptr addrspace(1) %67, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %69 = insertelement <3 x i32> %66, i32 %68, i64 2
  %70 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 20
  %71 = bitcast ptr addrspace(1) %70 to ptr addrspace(1)
  %72 = load float, ptr addrspace(1) %71, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %73 = fptrunc float %72 to half
  %74 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 21
  %75 = bitcast ptr addrspace(1) %74 to ptr addrspace(1)
  %76 = load float, ptr addrspace(1) %75, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %77 = fptrunc float %76 to half
  %78 = insertelement <3 x half> undef, half %77, i64 0
  %79 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 22
  %80 = bitcast ptr addrspace(1) %79 to ptr addrspace(1)
  %81 = load float, ptr addrspace(1) %80, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %82 = fptrunc float %81 to half
  %83 = insertelement <3 x half> %78, half %82, i64 1
  %84 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 23
  %85 = bitcast ptr addrspace(1) %84 to ptr addrspace(1)
  %86 = load float, ptr addrspace(1) %85, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %87 = fptrunc float %86 to half
  %88 = insertelement <3 x half> %83, half %87, i64 2
  %89 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 24
  %90 = bitcast ptr addrspace(1) %89 to ptr addrspace(1)
  %91 = load float, ptr addrspace(1) %90, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %92 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 25
  %93 = bitcast ptr addrspace(1) %92 to ptr addrspace(1)
  %94 = load float, ptr addrspace(1) %93, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %95 = insertelement <2 x float> undef, float %94, i64 0
  %96 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 26
  %97 = bitcast ptr addrspace(1) %96 to ptr addrspace(1)
  %98 = load float, ptr addrspace(1) %97, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %99 = insertelement <2 x float> %95, float %98, i64 1
  %100 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 27
  %101 = bitcast ptr addrspace(1) %100 to ptr addrspace(1)
  %102 = load float, ptr addrspace(1) %101, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %103 = insertelement <2 x float> undef, float %102, i64 0
  %104 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 28
  %105 = bitcast ptr addrspace(1) %104 to ptr addrspace(1)
  %106 = load float, ptr addrspace(1) %105, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %107 = insertelement <2 x float> %103, float %106, i64 1
  %108 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 29
  %109 = bitcast ptr addrspace(1) %108 to ptr addrspace(1)
  %110 = load float, ptr addrspace(1) %109, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %111 = tail call fast float @air.fabs.f32(float %4) #2
  %112 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  store float %111, ptr addrspace(1) %112, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %113 = tail call fast float @air.clamp.f32(float %7, float 1.000000e+00, float 2.000000e+00) #2
  %114 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  %115 = bitcast ptr addrspace(1) %114 to ptr addrspace(1)
  store float %113, ptr addrspace(1) %115, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %116 = tail call <4 x i32> @air.clamp.u.v4i32(<4 x i32> %19, <4 x i32> splat (i32 2), <4 x i32> splat (i32 5)) #2
  %117 = extractelement <4 x i32> %116, i64 0
  %118 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %117, ptr addrspace(1) %118, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %119 = extractelement <4 x i32> %116, i64 1
  %120 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %119, ptr addrspace(1) %120, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %121 = extractelement <4 x i32> %116, i64 2
  %122 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %121, ptr addrspace(1) %122, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %123 = extractelement <4 x i32> %116, i64 3
  %124 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %123, ptr addrspace(1) %124, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %125 = tail call <3 x i16> @air.clamp.s.v3i16(<3 x i16> %31, <3 x i16> splat (i16 -3), <3 x i16> splat (i16 3)) #2
  %126 = extractelement <3 x i16> %125, i64 0
  %127 = sext i16 %126 to i32
  %128 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  store i32 %127, ptr addrspace(1) %128, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %129 = extractelement <3 x i16> %125, i64 1
  %130 = sext i16 %129 to i32
  %131 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  store i32 %130, ptr addrspace(1) %131, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %132 = extractelement <3 x i16> %125, i64 2
  %133 = sext i16 %132 to i32
  %134 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  store i32 %133, ptr addrspace(1) %134, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %135 = tail call <3 x i16> @air.min.u.v3i16(<3 x i16> %43, <3 x i16> splat (i16 4)) #2
  %136 = extractelement <3 x i16> %135, i64 0
  %137 = zext i16 %136 to i32
  %138 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 9
  store i32 %137, ptr addrspace(1) %138, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %139 = extractelement <3 x i16> %135, i64 1
  %140 = zext i16 %139 to i32
  %141 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 10
  store i32 %140, ptr addrspace(1) %141, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %142 = extractelement <3 x i16> %135, i64 2
  %143 = zext i16 %142 to i32
  %144 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 11
  store i32 %143, ptr addrspace(1) %144, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %145 = tail call <2 x i16> @air.min.s.v2i16(<2 x i16> %51, <2 x i16> splat (i16 1)) #2
  %146 = extractelement <2 x i16> %145, i64 0
  %147 = sext i16 %146 to i32
  %148 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 12
  store i32 %147, ptr addrspace(1) %148, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %149 = extractelement <2 x i16> %145, i64 1
  %150 = sext i16 %149 to i32
  %151 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 13
  store i32 %150, ptr addrspace(1) %151, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %152 = tail call <3 x i32> @air.min.s.v3i32(<3 x i32> %60, <3 x i32> splat (i32 7)) #2
  %153 = extractelement <3 x i32> %152, i64 0
  %154 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 14
  store i32 %153, ptr addrspace(1) %154, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %155 = extractelement <3 x i32> %152, i64 1
  %156 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 15
  store i32 %155, ptr addrspace(1) %156, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %157 = extractelement <3 x i32> %152, i64 2
  %158 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 16
  store i32 %157, ptr addrspace(1) %158, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %159 = tail call <3 x i32> @air.min.u.v3i32(<3 x i32> %69, <3 x i32> splat (i32 6)) #2
  %160 = extractelement <3 x i32> %159, i64 0
  %161 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 17
  store i32 %160, ptr addrspace(1) %161, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %162 = extractelement <3 x i32> %159, i64 1
  %163 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 18
  store i32 %162, ptr addrspace(1) %163, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %164 = extractelement <3 x i32> %159, i64 2
  %165 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 19
  store i32 %164, ptr addrspace(1) %165, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %166 = tail call fast half @air.fmod.f16(half %73, half 0xH4000) #2
  %167 = bitcast half %166 to i16
  %168 = zext i16 %167 to i32
  %169 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 20
  store i32 %168, ptr addrspace(1) %169, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %170 = tail call fast <3 x half> @air.fmod.v3f16(<3 x half> %88, <3 x half> splat (half 0xH4400)) #2
  %171 = bitcast <3 x half> %170 to <3 x i16>
  %172 = extractelement <3 x i16> %171, i64 0
  %173 = zext i16 %172 to i32
  %174 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 21
  store i32 %173, ptr addrspace(1) %174, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %175 = extractelement <3 x i16> %171, i64 1
  %176 = zext i16 %175 to i32
  %177 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 22
  store i32 %176, ptr addrspace(1) %177, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %178 = extractelement <3 x i16> %171, i64 2
  %179 = zext i16 %178 to i32
  %180 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 23
  store i32 %179, ptr addrspace(1) %180, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %181 = tail call fast float @air.fmod.f32(float %91, float 4.000000e+00) #2
  %182 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 24
  %183 = bitcast ptr addrspace(1) %182 to ptr addrspace(1)
  store float %181, ptr addrspace(1) %183, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %184 = tail call fast <2 x float> @air.fast_rsqrt.v2f32(<2 x float> %99) #2
  %185 = bitcast <2 x float> %184 to <2 x i32>
  %186 = extractelement <2 x i32> %185, i64 0
  %187 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 25
  store i32 %186, ptr addrspace(1) %187, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %188 = extractelement <2 x i32> %185, i64 1
  %189 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 26
  store i32 %188, ptr addrspace(1) %189, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %190 = tail call fast <2 x float> @air.fast_log.v2f32(<2 x float> %107) #2
  %191 = bitcast <2 x float> %190 to <2 x i32>
  %192 = extractelement <2 x i32> %191, i64 0
  %193 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 27
  store i32 %192, ptr addrspace(1) %193, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %194 = extractelement <2 x i32> %191, i64 1
  %195 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 28
  store i32 %194, ptr addrspace(1) %195, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %196 = tail call fast float @air.fast_sinpi.f32(float %110) #2
  %197 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 29
  %198 = bitcast ptr addrspace(1) %197 to ptr addrspace(1)
  store float %196, ptr addrspace(1) %198, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fabs.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.clamp.f32(float, float, float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.clamp.u.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i16> @air.clamp.s.v3i16(<3 x i16>, <3 x i16>, <3 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i16> @air.min.u.v3i16(<3 x i16>, <3 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i16> @air.min.s.v2i16(<2 x i16>, <2 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i32> @air.min.s.v3i32(<3 x i32>, <3 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i32> @air.min.u.v3i32(<3 x i32>, <3 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.fmod.f16(half, half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x half> @air.fmod.v3f16(<3 x half>, <3 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fmod.f32(float, float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_rsqrt.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_log.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_sinpi.f32(float) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!14, !15, !16}
!llvm.ident = !{!17}
!air.version = !{!18}
!air.language_version = !{!19}
!air.source_file_name = !{!20}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_precise_clamp_min_fmod_and_fast_roots, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 3, i32 2, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_precise_clamp_min_fmod_and_fast_roots.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(1)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_precise_clamp_min_fmod_and_fast_roots)"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"air-alias-scope-arg(0)"}
