; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'psc.bc'
source_filename = "validation/fixtures/public/kernel_pi_scaled_sin_cos.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_pi_scaled_sin_cos(ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = load float, ptr addrspace(1) %1, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %4 = tail call fast float @air.sinpi.f32(float %3) #2
  %5 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  store float %4, ptr addrspace(1) %5, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %6 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  %7 = load float, ptr addrspace(1) %6, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %8 = tail call fast float @air.sinpi.f32(float %7) #2
  %9 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  %10 = bitcast ptr addrspace(1) %9 to ptr addrspace(1)
  store float %8, ptr addrspace(1) %10, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %11 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  %12 = load float, ptr addrspace(1) %11, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %13 = tail call fast float @air.sinpi.f32(float %12) #2
  %14 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  %15 = bitcast ptr addrspace(1) %14 to ptr addrspace(1)
  store float %13, ptr addrspace(1) %15, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %16 = getelementptr inbounds float, ptr addrspace(1) %1, i64 3
  %17 = load float, ptr addrspace(1) %16, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %18 = tail call fast float @air.sinpi.f32(float %17) #2
  %19 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  %20 = bitcast ptr addrspace(1) %19 to ptr addrspace(1)
  store float %18, ptr addrspace(1) %20, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %21 = getelementptr inbounds float, ptr addrspace(1) %1, i64 4
  %22 = load float, ptr addrspace(1) %21, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %23 = tail call fast float @air.sinpi.f32(float %22) #2
  %24 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  %25 = bitcast ptr addrspace(1) %24 to ptr addrspace(1)
  store float %23, ptr addrspace(1) %25, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %26 = getelementptr inbounds float, ptr addrspace(1) %1, i64 5
  %27 = load float, ptr addrspace(1) %26, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %28 = tail call fast float @air.sinpi.f32(float %27) #2
  %29 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  %30 = bitcast ptr addrspace(1) %29 to ptr addrspace(1)
  store float %28, ptr addrspace(1) %30, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %31 = getelementptr inbounds float, ptr addrspace(1) %1, i64 6
  %32 = load float, ptr addrspace(1) %31, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %33 = tail call fast float @air.sinpi.f32(float %32) #2
  %34 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  %35 = bitcast ptr addrspace(1) %34 to ptr addrspace(1)
  store float %33, ptr addrspace(1) %35, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %36 = getelementptr inbounds float, ptr addrspace(1) %1, i64 7
  %37 = load float, ptr addrspace(1) %36, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %38 = tail call fast float @air.sinpi.f32(float %37) #2
  %39 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  %40 = bitcast ptr addrspace(1) %39 to ptr addrspace(1)
  store float %38, ptr addrspace(1) %40, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %41 = getelementptr inbounds float, ptr addrspace(1) %1, i64 8
  %42 = load float, ptr addrspace(1) %41, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %43 = tail call fast float @air.sinpi.f32(float %42) #2
  %44 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  %45 = bitcast ptr addrspace(1) %44 to ptr addrspace(1)
  store float %43, ptr addrspace(1) %45, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %46 = getelementptr inbounds float, ptr addrspace(1) %1, i64 9
  %47 = load float, ptr addrspace(1) %46, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %48 = tail call fast float @air.sinpi.f32(float %47) #2
  %49 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 9
  %50 = bitcast ptr addrspace(1) %49 to ptr addrspace(1)
  store float %48, ptr addrspace(1) %50, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %51 = getelementptr inbounds float, ptr addrspace(1) %1, i64 10
  %52 = load float, ptr addrspace(1) %51, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %53 = tail call fast float @air.sinpi.f32(float %52) #2
  %54 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 10
  %55 = bitcast ptr addrspace(1) %54 to ptr addrspace(1)
  store float %53, ptr addrspace(1) %55, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %56 = getelementptr inbounds float, ptr addrspace(1) %1, i64 11
  %57 = load float, ptr addrspace(1) %56, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %58 = tail call fast float @air.sinpi.f32(float %57) #2
  %59 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 11
  %60 = bitcast ptr addrspace(1) %59 to ptr addrspace(1)
  store float %58, ptr addrspace(1) %60, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %61 = getelementptr inbounds float, ptr addrspace(1) %1, i64 12
  %62 = load float, ptr addrspace(1) %61, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %63 = tail call fast float @air.sinpi.f32(float %62) #2
  %64 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 12
  %65 = bitcast ptr addrspace(1) %64 to ptr addrspace(1)
  store float %63, ptr addrspace(1) %65, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %66 = getelementptr inbounds float, ptr addrspace(1) %1, i64 13
  %67 = load float, ptr addrspace(1) %66, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %68 = tail call fast float @air.sinpi.f32(float %67) #2
  %69 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 13
  %70 = bitcast ptr addrspace(1) %69 to ptr addrspace(1)
  store float %68, ptr addrspace(1) %70, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %71 = getelementptr inbounds float, ptr addrspace(1) %1, i64 14
  %72 = load float, ptr addrspace(1) %71, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %73 = tail call fast float @air.sinpi.f32(float %72) #2
  %74 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 14
  %75 = bitcast ptr addrspace(1) %74 to ptr addrspace(1)
  store float %73, ptr addrspace(1) %75, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %76 = getelementptr inbounds float, ptr addrspace(1) %1, i64 15
  %77 = load float, ptr addrspace(1) %76, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %78 = tail call fast float @air.sinpi.f32(float %77) #2
  %79 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 15
  %80 = bitcast ptr addrspace(1) %79 to ptr addrspace(1)
  store float %78, ptr addrspace(1) %80, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %81 = getelementptr inbounds float, ptr addrspace(1) %1, i64 16
  %82 = load float, ptr addrspace(1) %81, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %83 = tail call fast float @air.sinpi.f32(float %82) #2
  %84 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 16
  %85 = bitcast ptr addrspace(1) %84 to ptr addrspace(1)
  store float %83, ptr addrspace(1) %85, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %86 = getelementptr inbounds float, ptr addrspace(1) %1, i64 17
  %87 = load float, ptr addrspace(1) %86, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %88 = tail call fast float @air.sinpi.f32(float %87) #2
  %89 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 17
  %90 = bitcast ptr addrspace(1) %89 to ptr addrspace(1)
  store float %88, ptr addrspace(1) %90, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %91 = getelementptr inbounds float, ptr addrspace(1) %1, i64 18
  %92 = load float, ptr addrspace(1) %91, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %93 = tail call fast float @air.sinpi.f32(float %92) #2
  %94 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 18
  %95 = bitcast ptr addrspace(1) %94 to ptr addrspace(1)
  store float %93, ptr addrspace(1) %95, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %96 = getelementptr inbounds float, ptr addrspace(1) %1, i64 19
  %97 = load float, ptr addrspace(1) %96, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %98 = tail call fast float @air.sinpi.f32(float %97) #2
  %99 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 19
  %100 = bitcast ptr addrspace(1) %99 to ptr addrspace(1)
  store float %98, ptr addrspace(1) %100, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %101 = getelementptr inbounds float, ptr addrspace(1) %1, i64 20
  %102 = load float, ptr addrspace(1) %101, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %103 = tail call fast float @air.sinpi.f32(float %102) #2
  %104 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 20
  %105 = bitcast ptr addrspace(1) %104 to ptr addrspace(1)
  store float %103, ptr addrspace(1) %105, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %106 = getelementptr inbounds float, ptr addrspace(1) %1, i64 21
  %107 = load float, ptr addrspace(1) %106, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %108 = tail call fast float @air.sinpi.f32(float %107) #2
  %109 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 21
  %110 = bitcast ptr addrspace(1) %109 to ptr addrspace(1)
  store float %108, ptr addrspace(1) %110, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %111 = getelementptr inbounds float, ptr addrspace(1) %1, i64 22
  %112 = load float, ptr addrspace(1) %111, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %113 = tail call fast float @air.sinpi.f32(float %112) #2
  %114 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 22
  %115 = bitcast ptr addrspace(1) %114 to ptr addrspace(1)
  store float %113, ptr addrspace(1) %115, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %116 = getelementptr inbounds float, ptr addrspace(1) %1, i64 23
  %117 = load float, ptr addrspace(1) %116, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %118 = tail call fast float @air.sinpi.f32(float %117) #2
  %119 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 23
  %120 = bitcast ptr addrspace(1) %119 to ptr addrspace(1)
  store float %118, ptr addrspace(1) %120, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %121 = getelementptr inbounds float, ptr addrspace(1) %1, i64 24
  %122 = load float, ptr addrspace(1) %121, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %123 = tail call fast float @air.sinpi.f32(float %122) #2
  %124 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 24
  %125 = bitcast ptr addrspace(1) %124 to ptr addrspace(1)
  store float %123, ptr addrspace(1) %125, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %126 = getelementptr inbounds float, ptr addrspace(1) %1, i64 25
  %127 = load float, ptr addrspace(1) %126, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %128 = tail call fast float @air.sinpi.f32(float %127) #2
  %129 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 25
  %130 = bitcast ptr addrspace(1) %129 to ptr addrspace(1)
  store float %128, ptr addrspace(1) %130, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %131 = getelementptr inbounds float, ptr addrspace(1) %1, i64 26
  %132 = load float, ptr addrspace(1) %131, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %133 = tail call fast float @air.sinpi.f32(float %132) #2
  %134 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 26
  %135 = bitcast ptr addrspace(1) %134 to ptr addrspace(1)
  store float %133, ptr addrspace(1) %135, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %136 = getelementptr inbounds float, ptr addrspace(1) %1, i64 27
  %137 = load float, ptr addrspace(1) %136, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %138 = tail call fast float @air.sinpi.f32(float %137) #2
  %139 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 27
  %140 = bitcast ptr addrspace(1) %139 to ptr addrspace(1)
  store float %138, ptr addrspace(1) %140, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %141 = getelementptr inbounds float, ptr addrspace(1) %1, i64 28
  %142 = load float, ptr addrspace(1) %141, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %143 = tail call fast float @air.sinpi.f32(float %142) #2
  %144 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 28
  %145 = bitcast ptr addrspace(1) %144 to ptr addrspace(1)
  store float %143, ptr addrspace(1) %145, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %146 = getelementptr inbounds float, ptr addrspace(1) %1, i64 29
  %147 = load float, ptr addrspace(1) %146, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %148 = tail call fast float @air.sinpi.f32(float %147) #2
  %149 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 29
  %150 = bitcast ptr addrspace(1) %149 to ptr addrspace(1)
  store float %148, ptr addrspace(1) %150, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %151 = getelementptr inbounds float, ptr addrspace(1) %1, i64 30
  %152 = load float, ptr addrspace(1) %151, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %153 = tail call fast float @air.sinpi.f32(float %152) #2
  %154 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 30
  %155 = bitcast ptr addrspace(1) %154 to ptr addrspace(1)
  store float %153, ptr addrspace(1) %155, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %156 = getelementptr inbounds float, ptr addrspace(1) %1, i64 31
  %157 = load float, ptr addrspace(1) %156, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %158 = tail call fast float @air.sinpi.f32(float %157) #2
  %159 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 31
  %160 = bitcast ptr addrspace(1) %159 to ptr addrspace(1)
  store float %158, ptr addrspace(1) %160, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %161 = tail call fast float @air.cospi.f32(float %3) #2
  %162 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 32
  %163 = bitcast ptr addrspace(1) %162 to ptr addrspace(1)
  store float %161, ptr addrspace(1) %163, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %164 = tail call fast float @air.cospi.f32(float %7) #2
  %165 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 33
  %166 = bitcast ptr addrspace(1) %165 to ptr addrspace(1)
  store float %164, ptr addrspace(1) %166, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %167 = tail call fast float @air.cospi.f32(float %12) #2
  %168 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 34
  %169 = bitcast ptr addrspace(1) %168 to ptr addrspace(1)
  store float %167, ptr addrspace(1) %169, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %170 = tail call fast float @air.cospi.f32(float %17) #2
  %171 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 35
  %172 = bitcast ptr addrspace(1) %171 to ptr addrspace(1)
  store float %170, ptr addrspace(1) %172, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %173 = tail call fast float @air.cospi.f32(float %22) #2
  %174 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 36
  %175 = bitcast ptr addrspace(1) %174 to ptr addrspace(1)
  store float %173, ptr addrspace(1) %175, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %176 = tail call fast float @air.cospi.f32(float %27) #2
  %177 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 37
  %178 = bitcast ptr addrspace(1) %177 to ptr addrspace(1)
  store float %176, ptr addrspace(1) %178, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %179 = tail call fast float @air.cospi.f32(float %32) #2
  %180 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 38
  %181 = bitcast ptr addrspace(1) %180 to ptr addrspace(1)
  store float %179, ptr addrspace(1) %181, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %182 = tail call fast float @air.cospi.f32(float %37) #2
  %183 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 39
  %184 = bitcast ptr addrspace(1) %183 to ptr addrspace(1)
  store float %182, ptr addrspace(1) %184, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %185 = tail call fast float @air.cospi.f32(float %42) #2
  %186 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 40
  %187 = bitcast ptr addrspace(1) %186 to ptr addrspace(1)
  store float %185, ptr addrspace(1) %187, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %188 = tail call fast float @air.cospi.f32(float %47) #2
  %189 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 41
  %190 = bitcast ptr addrspace(1) %189 to ptr addrspace(1)
  store float %188, ptr addrspace(1) %190, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %191 = tail call fast float @air.cospi.f32(float %52) #2
  %192 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 42
  %193 = bitcast ptr addrspace(1) %192 to ptr addrspace(1)
  store float %191, ptr addrspace(1) %193, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %194 = tail call fast float @air.cospi.f32(float %57) #2
  %195 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 43
  %196 = bitcast ptr addrspace(1) %195 to ptr addrspace(1)
  store float %194, ptr addrspace(1) %196, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %197 = tail call fast float @air.cospi.f32(float %62) #2
  %198 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 44
  %199 = bitcast ptr addrspace(1) %198 to ptr addrspace(1)
  store float %197, ptr addrspace(1) %199, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %200 = tail call fast float @air.cospi.f32(float %67) #2
  %201 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 45
  %202 = bitcast ptr addrspace(1) %201 to ptr addrspace(1)
  store float %200, ptr addrspace(1) %202, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %203 = tail call fast float @air.cospi.f32(float %72) #2
  %204 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 46
  %205 = bitcast ptr addrspace(1) %204 to ptr addrspace(1)
  store float %203, ptr addrspace(1) %205, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %206 = tail call fast float @air.cospi.f32(float %77) #2
  %207 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 47
  %208 = bitcast ptr addrspace(1) %207 to ptr addrspace(1)
  store float %206, ptr addrspace(1) %208, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %209 = tail call fast float @air.cospi.f32(float %82) #2
  %210 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 48
  %211 = bitcast ptr addrspace(1) %210 to ptr addrspace(1)
  store float %209, ptr addrspace(1) %211, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %212 = tail call fast float @air.cospi.f32(float %87) #2
  %213 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 49
  %214 = bitcast ptr addrspace(1) %213 to ptr addrspace(1)
  store float %212, ptr addrspace(1) %214, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %215 = tail call fast float @air.cospi.f32(float %92) #2
  %216 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 50
  %217 = bitcast ptr addrspace(1) %216 to ptr addrspace(1)
  store float %215, ptr addrspace(1) %217, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %218 = tail call fast float @air.cospi.f32(float %97) #2
  %219 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 51
  %220 = bitcast ptr addrspace(1) %219 to ptr addrspace(1)
  store float %218, ptr addrspace(1) %220, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %221 = tail call fast float @air.cospi.f32(float %102) #2
  %222 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 52
  %223 = bitcast ptr addrspace(1) %222 to ptr addrspace(1)
  store float %221, ptr addrspace(1) %223, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %224 = tail call fast float @air.cospi.f32(float %107) #2
  %225 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 53
  %226 = bitcast ptr addrspace(1) %225 to ptr addrspace(1)
  store float %224, ptr addrspace(1) %226, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %227 = tail call fast float @air.cospi.f32(float %112) #2
  %228 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 54
  %229 = bitcast ptr addrspace(1) %228 to ptr addrspace(1)
  store float %227, ptr addrspace(1) %229, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %230 = tail call fast float @air.cospi.f32(float %117) #2
  %231 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 55
  %232 = bitcast ptr addrspace(1) %231 to ptr addrspace(1)
  store float %230, ptr addrspace(1) %232, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %233 = tail call fast float @air.cospi.f32(float %122) #2
  %234 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 56
  %235 = bitcast ptr addrspace(1) %234 to ptr addrspace(1)
  store float %233, ptr addrspace(1) %235, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %236 = tail call fast float @air.cospi.f32(float %127) #2
  %237 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 57
  %238 = bitcast ptr addrspace(1) %237 to ptr addrspace(1)
  store float %236, ptr addrspace(1) %238, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %239 = tail call fast float @air.cospi.f32(float %132) #2
  %240 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 58
  %241 = bitcast ptr addrspace(1) %240 to ptr addrspace(1)
  store float %239, ptr addrspace(1) %241, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %242 = tail call fast float @air.cospi.f32(float %137) #2
  %243 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 59
  %244 = bitcast ptr addrspace(1) %243 to ptr addrspace(1)
  store float %242, ptr addrspace(1) %244, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %245 = tail call fast float @air.cospi.f32(float %142) #2
  %246 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 60
  %247 = bitcast ptr addrspace(1) %246 to ptr addrspace(1)
  store float %245, ptr addrspace(1) %247, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %248 = tail call fast float @air.cospi.f32(float %147) #2
  %249 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 61
  %250 = bitcast ptr addrspace(1) %249 to ptr addrspace(1)
  store float %248, ptr addrspace(1) %250, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %251 = tail call fast float @air.cospi.f32(float %152) #2
  %252 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 62
  %253 = bitcast ptr addrspace(1) %252 to ptr addrspace(1)
  store float %251, ptr addrspace(1) %253, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %254 = tail call fast float @air.cospi.f32(float %157) #2
  %255 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 63
  %256 = bitcast ptr addrspace(1) %255 to ptr addrspace(1)
  store float %254, ptr addrspace(1) %256, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.sinpi.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.cospi.f32(float) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_pi_scaled_sin_cos, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_pi_scaled_sin_cos.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"float", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(1)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_pi_scaled_sin_cos)"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"air-alias-scope-arg(0)"}
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !23, i64 0}
