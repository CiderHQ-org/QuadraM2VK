; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'a.bc'
source_filename = "validation/fixtures/public/kernel_exact_convert_and_fma.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_exact_convert_and_fma(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %4, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %5, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %6) local_unnamed_addr #0 {
  %8 = load float, ptr addrspace(1) %1, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %9 = insertelement <3 x float> undef, float %8, i64 0
  %10 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  %11 = load float, ptr addrspace(1) %10, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %12 = insertelement <3 x float> %9, float %11, i64 1
  %13 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  %14 = load float, ptr addrspace(1) %13, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %15 = insertelement <3 x float> %12, float %14, i64 2
  %16 = getelementptr inbounds float, ptr addrspace(1) %1, i64 3
  %17 = load float, ptr addrspace(1) %16, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %18 = insertelement <3 x float> undef, float %17, i64 0
  %19 = getelementptr inbounds float, ptr addrspace(1) %1, i64 4
  %20 = load float, ptr addrspace(1) %19, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %21 = insertelement <3 x float> %18, float %20, i64 1
  %22 = getelementptr inbounds float, ptr addrspace(1) %1, i64 5
  %23 = load float, ptr addrspace(1) %22, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %24 = insertelement <3 x float> %21, float %23, i64 2
  %25 = getelementptr inbounds float, ptr addrspace(1) %1, i64 6
  %26 = load float, ptr addrspace(1) %25, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %27 = insertelement <3 x float> undef, float %26, i64 0
  %28 = getelementptr inbounds float, ptr addrspace(1) %1, i64 7
  %29 = load float, ptr addrspace(1) %28, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %30 = insertelement <3 x float> %27, float %29, i64 1
  %31 = getelementptr inbounds float, ptr addrspace(1) %1, i64 8
  %32 = load float, ptr addrspace(1) %31, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %33 = insertelement <3 x float> %30, float %32, i64 2
  %34 = tail call fast <3 x float> @air.fma.v3f32(<3 x float> %15, <3 x float> %24, <3 x float> %33) #2
  %35 = load i64, ptr addrspace(1) %3, align 8, !tbaa !40, !alias.scope !42, !noalias !43
  %36 = insertelement <4 x i64> undef, i64 %35, i64 0
  %37 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 1
  %38 = load i64, ptr addrspace(1) %37, align 8, !tbaa !40, !alias.scope !42, !noalias !43
  %39 = insertelement <4 x i64> %36, i64 %38, i64 1
  %40 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 2
  %41 = load i64, ptr addrspace(1) %40, align 8, !tbaa !40, !alias.scope !42, !noalias !43
  %42 = insertelement <4 x i64> %39, i64 %41, i64 2
  %43 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 3
  %44 = load i64, ptr addrspace(1) %43, align 8, !tbaa !40, !alias.scope !42, !noalias !43
  %45 = insertelement <4 x i64> %42, i64 %44, i64 3
  %46 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 4
  %47 = load i64, ptr addrspace(1) %46, align 8, !tbaa !40, !alias.scope !42, !noalias !43
  %48 = insertelement <4 x i64> undef, i64 %47, i64 0
  %49 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 5
  %50 = load i64, ptr addrspace(1) %49, align 8, !tbaa !40, !alias.scope !42, !noalias !43
  %51 = insertelement <4 x i64> %48, i64 %50, i64 1
  %52 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 6
  %53 = load i64, ptr addrspace(1) %52, align 8, !tbaa !40, !alias.scope !42, !noalias !43
  %54 = insertelement <4 x i64> %51, i64 %53, i64 2
  %55 = getelementptr inbounds i64, ptr addrspace(1) %3, i64 7
  %56 = load i64, ptr addrspace(1) %55, align 8, !tbaa !40, !alias.scope !42, !noalias !43
  %57 = insertelement <4 x i64> %54, i64 %56, i64 3
  %58 = tail call fast <4 x float> @air.convert.f.v4f32.u.v4i64(<4 x i64> %45) #2
  %59 = tail call fast <4 x float> @air.convert.f.v4f32.s.v4i64(<4 x i64> %57) #2
  %60 = getelementptr inbounds float, ptr addrspace(1) %1, i64 9
  %61 = load float, ptr addrspace(1) %60, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %62 = insertelement <4 x float> undef, float %61, i64 0
  %63 = getelementptr inbounds float, ptr addrspace(1) %1, i64 10
  %64 = load float, ptr addrspace(1) %63, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %65 = insertelement <4 x float> %62, float %64, i64 1
  %66 = getelementptr inbounds float, ptr addrspace(1) %1, i64 11
  %67 = load float, ptr addrspace(1) %66, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %68 = insertelement <4 x float> %65, float %67, i64 2
  %69 = getelementptr inbounds float, ptr addrspace(1) %1, i64 12
  %70 = load float, ptr addrspace(1) %69, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %71 = insertelement <4 x float> %68, float %70, i64 3
  %72 = tail call <4 x i64> @air.convert.s.v4i64.f.v4f32(<4 x float> %71) #2
  %73 = load half, ptr addrspace(1) %4, align 2, !tbaa !44, !alias.scope !46, !noalias !47
  %74 = insertelement <2 x half> undef, half %73, i64 0
  %75 = getelementptr inbounds half, ptr addrspace(1) %4, i64 1
  %76 = load half, ptr addrspace(1) %75, align 2, !tbaa !44, !alias.scope !46, !noalias !47
  %77 = insertelement <2 x half> %74, half %76, i64 1
  %78 = tail call <2 x i32> @air.convert.s.v2i32.f.v2f16(<2 x half> %77) #2
  %79 = load i16, ptr addrspace(1) %5, align 2, !tbaa !48, !alias.scope !50, !noalias !51
  %80 = insertelement <3 x i16> undef, i16 %79, i64 0
  %81 = getelementptr inbounds i16, ptr addrspace(1) %5, i64 1
  %82 = load i16, ptr addrspace(1) %81, align 2, !tbaa !48, !alias.scope !50, !noalias !51
  %83 = insertelement <3 x i16> %80, i16 %82, i64 1
  %84 = getelementptr inbounds i16, ptr addrspace(1) %5, i64 2
  %85 = load i16, ptr addrspace(1) %84, align 2, !tbaa !48, !alias.scope !50, !noalias !51
  %86 = insertelement <3 x i16> %83, i16 %85, i64 2
  %87 = tail call fast <3 x half> @air.convert.f.v3f16.u.v3i16(<3 x i16> %86) #2
  %88 = load i32, ptr addrspace(1) %2, align 4, !tbaa !52, !alias.scope !54, !noalias !55
  %89 = insertelement <3 x i32> undef, i32 %88, i64 0
  %90 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1
  %91 = load i32, ptr addrspace(1) %90, align 4, !tbaa !52, !alias.scope !54, !noalias !55
  %92 = insertelement <3 x i32> %89, i32 %91, i64 1
  %93 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 2
  %94 = load i32, ptr addrspace(1) %93, align 4, !tbaa !52, !alias.scope !54, !noalias !55
  %95 = insertelement <3 x i32> %92, i32 %94, i64 2
  %96 = tail call <3 x i16> @air.convert.u.v3i16.u.v3i32(<3 x i32> %95) #2
  %97 = getelementptr inbounds float, ptr addrspace(1) %1, i64 13
  %98 = load float, ptr addrspace(1) %97, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %99 = insertelement <4 x float> undef, float %98, i64 0
  %100 = getelementptr inbounds float, ptr addrspace(1) %1, i64 14
  %101 = load float, ptr addrspace(1) %100, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %102 = insertelement <4 x float> %99, float %101, i64 1
  %103 = getelementptr inbounds float, ptr addrspace(1) %1, i64 15
  %104 = load float, ptr addrspace(1) %103, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %105 = insertelement <4 x float> %102, float %104, i64 2
  %106 = getelementptr inbounds float, ptr addrspace(1) %1, i64 16
  %107 = load float, ptr addrspace(1) %106, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %108 = insertelement <4 x float> %105, float %107, i64 3
  %109 = tail call <4 x i16> @air.convert.s.v4i16.f.v4f32(<4 x float> %108) #2
  %110 = getelementptr inbounds half, ptr addrspace(1) %4, i64 2
  %111 = load half, ptr addrspace(1) %110, align 2, !tbaa !44, !alias.scope !46, !noalias !47
  %112 = insertelement <2 x half> undef, half %111, i64 0
  %113 = getelementptr inbounds half, ptr addrspace(1) %4, i64 3
  %114 = load half, ptr addrspace(1) %113, align 2, !tbaa !44, !alias.scope !46, !noalias !47
  %115 = insertelement <2 x half> %112, half %114, i64 1
  %116 = tail call fast <2 x half> @air.saturate.v2f16(<2 x half> %115) #2
  %117 = getelementptr inbounds half, ptr addrspace(1) %4, i64 4
  %118 = load half, ptr addrspace(1) %117, align 2, !tbaa !44, !alias.scope !46, !noalias !47
  %119 = insertelement <2 x half> undef, half %118, i64 0
  %120 = getelementptr inbounds half, ptr addrspace(1) %4, i64 5
  %121 = load half, ptr addrspace(1) %120, align 2, !tbaa !44, !alias.scope !46, !noalias !47
  %122 = insertelement <2 x half> %119, half %121, i64 1
  %123 = tail call fast <2 x half> @air.saturate.v2f16(<2 x half> %122) #2
  %124 = getelementptr inbounds float, ptr addrspace(1) %1, i64 17
  %125 = load float, ptr addrspace(1) %124, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %126 = insertelement <2 x float> undef, float %125, i64 0
  %127 = getelementptr inbounds float, ptr addrspace(1) %1, i64 18
  %128 = load float, ptr addrspace(1) %127, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %129 = insertelement <2 x float> %126, float %128, i64 1
  %130 = tail call fast <2 x float> @air.fast_trunc.v2f32(<2 x float> %129) #2
  %131 = getelementptr inbounds float, ptr addrspace(1) %1, i64 19
  %132 = load float, ptr addrspace(1) %131, align 4, !tbaa !26, !alias.scope !30, !noalias !33
  %133 = load i32, ptr addrspace(1) %6, align 4, !tbaa !52, !alias.scope !56, !noalias !57
  %134 = tail call fast float @air.fast_ldexp.f32(float %132, i32 %133) #2
  %135 = bitcast <3 x float> %34 to <3 x i32>
  %136 = extractelement <3 x i32> %135, i64 0
  store i32 %136, ptr addrspace(1) %0, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %137 = extractelement <3 x i32> %135, i64 1
  %138 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %137, ptr addrspace(1) %138, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %139 = extractelement <3 x i32> %135, i64 2
  %140 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %139, ptr addrspace(1) %140, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %141 = bitcast <4 x float> %58 to <4 x i32>
  %142 = extractelement <4 x i32> %141, i64 0
  %143 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %142, ptr addrspace(1) %143, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %144 = extractelement <4 x i32> %141, i64 1
  %145 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %144, ptr addrspace(1) %145, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %146 = extractelement <4 x i32> %141, i64 2
  %147 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %146, ptr addrspace(1) %147, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %148 = extractelement <4 x i32> %141, i64 3
  %149 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  store i32 %148, ptr addrspace(1) %149, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %150 = bitcast <4 x float> %59 to <4 x i32>
  %151 = extractelement <4 x i32> %150, i64 0
  %152 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  store i32 %151, ptr addrspace(1) %152, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %153 = extractelement <4 x i32> %150, i64 1
  %154 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  store i32 %153, ptr addrspace(1) %154, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %155 = extractelement <4 x i32> %150, i64 2
  %156 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 9
  store i32 %155, ptr addrspace(1) %156, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %157 = extractelement <4 x i32> %150, i64 3
  %158 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 10
  store i32 %157, ptr addrspace(1) %158, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %159 = extractelement <4 x i64> %72, i64 0
  %160 = trunc i64 %159 to i32
  %161 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 11
  store i32 %160, ptr addrspace(1) %161, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %162 = lshr i64 %159, 32
  %163 = trunc i64 %162 to i32
  %164 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 12
  store i32 %163, ptr addrspace(1) %164, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %165 = extractelement <4 x i64> %72, i64 1
  %166 = trunc i64 %165 to i32
  %167 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 13
  store i32 %166, ptr addrspace(1) %167, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %168 = lshr i64 %165, 32
  %169 = trunc i64 %168 to i32
  %170 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 14
  store i32 %169, ptr addrspace(1) %170, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %171 = extractelement <4 x i64> %72, i64 2
  %172 = trunc i64 %171 to i32
  %173 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 15
  store i32 %172, ptr addrspace(1) %173, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %174 = lshr i64 %171, 32
  %175 = trunc i64 %174 to i32
  %176 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 16
  store i32 %175, ptr addrspace(1) %176, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %177 = extractelement <4 x i64> %72, i64 3
  %178 = trunc i64 %177 to i32
  %179 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 17
  store i32 %178, ptr addrspace(1) %179, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %180 = lshr i64 %177, 32
  %181 = trunc i64 %180 to i32
  %182 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 18
  store i32 %181, ptr addrspace(1) %182, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %183 = extractelement <2 x i32> %78, i64 0
  %184 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 19
  store i32 %183, ptr addrspace(1) %184, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %185 = extractelement <2 x i32> %78, i64 1
  %186 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 20
  store i32 %185, ptr addrspace(1) %186, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %187 = bitcast <3 x half> %87 to <3 x i16>
  %188 = extractelement <3 x i16> %187, i64 0
  %189 = zext i16 %188 to i32
  %190 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 21
  store i32 %189, ptr addrspace(1) %190, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %191 = extractelement <3 x i16> %187, i64 1
  %192 = zext i16 %191 to i32
  %193 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 22
  store i32 %192, ptr addrspace(1) %193, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %194 = extractelement <3 x i16> %187, i64 2
  %195 = zext i16 %194 to i32
  %196 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 23
  store i32 %195, ptr addrspace(1) %196, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %197 = extractelement <3 x i16> %96, i64 0
  %198 = zext i16 %197 to i32
  %199 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 24
  store i32 %198, ptr addrspace(1) %199, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %200 = extractelement <3 x i16> %96, i64 1
  %201 = zext i16 %200 to i32
  %202 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 25
  store i32 %201, ptr addrspace(1) %202, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %203 = extractelement <3 x i16> %96, i64 2
  %204 = zext i16 %203 to i32
  %205 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 26
  store i32 %204, ptr addrspace(1) %205, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %206 = extractelement <4 x i16> %109, i64 0
  %207 = zext i16 %206 to i32
  %208 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 27
  store i32 %207, ptr addrspace(1) %208, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %209 = extractelement <4 x i16> %109, i64 1
  %210 = zext i16 %209 to i32
  %211 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 28
  store i32 %210, ptr addrspace(1) %211, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %212 = extractelement <4 x i16> %109, i64 2
  %213 = zext i16 %212 to i32
  %214 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 29
  store i32 %213, ptr addrspace(1) %214, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %215 = extractelement <4 x i16> %109, i64 3
  %216 = zext i16 %215 to i32
  %217 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 30
  store i32 %216, ptr addrspace(1) %217, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %218 = bitcast <2 x half> %116 to <2 x i16>
  %219 = extractelement <2 x i16> %218, i64 0
  %220 = zext i16 %219 to i32
  %221 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 31
  store i32 %220, ptr addrspace(1) %221, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %222 = extractelement <2 x i16> %218, i64 1
  %223 = zext i16 %222 to i32
  %224 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 32
  store i32 %223, ptr addrspace(1) %224, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %225 = bitcast <2 x half> %123 to <2 x i16>
  %226 = extractelement <2 x i16> %225, i64 0
  %227 = zext i16 %226 to i32
  %228 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 33
  store i32 %227, ptr addrspace(1) %228, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %229 = extractelement <2 x i16> %225, i64 1
  %230 = zext i16 %229 to i32
  %231 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 34
  store i32 %230, ptr addrspace(1) %231, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %232 = bitcast <2 x float> %130 to <2 x i32>
  %233 = extractelement <2 x i32> %232, i64 0
  %234 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 35
  store i32 %233, ptr addrspace(1) %234, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %235 = extractelement <2 x i32> %232, i64 1
  %236 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 36
  store i32 %235, ptr addrspace(1) %236, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  %237 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 37
  %238 = bitcast ptr addrspace(1) %237 to ptr addrspace(1)
  store float %134, ptr addrspace(1) %238, align 4, !tbaa !52, !alias.scope !58, !noalias !59
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x float> @air.convert.f.v4f32.u.v4i64(<4 x i64>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x float> @air.convert.f.v4f32.s.v4i64(<4 x i64>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i64> @air.convert.s.v4i64.f.v4f32(<4 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i32> @air.convert.s.v2i32.f.v2f16(<2 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x half> @air.convert.f.v3f16.u.v3i16(<3 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i16> @air.convert.u.v3i16.u.v3i32(<3 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i16> @air.convert.s.v4i16.f.v4f32(<4 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fma.v3f32(<3 x float>, <3 x float>, <3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x half> @air.saturate.v2f16(<2 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_trunc.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_ldexp.f32(float, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!19, !20, !21}
!llvm.ident = !{!22}
!air.version = !{!23}
!air.language_version = !{!24}
!air.source_file_name = !{!25}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_exact_convert_and_fma, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"uin"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"lin"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hin"}
!17 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort", !"air.arg_name", !"sin_"}
!18 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"iin"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 4, i32 0, i32 0}
!25 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_exact_convert_and_fma.metal"}
!26 = !{!27, !27, i64 0}
!27 = !{!"float", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"air-alias-scope-arg(1)"}
!32 = distinct !{!32, !"air-alias-scopes(kernel_exact_convert_and_fma)"}
!33 = !{!34, !35, !36, !37, !38, !39}
!34 = distinct !{!34, !32, !"air-alias-scope-arg(0)"}
!35 = distinct !{!35, !32, !"air-alias-scope-arg(2)"}
!36 = distinct !{!36, !32, !"air-alias-scope-arg(3)"}
!37 = distinct !{!37, !32, !"air-alias-scope-arg(4)"}
!38 = distinct !{!38, !32, !"air-alias-scope-arg(5)"}
!39 = distinct !{!39, !32, !"air-alias-scope-arg(6)"}
!40 = !{!41, !41, i64 0}
!41 = !{!"long", !28, i64 0}
!42 = !{!36}
!43 = !{!34, !31, !35, !37, !38, !39}
!44 = !{!45, !45, i64 0}
!45 = !{!"half", !28, i64 0}
!46 = !{!37}
!47 = !{!34, !31, !35, !36, !38, !39}
!48 = !{!49, !49, i64 0}
!49 = !{!"short", !28, i64 0}
!50 = !{!38}
!51 = !{!34, !31, !35, !36, !37, !39}
!52 = !{!53, !53, i64 0}
!53 = !{!"int", !28, i64 0}
!54 = !{!35}
!55 = !{!34, !31, !36, !37, !38, !39}
!56 = !{!39}
!57 = !{!34, !31, !35, !36, !37, !38}
!58 = !{!34}
!59 = !{!31, !35, !36, !37, !38, !39}
