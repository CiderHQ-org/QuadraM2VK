; ModuleID = '/tmp/da/tw.bc'
source_filename = "validation/fixtures/public/kernel_three_way_and_shuffle_math.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_three_way_and_shuffle_math(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call fast float @air.convert.f.f32.u.i32(i32 %1) #3
  %4 = insertelement <3 x float> undef, float %3, i64 0
  %5 = fsub fast float 8.000000e+00, %3
  %6 = insertelement <3 x float> %4, float %5, i64 1
  %7 = fmul fast float %3, 2.000000e+00
  %8 = fadd fast float %7, -3.000000e+00
  %9 = insertelement <3 x float> %6, float %8, i64 2
  %10 = fadd fast float %3, 1.600000e+01
  %11 = insertelement <3 x float> undef, float %10, i64 0
  %12 = fsub fast float 3.000000e+00, %3
  %13 = insertelement <3 x float> %11, float %12, i64 1
  %14 = fmul fast float %3, 4.000000e+00
  %15 = insertelement <3 x float> %13, float %14, i64 2
  %16 = fadd fast float %7, -5.000000e+00
  %17 = insertelement <3 x float> undef, float %16, i64 0
  %18 = fadd fast float %3, 1.000000e+00
  %19 = insertelement <3 x float> %17, float %18, i64 1
  %20 = fsub fast float 9.000000e+00, %3
  %21 = insertelement <3 x float> %19, float %20, i64 2
  %22 = tail call fast <3 x float> @air.fast_fmin3.v3f32(<3 x float> %9, <3 x float> %15, <3 x float> %21) #3
  %23 = tail call fast <3 x float> @air.fast_fmax3.v3f32(<3 x float> %9, <3 x float> %15, <3 x float> %21) #3
  %24 = fmul fast float %3, 3.000000e+00
  %25 = fadd fast float %24, -1.000000e+00
  %26 = insertelement <2 x float> undef, float %25, i64 0
  %27 = fsub fast float 7.000000e+00, %7
  %28 = insertelement <2 x float> %26, float %27, i64 1
  %29 = fsub fast float 1.100000e+01, %3
  %30 = insertelement <2 x float> undef, float %29, i64 0
  %31 = fadd fast float %14, -6.000000e+00
  %32 = insertelement <2 x float> %30, float %31, i64 1
  %33 = fadd fast float %3, 2.000000e+00
  %34 = insertelement <2 x float> undef, float %33, i64 0
  %35 = fsub fast float 5.000000e+00, %3
  %36 = insertelement <2 x float> %34, float %35, i64 1
  %37 = tail call fast <2 x float> @air.fast_fmin3.v2f32(<2 x float> %28, <2 x float> %32, <2 x float> %36) #3
  %38 = tail call fast <2 x float> @air.fast_fmax3.v2f32(<2 x float> %28, <2 x float> %32, <2 x float> %36) #3
  %39 = fptrunc float %3 to half
  %40 = fadd fast half %39, 0xH3800
  %41 = insertelement <3 x half> undef, half %40, i64 0
  %42 = fsub fast half 0xH4680, %39
  %43 = insertelement <3 x half> %41, half %42, i64 1
  %44 = fmul fast half %39, 0xH3E00
  %45 = insertelement <3 x half> %43, half %44, i64 2
  %46 = fmul fast half %39, 0xH4000
  %47 = insertelement <3 x half> <half poison, half 0xH3E00, half poison>, half %46, i64 0
  %48 = fsub fast half 0xH48C0, %39
  %49 = insertelement <3 x half> %47, half %48, i64 2
  %50 = fadd fast half %39, 0xH4480
  %51 = insertelement <3 x half> <half 0xH4300, half poison, half poison>, half %50, i64 1
  %52 = fadd fast half %39, 0xHC100
  %53 = insertelement <3 x half> %51, half %52, i64 2
  %54 = tail call fast <3 x half> @air.fmax3.v3f16(<3 x half> %45, <3 x half> %49, <3 x half> %53) #3
  %55 = shufflevector <3 x half> %45, <3 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %56 = fadd fast half %39, 0xHC300
  %57 = insertelement <4 x half> %55, half %56, i64 3
  %58 = shufflevector <3 x half> %49, <3 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %59 = insertelement <4 x half> %58, half 0xH4100, i64 3
  %60 = shufflevector <3 x half> %53, <3 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %61 = fsub fast half 0xH4780, %39
  %62 = insertelement <4 x half> %60, half %61, i64 3
  %63 = tail call fast <4 x half> @air.fmedian3.v4f16(<4 x half> %57, <4 x half> %59, <4 x half> %62) #3
  %64 = trunc i32 %1 to i16
  %65 = add i16 %64, -3
  %66 = insertelement <2 x i16> undef, i16 %65, i64 0
  %67 = shl i16 %64, 1
  %68 = add i16 %67, -7
  %69 = insertelement <2 x i16> %66, i16 %68, i64 1
  %70 = sub i16 5, %64
  %71 = insertelement <2 x i16> undef, i16 %70, i64 0
  %72 = insertelement <2 x i16> %71, i16 %64, i64 1
  %73 = tail call <2 x i16> @air.max.s.v2i16(<2 x i16> %69, <2 x i16> %72) #3
  %74 = tail call i32 @air.min3.u.i32(i32 %1, i32 5, i32 3) #3
  %75 = add i32 %1, 2
  %76 = tail call i32 @air.min3.u.i32(i32 %75, i32 4, i32 9) #3
  %77 = tail call fast half @air.trunc.f16(half %56) #3
  %78 = fsub fast half 0xHB400, %44
  %79 = tail call fast half @air.trunc.f16(half %78) #3
  %80 = tail call fast <3 x float> @air.fast_tanh.v3f32(<3 x float> zeroinitializer) #3
  %81 = tail call fast <2 x float> @air.fast_tanh.v2f32(<2 x float> zeroinitializer) #3
  %82 = and i32 %1, 31
  %83 = shl nuw i32 1, %82
  %84 = tail call fast float @air.convert.f.f32.u.i32(i32 %83) #3
  %85 = insertelement <2 x float> undef, float %84, i64 0
  %86 = and i32 %75, 31
  %87 = shl nuw i32 1, %86
  %88 = tail call fast float @air.convert.f.f32.u.i32(i32 %87) #3
  %89 = insertelement <2 x float> %85, float %88, i64 1
  %90 = tail call fast <2 x float> @air.fast_log2.v2f32(<2 x float> %89) #3
  %91 = insertelement <4 x float> undef, float %3, i64 0
  %92 = insertelement <4 x float> <float poison, float 1.000000e+00, float 2.000000e+00, float 2.500000e-01>, float %3, i64 0
  %93 = tail call fast <4 x float> @air.pow.v4f32(<4 x float> <float 2.000000e+00, float 4.000000e+00, float 5.000000e-01, float 1.600000e+01>, <4 x float> %92) #3
  %94 = freeze float %3
  %95 = fadd reassoc nsz arcp contract afn float %94, 1.000000e+02
  %96 = tail call fast float @air.quad_shuffle.f32(float %95, i16 2) #4
  %97 = tail call fast float @air.quad_shuffle.f32(float %95, i16 0) #4
  %98 = insertelement <4 x float> %91, float %7, i64 1
  %99 = insertelement <4 x float> %98, float %24, i64 2
  %100 = insertelement <4 x float> %99, float %14, i64 3
  %101 = freeze <4 x float> %100
  %102 = tail call fast <4 x float> @air.quad_shuffle.v4f32(<4 x float> %101, i16 1) #4
  %103 = mul i32 %1, 38
  %104 = bitcast <3 x float> %22 to <3 x i32>
  %105 = extractelement <3 x i32> %104, i64 0
  %106 = zext i32 %103 to i64
  %107 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %106
  store i32 %105, ptr addrspace(1) %107, align 4, !tbaa !21, !alias.scope !25
  %108 = extractelement <3 x i32> %104, i64 1
  %109 = or i32 %103, 1
  %110 = zext i32 %109 to i64
  %111 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %110
  store i32 %108, ptr addrspace(1) %111, align 4, !tbaa !21, !alias.scope !25
  %112 = extractelement <3 x i32> %104, i64 2
  %113 = add i32 %103, 2
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %114
  store i32 %112, ptr addrspace(1) %115, align 4, !tbaa !21, !alias.scope !25
  %116 = bitcast <3 x float> %23 to <3 x i32>
  %117 = extractelement <3 x i32> %116, i64 0
  %118 = add i32 %103, 3
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %119
  store i32 %117, ptr addrspace(1) %120, align 4, !tbaa !21, !alias.scope !25
  %121 = extractelement <3 x i32> %116, i64 1
  %122 = add i32 %103, 4
  %123 = zext i32 %122 to i64
  %124 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %123
  store i32 %121, ptr addrspace(1) %124, align 4, !tbaa !21, !alias.scope !25
  %125 = extractelement <3 x i32> %116, i64 2
  %126 = add i32 %103, 5
  %127 = zext i32 %126 to i64
  %128 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %127
  store i32 %125, ptr addrspace(1) %128, align 4, !tbaa !21, !alias.scope !25
  %129 = bitcast <2 x float> %37 to <2 x i32>
  %130 = extractelement <2 x i32> %129, i64 0
  %131 = add i32 %103, 6
  %132 = zext i32 %131 to i64
  %133 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %132
  store i32 %130, ptr addrspace(1) %133, align 4, !tbaa !21, !alias.scope !25
  %134 = extractelement <2 x i32> %129, i64 1
  %135 = add i32 %103, 7
  %136 = zext i32 %135 to i64
  %137 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %136
  store i32 %134, ptr addrspace(1) %137, align 4, !tbaa !21, !alias.scope !25
  %138 = bitcast <2 x float> %38 to <2 x i32>
  %139 = extractelement <2 x i32> %138, i64 0
  %140 = add i32 %103, 8
  %141 = zext i32 %140 to i64
  %142 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %141
  store i32 %139, ptr addrspace(1) %142, align 4, !tbaa !21, !alias.scope !25
  %143 = extractelement <2 x i32> %138, i64 1
  %144 = add i32 %103, 9
  %145 = zext i32 %144 to i64
  %146 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %145
  store i32 %143, ptr addrspace(1) %146, align 4, !tbaa !21, !alias.scope !25
  %147 = bitcast <3 x half> %54 to <3 x i16>
  %148 = extractelement <3 x i16> %147, i64 0
  %149 = zext i16 %148 to i32
  %150 = add i32 %103, 10
  %151 = zext i32 %150 to i64
  %152 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %151
  store i32 %149, ptr addrspace(1) %152, align 4, !tbaa !21, !alias.scope !25
  %153 = extractelement <3 x i16> %147, i64 1
  %154 = zext i16 %153 to i32
  %155 = add i32 %103, 11
  %156 = zext i32 %155 to i64
  %157 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %156
  store i32 %154, ptr addrspace(1) %157, align 4, !tbaa !21, !alias.scope !25
  %158 = extractelement <3 x i16> %147, i64 2
  %159 = zext i16 %158 to i32
  %160 = add i32 %103, 12
  %161 = zext i32 %160 to i64
  %162 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %161
  store i32 %159, ptr addrspace(1) %162, align 4, !tbaa !21, !alias.scope !25
  %163 = bitcast <4 x half> %63 to <4 x i16>
  %164 = extractelement <4 x i16> %163, i64 0
  %165 = zext i16 %164 to i32
  %166 = add i32 %103, 13
  %167 = zext i32 %166 to i64
  %168 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %167
  store i32 %165, ptr addrspace(1) %168, align 4, !tbaa !21, !alias.scope !25
  %169 = extractelement <4 x i16> %163, i64 1
  %170 = zext i16 %169 to i32
  %171 = add i32 %103, 14
  %172 = zext i32 %171 to i64
  %173 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %172
  store i32 %170, ptr addrspace(1) %173, align 4, !tbaa !21, !alias.scope !25
  %174 = extractelement <4 x i16> %163, i64 2
  %175 = zext i16 %174 to i32
  %176 = add i32 %103, 15
  %177 = zext i32 %176 to i64
  %178 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %177
  store i32 %175, ptr addrspace(1) %178, align 4, !tbaa !21, !alias.scope !25
  %179 = extractelement <4 x i16> %163, i64 3
  %180 = zext i16 %179 to i32
  %181 = add i32 %103, 16
  %182 = zext i32 %181 to i64
  %183 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %182
  store i32 %180, ptr addrspace(1) %183, align 4, !tbaa !21, !alias.scope !25
  %184 = extractelement <2 x i16> %73, i64 0
  %185 = zext i16 %184 to i32
  %186 = add i32 %103, 17
  %187 = zext i32 %186 to i64
  %188 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %187
  store i32 %185, ptr addrspace(1) %188, align 4, !tbaa !21, !alias.scope !25
  %189 = extractelement <2 x i16> %73, i64 1
  %190 = zext i16 %189 to i32
  %191 = add i32 %103, 18
  %192 = zext i32 %191 to i64
  %193 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %192
  store i32 %190, ptr addrspace(1) %193, align 4, !tbaa !21, !alias.scope !25
  %194 = add i32 %103, 19
  %195 = zext i32 %194 to i64
  %196 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %195
  store i32 %74, ptr addrspace(1) %196, align 4, !tbaa !21, !alias.scope !25
  %197 = add i32 %103, 20
  %198 = zext i32 %197 to i64
  %199 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %198
  store i32 %76, ptr addrspace(1) %199, align 4, !tbaa !21, !alias.scope !25
  %200 = bitcast half %77 to i16
  %201 = zext i16 %200 to i32
  %202 = add i32 %103, 21
  %203 = zext i32 %202 to i64
  %204 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %203
  store i32 %201, ptr addrspace(1) %204, align 4, !tbaa !21, !alias.scope !25
  %205 = bitcast half %79 to i16
  %206 = zext i16 %205 to i32
  %207 = add i32 %103, 22
  %208 = zext i32 %207 to i64
  %209 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %208
  store i32 %206, ptr addrspace(1) %209, align 4, !tbaa !21, !alias.scope !25
  %210 = bitcast <3 x float> %80 to <3 x i32>
  %211 = extractelement <3 x i32> %210, i64 0
  %212 = add i32 %103, 23
  %213 = zext i32 %212 to i64
  %214 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %213
  store i32 %211, ptr addrspace(1) %214, align 4, !tbaa !21, !alias.scope !25
  %215 = extractelement <3 x i32> %210, i64 2
  %216 = add i32 %103, 24
  %217 = zext i32 %216 to i64
  %218 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %217
  store i32 %215, ptr addrspace(1) %218, align 4, !tbaa !21, !alias.scope !25
  %219 = bitcast <2 x float> %81 to <2 x i32>
  %220 = extractelement <2 x i32> %219, i64 1
  %221 = add i32 %103, 25
  %222 = zext i32 %221 to i64
  %223 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %222
  store i32 %220, ptr addrspace(1) %223, align 4, !tbaa !21, !alias.scope !25
  %224 = bitcast <2 x float> %90 to <2 x i32>
  %225 = extractelement <2 x i32> %224, i64 0
  %226 = add i32 %103, 26
  %227 = zext i32 %226 to i64
  %228 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %227
  store i32 %225, ptr addrspace(1) %228, align 4, !tbaa !21, !alias.scope !25
  %229 = extractelement <2 x i32> %224, i64 1
  %230 = add i32 %103, 27
  %231 = zext i32 %230 to i64
  %232 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %231
  store i32 %229, ptr addrspace(1) %232, align 4, !tbaa !21, !alias.scope !25
  %233 = bitcast <4 x float> %93 to <4 x i32>
  %234 = extractelement <4 x i32> %233, i64 0
  %235 = add i32 %103, 28
  %236 = zext i32 %235 to i64
  %237 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %236
  store i32 %234, ptr addrspace(1) %237, align 4, !tbaa !21, !alias.scope !25
  %238 = extractelement <4 x i32> %233, i64 1
  %239 = add i32 %103, 29
  %240 = zext i32 %239 to i64
  %241 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %240
  store i32 %238, ptr addrspace(1) %241, align 4, !tbaa !21, !alias.scope !25
  %242 = extractelement <4 x i32> %233, i64 2
  %243 = add i32 %103, 30
  %244 = zext i32 %243 to i64
  %245 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %244
  store i32 %242, ptr addrspace(1) %245, align 4, !tbaa !21, !alias.scope !25
  %246 = extractelement <4 x i32> %233, i64 3
  %247 = add i32 %103, 31
  %248 = zext i32 %247 to i64
  %249 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %248
  store i32 %246, ptr addrspace(1) %249, align 4, !tbaa !21, !alias.scope !25
  %250 = add i32 %103, 32
  %251 = zext i32 %250 to i64
  %252 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %251
  %253 = bitcast ptr addrspace(1) %252 to ptr addrspace(1)
  store float %96, ptr addrspace(1) %253, align 4, !tbaa !21, !alias.scope !25
  %254 = add i32 %103, 33
  %255 = zext i32 %254 to i64
  %256 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %255
  %257 = bitcast ptr addrspace(1) %256 to ptr addrspace(1)
  store float %97, ptr addrspace(1) %257, align 4, !tbaa !21, !alias.scope !25
  %258 = bitcast <4 x float> %102 to <4 x i32>
  %259 = extractelement <4 x i32> %258, i64 0
  %260 = add i32 %103, 34
  %261 = zext i32 %260 to i64
  %262 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %261
  store i32 %259, ptr addrspace(1) %262, align 4, !tbaa !21, !alias.scope !25
  %263 = extractelement <4 x i32> %258, i64 1
  %264 = add i32 %103, 35
  %265 = zext i32 %264 to i64
  %266 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %265
  store i32 %263, ptr addrspace(1) %266, align 4, !tbaa !21, !alias.scope !25
  %267 = extractelement <4 x i32> %258, i64 2
  %268 = add i32 %103, 36
  %269 = zext i32 %268 to i64
  %270 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %269
  store i32 %267, ptr addrspace(1) %270, align 4, !tbaa !21, !alias.scope !25
  %271 = extractelement <4 x i32> %258, i64 3
  %272 = add i32 %103, 37
  %273 = zext i32 %272 to i64
  %274 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %273
  store i32 %271, ptr addrspace(1) %274, align 4, !tbaa !21, !alias.scope !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_fmin3.v3f32(<3 x float>, <3 x float>, <3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_fmax3.v3f32(<3 x float>, <3 x float>, <3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_fmin3.v2f32(<2 x float>, <2 x float>, <2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_fmax3.v2f32(<2 x float>, <2 x float>, <2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x half> @air.fmax3.v3f16(<3 x half>, <3 x half>, <3 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.fmedian3.v4f16(<4 x half>, <4 x half>, <4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i16> @air.max.s.v2i16(<2 x i16>, <2 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.min3.u.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.trunc.f16(half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_tanh.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_tanh.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_log2.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x float> @air.pow.v4f32(<4 x float>, <4 x float>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.quad_shuffle.f32(float, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x float> @air.quad_shuffle.v4f32(<4 x float>, i16) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent mustprogress nounwind willreturn }
attributes #3 = { nounwind willreturn memory(none) }
attributes #4 = { convergent nounwind willreturn }

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
!9 = !{ptr @kernel_three_way_and_shuffle_math, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_three_way_and_shuffle_math.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_three_way_and_shuffle_math)"}
