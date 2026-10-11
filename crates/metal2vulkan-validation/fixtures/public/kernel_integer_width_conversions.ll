; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'iwc.bc'
source_filename = "validation/fixtures/public/kernel_integer_width_conversions.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: write)
define void @kernel_integer_width_conversions(ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add nsw i32 %1, -3
  %4 = insertelement <4 x i32> undef, i32 %3, i64 0
  %5 = mul i32 %1, 300
  %6 = add nsw i32 %5, -500
  %7 = insertelement <4 x i32> %4, i32 %6, i64 1
  %8 = mul i32 %1, -70000
  %9 = or i32 %8, 5
  %10 = insertelement <4 x i32> %7, i32 %9, i64 2
  %11 = mul i32 %1, 7
  %12 = insertelement <4 x i32> %10, i32 %11, i64 3
  %13 = add i32 %1, 1
  %14 = insertelement <4 x i32> undef, i32 %13, i64 0
  %15 = insertelement <4 x i32> %14, i32 %5, i64 1
  %16 = mul i32 %1, 70000
  %17 = insertelement <4 x i32> %15, i32 %16, i64 2
  %18 = add i32 %1, -256
  %19 = insertelement <4 x i32> %17, i32 %18, i64 3
  %20 = trunc i32 %1 to i16
  %21 = add i16 %20, -4
  %22 = insertelement <4 x i16> undef, i16 %21, i64 0
  %23 = mul i16 %20, 100
  %24 = insertelement <4 x i16> %22, i16 %23, i64 1
  %25 = mul i16 %20, -300
  %26 = insertelement <4 x i16> %24, i16 %25, i64 2
  %27 = add i16 %20, 9
  %28 = insertelement <4 x i16> %26, i16 %27, i64 3
  %29 = trunc i32 %1 to i8
  %30 = add i8 %29, -3
  %31 = insertelement <4 x i8> undef, i8 %30, i64 0
  %32 = mul i8 %29, 5
  %33 = add i8 %32, -60
  %34 = insertelement <4 x i8> %31, i8 %33, i64 1
  %35 = mul i8 %29, -7
  %36 = insertelement <4 x i8> %34, i8 %35, i64 2
  %37 = add i8 %29, 1
  %38 = insertelement <4 x i8> %36, i8 %37, i64 3
  %39 = insertelement <3 x i8> undef, i8 %29, i64 0
  %40 = mul i8 %29, 30
  %41 = insertelement <3 x i8> %39, i8 %40, i64 1
  %42 = add nsw i32 %1, 200
  %43 = trunc i32 %42 to i8
  %44 = insertelement <3 x i8> %41, i8 %43, i64 2
  %45 = tail call fast half @air.convert.f.f16.s.i32(i32 %1) #2
  %46 = fadd fast half %45, 0xH3A00
  %47 = insertelement <4 x half> undef, half %46, i64 0
  %48 = fmul fast half %45, 0xH4000
  %49 = fadd fast half %48, 0xH3800
  %50 = insertelement <4 x half> %47, half %49, i64 1
  %51 = fmul fast half %45, 0xH4800
  %52 = insertelement <4 x half> %50, half %51, i64 2
  %53 = tail call fast half @air.convert.f.f16.s.i32(i32 %42) #2
  %54 = insertelement <4 x half> %52, half %53, i64 3
  %55 = shufflevector <4 x half> %52, <4 x half> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %56 = tail call <4 x i64> @air.convert.s.v4i64.s.v4i32(<4 x i32> %12) #2
  %57 = mul <4 x i64> %56, splat (i64 1000000)
  %58 = tail call <4 x i64> @air.convert.u.v4i64.u.v4i32(<4 x i32> %19) #2
  %59 = mul <4 x i64> %58, splat (i64 1000)
  %60 = icmp sgt <4 x i32> %12, zeroinitializer
  %61 = shufflevector <4 x i1> %60, <4 x i1> poison, <2 x i32> <i32 0, i32 1>
  %62 = tail call <4 x i16> @air.convert.u.v4i16.u.v4i1(<4 x i1> %60) #2
  %63 = tail call <4 x i16> @air.convert.u.v4i16.f.v4f16(<4 x half> %54) #2
  %64 = tail call <4 x i32> @air.convert.u.v4i32.u.v4i16(<4 x i16> %63) #2
  %65 = tail call <3 x i16> @air.convert.u.v3i16.u.v3i8(<3 x i8> %44) #2
  %66 = shufflevector <4 x i8> %34, <4 x i8> poison, <2 x i32> <i32 0, i32 1>
  %67 = tail call <2 x i32> @air.convert.s.v2i32.s.v2i8(<2 x i8> %66) #2
  %68 = shufflevector <4 x i32> %7, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %69 = tail call <2 x i16> @air.convert.s.v2i16.s.v2i32(<2 x i32> %68) #2
  %70 = tail call <4 x i16> @air.convert.u.v4i16.u.v4i32(<4 x i32> %19) #2
  %71 = tail call fast <4 x half> @air.convert.f.v4f16.u.v4i16(<4 x i16> %70) #2
  %72 = tail call <4 x i8> @air.convert.u.v4i8.s.v4i32(<4 x i32> %12) #2
  %73 = tail call <4 x i64> @air.convert.u.v4i64.s.v4i32(<4 x i32> %12) #2
  %74 = tail call <3 x i8> @air.convert.u.v3i8.f.v3f16(<3 x half> %55) #2
  %75 = tail call <3 x i32> @air.convert.u.v3i32.u.v3i8(<3 x i8> %44) #2
  %76 = tail call <3 x i16> @air.convert.u.v3i16.f.v3f16(<3 x half> %55) #2
  %77 = shufflevector <4 x i32> %15, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %78 = tail call <2 x i64> @air.convert.u.v2i64.u.v2i32(<2 x i32> %77) #2
  %79 = tail call <4 x i8> @air.convert.s.v4i8.u.v4i32(<4 x i32> %19) #2
  %80 = tail call <4 x i8> @air.convert.s.v4i8.s.v4i32(<4 x i32> %12) #2
  %81 = tail call <4 x i8> @air.convert.u.v4i8.u.v4i32(<4 x i32> %19) #2
  %82 = tail call <4 x i32> @air.convert.s.v4i32.u.v4i8(<4 x i8> %81) #2
  %83 = tail call <4 x i32> @air.convert.s.v4i32.u.v4i64(<4 x i64> %59) #2
  %84 = tail call <4 x i32> @air.convert.s.v4i32.s.v4i8(<4 x i8> %38) #2
  %85 = tail call <4 x i32> @air.convert.s.v4i32.s.v4i64(<4 x i64> %57) #2
  %86 = tail call <4 x i32> @air.convert.s.v4i32.s.v4i16(<4 x i16> %28) #2
  %87 = tail call <4 x i16> @air.convert.s.v4i16.s.v4i32(<4 x i32> %12) #2
  %88 = tail call <2 x i16> @air.convert.s.v2i16.u.v2i1(<2 x i1> %61) #2
  %89 = mul i32 %1, 3
  %90 = insertelement <4 x i32> %14, i32 %89, i64 1
  %91 = mul i32 %1, 5
  %92 = add i32 %91, 2
  %93 = insertelement <4 x i32> %90, i32 %92, i64 2
  %94 = add i32 %11, 9
  %95 = insertelement <4 x i32> %93, i32 %94, i64 3
  %96 = tail call fast <4 x half> @air.convert.f.v4f16.u.v4i32(<4 x i32> %95) #2
  %97 = mul i32 %1, 90
  %98 = extractelement <4 x i16> %62, i64 0
  %99 = zext i16 %98 to i32
  %100 = zext i32 %97 to i64
  %101 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %100
  store i32 %99, ptr addrspace(1) %101, align 4, !tbaa !21, !alias.scope !25
  %102 = extractelement <4 x i16> %62, i64 1
  %103 = zext i16 %102 to i32
  %104 = or i32 %97, 1
  %105 = zext i32 %104 to i64
  %106 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %105
  store i32 %103, ptr addrspace(1) %106, align 4, !tbaa !21, !alias.scope !25
  %107 = extractelement <4 x i16> %62, i64 2
  %108 = zext i16 %107 to i32
  %109 = add i32 %97, 2
  %110 = zext i32 %109 to i64
  %111 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %110
  store i32 %108, ptr addrspace(1) %111, align 4, !tbaa !21, !alias.scope !25
  %112 = extractelement <4 x i16> %62, i64 3
  %113 = zext i16 %112 to i32
  %114 = add i32 %97, 3
  %115 = zext i32 %114 to i64
  %116 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %115
  store i32 %113, ptr addrspace(1) %116, align 4, !tbaa !21, !alias.scope !25
  %117 = extractelement <4 x i32> %64, i64 0
  %118 = add i32 %97, 4
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %119
  store i32 %117, ptr addrspace(1) %120, align 4, !tbaa !21, !alias.scope !25
  %121 = extractelement <4 x i32> %64, i64 1
  %122 = add i32 %97, 5
  %123 = zext i32 %122 to i64
  %124 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %123
  store i32 %121, ptr addrspace(1) %124, align 4, !tbaa !21, !alias.scope !25
  %125 = extractelement <4 x i32> %64, i64 2
  %126 = add i32 %97, 6
  %127 = zext i32 %126 to i64
  %128 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %127
  store i32 %125, ptr addrspace(1) %128, align 4, !tbaa !21, !alias.scope !25
  %129 = extractelement <4 x i32> %64, i64 3
  %130 = add i32 %97, 7
  %131 = zext i32 %130 to i64
  %132 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %131
  store i32 %129, ptr addrspace(1) %132, align 4, !tbaa !21, !alias.scope !25
  %133 = extractelement <3 x i16> %65, i64 0
  %134 = zext i16 %133 to i32
  %135 = add i32 %97, 8
  %136 = zext i32 %135 to i64
  %137 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %136
  store i32 %134, ptr addrspace(1) %137, align 4, !tbaa !21, !alias.scope !25
  %138 = extractelement <3 x i16> %65, i64 1
  %139 = zext i16 %138 to i32
  %140 = add i32 %97, 9
  %141 = zext i32 %140 to i64
  %142 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %141
  store i32 %139, ptr addrspace(1) %142, align 4, !tbaa !21, !alias.scope !25
  %143 = extractelement <3 x i16> %65, i64 2
  %144 = zext i16 %143 to i32
  %145 = add i32 %97, 10
  %146 = zext i32 %145 to i64
  %147 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %146
  store i32 %144, ptr addrspace(1) %147, align 4, !tbaa !21, !alias.scope !25
  %148 = extractelement <2 x i32> %67, i64 0
  %149 = add i32 %97, 11
  %150 = zext i32 %149 to i64
  %151 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %150
  store i32 %148, ptr addrspace(1) %151, align 4, !tbaa !21, !alias.scope !25
  %152 = extractelement <2 x i32> %67, i64 1
  %153 = add i32 %97, 12
  %154 = zext i32 %153 to i64
  %155 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %154
  store i32 %152, ptr addrspace(1) %155, align 4, !tbaa !21, !alias.scope !25
  %156 = extractelement <2 x i16> %69, i64 0
  %157 = zext i16 %156 to i32
  %158 = add i32 %97, 13
  %159 = zext i32 %158 to i64
  %160 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %159
  store i32 %157, ptr addrspace(1) %160, align 4, !tbaa !21, !alias.scope !25
  %161 = extractelement <2 x i16> %69, i64 1
  %162 = zext i16 %161 to i32
  %163 = add i32 %97, 14
  %164 = zext i32 %163 to i64
  %165 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %164
  store i32 %162, ptr addrspace(1) %165, align 4, !tbaa !21, !alias.scope !25
  %166 = bitcast <4 x half> %71 to <4 x i16>
  %167 = extractelement <4 x i16> %166, i64 0
  %168 = zext i16 %167 to i32
  %169 = add i32 %97, 15
  %170 = zext i32 %169 to i64
  %171 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %170
  store i32 %168, ptr addrspace(1) %171, align 4, !tbaa !21, !alias.scope !25
  %172 = extractelement <4 x i16> %166, i64 1
  %173 = zext i16 %172 to i32
  %174 = add i32 %97, 16
  %175 = zext i32 %174 to i64
  %176 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %175
  store i32 %173, ptr addrspace(1) %176, align 4, !tbaa !21, !alias.scope !25
  %177 = extractelement <4 x i16> %166, i64 2
  %178 = zext i16 %177 to i32
  %179 = add i32 %97, 17
  %180 = zext i32 %179 to i64
  %181 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %180
  store i32 %178, ptr addrspace(1) %181, align 4, !tbaa !21, !alias.scope !25
  %182 = extractelement <4 x i16> %166, i64 3
  %183 = zext i16 %182 to i32
  %184 = add i32 %97, 18
  %185 = zext i32 %184 to i64
  %186 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %185
  store i32 %183, ptr addrspace(1) %186, align 4, !tbaa !21, !alias.scope !25
  %187 = extractelement <4 x i8> %72, i64 0
  %188 = zext i8 %187 to i32
  %189 = add i32 %97, 19
  %190 = zext i32 %189 to i64
  %191 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %190
  store i32 %188, ptr addrspace(1) %191, align 4, !tbaa !21, !alias.scope !25
  %192 = extractelement <4 x i8> %72, i64 1
  %193 = zext i8 %192 to i32
  %194 = add i32 %97, 20
  %195 = zext i32 %194 to i64
  %196 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %195
  store i32 %193, ptr addrspace(1) %196, align 4, !tbaa !21, !alias.scope !25
  %197 = extractelement <4 x i8> %72, i64 2
  %198 = zext i8 %197 to i32
  %199 = add i32 %97, 21
  %200 = zext i32 %199 to i64
  %201 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %200
  store i32 %198, ptr addrspace(1) %201, align 4, !tbaa !21, !alias.scope !25
  %202 = extractelement <4 x i8> %72, i64 3
  %203 = zext i8 %202 to i32
  %204 = add i32 %97, 22
  %205 = zext i32 %204 to i64
  %206 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %205
  store i32 %203, ptr addrspace(1) %206, align 4, !tbaa !21, !alias.scope !25
  %207 = extractelement <4 x i64> %58, i64 0
  %208 = trunc i64 %207 to i32
  %209 = add i32 %97, 23
  %210 = zext i32 %209 to i64
  %211 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %210
  store i32 %208, ptr addrspace(1) %211, align 4, !tbaa !21, !alias.scope !25
  %212 = lshr i64 %207, 32
  %213 = trunc i64 %212 to i32
  %214 = add i32 %97, 24
  %215 = zext i32 %214 to i64
  %216 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %215
  store i32 %213, ptr addrspace(1) %216, align 4, !tbaa !21, !alias.scope !25
  %217 = extractelement <4 x i64> %58, i64 1
  %218 = trunc i64 %217 to i32
  %219 = add i32 %97, 25
  %220 = zext i32 %219 to i64
  %221 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %220
  store i32 %218, ptr addrspace(1) %221, align 4, !tbaa !21, !alias.scope !25
  %222 = lshr i64 %217, 32
  %223 = trunc i64 %222 to i32
  %224 = add i32 %97, 26
  %225 = zext i32 %224 to i64
  %226 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %225
  store i32 %223, ptr addrspace(1) %226, align 4, !tbaa !21, !alias.scope !25
  %227 = extractelement <4 x i64> %58, i64 2
  %228 = trunc i64 %227 to i32
  %229 = add i32 %97, 27
  %230 = zext i32 %229 to i64
  %231 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %230
  store i32 %228, ptr addrspace(1) %231, align 4, !tbaa !21, !alias.scope !25
  %232 = lshr i64 %227, 32
  %233 = trunc i64 %232 to i32
  %234 = add i32 %97, 28
  %235 = zext i32 %234 to i64
  %236 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %235
  store i32 %233, ptr addrspace(1) %236, align 4, !tbaa !21, !alias.scope !25
  %237 = extractelement <4 x i64> %58, i64 3
  %238 = trunc i64 %237 to i32
  %239 = add i32 %97, 29
  %240 = zext i32 %239 to i64
  %241 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %240
  store i32 %238, ptr addrspace(1) %241, align 4, !tbaa !21, !alias.scope !25
  %242 = lshr i64 %237, 32
  %243 = trunc i64 %242 to i32
  %244 = add i32 %97, 30
  %245 = zext i32 %244 to i64
  %246 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %245
  store i32 %243, ptr addrspace(1) %246, align 4, !tbaa !21, !alias.scope !25
  %247 = extractelement <4 x i64> %73, i64 0
  %248 = trunc i64 %247 to i32
  %249 = add i32 %97, 31
  %250 = zext i32 %249 to i64
  %251 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %250
  store i32 %248, ptr addrspace(1) %251, align 4, !tbaa !21, !alias.scope !25
  %252 = lshr i64 %247, 32
  %253 = trunc i64 %252 to i32
  %254 = add i32 %97, 32
  %255 = zext i32 %254 to i64
  %256 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %255
  store i32 %253, ptr addrspace(1) %256, align 4, !tbaa !21, !alias.scope !25
  %257 = extractelement <4 x i64> %73, i64 1
  %258 = trunc i64 %257 to i32
  %259 = add i32 %97, 33
  %260 = zext i32 %259 to i64
  %261 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %260
  store i32 %258, ptr addrspace(1) %261, align 4, !tbaa !21, !alias.scope !25
  %262 = lshr i64 %257, 32
  %263 = trunc i64 %262 to i32
  %264 = add i32 %97, 34
  %265 = zext i32 %264 to i64
  %266 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %265
  store i32 %263, ptr addrspace(1) %266, align 4, !tbaa !21, !alias.scope !25
  %267 = extractelement <4 x i64> %73, i64 2
  %268 = trunc i64 %267 to i32
  %269 = add i32 %97, 35
  %270 = zext i32 %269 to i64
  %271 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %270
  store i32 %268, ptr addrspace(1) %271, align 4, !tbaa !21, !alias.scope !25
  %272 = lshr i64 %267, 32
  %273 = trunc i64 %272 to i32
  %274 = add i32 %97, 36
  %275 = zext i32 %274 to i64
  %276 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %275
  store i32 %273, ptr addrspace(1) %276, align 4, !tbaa !21, !alias.scope !25
  %277 = extractelement <4 x i64> %73, i64 3
  %278 = trunc i64 %277 to i32
  %279 = add i32 %97, 37
  %280 = zext i32 %279 to i64
  %281 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %280
  store i32 %278, ptr addrspace(1) %281, align 4, !tbaa !21, !alias.scope !25
  %282 = lshr i64 %277, 32
  %283 = trunc i64 %282 to i32
  %284 = add i32 %97, 38
  %285 = zext i32 %284 to i64
  %286 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %285
  store i32 %283, ptr addrspace(1) %286, align 4, !tbaa !21, !alias.scope !25
  %287 = extractelement <3 x i8> %74, i64 0
  %288 = zext i8 %287 to i32
  %289 = add i32 %97, 39
  %290 = zext i32 %289 to i64
  %291 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %290
  store i32 %288, ptr addrspace(1) %291, align 4, !tbaa !21, !alias.scope !25
  %292 = extractelement <3 x i8> %74, i64 1
  %293 = zext i8 %292 to i32
  %294 = add i32 %97, 40
  %295 = zext i32 %294 to i64
  %296 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %295
  store i32 %293, ptr addrspace(1) %296, align 4, !tbaa !21, !alias.scope !25
  %297 = extractelement <3 x i8> %74, i64 2
  %298 = zext i8 %297 to i32
  %299 = add i32 %97, 41
  %300 = zext i32 %299 to i64
  %301 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %300
  store i32 %298, ptr addrspace(1) %301, align 4, !tbaa !21, !alias.scope !25
  %302 = extractelement <3 x i32> %75, i64 0
  %303 = add i32 %97, 42
  %304 = zext i32 %303 to i64
  %305 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %304
  store i32 %302, ptr addrspace(1) %305, align 4, !tbaa !21, !alias.scope !25
  %306 = extractelement <3 x i32> %75, i64 1
  %307 = add i32 %97, 43
  %308 = zext i32 %307 to i64
  %309 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %308
  store i32 %306, ptr addrspace(1) %309, align 4, !tbaa !21, !alias.scope !25
  %310 = extractelement <3 x i32> %75, i64 2
  %311 = add i32 %97, 44
  %312 = zext i32 %311 to i64
  %313 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %312
  store i32 %310, ptr addrspace(1) %313, align 4, !tbaa !21, !alias.scope !25
  %314 = extractelement <3 x i16> %76, i64 0
  %315 = zext i16 %314 to i32
  %316 = add i32 %97, 45
  %317 = zext i32 %316 to i64
  %318 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %317
  store i32 %315, ptr addrspace(1) %318, align 4, !tbaa !21, !alias.scope !25
  %319 = extractelement <3 x i16> %76, i64 1
  %320 = zext i16 %319 to i32
  %321 = add i32 %97, 46
  %322 = zext i32 %321 to i64
  %323 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %322
  store i32 %320, ptr addrspace(1) %323, align 4, !tbaa !21, !alias.scope !25
  %324 = extractelement <3 x i16> %76, i64 2
  %325 = zext i16 %324 to i32
  %326 = add i32 %97, 47
  %327 = zext i32 %326 to i64
  %328 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %327
  store i32 %325, ptr addrspace(1) %328, align 4, !tbaa !21, !alias.scope !25
  %329 = extractelement <2 x i64> %78, i64 0
  %330 = trunc i64 %329 to i32
  %331 = add i32 %97, 48
  %332 = zext i32 %331 to i64
  %333 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %332
  store i32 %330, ptr addrspace(1) %333, align 4, !tbaa !21, !alias.scope !25
  %334 = lshr i64 %329, 32
  %335 = trunc i64 %334 to i32
  %336 = add i32 %97, 49
  %337 = zext i32 %336 to i64
  %338 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %337
  store i32 %335, ptr addrspace(1) %338, align 4, !tbaa !21, !alias.scope !25
  %339 = extractelement <2 x i64> %78, i64 1
  %340 = trunc i64 %339 to i32
  %341 = add i32 %97, 50
  %342 = zext i32 %341 to i64
  %343 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %342
  store i32 %340, ptr addrspace(1) %343, align 4, !tbaa !21, !alias.scope !25
  %344 = lshr i64 %339, 32
  %345 = trunc i64 %344 to i32
  %346 = add i32 %97, 51
  %347 = zext i32 %346 to i64
  %348 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %347
  store i32 %345, ptr addrspace(1) %348, align 4, !tbaa !21, !alias.scope !25
  %349 = extractelement <4 x i8> %79, i64 0
  %350 = zext i8 %349 to i32
  %351 = add i32 %97, 52
  %352 = zext i32 %351 to i64
  %353 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %352
  store i32 %350, ptr addrspace(1) %353, align 4, !tbaa !21, !alias.scope !25
  %354 = extractelement <4 x i8> %79, i64 1
  %355 = zext i8 %354 to i32
  %356 = add i32 %97, 53
  %357 = zext i32 %356 to i64
  %358 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %357
  store i32 %355, ptr addrspace(1) %358, align 4, !tbaa !21, !alias.scope !25
  %359 = extractelement <4 x i8> %79, i64 2
  %360 = zext i8 %359 to i32
  %361 = add i32 %97, 54
  %362 = zext i32 %361 to i64
  %363 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %362
  store i32 %360, ptr addrspace(1) %363, align 4, !tbaa !21, !alias.scope !25
  %364 = extractelement <4 x i8> %79, i64 3
  %365 = zext i8 %364 to i32
  %366 = add i32 %97, 55
  %367 = zext i32 %366 to i64
  %368 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %367
  store i32 %365, ptr addrspace(1) %368, align 4, !tbaa !21, !alias.scope !25
  %369 = extractelement <4 x i8> %80, i64 0
  %370 = zext i8 %369 to i32
  %371 = add i32 %97, 56
  %372 = zext i32 %371 to i64
  %373 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %372
  store i32 %370, ptr addrspace(1) %373, align 4, !tbaa !21, !alias.scope !25
  %374 = extractelement <4 x i8> %80, i64 1
  %375 = zext i8 %374 to i32
  %376 = add i32 %97, 57
  %377 = zext i32 %376 to i64
  %378 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %377
  store i32 %375, ptr addrspace(1) %378, align 4, !tbaa !21, !alias.scope !25
  %379 = extractelement <4 x i8> %80, i64 2
  %380 = zext i8 %379 to i32
  %381 = add i32 %97, 58
  %382 = zext i32 %381 to i64
  %383 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %382
  store i32 %380, ptr addrspace(1) %383, align 4, !tbaa !21, !alias.scope !25
  %384 = extractelement <4 x i8> %80, i64 3
  %385 = zext i8 %384 to i32
  %386 = add i32 %97, 59
  %387 = zext i32 %386 to i64
  %388 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %387
  store i32 %385, ptr addrspace(1) %388, align 4, !tbaa !21, !alias.scope !25
  %389 = extractelement <4 x i32> %82, i64 0
  %390 = add i32 %97, 60
  %391 = zext i32 %390 to i64
  %392 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %391
  store i32 %389, ptr addrspace(1) %392, align 4, !tbaa !21, !alias.scope !25
  %393 = extractelement <4 x i32> %82, i64 1
  %394 = add i32 %97, 61
  %395 = zext i32 %394 to i64
  %396 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %395
  store i32 %393, ptr addrspace(1) %396, align 4, !tbaa !21, !alias.scope !25
  %397 = extractelement <4 x i32> %82, i64 2
  %398 = add i32 %97, 62
  %399 = zext i32 %398 to i64
  %400 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %399
  store i32 %397, ptr addrspace(1) %400, align 4, !tbaa !21, !alias.scope !25
  %401 = extractelement <4 x i32> %82, i64 3
  %402 = add i32 %97, 63
  %403 = zext i32 %402 to i64
  %404 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %403
  store i32 %401, ptr addrspace(1) %404, align 4, !tbaa !21, !alias.scope !25
  %405 = extractelement <4 x i32> %83, i64 0
  %406 = add i32 %97, 64
  %407 = zext i32 %406 to i64
  %408 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %407
  store i32 %405, ptr addrspace(1) %408, align 4, !tbaa !21, !alias.scope !25
  %409 = extractelement <4 x i32> %83, i64 1
  %410 = add i32 %97, 65
  %411 = zext i32 %410 to i64
  %412 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %411
  store i32 %409, ptr addrspace(1) %412, align 4, !tbaa !21, !alias.scope !25
  %413 = extractelement <4 x i32> %83, i64 2
  %414 = add i32 %97, 66
  %415 = zext i32 %414 to i64
  %416 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %415
  store i32 %413, ptr addrspace(1) %416, align 4, !tbaa !21, !alias.scope !25
  %417 = extractelement <4 x i32> %83, i64 3
  %418 = add i32 %97, 67
  %419 = zext i32 %418 to i64
  %420 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %419
  store i32 %417, ptr addrspace(1) %420, align 4, !tbaa !21, !alias.scope !25
  %421 = extractelement <4 x i32> %84, i64 0
  %422 = add i32 %97, 68
  %423 = zext i32 %422 to i64
  %424 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %423
  store i32 %421, ptr addrspace(1) %424, align 4, !tbaa !21, !alias.scope !25
  %425 = extractelement <4 x i32> %84, i64 1
  %426 = add i32 %97, 69
  %427 = zext i32 %426 to i64
  %428 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %427
  store i32 %425, ptr addrspace(1) %428, align 4, !tbaa !21, !alias.scope !25
  %429 = extractelement <4 x i32> %84, i64 2
  %430 = add i32 %97, 70
  %431 = zext i32 %430 to i64
  %432 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %431
  store i32 %429, ptr addrspace(1) %432, align 4, !tbaa !21, !alias.scope !25
  %433 = extractelement <4 x i32> %84, i64 3
  %434 = add i32 %97, 71
  %435 = zext i32 %434 to i64
  %436 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %435
  store i32 %433, ptr addrspace(1) %436, align 4, !tbaa !21, !alias.scope !25
  %437 = extractelement <4 x i32> %85, i64 0
  %438 = add i32 %97, 72
  %439 = zext i32 %438 to i64
  %440 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %439
  store i32 %437, ptr addrspace(1) %440, align 4, !tbaa !21, !alias.scope !25
  %441 = extractelement <4 x i32> %85, i64 1
  %442 = add i32 %97, 73
  %443 = zext i32 %442 to i64
  %444 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %443
  store i32 %441, ptr addrspace(1) %444, align 4, !tbaa !21, !alias.scope !25
  %445 = extractelement <4 x i32> %85, i64 2
  %446 = add i32 %97, 74
  %447 = zext i32 %446 to i64
  %448 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %447
  store i32 %445, ptr addrspace(1) %448, align 4, !tbaa !21, !alias.scope !25
  %449 = extractelement <4 x i32> %85, i64 3
  %450 = add i32 %97, 75
  %451 = zext i32 %450 to i64
  %452 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %451
  store i32 %449, ptr addrspace(1) %452, align 4, !tbaa !21, !alias.scope !25
  %453 = extractelement <4 x i32> %86, i64 0
  %454 = add i32 %97, 76
  %455 = zext i32 %454 to i64
  %456 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %455
  store i32 %453, ptr addrspace(1) %456, align 4, !tbaa !21, !alias.scope !25
  %457 = extractelement <4 x i32> %86, i64 1
  %458 = add i32 %97, 77
  %459 = zext i32 %458 to i64
  %460 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %459
  store i32 %457, ptr addrspace(1) %460, align 4, !tbaa !21, !alias.scope !25
  %461 = extractelement <4 x i32> %86, i64 2
  %462 = add i32 %97, 78
  %463 = zext i32 %462 to i64
  %464 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %463
  store i32 %461, ptr addrspace(1) %464, align 4, !tbaa !21, !alias.scope !25
  %465 = extractelement <4 x i32> %86, i64 3
  %466 = add i32 %97, 79
  %467 = zext i32 %466 to i64
  %468 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %467
  store i32 %465, ptr addrspace(1) %468, align 4, !tbaa !21, !alias.scope !25
  %469 = extractelement <4 x i16> %87, i64 0
  %470 = zext i16 %469 to i32
  %471 = add i32 %97, 80
  %472 = zext i32 %471 to i64
  %473 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %472
  store i32 %470, ptr addrspace(1) %473, align 4, !tbaa !21, !alias.scope !25
  %474 = extractelement <4 x i16> %87, i64 1
  %475 = zext i16 %474 to i32
  %476 = add i32 %97, 81
  %477 = zext i32 %476 to i64
  %478 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %477
  store i32 %475, ptr addrspace(1) %478, align 4, !tbaa !21, !alias.scope !25
  %479 = extractelement <4 x i16> %87, i64 2
  %480 = zext i16 %479 to i32
  %481 = add i32 %97, 82
  %482 = zext i32 %481 to i64
  %483 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %482
  store i32 %480, ptr addrspace(1) %483, align 4, !tbaa !21, !alias.scope !25
  %484 = extractelement <4 x i16> %87, i64 3
  %485 = zext i16 %484 to i32
  %486 = add i32 %97, 83
  %487 = zext i32 %486 to i64
  %488 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %487
  store i32 %485, ptr addrspace(1) %488, align 4, !tbaa !21, !alias.scope !25
  %489 = extractelement <2 x i16> %88, i64 0
  %490 = zext i16 %489 to i32
  %491 = add i32 %97, 84
  %492 = zext i32 %491 to i64
  %493 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %492
  store i32 %490, ptr addrspace(1) %493, align 4, !tbaa !21, !alias.scope !25
  %494 = extractelement <2 x i16> %88, i64 1
  %495 = zext i16 %494 to i32
  %496 = add i32 %97, 85
  %497 = zext i32 %496 to i64
  %498 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %497
  store i32 %495, ptr addrspace(1) %498, align 4, !tbaa !21, !alias.scope !25
  %499 = bitcast <4 x half> %96 to <4 x i16>
  %500 = extractelement <4 x i16> %499, i64 0
  %501 = zext i16 %500 to i32
  %502 = add i32 %97, 86
  %503 = zext i32 %502 to i64
  %504 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %503
  store i32 %501, ptr addrspace(1) %504, align 4, !tbaa !21, !alias.scope !25
  %505 = extractelement <4 x i16> %499, i64 1
  %506 = zext i16 %505 to i32
  %507 = add i32 %97, 87
  %508 = zext i32 %507 to i64
  %509 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %508
  store i32 %506, ptr addrspace(1) %509, align 4, !tbaa !21, !alias.scope !25
  %510 = extractelement <4 x i16> %499, i64 2
  %511 = zext i16 %510 to i32
  %512 = add i32 %97, 88
  %513 = zext i32 %512 to i64
  %514 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %513
  store i32 %511, ptr addrspace(1) %514, align 4, !tbaa !21, !alias.scope !25
  %515 = extractelement <4 x i16> %499, i64 3
  %516 = zext i16 %515 to i32
  %517 = add i32 %97, 89
  %518 = zext i32 %517 to i64
  %519 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %518
  store i32 %516, ptr addrspace(1) %519, align 4, !tbaa !21, !alias.scope !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.s.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i64> @air.convert.s.v4i64.s.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i64> @air.convert.u.v4i64.u.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i16> @air.convert.u.v4i16.u.v4i1(<4 x i1>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i16> @air.convert.u.v4i16.f.v4f16(<4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.convert.u.v4i32.u.v4i16(<4 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i16> @air.convert.u.v3i16.u.v3i8(<3 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i32> @air.convert.s.v2i32.s.v2i8(<2 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i16> @air.convert.s.v2i16.s.v2i32(<2 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i16> @air.convert.u.v4i16.u.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.convert.f.v4f16.u.v4i16(<4 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i8> @air.convert.u.v4i8.s.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i64> @air.convert.u.v4i64.s.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i8> @air.convert.u.v3i8.f.v3f16(<3 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i32> @air.convert.u.v3i32.u.v3i8(<3 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i16> @air.convert.u.v3i16.f.v3f16(<3 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i64> @air.convert.u.v2i64.u.v2i32(<2 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i8> @air.convert.s.v4i8.u.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i8> @air.convert.s.v4i8.s.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i8> @air.convert.u.v4i8.u.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.convert.s.v4i32.u.v4i8(<4 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.convert.s.v4i32.u.v4i64(<4 x i64>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.convert.s.v4i32.s.v4i8(<4 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.convert.s.v4i32.s.v4i64(<4 x i64>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.convert.s.v4i32.s.v4i16(<4 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i16> @air.convert.s.v4i16.s.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i16> @air.convert.s.v2i16.u.v2i1(<2 x i1>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.convert.f.v4f16.u.v4i32(<4 x i32>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_integer_width_conversions, !10, !11}
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
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_integer_width_conversions.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_integer_width_conversions)"}
