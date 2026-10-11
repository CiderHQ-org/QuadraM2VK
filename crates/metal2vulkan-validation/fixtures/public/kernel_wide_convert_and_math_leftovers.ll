; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. Sixteen wide-vector
; conversions and small elementwise `air.*` symbols that the corpus reaches a handful of times each
; and that no authored case covered. `fmin`/`fmax` are spelled `precise::` because the default
; compile emits `air.fast_fmin`/`air.fast_fmax`, which are already covered.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_wide_convert_and_math_leftovers.bc'
source_filename = "validation/fixtures/public/kernel_wide_convert_and_math_leftovers.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_wide_convert_and_math_leftovers(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %4, ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %5) local_unnamed_addr #0 {
  %7 = load half, ptr addrspace(1) %0, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %8 = insertelement <16 x half> undef, half %7, i64 0
  %9 = getelementptr inbounds half, ptr addrspace(1) %0, i64 1
  %10 = load half, ptr addrspace(1) %9, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %11 = insertelement <16 x half> %8, half %10, i64 1
  %12 = getelementptr inbounds half, ptr addrspace(1) %0, i64 2
  %13 = load half, ptr addrspace(1) %12, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %14 = insertelement <16 x half> %11, half %13, i64 2
  %15 = getelementptr inbounds half, ptr addrspace(1) %0, i64 3
  %16 = load half, ptr addrspace(1) %15, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %17 = insertelement <16 x half> %14, half %16, i64 3
  %18 = getelementptr inbounds half, ptr addrspace(1) %0, i64 4
  %19 = load half, ptr addrspace(1) %18, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %20 = insertelement <16 x half> %17, half %19, i64 4
  %21 = getelementptr inbounds half, ptr addrspace(1) %0, i64 5
  %22 = load half, ptr addrspace(1) %21, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %23 = insertelement <16 x half> %20, half %22, i64 5
  %24 = getelementptr inbounds half, ptr addrspace(1) %0, i64 6
  %25 = load half, ptr addrspace(1) %24, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %26 = insertelement <16 x half> %23, half %25, i64 6
  %27 = getelementptr inbounds half, ptr addrspace(1) %0, i64 7
  %28 = load half, ptr addrspace(1) %27, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %29 = insertelement <16 x half> %26, half %28, i64 7
  %30 = getelementptr inbounds half, ptr addrspace(1) %0, i64 8
  %31 = load half, ptr addrspace(1) %30, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %32 = insertelement <16 x half> %29, half %31, i64 8
  %33 = getelementptr inbounds half, ptr addrspace(1) %0, i64 9
  %34 = load half, ptr addrspace(1) %33, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %35 = insertelement <16 x half> %32, half %34, i64 9
  %36 = getelementptr inbounds half, ptr addrspace(1) %0, i64 10
  %37 = load half, ptr addrspace(1) %36, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %38 = insertelement <16 x half> %35, half %37, i64 10
  %39 = getelementptr inbounds half, ptr addrspace(1) %0, i64 11
  %40 = load half, ptr addrspace(1) %39, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %41 = insertelement <16 x half> %38, half %40, i64 11
  %42 = getelementptr inbounds half, ptr addrspace(1) %0, i64 12
  %43 = load half, ptr addrspace(1) %42, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %44 = insertelement <16 x half> %41, half %43, i64 12
  %45 = getelementptr inbounds half, ptr addrspace(1) %0, i64 13
  %46 = load half, ptr addrspace(1) %45, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %47 = insertelement <16 x half> %44, half %46, i64 13
  %48 = getelementptr inbounds half, ptr addrspace(1) %0, i64 14
  %49 = load half, ptr addrspace(1) %48, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %50 = insertelement <16 x half> %47, half %49, i64 14
  %51 = getelementptr inbounds half, ptr addrspace(1) %0, i64 15
  %52 = load half, ptr addrspace(1) %51, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %53 = insertelement <16 x half> %50, half %52, i64 15
  %54 = getelementptr inbounds half, ptr addrspace(1) %0, i64 16
  %55 = load half, ptr addrspace(1) %54, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %56 = insertelement <8 x half> undef, half %55, i64 0
  %57 = getelementptr inbounds half, ptr addrspace(1) %0, i64 17
  %58 = load half, ptr addrspace(1) %57, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %59 = insertelement <8 x half> %56, half %58, i64 1
  %60 = getelementptr inbounds half, ptr addrspace(1) %0, i64 18
  %61 = load half, ptr addrspace(1) %60, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %62 = insertelement <8 x half> %59, half %61, i64 2
  %63 = getelementptr inbounds half, ptr addrspace(1) %0, i64 19
  %64 = load half, ptr addrspace(1) %63, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %65 = insertelement <8 x half> %62, half %64, i64 3
  %66 = getelementptr inbounds half, ptr addrspace(1) %0, i64 20
  %67 = load half, ptr addrspace(1) %66, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %68 = insertelement <8 x half> %65, half %67, i64 4
  %69 = getelementptr inbounds half, ptr addrspace(1) %0, i64 21
  %70 = load half, ptr addrspace(1) %69, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %71 = insertelement <8 x half> %68, half %70, i64 5
  %72 = getelementptr inbounds half, ptr addrspace(1) %0, i64 22
  %73 = load half, ptr addrspace(1) %72, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %74 = insertelement <8 x half> %71, half %73, i64 6
  %75 = getelementptr inbounds half, ptr addrspace(1) %0, i64 23
  %76 = load half, ptr addrspace(1) %75, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %77 = insertelement <8 x half> %74, half %76, i64 7
  %78 = load bfloat, ptr addrspace(1) %1, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %79 = insertelement <8 x bfloat> undef, bfloat %78, i64 0
  %80 = getelementptr inbounds bfloat, ptr addrspace(1) %1, i64 1
  %81 = load bfloat, ptr addrspace(1) %80, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %82 = insertelement <8 x bfloat> %79, bfloat %81, i64 1
  %83 = getelementptr inbounds bfloat, ptr addrspace(1) %1, i64 2
  %84 = load bfloat, ptr addrspace(1) %83, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %85 = insertelement <8 x bfloat> %82, bfloat %84, i64 2
  %86 = getelementptr inbounds bfloat, ptr addrspace(1) %1, i64 3
  %87 = load bfloat, ptr addrspace(1) %86, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %88 = insertelement <8 x bfloat> %85, bfloat %87, i64 3
  %89 = getelementptr inbounds bfloat, ptr addrspace(1) %1, i64 4
  %90 = load bfloat, ptr addrspace(1) %89, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %91 = insertelement <8 x bfloat> %88, bfloat %90, i64 4
  %92 = getelementptr inbounds bfloat, ptr addrspace(1) %1, i64 5
  %93 = load bfloat, ptr addrspace(1) %92, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %94 = insertelement <8 x bfloat> %91, bfloat %93, i64 5
  %95 = getelementptr inbounds bfloat, ptr addrspace(1) %1, i64 6
  %96 = load bfloat, ptr addrspace(1) %95, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %97 = insertelement <8 x bfloat> %94, bfloat %96, i64 6
  %98 = getelementptr inbounds bfloat, ptr addrspace(1) %1, i64 7
  %99 = load bfloat, ptr addrspace(1) %98, align 2, !tbaa !38, !alias.scope !40, !noalias !41
  %100 = insertelement <8 x bfloat> %97, bfloat %99, i64 7
  %101 = getelementptr inbounds half, ptr addrspace(1) %0, i64 24
  %102 = load half, ptr addrspace(1) %101, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %103 = insertelement <4 x half> undef, half %102, i64 0
  %104 = getelementptr inbounds half, ptr addrspace(1) %0, i64 25
  %105 = load half, ptr addrspace(1) %104, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %106 = insertelement <4 x half> %103, half %105, i64 1
  %107 = getelementptr inbounds half, ptr addrspace(1) %0, i64 26
  %108 = load half, ptr addrspace(1) %107, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %109 = insertelement <4 x half> %106, half %108, i64 2
  %110 = getelementptr inbounds half, ptr addrspace(1) %0, i64 27
  %111 = load half, ptr addrspace(1) %110, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %112 = insertelement <4 x half> %109, half %111, i64 3
  %113 = tail call fast <16 x float> @air.convert.f.v16f32.f.v16f16(<16 x half> %53) #2
  %114 = bitcast <16 x float> %113 to <16 x i32>
  %115 = extractelement <16 x i32> %114, i64 0
  store i32 %115, ptr addrspace(1) %5, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %116 = extractelement <16 x i32> %114, i64 1
  %117 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 1
  store i32 %116, ptr addrspace(1) %117, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %118 = extractelement <16 x i32> %114, i64 2
  %119 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 2
  store i32 %118, ptr addrspace(1) %119, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %120 = extractelement <16 x i32> %114, i64 3
  %121 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 3
  store i32 %120, ptr addrspace(1) %121, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %122 = extractelement <16 x i32> %114, i64 4
  %123 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 4
  store i32 %122, ptr addrspace(1) %123, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %124 = extractelement <16 x i32> %114, i64 5
  %125 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 5
  store i32 %124, ptr addrspace(1) %125, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %126 = extractelement <16 x i32> %114, i64 6
  %127 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 6
  store i32 %126, ptr addrspace(1) %127, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %128 = extractelement <16 x i32> %114, i64 7
  %129 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 7
  store i32 %128, ptr addrspace(1) %129, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %130 = extractelement <16 x i32> %114, i64 8
  %131 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 8
  store i32 %130, ptr addrspace(1) %131, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %132 = extractelement <16 x i32> %114, i64 9
  %133 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 9
  store i32 %132, ptr addrspace(1) %133, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %134 = extractelement <16 x i32> %114, i64 10
  %135 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 10
  store i32 %134, ptr addrspace(1) %135, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %136 = extractelement <16 x i32> %114, i64 11
  %137 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 11
  store i32 %136, ptr addrspace(1) %137, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %138 = extractelement <16 x i32> %114, i64 12
  %139 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 12
  store i32 %138, ptr addrspace(1) %139, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %140 = extractelement <16 x i32> %114, i64 13
  %141 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 13
  store i32 %140, ptr addrspace(1) %141, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %142 = extractelement <16 x i32> %114, i64 14
  %143 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 14
  store i32 %142, ptr addrspace(1) %143, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %144 = extractelement <16 x i32> %114, i64 15
  %145 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 15
  store i32 %144, ptr addrspace(1) %145, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %146 = tail call fast <16 x bfloat> @air.convert.f.v16bf16.f.v16f16(<16 x half> %53) #2
  %147 = bitcast <16 x bfloat> %146 to <16 x i16>
  %148 = extractelement <16 x i16> %147, i64 0
  %149 = zext i16 %148 to i32
  %150 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 16
  store i32 %149, ptr addrspace(1) %150, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %151 = extractelement <16 x i16> %147, i64 1
  %152 = zext i16 %151 to i32
  %153 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 17
  store i32 %152, ptr addrspace(1) %153, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %154 = extractelement <16 x i16> %147, i64 2
  %155 = zext i16 %154 to i32
  %156 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 18
  store i32 %155, ptr addrspace(1) %156, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %157 = extractelement <16 x i16> %147, i64 3
  %158 = zext i16 %157 to i32
  %159 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 19
  store i32 %158, ptr addrspace(1) %159, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %160 = extractelement <16 x i16> %147, i64 4
  %161 = zext i16 %160 to i32
  %162 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 20
  store i32 %161, ptr addrspace(1) %162, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %163 = extractelement <16 x i16> %147, i64 5
  %164 = zext i16 %163 to i32
  %165 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 21
  store i32 %164, ptr addrspace(1) %165, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %166 = extractelement <16 x i16> %147, i64 6
  %167 = zext i16 %166 to i32
  %168 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 22
  store i32 %167, ptr addrspace(1) %168, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %169 = extractelement <16 x i16> %147, i64 7
  %170 = zext i16 %169 to i32
  %171 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 23
  store i32 %170, ptr addrspace(1) %171, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %172 = extractelement <16 x i16> %147, i64 8
  %173 = zext i16 %172 to i32
  %174 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 24
  store i32 %173, ptr addrspace(1) %174, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %175 = extractelement <16 x i16> %147, i64 9
  %176 = zext i16 %175 to i32
  %177 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 25
  store i32 %176, ptr addrspace(1) %177, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %178 = extractelement <16 x i16> %147, i64 10
  %179 = zext i16 %178 to i32
  %180 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 26
  store i32 %179, ptr addrspace(1) %180, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %181 = extractelement <16 x i16> %147, i64 11
  %182 = zext i16 %181 to i32
  %183 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 27
  store i32 %182, ptr addrspace(1) %183, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %184 = extractelement <16 x i16> %147, i64 12
  %185 = zext i16 %184 to i32
  %186 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 28
  store i32 %185, ptr addrspace(1) %186, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %187 = extractelement <16 x i16> %147, i64 13
  %188 = zext i16 %187 to i32
  %189 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 29
  store i32 %188, ptr addrspace(1) %189, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %190 = extractelement <16 x i16> %147, i64 14
  %191 = zext i16 %190 to i32
  %192 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 30
  store i32 %191, ptr addrspace(1) %192, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %193 = extractelement <16 x i16> %147, i64 15
  %194 = zext i16 %193 to i32
  %195 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 31
  store i32 %194, ptr addrspace(1) %195, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %196 = tail call fast <8 x bfloat> @air.convert.f.v8bf16.f.v8f16(<8 x half> %77) #2
  %197 = bitcast <8 x bfloat> %196 to <8 x i16>
  %198 = extractelement <8 x i16> %197, i64 0
  %199 = zext i16 %198 to i32
  %200 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 32
  store i32 %199, ptr addrspace(1) %200, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %201 = extractelement <8 x i16> %197, i64 1
  %202 = zext i16 %201 to i32
  %203 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 33
  store i32 %202, ptr addrspace(1) %203, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %204 = extractelement <8 x i16> %197, i64 2
  %205 = zext i16 %204 to i32
  %206 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 34
  store i32 %205, ptr addrspace(1) %206, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %207 = extractelement <8 x i16> %197, i64 3
  %208 = zext i16 %207 to i32
  %209 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 35
  store i32 %208, ptr addrspace(1) %209, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %210 = extractelement <8 x i16> %197, i64 4
  %211 = zext i16 %210 to i32
  %212 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 36
  store i32 %211, ptr addrspace(1) %212, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %213 = extractelement <8 x i16> %197, i64 5
  %214 = zext i16 %213 to i32
  %215 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 37
  store i32 %214, ptr addrspace(1) %215, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %216 = extractelement <8 x i16> %197, i64 6
  %217 = zext i16 %216 to i32
  %218 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 38
  store i32 %217, ptr addrspace(1) %218, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %219 = extractelement <8 x i16> %197, i64 7
  %220 = zext i16 %219 to i32
  %221 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 39
  store i32 %220, ptr addrspace(1) %221, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %222 = tail call fast <8 x half> @air.convert.f.v8f16.f.v8bf16(<8 x bfloat> %100) #2
  %223 = bitcast <8 x half> %222 to <8 x i16>
  %224 = extractelement <8 x i16> %223, i64 0
  %225 = zext i16 %224 to i32
  %226 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 40
  store i32 %225, ptr addrspace(1) %226, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %227 = extractelement <8 x i16> %223, i64 1
  %228 = zext i16 %227 to i32
  %229 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 41
  store i32 %228, ptr addrspace(1) %229, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %230 = extractelement <8 x i16> %223, i64 2
  %231 = zext i16 %230 to i32
  %232 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 42
  store i32 %231, ptr addrspace(1) %232, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %233 = extractelement <8 x i16> %223, i64 3
  %234 = zext i16 %233 to i32
  %235 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 43
  store i32 %234, ptr addrspace(1) %235, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %236 = extractelement <8 x i16> %223, i64 4
  %237 = zext i16 %236 to i32
  %238 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 44
  store i32 %237, ptr addrspace(1) %238, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %239 = extractelement <8 x i16> %223, i64 5
  %240 = zext i16 %239 to i32
  %241 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 45
  store i32 %240, ptr addrspace(1) %241, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %242 = extractelement <8 x i16> %223, i64 6
  %243 = zext i16 %242 to i32
  %244 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 46
  store i32 %243, ptr addrspace(1) %244, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %245 = extractelement <8 x i16> %223, i64 7
  %246 = zext i16 %245 to i32
  %247 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 47
  store i32 %246, ptr addrspace(1) %247, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %248 = tail call fast <4 x bfloat> @air.convert.f.v4bf16.f.v4f16(<4 x half> %112) #2
  %249 = bitcast <4 x bfloat> %248 to <4 x i16>
  %250 = extractelement <4 x i16> %249, i64 0
  %251 = zext i16 %250 to i32
  %252 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 48
  store i32 %251, ptr addrspace(1) %252, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %253 = extractelement <4 x i16> %249, i64 1
  %254 = zext i16 %253 to i32
  %255 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 49
  store i32 %254, ptr addrspace(1) %255, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %256 = extractelement <4 x i16> %249, i64 2
  %257 = zext i16 %256 to i32
  %258 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 50
  store i32 %257, ptr addrspace(1) %258, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %259 = extractelement <4 x i16> %249, i64 3
  %260 = zext i16 %259 to i32
  %261 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 51
  store i32 %260, ptr addrspace(1) %261, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %262 = load i32, ptr addrspace(1) %2, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %263 = insertelement <4 x i32> undef, i32 %262, i64 0
  %264 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1
  %265 = load i32, ptr addrspace(1) %264, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %266 = insertelement <4 x i32> %263, i32 %265, i64 1
  %267 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 2
  %268 = load i32, ptr addrspace(1) %267, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %269 = insertelement <4 x i32> %266, i32 %268, i64 2
  %270 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 3
  %271 = load i32, ptr addrspace(1) %270, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %272 = insertelement <4 x i32> %269, i32 %271, i64 3
  %273 = tail call fast <4 x half> @air.convert.f.v4f16.s.v4i32(<4 x i32> %272) #2
  %274 = bitcast <4 x half> %273 to <4 x i16>
  %275 = extractelement <4 x i16> %274, i64 0
  %276 = zext i16 %275 to i32
  %277 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 52
  store i32 %276, ptr addrspace(1) %277, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %278 = extractelement <4 x i16> %274, i64 1
  %279 = zext i16 %278 to i32
  %280 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 53
  store i32 %279, ptr addrspace(1) %280, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %281 = extractelement <4 x i16> %274, i64 2
  %282 = zext i16 %281 to i32
  %283 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 54
  store i32 %282, ptr addrspace(1) %283, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %284 = extractelement <4 x i16> %274, i64 3
  %285 = zext i16 %284 to i32
  %286 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 55
  store i32 %285, ptr addrspace(1) %286, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %287 = getelementptr inbounds half, ptr addrspace(1) %0, i64 28
  %288 = load half, ptr addrspace(1) %287, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %289 = insertelement <4 x half> undef, half %288, i64 0
  %290 = getelementptr inbounds half, ptr addrspace(1) %0, i64 29
  %291 = load half, ptr addrspace(1) %290, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %292 = insertelement <4 x half> %289, half %291, i64 1
  %293 = getelementptr inbounds half, ptr addrspace(1) %0, i64 30
  %294 = load half, ptr addrspace(1) %293, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %295 = insertelement <4 x half> %292, half %294, i64 2
  %296 = getelementptr inbounds half, ptr addrspace(1) %0, i64 31
  %297 = load half, ptr addrspace(1) %296, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %298 = insertelement <4 x half> %295, half %297, i64 3
  %299 = getelementptr inbounds half, ptr addrspace(1) %0, i64 32
  %300 = load half, ptr addrspace(1) %299, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %301 = insertelement <4 x half> undef, half %300, i64 0
  %302 = getelementptr inbounds half, ptr addrspace(1) %0, i64 33
  %303 = load half, ptr addrspace(1) %302, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %304 = insertelement <4 x half> %301, half %303, i64 1
  %305 = getelementptr inbounds half, ptr addrspace(1) %0, i64 34
  %306 = load half, ptr addrspace(1) %305, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %307 = insertelement <4 x half> %304, half %306, i64 2
  %308 = getelementptr inbounds half, ptr addrspace(1) %0, i64 35
  %309 = load half, ptr addrspace(1) %308, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %310 = insertelement <4 x half> %307, half %309, i64 3
  %311 = tail call fast <4 x half> @air.powr.v4f16(<4 x half> %298, <4 x half> %310) #2
  %312 = bitcast <4 x half> %311 to <4 x i16>
  %313 = extractelement <4 x i16> %312, i64 0
  %314 = zext i16 %313 to i32
  %315 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 56
  store i32 %314, ptr addrspace(1) %315, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %316 = extractelement <4 x i16> %312, i64 1
  %317 = zext i16 %316 to i32
  %318 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 57
  store i32 %317, ptr addrspace(1) %318, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %319 = extractelement <4 x i16> %312, i64 2
  %320 = zext i16 %319 to i32
  %321 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 58
  store i32 %320, ptr addrspace(1) %321, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %322 = extractelement <4 x i16> %312, i64 3
  %323 = zext i16 %322 to i32
  %324 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 59
  store i32 %323, ptr addrspace(1) %324, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %325 = load i32, ptr addrspace(1) %3, align 4, !tbaa !42, !alias.scope !48, !noalias !49
  %326 = insertelement <3 x i32> undef, i32 %325, i64 0
  %327 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 1
  %328 = load i32, ptr addrspace(1) %327, align 4, !tbaa !42, !alias.scope !48, !noalias !49
  %329 = insertelement <3 x i32> %326, i32 %328, i64 1
  %330 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 2
  %331 = load i32, ptr addrspace(1) %330, align 4, !tbaa !42, !alias.scope !48, !noalias !49
  %332 = insertelement <3 x i32> %329, i32 %331, i64 2
  %333 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 3
  %334 = load i32, ptr addrspace(1) %333, align 4, !tbaa !42, !alias.scope !48, !noalias !49
  %335 = insertelement <3 x i32> undef, i32 %334, i64 0
  %336 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 4
  %337 = load i32, ptr addrspace(1) %336, align 4, !tbaa !42, !alias.scope !48, !noalias !49
  %338 = insertelement <3 x i32> %335, i32 %337, i64 1
  %339 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 5
  %340 = load i32, ptr addrspace(1) %339, align 4, !tbaa !42, !alias.scope !48, !noalias !49
  %341 = insertelement <3 x i32> %338, i32 %340, i64 2
  %342 = tail call <3 x i32> @air.max.u.v3i32(<3 x i32> %332, <3 x i32> %341) #2
  %343 = extractelement <3 x i32> %342, i64 0
  %344 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 60
  store i32 %343, ptr addrspace(1) %344, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %345 = extractelement <3 x i32> %342, i64 1
  %346 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 61
  store i32 %345, ptr addrspace(1) %346, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %347 = extractelement <3 x i32> %342, i64 2
  %348 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 62
  store i32 %347, ptr addrspace(1) %348, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %349 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 4
  %350 = load i32, ptr addrspace(1) %349, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %351 = insertelement <3 x i32> undef, i32 %350, i64 0
  %352 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 5
  %353 = load i32, ptr addrspace(1) %352, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %354 = insertelement <3 x i32> %351, i32 %353, i64 1
  %355 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 6
  %356 = load i32, ptr addrspace(1) %355, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %357 = insertelement <3 x i32> %354, i32 %356, i64 2
  %358 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 7
  %359 = load i32, ptr addrspace(1) %358, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %360 = insertelement <3 x i32> undef, i32 %359, i64 0
  %361 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 8
  %362 = load i32, ptr addrspace(1) %361, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %363 = insertelement <3 x i32> %360, i32 %362, i64 1
  %364 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 9
  %365 = load i32, ptr addrspace(1) %364, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %366 = insertelement <3 x i32> %363, i32 %365, i64 2
  %367 = tail call <3 x i32> @air.max.s.v3i32(<3 x i32> %357, <3 x i32> %366) #2
  %368 = extractelement <3 x i32> %367, i64 0
  %369 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 63
  store i32 %368, ptr addrspace(1) %369, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %370 = extractelement <3 x i32> %367, i64 1
  %371 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 64
  store i32 %370, ptr addrspace(1) %371, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %372 = extractelement <3 x i32> %367, i64 2
  %373 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 65
  store i32 %372, ptr addrspace(1) %373, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %374 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 10
  %375 = load i32, ptr addrspace(1) %374, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %376 = insertelement <3 x i32> undef, i32 %375, i64 0
  %377 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 11
  %378 = load i32, ptr addrspace(1) %377, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %379 = insertelement <3 x i32> %376, i32 %378, i64 1
  %380 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 12
  %381 = load i32, ptr addrspace(1) %380, align 4, !tbaa !42, !alias.scope !46, !noalias !47
  %382 = insertelement <3 x i32> %379, i32 %381, i64 2
  %383 = tail call <3 x i32> @air.abs.s.v3i32(<3 x i32> %382) #2
  %384 = extractelement <3 x i32> %383, i64 0
  %385 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 66
  store i32 %384, ptr addrspace(1) %385, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %386 = extractelement <3 x i32> %383, i64 1
  %387 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 67
  store i32 %386, ptr addrspace(1) %387, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %388 = extractelement <3 x i32> %383, i64 2
  %389 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 68
  store i32 %388, ptr addrspace(1) %389, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %390 = getelementptr inbounds half, ptr addrspace(1) %0, i64 36
  %391 = load half, ptr addrspace(1) %390, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %392 = insertelement <2 x half> undef, half %391, i64 0
  %393 = getelementptr inbounds half, ptr addrspace(1) %0, i64 37
  %394 = load half, ptr addrspace(1) %393, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %395 = insertelement <2 x half> %392, half %394, i64 1
  %396 = tail call fast <2 x half> @air.fract.v2f16(<2 x half> %395) #2
  %397 = bitcast <2 x half> %396 to <2 x i16>
  %398 = extractelement <2 x i16> %397, i64 0
  %399 = zext i16 %398 to i32
  %400 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 69
  store i32 %399, ptr addrspace(1) %400, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %401 = extractelement <2 x i16> %397, i64 1
  %402 = zext i16 %401 to i32
  %403 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 70
  store i32 %402, ptr addrspace(1) %403, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %404 = load float, ptr addrspace(1) %4, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %405 = insertelement <3 x float> undef, float %404, i64 0
  %406 = getelementptr inbounds float, ptr addrspace(1) %4, i64 1
  %407 = load float, ptr addrspace(1) %406, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %408 = insertelement <3 x float> %405, float %407, i64 1
  %409 = getelementptr inbounds float, ptr addrspace(1) %4, i64 2
  %410 = load float, ptr addrspace(1) %409, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %411 = insertelement <3 x float> %408, float %410, i64 2
  %412 = getelementptr inbounds float, ptr addrspace(1) %4, i64 3
  %413 = load float, ptr addrspace(1) %412, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %414 = insertelement <3 x float> undef, float %413, i64 0
  %415 = getelementptr inbounds float, ptr addrspace(1) %4, i64 4
  %416 = load float, ptr addrspace(1) %415, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %417 = insertelement <3 x float> %414, float %416, i64 1
  %418 = getelementptr inbounds float, ptr addrspace(1) %4, i64 5
  %419 = load float, ptr addrspace(1) %418, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %420 = insertelement <3 x float> %417, float %419, i64 2
  %421 = tail call fast <3 x float> @air.fmin.v3f32(<3 x float> %411, <3 x float> %420) #2
  %422 = bitcast <3 x float> %421 to <3 x i32>
  %423 = extractelement <3 x i32> %422, i64 0
  %424 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 71
  store i32 %423, ptr addrspace(1) %424, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %425 = extractelement <3 x i32> %422, i64 1
  %426 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 72
  store i32 %425, ptr addrspace(1) %426, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %427 = extractelement <3 x i32> %422, i64 2
  %428 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 73
  store i32 %427, ptr addrspace(1) %428, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %429 = tail call fast <3 x float> @air.fmax.v3f32(<3 x float> %411, <3 x float> %420) #2
  %430 = bitcast <3 x float> %429 to <3 x i32>
  %431 = extractelement <3 x i32> %430, i64 0
  %432 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 74
  store i32 %431, ptr addrspace(1) %432, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %433 = extractelement <3 x i32> %430, i64 1
  %434 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 75
  store i32 %433, ptr addrspace(1) %434, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %435 = extractelement <3 x i32> %430, i64 2
  %436 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 76
  store i32 %435, ptr addrspace(1) %436, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %437 = getelementptr inbounds half, ptr addrspace(1) %0, i64 38
  %438 = load half, ptr addrspace(1) %437, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %439 = insertelement <3 x half> undef, half %438, i64 0
  %440 = getelementptr inbounds half, ptr addrspace(1) %0, i64 39
  %441 = load half, ptr addrspace(1) %440, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %442 = insertelement <3 x half> %439, half %441, i64 1
  %443 = getelementptr inbounds half, ptr addrspace(1) %0, i64 40
  %444 = load half, ptr addrspace(1) %443, align 2, !tbaa !25, !alias.scope !29, !noalias !32
  %445 = insertelement <3 x half> %442, half %444, i64 2
  %446 = tail call fast <3 x half> @air.floor.v3f16(<3 x half> %445) #2
  %447 = bitcast <3 x half> %446 to <3 x i16>
  %448 = extractelement <3 x i16> %447, i64 0
  %449 = zext i16 %448 to i32
  %450 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 77
  store i32 %449, ptr addrspace(1) %450, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %451 = extractelement <3 x i16> %447, i64 1
  %452 = zext i16 %451 to i32
  %453 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 78
  store i32 %452, ptr addrspace(1) %453, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %454 = extractelement <3 x i16> %447, i64 2
  %455 = zext i16 %454 to i32
  %456 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 79
  store i32 %455, ptr addrspace(1) %456, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %457 = getelementptr inbounds float, ptr addrspace(1) %4, i64 6
  %458 = load float, ptr addrspace(1) %457, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %459 = getelementptr inbounds float, ptr addrspace(1) %4, i64 7
  %460 = load float, ptr addrspace(1) %459, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %461 = getelementptr inbounds float, ptr addrspace(1) %4, i64 8
  %462 = load float, ptr addrspace(1) %461, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %463 = tail call fast float @air.fast_fmedian3.f32(float %458, float %460, float %462) #2
  %464 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 80
  %465 = bitcast ptr addrspace(1) %464 to ptr addrspace(1)
  store float %463, ptr addrspace(1) %465, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %466 = getelementptr inbounds float, ptr addrspace(1) %4, i64 9
  %467 = load float, ptr addrspace(1) %466, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %468 = getelementptr inbounds float, ptr addrspace(1) %4, i64 10
  %469 = load float, ptr addrspace(1) %468, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %470 = getelementptr inbounds float, ptr addrspace(1) %4, i64 11
  %471 = load float, ptr addrspace(1) %470, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %472 = tail call fast float @air.fast_fmedian3.f32(float %467, float %469, float %471) #2
  %473 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 81
  %474 = bitcast ptr addrspace(1) %473 to ptr addrspace(1)
  store float %472, ptr addrspace(1) %474, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %475 = getelementptr inbounds float, ptr addrspace(1) %4, i64 12
  %476 = load float, ptr addrspace(1) %475, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %477 = tail call fast float @air.fast_cospi.f32(float %476) #2
  %478 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 82
  %479 = bitcast ptr addrspace(1) %478 to ptr addrspace(1)
  store float %477, ptr addrspace(1) %479, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %480 = getelementptr inbounds float, ptr addrspace(1) %4, i64 13
  %481 = load float, ptr addrspace(1) %480, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %482 = tail call fast float @air.fast_cospi.f32(float %481) #2
  %483 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 83
  %484 = bitcast ptr addrspace(1) %483 to ptr addrspace(1)
  store float %482, ptr addrspace(1) %484, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %485 = getelementptr inbounds float, ptr addrspace(1) %4, i64 14
  %486 = load float, ptr addrspace(1) %485, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %487 = tail call fast float @air.fast_cospi.f32(float %486) #2
  %488 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 84
  %489 = bitcast ptr addrspace(1) %488 to ptr addrspace(1)
  store float %487, ptr addrspace(1) %489, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  %490 = getelementptr inbounds float, ptr addrspace(1) %4, i64 15
  %491 = load float, ptr addrspace(1) %490, align 4, !tbaa !50, !alias.scope !52, !noalias !53
  %492 = tail call fast float @air.fast_cospi.f32(float %491) #2
  %493 = getelementptr inbounds i32, ptr addrspace(1) %5, i64 85
  %494 = bitcast ptr addrspace(1) %493 to ptr addrspace(1)
  store float %492, ptr addrspace(1) %494, align 4, !tbaa !42, !alias.scope !44, !noalias !45
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <16 x float> @air.convert.f.v16f32.f.v16f16(<16 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <16 x bfloat> @air.convert.f.v16bf16.f.v16f16(<16 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <8 x bfloat> @air.convert.f.v8bf16.f.v8f16(<8 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <8 x half> @air.convert.f.v8f16.f.v8bf16(<8 x bfloat>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x bfloat> @air.convert.f.v4bf16.f.v4f16(<4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.convert.f.v4f16.s.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.powr.v4f16(<4 x half>, <4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i32> @air.max.u.v3i32(<3 x i32>, <3 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i32> @air.max.s.v3i32(<3 x i32>, <3 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i32> @air.abs.s.v3i32(<3 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x half> @air.fract.v2f16(<2 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fmin.v3f32(<3 x float>, <3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fmax.v3f32(<3 x float>, <3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x half> @air.floor.v3f16(<3 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_fmedian3.f32(float, float, float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_cospi.f32(float) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_wide_convert_and_math_leftovers, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hs"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"bs"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"is"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"us"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fs"}
!17 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
!24 = !{!"/private/tmp/mathv/k.metal"}
!25 = !{!26, !26, i64 0}
!26 = !{!"half", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C++ TBAA"}
!29 = !{!30}
!30 = distinct !{!30, !31, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !"air-alias-scopes(kernel_wide_convert_and_math_leftovers)"}
!32 = !{!33, !34, !35, !36, !37}
!33 = distinct !{!33, !31, !"air-alias-scope-arg(1)"}
!34 = distinct !{!34, !31, !"air-alias-scope-arg(2)"}
!35 = distinct !{!35, !31, !"air-alias-scope-arg(3)"}
!36 = distinct !{!36, !31, !"air-alias-scope-arg(4)"}
!37 = distinct !{!37, !31, !"air-alias-scope-arg(5)"}
!38 = !{!39, !39, i64 0}
!39 = !{!"bfloat", !27, i64 0}
!40 = !{!33}
!41 = !{!30, !34, !35, !36, !37}
!42 = !{!43, !43, i64 0}
!43 = !{!"int", !27, i64 0}
!44 = !{!37}
!45 = !{!30, !33, !34, !35, !36}
!46 = !{!34}
!47 = !{!30, !33, !35, !36, !37}
!48 = !{!35}
!49 = !{!30, !33, !34, !36, !37}
!50 = !{!51, !51, i64 0}
!51 = !{!"float", !27, i64 0}
!52 = !{!36}
!53 = !{!30, !33, !34, !35, !37}
