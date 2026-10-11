; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. Every conversion is
; written out longhand rather than in a loop, so the module is loop-free and the case needs no
; execution bound.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_float_to_integer_saturates.bc'
source_filename = "validation/fixtures/public/kernel_float_to_integer_saturates.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_float_to_integer_saturates(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %2) local_unnamed_addr #0 {
  %4 = load float, ptr addrspace(1) %0, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %5 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  %6 = load float, ptr addrspace(1) %5, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %7 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  %8 = load float, ptr addrspace(1) %7, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %9 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  %10 = load float, ptr addrspace(1) %9, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %11 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  %12 = load float, ptr addrspace(1) %11, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %13 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  %14 = load float, ptr addrspace(1) %13, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %15 = load half, ptr addrspace(1) %1, align 2, !tbaa !32, !alias.scope !34, !noalias !35
  %16 = getelementptr inbounds half, ptr addrspace(1) %1, i64 1
  %17 = load half, ptr addrspace(1) %16, align 2, !tbaa !32, !alias.scope !34, !noalias !35
  %18 = getelementptr inbounds half, ptr addrspace(1) %1, i64 2
  %19 = load half, ptr addrspace(1) %18, align 2, !tbaa !32, !alias.scope !34, !noalias !35
  %20 = getelementptr inbounds half, ptr addrspace(1) %1, i64 3
  %21 = load half, ptr addrspace(1) %20, align 2, !tbaa !32, !alias.scope !34, !noalias !35
  %22 = getelementptr inbounds half, ptr addrspace(1) %1, i64 4
  %23 = load half, ptr addrspace(1) %22, align 2, !tbaa !32, !alias.scope !34, !noalias !35
  %24 = getelementptr inbounds half, ptr addrspace(1) %1, i64 5
  %25 = load half, ptr addrspace(1) %24, align 2, !tbaa !32, !alias.scope !34, !noalias !35
  %26 = tail call i32 @air.convert.s.i32.f.f32(float %4) #2
  store i32 %26, ptr addrspace(1) %2, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %27 = tail call i32 @air.convert.u.i32.f.f32(float %4) #2
  %28 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1
  store i32 %27, ptr addrspace(1) %28, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %29 = tail call i16 @air.convert.s.i16.f.f32(float %4) #2
  %30 = sext i16 %29 to i32
  %31 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 2
  store i32 %30, ptr addrspace(1) %31, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %32 = tail call i16 @air.convert.u.i16.f.f32(float %4) #2
  %33 = zext i16 %32 to i32
  %34 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 3
  store i32 %33, ptr addrspace(1) %34, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %35 = tail call i8 @air.convert.s.i8.f.f32(float %4) #2
  %36 = sext i8 %35 to i32
  %37 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 4
  store i32 %36, ptr addrspace(1) %37, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %38 = tail call i8 @air.convert.u.i8.f.f32(float %4) #2
  %39 = zext i8 %38 to i32
  %40 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 5
  store i32 %39, ptr addrspace(1) %40, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %41 = tail call i32 @air.convert.s.i32.f.f32(float %6) #2
  %42 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 6
  store i32 %41, ptr addrspace(1) %42, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %43 = tail call i32 @air.convert.u.i32.f.f32(float %6) #2
  %44 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 7
  store i32 %43, ptr addrspace(1) %44, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %45 = tail call i16 @air.convert.s.i16.f.f32(float %6) #2
  %46 = sext i16 %45 to i32
  %47 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 8
  store i32 %46, ptr addrspace(1) %47, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %48 = tail call i16 @air.convert.u.i16.f.f32(float %6) #2
  %49 = zext i16 %48 to i32
  %50 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 9
  store i32 %49, ptr addrspace(1) %50, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %51 = tail call i8 @air.convert.s.i8.f.f32(float %6) #2
  %52 = sext i8 %51 to i32
  %53 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 10
  store i32 %52, ptr addrspace(1) %53, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %54 = tail call i8 @air.convert.u.i8.f.f32(float %6) #2
  %55 = zext i8 %54 to i32
  %56 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 11
  store i32 %55, ptr addrspace(1) %56, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %57 = tail call i32 @air.convert.s.i32.f.f32(float %8) #2
  %58 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 12
  store i32 %57, ptr addrspace(1) %58, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %59 = tail call i32 @air.convert.u.i32.f.f32(float %8) #2
  %60 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 13
  store i32 %59, ptr addrspace(1) %60, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %61 = tail call i16 @air.convert.s.i16.f.f32(float %8) #2
  %62 = sext i16 %61 to i32
  %63 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 14
  store i32 %62, ptr addrspace(1) %63, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %64 = tail call i16 @air.convert.u.i16.f.f32(float %8) #2
  %65 = zext i16 %64 to i32
  %66 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 15
  store i32 %65, ptr addrspace(1) %66, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %67 = tail call i8 @air.convert.s.i8.f.f32(float %8) #2
  %68 = sext i8 %67 to i32
  %69 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 16
  store i32 %68, ptr addrspace(1) %69, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %70 = tail call i8 @air.convert.u.i8.f.f32(float %8) #2
  %71 = zext i8 %70 to i32
  %72 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 17
  store i32 %71, ptr addrspace(1) %72, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %73 = tail call i32 @air.convert.s.i32.f.f32(float %10) #2
  %74 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 18
  store i32 %73, ptr addrspace(1) %74, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %75 = tail call i32 @air.convert.u.i32.f.f32(float %10) #2
  %76 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 19
  store i32 %75, ptr addrspace(1) %76, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %77 = tail call i16 @air.convert.s.i16.f.f32(float %10) #2
  %78 = sext i16 %77 to i32
  %79 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 20
  store i32 %78, ptr addrspace(1) %79, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %80 = tail call i16 @air.convert.u.i16.f.f32(float %10) #2
  %81 = zext i16 %80 to i32
  %82 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 21
  store i32 %81, ptr addrspace(1) %82, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %83 = tail call i8 @air.convert.s.i8.f.f32(float %10) #2
  %84 = sext i8 %83 to i32
  %85 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 22
  store i32 %84, ptr addrspace(1) %85, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %86 = tail call i8 @air.convert.u.i8.f.f32(float %10) #2
  %87 = zext i8 %86 to i32
  %88 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 23
  store i32 %87, ptr addrspace(1) %88, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %89 = tail call i32 @air.convert.s.i32.f.f32(float %12) #2
  %90 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 24
  store i32 %89, ptr addrspace(1) %90, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %91 = tail call i32 @air.convert.u.i32.f.f32(float %12) #2
  %92 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 25
  store i32 %91, ptr addrspace(1) %92, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %93 = tail call i16 @air.convert.s.i16.f.f32(float %12) #2
  %94 = sext i16 %93 to i32
  %95 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 26
  store i32 %94, ptr addrspace(1) %95, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %96 = tail call i16 @air.convert.u.i16.f.f32(float %12) #2
  %97 = zext i16 %96 to i32
  %98 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 27
  store i32 %97, ptr addrspace(1) %98, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %99 = tail call i8 @air.convert.s.i8.f.f32(float %12) #2
  %100 = sext i8 %99 to i32
  %101 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 28
  store i32 %100, ptr addrspace(1) %101, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %102 = tail call i8 @air.convert.u.i8.f.f32(float %12) #2
  %103 = zext i8 %102 to i32
  %104 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 29
  store i32 %103, ptr addrspace(1) %104, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %105 = tail call i32 @air.convert.s.i32.f.f32(float %14) #2
  %106 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 30
  store i32 %105, ptr addrspace(1) %106, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %107 = tail call i32 @air.convert.u.i32.f.f32(float %14) #2
  %108 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 31
  store i32 %107, ptr addrspace(1) %108, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %109 = tail call i16 @air.convert.s.i16.f.f32(float %14) #2
  %110 = sext i16 %109 to i32
  %111 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 32
  store i32 %110, ptr addrspace(1) %111, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %112 = tail call i16 @air.convert.u.i16.f.f32(float %14) #2
  %113 = zext i16 %112 to i32
  %114 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 33
  store i32 %113, ptr addrspace(1) %114, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %115 = tail call i8 @air.convert.s.i8.f.f32(float %14) #2
  %116 = sext i8 %115 to i32
  %117 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 34
  store i32 %116, ptr addrspace(1) %117, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %118 = tail call i8 @air.convert.u.i8.f.f32(float %14) #2
  %119 = zext i8 %118 to i32
  %120 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 35
  store i32 %119, ptr addrspace(1) %120, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %121 = tail call i32 @air.convert.s.i32.f.f16(half %15) #2
  %122 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 36
  store i32 %121, ptr addrspace(1) %122, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %123 = tail call i32 @air.convert.u.i32.f.f16(half %15) #2
  %124 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 37
  store i32 %123, ptr addrspace(1) %124, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %125 = tail call i16 @air.convert.s.i16.f.f16(half %15) #2
  %126 = sext i16 %125 to i32
  %127 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 38
  store i32 %126, ptr addrspace(1) %127, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %128 = tail call i16 @air.convert.u.i16.f.f16(half %15) #2
  %129 = zext i16 %128 to i32
  %130 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 39
  store i32 %129, ptr addrspace(1) %130, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %131 = tail call i8 @air.convert.s.i8.f.f16(half %15) #2
  %132 = sext i8 %131 to i32
  %133 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 40
  store i32 %132, ptr addrspace(1) %133, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %134 = tail call i8 @air.convert.u.i8.f.f16(half %15) #2
  %135 = zext i8 %134 to i32
  %136 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 41
  store i32 %135, ptr addrspace(1) %136, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %137 = tail call i32 @air.convert.s.i32.f.f16(half %17) #2
  %138 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 42
  store i32 %137, ptr addrspace(1) %138, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %139 = tail call i32 @air.convert.u.i32.f.f16(half %17) #2
  %140 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 43
  store i32 %139, ptr addrspace(1) %140, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %141 = tail call i16 @air.convert.s.i16.f.f16(half %17) #2
  %142 = sext i16 %141 to i32
  %143 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 44
  store i32 %142, ptr addrspace(1) %143, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %144 = tail call i16 @air.convert.u.i16.f.f16(half %17) #2
  %145 = zext i16 %144 to i32
  %146 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 45
  store i32 %145, ptr addrspace(1) %146, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %147 = tail call i8 @air.convert.s.i8.f.f16(half %17) #2
  %148 = sext i8 %147 to i32
  %149 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 46
  store i32 %148, ptr addrspace(1) %149, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %150 = tail call i8 @air.convert.u.i8.f.f16(half %17) #2
  %151 = zext i8 %150 to i32
  %152 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 47
  store i32 %151, ptr addrspace(1) %152, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %153 = tail call i32 @air.convert.s.i32.f.f16(half %19) #2
  %154 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 48
  store i32 %153, ptr addrspace(1) %154, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %155 = tail call i32 @air.convert.u.i32.f.f16(half %19) #2
  %156 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 49
  store i32 %155, ptr addrspace(1) %156, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %157 = tail call i16 @air.convert.s.i16.f.f16(half %19) #2
  %158 = sext i16 %157 to i32
  %159 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 50
  store i32 %158, ptr addrspace(1) %159, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %160 = tail call i16 @air.convert.u.i16.f.f16(half %19) #2
  %161 = zext i16 %160 to i32
  %162 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 51
  store i32 %161, ptr addrspace(1) %162, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %163 = tail call i8 @air.convert.s.i8.f.f16(half %19) #2
  %164 = sext i8 %163 to i32
  %165 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 52
  store i32 %164, ptr addrspace(1) %165, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %166 = tail call i8 @air.convert.u.i8.f.f16(half %19) #2
  %167 = zext i8 %166 to i32
  %168 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 53
  store i32 %167, ptr addrspace(1) %168, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %169 = tail call i32 @air.convert.s.i32.f.f16(half %21) #2
  %170 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 54
  store i32 %169, ptr addrspace(1) %170, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %171 = tail call i32 @air.convert.u.i32.f.f16(half %21) #2
  %172 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 55
  store i32 %171, ptr addrspace(1) %172, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %173 = tail call i16 @air.convert.s.i16.f.f16(half %21) #2
  %174 = sext i16 %173 to i32
  %175 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 56
  store i32 %174, ptr addrspace(1) %175, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %176 = tail call i16 @air.convert.u.i16.f.f16(half %21) #2
  %177 = zext i16 %176 to i32
  %178 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 57
  store i32 %177, ptr addrspace(1) %178, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %179 = tail call i8 @air.convert.s.i8.f.f16(half %21) #2
  %180 = sext i8 %179 to i32
  %181 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 58
  store i32 %180, ptr addrspace(1) %181, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %182 = tail call i8 @air.convert.u.i8.f.f16(half %21) #2
  %183 = zext i8 %182 to i32
  %184 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 59
  store i32 %183, ptr addrspace(1) %184, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %185 = tail call i32 @air.convert.s.i32.f.f16(half %23) #2
  %186 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 60
  store i32 %185, ptr addrspace(1) %186, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %187 = tail call i32 @air.convert.u.i32.f.f16(half %23) #2
  %188 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 61
  store i32 %187, ptr addrspace(1) %188, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %189 = tail call i16 @air.convert.s.i16.f.f16(half %23) #2
  %190 = sext i16 %189 to i32
  %191 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 62
  store i32 %190, ptr addrspace(1) %191, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %192 = tail call i16 @air.convert.u.i16.f.f16(half %23) #2
  %193 = zext i16 %192 to i32
  %194 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 63
  store i32 %193, ptr addrspace(1) %194, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %195 = tail call i8 @air.convert.s.i8.f.f16(half %23) #2
  %196 = sext i8 %195 to i32
  %197 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 64
  store i32 %196, ptr addrspace(1) %197, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %198 = tail call i8 @air.convert.u.i8.f.f16(half %23) #2
  %199 = zext i8 %198 to i32
  %200 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 65
  store i32 %199, ptr addrspace(1) %200, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %201 = tail call i32 @air.convert.s.i32.f.f16(half %25) #2
  %202 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 66
  store i32 %201, ptr addrspace(1) %202, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %203 = tail call i32 @air.convert.u.i32.f.f16(half %25) #2
  %204 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 67
  store i32 %203, ptr addrspace(1) %204, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %205 = tail call i16 @air.convert.s.i16.f.f16(half %25) #2
  %206 = sext i16 %205 to i32
  %207 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 68
  store i32 %206, ptr addrspace(1) %207, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %208 = tail call i16 @air.convert.u.i16.f.f16(half %25) #2
  %209 = zext i16 %208 to i32
  %210 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 69
  store i32 %209, ptr addrspace(1) %210, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %211 = tail call i8 @air.convert.s.i8.f.f16(half %25) #2
  %212 = sext i8 %211 to i32
  %213 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 70
  store i32 %212, ptr addrspace(1) %213, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %214 = tail call i8 @air.convert.u.i8.f.f16(half %25) #2
  %215 = zext i8 %214 to i32
  %216 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 71
  store i32 %215, ptr addrspace(1) %216, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.convert.s.i32.f.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.convert.u.i32.f.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.convert.s.i16.f.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.convert.u.i16.f.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i8 @air.convert.s.i8.f.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i8 @air.convert.u.i8.f.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.convert.s.i32.f.f16(half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.convert.u.i32.f.f16(half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.convert.s.i16.f.f16(half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.convert.u.i16.f.f16(half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i8 @air.convert.s.i8.f.f16(half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i8 @air.convert.u.i8.f.f16(half) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_float_to_integer_saturates, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fin"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_float_to_integer_saturates.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_float_to_integer_saturates)"}
!29 = !{!30, !31}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = distinct !{!31, !28, !"air-alias-scope-arg(2)"}
!32 = !{!33, !33, i64 0}
!33 = !{!"half", !24, i64 0}
!34 = !{!30}
!35 = !{!27, !31}
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !24, i64 0}
!38 = !{!31}
!39 = !{!27, !30}
