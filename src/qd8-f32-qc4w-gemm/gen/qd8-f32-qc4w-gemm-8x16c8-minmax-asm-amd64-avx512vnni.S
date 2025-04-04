// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

.PERMUTATION:
      .long   0
      .long   2
      .long   4
      .long   6
      .long   8
      .long   10
      .long   12
      .long   14
      .long   16
      .long   18
      .long   20
      .long   22
      .long   24
      .long   26
      .long   28
      .long   30
.MASK:
      .quad   -1085102592571150096

BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni

      .intel_syntax noprefix
      # Free up GP registers.
      # Save register arguments for tail call to msan annotation helper.
      push rdi
      push rsi
      push rbx
      push rbp
      push r15
      push r14
      push r13
      push r12

      # load params to free up a GP registers
      mov r13, [rsp + 96] # params
      vbroadcastss zmm0, DWORD PTR [r13]
      vbroadcastss zmm1, DWORD PTR [r13 + 4]

      # Load c pointer.
      mov r10, [rsp + 72]
      # Load cm_stride.
      mov r11, [rsp + 80]

      add rdx, 7
      and rdx, -8

      # Move stack parameters which have not yet been loaded
      mov r12, [rsp + 104]

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13
      # Push additional stack parameters to the new stack
      mov [rsp + 8], r12

      # Allocate some space on the stack.
      sub rsp, 704
      # Write rsi (a pointer) to the stack as we need the register.
      mov [rsp + 16], rcx
      # Write r10 (c pointer) to the stack as we need the register.
      mov [rsp + 24], r10

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 32], rax
      mov [rsp + 40], r13

      # Clamp a & c pointers if mr <= 2
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 2
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 48], rcx
      mov [rsp + 56], r10

      # Clamp a & c pointers if mr <= 3
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 3
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 64], rax
      mov [rsp + 72], r13

      # Clamp a & c pointers if mr <= 4
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 4
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 80], rcx
      mov [rsp + 88], r10

      # Clamp a & c pointers if mr <= 5
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 5
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 96], rax
      mov [rsp + 104], r13

      # Clamp a & c pointers if mr <= 6
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 6
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 112], rcx
      mov [rsp + 120], r10

      # Clamp a & c pointers if mr <= 7
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 7
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 128], rax
      mov [rsp + 136], r13

      # Load quantization_params pointer from stack
      mov r11, [rsp + 712]
      mov edi, [r11 + 0]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 192], zmm6
      mov edi, [r11 + 8]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 256], zmm6
      mov edi, [r11 + 16]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 320], zmm6
      mov edi, [r11 + 24]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 384], zmm6
      mov edi, [r11 + 32]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 448], zmm6
      mov edi, [r11 + 40]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 512], zmm6
      mov edi, [r11 + 48]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 576], zmm6
      mov edi, [r11 + 56]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 640], zmm6

      mov r11, [rsp + 88]
      # Load 0xF0 for masking the weights
      vbroadcastsd  zmm13, qword ptr [rip + .MASK]


.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Read a pointers from stack into GP registers.
      mov rcx, [rsp + 16]
      mov rax, [rsp + 32]
      mov r15, [rsp + 48]
      mov r14, [rsp + 64]
      mov r12, [rsp + 80]
      mov r10, [rsp + 96]
      mov r13, [rsp + 112]
      mov rbx, [rsp + 128]

      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 192]
      vpmulld zmm12, zmm6, ZMMWORD PTR [rsp + 256]
      vpmulld zmm14, zmm6, ZMMWORD PTR [rsp + 320]
      vpmulld zmm15, zmm6, ZMMWORD PTR [rsp + 384]
      vpmulld zmm16, zmm6, ZMMWORD PTR [rsp + 448]
      vpmulld zmm17, zmm6, ZMMWORD PTR [rsp + 512]
      vpmulld zmm18, zmm6, ZMMWORD PTR [rsp + 576]
      vpmulld zmm19, zmm6, ZMMWORD PTR [rsp + 640]
      add r9, 64
      # Interleave with zeros.
      vextracti64x4 ymm20, zmm5, 1
      vpmovzxdq zmm20, ymm20
      vpmovzxdq zmm5, ymm5
      vextracti64x4 ymm21, zmm12, 1
      vpmovzxdq zmm21, ymm21
      vpmovzxdq zmm12, ymm12
      vextracti64x4 ymm22, zmm14, 1
      vpmovzxdq zmm22, ymm22
      vpmovzxdq zmm14, ymm14
      vextracti64x4 ymm23, zmm15, 1
      vpmovzxdq zmm23, ymm23
      vpmovzxdq zmm15, ymm15
      vextracti64x4 ymm24, zmm16, 1
      vpmovzxdq zmm24, ymm24
      vpmovzxdq zmm16, ymm16
      vextracti64x4 ymm25, zmm17, 1
      vpmovzxdq zmm25, ymm25
      vpmovzxdq zmm17, ymm17
      vextracti64x4 ymm26, zmm18, 1
      vpmovzxdq zmm26, ymm26
      vpmovzxdq zmm18, ymm18
      vextracti64x4 ymm27, zmm19, 1
      vpmovzxdq zmm27, ymm27
      vpmovzxdq zmm19, ymm19

.Linner_loop:
      vmovaps zmm7, [r9 + 0]
      vpslld zmm6, zmm7, 4
      vpandd zmm6, zmm6, zmm13
      vpandd zmm7, zmm7, zmm13
      add r9, 64
      vbroadcasti32x2 zmm2, QWORD PTR [rcx + r11]
      vpdpbusd  zmm5, zmm2, zmm6
      vpdpbusd  zmm20, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [rax + r11]
      vpdpbusd  zmm12, zmm2, zmm6
      vpdpbusd  zmm21, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [r15 + r11]
      vpdpbusd  zmm14, zmm2, zmm6
      vpdpbusd  zmm22, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [r14 + r11]
      vpdpbusd  zmm15, zmm2, zmm6
      vpdpbusd  zmm23, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [r12 + r11]
      vpdpbusd  zmm16, zmm2, zmm6
      vpdpbusd  zmm24, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [r10 + r11]
      vpdpbusd  zmm17, zmm2, zmm6
      vpdpbusd  zmm25, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [r13 + r11]
      vpdpbusd  zmm18, zmm2, zmm6
      vpdpbusd  zmm26, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [rbx + r11]
      vpdpbusd  zmm19, zmm2, zmm6
      vpdpbusd  zmm27, zmm2, zmm7

      add r11, 8
      cmp rdx, r11
      jne .Linner_loop
.Linner_loop_end:
      vpsrlq zmm6, zmm5, 32
      vpaddd zmm5, zmm5, zmm6
      vpsrlq zmm6, zmm12, 32
      vpaddd zmm12, zmm12, zmm6
      vpsrlq zmm6, zmm14, 32
      vpaddd zmm14, zmm14, zmm6
      vpsrlq zmm6, zmm15, 32
      vpaddd zmm15, zmm15, zmm6
      vpsrlq zmm6, zmm16, 32
      vpaddd zmm16, zmm16, zmm6
      vpsrlq zmm6, zmm17, 32
      vpaddd zmm17, zmm17, zmm6
      vpsrlq zmm6, zmm18, 32
      vpaddd zmm18, zmm18, zmm6
      vpsrlq zmm6, zmm19, 32
      vpaddd zmm19, zmm19, zmm6
      vpsrlq zmm6, zmm20, 32
      vpaddd zmm20, zmm20, zmm6
      vpsrlq zmm6, zmm21, 32
      vpaddd zmm21, zmm21, zmm6
      vpsrlq zmm6, zmm22, 32
      vpaddd zmm22, zmm22, zmm6
      vpsrlq zmm6, zmm23, 32
      vpaddd zmm23, zmm23, zmm6
      vpsrlq zmm6, zmm24, 32
      vpaddd zmm24, zmm24, zmm6
      vpsrlq zmm6, zmm25, 32
      vpaddd zmm25, zmm25, zmm6
      vpsrlq zmm6, zmm26, 32
      vpaddd zmm26, zmm26, zmm6
      vpsrlq zmm6, zmm27, 32
      vpaddd zmm27, zmm27, zmm6
      vmovups zmm6, zmmword ptr [rip + .PERMUTATION]
      vpermt2ps zmm5, zmm6, zmm20
      vpermt2ps zmm12, zmm6, zmm21
      vpermt2ps zmm14, zmm6, zmm22
      vpermt2ps zmm15, zmm6, zmm23
      vpermt2ps zmm16, zmm6, zmm24
      vpermt2ps zmm17, zmm6, zmm25
      vpermt2ps zmm18, zmm6, zmm26
      vpermt2ps zmm19, zmm6, zmm27

      # Convert from int32 to float.
      vpsrad zmm5, zmm5, 4
      vcvtdq2ps zmm5, zmm5
      vpsrad zmm12, zmm12, 4
      vcvtdq2ps zmm12, zmm12
      vpsrad zmm14, zmm14, 4
      vcvtdq2ps zmm14, zmm14
      vpsrad zmm15, zmm15, 4
      vcvtdq2ps zmm15, zmm15
      vpsrad zmm16, zmm16, 4
      vcvtdq2ps zmm16, zmm16
      vpsrad zmm17, zmm17, 4
      vcvtdq2ps zmm17, zmm17
      vpsrad zmm18, zmm18, 4
      vcvtdq2ps zmm18, zmm18
      vpsrad zmm19, zmm19, 4
      vcvtdq2ps zmm19, zmm19
      # Load quantization_params pointer from stack
      mov r11, [rsp + 712]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm12, zmm12, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm14, zmm14, DWORD PTR [r11 + 20]{1to16}
      vmulps zmm15, zmm15, DWORD PTR [r11 + 28]{1to16}
      vmulps zmm16, zmm16, DWORD PTR [r11 + 36]{1to16}
      vmulps zmm17, zmm17, DWORD PTR [r11 + 44]{1to16}
      vmulps zmm18, zmm18, DWORD PTR [r11 + 52]{1to16}
      vmulps zmm19, zmm19, DWORD PTR [r11 + 60]{1to16}
      vmovaps zmm10, [r9 + 0]
      add r9, 64
      vmovaps zmm6, [r9 + 0]
      add r9, 64
      vfmadd213ps zmm5, zmm10, zmm6
      vfmadd213ps zmm12, zmm10, zmm6
      vfmadd213ps zmm14, zmm10, zmm6
      vfmadd213ps zmm15, zmm10, zmm6
      vfmadd213ps zmm16, zmm10, zmm6
      vfmadd213ps zmm17, zmm10, zmm6
      vfmadd213ps zmm18, zmm10, zmm6
      vfmadd213ps zmm19, zmm10, zmm6
      # Min/max clamping.
      vminps  zmm5, zmm1, zmm5
      vminps  zmm12, zmm1, zmm12
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vminps  zmm16, zmm1, zmm16
      vminps  zmm17, zmm1, zmm17
      vminps  zmm18, zmm1, zmm18
      vminps  zmm19, zmm1, zmm19
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm19, zmm0, zmm19

      # Pop output pointers from the stack.
      mov rcx, [rsp + 24]
      mov rax, [rsp + 40]
      mov r15, [rsp + 56]
      mov r14, [rsp + 72]
      mov r12, [rsp + 88]
      mov r10, [rsp + 104]
      mov r13, [rsp + 120]
      mov rbx, [rsp + 136]

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail

      vmovups  [rcx], zmm5
      vmovups  [rax], zmm12
      vmovups  [r15], zmm14
      vmovups  [r14], zmm15
      vmovups  [r12], zmm16
      vmovups  [r10], zmm17
      vmovups  [r13], zmm18
      vmovups  [rbx], zmm19
      add rcx, 64
      add rax, 64
      add r15, 64
      add r14, 64
      add r12, 64
      add r10, 64
      add r13, 64
      add rbx, 64

      # Write output pointers to the stack.
      mov [rsp + 24], rcx
      mov [rsp + 40], rax
      mov [rsp + 56], r15
      mov [rsp + 72], r14
      mov [rsp + 88], r12
      mov [rsp + 104], r10
      mov [rsp + 120], r13
      mov [rsp + 136], rbx

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  ZMMWORD PTR [rcx]{k1}, zmm5
      vmovups  ZMMWORD PTR [rax]{k1}, zmm12
      vmovups  ZMMWORD PTR [r15]{k1}, zmm14
      vmovups  ZMMWORD PTR [r14]{k1}, zmm15
      vmovups  ZMMWORD PTR [r12]{k1}, zmm16
      vmovups  ZMMWORD PTR [r10]{k1}, zmm17
      vmovups  ZMMWORD PTR [r13]{k1}, zmm18
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm19

.Lreturn:
      add rsp, 704
      mov r13, [rsp]
      mov rsp, r13
      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      pop rsi
      pop rdi
      #if XNN_HAS_FEATURE(memory_sanitizer)
      jmp xnn_gemm_ukernel_msan_sizeof_c_4
      #else
      ret
      #endif
END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__asm_amd64_avx512vnni.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__