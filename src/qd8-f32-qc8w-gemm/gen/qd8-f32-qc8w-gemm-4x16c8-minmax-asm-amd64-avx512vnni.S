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

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni

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
      sub rsp, 384

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      # Clamp a & c pointers if mr <= 2
      mov r15, rax
      add r15, r8
      mov rbx, r13
      add rbx, r11
      cmp rdi, 2
      cmovle r15, rax
      cmovle rbx, r13

      # Clamp a & c pointers if mr <= 3
      mov r14, r15
      add r14, r8
      mov rbp, rbx
      add rbp, r11
      cmp rdi, 3
      cmovle r14, r15
      cmovle rbp, rbx
      # Load quantization_params pointer from stack
      mov r11, [rsp + 392]
      mov edi, [r11 + 0]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 128], zmm6
      mov edi, [r11 + 8]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 192], zmm6
      mov edi, [r11 + 16]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 256], zmm6
      mov edi, [r11 + 24]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 320], zmm6

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 128]
      vpmulld zmm12, zmm6, ZMMWORD PTR [rsp + 192]
      vpmulld zmm14, zmm6, ZMMWORD PTR [rsp + 256]
      vpmulld zmm15, zmm6, ZMMWORD PTR [rsp + 320]
      add r9, 64
      # Interleave with zeros.
      vextracti64x4 ymm16, zmm5, 1
      vpmovzxdq zmm16, ymm16
      vpmovzxdq zmm5, ymm5
      vextracti64x4 ymm17, zmm12, 1
      vpmovzxdq zmm17, ymm17
      vpmovzxdq zmm12, ymm12
      vextracti64x4 ymm18, zmm14, 1
      vpmovzxdq zmm18, ymm18
      vpmovzxdq zmm14, ymm14
      vextracti64x4 ymm19, zmm15, 1
      vpmovzxdq zmm19, ymm19
      vpmovzxdq zmm15, ymm15

.Linner_loop:
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      add r9, 128
      vbroadcasti32x2 zmm2, QWORD PTR [rcx + r11]
      vpdpbusd  zmm5, zmm2, zmm6
      vpdpbusd  zmm16, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [rax + r11]
      vpdpbusd  zmm12, zmm2, zmm6
      vpdpbusd  zmm17, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [r15 + r11]
      vpdpbusd  zmm14, zmm2, zmm6
      vpdpbusd  zmm18, zmm2, zmm7
      vbroadcasti32x2 zmm2, QWORD PTR [r14 + r11]
      vpdpbusd  zmm15, zmm2, zmm6
      vpdpbusd  zmm19, zmm2, zmm7

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
      vmovups zmm6, zmmword ptr [rip + .PERMUTATION]
      vpermt2ps zmm5, zmm6, zmm16
      vpermt2ps zmm12, zmm6, zmm17
      vpermt2ps zmm14, zmm6, zmm18
      vpermt2ps zmm15, zmm6, zmm19

      # Convert from int32 to float.
      vcvtdq2ps zmm5, zmm5
      vcvtdq2ps zmm12, zmm12
      vcvtdq2ps zmm14, zmm14
      vcvtdq2ps zmm15, zmm15
      # Load quantization_params pointer from stack
      mov r11, [rsp + 392]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm12, zmm12, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm14, zmm14, DWORD PTR [r11 + 20]{1to16}
      vmulps zmm15, zmm15, DWORD PTR [r11 + 28]{1to16}
      vmovaps zmm10, [r9 + 0]
      add r9, 64
      vmovaps zmm6, [r9 + 0]
      add r9, 64
      vfmadd213ps zmm5, zmm10, zmm6
      vfmadd213ps zmm12, zmm10, zmm6
      vfmadd213ps zmm14, zmm10, zmm6
      vfmadd213ps zmm15, zmm10, zmm6
      # Min/max clamping.
      vminps  zmm5, zmm1, zmm5
      vminps  zmm12, zmm1, zmm12
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail

      vmovups  [r10], zmm5
      vmovups  [r13], zmm12
      vmovups  [rbx], zmm14
      vmovups  [rbp], zmm15
      add r10, 64
      add r13, 64
      add rbx, 64
      add rbp, 64

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  ZMMWORD PTR [r10]{k1}, zmm5
      vmovups  ZMMWORD PTR [r13]{k1}, zmm12
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm14
      vmovups  ZMMWORD PTR [rbp]{k1}, zmm15

.Lreturn:
      add rsp, 384
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
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__asm_amd64_avx512vnni.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__