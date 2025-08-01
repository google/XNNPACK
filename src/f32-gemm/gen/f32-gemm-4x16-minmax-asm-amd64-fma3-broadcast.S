// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_fma3_broadcast

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

      # load params to free up GP registers
      mov r13, [rsp + 96] # params
      vbroadcastss ymm0, dword ptr [r13]
      vbroadcastss ymm1, dword ptr [r13 + 4]

      # Load c pointer.
      mov r10, [rsp + 72]
      # Load cm_stride.
      mov r11, [rsp + 80]

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13

      # Allocate some space on the stack.
      sub rsp, 128

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r12, r10
      add r12, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r12, r10

      # Clamp a & c pointers if mr <= 2
      mov r15, rax
      add r15, r8
      mov r13, r12
      add r13, r11
      cmp rdi, 2
      cmovle r15, rax
      cmovle r13, r12

      # Clamp a & c pointers if mr <= 3
      mov r14, r15
      add r14, r8
      mov rbx, r13
      add rbx, r11
      cmp rdi, 3
      cmovle r14, r15
      cmovle rbx, r13

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with the biases.
      vmovaps  ymm6, [r9 + 0]
      vmovaps  ymm10, [r9 + 32]
      vmovaps ymm7, ymm6
      vmovaps ymm8, ymm6
      vmovaps ymm9, ymm6
      vmovaps ymm11, ymm10
      vmovaps ymm12, ymm10
      vmovaps ymm13, ymm10
      add r9, 64

.Linner_loop:
      vmovaps  ymm14, [r9 + 0]
      vmovaps  ymm15, [r9 + 32]
      add r9, 64
      vbroadcastss ymm2, dword ptr [rcx + r11]
      vfmadd231ps  ymm6, ymm2, ymm14
      vfmadd231ps  ymm10, ymm2, ymm15
      vbroadcastss ymm3, dword ptr [rax + r11]
      vfmadd231ps  ymm7, ymm3, ymm14
      vfmadd231ps  ymm11, ymm3, ymm15
      vbroadcastss ymm4, dword ptr [r15 + r11]
      vfmadd231ps  ymm8, ymm4, ymm14
      vfmadd231ps  ymm12, ymm4, ymm15
      vbroadcastss ymm5, dword ptr [r14 + r11]
      vfmadd231ps  ymm9, ymm5, ymm14
      vfmadd231ps  ymm13, ymm5, ymm15

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:
      # Min/max clamping.
      vminps  ymm6, ymm1, ymm6
      vminps  ymm8, ymm1, ymm8
      vminps  ymm10, ymm1, ymm10
      vminps  ymm12, ymm1, ymm12
      vminps  ymm7, ymm1, ymm7
      vminps  ymm9, ymm1, ymm9
      vminps  ymm11, ymm1, ymm11
      vminps  ymm13, ymm1, ymm13
      vmaxps  ymm6, ymm0, ymm6
      vmaxps  ymm8, ymm0, ymm8
      vmaxps  ymm10, ymm0, ymm10
      vmaxps  ymm12, ymm0, ymm12
      vmaxps  ymm7, ymm0, ymm7
      vmaxps  ymm9, ymm0, ymm9
      vmaxps  ymm11, ymm0, ymm11
      vmaxps  ymm13, ymm0, ymm13

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail_8
      vmovups  [r10], ymm6
      vmovups  [r10 + 32], ymm10
      vmovups  [r12], ymm7
      vmovups  [r12 + 32], ymm11
      vmovups  [r13], ymm8
      vmovups  [r13 + 32], ymm12
      vmovups  [rbx], ymm9
      vmovups  [rbx + 32], ymm13
      add r10, 64
      add r12, 64
      add r13, 64
      add rbx, 64

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn
.Ltail_8:
      test sil, 8
      jz .Ltail_4
      vmovups  [r10], ymm6
      vmovups  [r12], ymm7
      vmovups  [r13], ymm8
      vmovups  [rbx], ymm9
      vmovaps  ymm6, ymm10
      vmovaps  ymm7, ymm11
      vmovaps  ymm8, ymm12
      vmovaps  ymm9, ymm13
      add r10, 32
      add r12, 32
      add r13, 32
      add rbx, 32


.Ltail_4:
      test sil, 4
      jz .Ltail_2
      vmovups  [r10], xmm6
      vmovups  [r12], xmm7
      vmovups  [r13], xmm8
      vmovups  [rbx], xmm9
      add  r10, 16
      add  r12, 16
      add  r13, 16
      add  rbx, 16
      vextractf128 xmm6, ymm6, 1
      vextractf128 xmm7, ymm7, 1
      vextractf128 xmm8, ymm8, 1
      vextractf128 xmm9, ymm9, 1


.Ltail_2:
      test sil, 2
      jz .Ltail_1
      vmovlps  qword ptr [r10], xmm6
      vmovlps  qword ptr [r12], xmm7
      vmovlps  qword ptr [r13], xmm8
      vmovlps  qword ptr [rbx], xmm9
      add r10, 8
      add r12, 8
      add r13, 8
      add rbx, 8
      vmovhlps xmm6, xmm6, xmm6
      vmovhlps xmm7, xmm7, xmm7
      vmovhlps xmm8, xmm8, xmm8
      vmovhlps xmm9, xmm9, xmm9


.Ltail_1:
      test sil, 1
      jz .Lreturn
      vmovss  dword ptr [r10], xmm6
      vmovss  dword ptr [r12], xmm7
      vmovss  dword ptr [r13], xmm8
      vmovss  dword ptr [rbx], xmm9

.Lreturn:
      add rsp, 128
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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_fma3_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_fma3_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x16__asm_amd64_fma3_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__