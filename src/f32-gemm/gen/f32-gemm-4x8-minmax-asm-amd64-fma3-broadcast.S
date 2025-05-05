// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_amd64_fma3_broadcast

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
      vbroadcastss ymm0, DWORD PTR [r13]
      vbroadcastss ymm1, DWORD PTR [r13 + 4]

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
      vmovaps ymm7, ymm6
      vmovaps ymm8, ymm6
      vmovaps ymm9, ymm6
      add r9, 32

.Linner_loop:
      vmovaps  ymm14, [r9 + 0]
      add r9, 32
      vbroadcastss ymm2, DWORD PTR [rcx + r11]
      vfmadd231ps  ymm6, ymm2, ymm14
      vbroadcastss ymm3, DWORD PTR [rax + r11]
      vfmadd231ps  ymm7, ymm3, ymm14
      vbroadcastss ymm4, DWORD PTR [r15 + r11]
      vfmadd231ps  ymm8, ymm4, ymm14
      vbroadcastss ymm5, DWORD PTR [r14 + r11]
      vfmadd231ps  ymm9, ymm5, ymm14

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:
      # Min/max clamping.
      vminps  ymm6, ymm1, ymm6
      vminps  ymm7, ymm1, ymm7
      vminps  ymm8, ymm1, ymm8
      vminps  ymm9, ymm1, ymm9
      vmaxps  ymm6, ymm0, ymm6
      vmaxps  ymm7, ymm0, ymm7
      vmaxps  ymm8, ymm0, ymm8
      vmaxps  ymm9, ymm0, ymm9

      # Check whether full or partial store.
      cmp rsi, 8
      jl .Ltail_4
      vmovups  [r10], ymm6
      vmovups  [r12], ymm7
      vmovups  [r13], ymm8
      vmovups  [rbx], ymm9
      add r10, 32
      add r12, 32
      add r13, 32
      add rbx, 32

      sub rsi, 8
      jne .Louter_loop
      jmp .Lreturn

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
      vmovlps  QWORD PTR [r10], xmm6
      vmovlps  QWORD PTR [r12], xmm7
      vmovlps  QWORD PTR [r13], xmm8
      vmovlps  QWORD PTR [rbx], xmm9
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
      vmovss  DWORD PTR [r10], xmm6
      vmovss  DWORD PTR [r12], xmm7
      vmovss  DWORD PTR [r13], xmm8
      vmovss  DWORD PTR [rbx], xmm9

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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_amd64_fma3_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_amd64_fma3_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_amd64_fma3_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__