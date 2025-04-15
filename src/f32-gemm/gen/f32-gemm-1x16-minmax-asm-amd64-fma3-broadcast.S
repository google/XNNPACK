// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_1x16__asm_amd64_fma3_broadcast

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

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with the biases.
      vmovaps  ymm6, [r9 + 0]
      vmovaps  ymm7, [r9 + 32]
      add r9, 64

.Linner_loop:
      vmovaps  ymm14, [r9 + 0]
      vmovaps  ymm15, [r9 + 32]
      add r9, 64
      vbroadcastss ymm2, DWORD PTR [rcx + r11]
      vfmadd231ps  ymm6, ymm2, ymm14
      vfmadd231ps  ymm7, ymm2, ymm15

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:
      # Min/max clamping.
      vminps  ymm6, ymm1, ymm6
      vminps  ymm7, ymm1, ymm7
      vmaxps  ymm6, ymm0, ymm6
      vmaxps  ymm7, ymm0, ymm7

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail_8
      vmovups  [r10], ymm6
      vmovups  [r10 + 32], ymm7
      add r10, 64

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn
.Ltail_8:
      test sil, 8
      jz .Ltail_4
      vmovups  [r10], ymm6
      vmovaps  ymm6, ymm7
      add r10, 32


.Ltail_4:
      test sil, 4
      jz .Ltail_2
      vmovups  [r10], xmm6
      add  r10, 16
      vextractf128 xmm6, ymm6, 1


.Ltail_2:
      test sil, 2
      jz .Ltail_1
      vmovlps  QWORD PTR [r10], xmm6
      add r10, 8
      vmovhlps xmm6, xmm6, xmm6


.Ltail_1:
      test sil, 1
      jz .Lreturn
      vmovss  DWORD PTR [r10], xmm6

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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_1x16__asm_amd64_fma3_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_1x16__asm_amd64_fma3_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_1x16__asm_amd64_fma3_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__