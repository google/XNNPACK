// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_2x32__asm_amd64_avx512f_broadcast

      .intel_syntax noprefix

      # Free up GP registers.
      push rbx
      push rbp
      push r15
      push r14
      push r13
      push r12

      # load params to free up a GP registers
      mov r13, [rsp + 80] # params
      vbroadcastss zmm0, DWORD PTR [r13]
      vbroadcastss zmm1, DWORD PTR [r13 + 4]

      # Load c pointer.
      mov r10, [rsp + 56]
      # Load cm_stride.
      mov r11, [rsp + 64]

      # Allocate some space on the stack.
      sub rsp, 64

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with the biases.
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm9, [r9 + 64]
      vmovaps zmm8, zmm7
      vmovaps zmm14, zmm9
      add r9, 128

inner_loop:
      vmovaps  zmm10, [r9 + 0]
      vmovaps  zmm11, [r9 + 64]
      add r9, 128
      vbroadcastss zmm2, DWORD PTR [rcx + r11]
      vbroadcastss zmm3, DWORD PTR [rax + r11]
      vfmadd231ps  zmm7, zmm2, zmm10
      vfmadd231ps  zmm9, zmm2, zmm11
      vfmadd231ps  zmm8, zmm3, zmm10
      vfmadd231ps  zmm14, zmm3, zmm11

      add r11, 4
      cmp rdx, r11
      jne inner_loop
inner_loop_end:
      # Min/max clamping.
      vminps  zmm7, zmm1, zmm7
      vminps  zmm8, zmm1, zmm8
      vminps  zmm9, zmm1, zmm9
      vminps  zmm14, zmm1, zmm14
      vmaxps  zmm7, zmm0, zmm7
      vmaxps  zmm8, zmm0, zmm8
      vmaxps  zmm9, zmm0, zmm9
      vmaxps  zmm14, zmm0, zmm14

      # Check whether full or partial store.
      cmp rsi, 32
      jl tail

      vmovups  [r10], zmm7
      vmovups  [r10 + 64], zmm9
      vmovups  [r13], zmm8
      vmovups  [r13 + 64], zmm14
      add r10, 128
      add r13, 128

      sub rsi, 32
      jne outer_loop
      jmp return

tail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      shr r11d, 16
      kmovw k2, r11d
      vmovups  ZMMWORD PTR [r10]{k1}, zmm7
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm9
      vmovups  ZMMWORD PTR [r13]{k1}, zmm8
      vmovups  ZMMWORD PTR [r13 + 64]{k2}, zmm14

return:
      add rsp, 64

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_2x32__asm_amd64_avx512f_broadcast