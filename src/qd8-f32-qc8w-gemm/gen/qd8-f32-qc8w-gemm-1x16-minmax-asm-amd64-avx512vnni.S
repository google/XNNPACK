// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__asm_amd64_avx512vnni

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

      add rdx, 3
      and rdx, -4

      # Allocate some space on the stack.
      sub rsp, 528

      # Load quantization params pointer from stack
      mov r11, [rsp + 616]
      mov edi, [r11 + 0]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 464], zmm6

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 464]
      add r9, 64

inner_loop:
      vmovaps  zmm6, [r9 + 0]
      add r9, 64
      vpbroadcastd zmm2, [rcx + r11]
      vpdpbusd  zmm5, zmm2, zmm6

      add r11, 4
      cmp rdx, r11
      jne inner_loop
inner_loop_end:

      # Convert from int32 to float.
      vcvtdq2ps zmm5, zmm5
      # Load quantization_params pointer from stack
      mov r11, [rsp + 616]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmovaps zmm10, [r9 + 0]
      add r9, 64
      vmovaps zmm6, [r9 + 0]
      add r9, 64
      vfmadd213ps zmm5, zmm10, zmm6
      # Min/max clamping.
      vminps  zmm5, zmm1, zmm5
      vmaxps  zmm5, zmm0, zmm5

      # Check whether full or partial store.
      cmp rsi, 16
      jl tail

      vmovups  [r10], zmm5
      add r10, 64

      sub rsi, 16
      jne outer_loop
      jmp return

tail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  ZMMWORD PTR [r10]{k1}, zmm5

return:
      add rsp, 528

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__asm_amd64_avx512vnni