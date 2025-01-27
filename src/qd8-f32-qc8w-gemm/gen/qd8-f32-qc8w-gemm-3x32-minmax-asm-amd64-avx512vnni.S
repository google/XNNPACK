// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x32c4__asm_amd64_avx512vnni

      .intel_syntax noprefix

      # Free up GP registers.
      push rbx
      push rbp
      push r15
      push r14
      push r13
      push r12

      # Swap rsi & rcx because sal can only use cl.
      mov r15, rsi
      mov rsi, rcx
      mov rcx, r15

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
      sub rsp, 656

      # Clamp a & c pointers if mr <= 1
      mov rax, rsi
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rsi
      cmovle r13, r10

      # Clamp a & c pointers if mr <= 2
      mov r15, rax
      add r15, r8
      mov rbx, r13
      add rbx, r11
      cmp rdi, 2
      cmovle r15, rax
      cmovle rbx, r13

      # Load quantization params pointer from stack
      mov r11, [rsp + 744]
      mov edi, [r11 + 0]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 464], zmm6
      mov edi, [r11 + 8]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 528], zmm6
      mov edi, [r11 + 16]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 592], zmm6

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 464]
      vpmulld zmm12, zmm6, ZMMWORD PTR [rsp + 528]
      vpmulld zmm14, zmm6, ZMMWORD PTR [rsp + 592]
      vpmulld zmm15, zmm7, ZMMWORD PTR [rsp + 464]
      vpmulld zmm16, zmm7, ZMMWORD PTR [rsp + 528]
      vpmulld zmm17, zmm7, ZMMWORD PTR [rsp + 592]
      add r9, 128

inner_loop:
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      add r9, 128
      vpbroadcastd zmm2, [rsi + r11]
      vpdpbusd  zmm5, zmm2, zmm6
      vpdpbusd  zmm15, zmm2, zmm7
      vpbroadcastd zmm2, [rax + r11]
      vpdpbusd  zmm12, zmm2, zmm6
      vpdpbusd  zmm16, zmm2, zmm7
      vpbroadcastd zmm2, [r15 + r11]
      vpdpbusd  zmm14, zmm2, zmm6
      vpdpbusd  zmm17, zmm2, zmm7

      add r11, 4
      cmp rdx, r11
      jne inner_loop
inner_loop_end:

      # Convert from int32 to float.
      vcvtdq2ps zmm5, zmm5
      vcvtdq2ps zmm12, zmm12
      vcvtdq2ps zmm14, zmm14
      vcvtdq2ps zmm15, zmm15
      vcvtdq2ps zmm16, zmm16
      vcvtdq2ps zmm17, zmm17
      # Load quantization_params pointer from stack
      mov r11, [rsp + 744]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm12, zmm12, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm14, zmm14, DWORD PTR [r11 + 20]{1to16}
      vmulps zmm15, zmm15, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm16, zmm16, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm17, zmm17, DWORD PTR [r11 + 20]{1to16}
      vmovaps zmm10, [r9 + 0]
      vmovaps zmm11, [r9 + 64]
      add r9, 128
      vmovaps zmm6, [r9 + 0]
      vmovaps zmm7, [r9 + 64]
      add r9, 128
      vfmadd213ps zmm5, zmm10, zmm6
      vfmadd213ps zmm12, zmm10, zmm6
      vfmadd213ps zmm14, zmm10, zmm6
      vfmadd213ps zmm15, zmm11, zmm7
      vfmadd213ps zmm16, zmm11, zmm7
      vfmadd213ps zmm17, zmm11, zmm7
      # Min/max clamping.
      vminps  zmm5, zmm1, zmm5
      vminps  zmm12, zmm1, zmm12
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vminps  zmm16, zmm1, zmm16
      vminps  zmm17, zmm1, zmm17
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm17, zmm0, zmm17

      # Check whether full or partial store.
      cmp rcx, 32
      jl tail

      vmovups  [r10], zmm5
      vmovups  [r10 + 64], zmm15
      vmovups  [r13], zmm12
      vmovups  [r13 + 64], zmm16
      vmovups  [rbx], zmm14
      vmovups  [rbx + 64], zmm17
      add r10, 128
      add r13, 128
      add rbx, 128

      sub rcx, 32
      jne outer_loop
      jmp return

tail:
      mov r11d, -1
      sal r11d, cl
      not r11d
      kmovw k1, r11d
      shr r11d, 16
      kmovw k2, r11d
      vmovups  ZMMWORD PTR [r10]{k1}, zmm5
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm15
      vmovups  ZMMWORD PTR [r13]{k1}, zmm12
      vmovups  ZMMWORD PTR [r13 + 64]{k2}, zmm16
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm14
      vmovups  ZMMWORD PTR [rbx + 64]{k2}, zmm17

return:
      add rsp, 656

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x32c4__asm_amd64_avx512vnni