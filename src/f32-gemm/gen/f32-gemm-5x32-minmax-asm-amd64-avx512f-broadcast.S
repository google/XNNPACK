// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_5x32__asm_amd64_avx512f_broadcast

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

      # Clamp a & c pointers if mr <= 3
      mov r14, r15
      add r14, r8
      mov rbp, rbx
      add rbp, r11
      cmp rdi, 3
      cmovle r14, r15
      cmovle rbp, rbx

      # Clamp a & c pointers if mr <= 4
      mov r12, r14
      add r12, r8
      mov r8, rbp
      add r8, r11
      cmp rdi, 4
      cmovle r12, r14
      cmovle r8, rbp

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with the biases.
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm16, [r9 + 64]
      vmovaps zmm8, zmm7
      vmovaps zmm9, zmm7
      vmovaps zmm14, zmm7
      vmovaps zmm15, zmm7
      vmovaps zmm17, zmm16
      vmovaps zmm18, zmm16
      vmovaps zmm19, zmm16
      vmovaps zmm20, zmm16
      add r9, 128

inner_loop:
      vmovaps  zmm10, [r9 + 0]
      vmovaps  zmm11, [r9 + 64]
      add r9, 128
      vbroadcastss zmm2, DWORD PTR [rsi + r11]
      vfmadd231ps  zmm7, zmm2, zmm10
      vfmadd231ps  zmm16, zmm2, zmm11
      vbroadcastss zmm3, DWORD PTR [rax + r11]
      vfmadd231ps  zmm8, zmm3, zmm10
      vfmadd231ps  zmm17, zmm3, zmm11
      vbroadcastss zmm4, DWORD PTR [r15 + r11]
      vfmadd231ps  zmm9, zmm4, zmm10
      vfmadd231ps  zmm18, zmm4, zmm11
      vbroadcastss zmm5, DWORD PTR [r14 + r11]
      vfmadd231ps  zmm14, zmm5, zmm10
      vfmadd231ps  zmm19, zmm5, zmm11
      vbroadcastss zmm6, DWORD PTR [r12 + r11]
      vfmadd231ps  zmm15, zmm6, zmm10
      vfmadd231ps  zmm20, zmm6, zmm11

      add r11, 4
      cmp rdx, r11
      jne inner_loop
inner_loop_end:
      # Min/max clamping.
      vminps  zmm7, zmm1, zmm7
      vminps  zmm8, zmm1, zmm8
      vminps  zmm9, zmm1, zmm9
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vminps  zmm16, zmm1, zmm16
      vminps  zmm17, zmm1, zmm17
      vminps  zmm18, zmm1, zmm18
      vminps  zmm19, zmm1, zmm19
      vminps  zmm20, zmm1, zmm20
      vmaxps  zmm7, zmm0, zmm7
      vmaxps  zmm8, zmm0, zmm8
      vmaxps  zmm9, zmm0, zmm9
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm19, zmm0, zmm19
      vmaxps  zmm20, zmm0, zmm20

      # Check whether full or partial store.
      cmp rcx, 32
      jl tail

      vmovups  [r10], zmm7
      vmovups  [r10 + 64], zmm16
      vmovups  [r13], zmm8
      vmovups  [r13 + 64], zmm17
      vmovups  [rbx], zmm9
      vmovups  [rbx + 64], zmm18
      vmovups  [rbp], zmm14
      vmovups  [rbp + 64], zmm19
      vmovups  [r8], zmm15
      vmovups  [r8 + 64], zmm20
      add r10, 128
      add r13, 128
      add rbx, 128
      add rbp, 128
      add r8, 128

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
      vmovups  ZMMWORD PTR [r10]{k1}, zmm7
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm16
      vmovups  ZMMWORD PTR [r13]{k1}, zmm8
      vmovups  ZMMWORD PTR [r13 + 64]{k2}, zmm17
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm9
      vmovups  ZMMWORD PTR [rbx + 64]{k2}, zmm18
      vmovups  ZMMWORD PTR [rbp]{k1}, zmm14
      vmovups  ZMMWORD PTR [rbp + 64]{k2}, zmm19
      vmovups  ZMMWORD PTR [r8]{k1}, zmm15
      vmovups  ZMMWORD PTR [r8 + 64]{k2}, zmm20

return:

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_5x32__asm_amd64_avx512f_broadcast