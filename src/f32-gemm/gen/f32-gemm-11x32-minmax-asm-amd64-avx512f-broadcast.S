// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_11x32__asm_amd64_avx512f_broadcast

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
      # Write rsi (a pointer) to the stack as we need the register.
      mov [rsp - 128], rsi
      # Write r10 (c pointer) to the stack as we need the register.
      mov [rsp - 136], r10

      # Clamp a & c pointers if mr <= 1
      mov rax, rsi
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rsi
      cmovle r13, r10

      mov [rsp - 144], rax
      mov [rsp - 152], r13

      # Clamp a & c pointers if mr <= 2
      mov rsi, rax
      add rsi, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 2
      cmovle rsi, rax
      cmovle r10, r13

      mov [rsp - 160], rsi
      mov [rsp - 168], r10

      # Clamp a & c pointers if mr <= 3
      mov rax, rsi
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 3
      cmovle rax, rsi
      cmovle r13, r10

      mov [rsp - 176], rax
      mov [rsp - 184], r13

      # Clamp a & c pointers if mr <= 4
      mov rsi, rax
      add rsi, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 4
      cmovle rsi, rax
      cmovle r10, r13

      mov [rsp - 192], rsi
      mov [rsp - 200], r10

      # Clamp a & c pointers if mr <= 5
      mov rax, rsi
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 5
      cmovle rax, rsi
      cmovle r13, r10

      mov [rsp - 208], rax
      mov [rsp - 216], r13

      # Clamp a & c pointers if mr <= 6
      mov rsi, rax
      add rsi, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 6
      cmovle rsi, rax
      cmovle r10, r13

      mov [rsp - 224], rsi
      mov [rsp - 232], r10

      # Clamp a & c pointers if mr <= 7
      mov rax, rsi
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 7
      cmovle rax, rsi
      cmovle r13, r10

      mov [rsp - 240], rax
      mov [rsp - 248], r13

      # Clamp a & c pointers if mr <= 8
      mov rsi, rax
      add rsi, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 8
      cmovle rsi, rax
      cmovle r10, r13

      mov [rsp - 256], rsi
      mov [rsp - 264], r10

      # Clamp a & c pointers if mr <= 9
      mov rax, rsi
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 9
      cmovle rax, rsi
      cmovle r13, r10

      mov [rsp - 272], rax
      mov [rsp - 280], r13

      # Clamp a & c pointers if mr <= 10
      mov rsi, rax
      add rsi, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 10
      cmovle rsi, rax
      cmovle r10, r13

      mov [rsp - 288], rsi
      mov [rsp - 296], r10

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Read a pointers from stack into GP registers.
      mov rsi, [rsp - 128]
      mov rax, [rsp - 144]
      mov r15, [rsp - 160]
      mov r14, [rsp - 176]
      mov r12, [rsp - 192]
      mov r10, [rsp - 208]
      mov r13, [rsp - 224]
      mov rbx, [rsp - 240]
      mov rbp, [rsp - 256]
      mov r8, [rsp - 272]
      mov rdi, [rsp - 288]

      # Initialize accumulators with the biases.
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm22, [r9 + 64]
      vmovaps zmm8, zmm7
      vmovaps zmm9, zmm7
      vmovaps zmm14, zmm7
      vmovaps zmm15, zmm7
      vmovaps zmm16, zmm7
      vmovaps zmm17, zmm7
      vmovaps zmm18, zmm7
      vmovaps zmm19, zmm7
      vmovaps zmm20, zmm7
      vmovaps zmm21, zmm7
      vmovaps zmm23, zmm22
      vmovaps zmm24, zmm22
      vmovaps zmm25, zmm22
      vmovaps zmm26, zmm22
      vmovaps zmm27, zmm22
      vmovaps zmm28, zmm22
      vmovaps zmm29, zmm22
      vmovaps zmm30, zmm22
      vmovaps zmm12, zmm22
      vmovaps zmm13, zmm22
      add r9, 128

inner_loop:
      vmovaps  zmm10, [r9 + 0]
      vmovaps  zmm11, [r9 + 64]
      add r9, 128
      vbroadcastss zmm2, DWORD PTR [rsi + r11]
      vfmadd231ps  zmm7, zmm2, zmm10
      vfmadd231ps  zmm22, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [rax + r11]
      vfmadd231ps  zmm8, zmm2, zmm10
      vfmadd231ps  zmm23, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r15 + r11]
      vfmadd231ps  zmm9, zmm2, zmm10
      vfmadd231ps  zmm24, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r14 + r11]
      vfmadd231ps  zmm14, zmm2, zmm10
      vfmadd231ps  zmm25, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r12 + r11]
      vfmadd231ps  zmm15, zmm2, zmm10
      vfmadd231ps  zmm26, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r10 + r11]
      vfmadd231ps  zmm16, zmm2, zmm10
      vfmadd231ps  zmm27, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r13 + r11]
      vfmadd231ps  zmm17, zmm2, zmm10
      vfmadd231ps  zmm28, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [rbx + r11]
      vfmadd231ps  zmm18, zmm2, zmm10
      vfmadd231ps  zmm29, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [rbp + r11]
      vfmadd231ps  zmm19, zmm2, zmm10
      vfmadd231ps  zmm30, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r8 + r11]
      vfmadd231ps  zmm20, zmm2, zmm10
      vfmadd231ps  zmm12, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [rdi + r11]
      vfmadd231ps  zmm21, zmm2, zmm10
      vfmadd231ps  zmm13, zmm2, zmm11

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
      vminps  zmm21, zmm1, zmm21
      vminps  zmm22, zmm1, zmm22
      vminps  zmm23, zmm1, zmm23
      vminps  zmm24, zmm1, zmm24
      vminps  zmm25, zmm1, zmm25
      vminps  zmm26, zmm1, zmm26
      vminps  zmm27, zmm1, zmm27
      vminps  zmm28, zmm1, zmm28
      vminps  zmm29, zmm1, zmm29
      vminps  zmm30, zmm1, zmm30
      vminps  zmm12, zmm1, zmm12
      vminps  zmm13, zmm1, zmm13
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
      vmaxps  zmm21, zmm0, zmm21
      vmaxps  zmm22, zmm0, zmm22
      vmaxps  zmm23, zmm0, zmm23
      vmaxps  zmm24, zmm0, zmm24
      vmaxps  zmm25, zmm0, zmm25
      vmaxps  zmm26, zmm0, zmm26
      vmaxps  zmm27, zmm0, zmm27
      vmaxps  zmm28, zmm0, zmm28
      vmaxps  zmm29, zmm0, zmm29
      vmaxps  zmm30, zmm0, zmm30
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm13, zmm0, zmm13

      # Pop output pointers from the stack.
      mov rsi, [rsp - 136]
      mov rax, [rsp - 152]
      mov r15, [rsp - 168]
      mov r14, [rsp - 184]
      mov r12, [rsp - 200]
      mov r10, [rsp - 216]
      mov r13, [rsp - 232]
      mov rbx, [rsp - 248]
      mov rbp, [rsp - 264]
      mov r8, [rsp - 280]
      mov rdi, [rsp - 296]

      # Check whether full or partial store.
      cmp rcx, 32
      jl tail

      vmovups  [rsi], zmm7
      vmovups  [rsi + 64], zmm22
      vmovups  [rax], zmm8
      vmovups  [rax + 64], zmm23
      vmovups  [r15], zmm9
      vmovups  [r15 + 64], zmm24
      vmovups  [r14], zmm14
      vmovups  [r14 + 64], zmm25
      vmovups  [r12], zmm15
      vmovups  [r12 + 64], zmm26
      vmovups  [r10], zmm16
      vmovups  [r10 + 64], zmm27
      vmovups  [r13], zmm17
      vmovups  [r13 + 64], zmm28
      vmovups  [rbx], zmm18
      vmovups  [rbx + 64], zmm29
      vmovups  [rbp], zmm19
      vmovups  [rbp + 64], zmm30
      vmovups  [r8], zmm20
      vmovups  [r8 + 64], zmm12
      vmovups  [rdi], zmm21
      vmovups  [rdi + 64], zmm13
      add rsi, 128
      add rax, 128
      add r15, 128
      add r14, 128
      add r12, 128
      add r10, 128
      add r13, 128
      add rbx, 128
      add rbp, 128
      add r8, 128
      add rdi, 128

      # Write output pointers to the stack.
      mov [rsp - 136], rsi
      mov [rsp - 152], rax
      mov [rsp - 168], r15
      mov [rsp - 184], r14
      mov [rsp - 200], r12
      mov [rsp - 216], r10
      mov [rsp - 232], r13
      mov [rsp - 248], rbx
      mov [rsp - 264], rbp
      mov [rsp - 280], r8
      mov [rsp - 296], rdi

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
      vmovups  ZMMWORD PTR [rsi]{k1}, zmm7
      vmovups  ZMMWORD PTR [rsi + 64]{k2}, zmm22
      vmovups  ZMMWORD PTR [rax]{k1}, zmm8
      vmovups  ZMMWORD PTR [rax + 64]{k2}, zmm23
      vmovups  ZMMWORD PTR [r15]{k1}, zmm9
      vmovups  ZMMWORD PTR [r15 + 64]{k2}, zmm24
      vmovups  ZMMWORD PTR [r14]{k1}, zmm14
      vmovups  ZMMWORD PTR [r14 + 64]{k2}, zmm25
      vmovups  ZMMWORD PTR [r12]{k1}, zmm15
      vmovups  ZMMWORD PTR [r12 + 64]{k2}, zmm26
      vmovups  ZMMWORD PTR [r10]{k1}, zmm16
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm27
      vmovups  ZMMWORD PTR [r13]{k1}, zmm17
      vmovups  ZMMWORD PTR [r13 + 64]{k2}, zmm28
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm18
      vmovups  ZMMWORD PTR [rbx + 64]{k2}, zmm29
      vmovups  ZMMWORD PTR [rbp]{k1}, zmm19
      vmovups  ZMMWORD PTR [rbp + 64]{k2}, zmm30
      vmovups  ZMMWORD PTR [r8]{k1}, zmm20
      vmovups  ZMMWORD PTR [r8 + 64]{k2}, zmm12
      vmovups  ZMMWORD PTR [rdi]{k1}, zmm21
      vmovups  ZMMWORD PTR [rdi + 64]{k2}, zmm13

return:

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_11x32__asm_amd64_avx512f_broadcast