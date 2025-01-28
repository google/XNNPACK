// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_11x16__asm_amd64_avx512f_broadcast

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
      sub rsp, 208
      # Write rsi (a pointer) to the stack as we need the register.
      mov [rsp], rcx
      # Write r10 (c pointer) to the stack as we need the register.
      mov [rsp + 8], r10

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 16], rax
      mov [rsp + 24], r13

      # Clamp a & c pointers if mr <= 2
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 2
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 32], rcx
      mov [rsp + 40], r10

      # Clamp a & c pointers if mr <= 3
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 3
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 48], rax
      mov [rsp + 56], r13

      # Clamp a & c pointers if mr <= 4
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 4
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 64], rcx
      mov [rsp + 72], r10

      # Clamp a & c pointers if mr <= 5
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 5
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 80], rax
      mov [rsp + 88], r13

      # Clamp a & c pointers if mr <= 6
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 6
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 96], rcx
      mov [rsp + 104], r10

      # Clamp a & c pointers if mr <= 7
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 7
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 112], rax
      mov [rsp + 120], r13

      # Clamp a & c pointers if mr <= 8
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 8
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 128], rcx
      mov [rsp + 136], r10

      # Clamp a & c pointers if mr <= 9
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 9
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 144], rax
      mov [rsp + 152], r13

      # Clamp a & c pointers if mr <= 10
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 10
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 160], rcx
      mov [rsp + 168], r10

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Read a pointers from stack into GP registers.
      mov rcx, [rsp + 0]
      mov rax, [rsp + 16]
      mov r15, [rsp + 32]
      mov r14, [rsp + 48]
      mov r12, [rsp + 64]
      mov r10, [rsp + 80]
      mov r13, [rsp + 96]
      mov rbx, [rsp + 112]
      mov rbp, [rsp + 128]
      mov r8, [rsp + 144]
      mov rdi, [rsp + 160]

      # Initialize accumulators with the biases.
      vmovaps  zmm7, [r9 + 0]
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
      add r9, 64

inner_loop:
      vmovaps  zmm10, [r9 + 0]
      add r9, 64
      vbroadcastss zmm2, DWORD PTR [rcx + r11]
      vfmadd231ps  zmm7, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [rax + r11]
      vfmadd231ps  zmm8, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [r15 + r11]
      vfmadd231ps  zmm9, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [r14 + r11]
      vfmadd231ps  zmm14, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [r12 + r11]
      vfmadd231ps  zmm15, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [r10 + r11]
      vfmadd231ps  zmm16, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [r13 + r11]
      vfmadd231ps  zmm17, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [rbx + r11]
      vfmadd231ps  zmm18, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [rbp + r11]
      vfmadd231ps  zmm19, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [r8 + r11]
      vfmadd231ps  zmm20, zmm2, zmm10
      vbroadcastss zmm2, DWORD PTR [rdi + r11]
      vfmadd231ps  zmm21, zmm2, zmm10

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

      # Pop output pointers from the stack.
      mov rcx, [rsp + 8]
      mov rax, [rsp + 24]
      mov r15, [rsp + 40]
      mov r14, [rsp + 56]
      mov r12, [rsp + 72]
      mov r10, [rsp + 88]
      mov r13, [rsp + 104]
      mov rbx, [rsp + 120]
      mov rbp, [rsp + 136]
      mov r8, [rsp + 152]
      mov rdi, [rsp + 168]

      # Check whether full or partial store.
      cmp rsi, 16
      jl tail

      vmovups  [rcx], zmm7
      vmovups  [rax], zmm8
      vmovups  [r15], zmm9
      vmovups  [r14], zmm14
      vmovups  [r12], zmm15
      vmovups  [r10], zmm16
      vmovups  [r13], zmm17
      vmovups  [rbx], zmm18
      vmovups  [rbp], zmm19
      vmovups  [r8], zmm20
      vmovups  [rdi], zmm21
      add rcx, 64
      add rax, 64
      add r15, 64
      add r14, 64
      add r12, 64
      add r10, 64
      add r13, 64
      add rbx, 64
      add rbp, 64
      add r8, 64
      add rdi, 64

      # Write output pointers to the stack.
      mov [rsp + 8], rcx
      mov [rsp + 24], rax
      mov [rsp + 40], r15
      mov [rsp + 56], r14
      mov [rsp + 72], r12
      mov [rsp + 88], r10
      mov [rsp + 104], r13
      mov [rsp + 120], rbx
      mov [rsp + 136], rbp
      mov [rsp + 152], r8
      mov [rsp + 168], rdi

      sub rsi, 16
      jne outer_loop
      jmp return

tail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  ZMMWORD PTR [rcx]{k1}, zmm7
      vmovups  ZMMWORD PTR [rax]{k1}, zmm8
      vmovups  ZMMWORD PTR [r15]{k1}, zmm9
      vmovups  ZMMWORD PTR [r14]{k1}, zmm14
      vmovups  ZMMWORD PTR [r12]{k1}, zmm15
      vmovups  ZMMWORD PTR [r10]{k1}, zmm16
      vmovups  ZMMWORD PTR [r13]{k1}, zmm17
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm18
      vmovups  ZMMWORD PTR [rbp]{k1}, zmm19
      vmovups  ZMMWORD PTR [r8]{k1}, zmm20
      vmovups  ZMMWORD PTR [rdi]{k1}, zmm21

return:
      add rsp, 208

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_11x16__asm_amd64_avx512f_broadcast