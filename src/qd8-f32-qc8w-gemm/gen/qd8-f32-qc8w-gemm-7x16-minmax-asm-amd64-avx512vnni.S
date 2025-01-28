// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__asm_amd64_avx512vnni

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
      sub rsp, 912
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

      # Load quantization params pointer from stack
      mov r11, [rsp + 1000]
      mov edi, [r11 + 0]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 464], zmm6
      mov edi, [r11 + 8]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 528], zmm6
      mov edi, [r11 + 16]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 592], zmm6
      mov edi, [r11 + 24]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 656], zmm6
      mov edi, [r11 + 32]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 720], zmm6
      mov edi, [r11 + 40]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 784], zmm6
      mov edi, [r11 + 48]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 848], zmm6

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

      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 464]
      vpmulld zmm12, zmm6, ZMMWORD PTR [rsp + 528]
      vpmulld zmm14, zmm6, ZMMWORD PTR [rsp + 592]
      vpmulld zmm15, zmm6, ZMMWORD PTR [rsp + 656]
      vpmulld zmm16, zmm6, ZMMWORD PTR [rsp + 720]
      vpmulld zmm17, zmm6, ZMMWORD PTR [rsp + 784]
      vpmulld zmm18, zmm6, ZMMWORD PTR [rsp + 848]
      add r9, 64

inner_loop:
      vmovaps  zmm6, [r9 + 0]
      add r9, 64
      vpbroadcastd zmm2, [rcx + r11]
      vpdpbusd  zmm5, zmm2, zmm6
      vpbroadcastd zmm2, [rax + r11]
      vpdpbusd  zmm12, zmm2, zmm6
      vpbroadcastd zmm2, [r15 + r11]
      vpdpbusd  zmm14, zmm2, zmm6
      vpbroadcastd zmm2, [r14 + r11]
      vpdpbusd  zmm15, zmm2, zmm6
      vpbroadcastd zmm2, [r12 + r11]
      vpdpbusd  zmm16, zmm2, zmm6
      vpbroadcastd zmm2, [r10 + r11]
      vpdpbusd  zmm17, zmm2, zmm6
      vpbroadcastd zmm2, [r13 + r11]
      vpdpbusd  zmm18, zmm2, zmm6

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
      vcvtdq2ps zmm18, zmm18
      # Load quantization_params pointer from stack
      mov r11, [rsp + 1000]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm12, zmm12, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm14, zmm14, DWORD PTR [r11 + 20]{1to16}
      vmulps zmm15, zmm15, DWORD PTR [r11 + 28]{1to16}
      vmulps zmm16, zmm16, DWORD PTR [r11 + 36]{1to16}
      vmulps zmm17, zmm17, DWORD PTR [r11 + 44]{1to16}
      vmulps zmm18, zmm18, DWORD PTR [r11 + 52]{1to16}
      vmovaps zmm10, [r9 + 0]
      add r9, 64
      vmovaps zmm6, [r9 + 0]
      add r9, 64
      vfmadd213ps zmm5, zmm10, zmm6
      vfmadd213ps zmm12, zmm10, zmm6
      vfmadd213ps zmm14, zmm10, zmm6
      vfmadd213ps zmm15, zmm10, zmm6
      vfmadd213ps zmm16, zmm10, zmm6
      vfmadd213ps zmm17, zmm10, zmm6
      vfmadd213ps zmm18, zmm10, zmm6
      # Min/max clamping.
      vminps  zmm5, zmm1, zmm5
      vminps  zmm12, zmm1, zmm12
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vminps  zmm16, zmm1, zmm16
      vminps  zmm17, zmm1, zmm17
      vminps  zmm18, zmm1, zmm18
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm18, zmm0, zmm18

      # Pop output pointers from the stack.
      mov rcx, [rsp + 8]
      mov rax, [rsp + 24]
      mov r15, [rsp + 40]
      mov r14, [rsp + 56]
      mov r12, [rsp + 72]
      mov r10, [rsp + 88]
      mov r13, [rsp + 104]

      # Check whether full or partial store.
      cmp rsi, 16
      jl tail

      vmovups  [rcx], zmm5
      vmovups  [rax], zmm12
      vmovups  [r15], zmm14
      vmovups  [r14], zmm15
      vmovups  [r12], zmm16
      vmovups  [r10], zmm17
      vmovups  [r13], zmm18
      add rcx, 64
      add rax, 64
      add r15, 64
      add r14, 64
      add r12, 64
      add r10, 64
      add r13, 64

      # Write output pointers to the stack.
      mov [rsp + 8], rcx
      mov [rsp + 24], rax
      mov [rsp + 40], r15
      mov [rsp + 56], r14
      mov [rsp + 72], r12
      mov [rsp + 88], r10
      mov [rsp + 104], r13

      sub rsi, 16
      jne outer_loop
      jmp return

tail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  ZMMWORD PTR [rcx]{k1}, zmm5
      vmovups  ZMMWORD PTR [rax]{k1}, zmm12
      vmovups  ZMMWORD PTR [r15]{k1}, zmm14
      vmovups  ZMMWORD PTR [r14]{k1}, zmm15
      vmovups  ZMMWORD PTR [r12]{k1}, zmm16
      vmovups  ZMMWORD PTR [r10]{k1}, zmm17
      vmovups  ZMMWORD PTR [r13]{k1}, zmm18

return:
      add rsp, 912

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__asm_amd64_avx512vnni