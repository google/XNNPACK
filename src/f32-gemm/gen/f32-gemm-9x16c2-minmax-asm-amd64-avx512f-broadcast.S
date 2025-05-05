// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"
.PERMUTATION:
      .long   0
      .long   2
      .long   4
      .long   6
      .long   8
      .long   10
      .long   12
      .long   14
      .long   16
      .long   18
      .long   20
      .long   22
      .long   24
      .long   26
      .long   28
      .long   30

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_9x16c2__asm_amd64_avx512f_broadcast

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
      vbroadcastss zmm0, DWORD PTR [r13]
      vbroadcastss zmm1, DWORD PTR [r13 + 4]

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
      sub rsp, 256
      # Write rsi (a pointer) to the stack as we need the register.
      mov [rsp + 16], rcx
      # Write r10 (c pointer) to the stack as we need the register.
      mov [rsp + 24], r10

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 32], rax
      mov [rsp + 40], r13

      # Clamp a & c pointers if mr <= 2
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 2
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 48], rcx
      mov [rsp + 56], r10

      # Clamp a & c pointers if mr <= 3
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 3
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 64], rax
      mov [rsp + 72], r13

      # Clamp a & c pointers if mr <= 4
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 4
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 80], rcx
      mov [rsp + 88], r10

      # Clamp a & c pointers if mr <= 5
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 5
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 96], rax
      mov [rsp + 104], r13

      # Clamp a & c pointers if mr <= 6
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 6
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 112], rcx
      mov [rsp + 120], r10

      # Clamp a & c pointers if mr <= 7
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 7
      cmovle rax, rcx
      cmovle r13, r10

      mov [rsp + 128], rax
      mov [rsp + 136], r13

      # Clamp a & c pointers if mr <= 8
      mov rcx, rax
      add rcx, r8
      mov r10, r13
      add r10, r11
      cmp rdi, 8
      cmovle rcx, rax
      cmovle r10, r13

      mov [rsp + 144], rcx
      mov [rsp + 152], r10

      # Copy k and flip bit.
      mov r11, rdx
      and r11, 0x4
      and rdx, 0xFFFFFFFFFFFFFFFB
      mov [rsp + 168], r11
      mov r11, 0x5555
      kmovw k3, r11d

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Read a pointers from stack into GP registers.
      mov rcx, [rsp + 16]
      mov rax, [rsp + 32]
      mov r15, [rsp + 48]
      mov r14, [rsp + 64]
      mov r12, [rsp + 80]
      mov r10, [rsp + 96]
      mov r13, [rsp + 112]
      mov rbx, [rsp + 128]
      mov rbp, [rsp + 144]

      vmovaps  zmm7, [r9 + 0]
      # Interleave with zeros.
      vpmovzxdq zmm11, ymm7
      vextracti64x4 ymm7, zmm7, 1
      vpmovzxdq zmm20, ymm7
      vmovaps zmm12, zmm11
      vmovaps zmm13, zmm11
      vmovaps zmm14, zmm11
      vmovaps zmm15, zmm11
      vmovaps zmm16, zmm11
      vmovaps zmm17, zmm11
      vmovaps zmm18, zmm11
      vmovaps zmm19, zmm11
      vmovaps zmm21, zmm20
      vmovaps zmm22, zmm20
      vmovaps zmm23, zmm20
      vmovaps zmm24, zmm20
      vmovaps zmm25, zmm20
      vmovaps zmm26, zmm20
      vmovaps zmm27, zmm20
      vmovaps zmm28, zmm20
      add r9, 64

      # Are there at least 8 bytes?
      cmp rdx, 8
      js .Linner_loop_tail

.Linner_loop:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      add r9, 128
      vbroadcastsd zmm2, QWORD PTR [rcx + r11]
      vfmadd231ps  zmm11, zmm2, zmm7
      vfmadd231ps  zmm20, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [rax + r11]
      vfmadd231ps  zmm12, zmm2, zmm7
      vfmadd231ps  zmm21, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r15 + r11]
      vfmadd231ps  zmm13, zmm2, zmm7
      vfmadd231ps  zmm22, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r14 + r11]
      vfmadd231ps  zmm14, zmm2, zmm7
      vfmadd231ps  zmm23, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r12 + r11]
      vfmadd231ps  zmm15, zmm2, zmm7
      vfmadd231ps  zmm24, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r10 + r11]
      vfmadd231ps  zmm16, zmm2, zmm7
      vfmadd231ps  zmm25, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r13 + r11]
      vfmadd231ps  zmm17, zmm2, zmm7
      vfmadd231ps  zmm26, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [rbx + r11]
      vfmadd231ps  zmm18, zmm2, zmm7
      vfmadd231ps  zmm27, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [rbp + r11]
      vfmadd231ps  zmm19, zmm2, zmm7
      vfmadd231ps  zmm28, zmm2, zmm8

      add r11, 8
      cmp rdx, r11
      jne .Linner_loop

      # Store nc_register.
      mov [rsp + 176], rsi
      # Load odd k bit.
      mov rsi, [rsp + 168]
      # Check if channels are odd.
      test rsi, rsi
      mov rsi, [rsp + 176]
      jz .Linner_loop_end

.Linner_loop_tail:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      add r9, 128
      vbroadcastsd zmm2, QWORD PTR [rcx + r11]
      vfmadd231ps  zmm11{k3}, zmm2, zmm7
      vfmadd231ps  zmm20{k3}, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [rax + r11]
      vfmadd231ps  zmm12{k3}, zmm2, zmm7
      vfmadd231ps  zmm21{k3}, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r15 + r11]
      vfmadd231ps  zmm13{k3}, zmm2, zmm7
      vfmadd231ps  zmm22{k3}, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r14 + r11]
      vfmadd231ps  zmm14{k3}, zmm2, zmm7
      vfmadd231ps  zmm23{k3}, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r12 + r11]
      vfmadd231ps  zmm15{k3}, zmm2, zmm7
      vfmadd231ps  zmm24{k3}, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r10 + r11]
      vfmadd231ps  zmm16{k3}, zmm2, zmm7
      vfmadd231ps  zmm25{k3}, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [r13 + r11]
      vfmadd231ps  zmm17{k3}, zmm2, zmm7
      vfmadd231ps  zmm26{k3}, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [rbx + r11]
      vfmadd231ps  zmm18{k3}, zmm2, zmm7
      vfmadd231ps  zmm27{k3}, zmm2, zmm8
      vbroadcastsd zmm2, QWORD PTR [rbp + r11]
      vfmadd231ps  zmm19{k3}, zmm2, zmm7
      vfmadd231ps  zmm28{k3}, zmm2, zmm8

.Linner_loop_end:
      vpsrlq zmm7, zmm11, 32
      vaddps zmm11, zmm11, zmm7
      vpsrlq zmm7, zmm12, 32
      vaddps zmm12, zmm12, zmm7
      vpsrlq zmm7, zmm13, 32
      vaddps zmm13, zmm13, zmm7
      vpsrlq zmm7, zmm14, 32
      vaddps zmm14, zmm14, zmm7
      vpsrlq zmm7, zmm15, 32
      vaddps zmm15, zmm15, zmm7
      vpsrlq zmm7, zmm16, 32
      vaddps zmm16, zmm16, zmm7
      vpsrlq zmm7, zmm17, 32
      vaddps zmm17, zmm17, zmm7
      vpsrlq zmm7, zmm18, 32
      vaddps zmm18, zmm18, zmm7
      vpsrlq zmm7, zmm19, 32
      vaddps zmm19, zmm19, zmm7
      vpsrlq zmm7, zmm20, 32
      vaddps zmm20, zmm20, zmm7
      vpsrlq zmm7, zmm21, 32
      vaddps zmm21, zmm21, zmm7
      vpsrlq zmm7, zmm22, 32
      vaddps zmm22, zmm22, zmm7
      vpsrlq zmm7, zmm23, 32
      vaddps zmm23, zmm23, zmm7
      vpsrlq zmm7, zmm24, 32
      vaddps zmm24, zmm24, zmm7
      vpsrlq zmm7, zmm25, 32
      vaddps zmm25, zmm25, zmm7
      vpsrlq zmm7, zmm26, 32
      vaddps zmm26, zmm26, zmm7
      vpsrlq zmm7, zmm27, 32
      vaddps zmm27, zmm27, zmm7
      vpsrlq zmm7, zmm28, 32
      vaddps zmm28, zmm28, zmm7
      vmovups zmm7, zmmword ptr [rip + .PERMUTATION]
      vpermt2ps zmm11, zmm7, zmm20
      vpermt2ps zmm12, zmm7, zmm21
      vpermt2ps zmm13, zmm7, zmm22
      vpermt2ps zmm14, zmm7, zmm23
      vpermt2ps zmm15, zmm7, zmm24
      vpermt2ps zmm16, zmm7, zmm25
      vpermt2ps zmm17, zmm7, zmm26
      vpermt2ps zmm18, zmm7, zmm27
      vpermt2ps zmm19, zmm7, zmm28
      # Min/max clamping.
      vminps  zmm11, zmm1, zmm11
      vminps  zmm12, zmm1, zmm12
      vminps  zmm13, zmm1, zmm13
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vminps  zmm16, zmm1, zmm16
      vminps  zmm17, zmm1, zmm17
      vminps  zmm18, zmm1, zmm18
      vminps  zmm19, zmm1, zmm19
      vmaxps  zmm11, zmm0, zmm11
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm13, zmm0, zmm13
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm19, zmm0, zmm19

      # Pop output pointers from the stack.
      mov rcx, [rsp + 24]
      mov rax, [rsp + 40]
      mov r15, [rsp + 56]
      mov r14, [rsp + 72]
      mov r12, [rsp + 88]
      mov r10, [rsp + 104]
      mov r13, [rsp + 120]
      mov rbx, [rsp + 136]
      mov rbp, [rsp + 152]

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail

      vmovups  [rcx], zmm11
      vmovups  [rax], zmm12
      vmovups  [r15], zmm13
      vmovups  [r14], zmm14
      vmovups  [r12], zmm15
      vmovups  [r10], zmm16
      vmovups  [r13], zmm17
      vmovups  [rbx], zmm18
      vmovups  [rbp], zmm19
      add rcx, 64
      add rax, 64
      add r15, 64
      add r14, 64
      add r12, 64
      add r10, 64
      add r13, 64
      add rbx, 64
      add rbp, 64

      # Write output pointers to the stack.
      mov [rsp + 24], rcx
      mov [rsp + 40], rax
      mov [rsp + 56], r15
      mov [rsp + 72], r14
      mov [rsp + 88], r12
      mov [rsp + 104], r10
      mov [rsp + 120], r13
      mov [rsp + 136], rbx
      mov [rsp + 152], rbp

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  ZMMWORD PTR [rcx]{k1}, zmm11
      vmovups  ZMMWORD PTR [rax]{k1}, zmm12
      vmovups  ZMMWORD PTR [r15]{k1}, zmm13
      vmovups  ZMMWORD PTR [r14]{k1}, zmm14
      vmovups  ZMMWORD PTR [r12]{k1}, zmm15
      vmovups  ZMMWORD PTR [r10]{k1}, zmm16
      vmovups  ZMMWORD PTR [r13]{k1}, zmm17
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm18
      vmovups  ZMMWORD PTR [rbp]{k1}, zmm19

.Lreturn:
      add rsp, 256
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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_9x16c2__asm_amd64_avx512f_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_9x16c2__asm_amd64_avx512f_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_9x16c2__asm_amd64_avx512f_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__