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

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_5x32c2__asm_amd64_avx512f_broadcast

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
      sub rsp, 192

      # Clamp a & c pointers if mr <= 1
      mov rax, rcx
      add rax, r8
      mov r13, r10
      add r13, r11
      cmp rdi, 1
      cmovle rax, rcx
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

      # Copy k and flip bit.
      mov r11, rdx
      and r11, 0x4
      and rdx, 0xFFFFFFFFFFFFFFFB
      mov [rsp + 104], r11
      mov r11, 0x5555
      kmovw k3, r11d

.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      # Interleave with zeros.
      vpmovzxdq zmm11, ymm7
      vextracti64x4 ymm7, zmm7, 1
      vpmovzxdq zmm16, ymm7
      vpmovzxdq zmm21, ymm8
      vextracti64x4 ymm8, zmm8, 1
      vpmovzxdq zmm26, ymm8
      vmovaps zmm12, zmm11
      vmovaps zmm13, zmm11
      vmovaps zmm14, zmm11
      vmovaps zmm15, zmm11
      vmovaps zmm17, zmm16
      vmovaps zmm18, zmm16
      vmovaps zmm19, zmm16
      vmovaps zmm20, zmm16
      vmovaps zmm22, zmm21
      vmovaps zmm23, zmm21
      vmovaps zmm24, zmm21
      vmovaps zmm25, zmm21
      vmovaps zmm27, zmm26
      vmovaps zmm28, zmm26
      vmovaps zmm29, zmm26
      vmovaps zmm30, zmm26
      add r9, 128

      # Are there at least 8 bytes?
      cmp rdx, 8
      js .Linner_loop_tail

.Linner_loop:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      vmovaps  zmm9, [r9 + 128]
      vmovaps  zmm10, [r9 + 192]
      add r9, 256
      vbroadcastsd zmm2, QWORD PTR [rcx + r11]
      vfmadd231ps  zmm11, zmm2, zmm7
      vfmadd231ps  zmm16, zmm2, zmm8
      vfmadd231ps  zmm21, zmm2, zmm9
      vfmadd231ps  zmm26, zmm2, zmm10
      vbroadcastsd zmm3, QWORD PTR [rax + r11]
      vfmadd231ps  zmm12, zmm3, zmm7
      vfmadd231ps  zmm17, zmm3, zmm8
      vfmadd231ps  zmm22, zmm3, zmm9
      vfmadd231ps  zmm27, zmm3, zmm10
      vbroadcastsd zmm4, QWORD PTR [r15 + r11]
      vfmadd231ps  zmm13, zmm4, zmm7
      vfmadd231ps  zmm18, zmm4, zmm8
      vfmadd231ps  zmm23, zmm4, zmm9
      vfmadd231ps  zmm28, zmm4, zmm10
      vbroadcastsd zmm5, QWORD PTR [r14 + r11]
      vfmadd231ps  zmm14, zmm5, zmm7
      vfmadd231ps  zmm19, zmm5, zmm8
      vfmadd231ps  zmm24, zmm5, zmm9
      vfmadd231ps  zmm29, zmm5, zmm10
      vbroadcastsd zmm6, QWORD PTR [r12 + r11]
      vfmadd231ps  zmm15, zmm6, zmm7
      vfmadd231ps  zmm20, zmm6, zmm8
      vfmadd231ps  zmm25, zmm6, zmm9
      vfmadd231ps  zmm30, zmm6, zmm10

      add r11, 8
      cmp rdx, r11
      jne .Linner_loop

      # Store nc_register.
      mov [rsp + 112], rsi
      # Load odd k bit.
      mov rsi, [rsp + 104]
      # Check if channels are odd.
      test rsi, rsi
      mov rsi, [rsp + 112]
      jz .Linner_loop_end

.Linner_loop_tail:
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm8, [r9 + 64]
      vmovaps  zmm9, [r9 + 128]
      vmovaps  zmm10, [r9 + 192]
      add r9, 256
      vbroadcastsd zmm2, QWORD PTR [rcx + r11]
      vfmadd231ps  zmm11{k3}, zmm2, zmm7
      vfmadd231ps  zmm16{k3}, zmm2, zmm8
      vfmadd231ps  zmm21{k3}, zmm2, zmm9
      vfmadd231ps  zmm26{k3}, zmm2, zmm10
      vbroadcastsd zmm3, QWORD PTR [rax + r11]
      vfmadd231ps  zmm12{k3}, zmm3, zmm7
      vfmadd231ps  zmm17{k3}, zmm3, zmm8
      vfmadd231ps  zmm22{k3}, zmm3, zmm9
      vfmadd231ps  zmm27{k3}, zmm3, zmm10
      vbroadcastsd zmm4, QWORD PTR [r15 + r11]
      vfmadd231ps  zmm13{k3}, zmm4, zmm7
      vfmadd231ps  zmm18{k3}, zmm4, zmm8
      vfmadd231ps  zmm23{k3}, zmm4, zmm9
      vfmadd231ps  zmm28{k3}, zmm4, zmm10
      vbroadcastsd zmm5, QWORD PTR [r14 + r11]
      vfmadd231ps  zmm14{k3}, zmm5, zmm7
      vfmadd231ps  zmm19{k3}, zmm5, zmm8
      vfmadd231ps  zmm24{k3}, zmm5, zmm9
      vfmadd231ps  zmm29{k3}, zmm5, zmm10
      vbroadcastsd zmm6, QWORD PTR [r12 + r11]
      vfmadd231ps  zmm15{k3}, zmm6, zmm7
      vfmadd231ps  zmm20{k3}, zmm6, zmm8
      vfmadd231ps  zmm25{k3}, zmm6, zmm9
      vfmadd231ps  zmm30{k3}, zmm6, zmm10
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
      vpsrlq zmm7, zmm29, 32
      vaddps zmm29, zmm29, zmm7
      vpsrlq zmm7, zmm30, 32
      vaddps zmm30, zmm30, zmm7
      vmovups zmm7, zmmword ptr [rip + .PERMUTATION]
      vpermt2ps zmm11, zmm7, zmm16
      vpermt2ps zmm12, zmm7, zmm17
      vpermt2ps zmm13, zmm7, zmm18
      vpermt2ps zmm14, zmm7, zmm19
      vpermt2ps zmm15, zmm7, zmm20
      vpermt2ps zmm21, zmm7, zmm26
      vpermt2ps zmm22, zmm7, zmm27
      vpermt2ps zmm23, zmm7, zmm28
      vpermt2ps zmm24, zmm7, zmm29
      vpermt2ps zmm25, zmm7, zmm30
      # Min/max clamping.
      vminps  zmm11, zmm1, zmm11
      vminps  zmm12, zmm1, zmm12
      vminps  zmm13, zmm1, zmm13
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vminps  zmm16, zmm1, zmm21
      vminps  zmm17, zmm1, zmm22
      vminps  zmm18, zmm1, zmm23
      vminps  zmm19, zmm1, zmm24
      vminps  zmm20, zmm1, zmm25
      vmaxps  zmm11, zmm0, zmm11
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm13, zmm0, zmm13
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm19, zmm0, zmm19
      vmaxps  zmm20, zmm0, zmm20

      # Check whether full or partial store.
      cmp rsi, 32
      jl .Ltail

      vmovups  [r10], zmm11
      vmovups  [r10 + 64], zmm16
      vmovups  [r13], zmm12
      vmovups  [r13 + 64], zmm17
      vmovups  [rbx], zmm13
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

      sub rsi, 32
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      shr r11d, 16
      kmovw k2, r11d
      vmovups  ZMMWORD PTR [r10]{k1}, zmm11
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm16
      vmovups  ZMMWORD PTR [r13]{k1}, zmm12
      vmovups  ZMMWORD PTR [r13 + 64]{k2}, zmm17
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm13
      vmovups  ZMMWORD PTR [rbx + 64]{k2}, zmm18
      vmovups  ZMMWORD PTR [rbp]{k1}, zmm14
      vmovups  ZMMWORD PTR [rbp + 64]{k2}, zmm19
      vmovups  ZMMWORD PTR [r8]{k1}, zmm15
      vmovups  ZMMWORD PTR [r8 + 64]{k2}, zmm20

.Lreturn:
      add rsp, 192
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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_5x32c2__asm_amd64_avx512f_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_5x32c2__asm_amd64_avx512f_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_5x32c2__asm_amd64_avx512f_broadcast.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__