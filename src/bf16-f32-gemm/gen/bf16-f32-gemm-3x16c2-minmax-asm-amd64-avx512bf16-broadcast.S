// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_bf16_f32_gemm_minmax_ukernel_3x16c2__asm_amd64_avx512bf16_broadcast

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
      sub rsp, 128

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

      # Copy k and flip bit.
      mov r11, rdx
      and r11, 0x2
      and rdx, 0xFFFFFFFFFFFFFFFD
      mov [rsp + 72], r11

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with the biases.
      vmovaps  zmm7, [r9 + 0]
      vmovaps zmm8, zmm7
      vmovaps zmm9, zmm7
      add r9, 64

      # Are there at least 4 bytes?
      cmp rdx, 4
      js inner_loop_tail

inner_loop:
      vmovaps  zmm10, [r9 + 0]
      add r9, 64
      vbroadcastss zmm2, DWORD PTR [rcx + r11]
      vdpbf16ps  zmm7, zmm2, zmm10
      vbroadcastss zmm3, DWORD PTR [rax + r11]
      vdpbf16ps  zmm8, zmm3, zmm10
      vbroadcastss zmm4, DWORD PTR [r15 + r11]
      vdpbf16ps  zmm9, zmm4, zmm10

      add r11, 4
      cmp rdx, r11
      jne inner_loop

      # Store nc_register.
      mov [rsp + 80], rsi
      # Load odd k bit.
      mov rsi, [rsp + 72]
      # Check if channels are odd.
      test rsi, rsi
      mov rsi, [rsp + 80]
      jz inner_loop_end

inner_loop_tail:
      vmovaps  zmm10, [r9 + 0]
      add r9, 64
      vbroadcastss zmm2, DWORD PTR [rcx + r11]
      vpslld zmm2, zmm2, 16
      vpsrad zmm2, zmm2, 16
      vdpbf16ps  zmm7, zmm2, zmm10

      vbroadcastss zmm3, DWORD PTR [rax + r11]
      vpslld zmm3, zmm3, 16
      vpsrad zmm3, zmm3, 16
      vdpbf16ps  zmm8, zmm3, zmm10

      vbroadcastss zmm4, DWORD PTR [r15 + r11]
      vpslld zmm4, zmm4, 16
      vpsrad zmm4, zmm4, 16
      vdpbf16ps  zmm9, zmm4, zmm10

inner_loop_end:
      # Min/max clamping.
      vminps  zmm7, zmm1, zmm7
      vminps  zmm8, zmm1, zmm8
      vminps  zmm9, zmm1, zmm9
      vmaxps  zmm7, zmm0, zmm7
      vmaxps  zmm8, zmm0, zmm8
      vmaxps  zmm9, zmm0, zmm9

      # Check whether full or partial store.
      cmp rsi, 16
      jl tail

      vmovups  [r10], zmm7
      vmovups  [r13], zmm8
      vmovups  [rbx], zmm9
      add r10, 64
      add r13, 64
      add rbx, 64

      sub rsi, 16
      jne outer_loop
      jmp return

tail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovups  ZMMWORD PTR [r10]{k1}, zmm7
      vmovups  ZMMWORD PTR [r13]{k1}, zmm8
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm9

return:
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
END_FUNCTION xnn_bf16_f32_gemm_minmax_ukernel_3x16c2__asm_amd64_avx512bf16_broadcast

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_bf16_f32_gemm_minmax_ukernel_3x16c2__asm_amd64_avx512bf16_broadcast.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_bf16_f32_gemm_minmax_ukernel_3x16c2__asm_amd64_avx512bf16_broadcast.dfsan
      #endif