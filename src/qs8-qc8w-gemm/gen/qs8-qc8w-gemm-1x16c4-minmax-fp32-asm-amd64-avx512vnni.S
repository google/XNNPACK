// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"
.SIGN_MASK:
      .quad   -9187201950435737472  # 0x8080808080808080

BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_amd64_avx512vnni

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

      # load params to free up GP registers
      mov r13, [rsp + 96] # params

      movsx         eax, WORD PTR [r13]
      vpbroadcastd zmm31, eax

      vpbroadcastb xmm0, BYTE PTR [r13 + 2]

      movsx         eax, WORD PTR [r13 + 4]
      vpbroadcastd  zmm1, eax
      vpsubd        zmm1, zmm1, zmm31
      vcvtdq2ps     zmm1, zmm1


      # Load c pointer.
      mov r10, [rsp + 72]
      # Load cm_stride.
      mov r11, [rsp + 80]

      add rdx, 3
      and rdx, -4

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13

      # Allocate some space on the stack.
      sub rsp, 128

      # Load 0x80 for xoring the weights
      vbroadcastsd  zmm13, qword ptr [rip + .SIGN_MASK]


.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with bias
      vmovaps zmm5, [r9 + 0]
      add r9, 64

.Linner_loop:
      vmovaps  zmm6, [r9 + 0]
      add r9, 64
      vpxord zmm2, zmm13, DWORD PTR [rcx + r11]{1to16}
      vpdpbusd  zmm5, zmm2, zmm6

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:

      # Convert from int32 to float.
      vcvtdq2ps zmm5, zmm5
      vmovaps zmm10, [r9 + 0]
      add r9, 64
      vmulps zmm5, zmm5, zmm10
      vminps zmm5, zmm5, zmm1
      vcvtps2dq zmm5, zmm5
      vpaddd zmm5, zmm5, zmm31
      vpmovsdb xmm5, zmm5
      vpmaxsb xmm5, xmm5, xmm0

      # Check whether full or partial store.
      cmp rsi, 16
      jl .Ltail

      vmovups  [r10], xmm5
      add r10, 16

      sub rsi, 16
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      vmovdqu8  XMMWORD PTR [r10]{k1}, xmm5

.Lreturn:
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
END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_amd64_avx512vnni

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_amd64_avx512vnni.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_amd64_avx512vnni.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__