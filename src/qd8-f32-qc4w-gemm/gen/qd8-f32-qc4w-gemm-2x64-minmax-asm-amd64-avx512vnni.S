// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"
.MASK:
      .quad   -1085102592571150096

BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni

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

      add rdx, 3
      and rdx, -4

      # Move stack parameters which have not yet been loaded
      mov r12, [rsp + 104]

      # Align the stack pointer.
      mov r13, rsp
      sub rsp, 64
      and rsp, 0xFFFFFFFFFFFFFFC0
      # Store the old stack pointer containing the return address
      mov [rsp], r13
      # Push additional stack parameters to the new stack
      mov [rsp + 8], r12

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
      # Load quantization_params pointer from stack
      mov r11, [rsp + 200]
      mov edi, [r11 + 0]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 64], zmm6
      mov edi, [r11 + 8]
      vpbroadcastd zmm6, edi
      vmovaps zmmword ptr [rsp + 128], zmm6

      mov r11, [rsp + 88]
      # Load 0xF0 for masking the weights
      vbroadcastsd  zmm13, qword ptr [rip + .MASK]


.Louter_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      vmovaps  zmm8, [r9 + 128]
      vmovaps  zmm9, [r9 + 192]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 64]
      vpmulld zmm12, zmm6, ZMMWORD PTR [rsp + 128]
      vpmulld zmm14, zmm7, ZMMWORD PTR [rsp + 64]
      vpmulld zmm15, zmm7, ZMMWORD PTR [rsp + 128]
      vpmulld zmm16, zmm8, ZMMWORD PTR [rsp + 64]
      vpmulld zmm17, zmm8, ZMMWORD PTR [rsp + 128]
      vpmulld zmm18, zmm9, ZMMWORD PTR [rsp + 64]
      vpmulld zmm19, zmm9, ZMMWORD PTR [rsp + 128]
      add r9, 256

.Linner_loop:
      vmovaps zmm7, [r9 + 0]
      vpslld zmm6, zmm7, 4
      vpandd zmm6, zmm6, zmm13
      vpandd zmm7, zmm7, zmm13
      vmovaps zmm9, [r9 + 64]
      vpslld zmm8, zmm9, 4
      vpandd zmm8, zmm8, zmm13
      vpandd zmm9, zmm9, zmm13
      add r9, 128
      vpbroadcastd zmm2, [rcx + r11]
      vpdpbusd  zmm5, zmm2, zmm6
      vpdpbusd  zmm14, zmm2, zmm7
      vpdpbusd  zmm16, zmm2, zmm8
      vpdpbusd  zmm18, zmm2, zmm9
      vpbroadcastd zmm2, [rax + r11]
      vpdpbusd  zmm12, zmm2, zmm6
      vpdpbusd  zmm15, zmm2, zmm7
      vpdpbusd  zmm17, zmm2, zmm8
      vpdpbusd  zmm19, zmm2, zmm9

      add r11, 4
      cmp rdx, r11
      jne .Linner_loop

.Linner_loop_end:

      # Convert from int32 to float.
      vpsrad zmm5, zmm5, 4
      vcvtdq2ps zmm5, zmm5
      vpsrad zmm12, zmm12, 4
      vcvtdq2ps zmm12, zmm12
      vpsrad zmm14, zmm14, 4
      vcvtdq2ps zmm14, zmm14
      vpsrad zmm15, zmm15, 4
      vcvtdq2ps zmm15, zmm15
      vpsrad zmm16, zmm16, 4
      vcvtdq2ps zmm16, zmm16
      vpsrad zmm17, zmm17, 4
      vcvtdq2ps zmm17, zmm17
      vpsrad zmm18, zmm18, 4
      vcvtdq2ps zmm18, zmm18
      vpsrad zmm19, zmm19, 4
      vcvtdq2ps zmm19, zmm19
      # Load quantization_params pointer from stack
      mov r11, [rsp + 200]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm12, zmm12, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm14, zmm14, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm15, zmm15, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm16, zmm16, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm17, zmm17, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm18, zmm18, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm19, zmm19, DWORD PTR [r11 + 12]{1to16}
      vmovaps zmm10, [r9 + 0]
      vmovaps zmm11, [r9 + 64]
      vmovaps zmm2, [r9 + 128]
      vmovaps zmm3, [r9 + 192]
      add r9, 256
      vmovaps zmm6, [r9 + 0]
      vmovaps zmm7, [r9 + 64]
      vmovaps zmm8, [r9 + 128]
      vmovaps zmm9, [r9 + 192]
      add r9, 256
      vfmadd213ps zmm5, zmm10, zmm6
      vfmadd213ps zmm12, zmm10, zmm6
      vfmadd213ps zmm14, zmm11, zmm7
      vfmadd213ps zmm15, zmm11, zmm7
      vfmadd213ps zmm16, zmm2, zmm8
      vfmadd213ps zmm17, zmm2, zmm8
      vfmadd213ps zmm18, zmm3, zmm9
      vfmadd213ps zmm19, zmm3, zmm9
      # Min/max clamping.
      vminps  zmm5, zmm1, zmm5
      vminps  zmm16, zmm1, zmm16
      vminps  zmm12, zmm1, zmm12
      vminps  zmm17, zmm1, zmm17
      vminps  zmm14, zmm1, zmm14
      vminps  zmm18, zmm1, zmm18
      vminps  zmm15, zmm1, zmm15
      vminps  zmm19, zmm1, zmm19
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm19, zmm0, zmm19

      # Check whether full or partial store.
      cmp rsi, 64
      jl .Ltail

      vmovups  [r10], zmm5
      vmovups  [r10 + 64], zmm14
      vmovups  [r10 + 128], zmm16
      vmovups  [r10 + 192], zmm18
      vmovups  [r13], zmm12
      vmovups  [r13 + 64], zmm15
      vmovups  [r13 + 128], zmm17
      vmovups  [r13 + 192], zmm19
      add r10, 256
      add r13, 256

      sub rsi, 64
      jne .Louter_loop
      jmp .Lreturn

.Ltail:
      mov r11, -1
      shlx r11, r11, rsi
      not r11
      kmovw k1, r11d
      shr r11, 16
      kmovw k2, r11d
      shr r11, 16
      kmovw k3, r11d
      shr r11, 16
      kmovw k4, r11d

      vmovups  ZMMWORD PTR [r10]{k1}, zmm5
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm14
      vmovups  ZMMWORD PTR [r10 + 128]{k3}, zmm16
      vmovups  ZMMWORD PTR [r10 + 192]{k4}, zmm18
      vmovups  ZMMWORD PTR [r13]{k1}, zmm12
      vmovups  ZMMWORD PTR [r13 + 64]{k2}, zmm15
      vmovups  ZMMWORD PTR [r13 + 128]{k3}, zmm17
      vmovups  ZMMWORD PTR [r13 + 192]{k4}, zmm19

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
END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni

      #if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x64c4__asm_amd64_avx512vnni.dfsan
      #endif

      #ifdef __ELF__
      .section .note.GNU-stack, "", @progbits
      #endif  // __ELF__