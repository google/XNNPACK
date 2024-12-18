#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x32c4__asm_amd64_avx512vnni

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
      sub rsp, 528

      # Load quantization params pointer from stack
      mov r11, [rsp + 616]
      mov edi, [r11 + 0]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 464], zmm6

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 464]
      vpmulld zmm12, zmm7, ZMMWORD PTR [rsp + 464]
      add r9, 128

inner_loop:
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      add r9, 128
      vpbroadcastd zmm2, [rsi + r11]
      vpdpbusd  zmm5, zmm2, zmm6
      vpdpbusd  zmm12, zmm2, zmm7

      add r11, 4
      cmp rdx, r11
      jne inner_loop

      # Convert from int32 to float.
      vcvtdq2ps zmm5, zmm5
      vcvtdq2ps zmm12, zmm12
      # Load quantization_params pointer from stack
      mov r11, [rsp + 616]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm12, zmm12, DWORD PTR [r11 + 4]{1to16}
      vmovaps zmm10, [r9 + 0]
      vmovaps zmm11, [r9 + 64]
      add r9, 128
      vmovaps zmm6, [r9 + 0]
      vmovaps zmm7, [r9 + 64]
      add r9, 128
      vfmadd213ps zmm5, zmm10, zmm6
      vfmadd213ps zmm12, zmm11, zmm7
      # Min/max clamping..
      vminps  zmm5, zmm1, zmm5
      vminps  zmm12, zmm1, zmm12
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm12, zmm0, zmm12

      # Check whether full or partial store.
      cmp rcx, 32
      jl tail

      vmovups  [r10], zmm5
      vmovups  [r10 + 64], zmm12
      add r10, 128

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
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm12

return:
      add rsp, 528

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x32c4__asm_amd64_avx512vnni