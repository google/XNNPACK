#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_11x32c4__asm_amd64_avx512vnni

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
      sub rsp, 1168
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

      # Load quantization params pointer from stack
      mov r11, [rsp + 1256]
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
      mov edi, [r11 + 56]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 912], zmm6
      mov edi, [r11 + 64]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 976], zmm6
      mov edi, [r11 + 72]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 1040], zmm6
      mov edi, [r11 + 80]
      vpbroadcastd zmm6, edi
      vmovups zmmword ptr [rsp + 1104], zmm6

outer_loop:
      # Zero k counter.
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

      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 464]
      vpmulld zmm12, zmm6, ZMMWORD PTR [rsp + 528]
      vpmulld zmm14, zmm6, ZMMWORD PTR [rsp + 592]
      vpmulld zmm15, zmm6, ZMMWORD PTR [rsp + 656]
      vpmulld zmm16, zmm6, ZMMWORD PTR [rsp + 720]
      vpmulld zmm17, zmm6, ZMMWORD PTR [rsp + 784]
      vpmulld zmm18, zmm6, ZMMWORD PTR [rsp + 848]
      vpmulld zmm19, zmm6, ZMMWORD PTR [rsp + 912]
      vpmulld zmm20, zmm6, ZMMWORD PTR [rsp + 976]
      vpmulld zmm21, zmm6, ZMMWORD PTR [rsp + 1040]
      vpmulld zmm22, zmm6, ZMMWORD PTR [rsp + 1104]
      vpmulld zmm23, zmm7, ZMMWORD PTR [rsp + 464]
      vpmulld zmm24, zmm7, ZMMWORD PTR [rsp + 528]
      vpmulld zmm25, zmm7, ZMMWORD PTR [rsp + 592]
      vpmulld zmm26, zmm7, ZMMWORD PTR [rsp + 656]
      vpmulld zmm27, zmm7, ZMMWORD PTR [rsp + 720]
      vpmulld zmm28, zmm7, ZMMWORD PTR [rsp + 784]
      vpmulld zmm29, zmm7, ZMMWORD PTR [rsp + 848]
      vpmulld zmm30, zmm7, ZMMWORD PTR [rsp + 912]
      vpmulld zmm4, zmm7, ZMMWORD PTR [rsp + 976]
      vpmulld zmm8, zmm7, ZMMWORD PTR [rsp + 1040]
      vpmulld zmm9, zmm7, ZMMWORD PTR [rsp + 1104]
      add r9, 128

inner_loop:
      vmovaps  zmm6, [r9 + 0]
      vmovaps  zmm7, [r9 + 64]
      add r9, 128
      vpbroadcastd zmm2, [rsi + r11]
      vpdpbusd  zmm5, zmm2, zmm6
      vpdpbusd  zmm23, zmm2, zmm7
      vpbroadcastd zmm2, [rax + r11]
      vpdpbusd  zmm12, zmm2, zmm6
      vpdpbusd  zmm24, zmm2, zmm7
      vpbroadcastd zmm2, [r15 + r11]
      vpdpbusd  zmm14, zmm2, zmm6
      vpdpbusd  zmm25, zmm2, zmm7
      vpbroadcastd zmm2, [r14 + r11]
      vpdpbusd  zmm15, zmm2, zmm6
      vpdpbusd  zmm26, zmm2, zmm7
      vpbroadcastd zmm2, [r12 + r11]
      vpdpbusd  zmm16, zmm2, zmm6
      vpdpbusd  zmm27, zmm2, zmm7
      vpbroadcastd zmm2, [r10 + r11]
      vpdpbusd  zmm17, zmm2, zmm6
      vpdpbusd  zmm28, zmm2, zmm7
      vpbroadcastd zmm2, [r13 + r11]
      vpdpbusd  zmm18, zmm2, zmm6
      vpdpbusd  zmm29, zmm2, zmm7
      vpbroadcastd zmm2, [rbx + r11]
      vpdpbusd  zmm19, zmm2, zmm6
      vpdpbusd  zmm30, zmm2, zmm7
      vpbroadcastd zmm2, [rbp + r11]
      vpdpbusd  zmm20, zmm2, zmm6
      vpdpbusd  zmm4, zmm2, zmm7
      vpbroadcastd zmm2, [r8 + r11]
      vpdpbusd  zmm21, zmm2, zmm6
      vpdpbusd  zmm8, zmm2, zmm7
      vpbroadcastd zmm2, [rdi + r11]
      vpdpbusd  zmm22, zmm2, zmm6
      vpdpbusd  zmm9, zmm2, zmm7

      add r11, 4
      cmp rdx, r11
      jne inner_loop

      # Convert from int32 to float.
      vcvtdq2ps zmm5, zmm5
      vcvtdq2ps zmm12, zmm12
      vcvtdq2ps zmm14, zmm14
      vcvtdq2ps zmm15, zmm15
      vcvtdq2ps zmm16, zmm16
      vcvtdq2ps zmm17, zmm17
      vcvtdq2ps zmm18, zmm18
      vcvtdq2ps zmm19, zmm19
      vcvtdq2ps zmm20, zmm20
      vcvtdq2ps zmm21, zmm21
      vcvtdq2ps zmm22, zmm22
      vcvtdq2ps zmm23, zmm23
      vcvtdq2ps zmm24, zmm24
      vcvtdq2ps zmm25, zmm25
      vcvtdq2ps zmm26, zmm26
      vcvtdq2ps zmm27, zmm27
      vcvtdq2ps zmm28, zmm28
      vcvtdq2ps zmm29, zmm29
      vcvtdq2ps zmm30, zmm30
      vcvtdq2ps zmm4, zmm4
      vcvtdq2ps zmm8, zmm8
      vcvtdq2ps zmm9, zmm9
      # Load quantization_params pointer from stack
      mov r11, [rsp + 1256]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm12, zmm12, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm14, zmm14, DWORD PTR [r11 + 20]{1to16}
      vmulps zmm15, zmm15, DWORD PTR [r11 + 28]{1to16}
      vmulps zmm16, zmm16, DWORD PTR [r11 + 36]{1to16}
      vmulps zmm17, zmm17, DWORD PTR [r11 + 44]{1to16}
      vmulps zmm18, zmm18, DWORD PTR [r11 + 52]{1to16}
      vmulps zmm19, zmm19, DWORD PTR [r11 + 60]{1to16}
      vmulps zmm20, zmm20, DWORD PTR [r11 + 68]{1to16}
      vmulps zmm21, zmm21, DWORD PTR [r11 + 76]{1to16}
      vmulps zmm22, zmm22, DWORD PTR [r11 + 84]{1to16}
      vmulps zmm23, zmm23, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm24, zmm24, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm25, zmm25, DWORD PTR [r11 + 20]{1to16}
      vmulps zmm26, zmm26, DWORD PTR [r11 + 28]{1to16}
      vmulps zmm27, zmm27, DWORD PTR [r11 + 36]{1to16}
      vmulps zmm28, zmm28, DWORD PTR [r11 + 44]{1to16}
      vmulps zmm29, zmm29, DWORD PTR [r11 + 52]{1to16}
      vmulps zmm30, zmm30, DWORD PTR [r11 + 60]{1to16}
      vmulps zmm4, zmm4, DWORD PTR [r11 + 68]{1to16}
      vmulps zmm8, zmm8, DWORD PTR [r11 + 76]{1to16}
      vmulps zmm9, zmm9, DWORD PTR [r11 + 84]{1to16}
      vmovaps zmm10, [r9 + 0]
      vmovaps zmm11, [r9 + 64]
      add r9, 128
      vmovaps zmm6, [r9 + 0]
      vmovaps zmm7, [r9 + 64]
      add r9, 128
      vfmadd213ps zmm5, zmm10, zmm6
      vfmadd213ps zmm12, zmm10, zmm6
      vfmadd213ps zmm14, zmm10, zmm6
      vfmadd213ps zmm15, zmm10, zmm6
      vfmadd213ps zmm16, zmm10, zmm6
      vfmadd213ps zmm17, zmm10, zmm6
      vfmadd213ps zmm18, zmm10, zmm6
      vfmadd213ps zmm19, zmm10, zmm6
      vfmadd213ps zmm20, zmm10, zmm6
      vfmadd213ps zmm21, zmm10, zmm6
      vfmadd213ps zmm22, zmm10, zmm6
      vfmadd213ps zmm23, zmm11, zmm7
      vfmadd213ps zmm24, zmm11, zmm7
      vfmadd213ps zmm25, zmm11, zmm7
      vfmadd213ps zmm26, zmm11, zmm7
      vfmadd213ps zmm27, zmm11, zmm7
      vfmadd213ps zmm28, zmm11, zmm7
      vfmadd213ps zmm29, zmm11, zmm7
      vfmadd213ps zmm30, zmm11, zmm7
      vfmadd213ps zmm4, zmm11, zmm7
      vfmadd213ps zmm8, zmm11, zmm7
      vfmadd213ps zmm9, zmm11, zmm7
      # Min/max clamping..
      vminps  zmm5, zmm1, zmm5
      vminps  zmm12, zmm1, zmm12
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
      vminps  zmm4, zmm1, zmm4
      vminps  zmm8, zmm1, zmm8
      vminps  zmm9, zmm1, zmm9
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm12, zmm0, zmm12
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
      vmaxps  zmm4, zmm0, zmm4
      vmaxps  zmm8, zmm0, zmm8
      vmaxps  zmm9, zmm0, zmm9

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

      vmovups  [rsi], zmm5
      vmovups  [rsi + 64], zmm23
      vmovups  [rax], zmm12
      vmovups  [rax + 64], zmm24
      vmovups  [r15], zmm14
      vmovups  [r15 + 64], zmm25
      vmovups  [r14], zmm15
      vmovups  [r14 + 64], zmm26
      vmovups  [r12], zmm16
      vmovups  [r12 + 64], zmm27
      vmovups  [r10], zmm17
      vmovups  [r10 + 64], zmm28
      vmovups  [r13], zmm18
      vmovups  [r13 + 64], zmm29
      vmovups  [rbx], zmm19
      vmovups  [rbx + 64], zmm30
      vmovups  [rbp], zmm20
      vmovups  [rbp + 64], zmm4
      vmovups  [r8], zmm21
      vmovups  [r8 + 64], zmm8
      vmovups  [rdi], zmm22
      vmovups  [rdi + 64], zmm9
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
      vmovups  ZMMWORD PTR [rsi]{k1}, zmm5
      vmovups  ZMMWORD PTR [rsi + 64]{k2}, zmm23
      vmovups  ZMMWORD PTR [rax]{k1}, zmm12
      vmovups  ZMMWORD PTR [rax + 64]{k2}, zmm24
      vmovups  ZMMWORD PTR [r15]{k1}, zmm14
      vmovups  ZMMWORD PTR [r15 + 64]{k2}, zmm25
      vmovups  ZMMWORD PTR [r14]{k1}, zmm15
      vmovups  ZMMWORD PTR [r14 + 64]{k2}, zmm26
      vmovups  ZMMWORD PTR [r12]{k1}, zmm16
      vmovups  ZMMWORD PTR [r12 + 64]{k2}, zmm27
      vmovups  ZMMWORD PTR [r10]{k1}, zmm17
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm28
      vmovups  ZMMWORD PTR [r13]{k1}, zmm18
      vmovups  ZMMWORD PTR [r13 + 64]{k2}, zmm29
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm19
      vmovups  ZMMWORD PTR [rbx + 64]{k2}, zmm30
      vmovups  ZMMWORD PTR [rbp]{k1}, zmm20
      vmovups  ZMMWORD PTR [rbp + 64]{k2}, zmm4
      vmovups  ZMMWORD PTR [r8]{k1}, zmm21
      vmovups  ZMMWORD PTR [r8 + 64]{k2}, zmm8
      vmovups  ZMMWORD PTR [rdi]{k1}, zmm22
      vmovups  ZMMWORD PTR [rdi + 64]{k2}, zmm9

return:
      add rsp, 1168

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_11x32c4__asm_amd64_avx512vnni