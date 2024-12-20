#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__asm_amd64_avx512vnni

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
      sub rsp, 976
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

      # Load quantization params pointer from stack
      mov r11, [rsp + 1064]
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

      # Initialize accumulators with k_sum * input zero point.
      vmovaps  zmm6, [r9 + 0]
      vpmulld zmm5, zmm6, ZMMWORD PTR [rsp + 464]
      vpmulld zmm12, zmm6, ZMMWORD PTR [rsp + 528]
      vpmulld zmm14, zmm6, ZMMWORD PTR [rsp + 592]
      vpmulld zmm15, zmm6, ZMMWORD PTR [rsp + 656]
      vpmulld zmm16, zmm6, ZMMWORD PTR [rsp + 720]
      vpmulld zmm17, zmm6, ZMMWORD PTR [rsp + 784]
      vpmulld zmm18, zmm6, ZMMWORD PTR [rsp + 848]
      vpmulld zmm19, zmm6, ZMMWORD PTR [rsp + 912]
      add r9, 64

inner_loop:
      vmovaps  zmm6, [r9 + 0]
      add r9, 64
      vpbroadcastd zmm2, [rsi + r11]
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
      vpbroadcastd zmm2, [rbx + r11]
      vpdpbusd  zmm19, zmm2, zmm6

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
      # Load quantization_params pointer from stack
      mov r11, [rsp + 1064]
      vmulps zmm5, zmm5, DWORD PTR [r11 + 4]{1to16}
      vmulps zmm12, zmm12, DWORD PTR [r11 + 12]{1to16}
      vmulps zmm14, zmm14, DWORD PTR [r11 + 20]{1to16}
      vmulps zmm15, zmm15, DWORD PTR [r11 + 28]{1to16}
      vmulps zmm16, zmm16, DWORD PTR [r11 + 36]{1to16}
      vmulps zmm17, zmm17, DWORD PTR [r11 + 44]{1to16}
      vmulps zmm18, zmm18, DWORD PTR [r11 + 52]{1to16}
      vmulps zmm19, zmm19, DWORD PTR [r11 + 60]{1to16}
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
      vfmadd213ps zmm19, zmm10, zmm6
      # Min/max clamping..
      vminps  zmm5, zmm1, zmm5
      vminps  zmm12, zmm1, zmm12
      vminps  zmm14, zmm1, zmm14
      vminps  zmm15, zmm1, zmm15
      vminps  zmm16, zmm1, zmm16
      vminps  zmm17, zmm1, zmm17
      vminps  zmm18, zmm1, zmm18
      vminps  zmm19, zmm1, zmm19
      vmaxps  zmm5, zmm0, zmm5
      vmaxps  zmm12, zmm0, zmm12
      vmaxps  zmm14, zmm0, zmm14
      vmaxps  zmm15, zmm0, zmm15
      vmaxps  zmm16, zmm0, zmm16
      vmaxps  zmm17, zmm0, zmm17
      vmaxps  zmm18, zmm0, zmm18
      vmaxps  zmm19, zmm0, zmm19

      # Pop output pointers from the stack.
      mov rsi, [rsp - 136]
      mov rax, [rsp - 152]
      mov r15, [rsp - 168]
      mov r14, [rsp - 184]
      mov r12, [rsp - 200]
      mov r10, [rsp - 216]
      mov r13, [rsp - 232]
      mov rbx, [rsp - 248]

      # Check whether full or partial store.
      cmp rcx, 16
      jl tail

      vmovups  [rsi], zmm5
      vmovups  [rax], zmm12
      vmovups  [r15], zmm14
      vmovups  [r14], zmm15
      vmovups  [r12], zmm16
      vmovups  [r10], zmm17
      vmovups  [r13], zmm18
      vmovups  [rbx], zmm19
      add rsi, 64
      add rax, 64
      add r15, 64
      add r14, 64
      add r12, 64
      add r10, 64
      add r13, 64
      add rbx, 64

      # Write output pointers to the stack.
      mov [rsp - 136], rsi
      mov [rsp - 152], rax
      mov [rsp - 168], r15
      mov [rsp - 184], r14
      mov [rsp - 200], r12
      mov [rsp - 216], r10
      mov [rsp - 232], r13
      mov [rsp - 248], rbx

      sub rcx, 16
      jne outer_loop
      jmp return

tail:
      mov r11d, -1
      sal r11d, cl
      not r11d
      kmovw k1, r11d
      vmovups  ZMMWORD PTR [rsi]{k1}, zmm5
      vmovups  ZMMWORD PTR [rax]{k1}, zmm12
      vmovups  ZMMWORD PTR [r15]{k1}, zmm14
      vmovups  ZMMWORD PTR [r14]{k1}, zmm15
      vmovups  ZMMWORD PTR [r12]{k1}, zmm16
      vmovups  ZMMWORD PTR [r10]{k1}, zmm17
      vmovups  ZMMWORD PTR [r13]{k1}, zmm18
      vmovups  ZMMWORD PTR [rbx]{k1}, zmm19

return:
      add rsp, 976

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__asm_amd64_avx512vnni