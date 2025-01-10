#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_7x32__asm_amd64_avx512f_broadcast

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

outer_loop:
      # Initialize k counter.
      mov r11, 0
      # Read a pointers from stack into GP registers.
      mov rsi, [rsp - 128]
      mov rax, [rsp - 144]
      mov r15, [rsp - 160]
      mov r14, [rsp - 176]
      mov r12, [rsp - 192]
      mov r10, [rsp - 208]
      mov r13, [rsp - 224]

      # Initialize accumulators with the biases.
      vmovaps  zmm7, [r9 + 0]
      vmovaps  zmm18, [r9 + 64]
      vmovaps zmm8, zmm7
      vmovaps zmm9, zmm7
      vmovaps zmm14, zmm7
      vmovaps zmm15, zmm7
      vmovaps zmm16, zmm7
      vmovaps zmm17, zmm7
      vmovaps zmm19, zmm18
      vmovaps zmm20, zmm18
      vmovaps zmm21, zmm18
      vmovaps zmm22, zmm18
      vmovaps zmm23, zmm18
      vmovaps zmm24, zmm18
      add r9, 128

inner_loop:
      vmovaps  zmm10, [r9 + 0]
      vmovaps  zmm11, [r9 + 64]
      add r9, 128
      vbroadcastss zmm2, DWORD PTR [rsi + r11]
      vfmadd231ps  zmm7, zmm2, zmm10
      vfmadd231ps  zmm18, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [rax + r11]
      vfmadd231ps  zmm8, zmm2, zmm10
      vfmadd231ps  zmm19, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r15 + r11]
      vfmadd231ps  zmm9, zmm2, zmm10
      vfmadd231ps  zmm20, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r14 + r11]
      vfmadd231ps  zmm14, zmm2, zmm10
      vfmadd231ps  zmm21, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r12 + r11]
      vfmadd231ps  zmm15, zmm2, zmm10
      vfmadd231ps  zmm22, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r10 + r11]
      vfmadd231ps  zmm16, zmm2, zmm10
      vfmadd231ps  zmm23, zmm2, zmm11
      vbroadcastss zmm2, DWORD PTR [r13 + r11]
      vfmadd231ps  zmm17, zmm2, zmm10
      vfmadd231ps  zmm24, zmm2, zmm11

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
      vminps  zmm22, zmm1, zmm22
      vminps  zmm23, zmm1, zmm23
      vminps  zmm24, zmm1, zmm24
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
      vmaxps  zmm22, zmm0, zmm22
      vmaxps  zmm23, zmm0, zmm23
      vmaxps  zmm24, zmm0, zmm24

      # Pop output pointers from the stack.
      mov rsi, [rsp - 136]
      mov rax, [rsp - 152]
      mov r15, [rsp - 168]
      mov r14, [rsp - 184]
      mov r12, [rsp - 200]
      mov r10, [rsp - 216]
      mov r13, [rsp - 232]

      # Check whether full or partial store.
      cmp rcx, 32
      jl tail

      vmovups  [rsi], zmm7
      vmovups  [rsi + 64], zmm18
      vmovups  [rax], zmm8
      vmovups  [rax + 64], zmm19
      vmovups  [r15], zmm9
      vmovups  [r15 + 64], zmm20
      vmovups  [r14], zmm14
      vmovups  [r14 + 64], zmm21
      vmovups  [r12], zmm15
      vmovups  [r12 + 64], zmm22
      vmovups  [r10], zmm16
      vmovups  [r10 + 64], zmm23
      vmovups  [r13], zmm17
      vmovups  [r13 + 64], zmm24
      add rsi, 128
      add rax, 128
      add r15, 128
      add r14, 128
      add r12, 128
      add r10, 128
      add r13, 128

      # Write output pointers to the stack.
      mov [rsp - 136], rsi
      mov [rsp - 152], rax
      mov [rsp - 168], r15
      mov [rsp - 184], r14
      mov [rsp - 200], r12
      mov [rsp - 216], r10
      mov [rsp - 232], r13

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
      vmovups  ZMMWORD PTR [rsi]{k1}, zmm7
      vmovups  ZMMWORD PTR [rsi + 64]{k2}, zmm18
      vmovups  ZMMWORD PTR [rax]{k1}, zmm8
      vmovups  ZMMWORD PTR [rax + 64]{k2}, zmm19
      vmovups  ZMMWORD PTR [r15]{k1}, zmm9
      vmovups  ZMMWORD PTR [r15 + 64]{k2}, zmm20
      vmovups  ZMMWORD PTR [r14]{k1}, zmm14
      vmovups  ZMMWORD PTR [r14 + 64]{k2}, zmm21
      vmovups  ZMMWORD PTR [r12]{k1}, zmm15
      vmovups  ZMMWORD PTR [r12 + 64]{k2}, zmm22
      vmovups  ZMMWORD PTR [r10]{k1}, zmm16
      vmovups  ZMMWORD PTR [r10 + 64]{k2}, zmm23
      vmovups  ZMMWORD PTR [r13]{k1}, zmm17
      vmovups  ZMMWORD PTR [r13 + 64]{k2}, zmm24

return:

      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_7x32__asm_amd64_avx512f_broadcast