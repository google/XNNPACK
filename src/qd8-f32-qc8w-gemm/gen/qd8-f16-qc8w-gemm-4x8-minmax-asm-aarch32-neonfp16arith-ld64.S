// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8__asm_aarch32_neonfp16arith_ld64_2

      # Free up GP registers. Decrement sp by 36.
      push {r4, r5, r6, r7, r8, r9, r10, r11, r14}

      # Preserve callee saved q4-q7 registers. Decrement sp by 64.
      vpush {d8-d15}

      # Load weight's ptr.
      ldr r5, [sp, #104]

      # Load c ptr.
      ldr r6, [sp, #108]

      # Load params.
      ldr r4, [sp, #124]

      # Load min/max values.
      vld1.8 {q8, q9},  [r4]

      # Load quantization params
      ldr r7, [sp, #124]
      # Load minmax pointer.
      ldr r11, [sp, #120]
      # Load dynamic quantization params.
      vld1.32 {q4, q5}, [r7]
      # Setup and alias a & c pointers.
      # Load a and cm stride registers.
      ldr r4, [sp, #100]
      ldr r12, [sp, #112]
      add r7, r3, r4
      add r9, r7, r4
      add r10, r9, r4
      add r4, r6, r12
      add r8, r4, r12
      add r14, r8, r12

      cmp r0, #2
      movlo  r7, r3
      movlo  r4, r6
      movls  r9, r7
      movls  r8, r4

      cmp r0, #4
      movlo  r10, r9
      movlo  r14, r8

.Louter_loop:
      # Initialize k counter.
      subs r0, r2, #8
      vld1.32 {q6, q7}, [r5]!
      # Initialize accumulators with k_sum * input zero point.
      vmul.s32 q8, q6, d8[0]
      vmul.s32 q10, q6, d9[0]
      vmul.s32 q12, q6, d10[0]
      vmul.s32 q14, q6, d11[0]
      vmul.s32 q9, q7, d8[0]
      vmul.s32 q11, q7, d9[0]
      vmul.s32 q13, q7, d10[0]
      vmul.s32 q15, q7, d11[0]

      # jump to epilogue if lower than 8
      blo .Lepilogue

      # Load 4 As and B0
      vld1.8 d12, [r5]!
      vld1.8 d0, [r3]!
      vld1.8 d2, [r7]!
      vld1.8 d4, [r9]!
      vld1.8 d6, [r10]!

      # Are there at least 8 bytes?
      subs r0, r0, #8
      blo .Lfinal_iteration

.Linner_loop:
      vmovl.s8 q6, d12
      vmovl.s8 q0, d0
      vmovl.s8 q1, d2
      vmovl.s8 q2, d4
      vmovl.s8 q3, d6
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q10, d12, d2[0]
      vmlal.s16  q12, d12, d4[0]
      vmlal.s16  q14, d12, d6[0]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q11, d13, d2[0]
      vmlal.s16  q13, d13, d4[0]
      vmlal.s16  q15, d13, d6[0]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d0[1]
      vmlal.s16  q10, d14, d2[1]
      vmlal.s16  q12, d14, d4[1]
      vmlal.s16  q14, d14, d6[1]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d0[1]
      vmlal.s16  q11, d15, d2[1]
      vmlal.s16  q13, d15, d4[1]
      vmlal.s16  q15, d15, d6[1]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d0[2]
      vmlal.s16  q10, d12, d2[2]
      vmlal.s16  q12, d12, d4[2]
      vmlal.s16  q14, d12, d6[2]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d0[2]
      vmlal.s16  q11, d13, d2[2]
      vmlal.s16  q13, d13, d4[2]
      vmlal.s16  q15, d13, d6[2]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d0[3]
      vmlal.s16  q10, d14, d2[3]
      vmlal.s16  q12, d14, d4[3]
      vmlal.s16  q14, d14, d6[3]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d0[3]
      vmlal.s16  q11, d15, d2[3]
      vmlal.s16  q13, d15, d4[3]
      vmlal.s16  q15, d15, d6[3]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q10, d12, d3[0]
      vmlal.s16  q12, d12, d5[0]
      vmlal.s16  q14, d12, d7[0]
      vmovl.s8 q7, d14
      vld1.8 d0, [r3]!
      vmlal.s16  q9, d13, d1[0]
      vmlal.s16  q11, d13, d3[0]
      vmlal.s16  q13, d13, d5[0]
      vmlal.s16  q15, d13, d7[0]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d1[1]
      vmlal.s16  q10, d14, d3[1]
      vmlal.s16  q12, d14, d5[1]
      vmlal.s16  q14, d14, d7[1]
      vmovl.s8 q6, d12
      vld1.8 d2, [r7]!
      vmlal.s16  q9, d15, d1[1]
      vmlal.s16  q11, d15, d3[1]
      vmlal.s16  q13, d15, d5[1]
      vmlal.s16  q15, d15, d7[1]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d1[2]
      vmlal.s16  q10, d12, d3[2]
      vmlal.s16  q12, d12, d5[2]
      vmlal.s16  q14, d12, d7[2]
      vmovl.s8 q7, d14
      vld1.8 d4, [r9]!
      vmlal.s16  q9, d13, d1[2]
      vmlal.s16  q11, d13, d3[2]
      vmlal.s16  q13, d13, d5[2]
      vmlal.s16  q15, d13, d7[2]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d1[3]
      vmlal.s16  q10, d14, d3[3]
      vmlal.s16  q12, d14, d5[3]
      vmlal.s16  q14, d14, d7[3]
      vld1.8 d6, [r10]!
      vmlal.s16  q9, d15, d1[3]
      vmlal.s16  q11, d15, d3[3]
      vmlal.s16  q13, d15, d5[3]
      vmlal.s16  q15, d15, d7[3]
      subs r0, r0, #8
      bhs .Linner_loop

.Lfinal_iteration:
      vmovl.s8 q6, d12
      vmovl.s8 q0, d0
      vmovl.s8 q1, d2
      vmovl.s8 q2, d4
      vmovl.s8 q3, d6
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q10, d12, d2[0]
      vmlal.s16  q12, d12, d4[0]
      vmlal.s16  q14, d12, d6[0]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q11, d13, d2[0]
      vmlal.s16  q13, d13, d4[0]
      vmlal.s16  q15, d13, d6[0]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d0[1]
      vmlal.s16  q10, d14, d2[1]
      vmlal.s16  q12, d14, d4[1]
      vmlal.s16  q14, d14, d6[1]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d0[1]
      vmlal.s16  q11, d15, d2[1]
      vmlal.s16  q13, d15, d4[1]
      vmlal.s16  q15, d15, d6[1]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d0[2]
      vmlal.s16  q10, d12, d2[2]
      vmlal.s16  q12, d12, d4[2]
      vmlal.s16  q14, d12, d6[2]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d0[2]
      vmlal.s16  q11, d13, d2[2]
      vmlal.s16  q13, d13, d4[2]
      vmlal.s16  q15, d13, d6[2]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d0[3]
      vmlal.s16  q10, d14, d2[3]
      vmlal.s16  q12, d14, d4[3]
      vmlal.s16  q14, d14, d6[3]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d0[3]
      vmlal.s16  q11, d15, d2[3]
      vmlal.s16  q13, d15, d4[3]
      vmlal.s16  q15, d15, d6[3]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q10, d12, d3[0]
      vmlal.s16  q12, d12, d5[0]
      vmlal.s16  q14, d12, d7[0]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d1[0]
      vmlal.s16  q11, d13, d3[0]
      vmlal.s16  q13, d13, d5[0]
      vmlal.s16  q15, d13, d7[0]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d1[1]
      vmlal.s16  q10, d14, d3[1]
      vmlal.s16  q12, d14, d5[1]
      vmlal.s16  q14, d14, d7[1]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d1[1]
      vmlal.s16  q11, d15, d3[1]
      vmlal.s16  q13, d15, d5[1]
      vmlal.s16  q15, d15, d7[1]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d1[2]
      vmlal.s16  q10, d12, d3[2]
      vmlal.s16  q12, d12, d5[2]
      vmlal.s16  q14, d12, d7[2]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d1[2]
      vmlal.s16  q11, d13, d3[2]
      vmlal.s16  q13, d13, d5[2]
      vmlal.s16  q15, d13, d7[2]
      vmlal.s16  q8, d14, d1[3]
      vmlal.s16  q10, d14, d3[3]
      vmlal.s16  q12, d14, d5[3]
      vmlal.s16  q14, d14, d7[3]
      vmlal.s16  q9, d15, d1[3]
      vmlal.s16  q11, d15, d3[3]
      vmlal.s16  q13, d15, d5[3]
      vmlal.s16  q15, d15, d7[3]
      adds r0, r0, #8
      bne .Lepilogue


.Linner_loop_end:

      # Convert from int32 to float.
      vcvt.f32.s32 q8, q8
      vcvt.f32.s32 q9, q9
      vcvt.f32.s32 q10, q10
      vcvt.f32.s32 q11, q11
      vcvt.f32.s32 q12, q12
      vcvt.f32.s32 q13, q13
      vcvt.f32.s32 q14, q14
      vcvt.f32.s32 q15, q15
      # Multiply by input scale.
      vmul.f32 q8, q8, d8[1]
      vmul.f32 q10, q10, d9[1]
      vmul.f32 q12, q12, d10[1]
      vmul.f32 q14, q14, d11[1]
      vmul.f32 q9, q9, d8[1]
      vmul.f32 q11, q11, d9[1]
      vmul.f32 q13, q13, d10[1]
      vmul.f32 q15, q15, d11[1]
      # Load weights scale.
      vld1.32 {d0, d1}, [r5]!
      vld1.32 {d2, d3}, [r5]!
      # Load biases.
      vld1.32 {d12, d13}, [r5]!
      vld1.32 {d14, d15}, [r5]!
      # Multiply by weight's scale.
      vmul.f32 q8, q8, q0
      vmul.f32 q10, q10, q0
      vmul.f32 q12, q12, q0
      vmul.f32 q14, q14, q0
      vmul.f32 q9, q9, q1
      vmul.f32 q11, q11, q1
      vmul.f32 q13, q13, q1
      vmul.f32 q15, q15, q1
      # Load min/max into registers.
      vld1.32 {d2[0]}, [r11]
      vdup.16 d0, d2[0]
      vdup.16 d2, d2[1]
      # Add bias.
      vadd.f32 q8, q8, q6
      vadd.f32 q10, q10, q6
      vadd.f32 q12, q12, q6
      vadd.f32 q14, q14, q6
      vadd.f32 q9, q9, q7
      vadd.f32 q11, q11, q7
      vadd.f32 q13, q13, q7
      vadd.f32 q15, q15, q7
      # Min/max clamping.
      vcvt.f16.f32 d16, q8
      vmin.f16 d16, d16, d2
      vcvt.f16.f32 d20, q10
      vmin.f16 d20, d20, d2
      vcvt.f16.f32 d24, q12
      vmin.f16 d24, d24, d2
      vcvt.f16.f32 d28, q14
      vmin.f16 d28, d28, d2
      vcvt.f16.f32 d18, q9
      vmin.f16 d18, d18, d2
      vcvt.f16.f32 d22, q11
      vmin.f16 d22, d22, d2
      vcvt.f16.f32 d26, q13
      vmin.f16 d26, d26, d2
      vcvt.f16.f32 d30, q15
      vmin.f16 d30, d30, d2
      vmax.f16 d16, d16, d0
      vmax.f16 d20, d20, d0
      vmax.f16 d24, d24, d0
      vmax.f16 d28, d28, d0
      vmax.f16 d18, d18, d0
      vmax.f16 d22, d22, d0
      vmax.f16 d26, d26, d0
      vmax.f16 d30, d30, d0

      # Check whether full or partial store.
      cmp r1, #8
      blo .Ltail_4
      vst1.16  d16, [r6]!
      vst1.16  d18, [r6]!
      vst1.16  d20, [r4]!
      vst1.16  d22, [r4]!
      vst1.16  d24, [r8]!
      vst1.16  d26, [r8]!
      vst1.16  d28, [r14]!
      vst1.16  d30, [r14]!
      sub r3, r3, r2
      sub r7, r7, r2
      sub r9, r9, r2
      sub r10, r10, r2

      sub r1, r1, #8
      bne .Louter_loop
      b .Lreturn


.Ltail_4:
      tst r1, #4
      beq .Ltail_2
      vst1.16  {d16}, [r6]!
      vst1.16  {d20}, [r4]!
      vst1.16  {d24}, [r8]!
      vst1.16  {d28}, [r14]!
      vmov  d16, d18
      vmov  d20, d22
      vmov  d24, d26
      vmov  d28, d30


.Ltail_2:
      tst r1, #2
      beq .Ltail_1
      vst1.32  {d16[0]}, [r6]!
      vst1.32  {d20[0]}, [r4]!
      vst1.32  {d24[0]}, [r8]!
      vst1.32  {d28[0]}, [r14]!
      vext.8 d16, d16, d17, #4
      vext.8 d20, d20, d21, #4
      vext.8 d24, d24, d25, #4
      vext.8 d28, d28, d29, #4


.Ltail_1:
      tst r1, #1
      beq .Lreturn
      vst1.16  {d16[0]}, [r6]
      vst1.16  {d20[0]}, [r4]
      vst1.16  {d24[0]}, [r8]
      vst1.16  {d28[0]}, [r14]

.Lreturn:
      # Restore callee saved q4-q7 registers.
      vpop       {d8-d15}

      # Restore the callee saved GP registers.
      pop {r4, r5, r6, r7, r8, r9, r10, r11, r14}

      bx lr

.Lepilogue:
      and r0, r0, #7

      # Load 4 As and B0
      vld1.8 d0, [r3]
      add r3, r0
      vld1.8 d2, [r7]
      add r7, r0
      vld1.8 d4, [r9]
      add r9, r0
      vld1.8 d6, [r10]
      add r10, r0
      vmovl.s8 q0, d0
      vmovl.s8 q1, d2
      vmovl.s8 q2, d4
      vmovl.s8 q3, d6
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q10, d12, d2[0]
      vmlal.s16  q12, d12, d4[0]
      vmlal.s16  q14, d12, d6[0]
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q11, d13, d2[0]
      vmlal.s16  q13, d13, d4[0]
      vmlal.s16  q15, d13, d6[0]
      cmp r0, #2
      blo .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d0[1]
      vmlal.s16  q10, d12, d2[1]
      vmlal.s16  q12, d12, d4[1]
      vmlal.s16  q14, d12, d6[1]
      vmlal.s16  q9, d13, d0[1]
      vmlal.s16  q11, d13, d2[1]
      vmlal.s16  q13, d13, d4[1]
      vmlal.s16  q15, d13, d6[1]
      beq .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d0[2]
      vmlal.s16  q10, d12, d2[2]
      vmlal.s16  q12, d12, d4[2]
      vmlal.s16  q14, d12, d6[2]
      vmlal.s16  q9, d13, d0[2]
      vmlal.s16  q11, d13, d2[2]
      vmlal.s16  q13, d13, d4[2]
      vmlal.s16  q15, d13, d6[2]
      cmp r0, #4
      blo .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d0[3]
      vmlal.s16  q10, d12, d2[3]
      vmlal.s16  q12, d12, d4[3]
      vmlal.s16  q14, d12, d6[3]
      vmlal.s16  q9, d13, d0[3]
      vmlal.s16  q11, d13, d2[3]
      vmlal.s16  q13, d13, d4[3]
      vmlal.s16  q15, d13, d6[3]
      beq .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q10, d12, d3[0]
      vmlal.s16  q12, d12, d5[0]
      vmlal.s16  q14, d12, d7[0]
      vmlal.s16  q9, d13, d1[0]
      vmlal.s16  q11, d13, d3[0]
      vmlal.s16  q13, d13, d5[0]
      vmlal.s16  q15, d13, d7[0]
      cmp r0, #6
      blo .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d1[1]
      vmlal.s16  q10, d12, d3[1]
      vmlal.s16  q12, d12, d5[1]
      vmlal.s16  q14, d12, d7[1]
      vmlal.s16  q9, d13, d1[1]
      vmlal.s16  q11, d13, d3[1]
      vmlal.s16  q13, d13, d5[1]
      vmlal.s16  q15, d13, d7[1]
      beq .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d1[2]
      vmlal.s16  q10, d12, d3[2]
      vmlal.s16  q12, d12, d5[2]
      vmlal.s16  q14, d12, d7[2]
      vmlal.s16  q9, d13, d1[2]
      vmlal.s16  q11, d13, d3[2]
      vmlal.s16  q13, d13, d5[2]
      vmlal.s16  q15, d13, d7[2]
      b .Linner_loop_end

END_FUNCTION xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8__asm_aarch32_neonfp16arith_ld64_2