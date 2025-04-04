// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2

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

.Louter_loop:
      # Initialize k counter.
      subs r0, r2, #8
      vld1.32 {q6, q7}, [r5]!
      # Initialize accumulators with k_sum * input zero point.
      vmul.s32 q8, q6, d8[0]
      vmul.s32 q9, q7, d8[0]

      # jump to epilogue if lower than 8
      blo .Lepilogue

      # Load 1 As and B0
      vld1.8 d12, [r5]!
      vld1.8 d0, [r3]!

      # Are there at least 8 bytes?
      subs r0, r0, #8
      blo .Lfinal_iteration

.Linner_loop:
      vmovl.s8 q6, d12
      vmovl.s8 q0, d0
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d0[0]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d0[0]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d0[1]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d0[1]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d0[2]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d0[2]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d0[3]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d0[3]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d1[0]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d1[0]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d1[1]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d1[1]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d1[2]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d1[2]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d1[3]
      vld1.8 d0, [r3]!
      vmlal.s16  q9, d15, d1[3]
      subs r0, r0, #8
      bhs .Linner_loop

.Lfinal_iteration:
      vmovl.s8 q6, d12
      vmovl.s8 q0, d0
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d0[0]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d0[0]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d0[1]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d0[1]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d0[2]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d0[2]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d0[3]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d0[3]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d1[0]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d1[0]
      vld1.8 d12, [r5]!
      vmlal.s16  q8, d14, d1[1]
      vmovl.s8 q6, d12
      vmlal.s16  q9, d15, d1[1]
      vld1.8 d14, [r5]!
      vmlal.s16  q8, d12, d1[2]
      vmovl.s8 q7, d14
      vmlal.s16  q9, d13, d1[2]
      vmlal.s16  q8, d14, d1[3]
      vmlal.s16  q9, d15, d1[3]
      adds r0, r0, #8
      bne .Lepilogue

.Linner_loop_end:

      # Convert from int32 to float.
      vcvt.f32.s32 q8, q8
      vcvt.f32.s32 q9, q9
      # Multiply by input scale.
      vmul.f32 q8, q8, d8[1]
      vmul.f32 q9, q9, d8[1]
      # Load weights scale.
      vld1.32 {d0, d1}, [r5]!
      vld1.32 {d2, d3}, [r5]!
      # Load biases.
      vld1.32 {d12, d13}, [r5]!
      vld1.32 {d14, d15}, [r5]!
      # Multiply by weight's scale.
      vmul.f32 q8, q8, q0
      vmul.f32 q9, q9, q1
      # Load min/max into registers.
      vld1.32 {d0[], d1[]}, [r11]!
      vld1.32 {d2[], d3[]}, [r11]
      sub r11, r11, #4
      # Add bias.
      vadd.f32 q8, q8, q6
      vadd.f32 q9, q9, q7
      # Min/max clamping.
      vmin.f32 q8, q8, q1
      vmin.f32 q9, q9, q1
      vmax.f32 q8, q8, q0
      vmax.f32 q9, q9, q0

      # Check whether full or partial store.
      cmp r1, #8
      blo .Ltail_4
      vst1.32  {d16, d17}, [r6]!
      vst1.32  {d18, d19}, [r6]!
      sub r3, r3, r2

      sub r1, r1, #8
      bne .Louter_loop
      b .Lreturn


.Ltail_4:
      tst r1, #4
      beq .Ltail_2
      vst1.32  {q8}, [r6]!
      vmov  q8, q9


.Ltail_2:
      tst r1, #2
      beq .Ltail_1
      vst1.32  d16, [r6]!
      vmov d16, d17


.Ltail_1:
      tst r1, #1
      beq .Lreturn
      vst1.32  {d16[0]}, [r6]

.Lreturn:
      # Restore callee saved q4-q7 registers.
      vpop       {d8-d15}

      # Restore the callee saved GP registers.
      pop {r4, r5, r6, r7, r8, r9, r10, r11, r14}

      bx lr

.Lepilogue:
      and r0, r0, #7

      # Load 1 As and B0
      vld1.8 d0, [r3]
      add r3, r0
      vmovl.s8 q0, d0
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q9, d13, d0[0]
      cmp r0, #2
      blo .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d0[1]
      vmlal.s16  q9, d13, d0[1]
      beq .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d0[2]
      vmlal.s16  q9, d13, d0[2]
      cmp r0, #4
      blo .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d0[3]
      vmlal.s16  q9, d13, d0[3]
      beq .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q9, d13, d1[0]
      cmp r0, #6
      blo .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d1[1]
      vmlal.s16  q9, d13, d1[1]
      beq .Linner_loop_end
      vld1.8 d12, [r5]!
      vmovl.s8 q6, d12
      vmlal.s16  q8, d12, d1[2]
      vmlal.s16  q9, d13, d1[2]
      b .Linner_loop_end

END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2