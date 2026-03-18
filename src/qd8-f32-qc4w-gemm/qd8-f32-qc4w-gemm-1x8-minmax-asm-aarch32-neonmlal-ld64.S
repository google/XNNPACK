// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2

      # void fn(
      #   size_t mr,        // r0
      #   size_t nr,        // r1
      #   size_t k,         // r2
      #   const int8_t* a,  // r3
      #   size_t a_stride,
      #   const void* w,
      #   float* c,
      #   size_t cm_stride,
      #   size_t cn_stride,
      #   const struct xnn_f32_minmax_params* params,
      #   const struct xnn_qd8_quantization_params* quantization_params
      # )

      # Free up GP registers. Decrement sp by 36.
      push {r4, r5, r6, r7, r8, r9, r10, r11, r14}

      # Preserve callee saved q4-q7 registers. Decrement sp by 64.
      vpush {d8-d15}

      # Load weight's ptr.
      ldr r5, [sp, #104]  /* (const void*)w */

      # Load c ptr.
      ldr r6, [sp, #108]  /* (float*)c */

      # Load quantization params
      ldr r7, [sp, #124]  /* (void *)quantization_params */

      # Load minmax pointer.
      ldr r11, [sp, #120]  /* (void *)minmax_params */

      # Load dynamic quantization params.
      vld1.32 d8, [r7]

      # Setup and alias a & c pointers.
      # Load a and cm stride registers.
      ldr r4, [sp, #100]  /* (size_t)a_stride */
      ldr r12, [sp, #112]  /* (size_t)cm_stride */

.Louter_loop:
      # Initialize k counter.
      subs r0, r2, #8  /* r0 = k - 8 */
      vld1.32 {q6, q7}, [r5]!  /* load 2*4*32 bits from w */

      # Initialize the first set of accumulators with k_sum * input zero point.
      vmul.s32 q8, q6, d8[0]  /* acc_a0_w0123 */
      vmul.s32 q9, q7, d8[0]  /* acc_a0_w4567 */
      vmov.u8 q10, #0  /* acc_a1_w0123 */
      vmov.u8 q11, #0  /* acc_a1_w4567 */
      vmov.u8 q12, #0  /* acc_a2_w0123 */
      vmov.u8 q13, #0  /* acc_a2_w4567 */
      vmov.u8 q14, #0  /* acc_a3_w0123 */
      vmov.u8 q15, #0  /* acc_a3_w4567 */

      # jump to epilogue if lower than 8
      blo .Lepilogue

      # Load a chunk of weights (four ks worth).
      vld1.8 {d12}, [r5]!  /* load 8 bytes of weights (four ks worth) */
      vld1.8 {d4}, [r5]!  /* load 8 bytes of weights (four ks worth) */

      # Load a chunk of inputs (eight ks worth).
      vld1.8 d0, [r3]!   /* load 8 bytes of inputs from a0 (eight ks worth). */

      # Are there at least 8 ks?
      subs r0, r0, #8
      blo .Lfinal_iteration 

.Linner_loop:
      # (Pre-)Load a chunk of weights (two ks worth).
      vld1.8 {d10}, [r5]!  /* load 8 bytes of weights (tw0 ks worth) */

      # Expand the 8-bit a0 values into q0 as 16-bit values.
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */

      # Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13.
      # Expand the 4-bit weights in d4 to 8-bit weights in d4 and d5.
      vshr.s8 d13, d12, #4  /* k1 values in d13 (right-shifted by 4 bits) */
      vshr.s8 d5, d4, #4  /* k3 values in d5 (right-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k0/k1 values in d12/d13 as signed-extended 8-bit. */
      vshl.i8 q2, q2, #4  /* k2/k3 values in d4/d5 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k0/k1 in d12/d13 into 16-bit values in q6/q7
      # Expand the 8-bit weights for k2/k3 in d4/d5 into 16-bit values in q2/q3
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # k = 0.

      # Multiply a_m0_k0 with w_n0123_k0 into acc_m0_n0123.
      vmlal.s16  q8, d12, d0[0]

      # Multiply a_m0_k0 with w_n4567_k0 into acc_m0_n4567.
      vmlal.s16  q9, d13, d0[0]

      # k = 1.

      # Multiply a_m0_k1 with w_n0123_k1 into acc_m1_n0123.
      vmlal.s16  q10, d14, d0[1]

      # Multiply a_m0_k1 with w_n4567_k1 into acc_m1_n4567.
      vmlal.s16  q11, d15, d0[1]

      # (Pre-)Load a chunk of weights (two ks worth).
      vld1.8 {d2}, [r5]!  /* load 8 bytes of weights (four ks worth) */

      # k = 2.

      # Multiply a_m0_k2 with w_n0123_k2 into acc_m0_n0123.
      vmlal.s16  q12, d4, d0[2]

      # Multiply a_m0_k2 with w_n4567_k2 into acc_m0_n4567.
      vmlal.s16  q13, d5, d0[2]

      # k = 3.

      # Multiply a_m0_k3 with w_n0123_k3 into acc_m0_n0123.
      vmlal.s16  q14, d6, d0[3]

      # Multiply a_m0_k3 with w_n4567_k3 into acc_m0_n4567.
      vmlal.s16  q15, d7, d0[3]

      # (Pre-)Load a chunk of inputs (eight ks worth).
      vld1.8 d0, [r3]!   /* load 8 bytes of inputs from a0 (eight ks worth). */

      # (Pre-)Load a chunk of weights (two ks worth).
      vld1.8 {d12}, [r5]!  /* load 8 bytes of weights (two ks worth) */

      # Expand the 4-bit weights in d4 to 8-bit weights in d10 and d11.
      # Expand the 4-bit weights in d2 to 8-bit weights in d2 and d3.
      vshr.s8 d11, d10, #4  /* k5 values in d11 (right-shifted by 4 bits) */
      vshr.s8 d3, d2, #4  /* k7 values in d3 (right-shifted by 4 bits) */
      vshl.i8 q5, q5, #4  /* k4/k5 values in d10/d11 as signed-extended 8-bit. */
      vshl.i8 q1, q1, #4  /* k6/k7 values in d2/d3 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k4/k5 in d10/d11 into 16-bit values in q5/q7
      # Expand the 8-bit weights for k6/k7 in d2/d3 into 16-bit values in q2/q3
      vmovl.s8 q7, d11  /* expand d11 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q3, d3  /* expand d3 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q5, d10  /* expand d10 into q5 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q1, d2  /* expand d2 into q2 (8-bit to 16-bit, sign extended) */

      # k = 4.

      # Multiply a_m0_k4 with w_n0123_k4 into acc_m0_n0123.
      vmlal.s16  q8, d10, d1[0]

      # Multiply a_m0_k4 with w_n4567_k4 into acc_m0_n4567.
      vmlal.s16  q9, d11, d1[0]

      # k = 5.

      # Multiply a_m0_k5 with w_n0123_k5 into acc_m1_n0123.
      vmlal.s16  q10, d14, d1[1]

      # Multiply a_m0_k5 with w_n4567_k5 into acc_m1_n4567.
      vmlal.s16  q11, d15, d1[1]

      # (Pre-)Load a chunk of weights (two ks worth).
      vld1.8 {d4}, [r5]!  /* load 8 bytes of weights (two ks worth) */

      # k = 6.

      # Multiply a_m0_k6 with w_n0123_k6 into acc_m0_n0123.
      vmlal.s16  q12, d2, d1[2]

      # Multiply a_m0_k6 with w_n4567_k6 into acc_m0_n4567.
      vmlal.s16  q13, d3, d1[2]

      # k = 7.

      # Multiply a_m0_k7 with w_n0123_k7 into acc_m0_n0123.
      vmlal.s16  q14, d6, d1[3]

      # Multiply a_m0_k7 with w_n4567_k7 into acc_m0_n4567.
      vmlal.s16  q15, d7, d1[3]

      # Decrement ks as jump back to the top of the loop if we have at least 8.
      subs r0, r0, #8
      bhs .Linner_loop

.Lfinal_iteration:
      # (Pre-)Load a chunk of weights (two ks worth).
      vld1.8 {d10}, [r5]!  /* load 8 bytes of weights (tw0 ks worth) */

      # Expand the 8-bit a0 values into q0 as 16-bit values.
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */

      # Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13.
      # Expand the 4-bit weights in d4 to 8-bit weights in d4 and d5.
      vshr.s8 d13, d12, #4  /* k1 values in d13 (right-shifted by 4 bits) */
      vshr.s8 d5, d4, #4  /* k3 values in d5 (right-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k0/k1 values in d12/d13 as signed-extended 8-bit. */
      vshl.i8 q2, q2, #4  /* k2/k3 values in d4/d5 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k0/k1 in d12/d13 into 16-bit values in q6/q7
      # Expand the 8-bit weights for k2/k3 in d4/d5 into 16-bit values in q2/q3
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # k = 0.

      # Multiply a_m0_k0 with w_n0123_k0 into acc_m0_n0123.
      vmlal.s16  q8, d12, d0[0]

      # Multiply a_m0_k0 with w_n4567_k0 into acc_m0_n4567.
      vmlal.s16  q9, d13, d0[0]

      # k = 1.

      # Multiply a_m0_k1 with w_n0123_k1 into acc_m1_n0123.
      vmlal.s16  q10, d14, d0[1]

      # Multiply a_m0_k1 with w_n4567_k1 into acc_m1_n4567.
      vmlal.s16  q11, d15, d0[1]

      # (Pre-)Load a chunk of weights (two ks worth).
      vld1.8 {d2}, [r5]!  /* load 8 bytes of weights (four ks worth) */

      # k = 2.

      # Multiply a_m0_k2 with w_n0123_k2 into acc_m0_n0123.
      vmlal.s16  q12, d4, d0[2]

      # Multiply a_m0_k2 with w_n4567_k2 into acc_m0_n4567.
      vmlal.s16  q13, d5, d0[2]

      # k = 3.

      # Multiply a_m0_k3 with w_n0123_k3 into acc_m0_n0123.
      vmlal.s16  q14, d6, d0[3]

      # Multiply a_m0_k3 with w_n4567_k3 into acc_m0_n4567.
      vmlal.s16  q15, d7, d0[3]

      # Don't pre-oad a chunk of inputs (eight ks worth).

      # Don't pre-load a chunk of weights (two ks worth).

      # Expand the 4-bit weights in d4 to 8-bit weights in d10 and d11.
      # Expand the 4-bit weights in d2 to 8-bit weights in d2 and d3.
      vshr.s8 d11, d10, #4  /* k5 values in d11 (right-shifted by 4 bits) */
      vshr.s8 d3, d2, #4  /* k7 values in d3 (right-shifted by 4 bits) */
      vshl.i8 q5, q5, #4  /* k4/k5 values in d10/d11 as signed-extended 8-bit. */
      vshl.i8 q1, q1, #4  /* k6/k7 values in d2/d3 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k4/k5 in d10/d11 into 16-bit values in q5/q7
      # Expand the 8-bit weights for k6/k7 in d2/d3 into 16-bit values in q2/q3
      vmovl.s8 q7, d11  /* expand d11 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q3, d3  /* expand d3 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q5, d10  /* expand d10 into q5 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q1, d2  /* expand d2 into q2 (8-bit to 16-bit, sign extended) */

      # k = 4.

      # Multiply a_m0_k4 with w_n0123_k4 into acc_m0_n0123.
      vmlal.s16  q8, d10, d1[0]

      # Multiply a_m0_k4 with w_n4567_k4 into acc_m0_n4567.
      vmlal.s16  q9, d11, d1[0]

      # k = 5.

      # Multiply a_m0_k5 with w_n0123_k5 into acc_m1_n0123.
      vmlal.s16  q10, d14, d1[1]

      # Multiply a_m0_k5 with w_n4567_k5 into acc_m1_n4567.
      vmlal.s16  q11, d15, d1[1]

      # Don't (pre-)load a chunk of weights (two ks worth).

      # k = 6.

      # Multiply a_m0_k6 with w_n0123_k6 into acc_m0_n0123.
      vmlal.s16  q12, d2, d1[2]

      # Multiply a_m0_k6 with w_n4567_k6 into acc_m0_n4567.
      vmlal.s16  q13, d3, d1[2]

      # k = 7.

      # Multiply a_m0_k7 with w_n0123_k7 into acc_m0_n0123.
      vmlal.s16  q14, d6, d1[3]

      # Multiply a_m0_k7 with w_n4567_k7 into acc_m0_n4567.
      vmlal.s16  q15, d7, d1[3]

      # Jump to the epilogue if there are leftover ks.
      adds r0, r0, #8
      bne .Lepilogue

.Linner_loop_end:

      # Accumulate the accumulators into q8/q9.
      vadd.s32 q10, q10, q12
      vadd.s32 q11, q11, q13
      vadd.s32 q8, q8, q14
      vadd.s32 q9, q9, q15
      vadd.s32 q8, q8, q10
      vadd.s32 q9, q9, q11

      # Convert from int32 to float.
      vcvt.f32.s32 q8, q8, #4
      vcvt.f32.s32 q9, q9, #4

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
      sub r7, r7, r2
      sub r9, r9, r2
      sub r10, r10, r2

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
      # Make sure we've only got the last three bits of k.
      and r0, r0, #7

      # Load next 8 bytes of weights (two ks worth).
      vld1.8 d12, [r5]!

      # Load the next chunk of the inputs, but only increase the pointers by k.
      vld1.8 d0, [r3]
      add r3, r0

      # Expand the 8-bit a0 values into q0 as 16-bit values.
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */

      # Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13.
      vshr.s8 d13, d12, #4  /* k0 values in d13 (left-shifted by 4 bits). */
      vshl.i8 q6, q6, #4  /* k1/k0 values in d12/d13 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k0/k1 in d13/d12 into 16-bit values in q6/q7.
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k0 with w_n0123_k0 into acc_m0_n0123.
      vmlal.s16  q8, d12, d0[0]

      # Multiply a_m0_k0 with w_n4567_k0 into acc_m0_n4567.
      vmlal.s16  q9, d13, d0[0]

      # If k < 2, we're done.
      cmp r0, #2
      blo .Linner_loop_end

      # Multiply a_m0_k1 with w_n0123_k1 into acc_m0_n0123.
      vmlal.s16  q8, d14, d0[1]

      # Multiply a_m0_k1 with w_n4567_k1 into acc_m0_n4567.
      vmlal.s16  q9, d15, d0[1]

      # If k == 2, we're done.
      beq .Linner_loop_end

      # Load next 8 bytes of weights (two ks worth),
      vld1.8 d12, [r5]!

      # Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13.
      vshr.s8 d13, d12, #4  /* k2 values in d13 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k3/k2 values in d12/d13 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k2/k3 in d13/d12 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k2 with w_n0123_k2 into acc_m0_n0123.
      vmlal.s16  q8, d12, d0[2]

      # Multiply a_m0_k2 with w_n4567_k2 into acc_m0_n4567.
      vmlal.s16  q9, d13, d0[2]

      # If k < 4, we're done.
      cmp r0, #4
      blo .Linner_loop_end

      # Multiply a_m0_k3 with w_n0123_k3 into acc_m0_n0123.
      vmlal.s16  q8, d14, d0[3]

      # Multiply a_m0_k3 with w_n4567_k3 into acc_m0_n4567.
      vmlal.s16  q9, d15, d0[3]

      # If k == 4, we're done.
      beq .Linner_loop_end

      # Load next 8 bytes of weights (two ks worth),
      vld1.8 d12, [r5]!

      # Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13.
      vshr.s8 d13, d12, #4  /* k4 values in d13 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k5/k4 values in d12/d13 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k4/k5 in d13/d12 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k4 with w_n0123_k4 into acc_m0_n0123.
      vmlal.s16  q8, d12, d1[0]

      # Multiply a_m0_k4 with w_n4567_k4 into acc_m0_n4567.
      vmlal.s16  q9, d13, d1[0]

      # If k < 6, we're done.
      cmp r0, #6
      blo .Linner_loop_end

      # Multiply a_m0_k5 with w_n0123_k5 into acc_m0_n0123.
      vmlal.s16  q8, d14, d1[1]

      # Multiply a_m0_k5 with w_n4567_k5 into acc_m0_n4567.
      vmlal.s16  q9, d15, d1[1]

      # If k == 6, we're done.
      beq .Linner_loop_end

      # Load next 8 bytes of weights (two ks worth),
      vld1.8 d12, [r5]!

      # Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13.
      vshr.s8 d13, d12, #4  /* k6 values in d13 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k7/k6 values in d12/d13 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k6/k7 in d13/d12 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k6 with w_n0123_k6 into acc_m0_n0123.
      vmlal.s16  q8, d12, d1[2]

      # Multiply a_m0_k6 with w_n4567_k6 into acc_m0_n4567.
      vmlal.s16  q9, d13, d1[2]

      # Jump back to the end of the inner loop.
      b .Linner_loop_end

END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch32_neonmlal_ld64_2