// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__asm_aarch32_neonmlal_ld64_2

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

      # Free up GP registers. Decrement sp by 20.
      push {r4, r5, r6, r7, r11}

      # Preserve callee saved q4-q7 registers. Decrement sp by 64.
      vpush {d8-d15}

      # Load weight's ptr.
      ldr r5, [sp, #88]  /* (const void*)w */

      # Load c ptr.
      ldr r6, [sp, #92]  /* (float*)c */

      # Load quantization params
      ldr r7, [sp, #108]  /* (void *)quantization_params */

      # Load minmax pointer.
      ldr r11, [sp, #104]  /* (void *)minmax_params */

      # Load dynamic quantization params.
      vld1.32 {d8, d9}, [r7]

      # Setup and alias a & c pointers.
      # Load a and cm stride registers.
      ldr r4, [sp, #84]  /* (size_t)a_stride */
      ldr r12, [sp, #96]  /* (size_t)cm_stride */
      add r7, r3, r4  /* a1 = a + a_stride */
      add r4, r6, r12  /* c1 = c + cm_stride */

      cmp r0, #2
      movlo  r7, r3  /* if (mr < 2) a1 = a */
      movlo  r4, r6  /* if (mr < 2) c1 = c */

.Louter_loop:
      # Initialize k counter.
      subs r0, r2, #8  /* r0 = k - 8 */
      vld1.32 {q6, q7}, [r5]!  /* load 2*4*32 bits from w */
      vld1.32 {q2, q3}, [r5]!  /* load 2*4*32 bits from w */

      # Initialize accumulators with k_sum * input zero point.
      vmul.s32 q8, q6, d8[0]  /* acc_a0_w0123 */
      vmul.s32 q9, q7, d8[0]  /* acc_a0_w4567 */
      vmul.s32 q10, q2, d8[0]  /* acc_a0_w89AB */
      vmul.s32 q11, q3, d8[0]  /* acc_a0_wCDEF */
      vmul.s32 q12, q6, d9[0]  /* acc_a1_w0123 */
      vmul.s32 q13, q7, d9[0]  /* acc_a1_w4567 */
      vmul.s32 q14, q2, d9[0]  /* acc_a1_w89AB */
      vmul.s32 q15, q3, d9[0]  /* acc_a1_wCDEF */

      # jump to epilogue if lower than 8
      blo .Lepilogue

      # (Pre-)load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # Load 2 As and B0
      vld1.8 d0, [r3]!   /* load 8 bytes of inputs from a0. */
      vld1.8 d2, [r7]!   /* load 8 bytes of inputs from a1. */

      # Are there at least 8 ks?
      subs r0, r0, #8
      blo .Lfinal_iteration

.Linner_loop:
      # Expand the 8-bit a0/a1 values into q0/q1 as 16-bit values.
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */
      vmovl.s8 q1, d2  /* expand a1 bytes into q1 int16 */

      # k = 0.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k1 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k0 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k0 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k0 with w_n*_k0 into acc_m0_n*.
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q10, d14, d0[0]
      vmlal.s16  q11, d15, d0[0]

      # Multiply a_m1_k0 with w_n*_k0 into acc_m1_n*.
      vmlal.s16  q12, d12, d2[0]
      vmlal.s16  q13, d13, d2[0]
      vmlal.s16  q14, d14, d2[0]
      vmlal.s16  q15, d15, d2[0]

      # (Pre-)load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # k = 1.

      # Expand the 8-bit weights for k1 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k1 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k1 with w_n*_k1 into acc_m0_n*.
      vmlal.s16  q8, d4, d0[1]
      vmlal.s16  q9, d5, d0[1]
      vmlal.s16  q10, d6, d0[1]
      vmlal.s16  q11, d7, d0[1]

      # Multiply a_m1_k1 with w_n*_k1 into acc_m1_n*.
      vmlal.s16  q12, d4, d2[1]
      vmlal.s16  q13, d5, d2[1]
      vmlal.s16  q14, d6, d2[1]
      vmlal.s16  q15, d7, d2[1]

      # k = 2.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k3 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k2 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k2 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k2 with w_n*_k2 into acc_m0_n*.
      vmlal.s16  q8, d12, d0[2]
      vmlal.s16  q9, d13, d0[2]
      vmlal.s16  q10, d14, d0[2]
      vmlal.s16  q11, d15, d0[2]

      # Multiply a_m1_k2 with w_n*_k2 into acc_m1_n*.
      vmlal.s16  q12, d12, d2[2]
      vmlal.s16  q13, d13, d2[2]
      vmlal.s16  q14, d14, d2[2]
      vmlal.s16  q15, d15, d2[2]

      # (Pre-)load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # k = 3.

      # Expand the 8-bit weights for k3 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k3 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k3 with w_n*_k3 into acc_m0_n*.
      vmlal.s16  q8, d4, d0[3]
      vmlal.s16  q9, d5, d0[3]
      vmlal.s16  q10, d6, d0[3]
      vmlal.s16  q11, d7, d0[3]

      # Multiply a_m1_k3 with w_n*_k3 into acc_m1_n*.
      vmlal.s16  q12, d4, d2[3]
      vmlal.s16  q13, d5, d2[3]
      vmlal.s16  q14, d6, d2[3]
      vmlal.s16  q15, d7, d2[3]

      # (Pre-)load the next a0/a1.
      vld1.8 d0, [r3]!   /* load 8 bytes of inputs from a0. */
      vld1.8 d2, [r7]!   /* load 8 bytes of inputs from a1. */

      # k = 4.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k5 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k4 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k4 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k4 with w_n*_k4 into acc_m0_n*.
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q9, d13, d1[0]
      vmlal.s16  q10, d14, d1[0]
      vmlal.s16  q11, d15, d1[0]

      # Multiply a_m1_k4 with w_n*_k4 into acc_m1_n*.
      vmlal.s16  q12, d12, d3[0]
      vmlal.s16  q13, d13, d3[0]
      vmlal.s16  q14, d14, d3[0]
      vmlal.s16  q15, d15, d3[0]

      # (Pre-)load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # k = 5.

      # Expand the 8-bit weights for k5 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k5 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k5 with w_n*_k5 into acc_m0_n*.
      vmlal.s16  q8, d4, d1[1]
      vmlal.s16  q9, d5, d1[1]
      vmlal.s16  q10, d6, d1[1]
      vmlal.s16  q11, d7, d1[1]

      # Multiply a_m1_k5 with w_n*_k5 into acc_m1_n*.
      vmlal.s16  q12, d4, d3[1]
      vmlal.s16  q13, d5, d3[1]
      vmlal.s16  q14, d6, d3[1]
      vmlal.s16  q15, d7, d3[1]

      # k = 6.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k7 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k6 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k6 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k6 with w_n*_k6 into acc_m0_n*.
      vmlal.s16  q8, d12, d1[2]
      vmlal.s16  q9, d13, d1[2]
      vmlal.s16  q10, d14, d1[2]
      vmlal.s16  q11, d15, d1[2]

      # Multiply a_m1_k6 with w_n*_k6 into acc_m1_n*.
      vmlal.s16  q12, d12, d3[2]
      vmlal.s16  q13, d13, d3[2]
      vmlal.s16  q14, d14, d3[2]
      vmlal.s16  q15, d15, d3[2]

      # (Pre-)load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # k = 7.

      # Expand the 8-bit weights for k7 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k7 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k7 with w_n*_k7 into acc_m0_n*.
      vmlal.s16  q8, d4, d1[3]
      vmlal.s16  q9, d5, d1[3]
      vmlal.s16  q10, d6, d1[3]
      vmlal.s16  q11, d7, d1[3]

      # Multiply a_m1_k7 with w_n*_k7 into acc_m1_n*.
      vmlal.s16  q12, d4, d3[3]
      vmlal.s16  q13, d5, d3[3]
      vmlal.s16  q14, d6, d3[3]
      vmlal.s16  q15, d7, d3[3]

      # Decrement ks as jump back to the top of the loop if we have at least 8.
      subs r0, r0, #8
      bhs .Linner_loop

.Lfinal_iteration:
      # Expand the 8-bit a0/a1 values into q0/q1 as 16-bit values.
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */
      vmovl.s8 q1, d2  /* expand a1 bytes into q1 int16 */

      # k = 0.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k1 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k0 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k0 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k0 with w_n*_k0 into acc_m0_n*.
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q10, d14, d0[0]
      vmlal.s16  q11, d15, d0[0]

      # Multiply a_m1_k0 with w_n*_k0 into acc_m1_n*.
      vmlal.s16  q12, d12, d2[0]
      vmlal.s16  q13, d13, d2[0]
      vmlal.s16  q14, d14, d2[0]
      vmlal.s16  q15, d15, d2[0]

      # (Pre-)load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # k = 1.

      # Expand the 8-bit weights for k1 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k1 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k1 with w_n*_k1 into acc_m0_n*.
      vmlal.s16  q8, d4, d0[1]
      vmlal.s16  q9, d5, d0[1]
      vmlal.s16  q10, d6, d0[1]
      vmlal.s16  q11, d7, d0[1]

      # Multiply a_m1_k1 with w_n*_k1 into acc_m1_n*.
      vmlal.s16  q12, d4, d2[1]
      vmlal.s16  q13, d5, d2[1]
      vmlal.s16  q14, d6, d2[1]
      vmlal.s16  q15, d7, d2[1]

      # k = 2.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k3 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k2 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k2 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k2 with w_n*_k2 into acc_m0_n*.
      vmlal.s16  q8, d12, d0[2]
      vmlal.s16  q9, d13, d0[2]
      vmlal.s16  q10, d14, d0[2]
      vmlal.s16  q11, d15, d0[2]

      # Multiply a_m1_k2 with w_n*_k2 into acc_m1_n*.
      vmlal.s16  q12, d12, d2[2]
      vmlal.s16  q13, d13, d2[2]
      vmlal.s16  q14, d14, d2[2]
      vmlal.s16  q15, d15, d2[2]

      # (Pre-)load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # k = 3.

      # Expand the 8-bit weights for k3 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k3 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k3 with w_n*_k3 into acc_m0_n*.
      vmlal.s16  q8, d4, d0[3]
      vmlal.s16  q9, d5, d0[3]
      vmlal.s16  q10, d6, d0[3]
      vmlal.s16  q11, d7, d0[3]

      # Multiply a_m1_k3 with w_n*_k3 into acc_m1_n*.
      vmlal.s16  q12, d4, d2[3]
      vmlal.s16  q13, d5, d2[3]
      vmlal.s16  q14, d6, d2[3]
      vmlal.s16  q15, d7, d2[3]

      # k = 4.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k5 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k4 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k4 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k4 with w_n*_k4 into acc_m0_n*.
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q9, d13, d1[0]
      vmlal.s16  q10, d14, d1[0]
      vmlal.s16  q11, d15, d1[0]

      # Multiply a_m1_k4 with w_n*_k4 into acc_m1_n*.
      vmlal.s16  q12, d12, d3[0]
      vmlal.s16  q13, d13, d3[0]
      vmlal.s16  q14, d14, d3[0]
      vmlal.s16  q15, d15, d3[0]

      # (Pre-)load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # k = 5.

      # Expand the 8-bit weights for k5 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k5 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k5 with w_n*_k5 into acc_m0_n*.
      vmlal.s16  q8, d4, d1[1]
      vmlal.s16  q9, d5, d1[1]
      vmlal.s16  q10, d6, d1[1]
      vmlal.s16  q11, d7, d1[1]

      # Multiply a_m1_k5 with w_n*_k5 into acc_m1_n*.
      vmlal.s16  q12, d4, d3[1]
      vmlal.s16  q13, d5, d3[1]
      vmlal.s16  q14, d6, d3[1]
      vmlal.s16  q15, d7, d3[1]

      # k = 6.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k7 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k6 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k6 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k6 with w_n*_k6 into acc_m0_n*.
      vmlal.s16  q8, d12, d1[2]
      vmlal.s16  q9, d13, d1[2]
      vmlal.s16  q10, d14, d1[2]
      vmlal.s16  q11, d15, d1[2]

      # Multiply a_m1_k6 with w_n*_k6 into acc_m1_n*.
      vmlal.s16  q12, d12, d3[2]
      vmlal.s16  q13, d13, d3[2]
      vmlal.s16  q14, d14, d3[2]
      vmlal.s16  q15, d15, d3[2]

      # Don't (pre-)load next 16 bytes of weights (two ks worth).

      # k = 7.

      # Expand the 8-bit weights for k7 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k7 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k7 with w_n*_k7 into acc_m0_n*.
      vmlal.s16  q8, d4, d1[3]
      vmlal.s16  q9, d5, d1[3]
      vmlal.s16  q10, d6, d1[3]
      vmlal.s16  q11, d7, d1[3]

      # Multiply a_m1_k7 with w_n*_k7 into acc_m1_n*.
      vmlal.s16  q12, d4, d3[3]
      vmlal.s16  q13, d5, d3[3]
      vmlal.s16  q14, d6, d3[3]
      vmlal.s16  q15, d7, d3[3]

      # Jump to the epilogue if there are leftover ks.
      adds r0, r0, #8
      bne .Lepilogue

.Linner_loop_end:

      # Convert from int32 to float.
      vcvt.f32.s32 q8, q8, #4
      vcvt.f32.s32 q9, q9, #4
      vcvt.f32.s32 q10, q10, #4
      vcvt.f32.s32 q11, q11, #4
      vcvt.f32.s32 q12, q12, #4
      vcvt.f32.s32 q13, q13, #4
      vcvt.f32.s32 q14, q14, #4
      vcvt.f32.s32 q15, q15, #4

      # Load weights scale.
      vld1.32 {q0, q1}, [r5]!
      vld1.32 {q2, q3}, [r5]!

      # Multiply by input scale.
      vmul.f32 q8, q8, d8[1]
      vmul.f32 q9, q9, d8[1]
      vmul.f32 q10, q10, d8[1]
      vmul.f32 q11, q11, d8[1]
      vmul.f32 q12, q12, d9[1]
      vmul.f32 q13, q13, d9[1]
      vmul.f32 q14, q14, d9[1]
      vmul.f32 q15, q15, d9[1]

      # Load biases.
      vld1.32 {d12, d13}, [r5]!
      vld1.32 {d14, d15}, [r5]!

      # Multiply by weight's scale.
      vmul.f32 q8, q8, q0
      vmul.f32 q12, q12, q0

      # Load biases.
      vld1.32 {d0, d1}, [r5]!
      
      vmul.f32 q9, q9, q1
      vmul.f32 q13, q13, q1

      # Load biases.
      vld1.32 {d2, d3}, [r5]!
      
      vmul.f32 q10, q10, q2
      vmul.f32 q14, q14, q2
      vmul.f32 q11, q11, q3
      vmul.f32 q15, q15, q3

      # Load min/max into registers.
      vld1.32 {d4[], d5[]}, [r11]!
      vld1.32 {d6[], d7[]}, [r11]
      sub r11, r11, #4

      # Add bias.
      vadd.f32 q8, q8, q6
      vadd.f32 q12, q12, q6
      vadd.f32 q9, q9, q7
      vadd.f32 q13, q13, q7
      vadd.f32 q10, q10, q0
      vadd.f32 q14, q14, q0
      vadd.f32 q11, q11, q1
      vadd.f32 q15, q15, q1

      # Min/max clamping.
      vmin.f32 q8, q8, q3
      vmin.f32 q10, q10, q3
      vmin.f32 q12, q12, q3
      vmin.f32 q14, q14, q3
      vmin.f32 q9, q9, q3
      vmin.f32 q11, q11, q3
      vmin.f32 q13, q13, q3
      vmin.f32 q15, q15, q3
      vmax.f32 q8, q8, q2
      vmax.f32 q10, q10, q2
      vmax.f32 q12, q12, q2
      vmax.f32 q14, q14, q2
      vmax.f32 q9, q9, q2
      vmax.f32 q11, q11, q2
      vmax.f32 q13, q13, q2
      vmax.f32 q15, q15, q2

      # Check whether full or partial store.
      cmp r1, #16
      blo .Ltail_8
      vst1.32  {q8, q9}, [r6]!
      vst1.32  {q10, q11}, [r6]!
      vst1.32  {q12, q13}, [r4]!
      vst1.32  {q14, q15}, [r4]!
      sub r3, r3, r2
      sub r7, r7, r2

      sub r1, r1, #16
      bne .Louter_loop
      b .Lreturn

      # Check whether full or partial store.
.Ltail_8:
      tst r1, #8
      beq .Ltail_4
      vst1.32  {d16, d17}, [r6]!
      vst1.32  {d18, d19}, [r6]!
      vst1.32  {d24, d25}, [r4]!
      vst1.32  {d26, d27}, [r4]!
      vmov q8, q10
      vmov q9, q11
      vmov q12, q14
      vmov q13, q15

.Ltail_4:
      tst r1, #4
      beq .Ltail_2
      vst1.32  {q8}, [r6]!
      vst1.32  {q12}, [r4]!
      vmov  q8, q9
      vmov  q12, q13


.Ltail_2:
      tst r1, #2
      beq .Ltail_1
      vst1.32  d16, [r6]!
      vst1.32  d24, [r4]!
      vmov d16, d17
      vmov d24, d25


.Ltail_1:
      tst r1, #1
      beq .Lreturn
      vst1.32  {d16[0]}, [r6]
      vst1.32  {d24[0]}, [r4]

.Lreturn:
      # Restore callee saved q4-q7 registers.
      vpop       {d8-d15}

      # Restore the callee saved GP registers.
      pop {r4, r5, r6, r7, r11}

      bx lr

.Lepilogue:
      # Make sure we've only got the last three bits of k.
      and r0, r0, #7

      # Load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # Load 4 As and B0, but only increase the pointers by k.
      vld1.8 d0, [r3]
      add r3, r0
      vld1.8 d2, [r7]
      add r7, r0

      # Expand the 8-bit a0/a1 values into q0/q1 as 16-bit values.
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */
      vmovl.s8 q1, d2  /* expand a1 bytes into q1 int16 */

      # k = 0.

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k1 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k0 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k0 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k0 with w_n*_k0 into acc_m0_n*.
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q10, d14, d0[0]
      vmlal.s16  q11, d15, d0[0]

      # Multiply a_m1_k0 with w_n*_k0 into acc_m1_n*.
      vmlal.s16  q12, d12, d2[0]
      vmlal.s16  q13, d13, d2[0]
      vmlal.s16  q14, d14, d2[0]
      vmlal.s16  q15, d15, d2[0]

      # If k < 2, we're done.
      cmp r0, #2
      blo .Linner_loop_end

      # k = 1.

      # Expand the 8-bit weights for k1 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k1 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k1 with w_n*_k1 into acc_m0_n*.
      vmlal.s16  q8, d4, d0[1]
      vmlal.s16  q9, d5, d0[1]
      vmlal.s16  q10, d6, d0[1]
      vmlal.s16  q11, d7, d0[1]

      # Multiply a_m1_k1 with w_n*_k1 into acc_m1_n*.
      vmlal.s16  q12, d4, d2[1]
      vmlal.s16  q13, d5, d2[1]
      vmlal.s16  q14, d6, d2[1]
      vmlal.s16  q15, d7, d2[1]

      # If k == 2, we're done.
      beq .Linner_loop_end

      # k = 2.

      # Load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k3 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k2 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k2 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k2 with w_n*_k2 into acc_m0_n*.
      vmlal.s16  q8, d12, d0[2]
      vmlal.s16  q9, d13, d0[2]
      vmlal.s16  q10, d14, d0[2]
      vmlal.s16  q11, d15, d0[2]

      # Multiply a_m1_k2 with w_n*_k2 into acc_m1_n*.
      vmlal.s16  q12, d12, d2[2]
      vmlal.s16  q13, d13, d2[2]
      vmlal.s16  q14, d14, d2[2]
      vmlal.s16  q15, d15, d2[2]

      # If k < 4, we're done.
      cmp r0, #4
      blo .Linner_loop_end

      # k = 3.

      # Expand the 8-bit weights for k3 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k3 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k3 with w_n*_k3 into acc_m0_n*.
      vmlal.s16  q8, d4, d0[3]
      vmlal.s16  q9, d5, d0[3]
      vmlal.s16  q10, d6, d0[3]
      vmlal.s16  q11, d7, d0[3]

      # Multiply a_m1_k3 with w_n*_k3 into acc_m1_n*.
      vmlal.s16  q12, d4, d2[3]
      vmlal.s16  q13, d5, d2[3]
      vmlal.s16  q14, d6, d2[3]
      vmlal.s16  q15, d7, d2[3]

      # If k == 4, we're done.
      beq .Linner_loop_end

      # k = 4.

      # Load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k5 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k4 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k4 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k4 with w_n*_k4 into acc_m0_n*.
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q9, d13, d1[0]
      vmlal.s16  q10, d14, d1[0]
      vmlal.s16  q11, d15, d1[0]

      # Multiply a_m1_k4 with w_n*_k4 into acc_m1_n*.
      vmlal.s16  q12, d12, d3[0]
      vmlal.s16  q13, d13, d3[0]
      vmlal.s16  q14, d14, d3[0]
      vmlal.s16  q15, d15, d3[0]

      # If k < 6, we're done.
      cmp r0, #6
      blo .Linner_loop_end

      # k = 5.

      # Expand the 8-bit weights for k5 in q2 into 16-bit values in q2/q3
      vshl.i8 q2, q2, #4  /* k5 values in q2 as signed-extended 8-bit. */
      vmovl.s8 q3, d5  /* expand d5 into q3 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand d4 into q2 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k5 with w_n*_k5 into acc_m0_n*.
      vmlal.s16  q8, d4, d1[1]
      vmlal.s16  q9, d5, d1[1]
      vmlal.s16  q10, d6, d1[1]
      vmlal.s16  q11, d7, d1[1]

      # Multiply a_m1_k5 with w_n*_k5 into acc_m1_n*.
      vmlal.s16  q12, d4, d3[1]
      vmlal.s16  q13, d5, d3[1]
      vmlal.s16  q14, d6, d3[1]
      vmlal.s16  q15, d7, d3[1]

      # If k == 6, we're done.
      beq .Linner_loop_end

      # k = 6.

      # Load next 16 bytes of weights (two ks worth).
      vld1.8 {d12, d13}, [r5]!  /* load 16 bytes of weights (two ks worth) */

      # Expand the 4-bit weights in q6 to 8-bit weights in q6 and q2.
      vshr.s8 q2, q6, #4  /* k7 values in q2 (left-shifted by 4 bits) */
      vshl.i8 q6, q6, #4  /* k6 values in q6 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k6 in q6 into 16-bit values in q6/q7
      vmovl.s8 q7, d13  /* expand d13 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d12 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m0_k6 with w_n*_k6 into acc_m0_n*.
      vmlal.s16  q8, d12, d1[2]
      vmlal.s16  q9, d13, d1[2]
      vmlal.s16  q10, d14, d1[2]
      vmlal.s16  q11, d15, d1[2]

      # Multiply a_m1_k6 with w_n*_k6 into acc_m1_n*.
      vmlal.s16  q12, d12, d3[2]
      vmlal.s16  q13, d13, d3[2]
      vmlal.s16  q14, d14, d3[2]
      vmlal.s16  q15, d15, d3[2]

      # Jump back to the end of the inner loop.
      b .Linner_loop_end

END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__asm_aarch32_neonmlal_ld64_2