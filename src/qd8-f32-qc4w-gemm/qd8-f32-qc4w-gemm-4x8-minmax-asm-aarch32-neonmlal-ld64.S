// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch32_neonmlal_ld64_2

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
      vld1.32 {q4, q5}, [r7]

      # Setup and alias a & c pointers.
      # Load a and cm stride registers.
      ldr r4, [sp, #100]  /* (size_t)a_stride */
      ldr r12, [sp, #112]  /* (size_t)cm_stride */
      add r7, r3, r4  /* a1 = a + a_stride */
      add r9, r7, r4  /* a2 = a1 + a_stride */
      add r10, r9, r4  /* a3 = a2 + a_stride */
      add r4, r6, r12  /* c1 = c + cm_stride */
      add r8, r4, r12  /* c2 = c1 + cm_stride */
      add r14, r8, r12  /* c3 = c2 + cm_stride */

      cmp r0, #2
      movlo  r7, r3  /* if (mr < 2) a1 = a */
      movlo  r4, r6  /* if (mr < 2) c1 = c */
      movls  r9, r7  /* if (mr <= 2) a2 = a */
      movls  r8, r4  /* if (mr <= 2) c2 = c */

      cmp r0, #4
      movlo  r10, r9  /* if (mr < 4) a3 = a */
      movlo  r14, r8  /* if (mr < 4) c3 = c */

.Louter_loop:
      # Initialize k counter.
      subs r0, r2, #8  /* r0 = k - 8 */
      vld1.32 {q6, q7}, [r5]!  /* load 2*4*32 bits from w */

      # Initialize accumulators with k_sum * input zero point.
      vmul.s32 q8, q6, d8[0]  /* acc_a0_w0123 */
      vmul.s32 q10, q6, d9[0]  /* acc_a1_w0123 */
      vmul.s32 q12, q6, d10[0]  /* acc_a2_w0123 */
      vmul.s32 q14, q6, d11[0]  /* acc_a3_w0123 */
      vmul.s32 q9, q7, d8[0]  /* acc_a0_w4567 */
      vmul.s32 q11, q7, d9[0] /* acc_a1_w4567 */
      vmul.s32 q13, q7, d10[0]  /* acc_a2_w4567 */
      vmul.s32 q15, q7, d11[0]  /* acc_a3_w4567 */

      # Push the contents of `q5` (dynamic quantization params) onto the stack
      # so that we can re-use these registers in the innter loop.
      vpush {d10-d11}

      # jump to epilogue if lower than 8
      blo .Lepilogue

      # Load 4 As and B0
      vld1.8 d12, [r5]!  /* load 8 bytes of weights (two ks worth) */
      vld1.8 d0, [r3]!   /* load 8 bytes of inputs from a0. */
      vld1.8 d2, [r7]!   /* load 8 bytes of inputs from a1. */
      vld1.8 d4, [r9]!   /* load 8 bytes of inputs from a2. */
      vld1.8 d6, [r10]!   /* load 8 bytes of inputs from a3. */

      # Are there at least 8 ks?
      subs r0, r0, #8
      blo .Lfinal_iteration

.Linner_loop:
      # (Pre-)load next 8 bytes of weights (two ks worth).
      vld1.8 d10, [r5]!

      # The following two blocks of operations are interleaved to allow for
      # better instruction-level parallelism:
      #  1. Expand the 8-bit a0/a1/a2/a3 values into q0/q1/q2/q3 as 16-bit
      #     values.
      #  2. Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13,
      #     Expand the 8-bit weights for k0/k1 in d13/d12 into 16-bit values in
      #     q6/q7.
      vshr.s8 d13, d12, #4  /* k0 values in d13 (left-shifted by 4 bits) */
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */
      vshl.i8 q6, q6, #4  /* k1/k0 values in d12/d13 as signed-extended 8-bit. */
      vmovl.s8 q1, d2  /* expand a1 bytes into q1 int16 */
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand a2 bytes into q2 int16 */
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q3, d6  /* expand a3 bytes into q3 int16 */

      # k = 0.

      # Multiply a_m[0-3]_k0 with w_n0123_k0 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q10, d12, d2[0]
      vmlal.s16  q12, d12, d4[0]
      vmlal.s16  q14, d12, d6[0]

      # Multiply a_m[0-3]_k0 with w_n4567_k0 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q11, d13, d2[0]
      vmlal.s16  q13, d13, d4[0]
      vmlal.s16  q15, d13, d6[0]

      # k = 1.

      # Multiply a_m[0-3]_k1 with w_n0123_k1 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d0[1]
      vmlal.s16  q10, d14, d2[1]
      vmlal.s16  q12, d14, d4[1]
      vmlal.s16  q14, d14, d6[1]

      # The following two blocks of operations are interleaved to allow for
      # better instruction-level parallelism:
      #  1. Multiply a_m[0-3]_k1 with w_n4567_k1 into acc_m[0-3]_n4567.
      #  2. Expand the 4-bit weights in d10 to 8-bit weights in d10 and d11,
      #     Expand the 8-bit weights for k2/k3 in d11/d10 into 16-bit values in
      #     q5/q7.
      vmlal.s16  q9, d15, d0[1]
      vshr.s8 d11, d10, #4  /* k2 values in d11 (left-shifted by 4 bits) */
      vmlal.s16  q11, d15, d2[1]
      vshl.i8 q5, q5, #4  /* k3/k2 values in d10/d11 as signed-extended 8-bit. */
      vmlal.s16  q13, d15, d4[1]
      vmovl.s8 q6, d11  /* expand d11 into q6 (8-bit to 16-bit, sign extended) */
      vmlal.s16  q15, d15, d6[1]
      vmovl.s8 q5, d10  /* expand d10 into q5 (8-bit to 16-bit, sign extended) */

      # k = 3.

      # Multiply a_m[0-3]_k3 with w_n0123_k3 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d0[3]
      vmlal.s16  q10, d12, d2[3]
      vmlal.s16  q12, d12, d4[3]
      vmlal.s16  q14, d12, d6[3]

      # (Pre-)load next 8 bytes of weights (two ks worth).
      vld1.8 d12, [r5]!

      # Multiply a_m[0-3]_k3 with w_n4567_k3 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d13, d0[3]
      vmlal.s16  q11, d13, d2[3]
      vmlal.s16  q13, d13, d4[3]
      vmlal.s16  q15, d13, d6[3]

      # k = 2.

      # Multiply a_m[0-3]_k2 with w_n0123_k2 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d10, d0[2]
      vmlal.s16  q10, d10, d2[2]
      vmlal.s16  q12, d10, d4[2]
      vmlal.s16  q14, d10, d6[2]

      # The following two blocks of operations are interleaved to allow for
      # better instruction-level parallelism:
      #  1. Multiply a_m[0-3]_k2 with w_n4567_k2 into acc_m[0-3]_n4567.
      #  2. Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13,
      #     Expand the 8-bit weights for k4/k5 in d13/d12 into 16-bit values in
      #     q6/q7.
      vmlal.s16  q9, d11, d0[2]
      vshr.s8 d13, d12, #4  /* k4 values in d13 (left-shifted by 4 bits) */
      vmlal.s16  q11, d11, d2[2]
      vshl.i8 q6, q6, #4  /* k5/k4 values in d12/d13 as signed-extended 8-bit. */
      vmlal.s16  q13, d11, d4[2]
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmlal.s16  q15, d11, d6[2]
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */

      # (Pre-)load next 8 bytes of weights (two ks worth).
      vld1.8 d10, [r5]!

      # (Pre-)load the next a0/a1/a2/a3.
      vld1.8 d0, [r3]!   /* load 8 bytes of inputs from a0. */
      vld1.8 d2, [r7]!   /* load 8 bytes of inputs from a1. */
      vld1.8 d4, [r9]!   /* load 8 bytes of inputs from a2. */
      vld1.8 d6, [r10]!   /* load 8 bytes of inputs from a3. */

      # k = 5.

      # Multiply a_m[0-3]_k5 with w_n0123_k5 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d1[1]
      vmlal.s16  q10, d14, d3[1]
      vmlal.s16  q12, d14, d5[1]
      vmlal.s16  q14, d14, d7[1]

      # Multiply a_m[0-3]_k5 with w_n4567_k5 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d15, d1[1]
      vmlal.s16  q11, d15, d3[1]
      vmlal.s16  q13, d15, d5[1]
      vmlal.s16  q15, d15, d7[1]

      # k = 4.

      # Multiply a_m[0-3]_k4 with w_n0123_k4 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q10, d12, d3[0]
      vmlal.s16  q12, d12, d5[0]
      vmlal.s16  q14, d12, d7[0]

      # The following two blocks of operations are interleaved to allow for
      # better instruction-level parallelism:
      #  1. Multiply a_m[0-3]_k4 with w_n4567_k4 into acc_m[0-3]_n4567.
      #  2. Expand the 4-bit weights in d10 to 8-bit weights in d10 and d11,
      #     Expand the 8-bit weights for k6/k7 in d11/d10 into 16-bit values in
      #     q5/q7.
      vmlal.s16  q9, d13, d1[0]
      vshr.s8 d11, d10, #4  /* k6 values in d11 (left-shifted by 4 bits) */
      vmlal.s16  q11, d13, d3[0]
      vshl.i8 q5, q5, #4  /* k7/k6 values in d10/d11 as signed-extended 8-bit. */
      vmlal.s16  q13, d13, d5[0]
      vmovl.s8 q7, d11  /* expand d10 into q6 (8-bit to 16-bit, sign extended) */
      vmlal.s16  q15, d13, d7[0]
      vmovl.s8 q5, d10  /* expand d11 into q5 (8-bit to 16-bit, sign extended) */

      # (Pre-)load next 8 bytes of weights (two ks worth).
      vld1.8 d12, [r5]!

      # k = 6.

      # Multiply a_m[0-3]_k6 with w_n0123_k6 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d10, d1[2]
      vmlal.s16  q10, d10, d3[2]
      vmlal.s16  q12, d10, d5[2]
      vmlal.s16  q14, d10, d7[2]

      # Multiply a_m[0-3]_k6 with w_n4567_k6 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d11, d1[2]
      vmlal.s16  q11, d11, d3[2]
      vmlal.s16  q13, d11, d5[2]
      vmlal.s16  q15, d11, d7[2]

      # k = 7.

      # Multiply a_m[0-3]_k7 with w_n0123_k7 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d1[3]
      vmlal.s16  q10, d14, d3[3]
      vmlal.s16  q12, d14, d5[3]
      vmlal.s16  q14, d14, d7[3]

      # Multiply a_m[0-3]_k7 with w_n4567_k7 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d15, d1[3]
      vmlal.s16  q11, d15, d3[3]
      vmlal.s16  q13, d15, d5[3]
      vmlal.s16  q15, d15, d7[3]

      # Decrement ks as jump back to the top of the loop if we have at least 8.
      subs r0, r0, #8
      bhs .Linner_loop

.Lfinal_iteration:
      # (Pre-)load next 8 bytes of weights (two ks worth).
      vld1.8 d10, [r5]!

      # The following two blocks of operations are interleaved to allow for
      # better instruction-level parallelism:
      #  1. Expand the 8-bit a0/a1/a2/a3 values into q0/q1/q2/q3 as 16-bit
      #     values.
      #  2. Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13,
      #     Expand the 8-bit weights for k0/k1 in d13/d12 into 16-bit values in
      #     q6/q7.
      vshr.s8 d13, d12, #4  /* k0 values in d13 (left-shifted by 4 bits) */
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */
      vshl.i8 q6, q6, #4  /* k1/k0 values in d12/d13 as signed-extended 8-bit. */
      vmovl.s8 q1, d2  /* expand a1 bytes into q1 int16 */
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q2, d4  /* expand a2 bytes into q2 int16 */
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q3, d6  /* expand a3 bytes into q3 int16 */

      # k = 0.

      # Multiply a_m[0-3]_k0 with w_n0123_k0 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q10, d12, d2[0]
      vmlal.s16  q12, d12, d4[0]
      vmlal.s16  q14, d12, d6[0]

      # Multiply a_m[0-3]_k0 with w_n4567_k0 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q11, d13, d2[0]
      vmlal.s16  q13, d13, d4[0]
      vmlal.s16  q15, d13, d6[0]

      # k = 1.

      # Multiply a_m[0-3]_k1 with w_n0123_k1 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d0[1]
      vmlal.s16  q10, d14, d2[1]
      vmlal.s16  q12, d14, d4[1]
      vmlal.s16  q14, d14, d6[1]

      # The following two blocks of operations are interleaved to allow for
      # better instruction-level parallelism:
      #  1. Multiply a_m[0-3]_k1 with w_n4567_k1 into acc_m[0-3]_n4567.
      #  2. Expand the 4-bit weights in d10 to 8-bit weights in d10 and d11,
      #     Expand the 8-bit weights for k2/k3 in d11/d10 into 16-bit values in
      #     q5/q7.
      vmlal.s16  q9, d15, d0[1]
      vshr.s8 d11, d10, #4  /* k2 values in d11 (left-shifted by 4 bits) */
      vmlal.s16  q11, d15, d2[1]
      vshl.i8 q5, q5, #4  /* k3/k2 values in d10/d11 as signed-extended 8-bit. */
      vmlal.s16  q13, d15, d4[1]
      vmovl.s8 q6, d11  /* expand d11 into q6 (8-bit to 16-bit, sign extended) */
      vmlal.s16  q15, d15, d6[1]
      vmovl.s8 q5, d10  /* expand d10 into q5 (8-bit to 16-bit, sign extended) */

      # k = 3.

      # Multiply a_m[0-3]_k3 with w_n0123_k3 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d0[3]
      vmlal.s16  q10, d12, d2[3]
      vmlal.s16  q12, d12, d4[3]
      vmlal.s16  q14, d12, d6[3]

      # (Pre-)load next 8 bytes of weights (two ks worth).
      vld1.8 d12, [r5]!

      # Multiply a_m[0-3]_k3 with w_n4567_k3 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d13, d0[3]
      vmlal.s16  q11, d13, d2[3]
      vmlal.s16  q13, d13, d4[3]
      vmlal.s16  q15, d13, d6[3]

      # k = 2.

      # Multiply a_m[0-3]_k2 with w_n0123_k2 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d10, d0[2]
      vmlal.s16  q10, d10, d2[2]
      vmlal.s16  q12, d10, d4[2]
      vmlal.s16  q14, d10, d6[2]

      # The following two blocks of operations are interleaved to allow for
      # better instruction-level parallelism:
      #  1. Multiply a_m[0-3]_k2 with w_n4567_k2 into acc_m[0-3]_n4567.
      #  2. Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13,
      #     Expand the 8-bit weights for k4/k5 in d13/d12 into 16-bit values in
      #     q6/q7.
      vmlal.s16  q9, d11, d0[2]
      vshr.s8 d13, d12, #4  /* k4 values in d13 (left-shifted by 4 bits) */
      vmlal.s16  q11, d11, d2[2]
      vshl.i8 q6, q6, #4  /* k5/k4 values in d12/d13 as signed-extended 8-bit. */
      vmlal.s16  q13, d11, d4[2]
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmlal.s16  q15, d11, d6[2]
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */

      # (Pre-)load next 8 bytes of weights (two ks worth).
      vld1.8 d10, [r5]!

      # Don't pre-load the next a0/a1/a2/a3.

      # k = 5.

      # Multiply a_m[0-3]_k5 with w_n0123_k5 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d1[1]
      vmlal.s16  q10, d14, d3[1]
      vmlal.s16  q12, d14, d5[1]
      vmlal.s16  q14, d14, d7[1]

      # Multiply a_m[0-3]_k5 with w_n4567_k5 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d15, d1[1]
      vmlal.s16  q11, d15, d3[1]
      vmlal.s16  q13, d15, d5[1]
      vmlal.s16  q15, d15, d7[1]

      # k = 4.

      # Multiply a_m[0-3]_k4 with w_n0123_k4 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q10, d12, d3[0]
      vmlal.s16  q12, d12, d5[0]
      vmlal.s16  q14, d12, d7[0]

      # The following two blocks of operations are interleaved to allow for
      # better instruction-level parallelism:
      #  1. Multiply a_m[0-3]_k4 with w_n4567_k4 into acc_m[0-3]_n4567.
      #  2. Expand the 4-bit weights in d10 to 8-bit weights in d10 and d11,
      #     Expand the 8-bit weights for k6/k7 in d11/d10 into 16-bit values in
      #     q5/q7.
      vmlal.s16  q9, d13, d1[0]
      vshr.s8 d11, d10, #4  /* k6 values in d11 (left-shifted by 4 bits) */
      vmlal.s16  q11, d13, d3[0]
      vshl.i8 q5, q5, #4  /* k7/k6 values in d10/d11 as signed-extended 8-bit. */
      vmlal.s16  q13, d13, d5[0]
      vmovl.s8 q7, d11  /* expand d10 into q6 (8-bit to 16-bit, sign extended) */
      vmlal.s16  q15, d13, d7[0]
      vmovl.s8 q5, d10  /* expand d11 into q5 (8-bit to 16-bit, sign extended) */

      # Don't pre-load next 8 bytes of weights (two ks worth).

      # k = 6.

      # Multiply a_m[0-3]_k6 with w_n0123_k6 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d10, d1[2]
      vmlal.s16  q10, d10, d3[2]
      vmlal.s16  q12, d10, d5[2]
      vmlal.s16  q14, d10, d7[2]

      # Multiply a_m[0-3]_k6 with w_n4567_k6 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d11, d1[2]
      vmlal.s16  q11, d11, d3[2]
      vmlal.s16  q13, d11, d5[2]
      vmlal.s16  q15, d11, d7[2]

      # k = 7.

      # Multiply a_m[0-3]_k7 with w_n0123_k7 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d1[3]
      vmlal.s16  q10, d14, d3[3]
      vmlal.s16  q12, d14, d5[3]
      vmlal.s16  q14, d14, d7[3]

      # Multiply a_m[0-3]_k7 with w_n4567_k7 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d15, d1[3]
      vmlal.s16  q11, d15, d3[3]
      vmlal.s16  q13, d15, d5[3]
      vmlal.s16  q15, d15, d7[3]

      # Jump to the epilogue if there are leftover ks.
      adds r0, r0, #8
      bne .Lepilogue


.Linner_loop_end:

      # Recover the contents of `q5` (dynamic quantization params) from the
      # stack.
      vpop {d10-d11}

      # Convert from int32 to float.
      vcvt.f32.s32 q8, q8, #4
      vcvt.f32.s32 q9, q9, #4
      vcvt.f32.s32 q10, q10, #4
      vcvt.f32.s32 q11, q11, #4
      vcvt.f32.s32 q12, q12, #4
      vcvt.f32.s32 q13, q13, #4
      vcvt.f32.s32 q14, q14, #4
      vcvt.f32.s32 q15, q15, #4

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
      vld1.32 {d0[], d1[]}, [r11]!
      vld1.32 {d2[], d3[]}, [r11]
      sub r11, r11, #4

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
      vmin.f32 q8, q8, q1
      vmin.f32 q10, q10, q1
      vmin.f32 q12, q12, q1
      vmin.f32 q14, q14, q1
      vmin.f32 q9, q9, q1
      vmin.f32 q11, q11, q1
      vmin.f32 q13, q13, q1
      vmin.f32 q15, q15, q1
      vmax.f32 q8, q8, q0
      vmax.f32 q10, q10, q0
      vmax.f32 q12, q12, q0
      vmax.f32 q14, q14, q0
      vmax.f32 q9, q9, q0
      vmax.f32 q11, q11, q0
      vmax.f32 q13, q13, q0
      vmax.f32 q15, q15, q0

      # Check whether full or partial store.
      cmp r1, #8
      blo .Ltail_4
      vst1.32  {d16, d17}, [r6]!
      vst1.32  {d18, d19}, [r6]!
      vst1.32  {d20, d21}, [r4]!
      vst1.32  {d22, d23}, [r4]!
      vst1.32  {d24, d25}, [r8]!
      vst1.32  {d26, d27}, [r8]!
      vst1.32  {d28, d29}, [r14]!
      vst1.32  {d30, d31}, [r14]!
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
      vst1.32  {q10}, [r4]!
      vst1.32  {q12}, [r8]!
      vst1.32  {q14}, [r14]!
      vmov  q8, q9
      vmov  q10, q11
      vmov  q12, q13
      vmov  q14, q15


.Ltail_2:
      tst r1, #2
      beq .Ltail_1
      vst1.32  d16, [r6]!
      vst1.32  d20, [r4]!
      vst1.32  d24, [r8]!
      vst1.32  d28, [r14]!
      vmov d16, d17
      vmov d20, d21
      vmov d24, d25
      vmov d28, d29


.Ltail_1:
      tst r1, #1
      beq .Lreturn
      vst1.32  {d16[0]}, [r6]
      vst1.32  {d20[0]}, [r4]
      vst1.32  {d24[0]}, [r8]
      vst1.32  {d28[0]}, [r14]

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

      # Load 4 As and B0, but only increase the pointers by k.
      vld1.8 d0, [r3]
      add r3, r0
      vld1.8 d2, [r7]
      add r7, r0
      vld1.8 d4, [r9]
      add r9, r0
      vld1.8 d6, [r10]
      add r10, r0

      # Expand the 8-bit a0/a1/a2/a3 values into q0/q1/q2/q3 as 16-bit values.
      vmovl.s8 q0, d0  /* expand a0 bytes into q0 int16 */
      vmovl.s8 q1, d2  /* expand a1 bytes into q1 int16 */
      vmovl.s8 q2, d4  /* expand a2 bytes into q2 int16 */
      vmovl.s8 q3, d6  /* expand a3 bytes into q3 int16 */

      # Expand the 4-bit weights in d12 to 8-bit weights in d12 and d13.
      vshr.s8 d13, d12, #4  /* k0 values in d13 (left-shifted by 4 bits). */
      vshl.i8 q6, q6, #4  /* k1/k0 values in d12/d13 as signed-extended 8-bit. */

      # Expand the 8-bit weights for k0/k1 in d13/d12 into 16-bit values in q6/q7.
      vmovl.s8 q7, d13  /* expand d12 into q7 (8-bit to 16-bit, sign extended) */
      vmovl.s8 q6, d12  /* expand d13 into q6 (8-bit to 16-bit, sign extended) */

      # Multiply a_m[0-3]_k0 with w_n0123_k0 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d0[0]
      vmlal.s16  q10, d12, d2[0]
      vmlal.s16  q12, d12, d4[0]
      vmlal.s16  q14, d12, d6[0]

      # Multiply a_m[0-3]_k0 with w_n4567_k0 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d13, d0[0]
      vmlal.s16  q11, d13, d2[0]
      vmlal.s16  q13, d13, d4[0]
      vmlal.s16  q15, d13, d6[0]

      # If k < 2, we're done.
      cmp r0, #2
      blo .Linner_loop_end

      # Multiply a_m[0-3]_k1 with w_n0123_k1 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d0[1]
      vmlal.s16  q10, d14, d2[1]
      vmlal.s16  q12, d14, d4[1]
      vmlal.s16  q14, d14, d6[1]

      # Multiply a_m[0-3]_k1 with w_n4567_k1 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d15, d0[1]
      vmlal.s16  q11, d15, d2[1]
      vmlal.s16  q13, d15, d4[1]
      vmlal.s16  q15, d15, d6[1]

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

      # Multiply a_m[0-3]_k2 with w_n0123_k2 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d0[2]
      vmlal.s16  q10, d12, d2[2]
      vmlal.s16  q12, d12, d4[2]
      vmlal.s16  q14, d12, d6[2]

      # Multiply a_m[0-3]_k2 with w_n4567_k2 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d13, d0[2]
      vmlal.s16  q11, d13, d2[2]
      vmlal.s16  q13, d13, d4[2]
      vmlal.s16  q15, d13, d6[2]

      # If k < 4, we're done.
      cmp r0, #4
      blo .Linner_loop_end

      # Multiply a_m[0-3]_k3 with w_n0123_k3 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d0[3]
      vmlal.s16  q10, d14, d2[3]
      vmlal.s16  q12, d14, d4[3]
      vmlal.s16  q14, d14, d6[3]

      # Multiply a_m[0-3]_k3 with w_n4567_k3 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d15, d0[3]
      vmlal.s16  q11, d15, d2[3]
      vmlal.s16  q13, d15, d4[3]
      vmlal.s16  q15, d15, d6[3]

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

      # Multiply a_m[0-3]_k4 with w_n0123_k4 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d1[0]
      vmlal.s16  q10, d12, d3[0]
      vmlal.s16  q12, d12, d5[0]
      vmlal.s16  q14, d12, d7[0]

      # Multiply a_m[0-3]_k4 with w_n4567_k4 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d13, d1[0]
      vmlal.s16  q11, d13, d3[0]
      vmlal.s16  q13, d13, d5[0]
      vmlal.s16  q15, d13, d7[0]

      # If k < 6, we're done.
      cmp r0, #6
      blo .Linner_loop_end

      # Multiply a_m[0-3]_k5 with w_n0123_k5 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d14, d1[1]
      vmlal.s16  q10, d14, d3[1]
      vmlal.s16  q12, d14, d5[1]
      vmlal.s16  q14, d14, d7[1]

      # Multiply a_m[0-3]_k5 with w_n4567_k5 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d15, d1[1]
      vmlal.s16  q11, d15, d3[1]
      vmlal.s16  q13, d15, d5[1]
      vmlal.s16  q15, d15, d7[1]

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

      # Multiply a_m[0-3]_k6 with w_n0123_k6 into acc_m[0-3]_n0123.
      vmlal.s16  q8, d12, d1[2]
      vmlal.s16  q10, d12, d3[2]
      vmlal.s16  q12, d12, d5[2]
      vmlal.s16  q14, d12, d7[2]

      # Multiply a_m[0-3]_k6 with w_n4567_k6 into acc_m[0-3]_n4567.
      vmlal.s16  q9, d13, d1[2]
      vmlal.s16  q11, d13, d3[2]
      vmlal.s16  q13, d13, d5[2]
      vmlal.s16  q15, d13, d7[2]

      # Jump back to the end of the inner loop.
      b .Linner_loop_end

END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch32_neonmlal_ld64_2