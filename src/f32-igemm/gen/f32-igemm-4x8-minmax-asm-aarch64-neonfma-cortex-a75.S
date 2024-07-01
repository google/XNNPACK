// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/4x8-aarch64-neonfma-cortex-a75.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float** restrict a,           x4
#     const float* restrict w,            x5
#     float* restrict c,                  x6
#     size_t cm_stride,                  x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_minmax_params params [sp + 24] -> x8

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0 x20  v0  v4
# A1 x13  v1  v5
# A2 x14  v2  v6
# A3 x15  v3  v7
# B   x5  v8  v9 v10 v11
# B       v12 v13 v14 v15
# B       v16 v17 v18 v19
# B       v20 v21 v22 v23
# C0  x6  v24 v25
# C1 x16  v26 v27
# C2 x17  v28 v29
# C3  x7  v30 v31
# Clamp v4 v5

BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75

        # Load cn_stride, a_offset
        LDP         x10, x11, [sp]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]

        # Save x20 on stack
        STR         x20, [sp, -80]!

        # Save d8-d15 on stack
        STP         d8,  d9, [sp, 16]
        STP         d10, d11, [sp, 32]
        STP         d12, d13, [sp, 48]
        STP         d14, d15, [sp, 64]

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x16, x6, x16, LO        //   c1 = c0

        ADD         x17, x16, x7            // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x17, x16, x17, LS       //   c2 = c1

        CMP         x0, 4                   // if mr < 4
        ADD         x7, x17, x7             // c3 = c2 + cm_stride
        CSEL        x7, x17, x7, LO         //   c3 = c2

0:
        # Load initial bias from w into accumulators
        LDP         q24, q25, [x5], 32
        MOV         v26.16b, v24.16b
        MOV         v27.16b, v25.16b
        MOV         v28.16b, v24.16b
        MOV         v29.16b, v25.16b
        MOV         v30.16b, v24.16b
        MOV         v31.16b, v25.16b

        MOV         x9, x3                  // p = ks

1:
        # Load next 4 A pointers
        LDR         x20, [x4], 8
        LDR         x13, [x4], 8
        LDR         x14, [x4], 8
        LDR         x15, [x4], 8

        CMP         x20, x12                // if a0 == zero
        ADD         x20, x20, x11           // a0 += a_offset
        CSEL        x20, x12, x20, EQ       //   a0 = zero, else += a0 + a_offset
        CMP         x13, x12                // if a1 == zero
        ADD         x13, x13, x11           // a1 += a_offset
        CSEL        x13, x12, x13, EQ       //   a1 = zero, else += a1 + a_offset
        CMP         x14, x12                // if a2 == zero
        ADD         x14, x14, x11           // a2 += a_offset
        CSEL        x14, x12, x14, EQ       //   a2 = zero, else += a2 + a_offset
        CMP         x15, x12                // if a3 == zero
        ADD         x15, x15, x11           // a3 += a_offset
        CSEL        x15, x12, x15, EQ       //   a3 = zero, else += a3 + a_offset

        # Is there at least 8 floats (32 bytes) for prologue + epilogue?
        SUBS        x0, x2, 32              // k = kc - 32
        B.LO        4f

        # 16 prologue
        # Read first block of 4 A and B.
        LDR         q0, [x20], 16
        LDP         q16, q17, [x5], 32
        LDR         q1, [x13], 16
        LDR         q2, [x14], 16
        LDR         q3, [x15], 16
        LDP         q18, q19, [x5], 32
        LDP         q20, q21, [x5], 32
        LDP         q22, q23, [x5], 32

        # Is there at least 32.  yes do main loop
        SUBS        x0, x0, 32
        B.LO        3f

        # Main loop - 8 floats of A
2:
        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA        v24.4s, v16.4s, v0.s[0]
        LDP         q8, q9, [x5], 32
        FMLA        v25.4s, v17.4s, v0.s[0]
        FMLA        v26.4s, v16.4s, v1.s[0]
        LDP         q10, q11, [x5], 32
        FMLA        v27.4s, v17.4s, v1.s[0]
        FMLA        v28.4s, v16.4s, v2.s[0]
        LDP         q12, q13, [x5], 32
        FMLA        v29.4s, v17.4s, v2.s[0]
        FMLA        v30.4s, v16.4s, v3.s[0]
        LDP         q14, q15, [x5], 32
        FMLA        v31.4s, v17.4s, v3.s[0]
        FMLA        v24.4s, v18.4s, v0.s[1]
        LDR         q4, [x20], 16
        FMLA        v25.4s, v19.4s, v0.s[1]
        FMLA        v26.4s, v18.4s, v1.s[1]
        LDR         q5, [x13], 16
        FMLA        v27.4s, v19.4s, v1.s[1]
        FMLA        v28.4s, v18.4s, v2.s[1]
        LDR         q6, [x14], 16
        FMLA        v29.4s, v19.4s, v2.s[1]
        FMLA        v30.4s, v18.4s, v3.s[1]
        LDR         q7, [x15], 16
        FMLA        v31.4s, v19.4s, v3.s[1]
        FMLA        v24.4s, v20.4s, v0.s[2]
        FMLA        v25.4s, v21.4s, v0.s[2]
        FMLA        v26.4s, v20.4s, v1.s[2]
        FMLA        v27.4s, v21.4s, v1.s[2]
        FMLA        v28.4s, v20.4s, v2.s[2]
        FMLA        v29.4s, v21.4s, v2.s[2]
        FMLA        v30.4s, v20.4s, v3.s[2]
        FMLA        v31.4s, v21.4s, v3.s[2]
        FMLA        v24.4s, v22.4s, v0.s[3]
        FMLA        v25.4s, v23.4s, v0.s[3]
        FMLA        v26.4s, v22.4s, v1.s[3]
        FMLA        v27.4s, v23.4s, v1.s[3]
        FMLA        v28.4s, v22.4s, v2.s[3]
        FMLA        v29.4s, v23.4s, v2.s[3]
        FMLA        v30.4s, v22.4s, v3.s[3]
        FMLA        v31.4s, v23.4s, v3.s[3]

        # Second block of 4.  FMA for second 4, loads for 1st block of 4.
        FMLA        v24.4s, v8.4s, v4.s[0]
        LDP         q16, q17, [x5], 32
        FMLA        v25.4s, v9.4s, v4.s[0]
        FMLA        v26.4s, v8.4s, v5.s[0]
        LDP         q18, q19, [x5], 32
        FMLA        v27.4s, v9.4s, v5.s[0]
        FMLA        v28.4s, v8.4s, v6.s[0]
        LDP         q20, q21, [x5], 32
        FMLA        v29.4s, v9.4s, v6.s[0]
        FMLA        v30.4s, v8.4s, v7.s[0]
        LDP         q22, q23, [x5], 32
        FMLA        v31.4s, v9.4s, v7.s[0]
        FMLA        v24.4s, v10.4s, v4.s[1]
        LDR         q0, [x20], 16
        FMLA        v25.4s, v11.4s, v4.s[1]
        FMLA        v26.4s, v10.4s, v5.s[1]
        LDR         q1, [x13], 16
        FMLA        v27.4s, v11.4s, v5.s[1]
        FMLA        v28.4s, v10.4s, v6.s[1]
        LDR         q2, [x14], 16
        FMLA        v29.4s, v11.4s, v6.s[1]
        FMLA        v30.4s, v10.4s, v7.s[1]
        LDR         q3, [x15], 16
        FMLA        v31.4s, v11.4s, v7.s[1]
        FMLA        v24.4s, v12.4s, v4.s[2]
        FMLA        v25.4s, v13.4s, v4.s[2]
        FMLA        v26.4s, v12.4s, v5.s[2]
        FMLA        v27.4s, v13.4s, v5.s[2]
        FMLA        v28.4s, v12.4s, v6.s[2]
        FMLA        v29.4s, v13.4s, v6.s[2]
        FMLA        v30.4s, v12.4s, v7.s[2]
        FMLA        v31.4s, v13.4s, v7.s[2]
        FMLA        v24.4s, v14.4s, v4.s[3]
        FMLA        v25.4s, v15.4s, v4.s[3]
        FMLA        v26.4s, v14.4s, v5.s[3]
        FMLA        v27.4s, v15.4s, v5.s[3]
        FMLA        v28.4s, v14.4s, v6.s[3]
        FMLA        v29.4s, v15.4s, v6.s[3]
        SUBS        x0, x0, 32
        FMLA        v30.4s, v14.4s, v7.s[3]
        FMLA        v31.4s, v15.4s, v7.s[3]

        B.HS        2b

3:
        # Epilogue
        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA        v24.4s, v16.4s, v0.s[0]
        LDP         q8, q9, [x5], 32
        FMLA        v25.4s, v17.4s, v0.s[0]
        FMLA        v26.4s, v16.4s, v1.s[0]
        LDP         q10, q11, [x5], 32
        FMLA        v27.4s, v17.4s, v1.s[0]
        FMLA        v28.4s, v16.4s, v2.s[0]
        LDP         q12, q13, [x5], 32
        FMLA        v29.4s, v17.4s, v2.s[0]
        FMLA        v30.4s, v16.4s, v3.s[0]
        LDP         q14, q15, [x5], 32
        FMLA        v31.4s, v17.4s, v3.s[0]
        FMLA        v24.4s, v18.4s, v0.s[1]
        LDR         q4, [x20], 16
        FMLA        v25.4s, v19.4s, v0.s[1]
        FMLA        v26.4s, v18.4s, v1.s[1]
        LDR         q5, [x13], 16
        FMLA        v27.4s, v19.4s, v1.s[1]
        FMLA        v28.4s, v18.4s, v2.s[1]
        LDR         q6, [x14], 16
        FMLA        v29.4s, v19.4s, v2.s[1]
        FMLA        v30.4s, v18.4s, v3.s[1]
        LDR         q7, [x15], 16
        FMLA        v31.4s, v19.4s, v3.s[1]
        FMLA        v24.4s, v20.4s, v0.s[2]
        FMLA        v25.4s, v21.4s, v0.s[2]
        FMLA        v26.4s, v20.4s, v1.s[2]
        FMLA        v27.4s, v21.4s, v1.s[2]
        FMLA        v28.4s, v20.4s, v2.s[2]
        FMLA        v29.4s, v21.4s, v2.s[2]
        FMLA        v30.4s, v20.4s, v3.s[2]
        FMLA        v31.4s, v21.4s, v3.s[2]
        FMLA        v24.4s, v22.4s, v0.s[3]
        FMLA        v25.4s, v23.4s, v0.s[3]
        FMLA        v26.4s, v22.4s, v1.s[3]
        FMLA        v27.4s, v23.4s, v1.s[3]
        FMLA        v28.4s, v22.4s, v2.s[3]
        FMLA        v29.4s, v23.4s, v2.s[3]
        FMLA        v30.4s, v22.4s, v3.s[3]
        FMLA        v31.4s, v23.4s, v3.s[3]

        # Second block of 4.  FMA for second 4, noloads
        FMLA        v24.4s, v8.4s, v4.s[0]
        FMLA        v25.4s, v9.4s, v4.s[0]
        FMLA        v26.4s, v8.4s, v5.s[0]
        FMLA        v27.4s, v9.4s, v5.s[0]
        FMLA        v28.4s, v8.4s, v6.s[0]
        FMLA        v29.4s, v9.4s, v6.s[0]
        FMLA        v30.4s, v8.4s, v7.s[0]
        FMLA        v31.4s, v9.4s, v7.s[0]
        FMLA        v24.4s, v10.4s, v4.s[1]
        FMLA        v25.4s, v11.4s, v4.s[1]
        FMLA        v26.4s, v10.4s, v5.s[1]
        FMLA        v27.4s, v11.4s, v5.s[1]
        FMLA        v28.4s, v10.4s, v6.s[1]
        FMLA        v29.4s, v11.4s, v6.s[1]
        FMLA        v30.4s, v10.4s, v7.s[1]
        FMLA        v31.4s, v11.4s, v7.s[1]
        FMLA        v24.4s, v12.4s, v4.s[2]
        FMLA        v25.4s, v13.4s, v4.s[2]
        FMLA        v26.4s, v12.4s, v5.s[2]
        FMLA        v27.4s, v13.4s, v5.s[2]
        FMLA        v28.4s, v12.4s, v6.s[2]
        FMLA        v29.4s, v13.4s, v6.s[2]
        FMLA        v30.4s, v12.4s, v7.s[2]
        FMLA        v31.4s, v13.4s, v7.s[2]

        FMLA        v24.4s, v14.4s, v4.s[3]
        FMLA        v25.4s, v15.4s, v4.s[3]
        FMLA        v26.4s, v14.4s, v5.s[3]
        FMLA        v27.4s, v15.4s, v5.s[3]

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]

        FMLA        v28.4s, v14.4s, v6.s[3]
        FMLA        v29.4s, v15.4s, v6.s[3]
        FMLA        v30.4s, v14.4s, v7.s[3]
        FMLA        v31.4s, v15.4s, v7.s[3]

4:
        # Remainder- 4 floats of A
        TBZ         x0, 4, 5f

        LDR         q0, [x20], 16
        LDP         q16, q17, [x5], 32
        LDR         q1, [x13], 16
        LDR         q2, [x14], 16
        LDR         q3, [x15], 16
        FMLA        v24.4s, v16.4s, v0.s[0]
        FMLA        v25.4s, v17.4s, v0.s[0]
        LDP         q18, q19, [x5], 32
        FMLA        v26.4s, v16.4s, v1.s[0]
        FMLA        v27.4s, v17.4s, v1.s[0]
        LDP         q20, q21, [x5], 32
        FMLA        v28.4s, v16.4s, v2.s[0]
        FMLA        v29.4s, v17.4s, v2.s[0]
        LDP         q22, q23, [x5], 32
        FMLA        v30.4s, v16.4s, v3.s[0]
        FMLA        v31.4s, v17.4s, v3.s[0]
        FMLA        v24.4s, v18.4s, v0.s[1]
        FMLA        v25.4s, v19.4s, v0.s[1]
        FMLA        v26.4s, v18.4s, v1.s[1]
        FMLA        v27.4s, v19.4s, v1.s[1]
        FMLA        v28.4s, v18.4s, v2.s[1]
        FMLA        v29.4s, v19.4s, v2.s[1]
        FMLA        v30.4s, v18.4s, v3.s[1]
        FMLA        v31.4s, v19.4s, v3.s[1]
        FMLA        v24.4s, v20.4s, v0.s[2]
        FMLA        v25.4s, v21.4s, v0.s[2]
        FMLA        v26.4s, v20.4s, v1.s[2]
        FMLA        v27.4s, v21.4s, v1.s[2]
        FMLA        v28.4s, v20.4s, v2.s[2]
        FMLA        v29.4s, v21.4s, v2.s[2]
        FMLA        v30.4s, v20.4s, v3.s[2]
        FMLA        v31.4s, v21.4s, v3.s[2]
        FMLA        v24.4s, v22.4s, v0.s[3]
        FMLA        v25.4s, v23.4s, v0.s[3]
        FMLA        v26.4s, v22.4s, v1.s[3]
        FMLA        v27.4s, v23.4s, v1.s[3]
        FMLA        v28.4s, v22.4s, v2.s[3]
        FMLA        v29.4s, v23.4s, v2.s[3]
        FMLA        v30.4s, v22.4s, v3.s[3]
        FMLA        v31.4s, v23.4s, v3.s[3]

5:
        # Remainder- 2 floats of A
        TBZ         x0, 3, 6f

        LDR         d0, [x20], 8
        LDP         q16, q17, [x5], 32
        LDR         d1, [x13], 8
        LDR         d2, [x14], 8
        LDR         d3, [x15], 8
        FMLA        v24.4s, v16.4s, v0.s[0]
        FMLA        v25.4s, v17.4s, v0.s[0]
        LDP         q18, q19, [x5], 32
        FMLA        v26.4s, v16.4s, v1.s[0]
        FMLA        v27.4s, v17.4s, v1.s[0]
        FMLA        v28.4s, v16.4s, v2.s[0]
        FMLA        v29.4s, v17.4s, v2.s[0]
        FMLA        v30.4s, v16.4s, v3.s[0]
        FMLA        v31.4s, v17.4s, v3.s[0]
        FMLA        v24.4s, v18.4s, v0.s[1]
        FMLA        v25.4s, v19.4s, v0.s[1]
        FMLA        v26.4s, v18.4s, v1.s[1]
        FMLA        v27.4s, v19.4s, v1.s[1]
        FMLA        v28.4s, v18.4s, v2.s[1]
        FMLA        v29.4s, v19.4s, v2.s[1]
        FMLA        v30.4s, v18.4s, v3.s[1]
        FMLA        v31.4s, v19.4s, v3.s[1]

6:
        # Remainder- 1 float of A
        TBZ         x0, 2, 7f

        LDR         s0, [x20], 4
        LDP         q16, q17, [x5], 32
        LDR         s1, [x13], 4
        LDR         s2, [x14], 4
        LDR         s3, [x15], 4
        FMLA        v24.4s, v16.4s, v0.s[0]
        FMLA        v25.4s, v17.4s, v0.s[0]
        FMLA        v26.4s, v16.4s, v1.s[0]
        FMLA        v27.4s, v17.4s, v1.s[0]
        FMLA        v28.4s, v16.4s, v2.s[0]
        FMLA        v29.4s, v17.4s, v2.s[0]
        FMLA        v30.4s, v16.4s, v3.s[0]
        FMLA        v31.4s, v17.4s, v3.s[0]

7:
        # ks loop
        SUBS        x9, x9, 32              // ks -= MR * sizeof(void*)
        B.HI        1b

        # Clamp
        FMAX        v24.4s, v24.4s, v4.4s
        FMAX        v25.4s, v25.4s, v4.4s
        FMAX        v26.4s, v26.4s, v4.4s
        FMAX        v27.4s, v27.4s, v4.4s
        FMAX        v28.4s, v28.4s, v4.4s
        FMAX        v29.4s, v29.4s, v4.4s
        FMAX        v30.4s, v30.4s, v4.4s
        FMAX        v31.4s, v31.4s, v4.4s
        FMIN        v24.4s, v24.4s, v5.4s
        FMIN        v25.4s, v25.4s, v5.4s
        FMIN        v26.4s, v26.4s, v5.4s
        FMIN        v27.4s, v27.4s, v5.4s
        FMIN        v28.4s, v28.4s, v5.4s
        FMIN        v29.4s, v29.4s, v5.4s
        FMIN        v30.4s, v30.4s, v5.4s
        FMIN        v31.4s, v31.4s, v5.4s

        # Store full 4 x 8
        SUBS        x1, x1, 8
        B.LO        8f

        STP         q30, q31,  [x7]
        ADD         x7,  x7, x10
        STP         q28, q29, [x17]
        ADD         x17, x17, x10
        STP         q26, q27, [x16]
        ADD         x16, x16, x10
        STP         q24, q25,  [x6]
        ADD         x6,  x6, x10

        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b

        # Restore d8-d15 from stack
        LDP         d14, d15, [sp, 64]
        LDP         d12, d13, [sp, 48]
        LDP         d10, d11, [sp, 32]
        LDP         d8,  d9, [sp, 16]

        # Restore x20 from stack
        LDR         x20, [sp], 80
        RET

        # Store odd width
8:
        TBZ         x1, 2, 9f
        STR         q30, [x7], 16
        MOV         v30.16b, v31.16b
        STR         q28, [x17], 16
        MOV         v28.16b, v29.16b
        STR         q26, [x16], 16
        MOV         v26.16b, v27.16b
        STR         q24, [x6], 16
        MOV         v24.16b, v25.16b

9:
        TBZ         x1, 1, 10f
        STR         d30, [x7], 8
        STR         d28, [x17], 8
        DUP         d30, v30.d[1]
        DUP         d28, v28.d[1]
        STR         d26, [x16], 8
        STR         d24, [x6], 8
        DUP         d26, v26.d[1]
        DUP         d24, v24.d[1]

10:
        TBZ         x1, 0, 11f
        STR         s30,  [x7]
        STR         s28, [x17]
        STR         s26, [x16]
        STR         s24,  [x6]
11:
        # Restore d8-d15 from stack
        LDP         d14, d15, [sp, 64]
        LDP         d12, d13, [sp, 48]
        LDP         d10, d11, [sp, 32]
        LDP         d8,  d9, [sp, 16]

        # Restore x20 from stack
        LDR         x20, [sp], 80
        RET

END_FUNCTION xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
