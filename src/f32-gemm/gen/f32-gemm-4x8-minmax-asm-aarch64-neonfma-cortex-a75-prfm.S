// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          x4
#     const float* w,           x5
#     float* c,                 x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x14
#     const xnn_f32_minmax_params* params)  [sp + 8] -> x8

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Vector register usage
# A0  x3  v0  v4
# A1  x11 v1  v5
# A2  x12 v2  v6
# A3  x4  v3  v7
# B   x5  v8  v9 v10 v11
# B       v12 v13 v14 v15
# B       v16 v17 v18 v19
# B       v20 v21 v22 v23
# C   x6  v24 v25
# C   x9  v26 v27
# C   x10 v28 v29
# C   x7  v30 v31
# Clamp v4 v5

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]

        # Save d8-d15 on stack
        STP         d8,  d9, [sp, -64]!
        STP         d10, d11, [sp, 16]
        STP         d12, d13, [sp, 32]
        STP         d14, d15, [sp, 48]

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x11, x3, x4             // a1 = a0 + a_stride
        ADD         x9, x6, x7              // c1 = c0 + cm_stride
        CSEL        x11, x3, x11, LO        //   a1 = a0
        CSEL        x9, x6, x9, LO          //   c1 = c0

        ADD         x12, x11, x4            // a2 = a1 + a_stride
        ADD         x10, x9, x7             // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x12, x11, x12, LS       //   a2 = a1
        CSEL        x10, x9, x10, LS        //   c2 = c1

        CMP         x0, 4                   // if mr < 4
        ADD         x4, x12, x4             // a3 = a2 + a_stride
        ADD         x7, x10, x7             // c3 = c2 + cm_stride
        CSEL        x4, x12, x4, LO         //   a3 = a2
        CSEL        x7, x10, x7, LO         //   c3 = c2

0:
        # Load initial bias from w into accumulators
        LDP         q24, q25, [x5], 32
        MOV         v26.16b, v24.16b
        MOV         v27.16b, v25.16b
        MOV         v28.16b, v24.16b
        MOV         v29.16b, v25.16b
        MOV         v30.16b, v24.16b
        MOV         v31.16b, v25.16b

        # Is there at least 8 floats (32 bytes) for prologue + epilogue?
        SUBS        x0, x2, 32              // k = kc - 32
        B.LO        3f

        # 16 prologue
        # Read first block of 4 A and B.
        LDR         q0,  [x3], 16
        LDP         q16, q17, [x5], 32
        LDR         q1, [x11], 16
        LDR         q2, [x12], 16
        LDR         q3,  [x4], 16
        LDP         q18, q19, [x5], 32
        LDP         q20, q21, [x5], 32
        LDP         q22, q23, [x5], 32

        # Is there at least 32.  yes do main loop
        SUBS        x0, x0, 32
        B.LO        2f

        # Main loop - 8 floats of A (32 bytes)
1:
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
        LDR         q4, [x3], 16
        FMLA        v25.4s, v19.4s, v0.s[1]
        FMLA        v26.4s, v18.4s, v1.s[1]
        LDR         q5, [x11], 16
        FMLA        v27.4s, v19.4s, v1.s[1]
        FMLA        v28.4s, v18.4s, v2.s[1]
        LDR         q6, [x12], 16
        FMLA        v29.4s, v19.4s, v2.s[1]
        FMLA        v30.4s, v18.4s, v3.s[1]
        LDR         q7, [x4], 16
        FMLA        v31.4s, v19.4s, v3.s[1]
        FMLA        v24.4s, v20.4s, v0.s[2]
        PRFM        PLDL1KEEP, [x5, 128]
        FMLA        v25.4s, v21.4s, v0.s[2]
        FMLA        v26.4s, v20.4s, v1.s[2]
        PRFM        PLDL1KEEP, [x5, 192]
        FMLA        v27.4s, v21.4s, v1.s[2]
        FMLA        v28.4s, v20.4s, v2.s[2]
        PRFM        PLDL1KEEP, [x5, 256]
        FMLA        v29.4s, v21.4s, v2.s[2]
        FMLA        v30.4s, v20.4s, v3.s[2]
        PRFM        PLDL1KEEP, [x5, 320]
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
        LDR         q0, [x3], 16
        FMLA        v25.4s, v11.4s, v4.s[1]
        FMLA        v26.4s, v10.4s, v5.s[1]
        LDR         q1, [x11], 16
        FMLA        v27.4s, v11.4s, v5.s[1]
        FMLA        v28.4s, v10.4s, v6.s[1]
        LDR         q2, [x12], 16
        FMLA        v29.4s, v11.4s, v6.s[1]
        FMLA        v30.4s, v10.4s, v7.s[1]
        LDR         q3, [x4], 16
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
        B.HS        1b

2:
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
        LDR         q4, [x3], 16
        FMLA        v25.4s, v19.4s, v0.s[1]
        FMLA        v26.4s, v18.4s, v1.s[1]
        LDR         q5, [x11], 16
        FMLA        v27.4s, v19.4s, v1.s[1]
        FMLA        v28.4s, v18.4s, v2.s[1]
        LDR         q6, [x12], 16
        FMLA        v29.4s, v19.4s, v2.s[1]
        FMLA        v30.4s, v18.4s, v3.s[1]
        LDR         q7, [x4], 16
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

3:
        # Remainder- 4 floats of A (16 bytes)
        TBZ         x0, 4, 4f

        LDR         q0,  [x3], 16
        LDP         q16, q17, [x5], 32
        LDR         q1, [x11], 16
        LDR         q2, [x12], 16
        LDR         q3,  [x4], 16
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

4:
        # Remainder- 2 floats of A (8 bytes)
        TBZ         x0, 3, 5f

        LDR         d0,  [x3], 8
        LDP         q16, q17, [x5], 32
        LDR         d1, [x11], 8
        LDR         d2, [x12], 8
        LDR         d3,  [x4], 8
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

5:
        # Remainder- 1 float of A (4 bytes)
        TBZ         x0, 2, 6f

        LDR         s0,  [x3], 4
        LDP         q16, q17, [x5], 32
        LDR         s1, [x11], 4
        LDR         s2, [x12], 4
        LDR         s3,  [x4], 4
        FMLA        v24.4s, v16.4s, v0.s[0]
        FMLA        v25.4s, v17.4s, v0.s[0]
        FMLA        v26.4s, v16.4s, v1.s[0]
        FMLA        v27.4s, v17.4s, v1.s[0]
        FMLA        v28.4s, v16.4s, v2.s[0]
        FMLA        v29.4s, v17.4s, v2.s[0]
        FMLA        v30.4s, v16.4s, v3.s[0]
        FMLA        v31.4s, v17.4s, v3.s[0]

6:
        # Clamp
        FMAX        v24.4s, v24.4s, v4.4s
        SUBS        x1, x1, 8
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
        B.LO        7f

        STP         q24, q25,  [x6]
        SUB         x3,  x3, x2             // a0 -= kc
        ADD         x6,  x6, x14
        STP         q26, q27,  [x9]
        SUB         x11, x11, x2            // a1 -= kc
        ADD         x9,  x9, x14
        STP         q28, q29, [x10]
        SUB         x12, x12, x2            // a2 -= kc
        ADD         x10, x10, x14
        STP         q30, q31,  [x7]
        SUB         x4,  x4, x2             // a3 -= kc
        ADD         x7,  x7, x14

        B.HI        0b

        # Restore d8-d15 from stack
        LDP         d14, d15, [sp, 48]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8,  d9, [sp], 64
        RET

        # Store odd width
7:
        TBZ         x1, 2, 8f
        STR         q24, [x6], 16
        MOV         v24.16b, v25.16b
        STR         q26, [x9], 16
        MOV         v26.16b, v27.16b
        STR         q28, [x10], 16
        MOV         v28.16b, v29.16b
        STR         q30, [x7], 16
        MOV         v30.16b, v31.16b

8:
        TBZ         x1, 1, 9f
        STR         d24, [x6], 8
        STR         d26, [x9], 8
        DUP         d24, v24.d[1]
        DUP         d26, v26.d[1]
        STR         d28, [x10], 8
        STR         d30, [x7], 8
        DUP         d28, v28.d[1]
        DUP         d30, v30.d[1]

9:
        TBZ         x1, 0, 10f
        STR         s24,  [x6]
        STR         s26,  [x9]
        STR         s28, [x10]
        STR         s30,  [x7]
10:
        # Restore d8-d15 from stack
        LDP         d14, d15, [sp, 48]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8,  d9, [sp], 64
        RET


END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
