// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/6x8-aarch64-neonfma-cortex-a73.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_gemminc_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const uint8_t*restrict a, x3
#     size_t a_stride,          x4
#     const void*restrict w,    x5
#     uint8_t*restrict c,       x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x14
#     const float*restrict acc,  [sp + 8] -> x15
#     const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 16] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointers
#  x3 a0
#  x9 a1
# x10 a2
# x11 a3
# x12 a4
#  x4 a5

# C pointers
#  x6 c0
# x16 c1
# x17 c2
# x18 c3
# x13 c4
#  x7 c5

# Vector register usage
# A0   v0  v6
# A1   v1  v7
# A2   v2  v8
# A3   v3  v9
# A4   v4 v10
# A5   v5 v11
# B   v12 v13 v14 v15
# B   v16 v17 v18 v19
# C   v20 v21
# C   v22 v23
# C   v24 v25
# C   v26 v27
# C   v28 v29
# C   v30 v31
# Clamp v6 v7

BEGIN_FUNCTION xnn_f32_gemminc_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73

        # Load cn_stride, acc
        LDP x14, x15, [sp]
        # Load params pointer
        LDR x8, [sp, 16]

        # Clamp A and C pointers / Save d8-d15 on stack
        STP  d8,  d9, [sp, -64]!
        CMP x0, 2                // if mr < 2
        ADD x9, x3, x4           // a1 = a0 + a_stride
        ADD x16, x6, x7          // c1 = c0 + cm_stride
        CSEL x9, x3, x9, LO      //   a1 = a0
        CSEL x16, x6, x16, LO    //   c1 = c0

        STP d10, d11, [sp, 16]
        ADD x10, x9, x4          // a2 = a1 + a_stride
        ADD x17, x16, x7         // c2 = c1 + cm_stride
                                 // if mr <= 2
        CSEL x10, x9, x10, LS    //   a2 = a1
        CSEL x17, x16, x17, LS   //   c2 = c1

        STP d12, d13, [sp, 32]
        CMP x0, 4                // if mr < 4
        ADD x11, x10, x4         // a3 = a2 + a_stride
        ADD x18, x17, x7         // c3 = c2 + cm_stride
        CSEL x11, x10, x11, LO   //   a3 = a2
        CSEL x18, x17, x18, LO   //   c3 = c2

        STP d14, d15, [sp, 48]
        ADD x12, x11, x4         // a4 = a3 + a_stride
        ADD x13, x18, x7         // c4 = c3 + cm_stride
                                 // if mr <= 4
        CSEL x12, x11, x12, LS   //   a4 = a3
        CSEL x13, x18, x13, LS   //   c4 = c3

        CMP x0, 6                // if mr < 6
        ADD x4, x12, x4          // a5 = a4 + a_stride
        ADD x7, x13, x7          // c5 = c4 + cm_stride
        CSEL x4, x12, x4, LO     //   a5 = a4
        CSEL x7, x13, x7, LO     //   c5 = c4

        .p2align 3
0:
        # Load initial accumulators
        LDP q20, q21, [x15], 32
        LDP q22, q23, [x15], 32
        LDP q24, q25, [x15], 32
        LDP q26, q27, [x15], 32
        LDP q28, q29, [x15], 32
        LDP q30, q31, [x15], 32
        PRFM PLDL1KEEP, [x5, 0]  // Prefetch B
        PRFM PLDL1KEEP, [x5, 64]
        PRFM PLDL1KEEP, [x5, 128]
        PRFM PLDL1KEEP, [x5, 192]
        PRFM PLDL1KEEP,  [x3]    // Prefetch A
        PRFM PLDL1KEEP,  [x9]
        PRFM PLDL1KEEP, [x10]
        PRFM PLDL1KEEP, [x11]
        PRFM PLDL1KEEP, [x12]
        PRFM PLDL1KEEP,  [x4]

        # Is there at least 8 floats (32 bytes) for prologue + epilogue?
        SUBS x0, x2, 32  // k = kc - 32
        B.LO 4f

        # Prologue - loads for main loop of 96 FMA
        # load A0 to A4 but not A5
        LDP  q0,  q6,  [x3], 32
        LDP  q1,  q7,  [x9], 32
        LDP  q2,  q8, [x10], 32
        LDP  q3,  q9, [x11], 32
        LDP  q4,  q10, [x12], 32
        # load first set of B
        LDP q12, q13, [x5], 32
        LDP q14, q15, [x5], 32

        # Is there at least 8 floats (32 bytes) for main loop?
        SUBS x0, x0, 32
        B.LO 2f

        # Main loop - 8 floats of A (32 bytes)
        # 96 FMA + 6 LDP A + 8 LDP B
        .p2align 3
1:
        # First group of 4 A.  48 FMA.  Loads A5

        LDP  q5, q11, [x4], 32
        FMLA v20.4s, v12.4s,  v0.s[0]
        FMLA v22.4s, v12.4s,  v1.s[0]
        LDP  q16,  q17, [x5], 32
        FMLA v24.4s, v12.4s,  v2.s[0]
        FMLA v26.4s, v12.4s,  v3.s[0]
        LDP  q18,  q19, [x5], 32
        FMLA v28.4s, v12.4s,  v4.s[0]
        FMLA v30.4s, v12.4s,  v5.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v1.s[0]
        FMLA v25.4s, v13.4s,  v2.s[0]
        FMLA v27.4s, v13.4s,  v3.s[0]
        FMLA v29.4s, v13.4s,  v4.s[0]
        FMLA v31.4s, v13.4s,  v5.s[0]

        FMLA v20.4s, v14.4s,  v0.s[1]
        FMLA v22.4s, v14.4s,  v1.s[1]
        FMLA v24.4s, v14.4s,  v2.s[1]
        FMLA v26.4s, v14.4s,  v3.s[1]
        FMLA v28.4s, v14.4s,  v4.s[1]
        FMLA v30.4s, v14.4s,  v5.s[1]
        FMLA v21.4s, v15.4s,  v0.s[1]
        FMLA v23.4s, v15.4s,  v1.s[1]
        FMLA v25.4s, v15.4s,  v2.s[1]
        FMLA v27.4s, v15.4s,  v3.s[1]
        FMLA v29.4s, v15.4s,  v4.s[1]
        FMLA v31.4s, v15.4s,  v5.s[1]

        LDP  q12,  q13, [x5], 32
        FMLA v20.4s, v16.4s,  v0.s[2]
        FMLA v22.4s, v16.4s,  v1.s[2]
        LDP  q14,  q15, [x5], 32
        FMLA v24.4s, v16.4s,  v2.s[2]
        FMLA v26.4s, v16.4s,  v3.s[2]
        PRFM PLDL1KEEP, [x5, 128]      // Prefetch B
        FMLA v28.4s, v16.4s,  v4.s[2]
        FMLA v30.4s, v16.4s,  v5.s[2]
        PRFM PLDL1KEEP, [x5, 256]
        FMLA v21.4s, v17.4s,  v0.s[2]
        FMLA v23.4s, v17.4s,  v1.s[2]
        FMLA v25.4s, v17.4s,  v2.s[2]
        FMLA v27.4s, v17.4s,  v3.s[2]
        FMLA v29.4s, v17.4s,  v4.s[2]
        FMLA v31.4s, v17.4s,  v5.s[2]

        FMLA v20.4s, v18.4s,  v0.s[3]
        FMLA v22.4s, v18.4s,  v1.s[3]
        FMLA v24.4s, v18.4s,  v2.s[3]
        FMLA v26.4s, v18.4s,  v3.s[3]
        FMLA v28.4s, v18.4s,  v4.s[3]
        FMLA v30.4s, v18.4s,  v5.s[3]
        FMLA v21.4s, v19.4s,  v0.s[3]
        FMLA v23.4s, v19.4s,  v1.s[3]
        FMLA v25.4s, v19.4s,  v2.s[3]
        FMLA v27.4s, v19.4s,  v3.s[3]
        FMLA v29.4s, v19.4s,  v4.s[3]
        FMLA v31.4s, v19.4s,  v5.s[3]

        # Second group of 4 A.  48 FMA.  Loads A0 - A4

        LDP  q16,  q17, [x5], 32
        FMLA v20.4s, v12.4s,  v6.s[0]
        FMLA v22.4s, v12.4s,  v7.s[0]
        LDP  q18,  q19, [x5], 32
        FMLA v24.4s, v12.4s,  v8.s[0]
        FMLA v26.4s, v12.4s,  v9.s[0]
        FMLA v28.4s, v12.4s, v10.s[0]
        FMLA v30.4s, v12.4s, v11.s[0]
        FMLA v21.4s, v13.4s,  v6.s[0]
        FMLA v23.4s, v13.4s,  v7.s[0]
        FMLA v25.4s, v13.4s,  v8.s[0]
        FMLA v27.4s, v13.4s,  v9.s[0]
        FMLA v29.4s, v13.4s, v10.s[0]
        FMLA v31.4s, v13.4s, v11.s[0]

        FMLA v20.4s, v14.4s,  v6.s[1]
        FMLA v22.4s, v14.4s,  v7.s[1]
        FMLA v24.4s, v14.4s,  v8.s[1]
        FMLA v26.4s, v14.4s,  v9.s[1]
        FMLA v28.4s, v14.4s, v10.s[1]
        FMLA v30.4s, v14.4s, v11.s[1]
        FMLA v21.4s, v15.4s,  v6.s[1]
        FMLA v23.4s, v15.4s,  v7.s[1]
        FMLA v25.4s, v15.4s,  v8.s[1]
        FMLA v27.4s, v15.4s,  v9.s[1]
        FMLA v29.4s, v15.4s, v10.s[1]
        FMLA v31.4s, v15.4s, v11.s[1]

        LDP  q12,  q13, [x5], 32
        FMLA v20.4s, v16.4s,  v6.s[2]
        FMLA v20.4s, v18.4s,  v6.s[3]
        LDP  q14,  q15, [x5], 32
        FMLA v21.4s, v17.4s,  v6.s[2]
        FMLA v21.4s, v19.4s,  v6.s[3]
        LDP  q0,  q6, [x3], 32
        FMLA v22.4s, v16.4s,  v7.s[2]
        FMLA v22.4s, v18.4s,  v7.s[3]
        FMLA v23.4s, v17.4s,  v7.s[2]
        FMLA v23.4s, v19.4s,  v7.s[3]
        LDP  q1,  q7, [x9], 32
        FMLA v24.4s, v16.4s,  v8.s[2]
        FMLA v24.4s, v18.4s,  v8.s[3]
        FMLA v25.4s, v17.4s,  v8.s[2]
        FMLA v25.4s, v19.4s,  v8.s[3]
        LDP  q2,  q8, [x10], 32
        FMLA v26.4s, v16.4s,  v9.s[2]
        FMLA v26.4s, v18.4s,  v9.s[3]
        FMLA v27.4s, v17.4s,  v9.s[2]
        FMLA v27.4s, v19.4s,  v9.s[3]
        LDP  q3,  q9, [x11], 32
        FMLA v28.4s, v16.4s, v10.s[2]
        FMLA v28.4s, v18.4s, v10.s[3]
        FMLA v29.4s, v17.4s, v10.s[2]
        FMLA v29.4s, v19.4s, v10.s[3]
        LDP  q4,  q10, [x12], 32
        FMLA v30.4s, v16.4s, v11.s[2]
        FMLA v30.4s, v18.4s, v11.s[3]
        SUBS x0, x0, 32
        FMLA v31.4s, v17.4s, v11.s[2]
        FMLA v31.4s, v19.4s, v11.s[3]
        B.HS 1b

        # Epilogue - 8 floats of A (32 bytes)
        # 96 FMA + 6 LDP A + 8 LDP B
        # First block same as main loop.  Second block has no preloads.
2:
        # First group of 4 A.  48 FMA.  Loads A5

        LDP  q5, q11, [x4], 32
        FMLA v20.4s, v12.4s,  v0.s[0]
        FMLA v22.4s, v12.4s,  v1.s[0]
        LDP  q16,  q17, [x5], 32
        FMLA v24.4s, v12.4s,  v2.s[0]
        FMLA v26.4s, v12.4s,  v3.s[0]
        LDP  q18,  q19, [x5], 32
        FMLA v28.4s, v12.4s,  v4.s[0]
        FMLA v30.4s, v12.4s,  v5.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v1.s[0]
        FMLA v25.4s, v13.4s,  v2.s[0]
        FMLA v27.4s, v13.4s,  v3.s[0]
        FMLA v29.4s, v13.4s,  v4.s[0]
        FMLA v31.4s, v13.4s,  v5.s[0]

        FMLA v20.4s, v14.4s,  v0.s[1]
        FMLA v22.4s, v14.4s,  v1.s[1]
        FMLA v24.4s, v14.4s,  v2.s[1]
        FMLA v26.4s, v14.4s,  v3.s[1]
        FMLA v28.4s, v14.4s,  v4.s[1]
        FMLA v30.4s, v14.4s,  v5.s[1]
        FMLA v21.4s, v15.4s,  v0.s[1]
        FMLA v23.4s, v15.4s,  v1.s[1]
        FMLA v25.4s, v15.4s,  v2.s[1]
        FMLA v27.4s, v15.4s,  v3.s[1]
        FMLA v29.4s, v15.4s,  v4.s[1]
        FMLA v31.4s, v15.4s,  v5.s[1]

        LDP  q12,  q13, [x5], 32
        FMLA v20.4s, v16.4s,  v0.s[2]
        FMLA v22.4s, v16.4s,  v1.s[2]
        LDP  q14,  q15, [x5], 32
        FMLA v24.4s, v16.4s,  v2.s[2]
        FMLA v26.4s, v16.4s,  v3.s[2]
        FMLA v28.4s, v16.4s,  v4.s[2]
        FMLA v30.4s, v16.4s,  v5.s[2]
        FMLA v21.4s, v17.4s,  v0.s[2]
        FMLA v23.4s, v17.4s,  v1.s[2]
        FMLA v25.4s, v17.4s,  v2.s[2]
        FMLA v27.4s, v17.4s,  v3.s[2]
        FMLA v29.4s, v17.4s,  v4.s[2]
        FMLA v31.4s, v17.4s,  v5.s[2]

        FMLA v20.4s, v18.4s,  v0.s[3]
        FMLA v22.4s, v18.4s,  v1.s[3]
        FMLA v24.4s, v18.4s,  v2.s[3]
        FMLA v26.4s, v18.4s,  v3.s[3]
        FMLA v28.4s, v18.4s,  v4.s[3]
        FMLA v30.4s, v18.4s,  v5.s[3]
        FMLA v21.4s, v19.4s,  v0.s[3]
        FMLA v23.4s, v19.4s,  v1.s[3]
        FMLA v25.4s, v19.4s,  v2.s[3]
        FMLA v27.4s, v19.4s,  v3.s[3]
        FMLA v29.4s, v19.4s,  v4.s[3]
        FMLA v31.4s, v19.4s,  v5.s[3]

        # Second group of 4 A.  48 FMA. No A Loads, No last B load

        LDP  q16,  q17, [x5], 32
        FMLA v20.4s, v12.4s,  v6.s[0]
        FMLA v22.4s, v12.4s,  v7.s[0]
        LDP  q18,  q19, [x5], 32
        FMLA v24.4s, v12.4s,  v8.s[0]
        FMLA v26.4s, v12.4s,  v9.s[0]
        FMLA v28.4s, v12.4s, v10.s[0]
        FMLA v30.4s, v12.4s, v11.s[0]
        FMLA v21.4s, v13.4s,  v6.s[0]
        FMLA v23.4s, v13.4s,  v7.s[0]
        FMLA v25.4s, v13.4s,  v8.s[0]
        FMLA v27.4s, v13.4s,  v9.s[0]
        FMLA v29.4s, v13.4s, v10.s[0]
        FMLA v31.4s, v13.4s, v11.s[0]

        FMLA v20.4s, v14.4s,  v6.s[1]
        FMLA v22.4s, v14.4s,  v7.s[1]
        FMLA v24.4s, v14.4s,  v8.s[1]
        FMLA v26.4s, v14.4s,  v9.s[1]
        FMLA v28.4s, v14.4s, v10.s[1]
        FMLA v30.4s, v14.4s, v11.s[1]
        FMLA v21.4s, v15.4s,  v6.s[1]
        FMLA v23.4s, v15.4s,  v7.s[1]
        FMLA v25.4s, v15.4s,  v8.s[1]
        FMLA v27.4s, v15.4s,  v9.s[1]
        FMLA v29.4s, v15.4s, v10.s[1]
        FMLA v31.4s, v15.4s, v11.s[1]

        # Last part of epilogue has loads removed.

        FMLA v20.4s, v16.4s,  v6.s[2]
        FMLA v22.4s, v16.4s,  v7.s[2]
        FMLA v24.4s, v16.4s,  v8.s[2]
        FMLA v26.4s, v16.4s,  v9.s[2]
        FMLA v28.4s, v16.4s, v10.s[2]
        FMLA v30.4s, v16.4s, v11.s[2]
        FMLA v21.4s, v17.4s,  v6.s[2]
        FMLA v23.4s, v17.4s,  v7.s[2]
        FMLA v25.4s, v17.4s,  v8.s[2]
        FMLA v27.4s, v17.4s,  v9.s[2]
        FMLA v29.4s, v17.4s, v10.s[2]
        FMLA v31.4s, v17.4s, v11.s[2]

        FMLA v20.4s, v18.4s,  v6.s[3]
        FMLA v22.4s, v18.4s,  v7.s[3]
        FMLA v24.4s, v18.4s,  v8.s[3]
        FMLA v26.4s, v18.4s,  v9.s[3]
        FMLA v28.4s, v18.4s, v10.s[3]
        FMLA v30.4s, v18.4s, v11.s[3]
        FMLA v21.4s, v19.4s,  v6.s[3]
        FMLA v23.4s, v19.4s,  v7.s[3]

        # Load min/max values
        LD2R {v6.4s, v7.4s}, [x8]

        FMLA v25.4s, v19.4s,  v8.s[3]
        FMLA v27.4s, v19.4s,  v9.s[3]
        # Is there a remainder?- 4 floats of A (16 bytes) or less
        TST x0, 31
        FMLA v29.4s, v19.4s, v10.s[3]
        FMLA v31.4s, v19.4s, v11.s[3]
        B.NE 4f

        .p2align 3

        # Clamp
3:
        SUBS x1, x1, 8
        FMAX v20.4s, v20.4s, v6.4s
        FMAX v21.4s, v21.4s, v6.4s
        FMAX v22.4s, v22.4s, v6.4s
        FMAX v23.4s, v23.4s, v6.4s
        FMAX v24.4s, v24.4s, v6.4s
        FMAX v25.4s, v25.4s, v6.4s
        FMAX v26.4s, v26.4s, v6.4s
        FMAX v27.4s, v27.4s, v6.4s
        FMAX v28.4s, v28.4s, v6.4s
        FMAX v29.4s, v29.4s, v6.4s
        FMAX v30.4s, v30.4s, v6.4s
        FMAX v31.4s, v31.4s, v6.4s
        FMIN v20.4s, v20.4s, v7.4s
        FMIN v21.4s, v21.4s, v7.4s
        FMIN v22.4s, v22.4s, v7.4s
        FMIN v23.4s, v23.4s, v7.4s
        FMIN v24.4s, v24.4s, v7.4s
        FMIN v25.4s, v25.4s, v7.4s
        FMIN v26.4s, v26.4s, v7.4s
        FMIN v27.4s, v27.4s, v7.4s
        FMIN v28.4s, v28.4s, v7.4s
        FMIN v29.4s, v29.4s, v7.4s
        FMIN v30.4s, v30.4s, v7.4s
        FMIN v31.4s, v31.4s, v7.4s

        # Store full 6 x 8
        NOP
        B.LO 7f

        STP q30, q31,  [x7]
        ADD x7, x7, x14
        SUB  x3,  x3, x2 // a0 -= kc
        STP q28, q29, [x13]
        ADD x13, x13, x14
        SUB  x9,  x9, x2 // a1 -= kc
        STP q26, q27, [x18]
        ADD x18, x18, x14
        SUB x10, x10, x2 // a2 -= kc
        STP q24, q25, [x17]
        ADD x17, x17, x14
        SUB x11, x11, x2 // a3 -= kc
        STP q22, q23, [x16]
        ADD x16, x16, x14
        SUB x12, x12, x2 // a4 -= kc
        STP q20, q21,  [x6]
        ADD  x6,  x6, x14
        SUB  x4,  x4, x2 // a5 -= kc

        NOP
        B.HI 0b

        # Restore d8-d15 from stack
        LDP d14, d15, [sp, 48]
        LDP d12, d13, [sp, 32]
        LDP d10, d11, [sp, 16]
        LDP  d8,  d9, [sp], 64
        RET

        .p2align 3
4:
        # Load min/max values
        LD2R {v6.4s, v7.4s}, [x8]

        # Is there a remainder?- 4 floats of A (16 bytes)
        TBZ x0, 4, 5f

        # Remainder- 4 floats of A (16 bytes)
        # Load A
        LDR   q0,  [x3], 16
        LDR   q1,  [x9], 16
        LDR   q2, [x10], 16
        LDR   q3, [x11], 16
        LDR   q4, [x12], 16
        LDR   q5,  [x4], 16
        # Load B
        LDP  q12,  q13, [x5], 32
        LDP  q14,  q15, [x5], 32
        LDP  q16,  q17, [x5], 32
        LDP  q18,  q19, [x5], 32

        FMLA v20.4s, v12.4s,  v0.s[0]
        FMLA v22.4s, v12.4s,  v1.s[0]
        FMLA v24.4s, v12.4s,  v2.s[0]
        FMLA v26.4s, v12.4s,  v3.s[0]
        FMLA v28.4s, v12.4s,  v4.s[0]
        FMLA v30.4s, v12.4s,  v5.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v1.s[0]
        FMLA v25.4s, v13.4s,  v2.s[0]
        FMLA v27.4s, v13.4s,  v3.s[0]
        FMLA v29.4s, v13.4s,  v4.s[0]
        FMLA v31.4s, v13.4s,  v5.s[0]

        FMLA v20.4s, v14.4s,  v0.s[1]
        FMLA v22.4s, v14.4s,  v1.s[1]
        FMLA v24.4s, v14.4s,  v2.s[1]
        FMLA v26.4s, v14.4s,  v3.s[1]
        FMLA v28.4s, v14.4s,  v4.s[1]
        FMLA v30.4s, v14.4s,  v5.s[1]
        FMLA v21.4s, v15.4s,  v0.s[1]
        FMLA v23.4s, v15.4s,  v1.s[1]
        FMLA v25.4s, v15.4s,  v2.s[1]
        FMLA v27.4s, v15.4s,  v3.s[1]
        FMLA v29.4s, v15.4s,  v4.s[1]
        FMLA v31.4s, v15.4s,  v5.s[1]

        FMLA v20.4s, v16.4s,  v0.s[2]
        FMLA v22.4s, v16.4s,  v1.s[2]
        FMLA v24.4s, v16.4s,  v2.s[2]
        FMLA v26.4s, v16.4s,  v3.s[2]
        FMLA v28.4s, v16.4s,  v4.s[2]
        FMLA v30.4s, v16.4s,  v5.s[2]
        FMLA v21.4s, v17.4s,  v0.s[2]
        FMLA v23.4s, v17.4s,  v1.s[2]
        FMLA v25.4s, v17.4s,  v2.s[2]
        FMLA v27.4s, v17.4s,  v3.s[2]
        FMLA v29.4s, v17.4s,  v4.s[2]
        FMLA v31.4s, v17.4s,  v5.s[2]

        FMLA v20.4s, v18.4s,  v0.s[3]
        FMLA v22.4s, v18.4s,  v1.s[3]
        FMLA v24.4s, v18.4s,  v2.s[3]
        FMLA v26.4s, v18.4s,  v3.s[3]
        FMLA v28.4s, v18.4s,  v4.s[3]
        FMLA v30.4s, v18.4s,  v5.s[3]
        FMLA v21.4s, v19.4s,  v0.s[3]
        FMLA v23.4s, v19.4s,  v1.s[3]
        FMLA v25.4s, v19.4s,  v2.s[3]
        FMLA v27.4s, v19.4s,  v3.s[3]
        FMLA v29.4s, v19.4s,  v4.s[3]
        FMLA v31.4s, v19.4s,  v5.s[3]

        # Is there a remainder?- 2 floats of A (8 bytes)
5:
        TBZ x0, 3, 6f

        # Remainder- 2 floats of A (8 bytes)
        # Load A
        LDR   d0,  [x3], 8
        LDR   d1,  [x9], 8
        LDR   d2, [x10], 8
        LDR   d3, [x11], 8
        LDR   d4, [x12], 8
        LDR   d5,  [x4], 8
        # Load B
        LDP  q12,  q13, [x5], 32
        LDP  q14,  q15, [x5], 32

        FMLA v20.4s, v12.4s,  v0.s[0]
        FMLA v22.4s, v12.4s,  v1.s[0]
        FMLA v24.4s, v12.4s,  v2.s[0]
        FMLA v26.4s, v12.4s,  v3.s[0]
        FMLA v28.4s, v12.4s,  v4.s[0]
        FMLA v30.4s, v12.4s,  v5.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v1.s[0]
        FMLA v25.4s, v13.4s,  v2.s[0]
        FMLA v27.4s, v13.4s,  v3.s[0]
        FMLA v29.4s, v13.4s,  v4.s[0]
        FMLA v31.4s, v13.4s,  v5.s[0]

        FMLA v20.4s, v14.4s,  v0.s[1]
        FMLA v22.4s, v14.4s,  v1.s[1]
        FMLA v24.4s, v14.4s,  v2.s[1]
        FMLA v26.4s, v14.4s,  v3.s[1]
        FMLA v28.4s, v14.4s,  v4.s[1]
        FMLA v30.4s, v14.4s,  v5.s[1]
        FMLA v21.4s, v15.4s,  v0.s[1]
        FMLA v23.4s, v15.4s,  v1.s[1]
        FMLA v25.4s, v15.4s,  v2.s[1]
        FMLA v27.4s, v15.4s,  v3.s[1]
        FMLA v29.4s, v15.4s,  v4.s[1]
        FMLA v31.4s, v15.4s,  v5.s[1]

        # Is there a remainder?- 1 float of A (4 bytes)
6:
        TBZ x0, 2, 3b

        # Remainder- 1 float of A (4 bytes)
        # Load A
        LDR   s0,  [x3], 4
        LDR   s1,  [x9], 4
        LDR   s2, [x10], 4
        LDR   s3, [x11], 4
        LDR   s4, [x12], 4
        LDR   s5,  [x4], 4
        # Load B
        LDP  q12,  q13, [x5], 32

        FMLA v20.4s, v12.4s,  v0.s[0]
        FMLA v22.4s, v12.4s,  v1.s[0]
        FMLA v24.4s, v12.4s,  v2.s[0]
        FMLA v26.4s, v12.4s,  v3.s[0]
        FMLA v28.4s, v12.4s,  v4.s[0]
        FMLA v30.4s, v12.4s,  v5.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v1.s[0]
        FMLA v25.4s, v13.4s,  v2.s[0]
        FMLA v27.4s, v13.4s,  v3.s[0]
        FMLA v29.4s, v13.4s,  v4.s[0]
        FMLA v31.4s, v13.4s,  v5.s[0]
        B 3b

        .p2align 3

        # Store odd width
7:
        TBZ x1, 2, 8f
        STR q30,  [x7], 16
        MOV v30.16b, v31.16b
        STR q28, [x13], 16
        MOV v28.16b, v29.16b
        STR q26, [x18], 16
        MOV v26.16b, v27.16b
        STR q24, [x17], 16
        MOV v24.16b, v25.16b
        STR q22, [x16], 16
        MOV v22.16b, v23.16b
        STR q20,  [x6], 16
        MOV v20.16b, v21.16b
8:
        TBZ x1, 1, 9f
        STR d30,  [x7], 8
        DUP d30, v30.d[1]
        STR d28, [x13], 8
        DUP d28, v28.d[1]
        STR d26, [x18], 8
        DUP d26, v26.d[1]
        STR d24, [x17], 8
        DUP d24, v24.d[1]
        STR d22, [x16], 8
        DUP d22, v22.d[1]
        STR d20,  [x6], 8
        DUP d20, v20.d[1]

9:
        TBZ x1, 0, 10f
        STR s30,  [x7]
        STR s28, [x13]
        STR s26, [x18]
        STR s24, [x17]
        STR s22, [x16]
        STR s20,  [x6]
10:
        # Restore d8-d15 from stack
        LDP d14, d15, [sp, 48]
        LDP d12, d13, [sp, 32]
        LDP d10, d11, [sp, 16]
        LDP  d8,  d9, [sp], 64
        RET

END_FUNCTION xnn_f32_gemminc_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
