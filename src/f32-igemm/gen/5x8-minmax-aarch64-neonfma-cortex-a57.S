// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/5x8-aarch64-neonfma-cortex-a75.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_igemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a57(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float**restrict a,           x4
#     const void*restrict w,             x5
#     uint8_t*restrict c,                x6
#     size_t cm_stride,                  x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_minmax_params params [sp + 24] -> x8

# 5x8 strips the following out of 5x8
# x23 a5
#  x7 c5  x13 unused
# A5  v10 v11
# C   v30 v31

# d8-d15 need to be preserved if used.
# x19-x30 need to be preserved if used.  x18 is reserved for OS.

# A pointers
# x14 a0
# x15 a1
# x20 a2
# x21 a3
#  x8 a4

# C pointers
#  x6 c0
# x16 c1
# x17 c2
# x13 c3
#  x7 c4

# Vector register usage
# A0   v0  v1
# A1   v2  v3
# A2   v4  v5
# A3   v6  v7
# A4   v8  v9
# B   v12 v13 v14 v15
# B   v16 v17 v18 v19
# C   v20 v21
# C   v22 v23
# C   v24 v25
# C   v26 v27
# C   v28 v29
# Clamp v30 v31

BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a57

        # Clamp C pointers / Save d8-d15 on stack
        STP  d8,  d9, [sp, -64]!
        CMP x0, 2                // if mr < 2
        ADD x16, x6, x7          // c1 = c0 + cm_stride
        CSEL x16, x6, x16, LO    //   c1 = c0

        STP d12, d13, [sp, 16]
        ADD x17, x16, x7         // c2 = c1 + cm_stride
                                 // if mr <= 2
        CSEL x17, x16, x17, LS   //   c2 = c1

        STP d14, d15, [sp, 32]
        CMP x0, 4                // if mr < 4
        ADD x13, x17, x7         // c3 = c2 + cm_stride
        CSEL x13, x17, x13, LO   //   c3 = c2

        # Load zero, clamping params pointer
        LDP x12, x8, [sp, 80]
        ADD x7, x13, x7          // c4 = c3 + cm_stride
                                 // if mr <= 4
        CSEL x7, x13, x7, LS     //   c4 = c3

        # Save x20,x21 on stack
        STP x20, x21, [sp, 48]

        # Load clamp values
        LD2R {v30.4s, v31.4s}, [x8]

        # Load cn_stride, a_offset
        LDP x10, x11, [sp, 64]

0:
        # Load initial bias from w into accumulators
        LDP q20, q21, [x5], 32
        MOV v22.16b, v20.16b
        MOV v23.16b, v21.16b
        MOV v24.16b, v20.16b
        MOV v25.16b, v21.16b
        MOV v26.16b, v20.16b
        MOV v27.16b, v21.16b
        MOV v28.16b, v20.16b
        MOV v29.16b, v21.16b

        MOV x9, x3  // p = ks

1:
        # Load next 5 A pointers
        LDP x14, x15, [x4], 16
        LDP x20, x21, [x4], 16
        LDR x8, [x4], 8

        CMP x14, x12            // if a0 == zero
        ADD x14, x14, x11       // a0 += a_offset
        CSEL x14, x12, x14, EQ  //   a0 = zero, else += a0 + a_offset
        CMP x15, x12            // if a1 == zero
        ADD x15, x15, x11       // a1 += a_offset
        CSEL x15, x12, x15, EQ  //   a1 = zero, else += a1 + a_offset
        CMP x20, x12            // if a2 == zero
        ADD x20, x20, x11       // a2 += a_offset
        CSEL x20, x12, x20, EQ  //   a2 = zero, else += a2 + a_offset
        CMP x21, x12            // if a3 == zero
        ADD x21, x21, x11       // a3 += a_offset
        CSEL x21, x12, x21, EQ  //   a3 = zero, else += a3 + a_offset
        CMP x8, x12            // if a4 == zero
        ADD x8, x8, x11       // a4 += a_offset
        CSEL x8, x12, x8, EQ  //   a4 = zero, else += a4 + a_offset

        # Is there at least 8 floats (32 bytes) for prologue + epilogue?
        SUBS x0, x2, 32  // k = kc - 32
        B.LO 5f

        # Prologue - loads for main loop of 96 FMA
        LDR   q0, [x14], 16
        LDR   q2, [x15], 16
        LDR   q4, [x20], 16
        LDR   q6, [x21], 16
        LDR   q8, [x8], 16
        LDP  q12, q13, [x5], 32  // Fetch 3 B (4th deferred)
        LDP  q14, q15, [x5], 32
        LDP  q16, q17, [x5], 32

        # Is there at least 8 floats (32 bytes) for main loop?
        SUBS x0, x0, 32
        B.LO 3f

        # Main loop - 8 floats of A (32 bytes)
        # 80 FMA + 5 LDP A + 8 LDP B
2:
        # First group of 4 A.  40 FMA.
        FMLA v20.4s, v12.4s,  v0.s[0]
        LDP  q18, q19, [x5], 32        // Load last B
        FMLA v22.4s, v12.4s,  v2.s[0]
        FMLA v24.4s, v12.4s,  v4.s[0]
        FMLA v26.4s, v12.4s,  v6.s[0]
        FMLA v28.4s, v12.4s,  v8.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v2.s[0]
        FMLA v25.4s, v13.4s,  v4.s[0]
        FMLA v27.4s, v13.4s,  v6.s[0]
        FMLA v29.4s, v13.4s,  v8.s[0]
        LDR   q1, [x14], 16            // Load next 5 A

        FMLA v20.4s, v14.4s,  v0.s[1]
        FMLA v22.4s, v14.4s,  v2.s[1]
        FMLA v24.4s, v14.4s,  v4.s[1]
        LDR   q3, [x15], 16
        FMLA v26.4s, v14.4s,  v6.s[1]
        FMLA v28.4s, v14.4s,  v8.s[1]
        FMLA v21.4s, v15.4s,  v0.s[1]
        LDR   q5, [x20], 16
        FMLA v23.4s, v15.4s,  v2.s[1]
        FMLA v25.4s, v15.4s,  v4.s[1]
        FMLA v27.4s, v15.4s,  v6.s[1]
        LDR   q7, [x21], 16
        FMLA v29.4s, v15.4s,  v8.s[1]

        FMLA v20.4s, v16.4s,  v0.s[2]
        FMLA v22.4s, v16.4s,  v2.s[2]
        LDR   q9, [x8], 16
        FMLA v24.4s, v16.4s,  v4.s[2]
        FMLA v26.4s, v16.4s,  v6.s[2]
        FMLA v28.4s, v16.4s,  v8.s[2]
        LDP  q12, q13, [x5], 32        // Load 4 B
        FMLA v21.4s, v17.4s,  v0.s[2]
        FMLA v23.4s, v17.4s,  v2.s[2]
        FMLA v25.4s, v17.4s,  v4.s[2]
        FMLA v27.4s, v17.4s,  v6.s[2]
        FMLA v29.4s, v17.4s,  v8.s[2]

        FMLA v20.4s, v18.4s,  v0.s[3]
        FMLA v22.4s, v18.4s,  v2.s[3]
        FMLA v24.4s, v18.4s,  v4.s[3]
        FMLA v26.4s, v18.4s,  v6.s[3]
        LDP  q14, q15, [x5], 32
        FMLA v28.4s, v18.4s,  v8.s[3]
        FMLA v21.4s, v19.4s,  v0.s[3]
        FMLA v23.4s, v19.4s,  v2.s[3]
        LDP  q16, q17, [x5], 32
        FMLA v25.4s, v19.4s,  v4.s[3]
        FMLA v27.4s, v19.4s,  v6.s[3]
        FMLA v29.4s, v19.4s,  v8.s[3]
        LDP  q18, q19, [x5], 32

        # Second group of 4 A.  40 FMA.
        FMLA v20.4s, v12.4s,  v1.s[0]
        FMLA v22.4s, v12.4s,  v3.s[0]
        FMLA v24.4s, v12.4s,  v5.s[0]
        LDR   q0, [x14], 16            // Load next 5 A
        FMLA v26.4s, v12.4s,  v7.s[0]
        FMLA v28.4s, v12.4s,  v9.s[0]
        FMLA v21.4s, v13.4s,  v1.s[0]
        LDR   q2, [x15], 16
        FMLA v23.4s, v13.4s,  v3.s[0]
        FMLA v25.4s, v13.4s,  v5.s[0]
        FMLA v27.4s, v13.4s,  v7.s[0]
        LDR   q4, [x20], 16
        FMLA v29.4s, v13.4s,  v9.s[0]

        FMLA v20.4s, v14.4s,  v1.s[1]
        FMLA v22.4s, v14.4s,  v3.s[1]
        LDR   q6, [x21], 16
        FMLA v24.4s, v14.4s,  v5.s[1]
        FMLA v26.4s, v14.4s,  v7.s[1]
        FMLA v28.4s, v14.4s,  v9.s[1]
        LDR   q8, [x8], 16
        FMLA v21.4s, v15.4s,  v1.s[1]
        FMLA v23.4s, v15.4s,  v3.s[1]
        FMLA v25.4s, v15.4s,  v5.s[1]
        LDP  q12, q13, [x5], 32        // Load next 3 B (not last)
        FMLA v27.4s, v15.4s,  v7.s[1]
        FMLA v29.4s, v15.4s,  v9.s[1]

        FMLA v20.4s, v16.4s,  v1.s[2]
        FMLA v22.4s, v16.4s,  v3.s[2]
        FMLA v24.4s, v16.4s,  v5.s[2]
        FMLA v26.4s, v16.4s,  v7.s[2]
        FMLA v28.4s, v16.4s,  v9.s[2]
        FMLA v21.4s, v17.4s,  v1.s[2]
        FMLA v23.4s, v17.4s,  v3.s[2]
        LDP  q14, q15, [x5], 32
        FMLA v25.4s, v17.4s,  v5.s[2]
        FMLA v27.4s, v17.4s,  v7.s[2]
        FMLA v29.4s, v17.4s,  v9.s[2]
        LDP  q16,  q17, [x5], 32

        FMLA v20.4s, v18.4s,  v1.s[3]
        FMLA v22.4s, v18.4s,  v3.s[3]
        SUBS x0, x0, 32
        FMLA v24.4s, v18.4s,  v5.s[3]
        FMLA v26.4s, v18.4s,  v7.s[3]
        FMLA v28.4s, v18.4s,  v9.s[3]
        FMLA v21.4s, v19.4s,  v1.s[3]
        FMLA v23.4s, v19.4s,  v3.s[3]
        FMLA v25.4s, v19.4s,  v5.s[3]
        FMLA v27.4s, v19.4s,  v7.s[3]
        FMLA v29.4s, v19.4s,  v9.s[3]
        B.HS 2b

        # Epilogue - 8 floats of A (32 bytes)
        # 80 FMA + 5 LDP A + 8 LDP B
        # First block same as main loop.  Second block has no preloads.
3:
        # First group of 4 A.  40 FMA.
        FMLA v20.4s, v12.4s,  v0.s[0]
        LDP  q18, q19, [x5], 32        // Load last B
        FMLA v22.4s, v12.4s,  v2.s[0]
        FMLA v24.4s, v12.4s,  v4.s[0]
        FMLA v26.4s, v12.4s,  v6.s[0]
        FMLA v28.4s, v12.4s,  v8.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v2.s[0]
        FMLA v25.4s, v13.4s,  v4.s[0]
        FMLA v27.4s, v13.4s,  v6.s[0]
        FMLA v29.4s, v13.4s,  v8.s[0]
        LDR   q1, [x14], 16            // Load next 5 A

        FMLA v20.4s, v14.4s,  v0.s[1]
        FMLA v22.4s, v14.4s,  v2.s[1]
        FMLA v24.4s, v14.4s,  v4.s[1]
        LDR   q3, [x15], 16
        FMLA v26.4s, v14.4s,  v6.s[1]
        FMLA v28.4s, v14.4s,  v8.s[1]
        FMLA v21.4s, v15.4s,  v0.s[1]
        LDR   q5, [x20], 16
        FMLA v23.4s, v15.4s,  v2.s[1]
        FMLA v25.4s, v15.4s,  v4.s[1]
        FMLA v27.4s, v15.4s,  v6.s[1]
        LDR   q7, [x21], 16
        FMLA v29.4s, v15.4s,  v8.s[1]

        FMLA v20.4s, v16.4s,  v0.s[2]
        FMLA v22.4s, v16.4s,  v2.s[2]
        LDR   q9, [x8], 16
        FMLA v24.4s, v16.4s,  v4.s[2]
        FMLA v26.4s, v16.4s,  v6.s[2]
        FMLA v28.4s, v16.4s,  v8.s[2]
        LDP  q12, q13, [x5], 32        // Load 4 B
        FMLA v21.4s, v17.4s,  v0.s[2]
        FMLA v23.4s, v17.4s,  v2.s[2]
        FMLA v25.4s, v17.4s,  v4.s[2]
        FMLA v27.4s, v17.4s,  v6.s[2]
        FMLA v29.4s, v17.4s,  v8.s[2]

        FMLA v20.4s, v18.4s,  v0.s[3]
        FMLA v22.4s, v18.4s,  v2.s[3]
        FMLA v24.4s, v18.4s,  v4.s[3]
        FMLA v26.4s, v18.4s,  v6.s[3]
        LDP  q14, q15, [x5], 32
        FMLA v28.4s, v18.4s,  v8.s[3]
        FMLA v21.4s, v19.4s,  v0.s[3]
        FMLA v23.4s, v19.4s,  v2.s[3]
        LDP  q16, q17, [x5], 32
        FMLA v25.4s, v19.4s,  v4.s[3]
        FMLA v27.4s, v19.4s,  v6.s[3]
        FMLA v29.4s, v19.4s,  v8.s[3]
        LDP  q18, q19, [x5], 32

        # Second group of 4 A.  40 FMA.
        FMLA v20.4s, v12.4s,  v1.s[0]
        FMLA v22.4s, v12.4s,  v3.s[0]
        FMLA v24.4s, v12.4s,  v5.s[0]
        FMLA v26.4s, v12.4s,  v7.s[0]
        FMLA v28.4s, v12.4s,  v9.s[0]
        FMLA v21.4s, v13.4s,  v1.s[0]
        FMLA v23.4s, v13.4s,  v3.s[0]
        FMLA v25.4s, v13.4s,  v5.s[0]
        FMLA v27.4s, v13.4s,  v7.s[0]
        FMLA v29.4s, v13.4s,  v9.s[0]

        FMLA v20.4s, v14.4s,  v1.s[1]
        FMLA v22.4s, v14.4s,  v3.s[1]
        FMLA v24.4s, v14.4s,  v5.s[1]
        FMLA v26.4s, v14.4s,  v7.s[1]
        FMLA v28.4s, v14.4s,  v9.s[1]
        FMLA v21.4s, v15.4s,  v1.s[1]
        FMLA v23.4s, v15.4s,  v3.s[1]
        FMLA v25.4s, v15.4s,  v5.s[1]
        FMLA v27.4s, v15.4s,  v7.s[1]
        FMLA v29.4s, v15.4s,  v9.s[1]

        FMLA v20.4s, v16.4s,  v1.s[2]
        FMLA v22.4s, v16.4s,  v3.s[2]
        FMLA v24.4s, v16.4s,  v5.s[2]
        FMLA v26.4s, v16.4s,  v7.s[2]
        FMLA v28.4s, v16.4s,  v9.s[2]
        FMLA v21.4s, v17.4s,  v1.s[2]
        FMLA v23.4s, v17.4s,  v3.s[2]
        FMLA v25.4s, v17.4s,  v5.s[2]
        FMLA v27.4s, v17.4s,  v7.s[2]
        FMLA v29.4s, v17.4s,  v9.s[2]

        FMLA v20.4s, v18.4s,  v1.s[3]
        FMLA v22.4s, v18.4s,  v3.s[3]
        FMLA v24.4s, v18.4s,  v5.s[3]
        FMLA v26.4s, v18.4s,  v7.s[3]
        FMLA v28.4s, v18.4s,  v9.s[3]
        FMLA v21.4s, v19.4s,  v1.s[3]
        FMLA v23.4s, v19.4s,  v3.s[3]
        FMLA v25.4s, v19.4s,  v5.s[3]
        FMLA v27.4s, v19.4s,  v7.s[3]
        FMLA v29.4s, v19.4s,  v9.s[3]
        # Is there a remainder?- 4 floats of A (16 bytes) or less
        TST x0, 31
        B.NE 5f

4:
        # ks loop
        SUBS x9, x9, 40  // ks -= MR * sizeof(void*)
        B.HI 1b

        # Clamp
        FMAX v20.4s, v20.4s, v30.4s
        FMAX v21.4s, v21.4s, v30.4s
        FMAX v22.4s, v22.4s, v30.4s
        FMAX v23.4s, v23.4s, v30.4s
        FMAX v24.4s, v24.4s, v30.4s
        FMAX v25.4s, v25.4s, v30.4s
        FMAX v26.4s, v26.4s, v30.4s
        FMAX v27.4s, v27.4s, v30.4s
        FMAX v28.4s, v28.4s, v30.4s
        FMAX v29.4s, v29.4s, v30.4s
        FMIN v20.4s, v20.4s, v31.4s
        FMIN v21.4s, v21.4s, v31.4s
        FMIN v22.4s, v22.4s, v31.4s
        FMIN v23.4s, v23.4s, v31.4s
        FMIN v24.4s, v24.4s, v31.4s
        FMIN v25.4s, v25.4s, v31.4s
        FMIN v26.4s, v26.4s, v31.4s
        FMIN v27.4s, v27.4s, v31.4s
        FMIN v28.4s, v28.4s, v31.4s
        FMIN v29.4s, v29.4s, v31.4s

        # Store full 5 x 8
        SUBS x1, x1, 8
        B.LO 8f

        STP q28, q29, [x7]
        ADD x7, x7, x10
        STP q26, q27, [x13]
        ADD x13, x13, x10
        STP q24, q25, [x17]
        ADD x17, x17, x10
        STP q22, q23, [x16]
        ADD x16, x16, x10
        STP q20, q21,  [x6]
        ADD  x6,  x6, x10

        SUB x4, x4, x3  // a -= ks

        # nc loop
        B.HI 0b

        # Restore x20,x21 from stack
        LDP x20, x21, [sp, 48]

        # Restore d8-d15 from stack
        LDP d14, d15, [sp, 32]
        LDP d12, d13, [sp, 16]
        LDP  d8,  d9, [sp], 64
        RET

5:
        # Is there a remainder?- 4 floats of A (16 bytes)
        TBZ x0, 4, 6f

        # Remainder- 4 floats of A (16 bytes)
        # Load A
        LDR   q0, [x14], 16
        LDR   q2, [x15], 16
        LDR   q4, [x20], 16
        LDR   q6, [x21], 16
        LDR   q8, [x8], 16
        # Load B
        LDP  q12, q13, [x5], 32
        LDP  q14, q15, [x5], 32
        LDP  q16, q17, [x5], 32
        LDP  q18, q19, [x5], 32

        FMLA v20.4s, v12.4s,  v0.s[0]
        FMLA v22.4s, v12.4s,  v2.s[0]
        FMLA v24.4s, v12.4s,  v4.s[0]
        FMLA v26.4s, v12.4s,  v6.s[0]
        FMLA v28.4s, v12.4s,  v8.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v2.s[0]
        FMLA v25.4s, v13.4s,  v4.s[0]
        FMLA v27.4s, v13.4s,  v6.s[0]
        FMLA v29.4s, v13.4s,  v8.s[0]

        FMLA v20.4s, v14.4s,  v0.s[1]
        FMLA v22.4s, v14.4s,  v2.s[1]
        FMLA v24.4s, v14.4s,  v4.s[1]
        FMLA v26.4s, v14.4s,  v6.s[1]
        FMLA v28.4s, v14.4s,  v8.s[1]
        FMLA v21.4s, v15.4s,  v0.s[1]
        FMLA v23.4s, v15.4s,  v2.s[1]
        FMLA v25.4s, v15.4s,  v4.s[1]
        FMLA v27.4s, v15.4s,  v6.s[1]
        FMLA v29.4s, v15.4s,  v8.s[1]

        FMLA v20.4s, v16.4s,  v0.s[2]
        FMLA v22.4s, v16.4s,  v2.s[2]
        FMLA v24.4s, v16.4s,  v4.s[2]
        FMLA v26.4s, v16.4s,  v6.s[2]
        FMLA v28.4s, v16.4s,  v8.s[2]
        FMLA v21.4s, v17.4s,  v0.s[2]
        FMLA v23.4s, v17.4s,  v2.s[2]
        FMLA v25.4s, v17.4s,  v4.s[2]
        FMLA v27.4s, v17.4s,  v6.s[2]
        FMLA v29.4s, v17.4s,  v8.s[2]

        FMLA v20.4s, v18.4s,  v0.s[3]
        FMLA v22.4s, v18.4s,  v2.s[3]
        FMLA v24.4s, v18.4s,  v4.s[3]
        FMLA v26.4s, v18.4s,  v6.s[3]
        FMLA v28.4s, v18.4s,  v8.s[3]
        FMLA v21.4s, v19.4s,  v0.s[3]
        FMLA v23.4s, v19.4s,  v2.s[3]
        FMLA v25.4s, v19.4s,  v4.s[3]
        FMLA v27.4s, v19.4s,  v6.s[3]
        FMLA v29.4s, v19.4s,  v8.s[3]

        # Is there a remainder?- 2 floats of A (8 bytes)
6:
        TBZ x0, 3, 7f

        # Remainder- 2 floats of A (8 bytes)
        # Load A
        LDR   d0, [x14], 8
        LDR   d2, [x15], 8
        LDR   d4, [x20], 8
        LDR   d6, [x21], 8
        LDR   d8, [x8], 8
        # Load B
        LDP  q12, q13, [x5], 32
        LDP  q14, q15, [x5], 32

        FMLA v20.4s, v12.4s,  v0.s[0]
        FMLA v22.4s, v12.4s,  v2.s[0]
        FMLA v24.4s, v12.4s,  v4.s[0]
        FMLA v26.4s, v12.4s,  v6.s[0]
        FMLA v28.4s, v12.4s,  v8.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v2.s[0]
        FMLA v25.4s, v13.4s,  v4.s[0]
        FMLA v27.4s, v13.4s,  v6.s[0]
        FMLA v29.4s, v13.4s,  v8.s[0]

        FMLA v20.4s, v14.4s,  v0.s[1]
        FMLA v22.4s, v14.4s,  v2.s[1]
        FMLA v24.4s, v14.4s,  v4.s[1]
        FMLA v26.4s, v14.4s,  v6.s[1]
        FMLA v28.4s, v14.4s,  v8.s[1]
        FMLA v21.4s, v15.4s,  v0.s[1]
        FMLA v23.4s, v15.4s,  v2.s[1]
        FMLA v25.4s, v15.4s,  v4.s[1]
        FMLA v27.4s, v15.4s,  v6.s[1]
        FMLA v29.4s, v15.4s,  v8.s[1]

        # Is there a remainder?- 1 float of A (4 bytes)
7:
        TBZ x0, 2, 4b

        # Remainder- 1 float of A (4 bytes)
        # Load A
        LDR   s0, [x14], 4
        LDR   s2, [x15], 4
        LDR   s4, [x20], 4
        LDR   s6, [x21], 4
        LDR   s8, [x8], 4
        # Load B
        LDP  q12, q13, [x5], 32

        FMLA v20.4s, v12.4s,  v0.s[0]
        FMLA v22.4s, v12.4s,  v2.s[0]
        FMLA v24.4s, v12.4s,  v4.s[0]
        FMLA v26.4s, v12.4s,  v6.s[0]
        FMLA v28.4s, v12.4s,  v8.s[0]
        FMLA v21.4s, v13.4s,  v0.s[0]
        FMLA v23.4s, v13.4s,  v2.s[0]
        FMLA v25.4s, v13.4s,  v4.s[0]
        FMLA v27.4s, v13.4s,  v6.s[0]
        FMLA v29.4s, v13.4s,  v8.s[0]
        B 4b

        # Store odd width
8:
        TBZ x1, 2, 9f
        STR q28, [x7], 16
        MOV v28.16b, v29.16b
        STR q26, [x13], 16
        MOV v26.16b, v27.16b
        STR q24, [x17], 16
        MOV v24.16b, v25.16b
        STR q22, [x16], 16
        MOV v22.16b, v23.16b
        STR q20,  [x6], 16
        MOV v20.16b, v21.16b
9:
        TBZ x1, 1, 10f
        STR d28, [x7], 8
        DUP d28, v28.d[1]
        STR d26, [x13], 8
        DUP d26, v26.d[1]
        STR d24, [x17], 8
        DUP d24, v24.d[1]
        STR d22, [x16], 8
        DUP d22, v22.d[1]
        STR d20,  [x6], 8
        DUP d20, v20.d[1]

10:
        TBZ x1, 0, 11f
        STR s28, [x7]
        STR s26, [x13]
        STR s24, [x17]
        STR s22, [x16]
        STR s20,  [x6]
11:
        # Restore x20,x21 from stack
        LDP x20, x21, [sp, 48]

        # Restore d8-d15 from stack
        LDP d14, d15, [sp, 32]
        LDP d12, d13, [sp, 16]
        LDP  d8,  d9, [sp], 64
        RET

END_FUNCTION xnn_f32_igemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a57

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
