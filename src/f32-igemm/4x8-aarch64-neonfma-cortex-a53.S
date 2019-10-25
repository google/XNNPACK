// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53(
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
#     const xnn_f32_output_params params [sp + 24] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointers
# x14 a0
# x15 a1
# x20 a2
# x21 a3

# C pointers
#  x6 c0
# x16 c1
# x17 c2
# x7  c3

# Vector register usage
# A0  v0
# A1  v0[1]
# A2  v1
# A3  v1[1]
# B   v16 v17 v18 v19
# C   v20 v21
# C   v22 v23
# C   v24 v25
# C   v26 v27
# Clamp v6 v7
# unused A   v2 v3 v4 v5 v8 v9 v10 v11
# unused B   v12 v13 v14 v15
# unused C   v28 c29 v30 v31

BEGIN_FUNCTION xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53

        # Clamp C pointers
        ADD x16, x6, x7          // c1 = c0 + cm_stride
        CMP x0, 2                // if mr < 2
        CSEL x16, x6, x16, LO    //   c1 = c0

        ADD x17, x16, x7         // c2 = c1 + cm_stride
                                 // if mr <= 2
        CSEL x17, x16, x17, LS   //   c2 = c1

        ADD x7, x17, x7         // c3 = c2 + cm_stride
        CMP x0, 4                // if mr < 4
        CSEL x7, x17, x7, LO   //   c3 = c2

        # Save x20,x21,x22,x23 on stack
        STP x20, x21, [sp, -32]!
        STP x22, x23, [sp, 16]

        # Load cn_stride, a_offset
        LDP x10, x11, [sp, 32]

        # Load zero, clamping params pointer
        LDP x12, x8, [sp, 48]

        # Load clamping_params values
        LD2R {v6.4s, v7.4s}, [x8]

0:
        # Load initial bias from w into accumulators
        LDP q20, q21, [x5], 32
        MOV v22.16b, v20.16b
        MOV v23.16b, v21.16b
        PRFM PLDL1KEEP, [x5, 0]  // Prefetch B
        MOV v24.16b, v20.16b
        MOV v25.16b, v21.16b
        PRFM PLDL1KEEP, [x5, 64]
        MOV v26.16b, v20.16b
        MOV v27.16b, v21.16b
        PRFM PLDL1KEEP, [x5, 128]

        MOV x9, x3  // p = ks

1:
        # Load next 4 A pointers
        LDP x14, x15, [x4], 16
        LDP x20, x21, [x4], 16

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

        # Is there at least 2 floats (8 bytes) for main loop?
        SUBS x0, x2, 8  // k = kc - 8
        B.LO 3f

        # Main loop - 2 floats of A (8 bytes)
        # 16 FMA + 4 LD64 A + 2 LDP B
2:
        LDR   d0,  [x14], 8
        LDP  q16,  q17, [x5], 32
        LD1   {v0.d}[1], [x15], 8
        LDR   d1, [x20], 8
        LD1   {v1.d}[1], [x21], 8

        FMLA v20.4s, v16.4s,  v0.s[0]
        FMLA v22.4s, v16.4s,  v0.s[2]
        FMLA v24.4s, v16.4s,  v1.s[0]
        FMLA v26.4s, v16.4s,  v1.s[2]
        LDP  q18,  q19, [x5], 32
        FMLA v21.4s, v17.4s,  v0.s[0]
        FMLA v23.4s, v17.4s,  v0.s[2]
        FMLA v25.4s, v17.4s,  v1.s[0]
        FMLA v27.4s, v17.4s,  v1.s[2]

        FMLA v20.4s, v18.4s,  v0.s[1]
        FMLA v22.4s, v18.4s,  v0.s[3]
        FMLA v24.4s, v18.4s,  v1.s[1]
        FMLA v26.4s, v18.4s,  v1.s[3]
        FMLA v21.4s, v19.4s,  v0.s[1]
        FMLA v23.4s, v19.4s,  v0.s[3]
        SUBS x0, x0, 8
        FMLA v25.4s, v19.4s,  v1.s[1]
        FMLA v27.4s, v19.4s,  v1.s[3]
        B.HS 2b
3:
        # Is there a remainder?- 1 floats of A (4 bytes)
        TBNZ x0, 2, 5f

4:
        # ks loop
        SUBS x9, x9, 32  // ks -= MR * sizeof(void*)
        B.NE 1b

        # Clamp
        FMIN v20.4s, v20.4s, v6.4s
        FMIN v21.4s, v21.4s, v6.4s
        FMIN v22.4s, v22.4s, v6.4s
        FMIN v23.4s, v23.4s, v6.4s
        FMIN v24.4s, v24.4s, v6.4s
        FMIN v25.4s, v25.4s, v6.4s
        FMIN v26.4s, v26.4s, v6.4s
        FMIN v27.4s, v27.4s, v6.4s
        FMAX v20.4s, v20.4s, v7.4s
        FMAX v21.4s, v21.4s, v7.4s
        FMAX v22.4s, v22.4s, v7.4s
        FMAX v23.4s, v23.4s, v7.4s
        FMAX v24.4s, v24.4s, v7.4s
        FMAX v25.4s, v25.4s, v7.4s
        FMAX v26.4s, v26.4s, v7.4s
        FMAX v27.4s, v27.4s, v7.4s

        # Store full 4 x 8
        CMP x1, 8
        B.LO 8f

        STP q26, q27, [x7]
        ADD x7, x7, x10
        STP q24, q25, [x17]
        ADD x17, x17, x10
        STP q22, q23, [x16]
        ADD x16, x16, x10
        STP q20, q21,  [x6]
        ADD  x6,  x6, x10

        SUB x4, x4, x3  // a -= ks

        # nc loop
        SUBS x1, x1, 8
        B.HI 0b

        # Restore x20,x21,x22,x23 from stack
        LDP x22, x23, [sp, 16]
        LDP x20, x21, [sp], 32
        RET

5:
        # Remainder- 1 floats of A (4 bytes)
        LDR   s0,  [x14], 4
        LDP  q16,  q17, [x5], 32
        LD1   {v0.s}[2], [x15], 4
        LDR   s1, [x20], 4
        LD1   {v1.s}[2], [x21], 4

        FMLA v20.4s, v16.4s,  v0.s[0]
        FMLA v22.4s, v16.4s,  v0.s[2]
        FMLA v24.4s, v16.4s,  v1.s[0]
        FMLA v26.4s, v16.4s,  v1.s[2]
        FMLA v21.4s, v17.4s,  v0.s[0]
        FMLA v23.4s, v17.4s,  v0.s[2]
        FMLA v25.4s, v17.4s,  v1.s[0]
        FMLA v27.4s, v17.4s,  v1.s[2]
        B 4b

        # Store odd width
8:
        TBZ x1, 2, 9f
        STR q26,  [x7], 16
        MOV v26.16b, v27.16b
        STR q24, [x17], 16
        MOV v24.16b, v25.16b
        STR q22, [x16], 16
        MOV v22.16b, v23.16b
        STR q20,  [x6], 16
        MOV v20.16b, v21.16b
9:
        TBZ x1, 1, 10f
        STR d26,  [x7], 8
        DUP d26, v26.d[1]
        STR d24, [x17], 8
        DUP d24, v24.d[1]
        STR d22, [x16], 8
        DUP d22, v22.d[1]
        STR d20,  [x6], 8
        DUP d20, v20.d[1]

10:
        TBZ x1, 0, 11f
        STR s26,  [x7]
        STR s24, [x17]
        STR s22, [x16]
        STR s20,  [x6]
11:
        # Restore x20,x21,x22,x23 from stack
        LDP x22, x23, [sp, 16]
        LDP x20, x21, [sp], 32
        RET

END_FUNCTION xnn_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
