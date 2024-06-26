// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/4x8-aarch64-neonfma-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128(
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
#     const xnn_f32_minmax_params params [sp + 24] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0 x8   v0
# A1 x13  v1
# A2 x14  v2
# A3 x15  v3
# B   x5 v20 v21 v22 v23
# C0  x6  v24 v25
# C1  x16 v26 v27
# C2  x17 v28 v29
# C3  x7  v30 v31
# Clamp v4 v5

BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128

        # Load cn_stride, a_offset
        LDP         x10, x11, [sp]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x16, x6, x16, LO        //   c1 = c0

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]

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
        LDP         x8, x13, [x4], 16
        LDP         x14, x15, [x4], 16

        CMP         x8, x12                 // if a0 == zero
        ADD         x8, x8, x11             // a0 += a_offset
        CSEL        x8, x12, x8, EQ         //   a0 = zero, else += a0 + a_offset
        CMP         x13, x12                // if a1 == zero
        ADD         x13, x13, x11           // a1 += a_offset
        CSEL        x13, x12, x13, EQ       //   a1 = zero, else += a1 + a_offset
        CMP         x14, x12                // if a2 == zero
        ADD         x14, x14, x11           // a2 += a_offset
        CSEL        x14, x12, x14, EQ       //   a2 = zero, else += a2 + a_offset
        CMP         x15, x12                // if a3 == zero
        ADD         x15, x15, x11           // a3 += a_offset
        CSEL        x15, x12, x15, EQ       //   a3 = zero, else += a3 + a_offset

        # Is there at least 4 floats (16 bytes)?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        4f

        # Main loop - 4 floats of A (16 bytes)
2:
        LDR         q0,  [x8], 16
        LDP         q20, q21, [x5], 32
        LDR         q1, [x13], 16
        LDR         q2, [x14], 16
        LDR         q3, [x15], 16
        FMLA        v24.4s, v20.4s, v0.s[0]
        FMLA        v25.4s, v21.4s, v0.s[0]
        FMLA        v26.4s, v20.4s, v1.s[0]
        FMLA        v27.4s, v21.4s, v1.s[0]
        LDP         q22, q23, [x5], 32
        FMLA        v28.4s, v20.4s, v2.s[0]
        FMLA        v29.4s, v21.4s, v2.s[0]
        FMLA        v30.4s, v20.4s, v3.s[0]
        FMLA        v31.4s, v21.4s, v3.s[0]
        LDP         q16, q17, [x5], 32
        FMLA        v24.4s, v22.4s, v0.s[1]
        FMLA        v25.4s, v23.4s, v0.s[1]
        FMLA        v26.4s, v22.4s, v1.s[1]
        FMLA        v27.4s, v23.4s, v1.s[1]
        LDP         q18, q19, [x5], 32
        FMLA        v28.4s, v22.4s, v2.s[1]
        FMLA        v29.4s, v23.4s, v2.s[1]
        FMLA        v30.4s, v22.4s, v3.s[1]
        FMLA        v31.4s, v23.4s, v3.s[1]
        FMLA        v24.4s, v16.4s, v0.s[2]
        FMLA        v25.4s, v17.4s, v0.s[2]
        FMLA        v26.4s, v16.4s, v1.s[2]
        FMLA        v27.4s, v17.4s, v1.s[2]
        FMLA        v28.4s, v16.4s, v2.s[2]
        FMLA        v29.4s, v17.4s, v2.s[2]
        FMLA        v30.4s, v16.4s, v3.s[2]
        FMLA        v31.4s, v17.4s, v3.s[2]
        FMLA        v24.4s, v18.4s, v0.s[3]
        FMLA        v25.4s, v19.4s, v0.s[3]
        FMLA        v26.4s, v18.4s, v1.s[3]
        FMLA        v27.4s, v19.4s, v1.s[3]
        FMLA        v28.4s, v18.4s, v2.s[3]
        FMLA        v29.4s, v19.4s, v2.s[3]
        SUBS        x0, x0, 16
        FMLA        v30.4s, v18.4s, v3.s[3]
        FMLA        v31.4s, v19.4s, v3.s[3]
        B.HS        2b

        # Is there a remainder?- 2 floats of A (8 bytes) or less
        TST         x0, 15
        B.NE        4f
3:
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
        B.LO        6f

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
        RET

        # Remainder- 2 floats of A (8 bytes)
4:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBZ         x0, 3, 5f

        # Remainder- 2 floats of A (8 bytes)
        LDP         q20, q21, [x5], 32
        LDR         d0,  [x8], 8
        LDR         d1, [x13], 8
        LDR         d2, [x14], 8
        LDR         d3, [x15], 8
        FMLA        v24.4s, v20.4s, v0.s[0]
        FMLA        v25.4s, v21.4s, v0.s[0]
        FMLA        v26.4s, v20.4s, v1.s[0]
        FMLA        v27.4s, v21.4s, v1.s[0]
        LDP         q22, q23, [x5], 32
        FMLA        v28.4s, v20.4s, v2.s[0]
        FMLA        v29.4s, v21.4s, v2.s[0]
        FMLA        v30.4s, v20.4s, v3.s[0]
        FMLA        v31.4s, v21.4s, v3.s[0]
        FMLA        v24.4s, v22.4s, v0.s[1]
        FMLA        v25.4s, v23.4s, v0.s[1]
        FMLA        v26.4s, v22.4s, v1.s[1]
        FMLA        v27.4s, v23.4s, v1.s[1]
        FMLA        v28.4s, v22.4s, v2.s[1]
        FMLA        v29.4s, v23.4s, v2.s[1]
        FMLA        v30.4s, v22.4s, v3.s[1]
        FMLA        v31.4s, v23.4s, v3.s[1]

        # Is there a remainder?- 1 float of A (4 bytes)
        TBZ         x0, 2, 3b

        # Remainder- 1 float of A
5:
        LDR         s0, [x8], 4
        LDP         q20, q21, [x5], 32
        LDR         s1, [x13], 4
        LDR         s2, [x14], 4
        LDR         s3, [x15], 4
        FMLA        v24.4s, v20.4s, v0.s[0]
        FMLA        v25.4s, v21.4s, v0.s[0]
        FMLA        v26.4s, v20.4s, v1.s[0]
        FMLA        v27.4s, v21.4s, v1.s[0]
        FMLA        v28.4s, v20.4s, v2.s[0]
        FMLA        v29.4s, v21.4s, v2.s[0]
        FMLA        v30.4s, v20.4s, v3.s[0]
        FMLA        v31.4s, v21.4s, v3.s[0]
        B           3b

        # Store odd width
6:
        TBZ         x1, 2, 7f
        STR         q30, [x7], 16
        MOV         v30.16b, v31.16b
        STR         q28, [x17], 16
        MOV         v28.16b, v29.16b
        STR         q26, [x16], 16
        MOV         v26.16b, v27.16b
        STR         q24, [x6], 16
        MOV         v24.16b, v25.16b

7:
        TBZ         x1, 1, 8f
        STR         d30, [x7], 8
        STR         d28, [x17], 8
        DUP         d30, v30.d[1]
        DUP         d28, v28.d[1]
        STR         d26, [x16], 8
        STR         d24, [x6], 8
        DUP         d26, v26.d[1]
        DUP         d24, v24.d[1]

8:
        TBZ         x1, 0, 9f
        STR         s30,  [x7]
        STR         s28, [x17]
        STR         s26, [x16]
        STR         s24,  [x6]
9:
        RET

END_FUNCTION xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
