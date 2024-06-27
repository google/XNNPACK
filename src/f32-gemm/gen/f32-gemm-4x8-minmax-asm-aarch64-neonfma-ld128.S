// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x8-aarch64-neonfma-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          x4
#     const void* w,            x5
#     float* c,                 x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x14
#     const xnn_f32_minmax_params* params)  [sp + 8] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x3  v0
# A1  x11 v1
# A2  x12 v2
# A3  x4  v3
# B   x5  v20 v24 v21 v25 v22 v26 v23 v27
# C0  x6  v16 v17
# C1  x9  v18 v19
# C2  x10 v28 v29
# C3  x7  v30 v31
# Clamp v4 v5

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]

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
        LDP         q16, q17, [x5], 32
        MOV         v18.16b, v16.16b
        MOV         v19.16b, v17.16b
        MOV         v28.16b, v16.16b
        MOV         v29.16b, v17.16b
        MOV         v30.16b, v16.16b
        MOV         v31.16b, v17.16b

        # Is there at least 4 floats (16 bytes)?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
1:
        LDR         q0, [x3], 16
        LDP         q20, q24, [x5], 32      // 8 F32 weights
        LDR         q1, [x11], 16
        LDR         q2, [x12], 16
        LDR         q3, [x4], 16
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v18.4s, v20.4s, v1.s[0]
        FMLA        v28.4s, v20.4s, v2.s[0]
        FMLA        v30.4s, v20.4s, v3.s[0]
        LDP         q21, q25, [x5], 32      // 8 F32 weights
        FMLA        v17.4s, v24.4s, v0.s[0]
        FMLA        v19.4s, v24.4s, v1.s[0]
        FMLA        v29.4s, v24.4s, v2.s[0]
        FMLA        v31.4s, v24.4s, v3.s[0]
        LDP         q22, q26, [x5], 32      // 8 F32 weights
        FMLA        v16.4s, v21.4s, v0.s[1]
        FMLA        v18.4s, v21.4s, v1.s[1]
        FMLA        v28.4s, v21.4s, v2.s[1]
        FMLA        v30.4s, v21.4s, v3.s[1]
        LDP         q23, q27, [x5], 32      // 8 F32 weights
        FMLA        v17.4s, v25.4s, v0.s[1]
        FMLA        v19.4s, v25.4s, v1.s[1]
        FMLA        v29.4s, v25.4s, v2.s[1]
        FMLA        v31.4s, v25.4s, v3.s[1]
        FMLA        v16.4s, v22.4s, v0.s[2]
        FMLA        v18.4s, v22.4s, v1.s[2]
        FMLA        v28.4s, v22.4s, v2.s[2]
        FMLA        v30.4s, v22.4s, v3.s[2]
        FMLA        v17.4s, v26.4s, v0.s[2]
        FMLA        v19.4s, v26.4s, v1.s[2]
        FMLA        v29.4s, v26.4s, v2.s[2]
        FMLA        v31.4s, v26.4s, v3.s[2]
        FMLA        v16.4s, v23.4s, v0.s[3]
        FMLA        v18.4s, v23.4s, v1.s[3]
        FMLA        v28.4s, v23.4s, v2.s[3]
        FMLA        v30.4s, v23.4s, v3.s[3]
        SUBS        x0, x0, 16
        FMLA        v17.4s, v27.4s, v0.s[3]
        FMLA        v19.4s, v27.4s, v1.s[3]
        FMLA        v29.4s, v27.4s, v2.s[3]
        FMLA        v31.4s, v27.4s, v3.s[3]
        B.HS        1b

        TST         x0, 15
        B.NE        3f

2:
        # Clamp
        FMAX        v16.4s, v16.4s, v4.4s
        SUBS        x1, x1, 8
        FMAX        v17.4s, v17.4s, v4.4s
        FMAX        v18.4s, v18.4s, v4.4s
        FMAX        v19.4s, v19.4s, v4.4s
        FMAX        v28.4s, v28.4s, v4.4s
        FMAX        v29.4s, v29.4s, v4.4s
        FMAX        v30.4s, v30.4s, v4.4s
        FMAX        v31.4s, v31.4s, v4.4s
        FMIN        v16.4s, v16.4s, v5.4s
        FMIN        v17.4s, v17.4s, v5.4s
        FMIN        v18.4s, v18.4s, v5.4s
        FMIN        v19.4s, v19.4s, v5.4s
        FMIN        v28.4s, v28.4s, v5.4s
        FMIN        v29.4s, v29.4s, v5.4s
        FMIN        v30.4s, v30.4s, v5.4s
        FMIN        v31.4s, v31.4s, v5.4s

        # Store full 4 x 8
        B.LO        5f


        ST1         {v16.16b, v17.16b},  [x6], x14
        SUB         x3,  x3, x2             // a0 -= kc
        ST1         {v18.16b, v19.16b},  [x9], x14
        SUB         x11, x11, x2            // a1 -= kc
        ST1         {v28.16b, v29.16b}, [x10], x14
        SUB         x12, x12, x2            // a2 -= kc
        ST1         {v30.16b, v31.16b},  [x7], x14
        SUB         x4,  x4, x2             // a3 -= kc

        B.HI        0b
        RET

        # Remainder- 2 floats of A (8 bytes)
3:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBZ         x0, 3, 4f

        # Remainder- 2 floats of A (8 bytes)
        LDP         q20, q24, [x5], 32      // 16 F32 weights
        LDP         q21, q25, [x5], 32
        LDR         d0, [x3], 8
        LDR         d1, [x11], 8
        LDR         d2, [x12], 8
        LDR         d3, [x4], 8
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v18.4s, v20.4s, v1.s[0]
        FMLA        v28.4s, v20.4s, v2.s[0]
        FMLA        v30.4s, v20.4s, v3.s[0]
        FMLA        v17.4s, v24.4s, v0.s[0]
        FMLA        v19.4s, v24.4s, v1.s[0]
        FMLA        v29.4s, v24.4s, v2.s[0]
        FMLA        v31.4s, v24.4s, v3.s[0]
        FMLA        v16.4s, v21.4s, v0.s[1]
        FMLA        v18.4s, v21.4s, v1.s[1]
        FMLA        v28.4s, v21.4s, v2.s[1]
        FMLA        v30.4s, v21.4s, v3.s[1]
        FMLA        v17.4s, v25.4s, v0.s[1]
        FMLA        v19.4s, v25.4s, v1.s[1]
        FMLA        v29.4s, v25.4s, v2.s[1]
        FMLA        v31.4s, v25.4s, v3.s[1]

        # Is there a remainder?- 1 float of A (4 bytes)
        TBZ         x0, 2, 2b

        # Remainder- 1 float of A (4 bytes)
4:
        # Remainder- 2 floats of A (8 bytes)
        LDP         q20, q24, [x5], 32      // 8 F32 weights
        LDR         s0, [x3], 4
        LDR         s1, [x11], 4
        LDR         s2, [x12], 4
        LDR         s3, [x4], 4
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v18.4s, v20.4s, v1.s[0]
        FMLA        v28.4s, v20.4s, v2.s[0]
        FMLA        v30.4s, v20.4s, v3.s[0]
        FMLA        v17.4s, v24.4s, v0.s[0]
        FMLA        v19.4s, v24.4s, v1.s[0]
        FMLA        v29.4s, v24.4s, v2.s[0]
        FMLA        v31.4s, v24.4s, v3.s[0]
        B           2b

        # Store odd width
5:
        TBZ         x1, 2, 6f
        STR         q16, [x6], 16
        MOV         v16.16b, v17.16b
        STR         q18, [x9], 16
        MOV         v18.16b, v19.16b
        STR         q28, [x10], 16
        MOV         v28.16b, v29.16b
        STR         q30, [x7], 16
        MOV         v30.16b, v31.16b

6:
        TBZ         x1, 1, 7f
        STR         d16, [x6], 8
        STR         d18, [x9], 8
        DUP         d16, v16.d[1]
        DUP         d18, v18.d[1]
        STR         d28, [x10], 8
        STR         d30, [x7], 8
        DUP         d28, v28.d[1]
        DUP         d30, v30.d[1]

7:
        TBZ         x1, 0, 8f
        STR         s16,  [x6]
        STR         s18,  [x9]
        STR         s28, [x10]
        STR         s30,  [x7]

8:
        RET

END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
