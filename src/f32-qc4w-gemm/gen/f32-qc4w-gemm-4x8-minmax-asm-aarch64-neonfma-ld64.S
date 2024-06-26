// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x8-aarch64-neonfma-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          x4
#     const float* w,           x5
#     float* c,                 x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x14
#     const xnn_f32_qc4w_minmax_params* params)  [sp + 8] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x3  v0
# A1  x11 v1
# A2  x12 v2
# A3  x4  v3
# B   x5  v20 v21 v22 v23
# C0  x6  v24 v25
# C1  x9  v26 v27
# C2  x10 v28 v29
# C3  x7  v30 v31
# Clamp v4 v5
# ZeroPoint v6

BEGIN_FUNCTION xnn_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x11, x3, x4             // a1 = a0 + a_stride
        ADD         x9, x6, x7              // c1 = c0 + cm_stride
        CSEL        x11, x3, x11, LO        //   a1 = a0
        CSEL        x9, x6, x9, LO          //   c1 = c0

        # Load min/max/zerop values
        LD3R        {v4.4s, v5.4s, v6.4s}, [x8]

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

        # Is there at least 2 floats (8 bytes)?
        SUBS        x0, x2, 8               // k = kc - 8
        B.LO        3f

        # Main loop - 2 floats of A (8 bytes)
1:
        LDR         d0,  [x3], 8
        LDR         d20, [x5], 8            // 16 QC4 weights
        UXTL        v21.8h, v20.8b
        USHR        v23.8h, v21.8h, 4       // second set of 8 weights
        BIC         v21.8h, 0xF0            // first set of 8 weights
        SADDW       v20.4s, v6.4s, v21.4h
        SADDW2      v21.4s, v6.4s, v21.8h
        SADDW       v22.4s, v6.4s, v23.4h
        SADDW2      v23.4s, v6.4s, v23.8h
        SCVTF       v20.4s, v20.4s
        SCVTF       v21.4s, v21.4s
        SCVTF       v22.4s, v22.4s
        SCVTF       v23.4s, v23.4s
        LDR         d1, [x11], 8
        LDR         d2, [x12], 8
        LDR         d3,  [x4], 8
        FMLA        v24.4s, v20.4s, v0.s[0]
        FMLA        v25.4s, v21.4s, v0.s[0]
        FMLA        v26.4s, v20.4s, v1.s[0]
        FMLA        v27.4s, v21.4s, v1.s[0]
        FMLA        v28.4s, v20.4s, v2.s[0]
        FMLA        v29.4s, v21.4s, v2.s[0]
        FMLA        v30.4s, v20.4s, v3.s[0]
        FMLA        v31.4s, v21.4s, v3.s[0]
        FMLA        v24.4s, v22.4s, v0.s[1]
        FMLA        v25.4s, v23.4s, v0.s[1]
        FMLA        v26.4s, v22.4s, v1.s[1]
        FMLA        v27.4s, v23.4s, v1.s[1]
        SUBS        x0, x0, 8
        FMLA        v28.4s, v22.4s, v2.s[1]
        FMLA        v29.4s, v23.4s, v2.s[1]
        FMLA        v30.4s, v22.4s, v3.s[1]
        FMLA        v31.4s, v23.4s, v3.s[1]
        B.HS        1b

        # Is there a remainder?- 1 float of A (4 bytes)
        TBNZ        x0, 2, 3f

2:
        # Scale
        LDP         q20, q21, [x5], 32
        FMUL        v24.4s, v24.4s, v20.4s
        FMUL        v25.4s, v25.4s, v21.4s
        FMUL        v26.4s, v26.4s, v20.4s
        FMUL        v27.4s, v27.4s, v21.4s
        FMUL        v28.4s, v28.4s, v20.4s
        FMUL        v29.4s, v29.4s, v21.4s
        FMUL        v30.4s, v30.4s, v20.4s
        FMUL        v31.4s, v31.4s, v21.4s
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
        B.LO        4f

        ST1         {v24.16b, v25.16b},  [x6], x14
        SUB         x3,  x3, x2             // a0 -= kc
        ST1         {v26.16b, v27.16b},  [x9], x14
        SUB         x11, x11, x2            // a1 -= kc
        ST1         {v28.16b, v29.16b}, [x10], x14
        SUB         x12, x12, x2            // a2 -= kc
        ST1         {v30.16b, v31.16b},  [x7], x14
        SUB         x4,  x4, x2             // a3 -= kc

        B.HI        0b
        RET

        # Remainder- 1 float of A (4 bytes)
3:
        LDR         s0,  [x3], 4
        LDR         d20, [x5], 8            // 8 QC4 weights
        SXTL        v21.8h, v20.8b
        SADDW       v20.4s, v6.4s, v21.4h
        SADDW2      v21.4s, v6.4s, v21.8h
        SCVTF       v20.4s, v20.4s
        SCVTF       v21.4s, v21.4s
        LDR         s1, [x11], 4
        LDR         s2, [x12], 4
        LDR         s3 , [x4], 4

        FMLA        v24.4s, v20.4s, v0.s[0]
        FMLA        v25.4s, v21.4s, v0.s[0]
        FMLA        v26.4s, v20.4s, v1.s[0]
        FMLA        v27.4s, v21.4s, v1.s[0]
        FMLA        v28.4s, v20.4s, v2.s[0]
        FMLA        v29.4s, v21.4s, v2.s[0]
        FMLA        v30.4s, v20.4s, v3.s[0]
        FMLA        v31.4s, v21.4s, v3.s[0]
        B           2b

        # Store odd width
4:
        TBZ         x1, 2, 5f
        STR         q24, [x6], 16
        MOV         v24.16b, v25.16b
        STR         q26, [x9], 16
        MOV         v26.16b, v27.16b
        STR         q28, [x10], 16
        MOV         v28.16b, v29.16b
        STR         q30, [x7], 16
        MOV         v30.16b, v31.16b

5:
        TBZ         x1, 1, 6f
        STR         d24, [x6], 8
        STR         d26, [x9], 8
        DUP         d24, v24.d[1]
        DUP         d26, v26.d[1]
        STR         d28, [x10], 8
        STR         d30, [x7], 8
        DUP         d28, v28.d[1]
        DUP         d30, v30.d[1]

6:
        TBZ         x1, 0, 7f
        STR         s24,  [x6]
        STR         s26,  [x9]
        STR         s28, [x10]
        STR         s30,  [x7]
7:
        RET

END_FUNCTION xnn_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
