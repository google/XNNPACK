// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x1-aarch64-neonfma-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_qc4w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128(
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
# B   x5  v20
# C0  x6  v24
# C1  x9  v26
# C2  x10 v28
# C3  x7  v30
# Clamp v4 v5
# zerop/mask v6 v7

BEGIN_FUNCTION xnn_f32_qc4w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x11, x3, x4             // a1 = a0 + a_stride
        ADD         x9, x6, x7              // c1 = c0 + cm_stride
        CSEL        x11, x3, x11, LO        //   a1 = a0
        CSEL        x9, x6, x9, LO          //   c1 = c0

        # Load min/max/zerop values
        LD3R        {v4.2s, v5.2s, v6.2s}, [x8]
        MOVI        v7.8b, 15

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
        MOVI        v24.4s, 0
        LDR         s24, [x5], 4
        MOV         v26.16b, v24.16b
        MOV         v28.16b, v24.16b
        MOV         v30.16b, v24.16b

        # Is there at least 4 floats (16 bytes)?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
1:
       LDR         h21, [x5], 2        // 4 QC4 weights
       LDR         q0,  [x3], 16
       AND         v20.8b, v21.8b, v7.8b   // first 2 weights
       USHR        v21.8b, v21.8b, 4       // next 2 weights
       INS         v20.h[1], v21.h[0]      // insert 2 weights
       SADDW       v20.8h, v6.8h, v20.8b
       LDR         q1, [x11], 16
       SXTL        v20.4s, v20.4h
       LDR         q2, [x12], 16
       SCVTF       v20.2s, v20.2s
       LDR         q3,  [x4], 16
        SUBS        x0, x0, 16
        FMLA        v24.4s, v20.4s, v0.4s
        FMLA        v26.4s, v20.4s, v1.4s
        FMLA        v28.4s, v20.4s, v2.4s
        FMLA        v30.4s, v20.4s, v3.4s
        B.HS        1b

        FADDP       v24.4s, v24.4s, v24.4s
        FADDP       v26.4s, v26.4s, v26.4s
        FADDP       v28.4s, v28.4s, v28.4s
        FADDP       v30.4s, v30.4s, v30.4s

        # Is there a remainder?- 1 halffloat of A (2 bytes)
        ANDS        x0, x0, 15

        FADDP       s24, v24.2s
        FADDP       s26, v26.2s
        FADDP       s28, v28.2s
        FADDP       s30, v30.2s

        B.NE        3f

2:
        # Scale
        LDR         s20, [x5], 4
        FMUL        s24, s24, v20.s[0]
        FMUL        s26, s26, v20.s[0]
        FMUL        s28, s28, v20.s[0]
        FMUL        s30, s30, v20.s[0]

        # Clamp
        FMAX        s24, s24, s4
        SUBS        x1, x1, 1
        FMAX        s26, s26, s4
        FMAX        s28, s28, s4
        FMAX        s30, s30, s4
        FMIN        s24, s24, s5
        FMIN        s26, s26, s5
        FMIN        s28, s28, s5
        FMIN        s30, s30, s5

        ST1         {v24.s}[0],  [x6], x14
        SUB         x3,  x3, x2             // a0 -= kc
        ST1         {v26.s}[0],  [x9], x14
        SUB         x11, x11, x2            // a1 -= kc
        ST1         {v28.s}[0], [x10], x14
        SUB         x12, x12, x2            // a2 -= kc
        ST1         {v30.s}[0],  [x7], x14
        SUB         x4,  x4, x2             // a3 -= kc

        B.HI        0b

        RET

3:
        AND         x0, x0, 15

        # Remainder- 1 float of A (4 bytes)
4:
        LDR         s0,  [x3], 4
        // TODO: This supports remainder of 1 or 2 but not 3.
        LDR         b20, [x5], 1
        SADDW       v20.8h, v6.8h, v20.8b
        SXTL        v20.4s, v20.4h
        SCVTF       v20.2s, v20.2s
        LDR         s1, [x11], 4
        LDR         s2, [x12], 4
        LDR         s3,  [x4], 4
        SUBS        x0, x0, 4
        FMLA        s24, s20, v0.s[0]
        FMLA        s26, s20, v1.s[0]
        FMLA        s28, s20, v2.s[0]
        FMLA        s30, s20, v3.s[0]
        B.NE        4b
        B           2b

END_FUNCTION xnn_f32_qc4w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
