// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x2-aarch64-neonfma-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          x4
#     const float* w,           x5
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
# B   x5  v20 v21
# C0  x6  v24 v25
# C1  x9  v26 v27
# C2  x10 v28 v29
# C3  x7  v30 v31
# Clamp v4 v5

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x11, x3, x4             // a1 = a0 + a_stride
        ADD         x9, x6, x7              // c1 = c0 + cm_stride
        CSEL        x11, x3, x11, LO        //   a1 = a0
        CSEL        x9, x6, x9, LO          //   c1 = c0

        # Load min/max values
        LD2R        {v4.2s, v5.2s}, [x8]

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
        MOVI        v25.4s, 0
        LD2         {v24.s, v25.s}[0], [x5], 8
        MOV         v26.16b, v24.16b
        MOV         v27.16b, v25.16b
        MOV         v28.16b, v24.16b
        MOV         v29.16b, v25.16b
        MOV         v30.16b, v24.16b
        MOV         v31.16b, v25.16b

        # Is there at least 4 floats (16 bytes)?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
1:
        LDR         q0,  [x3], 16
        LD2         {v20.4s, v21.4s}, [x5], 32
        LDR         q1, [x11], 16
        LDR         q2, [x12], 16
        LDR         q3,  [x4], 16
        SUBS        x0, x0, 16
        FMLA        v24.4s, v20.4s, v0.4s
        FMLA        v25.4s, v21.4s, v0.4s
        FMLA        v26.4s, v20.4s, v1.4s
        FMLA        v27.4s, v21.4s, v1.4s
        FMLA        v28.4s, v20.4s, v2.4s
        FMLA        v29.4s, v21.4s, v2.4s
        FMLA        v30.4s, v20.4s, v3.4s
        FMLA        v31.4s, v21.4s, v3.4s
        B.HS        1b

        FADDP       v24.4s, v24.4s, v25.4s
        FADDP       v26.4s, v26.4s, v27.4s
        FADDP       v28.4s, v28.4s, v29.4s
        FADDP       v30.4s, v30.4s, v31.4s

        # Is there a remainder?- 1-3 floats of A (4-12 bytes)
        ANDS        x0, x0, 15

        FADDP       v24.4s, v24.4s, v24.4s
        FADDP       v26.4s, v26.4s, v26.4s
        FADDP       v28.4s, v28.4s, v28.4s
        FADDP       v30.4s, v30.4s, v30.4s

        B.NE        4f

2:


        # Clamp
        FMAX        v24.2s, v24.2s, v4.2s
        SUBS        x1, x1, 2
        FMAX        v26.2s, v26.2s, v4.2s
        FMAX        v28.2s, v28.2s, v4.2s
        FMAX        v30.2s, v30.2s, v4.2s
        FMIN        v24.2s, v24.2s, v5.2s
        FMIN        v26.2s, v26.2s, v5.2s
        FMIN        v28.2s, v28.2s, v5.2s
        FMIN        v30.2s, v30.2s, v5.2s

        # Store full 4 x 2
        B.LO        5f

        ST1         {v24.8b},  [x6], x14
        SUB         x3,  x3, x2             // a0 -= kc
        ST1         {v26.8b},  [x9], x14
        SUB         x11, x11, x2            // a1 -= kc
        ST1         {v28.8b}, [x10], x14
        SUB         x12, x12, x2            // a2 -= kc
        ST1         {v30.8b},  [x7], x14
        SUB         x4,  x4, x2             // a3 -= kc

        B.HI        0b

        RET

3:
        ADD         x0, x0, 16

        FADDP       v24.4s, v24.4s, v25.4s
        FADDP       v26.4s, v26.4s, v27.4s
        FADDP       v28.4s, v28.4s, v29.4s
        FADDP       v30.4s, v30.4s, v31.4s
        FADDP       v24.4s, v24.4s, v24.4s
        FADDP       v26.4s, v26.4s, v26.4s
        FADDP       v28.4s, v28.4s, v28.4s
        FADDP       v30.4s, v30.4s, v30.4s

        # Remainder- 1 float of A (4 bytes)
4:
        LDR         s0,  [x3], 4
        LDR         d20, [x5], 8
        LDR         s1, [x11], 4
        LDR         s2, [x12], 4
        LDR         s3,  [x4], 4
        SUBS        x0, x0, 4
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        B.HI        4b
        B           2b

        # Store odd width
5:
        STR         s24,  [x6]
        STR         s26,  [x9]
        STR         s28, [x10]
        STR         s30,  [x7]
        RET

END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
