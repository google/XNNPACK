// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x2-aarch64-neonfma-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64(
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

BEGIN_FUNCTION xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64

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
        LDR         d24, [x5], 8
        MOV         v26.8b, v24.8b
        MOV         v28.8b, v24.8b
        MOV         v30.8b, v24.8b
        MOVI        v25.2s, 0
        MOVI        v27.2s, 0
        MOVI        v29.2s, 0
        MOVI        v31.2s, 0

        # Is there at least 2 floats (8 bytes)?
        SUBS        x0, x2, 8               // k = kc - 8
        B.LO        3f

        # Main loop - 2 floats of A (8 bytes)
1:
        LDR         s21, [x5], 4            // 4 QC8 weights
        SXTL        v20.8h, v21.8b
        LDR         d0,  [x3], 8
        SXTL        v20.4s, v20.4h
        LDR         d1, [x11], 8
        SCVTF       v20.4s, v20.4s
        LDR         d2, [x12], 8
        DUP         d21, v20.d[1]
        LDR         d3,  [x4], 8
        SUBS        x0, x0, 8
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        FMLA        v25.2s, v21.2s, v0.s[1]
        FMLA        v27.2s, v21.2s, v1.s[1]
        FMLA        v29.2s, v21.2s, v2.s[1]
        FMLA        v31.2s, v21.2s, v3.s[1]
        B.HS        1b

        # Is there a remainder?- 1 float of A (4 bytes)
        TBNZ        x0, 2, 3f

2:
        FADD        v24.2s, v24.2s, v25.2s
        FADD        v26.2s, v26.2s, v27.2s
        FADD        v28.2s, v28.2s, v29.2s
        FADD        v30.2s, v30.2s, v31.2s
        # Scale
        LDR         d20, [x5], 8
        FMUL        v24.2s, v24.2s, v20.2s
        FMUL        v26.2s, v26.2s, v20.2s
        FMUL        v28.2s, v28.2s, v20.2s
        FMUL        v30.2s, v30.2s, v20.2s

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
        B.LO        4f

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

        # Remainder- 1 float of A (4 bytes)
3:
        LDR         s0,  [x3], 4
        LDR         h21, [x5], 2            // 2 QC8 weights
        SXTL        v20.8h, v21.8b
        SXTL        v20.4s, v20.4h
        SCVTF       v20.2s, v20.2s
        LDR         s1, [x11], 4
        LDR         s2, [x12], 4
        LDR         s3,  [x4], 4
        SUBS        x0, x0, 4
        FMLA        v24.2s, v20.2s, v0.s[0]
        FMLA        v26.2s, v20.2s, v1.s[0]
        FMLA        v28.2s, v20.2s, v2.s[0]
        FMLA        v30.2s, v20.2s, v3.s[0]
        B           2b

        # Store odd width
4:
        STR         s24,  [x6]
        STR         s26,  [x9]
        STR         s28, [x10]
        STR         s30,  [x7]
        RET

END_FUNCTION xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
