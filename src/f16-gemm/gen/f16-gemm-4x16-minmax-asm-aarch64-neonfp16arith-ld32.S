// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/4x16-aarch64-neonfp16arith-ld32.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const void* restrict a,    x3
#     size_t a_stride,          x4
#     const void* restrict w,    x5
#     void* restrict c,          x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x14
#     const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3 v0
// A1 x11 v1
// A2 x12 v2
// A3  x4 v3
// B   x5 v20 v21 v22 v23
// C0  x6 v24 v25
// C1  x9 v26 v27
// C2 x10 v28 v29
// C3  x7 v30 v31
// clamp  v4, v5

BEGIN_FUNCTION xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Load params values
        LD2R        {v4.8h, v5.8h}, [x8]

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
        LDR         q24, [x5], 16
        LDR         q25, [x5], 16
        MOV         v26.16b, v24.16b
        MOV         v28.16b, v24.16b
        MOV         v30.16b, v24.16b
        MOV         v27.16b, v25.16b
        MOV         v29.16b, v25.16b
        MOV         v31.16b, v25.16b

        # Is there at least 2 halffloats (4 bytes)?
        SUBS        x0, x2, 4               // k = kc - 4
        B.LO        3f

       .p2align 3
        # Main loop - 2 halffloats of A (4 bytes)
1:
        LDR         s0,  [x3], 4
        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         s1, [x11], 4
        LDR         s2, [x12], 4
        LDR         s3,  [x4], 4
        LDR         q22, [x5], 16
        LDR         q23, [x5], 16
        SUBS        x0, x0, 4
        FMLA        v24.8h, v20.8h, v0.h[0]
        FMLA        v25.8h, v21.8h, v0.h[0]
        FMLA        v26.8h, v20.8h, v1.h[0]
        FMLA        v27.8h, v21.8h, v1.h[0]
        FMLA        v28.8h, v20.8h, v2.h[0]
        FMLA        v29.8h, v21.8h, v2.h[0]
        FMLA        v30.8h, v20.8h, v3.h[0]
        FMLA        v31.8h, v21.8h, v3.h[0]

        FMLA        v24.8h, v22.8h, v0.h[1]
        FMLA        v25.8h, v23.8h, v0.h[1]
        FMLA        v26.8h, v22.8h, v1.h[1]
        FMLA        v27.8h, v23.8h, v1.h[1]
        FMLA        v28.8h, v22.8h, v2.h[1]
        FMLA        v29.8h, v23.8h, v2.h[1]
        FMLA        v30.8h, v22.8h, v3.h[1]
        FMLA        v31.8h, v23.8h, v3.h[1]
        B.HS        1b

        # Is there a remainder?- 1 halffloat of A (2 bytes)
        TBNZ        x0, 1, 3f

2:
        # Clamp
        FMAX        v24.8h, v24.8h, v4.8h
        SUBS        x1, x1, 16
        FMAX        v25.8h, v25.8h, v4.8h
        FMAX        v26.8h, v26.8h, v4.8h
        FMAX        v27.8h, v27.8h, v4.8h
        FMAX        v28.8h, v28.8h, v4.8h
        FMAX        v29.8h, v29.8h, v4.8h
        FMAX        v30.8h, v30.8h, v4.8h
        FMAX        v31.8h, v31.8h, v4.8h
        FMIN        v24.8h, v24.8h, v5.8h
        FMIN        v25.8h, v25.8h, v5.8h
        FMIN        v26.8h, v26.8h, v5.8h
        FMIN        v27.8h, v27.8h, v5.8h
        FMIN        v28.8h, v28.8h, v5.8h
        FMIN        v29.8h, v29.8h, v5.8h
        FMIN        v30.8h, v30.8h, v5.8h
        FMIN        v31.8h, v31.8h, v5.8h

        # Store full 4 x 16
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

        # Remainder- 1 halffloat of A (2 bytes)
3:
        LDR         h0,  [x3], 2
        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         h1, [x11], 2
        LDR         h2, [x12], 2
        LDR         h3,  [x4], 2
        FMLA        v24.8h, v20.8h, v0.h[0]
        FMLA        v25.8h, v21.8h, v0.h[0]
        FMLA        v26.8h, v20.8h, v1.h[0]
        FMLA        v27.8h, v21.8h, v1.h[0]
        FMLA        v28.8h, v20.8h, v2.h[0]
        FMLA        v29.8h, v21.8h, v2.h[0]
        FMLA        v30.8h, v20.8h, v3.h[0]
        FMLA        v31.8h, v21.8h, v3.h[0]
        B           2b

        # Store odd width
4:
        TBZ         x1, 3, 5f
        STR         q24, [x6], 16
        MOV         v24.16b, v25.16b
        STR         q26, [x9], 16
        MOV         v26.16b, v27.16b
        STR         q28, [x10], 16
        MOV         v28.16b, v29.16b
        STR         q30, [x7], 16
        MOV         v30.16b, v31.16b

5:
        TBZ         x1, 2, 6f
        STR         d24, [x6], 8
        STR         d26, [x9], 8
        DUP         d24, v24.d[1]
        DUP         d26, v26.d[1]
        STR         d28, [x10], 8
        STR         d30, [x7], 8
        DUP         d28, v28.d[1]
        DUP         d30, v30.d[1]

6:
        TBZ         x1, 1, 7f
        STR         s24,  [x6], 4
        STR         s26,  [x9], 4
        DUP         s24, v24.s[1]
        DUP         s26, v26.s[1]
        STR         s28, [x10], 4
        STR         s30,  [x7], 4
        DUP         s28, v28.s[1]
        DUP         s30, v30.s[1]

7:
        TBZ         x1, 0, 8f
        STR         h24,  [x6]
        STR         h26,  [x9]
        STR         h28, [x10]
        STR         h30,  [x7]
8:
        RET

END_FUNCTION xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
