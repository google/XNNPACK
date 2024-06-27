// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/1x16-aarch64-neonfp16arith-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64(
#     size_t mr,                (x0) - unused.  mr = 1
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const void* restrict a,    x3
#     size_t a_stride,          (x4) - unused
#     const void* restrict w,    x5
#     void* restrict c,          x6
#     size_t cm_stride,         (x7) - unused
#     size_t cn_stride,         [sp] -> x14
#     const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x8 v0
// B   x5 v24 v25 v26 v27 v28 v29 v30 v31
// C0  x6 v16 v17 v18 v19 v20 v21 v22 v23
// clamp  v4, v5

BEGIN_FUNCTION xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64
        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Load params values
        LD2R        {v4.8h, v5.8h}, [x8]
0:
        # Load initial bias from w into accumulators
        LDP         q16, q17, [x5], 32
        MOVI        v18.8h, 0               // 4 sets of C for pipelining FMLA
        MOVI        v19.8h, 0
        MOVI        v20.8h, 0
        MOVI        v21.8h, 0
        MOVI        v22.8h, 0
        MOVI        v23.8h, 0

        # Is there at least 4 halffloats (8 bytes)
        SUBS        x0, x2, 8               // k = kc - 8
        B.LO        3f

       .p2align 3
        # Main loop - 4 halffloats of A (8 bytes)
1:
        LDR         d0, [x3], 8
        LDR         q24, [x5, 0]
        LDR         q25, [x5, 16]
        LDR         q26, [x5, 32]
        LDR         q27, [x5, 48]
        LDR         q28, [x5, 64]
        LDR         q29, [x5, 80]
        LDR         q30, [x5, 96]
        LDR         q31, [x5, 112]
        SUBS        x0, x0, 8
        FMLA        v16.8h, v24.8h, v0.h[0]
        FMLA        v17.8h, v25.8h, v0.h[0]
        FMLA        v18.8h, v26.8h, v0.h[1]
        FMLA        v19.8h, v27.8h, v0.h[1]
        FMLA        v20.8h, v28.8h, v0.h[2]
        FMLA        v21.8h, v29.8h, v0.h[2]
        FMLA        v22.8h, v30.8h, v0.h[3]
        FMLA        v23.8h, v31.8h, v0.h[3]
        ADD         x5, x5, 128
        B.HS        1b

        # Is there a remainder- 1 to 3 halffloats of A (2 to 6 bytes)
        ANDS        x0, x0, 7
        B.NE        3f

2:
        FADD        v16.8h, v16.8h, v18.8h
        FADD        v17.8h, v17.8h, v19.8h
        FADD        v20.8h, v20.8h, v22.8h
        FADD        v21.8h, v21.8h, v23.8h
        FADD        v16.8h, v16.8h, v20.8h
        FADD        v17.8h, v17.8h, v21.8h
        SUBS        x1, x1, 16

        # Clamp
        FMAX        v16.8h, v16.8h, v4.8h
        FMAX        v17.8h, v17.8h, v4.8h
        FMIN        v16.8h, v16.8h, v5.8h
        FMIN        v17.8h, v17.8h, v5.8h

        # Store full 1 x 16
        B.LO        5f

        STP         q16, q17, [x6]
        ADD         x6, x6, x14

        SUB         x3,  x3, x2             // a0 -= kc

        B.HI        0b

        RET

        # Remainder- 1 to 3 halffloats of A (2 to 6 bytes)
3:
        TBZ         x0, 2, 4f
        LDR         s0, [x3], 4
        LDR         q24, [x5, 0]
        LDR         q25, [x5, 16]
        LDR         q26, [x5, 32]
        LDR         q27, [x5, 48]
        FMLA        v16.8h, v24.8h, v0.h[0]
        FMLA        v17.8h, v25.8h, v0.h[0]
        FMLA        v18.8h, v26.8h, v0.h[1]
        FMLA        v19.8h, v27.8h, v0.h[1]
        ADD         x5, x5, 64
        TBZ         x0, 1, 2b

4:
        LDR         h0, [x3], 2
        LDR         q24, [x5, 0]
        LDR         q25, [x5, 16]
        FMLA        v16.8h, v24.8h, v0.h[0]
        FMLA        v17.8h, v25.8h, v0.h[0]
        ADD         x5, x5, 32
        B           2b

        # Store odd channels
5:
        TBZ         x1, 3, 6f
        STR         q16, [x6], 16
        MOV         v16.16b, v17.16b
6:
        TBZ         x1, 2, 7f
        STR         d16, [x6], 8
        DUP         d16, v16.d[1]
7:
        TBZ         x1, 1, 8f
        STR         s16, [x6], 4
        DUP         s16, v16.s[1]
8:
        TBZ         x1, 0, 9f
        STR         h16, [x6]
9:
        RET

END_FUNCTION xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
