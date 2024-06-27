// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64(
#     size_t mr,                         (x0) - unused.  mr = 1
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const void** restrict a,            x4
#     const void* restrict w,             x5
#     void* restrict c,                   x6
#     size_t cm_stride,                  (x7) - unused
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const void* zero,                  [sp + 16] -> x12
#     const xnn_f16_minmax_params params [sp + 24] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x8 v0
// B   x5 v24 v25 v26 v27 v28 v29 v30 v31
// C0  x6 v16 v17 v18 v19 v20 v21 v22 v23
// clamp  v4, v5

BEGIN_FUNCTION xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64
        # Load cn_stride, a_offset
        LDP         x10, x11, [sp]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Load params values
        LD2R        {v4.8h, v5.8h}, [x8]

0:
        # Load initial bias from w into accumulators
        LDR         q16, [x5], 16
        LDR         q17, [x5], 16
        MOVI        v18.8h, 0               // 4 sets of C for pipelining FMLA
        MOVI        v19.8h, 0
        MOVI        v20.8h, 0
        MOVI        v21.8h, 0
        MOVI        v22.8h, 0
        MOVI        v23.8h, 0

        MOV         x9, x3                  // p = ks

1:
        # Load next A pointer
        LDR         x8, [x4], 8

        CMP         x8, x12                 // if a0 == zero
        ADD         x8, x8, x11             // a0 += a_offset
        CSEL        x8, x12, x8, EQ         //   a0 = zero, else += a0 + a_offset

        # Is there at least 4 halffloats (8 bytes)?
        SUBS        x0, x2, 8               // k = kc - 8
        B.LO        4f

       .p2align 3
        # Main loop - 2 halffloats of A (4 bytes)
2:
        LDR         d0,  [x8], 8
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
        B.HS        2b

        # Is there a remainder?- 1 halffloat of A (2 bytes)
        ANDS        x0, x0, 7
        B.NE        4f

3:
        # ks loop
        SUBS        x9, x9, 8               // ks -= MR * sizeof(void*)
        B.HI        1b

        FADD        v16.8h, v16.8h, v18.8h
        FADD        v17.8h, v17.8h, v19.8h
        FADD        v20.8h, v20.8h, v22.8h
        FADD        v21.8h, v21.8h, v23.8h
        FADD        v16.8h, v16.8h, v20.8h
        FADD        v17.8h, v17.8h, v21.8h

        # Clamp
        FMAX        v16.8h, v16.8h, v4.8h
        FMAX        v17.8h, v17.8h, v4.8h
        FMIN        v16.8h, v16.8h, v5.8h
        FMIN        v17.8h, v17.8h, v5.8h

        # Store full 1 x 16
        SUBS        x1, x1, 16
        B.LO        6f

        STP         q16, q17,  [x6]
        ADD         x6,  x6, x10

        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b
        RET


        # Remainder- 1 to 3 halffloats of A (2 to 6 bytes)
4:
        TBZ         x0, 2, 5f
        LDR         s0,  [x8], 4
        LDR         q24, [x5, 0]
        LDR         q25, [x5, 16]
        LDR         q26, [x5, 32]
        LDR         q27, [x5, 48]
        FMLA        v16.8h, v24.8h, v0.h[0]
        FMLA        v17.8h, v25.8h, v0.h[0]
        FMLA        v18.8h, v26.8h, v0.h[1]
        FMLA        v19.8h, v27.8h, v0.h[1]
        ADD         x5, x5, 64
        TBZ         x0, 1, 3b
5:
        LDR         h0, [x8], 2
        LDR         q24, [x5, 0]
        LDR         q25, [x5, 16]
        FMLA        v16.8h, v24.8h, v0.h[0]
        FMLA        v17.8h, v25.8h, v0.h[0]
        ADD         x5, x5, 32
        B           3b

        # Store odd width
6:
        TBZ         x1, 3, 7f
        STR         q16, [x6], 16
        MOV         v16.16b, v17.16b
7:
        TBZ         x1, 2, 8f
        STR         d16, [x6], 8
        DUP         d16, v16.d[1]
8:
        TBZ         x1, 1, 9f
        STR         s16,  [x6], 4
        DUP         s16, v16.s[1]
9:
        TBZ         x1, 0, 10f
        STR         h16, [x6]
10:
        RET

END_FUNCTION xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
