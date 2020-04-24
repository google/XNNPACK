// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/1x8-aarch64-neonfp16arith-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f16_gemminc_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64(
#     size_t mr,                (x0) - unused.  mr = 1
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const uint8_t*restrict a, x3
#     size_t a_stride,          (x4) - unused
#     const void*restrict w,    x5
#     uint8_t*restrict c,       x6
#     size_t cm_stride,         (x7) - unused
#     size_t cn_stride,         [sp] -> x14
#     const float*restrict acc,  [sp + 8] -> x15
#     const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 16] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointer
# x3  a0

# C pointer
# x6  c0

# Vector register usage
# A0   v0 v1
# B   v20 v21 v22 v23
# C   v16 v18
# Clamp v4, v5, v6


BEGIN_FUNCTION xnn_f16_gemminc_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64

        # Load cn_stride, acc
        LDP x14, x15, [sp]
        # Load params pointer
        LDR x8, [sp, 16]

        # Load params values
        LD3R {v4.8h, v5.8h, v6.8h}, [x8]
0:
        # Load initial accumulators
        LDR q16, [x15], 16

        MOVI v18.8h, 0  // second set of C for pipelining FMLA

        # Is there at least 4 halffloats (8 bytes)
        SUBS x0, x2, 8  // k = kc - 8

        B.LO 6f

        # Main loop - 4 halffloats of A (8 bytes)
1:
        LDR  d0, [x3], 8
        LDR q20, [x5], 16
        LDR q21, [x5], 16
        LDR q22, [x5], 16
        LDR q23, [x5], 16
        SUBS x0, x0, 8
        FMLA v16.8h, v20.8h, v0.h[0]
        FMLA v18.8h, v21.8h, v0.h[1]
        FMLA v16.8h, v22.8h, v0.h[2]
        FMLA v18.8h, v23.8h, v0.h[3]
        B.HS 1b

        # Is there a remainder?- 2 halffloats of A (4 bytes)
        TBNZ x0, 2, 7f
        # Is there a remainder?- 1 halffloats of A (2 bytes)
        TBNZ x0, 1, 8f

4:
        FADD v16.8h, v16.8h, v18.8h
        SUBS x1, x1, 8

        # Scale and Clamp
        FMUL v16.8h, v16.8h, v4.8h
        FMAX v16.8h, v16.8h, v5.8h
        FMIN v16.8h, v16.8h, v6.8h

        # Store full 1 x 8
        B.LO 10f

        ST1 {v16.16b},  [x6], x14
        SUB  x3,  x3, x2 // a0 -= kc

        B.HI 0b
        RET

6:
        TBZ x0, 2, 8f
7:
        # Remainder- 2 halffloats of A (4 bytes)
        LDR  s0, [x3], 4
        LDR q20, [x5], 16
        LDR q21, [x5], 16
        FMLA v16.8h, v20.8h, v0.h[0]
        FMLA v18.8h, v21.8h, v0.h[1]
        TBZ x0, 1, 4b

8:
        # Remainder- 1 halffloat of A (2 bytes)
        LDR h0,  [x3], 2
        LDR q20, [x5], 16
        FMLA v16.8h, v20.8h, v0.h[0]
        B 4b

        # Store odd channels
10:
        TBZ x1, 2, 11f
        STR d16, [x6], 8
        DUP d16, v16.d[1]

11:
        TBZ x1, 1, 12f
        STR s16, [x6], 4
        DUP s16, v16.s[1]

12:
        TBZ x1, 0, 13f
        STR h16, [x6]
13:
        RET

END_FUNCTION xnn_f16_gemminc_minmax_ukernel_1x8__aarch64_neonfp16arith_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
