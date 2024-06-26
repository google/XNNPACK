// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/1x8-aarch64-neonfma-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128(
#     size_t mr,                (x0) - unused.  mr = 1
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          (x4) - unused
#     const void* w,            x5
#     float* c,                 x6
#     size_t cm_stride,         (x7) - unused
#     size_t cn_stride,         [sp] -> x14
#     const xnn_f32_minmax_params* params)  [sp + 8] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x3  v0
# B   x5  v20 v24 v21 v25 v22 v26 v23 v27
# C0  x6  v16 v17
# Clamp v4 v5

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]
0:
        # Load initial bias from w into accumulators
        LDP         q16, q17, [x5], 32
        # Is there at least 4 floats (16 bytes)
        SUBS        x0, x2, 16                  // k = kc - 16
        B.LO        3f


        # Main loop - 4 floats of A (16 bytes)
1:
        LDR         q0, [x3], 16
        LDP         q20, q24, [x5], 32
        LDP         q21, q25, [x5], 32
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v17.4s, v24.4s, v0.s[0]
        FMLA        v16.4s, v21.4s, v0.s[1]
        FMLA        v17.4s, v25.4s, v0.s[1]
        LDP         q22, q26, [x5], 32
        LDP         q23, q27, [x5], 32
        SUBS        x0, x0, 16
        FMLA        v16.4s, v22.4s, v0.s[2]
        FMLA        v17.4s, v26.4s, v0.s[2]
        FMLA        v16.4s, v23.4s, v0.s[3]
        FMLA        v17.4s, v27.4s, v0.s[3]
        B.HS        1b

        # Is there a remainder?- 2 float of A (8 bytes)
        TBNZ        x0, 3, 4f
        # Is there a remainder?- 1 float of A (4 bytes)
        TBNZ        x0, 2, 5f

2:
        SUBS        x1, x1, 8

        # Clamp
        FMAX        v16.4s, v16.4s, v4.4s
        FMAX        v17.4s, v17.4s, v4.4s
        FMIN        v16.4s, v16.4s, v5.4s
        FMIN        v17.4s, v17.4s, v5.4s

        # Store full 1 x 8
        B.LO        6f

        STP         q16, q17, [x6]
        ADD         x6, x6, x14

        SUB         x3,  x3, x2             // a0 -= kc
        B.HI        0b
        RET

3:
        TBZ         x0, 3, 5f

        # Remainder- 2 float of A (4 bytes)
4:
        # Remainder- 2 floats of A (8 bytes)
        LDP         q20, q24, [x5], 32
        LDP         q21, q25, [x5], 32
        LDR         d0, [x3], 8
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v17.4s, v24.4s, v0.s[0]
        FMLA        v16.4s, v21.4s, v0.s[1]
        FMLA        v17.4s, v25.4s, v0.s[1]
        TBZ         x0, 2, 2b

        # Remainder- 1 float of A (4 bytes)
5:
        # Remainder- 2 floats of A (8 bytes)
        LDP         q20, q24, [x5], 32
        LDR         s0, [x3], 4
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v17.4s, v24.4s, v0.s[0]
        B           2b

        # Store odd channels
6:
        TBZ         x1, 2, 7f
        STR         q16, [x6], 16
        MOV         v16.16b, v17.16b

7:
        TBZ         x1, 1, 8f
        STR         d16, [x6], 8
        DUP         d16, v16.d[1]

8:
        TBZ         x1, 0, 9f
        STR         s16, [x6]
9:
        RET

END_FUNCTION xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
