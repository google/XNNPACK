// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/1x8-aarch64-neonfma-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64(
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
# A0  x3 v0
# B   x5 v20 v21 v22 v23
# C0  x6 v16 v17
# Clamp v4 v5

BEGIN_FUNCTION xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]
0:
        # Load initial bias from w into accumulators
        LDP         q16, q17, [x5], 32
        SUBS        x0, x2, 8               // k = kc - 8
        # Is there at least 2 floats (8 bytes)
        B.LO        3f

        # Main loop - 2 floats of A (8 bytes)
1:
        LDR         d0, [x3], 8
        LDR         q22, [x5], 16           // 16 QC8 weights
        SXTL        v21.8h, v22.8b
        SXTL2       v23.8h, v22.16b
        SXTL        v20.4s, v21.4h
        SXTL        v22.4s, v23.4h
        SXTL2       v21.4s, v21.8h
        SXTL2       v23.4s, v23.8h
        SCVTF       v20.4s, v20.4s
        SCVTF       v21.4s, v21.4s
        SCVTF       v22.4s, v22.4s
        SCVTF       v23.4s, v23.4s
        SUBS        x0, x0, 8
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v17.4s, v21.4s, v0.s[0]
        FMLA        v16.4s, v22.4s, v0.s[1]
        FMLA        v17.4s, v23.4s, v0.s[1]
        B.HS        1b

        # Is there a remainder?- 1 float of A (4 bytes)
        TBNZ        x0, 2, 3f

2:
        # Scale
        LDP         q20, q21, [x5], 32
        FMUL        v16.4s, v16.4s, v20.4s
        FMUL        v17.4s, v17.4s, v21.4s
        SUBS        x1, x1, 8

        # Clamp
        FMAX        v16.4s, v16.4s, v4.4s
        FMAX        v17.4s, v17.4s, v4.4s
        FMIN        v16.4s, v16.4s, v5.4s
        FMIN        v17.4s, v17.4s, v5.4s

        # Store full 1 x 8
        B.LO        4f

        STP         q16, q17, [x6]
        ADD         x6, x6, x14

        SUB         x3,  x3, x2             // a0 -= kc
        B.HI        0b
        RET

3:
        # Remainder- 1 float of A (4 bytes)
        LDR         s0, [x3], 4
        LDR         d21, [x5], 8            // 8 QC8 weights
        SXTL        v21.8h, v21.8b
        SXTL        v20.4s, v21.4h
        SXTL2       v21.4s, v21.8h
        SCVTF       v20.4s, v20.4s
        SCVTF       v21.4s, v21.4s
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v17.4s, v21.4s, v0.s[0]
        B           2b

        # Store odd channels
4:
        TBZ         x1, 2, 5f
        STR         q16, [x6], 16
        MOV         v16.16b, v17.16b

5:
        TBZ         x1, 1, 6f
        STR         d16, [x6], 8
        DUP         d16, v16.d[1]

6:
        TBZ         x1, 0, 7f
        STR         s16, [x6]
7:
        RET

END_FUNCTION xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
