// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/1x8-aarch64-neon-ld128-acc2.S.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2(
#     size_t mr,                (x0) - unused.  mr = 1
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const float* a,           x3
#     size_t a_stride,          (x4) - unused
#     const void* w,            x5
#     float* c,                 x6
#     size_t cm_stride,         (x7) - unused
#     size_t cn_stride,         [sp] -> x14
#     const xnn_f32_qc4w_minmax_params* params)  [sp + 8] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x3 v0
# B   x5 v20 v21 v22 v23
# C0  x6 v16 v17 v18 v19 v26 v27 v28 v29
# Clamp v4 v5
# zerop/mask v6 v7

BEGIN_FUNCTION xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2

        # Load cn_stride, params pointer
        LDP         x14, x8, [sp]

        # Load min/max/zerop values
        LD3R        {v4.4s, v5.4s, v6.4s}, [x8]
        MOVI        v7.8b, 15
0:
        # Load initial bias from w into accumulators
        LDP         q16, q17, [x5], 32
        SUBS        x0, x2, 16                  // k = kc - 16
        MOVI        v18.4s, 0                   // second set of C for pipelining FMUL
        MOVI        v19.4s, 0
        MOVI        v26.4s, 0
        MOVI        v27.4s, 0
        MOVI        v28.4s, 0
        MOVI        v29.4s, 0
        # Is there at least 4 floats (16 bytes)
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
1:

        LDR         q22, [x5], 16
        FADD        v16.4s, v16.4s, v26.4s
        FADD        v17.4s, v17.4s, v27.4s
        FADD        v18.4s, v18.4s, v28.4s
        FADD        v19.4s, v19.4s, v29.4s
        SXTL        v21.8h, v22.8b
        SXTL2       v23.8h, v22.16b
        LDR         q0, [x3], 16
        SXTL        v20.4s, v21.4h
        SXTL        v22.4s, v23.4h
        SXTL2       v21.4s, v21.8h
        SXTL2       v23.4s, v23.8h
        SCVTF       v20.4s, v20.4s
        SCVTF       v21.4s, v21.4s
        SCVTF       v22.4s, v22.4s
        SCVTF       v23.4s, v23.4s
        FMUL        v26.4s, v20.4s, v0.s[0]
        FMUL        v27.4s, v21.4s, v0.s[0]
        FMUL        v28.4s, v22.4s, v0.s[1]
        FMUL        v29.4s, v23.4s, v0.s[1]

        LDR         q22, [x5], 16
        FADD        v16.4s, v16.4s, v26.4s
        FADD        v17.4s, v17.4s, v27.4s
        FADD        v18.4s, v18.4s, v28.4s
        FADD        v19.4s, v19.4s, v29.4s
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
        SUBS        x0, x0, 16
        FMUL        v26.4s, v20.4s, v0.s[2]
        FMUL        v27.4s, v21.4s, v0.s[2]
        FMUL        v28.4s, v22.4s, v0.s[3]
        FMUL        v29.4s, v23.4s, v0.s[3]
        B.HS        1b

        FADD        v16.4s, v16.4s, v26.4s
        FADD        v17.4s, v17.4s, v27.4s
        FADD        v18.4s, v18.4s, v28.4s
        FADD        v19.4s, v19.4s, v29.4s

        # Is there a remainder?- 2 float of A (8 bytes)
        TBNZ        x0, 3, 4f
        # Is there a remainder?- 1 float of A (4 bytes)
        TBNZ        x0, 2, 5f

2:
        # Load Scale
        LDP         q24, q25, [x5], 32
        FADD        v16.4s, v16.4s, v18.4s
        FADD        v17.4s, v17.4s, v19.4s
        # Scale
        FMUL        v16.4s, v16.4s, v24.4s
        FMUL        v17.4s, v17.4s, v25.4s
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
        LDR         d0, [x3], 8
        LDR         d22, [x5], 8            // 16 QC4 weights
        AND         v21.8b, v22.8b, v7.8b   // first set of 8 weights
        USHR        v23.8b, v22.8b, 4       // second set of 8 weights
        SADDW       v21.8h, v6.8h, v21.8b
        SADDW       v23.8h, v6.8h, v23.8b
        SXTL        v20.4s, v21.4h
        SXTL        v22.4s, v23.4h
        SXTL2       v21.4s, v21.8h
        SXTL2       v23.4s, v23.8h
        SCVTF       v20.4s, v20.4s
        SCVTF       v21.4s, v21.4s
        SCVTF       v22.4s, v22.4s
        SCVTF       v23.4s, v23.4s
        FMUL        v26.4s, v20.4s, v0.s[0]
        FMUL        v27.4s, v21.4s, v0.s[0]
        FMUL        v28.4s, v22.4s, v0.s[1]
        FMUL        v29.4s, v23.4s, v0.s[1]
        FADD        v16.4s, v16.4s, v26.4s
        FADD        v17.4s, v17.4s, v27.4s
        FADD        v18.4s, v18.4s, v28.4s
        FADD        v19.4s, v19.4s, v29.4s
        TBZ         x0, 2, 2b

5:
        # Remainder- 1 float of A (4 bytes)
        LDR         s0, [x3], 4
         LDR         d21, [x5], 8            // 8 QC4 weights
         SADDW       v21.8h, v6.8h, v21.8b
         SXTL        v20.4s, v21.4h
         SXTL2       v21.4s, v21.8h
         SCVTF       v20.4s, v20.4s
         SCVTF       v21.4s, v21.4s
        FMUL        v26.4s, v20.4s, v0.s[0]
        FMUL        v27.4s, v21.4s, v0.s[0]
        FADD        v16.4s, v16.4s, v26.4s
        FADD        v17.4s, v17.4s, v27.4s
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

END_FUNCTION xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
