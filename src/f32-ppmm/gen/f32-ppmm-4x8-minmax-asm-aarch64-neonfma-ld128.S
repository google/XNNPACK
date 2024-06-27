// Auto-generated file. Do not edit!
//   Template: src/f32-ppmm/4x8-aarch64-neonfma-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const uint8_t* restrict a, x3
#     const void* restrict w,    x4
#     uint8_t* restrict c,       x5
#     size_t cm_stride,         x6
#     size_t cn_stride,         x7
#     const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A   x3  v0 v1 v2 v3
# B   x4  v20 v21
# C0  x5  v16 v17
# C1  x9  v18 v19
# C2  x10 v28 v29
# C3  x6  v30 v31
# Clamp v4 v5


BEGIN_FUNCTION xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128

        # Load params pointer
        LDR         x8, [sp]

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x9, x5, x6              // c1 = c0 + cm_stride
        CSEL        x9, x5, x9, LO          //   c1 = c0

        ADD         x10, x9, x6             // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x10, x9, x10, LS        //   c2 = c1

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]

        CMP         x0, 4                   // if mr < 4
        ADD         x6, x10, x6             // c3 = c2 + cm_stride
        CSEL        x6, x10, x6, LO         //   c3 = c2


0:
        # Load initial bias from w into accumulators
        LDR         q16, [x4], 16
        LDR         q17, [x4], 16
        SUBS        x0, x2, 16              // k = kc - 16
        MOV         v18.16b, v16.16b
        MOV         v28.16b, v16.16b
        MOV         v30.16b, v16.16b
        MOV         v19.16b, v17.16b
        MOV         v29.16b, v17.16b
        MOV         v31.16b, v17.16b

        # Is there at least 4 floats (16 bytes)?
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
1:
        LDR         q0, [x3], 16
        LDR         q20, [x4], 16
        LDR         q21, [x4], 16
        LDR         q22, [x4], 16
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v18.4s, v20.4s, v0.s[1]
        LDR         q1, [x3], 16
        FMLA        v28.4s, v20.4s, v0.s[2]
        FMLA        v30.4s, v20.4s, v0.s[3]
        LDR         q23, [x4], 16
        FMLA        v17.4s, v21.4s, v0.s[0]
        FMLA        v19.4s, v21.4s, v0.s[1]
        LDR         q24, [x4], 16
        FMLA        v29.4s, v21.4s, v0.s[2]
        FMLA        v31.4s, v21.4s, v0.s[3]
        LDR         q2, [x3], 16
        FMLA        v16.4s, v22.4s, v1.s[0]
        FMLA        v18.4s, v22.4s, v1.s[1]
        LDR         q25, [x4], 16
        FMLA        v28.4s, v22.4s, v1.s[2]
        FMLA        v30.4s, v22.4s, v1.s[3]
        LDR         q26, [x4], 16
        FMLA        v17.4s, v23.4s, v1.s[0]
        FMLA        v19.4s, v23.4s, v1.s[1]
        LDR         q3, [x3], 16
        FMLA        v29.4s, v23.4s, v1.s[2]
        FMLA        v31.4s, v23.4s, v1.s[3]
        LDR         q27, [x4], 16
        FMLA        v16.4s, v24.4s, v2.s[0]
        FMLA        v18.4s, v24.4s, v2.s[1]
        FMLA        v28.4s, v24.4s, v2.s[2]
        FMLA        v30.4s, v24.4s, v2.s[3]
        FMLA        v17.4s, v25.4s, v2.s[0]
        FMLA        v19.4s, v25.4s, v2.s[1]
        FMLA        v29.4s, v25.4s, v2.s[2]
        FMLA        v31.4s, v25.4s, v2.s[3]
        FMLA        v16.4s, v26.4s, v3.s[0]
        FMLA        v18.4s, v26.4s, v3.s[1]
        FMLA        v28.4s, v26.4s, v3.s[2]
        FMLA        v30.4s, v26.4s, v3.s[3]
        SUBS        x0, x0, 16
        FMLA        v17.4s, v27.4s, v3.s[0]
        FMLA        v19.4s, v27.4s, v3.s[1]
        FMLA        v29.4s, v27.4s, v3.s[2]
        FMLA        v31.4s, v27.4s, v3.s[3]
        B.HS        1b

        TST         x0, 15
        B.NE        3f

2:
        # Clamp
        FMAX        v16.4s, v16.4s, v4.4s
        SUBS        x1, x1, 8
        FMAX        v17.4s, v17.4s, v4.4s
        FMAX        v18.4s, v18.4s, v4.4s
        FMAX        v19.4s, v19.4s, v4.4s
        FMAX        v28.4s, v28.4s, v4.4s
        FMAX        v29.4s, v29.4s, v4.4s
        FMAX        v30.4s, v30.4s, v4.4s
        FMAX        v31.4s, v31.4s, v4.4s
        FMIN        v16.4s, v16.4s, v5.4s
        FMIN        v17.4s, v17.4s, v5.4s
        FMIN        v18.4s, v18.4s, v5.4s
        FMIN        v19.4s, v19.4s, v5.4s
        FMIN        v28.4s, v28.4s, v5.4s
        FMIN        v29.4s, v29.4s, v5.4s
        FMIN        v30.4s, v30.4s, v5.4s
        FMIN        v31.4s, v31.4s, v5.4s

        # Store full 4 x 8
        B.LO        5f

        ST1         {v16.16b, v17.16b},  [x5], x7
        ST1         {v18.16b, v19.16b},  [x9], x7
        SUB         x3,  x3, x2, lsl #2             // a0 -= kc * 4
        ST1         {v28.16b, v29.16b}, [x10], x7
        ST1         {v30.16b, v31.16b},  [x6], x7

        B.HI        0b
        RET

        # Remainder- 2 floats of A (8 bytes)
3:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBZ         x0, 3, 4f

        # Remainder- 2 floats of A (8 bytes)
        LDR         q0, [x3], 16
        LDR         q20, [x4], 16
        LDR         q21, [x4], 16
        LDR         q22, [x4], 16
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v18.4s, v20.4s, v0.s[1]
        LDR         q1, [x3], 16
        FMLA        v28.4s, v20.4s, v0.s[2]
        FMLA        v30.4s, v20.4s, v0.s[3]
        LDR         q23, [x4], 16
        FMLA        v17.4s, v21.4s, v0.s[0]
        FMLA        v19.4s, v21.4s, v0.s[1]
        FMLA        v29.4s, v21.4s, v0.s[2]
        FMLA        v31.4s, v21.4s, v0.s[3]
        FMLA        v16.4s, v22.4s, v1.s[0]
        FMLA        v18.4s, v22.4s, v1.s[1]
        FMLA        v28.4s, v22.4s, v1.s[2]
        FMLA        v30.4s, v22.4s, v1.s[3]
        FMLA        v17.4s, v23.4s, v1.s[0]
        FMLA        v19.4s, v23.4s, v1.s[1]
        FMLA        v29.4s, v23.4s, v1.s[2]
        FMLA        v31.4s, v23.4s, v1.s[3]

        # Is there a remainder?- 1 float of A (4 bytes)
        TBZ         x0, 2, 2b

        # Remainder- 1 float of A (4 bytes)
4:
        LDR         q0, [x3], 16
        LDR         q20, [x4], 16
        LDR         q21, [x4], 16
        FMLA        v16.4s, v20.4s, v0.s[0]
        FMLA        v18.4s, v20.4s, v0.s[1]
        FMLA        v28.4s, v20.4s, v0.s[2]
        FMLA        v30.4s, v20.4s, v0.s[3]
        FMLA        v17.4s, v21.4s, v0.s[0]
        FMLA        v19.4s, v21.4s, v0.s[1]
        FMLA        v29.4s, v21.4s, v0.s[2]
        FMLA        v31.4s, v21.4s, v0.s[3]
        B           2b

        # Store odd width
5:
        TBZ         x1, 2, 6f
        STR         q16, [x5], 16
        MOV         v16.16b, v17.16b
        STR         q18, [x9], 16
        MOV         v18.16b, v19.16b
        STR         q28, [x10], 16
        MOV         v28.16b, v29.16b
        STR         q30, [x6], 16
        MOV         v30.16b, v31.16b

6:
        TBZ         x1, 1, 7f
        STR         d16, [x5], 8
        STR         d18, [x9], 8
        DUP         d16, v16.d[1]
        DUP         d18, v18.d[1]
        STR         d28, [x10], 8
        STR         d30, [x6], 8
        DUP         d28, v28.d[1]
        DUP         d30, v30.d[1]

7:
        TBZ         x1, 0, 8f
        STR         s16,  [x5]
        STR         s18,  [x9]
        STR         s28, [x10]
        STR         s30,  [x6]

8:
        RET

END_FUNCTION xnn_f32_ppmm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
