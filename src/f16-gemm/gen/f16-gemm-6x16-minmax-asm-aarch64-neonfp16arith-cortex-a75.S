// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/6x16-aarch64-neonfp16arith-cortex-a75.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const void* restrict a,    x3
#     size_t a_stride,          x4
#     const void* restrict w,    x5
#     void* restrict c,          x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x8

#     const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3 v0
// A1  x9 v1
// A2 x10 v2
// A3 x11 v3
// A4 x12 v4
// A5  x4 v5
// B   x5 v16 v17 v18 v19
// C0  x6  v20 v21
// C1 x16  v22 v23
// C2 x17  v24 v25
// C3 x14  v26 v27
// C4 x13  v28 v29
// C5  x7  v30 v31
// clamp  v6, (v4), (v5)
// unused     v7
// unused A   v8 v9 v10 v11
// unused B   v12 v13 v14 v15

BEGIN_FUNCTION xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75

        # Load params pointer
        LDR         x8, [sp, 8]

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x9, x3, x4              // a1 = a0 + a_stride
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x9, x3, x9, LO          //   a1 = a0
        CSEL        x16, x6, x16, LO        //   c1 = c0

        # Load params
        LDR         s6, [x8]

        ADD         x10, x9, x4             // a2 = a1 + a_stride
        ADD         x17, x16, x7            // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x10, x9, x10, LS        //   a2 = a1
        CSEL        x17, x16, x17, LS       //   c2 = c1

        CMP         x0, 4                   // if mr < 4
        ADD         x11, x10, x4            // a3 = a2 + a_stride
        ADD         x14, x17, x7            // c3 = c2 + cm_stride
        CSEL        x11, x10, x11, LO       //   a3 = a2
        CSEL        x14, x17, x14, LO       //   c3 = c2

        ADD         x12, x11, x4            // a4 = a3 + a_stride
        ADD         x13, x14, x7            // c4 = c3 + cm_stride
                                            // if mr <= 4
        CSEL        x12, x11, x12, LS       //   a4 = a3
        CSEL        x13, x14, x13, LS       //   c4 = c3

        CMP         x0, 6                   // if mr < 6
        ADD         x4, x12, x4             // a5 = a4 + a_stride
        ADD         x7, x13, x7             // c5 = c4 + cm_stride
        CSEL        x4, x12, x4, LO         //   a5 = a4
        CSEL        x7, x13, x7, LO         //   c5 = c4

        LDR         x8, [sp]                // load cn_stride

0:
        # Load initial bias from w into accumulators
        LDP         q20, q21, [x5], 32
        MOV         v22.16b, v20.16b
        MOV         v23.16b, v21.16b
        MOV         v24.16b, v20.16b
        MOV         v25.16b, v21.16b
        MOV         v26.16b, v20.16b
        MOV         v27.16b, v21.16b
        MOV         v28.16b, v20.16b
        MOV         v29.16b, v21.16b
        MOV         v30.16b, v20.16b
        MOV         v31.16b, v21.16b

        # Is there at least 4 halffloats (8 bytes)?
        SUBS        x0, x2, 8               // k = kc - 8
        B.LO        4f

        # Prologue - load 4 A and 2 B

        LDR         d0,  [x3], 8                // A0
        LDR         q16, [x5], 16               // B0
        LDR         q17, [x5], 16               // B1
        LDR         d1,  [x9], 8                // A1
        LDR         d2, [x10], 8                // A2
        LDR         d3, [x11], 8                // A3

        # Is there at least 4 halffloats for main loop?
        SUBS        x0, x0, 8
        B.LO        2f

       .p2align 3
        # Main loop - 4 halffloats of A (8 bytes)
        # 48 FMA + 6 ld32 A + 8 LDR B
1:
        FMLA        v20.8h, v16.8h,  v0.h[0]
        FMLA        v21.8h, v17.8h,  v0.h[0]
        LDR         d4, [x12], 8                // A4
        FMLA        v22.8h, v16.8h,  v1.h[0]
        FMLA        v23.8h, v17.8h,  v1.h[0]
        LDR         d5,  [x4], 8                // A5
        FMLA        v24.8h, v16.8h,  v2.h[0]
        FMLA        v25.8h, v17.8h,  v2.h[0]
        LDR         q18, [x5], 16               // B2
        FMLA        v26.8h, v16.8h,  v3.h[0]
        FMLA        v27.8h, v17.8h,  v3.h[0]
        LDR         q19, [x5], 16               // B3
        FMLA        v28.8h, v16.8h,  v4.h[0]
        FMLA        v29.8h, v17.8h,  v4.h[0]
        FMLA        v30.8h, v16.8h,  v5.h[0]
        FMLA        v31.8h, v17.8h,  v5.h[0]
        SUBS        x0, x0, 8

        FMLA        v20.8h, v18.8h,  v0.h[1]
        FMLA        v21.8h, v19.8h,  v0.h[1]
        LDR         q16, [x5], 16               // B4
        FMLA        v22.8h, v18.8h,  v1.h[1]
        FMLA        v23.8h, v19.8h,  v1.h[1]
        LDR         q17, [x5], 16               // B5
        FMLA        v24.8h, v18.8h,  v2.h[1]
        FMLA        v25.8h, v19.8h,  v2.h[1]
        FMLA        v26.8h, v18.8h,  v3.h[1]
        FMLA        v27.8h, v19.8h,  v3.h[1]
        FMLA        v28.8h, v18.8h,  v4.h[1]
        FMLA        v29.8h, v19.8h,  v4.h[1]
        FMLA        v30.8h, v18.8h,  v5.h[1]
        FMLA        v31.8h, v19.8h,  v5.h[1]

        FMLA        v20.8h, v16.8h,  v0.h[2]
        FMLA        v21.8h, v17.8h,  v0.h[2]
        LDR         q18, [x5], 16               // B6
        FMLA        v22.8h, v16.8h,  v1.h[2]
        FMLA        v23.8h, v17.8h,  v1.h[2]
        LDR         q19, [x5], 16               // B7
        FMLA        v24.8h, v16.8h,  v2.h[2]
        FMLA        v25.8h, v17.8h,  v2.h[2]
        FMLA        v26.8h, v16.8h,  v3.h[2]
        FMLA        v27.8h, v17.8h,  v3.h[2]
        FMLA        v28.8h, v16.8h,  v4.h[2]
        FMLA        v29.8h, v17.8h,  v4.h[2]
        FMLA        v30.8h, v16.8h,  v5.h[2]
        FMLA        v31.8h, v17.8h,  v5.h[2]

        LDR         q16, [x5], 16               // B0
        FMLA        v20.8h, v18.8h,  v0.h[3]
        FMLA        v21.8h, v19.8h,  v0.h[3]
        LDR         q17, [x5], 16               // B1
        FMLA        v22.8h, v18.8h,  v1.h[3]
        FMLA        v23.8h, v19.8h,  v1.h[3]
        LDR         d0,  [x3], 8                // A0
        FMLA        v24.8h, v18.8h,  v2.h[3]
        FMLA        v25.8h, v19.8h,  v2.h[3]
        LDR         d1,  [x9], 8                // A1
        FMLA        v26.8h, v18.8h,  v3.h[3]
        FMLA        v27.8h, v19.8h,  v3.h[3]
        LDR         d2, [x10], 8                // A2
        FMLA        v28.8h, v18.8h,  v4.h[3]
        FMLA        v29.8h, v19.8h,  v4.h[3]
        LDR         d3, [x11], 8                // A3
        FMLA        v30.8h, v18.8h,  v5.h[3]
        FMLA        v31.8h, v19.8h,  v5.h[3]
        B.HS        1b

        # Epilogue - same as main loop but no loads for next loop
2:
        FMLA        v20.8h, v16.8h,  v0.h[0]
        FMLA        v21.8h, v17.8h,  v0.h[0]
        LDR         d4, [x12], 8                // A4
        FMLA        v22.8h, v16.8h,  v1.h[0]
        FMLA        v23.8h, v17.8h,  v1.h[0]
        LDR         d5,  [x4], 8                // A5
        FMLA        v24.8h, v16.8h,  v2.h[0]
        FMLA        v25.8h, v17.8h,  v2.h[0]
        LDR         q18, [x5], 16               // B2
        FMLA        v26.8h, v16.8h,  v3.h[0]
        FMLA        v27.8h, v17.8h,  v3.h[0]
        LDR         q19, [x5], 16               // B3
        FMLA        v28.8h, v16.8h,  v4.h[0]
        FMLA        v29.8h, v17.8h,  v4.h[0]
        FMLA        v30.8h, v16.8h,  v5.h[0]
        FMLA        v31.8h, v17.8h,  v5.h[0]
        ADDS        x0, x0, 8

        FMLA        v20.8h, v18.8h,  v0.h[1]
        FMLA        v21.8h, v19.8h,  v0.h[1]
        LDR         q16, [x5], 16               // B4
        FMLA        v22.8h, v18.8h,  v1.h[1]
        FMLA        v23.8h, v19.8h,  v1.h[1]
        LDR         q17, [x5], 16               // B5
        FMLA        v24.8h, v18.8h,  v2.h[1]
        FMLA        v25.8h, v19.8h,  v2.h[1]
        FMLA        v26.8h, v18.8h,  v3.h[1]
        FMLA        v27.8h, v19.8h,  v3.h[1]
        FMLA        v28.8h, v18.8h,  v4.h[1]
        FMLA        v29.8h, v19.8h,  v4.h[1]
        FMLA        v30.8h, v18.8h,  v5.h[1]
        FMLA        v31.8h, v19.8h,  v5.h[1]

        FMLA        v20.8h, v16.8h,  v0.h[2]
        FMLA        v21.8h, v17.8h,  v0.h[2]
        LDR         q18, [x5], 16               // B6
        FMLA        v22.8h, v16.8h,  v1.h[2]
        FMLA        v23.8h, v17.8h,  v1.h[2]
        LDR         q19, [x5], 16               // B7
        FMLA        v24.8h, v16.8h,  v2.h[2]
        FMLA        v25.8h, v17.8h,  v2.h[2]
        FMLA        v26.8h, v16.8h,  v3.h[2]
        FMLA        v27.8h, v17.8h,  v3.h[2]
        FMLA        v28.8h, v16.8h,  v4.h[2]
        FMLA        v29.8h, v17.8h,  v4.h[2]
        FMLA        v30.8h, v16.8h,  v5.h[2]
        FMLA        v31.8h, v17.8h,  v5.h[2]

        FMLA        v20.8h, v18.8h,  v0.h[3]
        FMLA        v21.8h, v19.8h,  v0.h[3]
        FMLA        v22.8h, v18.8h,  v1.h[3]
        FMLA        v23.8h, v19.8h,  v1.h[3]
        FMLA        v24.8h, v18.8h,  v2.h[3]
        FMLA        v25.8h, v19.8h,  v2.h[3]
        FMLA        v26.8h, v18.8h,  v3.h[3]
        FMLA        v27.8h, v19.8h,  v3.h[3]
        FMLA        v28.8h, v18.8h,  v4.h[3]
        FMLA        v29.8h, v19.8h,  v4.h[3]
        FMLA        v30.8h, v18.8h,  v5.h[3]
        FMLA        v31.8h, v19.8h,  v5.h[3]

        # Is there a remainder?- 1-3 halffloats of A (2-6 bytes)
        B.NE        4f

3:
        # Clamp
        DUP         v4.8h, v6.h[0]
        DUP         v5.8h, v6.h[1]
        FMAX        v20.8h, v20.8h, v4.8h
        FMAX        v21.8h, v21.8h, v4.8h
        FMAX        v22.8h, v22.8h, v4.8h
        FMAX        v23.8h, v23.8h, v4.8h
        FMAX        v24.8h, v24.8h, v4.8h
        FMAX        v25.8h, v25.8h, v4.8h
        FMAX        v26.8h, v26.8h, v4.8h
        FMAX        v27.8h, v27.8h, v4.8h
        FMAX        v28.8h, v28.8h, v4.8h
        FMAX        v29.8h, v29.8h, v4.8h
        FMAX        v30.8h, v30.8h, v4.8h
        FMAX        v31.8h, v31.8h, v4.8h
        SUBS        x1, x1, 16
        FMIN        v20.8h, v20.8h, v5.8h
        FMIN        v21.8h, v21.8h, v5.8h
        FMIN        v22.8h, v22.8h, v5.8h
        FMIN        v23.8h, v23.8h, v5.8h
        FMIN        v24.8h, v24.8h, v5.8h
        FMIN        v25.8h, v25.8h, v5.8h
        FMIN        v26.8h, v26.8h, v5.8h
        FMIN        v27.8h, v27.8h, v5.8h
        FMIN        v28.8h, v28.8h, v5.8h
        FMIN        v29.8h, v29.8h, v5.8h
        FMIN        v30.8h, v30.8h, v5.8h
        FMIN        v31.8h, v31.8h, v5.8h

        # Store full 6 x 16
        B.LO        6f

        ST1         {v20.16b, v21.16b},  [x6], x8
        SUB         x3,  x3, x2             // a0 -= kc
        ST1         {v22.16b, v23.16b}, [x16], x8
        SUB         x9,  x9, x2             // a1 -= kc
        ST1         {v24.16b, v25.16b}, [x17], x8
        SUB         x10, x10, x2            // a2 -= kc
        ST1         {v26.16b, v27.16b}, [x14], x8
        SUB         x11, x11, x2            // a3 -= kc
        ST1         {v28.16b, v29.16b}, [x13], x8
        SUB         x12, x12, x2            // a4 -= kc
        ST1         {v30.16b, v31.16b},  [x7], x8
        SUB         x4,  x4, x2             // a5 -= kc

        B.HI        0b
        RET

        # Remainder- 1-3 halffloats of A (2-6 bytes)
4:
        TBZ         x0, 2, 5f
        LDR         s0,  [x3], 4
        LDR         q16, [x5], 16
        LDR         q17, [x5], 16
        LDR         s1,  [x9], 4
        LDR         s2, [x10], 4
        LDR         s3, [x11], 4
        LDR         s4, [x12], 4
        LDR         s5,  [x4], 4
        LDR         q18, [x5], 16
        LDR         q19, [x5], 16
        FMLA        v20.8h, v16.8h,  v0.h[0]
        FMLA        v22.8h, v16.8h,  v1.h[0]
        FMLA        v24.8h, v16.8h,  v2.h[0]
        FMLA        v26.8h, v16.8h,  v3.h[0]
        FMLA        v28.8h, v16.8h,  v4.h[0]
        FMLA        v30.8h, v16.8h,  v5.h[0]
        FMLA        v21.8h, v17.8h,  v0.h[0]
        FMLA        v23.8h, v17.8h,  v1.h[0]
        FMLA        v25.8h, v17.8h,  v2.h[0]
        FMLA        v27.8h, v17.8h,  v3.h[0]
        FMLA        v29.8h, v17.8h,  v4.h[0]
        FMLA        v31.8h, v17.8h,  v5.h[0]

        FMLA        v20.8h, v18.8h,  v0.h[1]
        FMLA        v22.8h, v18.8h,  v1.h[1]
        FMLA        v24.8h, v18.8h,  v2.h[1]
        FMLA        v26.8h, v18.8h,  v3.h[1]
        FMLA        v28.8h, v18.8h,  v4.h[1]
        FMLA        v30.8h, v18.8h,  v5.h[1]
        FMLA        v21.8h, v19.8h,  v0.h[1]
        FMLA        v23.8h, v19.8h,  v1.h[1]
        FMLA        v25.8h, v19.8h,  v2.h[1]
        FMLA        v27.8h, v19.8h,  v3.h[1]
        FMLA        v29.8h, v19.8h,  v4.h[1]
        FMLA        v31.8h, v19.8h,  v5.h[1]
        TBZ         x0, 1, 3b

5:
        LDR         h0,  [x3], 2
        LDR         q16, [x5], 16
        LDR         q17, [x5], 16
        LDR         h1,  [x9], 2
        LDR         h2, [x10], 2
        LDR         h3, [x11], 2
        LDR         h4, [x12], 2
        LDR         h5,  [x4], 2
        FMLA        v20.8h, v16.8h,  v0.h[0]
        FMLA        v22.8h, v16.8h,  v1.h[0]
        FMLA        v24.8h, v16.8h,  v2.h[0]
        FMLA        v26.8h, v16.8h,  v3.h[0]
        FMLA        v28.8h, v16.8h,  v4.h[0]
        FMLA        v30.8h, v16.8h,  v5.h[0]
        FMLA        v21.8h, v17.8h,  v0.h[0]
        FMLA        v23.8h, v17.8h,  v1.h[0]
        FMLA        v25.8h, v17.8h,  v2.h[0]
        FMLA        v27.8h, v17.8h,  v3.h[0]
        FMLA        v29.8h, v17.8h,  v4.h[0]
        FMLA        v31.8h, v17.8h,  v5.h[0]
        B           3b

        # Store odd width
6:
        TBZ         x1, 3, 7f
        STR         q20,  [x6], 16
        MOV         v20.16b, v21.16b
        STR         q22, [x16], 16
        MOV         v22.16b, v23.16b
        STR         q24, [x17], 16
        MOV         v24.16b, v25.16b
        STR         q26, [x14], 16
        MOV         v26.16b, v27.16b
        STR         q28, [x13], 16
        MOV         v28.16b, v29.16b
        STR         q30,  [x7], 16
        MOV         v30.16b, v31.16b

7:
        TBZ         x1, 2, 8f
        STR         d20,  [x6], 8
        STR         d22, [x16], 8
        DUP         d20, v20.d[1]
        DUP         d22, v22.d[1]
        STR         d24, [x17], 8
        STR         d26, [x14], 8
        DUP         d24, v24.d[1]
        DUP         d26, v26.d[1]
        STR         d28, [x13], 8
        STR         d30,  [x7], 8
        DUP         d28, v28.d[1]
        DUP         d30, v30.d[1]

8:
        TBZ         x1, 1, 9f
        STR         s20,  [x6], 4
        STR         s22, [x16], 4
        DUP         s20, v20.s[1]
        DUP         s22, v22.s[1]
        STR         s24, [x17], 4
        STR         s26, [x14], 4
        DUP         s24, v24.s[1]
        DUP         s26, v26.s[1]
        STR         s28, [x13], 4
        STR         s30,  [x7], 4
        DUP         s28, v28.s[1]
        DUP         s30, v30.s[1]

9:
        TBZ         x1, 0, 10f
        STR         h20,  [x6]
        STR         h22, [x16]
        STR         h24, [x17]
        STR         h26, [x14]
        STR         h28, [x13]
        STR         h30,  [x7]
10:
        RET

END_FUNCTION xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
