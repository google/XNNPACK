// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/1x16c4-aarch64-neondot-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     const int8_t* restrict a,  x3
#     size_t a_stride,           (x4)
#     const void* restrict w,    x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          (x7)
#     size_t cn_stride,          [sp] -> x12
#     const union xnn_qs8_qc8w_conv_minmax_params params)  [sp + 8] -> x11

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3 v0
// B   x5 v4  v5  v6  v7  v16  v17 v18 v19
// C0  x6 v28 v29 v30 v31
// unused v8 v9 v10 v11 v12 v13 v14 v15

BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64
        ADD         x2, x2, 3               // kc = (kc + 3) & ~3
        BIC         x2, x2, 3

        .p2align    3
0:
        # Load initial bias from w into accumulators
        LDP         q28, q29, [x5], 32
        SUBS        x0, x2, 8               // k = kc - 8
        LDP         q30, q31, [x5], 32
        LDR         x11, [sp, 8]            // params

        # Is there at least 8 bytes?
        B.LO        3f

        # Main loop - 8 bytes of A
        .p2align    3
1:
        LDR         d0,  [x3], 8
        LDR         q16, [x5, 0]
        LDR         q17, [x5, 16]
        SDOT        v28.4s, v16.16b, v0.4b[0]
        LDR         q18, [x5, 32]
        SDOT        v29.4s, v17.16b, v0.4b[0]
        LDR         q19, [x5, 48]
        SDOT        v30.4s, v18.16b, v0.4b[0]
        LDR         q4, [x5, 64]
        SDOT        v31.4s, v19.16b, v0.4b[0]
        LDR         q5, [x5, 80]
        SDOT        v28.4s, v4.16b,  v0.4b[1]
        LDR         q6, [x5, 96]
        SDOT        v29.4s, v5.16b,  v0.4b[1]
        LDR         q7, [x5, 112]
        SDOT        v30.4s, v6.16b,  v0.4b[1]
        ADD         x5, x5, 128
        SDOT        v31.4s, v7.16b,  v0.4b[1]
        SUBS        x0, x0, 8
        B.HS        1b

        # Is there a remainder?- 1 to 4 bytes of A
        TBNZ        x0, 2, 3f

2:
        # Load per channel scale values from weights
        SCVTF       v28.4s, v28.4s
        LDR         q4, [x5], 16
        SCVTF       v29.4s, v29.4s
        LDR         q5, [x5], 16
        SCVTF       v30.4s, v30.4s
        LDR         q6, [x5], 16
        SCVTF       v31.4s, v31.4s
        FMUL        v28.4s, v28.4s, v4.4s
        LDR         q4, [x5], 16
        FMUL        v29.4s, v29.4s, v5.4s
        FMUL        v30.4s, v30.4s, v6.4s
        FMUL        v31.4s, v31.4s, v4.4s

        FCVTNS      v28.4s, v28.4s
        FCVTNS      v29.4s, v29.4s
        FCVTNS      v30.4s, v30.4s
        FCVTNS      v31.4s, v31.4s

        LD1R        {v6.8h}, [x11], 2       // add bias
        SQXTN       v0.4h, v28.4s
        SQXTN       v2.4h, v30.4s
        SQXTN2      v0.8h, v29.4s
        SQXTN2      v2.8h, v31.4s

        LD2R        {v4.16b, v5.16b}, [x11] // clamp to min/max
        SQADD       v0.8h, v0.8h, v6.8h
        SQADD       v2.8h, v2.8h, v6.8h
        LDR         x12, [sp]               // cn_stride
        SQXTN       v0.8b, v0.8h
        SQXTN2      v0.16b, v2.8h
        SUBS        x1, x1, 16
        SMAX        v0.16b, v0.16b, v4.16b
        SMIN        v0.16b, v0.16b, v5.16b
        B.LO        4f

        # Store full 1 x 16
        ST1         {v0.16b}, [x6], x12
        SUB         x3,  x3, x2             // a0 -= kc
        B.NE        0b

        RET

        # Remainder - 4 bytes of A
        .p2align    3
3:
        LDR         s0,  [x3], 4
        LDR         q16, [x5, 0]
        LDR         q17, [x5, 16]
        SDOT        v28.4s, v16.16b, v0.4b[0]
        LDR         q18, [x5, 32]
        SDOT        v29.4s, v17.16b, v0.4b[0]
        LDR         q19, [x5, 48]
        SDOT        v30.4s, v18.16b, v0.4b[0]
        ADD         x5, x5, 64
        SDOT        v31.4s, v19.16b, v0.4b[0]
        B           2b

        # Store odd width
        .p2align    3
4:
        TBZ         x1, 3, 5f
        STR         d0, [x6], 8
        DUP         d0, v0.d[1]
5:
        TBZ         x1, 2, 6f
        STR         s0, [x6], 4
        DUP         s0, v0.s[1]
6:
        TBZ         x1, 1, 7f
        STR         h0, [x6], 2
        DUP         h0, v0.h[1]
7:
        TBZ         x1, 0, 8f
        STR         b0, [x6]
8:
        RET

END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
