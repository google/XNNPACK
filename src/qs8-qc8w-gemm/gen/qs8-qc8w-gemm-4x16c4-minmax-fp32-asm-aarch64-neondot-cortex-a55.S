// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     const int8_t* restrict a,  x3
#     size_t a_stride,           x4
#     const void* restrict w,    x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,          [sp] -> x12
#     const union xnn_qs8_qc8w_conv_minmax_params *params)  [sp + 8] -> x11

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3  v0  v4
// A1 x15  v1  v5
// A2 x13  v2  v6
// A3  x4  v3  v7
// B   x5  v8  v9 v10 v11
// C0  x6 v16 v20 v24 v28
// C1  x8 v17 v21 v25 v29
// C2  x9 v18 v22 v26 v30
// C3  x7 v19 v23 v27 v31
// temp x14 for Cortex-A55 loads
// unused v12 v13 v14 v15

BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2

        LDP         x12, x11, [sp]          // cn_stride, params

        ADD         x15, x3, x4             // a1 = a0 + a_stride
        ADD         x8, x6, x7              // c1 = c0 + cm_stride

        STP         d8,  d9, [sp, -32]!

        CSEL        x15, x3, x15, LO        //   a1 = a0
        CSEL        x8, x6,  x8, LO         //   c1 = c0
        ADD         x2, x2, 3               // kc = (kc + 3) & ~3

        ADD         x13, x15, x4            // a2 = a1 + a_stride
        ADD         x9,  x8, x7             // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x13, x15, x13, LS       //   a2 = a1
        CSEL        x9,  x8,  x9, LS        //   c2 = c1
        BIC         x2, x2, 3

        STP         d10, d11, [sp, 16]

        CMP         x0, 4                   // if mr < 4
        ADD         x4, x13, x4             // a3 = a2 + a_stride
        ADD         x7,  x9, x7             // c3 = c2 + cm_stride
        CSEL        x4, x13, x4, LO         //   a3 = a2
        CSEL        x7,  x9, x7, LO         //   c3 = c2

        .p2align    3
0:
        # Load initial bias from w into accumulators
        LDP         q16, q20, [x5], 32
        SUBS        x0, x2, 16              // k = kc - 16
        MOV         v17.16b, v16.16b
        MOV         v18.16b, v16.16b
        LDP         q24, q28, [x5], 32
        MOV         v19.16b, v16.16b
        MOV         v21.16b, v20.16b
        MOV         v22.16b, v20.16b
        MOV         v23.16b, v20.16b
        MOV         v25.16b, v24.16b
        MOV         v26.16b, v24.16b
        MOV         v27.16b, v24.16b
        MOV         v29.16b, v28.16b
        MOV         v30.16b, v28.16b
        MOV         v31.16b, v28.16b

        # Is there at least 16 bytes for prologue/epilogue?
        B.LO        4f

        # prologue - read A and B values for block 0 and 1
        LDR         d0,  [x3], 8
        LDR         q8,  [x5], 16
        LDR         d1, [x15], 8
        LDR         d2, [x13], 8
        LDR         d3,  [x4], 8
        SUBS        x0, x0, 16              // is there 16 for main loop?
        LDR         d9,  [x5], 8
        LDR         x14,  [x5], 8
        # Is there at least 16 bytes for main loop?
        B.LO        2f

        # Main loop - 16 bytes of A in 4 groups.
        # 4 row of 4 vectors wide = 16 sdot instructions for 4 channels
        # 4 LD64 for A
        # 4 LD128 for W. = 2 LD64 + INS.
        # for each 4 sdot, 1 LD64 for A, 2 LD64 for W + INS.

        .p2align    3
1:
        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v0.4b[0]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v1.4b[0]
        INS         v9.d[1], x14
        SDOT        v18.4s,  v8.16b, v2.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v19.4s,  v8.16b, v3.4b[0]
        LDR         d4,  [x3], 8

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v0.4b[0]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v1.4b[0]
        INS         v10.d[1], x14
        SDOT        v22.4s,  v9.16b, v2.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v23.4s,  v9.16b, v3.4b[0]
        LDR         d5, [x15], 8

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v0.4b[0]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v1.4b[0]
        INS         v11.d[1], x14
        SDOT        v26.4s, v10.16b, v2.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v27.4s, v10.16b, v3.4b[0]
        LDR         d6, [x13], 8

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v0.4b[0]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v1.4b[0]
        INS         v8.d[1], x14
        SDOT        v30.4s, v11.16b, v2.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v31.4s, v11.16b, v3.4b[0]
        LDR         d7,  [x4], 8

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v0.4b[1]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v1.4b[1]
        INS         v9.d[1], x14
        SDOT        v18.4s,  v8.16b, v2.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v19.4s,  v8.16b, v3.4b[1]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v0.4b[1]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v1.4b[1]
        INS         v10.d[1], x14
        SDOT        v22.4s,  v9.16b, v2.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v23.4s,  v9.16b, v3.4b[1]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v0.4b[1]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v1.4b[1]
        INS         v11.d[1], x14
        SDOT        v26.4s, v10.16b, v2.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v27.4s, v10.16b, v3.4b[1]

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v0.4b[1]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v1.4b[1]
        INS         v8.d[1], x14
        SDOT        v30.4s, v11.16b, v2.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v31.4s, v11.16b, v3.4b[1]

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v4.4b[0]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v5.4b[0]
        INS         v9.d[1], x14
        SDOT        v18.4s,  v8.16b, v6.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v19.4s,  v8.16b, v7.4b[0]
        LDR         d0,  [x3], 8

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v4.4b[0]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v5.4b[0]
        INS         v10.d[1], x14
        SDOT        v22.4s,  v9.16b, v6.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v23.4s,  v9.16b, v7.4b[0]
        LDR         d1, [x15], 8

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v4.4b[0]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v5.4b[0]
        INS         v11.d[1], x14
        SDOT        v26.4s, v10.16b, v6.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v27.4s, v10.16b, v7.4b[0]
        LDR         d2, [x13], 8

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v4.4b[0]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v5.4b[0]
        INS         v8.d[1], x14
        SDOT        v30.4s, v11.16b, v6.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v31.4s, v11.16b, v7.4b[0]
        LDR         d3,  [x4], 8

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v4.4b[1]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v5.4b[1]
        INS         v9.d[1], x14
        SDOT        v18.4s,  v8.16b, v6.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v19.4s,  v8.16b, v7.4b[1]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v4.4b[1]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v5.4b[1]
        INS         v10.d[1], x14
        SDOT        v22.4s,  v9.16b, v6.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v23.4s,  v9.16b, v7.4b[1]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v4.4b[1]
        LDR         d8,  [x5], 8            // First B values for block 0 and 1
        SDOT        v25.4s, v10.16b, v5.4b[1]
        INS         v11.d[1], x14
        SDOT        v26.4s, v10.16b, v6.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v27.4s, v10.16b, v7.4b[1]
        SUBS        x0, x0, 16

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v4.4b[1]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v5.4b[1]
        INS         v8.d[1], x14
        SDOT        v30.4s, v11.16b, v6.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v31.4s, v11.16b, v7.4b[1]
        B.HS        1b

        # Epilogue.  Same as main loop but no preloads in final group
2:
        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v0.4b[0]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v1.4b[0]
        INS         v9.d[1], x14
        SDOT        v18.4s,  v8.16b, v2.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v19.4s,  v8.16b, v3.4b[0]
        LDR         d4,  [x3], 8

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v0.4b[0]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v1.4b[0]
        INS         v10.d[1], x14
        SDOT        v22.4s,  v9.16b, v2.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v23.4s,  v9.16b, v3.4b[0]
        LDR         d5, [x15], 8

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v0.4b[0]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v1.4b[0]
        INS         v11.d[1], x14
        SDOT        v26.4s, v10.16b, v2.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v27.4s, v10.16b, v3.4b[0]
        LDR         d6, [x13], 8

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v0.4b[0]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v1.4b[0]
        INS         v8.d[1], x14
        SDOT        v30.4s, v11.16b, v2.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v31.4s, v11.16b, v3.4b[0]
        LDR         d7,  [x4], 8

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v0.4b[1]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v1.4b[1]
        INS         v9.d[1], x14
        SDOT        v18.4s,  v8.16b, v2.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v19.4s,  v8.16b, v3.4b[1]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v0.4b[1]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v1.4b[1]
        INS         v10.d[1], x14
        SDOT        v22.4s,  v9.16b, v2.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v23.4s,  v9.16b, v3.4b[1]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v0.4b[1]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v1.4b[1]
        INS         v11.d[1], x14
        SDOT        v26.4s, v10.16b, v2.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v27.4s, v10.16b, v3.4b[1]

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v0.4b[1]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v1.4b[1]
        INS         v8.d[1], x14
        SDOT        v30.4s, v11.16b, v2.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v31.4s, v11.16b, v3.4b[1]

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v4.4b[0]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v5.4b[0]
        INS         v9.d[1], x14
        SDOT        v18.4s,  v8.16b, v6.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v19.4s,  v8.16b, v7.4b[0]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v4.4b[0]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v5.4b[0]
        INS         v10.d[1], x14
        SDOT        v22.4s,  v9.16b, v6.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v23.4s,  v9.16b, v7.4b[0]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v4.4b[0]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v5.4b[0]
        INS         v11.d[1], x14
        SDOT        v26.4s, v10.16b, v6.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v27.4s, v10.16b, v7.4b[0]

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v4.4b[0]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v5.4b[0]
        INS         v8.d[1], x14
        SDOT        v30.4s, v11.16b, v6.4b[0]
        LDR         x14,  [x5], 8
        SDOT        v31.4s, v11.16b, v7.4b[0]

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v4.4b[1]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v5.4b[1]
        INS         v9.d[1], x14
        SDOT        v18.4s,  v8.16b, v6.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v19.4s,  v8.16b, v7.4b[1]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v4.4b[1]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v5.4b[1]
        INS         v10.d[1], x14
        SDOT        v22.4s,  v9.16b, v6.4b[1]
        LDR         x14,  [x5], 8
        SDOT        v23.4s,  v9.16b, v7.4b[1]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v4.4b[1]
        SDOT        v25.4s, v10.16b, v5.4b[1]
        INS         v11.d[1], x14
        SDOT        v26.4s, v10.16b, v6.4b[1]
        SDOT        v27.4s, v10.16b, v7.4b[1]
        AND         x0, x2, 15              // kc remainder 0 to 12

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v4.4b[1]
        SDOT        v29.4s, v11.16b, v5.4b[1]
        SDOT        v30.4s, v11.16b, v6.4b[1]
        SDOT        v31.4s, v11.16b, v7.4b[1]

        # Is there a remainder?- 4 to 12 bytes of A
        CBNZ        x0, 5f

        .p2align    3
3:
        SCVTF       v16.4s, v16.4s
        SCVTF       v17.4s, v17.4s
        # Load per channel scale values from weights
        LDR         q4, [x5], 16
        SCVTF       v18.4s, v18.4s
        SCVTF       v19.4s, v19.4s
        LDR         q5, [x5], 16
        SCVTF       v20.4s, v20.4s
        SCVTF       v21.4s, v21.4s
        SCVTF       v22.4s, v22.4s
        SCVTF       v23.4s, v23.4s
        SCVTF       v24.4s, v24.4s
        SCVTF       v25.4s, v25.4s
        SCVTF       v26.4s, v26.4s
        SCVTF       v27.4s, v27.4s
        SCVTF       v28.4s, v28.4s
        SCVTF       v29.4s, v29.4s
        SCVTF       v30.4s, v30.4s
        SCVTF       v31.4s, v31.4s

        LDR         q6, [x5], 16
        FMUL        v16.4s, v16.4s, v4.4s
        FMUL        v17.4s, v17.4s, v4.4s
        FMUL        v18.4s, v18.4s, v4.4s
        FMUL        v19.4s, v19.4s, v4.4s
        FMUL        v20.4s, v20.4s, v5.4s
        LDR         q4, [x5], 16
        FMUL        v21.4s, v21.4s, v5.4s
        FMUL        v22.4s, v22.4s, v5.4s
        FMUL        v23.4s, v23.4s, v5.4s
        FMUL        v24.4s, v24.4s, v6.4s
        FMUL        v25.4s, v25.4s, v6.4s
        FMUL        v26.4s, v26.4s, v6.4s
        FMUL        v27.4s, v27.4s, v6.4s
        FMUL        v28.4s, v28.4s, v4.4s
        FMUL        v29.4s, v29.4s, v4.4s
        FMUL        v30.4s, v30.4s, v4.4s
        FMUL        v31.4s, v31.4s, v4.4s

        FCVTNS      v16.4s, v16.4s
        FCVTNS      v17.4s, v17.4s
        FCVTNS      v18.4s, v18.4s
        FCVTNS      v19.4s, v19.4s
        FCVTNS      v20.4s, v20.4s
        FCVTNS      v21.4s, v21.4s
        FCVTNS      v22.4s, v22.4s
        FCVTNS      v23.4s, v23.4s
        FCVTNS      v24.4s, v24.4s
        FCVTNS      v25.4s, v25.4s
        FCVTNS      v26.4s, v26.4s
        FCVTNS      v27.4s, v27.4s
        FCVTNS      v28.4s, v28.4s
        FCVTNS      v29.4s, v29.4s
        FCVTNS      v30.4s, v30.4s
        FCVTNS      v31.4s, v31.4s

        SQXTN       v16.4h, v16.4s
        SQXTN       v17.4h, v17.4s
        SQXTN       v18.4h, v18.4s
        SQXTN       v19.4h, v19.4s
        SQXTN       v24.4h, v24.4s
        SQXTN       v25.4h, v25.4s
        SQXTN       v26.4h, v26.4s
        SQXTN       v27.4h, v27.4s
        LD1R        {v6.8h}, [x11], 2       // add bias

        SQXTN2      v16.8h, v20.4s
        SQXTN2      v17.8h, v21.4s
        SQXTN2      v18.8h, v22.4s
        SQXTN2      v19.8h, v23.4s
        SQXTN2      v24.8h, v28.4s
        SQXTN2      v25.8h, v29.4s
        SQXTN2      v26.8h, v30.4s
        SQXTN2      v27.8h, v31.4s

        SQADD       v16.8h, v16.8h, v6.8h
        SQADD       v17.8h, v17.8h, v6.8h
        SQADD       v18.8h, v18.8h, v6.8h
        SQADD       v19.8h, v19.8h, v6.8h
        SQADD       v24.8h, v24.8h, v6.8h
        SQADD       v25.8h, v25.8h, v6.8h
        SQADD       v26.8h, v26.8h, v6.8h
        SQADD       v27.8h, v27.8h, v6.8h
        LD1R        {v4.16b}, [x11], 1      // clamp min value

        SQXTN       v0.8b, v16.8h
        SQXTN       v1.8b, v17.8h
        SQXTN       v2.8b, v18.8h
        SQXTN       v3.8b, v19.8h
        LD1R        {v5.16b}, [x11]         // clamp max value
        SQXTN2      v0.16b, v24.8h
        SQXTN2      v1.16b, v25.8h
        SQXTN2      v2.16b, v26.8h
        SQXTN2      v3.16b, v27.8h
        SUB         x11, x11, 3               // rewind params pointer

        SMAX        v0.16b, v0.16b, v4.16b
        SMAX        v1.16b, v1.16b, v4.16b
        SMAX        v2.16b, v2.16b, v4.16b
        SMAX        v3.16b, v3.16b, v4.16b
        SUBS        x1, x1, 16
        SMIN        v0.16b, v0.16b, v5.16b
        SMIN        v1.16b, v1.16b, v5.16b
        SMIN        v2.16b, v2.16b, v5.16b
        SMIN        v3.16b, v3.16b, v5.16b
        B.LO        6f

        # Store full 4 x 16
        ST1         {v0.16b}, [x6], x12
        SUB         x3,  x3, x2             // a0 -= kc
        ST1         {v1.16b}, [x8], x12
        SUB         x15, x15, x2            // a1 -= kc
        ST1         {v2.16b}, [x9], x12
        SUB         x13, x13, x2            // a2 -= kc
        ST1         {v3.16b}, [x7], x12
        SUB         x4,  x4, x2             // a3 -= kc
        B.NE        0b

        # Restore d8-d11 from stack
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 32
        RET

        # Remainder- 4 to 12 bytes of A
        # Although C4, its safe to read 16 bytes.
        .p2align    3
4:
        AND         x0, x2, 15              // kc remainder 4 to 12
5:
        LDP         q8,  q9,  [x5], 32
        LDP         q10, q11,  [x5], 32
        LD1         {v0.16b},  [x3], x0
        LD1         {v1.16b}, [x15], x0
        LD1         {v2.16b}, [x13], x0
        LD1         {v3.16b},  [x4], x0
        SDOT        v16.4s,  v8.16b, v0.4b[0]
        SDOT        v17.4s,  v8.16b, v1.4b[0]
        SDOT        v18.4s,  v8.16b, v2.4b[0]
        SDOT        v19.4s,  v8.16b, v3.4b[0]
        SDOT        v20.4s,  v9.16b, v0.4b[0]
        SDOT        v21.4s,  v9.16b, v1.4b[0]
        SDOT        v22.4s,  v9.16b, v2.4b[0]
        SDOT        v23.4s,  v9.16b, v3.4b[0]
        SDOT        v24.4s, v10.16b, v0.4b[0]
        SDOT        v25.4s, v10.16b, v1.4b[0]
        SDOT        v26.4s, v10.16b, v2.4b[0]
        SDOT        v27.4s, v10.16b, v3.4b[0]
        SDOT        v28.4s, v11.16b, v0.4b[0]
        SDOT        v29.4s, v11.16b, v1.4b[0]
        SDOT        v30.4s, v11.16b, v2.4b[0]
        SDOT        v31.4s, v11.16b, v3.4b[0]
        CMP         x0, 4
        B.LS        3b
        LDP         q8,  q9,  [x5], 32
        LDP         q10, q11,  [x5], 32
        SDOT        v16.4s,  v8.16b, v0.4b[1]
        SDOT        v17.4s,  v8.16b, v1.4b[1]
        SDOT        v18.4s,  v8.16b, v2.4b[1]
        SDOT        v19.4s,  v8.16b, v3.4b[1]
        SDOT        v20.4s,  v9.16b, v0.4b[1]
        SDOT        v21.4s,  v9.16b, v1.4b[1]
        SDOT        v22.4s,  v9.16b, v2.4b[1]
        SDOT        v23.4s,  v9.16b, v3.4b[1]
        SDOT        v24.4s, v10.16b, v0.4b[1]
        SDOT        v25.4s, v10.16b, v1.4b[1]
        SDOT        v26.4s, v10.16b, v2.4b[1]
        SDOT        v27.4s, v10.16b, v3.4b[1]
        SDOT        v28.4s, v11.16b, v0.4b[1]
        SDOT        v29.4s, v11.16b, v1.4b[1]
        SDOT        v30.4s, v11.16b, v2.4b[1]
        SDOT        v31.4s, v11.16b, v3.4b[1]
        CMP         x0, 8
        B.LS        3b
        LDP         q8,  q9,  [x5], 32
        LDP         q10, q11,  [x5], 32
        SDOT        v16.4s,  v8.16b, v0.4b[2]
        SDOT        v17.4s,  v8.16b, v1.4b[2]
        SDOT        v18.4s,  v8.16b, v2.4b[2]
        SDOT        v19.4s,  v8.16b, v3.4b[2]
        SDOT        v20.4s,  v9.16b, v0.4b[2]
        SDOT        v21.4s,  v9.16b, v1.4b[2]
        SDOT        v22.4s,  v9.16b, v2.4b[2]
        SDOT        v23.4s,  v9.16b, v3.4b[2]
        SDOT        v24.4s, v10.16b, v0.4b[2]
        SDOT        v25.4s, v10.16b, v1.4b[2]
        SDOT        v26.4s, v10.16b, v2.4b[2]
        SDOT        v27.4s, v10.16b, v3.4b[2]
        SDOT        v28.4s, v11.16b, v0.4b[2]
        SDOT        v29.4s, v11.16b, v1.4b[2]
        SDOT        v30.4s, v11.16b, v2.4b[2]
        SDOT        v31.4s, v11.16b, v3.4b[2]
        B           3b

        # Store odd width
        .p2align    3
6:
        TBZ         x1, 3, 7f
        STR         d0, [x6], 8
        STR         d1, [x8], 8
        DUP         d0, v0.d[1]
        DUP         d1, v1.d[1]
        STR         d2, [x9], 8
        STR         d3, [x7], 8
        DUP         d2, v2.d[1]
        DUP         d3, v3.d[1]
7:
        TBZ         x1, 2, 8f
        STR         s0, [x6], 4
        STR         s1, [x8], 4
        DUP         s0, v0.s[1]
        DUP         s1, v1.s[1]
        STR         s2, [x9], 4
        STR         s3, [x7], 4
        DUP         s2, v2.s[1]
        DUP         s3, v3.s[1]
8:
        TBZ         x1, 1, 9f
        STR         h0, [x6], 2
        STR         h1, [x8], 2
        DUP         h0, v0.h[1]
        DUP         h1, v1.h[1]
        STR         h2, [x9], 2
        STR         h3, [x7], 2
        DUP         h2, v2.h[1]
        DUP         h3, v3.h[1]
9:
        TBZ         x1, 0, 10f
        STR         b0, [x6]
        STR         b1, [x8]
        STR         b2, [x9]
        STR         b3, [x7]
10:
        # Restore d8-d11 from stack
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 32
        RET

END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
