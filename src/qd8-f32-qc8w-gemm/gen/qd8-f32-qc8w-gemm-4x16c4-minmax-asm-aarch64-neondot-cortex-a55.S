// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     const int8_t* restrict a,  x3
#     size_t a_stride,           x4
#     const void* restrict w,    x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,          [sp] -> x12
#     const union xnn_f32_minmax_params *params,  [sp + 8]  -> x11
#     const struct xnn_qd8_quantization_params *quantization_params) [sp + 16] -> x16

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
// unused v14 v15

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2

        LDP         x12, x11, [sp]          // cn_stride, params

        ADD         x15, x3, x4             // a1 = a0 + a_stride
        ADD         x8, x6, x7              // c1 = c0 + cm_stride

        STP         d8,  d9, [sp, -48]!
        STP         d12,  d13, [sp, 32]
        LDR         x16, [sp, 64]           // &quantization_params[0].zero_point
        LD2         {v12.4s, v13.4s}, [x16] // v12 zero_point, v13 scale

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
        MUL         v17.4s, v16.4s, v12.s[1]
        MUL         v18.4s, v16.4s, v12.s[2]
        LDP         q24, q28, [x5], 32
        MUL         v19.4s, v16.4s, v12.s[3]
        MUL         v21.4s, v20.4s, v12.s[1]
        MUL         v22.4s, v20.4s, v12.s[2]
        MUL         v23.4s, v20.4s, v12.s[3]
        MUL         v25.4s, v24.4s, v12.s[1]
        MUL         v26.4s, v24.4s, v12.s[2]
        MUL         v27.4s, v24.4s, v12.s[3]
        MUL         v29.4s, v28.4s, v12.s[1]
        MUL         v30.4s, v28.4s, v12.s[2]
        MUL         v31.4s, v28.4s, v12.s[3]
        MUL         v24.4s, v24.4s, v12.s[0]
        MUL         v28.4s, v28.4s, v12.s[0]
        MUL         v16.4s, v16.4s, v12.s[0]
        MUL         v20.4s, v20.4s, v12.s[0]

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
        SCVTF       v18.4s, v18.4s
        SCVTF       v19.4s, v19.4s
        SCVTF       v20.4s, v20.4s
        SCVTF       v21.4s, v21.4s
        SCVTF       v22.4s, v22.4s
        SCVTF       v23.4s, v23.4s
        LDP         q0, q1, [x5], 32        // kernel_scale
        SCVTF       v24.4s, v24.4s
        SCVTF       v25.4s, v25.4s
        SCVTF       v26.4s, v26.4s
        SCVTF       v27.4s, v27.4s
        SCVTF       v28.4s, v28.4s
        SCVTF       v29.4s, v29.4s
        SCVTF       v30.4s, v30.4s
        SCVTF       v31.4s, v31.4s
        LDP         q2, q3, [x5], 32
        FMUL        v4.4s, v0.4s, v13.s[0]  // kernel_scale * scale
        FMUL        v5.4s, v1.4s, v13.s[0]
        FMUL        v6.4s, v2.4s, v13.s[0]
        FMUL        v7.4s, v3.4s, v13.s[0]
        FMUL        v8.4s, v0.4s, v13.s[1]
        FMUL        v9.4s, v1.4s, v13.s[1]
        FMUL        v10.4s, v2.4s, v13.s[1]
        FMUL        v11.4s, v3.4s, v13.s[1]
        FMUL        v16.4s, v16.4s, v4.4s
        FMUL        v20.4s, v20.4s, v5.4s
        FMUL        v24.4s, v24.4s, v6.4s
        FMUL        v28.4s, v28.4s, v7.4s
        FMUL        v17.4s, v17.4s, v8.4s
        FMUL        v21.4s, v21.4s, v9.4s
        FMUL        v25.4s, v25.4s, v10.4s
        FMUL        v29.4s, v29.4s, v11.4s
        FMUL        v4.4s, v0.4s, v13.s[2]
        FMUL        v5.4s, v1.4s, v13.s[2]
        FMUL        v6.4s, v2.4s, v13.s[2]
        FMUL        v7.4s, v3.4s, v13.s[2]
        FMUL        v8.4s, v0.4s, v13.s[3]
        FMUL        v9.4s, v1.4s, v13.s[3]
        FMUL        v10.4s, v2.4s, v13.s[3]
        FMUL        v11.4s, v3.4s, v13.s[3]
        LDP         q0, q1, [x5], 32        // bias
        FMUL        v18.4s, v18.4s, v4.4s
        FMUL        v22.4s, v22.4s, v5.4s
        FMUL        v26.4s, v26.4s, v6.4s
        FMUL        v30.4s, v30.4s, v7.4s
        FMUL        v19.4s, v19.4s, v8.4s
        FMUL        v23.4s, v23.4s, v9.4s
        FMUL        v27.4s, v27.4s, v10.4s
        FMUL        v31.4s, v31.4s, v11.4s
        LDP         q2, q3, [x5], 32
        FADD        v16.4s, v16.4s, v0.4s
        FADD        v17.4s, v17.4s, v0.4s
        FADD        v18.4s, v18.4s, v0.4s
        FADD        v19.4s, v19.4s, v0.4s
        FADD        v20.4s, v20.4s, v1.4s
        FADD        v21.4s, v21.4s, v1.4s
        FADD        v22.4s, v22.4s, v1.4s
        FADD        v23.4s, v23.4s, v1.4s
        LD2R        {v0.4s, v1.4s}, [x11]       // min max
        FADD        v24.4s, v24.4s, v2.4s
        FADD        v25.4s, v25.4s, v2.4s
        FADD        v26.4s, v26.4s, v2.4s
        FADD        v27.4s, v27.4s, v2.4s
        FADD        v28.4s, v28.4s, v3.4s
        FADD        v29.4s, v29.4s, v3.4s
        FADD        v30.4s, v30.4s, v3.4s
        FADD        v31.4s, v31.4s, v3.4s
        FMAX        v16.4s, v16.4s, v0.4s
        FMAX        v17.4s, v17.4s, v0.4s
        FMAX        v18.4s, v18.4s, v0.4s
        FMAX        v19.4s, v19.4s, v0.4s
        FMAX        v20.4s, v20.4s, v0.4s
        FMAX        v21.4s, v21.4s, v0.4s
        FMAX        v22.4s, v22.4s, v0.4s
        FMAX        v23.4s, v23.4s, v0.4s
        FMAX        v24.4s, v24.4s, v0.4s
        FMAX        v25.4s, v25.4s, v0.4s
        FMAX        v26.4s, v26.4s, v0.4s
        FMAX        v27.4s, v27.4s, v0.4s
        FMAX        v28.4s, v28.4s, v0.4s
        FMAX        v29.4s, v29.4s, v0.4s
        FMAX        v30.4s, v30.4s, v0.4s
        FMAX        v31.4s, v31.4s, v0.4s
        FMIN        v16.4s, v16.4s, v1.4s
        FMIN        v17.4s, v17.4s, v1.4s
        FMIN        v18.4s, v18.4s, v1.4s
        FMIN        v19.4s, v19.4s, v1.4s
        FMIN        v20.4s, v20.4s, v1.4s
        FMIN        v21.4s, v21.4s, v1.4s
        FMIN        v22.4s, v22.4s, v1.4s
        FMIN        v23.4s, v23.4s, v1.4s
        FMIN        v24.4s, v24.4s, v1.4s
        FMIN        v25.4s, v25.4s, v1.4s
        FMIN        v26.4s, v26.4s, v1.4s
        FMIN        v27.4s, v27.4s, v1.4s
        FMIN        v28.4s, v28.4s, v1.4s
        FMIN        v29.4s, v29.4s, v1.4s
        FMIN        v30.4s, v30.4s, v1.4s
        FMIN        v31.4s, v31.4s, v1.4s

        SUBS        x1, x1, 16
        B.LO        6f
        STP         q19, q23, [x7]
        STP         q27, q31, [x7, #32]
        ADD         x7, x7, x12
        STP         q18, q22, [x9]
        STP         q26, q30, [x9, #32]
        ADD         x9, x9, x12
        STP         q17, q21, [x8]
        STP         q25, q29, [x8, #32]
        ADD         x8, x8, x12
        STP         q16, q20, [x6]
        STP         q24, q28, [x6, #32]
        ADD         x6, x6, x12
        SUB         x3,  x3, x2             // a0 -= kc
        SUB         x15, x15, x2            // a1 -= kc
        SUB         x13, x13, x2            // a2 -= kc
        SUB         x4,  x4, x2             // a3 -= kc
        B.NE        0b


        # Restore d8-d13 from stack
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 48
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
        STP         q19, q23, [x7]
        STP         q18, q22, [x9]
        STP         q17, q21, [x8]
        STP         q16, q20, [x6]
        MOV         v16.16b, v24.16b
        MOV         v17.16b, v25.16b
        MOV         v18.16b, v26.16b
        MOV         v19.16b, v27.16b
        MOV         v20.16b, v28.16b
        MOV         v21.16b, v29.16b
        MOV         v22.16b, v30.16b
        MOV         v23.16b, v31.16b
        ADD         x6, x6, #32
        ADD         x7, x7, #32
        ADD         x8, x8, #32
        ADD         x9, x9, #32
7:
        TBZ         x1, 2, 8f
        STR         q19, [x7]
        STR         q18, [x9]
        STR         q17, [x8]
        STR         q16, [x6]
        MOV         v16.16b, v20.16b
        MOV         v17.16b, v21.16b
        MOV         v18.16b, v22.16b
        MOV         v19.16b, v23.16b
        ADD         x6, x6, #16
        ADD         x7, x7, #16
        ADD         x8, x8, #16
        ADD         x9, x9, #16
8:
        TBZ         x1, 1, 9f
        ST1         {v19.2s}, [x7]
        ST1         {v18.2s}, [x9]
        ST1         {v17.2s}, [x8]
        ST1         {v16.2s}, [x6]
        DUP         d16, v16.d[1]
        DUP         d17, v17.d[1]
        DUP         d18, v18.d[1]
        DUP         d19, v19.d[1]
        ADD         x6, x6, #8
        ADD         x7, x7, #8
        ADD         x8, x8, #8
        ADD         x9, x9, #8
9:
        TBZ         x1, 0, 10f
        STR         s19, [x7]
        STR         s18, [x9]
        STR         s17, [x8]
        STR         s16, [x6]
10:
        # Restore d8-d13 from stack
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 48
        RET

END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
