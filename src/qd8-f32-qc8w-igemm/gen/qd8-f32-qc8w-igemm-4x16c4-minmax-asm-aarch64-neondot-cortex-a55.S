// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/4x16c4-aarch64-neondot-cortex-a55.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qd8_f32_qc8w_igemm_minmax__ukernel_4x16c4__asm_aarch64_neondot_cortex_a55(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     size_t ks,                 x3 / x9
#     const int8_t** restrict a,  x4
#     const int8_t* restrict w,  x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,                  [sp] -> (x0)
#     size_t a_offset,                   [sp + 8] -> x8
#     const int8_t* zero,                [sp + 16] -> x12
#     const int8_t* zero_data,             [sp + 24] -> x19
#     const union xnn_f32_minmax_params *params,  [sp + 32]  -> x11
#     const struct xnn_qd8_quantization_params *quantization_params) [sp + 40] -> x17

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x13  v0  v4
// A1  x14  v1  v5
// A2  x15  v2  v6
// A3  x10  v3  v7
// B    x5  v8  v9 v10 v11
// C0   x6 v16 v20 v24 v28
// C1  x16 v17 v21 v25 v29
// C2  x17 v18 v22 v26 v30
// C3   x7 v19 v23 v27 v31
// unused v13, v14 v15

// x11 temp for Cortex-A55 loads

BEGIN_FUNCTION xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        LDR         x8, [sp, 8]             // Load a_offset
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        LDR         x12, [sp, 16]       // Load zero
        LDR         x11, [sp, 32]      // Load params pointer
        CSEL        x16, x6,  x16, LO       //   c1 = c0
        ADD         x2, x2, 3               // kc = (kc + 3) & ~3
        STP         d8,  d9, [sp, -48]!   // Save d8-d11 on stack
        STR         x19, [sp, 40]       // Save x19 to stack
        LDR         x19, [sp, 72]       // Load zero_data
        STR         d12,  [sp, 32]
        LDR         x17, [sp, 88]           // &quantization_params.zero_point
        LD1         {v12.4s}, [x17]         // zero point and scale

        ADD         x17, x16, x7            // c2 = c1 + cm_stride
        STP         d10, d11, [sp, 16]
                                            // if mr <= 2
        CSEL        x17, x16, x17, LS       //   c2 = c1
        BIC         x2, x2, 3

        CMP         x0, 4                   // if mr < 4
        ADD         x7,  x17, x7            // c3 = c2 + cm_stride
        CSEL        x7,  x17, x7, LO        //   c3 = c2

        .p2align    3
0:
        # Load initial bias from w into accumulators
        LDP         q16, q20, [x5], 32
        MUL         v17.4s, v16.4s, v12.s[0]
        MUL         v18.4s, v16.4s, v12.s[0]
        LDP         q24, q28, [x5], 32
        MUL         v19.4s, v16.4s, v12.s[0]
        MUL         v21.4s, v20.4s, v12.s[0]
        MUL         v22.4s, v20.4s, v12.s[0]
        MUL         v23.4s, v20.4s, v12.s[0]
        MUL         v25.4s, v24.4s, v12.s[0]
        MUL         v26.4s, v24.4s, v12.s[0]
        MUL         v27.4s, v24.4s, v12.s[0]
        MUL         v29.4s, v28.4s, v12.s[0]
        MUL         v30.4s, v28.4s, v12.s[0]
        MUL         v31.4s, v28.4s, v12.s[0]
        MUL         v24.4s, v24.4s, v12.s[0]
        MUL         v28.4s, v28.4s, v12.s[0]
        MUL         v16.4s, v16.4s, v12.s[0]
        MUL         v20.4s, v20.4s, v12.s[0]
        MOV         x9, x3                  // p = ks

        .p2align    3
1:
        # Load next 4 A pointers
        LDP         x13, x14, [x4], 16
        LDP         x15, x10, [x4], 16

        CMP         x13, x12                // if a0 == zero
        ADD         x13, x13, x8            // a0 += a_offset
        CSEL        x13, x19, x13, EQ       //   a0 = zero_data, else a0 += a_offset
        CMP         x14, x12                // if a1 == zero
        ADD         x14, x14, x8            // a1 += a_offset
        CSEL        x14, x19, x14, EQ       //   a1 = zero_data, else a1 += a_offset
        CMP         x15, x12                // if a2 == zero
        ADD         x15, x15, x8            // a2 += a_offset
        CSEL        x15, x19, x15, EQ       //   a2 = zero_data, else a2 += a_offset
        CMP         x10, x12                // if a3 == zero
        ADD         x10, x10, x8            // a3 += a_offset
        CSEL        x10, x19, x10, EQ       //   a3 = zero_data, else a3 += a_offset

        # Is there at least 16 bytes for prologue/epilogue?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        5f

        # prologue - read A and B values for block 0 and 1
        LDR         d0, [x13], 8
        LDR         q8,  [x5], 16
        LDR         d1, [x14], 8
        LDR         d2, [x15], 8
        LDR         d3, [x10], 8
        SUBS        x0, x0, 16              // is there 16 for main loop?
        LDR         d9,  [x5], 8
        LDR         x11,  [x5], 8
        # Is there at least 16 bytes for main loop?
        B.LO        3f

        # Main loop - 16 bytes of A in 4 groups.
        # 4 row of 4 vectors wide = 16 sdot instructions for 4 channels
        # 4 LD64 for A
        # 4 LD128 for W. = 2 LD64 + INS.
        # for each 4 sdot, 1 LD64 for A, 2 LD64 for W + INS.

        .p2align    3
2:
        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v0.4b[0]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v1.4b[0]
        INS         v9.d[1], x11
        SDOT        v18.4s,  v8.16b, v2.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v19.4s,  v8.16b, v3.4b[0]
        LDR         d4,  [x13], 8

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v0.4b[0]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v1.4b[0]
        INS         v10.d[1], x11
        SDOT        v22.4s,  v9.16b, v2.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v23.4s,  v9.16b, v3.4b[0]
        LDR         d5, [x14], 8

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v0.4b[0]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v1.4b[0]
        INS         v11.d[1], x11
        SDOT        v26.4s, v10.16b, v2.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v27.4s, v10.16b, v3.4b[0]
        LDR         d6, [x15], 8

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v0.4b[0]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v1.4b[0]
        INS         v8.d[1], x11
        SDOT        v30.4s, v11.16b, v2.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v31.4s, v11.16b, v3.4b[0]
        LDR         d7,  [x10], 8

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v0.4b[1]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v1.4b[1]
        INS         v9.d[1], x11
        SDOT        v18.4s,  v8.16b, v2.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v19.4s,  v8.16b, v3.4b[1]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v0.4b[1]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v1.4b[1]
        INS         v10.d[1], x11
        SDOT        v22.4s,  v9.16b, v2.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v23.4s,  v9.16b, v3.4b[1]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v0.4b[1]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v1.4b[1]
        INS         v11.d[1], x11
        SDOT        v26.4s, v10.16b, v2.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v27.4s, v10.16b, v3.4b[1]

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v0.4b[1]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v1.4b[1]
        INS         v8.d[1], x11
        SDOT        v30.4s, v11.16b, v2.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v31.4s, v11.16b, v3.4b[1]

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v4.4b[0]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v5.4b[0]
        INS         v9.d[1], x11
        SDOT        v18.4s,  v8.16b, v6.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v19.4s,  v8.16b, v7.4b[0]
        LDR         d0,  [x13], 8

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v4.4b[0]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v5.4b[0]
        INS         v10.d[1], x11
        SDOT        v22.4s,  v9.16b, v6.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v23.4s,  v9.16b, v7.4b[0]
        LDR         d1, [x14], 8

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v4.4b[0]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v5.4b[0]
        INS         v11.d[1], x11
        SDOT        v26.4s, v10.16b, v6.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v27.4s, v10.16b, v7.4b[0]
        LDR         d2, [x15], 8

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v4.4b[0]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v5.4b[0]
        INS         v8.d[1], x11
        SDOT        v30.4s, v11.16b, v6.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v31.4s, v11.16b, v7.4b[0]
        LDR         d3,  [x10], 8

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v4.4b[1]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v5.4b[1]
        INS         v9.d[1], x11
        SDOT        v18.4s,  v8.16b, v6.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v19.4s,  v8.16b, v7.4b[1]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v4.4b[1]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v5.4b[1]
        INS         v10.d[1], x11
        SDOT        v22.4s,  v9.16b, v6.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v23.4s,  v9.16b, v7.4b[1]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v4.4b[1]
        LDR         d8,  [x5], 8            // First B values for block 0 and 1
        SDOT        v25.4s, v10.16b, v5.4b[1]
        INS         v11.d[1], x11
        SDOT        v26.4s, v10.16b, v6.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v27.4s, v10.16b, v7.4b[1]
        SUBS        x0, x0, 16

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v4.4b[1]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v5.4b[1]
        INS         v8.d[1], x11
        SDOT        v30.4s, v11.16b, v6.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v31.4s, v11.16b, v7.4b[1]
        B.HS        2b

        # Epilogue.  Same as main loop but no preloads in final group
3:
        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v0.4b[0]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v1.4b[0]
        INS         v9.d[1], x11
        SDOT        v18.4s,  v8.16b, v2.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v19.4s,  v8.16b, v3.4b[0]
        LDR         d4,  [x13], 8

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v0.4b[0]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v1.4b[0]
        INS         v10.d[1], x11
        SDOT        v22.4s,  v9.16b, v2.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v23.4s,  v9.16b, v3.4b[0]
        LDR         d5, [x14], 8

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v0.4b[0]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v1.4b[0]
        INS         v11.d[1], x11
        SDOT        v26.4s, v10.16b, v2.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v27.4s, v10.16b, v3.4b[0]
        LDR         d6, [x15], 8

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v0.4b[0]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v1.4b[0]
        INS         v8.d[1], x11
        SDOT        v30.4s, v11.16b, v2.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v31.4s, v11.16b, v3.4b[0]
        LDR         d7,  [x10], 8

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v0.4b[1]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v1.4b[1]
        INS         v9.d[1], x11
        SDOT        v18.4s,  v8.16b, v2.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v19.4s,  v8.16b, v3.4b[1]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v0.4b[1]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v1.4b[1]
        INS         v10.d[1], x11
        SDOT        v22.4s,  v9.16b, v2.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v23.4s,  v9.16b, v3.4b[1]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v0.4b[1]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v1.4b[1]
        INS         v11.d[1], x11
        SDOT        v26.4s, v10.16b, v2.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v27.4s, v10.16b, v3.4b[1]

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v0.4b[1]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v1.4b[1]
        INS         v8.d[1], x11
        SDOT        v30.4s, v11.16b, v2.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v31.4s, v11.16b, v3.4b[1]

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v4.4b[0]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v5.4b[0]
        INS         v9.d[1], x11
        SDOT        v18.4s,  v8.16b, v6.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v19.4s,  v8.16b, v7.4b[0]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v4.4b[0]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v5.4b[0]
        INS         v10.d[1], x11
        SDOT        v22.4s,  v9.16b, v6.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v23.4s,  v9.16b, v7.4b[0]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v4.4b[0]
        LDR         d8,  [x5], 8
        SDOT        v25.4s, v10.16b, v5.4b[0]
        INS         v11.d[1], x11
        SDOT        v26.4s, v10.16b, v6.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v27.4s, v10.16b, v7.4b[0]

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v4.4b[0]
        LDR         d9,  [x5], 8
        SDOT        v29.4s, v11.16b, v5.4b[0]
        INS         v8.d[1], x11
        SDOT        v30.4s, v11.16b, v6.4b[0]
        LDR         x11,  [x5], 8
        SDOT        v31.4s, v11.16b, v7.4b[0]

        # BLOCK 0
        SDOT        v16.4s,  v8.16b, v4.4b[1]
        LDR         d10,  [x5], 8
        SDOT        v17.4s,  v8.16b, v5.4b[1]
        INS         v9.d[1], x11
        SDOT        v18.4s,  v8.16b, v6.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v19.4s,  v8.16b, v7.4b[1]

        # BLOCK 1
        SDOT        v20.4s,  v9.16b, v4.4b[1]
        LDR         d11,  [x5], 8
        SDOT        v21.4s,  v9.16b, v5.4b[1]
        INS         v10.d[1], x11
        SDOT        v22.4s,  v9.16b, v6.4b[1]
        LDR         x11,  [x5], 8
        SDOT        v23.4s,  v9.16b, v7.4b[1]

        # BLOCK 2
        SDOT        v24.4s, v10.16b, v4.4b[1]
        SDOT        v25.4s, v10.16b, v5.4b[1]
        INS         v11.d[1], x11
        SDOT        v26.4s, v10.16b, v6.4b[1]
        SDOT        v27.4s, v10.16b, v7.4b[1]
        AND         x0, x2, 15              // kc remainder 0 to 12

        # BLOCK 3
        SDOT        v28.4s, v11.16b, v4.4b[1]
        SDOT        v29.4s, v11.16b, v5.4b[1]
        LDR         x11, [sp, 80]                  // reload params pointer
        SDOT        v30.4s, v11.16b, v6.4b[1]
        SDOT        v31.4s, v11.16b, v7.4b[1]

        # Is there a remainder?- 4 to 12 bytes of A
        CBNZ        x0, 6f

        .p2align    3
4:
        # ks loop
        SUBS        x9, x9, 32              // ks -= MR * sizeof(int8_t*)
        B.HI        1b

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
        FMUL        v4.4s, v0.4s, v12.s[1]  // kernel_scale * scale
        FMUL        v5.4s, v1.4s, v12.s[1]
        FMUL        v6.4s, v2.4s, v12.s[1]
        FMUL        v7.4s, v3.4s, v12.s[1]
        FMUL        v8.4s, v0.4s, v12.s[1]
        FMUL        v9.4s, v1.4s, v12.s[1]
        FMUL        v10.4s, v2.4s, v12.s[1]
        FMUL        v11.4s, v3.4s, v12.s[1]
        FMUL        v16.4s, v16.4s, v4.4s
        FMUL        v20.4s, v20.4s, v5.4s
        FMUL        v24.4s, v24.4s, v6.4s
        FMUL        v28.4s, v28.4s, v7.4s
        FMUL        v17.4s, v17.4s, v8.4s
        FMUL        v21.4s, v21.4s, v9.4s
        FMUL        v25.4s, v25.4s, v10.4s
        FMUL        v29.4s, v29.4s, v11.4s
        FMUL        v4.4s, v0.4s, v12.s[1]
        FMUL        v5.4s, v1.4s, v12.s[1]
        FMUL        v6.4s, v2.4s, v12.s[1]
        FMUL        v7.4s, v3.4s, v12.s[1]
        FMUL        v8.4s, v0.4s, v12.s[1]
        FMUL        v9.4s, v1.4s, v12.s[1]
        FMUL        v10.4s, v2.4s, v12.s[1]
        FMUL        v11.4s, v3.4s, v12.s[1]
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
        LDR         x0, [sp, 48]            // cn_stride
        B.LO        7f
        STP         q19, q23, [x7]
        STP         q27, q31, [x7, #32]
        ADD         x7, x7, x0

        STP         q18, q22, [x17]
        STP         q26, q30, [x17, #32]
        ADD         x17, x17, x0

        STP         q17, q21, [x16]
        STP         q25, q29, [x16, #32]
        ADD         x16, x16, x0

        STP         q16, q20, [x6]
        STP         q24, q28, [x6, #32]
        ADD         x6, x6, x0
        SUB         x4, x4, x3              // a -= ks
        B.NE        0b


        # Restore d8-d12 from stack
        LDR         x19, [sp, 40]
        LDR         d12, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 48
        RET

        # Remainder- 4 to 12 bytes of A
        # Although C4, its safe to read 16 bytes.
        .p2align    3
5:
        AND         x0, x2, 15              // kc remainder 4 to 12
6:
        LDR         q0, [x13]
        LDP         q8,  q9,  [x5], 32
        LDR         q1, [x14]
        LDR         q2, [x15]
        LDR         q3, [x10]
        LDP         q10, q11, [x5], 32
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
        B.LS        4b
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
        B.LS        4b
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
        B           4b

        # Store odd width
        .p2align    3
7:
        TBZ         x1, 3, 8f
        STP         q19, q23, [x7]
        STP         q18, q22, [x17]
        STP         q17, q21, [x16]
        STP         q16, q20, [x6]
        MOV         v16.16b, v24.16b
        MOV         v17.16b, v25.16b
        MOV         v18.16b, v26.16b
        MOV         v19.16b, v27.16b
        MOV         v20.16b, v28.16b
        MOV         v21.16b, v29.16b
        MOV         v22.16b, v30.16b
        MOV         v23.16b, v31.16b
        ADD         x7, x7, #32
        ADD         x17, x17, #32
        ADD         x16, x16, #32
        ADD         x6, x6, #32
8:
        TBZ         x1, 2, 9f
        STR         q19, [x7]
        STR         q18, [x17]
        STR         q17, [x16]
        STR         q16, [x6]
        MOV         v16.16b, v20.16b
        MOV         v17.16b, v21.16b
        MOV         v18.16b, v22.16b
        MOV         v19.16b, v23.16b
        ADD         x7, x7, #16
        ADD         x17, x17, #16
        ADD         x16, x16, #16
        ADD         x6, x6, #16
9:
        TBZ         x1, 1, 10f
        ST1         {v19.2s}, [x7]
        ST1         {v18.2s}, [x17]
        ST1         {v17.2s}, [x16]
        ST1         {v16.2s}, [x6]
        DUP         d16, v16.d[1]
        DUP         d17, v17.d[1]
        DUP         d18, v18.d[1]
        DUP         d19, v19.d[1]
        ADD         x7, x7, #8
        ADD         x17, x17, #8
        ADD         x16, x16, #8
        ADD         x6, x6, #8
10:
        TBZ         x1, 0, 11f
        STR         s19, [x7]
        STR         s18, [x17]
        STR         s17, [x16]
        STR         s16, [x6]
11:
        # Restore d8-d12 from stack
        LDR         x19, [sp, 40]
        LDR         d12, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 48
        RET

END_FUNCTION xnn_qd8_f32_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
