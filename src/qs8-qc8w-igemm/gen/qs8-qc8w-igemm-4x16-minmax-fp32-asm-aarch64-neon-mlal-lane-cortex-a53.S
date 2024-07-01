// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     size_t ks,                 x3 / x9
#     const int8_t** restrict a, x4
#     const int8_t* restrict w,  x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x8
#     const int8_t* zero,                [sp + 16] -> x12
#     const xnn_qs8_conv_minmax_params params [sp + 24] -> (x11)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x13  v0
// A1  x14  v1
// A2  x15  v2
// A3  x20  v3
// B    x5  v4  v5  v6
// C0   x6 v16 v20 v24 v28
// C1  x16 v17 v21 v25 v29
// C2  x17 v18 v22 v26 v30
// C3   x7 v19 v23 v27 v31
# unused  v7 v8 v9 v10 v11 v12 v13 v14 v15
// x11, x21 temp for Cortex-A53 loads

BEGIN_FUNCTION xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        LDP         x10, x8, [sp]           // Load cn_stride, a_offset
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x16, x6,  x16, LO       //   c1 = c0

        ADD         x17, x16, x7            // c2 = c1 + cm_stride
        LDP         x12, x11, [sp, 16]      // Load zero, params pointer
                                            // if mr <= 2
        CSEL        x17, x16, x17, LS       //   c2 = c1

        CMP         x0, 4                   // if mr < 4
        STP         x20, x21, [sp, -16]!    // Save x20-x21 on stack
        ADD         x7,  x17, x7            // c3 = c2 + cm_stride
        CSEL        x7,  x17, x7, LO        //   c3 = c2


        .p2align    3
0:
        # Load initial bias from w into accumulators
        LDP         q16, q20, [x5], 32
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
        MOV         x9, x3                  // p = ks

        .p2align    3
1:
        # Load next 4 A pointers
        LDP         x13, x14, [x4], 16
        LDP         x15, x20, [x4], 16

        CMP         x13, x12                // if a0 == zero
        ADD         x13, x13, x8            // a0 += a_offset
        CSEL        x13, x12, x13, EQ       //   a0 = zero, else += a0 + a_offset
        CMP         x14, x12                // if a1 == zero
        ADD         x14, x14, x8            // a1 += a_offset
        CSEL        x14, x12, x14, EQ       //   a1 = zero, else += a1 + a_offset
        CMP         x15, x12                // if a2 == zero
        ADD         x15, x15, x8            // a2 += a_offset
        CSEL        x15, x12, x15, EQ       //   a2 = zero, else += a2 + a_offset
        CMP         x20, x12                // if a3 == zero
        ADD         x20, x20, x8            // a3 += a_offset
        CSEL        x20, x12, x20, EQ       //   a3 = zero, else += a3 + a_offset

        # Is there at least 8 bytes for epilogue?
        SUBS        x0, x2, 8               // k = kc - 8
        B.LO        5f

        # Prologue
        LDR         d0, [x13], 8
        LDP         d4, d6, [x5]
        LDR         d1, [x14], 8
        LDR         d2, [x15], 8
        LDR         d3, [x20], 8
        SXTL    v0.8h, v0.8b
        LDR         x11, [x5, 16]
        SXTL        v4.8h, v4.8b
        SXTL    v1.8h, v1.8b
        SXTL    v2.8h, v2.8b
        SXTL    v3.8h, v3.8b
        SXTL        v6.8h, v6.8b

        SUBS        x0, x0, 8               // k = k - 8
        # Is there at least 8 bytes for main loop?
        B.LO        3f

        # Main loop - 8 bytes of A
        .p2align    3
2:
        SMLAL       v16.4s, v4.4h, v0.h[0]
        SMLAL2      v20.4s, v4.8h, v0.h[0]
        SMLAL       v17.4s, v4.4h, v1.h[0]
        SMLAL2      v21.4s, v4.8h, v1.h[0]
        SMLAL       v18.4s, v4.4h, v2.h[0]
        SMLAL2      v22.4s, v4.8h, v2.h[0]
        SMLAL       v19.4s, v4.4h, v3.h[0]
        SMLAL2      v23.4s, v4.8h, v3.h[0]
        LDR         d4, [x5, 24]
        INS         v5.d[0], x11
        SMLAL       v24.4s, v6.4h, v0.h[0]
        SMLAL2      v28.4s, v6.8h, v0.h[0]
        SMLAL       v25.4s, v6.4h, v1.h[0]
        SMLAL2      v29.4s, v6.8h, v1.h[0]
        SXTL        v5.8h, v5.8b
        SMLAL       v26.4s, v6.4h, v2.h[0]
        SMLAL2      v30.4s, v6.8h, v2.h[0]
        SMLAL       v27.4s, v6.4h, v3.h[0]
        SMLAL2      v31.4s, v6.8h, v3.h[0]
        LDR         x11, [x5, 32]
        SMLAL       v16.4s, v5.4h, v0.h[1]
        SMLAL2      v20.4s, v5.8h, v0.h[1]
        SMLAL       v17.4s, v5.4h, v1.h[1]
        SMLAL2      v21.4s, v5.8h, v1.h[1]
        SXTL        v4.8h, v4.8b
        SMLAL       v18.4s, v5.4h, v2.h[1]
        SMLAL2      v22.4s, v5.8h, v2.h[1]
        SMLAL       v19.4s, v5.4h, v3.h[1]
        SMLAL2      v23.4s, v5.8h, v3.h[1]
        LDR         d5, [x5, 40]
        INS         v6.d[0], x11
        SMLAL       v24.4s, v4.4h, v0.h[1]
        SMLAL2      v28.4s, v4.8h, v0.h[1]
        SMLAL       v25.4s, v4.4h, v1.h[1]
        SMLAL2      v29.4s, v4.8h, v1.h[1]
        SXTL        v6.8h, v6.8b
        SMLAL       v26.4s, v4.4h, v2.h[1]
        SMLAL2      v30.4s, v4.8h, v2.h[1]
        SMLAL       v27.4s, v4.4h, v3.h[1]
        SMLAL2      v31.4s, v4.8h, v3.h[1]
        LDR         x11, [x5, 48]
        SMLAL       v16.4s, v6.4h, v0.h[2]
        SMLAL2      v20.4s, v6.8h, v0.h[2]
        SMLAL       v17.4s, v6.4h, v1.h[2]
        SXTL        v5.8h, v5.8b
        SMLAL2      v21.4s, v6.8h, v1.h[2]
        SMLAL       v18.4s, v6.4h, v2.h[2]
        SMLAL2      v22.4s, v6.8h, v2.h[2]
        SMLAL       v19.4s, v6.4h, v3.h[2]
        SMLAL2      v23.4s, v6.8h, v3.h[2]
        LDR         d6, [x5, 56]
        INS         v4.d[0], x11
        SMLAL       v24.4s, v5.4h, v0.h[2]
        SMLAL2      v28.4s, v5.8h, v0.h[2]
        SMLAL       v25.4s, v5.4h, v1.h[2]
        SMLAL2      v29.4s, v5.8h, v1.h[2]
        SXTL        v4.8h, v4.8b
        SMLAL       v26.4s, v5.4h, v2.h[2]
        SMLAL2      v30.4s, v5.8h, v2.h[2]
        SMLAL       v27.4s, v5.4h, v3.h[2]
        SMLAL2      v31.4s, v5.8h, v3.h[2]
        LDR         x11, [x5, 64]
        SMLAL       v16.4s, v4.4h, v0.h[3]
        SMLAL2      v20.4s, v4.8h, v0.h[3]
        SMLAL       v17.4s, v4.4h, v1.h[3]
        SMLAL2      v21.4s, v4.8h, v1.h[3]
        SXTL        v6.8h, v6.8b
        SMLAL       v18.4s, v4.4h, v2.h[3]
        SMLAL2      v22.4s, v4.8h, v2.h[3]
        SMLAL       v19.4s, v4.4h, v3.h[3]
        SMLAL2      v23.4s, v4.8h, v3.h[3]
        LDR         d4, [x5, 72]
        INS         v5.d[0], x11
        SMLAL       v24.4s, v6.4h, v0.h[3]
        SMLAL2      v28.4s, v6.8h, v0.h[3]
        SXTL        v5.8h, v5.8b
        SMLAL       v25.4s, v6.4h, v1.h[3]
        SMLAL2      v29.4s, v6.8h, v1.h[3]
        SMLAL       v26.4s, v6.4h, v2.h[3]
        SMLAL2      v30.4s, v6.8h, v2.h[3]
        SMLAL       v27.4s, v6.4h, v3.h[3]
        SMLAL2      v31.4s, v6.8h, v3.h[3]
        LDR         x11, [x5, 80]
        SMLAL       v16.4s, v5.4h, v0.h[4]
        SMLAL2      v20.4s, v5.8h, v0.h[4]
        SMLAL       v17.4s, v5.4h, v1.h[4]
        SMLAL2      v21.4s, v5.8h, v1.h[4]
        SXTL        v4.8h, v4.8b
        SMLAL       v18.4s, v5.4h, v2.h[4]
        SMLAL2      v22.4s, v5.8h, v2.h[4]
        SMLAL       v19.4s, v5.4h, v3.h[4]
        SMLAL2      v23.4s, v5.8h, v3.h[4]
        LDR         d5, [x5, 88]
        INS         v6.d[0], x11
        SMLAL       v24.4s, v4.4h, v0.h[4]
        SMLAL2      v28.4s, v4.8h, v0.h[4]
        SMLAL       v25.4s, v4.4h, v1.h[4]
        SMLAL2      v29.4s, v4.8h, v1.h[4]
        SXTL        v6.8h, v6.8b
        SMLAL       v26.4s, v4.4h, v2.h[4]
        SMLAL2      v30.4s, v4.8h, v2.h[4]
        SMLAL       v27.4s, v4.4h, v3.h[4]
        SMLAL2      v31.4s, v4.8h, v3.h[4]
        LDR         x11, [x5, 96]
        SMLAL       v16.4s, v6.4h, v0.h[5]
        SMLAL2      v20.4s, v6.8h, v0.h[5]
        SMLAL       v17.4s, v6.4h, v1.h[5]
        SMLAL2      v21.4s, v6.8h, v1.h[5]
        SXTL        v5.8h, v5.8b
        SMLAL       v18.4s, v6.4h, v2.h[5]
        SMLAL2      v22.4s, v6.8h, v2.h[5]
        SMLAL       v19.4s, v6.4h, v3.h[5]
        SMLAL2      v23.4s, v6.8h, v3.h[5]
        LDR         d6, [x5, 104]
        INS         v4.d[0], x11
        SMLAL       v24.4s, v5.4h, v0.h[5]
        SMLAL2      v28.4s, v5.8h, v0.h[5]
        SMLAL       v25.4s, v5.4h, v1.h[5]
        SMLAL2      v29.4s, v5.8h, v1.h[5]
        SXTL        v4.8h, v4.8b
        SMLAL       v26.4s, v5.4h, v2.h[5]
        SMLAL2      v30.4s, v5.8h, v2.h[5]
        SMLAL       v27.4s, v5.4h, v3.h[5]
        SMLAL2      v31.4s, v5.8h, v3.h[5]
        SXTL        v6.8h, v6.8b
        LDR         x11, [x5, 112]
        SMLAL       v16.4s, v4.4h, v0.h[6]
        SMLAL2      v20.4s, v4.8h, v0.h[6]
        SMLAL       v17.4s, v4.4h, v1.h[6]
        SMLAL2      v21.4s, v4.8h, v1.h[6]
        SMLAL       v18.4s, v4.4h, v2.h[6]
        SMLAL2      v22.4s, v4.8h, v2.h[6]
        SMLAL       v19.4s, v4.4h, v3.h[6]
        SMLAL2      v23.4s, v4.8h, v3.h[6]
        LDR         d5, [x5, 120]
        INS         v4.d[0], x11
        SMLAL       v24.4s, v6.4h, v0.h[6]
        SMLAL2      v28.4s, v6.8h, v0.h[6]
        SMLAL       v25.4s, v6.4h, v1.h[6]
        SMLAL2      v29.4s, v6.8h, v1.h[6]
        SXTL        v4.8h, v4.8b
        ADD         x5, x5, 128

        SMLAL       v26.4s, v6.4h, v2.h[6]
        SMLAL2      v30.4s, v6.8h, v2.h[6]
        LDR         x11, [x5]
        SMLAL       v27.4s, v6.4h, v3.h[6]
        SMLAL2      v31.4s, v6.8h, v3.h[6]
        SXTL        v5.8h, v5.8b
        LDR         x21, [x13], 8

        SMLAL       v16.4s, v4.4h, v0.h[7]
        SMLAL2      v20.4s, v4.8h, v0.h[7]
        SMLAL       v17.4s, v4.4h, v1.h[7]
        SMLAL2      v21.4s, v4.8h, v1.h[7]
        SMLAL       v18.4s, v4.4h, v2.h[7]
        SMLAL2      v22.4s, v4.8h, v2.h[7]
        SMLAL       v19.4s, v4.4h, v3.h[7]
        SMLAL2      v23.4s, v4.8h, v3.h[7]
        LDR         d6, [x5, 8]
        INS         v4.d[0], x11
        SMLAL       v24.4s, v5.4h, v0.h[7]
        SMLAL2      v28.4s, v5.8h, v0.h[7]
        LDR         x11, [x15], 8
        SMLAL       v25.4s, v5.4h, v1.h[7]
        SMLAL2      v29.4s, v5.8h, v1.h[7]
        LDR         d1, [x14], 8
        INS         v0.d[0], x21
        SMLAL       v26.4s, v5.4h, v2.h[7]
        SMLAL2      v30.4s, v5.8h, v2.h[7]
        SMLAL       v27.4s, v5.4h, v3.h[7]
        SMLAL2      v31.4s, v5.8h, v3.h[7]
        LDR         d3, [x20], 8
        INS         v2.d[0], x11

        SXTL    v0.8h, v0.8b
        SXTL    v1.8h, v1.8b
        LDR         x11, [x5, 16]
        SXTL        v4.8h, v4.8b
        SXTL    v2.8h, v2.8b
        SUBS        x0, x0, 8
        SXTL    v3.8h, v3.8b
        SXTL        v6.8h, v6.8b
        B.HS        2b

        # Epilogue.  Same as main loop but no preloads in final group

        .p2align    3
3:
        SMLAL       v16.4s, v4.4h, v0.h[0]
        SMLAL2      v20.4s, v4.8h, v0.h[0]
        SMLAL       v17.4s, v4.4h, v1.h[0]
        SMLAL2      v21.4s, v4.8h, v1.h[0]
        SMLAL       v18.4s, v4.4h, v2.h[0]
        SMLAL2      v22.4s, v4.8h, v2.h[0]
        SMLAL       v19.4s, v4.4h, v3.h[0]
        SMLAL2      v23.4s, v4.8h, v3.h[0]
        LDR         d4, [x5, 24]
        INS         v5.d[0], x11
        SMLAL       v24.4s, v6.4h, v0.h[0]
        SMLAL2      v28.4s, v6.8h, v0.h[0]
        SMLAL       v25.4s, v6.4h, v1.h[0]
        SMLAL2      v29.4s, v6.8h, v1.h[0]
        SXTL        v5.8h, v5.8b
        SMLAL       v26.4s, v6.4h, v2.h[0]
        SMLAL2      v30.4s, v6.8h, v2.h[0]
        SMLAL       v27.4s, v6.4h, v3.h[0]
        SMLAL2      v31.4s, v6.8h, v3.h[0]
        LDR         x11, [x5, 32]
        SMLAL       v16.4s, v5.4h, v0.h[1]
        SMLAL2      v20.4s, v5.8h, v0.h[1]
        SMLAL       v17.4s, v5.4h, v1.h[1]
        SMLAL2      v21.4s, v5.8h, v1.h[1]
        SXTL        v4.8h, v4.8b
        SMLAL       v18.4s, v5.4h, v2.h[1]
        SMLAL2      v22.4s, v5.8h, v2.h[1]
        SMLAL       v19.4s, v5.4h, v3.h[1]
        SMLAL2      v23.4s, v5.8h, v3.h[1]
        LDR         d5, [x5, 40]
        INS         v6.d[0], x11
        SMLAL       v24.4s, v4.4h, v0.h[1]
        SMLAL2      v28.4s, v4.8h, v0.h[1]
        SMLAL       v25.4s, v4.4h, v1.h[1]
        SMLAL2      v29.4s, v4.8h, v1.h[1]
        SXTL        v6.8h, v6.8b
        SMLAL       v26.4s, v4.4h, v2.h[1]
        SMLAL2      v30.4s, v4.8h, v2.h[1]
        SMLAL       v27.4s, v4.4h, v3.h[1]
        SMLAL2      v31.4s, v4.8h, v3.h[1]
        LDR         x11, [x5, 48]
        SMLAL       v16.4s, v6.4h, v0.h[2]
        SMLAL2      v20.4s, v6.8h, v0.h[2]
        SMLAL       v17.4s, v6.4h, v1.h[2]
        SXTL        v5.8h, v5.8b
        SMLAL2      v21.4s, v6.8h, v1.h[2]
        SMLAL       v18.4s, v6.4h, v2.h[2]
        SMLAL2      v22.4s, v6.8h, v2.h[2]
        SMLAL       v19.4s, v6.4h, v3.h[2]
        SMLAL2      v23.4s, v6.8h, v3.h[2]
        LDR         d6, [x5, 56]
        INS         v4.d[0], x11
        SMLAL       v24.4s, v5.4h, v0.h[2]
        SMLAL2      v28.4s, v5.8h, v0.h[2]
        SMLAL       v25.4s, v5.4h, v1.h[2]
        SMLAL2      v29.4s, v5.8h, v1.h[2]
        SXTL        v4.8h, v4.8b
        SMLAL       v26.4s, v5.4h, v2.h[2]
        SMLAL2      v30.4s, v5.8h, v2.h[2]
        SMLAL       v27.4s, v5.4h, v3.h[2]
        SMLAL2      v31.4s, v5.8h, v3.h[2]
        LDR         x11, [x5, 64]
        SMLAL       v16.4s, v4.4h, v0.h[3]
        SMLAL2      v20.4s, v4.8h, v0.h[3]
        SMLAL       v17.4s, v4.4h, v1.h[3]
        SMLAL2      v21.4s, v4.8h, v1.h[3]
        SXTL        v6.8h, v6.8b
        SMLAL       v18.4s, v4.4h, v2.h[3]
        SMLAL2      v22.4s, v4.8h, v2.h[3]
        SMLAL       v19.4s, v4.4h, v3.h[3]
        SMLAL2      v23.4s, v4.8h, v3.h[3]
        LDR         d4, [x5, 72]
        INS         v5.d[0], x11
        SMLAL       v24.4s, v6.4h, v0.h[3]
        SMLAL2      v28.4s, v6.8h, v0.h[3]
        SXTL        v5.8h, v5.8b
        SMLAL       v25.4s, v6.4h, v1.h[3]
        SMLAL2      v29.4s, v6.8h, v1.h[3]
        SMLAL       v26.4s, v6.4h, v2.h[3]
        SMLAL2      v30.4s, v6.8h, v2.h[3]
        SMLAL       v27.4s, v6.4h, v3.h[3]
        SMLAL2      v31.4s, v6.8h, v3.h[3]
        LDR         x11, [x5, 80]
        SMLAL       v16.4s, v5.4h, v0.h[4]
        SMLAL2      v20.4s, v5.8h, v0.h[4]
        SMLAL       v17.4s, v5.4h, v1.h[4]
        SMLAL2      v21.4s, v5.8h, v1.h[4]
        SXTL        v4.8h, v4.8b
        SMLAL       v18.4s, v5.4h, v2.h[4]
        SMLAL2      v22.4s, v5.8h, v2.h[4]
        SMLAL       v19.4s, v5.4h, v3.h[4]
        SMLAL2      v23.4s, v5.8h, v3.h[4]
        LDR         d5, [x5, 88]
        INS         v6.d[0], x11
        SMLAL       v24.4s, v4.4h, v0.h[4]
        SMLAL2      v28.4s, v4.8h, v0.h[4]
        SMLAL       v25.4s, v4.4h, v1.h[4]
        SMLAL2      v29.4s, v4.8h, v1.h[4]
        SXTL        v6.8h, v6.8b
        SMLAL       v26.4s, v4.4h, v2.h[4]
        SMLAL2      v30.4s, v4.8h, v2.h[4]
        SMLAL       v27.4s, v4.4h, v3.h[4]
        SMLAL2      v31.4s, v4.8h, v3.h[4]
        LDR         x11, [x5, 96]
        SMLAL       v16.4s, v6.4h, v0.h[5]
        SMLAL2      v20.4s, v6.8h, v0.h[5]
        SMLAL       v17.4s, v6.4h, v1.h[5]
        SMLAL2      v21.4s, v6.8h, v1.h[5]
        SXTL        v5.8h, v5.8b
        SMLAL       v18.4s, v6.4h, v2.h[5]
        SMLAL2      v22.4s, v6.8h, v2.h[5]
        SMLAL       v19.4s, v6.4h, v3.h[5]
        SMLAL2      v23.4s, v6.8h, v3.h[5]
        LDR         d6, [x5, 104]
        INS         v4.d[0], x11
        SMLAL       v24.4s, v5.4h, v0.h[5]
        SMLAL2      v28.4s, v5.8h, v0.h[5]
        SMLAL       v25.4s, v5.4h, v1.h[5]
        SMLAL2      v29.4s, v5.8h, v1.h[5]
        SXTL        v4.8h, v4.8b
        SMLAL       v26.4s, v5.4h, v2.h[5]
        SMLAL2      v30.4s, v5.8h, v2.h[5]
        SMLAL       v27.4s, v5.4h, v3.h[5]
        SMLAL2      v31.4s, v5.8h, v3.h[5]
        SXTL        v6.8h, v6.8b
        SMLAL       v16.4s, v4.4h, v0.h[6]
        SMLAL2      v20.4s, v4.8h, v0.h[6]
        SMLAL       v17.4s, v4.4h, v1.h[6]
        SMLAL2      v21.4s, v4.8h, v1.h[6]
        SMLAL       v18.4s, v4.4h, v2.h[6]
        SMLAL2      v22.4s, v4.8h, v2.h[6]
        SMLAL       v19.4s, v4.4h, v3.h[6]
        SMLAL2      v23.4s, v4.8h, v3.h[6]
        LDR         x11, [x5, 112]
        SMLAL       v24.4s, v6.4h, v0.h[6]
        SMLAL2      v28.4s, v6.8h, v0.h[6]
        SMLAL       v25.4s, v6.4h, v1.h[6]
        SMLAL2      v29.4s, v6.8h, v1.h[6]
        LDR         d5, [x5, 120]
        INS         v4.d[0], x11
        SXTL        v4.8h, v4.8b
        SMLAL       v26.4s, v6.4h, v2.h[6]
        SMLAL2      v30.4s, v6.8h, v2.h[6]
        SMLAL       v27.4s, v6.4h, v3.h[6]
        SMLAL2      v31.4s, v6.8h, v3.h[6]
        SMLAL       v16.4s, v4.4h, v0.h[7]
        SMLAL2      v20.4s, v4.8h, v0.h[7]
        SMLAL       v17.4s, v4.4h, v1.h[7]
        SMLAL2      v21.4s, v4.8h, v1.h[7]
        SXTL        v5.8h, v5.8b
        SMLAL       v18.4s, v4.4h, v2.h[7]
        SMLAL2      v22.4s, v4.8h, v2.h[7]
        SMLAL       v19.4s, v4.4h, v3.h[7]
        SMLAL2      v23.4s, v4.8h, v3.h[7]
        ADD         x5, x5, 128
        SMLAL       v24.4s, v5.4h, v0.h[7]
        SMLAL2      v28.4s, v5.8h, v0.h[7]
        SMLAL       v25.4s, v5.4h, v1.h[7]
        SMLAL2      v29.4s, v5.8h, v1.h[7]
        AND         x0, x2, 7               // kc remainder 0 to 7
        SMLAL       v26.4s, v5.4h, v2.h[7]
        SMLAL2      v30.4s, v5.8h, v2.h[7]
        LDR         x11, [sp, 40]               // reload params pointer
        SMLAL       v27.4s, v5.4h, v3.h[7]
        SMLAL2      v31.4s, v5.8h, v3.h[7]

        # Is there a remainder?- 1 to 7 bytes of A
        CBNZ        x0, 5f

4:
        # ks loop
        SUBS        x9, x9, 32              // ks -= MR * sizeof(int8_t*)
        B.HI        1b

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
        LD1R        {v6.8h}, [x11], 2           // add bias

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
        LD1R        {v4.16b}, [x11], 1          // clamp min value

        SQXTN   v0.8b, v16.8h
        SQXTN   v1.8b, v17.8h
        SQXTN   v2.8b, v18.8h
        SQXTN   v3.8b, v19.8h
        LD1R        {v5.16b}, [x11]             // clamp max value
        SQXTN2  v0.16b, v24.8h
        SQXTN2  v1.16b, v25.8h
        SQXTN2  v2.16b, v26.8h
        SQXTN2  v3.16b, v27.8h
        SUB         x11, x11, 3               // rewind params pointer

        SMAX    v0.16b, v0.16b, v4.16b
        SMAX    v1.16b, v1.16b, v4.16b
        SMAX    v2.16b, v2.16b, v4.16b
        SMAX    v3.16b, v3.16b, v4.16b
        SUBS        x1, x1, 16
        SMIN    v0.16b, v0.16b, v5.16b
        SMIN    v1.16b, v1.16b, v5.16b
        SMIN    v2.16b, v2.16b, v5.16b
        SMIN    v3.16b, v3.16b, v5.16b
        B.LO        6f

        # Store full 4 x 16
        ST1         {v3.16b},  [x7], x10
        ST1         {v2.16b}, [x17], x10
        ST1         {v1.16b}, [x16], x10
        ST1         {v0.16b},  [x6], x10

        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b

        # Restore x20-x21 from stack
        LDP         x20, x21, [sp], 16
        RET

        # Remainder- 1 to 7 bytes of A
        .p2align    3
5:
        AND         x0, x2, 7               // kc remainder 1 to 7

        LD1         {v0.8b}, [x13], x0
        LDP         d4, d5, [x5], 16
        LD1         {v1.8b}, [x14], x0
        LD1         {v2.8b}, [x15], x0
        LD1         {v3.8b}, [x20], x0
        SXTL    v0.8h, v0.8b
        SXTL        v4.8h, v4.8b
        SXTL        v5.8h, v5.8b
        SXTL    v1.8h, v1.8b
        SXTL    v2.8h, v2.8b
        SXTL    v3.8h, v3.8b
        SMLAL       v16.4s, v4.4h, v0.h[0]
        SMLAL2      v20.4s, v4.8h, v0.h[0]
        SMLAL       v24.4s, v5.4h, v0.h[0]
        SMLAL2      v28.4s, v5.8h, v0.h[0]
        SMLAL       v17.4s, v4.4h, v1.h[0]
        SMLAL2      v21.4s, v4.8h, v1.h[0]
        SMLAL       v25.4s, v5.4h, v1.h[0]
        SMLAL2      v29.4s, v5.8h, v1.h[0]
        SMLAL       v18.4s, v4.4h, v2.h[0]
        SMLAL2      v22.4s, v4.8h, v2.h[0]
        SMLAL       v26.4s, v5.4h, v2.h[0]
        SMLAL2      v30.4s, v5.8h, v2.h[0]
        SMLAL       v19.4s, v4.4h, v3.h[0]
        SMLAL2      v23.4s, v4.8h, v3.h[0]
        SMLAL       v27.4s, v5.4h, v3.h[0]
        SMLAL2      v31.4s, v5.8h, v3.h[0]
        CMP         x0, 2
        B.LO        4b

        LDP         d4, d5, [x5], 16
        SXTL        v4.8h, v4.8b
        SXTL        v5.8h, v5.8b
        SMLAL       v16.4s, v4.4h, v0.h[1]
        SMLAL2      v20.4s, v4.8h, v0.h[1]
        SMLAL       v24.4s, v5.4h, v0.h[1]
        SMLAL2      v28.4s, v5.8h, v0.h[1]
        SMLAL       v17.4s, v4.4h, v1.h[1]
        SMLAL2      v21.4s, v4.8h, v1.h[1]
        SMLAL       v25.4s, v5.4h, v1.h[1]
        SMLAL2      v29.4s, v5.8h, v1.h[1]
        SMLAL       v18.4s, v4.4h, v2.h[1]
        SMLAL2      v22.4s, v4.8h, v2.h[1]
        SMLAL       v26.4s, v5.4h, v2.h[1]
        SMLAL2      v30.4s, v5.8h, v2.h[1]
        SMLAL       v19.4s, v4.4h, v3.h[1]
        SMLAL2      v23.4s, v4.8h, v3.h[1]
        SMLAL       v27.4s, v5.4h, v3.h[1]
        SMLAL2      v31.4s, v5.8h, v3.h[1]
        B.EQ        4b

        LDP         d4, d5, [x5], 16
        SXTL        v4.8h, v4.8b
        SXTL        v5.8h, v5.8b
        SMLAL       v16.4s, v4.4h, v0.h[2]
        SMLAL2      v20.4s, v4.8h, v0.h[2]
        SMLAL       v24.4s, v5.4h, v0.h[2]
        SMLAL2      v28.4s, v5.8h, v0.h[2]
        SMLAL       v17.4s, v4.4h, v1.h[2]
        SMLAL2      v21.4s, v4.8h, v1.h[2]
        SMLAL       v25.4s, v5.4h, v1.h[2]
        SMLAL2      v29.4s, v5.8h, v1.h[2]
        SMLAL       v18.4s, v4.4h, v2.h[2]
        SMLAL2      v22.4s, v4.8h, v2.h[2]
        SMLAL       v26.4s, v5.4h, v2.h[2]
        SMLAL2      v30.4s, v5.8h, v2.h[2]
        SMLAL       v19.4s, v4.4h, v3.h[2]
        SMLAL2      v23.4s, v4.8h, v3.h[2]
        SMLAL       v27.4s, v5.4h, v3.h[2]
        SMLAL2      v31.4s, v5.8h, v3.h[2]
        CMP         x0, 4
        B.LO        4b

        LDP         d4, d5, [x5], 16
        SXTL        v4.8h, v4.8b
        SXTL        v5.8h, v5.8b
        SMLAL       v16.4s, v4.4h, v0.h[3]
        SMLAL2      v20.4s, v4.8h, v0.h[3]
        SMLAL       v24.4s, v5.4h, v0.h[3]
        SMLAL2      v28.4s, v5.8h, v0.h[3]
        SMLAL       v17.4s, v4.4h, v1.h[3]
        SMLAL2      v21.4s, v4.8h, v1.h[3]
        SMLAL       v25.4s, v5.4h, v1.h[3]
        SMLAL2      v29.4s, v5.8h, v1.h[3]
        SMLAL       v18.4s, v4.4h, v2.h[3]
        SMLAL2      v22.4s, v4.8h, v2.h[3]
        SMLAL       v26.4s, v5.4h, v2.h[3]
        SMLAL2      v30.4s, v5.8h, v2.h[3]
        SMLAL       v19.4s, v4.4h, v3.h[3]
        SMLAL2      v23.4s, v4.8h, v3.h[3]
        SMLAL       v27.4s, v5.4h, v3.h[3]
        SMLAL2      v31.4s, v5.8h, v3.h[3]
        B.EQ        4b

        LDP         d4, d5, [x5], 16
        SXTL        v4.8h, v4.8b
        SXTL        v5.8h, v5.8b
        SMLAL       v16.4s, v4.4h, v0.h[4]
        SMLAL2      v20.4s, v4.8h, v0.h[4]
        SMLAL       v24.4s, v5.4h, v0.h[4]
        SMLAL2      v28.4s, v5.8h, v0.h[4]
        SMLAL       v17.4s, v4.4h, v1.h[4]
        SMLAL2      v21.4s, v4.8h, v1.h[4]
        SMLAL       v25.4s, v5.4h, v1.h[4]
        SMLAL2      v29.4s, v5.8h, v1.h[4]
        SMLAL       v18.4s, v4.4h, v2.h[4]
        SMLAL2      v22.4s, v4.8h, v2.h[4]
        SMLAL       v26.4s, v5.4h, v2.h[4]
        SMLAL2      v30.4s, v5.8h, v2.h[4]
        SMLAL       v19.4s, v4.4h, v3.h[4]
        SMLAL2      v23.4s, v4.8h, v3.h[4]
        SMLAL       v27.4s, v5.4h, v3.h[4]
        SMLAL2      v31.4s, v5.8h, v3.h[4]
        CMP         x0, 6
        B.LO        4b

        LDP         d4, d5, [x5], 16
        SXTL        v4.8h, v4.8b
        SXTL        v5.8h, v5.8b
        SMLAL       v16.4s, v4.4h, v0.h[5]
        SMLAL2      v20.4s, v4.8h, v0.h[5]
        SMLAL       v24.4s, v5.4h, v0.h[5]
        SMLAL2      v28.4s, v5.8h, v0.h[5]
        SMLAL       v17.4s, v4.4h, v1.h[5]
        SMLAL2      v21.4s, v4.8h, v1.h[5]
        SMLAL       v25.4s, v5.4h, v1.h[5]
        SMLAL2      v29.4s, v5.8h, v1.h[5]
        SMLAL       v18.4s, v4.4h, v2.h[5]
        SMLAL2      v22.4s, v4.8h, v2.h[5]
        SMLAL       v26.4s, v5.4h, v2.h[5]
        SMLAL2      v30.4s, v5.8h, v2.h[5]
        SMLAL       v19.4s, v4.4h, v3.h[5]
        SMLAL2      v23.4s, v4.8h, v3.h[5]
        SMLAL       v27.4s, v5.4h, v3.h[5]
        SMLAL2      v31.4s, v5.8h, v3.h[5]
        B.EQ        4b

        LDP         d4, d5, [x5], 16
        SXTL        v4.8h, v4.8b
        SXTL        v5.8h, v5.8b
        SMLAL       v16.4s, v4.4h, v0.h[6]
        SMLAL2      v20.4s, v4.8h, v0.h[6]
        SMLAL       v24.4s, v5.4h, v0.h[6]
        SMLAL2      v28.4s, v5.8h, v0.h[6]
        SMLAL       v17.4s, v4.4h, v1.h[6]
        SMLAL2      v21.4s, v4.8h, v1.h[6]
        SMLAL       v25.4s, v5.4h, v1.h[6]
        SMLAL2      v29.4s, v5.8h, v1.h[6]
        SMLAL       v18.4s, v4.4h, v2.h[6]
        SMLAL2      v22.4s, v4.8h, v2.h[6]
        SMLAL       v26.4s, v5.4h, v2.h[6]
        SMLAL2      v30.4s, v5.8h, v2.h[6]
        SMLAL       v19.4s, v4.4h, v3.h[6]
        SMLAL2      v23.4s, v4.8h, v3.h[6]
        SMLAL       v27.4s, v5.4h, v3.h[6]
        SMLAL2      v31.4s, v5.8h, v3.h[6]
        B           4b

        # Store odd width
        .p2align    3
6:
        TBZ         x1, 3, 7f
        STR         d3, [x7], 8
        STR         d2, [x17], 8
        DUP         d3, v3.d[1]
        DUP         d2, v2.d[1]
        STR         d1, [x16], 8
        STR         d0, [x6], 8
        DUP         d1, v1.d[1]
        DUP         d0, v0.d[1]
7:
        TBZ         x1, 2, 8f
        STR         s3, [x7], 4
        STR         s2, [x17], 4
        DUP         s3, v3.s[1]
        DUP         s2, v2.s[1]
        STR         s1, [x16], 4
        STR         s0, [x6], 4
        DUP         s1, v1.s[1]
        DUP         s0, v0.s[1]
8:
        TBZ         x1, 1, 9f
        STR         h3, [x7], 2
        STR         h2, [x17], 2
        DUP         h3, v3.h[1]
        DUP         h2, v2.h[1]
        STR         h1, [x16], 2
        STR         h0, [x6], 2
        DUP         h1, v1.h[1]
        DUP         h0, v0.h[1]
9:
        TBZ         x1, 0, 10f
        STR         b3, [x7]
        STR         b2, [x17]
        STR         b1, [x16]
        STR         b0, [x6]
10:
        # Restore x20-x21 from stack
        LDP         x20, x21, [sp], 16
        RET

END_FUNCTION xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
