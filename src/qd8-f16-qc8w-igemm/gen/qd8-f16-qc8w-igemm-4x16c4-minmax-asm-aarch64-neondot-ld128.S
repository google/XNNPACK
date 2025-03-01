// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/4x16c4-aarch64-neondot-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128(
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
#     const union xnn_f16_minmax_params *params,  [sp + 32]  -> x11
#     const struct xnn_qd8_quantization_params *quantization_params) [sp + 40] -> x16

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x13  v0
// A1  x14  v1
// A2  x15  v2
// A3  x10  v3
// B    x5  v4  v5  v6  v7
// C0   x6 v16 v20 v24 v28
// C1  x16 v17 v21 v25 v29
// C2  x17 v18 v22 v26 v30
// C3   x7 v19 v23 v27 v31
// unused v8 v9 v10 v11 v12 v13 v14 v15

BEGIN_FUNCTION xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        LDR         x8, [sp, 8]             // Load a_offset
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x16, x6,  x16, LO       //   c1 = c0
        ADD         x2, x2, 3               // kc = (kc + 3) & ~3

        ADD         x17, x16, x7            // c2 = c1 + cm_stride
        LDR         x12, [sp, 16]       // Load zero
        LDR         x11, [sp, 32]      // Load params pointer
                                            // if mr <= 2
        CSEL        x17, x16, x17, LS       //   c2 = c1
        BIC         x2, x2, 3
        STR         x19, [sp, -56]      // Push x19 to the stack
        LDR         x19, [sp, 24]       // Load zero_data
        LDR         x15, [sp, 40]           // &quantization_params[0].zero_point
        STP         d8,  d9, [sp, -48]!
        STP         d10, d11, [sp, 16]
        STP         d12,  d13, [sp, 32]
        LD1         {v12.4s}, [x15]         // v12 & v13 interleaved zero_point & scale

        CMP         x0, 4                   // if mr < 4
        ADD         x7,  x17, x7            // c3 = c2 + cm_stride
        CSEL        x7,  x17, x7, LO        //   c3 = c2

        .p2align    3
0:
        # Load initial bias from w into accumulators
        LDP         q16, q20, [x5], 32
        MUL         v16.4s, v16.4s, v12.s[0]
        MUL         v20.4s, v20.4s, v12.s[0]
        MOV         v17.16b, v16.16b
        MOV         v18.16b, v16.16b
        LDP         q24, q28, [x5], 32
        MUL         v24.4s, v24.4s, v12.s[0]
        MUL         v28.4s, v28.4s, v12.s[0]
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
        CMP         x10, x12                    // if a3 == zero
        ADD         x10, x10, x8                // a3 += a_offset
        CSEL        x10, x19, x10, EQ       //   a3 = zero_data, else a3 += a_offset

        # Is there at least 16 bytes for main loop?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        4f

        # Main loop - 16 bytes of A
        .p2align    3
2:
        LDR         q0, [x13], 16
        LDR         q4,  [x5], 16
        LDR         q1, [x14], 16
        LDR         q2, [x15], 16
        LDR         q3,  [x10], 16
        LDR         q5,  [x5], 16
        SDOT        v16.4s, v4.16b,  v0.4b[0]
        SDOT        v17.4s, v4.16b,  v1.4b[0]
        LDP         q6, q7, [x5], 32
        SDOT        v18.4s, v4.16b,  v2.4b[0]
        SDOT        v19.4s, v4.16b,  v3.4b[0]
        SDOT        v20.4s, v5.16b,  v0.4b[0]
        SDOT        v21.4s, v5.16b,  v1.4b[0]
        SDOT        v22.4s, v5.16b,  v2.4b[0]
        SDOT        v23.4s, v5.16b,  v3.4b[0]
        SDOT        v24.4s, v6.16b, v0.4b[0]
        SDOT        v25.4s, v6.16b, v1.4b[0]
        LDP         q4, q5, [x5], 32
        SDOT        v26.4s, v6.16b, v2.4b[0]
        SDOT        v27.4s, v6.16b, v3.4b[0]
        SDOT        v28.4s, v7.16b, v0.4b[0]
        SDOT        v29.4s, v7.16b, v1.4b[0]
        SDOT        v30.4s, v7.16b, v2.4b[0]
        SDOT        v31.4s, v7.16b, v3.4b[0]

        SDOT        v16.4s, v4.16b,  v0.4b[1]
        SDOT        v17.4s, v4.16b,  v1.4b[1]
        LDP         q6, q7, [x5], 32
        SDOT        v18.4s, v4.16b,  v2.4b[1]
        SDOT        v19.4s, v4.16b,  v3.4b[1]
        SDOT        v20.4s, v5.16b,  v0.4b[1]
        SDOT        v21.4s, v5.16b,  v1.4b[1]
        SDOT        v22.4s, v5.16b,  v2.4b[1]
        SDOT        v23.4s, v5.16b,  v3.4b[1]
        SDOT        v24.4s, v6.16b,  v0.4b[1]
        SDOT        v25.4s, v6.16b,  v1.4b[1]
        LDP         q4, q5, [x5], 32
        SDOT        v26.4s, v6.16b,  v2.4b[1]
        SDOT        v27.4s, v6.16b,  v3.4b[1]
        SDOT        v28.4s, v7.16b,  v0.4b[1]
        SDOT        v29.4s, v7.16b,  v1.4b[1]
        SDOT        v30.4s, v7.16b,  v2.4b[1]
        SDOT        v31.4s, v7.16b,  v3.4b[1]

        SDOT        v16.4s, v4.16b,  v0.4b[2]
        SDOT        v17.4s, v4.16b,  v1.4b[2]
        LDP         q6, q7, [x5], 32
        SDOT        v18.4s, v4.16b,  v2.4b[2]
        SDOT        v19.4s, v4.16b,  v3.4b[2]
        SDOT        v20.4s, v5.16b,  v0.4b[2]
        SDOT        v21.4s, v5.16b,  v1.4b[2]
        SDOT        v22.4s, v5.16b,  v2.4b[2]
        SDOT        v23.4s, v5.16b,  v3.4b[2]
        SDOT        v24.4s, v6.16b,  v0.4b[2]
        SDOT        v25.4s, v6.16b,  v1.4b[2]
        LDP         q4, q5, [x5], 32
        SDOT        v26.4s, v6.16b,  v2.4b[2]
        SDOT        v27.4s, v6.16b,  v3.4b[2]
        SDOT        v28.4s, v7.16b,  v0.4b[2]
        SDOT        v29.4s, v7.16b,  v1.4b[2]
        SDOT        v30.4s, v7.16b,  v2.4b[2]
        SDOT        v31.4s, v7.16b,  v3.4b[2]

        SDOT        v16.4s, v4.16b,  v0.4b[3]
        SDOT        v17.4s, v4.16b,  v1.4b[3]
        LDP         q6, q7, [x5], 32
        SDOT        v18.4s, v4.16b,  v2.4b[3]
        SDOT        v19.4s, v4.16b,  v3.4b[3]
        SDOT        v20.4s, v5.16b,  v0.4b[3]
        SDOT        v21.4s, v5.16b,  v1.4b[3]
        SDOT        v22.4s, v5.16b,  v2.4b[3]
        SDOT        v23.4s, v5.16b,  v3.4b[3]
        SDOT        v24.4s, v6.16b,  v0.4b[3]
        SDOT        v25.4s, v6.16b,  v1.4b[3]
        SDOT        v26.4s, v6.16b,  v2.4b[3]
        SDOT        v27.4s, v6.16b,  v3.4b[3]
        SUBS        x0, x0, 16
        SDOT        v28.4s, v7.16b,  v0.4b[3]
        SDOT        v29.4s, v7.16b,  v1.4b[3]
        SDOT        v30.4s, v7.16b,  v2.4b[3]
        SDOT        v31.4s, v7.16b,  v3.4b[3]
        B.HS        2b

        # Is there a remainder?- 4 to 12 bytes of A
        TST         x0, 15
        B.NE        4f

3:
        # ks loop
        SUBS        x9, x9, 32              // ks -= MR * sizeof(int8_t*)
        B.HI        1b

        LDP         q0, q1, [x5], 32        // kernel_scale
        SCVTF       v19.4s, v19.4s
        SCVTF       v23.4s, v23.4s
        SCVTF       v27.4s, v27.4s
        SCVTF       v31.4s, v31.4s
        SCVTF       v18.4s, v18.4s
        SCVTF       v22.4s, v22.4s
        SCVTF       v26.4s, v26.4s
        LDP         q2, q3, [x5], 32
        SCVTF       v30.4s, v30.4s
        SCVTF       v17.4s, v17.4s
        SCVTF       v21.4s, v21.4s
        SCVTF       v25.4s, v25.4s
        SCVTF       v29.4s, v29.4s
        SCVTF       v16.4s, v16.4s
        SCVTF       v20.4s, v20.4s
        SCVTF       v24.4s, v24.4s
        SCVTF       v28.4s, v28.4s
        FMUL        v8.4s, v0.4s, v12.s[1]      // kernel_scale * scale
        FMUL        v9.4s, v1.4s, v12.s[1]
        FMUL        v10.4s, v2.4s, v12.s[1]
        FMUL        v11.4s, v3.4s, v12.s[1]
        FMUL        v4.4s, v0.4s, v12.s[1]
        FMUL        v5.4s, v1.4s, v12.s[1]
        FMUL        v6.4s, v2.4s, v12.s[1]
        FMUL        v7.4s, v3.4s, v12.s[1]
        LDP         q0, q1, [x5], 32        // bias
        FMUL        v19.4s, v19.4s, v8.4s
        FMUL        v23.4s, v23.4s, v9.4s
        FMUL        v27.4s, v27.4s, v10.4s
        FMUL        v31.4s, v31.4s, v11.4s
        FMUL        v18.4s, v18.4s, v4.4s
        FMUL        v22.4s, v22.4s, v5.4s
        FMUL        v26.4s, v26.4s, v6.4s
        FMUL        v30.4s, v30.4s, v7.4s
        LDP         q2, q3, [x5], 32
        FMUL        v17.4s, v17.4s, v8.4s
        FMUL        v21.4s, v21.4s, v9.4s
        FMUL        v25.4s, v25.4s, v10.4s
        FMUL        v29.4s, v29.4s, v11.4s
        FMUL        v16.4s, v16.4s, v4.4s
        FMUL        v20.4s, v20.4s, v5.4s
        FMUL        v24.4s, v24.4s, v6.4s
        FMUL        v28.4s, v28.4s, v7.4s
        LD2R        {v4.8h, v5.8h}, [x11]       // min max
        FADD        v19.4s, v19.4s, v0.4s
        FADD        v23.4s, v23.4s, v1.4s
        FADD        v27.4s, v27.4s, v2.4s
        FADD        v31.4s, v31.4s, v3.4s
        FADD        v18.4s, v18.4s, v0.4s
        FADD        v22.4s, v22.4s, v1.4s
        FADD        v26.4s, v26.4s, v2.4s
        FADD        v30.4s, v30.4s, v3.4s
        FADD        v17.4s, v17.4s, v0.4s
        FADD        v21.4s, v21.4s, v1.4s
        FADD        v25.4s, v25.4s, v2.4s
        FADD        v29.4s, v29.4s, v3.4s
        FADD        v16.4s, v16.4s, v0.4s
        FADD        v20.4s, v20.4s, v1.4s
        FADD        v24.4s, v24.4s, v2.4s
        FADD        v28.4s, v28.4s, v3.4s
        FCVTN       v19.4h, v19.4s
        FCVTN       v27.4h, v27.4s
        FCVTN       v18.4h, v18.4s
        FCVTN       v26.4h, v26.4s
        FCVTN       v17.4h, v17.4s
        FCVTN       v25.4h, v25.4s
        FCVTN       v16.4h, v16.4s
        FCVTN       v24.4h, v24.4s
        FCVTN2      v19.8h, v23.4s
        FCVTN2      v27.8h, v31.4s
        FCVTN2      v18.8h, v22.4s
        FCVTN2      v26.8h, v30.4s
        FCVTN2      v17.8h, v21.4s
        FCVTN2      v25.8h, v29.4s
        FCVTN2      v16.8h, v20.4s
        FCVTN2      v24.8h, v28.4s
        LDR         x0, [sp, 48]                // cn_stride
        FMAX        v19.8h, v19.8h, v4.8h
        FMAX        v27.8h, v27.8h, v4.8h
        FMAX        v18.8h, v18.8h, v4.8h
        FMAX        v26.8h, v26.8h, v4.8h
        FMAX        v17.8h, v17.8h, v4.8h
        FMAX        v25.8h, v25.8h, v4.8h
        FMAX        v16.8h, v16.8h, v4.8h
        FMAX        v24.8h, v24.8h, v4.8h
        SUBS        x1, x1, 16
        FMIN        v19.8h, v19.8h, v5.8h
        FMIN        v27.8h, v27.8h, v5.8h
        FMIN        v18.8h, v18.8h, v5.8h
        FMIN        v26.8h, v26.8h, v5.8h
        FMIN        v17.8h, v17.8h, v5.8h
        FMIN        v25.8h, v25.8h, v5.8h
        FMIN        v16.8h, v16.8h, v5.8h
        FMIN        v24.8h, v24.8h, v5.8h
        B.LO        6f

        STP         q19, q27, [x7]
        ADD         x7, x7, x0
        STP         q18, q26, [x17]
        ADD         x17, x17, x0
        STP         q17, q25, [x16]
        ADD         x16, x16, x0
        STP         q16, q24, [x6]
        ADD         x6, x6, x0
        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b
        # Restore d8-d13 from stack
        LDR         x19, [sp, -8]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 48
        RET


        # Remainder- 8 bytes of A
        .p2align    3
4:
        # Is there a remainder?- 8 bytes of A
        TBZ         x0, 3, 5f

        LDR         d0, [x13], 8
        LDR         q4,  [x5], 16
        LDR         d1, [x14], 8
        LDR         d2, [x15], 8
        LDR         d3,  [x10], 8
        LDR         q5,  [x5], 16
        SDOT        v16.4s, v4.16b,  v0.4b[0]
        SDOT        v17.4s, v4.16b,  v1.4b[0]
        LDP         q6, q7, [x5], 32
        SDOT        v18.4s, v4.16b,  v2.4b[0]
        SDOT        v19.4s, v4.16b,  v3.4b[0]
        SDOT        v20.4s, v5.16b,  v0.4b[0]
        SDOT        v21.4s, v5.16b,  v1.4b[0]
        SDOT        v22.4s, v5.16b,  v2.4b[0]
        SDOT        v23.4s, v5.16b,  v3.4b[0]
        SDOT        v24.4s, v6.16b, v0.4b[0]
        SDOT        v25.4s, v6.16b, v1.4b[0]
        LDP         q4, q5, [x5], 32
        SDOT        v26.4s, v6.16b, v2.4b[0]
        SDOT        v27.4s, v6.16b, v3.4b[0]
        SDOT        v28.4s, v7.16b, v0.4b[0]
        SDOT        v29.4s, v7.16b, v1.4b[0]
        SDOT        v30.4s, v7.16b, v2.4b[0]
        SDOT        v31.4s, v7.16b, v3.4b[0]
        SDOT        v16.4s, v4.16b,  v0.4b[1]
        SDOT        v17.4s, v4.16b,  v1.4b[1]
        LDP         q6, q7, [x5], 32
        SDOT        v18.4s, v4.16b,  v2.4b[1]
        SDOT        v19.4s, v4.16b,  v3.4b[1]
        SDOT        v20.4s, v5.16b,  v0.4b[1]
        SDOT        v21.4s, v5.16b,  v1.4b[1]
        SDOT        v22.4s, v5.16b,  v2.4b[1]
        SDOT        v23.4s, v5.16b,  v3.4b[1]
        SDOT        v24.4s, v6.16b,  v0.4b[1]
        SDOT        v25.4s, v6.16b,  v1.4b[1]
        SDOT        v26.4s, v6.16b,  v2.4b[1]
        SDOT        v27.4s, v6.16b,  v3.4b[1]
        SDOT        v28.4s, v7.16b,  v0.4b[1]
        SDOT        v29.4s, v7.16b,  v1.4b[1]
        SDOT        v30.4s, v7.16b,  v2.4b[1]
        SDOT        v31.4s, v7.16b,  v3.4b[1]
        # Is there a remainder?- 4 bytes of A
        TBZ         x0, 2, 3b

        # Remainder- 4 bytes of A
5:
        LDR         s0, [x13], 4
        LDR         q4, [x5], 16
        LDR         s1, [x14], 4
        LDR         s2, [x15], 4
        LDR         s3,  [x10], 4
        LDR         q5, [x5], 16
        SDOT        v16.4s, v4.16b,  v0.4b[0]
        SDOT        v17.4s, v4.16b,  v1.4b[0]
        LDP         q6, q7, [x5], 32
        SDOT        v18.4s, v4.16b,  v2.4b[0]
        SDOT        v19.4s, v4.16b,  v3.4b[0]
        SDOT        v20.4s, v5.16b,  v0.4b[0]
        SDOT        v21.4s, v5.16b,  v1.4b[0]
        SDOT        v22.4s, v5.16b,  v2.4b[0]
        SDOT        v23.4s, v5.16b,  v3.4b[0]
        SDOT        v24.4s, v6.16b, v0.4b[0]
        SDOT        v25.4s, v6.16b, v1.4b[0]
        SDOT        v26.4s, v6.16b, v2.4b[0]
        SDOT        v27.4s, v6.16b, v3.4b[0]
        SDOT        v28.4s, v7.16b, v0.4b[0]
        SDOT        v29.4s, v7.16b, v1.4b[0]
        SDOT        v30.4s, v7.16b, v2.4b[0]
        SDOT        v31.4s, v7.16b, v3.4b[0]
        B           3b

        # Store odd width
        .p2align    3
6:
        TBZ         x1, 3, 7f
        STR         q19, [x7], 16
        STR         q18, [x17], 16
        MOV         v19.16b, v27.16b
        MOV         v18.16b, v26.16b
        STR         q17, [x16], 16
        STR         q16, [x6], 16
        MOV         v17.16b, v25.16b
        MOV         v16.16b, v24.16b
7:
        TBZ         x1, 2, 8f
        STR         d19, [x7], 8
        STR         d18, [x17], 8
        DUP         d19, v19.d[1]
        DUP         d18, v18.d[1]
        STR         d17, [x16], 8
        STR         d16, [x6], 8
        DUP         d17, v17.d[1]
        DUP         d16, v16.d[1]
8:
        TBZ         x1, 1, 9f
        STR         s19, [x7], 4
        STR         s18, [x17], 4
        DUP         s19, v19.s[1]
        DUP         s18, v18.s[1]
        STR         s17, [x16], 4
        STR         s16, [x6], 4
        DUP         s17, v17.s[1]
        DUP         s16, v16.s[1]
9:
        TBZ         x1, 0, 10f
        STR         h19, [x7]
        STR         h18, [x17]
        STR         h17, [x16]
        STR         h16, [x6]
10:
        # Restore d8-d13 from stack
        LDR         x19, [sp, -8]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 48
        RET

END_FUNCTION xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
