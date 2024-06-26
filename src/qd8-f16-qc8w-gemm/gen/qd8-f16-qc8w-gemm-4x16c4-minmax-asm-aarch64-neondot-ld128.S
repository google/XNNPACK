// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     const int8_t* restrict a,  x3
#     size_t a_stride,           x4
#     const void* restrict w,    x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,          [sp] -> x12
#     const union xnn_f16_minmax_params *params,  [sp + 8]  -> x11
#     const struct xnn_qd8_quantization_params *quantization_params) [sp + 16] -> x16

# params structure is 8 bytes
# struct {
#  int32_t zero_point;
#  float scale;
# } scalar;

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3 v0
// A1 x15 v1
// A2 x13 v2
// A3  x4 v3
// B   x5 v4  v5  v6  v7
// C0  x6 v16 v20 v24 v28
// C1  x8 v17 v21 v25 v29
// C2  x9 v18 v22 v26 v30
// C3  x7 v19 v23 v27 v31
// unused v14 v15

BEGIN_FUNCTION xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128

        # Clamp A and C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x2, x2, 3               // kc = (kc + 3) & ~3
        ADD         x15, x3, x4             // a1 = a0 + a_stride
        ADD         x8, x6, x7              // c1 = c0 + cm_stride
        CSEL        x15, x3, x15, LO        //   a1 = a0
        CSEL        x8, x6,  x8, LO         //   c1 = c0
        BIC         x2, x2, 3

        ADD         x13, x15, x4            // a2 = a1 + a_stride
        ADD         x9,  x8, x7             // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x13, x15, x13, LS       //   a2 = a1
        CSEL        x9,  x8,  x9, LS        //   c2 = c1

        LDP         x12, x11, [sp]          // cn_stride, params
        LDR         x16, [sp, 16]           // &quantization_params[0].zero_point
        STP         d8,  d9, [sp, -48]!
        STP         d10, d11, [sp, 16]
        STP         d12,  d13, [sp, 32]
        LDP         q12, q13, [x16]         // v12 & v13 interleaved zero_point & scale

        CMP         x0, 4                   // if mr < 4
        ADD         x4, x13, x4             // a3 = a2 + a_stride
        ADD         x7,  x9, x7             // c3 = c2 + cm_stride
        CSEL        x4, x13, x4, LO         //   a3 = a2
        CSEL        x7,  x9, x7, LO         //   c3 = c2

        .p2align    3
0:
        # Load initial bias from w into accumulators
        SUBS        x0, x2, 16              // k = kc - 16
        LDP         q0, q1, [x5], 32
        MUL         v16.4s, v0.4s, v12.s[0]
        MUL         v17.4s, v0.4s, v12.s[2]
        MUL         v18.4s, v0.4s, v13.s[0]
        LDP         q2, q3, [x5], 32
        MUL         v19.4s, v0.4s, v13.s[2]
        MUL         v20.4s, v1.4s, v12.s[0]
        MUL         v21.4s, v1.4s, v12.s[2]
        MUL         v22.4s, v1.4s, v13.s[0]
        MUL         v23.4s, v1.4s, v13.s[2]
        MUL         v24.4s, v2.4s, v12.s[0]
        MUL         v25.4s, v2.4s, v12.s[2]
        MUL         v26.4s, v2.4s, v13.s[0]
        MUL         v27.4s, v2.4s, v13.s[2]
        MUL         v28.4s, v3.4s, v12.s[0]
        MUL         v29.4s, v3.4s, v12.s[2]
        MUL         v30.4s, v3.4s, v13.s[0]
        MUL         v31.4s, v3.4s, v13.s[2]

        # Is there at least 16 bytes?
        B.LO        3f

        # Main loop - 16 bytes of A
        .p2align    3
1:
        LDR         q0,  [x3], 16
        LDR         q4,  [x5], 16
        LDR         q1, [x15], 16
        LDR         q2, [x13], 16
        LDR         q3,  [x4], 16
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
        B.HS        1b

        # Is there a remainder?- 4 to 12 bytes of A
        TST         x0, 15
        B.NE        3f

2:
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
        FMUL        v8.4s, v0.4s, v13.s[3]      // kernel_scale * scale
        FMUL        v9.4s, v1.4s, v13.s[3]
        FMUL        v10.4s, v2.4s, v13.s[3]
        FMUL        v11.4s, v3.4s, v13.s[3]
        FMUL        v4.4s, v0.4s, v13.s[1]
        FMUL        v5.4s, v1.4s, v13.s[1]
        FMUL        v6.4s, v2.4s, v13.s[1]
        FMUL        v7.4s, v3.4s, v13.s[1]
        FMUL        v19.4s, v19.4s, v8.4s
        FMUL        v8.4s, v0.4s, v12.s[3]
        FMUL        v23.4s, v23.4s, v9.4s
        FMUL        v9.4s, v1.4s, v12.s[3]
        FMUL        v27.4s, v27.4s, v10.4s
        FMUL        v10.4s, v2.4s, v12.s[3]
        FMUL        v31.4s, v31.4s, v11.4s
        FMUL        v11.4s, v3.4s, v12.s[3]
        FMUL        v18.4s, v18.4s, v4.4s
        FMUL        v4.4s, v0.4s, v12.s[1]
        FMUL        v22.4s, v22.4s, v5.4s
        FMUL        v5.4s, v1.4s, v12.s[1]
        LDP         q0, q1, [x5], 32        // bias
        FMUL        v26.4s, v26.4s, v6.4s
        FMUL        v6.4s, v2.4s, v12.s[1]
        FMUL        v30.4s, v30.4s, v7.4s
        FMUL        v7.4s, v3.4s, v12.s[1]
        FMUL        v17.4s, v17.4s, v8.4s
        FMUL        v21.4s, v21.4s, v9.4s
        FMUL        v25.4s, v25.4s, v10.4s
        FMUL        v29.4s, v29.4s, v11.4s
        LDP         q2, q3, [x5], 32
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
        FMAX        v19.8h, v19.8h, v4.8h
        FMAX        v27.8h, v27.8h, v4.8h
        FMAX        v18.8h, v18.8h, v4.8h
        FMAX        v26.8h, v26.8h, v4.8h
        FMAX        v17.8h, v17.8h, v4.8h
        FMAX        v25.8h, v25.8h, v4.8h
        FMAX        v16.8h, v16.8h, v4.8h
        FMAX        v24.8h, v24.8h, v4.8h
        FMIN        v19.8h, v19.8h, v5.8h
        FMIN        v27.8h, v27.8h, v5.8h
        FMIN        v18.8h, v18.8h, v5.8h
        FMIN        v26.8h, v26.8h, v5.8h
        FMIN        v17.8h, v17.8h, v5.8h
        FMIN        v25.8h, v25.8h, v5.8h
        FMIN        v16.8h, v16.8h, v5.8h
        FMIN        v24.8h, v24.8h, v5.8h

        SUBS        x1, x1, 16
        B.LO        5f
        STP         q19, q27, [x7]
        ADD         x7, x7, x12
        STP         q18, q26, [x9]
        ADD         x9, x9, x12
        STP         q17, q25, [x8]
        ADD         x8, x8, x12
        STP         q16, q24, [x6]
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


        # Remainder- 8 bytes of A
        .p2align    3
3:
        # Is there a remainder?- 8 bytes of A
        TBZ         x0, 3, 4f

        LDR         d0,  [x3], 8
        LDR         q4,  [x5], 16
        LDR         d1, [x15], 8
        LDR         d2, [x13], 8
        LDR         d3,  [x4], 8
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
        TBZ         x0, 2, 2b

        # Remainder- 4 bytes of A
4:
        LDR         s0,  [x3], 4
        LDR         q4, [x5], 16
        LDR         s1, [x15], 4
        LDR         s2, [x13], 4
        LDR         s3,  [x4], 4
        SDOT        v16.4s, v4.16b,  v0.4b[0]
        LDR         q5, [x5], 16
        SDOT        v17.4s, v4.16b,  v1.4b[0]
        SDOT        v18.4s, v4.16b,  v2.4b[0]
        SDOT        v19.4s, v4.16b,  v3.4b[0]
        SDOT        v20.4s, v5.16b,  v0.4b[0]
        LDP         q6, q7, [x5], 32
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
        B           2b

        # Store odd width
        .p2align    3
5:
        TBZ         x1, 3, 6f
        STR         q19, [x7], 16
        STR         q18, [x9], 16
        MOV         v19.16b, v27.16b
        MOV         v18.16b, v26.16b
        STR         q17, [x8], 16
        STR         q16, [x6], 16
        MOV         v17.16b, v25.16b
        MOV         v16.16b, v24.16b
6:
        TBZ         x1, 2, 7f
        STR         d19, [x7], 8
        STR         d18, [x9], 8
        DUP         d19, v19.d[1]
        DUP         d18, v18.d[1]
        STR         d17, [x8], 8
        STR         d16, [x6], 8
        DUP         d17, v17.d[1]
        DUP         d16, v16.d[1]
7:
        TBZ         x1, 1, 8f
        STR         s19, [x7], 4
        STR         s18, [x9], 4
        DUP         s19, v19.s[1]
        DUP         s18, v18.s[1]
        STR         s17, [x8], 4
        STR         s16, [x6], 4
        DUP         s17, v17.s[1]
        DUP         s16, v16.s[1]
8:
        TBZ         x1, 0, 9f
        STR         h19, [x7]
        STR         h18, [x9]
        STR         h17, [x8]
        STR         h16, [x6]
9:
        # Restore d8-d13 from stack
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 48
        RET

END_FUNCTION xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
