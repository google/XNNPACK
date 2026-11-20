// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/2x16c4s2-aarch64-neon-mlal.S.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "src/xnnpack/assembly.h"

# void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4s2__asm_aarch64_neon_mlal(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     const int8_t* restrict a,  x3
#     size_t a_stride,           x4
#     const void* restrict w,    x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,          [sp] -> x10
#     const union xnn_qs8_qc8w_conv_minmax_params params)  [sp + 8] -> x11

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0   x3  v0  v2
// A1   x8  v1  v3
// B    x5  v4  v5  v6  v7
// C0   x6  v16 v17 v18 v19 v20 v21 v22 v23
// C1  x12  v24 v25 v26 v27 v28 v29 v30 v31
// temp0    v8  v10 v12 v14
// temp1    v9  v11 v13 v15

BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4s2__asm_aarch64_neon_mlal
        LDP         x10, x11, [sp]          // cn_stride, params
        STP         d8, d9, [sp, -64]!
        CMP         x0, 2                   // if mr < 2
        STP         d10, d11, [sp, 16]
        ADD         x8, x3, x4              // a1 = a0 + a_stride
        STP         d12, d13, [sp, 32]
        ADD         x12, x6, x7             // c1 = c0 + cm_stride
        STP         d14, d15, [sp, 48]
        CSEL        x8, x3, x8, LO          //   a1 = a0
        ADD         x2, x2, 7               // kc = (kc + 7) & ~7
        CSEL        x12, x6, x12, LO        //   c1 = c0
        BIC         x2, x2, 7

        .p2align    3
0:
        # Load initial bias from w into accumulators
        LDP         q0, q1, [x5], 32
        UXTL        v16.2d, v0.2s
        UXTL2       v17.2d, v0.4s
        UXTL        v18.2d, v1.2s
        UXTL2       v19.2d, v1.4s
        LDP         q0, q1, [x5], 32
        UXTL        v20.2d, v0.2s
        UXTL2       v21.2d, v0.4s
        UXTL        v22.2d, v1.2s
        UXTL2       v23.2d, v1.4s
        MOV         v24.16b, v16.16b
        MOV         v25.16b, v17.16b
        MOV         v26.16b, v18.16b
        MOV         v27.16b, v19.16b
        MOV         v28.16b, v20.16b
        MOV         v29.16b, v21.16b
        MOV         v30.16b, v22.16b
        MOV         v31.16b, v23.16b
        SUBS        x0, x2, 16
        B.LO        2f

        .p2align    3
1:
        LDP         d0, d2, [x3], 16
        LDP         d1, d3, [x8], 16
        LDP         d4, d5, [x5]
        LDP         d6, d7, [x5, 128]
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 16]
        SMLAL       v8.8h, v6.8b, v2.8b
        SMLAL       v9.8h, v6.8b, v3.8b
        SMLAL       v10.8h, v7.8b, v2.8b
        SMLAL       v11.8h, v7.8b, v3.8b
        LDP         d6, d7, [x5, 144]
        SADALP      v16.4s, v8.8h
        SADALP      v24.4s, v9.8h
        SADALP      v17.4s, v10.8h
        SADALP      v25.4s, v11.8h
        SMULL       v12.8h, v4.8b, v0.8b
        SMULL       v13.8h, v4.8b, v1.8b
        SMULL       v14.8h, v5.8b, v0.8b
        SMULL       v15.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 32]
        SMLAL       v12.8h, v6.8b, v2.8b
        SMLAL       v13.8h, v6.8b, v3.8b
        SMLAL       v14.8h, v7.8b, v2.8b
        SMLAL       v15.8h, v7.8b, v3.8b
        LDP         d6, d7, [x5, 160]
        SADALP      v18.4s, v12.8h
        SADALP      v26.4s, v13.8h
        SADALP      v19.4s, v14.8h
        SADALP      v27.4s, v15.8h
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 48]
        SMLAL       v8.8h, v6.8b, v2.8b
        SMLAL       v9.8h, v6.8b, v3.8b
        SMLAL       v10.8h, v7.8b, v2.8b
        SMLAL       v11.8h, v7.8b, v3.8b
        LDP         d6, d7, [x5, 176]
        SADALP      v20.4s, v8.8h
        SADALP      v28.4s, v9.8h
        SADALP      v21.4s, v10.8h
        SADALP      v29.4s, v11.8h
        SMULL       v12.8h, v4.8b, v0.8b
        SMULL       v13.8h, v4.8b, v1.8b
        SMULL       v14.8h, v5.8b, v0.8b
        SMULL       v15.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 64]
        SMLAL       v12.8h, v6.8b, v2.8b
        SMLAL       v13.8h, v6.8b, v3.8b
        SMLAL       v14.8h, v7.8b, v2.8b
        SMLAL       v15.8h, v7.8b, v3.8b
        LDP         d6, d7, [x5, 192]
        SADALP      v22.4s, v12.8h
        SADALP      v30.4s, v13.8h
        SADALP      v23.4s, v14.8h
        SADALP      v31.4s, v15.8h
        EXT         v0.8b, v0.8b, v0.8b, 4
        EXT         v2.8b, v2.8b, v2.8b, 4
        EXT         v1.8b, v1.8b, v1.8b, 4
        EXT         v3.8b, v3.8b, v3.8b, 4
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 80]
        SMLAL       v8.8h, v6.8b, v2.8b
        SMLAL       v9.8h, v6.8b, v3.8b
        SMLAL       v10.8h, v7.8b, v2.8b
        SMLAL       v11.8h, v7.8b, v3.8b
        LDP         d6, d7, [x5, 208]
        SADALP      v16.4s, v8.8h
        SADALP      v24.4s, v9.8h
        SADALP      v17.4s, v10.8h
        SADALP      v25.4s, v11.8h
        SMULL       v12.8h, v4.8b, v0.8b
        SMULL       v13.8h, v4.8b, v1.8b
        SMULL       v14.8h, v5.8b, v0.8b
        SMULL       v15.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 96]
        SMLAL       v12.8h, v6.8b, v2.8b
        SMLAL       v13.8h, v6.8b, v3.8b
        SMLAL       v14.8h, v7.8b, v2.8b
        SMLAL       v15.8h, v7.8b, v3.8b
        LDP         d6, d7, [x5, 224]
        SADALP      v18.4s, v12.8h
        SADALP      v26.4s, v13.8h
        SADALP      v19.4s, v14.8h
        SADALP      v27.4s, v15.8h
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 112]
        SMLAL       v8.8h, v6.8b, v2.8b
        SMLAL       v9.8h, v6.8b, v3.8b
        SMLAL       v10.8h, v7.8b, v2.8b
        SMLAL       v11.8h, v7.8b, v3.8b
        LDP         d6, d7, [x5, 240]
        SADALP      v20.4s, v8.8h
        SADALP      v28.4s, v9.8h
        SADALP      v21.4s, v10.8h
        SADALP      v29.4s, v11.8h
        SMULL       v12.8h, v4.8b, v0.8b
        SMULL       v13.8h, v4.8b, v1.8b
        SMULL       v14.8h, v5.8b, v0.8b
        SMULL       v15.8h, v5.8b, v1.8b
        SMLAL       v12.8h, v6.8b, v2.8b
        SMLAL       v13.8h, v6.8b, v3.8b
        SMLAL       v14.8h, v7.8b, v2.8b
        SMLAL       v15.8h, v7.8b, v3.8b
        SADALP      v22.4s, v12.8h
        SADALP      v30.4s, v13.8h
        SADALP      v23.4s, v14.8h
        SADALP      v31.4s, v15.8h
        ADD         x5, x5, 256
        SUBS        x0, x0, 16
        B.HS        1b

        TBZ         x0, 3, 3f

        # Remainder - 8 bytes of A
        .p2align    3
2:
        LDR         d0, [x3], 8
        LDR         d1, [x8], 8
        LDP         d4, d5, [x5]
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SADALP      v16.4s, v8.8h
        SADALP      v24.4s, v9.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 16]
        SADALP      v17.4s, v10.8h
        SADALP      v25.4s, v11.8h
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SADALP      v18.4s, v8.8h
        SADALP      v26.4s, v9.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 32]
        SADALP      v19.4s, v10.8h
        SADALP      v27.4s, v11.8h
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SADALP      v20.4s, v8.8h
        SADALP      v28.4s, v9.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 48]
        SADALP      v21.4s, v10.8h
        SADALP      v29.4s, v11.8h
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SADALP      v22.4s, v8.8h
        SADALP      v30.4s, v9.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 64]
        SADALP      v23.4s, v10.8h
        SADALP      v31.4s, v11.8h
        EXT         v0.8b, v0.8b, v0.8b, 4
        EXT         v1.8b, v1.8b, v1.8b, 4
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SADALP      v16.4s, v8.8h
        SADALP      v24.4s, v9.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 80]
        SADALP      v17.4s, v10.8h
        SADALP      v25.4s, v11.8h
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SADALP      v18.4s, v8.8h
        SADALP      v26.4s, v9.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 96]
        SADALP      v19.4s, v10.8h
        SADALP      v27.4s, v11.8h
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SADALP      v20.4s, v8.8h
        SADALP      v28.4s, v9.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDP         d4, d5, [x5, 112]
        SADALP      v21.4s, v10.8h
        SADALP      v29.4s, v11.8h
        SMULL       v8.8h, v4.8b, v0.8b
        SMULL       v9.8h, v4.8b, v1.8b
        SADALP      v22.4s, v8.8h
        SADALP      v30.4s, v9.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        SADALP      v23.4s, v10.8h
        SADALP      v31.4s, v11.8h
        ADD         x5, x5, 128

        .p2align    3
3:
        ADDP        v16.4s, v16.4s, v17.4s
        ADDP        v18.4s, v18.4s, v19.4s
        ADDP        v20.4s, v20.4s, v21.4s
        ADDP        v22.4s, v22.4s, v23.4s
        ADDP        v24.4s, v24.4s, v25.4s
        ADDP        v26.4s, v26.4s, v27.4s
        ADDP        v28.4s, v28.4s, v29.4s
        ADDP        v30.4s, v30.4s, v31.4s

        SCVTF       v16.4s, v16.4s
        SCVTF       v18.4s, v18.4s
        SCVTF       v20.4s, v20.4s
        SCVTF       v22.4s, v22.4s
        SCVTF       v24.4s, v24.4s
        SCVTF       v26.4s, v26.4s
        SCVTF       v28.4s, v28.4s
        SCVTF       v30.4s, v30.4s
        LDP         q4, q5, [x5], 32
        LDP         q6, q7, [x5], 32
        FMUL        v16.4s, v16.4s, v4.4s
        FMUL        v18.4s, v18.4s, v5.4s
        FMUL        v20.4s, v20.4s, v6.4s
        FMUL        v22.4s, v22.4s, v7.4s
        FMUL        v24.4s, v24.4s, v4.4s
        FMUL        v26.4s, v26.4s, v5.4s
        FMUL        v28.4s, v28.4s, v6.4s
        FMUL        v30.4s, v30.4s, v7.4s
        FCVTNS      v16.4s, v16.4s
        FCVTNS      v18.4s, v18.4s
        FCVTNS      v20.4s, v20.4s
        FCVTNS      v22.4s, v22.4s
        FCVTNS      v24.4s, v24.4s
        FCVTNS      v26.4s, v26.4s
        FCVTNS      v28.4s, v28.4s
        FCVTNS      v30.4s, v30.4s

        LD1R        {v6.8h}, [x11], 2
        SQXTN       v16.4h, v16.4s
        SQXTN2      v16.8h, v18.4s
        SQXTN       v20.4h, v20.4s
        SQXTN2      v20.8h, v22.4s
        SQXTN       v24.4h, v24.4s
        SQXTN2      v24.8h, v26.4s
        SQXTN       v28.4h, v28.4s
        SQXTN2      v28.8h, v30.4s
        SQADD       v16.8h, v16.8h, v6.8h
        SQADD       v20.8h, v20.8h, v6.8h
        SQADD       v24.8h, v24.8h, v6.8h
        SQADD       v28.8h, v28.8h, v6.8h
        LD1R        {v4.16b}, [x11], 1
        LD1R        {v5.16b}, [x11]
        SUB         x11, x11, 3
        SQXTN       v0.8b, v16.8h
        SQXTN2      v0.16b, v20.8h
        SQXTN       v1.8b, v24.8h
        SQXTN2      v1.16b, v28.8h
        SUBS        x1, x1, 16
        SMAX        v0.16b, v0.16b, v4.16b
        SMAX        v1.16b, v1.16b, v4.16b
        SMIN        v0.16b, v0.16b, v5.16b
        SMIN        v1.16b, v1.16b, v5.16b
        B.LO        4f

        # Store full 2 x 16
        ST1         {v1.16b}, [x12], x10
        SUB         x8, x8, x2              // a1 -= kc
        ST1         {v0.16b}, [x6], x10
        SUB         x3, x3, x2              // a0 -= kc
        B.HI        0b

        LDP         d14, d15, [sp, 48]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 64
        RET

        # Store odd width
        .p2align    3
4:
        TBZ         x1, 3, 5f
        STR         d1, [x12], 8
        DUP         d1, v1.d[1]
        STR         d0, [x6], 8
        DUP         d0, v0.d[1]
5:
        TBZ         x1, 2, 6f
        STR         s1, [x12], 4
        DUP         s1, v1.s[1]
        STR         s0, [x6], 4
        DUP         s0, v0.s[1]
6:
        TBZ         x1, 1, 7f
        STR         h1, [x12], 2
        DUP         h1, v1.h[1]
        STR         h0, [x6], 2
        DUP         h0, v0.h[1]
7:
        TBZ         x1, 0, 8f
        STR         b1, [x12]
        STR         b0, [x6]
8:
        LDP         d14, d15, [sp, 48]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 64
        RET

END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4s2__asm_aarch64_neon_mlal

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
