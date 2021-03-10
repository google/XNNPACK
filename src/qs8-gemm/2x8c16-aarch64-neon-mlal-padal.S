// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_qs8_gemm_minmax_ukernel_2x8c16__aarch64_neon_mlal_padal(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     const int8_t* restrict a,  x3
#     size_t a_stride,           x4
#     const void* restrict w,    x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,          [sp] -> x10
#     const union xnn_qs8_gemm_params params)  [sp + 8] -> x9

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x3 v0
# A1  x4 v1
# B   x5 v4  v5  v6  v7
# C0  x7 v2 v10 v12 v14 v16 v18 v20 v22 v24 v26 v28 v30
# C1  x8 v3 v11 v13 v15 v17 v19 v21 v23 v25 v27 v29 v31
# unused v8 v9

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_2x8c16__aarch64_neon_mlal_padal

        # Clamp A and C pointers
        CMP     x0, 2             // if mr < 2
        ADD     x4, x3, x4        // a1 = a0 + a_stride
        ADD     x7, x6, x7        // c1 = c0 + cm_stride
        CSEL    x4, x3, x4, LO    //   a1 = a0
        CSEL    x7, x6, x7, LO    //   c1 = c0

        # Save d10-d15 on stack
        STP     d10, d11, [sp, -48]!
        STP     d12, d13, [sp, 16]
        STP     d14, d15, [sp, 32]

        ADD     x2, x2, 15        // kc = (kc + 15) & ~15
        BIC     x2, x2, 15


        .p2align 3
0:
        # Load initial bias from w into accumulators

        MOVI    v16.4s, 0
        LD1     {v16.s}[0], [x5], 4
        MOVI    v18.4s, 0
        LD1     {v18.s}[0], [x5], 4
        MOVI    v20.4s, 0
        LD1     {v20.s}[0], [x5], 4
        MOVI    v22.4s, 0
        LD1     {v22.s}[0], [x5], 4
        MOVI    v24.4s, 0
        LD1     {v24.s}[0], [x5], 4
        MOVI    v26.4s, 0
        LD1     {v26.s}[0], [x5], 4
        MOVI    v28.4s, 0
        LD1     {v28.s}[0], [x5], 4
        MOVI    v30.4s, 0
        LD1     {v30.s}[0], [x5], 4
        MOV     v17.4s, v16.4s
        LDP     x10, x9, [sp, 48]  // cn_stride, params
        MOV     v19.4s, v18.4s
        MOV     v21.4s, v20.4s
        MOV     v23.4s, v22.4s
        MOV     v25.4s, v24.4s
        MOV     v27.4s, v26.4s
        MOV     v29.4s, v28.4s
        MOV     v31.4s, v30.4s

        # There is 16 bytes for main loop
        MOV     x0, x2   // k = kc

        # Main loop - 16 bytes of A
        .p2align 3
1:
        LDR     q0, [x3], 16
        LDP     q4, q5, [x5], 32
        LDR     q1, [x4], 16
        LDP     q6, q7, [x5], 32
        SMULL    v2.8h, v4.8b, v0.8b
        SMULL    v3.8h, v4.8b, v1.8b
        SMULL   v10.8h, v5.8b, v0.8b
        SMULL   v11.8h, v5.8b, v1.8b
        SMULL   v12.8h, v6.8b, v0.8b
        SMULL   v13.8h, v6.8b, v1.8b
        SMULL   v14.8h, v7.8b, v0.8b
        SMULL   v15.8h, v7.8b, v1.8b

        SMLAL2   v2.8h, v4.16b, v0.16b
        SMLAL2   v3.8h, v4.16b, v1.16b
        SMLAL2  v10.8h, v5.16b, v0.16b
        SMLAL2  v11.8h, v5.16b, v1.16b
        SMLAL2  v12.8h, v6.16b, v0.16b
        SMLAL2  v13.8h, v6.16b, v1.16b
        SMLAL2  v14.8h, v7.16b, v0.16b
        SMLAL2  v15.8h, v7.16b, v1.16b

        LDP     q4, q5, [x5], 32
        SADALP  v16.4s,  v2.8h
        SADALP  v17.4s,  v3.8h
        SADALP  v18.4s, v10.8h
        SADALP  v19.4s, v11.8h
        LDP     q6, q7, [x5], 32
        SADALP  v20.4s, v12.8h
        SADALP  v21.4s, v13.8h
        SADALP  v22.4s, v14.8h
        SADALP  v23.4s, v15.8h

        SMULL    v2.8h, v4.8b, v0.8b
        SMULL    v3.8h, v4.8b, v1.8b
        SMULL   v10.8h, v5.8b, v0.8b
        SMULL   v11.8h, v5.8b, v1.8b
        SMULL   v12.8h, v6.8b, v0.8b
        SMULL   v13.8h, v6.8b, v1.8b
        SMULL   v14.8h, v7.8b, v0.8b
        SMULL   v15.8h, v7.8b, v1.8b

        SMLAL2   v2.8h, v4.16b, v0.16b
        SMLAL2   v3.8h, v4.16b, v1.16b
        SMLAL2  v10.8h, v5.16b, v0.16b
        SMLAL2  v11.8h, v5.16b, v1.16b
        SMLAL2  v12.8h, v6.16b, v0.16b
        SMLAL2  v13.8h, v6.16b, v1.16b
        SMLAL2  v14.8h, v7.16b, v0.16b
        SMLAL2  v15.8h, v7.16b, v1.16b
        SUBS    x0, x0, 16

        SADALP  v24.4s,  v2.8h
        SADALP  v25.4s,  v3.8h
        SADALP  v26.4s, v10.8h
        SADALP  v27.4s, v11.8h
        SADALP  v28.4s, v12.8h
        SADALP  v29.4s, v13.8h
        SADALP  v30.4s, v14.8h
        SADALP  v31.4s, v15.8h
        B.HI    1b

        # Add columns
        ADDP    v16.4s, v16.4s, v18.4s
        ADDP    v20.4s, v20.4s, v22.4s
        LD1R    {v4.4s}, [x9], 4
        ADDP    v24.4s, v24.4s, v26.4s
        ADDP    v28.4s, v28.4s, v30.4s
        LD1R    {v7.4s}, [x9], 4
        ADDP    v17.4s, v17.4s, v19.4s
        ADDP    v21.4s, v21.4s, v23.4s
        ADDP    v25.4s, v25.4s, v27.4s
        ADDP    v29.4s, v29.4s, v31.4s
        ADDP    v0.4s, v16.4s, v20.4s
        ADDP    v1.4s, v24.4s, v28.4s
        ADDP    v2.4s, v17.4s, v21.4s
        ADDP    v3.4s, v25.4s, v29.4s

        # Apply params - scale, shift, bias and clamp
        SQRDMULH        v0.4s, v0.4s, v4.4s
        SQRDMULH        v1.4s, v1.4s, v4.4s
        SQRDMULH        v2.4s, v2.4s, v4.4s
        SQRDMULH        v3.4s, v3.4s, v4.4s
        CMEQ    v4.4s, v7.4s, 0
        LD1R    {v5.8h}, [x9], 2
        BIC     v6.16b, v0.16b, v4.16b
        BIC     v16.16b, v1.16b, v4.16b
        BIC     v17.16b, v2.16b, v4.16b
        BIC     v4.16b, v3.16b, v4.16b
        SSRA    v0.4s, v6.4s, 31
        SSRA    v1.4s, v16.4s, 31
        SSRA    v2.4s, v17.4s, 31
        SSRA    v3.4s, v4.4s, 31
        SRSHL   v0.4s, v0.4s, v7.4s
        SRSHL   v1.4s, v1.4s, v7.4s
        SRSHL   v2.4s, v2.4s, v7.4s
        SRSHL   v3.4s, v3.4s, v7.4s
        SQXTN   v0.4h, v0.4s
        SQXTN   v2.4h, v2.4s
        SQXTN2  v0.8h, v1.4s
        SQXTN2  v2.8h, v3.4s
        SQADD   v0.8h, v0.8h, v5.8h
        SQADD   v1.8h, v2.8h, v5.8h
        SQXTN   v0.8b, v0.8h
        SQXTN2  v0.16b, v1.8h
        LD1R    {v1.16b}, [x9], 1
        LD1R    {v2.16b}, [x9]
        SUBS    x1, x1, 8
        SMAX    v0.16b, v0.16b, v1.16b
        SMIN    v0.16b, v0.16b, v2.16b
        B.LO    4f

        # Store full 2 x 8
        ST1     {v0.8b}, [x6], x10
        SUB     x3, x3, x2     // a0 -= kc
        ST1     {v0.d}[1], [x7], x10
        SUB     x4, x4, x2     // a1 -= kc
        B.HI    0b

        # Restore d10-d15 from stack
        LDP     d14, d15, [sp, 32]
        LDP     d12, d13, [sp, 16]
        LDP     d10, d11, [sp], 48
        RET

        # Store odd width
        .p2align 3
4:
        TBZ     x1, 2, 5f
        STR     s0, [x6], 4
        ST1     {v0.s}[2], [x7], 4
        EXT     v0.16b, v0.16b, v0.16b, 4

5:
        TBZ     x1, 1, 6f
        ST1     {v0.h}[0], [x6], 2
        ST1     {v0.h}[4], [x7], 2
        EXT     v0.16b, v0.16b, v0.16b, 2
6:
        TBZ     x1, 0, 7f
        ST1     {v0.b}[0], [x6]
        ST1     {v0.b}[8], [x7]
7:
        # Restore d10-d15 from stack
        LDP     d14, d15, [sp, 32]
        LDP     d12, d13, [sp, 16]
        LDP     d10, d11, [sp], 48
        RET

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_2x8c16__aarch64_neon_mlal_padal

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

