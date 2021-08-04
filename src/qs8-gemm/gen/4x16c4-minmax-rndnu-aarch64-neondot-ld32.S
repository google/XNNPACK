// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/4x16c4-aarch64-neondot-ld32.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <xnnpack/assembly.h>

# void xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld32(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     const int8_t* restrict a,  x3
#     size_t a_stride,           x4
#     const void* restrict w,    x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,          [sp] -> x12
#     const union xnn_qs8_conv_minmax_params params)  [sp + 8] -> x11

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x3 v0
# A1 x15 v1
# A2 x13 v2
# A3  x4 v3
# B   x5 v4  v5  v6  v7
# C0  x6 v16 v20 v24 v28
# C1  x8 v17 v21 v25 v29
# C2  x9 v18 v22 v26 v30
# C3  x7 v19 v23 v27 v31
# unused v8 v9 v10 v11 v12 v13 v14 v15

BEGIN_FUNCTION xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld32

        # Clamp A and C pointers
        CMP     x0, 2                   // if mr < 2
        ADD     x2, x2, 3               // kc = (kc + 3) & ~3
        ADD     x15, x3, x4             // a1 = a0 + a_stride
        ADD     x8, x6, x7              // c1 = c0 + cm_stride
        CSEL    x15, x3, x15, LO        //   a1 = a0
        CSEL    x8, x6,  x8, LO         //   c1 = c0
        BIC     x2, x2, 3

        ADD     x13, x15, x4            // a2 = a1 + a_stride
        ADD     x9,  x8, x7             // c2 = c1 + cm_stride
                                        // if mr <= 2
        CSEL    x13, x15, x13, LS       //   a2 = a1
        CSEL    x9,  x8,  x9, LS        //   c2 = c1

        CMP     x0, 4                   // if mr < 4
        ADD     x4, x13, x4             // a3 = a2 + a_stride
        ADD     x7,  x9, x7             // c3 = c2 + cm_stride
        CSEL    x4, x13, x4, LO         //   a3 = a2
        CSEL    x7,  x9, x7, LO         //   c3 = c2

        .p2align 3
0:
        # Load initial bias from w into accumulators
        LDP     q16, q20, [x5], 32
        MOV     v17.16b, v16.16b
        MOV     v18.16b, v16.16b
        LDP     q24, q28, [x5], 32
        MOV     v19.16b, v16.16b
        MOV     v21.16b, v20.16b
        LDR     x11, [sp, 8]            // reload params
        MOV     v22.16b, v20.16b
        MOV     v23.16b, v20.16b
        MOV     x0, x2                  // k = kc.  assumes kc > 0
        MOV     v25.16b, v24.16b
        MOV     v26.16b, v24.16b
        MOV     v27.16b, v24.16b
        MOV     v29.16b, v28.16b
        MOV     v30.16b, v28.16b
        MOV     v31.16b, v28.16b

        # Main loop - 4 bytes of A
        .p2align 3
1:
        LD1R    {v0.4s},  [x3], 4
        LDR     q4, [x5], 16
        LD1R    {v1.4s}, [x15], 4
        LD1R    {v2.4s}, [x13], 4
        LD1R    {v3.4s},  [x4], 4
        SDOT    v16.4s, v4.16b, v0.16b
        SDOT    v17.4s, v4.16b, v1.16b
        LDR     q5, [x5], 16
        SDOT    v18.4s, v4.16b, v2.16b
        SDOT    v19.4s, v4.16b, v3.16b
        LDR     q6, [x5], 16
        SDOT    v20.4s, v5.16b, v0.16b
        SDOT    v21.4s, v5.16b, v1.16b
        LDR     q7, [x5], 16
        SDOT    v22.4s, v5.16b, v2.16b
        SDOT    v23.4s, v5.16b, v3.16b
        SUBS    x0, x0, 4
        SDOT    v24.4s, v6.16b, v0.16b
        SDOT    v25.4s, v6.16b, v1.16b
        SDOT    v26.4s, v6.16b, v2.16b
        SDOT    v27.4s, v6.16b, v3.16b
        SDOT    v28.4s, v7.16b, v0.16b
        SDOT    v29.4s, v7.16b, v1.16b
        SDOT    v30.4s, v7.16b, v2.16b
        SDOT    v31.4s, v7.16b, v3.16b
        B.HI    1b

        # Apply params - preshift, scale, postshift, bias and clamp
        LD1R    {v4.4s}, [x11], 4
        SSHL    v16.4s, v16.4s, v4.4s   // shift to upper bits
        SSHL    v17.4s, v17.4s, v4.4s
        SSHL    v18.4s, v18.4s, v4.4s
        SSHL    v19.4s, v19.4s, v4.4s
        SSHL    v20.4s, v20.4s, v4.4s
        SSHL    v21.4s, v21.4s, v4.4s
        SSHL    v22.4s, v22.4s, v4.4s
        SSHL    v23.4s, v23.4s, v4.4s
        LD1R    {v5.4s}, [x11], 4
        SSHL    v24.4s, v24.4s, v4.4s
        SSHL    v25.4s, v25.4s, v4.4s
        SSHL    v26.4s, v26.4s, v4.4s
        SSHL    v27.4s, v27.4s, v4.4s
        SSHL    v28.4s, v28.4s, v4.4s
        SSHL    v29.4s, v29.4s, v4.4s
        SSHL    v30.4s, v30.4s, v4.4s
        SSHL    v31.4s, v31.4s, v4.4s
        LD1R    {v6.4s}, [x11], 4
        SQDMULH v16.4s, v16.4s, v5.4s   // scale without rounding
        SQDMULH v17.4s, v17.4s, v5.4s
        SQDMULH v18.4s, v18.4s, v5.4s
        SQDMULH v19.4s, v19.4s, v5.4s
        SQDMULH v20.4s, v20.4s, v5.4s
        SQDMULH v21.4s, v21.4s, v5.4s
        SQDMULH v22.4s, v22.4s, v5.4s
        SQDMULH v23.4s, v23.4s, v5.4s
        SQDMULH v24.4s, v24.4s, v5.4s
        SQDMULH v25.4s, v25.4s, v5.4s
        SQDMULH v26.4s, v26.4s, v5.4s
        SQDMULH v27.4s, v27.4s, v5.4s
        SQDMULH v28.4s, v28.4s, v5.4s
        SQDMULH v29.4s, v29.4s, v5.4s
        SQDMULH v30.4s, v30.4s, v5.4s
        SQDMULH v31.4s, v31.4s, v5.4s
        SRSHL   v16.4s, v16.4s, v6.4s   // signed rounding shift left
        SRSHL   v17.4s, v17.4s, v6.4s
        SRSHL   v18.4s, v18.4s, v6.4s
        SRSHL   v19.4s, v19.4s, v6.4s
        SRSHL   v20.4s, v20.4s, v6.4s
        SRSHL   v21.4s, v21.4s, v6.4s
        SRSHL   v22.4s, v22.4s, v6.4s
        SRSHL   v23.4s, v23.4s, v6.4s
        SRSHL   v24.4s, v24.4s, v6.4s
        SRSHL   v25.4s, v25.4s, v6.4s
        SRSHL   v26.4s, v26.4s, v6.4s
        SRSHL   v27.4s, v27.4s, v6.4s
        SRSHL   v28.4s, v28.4s, v6.4s
        SRSHL   v29.4s, v29.4s, v6.4s
        SRSHL   v30.4s, v30.4s, v6.4s
        SRSHL   v31.4s, v31.4s, v6.4s

        SQXTN   v16.4h, v16.4s
        SQXTN   v17.4h, v17.4s
        SQXTN   v18.4h, v18.4s
        SQXTN   v19.4h, v19.4s
        SQXTN   v24.4h, v24.4s
        SQXTN   v25.4h, v25.4s
        SQXTN   v26.4h, v26.4s
        SQXTN   v27.4h, v27.4s
        LD1R    {v6.8h}, [x11], 2       // add bias

        SQXTN2  v16.8h, v20.4s
        SQXTN2  v17.8h, v21.4s
        SQXTN2  v18.8h, v22.4s
        SQXTN2  v19.8h, v23.4s
        SQXTN2  v24.8h, v28.4s
        SQXTN2  v25.8h, v29.4s
        SQXTN2  v26.8h, v30.4s
        SQXTN2  v27.8h, v31.4s

        SQADD   v16.8h, v16.8h, v6.8h
        SQADD   v17.8h, v17.8h, v6.8h
        SQADD   v18.8h, v18.8h, v6.8h
        SQADD   v19.8h, v19.8h, v6.8h
        SQADD   v24.8h, v24.8h, v6.8h
        SQADD   v25.8h, v25.8h, v6.8h
        SQADD   v26.8h, v26.8h, v6.8h
        SQADD   v27.8h, v27.8h, v6.8h
        LD1R    {v4.16b}, [x11], 1      // clamp min value

        SQXTN   v0.8b, v16.8h
        SQXTN   v1.8b, v17.8h
        SQXTN   v2.8b, v18.8h
        SQXTN   v3.8b, v19.8h
        LD1R    {v5.16b}, [x11]         // clamp max value
        SQXTN2  v0.16b, v24.8h
        SQXTN2  v1.16b, v25.8h
        SQXTN2  v2.16b, v26.8h
        SQXTN2  v3.16b, v27.8h
        LDR     x12, [sp]               // cn_stride

        SMAX    v0.16b, v0.16b, v4.16b
        SMAX    v1.16b, v1.16b, v4.16b
        SMAX    v2.16b, v2.16b, v4.16b
        SMAX    v3.16b, v3.16b, v4.16b
        SUBS    x1, x1, 16
        SMIN    v0.16b, v0.16b, v5.16b
        SMIN    v1.16b, v1.16b, v5.16b
        SMIN    v2.16b, v2.16b, v5.16b
        SMIN    v3.16b, v3.16b, v5.16b
        B.LO    2f

        # Store full 4 x 16
        ST1     {v0.16b}, [x6], x12
        SUB     x3,  x3, x2             // a0 -= kc
        ST1     {v1.16b}, [x8], x12
        SUB     x15, x15, x2            // a1 -= kc
        ST1     {v2.16b}, [x9], x12
        SUB     x13, x13, x2            // a2 -= kc
        ST1     {v3.16b}, [x7], x12
        SUB     x4,  x4, x2             // a3 -= kc
        B.NE    0b
        RET

        # Store odd width
        .p2align 3
2:
        TBZ     x1, 3, 3f
        STR     d0, [x6], 8
        DUP     d0, v0.d[1]
        STR     d1, [x8], 8
        DUP     d1, v1.d[1]
        STR     d2, [x9], 8
        DUP     d2, v2.d[1]
        STR     d3, [x7], 8
        DUP     d3, v3.d[1]
3:
        TBZ     x1, 2, 4f
        STR     s0, [x6], 4
        DUP     s0, v0.s[1]
        STR     s1, [x8], 4
        DUP     s1, v1.s[1]
        STR     s2, [x9], 4
        DUP     s2, v2.s[1]
        STR     s3, [x7], 4
        DUP     s3, v3.s[1]
4:
        TBZ     x1, 1, 5f
        ST1     {v0.h}[0], [x6], 2
        DUP     h0, v0.h[1]
        ST1     {v1.h}[0], [x8], 2
        DUP     h1, v1.h[1]
        ST1     {v2.h}[0], [x9], 2
        DUP     h2, v2.h[1]
        ST1     {v3.h}[0], [x7], 2
        DUP     h3, v3.h[1]
5:
        TBZ     x1, 0, 6f
        ST1     {v0.b}[0], [x6]
        ST1     {v1.b}[0], [x8]
        ST1     {v2.b}[0], [x9]
        ST1     {v3.b}[0], [x7]
6:
        RET

END_FUNCTION xnn_qs8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld32

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
