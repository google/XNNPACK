// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/4x16c4-aarch64-neondot-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <xnnpack/assembly.h>

# void xnn_qs8_igemm_minmax_gemmlowp_ukernel_4x16c4__aarch64_neondot_ld64(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     size_t ks,                 x3 / x9
#     const int8_t**restrict a,  x4
#     const int8_t* restrict w,  x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x8
#     const int8_t* zero,                [sp + 16] -> x12
#     const union xnn_qs8_conv_minmax_params params [sp + 24] -> (x11)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x13  v0
# A1  x14  v1
# A2  x15  v2
# A3  x11  v3
# B    x5  v4  v5  v6  v7
# C0   x6 v16 v20 v24 v28
# C1  x16 v17 v21 v25 v29
# C2  x17 v18 v22 v26 v30
# C3   x7 v19 v23 v27 v31
# unused v8 v9 v10 v11 v12 v13 v14 v15

BEGIN_FUNCTION xnn_qs8_igemm_minmax_gemmlowp_ukernel_4x16c4__aarch64_neondot_ld64

        # Clamp C pointers
        CMP     x0, 2                   // if mr < 2
        LDP     x10, x8, [sp]           // Load cn_stride, a_offset
        ADD     x16, x6, x7             // c1 = c0 + cm_stride
        CSEL    x16, x6,  x16, LO       //   c1 = c0
        ADD     x2, x2, 3               // kc = (kc + 3) & ~3

        ADD     x17, x16, x7            // c2 = c1 + cm_stride
        LDR     x12, [sp, 16]           // Load zero pointer
                                        // if mr <= 2
        CSEL    x17, x16, x17, LS       //   c2 = c1
        BIC     x2, x2, 3

        CMP     x0, 4                   // if mr < 4
        ADD     x7,  x17, x7            // c3 = c2 + cm_stride
        CSEL    x7,  x17, x7, LO        //   c3 = c2

        .p2align 3
0:
        # Load initial bias from w into accumulators
        LDP     q16, q20, [x5], 32
        MOV     v17.16b, v16.16b
        MOV     v18.16b, v16.16b
        LDP     q24, q28, [x5], 32
        MOV     v19.16b, v16.16b
        MOV     v21.16b, v20.16b
        MOV     v22.16b, v20.16b
        MOV     v23.16b, v20.16b
        MOV     v25.16b, v24.16b
        MOV     v26.16b, v24.16b
        MOV     v27.16b, v24.16b
        MOV     v29.16b, v28.16b
        MOV     v30.16b, v28.16b
        MOV     v31.16b, v28.16b
        MOV     x9, x3                  // p = ks

        .p2align 3
1:
        # Load next 4 A pointers
        LDP     x13, x14, [x4], 16
        LDP     x15, x11, [x4], 16

        CMP     x13, x12                // if a0 == zero
        ADD     x13, x13, x8            // a0 += a_offset
        CSEL    x13, x12, x13, EQ       //   a0 = zero, else += a0 + a_offset
        CMP     x14, x12                // if a1 == zero
        ADD     x14, x14, x8            // a1 += a_offset
        CSEL    x14, x12, x14, EQ       //   a1 = zero, else += a1 + a_offset
        CMP     x15, x12                // if a2 == zero
        ADD     x15, x15, x8            // a2 += a_offset
        CSEL    x15, x12, x15, EQ       //   a2 = zero, else += a2 + a_offset
        CMP     x11, x12                // if a3 == zero
        ADD     x11, x11, x8            // a3 += a_offset
        CSEL    x11, x12, x11, EQ       //   a3 = zero, else += a3 + a_offset

        # Is there at least 8 bytes for main loop?
        SUBS    x0, x2, 8               // k = kc - 8
        B.LO    4f

        # Main loop - 8 bytes of A
        .p2align 3
2:
        LDR     d0, [x13], 8
        LDR     q4,  [x5], 16
        LDR     d1, [x14], 8
        LDR     d2, [x15], 8
        LDR     d3, [x11], 8
        LDR     q5,  [x5], 16
        SDOT    v16.4s, v4.16b,  v0.4b[0]
        SDOT    v17.4s, v4.16b,  v1.4b[0]
        LDP     q6, q7, [x5], 32
        SDOT    v18.4s, v4.16b,  v2.4b[0]
        SDOT    v19.4s, v4.16b,  v3.4b[0]
        SDOT    v20.4s, v5.16b,  v0.4b[0]
        SDOT    v21.4s, v5.16b,  v1.4b[0]
        SDOT    v22.4s, v5.16b,  v2.4b[0]
        SDOT    v23.4s, v5.16b,  v3.4b[0]
        SDOT    v24.4s, v6.16b, v0.4b[0]
        SDOT    v25.4s, v6.16b, v1.4b[0]
        LDP     q4, q5, [x5], 32
        SDOT    v26.4s, v6.16b, v2.4b[0]
        SDOT    v27.4s, v6.16b, v3.4b[0]
        SDOT    v28.4s, v7.16b, v0.4b[0]
        SDOT    v29.4s, v7.16b, v1.4b[0]
        SDOT    v30.4s, v7.16b, v2.4b[0]
        SDOT    v31.4s, v7.16b, v3.4b[0]
        SDOT    v16.4s, v4.16b,  v0.4b[1]
        SDOT    v17.4s, v4.16b,  v1.4b[1]
        LDP     q6, q7, [x5], 32
        SDOT    v18.4s, v4.16b,  v2.4b[1]
        SDOT    v19.4s, v4.16b,  v3.4b[1]
        SDOT    v20.4s, v5.16b,  v0.4b[1]
        SDOT    v21.4s, v5.16b,  v1.4b[1]
        SDOT    v22.4s, v5.16b,  v2.4b[1]
        SDOT    v23.4s, v5.16b,  v3.4b[1]
        SDOT    v24.4s, v6.16b,  v0.4b[1]
        SDOT    v25.4s, v6.16b,  v1.4b[1]
        SDOT    v26.4s, v6.16b,  v2.4b[1]
        SDOT    v27.4s, v6.16b,  v3.4b[1]
        SDOT    v28.4s, v7.16b,  v0.4b[1]
        SDOT    v29.4s, v7.16b,  v1.4b[1]
        SDOT    v30.4s, v7.16b,  v2.4b[1]
        SUBS    x0, x0, 8
        SDOT    v31.4s, v7.16b,  v3.4b[1]
        B.HS    2b

        # Is there a remainder?- 4 bytes of A
        TBNZ    x0, 2, 4f

        LDR     x11, [sp, 24]           // reload params pointer

        # ks loop
        SUBS    x9, x9, 32              // ks -= MR * sizeof(int8_t*)
        B.HI    1b

3:
        # Apply params - scale, shift, bias and clamp
        LD2R    {v4.4s, v5.4s}, [x11], 8
        CMEQ    v6.4s, v5.4s, 0

        BIC     v0.16b, v16.16b, v6.16b
        BIC     v1.16b, v17.16b, v6.16b
        BIC     v2.16b, v18.16b, v6.16b
        BIC     v3.16b, v19.16b, v6.16b

        SQRDMULH v16.4s, v16.4s, v4.4s
        SQRDMULH v17.4s, v17.4s, v4.4s
        SQRDMULH v18.4s, v18.4s, v4.4s
        SQRDMULH v19.4s, v19.4s, v4.4s

        SSRA    v16.4s, v0.4s, 31       // signed shift right accumulate
        SSRA    v17.4s, v1.4s, 31
        SSRA    v18.4s, v2.4s, 31
        SSRA    v19.4s, v3.4s, 31

        BIC     v0.16b, v20.16b, v6.16b
        BIC     v1.16b, v21.16b, v6.16b
        BIC     v2.16b, v22.16b, v6.16b
        BIC     v3.16b, v23.16b, v6.16b

        SQRDMULH v20.4s, v20.4s, v4.4s
        SQRDMULH v21.4s, v21.4s, v4.4s
        SQRDMULH v22.4s, v22.4s, v4.4s
        SQRDMULH v23.4s, v23.4s, v4.4s

        SSRA    v20.4s, v0.4s, 31
        SSRA    v21.4s, v1.4s, 31
        SSRA    v22.4s, v2.4s, 31
        SSRA    v23.4s, v3.4s, 31

        BIC     v0.16b, v24.16b, v6.16b
        BIC     v1.16b, v25.16b, v6.16b
        BIC     v2.16b, v26.16b, v6.16b
        BIC     v3.16b, v27.16b, v6.16b

        SQRDMULH v24.4s, v24.4s, v4.4s
        SQRDMULH v25.4s, v25.4s, v4.4s
        SQRDMULH v26.4s, v26.4s, v4.4s
        SQRDMULH v27.4s, v27.4s, v4.4s

        SSRA    v24.4s, v0.4s, 31
        SSRA    v25.4s, v1.4s, 31
        SSRA    v26.4s, v2.4s, 31
        SSRA    v27.4s, v3.4s, 31

        BIC     v0.16b, v28.16b, v6.16b
        BIC     v1.16b, v29.16b, v6.16b
        BIC     v2.16b, v30.16b, v6.16b
        BIC     v3.16b, v31.16b, v6.16b

        SQRDMULH v28.4s, v28.4s, v4.4s
        SQRDMULH v29.4s, v29.4s, v4.4s
        SQRDMULH v30.4s, v30.4s, v4.4s
        SQRDMULH v31.4s, v31.4s, v4.4s

        SSRA    v28.4s, v0.4s, 31
        SSRA    v29.4s, v1.4s, 31
        SSRA    v30.4s, v2.4s, 31
        SSRA    v31.4s, v3.4s, 31

        SRSHL   v16.4s, v16.4s, v5.4s   // signed rounding shift left
        SRSHL   v17.4s, v17.4s, v5.4s
        SRSHL   v18.4s, v18.4s, v5.4s
        SRSHL   v19.4s, v19.4s, v5.4s
        SRSHL   v20.4s, v20.4s, v5.4s
        SRSHL   v21.4s, v21.4s, v5.4s
        SRSHL   v22.4s, v22.4s, v5.4s
        SRSHL   v23.4s, v23.4s, v5.4s
        SRSHL   v24.4s, v24.4s, v5.4s
        SRSHL   v25.4s, v25.4s, v5.4s
        SRSHL   v26.4s, v26.4s, v5.4s
        SRSHL   v27.4s, v27.4s, v5.4s
        SRSHL   v28.4s, v28.4s, v5.4s
        SRSHL   v29.4s, v29.4s, v5.4s
        SRSHL   v30.4s, v30.4s, v5.4s
        SRSHL   v31.4s, v31.4s, v5.4s

        SQXTN   v16.4h, v16.4s
        SQXTN   v17.4h, v17.4s
        SQXTN   v18.4h, v18.4s
        SQXTN   v19.4h, v19.4s
        SQXTN   v24.4h, v24.4s
        SQXTN   v25.4h, v25.4s
        SQXTN   v26.4h, v26.4s
        SQXTN   v27.4h, v27.4s
        LD1R    {v6.8h}, [x11], 2        // add bias

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
        LD1R    {v4.16b}, [x11], 1       // clamp min value

        SQXTN   v0.8b, v16.8h
        SQXTN   v1.8b, v17.8h
        SQXTN   v2.8b, v18.8h
        SQXTN   v3.8b, v19.8h
        LD1R    {v5.16b}, [x11]          // clamp max value
        SQXTN2  v0.16b, v24.8h
        SQXTN2  v1.16b, v25.8h
        SQXTN2  v2.16b, v26.8h
        SQXTN2  v3.16b, v27.8h

        SMAX    v0.16b, v0.16b, v4.16b
        SMAX    v1.16b, v1.16b, v4.16b
        SMAX    v2.16b, v2.16b, v4.16b
        SMAX    v3.16b, v3.16b, v4.16b
        SUBS    x1, x1, 16
        SMIN    v0.16b, v0.16b, v5.16b
        SMIN    v1.16b, v1.16b, v5.16b
        SMIN    v2.16b, v2.16b, v5.16b
        SMIN    v3.16b, v3.16b, v5.16b
        B.LO    5f

        # Store full 4 x 16
        ST1     {v3.16b},  [x7], x10
        ST1     {v2.16b}, [x17], x10
        ST1     {v1.16b}, [x16], x10
        ST1     {v0.16b},  [x6], x10

        SUB     x4, x4, x3              // a -= ks

        # nc loop
        B.HI    0b
        RET

        # Remainder- 4 bytes of A
        .p2align 3
4:
        LDR     s0, [x13], 4
        LDR     q4, [x5], 16
        LDR     s1, [x14], 4
        LDR     s2, [x15], 4
        LDR     s3, [x11], 4
        LDR     q5, [x5], 16
        SDOT    v16.4s, v4.16b,  v0.4b[0]
        SDOT    v17.4s, v4.16b,  v1.4b[0]
        LDP     q6, q7, [x5], 32
        SDOT    v18.4s, v4.16b,  v2.4b[0]
        SDOT    v19.4s, v4.16b,  v3.4b[0]
        SDOT    v20.4s, v5.16b,  v0.4b[0]
        SDOT    v21.4s, v5.16b,  v1.4b[0]
        SDOT    v22.4s, v5.16b,  v2.4b[0]
        SDOT    v23.4s, v5.16b,  v3.4b[0]
        LDR     x11, [sp, 24]            // reload params pointer
        SDOT    v24.4s, v6.16b, v0.4b[0]
        SDOT    v25.4s, v6.16b, v1.4b[0]
        SDOT    v26.4s, v6.16b, v2.4b[0]
        SDOT    v27.4s, v6.16b, v3.4b[0]
        SDOT    v28.4s, v7.16b, v0.4b[0]
        SDOT    v29.4s, v7.16b, v1.4b[0]
        SDOT    v30.4s, v7.16b, v2.4b[0]
        SDOT    v31.4s, v7.16b, v3.4b[0]

        # ks loop
        SUBS    x9, x9, 32              // ks -= MR * sizeof(int8_t*)
        B.HI    1b
        B       3b

        # Store odd width
        .p2align 3
5:
        TBZ     x1, 3, 6f
        STR     d3, [x7], 8
        DUP     d3, v3.d[1]
        STR     d2, [x17], 8
        DUP     d2, v2.d[1]
        STR     d1, [x16], 8
        DUP     d1, v1.d[1]
        STR     d0, [x6], 8
        DUP     d0, v0.d[1]
6:
        TBZ     x1, 2, 7f
        STR     s3, [x7], 4
        DUP     s3, v3.s[1]
        STR     s2, [x17], 4
        DUP     s2, v2.s[1]
        STR     s1, [x16], 4
        DUP     s1, v1.s[1]
        STR     s0, [x6], 4
        DUP     s0, v0.s[1]
7:
        TBZ     x1, 1, 8f
        ST1     {v3.h}[0], [x7], 2
        DUP     h3, v3.h[1]
        ST1     {v2.h}[0], [x17], 2
        DUP     h2, v2.h[1]
        ST1     {v1.h}[0], [x16], 2
        DUP     h1, v1.h[1]
        ST1     {v0.h}[0], [x6], 2
        DUP     h0, v0.h[1]
8:
        TBZ     x1, 0, 9f
        ST1     {v3.b}[0], [x7]
        ST1     {v2.b}[0], [x17]
        ST1     {v1.b}[0], [x16]
        ST1     {v0.b}[0], [x6]
9:
        RET

END_FUNCTION xnn_qs8_igemm_minmax_gemmlowp_ukernel_4x16c4__aarch64_neondot_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
