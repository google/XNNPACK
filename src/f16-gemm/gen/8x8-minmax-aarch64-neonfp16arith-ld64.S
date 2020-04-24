// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/8x8-aarch64-neonfp16arith-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const uint8_t*restrict a, x3
#     size_t a_stride,          x4
#     const void*restrict w,    x5
#     uint8_t*restrict c,       x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x14
#     const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointers
#  x3 a0
#  x9 a1
# x10 a2
# x11 a3
# x12 a4
# x19 a5
# x20 a6
#  x4 a7

# C pointers
#  x6 c0
# x16 c1
# x17 c2
# x18 c3
# x13 c4
# x21 c5
# x22 c6
#  x7 c7

# Vector register usage
# A0   v0
# A1   v1
# A2   v2
# A3   v3
# A4   v4
# A5   v5
# A6   v6
# A7   v7
# B   v16 v17 v18 v19
# C   v24
# C   v25
# C   v26
# C   v27
# C   v28
# C   v29
# C   v30
# C   v31

# Clamp v20 v21 v22
# unused A   v8 v9 v10 v11
# unused B   v12 v13 v14 v15

BEGIN_FUNCTION xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64

        # Load cn_stride, params pointer
        LDP x14, x8, [sp]

        # Save x19,x20,x21,x22 on stack
        STP x19, x20, [sp, -32]!
        STP x21, x22, [sp, 16]

        # Clamp A and C pointers
        CMP x0, 2                // if mr < 2
        ADD x9, x3, x4           // a1 = a0 + a_stride
        ADD x16, x6, x7          // c1 = c0 + cm_stride
        CSEL x9, x3, x9, LO      //   a1 = a0
        CSEL x16, x6, x16, LO    //   c1 = c0

        ADD x10, x9, x4          // a2 = a1 + a_stride
        ADD x17, x16, x7         // c2 = c1 + cm_stride
                                 // if mr <= 2
        CSEL x10, x9, x10, LS    //   a2 = a1
        CSEL x17, x16, x17, LS   //   c2 = c1

        CMP x0, 4                // if mr < 4
        ADD x11, x10, x4         // a3 = a2 + a_stride
        ADD x18, x17, x7         // c3 = c2 + cm_stride
        CSEL x11, x10, x11, LO   //   a3 = a2
        CSEL x18, x17, x18, LO   //   c3 = c2

        ADD x12, x11, x4         // a4 = a3 + a_stride
        ADD x13, x18, x7         // c4 = c3 + cm_stride
                                 // if mr <= 4
        CSEL x12, x11, x12, LS   //   a4 = a3
        CSEL x13, x18, x13, LS   //   c4 = c3

        CMP x0, 6                // if mr < 6
        ADD x19, x12, x4         // a5 = a4 + a_stride
        ADD x21, x13, x7         // c5 = c4 + cm_stride
        CSEL x19, x12, x19, LO   //   a5 = a4
        CSEL x21, x13, x21, LO   //   c5 = c4

        ADD x20, x19, x4         // a6 = a5 + a_stride
        ADD x22, x21, x7         // c6 = c5 + cm_stride
                                 // if mr <= 6
        CSEL x20, x19, x20, LS   //   a6 = a5
        CSEL x22, x21, x22, LS   //   c6 = c5

        CMP x0, 8                // if mr < 8
        ADD x4, x20, x4          // a7 = a5 + a_stride
        ADD x7, x22, x7          // c7 = c5 + cm_stride
        CSEL x4, x20, x4, LO     //   a7 = a5
        CSEL x7, x22, x7, LO     //   c7 = c5

        # Load params scale value
        LD3R {v20.8h, v21.8h, v22.8h}, [x8]

0:
       # Load initial bias from w into accumulators
       LDR q24, [x5], 16
       MOV v25.16b, v24.16b
       MOV v26.16b, v24.16b
       MOV v27.16b, v24.16b
       MOV v28.16b, v24.16b
       MOV v29.16b, v24.16b
       MOV v30.16b, v24.16b
       MOV v31.16b, v24.16b

         # Is there at least 4 halffloats (8 bytes)?
        SUBS x0, x2, 8  // k = kc - 8
        B.LO 5f

        # Main loop - 4 halffloats of A (8 bytes)
        # 32 FMA + 8 ld64 A + 4 LDR B
1:
        LDR   d0,  [x3], 8
        LDR  q16,  [x5], 16
        LDR  q17,  [x5], 16
        LDR   d1,  [x9], 8
        LDR   d2, [x10], 8
        LDR   d3, [x11], 8
        LDR   d4, [x12], 8
        LDR   d5, [x19], 8
        LDR   d6, [x20], 8
        LDR   d7,  [x4], 8
        SUBS x0, x0, 8
        FMLA v24.8h, v16.8h,  v0.h[0]
        FMLA v25.8h, v16.8h,  v1.h[0]
        FMLA v26.8h, v16.8h,  v2.h[0]
        FMLA v27.8h, v16.8h,  v3.h[0]
        FMLA v28.8h, v16.8h,  v4.h[0]
        FMLA v29.8h, v16.8h,  v5.h[0]
        FMLA v30.8h, v16.8h,  v6.h[0]
        FMLA v31.8h, v16.8h,  v7.h[0]
        LDR  q18,  [x5], 16
        LDR  q19,  [x5], 16

        FMLA v24.8h, v17.8h,  v0.h[1]
        FMLA v25.8h, v17.8h,  v1.h[1]
        FMLA v26.8h, v17.8h,  v2.h[1]
        FMLA v27.8h, v17.8h,  v3.h[1]
        FMLA v28.8h, v17.8h,  v4.h[1]
        FMLA v29.8h, v17.8h,  v5.h[1]
        FMLA v30.8h, v17.8h,  v6.h[1]
        FMLA v31.8h, v17.8h,  v7.h[1]

        FMLA v24.8h, v18.8h,  v0.h[2]
        FMLA v25.8h, v18.8h,  v1.h[2]
        FMLA v26.8h, v18.8h,  v2.h[2]
        FMLA v27.8h, v18.8h,  v3.h[2]
        FMLA v28.8h, v18.8h,  v4.h[2]
        FMLA v29.8h, v18.8h,  v5.h[2]
        FMLA v30.8h, v18.8h,  v6.h[2]
        FMLA v31.8h, v18.8h,  v7.h[2]

        FMLA v24.8h, v19.8h,  v0.h[3]
        FMLA v25.8h, v19.8h,  v1.h[3]
        FMLA v26.8h, v19.8h,  v2.h[3]
        FMLA v27.8h, v19.8h,  v3.h[3]
        FMLA v28.8h, v19.8h,  v4.h[3]
        FMLA v29.8h, v19.8h,  v5.h[3]
        FMLA v30.8h, v19.8h,  v6.h[3]
        FMLA v31.8h, v19.8h,  v7.h[3]
        B.HS 1b

        # Is there a remainder?- 2 halffloats of A (4 bytes)
        TBNZ x0, 2, 6f
        # Is there a remainder?- 1 halffloats of A (2 bytes)
        TBNZ x0, 1, 7f
4:
        # Scale and Clamp
        FMUL v24.8h, v24.8h, v20.8h
        FMUL v25.8h, v25.8h, v20.8h
        FMUL v26.8h, v26.8h, v20.8h
        FMUL v27.8h, v27.8h, v20.8h
        FMUL v28.8h, v28.8h, v20.8h
        FMUL v29.8h, v29.8h, v20.8h
        FMUL v30.8h, v30.8h, v20.8h
        FMUL v31.8h, v31.8h, v20.8h

        SUBS x1, x1, 8
        FMAX v24.8h, v24.8h, v21.8h
        FMAX v25.8h, v25.8h, v21.8h
        FMAX v26.8h, v26.8h, v21.8h
        FMAX v27.8h, v27.8h, v21.8h
        FMAX v28.8h, v28.8h, v21.8h
        FMAX v29.8h, v29.8h, v21.8h
        FMAX v30.8h, v30.8h, v21.8h
        FMAX v31.8h, v31.8h, v21.8h

        FMIN v24.8h, v24.8h, v22.8h
        FMIN v25.8h, v25.8h, v22.8h
        FMIN v26.8h, v26.8h, v22.8h
        FMIN v27.8h, v27.8h, v22.8h
        FMIN v28.8h, v28.8h, v22.8h
        FMIN v29.8h, v29.8h, v22.8h
        FMIN v30.8h, v30.8h, v22.8h
        FMIN v31.8h, v31.8h, v22.8h

        # Store full 8 x 8
        B.LO 8f

        ST1 {v24.16b},  [x6], x14
        SUB  x3,  x3, x2 // a0 -= kc
        ST1 {v25.16b}, [x16], x14
        SUB  x9,  x9, x2 // a1 -= kc
        ST1 {v26.16b}, [x17], x14
        SUB x10, x10, x2 // a2 -= kc
        ST1 {v27.16b}, [x18], x14
        SUB x11, x11, x2 // a3 -= kc
        ST1 {v28.16b}, [x13], x14
        SUB x12, x12, x2 // a4 -= kc
        ST1 {v29.16b}, [x21], x14
        SUB x19, x19, x2 // a6 -= kc
        ST1 {v30.16b}, [x22], x14
        SUB x20, x20, x2 // a6 -= kc
        ST1 {v31.16b},  [x7], x14
        SUB  x4,  x4, x2 // a7 -= kc

        B.HI 0b

        # Restore x19,x20,x21,x22 from stack
        LDP x21, x22, [sp, 16]
        LDP x19, x20, [sp], 32
        RET

5:
        TBZ x0, 2, 7f
6:
        # Remainder- 2 halffloats of A (4 bytes)
        LDR   s0,  [x3], 4
        LDR  q16,  [x5], 16
        LDR  q17,  [x5], 16
        LDR   s1,  [x9], 4
        LDR   s2, [x10], 4
        LDR   s3, [x11], 4
        LDR   s4, [x12], 4
        LDR   s5, [x19], 4
        LDR   s6, [x20], 4
        LDR   s7,  [x4], 4

        FMLA v24.8h, v16.8h,  v0.h[0]
        FMLA v25.8h, v16.8h,  v1.h[0]
        FMLA v26.8h, v16.8h,  v2.h[0]
        FMLA v27.8h, v16.8h,  v3.h[0]
        FMLA v28.8h, v16.8h,  v4.h[0]
        FMLA v29.8h, v16.8h,  v5.h[0]
        FMLA v30.8h, v16.8h,  v6.h[0]
        FMLA v31.8h, v16.8h,  v7.h[0]

        FMLA v24.8h, v17.8h,  v0.h[1]
        FMLA v25.8h, v17.8h,  v1.h[1]
        FMLA v26.8h, v17.8h,  v2.h[1]
        FMLA v27.8h, v17.8h,  v3.h[1]
        FMLA v28.8h, v17.8h,  v4.h[1]
        FMLA v29.8h, v17.8h,  v5.h[1]
        FMLA v30.8h, v17.8h,  v6.h[1]
        FMLA v31.8h, v17.8h,  v7.h[1]

        TBZ x0, 1, 4b

7:
        # Remainder- 1 halffloat of A (2 bytes)
        LDR   h0,  [x3], 2
        LDR  q16,  [x5], 16
        LDR   h1,  [x9], 2
        LDR   h2, [x10], 2
        LDR   h3, [x11], 2
        LDR   h4, [x12], 2
        LDR   h5, [x19], 2
        LDR   h6, [x20], 2
        LDR   h7,  [x4], 2

        FMLA v24.8h, v16.8h,  v0.h[0]
        FMLA v25.8h, v16.8h,  v1.h[0]
        FMLA v26.8h, v16.8h,  v2.h[0]
        FMLA v27.8h, v16.8h,  v3.h[0]
        FMLA v28.8h, v16.8h,  v4.h[0]
        FMLA v29.8h, v16.8h,  v5.h[0]
        FMLA v30.8h, v16.8h,  v6.h[0]
        FMLA v31.8h, v16.8h,  v7.h[0]
        B 4b

        # Store odd width
8:
        TBZ x1, 2, 9f
        STR d24,  [x6], 8
        DUP d24, v24.d[1]
        STR d25, [x16], 8
        DUP d25, v25.d[1]
        STR d26, [x17], 8
        DUP d26, v26.d[1]
        STR d27, [x18], 8
        DUP d27, v27.d[1]
        STR d28, [x13], 8
        DUP d28, v28.d[1]
        STR d29, [x21], 8
        DUP d29, v29.d[1]
        STR d30, [x22], 8
        DUP d30, v30.d[1]
        STR d31,  [x7], 8
        DUP d31, v31.d[1]
9:
        TBZ x1, 1, 10f
        STR s24,  [x6], 4
        DUP s24, v24.s[1]
        STR s25, [x16], 4
        DUP s25, v25.s[1]
        STR s26, [x17], 4
        DUP s26, v26.s[1]
        STR s27, [x18], 4
        DUP s27, v27.s[1]
        STR s28, [x13], 4
        DUP s28, v28.s[1]
        STR s29, [x21], 4
        DUP s29, v29.s[1]
        STR s30, [x22], 4
        DUP s30, v30.s[1]
        STR s31,  [x7], 4
        DUP s31, v31.s[1]

10:
        TBZ x1, 0, 11f
        STR h24,  [x6]
        STR h25, [x16]
        STR h26, [x17]
        STR h27, [x18]
        STR h28, [x13]
        STR h29, [x21]
        STR h30, [x22]
        STR h31,  [x7]
11:
        # Restore x19,x20,x21,x22 from stack
        LDP x21, x22, [sp, 16]
        LDP x19, x20, [sp], 32
        RET

END_FUNCTION xnn_f16_gemm_minmax_ukernel_8x8__aarch64_neonfp16arith_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
