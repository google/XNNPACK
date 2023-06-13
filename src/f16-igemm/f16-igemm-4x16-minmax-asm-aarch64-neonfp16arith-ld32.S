// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const void** restrict a,            x4
#     const void* restrict w,             x5
#     void* restrict c,                   x6
#     size_t cm_stride,                  x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const void* zero,                  [sp + 16] -> x12
#     const xnn_f16_minmax_params params [sp + 24] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x8 v0
// A1 x13 v1
// A2 x14 v2
// A3 x15 v3
// B   x5 v20 v21 v22 v23
// C0  x6 v24 v25
// C1 x16 v26 v27
// C2 x17 v28 v29
// C3  x7 v30 v31
// clamp  v4, v5

BEGIN_FUNCTION xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32

        # Load cn_stride, a_offset
        LDP         x10, x11, [sp]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Load params values
        LD2R        {v4.8h, v5.8h}, [x8]

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x16, x6, x16, LO        //   c1 = c0
        ADD         x17, x16, x7            // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x17, x16, x17, LS       //   c2 = c1
        CMP         x0, 4                   // if mr < 4
        ADD         x7, x17, x7             // c3 = c2 + cm_stride
        CSEL        x7, x17, x7, LO         //   c3 = c2

0:
        # Load initial bias from w into accumulators
        LDR         q24, [x5], 16
        LDR         q25, [x5], 16
        MOV         v26.16b, v24.16b
        MOV         v28.16b, v24.16b
        MOV         v30.16b, v24.16b
        MOV         v27.16b, v25.16b
        MOV         v29.16b, v25.16b
        MOV         v31.16b, v25.16b

        MOV         x9, x3                  // p = ks

1:
        # Load next 4 A pointers
        LDP         x8, x13, [x4], 16
        LDP         x14, x15, [x4], 16

        CMP         x8, x12                 // if a0 == zero
        ADD         x8, x8, x11             // a0 += a_offset
        CSEL        x8, x12, x8, EQ         //   a0 = zero, else += a0 + a_offset
        CMP         x13, x12                // if a1 == zero
        ADD         x13, x13, x11           // a1 += a_offset
        CSEL        x13, x12, x13, EQ       //   a1 = zero, else += a1 + a_offset
        CMP         x14, x12                // if a2 == zero
        ADD         x14, x14, x11           // a2 += a_offset
        CSEL        x14, x12, x14, EQ       //   a2 = zero, else += a2 + a_offset
        CMP         x15, x12                // if a3 == zero
        ADD         x15, x15, x11           // a3 += a_offset
        CSEL        x15, x12, x15, EQ       //   a3 = zero, else += a3 + a_offset

        # Is there at least 2 halffloats (4 bytes)?
        SUBS        x0, x2, 4               // k = kc - 4
        B.LO        4f

       .p2align 3
        # Main loop - 2 halffloats of A (4 bytes)
2:
        LDR         s0,  [x8], 4
        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         s1, [x13], 4
        LDR         s2, [x14], 4
        LDR         s3, [x15], 4
        LDR         q22, [x5], 16
        LDR         q23, [x5], 16
        SUBS        x0, x0, 4
        FMLA        v24.8h, v20.8h, v0.h[0]
        FMLA        v25.8h, v21.8h, v0.h[0]
        FMLA        v26.8h, v20.8h, v1.h[0]
        FMLA        v27.8h, v21.8h, v1.h[0]
        FMLA        v28.8h, v20.8h, v2.h[0]
        FMLA        v29.8h, v21.8h, v2.h[0]
        FMLA        v30.8h, v20.8h, v3.h[0]
        FMLA        v31.8h, v21.8h, v3.h[0]

        FMLA        v24.8h, v22.8h, v0.h[1]
        FMLA        v25.8h, v23.8h, v0.h[1]
        FMLA        v26.8h, v22.8h, v1.h[1]
        FMLA        v27.8h, v23.8h, v1.h[1]
        FMLA        v28.8h, v22.8h, v2.h[1]
        FMLA        v29.8h, v23.8h, v2.h[1]
        FMLA        v30.8h, v22.8h, v3.h[1]
        FMLA        v31.8h, v23.8h, v3.h[1]
        B.HS        2b

        # Is there a remainder?- 1 halffloat of A (2 bytes)
        TBNZ        x0, 1, 4f

3:
        # ks loop
        SUBS        x9, x9, 32              // ks -= MR * sizeof(void*)
        B.HI        1b

        # Clamp
        FMAX        v24.8h, v24.8h, v4.8h
        FMAX        v25.8h, v25.8h, v4.8h
        FMAX        v26.8h, v26.8h, v4.8h
        FMAX        v27.8h, v27.8h, v4.8h
        FMAX        v28.8h, v28.8h, v4.8h
        FMAX        v29.8h, v29.8h, v4.8h
        FMAX        v30.8h, v30.8h, v4.8h
        FMAX        v31.8h, v31.8h, v4.8h
        FMIN        v24.8h, v24.8h, v5.8h
        FMIN        v25.8h, v25.8h, v5.8h
        FMIN        v26.8h, v26.8h, v5.8h
        FMIN        v27.8h, v27.8h, v5.8h
        FMIN        v28.8h, v28.8h, v5.8h
        FMIN        v29.8h, v29.8h, v5.8h
        FMIN        v30.8h, v30.8h, v5.8h
        FMIN        v31.8h, v31.8h, v5.8h

        # Store full 4 x 16
        SUBS        x1, x1, 16
        B.LO        5f

        STP         q30, q31,  [x7]
        ADD         x7,  x7, x10
        STP         q28, q29, [x17]
        ADD         x17, x17, x10
        STP         q26, q27, [x16]
        ADD         x16, x16, x10
        STP         q24, q25,  [x6]
        ADD         x6,  x6, x10

        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b
        RET

        # Remainder- 1 halffloat of A
4:
        LDR         h0, [x8], 2
        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        LDR         h1, [x13], 2
        LDR         h2, [x14], 2
        LDR         h3, [x15], 2
        FMLA        v24.8h, v20.8h, v0.h[0]
        FMLA        v25.8h, v21.8h, v0.h[0]
        FMLA        v26.8h, v20.8h, v1.h[0]
        FMLA        v27.8h, v21.8h, v1.h[0]
        FMLA        v28.8h, v20.8h, v2.h[0]
        FMLA        v29.8h, v21.8h, v2.h[0]
        FMLA        v30.8h, v20.8h, v3.h[0]
        FMLA        v31.8h, v21.8h, v3.h[0]
        B           3b

        # Store odd width
5:
        TBZ         x1, 3, 6f
        STR         q30, [x7], 16
        MOV         v30.16b, v31.16b
        STR         q28, [x17], 16
        MOV         v28.16b, v29.16b
        STR         q26, [x16], 16
        MOV         v26.16b, v27.16b
        STR         q24, [x6], 16
        MOV         v24.16b, v25.16b

6:
        TBZ         x1, 2, 7f
        STR         d30, [x7], 8
        STR         d28, [x17], 8
        DUP         d30, v30.d[1]
        DUP         d28, v28.d[1]
        STR         d26, [x16], 8
        STR         d24, [x6], 8
        DUP         d26, v26.d[1]
        DUP         d24, v24.d[1]

7:
        TBZ         x1, 1, 8f
        STR         s30,  [x7], 4
        STR         s28, [x17], 4
        DUP         s30, v30.s[1]
        DUP         s28, v28.s[1]
        STR         s26, [x16], 4
        STR         s24,  [x6], 4
        DUP         s26, v26.s[1]
        DUP         s24, v24.s[1]
8:
        TBZ         x1, 0, 9f
        STR         h30,  [x7]
        STR         h28, [x17]
        STR         h26, [x16]
        STR         h24,  [x6]
9:
        RET

END_FUNCTION xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
