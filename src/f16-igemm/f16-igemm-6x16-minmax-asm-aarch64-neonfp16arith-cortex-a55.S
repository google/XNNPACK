// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const void** restrict a,            x4
#     const void* restrict w,             x5
#     uint8_t* restrict c,                x6
#     size_t cm_stride,                  x7
#     size_t cn_stride,                  [sp] -> x8
#     size_t a_offset,                   [sp + 8] -> x11
#     const void* zero,                  [sp + 16] -> x12
#     const xnn_f16_minmax_params params [sp + 24] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0 x14 v0
// A1 x15 v1
// A2 x20 v2
// A3 x21 v3
// A4 x22 v4
// A5 x23 v5
// B   x5 v16 v17 v18 v19
// C0  x6  v20 v21
// C1 x16  v22 v23
// C2 x17  v24 v25
// C3 x10  v26 v27
// C4 x13  v28 v29
// C5  x7  v30 v31
// clamp  v6, (v4), (v5)
// unused     v7
// unused A   v8 v9 v10 v11
// unused B   v12 v13 v14 v15

BEGIN_FUNCTION xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x16, x6, x7             // c1 = c0 + cm_stride
        CSEL        x16, x6, x16, LO        //   c1 = c0
        ADD         x17, x16, x7            // c2 = c1 + cm_stride
                                            // if mr <= 2
        CSEL        x17, x16, x17, LS       //   c2 = c1

        # Load params
        LDR         s6, [x8]

        CMP         x0, 4                   // if mr < 4
        ADD         x10, x17, x7            // c3 = c2 + cm_stride
        CSEL        x10, x17, x10, LO       //   c3 = c2
        ADD         x13, x10, x7            // c4 = c3 + cm_stride
                                            // if mr <= 4
        CSEL        x13, x10, x13, LS       //   c4 = c3
        CMP         x0, 6                   // if mr < 6
        ADD         x7, x13, x7             // c5 = c4 + cm_stride
        CSEL        x7, x13, x7, LO         //   c5 = c4

        LDP         x8, x11, [sp]           // load cn_stride, a_offset

        # Save x20-x23 on stack
        STP         x20, x21, [sp, -32]!
        STP         x22, x23, [sp, 16]

0:
        # Load initial bias from w into accumulators
        LDP         q20, q21, [x5], 32
        MOV         x9, x3                  // p = ks
        MOV         v22.16b, v20.16b
        PRFM        PLDL1KEEP, [x5, 0]      // Prefetch B
        MOV         v23.16b, v21.16b
        PRFM        PLDL1KEEP, [x5, 64]
        MOV         v24.16b, v20.16b
        PRFM        PLDL1KEEP, [x5, 128]
        MOV         v25.16b, v21.16b
        PRFM        PLDL1KEEP, [x5, 192]
        MOV         v26.16b, v20.16b
        PRFM        PLDL1KEEP, [x5, 256]
        MOV         v27.16b, v21.16b
        PRFM        PLDL1KEEP, [x5, 320]
        MOV         v28.16b, v20.16b
        MOV         v29.16b, v21.16b
        MOV         v30.16b, v20.16b
        MOV         v31.16b, v21.16b

1:
        # Load next 6 A pointers
        LDP         x14, x15, [x4], 16
        LDP         x20, x21, [x4], 16
        LDP         x22, x23, [x4], 16

        CMP         x14, x12                // if a0 == zero
        ADD         x14, x14, x11           // a0 += a_offset
        CSEL        x14, x12, x14, EQ       //   a0 = zero, else += a0 + a_offset
        CMP         x15, x12                // if a1 == zero
        ADD         x15, x15, x11           // a1 += a_offset
        CSEL        x15, x12, x15, EQ       //   a1 = zero, else += a1 + a_offset
        CMP         x20, x12                // if a2 == zero
        ADD         x20, x20, x11           // a2 += a_offset
        CSEL        x20, x12, x20, EQ       //   a2 = zero, else += a2 + a_offset
        CMP         x21, x12                // if a3 == zero
        ADD         x21, x21, x11           // a3 += a_offset
        CSEL        x21, x12, x21, EQ       //   a3 = zero, else += a3 + a_offset
        CMP         x22, x12                // if a4 == zero
        ADD         x22, x22, x11           // a4 += a_offset
        CSEL        x22, x12, x22, EQ       //   a4 = zero, else += a4 + a_offset
        CMP         x23, x12                // if a5 == zero
        ADD         x23, x23, x11           // a5 += a_offset
        CSEL        x23, x12, x23, EQ       //   a5 = zero, else += a5 + a_offset

        # Is there at least 2 halffloats (4 bytes)?
        SUBS        x0, x2, 4               // k = kc - 4
        B.LO        5f

        # Prologue - load 4 A and 2 B
        LDR         s0, [x14], 4                // A0
        LDR         q16, [x5], 16               // B
        LDR         q17, [x5], 16               // B
        LDR         s1, [x15], 4                // A1
        LDR         s2, [x20], 4                // A2
        LDR         s3, [x21], 4                // A3

        # Is there at least 2 halffloats for main loop?
        SUBS        x0, x0, 4
        B.LO        3f

       .p2align 3
        # Main loop - 2 halffloats of A (4 bytes)
        # 24 FMA + 6 ld32 A + 4 LDR B
2:
        FMLA        v20.8h, v16.8h,  v0.h[0]
        LDR         s4, [x22], 4                // A4
        FMLA        v21.8h, v17.8h,  v0.h[0]
        LDR         s5, [x23], 4                // A5
        FMLA        v22.8h, v16.8h,  v1.h[0]
        LDR         d18, [x5], 8                // B0
        FMLA        v23.8h, v17.8h,  v1.h[0]
        LD1         {v18.d}[1], [x5], 8         // B1
        FMLA        v24.8h, v16.8h,  v2.h[0]
        LDR         d19, [x5], 8                // B2
        FMLA        v25.8h, v17.8h,  v2.h[0]
        LD1         {v19.d}[1], [x5], 8         // B3
        FMLA        v26.8h, v16.8h,  v3.h[0]
        FMLA        v27.8h, v17.8h,  v3.h[0]
        FMLA        v28.8h, v16.8h,  v4.h[0]
        FMLA        v29.8h, v17.8h,  v4.h[0]
        FMLA        v30.8h, v16.8h,  v5.h[0]
        FMLA        v31.8h, v17.8h,  v5.h[0]
        SUBS        x0, x0, 4

        FMLA        v20.8h, v18.8h,  v0.h[1]
        LDR         d16, [x5], 8                // B0
        FMLA        v21.8h, v19.8h,  v0.h[1]
        LD1         {v16.d}[1], [x5], 8         // B1
        FMLA        v22.8h, v18.8h,  v1.h[1]
        LDR         d17, [x5], 8                // B2
        FMLA        v23.8h, v19.8h,  v1.h[1]
        LD1         {v17.d}[1], [x5], 8         // B3
        FMLA        v24.8h, v18.8h,  v2.h[1]
        FMLA        v25.8h, v19.8h,  v2.h[1]
        FMLA        v26.8h, v18.8h,  v3.h[1]
        FMLA        v27.8h, v19.8h,  v3.h[1]
        LDR         s0,  [x14], 4               // A0
        FMLA        v28.8h, v18.8h,  v4.h[1]
        LDR         s1,  [x15], 4               // A1
        FMLA        v29.8h, v19.8h,  v4.h[1]
        LDR         s2, [x20], 4                // A2
        FMLA        v30.8h, v18.8h,  v5.h[1]
        LDR         s3, [x21], 4                // A3
        FMLA        v31.8h, v19.8h,  v5.h[1]
        B.HS        2b

        # Epilogue - same as main loop but no loads for next loop
3:
        FMLA        v20.8h, v16.8h,  v0.h[0]
        LDR         s4, [x22], 4                // A4
        FMLA        v21.8h, v17.8h,  v0.h[0]
        LDR         s5, [x23], 4                // A5
        FMLA        v22.8h, v16.8h,  v1.h[0]
        LDR         d18, [x5], 8                // B0
        FMLA        v23.8h, v17.8h,  v1.h[0]
        LD1         {v18.d}[1], [x5], 8         // B1
        FMLA        v24.8h, v16.8h,  v2.h[0]
        LDR         d19, [x5], 8                // B2
        FMLA        v25.8h, v17.8h,  v2.h[0]
        LD1         {v19.d}[1], [x5], 8         // B3
        FMLA        v26.8h, v16.8h,  v3.h[0]
        FMLA        v27.8h, v17.8h,  v3.h[0]
        FMLA        v28.8h, v16.8h,  v4.h[0]
        FMLA        v29.8h, v17.8h,  v4.h[0]
        FMLA        v30.8h, v16.8h,  v5.h[0]
        FMLA        v31.8h, v17.8h,  v5.h[0]

        FMLA        v20.8h, v18.8h,  v0.h[1]
        FMLA        v21.8h, v19.8h,  v0.h[1]
        FMLA        v22.8h, v18.8h,  v1.h[1]
        FMLA        v23.8h, v19.8h,  v1.h[1]
        FMLA        v24.8h, v18.8h,  v2.h[1]
        FMLA        v25.8h, v19.8h,  v2.h[1]
        FMLA        v26.8h, v18.8h,  v3.h[1]
        FMLA        v27.8h, v19.8h,  v3.h[1]
        FMLA        v28.8h, v18.8h,  v4.h[1]
        FMLA        v29.8h, v19.8h,  v4.h[1]
        FMLA        v30.8h, v18.8h,  v5.h[1]
        FMLA        v31.8h, v19.8h,  v5.h[1]

        # Is there a remainder?- 1 halffloat of A (2 bytes)
        TBNZ        x0, 1, 5f

4:
        # ks loop
        SUBS        x9, x9, 48              // ks -= MR * sizeof(void*)
        B.HI        1b

        # Clamp
        DUP         v4.8h, v6.h[0]
        DUP         v5.8h, v6.h[1]
        FMAX        v20.8h, v20.8h, v4.8h
        FMAX        v21.8h, v21.8h, v4.8h
        FMAX        v22.8h, v22.8h, v4.8h
        FMAX        v23.8h, v23.8h, v4.8h
        FMAX        v24.8h, v24.8h, v4.8h
        FMAX        v25.8h, v25.8h, v4.8h
        FMAX        v26.8h, v26.8h, v4.8h
        FMAX        v27.8h, v27.8h, v4.8h
        FMAX        v28.8h, v28.8h, v4.8h
        FMAX        v29.8h, v29.8h, v4.8h
        FMAX        v30.8h, v30.8h, v4.8h
        FMAX        v31.8h, v31.8h, v4.8h
        SUBS        x1, x1, 16
        FMIN        v20.8h, v20.8h, v5.8h
        FMIN        v21.8h, v21.8h, v5.8h
        FMIN        v22.8h, v22.8h, v5.8h
        FMIN        v23.8h, v23.8h, v5.8h
        FMIN        v24.8h, v24.8h, v5.8h
        FMIN        v25.8h, v25.8h, v5.8h
        FMIN        v26.8h, v26.8h, v5.8h
        FMIN        v27.8h, v27.8h, v5.8h
        FMIN        v28.8h, v28.8h, v5.8h
        FMIN        v29.8h, v29.8h, v5.8h
        FMIN        v30.8h, v30.8h, v5.8h
        FMIN        v31.8h, v31.8h, v5.8h

        # Store full 6 x 16
        B.LO        6f

        ST1         {v30.16b, v31.16b},  [x7], x8
        ST1         {v28.16b, v29.16b}, [x13], x8
        ST1         {v26.16b, v27.16b}, [x10], x8
        ST1         {v24.16b, v25.16b}, [x17], x8
        ST1         {v22.16b, v23.16b}, [x16], x8
        ST1         {v20.16b, v21.16b},  [x6], x8

        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b

        # Restore x20-x23 from stack
        LDP         x22, x23, [sp, 16]
        LDP         x20, x21, [sp], 32
        RET

5:
        # Remainder- 1 halffloat of A (2 bytes)
        LDR         h0, [x14], 2                // A0
        LDR         q16, [x5], 16               // B
        LDR         q17, [x5], 16               // B
        FMLA        v20.8h, v16.8h,  v0.h[0]
        LDR         h1, [x15], 2                // A1
        FMLA        v21.8h, v17.8h,  v0.h[0]
        LDR         h2, [x20], 2                // A2
        FMLA        v22.8h, v16.8h,  v1.h[0]
        LDR         h3, [x21], 2                // A3
        FMLA        v23.8h, v17.8h,  v1.h[0]
        LDR         h4,  [x22], 2               // A4
        FMLA        v24.8h, v16.8h,  v2.h[0]
        LDR         h5,  [x23], 2               // A5
        FMLA        v25.8h, v17.8h,  v2.h[0]
        FMLA        v26.8h, v16.8h,  v3.h[0]
        FMLA        v27.8h, v17.8h,  v3.h[0]
        FMLA        v28.8h, v16.8h,  v4.h[0]
        FMLA        v29.8h, v17.8h,  v4.h[0]
        FMLA        v30.8h, v16.8h,  v5.h[0]
        FMLA        v31.8h, v17.8h,  v5.h[0]
        B           4b

        # Store odd width
6:
        TBZ         x1, 3, 7f
        STR         q30,  [x7], 16
        MOV         v30.16b, v31.16b
        STR         q28, [x13], 16
        MOV         v28.16b, v29.16b
        STR         q26, [x10], 16
        MOV         v26.16b, v27.16b
        STR         q24, [x17], 16
        MOV         v24.16b, v25.16b
        STR         q22, [x16], 16
        MOV         v22.16b, v23.16b
        STR         q20,  [x6], 16
        MOV         v20.16b, v21.16b
7:
        TBZ         x1, 2, 8f
        STR         d30,  [x7], 8
        STR         d28, [x13], 8
        DUP         d30, v30.d[1]
        DUP         d28, v28.d[1]
        STR         d26, [x10], 8
        STR         d24, [x17], 8
        DUP         d26, v26.d[1]
        DUP         d24, v24.d[1]
        STR         d22, [x16], 8
        STR         d20,  [x6], 8
        DUP         d22, v22.d[1]
        DUP         d20, v20.d[1]

8:
        TBZ         x1, 1, 9f
        STR         s30,  [x7], 4
        STR         s28, [x13], 4
        DUP         s30, v30.s[1]
        DUP         s28, v28.s[1]
        STR         s26, [x10], 4
        STR         s24, [x17], 4
        DUP         s26, v26.s[1]
        DUP         s24, v24.s[1]
        STR         s22, [x16], 4
        STR         s20,  [x6], 4
        DUP         s22, v22.s[1]
        DUP         s20, v20.s[1]

9:
        TBZ         x1, 0, 10f
        STR         h30,  [x7]
        STR         h28, [x13]
        STR         h26, [x10]
        STR         h24, [x17]
        STR         h22, [x16]
        STR         h20,  [x6]
10:
        # Restore x20-x23 from stack
        LDP         x22, x23, [sp, 16]
        LDP         x20, x21, [sp], 32
        RET

END_FUNCTION xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
