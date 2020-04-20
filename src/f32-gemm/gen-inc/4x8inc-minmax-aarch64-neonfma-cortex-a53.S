// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/4x8-aarch64-neonfma-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_gemminc_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53(
#     size_t mr,                x0
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const uint8_t*restrict a, x3
#     size_t a_stride,          x4
#     const void*restrict w,    x5
#     uint8_t*restrict c,       x6
#     size_t cm_stride,         x7
#     size_t cn_stride,         [sp] -> x14
#     const float*restrict acc,  [sp + 8] -> x15
#     const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 16] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointers
#  x3 a0
#  x9 a1
# x10 a2
# x11 a3

# C pointers
#  x6 c0
# x16 c1
# x17 c2
# x18 c3

# x4 temporary vector shadow register

# Vector register usage
# A0  v0     v3
# A1  v0[1]  v3[1]
# A2  v1     v4
# A3  v1[1]  v4[1]

# B   v12 v13 v14 v15 second set of B
# B   v16 v17 v18 v19 first set
# C   v20 v21
# C   v22 v23
# C   v24 v25
# C   v26 v27
# Clamp v6 v7

# unused A   v8 v9 v10 v11
# x12 a4
# x13 c4
#  x7 c5
# A4  v2     v5
# A5  v2[1]  v5[1]
# C   v28 v29
# C   v30 v31

BEGIN_FUNCTION xnn_f32_gemminc_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53

        # Load cn_stride, acc
        LDP x14, x15, [sp]
        # Load params pointer
        LDR x8, [sp, 16]

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

        # Load min/max values
        LD2R {v6.4s, v7.4s}, [x8]

        // Save d12-d15 on stack
        STP d12, d13, [sp, -32]!
        STP d14, d15, [sp, 16]

0:
        # Load initial accumulators
        LDP q20, q21, [x15], 32
        LDP q22, q23, [x15], 32
        LDP q24, q25, [x15], 32
        LDP q26, q27, [x15], 32
        PRFM PLDL1KEEP,  [x3,  0]  // Prefetch A
        PRFM PLDL1KEEP,  [x3, 64]
        PRFM PLDL1KEEP,  [x9,  0]
        PRFM PLDL1KEEP,  [x9, 64]
        PRFM PLDL1KEEP, [x10,  0]
        PRFM PLDL1KEEP, [x10, 64]
        PRFM PLDL1KEEP, [x11,  0]
        PRFM PLDL1KEEP, [x11, 64]
        PRFM PLDL1KEEP, [x5,   0]  // Prefetch B
        PRFM PLDL1KEEP, [x5,  64]
        PRFM PLDL1KEEP, [x5, 128]
        PRFM PLDL1KEEP, [x5, 192]

        # Is there at least 4 floats (16 bytes) for prologue + epilogue?
        SUBS x0, x2, 16  // k = kc - 16
        B.LO 5f

        # Prologue - First group loads, no FMA
        LDR   d0, [x3], 8              // a0
        LDP q16, q17, [x5], 32         // b
        LDR   d1, [x10], 8             // a2
        LD1  {v0.d}[1],  [x9], 8       // a1
        LD1  {v1.d}[1], [x11], 8       // a3
        SUBS x0, x0, 16
        LDR  q18, [x5], 16
        LDR  d19, [x5], 8
        LDR  x4, [x5], 8   // ins is in BLOCK 0

        # Is there at least 4 floats (16 bytes) for main loop?
        B.LO 2f

        # Main loop - 4 floats of A (16 bytes)
        # 32 FMA + 8 LD64 A + 8 LDR B
1:
        # First group of 16 FMA, Second group loads
        // BLOCK 0
        LDR   d3, [x3], 8              // a0
        INS v19.d[1], x4               // b from second group
        FMLA v20.4s, v16.4s,  v0.s[0]
        LDR  x4, [x9], 8               // a1
        FMLA v22.4s, v16.4s,  v0.s[2]
        FMLA v24.4s, v16.4s,  v1.s[0]

        // BLOCK 1
        LDR  d12, [x5]
        INS v3.d[1], x4                // a1 ins
        FMLA v26.4s, v16.4s,  v1.s[2]
        LDR  x4, [x5, 8]   // b
        FMLA v21.4s, v17.4s,  v0.s[0]
        FMLA v23.4s, v17.4s,  v0.s[2]

        // BLOCK 2
        LDR   d4, [x10], 8             // a2
        INS v12.d[1], x4  // b  ins
        FMLA v25.4s, v17.4s,  v1.s[0]
        LDR  x4, [x11], 8              // a3
        FMLA v27.4s, v17.4s,  v1.s[2]
        FMLA v20.4s, v18.4s,  v0.s[1]

        // BLOCK 3
        LDR  d13, [x5, 16]
        INS v4.d[1], x4                // a3 ins
        FMLA v22.4s, v18.4s,  v0.s[3]
        LDR  x4, [x5, 24]
        FMLA v24.4s, v18.4s,  v1.s[1]
        FMLA v26.4s, v18.4s,  v1.s[3]

        // BLOCK 4
        LDR  d14, [x5, 32]
        INS v13.d[1], x4  // b
        FMLA v21.4s, v19.4s,  v0.s[1]
        LDR  x4, [x5, 40]
        FMLA v23.4s, v19.4s,  v0.s[3]
        FMLA v25.4s, v19.4s,  v1.s[1]

        // BLOCK 5
        // NOPs to ensure 4 cycle LDR lands on next LDR
        LDR  d15, [x5, 48]
        INS v14.d[1], x4  // b from previous
        FMLA v27.4s, v19.4s,  v1.s[3]
        LDR x4, [x5, 56]
        NOP
        NOP
        NOP
        NOP

        # Second group of 16 FMA, First group of loads
        // BLOCK 0
        LDR   d0, [x3], 8              // a0
        INS v15.d[1], x4  // b from previous
        FMLA v20.4s, v12.4s,  v3.s[0]
        LDR  x4, [x9], 8               // a1
        FMLA v22.4s, v12.4s,  v3.s[2]
        FMLA v24.4s, v12.4s,  v4.s[0]
        PRFM PLDL1KEEP, [x3, 128]      // Prefetch A0

        // BLOCK 1
        LDR  d16, [x5, 64]
        INS v0.d[1], x4                // a1 ins
        FMLA v26.4s, v12.4s,  v4.s[2]
        LDR  x4, [x5, 72]  // b
        FMLA v21.4s, v13.4s,  v3.s[0]
        FMLA v23.4s, v13.4s,  v3.s[2]
        PRFM PLDL1KEEP, [x9, 128]      // Prefetch A1

        // BLOCK 2
        LDR   d1, [x10], 8             // a2
        INS v16.d[1], x4  // b
        FMLA v25.4s, v13.4s,  v4.s[0]
        LDR  x4, [x11], 8              // a3
        FMLA v27.4s, v13.4s,  v4.s[2]
        FMLA v20.4s, v14.4s,  v3.s[1]
        PRFM PLDL1KEEP, [x10, 128]     // Prefetch A2

        // BLOCK 3
        LDR  d17, [x5, 80]
        INS v1.d[1], x4                // a3 ins
        FMLA v22.4s, v14.4s,  v3.s[3]
        LDR  x4, [x5, 88]
        FMLA v24.4s, v14.4s,  v4.s[1]
        FMLA v26.4s, v14.4s,  v4.s[3]
        PRFM PLDL1KEEP, [x11, 128]     // Prefetch A3

        // BLOCK 4
        LDR  d18, [x5, 96]
        INS v17.d[1], x4  // b
        FMLA v21.4s, v15.4s,  v3.s[1]
        LDR  x4, [x5, 104]
        FMLA v23.4s, v15.4s,  v3.s[3]
        FMLA v25.4s, v15.4s,  v4.s[1]
        PRFM PLDL1KEEP, [x5, 192]      // Prefetch B

        // BLOCK 5
        // NOTE that block needs to be 4 cycles for LDR not to stall
        LDR  d19, [x5, 112]
        INS v18.d[1], x4
        FMLA v27.4s, v15.4s,  v4.s[3]
        LDR  x4, [x5, 120]
        SUBS x0, x0, 16
        PRFM PLDL1KEEP, [x5, 256]      // Prefetch B
        ADD x5, x5, 128
        B.HS 1b

        # Epilogue - 4 floats of A (16 bytes)
        # 32 FMA + 8 LD64 A + 8 LDR B
2:
        # First group of 16 FMA, Second group loads
        // BLOCK 0
        LDR   d3, [x3], 8              // a0
        INS v19.d[1], x4               // b from second group
        FMLA v20.4s, v16.4s,  v0.s[0]
        LDR  x4, [x9], 8               // a1
        FMLA v22.4s, v16.4s,  v0.s[2]
        FMLA v24.4s, v16.4s,  v1.s[0]

        // BLOCK 1
        LDR  d12, [x5]
        INS v3.d[1], x4                // a1 ins
        FMLA v26.4s, v16.4s,  v1.s[2]
        LDR  x4, [x5, 8]   // b
        FMLA v21.4s, v17.4s,  v0.s[0]
        FMLA v23.4s, v17.4s,  v0.s[2]

        // BLOCK 2
        LDR   d4, [x10], 8             // a2
        INS v12.d[1], x4  // b  ins
        FMLA v25.4s, v17.4s,  v1.s[0]
        LDR  x4, [x11], 8              // a3
        FMLA v27.4s, v17.4s,  v1.s[2]
        FMLA v20.4s, v18.4s,  v0.s[1]

        // BLOCK 3
        LDR  d13, [x5, 16]
        INS v4.d[1], x4                // a3 ins
        FMLA v22.4s, v18.4s,  v0.s[3]
        LDR  x4, [x5, 24]
        FMLA v24.4s, v18.4s,  v1.s[1]
        FMLA v26.4s, v18.4s,  v1.s[3]

        // BLOCK 4
        LDR  d14, [x5, 32]
        INS v13.d[1], x4  // b
        FMLA v21.4s, v19.4s,  v0.s[1]
        LDR  x4, [x5, 40]
        FMLA v23.4s, v19.4s,  v0.s[3]
        FMLA v25.4s, v19.4s,  v1.s[1]

        // BLOCK 5
        // NOPs to ensure 4 cycle LDR lands on next LDR
        LDR  d15, [x5, 48]
        INS v14.d[1], x4
        FMLA v27.4s, v19.4s,  v1.s[3]
        LDR x4, [x5, 56]
        NOP // fma
        NOP
        NOP // fma
        NOP

        # Second group of 16 FMA, no loads
        // BLOCK 0
        INS v15.d[1], x4  // b from previous
        FMLA v20.4s, v12.4s,  v3.s[0]
        FMLA v22.4s, v12.4s,  v3.s[2]
        FMLA v24.4s, v12.4s,  v4.s[0]

        // BLOCK 1
        FMLA v26.4s, v12.4s,  v4.s[2]
        FMLA v21.4s, v13.4s,  v3.s[0]
        FMLA v23.4s, v13.4s,  v3.s[2]

        // BLOCK 2
        FMLA v25.4s, v13.4s,  v4.s[0]
        FMLA v27.4s, v13.4s,  v4.s[2]
        FMLA v20.4s, v14.4s,  v3.s[1]

        // BLOCK 3
        FMLA v22.4s, v14.4s,  v3.s[3]
        FMLA v24.4s, v14.4s,  v4.s[1]
        FMLA v26.4s, v14.4s,  v4.s[3]
        TST x0, 15

        // BLOCK 4
        FMLA v21.4s, v15.4s,  v3.s[1]
        FMLA v23.4s, v15.4s,  v3.s[3]
        FMLA v25.4s, v15.4s,  v4.s[1]
        ADD x5, x5, 64

        // BLOCK 5
        FMLA v27.4s, v15.4s,  v4.s[3]

        # Is there a remainder?- 2 floats of A (8 bytes) or less
        B.NE 5f

4:
        # Clamp
        FMAX v20.4s, v20.4s, v6.4s
        SUBS x1, x1, 8
        FMAX v21.4s, v21.4s, v6.4s
        FMAX v22.4s, v22.4s, v6.4s
        FMAX v23.4s, v23.4s, v6.4s
        FMAX v24.4s, v24.4s, v6.4s
        FMAX v25.4s, v25.4s, v6.4s
        FMAX v26.4s, v26.4s, v6.4s
        FMAX v27.4s, v27.4s, v6.4s
        FMIN v20.4s, v20.4s, v7.4s
        FMIN v21.4s, v21.4s, v7.4s
        FMIN v22.4s, v22.4s, v7.4s
        FMIN v23.4s, v23.4s, v7.4s
        FMIN v24.4s, v24.4s, v7.4s
        FMIN v25.4s, v25.4s, v7.4s
        FMIN v26.4s, v26.4s, v7.4s
        FMIN v27.4s, v27.4s, v7.4s

        # Store full 4 x 8
        B.LO 8f

        ST1 {v26.16b, v27.16b}, [x18], x14
        SUB  x3,  x3, x2 // a0 -= kc
        ST1 {v24.16b, v25.16b}, [x17], x14
        SUB  x9,  x9, x2 // a1 -= kc
        ST1 {v22.16b, v23.16b}, [x16], x14
        SUB x10, x10, x2 // a2 -= kc
        ST1 {v20.16b, v21.16b},  [x6], x14
        SUB x11, x11, x2 // a3 -= kc

        B.HI 0b

        // Restore d12-d15 from stack
        LDP d14, d15, [sp, 16]
        LDP d12, d13, [sp], 32
        RET

5:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBZ x0, 3, 6f

        # Remainder- 2 floats of A (8 bytes)
        LDR   d0,  [x3], 8
        LDR  q16, [x5], 16
        LD1   {v0.d}[1], [x9], 8
        LDR   d1, [x10], 8
        LD1   {v1.d}[1], [x11], 8
        LDR  q17, [x5], 16
        LDR  q18, [x5], 16
        LDR  q19, [x5], 16
        FMLA v20.4s, v16.4s,  v0.s[0]
        FMLA v22.4s, v16.4s,  v0.s[2]
        FMLA v24.4s, v16.4s,  v1.s[0]
        FMLA v26.4s, v16.4s,  v1.s[2]
        FMLA v21.4s, v17.4s,  v0.s[0]
        FMLA v23.4s, v17.4s,  v0.s[2]
        FMLA v25.4s, v17.4s,  v1.s[0]
        FMLA v27.4s, v17.4s,  v1.s[2]

        FMLA v20.4s, v18.4s,  v0.s[1]
        FMLA v22.4s, v18.4s,  v0.s[3]
        FMLA v24.4s, v18.4s,  v1.s[1]
        FMLA v26.4s, v18.4s,  v1.s[3]
        FMLA v21.4s, v19.4s,  v0.s[1]
        FMLA v23.4s, v19.4s,  v0.s[3]
        FMLA v25.4s, v19.4s,  v1.s[1]
        FMLA v27.4s, v19.4s,  v1.s[3]

        # Is there a remainder?- 1 floats of A (4 bytes)
        TBZ x0, 2, 4b

6:
        # Remainder- 1 floats of A (4 bytes)
        LDR   s0,  [x3], 4
        LDR  q16, [x5], 16
        LD1   {v0.s}[2], [x9], 4
        LDR   s1, [x10], 4
        LD1   {v1.s}[2], [x11], 4
        LDR  q17, [x5], 16

        FMLA v20.4s, v16.4s,  v0.s[0]
        FMLA v22.4s, v16.4s,  v0.s[2]
        FMLA v24.4s, v16.4s,  v1.s[0]
        FMLA v26.4s, v16.4s,  v1.s[2]
        FMLA v21.4s, v17.4s,  v0.s[0]
        FMLA v23.4s, v17.4s,  v0.s[2]
        FMLA v25.4s, v17.4s,  v1.s[0]
        FMLA v27.4s, v17.4s,  v1.s[2]
        B 4b

        # Store odd width
8:
        TBZ x1, 2, 9f
        STR q26, [x18], 16
        MOV v26.16b, v27.16b
        STR q24, [x17], 16
        MOV v24.16b, v25.16b
        STR q22, [x16], 16
        MOV v22.16b, v23.16b
        STR q20,  [x6], 16
        MOV v20.16b, v21.16b

9:
        TBZ x1, 1, 10f
        STR d26, [x18], 8
        DUP d26, v26.d[1]
        STR d24, [x17], 8
        DUP d24, v24.d[1]
        STR d22, [x16], 8
        DUP d22, v22.d[1]
        STR d20,  [x6], 8
        DUP d20, v20.d[1]

10:
        TBZ x1, 0, 11f
        STR s26, [x18]
        STR s24, [x17]
        STR s22, [x16]
        STR s20,  [x6]
11:
        // Restore d12-d15 from stack
        LDP d14, d15, [sp, 16]
        LDP d12, d13, [sp], 32
        RET

END_FUNCTION xnn_f32_gemminc_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
