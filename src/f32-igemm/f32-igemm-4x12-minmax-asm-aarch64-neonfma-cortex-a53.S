// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_igemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float** restrict a,           x4
#     const float* restrict w,            x5
#     float* restrict c,                  x6
#     size_t cm_stride,                  x7
#     size_t cn_stride,                  [sp] -> (x0)
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_minmax_params params [sp + 24] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x13  v0
# A1  x14  v0[1]
# A2  x15  v1
# A3  x16  v1[1]
# A0  x13  v2
# A1  x14  v2[1]
# A2  x15  v3
# A3  x16  v3[1]
# B    v6  v7  v8
# B    v9 v10 v11
# B   v14 v15 v16
# B   v17 v18 v19
# C0   x6 v20 v21 v22
# C1  x17 v23 v24 v25
# C2  x10 v26 v27 v28
# C3   x7 v29 v30 v31
# temporary vector shadow register x8
# Clamp v4 v5
# unused v12 v13

BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53

        # Load a_offset
        LDR         x11, [sp, 8]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Save d8-d11,d14,d15 on stack
        STP         d8,  d9, [sp, -48]!
        STP         d10, d11, [sp, 16]
        STP         d14, d15, [sp, 32]

        # Load min/max values
        LD2R        {v4.4s, v5.4s}, [x8]

        # Clamp C pointers
        CMP         x0, 2                   // if mr < 2
        ADD         x17, x6, x7             // c1 = c0 + cm_stride
        CSEL        x17, x6, x17, LO        //   c1 = c0

        ADD         x10, x17, x7            // c2 = c1 + cm_stride
                                            // if mr <= 2

        CSEL        x10, x17, x10, LS       //   c2 = c1

        CMP         x0, 4                   // if mr < 4
        ADD         x7, x10, x7             // c3 = c2 + cm_stride
        CSEL        x7, x10, x7, LO         //   c3 = c2

0:
        # Load initial bias from w into accumulators
        LD1         {v20.16b, v21.16b, v22.16b}, [x5], 48
        MOV         v23.16b, v20.16b
        PRFM        PLDL1KEEP, [x5,   0]    // Prefetch B
        MOV         v24.16b, v21.16b
        PRFM        PLDL1KEEP, [x5,  64]
        MOV         v25.16b, v22.16b
        PRFM        PLDL1KEEP, [x5, 128]
        MOV         v26.16b, v20.16b
        PRFM        PLDL1KEEP, [x5, 192]
        MOV         v27.16b, v21.16b
        PRFM        PLDL1KEEP, [x5, 256]
        MOV         v28.16b, v22.16b
        PRFM        PLDL1KEEP, [x5, 320]
        MOV         v29.16b, v20.16b
        MOV         v30.16b, v21.16b
        MOV         v31.16b, v22.16b

        MOV         x9, x3                  // p = ks

1:
        # Load next 4 A pointers
        LDP         x13, x14, [x4], 16
        LDP         x15, x16, [x4], 16

        CMP         x13, x12                // if a0 == zero
        ADD         x13, x13, x11           // a0 += a_offset
        CSEL        x13, x12, x13, EQ       //   a0 = zero, else += a0 + a_offset
        CMP         x14, x12                // if a1 == zero
        ADD         x14, x14, x11           // a1 += a_offset
        CSEL        x14, x12, x14, EQ       //   a1 = zero, else += a1 + a_offset
        CMP         x15, x12                // if a2 == zero
        ADD         x15, x15, x11           // a2 += a_offset
        CSEL        x15, x12, x15, EQ       //   a2 = zero, else += a2 + a_offset
        CMP         x16, x12                // if a3 == zero
        ADD         x16, x16, x11           // a3 += a_offset
        CSEL        x16, x12, x16, EQ       //   a3 = zero, else += a3 + a_offset

        # Is there at least 4 floats (16 bytes) for prologue + epilogue?
        SUBS        x0, x2, 16              // k = kc - 16

        PRFM        PLDL1KEEP, [x13,  0]    // Prefetch A
        PRFM        PLDL1KEEP, [x13, 64]
        PRFM        PLDL1KEEP, [x14,  0]
        PRFM        PLDL1KEEP, [x14, 64]
        PRFM        PLDL1KEEP, [x15,  0]
        PRFM        PLDL1KEEP, [x15, 64]
        PRFM        PLDL1KEEP, [x16,  0]
        PRFM        PLDL1KEEP, [x16, 64]
        B.LO        5f

        SUBS        x0, x0, 16              // 4 floats for main loop

        # Prologue - loads for first group of 24 FMA

        # Read first block of 4 A.
        LDR         d0, [x13], 8                // a0
        LDR         d1, [x15], 8                // a2
        LD1         {v0.d}[1], [x14], 8         // a1
        LD1         {v1.d}[1], [x16], 8         // a3

        LD1         {v6.16b, v7.16b, v8.16b}, [x5], 48
        LD1         {v9.16b, v10.16b}, [x5], 32
        LDR         d11, [x5], 8
        LDR         x8, [x5], 8

        # Is there at least 4 floats (16 bytes) for main loop?
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
2:
        # First group of 24 fma.  8 blocks of 4 cycles.  LDR + 3 FMA
        # A is loaded for 2nd group into v2/v3
        # INS is 4 blocks (16 cycles) after load

        # BLOCK 0
        LDR         d2, [x13], 8                // a0
        INS         v11.d[1], x8
        FMLA        v20.4s, v6.4s, v0.s[0]
        LDR         x8, [x14], 8                // a1
        FMLA        v23.4s, v6.4s, v0.s[2]
        FMLA        v26.4s, v6.4s, v1.s[0]
        PRFM        PLDL1KEEP, [x13, 128]       // Prefetch A0

        # BLOCK 1
        LDR         d3, [x15], 8                // a2
        INS         v2.d[1], x8                 // a1 was loaded in block 0
        FMLA        v29.4s, v6.4s, v1.s[2]
        LDR         x8, [x16], 8                // a3
        FMLA        v21.4s, v7.4s, v0.s[0]
        FMLA        v24.4s, v7.4s, v0.s[2]
        PRFM        PLDL1KEEP, [x14, 128]       // Prefetch A1

        # BLOCK 2
        LDR         d14, [x5]                   // vb0x0123
        INS         v3.d[1], x8                 // a3 was loaded in block 1
        FMLA        v27.4s, v7.4s, v1.s[0]
        LDR         x8, [x5, 8]
        FMLA        v30.4s, v7.4s, v1.s[2]
        FMLA        v22.4s, v8.4s, v0.s[0]
        PRFM        PLDL1KEEP, [x15, 128]       // Prefetch A2

        # BLOCK 3
        LDR         d15, [x5, 16]               // vb0x4567
        INS         v14.d[1], x8                // v14 was loaded in block 2
        FMLA        v25.4s, v8.4s, v0.s[2]
        LDR         x8, [x5, 24]
        FMLA        v28.4s, v8.4s, v1.s[0]
        FMLA        v31.4s, v8.4s, v1.s[2]
        PRFM        PLDL1KEEP, [x16, 128]       // Prefetch A3

        # BLOCK 4
        LDR         d16, [x5, 32]               // vb0x89AB
        INS         v15.d[1], x8
        FMLA        v20.4s, v9.4s, v0.s[1]
        LDR         x8, [x5, 40]
        FMLA        v23.4s, v9.4s, v0.s[3]
        FMLA        v26.4s, v9.4s, v1.s[1]
        PRFM        PLDL1KEEP, [x5, 320]        // Prefetch B

        # BLOCK 5
        LDR         d17, [x5, 48]               // vb1x0123
        INS         v16.d[1], x8
        FMLA        v29.4s, v9.4s, v1.s[3]
        LDR         x8, [x5, 56]
        FMLA        v21.4s, v10.4s, v0.s[1]
        FMLA        v24.4s, v10.4s, v0.s[3]
        PRFM        PLDL1KEEP, [x5, 384]        // Prefetch B

        # BLOCK 6
        LDR         d18, [x5, 64]               // vb1x4567
        INS         v17.d[1], x8
        FMLA        v27.4s, v10.4s, v1.s[1]
        LDR         x8, [x5, 72]
        FMLA        v30.4s, v10.4s, v1.s[3]
        FMLA        v22.4s, v11.4s, v0.s[1]
        PRFM        PLDL1KEEP, [x5, 448]        // Prefetch B

        # BLOCK 7
        LDR         d19, [x5, 80]               // vb1x89AB
        INS         v18.d[1], x8
        FMLA        v25.4s, v11.4s, v0.s[3]
        LDR         x8, [x5, 88]
        FMLA        v28.4s, v11.4s, v1.s[1]
        FMLA        v31.4s, v11.4s, v1.s[3]

        # Second group of 24 fma.  8 blocks of 4 cycles.  LDR + 3 FMA
        # A is loaded for 1st group into v0/v1

        # BLOCK 0
        LDR         d0, [x13], 8                // a0
        INS         v19.d[1], x8
        FMLA        v20.4s, v14.4s, v2.s[0]
        LDR         x8, [x14], 8                // a1
        FMLA        v23.4s, v14.4s, v2.s[2]
        FMLA        v26.4s, v14.4s, v3.s[0]

        # BLOCK 1
        LDR         d1, [x15], 8                // a2
        INS         v0.d[1], x8                 // a1
        FMLA        v29.4s, v14.4s, v3.s[2]
        LDR         x8, [x16], 8                // a3
        FMLA        v21.4s, v15.4s, v2.s[0]
        FMLA        v24.4s, v15.4s, v2.s[2]

        # BLOCK 2
        LDR         d6, [x5, 96]                // vb0x0123
        INS         v1.d[1], x8                 // a3
        FMLA        v27.4s, v15.4s, v3.s[0]
        LDR         x8, [x5, 104]
        FMLA        v30.4s, v15.4s, v3.s[2]
        FMLA        v22.4s, v16.4s, v2.s[0]

        # BLOCK 3
        LDR         d7, [x5, 112]               // vb0x4567
        INS         v6.d[1], x8
        FMLA        v25.4s, v16.4s, v2.s[2]
        LDR         x8, [x5, 120]
        FMLA        v28.4s, v16.4s, v3.s[0]
        FMLA        v31.4s, v16.4s, v3.s[2]

        # BLOCK 4
        LDR         d8, [x5, 128]               // vb0x89AB
        INS         v7.d[1], x8
        FMLA        v20.4s, v17.4s, v2.s[1]
        LDR         x8, [x5, 136]
        FMLA        v23.4s, v17.4s, v2.s[3]
        FMLA        v26.4s, v17.4s, v3.s[1]

        # BLOCK 5
        LDR         d9, [x5, 144]               // vb1x0123
        INS         v8.d[1], x8
        FMLA        v29.4s, v17.4s, v3.s[3]
        LDR         x8, [x5, 152]
        FMLA        v21.4s, v18.4s, v2.s[1]
        FMLA        v24.4s, v18.4s, v2.s[3]

        # BLOCK 6
        LDR         d10, [x5, 160]              // vb1x4567
        INS         v9.d[1], x8
        FMLA        v27.4s, v18.4s, v3.s[1]
        LDR         x8, [x5, 168]
        FMLA        v30.4s, v18.4s, v3.s[3]
        SUBS        x0, x0, 16
        FMLA        v22.4s, v19.4s, v2.s[1]

        # BLOCK 7
        LDR         d11, [x5, 176]              // vb1x89AB
        INS         v10.d[1], x8
        FMLA        v25.4s, v19.4s, v2.s[3]
        LDR         x8, [x5, 184]
        FMLA        v28.4s, v19.4s, v3.s[1]
        ADD         x5, x5, 192
        FMLA        v31.4s, v19.4s, v3.s[3]
        B.HS        2b

        # Epilogue
        # First block same as main loop.  Second block has no loads.
3:
        # BLOCK 0
        LDR         d2, [x13], 8                // a0
        INS         v11.d[1], x8
        FMLA        v20.4s, v6.4s, v0.s[0]
        LDR         x8, [x14], 8                // a1
        FMLA        v23.4s, v6.4s, v0.s[2]
        FMLA        v26.4s, v6.4s, v1.s[0]

        # BLOCK 1
        LDR         d3, [x15], 8                // a2
        INS         v2.d[1], x8                 // a1 was loaded in block 0
        FMLA        v29.4s, v6.4s, v1.s[2]
        LDR         x8, [x16], 8                // a3
        FMLA        v21.4s, v7.4s, v0.s[0]
        FMLA        v24.4s, v7.4s, v0.s[2]

        # BLOCK 2
        LDR         d14, [x5]                   // vb0x0123
        INS         v3.d[1], x8                 // a3 was loaded in block 1
        FMLA        v27.4s, v7.4s, v1.s[0]
        LDR         x8, [x5, 8]
        FMLA        v30.4s, v7.4s, v1.s[2]
        FMLA        v22.4s, v8.4s, v0.s[0]

        # BLOCK 3
        LDR         d15, [x5, 16]               // vb0x4567
        INS         v14.d[1], x8                // v14 was loaded in block 2
        FMLA        v25.4s, v8.4s, v0.s[2]
        LDR         x8, [x5, 24]
        FMLA        v28.4s, v8.4s, v1.s[0]
        FMLA        v31.4s, v8.4s, v1.s[2]

        # BLOCK 4
        LDR         d16, [x5, 32]               // vb0x89AB
        INS         v15.d[1], x8
        FMLA        v20.4s, v9.4s, v0.s[1]
        LDR         x8, [x5, 40]
        FMLA        v23.4s, v9.4s, v0.s[3]
        FMLA        v26.4s, v9.4s, v1.s[1]

        # BLOCK 5
        LDR         d17, [x5, 48]               // vb1x0123
        INS         v16.d[1], x8
        FMLA        v29.4s, v9.4s, v1.s[3]
        LDR         x8, [x5, 56]
        FMLA        v21.4s, v10.4s, v0.s[1]
        FMLA        v24.4s, v10.4s, v0.s[3]

        # BLOCK 6
        LDR         d18, [x5, 64]               // vb1x4567
        INS         v17.d[1], x8
        FMLA        v27.4s, v10.4s, v1.s[1]
        LDR         x8, [x5, 72]
        FMLA        v30.4s, v10.4s, v1.s[3]
        FMLA        v22.4s, v11.4s, v0.s[1]

        # BLOCK 7
        LDR         d19, [x5, 80]               // vb1x89AB
        INS         v18.d[1], x8
        FMLA        v25.4s, v11.4s, v0.s[3]
        LDR         x8, [x5, 88]
        FMLA        v28.4s, v11.4s, v1.s[1]
        FMLA        v31.4s, v11.4s, v1.s[3]

        # Second group of 24 fma.  8 blocks of 4 cycles.  LDR + 3 FMA
        # A is loaded for 1st group into v0/v1

        # BLOCK 0
        INS         v19.d[1], x8
        FMLA        v20.4s, v14.4s, v2.s[0]
        FMLA        v23.4s, v14.4s, v2.s[2]
        FMLA        v26.4s, v14.4s, v3.s[0]

        # BLOCK 1
        FMLA        v29.4s, v14.4s, v3.s[2]
        FMLA        v21.4s, v15.4s, v2.s[0]
        FMLA        v24.4s, v15.4s, v2.s[2]

        # BLOCK 2
        FMLA        v27.4s, v15.4s, v3.s[0]
        FMLA        v30.4s, v15.4s, v3.s[2]
        FMLA        v22.4s, v16.4s, v2.s[0]

        # BLOCK 3
        FMLA        v25.4s, v16.4s, v2.s[2]
        FMLA        v28.4s, v16.4s, v3.s[0]
        FMLA        v31.4s, v16.4s, v3.s[2]

        # BLOCK 4
        FMLA        v20.4s, v17.4s, v2.s[1]
        FMLA        v23.4s, v17.4s, v2.s[3]
        FMLA        v26.4s, v17.4s, v3.s[1]

        # BLOCK 5
        FMLA        v29.4s, v17.4s, v3.s[3]
        FMLA        v21.4s, v18.4s, v2.s[1]
        FMLA        v24.4s, v18.4s, v2.s[3]

        # BLOCK 6
        FMLA        v27.4s, v18.4s, v3.s[1]
        FMLA        v30.4s, v18.4s, v3.s[3]
        FMLA        v22.4s, v19.4s, v2.s[1]
        TST         x0, 15

        # BLOCK 7
        FMLA        v25.4s, v19.4s, v2.s[3]
        FMLA        v28.4s, v19.4s, v3.s[1]
        ADD         x5, x5, 96
        FMLA        v31.4s, v19.4s, v3.s[3]

        # Is there a remainder?- 2 floats of A (8 bytes) or less
        B.NE        5f

4:
        # ks loop
        SUBS        x9, x9, 32              // ks -= MR * sizeof(void*)
        B.HI        1b

        # Clamp
        FMAX        v20.4s, v20.4s, v4.4s
        # Load cn_stride
        LDR         x0, [sp, 48]
        FMAX        v21.4s, v21.4s, v4.4s
        FMAX        v22.4s, v22.4s, v4.4s
        FMAX        v23.4s, v23.4s, v4.4s
        FMAX        v24.4s, v24.4s, v4.4s
        FMAX        v25.4s, v25.4s, v4.4s
        FMAX        v26.4s, v26.4s, v4.4s
        FMAX        v27.4s, v27.4s, v4.4s
        FMAX        v28.4s, v28.4s, v4.4s
        FMAX        v29.4s, v29.4s, v4.4s
        FMAX        v30.4s, v30.4s, v4.4s
        FMAX        v31.4s, v31.4s, v4.4s
        SUBS        x1, x1, 12
        FMIN        v20.4s, v20.4s, v5.4s
        FMIN        v21.4s, v21.4s, v5.4s
        FMIN        v22.4s, v22.4s, v5.4s
        FMIN        v23.4s, v23.4s, v5.4s
        FMIN        v24.4s, v24.4s, v5.4s
        FMIN        v25.4s, v25.4s, v5.4s
        FMIN        v26.4s, v26.4s, v5.4s
        FMIN        v27.4s, v27.4s, v5.4s
        FMIN        v28.4s, v28.4s, v5.4s
        FMIN        v29.4s, v29.4s, v5.4s
        FMIN        v30.4s, v30.4s, v5.4s
        FMIN        v31.4s, v31.4s, v5.4s

        # Store full 4 x 12
        B.LO        7f

        ST1         {v29.16b, v30.16b, v31.16b},  [x7], x0
        ST1         {v26.16b, v27.16b, v28.16b}, [x10], x0
        ST1         {v23.16b, v24.16b, v25.16b}, [x17], x0
        ST1         {v20.16b, v21.16b, v22.16b},  [x6], x0
        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b

        # Restore d8-d11,d14,d15 from stack
        LDP         d14, d15, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8,  d9, [sp], 48
        RET

5:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBZ         x0, 3, 6f

        # Remainder- 2 floats of A (8 bytes)
        LDR         d0,  [x13], 8           // a0
        LD1         {v6.16b, v7.16b, v8.16b}, [x5], 48
        LDR         d1, [x14], 8            // a1
        LDR         d2, [x15], 8            // a2
        LDR         d3,  [x16], 8           // a3
        LD1         {v9.16b, v10.16b, v11.16b}, [x5], 48

        # First block of 3 B
        FMLA        v20.4s, v6.4s, v0.s[0]
        FMLA        v23.4s, v6.4s, v1.s[0]
        FMLA        v26.4s, v6.4s, v2.s[0]
        FMLA        v29.4s, v6.4s, v3.s[0]
        FMLA        v21.4s, v7.4s, v0.s[0]
        FMLA        v24.4s, v7.4s, v1.s[0]
        FMLA        v27.4s, v7.4s, v2.s[0]
        FMLA        v30.4s, v7.4s, v3.s[0]
        FMLA        v22.4s, v8.4s, v0.s[0]
        FMLA        v25.4s, v8.4s, v1.s[0]
        FMLA        v28.4s, v8.4s, v2.s[0]
        FMLA        v31.4s, v8.4s, v3.s[0]

        # Second block of 3 B
        FMLA        v20.4s, v9.4s, v0.s[1]
        FMLA        v23.4s, v9.4s, v1.s[1]
        FMLA        v26.4s, v9.4s, v2.s[1]
        FMLA        v29.4s, v9.4s, v3.s[1]
        FMLA        v21.4s, v10.4s, v0.s[1]
        FMLA        v24.4s, v10.4s, v1.s[1]
        FMLA        v27.4s, v10.4s, v2.s[1]
        FMLA        v30.4s, v10.4s, v3.s[1]
        FMLA        v22.4s, v11.4s, v0.s[1]
        FMLA        v25.4s, v11.4s, v1.s[1]
        FMLA        v28.4s, v11.4s, v2.s[1]
        FMLA        v31.4s, v11.4s, v3.s[1]

        # Is there a remainder?- 1 float of A (4 bytes)
        TBZ         x0, 2, 4b
6:
        # Remainder- 1 float of A (4 bytes)
        LDR         s0,  [x13], 4           // a0
        LD1         {v6.16b, v7.16b, v8.16b}, [x5], 48
        LDR         s1, [x14], 4            // a1
        LDR         s2, [x15], 4            // a2
        LDR         s3,  [x16], 4           // a3

        FMLA        v20.4s, v6.4s, v0.s[0]
        FMLA        v23.4s, v6.4s, v1.s[0]
        FMLA        v26.4s, v6.4s, v2.s[0]
        FMLA        v29.4s, v6.4s, v3.s[0]
        FMLA        v21.4s, v7.4s, v0.s[0]
        FMLA        v24.4s, v7.4s, v1.s[0]
        FMLA        v27.4s, v7.4s, v2.s[0]
        FMLA        v30.4s, v7.4s, v3.s[0]
        FMLA        v22.4s, v8.4s, v0.s[0]
        FMLA        v25.4s, v8.4s, v1.s[0]
        FMLA        v28.4s, v8.4s, v2.s[0]
        FMLA        v31.4s, v8.4s, v3.s[0]
        B           4b

7:
        ADD         x1, x1, 12
        # Store odd channels
        TBZ         x1, 3, 8f
        STP         q29, q30,  [x7], 32
        MOV         v29.16b, v31.16b
        STP         q26, q27, [x10], 32
        MOV         v26.16b, v28.16b
        STP         q23, q24, [x17], 32
        MOV         v23.16b, v25.16b
        STP         q20, q21,  [x6], 32
        MOV         v20.16b, v22.16b

8:
        TBZ         x1, 2, 9f
        STR         q29,  [x7], 16
        MOV         v29.16b, v30.16b
        STR         q26, [x10], 16
        MOV         v26.16b, v27.16b
        STR         q23,  [x17], 16
        MOV         v23.16b, v24.16b
        STR         q20,  [x6], 16
        MOV         v20.16b, v21.16b

9:
        TBZ         x1, 1, 10f
        STR         d29,  [x7], 8
        DUP         d29, v29.d[1]
        STR         d26, [x10], 8
        DUP         d26, v26.d[1]
        STR         d23, [x17], 8
        DUP         d23, v23.d[1]
        STR         d20,  [x6], 8
        DUP         d20, v20.d[1]

10:
        TBZ         x1, 0, 11f
        STR         s29,  [x7]
        STR         s26, [x10]
        STR         s23, [x17]
        STR         s20,  [x6]
11:
        # Restore d8-d11,d14,d15 from stack
        LDP         d14, d15, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8,  d9, [sp], 48
        RET

END_FUNCTION xnn_f32_igemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
