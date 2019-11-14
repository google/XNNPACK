// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_igemm_ukernel_4x12__aarch64_neonfma_cortex_a53(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float**restrict a,           x4
#     const float*restrict w,            x5
#     float*restrict c,                  x6
#     size_t cm_stride,                  x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_output_params params [sp + 24] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointers
# x22 a0
# x23 a1
# x24 a2
# x25 a3

# C pointers
# x6  c0
# x20 c1
# x21 c2
# x7  c3 / cm_stride

# Vector register usage and GPR shadows
# A0  v0           first set of A
# A1  v0[1]        x13
# A2  v1
# A3  v1[1]        x8
# A0  v2           second set of A
# A1  v2[1]        x13
# A2  v3
# A3  v3[1]        x8
# B   v4  v5  v6   x14 x15 x16  first set of B
# B   v7  v8  v9   x17 x18 x19
# B  v14 v15 v16   x14 x15 x16  second set of B (same x as first set)
# B  v17 v18 v19   x17 x18 x19
# C  v20 v21 v22
# C  v23 v24 v25
# C  v26 v27 v28
# C  v29 v30 v31
# Clamp v10 v11
# v12 to v13 unused.

BEGIN_FUNCTION xnn_f32_igemm_ukernel_4x12__aarch64_neonfma_cortex_a53

        # Load cn_stride, a_offset
        LDP x10, x11, [sp]

        # Load zero, clamping params pointer
        LDP x12, x8, [sp, 16]

        # Save x19-x25 on stack
        STR x19, [sp, -112]!
        STP x20, x21, [sp, 16]
        STP x22, x23, [sp, 32]
        STP x24, x25, [sp, 48]

        # Save d8-d11,d14,d15 on stack
        STP  d8,  d9, [sp, 64]
        STP d10, d11, [sp, 80]
        STP d14, d15, [sp, 96]

        # Load clamping_params values
        LD2R {v10.4s, v11.4s}, [x8]

        # Clamp C pointers
        ADD x20, x6, x7          // c1 = c0 + cm_stride

        CMP x0, 2                // if mr < 2
        CSEL x20, x6, x20, LO    //   c1 = c0

        ADD x21, x20, x7         // c2 = c1 + cm_stride
                                 // if mr <= 2

        CSEL x21, x20, x21, LS   //   c2 = c1

        ADD x7, x21, x7          // c3 = c2 + cm_stride
        CMP x0, 4                // if mr < 4
        CSEL x7, x21, x7, LO     //   c3 = c2

0:
        # Load initial bias from w into accumulators
        LD1 {v20.16b, v21.16b, v22.16b}, [x5], 48
        MOV v23.16b, v20.16b
        PRFM PLDL1KEEP, [x5]
        MOV v24.16b, v21.16b
        PRFM PLDL1KEEP, [x5, 64]
        MOV v25.16b, v22.16b
        PRFM PLDL1KEEP, [x5, 128]
        MOV v26.16b, v20.16b
        PRFM PLDL1KEEP, [x5, 192]
        MOV v27.16b, v21.16b
        PRFM PLDL1KEEP, [x5, 256]
        MOV v28.16b, v22.16b
        PRFM PLDL1KEEP, [x5, 320]
        MOV v29.16b, v20.16b
        MOV v30.16b, v21.16b
        MOV v31.16b, v22.16b

        MOV x9, x3  // p = ks

1:
        # Load next 4 A pointers
        LDP x22, x23, [x4], 16
        LDP x24, x25, [x4], 16

        CMP x22, x12            // if a0 == zero
        ADD x22, x22, x11       // a0 += a_offset
        CSEL x22, x12, x22, EQ  //   a0 = zero, else += a0 + a_offset
        CMP x23, x12            // if a1 == zero
        ADD x23, x23, x11       // a1 += a_offset
        CSEL x23, x12, x23, EQ  //   a1 = zero, else += a1 + a_offset
        CMP x24, x12            // if a2 == zero
        ADD x24, x24, x11       // a2 += a_offset
        CSEL x24, x12, x24, EQ  //   a2 = zero, else += a2 + a_offset
        CMP x25, x12            // if a3 == zero
        ADD x25, x25, x11       // a3 += a_offset
        CSEL x25, x12, x25, EQ  //   a3 = zero, else += a3 + a_offset

        # Is there at least 4 floats (16 bytes) for prologue + epilogue?
        SUBS x0, x2, 16  // k = kc - 16
        B.LO 4f

        SUBS x0, x0, 16  // 4 floats for main loop

        # Prologue - loads for first group of 24 FMA

        # Read first block of 4 A.
        LDR d0, [x22], 8   // a0
        LDR x13, [x23], 8  // a1
        LDR d1, [x24], 8   // a2
        LDR x8, [x25], 8   // a3

        LDR d4, [x5]       // vb0x0123
        LDR x14, [x5, 8]

        LDR d5, [x5, 16]   // vb0x25567
        LDR x15, [x5, 24]

        LDR d6, [x5, 32]   // vb0x89AB
        LDR x16, [x5, 40]

        LDR d7, [x5, 48]   // vb1x0123
        INS v0.d[1], x13
        LDR x17, [x5, 56]

        LDR d8, [x5, 64]   // vb1x25567
        INS v1.d[1], x8
        LDR x18, [x5, 72]

        LDR d9, [x5, 80]   // vb1x89AB
        LDR x19, [x5, 88]
        INS v4.d[1], x14
        ADD x5, x5, 96

        # Is there at least 4 floats (16 bytes) for main loop?
        B.LO 3f

        # Main loop - 4 floats of A (16 bytes)
2:
        # First group of 24 fma.  8 blocks of 4 cycles.  LDR + 3 FMA
        # A is loaded for 2nd group into v2/v3
        # INS is 4 blocks (16 cycles) after load

        # BLOCK 0
        LDR d2, [x22], 8         // a0
        INS v5.d[1], x15
        FMLA v20.4s, v4.4s, v0.s[0]
        LDR x13, [x23], 8        // a1
        FMLA v23.4s, v4.4s, v0.s[2]
        PRFM PLDL1KEEP, [x5, 192]
        FMLA v26.4s, v4.4s, v1.s[0]

        # BLOCK 1
        LDR d3, [x24], 8         // a2
        INS v6.d[1], x16
        FMLA v29.4s, v4.4s, v1.s[2]
        LDR x8, [x25], 8         // a3
        FMLA v21.4s, v5.4s, v0.s[0]
        PRFM PLDL1KEEP, [x5, 256]
        FMLA v24.4s, v5.4s, v0.s[2]

        # BLOCK 2
        LDR d14, [x5]       // vb0x0123
        INS v7.d[1], x17
        FMLA v27.4s, v5.4s, v1.s[0]
        LDR x14, [x5, 8]
        FMLA v30.4s, v5.4s, v1.s[2]
        PRFM PLDL1KEEP, [x5, 320]
        FMLA v22.4s, v6.4s, v0.s[0]

        # BLOCK 3
        LDR d15, [x5, 16]   // vb0x25567
        INS v8.d[1], x18
        FMLA v25.4s, v6.4s, v0.s[2]
        LDR x15, [x5, 24]
        FMLA v28.4s, v6.4s, v1.s[0]
        FMLA v31.4s, v6.4s, v1.s[2]

        # BLOCK 4
        LDR d16, [x5, 32]   // vb0x89AB
        INS v9.d[1], x19
        FMLA v20.4s, v7.4s, v0.s[1]
        LDR x16, [x5, 40]
        FMLA v23.4s, v7.4s, v0.s[3]
        FMLA v26.4s, v7.4s, v1.s[1]

        # BLOCK 5
        LDR d17, [x5, 48]   // vb1x0123
        INS v2.d[1], x13    // a1 was loaded in block 0
        FMLA v29.4s, v7.4s, v1.s[3]
        LDR x17, [x5, 56]
        FMLA v21.4s, v8.4s, v0.s[1]
        FMLA v24.4s, v8.4s, v0.s[3]

        # BLOCK 6
        LDR d18, [x5, 64]   // vb1x25567
        INS v3.d[1], x8     // a3 was loaded in block 1
        FMLA v27.4s, v8.4s, v1.s[1]
        LDR x18, [x5, 72]
        FMLA v30.4s, v8.4s, v1.s[3]
        FMLA v22.4s, v9.4s, v0.s[1]

        # BLOCK 7
        LDR d19, [x5, 80]   // vb1x89AB
        INS v14.d[1], x14   // v14 was loaded in block 2
        FMLA v25.4s, v9.4s, v0.s[3]
        LDR x19, [x5, 88]
        FMLA v28.4s, v9.4s, v1.s[1]
        FMLA v31.4s, v9.4s, v1.s[3]

        # Second group of 24 fma.  8 blocks of 4 cycles.  LDR + 3 FMA
        # A is loaded for 1st group into v0/v1

        # BLOCK 0
        LDR d0, [x22], 8         // a0
        INS v15.d[1], x15
        FMLA v20.4s, v14.4s, v2.s[0]
        LDR x13, [x23], 8        // a1
        FMLA v23.4s, v14.4s, v2.s[2]
        FMLA v26.4s, v14.4s, v3.s[0]

        # BLOCK 1
        LDR d1, [x24], 8         // a2
        INS v16.d[1], x16
        FMLA v29.4s, v14.4s, v3.s[2]
        LDR x8, [x25], 8        // a3
        FMLA v21.4s, v15.4s, v2.s[0]
        FMLA v24.4s, v15.4s, v2.s[2]

        # BLOCK 2
        LDR d4, [x5, 96]        // vb0x0123
        INS v17.d[1], x17
        FMLA v27.4s, v15.4s, v3.s[0]
        LDR x14, [x5, 104]
        FMLA v30.4s, v15.4s, v3.s[2]
        FMLA v22.4s, v16.4s, v2.s[0]

        # BLOCK 3
        LDR d5, [x5, 112]       // vb0x25567
        INS v18.d[1], x18
        FMLA v25.4s, v16.4s, v2.s[2]
        LDR x15, [x5, 120]
        FMLA v28.4s, v16.4s, v3.s[0]
        FMLA v31.4s, v16.4s, v3.s[2]

        # BLOCK 4
        LDR d6, [x5, 128]       // vb0x89AB
        INS v19.d[1], x19
        FMLA v20.4s, v17.4s, v2.s[1]
        LDR x16, [x5, 136]
        FMLA v23.4s, v17.4s, v2.s[3]
        FMLA v26.4s, v17.4s, v3.s[1]

        # BLOCK 5
        LDR d7, [x5, 144]       // vb1x0123
        INS v0.d[1], x13        // a1
        FMLA v29.4s, v17.4s, v3.s[3]
        LDR x17, [x5, 152]
        FMLA v21.4s, v18.4s, v2.s[1]
        FMLA v24.4s, v18.4s, v2.s[3]

        # BLOCK 6
        LDR d8, [x5, 160]       // vb1x25567
        INS v1.d[1], x8         // a3
        FMLA v27.4s, v18.4s, v3.s[1]
        LDR x18, [x5, 168]
        FMLA v30.4s, v18.4s, v3.s[3]
        SUBS x0, x0, 16
        FMLA v22.4s, v19.4s, v2.s[1]

        # BLOCK 7
        LDR d9, [x5, 176]       // vb1x89AB
        INS v4.d[1], x14
        FMLA v25.4s, v19.4s, v2.s[3]
        LDR x19, [x5, 184]
        FMLA v28.4s, v19.4s, v3.s[1]
        ADD x5, x5, 192
        FMLA v31.4s, v19.4s, v3.s[3]
        B.HS 2b

        # Epilogue
        # First block same as main loop.  Second block has no loads.
3:
        # BLOCK 0
        LDR d2, [x22], 8         // a0
        INS v5.d[1], x15
        FMLA v20.4s, v4.4s, v0.s[0]
        LDR x13, [x23], 8        // a1
        FMLA v23.4s, v4.4s, v0.s[2]
        PRFM PLDL1KEEP, [x5, 192]
        FMLA v26.4s, v4.4s, v1.s[0]

        # BLOCK 1
        LDR d3, [x24], 8         // a2
        INS v6.d[1], x16
        FMLA v29.4s, v4.4s, v1.s[2]
        LDR x8, [x25], 8         // a3
        FMLA v21.4s, v5.4s, v0.s[0]
        PRFM PLDL1KEEP, [x5, 256]
        FMLA v24.4s, v5.4s, v0.s[2]

        # BLOCK 2
        LDR d14, [x5]          // vb0x0123
        INS v7.d[1], x17
        FMLA v27.4s, v5.4s, v1.s[0]
        LDR x14, [x5, 8]
        FMLA v30.4s, v5.4s, v1.s[2]
        PRFM PLDL1KEEP, [x5, 320]
        FMLA v22.4s, v6.4s, v0.s[0]

        # BLOCK 3
        LDR d15, [x5, 16]      // vb0x25567
        INS v8.d[1], x18
        FMLA v25.4s, v6.4s, v0.s[2]
        LDR x15, [x5, 24]
        FMLA v28.4s, v6.4s, v1.s[0]
        FMLA v31.4s, v6.4s, v1.s[2]

        # BLOCK 4
        LDR d16, [x5, 32]      // vb0x89AB
        INS v9.d[1], x19
        FMLA v20.4s, v7.4s, v0.s[1]
        LDR x16, [x5, 40]
        FMLA v23.4s, v7.4s, v0.s[3]
        FMLA v26.4s, v7.4s, v1.s[1]

        # BLOCK 5
        LDR d17, [x5, 48]      // vb1x0123
        INS v2.d[1], x13       // a1 was loaded in block 0
        FMLA v29.4s, v7.4s, v1.s[3]
        LDR x17, [x5, 56]
        FMLA v21.4s, v8.4s, v0.s[1]
        FMLA v24.4s, v8.4s, v0.s[3]

        # BLOCK 6
        LDR d18, [x5, 64]      // vb1x25567
        INS v3.d[1], x8        // a3 was loaded in block 1
        FMLA v27.4s, v8.4s, v1.s[1]
        LDR x18, [x5, 72]
        FMLA v30.4s, v8.4s, v1.s[3]
        FMLA v22.4s, v9.4s, v0.s[1]

        # BLOCK 7
        LDR d19, [x5, 80]      // vb1x89AB
        INS v14.d[1], x14      // v14 was loaded in block 2
        FMLA v25.4s, v9.4s, v0.s[3]
        LDR x19, [x5, 88]
        FMLA v28.4s, v9.4s, v1.s[1]
        ADD x5, x5, 96
        FMLA v31.4s, v9.4s, v1.s[3]

        # Second group of 24 fma.  8 blocks of 4 cycles.
        # Epilogue version does no loads

        # BLOCK 0
        INS v15.d[1], x15
        FMLA v20.4s, v14.4s, v2.s[0]
        FMLA v23.4s, v14.4s, v2.s[2]
        FMLA v26.4s, v14.4s, v3.s[0]

        # BLOCK 1
        INS v16.d[1], x16
        FMLA v29.4s, v14.4s, v3.s[2]
        FMLA v21.4s, v15.4s, v2.s[0]
        FMLA v24.4s, v15.4s, v2.s[2]

        # BLOCK 2
        INS v17.d[1], x17
        FMLA v27.4s, v15.4s, v3.s[0]
        FMLA v30.4s, v15.4s, v3.s[2]
        FMLA v22.4s, v16.4s, v2.s[0]

        # BLOCK 3
        INS v18.d[1], x18
        FMLA v25.4s, v16.4s, v2.s[2]
        FMLA v28.4s, v16.4s, v3.s[0]
        FMLA v31.4s, v16.4s, v3.s[2]

        # BLOCK 4
        INS v19.d[1], x19
        FMLA v20.4s, v17.4s, v2.s[1]
        FMLA v23.4s, v17.4s, v2.s[3]
        FMLA v26.4s, v17.4s, v3.s[1]

        # BLOCK 5
        FMLA v29.4s, v17.4s, v3.s[3]
        FMLA v21.4s, v18.4s, v2.s[1]
        FMLA v24.4s, v18.4s, v2.s[3]

        # BLOCK 6
        FMLA v27.4s, v18.4s, v3.s[1]
        FMLA v30.4s, v18.4s, v3.s[3]
        FMLA v22.4s, v19.4s, v2.s[1]

        # BLOCK 7
        FMLA v25.4s, v19.4s, v2.s[3]
        FMLA v28.4s, v19.4s, v3.s[1]
        FMLA v31.4s, v19.4s, v3.s[3]

4:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBNZ x0, 3, 6f
        # Is there a remainder?- 1 floats of A (4 bytes)
        TBNZ x0, 2, 7f

5:
        # ks loop
        SUBS x9, x9, 32  // ks -= MR * sizeof(void*)
        B.NE 1b

        # Clamp
        FMIN v20.4s, v20.4s, v10.4s
        FMIN v21.4s, v21.4s, v10.4s
        FMIN v22.4s, v22.4s, v10.4s
        FMIN v23.4s, v23.4s, v10.4s
        FMIN v24.4s, v24.4s, v10.4s
        FMIN v25.4s, v25.4s, v10.4s
        FMIN v26.4s, v26.4s, v10.4s
        FMIN v27.4s, v27.4s, v10.4s
        FMIN v28.4s, v28.4s, v10.4s
        FMIN v29.4s, v29.4s, v10.4s
        FMIN v30.4s, v30.4s, v10.4s
        FMIN v31.4s, v31.4s, v10.4s
        FMAX v20.4s, v20.4s, v11.4s
        FMAX v21.4s, v21.4s, v11.4s
        FMAX v22.4s, v22.4s, v11.4s
        FMAX v23.4s, v23.4s, v11.4s
        FMAX v24.4s, v24.4s, v11.4s
        FMAX v25.4s, v25.4s, v11.4s
        FMAX v26.4s, v26.4s, v11.4s
        FMAX v27.4s, v27.4s, v11.4s
        FMAX v28.4s, v28.4s, v11.4s
        FMAX v29.4s, v29.4s, v11.4s
        FMAX v30.4s, v30.4s, v11.4s
        FMAX v31.4s, v31.4s, v11.4s

        # Store full 4 x 12
        CMP x1, 12
        B.LO 8f

        ST1 {v29.16b, v30.16b, v31.16b},  [x7], x10
        ST1 {v26.16b, v27.16b, v28.16b}, [x21], x10
        ST1 {v23.16b, v24.16b, v25.16b}, [x20], x10
        ST1 {v20.16b, v21.16b, v22.16b},  [x6], x10
        SUB x4, x4, x3  // a -= ks

        # nc loop
        SUBS x1, x1, 12
        B.HI 0b

        # Restore d8-d11,d14,d15 from stack
        LDP d14, d15, [sp, 96]
        LDP d10, d11, [sp, 80]
        LDP  d8,  d9, [sp, 64]

        # Restore x19-x25 from stack
        LDP x24, x25, [sp, 48]
        LDP x22, x23, [sp, 32]
        LDP x20, x21, [sp, 16]
        LDR x19, [sp], 112
        RET

6:
        # Remainder - 2 floats of A (8 bytes)
        # Read first block of 4 A.
        LDR d0, [x22], 8  // a0
        LDR d1, [x23], 8  // a1
        LDR d2, [x24], 8  // a2
        LDR d3, [x25], 8  // a3
        LD1 {v4.16b, v5.16b, v6.16b}, [x5], 48
        LD1 {v7.16b, v8.16b, v9.16b}, [x5], 48

        # First block of 3 B
        FMLA v20.4s, v4.4s, v0.s[0]
        FMLA v23.4s, v4.4s, v1.s[0]
        FMLA v26.4s, v4.4s, v2.s[0]
        FMLA v29.4s, v4.4s, v3.s[0]
        FMLA v21.4s, v5.4s, v0.s[0]
        FMLA v24.4s, v5.4s, v1.s[0]
        FMLA v27.4s, v5.4s, v2.s[0]
        FMLA v30.4s, v5.4s, v3.s[0]
        FMLA v22.4s, v6.4s, v0.s[0]
        FMLA v25.4s, v6.4s, v1.s[0]
        FMLA v28.4s, v6.4s, v2.s[0]
        FMLA v31.4s, v6.4s, v3.s[0]

        # Second block of 3 B
        FMLA v20.4s, v7.4s, v0.s[1]
        FMLA v23.4s, v7.4s, v1.s[1]
        FMLA v26.4s, v7.4s, v2.s[1]
        FMLA v29.4s, v7.4s, v3.s[1]
        FMLA v21.4s, v8.4s, v0.s[1]
        FMLA v24.4s, v8.4s, v1.s[1]
        FMLA v27.4s, v8.4s, v2.s[1]
        FMLA v30.4s, v8.4s, v3.s[1]
        FMLA v22.4s, v9.4s, v0.s[1]
        FMLA v25.4s, v9.4s, v1.s[1]
        FMLA v28.4s, v9.4s, v2.s[1]
        FMLA v31.4s, v9.4s, v3.s[1]

        TBZ x0, 2, 5b
7:
        # Remainder - 1 float of A (4 bytes)
        LDR s0, [x22], 4  // a0
        LDR s1, [x23], 4  // a1
        LDR s2, [x24], 4  // a2
        LDR s3, [x25], 4  // a3
        LD1 {v4.16b, v5.16b, v6.16b}, [x5], 48

        FMLA v20.4s, v4.4s, v0.s[0]
        FMLA v23.4s, v4.4s, v1.s[0]
        FMLA v26.4s, v4.4s, v2.s[0]
        FMLA v29.4s, v4.4s, v3.s[0]
        FMLA v21.4s, v5.4s, v0.s[0]
        FMLA v24.4s, v5.4s, v1.s[0]
        FMLA v27.4s, v5.4s, v2.s[0]
        FMLA v30.4s, v5.4s, v3.s[0]
        FMLA v22.4s, v6.4s, v0.s[0]
        FMLA v25.4s, v6.4s, v1.s[0]
        FMLA v28.4s, v6.4s, v2.s[0]
        FMLA v31.4s, v6.4s, v3.s[0]
        B 5b

8:
        # Store odd channels
        TBZ x1, 3, 9f
        STP q29, q30,  [x7], 32
        MOV v29.16b, v31.16b
        STP q26, q27, [x21], 32
        MOV v26.16b, v28.16b
        STP q23, q24, [x20], 32
        MOV v23.16b, v25.16b
        STP q20, q21,  [x6], 32
        MOV v20.16b, v22.16b

9:
        TBZ x1, 2, 10f
        STR q29,  [x7], 16
        MOV v29.16b, v30.16b
        STR q26, [x21], 16
        MOV v26.16b, v27.16b
        STR q23,  [x20], 16
        MOV v23.16b, v24.16b
        STR q20,  [x6], 16
        MOV v20.16b, v21.16b

10:
        TBZ x1, 1, 11f
        STR d29,  [x7], 8
        DUP d29, v29.d[1]
        STR d26, [x21], 8
        DUP d26, v26.d[1]
        STR d23, [x20], 8
        DUP d23, v23.d[1]
        STR d20,  [x6], 8
        DUP d20, v20.d[1]

11:
        TBZ x1, 0, 12f
        STR s29,  [x7]
        STR s26, [x21]
        STR s23, [x20]
        STR s20,  [x6]
12:
        # Restore d8-d11,d14,d15 from stack
        LDP d14, d15, [sp, 96]
        LDP d10, d11, [sp, 80]
        LDP  d8,  d9, [sp, 64]

        # Restore x19-x25 from stack
        LDP x24, x25, [sp, 48]
        LDP x22, x23, [sp, 32]
        LDP x20, x21, [sp, 16]
        LDR x19, [sp], 112
        RET

END_FUNCTION xnn_f32_igemm_ukernel_4x12__aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
