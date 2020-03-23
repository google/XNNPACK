// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/1x12-aarch64-neonfma-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_gemminc_ukernel_1x12__aarch64_neonfma_cortex_a53(
#     size_t mr,                (x0) - unused.  mr = 1
#     size_t nc,                x1
#     size_t kc,                x2 / x0
#     const uint8_t*restrict a, x3
#     size_t a_stride,          (x4) - unused
#     const void*restrict w,    x5
#     uint8_t*restrict c,       x6
#     size_t cm_stride,         (x7) - unused
#     size_t cn_stride,         [sp] -> x14
#     const float*restrict acc,  [sp + 8] -> x15
#     const union xnn_f32_output_params params[restrict static 1])  [sp + 16] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointer
# x3  a0

# C pointer
# x6  c0

# Vector register usage and GPR shadows
# a0  v0           first set of A
# a0  v1           second set of A
# B   v2  v3  v4    x7 x10 x16  first set of B
# B   v5  v6  v7   x17 x13  x9
# B  v23 v24 v25    x7 x10 x16  second set of B (same x as first set)
# B  v17 v18 v19   x17 x13  x9
# C  v20 v21 v22

BEGIN_FUNCTION xnn_f32_gemminc_ukernel_1x12__aarch64_neonfma_cortex_a53

        # Load cn_stride, acc
        LDP x14, x15, [sp]
        # Load params pointer
        LDR x8, [sp, 16]

        # Load clamping_params values
        LD2R {v30.4s, v31.4s}, [x8]

0:
        # Load initial accumulators
        LD1 {v20.16b, v21.16b, v22.16b}, [x15], 48

        PRFM PLDL1KEEP, [x5]
        PRFM PLDL1KEEP, [x5, 64]
        PRFM PLDL1KEEP, [x5, 128]

        # Is there at least 4 floats (16 bytes) for prologue + epilogue?
        SUBS x0, x2, 16  // k = kc - 16
        B.LO 5f

        # Prologue - loads for first group of 6 fma

        # Read first block of 1 A.
        LDR d0, [x3], 8     // a0

        LDR d2, [x5]       // vb0x0123
        LDR x7, [x5, 8]

        LDR d3, [x5, 16]   // vb0x4567
        LDR x10, [x5, 24]

        LDR d4, [x5, 32]   // vb0x89AB
        LDR x16, [x5, 40]

        LDR d5, [x5, 48]   // vb1x0123
        LDR x17, [x5, 56]

        LDR d6, [x5, 64]   // vb1x4567
        LDR x13, [x5, 72]

        LDR d7, [x5, 80]   // vb1x89AB
        LDR x9, [x5, 88]
        INS v2.d[1], x7
        ADD x5, x5, 96

        # Is there at least 4 floats (16 bytes) for main loop?
        SUBS x0, x0, 16
        B.LO 2f

        # Main loop - 4 floats of A (16 bytes)
1:
        # First group of 6 fma.
        # A is loaded for 2nd group into v1

        # BLOCK 0
        LDR d1, [x3], 8          // a0
        INS v3.d[1], x10
        FMLA v20.4s, v2.4s, v0.s[0]
        PRFM PLDL1KEEP, [x5, 96]

        # BLOCK 1
        INS v4.d[1], x16
        FMLA v21.4s, v3.4s, v0.s[0]
        PRFM PLDL1KEEP, [x5, 128]

        # BLOCK 2
        LDR d23, [x5]       // vb0x0123
        INS v5.d[1], x17
        LDR x7, [x5, 8]
        FMLA v22.4s, v4.4s, v0.s[0]

        # BLOCK 3
        LDR d24, [x5, 16]   // vb0x4567
        INS v6.d[1], x13
        LDR x10, [x5, 24]

        # BLOCK 4
        LDR d25, [x5, 32]   // vb0x89AB
        INS v7.d[1], x9
        FMLA v20.4s, v5.4s, v0.s[1]
        LDR x16, [x5, 40]

        # BLOCK 5
        LDR d17, [x5, 48]   // vb1x0123
        LDR x17, [x5, 56]
        FMLA v21.4s, v6.4s, v0.s[1]

        # BLOCK 6
        LDR d18, [x5, 64]   // vb1x4567
        LDR x13, [x5, 72]
        FMLA v22.4s, v7.4s, v0.s[1]

        # BLOCK 7
        LDR d19, [x5, 80]   // vb1x89AB
        INS v23.d[1], x7   // v23 was loaded in block 2
        LDR x9, [x5, 88]

        # Second group of 6 fma.
        # A is loaded for 1st group into v0

        # BLOCK 0
        LDR d0, [x3], 8          // a0
        INS v24.d[1], x10
        FMLA v20.4s, v23.4s, v1.s[0]

        # BLOCK 1
        INS v25.d[1], x16
        FMLA v21.4s, v24.4s, v1.s[0]

        # BLOCK 2
        LDR d2, [x5, 96]        // vb0x0123
        INS v17.d[1], x17
        LDR x7, [x5, 104]
        FMLA v22.4s, v25.4s, v1.s[0]

        # BLOCK 3
        LDR d3, [x5, 112]    // vb0x4567
        INS v18.d[1], x13
        LDR x10, [x5, 120]

        # BLOCK 4
        LDR d4, [x5, 128]   // vb0x89AB
        INS v19.d[1], x9
        FMLA v20.4s, v17.4s, v1.s[1]
        LDR x16, [x5, 136]

        # BLOCK 5
        LDR d5, [x5, 144]   // vb1x0123
        LDR x17, [x5, 152]
        FMLA v21.4s, v18.4s, v1.s[1]

        # BLOCK 6
        LDR d6, [x5, 160]   // vb1x4567
        LDR x13, [x5, 168]
        SUBS x0, x0, 16
        FMLA v22.4s, v19.4s, v1.s[1]

        # BLOCK 7
        LDR d7, [x5, 176]   // vb1x89AB
        INS v2.d[1], x7
        LDR x9, [x5, 184]
        ADD x5, x5, 192
        B.HS 1b

        # Epilogue
        # First block same as main loop.  Second block has no loads.
2:
        # BLOCK 0
        LDR d1, [x3], 8          // a0
        INS v3.d[1], x10
        FMLA v20.4s, v2.4s, v0.s[0]
        PRFM PLDL1KEEP, [x5, 96]

        # BLOCK 1
        INS v4.d[1], x16
        FMLA v21.4s, v3.4s, v0.s[0]
        PRFM PLDL1KEEP, [x5, 128]

        # BLOCK 2
        LDR d23, [x5]       // vb0x0123
        INS v5.d[1], x17
        LDR x7, [x5, 8]
        FMLA v22.4s, v4.4s, v0.s[0]

        # BLOCK 3
        LDR d24, [x5, 16]   // vb0x4567
        INS v6.d[1], x13
        LDR x10, [x5, 24]

        # BLOCK 4
        LDR d25, [x5, 32]   // vb0x89AB
        INS v7.d[1], x9
        FMLA v20.4s, v5.4s, v0.s[1]
        LDR x16, [x5, 40]

        # BLOCK 5
        LDR d17, [x5, 48]   // vb1x0123
        LDR x17, [x5, 56]
        FMLA v21.4s, v6.4s, v0.s[1]

        # BLOCK 6
        LDR d18, [x5, 64]   // vb1x4567
        LDR x13, [x5, 72]
        FMLA v22.4s, v7.4s, v0.s[1]

        # BLOCK 7
        LDR d19, [x5, 80]   // vb1x89AB
        INS v23.d[1], x7   // v23 was loaded in block 2
        LDR x9, [x5, 88]
        ADD x5, x5, 96

        # Second group of 6 fma.  8 blocks of 4 cycles.
        # Epilogue version does no loads

        # BLOCK 0
        INS v24.d[1], x10
        FMLA v20.4s, v23.4s, v1.s[0]

        # BLOCK 1
        INS v25.d[1], x16
        FMLA v21.4s, v24.4s, v1.s[0]

        # BLOCK 2
        INS v17.d[1], x17
        FMLA v22.4s, v25.4s, v1.s[0]

        # BLOCK 3
        INS v18.d[1], x13

        # BLOCK 4
        INS v19.d[1], x9
        FMLA v20.4s, v17.4s, v1.s[1]
        TST x0, 15

        # BLOCK 5
        FMLA v21.4s, v18.4s, v1.s[1]

        # BLOCK 6
        FMLA v22.4s, v19.4s, v1.s[1]

        # BLOCK 7
        # Is there a remainder?- 2 floats of A (8 bytes) or less
        B.NE 5f

4:
        # Clamp
        FMIN v20.4s, v20.4s, v30.4s
        SUBS x1, x1, 12
        FMIN v21.4s, v21.4s, v30.4s
        FMIN v22.4s, v22.4s, v30.4s
        FMAX v20.4s, v20.4s, v31.4s
        FMAX v21.4s, v21.4s, v31.4s
        FMAX v22.4s, v22.4s, v31.4s

        # Store full 1 x 12
        B.LO 7f

        ST1 {v20.16b, v21.16b, v22.16b}, [x6], x14
        SUB  x3,  x3, x2 // a0 -= kc
        B.HI 0b
        RET

5:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBZ x0, 3, 6f

        # Remainder - 2 floats of A (8 bytes)
        # Read first block of 1 A.
        LDR d0, [x3], 8   // a0
        LD1 {v2.16b, v3.16b, v4.16b}, [x5], 48
        LD1 {v5.16b, v6.16b, v7.16b}, [x5], 48

        # First block of 3 B
        FMLA v20.4s, v2.4s, v0.s[0]
        FMLA v21.4s, v3.4s, v0.s[0]
        FMLA v22.4s, v4.4s, v0.s[0]

        # Second block of 3 B
        FMLA v20.4s, v5.4s, v0.s[1]
        FMLA v21.4s, v6.4s, v0.s[1]
        FMLA v22.4s, v7.4s, v0.s[1]

        TBZ x0, 2, 4b
6:
        # Remainder - 1 float of A (4 bytes)
        LDR s0, [x3], 4   // a0
        LD1 {v2.16b, v3.16b, v4.16b}, [x5], 48

        FMLA v20.4s, v2.4s, v0.s[0]
        FMLA v21.4s, v3.4s, v0.s[0]
        FMLA v22.4s, v4.4s, v0.s[0]
        B 4b

7:
        ADD x1, x1, 12
        # Store odd channels
        TBZ x1, 3, 8f
        STP q20, q21, [x6], 32
        MOV v20.16b, v22.16b

8:
        TBZ x1, 2, 9f
        STR q20, [x6], 16
        MOV v20.16b, v21.16b

9:
        TBZ x1, 1, 10f
        STR d20, [x6], 8
        DUP d20, v20.d[1]

10:
        TBZ x1, 0, 11f
        STR s20, [x6]
11:
        RET

END_FUNCTION xnn_f32_gemminc_ukernel_1x12__aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
