// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53(
#     size_t mr,                         (x0) - unused.  mr = 1
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float**restrict a,           x4
#     const float*restrict w,            x5
#     float*restrict c,                  x6
#     size_t cm_stride,                  (x7) - unused
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_output_params params [sp + 24] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointer
# x8  a0

# C pointer
# x6  c0

# Vector register usage and GPR shadows
# a0  v0       first set of A
# a0  v1       second set of A
# B   v2  v3   x14 x15  first set of B
# B   v5  v6   x17 x18
# B  v23 v24   x14 x15  second set of B (same x as first set)
# B  v17 v18   x17 x18
# C  v20 v21

BEGIN_FUNCTION xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53

        # Load cn_stride, a_offset
        LDP x10, x11, [sp]

        # Load zero, clamping params pointer
        LDP x12, x8, [sp, 16]

        # Load clamping_params values
        LD2R {v30.4s, v31.4s}, [x8]

0:
        # Load initial bias from w into accumulators
        LD1 {v20.16b, v21.16b}, [x5], 32

        PRFM PLDL1KEEP, [x5]
        PRFM PLDL1KEEP, [x5, 64]
        PRFM PLDL1KEEP, [x5, 128]
        PRFM PLDL1KEEP, [x5, 192]

        MOV x9, x3  // p = ks

1:
        # Load next A pointer
        LDR x8, [x4], 8

        CMP x8, x12           // if a0 == zero
        ADD x8, x8, x11       // a0 += a_offset
        CSEL x8, x12, x8, EQ  //   a0 = zero, else += a0 + a_offset

        # Is there at least 4 floats (16 bytes) for prologue + epilogue?
        SUBS x0, x2, 16  // k = kc - 16
        B.LO 4f

        # Prologue - loads for first group of 4 fma

        # Read first block of 1 A.
        LDR d0, [x8], 8    // a0

        LDR d2, [x5]       // vb0x0123
        LDR x14, [x5, 8]

        LDR d3, [x5, 16]   // vb0x4567
        LDR x15, [x5, 24]

        LDR d5, [x5, 32]   // vb1x0123
        LDR x17, [x5, 40]

        LDR d6, [x5, 48]   // vb1x4567
        LDR x18, [x5, 56]

        INS v2.d[1], x14
        ADD x5, x5, 64

        # Is there at least 4 floats (16 bytes) for main loop?
        SUBS x0, x0, 16  // 4 floats for main loop
        B.LO 3f

        # Main loop - 4 floats of A (16 bytes)
2:
        # First group of 4 fma.
        # A is loaded for 2nd group into v1

        # BLOCK 0
        LDR d1, [x8], 8    // a0
        INS v3.d[1], x15
        FMLA v20.4s, v2.4s, v0.s[0]
        PRFM PLDL1KEEP, [x5, 128]

        # BLOCK 1
        FMLA v21.4s, v3.4s, v0.s[0]

        # BLOCK 2
        LDR d23, [x5]      // vb0x0123
        INS v5.d[1], x17
        LDR x14, [x5, 8]

        # BLOCK 3
        LDR d24, [x5, 16]  // vb0x4567
        INS v6.d[1], x18
        LDR x15, [x5, 24]

        # BLOCK 4
        FMLA v20.4s, v5.4s, v0.s[1]

        # BLOCK 5
        LDR d17, [x5, 32]  // vb1x0123
        LDR x17, [x5, 40]
        FMLA v21.4s, v6.4s, v0.s[1]

        # BLOCK 6
        LDR d18, [x5, 48]  // vb1x4567
        LDR x18, [x5, 56]

        # BLOCK 7
        INS v23.d[1], x14  // v23 was loaded in block 2

        # Second group of 4 fma.
        # A is loaded for 1st group into v0

        # BLOCK 0
        LDR d0, [x8], 8    // a0
        INS v24.d[1], x15
        FMLA v20.4s, v23.4s, v1.s[0]

        # BLOCK 1
        FMLA v21.4s, v24.4s, v1.s[0]

        # BLOCK 2
        LDR d2, [x5, 64]   // vb0x0123
        INS v17.d[1], x17
        LDR x14, [x5, 72]

        # BLOCK 3
        LDR d3, [x5, 80]  // vb0x4567
        INS v18.d[1], x18
        LDR x15, [x5, 88]

        # BLOCK 4
        FMLA v20.4s, v17.4s, v1.s[1]

        # BLOCK 5
        LDR d5, [x5, 96]  // vb1x0123
        LDR x17, [x5, 104]
        FMLA v21.4s, v18.4s, v1.s[1]

        # BLOCK 6
        LDR d6, [x5, 112]  // vb1x4567
        LDR x18, [x5, 120]
        SUBS x0, x0, 16

        # BLOCK 7
        INS v2.d[1], x14
        ADD x5, x5, 128
        B.HS 2b

        # Epilogue
        # First block same as main loop.  Second block has no loads.
3:
        # BLOCK 0
        LDR d1, [x8], 8    // a0
        INS v3.d[1], x15
        FMLA v20.4s, v2.4s, v0.s[0]

        # BLOCK 1
        FMLA v21.4s, v3.4s, v0.s[0]

        # BLOCK 2
        LDR d23, [x5]      // vb0x0123
        INS v5.d[1], x17
        LDR x14, [x5, 8]

        # BLOCK 3
        LDR d24, [x5, 16]  // vb0x4567
        INS v6.d[1], x18
        LDR x15, [x5, 24]

        # BLOCK 4
        FMLA v20.4s, v5.4s, v0.s[1]

        # BLOCK 5
        LDR d17, [x5, 32]  // vb1x0123
        LDR x17, [x5, 40]
        FMLA v21.4s, v6.4s, v0.s[1]

        # BLOCK 6
        LDR d18, [x5, 48]  // vb1x4567
        LDR x18, [x5, 56]

        # BLOCK 7
        INS v23.d[1], x14  // v23 was loaded in block 2
        ADD x5, x5, 64

        # Second group of 4 fma.
        # Epilogue version does no loads

        # BLOCK 0
        INS v24.d[1], x15
        FMLA v20.4s, v23.4s, v1.s[0]

        # BLOCK 1
        FMLA v21.4s, v24.4s, v1.s[0]

        # BLOCK 2
        INS v17.d[1], x17

        # BLOCK 3
        INS v18.d[1], x18

        # BLOCK 4
        FMLA v20.4s, v17.4s, v1.s[1]

        # BLOCK 5
        FMLA v21.4s, v18.4s, v1.s[1]

        # BLOCK 6

        # BLOCK 7

4:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBNZ x0, 3, 6f
        # Is there a remainder?- 1 floats of A (4 bytes)
        TBNZ x0, 2, 7f

5:
        # ks loop
        SUBS x9, x9, 8  // ks -= MR * sizeof(void*)
        B.NE 1b

        # Clamp
        FMIN v20.4s, v20.4s, v30.4s
        FMIN v21.4s, v21.4s, v30.4s
        FMAX v20.4s, v20.4s, v31.4s
        FMAX v21.4s, v21.4s, v31.4s

        # Store full 1 x 8
        CMP x1, 8
        B.LO 8f

        ST1 {v20.16b, v21.16b}, [x6], x10

        SUB x4, x4, x3  // a -= ks

        # nc loop
        SUBS x1, x1, 8
        B.HI 0b

        RET

6:
        # Remainder - 2 floats of A (8 bytes)
        # Read first block of 1 A.
        LDR d0, [x8], 8  // a0
        LD1 {v2.16b, v3.16b}, [x5], 32
        LD1 {v5.16b, v6.16b}, [x5], 32

        # First block of 2 B
        FMLA v20.4s, v2.4s, v0.s[0]
        FMLA v21.4s, v3.4s, v0.s[0]

        # Second block of 2 B
        FMLA v20.4s, v5.4s, v0.s[1]
        FMLA v21.4s, v6.4s, v0.s[1]

        TBZ x0, 2, 5b
7:
        # Remainder - 1 float of A (4 bytes)
        LDR s0, [x8], 4  // a0
        LD1 {v2.16b, v3.16b}, [x5], 32

        FMLA v20.4s, v2.4s, v0.s[0]
        FMLA v21.4s, v3.4s, v0.s[0]
        B 5b

8:
        # Store odd channels
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

END_FUNCTION xnn_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
