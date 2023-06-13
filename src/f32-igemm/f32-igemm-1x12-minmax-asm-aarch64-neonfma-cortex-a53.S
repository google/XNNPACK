// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_igemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53(
#     size_t mr,                         (x0) - unused.  mr = 1
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float** restrict a,           x4
#     const float* restrict w,            x5
#     float* restrict c,                  x6
#     size_t cm_stride,                  (x7) - unused
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_minmax_params params [sp + 24] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0   x8 v0           first set of A
# A0   x8 v1           second set of A
# B   x14 x15 x16  v2  v3  v4   first set of B
# B   x17 x13  x7  v5  v6  v7
# B   x14 x15 x16 v23 v24 v25   second set of B (same x as first set)
# B   x17 x13  x7 v17 v18 v19
# C0   x6 v20 v21 v22

BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53

        # Load cn_stride, a_offset
        LDP         x10, x11, [sp]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Load min/max values
        LD2R        {v30.4s, v31.4s}, [x8]

0:
        # Load initial bias from w into accumulators
        LD1         {v20.16b, v21.16b, v22.16b}, [x5], 48

        PRFM        PLDL1KEEP, [x5]
        PRFM        PLDL1KEEP, [x5, 64]
        PRFM        PLDL1KEEP, [x5, 128]
        PRFM        PLDL1KEEP, [x5, 192]
        PRFM        PLDL1KEEP, [x5, 256]
        PRFM        PLDL1KEEP, [x5, 320]

        MOV         x9, x3                  // p = ks

1:
        # Load next A pointer
        LDR         x8, [x4], 8

        CMP         x8, x12                 // if a0 == zero
        ADD         x8, x8, x11             // a0 += a_offset
        CSEL        x8, x12, x8, EQ         //   a0 = zero, else += a0 + a_offset

        # Is there at least 4 floats (16 bytes) for prologue + epilogue?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        5f

        # Prologue - loads for first group of 6 fma

        # Read first block of 1 A.
        LDR         d0, [x8], 8             // a0

        LDR         d2, [x5]                // vb0x0123
        LDR         x14, [x5, 8]

        LDR         d3, [x5, 16]            // vb0x25567
        LDR         x15, [x5, 24]

        LDR         d4, [x5, 32]            // vb0x89AB
        LDR         x16, [x5, 40]

        LDR         d5, [x5, 48]            // vb1x0123
        LDR         x17, [x5, 56]

        LDR         d6, [x5, 64]            // vb1x25567
        LDR         x13, [x5, 72]

        LDR         d7, [x5, 80]            // vb1x89AB
        LDR         x7, [x5, 88]
        INS         v2.d[1], x14
        ADD         x5, x5, 96

        # Is there at least 4 floats (16 bytes) for main loop?
        SUBS        x0, x0, 16              // 4 floats for main loop
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
2:
        # First group of 6 fma.
        # A is loaded for 2nd group into v1

        # BLOCK 0
        LDR         d1, [x8], 8             // a0
        INS         v3.d[1], x15
        FMLA        v20.4s, v2.4s, v0.s[0]
        PRFM        PLDL1KEEP, [x5, 192]

        # BLOCK 1
        INS         v4.d[1], x16
        FMLA        v21.4s, v3.4s, v0.s[0]
        PRFM        PLDL1KEEP, [x5, 256]

        # BLOCK 2
        LDR         d23, [x5]               // vb0x0123
        INS         v5.d[1], x17
        LDR         x14, [x5, 8]
        PRFM        PLDL1KEEP, [x5, 320]
        FMLA        v22.4s, v4.4s, v0.s[0]

        # BLOCK 3
        LDR         d24, [x5, 16]           // vb0x25567
        INS         v6.d[1], x13
        LDR         x15, [x5, 24]

        # BLOCK 4
        LDR         d25, [x5, 32]           // vb0x89AB
        INS         v7.d[1], x7
        FMLA        v20.4s, v5.4s, v0.s[1]
        LDR         x16, [x5, 40]

        # BLOCK 5
        LDR         d17, [x5, 48]           // vb1x0123
        LDR         x17, [x5, 56]
        FMLA        v21.4s, v6.4s, v0.s[1]

        # BLOCK 6
        LDR         d18, [x5, 64]           // vb1x25567
        LDR         x13, [x5, 72]
        FMLA        v22.4s, v7.4s, v0.s[1]

        # BLOCK 7
        LDR         d19, [x5, 80]           // vb1x89AB
        INS         v23.d[1], x14           // v23 was loaded in block 2
        LDR         x7, [x5, 88]

        # Second group of 6 fma.
        # A is loaded for 1st group into v0

        # BLOCK 0
        LDR         d0, [x8], 8             // a0
        INS         v24.d[1], x15
        FMLA        v20.4s, v23.4s, v1.s[0]

        # BLOCK 1
        INS         v25.d[1], x16
        FMLA        v21.4s, v24.4s, v1.s[0]

        # BLOCK 2
        LDR         d2, [x5, 96]            // vb0x0123
        INS         v17.d[1], x17
        LDR         x14, [x5, 104]
        FMLA        v22.4s, v25.4s, v1.s[0]

        # BLOCK 3
        LDR         d3, [x5, 112]           // vb0x25567
        INS         v18.d[1], x13
        LDR         x15, [x5, 120]

        # BLOCK 4
        LDR         d4, [x5, 128]           // vb0x89AB
        INS         v19.d[1], x7
        FMLA        v20.4s, v17.4s, v1.s[1]
        LDR         x16, [x5, 136]

        # BLOCK 5
        LDR         d5, [x5, 144]           // vb1x0123
        LDR         x17, [x5, 152]
        FMLA        v21.4s, v18.4s, v1.s[1]

        # BLOCK 6
        LDR         d6, [x5, 160]           // vb1x25567
        LDR         x13, [x5, 168]
        SUBS        x0, x0, 16
        FMLA        v22.4s, v19.4s, v1.s[1]

        # BLOCK 7
        LDR         d7, [x5, 176]           // vb1x89AB
        INS         v2.d[1], x14
        LDR         x7, [x5, 184]
        ADD         x5, x5, 192
        B.HS        2b

        # Epilogue
        # First block same as main loop.  Second block has no loads.
3:
        # BLOCK 0
        LDR         d1, [x8], 8             // a0
        INS         v3.d[1], x15
        FMLA        v20.4s, v2.4s, v0.s[0]
        PRFM        PLDL1KEEP, [x5, 192]

        # BLOCK 1
        INS         v4.d[1], x16
        FMLA        v21.4s, v3.4s, v0.s[0]
        PRFM        PLDL1KEEP, [x5, 256]

        # BLOCK 2
        LDR         d23, [x5]               // vb0x0123
        INS         v5.d[1], x17
        LDR         x14, [x5, 8]
        PRFM        PLDL1KEEP, [x5, 320]
        FMLA        v22.4s, v4.4s, v0.s[0]

        # BLOCK 3
        LDR         d24, [x5, 16]           // vb0x25567
        INS         v6.d[1], x13
        LDR         x15, [x5, 24]

        # BLOCK 4
        LDR         d25, [x5, 32]           // vb0x89AB
        INS         v7.d[1], x7
        FMLA        v20.4s, v5.4s, v0.s[1]
        LDR         x16, [x5, 40]

        # BLOCK 5
        LDR         d17, [x5, 48]           // vb1x0123
        LDR         x17, [x5, 56]
        FMLA        v21.4s, v6.4s, v0.s[1]

        # BLOCK 6
        LDR         d18, [x5, 64]           // vb1x25567
        LDR         x13, [x5, 72]
        FMLA        v22.4s, v7.4s, v0.s[1]

        # BLOCK 7
        LDR         d19, [x5, 80]           // vb1x89AB
        INS         v23.d[1], x14           // v23 was loaded in block 2
        LDR         x7, [x5, 88]
        ADD         x5, x5, 96

        # Second group of 6 fma.  8 blocks of 4 cycles.
        # Epilogue version does no loads

        # BLOCK 0
        INS         v24.d[1], x15
        FMLA        v20.4s, v23.4s, v1.s[0]

        # BLOCK 1
        INS         v25.d[1], x16
        FMLA        v21.4s, v24.4s, v1.s[0]

        # BLOCK 2
        INS         v17.d[1], x17
        FMLA        v22.4s, v25.4s, v1.s[0]

        # BLOCK 3
        INS         v18.d[1], x13

        # BLOCK 4
        INS         v19.d[1], x7
        FMLA        v20.4s, v17.4s, v1.s[1]
        TST         x0, 15

        # BLOCK 5
        FMLA        v21.4s, v18.4s, v1.s[1]

        # BLOCK 6
        FMLA        v22.4s, v19.4s, v1.s[1]

        # BLOCK 7
        # Is there a remainder?- 2 floats of A (8 bytes) or less
        B.NE        5f

4:
        # ks loop
        SUBS        x9, x9, 8               // ks -= MR * sizeof(void*)
        B.HI        1b

        # Clamp
        FMAX        v20.4s, v20.4s, v30.4s
        FMAX        v21.4s, v21.4s, v30.4s
        FMAX        v22.4s, v22.4s, v30.4s
        FMIN        v20.4s, v20.4s, v31.4s
        FMIN        v21.4s, v21.4s, v31.4s
        FMIN        v22.4s, v22.4s, v31.4s

        # Store full 1 x 12
        SUBS        x1, x1, 12
        B.LO        7f

        ST1         {v20.16b, v21.16b, v22.16b}, [x6], x10
        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b
        RET

5:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBZ         x0, 3, 6f

        # Remainder- 2 floats of A (8 bytes)
        LDR         d0, [x8], 8             // a0
        LD1         {v2.16b, v3.16b, v4.16b}, [x5], 48
        LD1         {v5.16b, v6.16b, v7.16b}, [x5], 48

        # First block of 3 B
        FMLA        v20.4s, v2.4s, v0.s[0]
        FMLA        v21.4s, v3.4s, v0.s[0]
        FMLA        v22.4s, v4.4s, v0.s[0]

        # Second block of 3 B
        FMLA        v20.4s, v5.4s, v0.s[1]
        FMLA        v21.4s, v6.4s, v0.s[1]
        FMLA        v22.4s, v7.4s, v0.s[1]

        TBZ         x0, 2, 4b
6:
        # Remainder - 1 float of A (4 bytes)
        LDR         s0, [x8], 4             // a0
        LD1         {v2.16b, v3.16b, v4.16b}, [x5], 48

        FMLA        v20.4s, v2.4s, v0.s[0]
        FMLA        v21.4s, v3.4s, v0.s[0]
        FMLA        v22.4s, v4.4s, v0.s[0]
        B           4b

7:
        ADD         x1, x1, 12
        # Store odd channels
        TBZ         x1, 3, 8f
        STP         q20, q21, [x6]
        ADD         x6, x6, 32
        MOV         v20.16b, v22.16b

8:
        TBZ         x1, 2, 9f
        STR         q20, [x6], 16
        MOV         v20.16b, v21.16b

9:
        TBZ         x1, 1, 10f
        STR         d20, [x6], 8
        DUP         d20, v20.d[1]

10:
        TBZ         x1, 0, 11f
        STR         s20, [x6]
11:
        RET

END_FUNCTION xnn_f32_igemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
