// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/1x8c8-aarch64-neon-mlal.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     size_t ks,                 x3 / x9
#     const int8_t** restrict a,  x4
#     const int8_t* restrict w,  x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          (x7)
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x8
#     const int8_t* zero,                [sp + 16] -> x12
#     const union xnn_qs8_qc8w_conv_minmax_params params [sp + 24] -> x11

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0 x13  v0  v6
// B   x5  v4  v5  v2  v3
// C0  x6 v16 v18 v20 v22 v24 v26 v28 v30
// temp0  v17 v19 v21 v23

BEGIN_FUNCTION xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm
        # Clamp C pointers
        LDP         x10, x8, [sp]           // Load cn_stride, a_offset
        ADD         x2, x2, 7               // kc = (kc + 7) & ~7
        LDP         x12, x11, [sp, 16]      // Load zero, params pointer
        BIC         x2, x2, 7

        .p2align    3
0:
        # Load initial bias from w into accumulators
        LDP         s16, s18, [x5], 8
        LDP         s20, s22, [x5], 8
        LDP         s24, s26, [x5], 8
        LDP         s28, s30, [x5], 8
        MOV         x9, x3                  // p = ks

        .p2align    3
1:
        # Load next A pointer
        LDR         x13, [x4], 8
        CMP         x13, x12                // if a0 == zero
        ADD         x13, x13, x8            // a0 += a_offset
        CSEL        x13, x12, x13, EQ       //   a0 = zero, else += a0 + a_offset

        # Is there at least 16 bytes for epilogue?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        5f

        # Prologue: load A0 and 4 B's
        LDP         d0, d6, [x13], 16       // Read A0
        LDP         d4, d5, [x5]            // Read B
        LDP         d2, d3, [x5, 64]        // Read B

        # Is there at least 16 bytes for main loop?
        SUBS        x0, x0, 16              // k = k - 16
        B.LO        3f

        # Main loop - 16 bytes of A
        # 4 groups of 2 mul/mla/adap = 6 cycles.
        # 2 load for A0, A1 = +4 cycle.  Total 36 cycles.

        .p2align    3
2:
        # BLOCK 0 - 4 cycles
        SMULL       v17.8h, v4.8b, v0.8b
        SMULL       v19.8h, v5.8b, v0.8b
        LDP         d4, d5, [x5, 16]
        SMLAL       v17.8h, v2.8b, v6.8b
        SMLAL       v19.8h, v3.8b, v6.8b
        LDP         d2, d3, [x5, 80]

        # BLOCK 1 - 6 cycles
        SMULL       v21.8h, v4.8b, v0.8b
        SMULL       v23.8h, v5.8b, v0.8b
        PRFM        PLDL1KEEP, [x5, 448]
        SADALP      v16.4s, v17.8h
        PRFM        PLDL1KEEP, [x5, 512]
        SADALP      v18.4s, v19.8h
        LDP         d4, d5, [x5, 32]
        SMLAL       v21.8h, v2.8b, v6.8b
        SMLAL       v23.8h, v3.8b, v6.8b
        LDP         d2, d3, [x5, 96]

        # BLOCK 2 - 6 cycles
        SMULL       v17.8h, v4.8b, v0.8b
        SMULL       v19.8h, v5.8b, v0.8b
        PRFM        PLDL1KEEP, [x13, 128]
        SADALP      v20.4s, v21.8h
        SADALP      v22.4s, v23.8h
        LDP         d4, d5, [x5, 48]
        SMLAL       v17.8h, v2.8b, v6.8b
        SMLAL       v19.8h, v3.8b, v6.8b
        LDP         d2, d3, [x5, 112]

        # BLOCK 3 - 14 cycles
        SMULL       v21.8h, v4.8b, v0.8b
        ADD         x5, x5, 128
        SMULL       v23.8h, v5.8b, v0.8b
        SADALP      v24.4s, v17.8h
        SUBS        x0, x0, 16
        SADALP      v26.4s, v19.8h
        LDP         d4, d5, [x5]            // Read B
        SMLAL       v21.8h, v2.8b, v6.8b
        SMLAL       v23.8h, v3.8b, v6.8b
        LDP         d0, d6, [x13], 16       // Read A0
        SADALP      v28.4s, v21.8h
        LDP         d2, d3, [x5, 64]        // Read B
        SADALP      v30.4s, v23.8h
        B.HS        2b

        # Epilogue
        # Same as main loop except no loads at end of loop

        .p2align    3
3:
       # BLOCK 0 - 4 cycles
        SMULL       v17.8h, v4.8b, v0.8b
        SMULL       v19.8h, v5.8b, v0.8b
        LDP         d4, d5, [x5, 16]
        SMLAL       v17.8h, v2.8b, v6.8b
        SMLAL       v19.8h, v3.8b, v6.8b
        LDP         d2, d3, [x5, 80]

        # BLOCK 1 - 6 cycles
        SMULL       v21.8h, v4.8b, v0.8b
        SMULL       v23.8h, v5.8b, v0.8b
        PRFM        PLDL1KEEP, [x5, 448]
        SADALP      v16.4s, v17.8h
        PRFM        PLDL1KEEP, [x5, 512]
        SADALP      v18.4s, v19.8h
        LDP         d4, d5, [x5, 32]
        SMLAL       v21.8h, v2.8b, v6.8b
        SMLAL       v23.8h, v3.8b, v6.8b
        LDP         d2, d3, [x5, 96]

        # BLOCK 2 - 6 cycles
        SMULL       v17.8h, v4.8b, v0.8b
        SMULL       v19.8h, v5.8b, v0.8b
        PRFM        PLDL1KEEP, [x13, 128]
        SADALP      v20.4s, v21.8h
        SADALP      v22.4s, v23.8h
        LDP         d4, d5, [x5, 48]
        SMLAL       v17.8h, v2.8b, v6.8b
        SMLAL       v19.8h, v3.8b, v6.8b
        LDP         d2, d3, [x5, 112]

        # BLOCK 3 - 8 cycles
        SMULL       v21.8h, v4.8b, v0.8b
        ADD         x5, x5, 128
        SMULL       v23.8h, v5.8b, v0.8b
        SADALP      v24.4s, v17.8h
        SUBS        x0, x0, 16
        SADALP      v26.4s, v19.8h
        SMLAL       v21.8h, v2.8b, v6.8b
        SMLAL       v23.8h, v3.8b, v6.8b
        SADALP      v28.4s, v21.8h
        SADALP      v30.4s, v23.8h

        # Is there a remainder?- 8 bytes of A
        TBNZ        x0, 3, 5f

        # ks loop
        SUBS        x9, x9, 8               // ks -= MR * sizeof(int8_t*)
        B.HI        1b

4:
        # Add columns
        ADDP        v16.4s, v16.4s, v18.4s
        ADDP        v20.4s, v20.4s, v22.4s
        ADDP        v24.4s, v24.4s, v26.4s
        ADDP        v28.4s, v28.4s, v30.4s
        ADDP        v0.4s, v16.4s, v20.4s
        ADDP        v1.4s, v24.4s, v28.4s

        # Load per channel scale values from weights
        SCVTF       v0.4s, v0.4s
        LDR         q4, [x5], 16
        SCVTF       v1.4s, v1.4s
        LDR         q5, [x5], 16
        FMUL        v0.4s, v0.4s, v4.4s
        FMUL        v1.4s, v1.4s, v5.4s

        FCVTNS      v0.4s, v0.4s
        FCVTNS      v1.4s, v1.4s

        LD1R        {v5.8h}, [x11], 2
        SQXTN       v0.4h, v0.4s
        SQXTN2      v0.8h, v1.4s
        SUBS        x1, x1, 8
        SQADD       v0.8h, v0.8h, v5.8h
        LD1R        {v1.16b}, [x11], 1
        SQXTN       v0.8b, v0.8h
        LD1R        {v17.16b}, [x11]
        SMAX        v0.8b, v0.8b, v1.8b
        SUB         x11, x11, 3           // rewind params pointer
        SMIN        v0.8b, v0.8b, v17.8b
        B.LO        6f

        # Store full 1 x 8
        ST1         {v0.8b}, [x6], x10
        SUB         x4, x4, x3              // a -= ks
        B.HI        0b
        RET

        # Remainder - 8 bytes of A
        .p2align    3
5:
        LDR         d0, [x13], 8
        LDP         d4, d5, [x5]
        LDP         d6, d7, [x5, 16]
        SMULL       v17.8h, v4.8b, v0.8b
        SMULL       v19.8h, v5.8b, v0.8b
        SMULL       v21.8h, v6.8b, v0.8b
        SMULL       v23.8h, v7.8b, v0.8b
        LDP         d4, d5, [x5, 32]
        LDP         d6, d7, [x5, 48]
        SADALP      v16.4s, v17.8h
        SADALP      v18.4s, v19.8h
        SADALP      v20.4s, v21.8h
        SADALP      v22.4s, v23.8h
        SMULL       v17.8h, v4.8b, v0.8b
        SMULL       v19.8h, v5.8b, v0.8b
        SMULL       v21.8h, v6.8b, v0.8b
        SMULL       v23.8h, v7.8b, v0.8b
        ADD         x5, x5, 64
        SADALP      v24.4s, v17.8h
        SADALP      v26.4s, v19.8h
        SADALP      v28.4s, v21.8h
        SADALP      v30.4s, v23.8h

        # ks loop
        SUBS        x9, x9, 8               // ks -= MR * sizeof(int8_t*)
        B.HI        1b
        B           4b

        # Store odd width
        .p2align    3
6:
        TBZ         x1, 2, 7f
        STR         s0, [x6], 4
        EXT         v0.16b, v0.16b, v0.16b, 4

7:
        TBZ         x1, 1, 8f
        STR         h0, [x6], 2
        EXT         v0.16b, v0.16b, v0.16b, 2
8:
        TBZ         x1, 0, 9f
        STR         b0, [x6]
9:
        RET

END_FUNCTION xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
