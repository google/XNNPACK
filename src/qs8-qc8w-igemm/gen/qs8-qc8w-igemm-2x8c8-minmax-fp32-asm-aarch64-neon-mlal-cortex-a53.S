// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

# void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53(
#     size_t mr,                 x0
#     size_t nc,                 x1
#     size_t kc,                 x2 / x0
#     size_t ks,                 x3 / x9
#     const int8_t** restrict a,  x4
#     const int8_t* restrict w,  x5
#     int8_t* restrict c,        x6
#     size_t cm_stride,          x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x8
#     const int8_t* zero,                [sp + 16] -> x12
#     const union xnn_qs8_qc8w_conv_minmax_params params [sp + 24] -> x11

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0 x13  v0  v6
// A1 x15  v1  v7
// B   x5  v4  v5  v8  v9
// C0  x6 v16 v18 v20 v22 v24 v26 v28 v30
// C1  x7 v17 v19 v21 v23 v25 v27 v29 v31
// temp0   v2 v10 v12 v14
// temp1   v3 v11 v13 v15
// x16, x17, x20, x21 tenporary a53 gpr load data


BEGIN_FUNCTION xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53

        # Clamp C pointers
        LDP         x10, x8, [sp]           // Load cn_stride, a_offset
        CMP         x0, 2                   // if mr < 2
        LDP         x12, x11, [sp, 16]      // Load zero, params pointer
        ADD         x7, x6, x7              // c1 = c0 + cm_stride
        STP         d8, d9, [sp, -80]!
        ADD         x2, x2, 7               // kc = (kc + 7) & ~7
        STP         d10, d11, [sp, 16]
        CSEL        x7, x6, x7, LO          //   c1 = c0
        STP         d12, d13, [sp, 32]
        BIC         x2, x2, 7
        STP         d14, d15, [sp, 48]
        STP         x20, x21, [sp, 64]      // Save x20,x21 on stack

        .p2align    3
0:
        # Load initial bias from w into accumulators
        LDP         s16, s18, [x5], 8
        MOV         v17.16b, v16.16b
        MOV         v19.16b, v18.16b
        LDP         s20, s22, [x5], 8
        MOV         v21.16b, v20.16b
        MOV         v23.16b, v22.16b
        LDP         s24, s26, [x5], 8
        MOV         v25.16b, v24.16b
        MOV         v27.16b, v26.16b
        LDP         s28, s30, [x5], 8
        MOV         v29.16b, v28.16b
        MOV         v31.16b, v30.16b
        MOV         x9, x3                  // p = ks

        .p2align    3
1:
        # Load next 2 A pointers
        LDP         x13, x15, [x4], 16
        CMP         x13, x12                // if a0 == zero
        ADD         x13, x13, x8            // a0 += a_offset
        CSEL        x13, x12, x13, EQ       //   a0 = zero, else += a0 + a_offset
        CMP         x15, x12                // if a1 == zero
        ADD         x15, x15, x8            // a1 += a_offset
        CSEL        x15, x12, x15, EQ       //   a1 = zero, else += a1 + a_offset

        # Is there at least 16 bytes for epilogue?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        5f

        # Prologue: load A0, A1 and 2 B's
        LDP         d4, d5, [x5]            // Read B
        LDP         d0, d6, [x13], 16
        LDP         d1, d7, [x15], 16
//        LDP     d8, d9, [x5, 64]
        LDR         x17, [x5, 64]           // Read B
        LDR         x16, [x5, 16]

        # Is there at least 16 bytes for main loop?
        SUBS        x0, x0, 16              // k = k - 16
        B.LO        3f

         # Main loop - 16 bytes of A
         # 4 groups of 4 mul/mla/adap + 2 load = 18 cycles.
         # 2 loads for A0 = +2 cycles.  Total 18 * 4 + 2 = 74 cycles.

        .p2align    3
2:
        # BLOCK 0 - 18 cycles - includes prfm
        LDR         d9, [x5, 72]            // Read B
        INS         v8.d[0], x17
        SMULL       v2.8h, v4.8b, v0.8b
        SMULL       v3.8h, v4.8b, v1.8b
        LDR         x17, [x5, 80]
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDR         d5, [x5, 24]
        INS         v4.d[0], x16
        SMLAL       v2.8h, v8.8b, v6.8b
        SMLAL       v3.8h, v8.8b, v7.8b
        LDR         x16, [x5, 32]
        SMLAL       v10.8h, v9.8b, v6.8b
        SMLAL       v11.8h, v9.8b, v7.8b
        SADALP      v16.4s,  v2.8h
        SADALP      v17.4s,  v3.8h
        SADALP      v18.4s, v10.8h
        SADALP      v19.4s, v11.8h

        # BLOCK 1- 18 cycles
        LDR         d9, [x5, 88]
        INS         v8.d[0], x17
        SMULL       v12.8h, v4.8b, v0.8b
        SMULL       v13.8h, v4.8b, v1.8b
        LDR         x17, [x5, 96]
        SMULL       v14.8h, v5.8b, v0.8b
        SMULL       v15.8h, v5.8b, v1.8b
        LDR         d5, [x5, 40]
        INS         v4.d[0], x16
        SMLAL       v12.8h, v8.8b, v6.8b
        SMLAL       v13.8h, v8.8b, v7.8b
        LDR         x16, [x5, 48]
        SMLAL       v14.8h, v9.8b, v6.8b
        SMLAL       v15.8h, v9.8b, v7.8b
        SADALP      v20.4s, v12.8h
        SADALP      v21.4s, v13.8h
        SADALP      v22.4s, v14.8h
        SADALP      v23.4s, v15.8h

        # BLOCK 2 - 18 cycles
        LDR         d9, [x5, 104]
        INS         v8.d[0], x17
        SMULL       v2.8h, v4.8b, v0.8b
        SMULL       v3.8h, v4.8b, v1.8b
        LDR         x17, [x5, 112]
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDR         d5, [x5, 56]
        INS         v4.d[0], x16
        SMLAL       v2.8h, v8.8b, v6.8b
        SMLAL       v3.8h, v8.8b, v7.8b
        LDR         x16, [x5, 128]
        SMLAL       v10.8h, v9.8b, v6.8b
        SMLAL       v11.8h, v9.8b, v7.8b
        SADALP      v24.4s,  v2.8h
        LDR         x20, [x13], 8           // Read A0
        SADALP      v25.4s,  v3.8h
        LDR         x21, [x15], 8           // Read A1
        SADALP      v26.4s, v10.8h
        SADALP      v27.4s, v11.8h
        SUBS        x0, x0, 16

        # BLOCK 3 - includes 2 cycles to read A0, A1 = 20 cycles
        LDR         d9, [x5, 120]
        INS         v8.d[0], x17
        SMULL       v12.8h, v4.8b, v0.8b
        SMULL       v13.8h, v4.8b, v1.8b
        LDR         x17, [x5, 192]          // Read B
        SMULL       v14.8h, v5.8b, v0.8b
        SMULL       v15.8h, v5.8b, v1.8b
        LDR         d5, [x5, 136]           // Read B
        INS         v4.d[0], x16
        SMLAL       v12.8h, v8.8b, v6.8b
        SMLAL       v13.8h, v8.8b, v7.8b
        LDR         x16, [x5, 144]
        SMLAL       v14.8h, v9.8b, v6.8b
        SMLAL       v15.8h, v9.8b, v7.8b
        LDR         d6, [x13], 8            // Read A0
        INS         v0.d[0], x20
        LDR         d7, [x15], 8            // Read A1
        INS         v1.d[0], x21
        SADALP      v28.4s, v12.8h
        SADALP      v29.4s, v13.8h
        ADD         x5, x5, 128
        SADALP      v30.4s, v14.8h
        SADALP      v31.4s, v15.8h
        B.HS        2b

        # Epilogue
        # Same as main loop except no loads at end of loop
        .p2align    3
3:
        # BLOCK 0 - 18 cycles
        LDR         d9, [x5, 72]            // Read B
        INS         v8.d[0], x17
        SMULL       v2.8h, v4.8b, v0.8b
        SMULL       v3.8h, v4.8b, v1.8b
        LDR         x17, [x5, 80]
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDR         d5, [x5, 24]
        INS         v4.d[0], x16
        SMLAL       v2.8h, v8.8b, v6.8b
        SMLAL       v3.8h, v8.8b, v7.8b
        LDR         x16, [x5, 32]
        SMLAL       v10.8h, v9.8b, v6.8b
        SMLAL       v11.8h, v9.8b, v7.8b
        SADALP      v16.4s,  v2.8h
        SADALP      v17.4s,  v3.8h
        SADALP      v18.4s, v10.8h
        SADALP      v19.4s, v11.8h

        # BLOCK 1- 18 cycles
        LDR         d9, [x5, 88]
        INS         v8.d[0], x17
        SMULL       v12.8h, v4.8b, v0.8b
        SMULL       v13.8h, v4.8b, v1.8b
        LDR         x17, [x5, 96]
        SMULL       v14.8h, v5.8b, v0.8b
        SMULL       v15.8h, v5.8b, v1.8b
        LDR         d5, [x5, 40]
        INS         v4.d[0], x16
        SMLAL       v12.8h, v8.8b, v6.8b
        SMLAL       v13.8h, v8.8b, v7.8b
        LDR         x16, [x5, 48]
        SMLAL       v14.8h, v9.8b, v6.8b
        SMLAL       v15.8h, v9.8b, v7.8b
        SADALP      v20.4s, v12.8h
        SADALP      v21.4s, v13.8h
        SADALP      v22.4s, v14.8h
        SADALP      v23.4s, v15.8h

        # BLOCK 2 - 18 cycles
        LDR         d9, [x5, 104]
        INS         v8.d[0], x17
        SMULL       v2.8h, v4.8b, v0.8b
        SMULL       v3.8h, v4.8b, v1.8b
        LDR         x17, [x5, 112]
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        LDR         d5, [x5, 56]
        INS         v4.d[0], x16
        SMLAL       v2.8h, v8.8b, v6.8b
        SMLAL       v3.8h, v8.8b, v7.8b
        SMLAL       v10.8h, v9.8b, v6.8b
        SMLAL       v11.8h, v9.8b, v7.8b
        SADALP      v24.4s,  v2.8h
        SADALP      v25.4s,  v3.8h
        SADALP      v26.4s, v10.8h
        SADALP      v27.4s, v11.8h

        # BLOCK 3 - 17 cycles
        LDR         d9, [x5, 120]
        INS         v8.d[0], x17
        SMULL       v12.8h, v4.8b, v0.8b
        SMULL       v13.8h, v4.8b, v1.8b
        SMULL       v14.8h, v5.8b, v0.8b
        SMULL       v15.8h, v5.8b, v1.8b
        SMLAL       v12.8h, v8.8b, v6.8b
        SMLAL       v13.8h, v8.8b, v7.8b
        SMLAL       v14.8h, v9.8b, v6.8b
        SMLAL       v15.8h, v9.8b, v7.8b
        SADALP      v28.4s, v12.8h
        SADALP      v29.4s, v13.8h
        ADD         x5, x5, 128
        SADALP      v30.4s, v14.8h
        SADALP      v31.4s, v15.8h

        # Is there a remainder?- 8 bytes of A
        TBNZ        x0, 3, 5f

        # ks loop
        SUBS        x9, x9, 16              // ks -= MR * sizeof(int8_t*)
        B.HI        1b

4:
        # Add columns
        ADDP        v16.4s, v16.4s, v18.4s
        ADDP        v20.4s, v20.4s, v22.4s
        ADDP        v24.4s, v24.4s, v26.4s
        ADDP        v28.4s, v28.4s, v30.4s
        ADDP        v17.4s, v17.4s, v19.4s
        ADDP        v21.4s, v21.4s, v23.4s
        ADDP        v25.4s, v25.4s, v27.4s
        ADDP        v29.4s, v29.4s, v31.4s
        ADDP        v0.4s, v16.4s, v20.4s
        ADDP        v1.4s, v24.4s, v28.4s
        ADDP        v2.4s, v17.4s, v21.4s
        ADDP        v3.4s, v25.4s, v29.4s

        # Load per channel scale values from weights
        SCVTF       v0.4s, v0.4s
        LDR         q4, [x5], 16
        SCVTF       v1.4s, v1.4s
        LDR         q5, [x5], 16
        SCVTF       v2.4s, v2.4s
        SCVTF       v3.4s, v3.4s
        FMUL        v0.4s, v0.4s, v4.4s
        FMUL        v1.4s, v1.4s, v5.4s
        FMUL        v2.4s, v2.4s, v4.4s
        FMUL        v3.4s, v3.4s, v5.4s

        FCVTNS      v0.4s, v0.4s
        FCVTNS      v1.4s, v1.4s
        FCVTNS      v2.4s, v2.4s
        FCVTNS      v3.4s, v3.4s

        LD1R        {v5.8h}, [x11], 2
        SQXTN       v0.4h, v0.4s
        SQXTN       v2.4h, v2.4s
        SQXTN2      v0.8h, v1.4s
        SQXTN2      v2.8h, v3.4s
        SUBS        x1, x1, 8
        SQADD       v0.8h, v0.8h, v5.8h
        SQADD       v1.8h, v2.8h, v5.8h
        SQXTN       v0.8b, v0.8h
        SQXTN2      v0.16b, v1.8h
        LD1R        {v1.16b}, [x11], 1
        LD1R        {v2.16b}, [x11]
        SMAX        v0.16b, v0.16b, v1.16b
        SUB         x11, x11, 3           // rewind params pointer
        SMIN        v0.16b, v0.16b, v2.16b
        B.LO        6f

        # Store full 2 x 8
        ST1         {v0.d}[1], [x7], x10
        ST1         {v0.8b}, [x6], x10

        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b

        # Restore x20,x21 from stack
        LDP         x20, x21, [sp, 64]

        # Restore d8-d15 from stack
        LDP         d14, d15, [sp, 48]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 80
        RET

        # Remainder - 8 bytes of A
        .p2align    3
5:
        LDR         d0, [x13], 8
        LDP         d4, d5, [x5]
        LDR         d1, [x15], 8
        LDP         d6, d7, [x5, 16]
        SMULL       v2.8h, v4.8b, v0.8b
        SMULL       v3.8h, v4.8b, v1.8b
        SMULL       v10.8h, v5.8b, v0.8b
        SMULL       v11.8h, v5.8b, v1.8b
        SMULL       v12.8h, v6.8b, v0.8b
        SADALP      v16.4s,  v2.8h
        SMULL       v13.8h, v6.8b, v1.8b
        SADALP      v17.4s,  v3.8h
        SMULL       v14.8h, v7.8b, v0.8b
        SADALP      v18.4s, v10.8h
        SMULL       v15.8h, v7.8b, v1.8b
        SADALP      v19.4s, v11.8h
        LDP         d4, d5, [x5, 32]
        SMULL       v2.8h, v4.8b, v0.8b
        SADALP      v20.4s, v12.8h
        SMULL       v3.8h, v4.8b, v1.8b
        SADALP      v21.4s, v13.8h
        SMULL       v10.8h, v5.8b, v0.8b
        SADALP      v22.4s, v14.8h
        SMULL       v11.8h, v5.8b, v1.8b
        SADALP      v23.4s, v15.8h
        LDP         d6, d7, [x5, 48]
        SMULL       v12.8h, v6.8b, v0.8b
        SADALP      v24.4s,  v2.8h
        SMULL       v13.8h, v6.8b, v1.8b
        SADALP      v25.4s,  v3.8h
        SMULL       v14.8h, v7.8b, v0.8b
        SADALP      v26.4s, v10.8h
        SMULL       v15.8h, v7.8b, v1.8b
        SADALP      v27.4s, v11.8h
        ADD         x5, x5, 64
        SADALP      v28.4s, v12.8h
        SADALP      v29.4s, v13.8h
        SADALP      v30.4s, v14.8h
        SADALP      v31.4s, v15.8h

        # ks loop
        SUBS        x9, x9, 16              // ks -= MR * sizeof(int8_t*)
        B.HI        1b
        B           4b

        # Store odd width
        .p2align    3
6:
        TBZ         x1, 2, 7f
        ST1         {v0.s}[2], [x7], 4
        STR         s0, [x6], 4
        EXT         v0.16b, v0.16b, v0.16b, 4

7:
        TBZ         x1, 1, 8f
        ST1         {v0.h}[4], [x7], 2
        STR         h0, [x6], 2
        EXT         v0.16b, v0.16b, v0.16b, 2
8:
        TBZ         x1, 0, 9f
        ST1         {v0.b}[8], [x7]
        STR         b0, [x6]
9:
        # Restore x20,x21 from stack
        LDP         x20, x21, [sp, 64]

        # Restore d8-d15 from stack
        LDP         d14, d15, [sp, 48]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         d8, d9, [sp], 80
        RET

END_FUNCTION xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
