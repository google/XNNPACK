// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/4x8-aarch64-neonfma-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

# void xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float** restrict a,           x4
#     const void* restrict w,             x5
#     uint8_t* restrict c,                x6
#     size_t cm_stride,                  x7
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_minmax_params params [sp + 24] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# Register usage
# A0  x13 v0     v3
# A1  x14 v0[1]  v3[1]
# A2  x15 v1     v4
# A3  x20 v1[1]  v4[1]
# B    x5 v12 v13 v14 v15 second set of B
# B       v16 v17 v18 v19 first set
# C    x6 v20 v21
# C   x16 v22 v23
# C   x17 v24 v25
# C    x7 v26 v27
# Clamp v6 v7
# temporary vector shadow register x19

# unused A   v8 v9 v10 v11
# x12 a4
#  x4 a5
# x13 c4
#  x7 c5
# A4  v2     v5
# A5  v2[1]  v5[1]
# C   v28 v29
# C   v30 v31

BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53

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

        # Load cn_stride, a_offset
        LDP         x10, x11, [sp]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Load min/max values
        LD2R        {v6.4s, v7.4s}, [x8]

        # Save x19, d12-d15 on stack
        STP         d12, d13, [sp, -48]!
        STP         d14, d15, [sp, 16]
        STP         x19, x20,  [sp, 32]

0:
        # Load initial bias from w into accumulators
        LDP         q20, q21, [x5], 32
        MOV         v22.16b, v20.16b
        MOV         v23.16b, v21.16b
        MOV         v24.16b, v20.16b
        MOV         v25.16b, v21.16b
        MOV         v26.16b, v20.16b
        MOV         v27.16b, v21.16b

        MOV         x9, x3                  // p = ks

1:
        # Load next 4 A pointers
        LDP         x13, x14, [x4], 16
        LDP         x15, x20, [x4], 16

        CMP         x13, x12                // if a0 == zero
        ADD         x13, x13, x11           // a0 += a_offset
        CSEL        x13, x12, x13, EQ       //   a0 = zero, else += a0 + a_offset
        CMP         x14, x12                // if a1 == zero
        ADD         x14, x14, x11           // a1 += a_offset
        CSEL        x14, x12, x14, EQ       //   a1 = zero, else += a1 + a_offset
        CMP         x15, x12                // if a2 == zero
        ADD         x15, x15, x11           // a2 += a_offset
        CSEL        x15, x12, x15, EQ       //   a2 = zero, else += a2 + a_offset
        CMP         x20, x12                    // if a3 == zero
        ADD         x20, x20, x11               // a3 += a_offset
        CSEL        x20, x12, x20, EQ           //   a3 = zero, else += a3 + a_offset

        # Is there at least 4 floats (16 bytes) for prologue + epilogue?
        SUBS        x0, x2, 16              // k = kc - 16
        B.LO        4f

        # Prologue - First group loads, no FMA
        LDR         d0, [x13], 8            // a0
        LDP         q16, q17, [x5], 32          // b
        LDR         d1, [x15], 8            // a2
        LD1         {v0.d}[1],  [x14], 8        // a1
        LD1         {v1.d}[1], [x20], 8         // a3
        SUBS        x0, x0, 16
        LDR         q18, [x5], 16
        LDR         d19, [x5], 8
        LDR         x19, [x5], 8            // ins is in BLOCK 0

        # Is there at least 4 floats (16 bytes) for main loop?
        B.LO        3f

        # Main loop - 4 floats of A (16 bytes)
        # 32 FMA + 8 LD64 A + 8 LDR B
2:
        # First group of 16 FMA, Second group loads
        # BLOCK 0
        LDR         d3, [x13], 8                // a0
        INS         v19.d[1], x19               // b from second group
        FMLA        v20.4s, v16.4s,  v0.s[0]
        LDR         x19, [x14], 8               // a1
        FMLA        v22.4s, v16.4s,  v0.s[2]
        FMLA        v24.4s, v16.4s,  v1.s[0]

        # BLOCK 1
        LDR         d12, [x5]
        INS         v3.d[1], x19                // a1 ins
        FMLA        v26.4s, v16.4s,  v1.s[2]
        LDR         x19, [x5, 8]            // b
        FMLA        v21.4s, v17.4s,  v0.s[0]
        FMLA        v23.4s, v17.4s,  v0.s[2]

        # BLOCK 2
        LDR         d4, [x15], 8                // a2
        INS         v12.d[1], x19           // b  ins
        FMLA        v25.4s, v17.4s,  v1.s[0]
        LDR         x19, [x20], 8               // a3
        FMLA        v27.4s, v17.4s,  v1.s[2]
        FMLA        v20.4s, v18.4s,  v0.s[1]

        # BLOCK 3
        LDR         d13, [x5, 16]
        INS         v4.d[1], x19                // a3 ins
        FMLA        v22.4s, v18.4s,  v0.s[3]
        LDR         x19, [x5, 24]
        FMLA        v24.4s, v18.4s,  v1.s[1]
        FMLA        v26.4s, v18.4s,  v1.s[3]

        # BLOCK 4
        LDR         d14, [x5, 32]
        INS         v13.d[1], x19           // b
        FMLA        v21.4s, v19.4s,  v0.s[1]
        LDR         x19, [x5, 40]
        FMLA        v23.4s, v19.4s,  v0.s[3]
        FMLA        v25.4s, v19.4s,  v1.s[1]

        # BLOCK 5
        # NOPs to ensure 4 cycle LDR lands on next LDR
        LDR         d15, [x5, 48]
        INS         v14.d[1], x19           // b from previous
        FMLA        v27.4s, v19.4s,  v1.s[3]
        LDR         x19, [x5, 56]
        NOP
        NOP
        NOP
        NOP

        # Second group of 16 FMA, First group of loads
        # BLOCK 0
        LDR         d0, [x13], 8                // a0
        INS         v15.d[1], x19           // b from previous
        FMLA        v20.4s, v12.4s,  v3.s[0]
        LDR         x19, [x14], 8               // a1
        FMLA        v22.4s, v12.4s,  v3.s[2]
        FMLA        v24.4s, v12.4s,  v4.s[0]

        # BLOCK 1
        LDR         d16, [x5, 64]
        INS         v0.d[1], x19                // a1 ins
        FMLA        v26.4s, v12.4s,  v4.s[2]
        LDR         x19, [x5, 72]           // b
        FMLA        v21.4s, v13.4s,  v3.s[0]
        FMLA        v23.4s, v13.4s,  v3.s[2]

        # BLOCK 2
        LDR         d1, [x15], 8                // a2
        INS         v16.d[1], x19           // b
        FMLA        v25.4s, v13.4s,  v4.s[0]
        LDR         x19, [x20], 8               // a3
        FMLA        v27.4s, v13.4s,  v4.s[2]
        FMLA        v20.4s, v14.4s,  v3.s[1]

        # BLOCK 3
        LDR         d17, [x5, 80]
        INS         v1.d[1], x19                // a3 ins
        FMLA        v22.4s, v14.4s,  v3.s[3]
        LDR         x19, [x5, 88]
        FMLA        v24.4s, v14.4s,  v4.s[1]
        FMLA        v26.4s, v14.4s,  v4.s[3]

        # BLOCK 4
        LDR         d18, [x5, 96]
        INS         v17.d[1], x19           // b
        FMLA        v21.4s, v15.4s,  v3.s[1]
        LDR         x19, [x5, 104]
        FMLA        v23.4s, v15.4s,  v3.s[3]
        FMLA        v25.4s, v15.4s,  v4.s[1]

        # BLOCK 5
        # NOTE that block needs to be 4 cycles for LDR not to stall
        LDR         d19, [x5, 112]
        INS         v18.d[1], x19
        FMLA        v27.4s, v15.4s,  v4.s[3]
        LDR         x19, [x5, 120]
        SUBS        x0, x0, 16
        ADD         x5, x5, 128
        B.HS        2b

        # Epilogue - 4 floats of A (16 bytes)
        # 32 FMA + 8 LD64 A + 8 LDR B
3:
        # First group of 16 FMA, Second group loads
        # BLOCK 0
        LDR         d3, [x13], 8                // a0
        INS         v19.d[1], x19               // b from second group
        FMLA        v20.4s, v16.4s,  v0.s[0]
        LDR         x19, [x14], 8               // a1
        FMLA        v22.4s, v16.4s,  v0.s[2]
        FMLA        v24.4s, v16.4s,  v1.s[0]

        # BLOCK 1
        LDR         d12, [x5]
        INS         v3.d[1], x19                // a1 ins
        FMLA        v26.4s, v16.4s,  v1.s[2]
        LDR         x19, [x5, 8]            // b
        FMLA        v21.4s, v17.4s,  v0.s[0]
        FMLA        v23.4s, v17.4s,  v0.s[2]

        # BLOCK 2
        LDR         d4, [x15], 8                // a2
        INS         v12.d[1], x19           // b  ins
        FMLA        v25.4s, v17.4s,  v1.s[0]
        LDR         x19, [x20], 8               // a3
        FMLA        v27.4s, v17.4s,  v1.s[2]
        FMLA        v20.4s, v18.4s,  v0.s[1]

        # BLOCK 3
        LDR         d13, [x5, 16]
        INS         v4.d[1], x19                // a3 ins
        FMLA        v22.4s, v18.4s,  v0.s[3]
        LDR         x19, [x5, 24]
        FMLA        v24.4s, v18.4s,  v1.s[1]
        FMLA        v26.4s, v18.4s,  v1.s[3]

        # BLOCK 4
        LDR         d14, [x5, 32]
        INS         v13.d[1], x19           // b
        FMLA        v21.4s, v19.4s,  v0.s[1]
        LDR         x19, [x5, 40]
        FMLA        v23.4s, v19.4s,  v0.s[3]
        FMLA        v25.4s, v19.4s,  v1.s[1]

        # BLOCK 5
        # NOPs to ensure 4 cycle LDR lands on next LDR
        LDR         d15, [x5, 48]
        INS         v14.d[1], x19
        FMLA        v27.4s, v19.4s,  v1.s[3]
        LDR         x19, [x5, 56]
        NOP         // fma
        NOP
        NOP         // fma
        NOP

        # Second group of 16 FMA, no loads
        # BLOCK 0
        INS         v15.d[1], x19           // b from previous
        FMLA        v20.4s, v12.4s,  v3.s[0]
        FMLA        v22.4s, v12.4s,  v3.s[2]
        FMLA        v24.4s, v12.4s,  v4.s[0]

        # BLOCK 1
        FMLA        v26.4s, v12.4s,  v4.s[2]
        FMLA        v21.4s, v13.4s,  v3.s[0]
        FMLA        v23.4s, v13.4s,  v3.s[2]

        # BLOCK 2
        FMLA        v25.4s, v13.4s,  v4.s[0]
        FMLA        v27.4s, v13.4s,  v4.s[2]
        FMLA        v20.4s, v14.4s,  v3.s[1]

        # BLOCK 3
        FMLA        v22.4s, v14.4s,  v3.s[3]
        FMLA        v24.4s, v14.4s,  v4.s[1]
        FMLA        v26.4s, v14.4s,  v4.s[3]

        # BLOCK 4
        FMLA        v21.4s, v15.4s,  v3.s[1]
        FMLA        v23.4s, v15.4s,  v3.s[3]
        FMLA        v25.4s, v15.4s,  v4.s[1]
        ADD         x5, x5, 64

        # BLOCK 5
        FMLA        v27.4s, v15.4s,  v4.s[3]

4:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBNZ        x0, 3, 6f
        # Is there a remainder?- 1 float of A (4 bytes)
        TBNZ        x0, 2, 7f
5:
        # ks loop
        SUBS        x9, x9, 32              // ks -= MR * sizeof(void*)
        B.HI        1b

        # Clamp
        FMAX        v20.4s, v20.4s, v6.4s
        FMAX        v21.4s, v21.4s, v6.4s
        FMAX        v22.4s, v22.4s, v6.4s
        FMAX        v23.4s, v23.4s, v6.4s
        FMAX        v24.4s, v24.4s, v6.4s
        FMAX        v25.4s, v25.4s, v6.4s
        FMAX        v26.4s, v26.4s, v6.4s
        FMAX        v27.4s, v27.4s, v6.4s
        FMIN        v20.4s, v20.4s, v7.4s
        FMIN        v21.4s, v21.4s, v7.4s
        FMIN        v22.4s, v22.4s, v7.4s
        FMIN        v23.4s, v23.4s, v7.4s
        FMIN        v24.4s, v24.4s, v7.4s
        FMIN        v25.4s, v25.4s, v7.4s
        FMIN        v26.4s, v26.4s, v7.4s
        FMIN        v27.4s, v27.4s, v7.4s

        # Store full 4 x 8
        SUBS        x1, x1, 8
        B.LO        8f

        STP         q26, q27, [x7]
        ADD         x7, x7, x10
        STP         q24, q25, [x17]
        ADD         x17, x17, x10
        STP         q22, q23, [x16]
        ADD         x16, x16, x10
        STP         q20, q21,  [x6]
        ADD         x6,  x6, x10

        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b

        # Restore x19, d12-d15 from stack
        LDP         x19, x20, [sp, 32]
        LDP         d14, d15, [sp, 16]
        LDP         d12, d13, [sp], 48
        RET

        # Remainder - 2 floats of A (8 bytes)
        # 16 FMA + 4 LD64 A + 2 LDP B
6:
        LDR         d0,  [x13], 8
        LDP         q16,  q17, [x5], 32
        LD1         {v0.d}[1], [x14], 8
        LDR         d1, [x15], 8
        LD1         {v1.d}[1], [x20], 8
        LDP         q18,  q19, [x5], 32
        FMLA        v20.4s, v16.4s,  v0.s[0]
        FMLA        v22.4s, v16.4s,  v0.s[2]
        FMLA        v24.4s, v16.4s,  v1.s[0]
        FMLA        v26.4s, v16.4s,  v1.s[2]
        FMLA        v21.4s, v17.4s,  v0.s[0]
        FMLA        v23.4s, v17.4s,  v0.s[2]
        FMLA        v25.4s, v17.4s,  v1.s[0]
        FMLA        v27.4s, v17.4s,  v1.s[2]

        FMLA        v20.4s, v18.4s,  v0.s[1]
        FMLA        v22.4s, v18.4s,  v0.s[3]
        FMLA        v24.4s, v18.4s,  v1.s[1]
        FMLA        v26.4s, v18.4s,  v1.s[3]
        FMLA        v21.4s, v19.4s,  v0.s[1]
        FMLA        v23.4s, v19.4s,  v0.s[3]
        FMLA        v25.4s, v19.4s,  v1.s[1]
        FMLA        v27.4s, v19.4s,  v1.s[3]

        # Is there a remainder?- 1 float of A (4 bytes)
        TBZ         x0, 2, 5b

7:
        # Remainder- 1 float of A (4 bytes)
        LDR         s0,  [x13], 4
        LDP         q16,  q17, [x5], 32
        LD1         {v0.s}[2], [x14], 4
        LDR         s1, [x15], 4
        LD1         {v1.s}[2], [x20], 4

        FMLA        v20.4s, v16.4s,  v0.s[0]
        FMLA        v22.4s, v16.4s,  v0.s[2]
        FMLA        v24.4s, v16.4s,  v1.s[0]
        FMLA        v26.4s, v16.4s,  v1.s[2]
        FMLA        v21.4s, v17.4s,  v0.s[0]
        FMLA        v23.4s, v17.4s,  v0.s[2]
        FMLA        v25.4s, v17.4s,  v1.s[0]
        FMLA        v27.4s, v17.4s,  v1.s[2]
        B           5b

        # Store odd width
8:
        TBZ         x1, 2, 9f
        STR         q26,  [x7], 16
        MOV         v26.16b, v27.16b
        STR         q24, [x17], 16
        MOV         v24.16b, v25.16b
        STR         q22, [x16], 16
        MOV         v22.16b, v23.16b
        STR         q20,  [x6], 16
        MOV         v20.16b, v21.16b
9:
        TBZ         x1, 1, 10f
        STR         d26,  [x7], 8
        STR         d24, [x17], 8
        DUP         d26, v26.d[1]
        DUP         d24, v24.d[1]
        STR         d22, [x16], 8
        STR         d20,  [x6], 8
        DUP         d22, v22.d[1]
        DUP         d20, v20.d[1]

10:
        TBZ         x1, 0, 11f
        STR         s26,  [x7]
        STR         s24, [x17]
        STR         s22, [x16]
        STR         s20,  [x6]
11:
        # Restore x19, d12-d15 from stack
        LDP         x19, x20, [sp, 32]
        LDP         d14, d15, [sp, 16]
        LDP         d12, d13, [sp], 48
        RET

END_FUNCTION xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
