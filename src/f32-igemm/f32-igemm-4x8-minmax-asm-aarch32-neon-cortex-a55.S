// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

.syntax unified

# LINT.IfChange
// void xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5
//     size_t ks,                            r3 -> sp + 64 -> r14
//     const float** restrict a,  sp + 104 -> (r5)
//     const void* restrict w,    sp + 108 -> r9
//     uint8_t* restrict c,       sp + 112 -> r11
//     size_t cm_stride,         sp + 116 -> (r6)
//     size_t cn_stride,         sp + 120 -> (r0)
//     size_t a_offset,          sp + 124 -> (r5)
//     const float* zero,        sp + 128 -> (r0)
//     minmax_params*params,     sp + 132 -> (r14)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0 d4
// A1  r12  d1 d5
// A2  r10  d2 d6
// A3   r7  d3 d7
// B    r9  d8,  d9, d10, d11
// B       d12, d13, d14, d15
// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15
// clamp  (r14) d4 d5 d6 d7

BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55
        .arm
#ifndef __APPLE__
        .arch       armv7-a
        .fpu        neon
#endif
        # Push 104 bytes
        PUSH        {r3, r4, r5, r6, r7, r8, r9, r10, r11, lr}      // +40
        VPUSH       {d8-d15}                                        // +64 = 104

        LDR         r11, [sp, 112]          // c
        LDR         r6, [sp, 116]           // cm_stride
        LDR         r5, [sp, 104]           // a
        LDR         r9, [sp, 108]           // w
        MOV         r14, r3                 // p = ks

        # Clamp C pointers
        CMP         r0, 2                   // if mr >= 2
        ADD         r4, r11, r6             //   c1 = c0 + cm_stride
        MOVLO       r4, r11                 // c1
                                     // if mr > 2
        ADD         r8, r4, r6              //   c2 = c1 + cm_stride
        MOVLS       r8, r4                  // c2
        CMP         r0, 4                   // if mr >=4
        ADD         r6, r8, r6              //   c3 = c2 + cm_stride
        MOVLO       r6, r8                  // c3


        .p2align    3
0:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}          // Bias

        VMOV        q10, q8
        VMOV        q11, q9
        VMOV        q12, q8
        VMOV        q13, q9
        PLD         [r9,   0]               // Prefetch B
        PLD         [r9,  64]
        VMOV        q14, q8
        PLD         [r9, 128]
        PLD         [r9, 192]
        VMOV        q15, q9
        PLD         [r9, 256]
        PLD         [r9, 320]

1:
        # Load next 4 A pointers
        LDR         r3, [r5,  0]
        LDR         r12, [r5,  4]
        LDR         r10, [r5,  8]
        LDR         r7, [r5, 12]
        ADD         r5, r5, 16
        PLD         [r3,  0]                // Prefetch A
        STR         r5, [sp, 104]           // a
        PLD         [r3, 64]
        LDR         r0, [sp, 128]           // zero
        PLD         [r12,  0]
        LDR         r5, [sp, 124]           // a_offset
        PLD         [r12, 64]
        PLD         [r10,  0]
        PLD         [r10, 64]
        PLD         [r7,  0]
        PLD         [r7, 64]

        # Add a_offset
        CMP         r3,  r0                 // if a0 == zero
        ADD         r3,  r3, r5             // a0 += a_offset
        MOVEQ       r3,  r0                 //   a0 = zero, else += a0 + a_offset
        CMP         r12,  r0                // if a1 == zero
        ADD         r12, r12, r5            // a1 += a_offset
        MOVEQ       r12,  r0                //   a1 = zero, else += a1 + a_offset
        CMP         r10,  r0                // if a2 == zero
        ADD         r10, r10, r5            // a2 += a_offset
        MOVEQ       r10,  r0                //   a2 = zero, else += a2 + a_offset
        CMP         r7,  r0                 // if a3 == zero
        ADD         r7,  r7, r5             // a3 += a_offset
        MOVEQ       r7,  r0                 //   a3 = zero, else += a3 + a_offset

        SUBS        r5, r2, 16              // kc - 16
        BLO         5f                      // less than 4 channels?

        # Prologue
        VLD1.32     {d0},  [r3]!            // A0
        VLD1.32     {d1}, [r12]!            // A1
        VLD1.32     {d2}, [r10]!            // A2
        VLD1.32     {d3},  [r7]!            // A3
        SUBS        r5, r5, 16
        VLDM        r9, {d8-d11}            // B0
        VLDR        d15, [r9, 56]           // B1CK 0
        VLDR        d13, [r9, 40]           // B1

        BLO         3f                      // less than 4 channels?  skip main loop

        # Main loop - 4 floats of A (16 bytes)
        # 32 FMA + 8 LD64 A + 8 LDR B
        .p2align    3
2:
        # First group of 16 FMA, Second group loads
        # BLOCK 0
        VMLA.F32    q8, q4, d0[0]
        VLD1.32     {d4}, [r3]!             // A0
        VMLA.F32    q10, q4, d1[0]
        VLD1.32     {d5}, [r12]!            // A1
        VMLA.F32    q12, q4, d2[0]

        # BLOCK 1
        VMLA.F32    q14, q4, d3[0]
        VLDR        d12, [r9, 32]           // B1
        VMLA.F32    q9, q5, d0[0]
        VLDR        d9, [r9, 72]            // B0
        VMLA.F32    q11, q5, d1[0]

        # BLOCK 2
        VMLA.F32    q13, q5, d2[0]
        VLD1.32     {d6}, [r10]!            // A2
        VMLA.F32    q15, q5, d3[0]
        VLD1.32     {d7}, [r7]!             // A3
        VMLA.F32    q8, q6, d0[1]

        # BLOCK 3
        VMLA.F32    q10, q6, d1[1]
        VLDR        d14, [r9, 48]           // B1
        VMLA.F32    q12, q6, d2[1]
        VLDR        d11, [r9, 88]           // B0
        VMLA.F32    q14, q6, d3[1]

        # BLOCK 4
        VMLA.F32    q9, q7, d0[1]
        VLDR        d8, [r9, 64]            // B0
        VMLA.F32    q11, q7, d1[1]
        VLDR        d13, [r9, 104]          // B1
        VMLA.F32    q13, q7, d2[1]
        VLDR        d10, [r9, 80]           // B0

        # BLOCK 5
        VMLA.F32    q15, q7, d3[1]
        VLDR        d15, [r9, 120]          // B1

        # Second group of 16 FMA, First group of loads
        # BLOCK 0
        VMLA.F32    q8, q4, d4[0]
        VLD1.32     {d0}, [r3]!             // A0
        VMLA.F32    q10, q4, d5[0]
        VLD1.32     {d1}, [r12]!            // A1
        VMLA.F32    q12, q4, d6[0]

        # BLOCK 1
        VMLA.F32    q14, q4, d7[0]
        VLDR        d12, [r9, 96]           // B1
        VMLA.F32    q9, q5, d4[0]
        VLDR        d9, [r9, 136]           // B0
        VMLA.F32    q11, q5, d5[0]

        # BLOCK 2
        VMLA.F32    q13, q5, d6[0]
        VLD1.32     {d2}, [r10]!            // A2
        VMLA.F32    q15, q5, d7[0]
        VLD1.32     {d3}, [r7]!             // A3
        VMLA.F32    q8, q6, d4[1]
        SUBS        r5, r5, 16

        # BLOCK 3
        VMLA.F32    q10, q6, d5[1]
        VLDR        d14, [r9, 112]          // B1
        VMLA.F32    q12, q6, d6[1]
        VLDR        d11, [r9, 152]          // B0
        VMLA.F32    q14, q6, d7[1]

        # BLOCK 4
        VMLA.F32    q9, q7, d4[1]
        VLDR        d8, [r9, 128]           // B0
        VMLA.F32    q11, q7, d5[1]
        VLDR        d13, [r9, 168]          // B1
        VMLA.F32    q13, q7, d6[1]
        VLDR        d10, [r9, 144]          // B0

        # BLOCK 5
        VMLA.F32    q15, q7, d7[1]
        VLDR        d15, [r9, 184]          // B1
        ADD         r9, r9, 128             // B++
        BHS         2b

        # Epilogue - 4 floats of A (16 bytes)
3:
        # First group of 16 FMA, Second group loads
        # BLOCK 0
        VMLA.F32    q8, q4, d0[0]
        VLD1.32     {d4}, [r3]!             // A0
        VMLA.F32    q10, q4, d1[0]
        VLD1.32     {d5}, [r12]!            // A1
        VMLA.F32    q12, q4, d2[0]

        # BLOCK 1
        VMLA.F32    q14, q4, d3[0]
        VLDR        d12, [r9, 32]           // B1
        VMLA.F32    q9, q5, d0[0]
        VLDR        d9, [r9, 72]            // B0
        VMLA.F32    q11, q5, d1[0]

        # BLOCK 2
        VMLA.F32    q13, q5, d2[0]
        VLD1.32     {d6}, [r10]!            // A2
        VMLA.F32    q15, q5, d3[0]
        VLD1.32     {d7}, [r7]!             // A3
        VMLA.F32    q8, q6, d0[1]

        # BLOCK 3
        VMLA.F32    q10, q6, d1[1]
        VLDR        d14, [r9, 48]           // B1
        VMLA.F32    q12, q6, d2[1]
        VLDR        d11, [r9, 88]           // B0
        VMLA.F32    q14, q6, d3[1]

        # BLOCK 4
        VMLA.F32    q9, q7, d0[1]
        VLDR        d8, [r9, 64]            // B0
        VMLA.F32    q11, q7, d1[1]
        VLDR        d13, [r9, 104]          // B1
        VMLA.F32    q13, q7, d2[1]
        VLDR        d10, [r9, 80]           // B0

        # BLOCK 5
        VMLA.F32    q15, q7, d3[1]
        VLDR        d15, [r9, 120]          // B1

        # Second group of 16 FMA, First group of loads
        # BLOCK 0
        VMLA.F32    q8, q4, d4[0]
        VLDR        d12, [r9, 96]           // B1
        VMLA.F32    q10, q4, d5[0]
        VMLA.F32    q12, q4, d6[0]

        # BLOCK 1
        VMLA.F32    q14, q4, d7[0]
        VLDR        d14, [r9, 112]          // B1
        VMLA.F32    q9, q5, d4[0]
        VMLA.F32    q11, q5, d5[0]

        # BLOCK 2
        VMLA.F32    q13, q5, d6[0]
        VMLA.F32    q15, q5, d7[0]
        VMLA.F32    q8, q6, d4[1]
        ADD         r9, r9, 128             // B++

        # BLOCK 3
        VMLA.F32    q10, q6, d5[1]
        VMLA.F32    q12, q6, d6[1]
        VMLA.F32    q14, q6, d7[1]
        TST         r5, 15

        # BLOCK 4
        VMLA.F32    q9, q7, d4[1]
        VMLA.F32    q11, q7, d5[1]
        VMLA.F32    q13, q7, d6[1]

        # BLOCK 5
        VMLA.F32    q15, q7, d7[1]

        # Is there a remainder?- 1 to 3 floats of A (4, 8 or 12 bytes)
        BNE         5f

        .p2align    3
4:
        LDR         r5, [sp, 104]           // a
        SUBS        r14, r14, 16            // ks -= MR * sizeof(void*)

        # ks loop
        BHI         1b

        # Load params pointer
        LDR         r14, [sp, 132]          // params
        # Load min/max values
        VLD1.32     {d4[],d5[]}, [r14]!
        VLD1.32     {d6[],d7[]}, [r14]
        SUBS        r1, r1, 8
        LDR         r0, [sp, 120]           // cn_stride

        # Clamp
        VMAX.F32    q8,  q8, q2
        VMAX.F32    q9,  q9, q2
        VMAX.F32    q10, q10, q2
        VMAX.F32    q11, q11, q2
        VMAX.F32    q12, q12, q2
        VMAX.F32    q13, q13, q2
        VMAX.F32    q14, q14, q2
        VMAX.F32    q15, q15, q2
        VMIN.F32    q8,  q8, q3
        VMIN.F32    q9,  q9, q3
        VMIN.F32    q10, q10, q3
        VMIN.F32    q11, q11, q3
        VMIN.F32    q12, q12, q3
        VMIN.F32    q13, q13, q3
        VMIN.F32    q14, q14, q3
        VMIN.F32    q15, q15, q3

        # Store full 4 x 8
        LDR         r14, [sp, 64]           // p = ks
        BLO         7f
        VST1.32     {d28-d31},  [r6], r0
        VST1.32     {d24-d27},  [r8], r0
        VST1.32     {d20-d23},  [r4], r0
        VST1.32     {d16-d19}, [r11], r0

        SUB         r5, r5, r14             // a -= ks

        BHI         0b

        VPOP        {d8-d15}
        ADD         sp, sp, 4               // skip r3
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

        .p2align    3
5:
        # Is there a remainder?- 2 floats of A (8 bytes)
        TST         r5, 8
        BEQ         6f

        # Remainder - 2 floats of A (8 bytes)
        VLD1.32     {d0}, [r3]!             // A0
        VLDM        r9!, {d8-d11}           // B0
        VLD1.32     {d1}, [r12]!            // A1
        VLD1.32     {d2}, [r10]!            // A2
        VLD1.32     {d3}, [ r7]!            // A3

        VMLA.F32    q8, q4, d0[0]
        VMLA.F32    q9, q5, d0[0]
        VMLA.F32    q10, q4, d1[0]
        VMLA.F32    q11, q5, d1[0]
        VLDM        r9!, {d12-d15}          // B1
        VMLA.F32    q12, q4, d2[0]
        VMLA.F32    q13, q5, d2[0]
        VMLA.F32    q14, q4, d3[0]
        VMLA.F32    q15, q5, d3[0]
        VMLA.F32    q8, q6, d0[1]
        VMLA.F32    q9, q7, d0[1]
        VMLA.F32    q10, q6, d1[1]
        VMLA.F32    q11, q7, d1[1]
        VMLA.F32    q12, q6, d2[1]
        VMLA.F32    q13, q7, d2[1]
        VMLA.F32    q14, q6, d3[1]
        VMLA.F32    q15, q7, d3[1]

        # Is there a remainder?- 1 float of A (4 bytes)
        TST         r5, 4
        BEQ         4b

6:
        # Remainder- 1 float of A (4 bytes)
        VLDM        r3!, {s0}               // A0
        VLDM        r9!, {d8-d11}           // B0
        VLDM        r12!, {s2}              // A1
        VLDM        r10!, {s4}              // A2
        VLDM        r7!, {s6}               // A3
        VMLA.F32    q8, q4, d0[0]
        VMLA.F32    q9, q5, d0[0]
        VMLA.F32    q10, q4, d1[0]
        VMLA.F32    q11, q5, d1[0]
        VMLA.F32    q12, q4, d2[0]
        VMLA.F32    q13, q5, d2[0]
        VMLA.F32    q14, q4, d3[0]
        VMLA.F32    q15, q5, d3[0]
        B           4b

        # Store odd width
7:
        TST         r1, 4
        BEQ         8f
        VST1.32     {d28-d29},  [r6]!
        VST1.32     {d24-d25},  [r8]!
        VMOV        q14, q15
        VMOV        q12, q13
        VST1.32     {d20-d21},  [r4]!
        VST1.32     {d16-d17}, [r11]!
        VMOV        q10, q11
        VMOV        q8,  q9

8:
        TST         r1, 2
        BEQ         9f
        VST1.32     {d28},  [r6]!
        VST1.32     {d24},  [r8]!
        VMOV        d28, d29
        VMOV        d24, d25
        VST1.32     {d20},  [r4]!
        VST1.32     {d16}, [r11]!
        VMOV        d20, d21
        VMOV        d16, d17

9:
        TST         r1, 1
        BEQ         10f
        VST1.32     {d28[0]},  [r6]!
        VST1.32     {d24[0]},  [r8]!
        VST1.32     {d20[0]},  [r4]!
        VST1.32     {d16[0]}, [r11]!

10:
        VPOP        {d8-d15}
        ADD         sp, sp, 4               // skip r3
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

END_FUNCTION xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55
# LINT.ThenChange(gen/f32-igemm-4x8-aarch32-neon-cortex-a55.cc)

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
