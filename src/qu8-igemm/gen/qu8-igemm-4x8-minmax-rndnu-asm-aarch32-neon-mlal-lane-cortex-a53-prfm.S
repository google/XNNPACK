// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

.syntax unified

// void xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm
//     size_t mr,                                     (r0)
//     size_t nc,                                      r1 -> sp + 56
//     size_t kc,                                     (r2) -> r5 -> sp + 60
//     size_t ks,                                     (r3) -> sp + 64 -> r14
//     const uint8_t** restrict a,            sp + 104  -> r2
//     const void* restrict w,              sp + 108  -> r9
//     uint8_t* restrict c,                   sp + 112  -> r11
//     size_t cm_stride,                   sp + 116  -> (r6)
//     size_t cn_stride,                   sp + 120  -> (r7)
//     size_t a_offset,                    sp + 124 -> (r5)
//     const uint8_t* zero,                  sp + 128 -> (r7)
//     xnn_qu8_conv_minmax_params*params); sp + 132 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0-d1 q0
// A1  r12  d2-d3 q1
// A2  r10  d4-d5 q2
// A3   r0  d6-d7 q3
// B    r9  d8-d9 q4 q5
// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15

// r1,r7 A53 gpr temporary loads
// unused d15

// params structure is 20 bytes
//  struct {
//    uint8_t kernel_zero_point[4];  d14
//    int32_t right_pre_shift;       d12[0]
//    int32_t multiplier;            d12[1]
//    int32_t right_post_shift;      d13[0]
//    int16_t output_zero_point;     d13[2]
//    uint8_t output_min;            d13[6]
//    uint8_t output_max;            d13[7]
//  } rndnu_neon;

BEGIN_FUNCTION xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm
        # Push 104 bytes
        # r1, r2 will be reloaded in outer loop.  r3 is ks
        PUSH        {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, lr}      // +48
        VPUSH       {d8-d14}                            // +56 = 104

        LDR         r11, [sp, 112]          // c
        LDR         r6, [sp, 116]           // cm_stride
        LDR         r2, [sp, 104]           // a
        LDR         r9, [sp, 108]           // w
        LDR         r5, [sp, 132]           // params
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

        # Load params values
        VLD1.32     {d14[]}, [r5]!          // QU8 kernel_zero_point
        VLDM        r5, {d12-d13}           // RNDNU params

        PLD         [r9,  64]               // Prefetch B
        PLD         [r9, 128]
        PLD         [r9, 192]
        PLD         [r9, 256]
        PLD         [r9, 320]
        PLD         [r9, 384]

        .p2align    3
0:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}          // Bias
        VMOV        q10, q8
        VMOV        q11, q9
        STR         r1, [sp, 56]            // save nc
        VMOV        q12, q8
        VMOV        q13, q9
        VMOV        q14, q8
        VMOV        q15, q9

        .p2align    3
1:
        # Load next 4 A pointers
        LDR         r3, [r2,  0]
        LDR         r12, [r2,  4]
        LDR         r10, [r2,  8]
        LDR         r0, [r2, 12]

        # Add a_offset
        LDR         r5, [sp, 124]           // a_offset
        LDR         r7, [sp, 128]           // zero
        ADD         r2, r2, 16
        CMP         r3,  r7                 // if a0 == zero
        ADD         r3,  r3, r5             // a0 += a_offset
        MOVEQ       r3,  r7                 //   a0 = zero, else += a0 + a_offset
        CMP         r12,  r7                // if a1 == zero
        ADD         r12, r12, r5            // a1 += a_offset
        MOVEQ       r12,  r7                //   a1 = zero, else += a1 + a_offset
        CMP         r10,  r7                // if a2 == zero
        ADD         r10, r10, r5            // a2 += a_offset
        MOVEQ       r10,  r7                //   a2 = zero, else += a2 + a_offset
        CMP         r0,  r7                 // if a3 == zero
        ADD         r0,  r0, r5             // a3 += a_offset
        LDR         r5, [sp, 60]            // kc
        MOVEQ       r0,  r7                 //   a3 = zero, else += a3 + a_offset
        SUBS        r5, r5, 8               // kc - 8
        BLO         5f                      // less than 8 channels?

        // Prologue - load 4A's and B0
        VLD1.8      {d0},  [r3]!            // A0
        VLD1.8      {d8},  [r9]!            // B0
        SUBS        r5, r5, 8               // k = k - 8
        VLD1.8      {d2}, [r12]!            // A1
        VLD1.8      {d4}, [r10]!            // A2
        VLD1.8      {d6},  [r0]!            // A3
        BLO         3f                      // less than 8 channels?

        // Main loop - 8 bytes
        // 64 bytes for weights.
        // 5 VMOVL = 4 A and 1 B = 5 cycles
        // 7 blocks with VLD B, VMOVL, 8 VMLA = 10 cycles
        // 1 blocks with VLD B, VMLA = 9 cycles
        // total = 84 cycles
        .p2align    3
2:
        // Extend - 5 cycles
        VMOVL.U8 q0, d0
        PLD         [r3, 128]
        VSUBL.U8    q4, d8, d14
        PLD         [r9, 448]
        VMOVL.U8 q1, d2
        PLD         [r12, 128]
        VMOVL.U8 q2, d4
        PLD         [r0, 128]
        VMOVL.U8 q3, d6
        PLD         [r10, 128]

        // BLOCK 0 - 10 cycles
        VLD1.8      {d10},  [r9]!           // B1
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        VMLAL.S16   q10, d8, d2[0]
        VMLAL.S16   q11, d9, d2[0]
        VSUBL.U8    q5, d10, d14
        VMLAL.S16   q12, d8, d4[0]
        VMLAL.S16   q13, d9, d4[0]
        VMLAL.S16   q14, d8, d6[0]
        VMLAL.S16   q15, d9, d6[0]

        // BLOCK 1 - 10 cycles
        VLD1.8      {d8},  [r9]!            // B2
        VMLAL.S16   q8, d10, d0[1]
        VMLAL.S16   q9, d11, d0[1]
        VMLAL.S16   q10, d10, d2[1]
        VMLAL.S16   q11, d11, d2[1]
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q12, d10, d4[1]
        VMLAL.S16   q13, d11, d4[1]
        VMLAL.S16   q14, d10, d6[1]
        VMLAL.S16   q15, d11, d6[1]

        // BLOCK 2 - 10 cycles
        VLD1.8      {d10},  [r9]!           // B3
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        VMLAL.S16   q10, d8, d2[2]
        VMLAL.S16   q11, d9, d2[2]
        VSUBL.U8    q5, d10, d14
        VMLAL.S16   q12, d8, d4[2]
        VMLAL.S16   q13, d9, d4[2]
        VMLAL.S16   q14, d8, d6[2]
        VMLAL.S16   q15, d9, d6[2]

        // BLOCK 3 - 10 cycles
        VLD1.8      {d8},  [r9]!            // B4
        VMLAL.S16   q8, d10, d0[3]
        VMLAL.S16   q9, d11, d0[3]
        VMLAL.S16   q10, d10, d2[3]
        VMLAL.S16   q11, d11, d2[3]
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q12, d10, d4[3]
        LDR         r1, [r3]                // A0 low
        VMLAL.S16   q13, d11, d4[3]
        LDR         r7, [r3, 4]             // A0 high
        VMLAL.S16   q14, d10, d6[3]
        ADD         r3, r3, 8
        VMLAL.S16   q15, d11, d6[3]

        // BLOCK 4 - 10 cycles
        VLD1.8      {d10},  [r9]!           // B5
        VMOV        d0, r1, r7              // A0 VMOV
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        VMLAL.S16   q10, d8, d3[0]
        VMLAL.S16   q11, d9, d3[0]
        VSUBL.U8    q5, d10, d14
        VMLAL.S16   q12, d8, d5[0]
        LDR         r1, [r12]               // A1 low
        VMLAL.S16   q13, d9, d5[0]
        LDR         r7, [r12, 4]            // A1 high
        VMLAL.S16   q14, d8, d7[0]
        ADD         r12, r12, 8
        VMLAL.S16   q15, d9, d7[0]

        // BLOCK 5 - 10 cycles
        VLD1.8      {d8},  [r9]!            // B6
        VMOV        d2, r1, r7              // A1 VMOV
        VMLAL.S16   q8, d10, d1[1]
        VMLAL.S16   q9, d11, d1[1]
        VMLAL.S16   q10, d10, d3[1]
        VMLAL.S16   q11, d11, d3[1]
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q12, d10, d5[1]
        LDR         r1, [r10]               // A2 low
        VMLAL.S16   q13, d11, d5[1]
        LDR         r7, [r10, 4]            // A2 high
        VMLAL.S16   q14, d10, d7[1]
        ADD         r10, r10, 8
        VMLAL.S16   q15, d11, d7[1]

        // BLOCK 6 - 10 cycles
        VLD1.8      {d10},  [r9]!           // B7
        VMOV        d4, r1, r7              // A2 VMOV
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        VMLAL.S16   q10, d8, d3[2]
        VMLAL.S16   q11, d9, d3[2]
        VSUBL.U8    q5, d10, d14
        VMLAL.S16   q12, d8, d5[2]
        LDR         r1, [r0]                // A3 low
        VMLAL.S16   q13, d9, d5[2]
        LDR         r7, [r0, 4]             // A3 high
        VMLAL.S16   q14, d8, d7[2]
        ADD         r0, r0, 8
        VMLAL.S16   q15, d9, d7[2]

        // BLOCK 7 - 9 cycles
        VLD1.8      {d8},  [r9]!            // B0
        VMOV        d6, r1, r7              // A3 VMOV
        VMLAL.S16   q8, d10, d1[3]
        VMLAL.S16   q9, d11, d1[3]
        VMLAL.S16   q10, d10, d3[3]
        VMLAL.S16   q11, d11, d3[3]
        VMLAL.S16   q12, d10, d5[3]
        VMLAL.S16   q13, d11, d5[3]
        SUBS        r5, r5, 8
        VMLAL.S16   q14, d10, d7[3]
        VMLAL.S16   q15, d11, d7[3]
        BHS         2b

        // Epilogue

        .p2align    3
3:
        VMOVL.U8 q0, d0
        VSUBL.U8    q4, d8, d14
        VMOVL.U8 q1, d2
        VMOVL.U8 q2, d4
        VMOVL.U8 q3, d6

        VLD1.8      {d10},  [r9]!           // B1
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        VMLAL.S16   q10, d8, d2[0]
        VMLAL.S16   q11, d9, d2[0]
        VSUBL.U8    q5, d10, d14
        VMLAL.S16   q12, d8, d4[0]
        VMLAL.S16   q13, d9, d4[0]
        VMLAL.S16   q14, d8, d6[0]
        VMLAL.S16   q15, d9, d6[0]

        VLD1.8      {d8},  [r9]!            // B2
        VMLAL.S16   q8, d10, d0[1]
        VMLAL.S16   q9, d11, d0[1]
        VMLAL.S16   q10, d10, d2[1]
        VMLAL.S16   q11, d11, d2[1]
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q12, d10, d4[1]
        VMLAL.S16   q13, d11, d4[1]
        VMLAL.S16   q14, d10, d6[1]
        VMLAL.S16   q15, d11, d6[1]

        VLD1.8      {d10},  [r9]!           // B3
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        VMLAL.S16   q10, d8, d2[2]
        VMLAL.S16   q11, d9, d2[2]
        VSUBL.U8    q5, d10, d14
        VMLAL.S16   q12, d8, d4[2]
        VMLAL.S16   q13, d9, d4[2]
        VMLAL.S16   q14, d8, d6[2]
        VMLAL.S16   q15, d9, d6[2]

        VLD1.8      {d8},  [r9]!            // B4
        VMLAL.S16   q8, d10, d0[3]
        VMLAL.S16   q9, d11, d0[3]
        VMLAL.S16   q10, d10, d2[3]
        VMLAL.S16   q11, d11, d2[3]
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q12, d10, d4[3]
        VMLAL.S16   q13, d11, d4[3]
        VMLAL.S16   q14, d10, d6[3]
        VMLAL.S16   q15, d11, d6[3]

        VLD1.8      {d10},  [r9]!           // B5
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        VMLAL.S16   q10, d8, d3[0]
        VMLAL.S16   q11, d9, d3[0]
        VSUBL.U8    q5, d10, d14
        VMLAL.S16   q12, d8, d5[0]
        VMLAL.S16   q13, d9, d5[0]
        VMLAL.S16   q14, d8, d7[0]
        VMLAL.S16   q15, d9, d7[0]

        VLD1.8      {d8},  [r9]!            // B6
        VMLAL.S16   q8, d10, d1[1]
        VMLAL.S16   q9, d11, d1[1]
        VMLAL.S16   q10, d10, d3[1]
        VMLAL.S16   q11, d11, d3[1]
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q12, d10, d5[1]
        VMLAL.S16   q13, d11, d5[1]
        VMLAL.S16   q14, d10, d7[1]
        VMLAL.S16   q15, d11, d7[1]

        VLD1.8      {d10},  [r9]!           // B7
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        VMLAL.S16   q10, d8, d3[2]
        VMLAL.S16   q11, d9, d3[2]
        VSUBL.U8    q5, d10, d14
        VMLAL.S16   q12, d8, d5[2]
        VMLAL.S16   q13, d9, d5[2]
        VMLAL.S16   q14, d8, d7[2]
        VMLAL.S16   q15, d9, d7[2]

        VMLAL.S16   q8, d10, d1[3]
        VMLAL.S16   q9, d11, d1[3]
        VMLAL.S16   q10, d10, d3[3]
        VMLAL.S16   q11, d11, d3[3]
        VMLAL.S16   q12, d10, d5[3]
        VMLAL.S16   q13, d11, d5[3]
        ADDS        r5, r5, 8
        VMLAL.S16   q14, d10, d7[3]
        VMLAL.S16   q15, d11, d7[3]

        # Is there a remainder?- 1-7 bytes of A
        BNE         6f

4:
        # ks loop
        SUBS        r14, r14, 16            // ks -= MR * sizeof(void*)
        BHI         1b

        LDR         r7, [sp, 120]           // cn_stride
        LDR         r14, [sp, 64]           // p = ks

        # RNDNU quantization
        VDUP.32     q0, d12[0]              // right_pre_shift

        VQSHL.S32   q8,  q8, q0
        VQSHL.S32   q9,  q9, q0
        VQSHL.S32   q10, q10, q0
        VQSHL.S32   q11, q11, q0
        VQSHL.S32   q12, q12, q0
        VQSHL.S32   q13, q13, q0
        VQSHL.S32   q14, q14, q0
        VQSHL.S32   q15, q15, q0

        VDUP.32     q2, d13[0]              // right_post_shift

        VQDMULH.S32 q8,  q8, d12[1]     // multiplier
        VQDMULH.S32 q9,  q9, d12[1]
        VQDMULH.S32 q10, q10, d12[1]
        VQDMULH.S32 q11, q11, d12[1]
        VQDMULH.S32 q12, q12, d12[1]
        VQDMULH.S32 q13, q13, d12[1]
        VQDMULH.S32 q14, q14, d12[1]
        VQDMULH.S32 q15, q15, d12[1]

        VRSHL.S32   q8,  q8, q2
        VRSHL.S32   q9,  q9, q2
        VRSHL.S32   q10, q10, q2
        VRSHL.S32   q11, q11, q2
        VRSHL.S32   q12, q12, q2
        VRSHL.S32   q13, q13, q2
        VRSHL.S32   q14, q14, q2
        VRSHL.S32   q15, q15, q2

        VDUP.16     q0, d13[2]              // output_zero_point

        VQMOVN.S32  d16, q8
        VQMOVN.S32  d17, q9
        VQMOVN.S32  d18, q10
        VQMOVN.S32  d19, q11
        VQMOVN.S32  d20, q12
        VQMOVN.S32  d21, q13
        VQMOVN.S32  d22, q14
        VQMOVN.S32  d23, q15

        VQADD.S16   q8,  q8, q0
        VQADD.S16   q9,  q9, q0
        VQADD.S16   q10, q10, q0
        VQADD.S16   q11, q11, q0

        LDR         r1, [sp, 56]            // restore nc
        VDUP.8      q12, d13[6]             // output_min

        VQMOVUN.S16 d0,  q8
        VQMOVUN.S16 d1,  q9
        VQMOVUN.S16 d2, q10
        VQMOVUN.S16 d3, q11

        VDUP.8      q13, d13[7]             // output_max

        VMAX.U8 q0, q0, q12
        VMAX.U8 q1, q1, q12

        SUBS        r1, r1, 8               // nc -= 8

        VMIN.U8 q0, q0, q13
        VMIN.U8 q1, q1, q13

        # Store full 4 x 8
        BLO         7f
        VST1.8      {d3}, [r6], r7
        VST1.8      {d2}, [r8], r7
        VST1.8      {d1}, [r4], r7
        VST1.8      {d0}, [r11], r7
        SUB         r2, r2, r14             // a -= ks
        BHI         0b

        VPOP        {d8-d14}
        ADD         sp, sp, 12              // skip r1, r2, r3
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

        # Remainder- 1 to 7 bytes of A
        .p2align    3
5:
        AND         r5, r5, 7               // kc remainder 1 to 7
6:
        VLD1.8      {d0},  [r3]
        VLD1.8      {d8},  [r9]!
        VLD1.8      {d2}, [r12]
        VLD1.8      {d4}, [r10]
        VLD1.8      {d6},  [r0]

        VMOVL.U8 q0, d0
        VSUBL.U8    q4, d8, d14
        VMOVL.U8 q1, d2
        VMOVL.U8 q2, d4
        VMOVL.U8 q3, d6
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        VMLAL.S16   q10, d8, d2[0]
        VMLAL.S16   q11, d9, d2[0]
        VMLAL.S16   q12, d8, d4[0]
        VMLAL.S16   q13, d9, d4[0]
        VMLAL.S16   q14, d8, d6[0]
        VMLAL.S16   q15, d9, d6[0]
        CMP         r5, 2
        BLO         4b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d0[1]
        VMLAL.S16   q9, d9, d0[1]
        VMLAL.S16   q10, d8, d2[1]
        VMLAL.S16   q11, d9, d2[1]
        VMLAL.S16   q12, d8, d4[1]
        VMLAL.S16   q13, d9, d4[1]
        VMLAL.S16   q14, d8, d6[1]
        VMLAL.S16   q15, d9, d6[1]
        BEQ         4b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        VMLAL.S16   q10, d8, d2[2]
        VMLAL.S16   q11, d9, d2[2]
        VMLAL.S16   q12, d8, d4[2]
        VMLAL.S16   q13, d9, d4[2]
        VMLAL.S16   q14, d8, d6[2]
        VMLAL.S16   q15, d9, d6[2]
        CMP         r5, 4
        BLO         4b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d0[3]
        VMLAL.S16   q9, d9, d0[3]
        VMLAL.S16   q10, d8, d2[3]
        VMLAL.S16   q11, d9, d2[3]
        VMLAL.S16   q12, d8, d4[3]
        VMLAL.S16   q13, d9, d4[3]
        VMLAL.S16   q14, d8, d6[3]
        VMLAL.S16   q15, d9, d6[3]
        BEQ         4b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        VMLAL.S16   q10, d8, d3[0]
        VMLAL.S16   q11, d9, d3[0]
        VMLAL.S16   q12, d8, d5[0]
        VMLAL.S16   q13, d9, d5[0]
        VMLAL.S16   q14, d8, d7[0]
        VMLAL.S16   q15, d9, d7[0]
        CMP         r5, 6
        BLO         4b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d1[1]
        VMLAL.S16   q9, d9, d1[1]
        VMLAL.S16   q10, d8, d3[1]
        VMLAL.S16   q11, d9, d3[1]
        VMLAL.S16   q12, d8, d5[1]
        VMLAL.S16   q13, d9, d5[1]
        VMLAL.S16   q14, d8, d7[1]
        VMLAL.S16   q15, d9, d7[1]
        BEQ         4b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        VMLAL.S16   q10, d8, d3[2]
        VMLAL.S16   q11, d9, d3[2]
        VMLAL.S16   q12, d8, d5[2]
        VMLAL.S16   q13, d9, d5[2]
        VMLAL.S16   q14, d8, d7[2]
        VMLAL.S16   q15, d9, d7[2]
        B           4b

        # Store odd width
        .p2align    3
7:
        TST         r1, 4
        BEQ         8f
        VST1.32     {d3[0]}, [r6]!
        VST1.32     {d2[0]}, [r8]!
        VST1.32     {d1[0]}, [r4]!
        VST1.32     {d0[0]}, [r11]!
        VEXT.8      q1, q1, q1, 4
        VEXT.8      q0, q0, q0, 4
8:
        TST         r1, 2
        BEQ         9f
        VST1.16     {d3[0]}, [r6]!
        VST1.16     {d2[0]}, [r8]!
        VST1.16     {d1[0]}, [r4]!
        VST1.16     {d0[0]}, [r11]!
        VEXT.8      q1, q1, q1, 2
        VEXT.8      q0, q0, q0, 2

9:
        TST         r1, 1
        BEQ         10f
        VST1.8      {d3[0]}, [r6]
        VST1.8      {d2[0]}, [r8]
        VST1.8      {d1[0]}, [r4]
        VST1.8      {d0[0]}, [r11]

10:
        VPOP        {d8-d14}
        ADD         sp, sp, 12              // skip r1, r2, r3
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

END_FUNCTION xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
