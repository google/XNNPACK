// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

.syntax unified

// void xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            (r2) -> r5
//     const uint8_t* restrict a,              r3
//     size_t a_stride,           sp + 96 -> (unused)
//     const void* restrict w,     sp + 100 -> r9
//     uint8_t* restrict c,         sp + 104 -> r11
//     size_t cm_stride,          sp + 108 -> (unused)
//     size_t cn_stride,          sp + 112 -> r7
//     xnn_qu8_conv_minmax_params params)  sp + 116 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Based on cortex_a53 microkernel but with Neon loads

// Register usage
// A0   r3  d0-d1 q0
// B    r9  d8-d9 q4 q5
// C0  r11 d16-d17  q8  d18-d19  q9
//         q2, q3 acc2
// unused r4, r6, r8, r10, r12, d15, q10-q15, q1-q3
# params structure is 20 bytes
#  struct {
#    uint8_t kernel_zero_point[4];  d14
#    int32_t right_pre_shift;       d12[0]
#    int32_t multiplier;            d12[1]
#    int32_t right_post_shift;      d13[0]
#    int16_t output_zero_point;     d13[2]
#    uint8_t output_min;            d13[6]
#    uint8_t output_max;            d13[7]
#  } rndnu_neon;

BEGIN_FUNCTION xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7
        # Push 96 bytes
        PUSH        {r5, r7, r9, r11}                   // 16
        SUB         sp, sp, 24                          // +24
        VPUSH       {d8-d14}                            // +56 = 96

        LDR         r11, [sp, 104]          // c
        LDR         r9, [sp, 100]           // w
        LDR         r5, [sp, 116]           // params

        # Load params values
        VLD1.32     {d14[]}, [r5]!          // QU8 kernel_zero_point
        VLDM        r5, {d12-d13}           // RNDNU params
        LDR         r7, [sp, 112]               // cn_stride


        .p2align    3
0:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}          // Bias
        VMOV.I32    q2, 0                   // second set of C for pipelining FMLA
        SUBS        r5, r2, 8               // k = kc - 8
        VMOV.I32    q3, 0
        BLO         4f                      // less than 8 channels?

        // Prologue - load A0 and B0
        VLD1.8      {d0},  [r3]!            // A0
        SUBS        r5, r5, 8               // k = k - 8
        VLD1.8      {d8},  [r9]!            // B0
        BLO         2f                      // less than 8 channels?

        // Main loop - 8 bytes
        // 64 bytes for weights.

        .p2align    3
1:
        // Extend
        VMOVL.U8 q0, d0
        VSUBL.U8    q4, d8, d14

        // BLOCK 0
        VLD1.8      {d10},  [r9]!           // B1
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        VSUBL.U8    q5, d10, d14

        // BLOCK 1
        VLD1.8      {d8},  [r9]!            // B2
        VMLAL.S16   q2, d10, d0[1]
        VMLAL.S16   q3, d11, d0[1]
        VSUBL.U8    q4, d8, d14

        // BLOCK 2
        VLD1.8      {d10},  [r9]!           // B3
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        VSUBL.U8    q5, d10, d14

        // BLOCK 3
        VLD1.8      {d8},  [r9]!            // B4
        VMLAL.S16   q2, d10, d0[3]
        VMLAL.S16   q3, d11, d0[3]
        VLD1.8      {d0},  [r3]!            // A0
        VSUBL.U8    q4, d8, d14

        // BLOCK 4
        VLD1.8      {d10},  [r9]!           // B5
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        VSUBL.U8    q5, d10, d14

        // BLOCK 5
        VLD1.8      {d8},  [r9]!            // B6
        VMLAL.S16   q2, d10, d1[1]
        VMLAL.S16   q3, d11, d1[1]
        VSUBL.U8    q4, d8, d14

        // BLOCK 6
        VLD1.8      {d10},  [r9]!           // B7
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        VSUBL.U8    q5, d10, d14

        // BLOCK 7
        VLD1.8      {d8},  [r9]!            // B0
        VMLAL.S16   q2, d10, d1[3]
        VMLAL.S16   q3, d11, d1[3]
        SUBS        r5, r5, 8
        BHS         1b

        // Epilogue

        .p2align    3
2:
        VMOVL.U8 q0, d0
        VSUBL.U8    q4, d8, d14

        VLD1.8      {d10},  [r9]!           // B1
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        VSUBL.U8    q5, d10, d14

        VLD1.8      {d8},  [r9]!            // B2
        VMLAL.S16   q2, d10, d0[1]
        VMLAL.S16   q3, d11, d0[1]
        VSUBL.U8    q4, d8, d14

        VLD1.8      {d10},  [r9]!           // B3
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        VSUBL.U8    q5, d10, d14

        VLD1.8      {d8},  [r9]!            // B4
        VMLAL.S16   q2, d10, d0[3]
        VMLAL.S16   q3, d11, d0[3]
        VSUBL.U8    q4, d8, d14

        VLD1.8      {d10},  [r9]!           // B5
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        VSUBL.U8    q5, d10, d14

        VLD1.8      {d8},  [r9]!            // B6
        VMLAL.S16   q2, d10, d1[1]
        VMLAL.S16   q3, d11, d1[1]
        VSUBL.U8    q4, d8, d14

        VLD1.8      {d10},  [r9]!           // B7
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        VSUBL.U8    q5, d10, d14
        ADDS        r5, r5, 8

        VMLAL.S16   q2, d10, d1[3]
        VMLAL.S16   q3, d11, d1[3]

        # Is there a remainder?- 1-7 bytes of A
        BNE         4f

3:
        VADD.S32    q8, q8, q2
        VADD.S32    q9, q9, q3

        # RNDNU quantization
        VDUP.32     q0, d12[0]              // right_pre_shift

        VQSHL.S32   q8,  q8, q0
        VQSHL.S32   q9,  q9, q0

        VDUP.32     q2, d13[0]              // right_post_shift

        VQDMULH.S32 q8,  q8, d12[1]     // multiplier
        VQDMULH.S32 q9,  q9, d12[1]

        VRSHL.S32   q8,  q8, q2
        VRSHL.S32   q9,  q9, q2

        VDUP.16     q0, d13[2]              // output_zero_point

        VQMOVN.S32  d16, q8
        VQMOVN.S32  d17, q9

        VQADD.S16   q8,  q8, q0

        VDUP.8      d24, d13[6]             // output_min

        VQMOVUN.S16 d0,  q8

        VDUP.8      d25, d13[7]             // output_max

        VMAX.U8 d0, d0, d24

        SUBS        r1, r1, 8

        VMIN.U8 d0, d0, d25

        # Store full 1 x 8
        BLO         5f
        VST1.8      {d0}, [r11], r7
        SUB         r3, r3, r2
        BHI         0b

        VPOP        {d8-d14}
        ADD         sp, sp, 8               // skip pad of 8
        ADD         sp, sp, 16
        POP         {r5, r7, r9, r11}
        BX          lr

        # Remainder- 1 to 7 bytes of A
        .p2align    3
4:
        AND         r5, r5, 7               // kc remainder 1 to 7

        VLD1.8      {d0},  [r3], r5
        VLD1.8      {d8},  [r9]!

        VMOVL.U8 q0, d0
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d0[0]
        VMLAL.S16   q9, d9, d0[0]
        CMP         r5, 2
        BLO         3b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d0[1]
        VMLAL.S16   q9, d9, d0[1]
        BEQ         3b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d0[2]
        VMLAL.S16   q9, d9, d0[2]
        CMP         r5, 4
        BLO         3b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d0[3]
        VMLAL.S16   q9, d9, d0[3]
        BEQ         3b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d1[0]
        VMLAL.S16   q9, d9, d1[0]
        CMP         r5, 6
        BLO         3b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d1[1]
        VMLAL.S16   q9, d9, d1[1]
        BEQ         3b

        VLD1.8      {d8},  [r9]!
        VSUBL.U8    q4, d8, d14
        VMLAL.S16   q8, d8, d1[2]
        VMLAL.S16   q9, d9, d1[2]
        B           3b

        # Store odd width
        .p2align    3
5:
        TST         r1, 4
        BEQ         6f
        VST1.32     {d0[0]}, [r11]!
        VEXT.8      q0, q0, q0, 4
6:
        TST         r1, 2
        BEQ         7f
        VST1.16     {d0[0]}, [r11]!
        VEXT.8      q0, q0, q0, 2
7:
        TST         r1, 1
        BEQ         8f
        VST1.8      {d0[0]}, [r11]
8:
        VPOP        {d8-d14}
        ADD         sp, sp, 8               // skip pad of 8
        ADD         sp, sp, 16
        POP         {r5, r7, r9, r11}
        BX          lr

END_FUNCTION xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

