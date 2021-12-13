// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5
//     const uint8_t*restrict a,             r3
//     size_t a_stride,          sp + 96  -> (r7)
//     const void*restrict w,    sp + 100 -> r9
//     uint8_t*restrict c,       sp + 104 -> r11
//     size_t cm_stride,         sp + 108 -> (r6)
//     size_t cn_stride,         sp + 112 -> r7
//     const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  sp + 116 -> (r7)


// inner loop registers

// A0  r7  d0
// A1  r6  d1
// A2  r2  d2
// A3  r3  d3

// B    r9  d8,  d9, d10, d11
// B       d12, d13, d14, d15

// C3  ip r12  [240]q9  q5
// C2  sl r10       q15 q7
// C1  fp r11       q2  q4
// C0  r0           q6  q14

BEGIN_FUNCTION xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64
        .arm
#ifndef __APPLE__
        .arch   armv7-a
        .fpu    neon
#endif
        # Push 104 bytes.  r2 is for kc reset
        VPUSH   {d8-d15}      // 64 bytes
        PUSH    {r2, r4, r5, r6, r7, r8, r9, sl, fp, lr}   // 40 bytes
        SUB     sp, sp, #456  // +456 = 560 bytes.  TODO eliminate
        MOV     lr, r1
        LDR     r1, [sp, #560]
        CMP     r0, #2
        MOV     r4, r3
        ADDCS   r4, r4, r1
        CMP     r0, #3
        LDR     r9, [sp, #580]
        MOV     r8, #15
        MOV     r6, r4
        LDR     ip, [sp, #568]
        ADDCS   r6, r6, r1
        CMP     r0, #4
        LDR     r5, [sp, #572]
        MOV     r7, r6
        MOV     sl, ip
        ADDEQ   r7, r7, r1
        MOV     r1, r9
        VLD1.32 {d16-d17}, [r1], r8
        CMP     r0, #2
        ADDCS   sl, sl, r5
        CMP     r0, #3
        VLD1.8  {d18-d19}, [r1]
        ADD     r1, r9, #4
        MOV     fp, sl
        VLD1.32 {d20-d21}, [r1]
        ADD     r1, r9, #8
        ADDCS   fp, fp, r5
        CMP     r0, #4
        VLD1.64 {d22-d23}, [r1]
        ADD     r1, r9, #12
        MOV     r0, fp
        VLD1.32 {d24-d25}, [r1]
        ADD     r1, r9, #14
        ADDEQ   r0, r0, r5
        MOV     r9, #32
        VLD1.16 {d26-d27}, [r1]
        MOV     r1, r2
        MOV     r2, r4
        ADD     r4, sp, #128
        VDUP.32 q0, d16[0]
        LDR     r5, [sp, #576]
        LDR     r8, [sp, #564]  // w
        VDUP.8  q8, d26[0]
        STR     r1, [sp, #56]
        VDUP.8  q1, d18[0]
        VSTMIA  r4, {d16-d17}
        ADD     r4, sp, #112
        VDUP.16 q8, d24[0]
        VSTMIA  r4, {d16-d17}
        ADD     r4, sp, #96
        VDUP.32 q8, d22[0]
        VSTMIA  r4, {d16-d17}
        ADD     r4, sp, #80
        VDUP.32 q8, d20[0]
        VSTMIA  r4, {d16-d17}
        ADD     r4, sp, #32
        VSTMIA  r4, {d0-d1}
        ADD     r4, sp, #16
        VSTMIA  r4, {d2-d3}
0:
        # Load initial bias from w into accumulators
        ADD     r4, r8, #16
        VLD1.8  {d16-d17}, [r8], r9   // Bias
        CMP     r1, #8
        STR     lr, [sp, #12]
        ADD     lr, sp, #240
        VLD1.8  {d10-d11}, [r4]
        VSTMIA  lr, {d16-d17}
        LDR     lr, [sp, #12]
        BCC     2f                     // less than 8 channels?  skip main loop
        STR     lr, [sp, #64]
        ADD     lr, sp, #240
        VORR    q7, q5, q5
        STR     ip, [sp, #68]
        VLDMIA  lr, {d6-d7}
        VORR    q14, q5, q5
        VORR    q6, q5, q5
        MOV     ip, #0
        VORR    q15, q3, q3
        MOV     r4, r1
        VORR    q4, q3, q3
        STR     r0, [sp, #72]
        STR     fp, [sp, #76]
        STR     sl, [sp, #60]
        STR     r7, [sp, #156]

        # Main loop - 8 bytes of A

1:
        MOV     r9, r4
        MOV     r4, r7
        LDR     lr, [r8, #4]
        MOV     r7, r6
        MOV     r6, r2
        LDR     sl, [r4, ip]!
        LDR     fp, [r8]
        LDR     r2, [r8, #8]
        LDR     r1, [r8, #12]
        STR     lr, [sp, #364]
        ADD     lr, sp, #192
        LDR     r0, [r8, #24]
        STR     fp, [sp, #360]
        LDR     r5, [r8, #20]
        STR     r2, [sp, #384]
        ADD     r2, sp, #360
        STR     r1, [sp, #388]
        VLD1.8  {d16}, [r2 :64]
        ADD     r2, sp, #384
        VLD1.8  {d17}, [r2 :64]
        VMOVL.S8 q10, d16
        LDR     r1, [r8, #28]
        LDR     r2, [r8, #16]
        STR     sl, [sp, #416]
        STR     r1, [sp, #380]
        STR     r0, [sp, #376]
        STR     r5, [sp, #372]
        STR     r2, [sp, #368]
        LDR     r0, [r4, #4]
        MOV     r4, r9
        STR     r0, [sp, #420]
        ADD     r0, sp, #416
        SUB     r4, r9, #8
        VLD1.8  {d18}, [r0 :64]
        ADD     r0, sp, #376
        CMP     r4, #7
        VMOVL.S8 q0, d18
        VLD1.8  {d16}, [r0 :64]
        VMOVL.S8 q9, d17
        ADD     r0, sp, #368
        VMLAL.S16 q6, d21, d0[0]
        VLD1.8  {d17}, [r0 :64]
        VORR    q11, q9, q9
        PLD     [r8, #480]
        LDR     r0, [r8, #32]
        VMLAL.S16 q3, d20, d0[0]
        LDR     r1, [r8, #36]
        VMLAL.S16 q6, d19, d0[1]
        LDR     r2, [r8, #40]
        VMOVL.S8 q9, d17
        LDR     r5, [r8, #44]
        STR     r0, [sp, #424]
        ADD     r0, sp, #424
        STR     r1, [sp, #428]
        VMLAL.S16 q6, d19, d0[2]
        VORR    q1, q9, q9
        VMOVL.S8 q9, d16
        VMLAL.S16 q6, d19, d0[3]
        VSTMIA  lr, {d18-d19}
        ADD     lr, sp, #176
        VORR    q9, q11, q11
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #432
        VMLAL.S16 q3, d18, d0[1]
        VMOVL.S8 q8, d16
        STR     r2, [sp, #432]
        STR     r5, [sp, #436]
        MOV     r2, r6
        MOV     r6, r7
        LDR     r7, [sp, #156]
        VMLAL.S16 q6, d17, d1[0]
        LDR     r1, [r8, #48]
        VSTMIA  lr, {d16-d17}
        ADD     lr, sp, #208
        VLD1.8  {d16}, [r0 :64]
        LDR     r0, [r8, #52]
        VMOVL.S8 q8, d16
        STR     r0, [sp, #444]
        ADD     r0, sp, #440
        STR     r1, [sp, #440]
        VLD1.8  {d24}, [r0 :64]
        VMLAL.S16 q6, d17, d1[1]
        VORR    q2, q8, q8
        LDR     r0, [r8, #60]
        VMOVL.S8 q8, d24
        LDR     r1, [r8, #56]
        ADD     r8, r8, #64
        STR     r0, [sp, #452]
        ADD     r0, sp, #448
        STR     r1, [sp, #448]
        VMLAL.S16 q6, d17, d1[2]
        VLD1.8  {d26}, [r0 :64]
        VORR    q12, q8, q8
        MOV     r0, r6
        VMOVL.S8 q13, d26
        LDR     r1, [r0, ip]!
        VMLAL.S16 q6, d27, d1[3]
        VSTMIA  lr, {d12-d13}
        VORR    q6, q10, q10
        VORR    q10, q1, q1
        ADD     lr, sp, #256
        VORR    q1, q12, q12
        VSTMIA  lr, {d22-d23}
        ADD     lr, sp, #192
        VMLAL.S16 q3, d20, d0[2]
        VLDMIA  lr, {d22-d23}
        ADD     lr, sp, #176
        VLDMIA  lr, {d16-d17}
        ADD     lr, sp, #160
        VMLAL.S16 q3, d22, d0[3]
        STR     r1, [sp, #408]
        LDR     r0, [r0, #4]
        STR     r0, [sp, #412]
        ADD     r0, sp, #408
        VMLAL.S16 q3, d16, d1[0]
        VMLAL.S16 q3, d4, d1[1]
        VMLAL.S16 q3, d24, d1[2]
        VORR    q12, q8, q8
        VMLAL.S16 q3, d26, d1[3]
        VLD1.8  {d0}, [r0 :64]
        MOV     r0, r2
        VMOVL.S8 q0, d0
        VSTMIA  lr, {d4-d5}
        ADD     lr, sp, #256
        LDR     r1, [r0, ip]!
        VMLAL.S16 q14, d13, d0[0]
        VMLAL.S16 q4, d12, d0[0]
        VMLAL.S16 q14, d19, d0[1]
        VORR    q9, q10, q10
        VMLAL.S16 q14, d21, d0[2]
        VORR    q10, q6, q6
        VMLAL.S16 q14, d23, d0[3]
        VMLAL.S16 q14, d17, d1[0]
        VORR    q8, q2, q2
        VLDMIA  lr, {d4-d5}
        ADD     lr, sp, #160
        VMLAL.S16 q4, d4, d0[1]
        STR     r1, [sp, #400]
        VMLAL.S16 q14, d17, d1[1]
        LDR     r0, [r0, #4]
        STR     r0, [sp, #404]
        ADD     r0, sp, #400
        VMLAL.S16 q4, d18, d0[2]
        VMLAL.S16 q14, d3, d1[2]
        VMLAL.S16 q4, d22, d0[3]
        VMLAL.S16 q14, d27, d1[3]
        VMLAL.S16 q4, d24, d1[0]
        VMLAL.S16 q4, d16, d1[1]
        VLDMIA  lr, {d16-d17}
        ADD     lr, sp, #224
        VMLAL.S16 q4, d2, d1[2]
        VMLAL.S16 q4, d26, d1[3]
        VLD1.8  {d0}, [r0 :64]
        MOV     r0, r3
        VMOVL.S8 q0, d0
        LDR     r1, [r0, ip]!
        ADD     ip, ip, #8
        VMLAL.S16 q7, d13, d0[0]
        VMLAL.S16 q15, d20, d0[0]
        VMLAL.S16 q7, d5, d0[1]
        VORR    q2, q9, q9
        VMLAL.S16 q7, d19, d0[2]
        VORR    q9, q1, q1
        VMLAL.S16 q7, d23, d0[3]
        VMLAL.S16 q7, d25, d1[0]
        VMLAL.S16 q7, d17, d1[1]
        VMLAL.S16 q7, d3, d1[2]
        VMLAL.S16 q7, d27, d1[3]
        VSTMIA  lr, {d14-d15}
        ADD     lr, sp, #240
        STR     r1, [sp, #392]
        LDR     r0, [r0, #4]
        STR     r0, [sp, #396]
        ADD     r0, sp, #392
        VLDMIA  lr, {d12-d13}
        ADD     lr, sp, #256
        VLD1.8  {d2}, [r0 :64]
        VMOVL.S8 q1, d2
        VLDMIA  lr, {d14-d15}
        ADD     lr, sp, #224
        VMLAL.S16 q15, d14, d0[1]
        VMLAL.S16 q6, d20, d2[0]
        VMLAL.S16 q5, d21, d2[0]
        VMLAL.S16 q15, d4, d0[2]
        VMLAL.S16 q6, d14, d2[1]
        VMLAL.S16 q5, d15, d2[1]
        VLDMIA  lr, {d14-d15}
        ADD     lr, sp, #240
        VMLAL.S16 q15, d22, d0[3]
        VMLAL.S16 q6, d4, d2[2]
        VMLAL.S16 q5, d5, d2[2]
        VMLAL.S16 q15, d24, d1[0]
        VMLAL.S16 q6, d22, d2[3]
        VMLAL.S16 q5, d23, d2[3]
        VMLAL.S16 q15, d16, d1[1]
        VMLAL.S16 q6, d24, d3[0]
        VMLAL.S16 q5, d25, d3[0]
        VMLAL.S16 q15, d18, d1[2]
        VMLAL.S16 q6, d16, d3[1]
        VMLAL.S16 q5, d17, d3[1]
        VMLAL.S16 q15, d26, d1[3]
        VMLAL.S16 q6, d18, d3[2]
        VMLAL.S16 q5, d19, d3[2]
        VMLAL.S16 q6, d26, d3[3]
        VMLAL.S16 q5, d27, d3[3]
        VSTMIA  lr, {d12-d13}
        ADD     lr, sp, #208
        VLDMIA  lr, {d12-d13}
        BHI     1b
        ADD     r5, sp, #32
        ADD     lr, sp, #56
        VORR    q2, q4, q4
        ADD     r7, r7, ip
        VLDMIA  r5, {d0-d1}
        ADD     r5, sp, #16
        VORR    q4, q14, q14
        ADD     r6, r6, ip
        VLDMIA  r5, {d2-d3}
        ADD     r2, r2, ip
        ADD     r3, r3, ip
        VORR    q14, q3, q3
        LDR     ip, [sp, #68]
        MOV     r9, #32
        LDR     fp, [sp, #76]
        LDR     r0, [sp, #72]
        LDR     r5, [sp, #576]
        LDM     lr, {r1, sl, lr}
        B       3f
2:
        STR     lr, [sp, #12]
        ADD     lr, sp, #240
        VORR    q15, q5, q5
        MOV     r4, r1
        VLDMIA  lr, {d16-d17}
        VORR    q6, q5, q5
        VORR    q4, q5, q5
        LDR     lr, [sp, #12]
        VORR    q14, q8, q8
        VORR    q2, q8, q8
        VORR    q7, q5, q5
        VORR    q15, q8, q8
3:
        CMP     r4, #0
        BNE     5f

        # rndnu quantization
        # C3 [240]q9  q5
        # C2      q15 q7
        # C1      q2  q4
        # C0      q6  q14

4:
        ADD     r4, sp, #80
        VSHL.S32 q11, q2, q0
        CMP     lr, #7
        VLDMIA  r4, {d4-d5}

        VSHL.S32 q13, q15, q0
        VLDR    q15, [sp, 240]           // q15 spilled

        VSHL.S32 q8, q6, q0
        VSHL.S32 q9, q14, q0
        VSHL.S32 q10, q4, q0
        VSHL.S32 q12, q7, q0
        VSHL.S32 q14, q5, q0
        VSHL.S32 q15, q15, q0

        VQDMULH.S32 q8, q8, q2
        VQDMULH.S32 q9, q9, q2
        VQDMULH.S32 q10, q10, q2
        VQDMULH.S32 q12, q12, q2
        VQDMULH.S32 q11, q11, q2
        VQDMULH.S32 q13, q13, q2
        VQDMULH.S32 q14, q14, q2
        VQDMULH.S32 q15, q15, q2
        VLDMIA  r4, {d4-d5}
        VRSHL.S32 q8, q8, q2
        VRSHL.S32 q9, q9, q2
        VRSHL.S32 q10, q10, q2
        VRSHL.S32 q12, q12, q2
        VRSHL.S32 q11, q11, q2
        VRSHL.S32 q13, q13, q2
        VRSHL.S32 q14, q14, q2
        VRSHL.S32 q15, q15, q2
        VQMOVN.S32 d17, q8
        VQMOVN.S32 d16, q9
        VQMOVN.S32 d19, q10
        VQMOVN.S32 d21, q12
        VLDMIA  r4, {d24-d25}
        VQMOVN.S32 d18, q11
        VQMOVN.S32 d20, q13
        VQMOVN.S32 d23, q14
        VQMOVN.S32 d22, q15
        VQADD.S16 q8, q8, q12
        VQADD.S16 q9, q9, q12
        VQADD.S16 q10, q10, q12
        VQADD.S16 q11, q11, q12
        VQMOVN.S16 d17, q8
        VQMOVN.S16 d16, q9
        VQMOVN.S16 d19, q10
        VQMOVN.S16 d18, q11
        VLDMIA  r4, {d20-d21}
        VMAX.S8 q8, q8, q10
        VMAX.S8 q10, q9, q10
        SUBS    lr, lr, #8
        VMIN.S8 q9, q8, q1
        VMIN.S8 q11, q10, q1
        BLS     9f

        # Store full 4 x 8
        VST1.8  {d22}, [ip], r5
        SUB     r7, r7, r1
        VST1.8  {d23}, [sl], r5
        SUB     r6, r6, r1
        VST1.8  {d18}, [fp], r5
        SUB     r2, r2, r1
        VST1.8  {d19}, [r0], r5
        SUB     r3, r3, r1
        BNE     0b

        ADD     sp, sp, #460  // skip over r2.
        POP     {r4, r5, r6, r7, r8, r9, sl, fp, lr}
        VPOP    {d8-d15}
        BX      lr

5:
        STR     r0, [sp, #72]
        ADD     r0, r8, #8
        STR     r0, [sp, #224]
        MOV     r5, r7
        MOV     r7, r6
        LDR     r0, [r8]
        STR     r0, [sp, #256]
        MOV     r6, r2
        LDR     r2, [r7]
        MOV     r9, r3
        LDR     r1, [r7, #4]
        CMP     r4, #1
        LDR     r0, [r5, #4]
        STR     fp, [sp, #76]
        LDR     fp, [r8, #4]
        LDR     r3, [r5]
        STR     r1, [sp, #340]
        STR     r2, [sp, #336]
        STR     r0, [sp, #348]
        LDR     r0, [r6]
        LDR     r2, [r6, #4]
        STR     r3, [sp, #344]
        MOV     r3, r9
        STR     fp, [sp, #356]
        ADD     r3, r4, r9
        LDR     r1, [sp, #256]
        STR     r1, [sp, #352]
        STR     r0, [sp, #328]
        ADD     r0, sp, #336
        STR     r2, [sp, #332]
        MOV     r2, r6
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #344
        MOV     r6, r7
        MOV     r7, r5
        VLD1.8  {d17}, [r0 :64]
        ADD     r0, sp, #352
        VMOVL.S8 q10, d16
        ADD     r7, r4, r5
        VLD1.8  {d18}, [r0 :64]
        ADD     r0, sp, #328
        VMOVL.S8 q3, d17
        ADD     r6, r4, r6
        VLD1.8  {d17}, [r0 :64]
        VMOVL.S8 q9, d18
        VORR    q11, q10, q10
        ADD     r2, r4, r2
        VMOVL.S8 q8, d17
        LDR     r0, [r9]
        LDR     r1, [r9, #4]
        VMLAL.S16 q6, d19, d6[0]
        STR     r0, [sp, #320]
        VORR    q12, q3, q3
        VMLAL.S16 q14, d18, d6[0]
        ADD     r0, sp, #320
        VORR    q3, q10, q10
        STR     r1, [sp, #324]
        VORR    q10, q8, q8
        VMLAL.S16 q4, d19, d6[0]
        VMLAL.S16 q2, d18, d6[0]
        VORR    q3, q8, q8
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #240
        VMLAL.S16 q7, d19, d6[0]
        VMLAL.S16 q15, d18, d6[0]
        VMOVL.S8 q3, d16
        VLDMIA  r0, {d16-d17}
        ADD     r0, sp, #240
        VMLAL.S16 q8, d18, d6[0]
        VMLAL.S16 q5, d19, d6[0]
        VSTMIA  r0, {d16-d17}
        BNE     6f
        LDR     r8, [sp, #224]
        MOV     r9, #32
        LDR     fp, [sp, #76]
        B       8f
6:
        LDR     r5, [sp, #224]
        VORR    q13, q3, q3
        VORR    q3, q12, q12
        CMP     r4, #3
        MOV     r9, #32
        LDR     r0, [r5]
        LDR     r1, [r5, #4]
        STR     r0, [sp, #312]
        ADD     r0, sp, #312
        STR     r1, [sp, #316]
        ADD     r1, r5, #8
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #240
        VMOVL.S8 q8, d16
        VLDMIA  r0, {d18-d19}
        ADD     r0, sp, #240
        LDR     fp, [sp, #76]
        VMLAL.S16 q6, d17, d6[1]
        VMLAL.S16 q14, d16, d6[1]
        VORR    q3, q11, q11
        VMLAL.S16 q4, d17, d6[1]
        VMLAL.S16 q2, d16, d6[1]
        VORR    q3, q10, q10
        VMLAL.S16 q7, d17, d6[1]
        VMLAL.S16 q15, d16, d6[1]
        VORR    q3, q13, q13
        VMLAL.S16 q9, d16, d6[1]
        VMLAL.S16 q5, d17, d6[1]
        VSTMIA  r0, {d18-d19}
        BCC     7f
        LDR     r0, [r1]
        VORR    q3, q12, q12
        LDR     r1, [r1, #4]
        CMP     r4, #3
        STR     r0, [sp, #304]
        ADD     r0, sp, #304
        STR     r1, [sp, #308]
        ADD     r1, r5, #16
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #240
        VMOVL.S8 q8, d16
        VLDMIA  r0, {d18-d19}
        ADD     r0, sp, #240
        VMLAL.S16 q6, d17, d6[2]
        VMLAL.S16 q14, d16, d6[2]
        VORR    q3, q11, q11
        VMLAL.S16 q4, d17, d6[2]
        VMLAL.S16 q2, d16, d6[2]
        VORR    q3, q10, q10
        VMLAL.S16 q7, d17, d6[2]
        VMLAL.S16 q15, d16, d6[2]
        VORR    q3, q13, q13
        VMLAL.S16 q9, d16, d6[2]
        VMLAL.S16 q5, d17, d6[2]
        VSTMIA  r0, {d18-d19}
        BEQ     7f
        LDR     r0, [r1]
        VORR    q3, q12, q12
        LDR     r1, [r1, #4]
        CMP     r4, #5
        STR     r0, [sp, #296]
        ADD     r0, sp, #296
        STR     r1, [sp, #300]
        ADD     r1, r5, #24
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #240
        VMOVL.S8 q8, d16
        VLDMIA  r0, {d18-d19}
        ADD     r0, sp, #240
        VMLAL.S16 q6, d17, d6[3]
        VMLAL.S16 q14, d16, d6[3]
        VORR    q3, q11, q11
        VMLAL.S16 q4, d17, d6[3]
        VMLAL.S16 q2, d16, d6[3]
        VORR    q3, q10, q10
        VMLAL.S16 q7, d17, d6[3]
        VMLAL.S16 q15, d16, d6[3]
        VORR    q3, q13, q13
        VMLAL.S16 q9, d16, d6[3]
        VMLAL.S16 q5, d17, d6[3]
        VSTMIA  r0, {d18-d19}
        BCC     7f
        LDR     r0, [r1]
        VORR    q3, q12, q12
        LDR     r1, [r1, #4]
        CMP     r4, #5
        STR     r0, [sp, #288]
        ADD     r0, sp, #288
        STR     r1, [sp, #292]
        ADD     r1, r5, #32
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #240
        VMOVL.S8 q8, d16
        VLDMIA  r0, {d18-d19}
        ADD     r0, sp, #240
        VMLAL.S16 q6, d17, d7[0]
        VMLAL.S16 q14, d16, d7[0]
        VORR    q3, q11, q11
        VMLAL.S16 q4, d17, d7[0]
        VMLAL.S16 q2, d16, d7[0]
        VORR    q3, q10, q10
        VMLAL.S16 q7, d17, d7[0]
        VMLAL.S16 q15, d16, d7[0]
        VORR    q3, q13, q13
        VMLAL.S16 q9, d16, d7[0]
        VMLAL.S16 q5, d17, d7[0]
        VSTMIA  r0, {d18-d19}
        BEQ     7f
        LDR     r0, [r1]
        VORR    q3, q12, q12
        LDR     r1, [r1, #4]
        CMP     r4, #7
        STR     r0, [sp, #280]
        ADD     r0, sp, #280
        STR     r1, [sp, #284]
        ADD     r1, r5, #40
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #240
        VMOVL.S8 q8, d16
        VLDMIA  r0, {d18-d19}
        ADD     r0, sp, #240
        VMLAL.S16 q6, d17, d7[1]
        VMLAL.S16 q14, d16, d7[1]
        VORR    q3, q11, q11
        VMLAL.S16 q4, d17, d7[1]
        VMLAL.S16 q2, d16, d7[1]
        VORR    q3, q10, q10
        VMLAL.S16 q7, d17, d7[1]
        VMLAL.S16 q15, d16, d7[1]
        VORR    q3, q13, q13
        VMLAL.S16 q9, d16, d7[1]
        VMLAL.S16 q5, d17, d7[1]
        VSTMIA  r0, {d18-d19}
        BCC     7f
        LDR     r0, [r1]
        VORR    q3, q12, q12
        LDR     r1, [r1, #4]
        ADD     r8, r8, #56
        STR     r0, [sp, #272]
        ADD     r0, sp, #272
        STR     r1, [sp, #276]
        VLD1.8  {d16}, [r0 :64]
        ADD     r0, sp, #240
        VMOVL.S8 q8, d16
        VLDMIA  r0, {d18-d19}
        ADD     r0, sp, #240
        VMLAL.S16 q6, d17, d7[2]
        VMLAL.S16 q14, d16, d7[2]
        VORR    q3, q11, q11
        VMLAL.S16 q4, d17, d7[2]
        VMLAL.S16 q2, d16, d7[2]
        VORR    q3, q10, q10
        VMLAL.S16 q7, d17, d7[2]
        VMLAL.S16 q15, d16, d7[2]
        VORR    q3, q13, q13
        VMLAL.S16 q9, d16, d7[2]
        VMLAL.S16 q5, d17, d7[2]
        VSTMIA  r0, {d18-d19}
        B       8f
7:
        MOV     r8, r1
8:
        LDR     r0, [sp, #72]
        LDR     r1, [sp, #56]
        LDR     r5, [sp, #576]
        B       4b

        # Store odd width
9:
        TST     lr, #4
        BEQ     10f
        VST1.32 {d22[0]}, [ip]!
        VST1.32 {d23[0]}, [sl]!
        VST1.32 {d18[0]}, [fp]!
        VST1.32 {d19[0]}, [r0]!
        VEXT.8  q9, q9, q9, #4
        VEXT.8  q11, q11, q11, #4
10:
        TST     lr, #2
        BEQ     11f
        VST1.16 {d22[0]}, [ip]!
        VST1.16 {d23[0]}, [sl]!
        VST1.16 {d18[0]}, [fp]!
        VST1.16 {d19[0]}, [r0]!
        VEXT.8  q9, q9, q9, #2
        VEXT.8  q11, q11, q11, #2
11:
        TST     lr, #1
        BEQ     12f
        VST1.8  {d22[0]}, [ip]
        VST1.8  {d23[0]}, [sl]
        VST1.8  {d18[0]}, [fp]
        VST1.8  {d19[0]}, [r0]
12:
        ADD     sp, sp, #460  // skip over r2.
        POP     {r4, r5, r6, r7, r8, r9, sl, fp, lr}
        VPOP    {d8-d15}
        BX      lr

END_FUNCTION xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
