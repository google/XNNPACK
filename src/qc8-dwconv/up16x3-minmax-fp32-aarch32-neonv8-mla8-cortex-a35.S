// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

BEGIN_FUNCTION xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__aarch32_neonv8_mla8_cortex_a35

        PUSH    {r0, r4, r5, r6, r7, r8, r9, r10, r11, lr}
        VPUSH   {d8, d9, d10, d11, d12, d13}
        SUB     sp, sp, #24
        LDR     r5, [sp, #100+32]
        MOV     r9, r0
        LDR     r4, [sp, #96+32]
        LDR     r12, [sp, #88+32]
        LDR     r10, [sp, #80+32]

# r9/r11 kc loop counter
# r5 i0

        VLD1.32 {d20[]}, [r5]
        VDUP.8  q9 , d20[2]              // output_min
        VDUP.8  q11, d20[3]              // output_max
        VDUP.16 q10, d20[0]              // output_zero_point
        LDR     r0, [sp, #92+32]         // input_offset

0:

        LDMIB   r2, {r5, r6}  // i0, i1
        LDR     r8, [r2]      // i2
        CMP     r5, r4        // i0 == zero?
        ADDNE   r5, r5, r0    // i0 += input_offset
        CMP     r6, r4        // i1 == zero?
        ADDNE   r6, r6, r0    // i1 += input_offset
        CMP     r8, r4        // i2 == zero?
        ADDNE   r8, r8, r0    // i2 += input_offset

        MOV     lr, r3

        SUBS    r11, r9, #16
        BLO     2f

        MOV     r0, #80

# Main loop
# lr weights.  r3 reset
# r9/r11 loop counter
# r5 i0
# r6 i1
# r8 i2
# q12 q13 q14 q15   accumulators


1:
        ADD     r9, lr, #64
        VLD1.8  {q4}, [r8]!        // i2
        VLD1.8  {q12}, [r9]!       // w0
        VLD1.8  {q5}, [r5]!        // i0
        VLD1.8  {q13}, [r9]!       // w1
        VLD1.8  {q6}, [r6]!        // i1
        VLD1.8  {q14}, [r9]        // w2

        VMULL.S8 q1, d8,  d24      // i2 * w0
        VMULL.S8 q2, d9,  d25
        VMLAL.S8 q1, d10, d26      // i0 * w1
        VMLAL.S8 q2, d11, d27
        VMULL.S8 q0, d12, d28      // i1 * w2
        VLD1.8  {q12, q13}, [lr]!  // load bias
        VMULL.S8 q3, d13, d29
        VLD1.8  {q14, q15}, [lr], r0

        VADDW.S16 q12, q12, d0
        VADDW.S16 q13, q13, d1
        VADDW.S16 q14, q14, d4
        VADDW.S16 q15, q15, d5
        VADDW.S16 q12, q12, d2
        VADDW.S16 q13, q13, d3
        VADDW.S16 q14, q14, d6
        VLD1.32 {q0, q1}, [lr]!   // quant per channel scale values
        VADDW.S16 q15, q15, d7
        VLD1.32 {q2, q3}, [lr]!

        # QC8 FP32 quantization

        VCVT.F32.S32 q12, q12
        VCVT.F32.S32 q13, q13
        VCVT.F32.S32 q14, q14
        VCVT.F32.S32 q15, q15

        VMUL.F32 q12, q0, q12
        VMUL.F32 q13, q1, q13
        VMUL.F32 q14, q2, q14
        VMUL.F32 q15, q3, q15

        VCVTN.S32.F32 q12, q12
        VCVTN.S32.F32 q13, q13
        VCVTN.S32.F32 q14, q14
        VCVTN.S32.F32 q15, q15

        VQMOVN.S32 d24, q12
        VQMOVN.S32 d25, q13
        VQMOVN.S32 d28, q14
        VQMOVN.S32 d29, q15

        VQADD.S16 q12, q12, q10
        VQADD.S16 q14, q14, q10
        VQMOVN.S16 d24, q12
        VQMOVN.S16 d25, q14
        VMIN.S8 q12, q12, q11
        VMAX.S8 q12, q12, q9
        SUBS    r11, r11, 16
        VST1.8  {q12}, [r10]!
        BHS     1b

        LDR     r9, [sp, #72]
        LDR     r0, [sp, #92+32]

2:

        ANDS    r11, r9, 15
        BNE     4f

3:
        LDR     r6, [sp, #84+32]
        ADD     r10, r10, r12   // output += output_increment
        SUBS    r1, r1, #1      // output_width--
        ADD     r2, r2, r6      // input += input_stride
        BNE     0b

        ADD     sp, sp, #24
        VPOP    {d8, d9, d10, d11, d12, d13}
        ADD     sp, sp, #4
        POP     {r4, r5, r6, r7, r8, r9, r10, r11, pc}

4:
        MOV     r12, r11
        ADD     r0, lr, #128
        STR     r0, [sp, #12]
        ADD     r0, lr, #64
        STR     r0, [sp, #4]
        MOV     r0, #0

// Remainder loop - 8 channels

5:
        LDR            r7, [sp, #4]
        ADD            r4, r5, r0
        STR            r12, [sp, #8]
        VLD1.8         {d28, d29}, [lr]!
        VLD1.8         {d24}, [r4]
        MOV            r4, #32
        ADD            r7, r7, r0
        ADD            r12, r7, #16
        VLD1.8         {d26}, [r7], r4
        ADD            r4, r8, r0
        VLD1.8         {d25}, [r12]
        VLD1.8         {d27}, [r4]
        ADD            r12, r6, r0
        MVN            r4, #15
        VLD1.8         {d30}, [r12]
        LDR            r12, [sp, #8]
        VLD1.8         {d31}, [r7]
        LDR            r7, [sp, #12]
        VMULL.S8       q12, d24, d25
        VMLAL.S8       q12, d27, d26
        VLD1.8         {d26, d27}, [lr]
        CMP            r12, #7
        VLD1.32        {d0, d1}, [r7], r4
        VMULL.S8       q15, d30, d31
        VADDW.S16      q13, q13, d25
        VADDW.S16      q12, q14, d24
        VADDW.S16      q13, q13, d31
        VADDW.S16      q12, q12, d30
        VLD1.32        {d28, d29}, [r7]
        VCVT.F32.S32   q13, q13
        VCVT.F32.S32   q12, q12
        VMUL.F32       q13, q0, q13
        VCVTN.S32.F32  q13, q13
        VQMOVN.S32     d27, q13
        VMUL.F32       q12, q14, q12
        VCVTN.S32.F32  q12, q12
        VQMOVN.S32     d26, q12
        VQADD.S16      q12, q13, q10
        VQMOVN.S16     d24, q12
        VMAX.S8        d24, d24, d18
        VMIN.S8        d24, d24, d22
        BLS            6f

        LDR            r4, [sp, #12]
        ADD            r0, r0, #8
        ADD            lr, lr, #16
        SUB            r12, r12, #8
        VST1.8         {d24}, [r10]!
        CMP            r11, r0
        ADD            r4, r4, #32
        STR            r4, [sp, #12]
        BNE            5b

        LDR     r4, [sp, #96+32]
        LDR     r0, [sp, #92+32]
        LDR     r12, [sp, #88+32]
        B       3b

// less than 8
6:
        TST     r12, #4
        BEQ     7f

        VST1.32 {d24[0]}, [r10]!
        VEXT.8  d24, d24, d24, #4

7:
        LDR     r6, [sp, #8]
        LDR     r4, [sp, #96+32]
        LDR     r0, [sp, #92+32]
        LDR     r12, [sp, #88+32]

        TST     r6, #2
        BEQ     8f

        VST1.16 {d24[0]}, [r10]!
        VEXT.8  d24, d24, d24, #2

8:
        TST     r6, #1
        BEQ     3b
        VST1.8  {d24[0]}, [r10]!
        B       3b



END_FUNCTION xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__aarch32_neonv8_mla8_cortex_a35
#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
