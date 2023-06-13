// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x1(
//     size_t rows,                          r0
//     const uint32_t* input,                r1
//     const uint8_t* weight_widths,         r2
//     const uint16_t* weights,              r3
//     uint64_t* output)                     sp -> r12

// d8-d15, r12-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// input   r1  d2
// weights r3  d3 d4 d5
// output  r12 d0 d1
// weight_widths r2 r4

BEGIN_FUNCTION xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x1
        .arm
#ifndef __APPLE__
        .arch       armv7-a
        .fpu        neon
#endif
        LDR         r12, [sp]               // output
        PUSH        {r4,lr}                 // push 8 bytes

        VMOV.U8     d0, 0                   // weight_accumulator

        // Compute unweight as initial weight
        LDRB        r4, [r2], 1             // weight_widths
        VMOV.U8     d1, 0                   // unweight_accumulator
0:
        VLD1.32     {d3[]}, [r3]!           // weight+unweight
        VLD1.32     {d2[]}, [r1]!           // input
        SUBS        r4, r4, 1
        VMOVL.U16   q2, d3
        VMLAL.U32   q0, d2, d4[1]           // unweight
        BHI         0b

        SUBS        r0, r0, 1
        BLS         3f

1:
        LDRB        r4, [r2], 1             // weight_widths
        VMOV.U8     d1, 0                   // unweight_accumulator
2:
        VLD1.32     {d3[]}, [r3]!           // weight+unweight
        VLD1.32     {d2[]}, [r1]!           // input
        SUBS        r4, r4, 1
        VMOVL.U16   q2, d3
        VMLAL.U32   q0, d4, d2
        BHI         2b

        VST1.64     {d0}, [r12]!
        SUBS        r0, r0, 1
        VMOV        d0, d1
        BNE         1b

3:
        // Final row only compute weight
        LDRB        r4, [r2], 1             // weight_widths
4:
        VLD1.32     {d3[]}, [r3]!           // weight+unweight
        VLD1.32     {d2[]}, [r1]!           // input
        SUBS        r4, r4, 1
        VMOVL.U16   q2, d3
        VMLAL.U32   q0, d2, d4[0]           // weight
        BHI         4b

        VST1.64     {d0}, [r12]!

        POP         {r4,pc}

END_FUNCTION xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x1

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
