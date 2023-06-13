// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x2(
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

BEGIN_FUNCTION xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x2
        .arm
#ifndef __APPLE__
        .arch       armv7-a
        .fpu        neon
#endif
        LDR         r12, [sp]               // output
        VMOV.U8     d0, 0                   // weight_accumulator
        PUSH        {r4,lr}                 // push 8 bytes

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

1:
        LDRB        r4, [r2], 1             // weight_widths
        SUBS        r4, r4, 1
        VMOV.U8     d1, 0                   // unweight_accumulator
        BLS         3f                      // less than 2 weights?

2:
        VLD1.16     {d3}, [r3]!             // weights
        VLD1.32     {d2}, [r1]!             // input
        SUBS        r4, r4, 2
        VMOVL.U16   q2, d3
        VMLAL.U32   q0, d4, d2[0]
        VMLAL.U32   q0, d5, d2[1]
        BHI         2b

        BLO         4f                      // is there a remainder?
3:
        VLD1.32     {d3[]}, [r3]!           // weights
        VLD1.32     {d2[]}, [r1]!           // input
        VMOVL.U16   q2, d3
        VMLAL.U32   q0, d4, d2

4:
        VST1.64     {d0}, [r12]!
        SUBS        r0, r0, 1
        VMOV        d0, d1
        BNE         1b

        POP         {r4,pc}

END_FUNCTION xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_neon_x2

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
