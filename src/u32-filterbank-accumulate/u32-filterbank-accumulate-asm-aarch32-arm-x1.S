// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

.syntax unified

// void xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_arm_x1(
//     size_t rows,                          r0
//     const uint32_t* input,                r1
//     const uint8_t* weight_widths,         r2
//     const uint16_t* weights,              r3
//     uint64_t* output)                     sp -> r12

// d8-d15, r12-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// input                r1  r6
// weights              r3  r5 r7
// weight_accumulator   r12 r8 r9
// unweight_accumulator r10 r11
// weight_widths        r2  r4

BEGIN_FUNCTION xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_arm_x1
        .arm
#ifndef __APPLE__
        .arch   armv7-a
        .fpu    neon
#endif
        LDR     r12, [sp]               // output
        ADD     r3, r3, 2               // advance weights pointer to unweight
        PUSH    {r4,r5,r6,r7,r8,r9,r10,r11}  // push 32 bytes
        MOV     r8, 0                   // weight_accumulator
        MOV     r9, 0

        // Compute unweight as initial weight
        LDRB    r4, [r2], 1             // weight_widths
0:
        LDRH    r5, [r3], 4             // unweight
        LDR     r6, [r1], 4             // input
        SUBS    r4, r4, 1
        UMLAL   r8, r9, r6, r5          // initial weight_accumulator
        BHI     0b

        SUBS    r0, r0, 1
        SUB     r3, r3, 2               // rewind weights pointer to weight

        BLS     3f

1:
        LDRB    r4, [r2], 1             // weight_widths
        MOV     r10, 0                  // unweight_accumulator
        MOV     r11, 0

2:
        LDR     r5, [r3], 4             // weight+unweight
        LDR     r6, [r1], 4             // input
        SUBS    r4, r4, 1
        UXTH    r7, r5                  // weight
        UXTH    r5, r5, ror #16         // unweight
        UMLAL   r8,  r9,  r6, r7        // weight_accumulator
        UMLAL   r10, r11, r6, r5        // unweight_accumulator
        BHI     2b

        STMIA   r12!, {r8, r9}
        SUBS    r0, r0, 1
        MOV     r8, r10                 // weight_accumulator = unweight_accumulator
        MOV     r9, r11
        BHI     1b

3:
        // Final row only compute weight
        LDRB    r4, [r2]                // last weight_widths
4:
        LDRH    r5, [r3], 4             // weight
        LDR     r6, [r1], 4             // input
        SUBS    r4, r4, 1
        UMLAL   r8,  r9,  r6, r5        // weight_accumulator
        BHI     4b

        STMIA   r12!, {r8, r9}

        POP     {r4,r5,r6,r7,r8,r9,r10,r11}
        BX      lr

END_FUNCTION xnn_u32_filterbank_accumulate_ukernel__asm_aarch32_arm_x1

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
