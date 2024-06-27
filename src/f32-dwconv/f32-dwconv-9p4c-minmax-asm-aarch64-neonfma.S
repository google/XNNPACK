// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma(
#     size_t channels,                   x0, x20
#     size_t output_width,               x1
#     const float** input,               x2
#     const float* weights,              x3, x19
#     float* output,                     x4
#     intptr_t input_stride,             x5
#     size_t output_increment,           x6
#     size_t input_offset,               x7
#     const float* zero,                 [sp + 80] -> x17
#     const xnn_f32_minmax_params params [sp + 88] -> (x16)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# inputs
# i0  x8 v21
# i1  x9 v22
# i2 x10 v23
# i3 x11 v24
# i4 x12 v25
# i5 x13 v26
# i6 x14 v27
# i7 x15 v28
# i8 x16 v29

# weights
# x19 v0 (acc) v1 v2 v3 v4 v5 v6 v7 v16 v17

# Clamp v30 v31

# unused v18 v19 v20

BEGIN_FUNCTION xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma

        # Load zero, params pointer
        LDP         x17, x16, [sp]

        # Save x19,x20 on stack
        STP         x19, x20, [sp, -16]!

        # Load min/max values
        LD2R        {v30.4s, v31.4s}, [x16]

0:
        # Load 9 input pointers
        LDP         x8, x9, [x2]
        LDP         x10, x11, [x2, 16]
        LDP         x12, x13, [x2, 32]
        LDP         x14, x15, [x2, 48]
        LDR         x16, [x2, 64]

        CMP         x8, x17                 // if i0 == zero
        ADD         x8, x8, x7              // i0 += input_offset
        CSEL        x8, x17, x8, EQ         //   i0 = zero, else += i0 + input_offset
        CMP         x9, x17                 // if i1 == zero
        ADD         x9, x9, x7              // i1 += input_offset
        CSEL        x9, x17, x9, EQ         //   i1 = zero, else += i1 + input_offset
        CMP         x10, x17                // if i2 == zero
        ADD         x10, x10, x7            // i2 += input_offset
        CSEL        x10, x17, x10, EQ       //   i2 = zero, else += i2 + input_offset
        CMP         x11, x17                // if i3 == zero
        ADD         x11, x11, x7            // i3 += input_offset
        CSEL        x11, x17, x11, EQ       //   i3 = zero, else += i3 + input_offset
        CMP         x12, x17                // if i4 == zero
        ADD         x12, x12, x7            // i4 += input_offset
        CSEL        x12, x17, x12, EQ       //   i4 = zero, else += i4 + input_offset
        CMP         x13, x17                // if i5 == zero
        ADD         x13, x13, x7            // i5 += input_offset
        CSEL        x13, x17, x13, EQ       //   i5 = zero, else += i5 + input_offset
        CMP         x14, x17                // if i6 == zero
        ADD         x14, x14, x7            // i6 += input_offset
        CSEL        x14, x17, x14, EQ       //   i6 = zero, else += i6 + input_offset
        CMP         x15, x17                // if i7 == zero
        ADD         x15, x15, x7            // i7 += input_offset
        CSEL        x15, x17, x15, EQ       //   i7 = zero, else += i7 + input_offset
        CMP         x16, x17                // if i8 == zero
        ADD         x16, x16, x7            // i8 += input_offset
        CSEL        x16, x17, x16, EQ       //   i8 = zero, else += i8 + input_offset

        # input += input_stride
        ADD         x2, x2, x5

        # x20 := c = channels
        # c -= 4
        SUBS        x20, x0, 4
        # x19 := w = weights
        MOV         x19, x3

        # skip main loop if c <= 4
        B.LO        2f
1:
        LDR         q21, [x8], 16           // load 9 inputs
        LDP         q0, q1, [x19], 32       // load bias and 9 weights
        LDR         q22, [x9], 16
        LDR         q23, [x10], 16
        LDR         q24, [x11], 16
        LDR         q25, [x12], 16
        LDR         q26, [x13], 16
        LDR         q27, [x14], 16
        LDR         q28, [x15], 16
        LDR         q29, [x16], 16
        LDP         q2, q3, [x19], 32
        LDP         q4, q5, [x19], 32
        LDP         q6, q7, [x19], 32
        LDP         q16, q17, [x19], 32

        FMLA        v0.4S, v1.4S, v21.4S
        FMLA        v0.4S, v2.4S, v22.4S
        FMLA        v0.4S, v3.4S, v23.4S
        FMLA        v0.4S, v4.4S, v24.4S
        FMLA        v0.4S, v5.4S, v25.4S
        FMLA        v0.4S, v6.4S, v26.4S
        FMLA        v0.4S, v7.4S, v27.4S
        FMLA        v0.4S, v16.4S, v28.4S
        FMLA        v0.4S, v17.4S, v29.4S
        SUBS        x20, x20, 4

        FMAX        v0.4S, v0.4S, v30.4S
        FMIN        v0.4S, v0.4S, v31.4S
        STR         q0, [x4], 16
        B.HS        1b

2:
        # Is there a remainder?- 1 to 3 channels
        TST         x20, 3
        B.EQ        4f

        LDR         q21, [x8], 16           // load 9 inputs
        LDP         q0, q1, [x19], 32       // load bias and 9 weights
        LDR         q22, [x9], 16
        LDR         q23, [x10], 16
        LDR         q24, [x11], 16
        LDR         q25, [x12], 16
        LDR         q26, [x13], 16
        LDR         q27, [x14], 16
        LDR         q28, [x15], 16
        LDR         q29, [x16], 16
        LDP         q2, q3, [x19], 32
        LDP         q4, q5, [x19], 32
        LDP         q6, q7, [x19], 32
        LDP         q16, q17, [x19], 32

        FMLA        v0.4S, v1.4S, v21.4S
        FMLA        v0.4S, v2.4S, v22.4S
        FMLA        v0.4S, v3.4S, v23.4S
        FMLA        v0.4S, v4.4S, v24.4S
        FMLA        v0.4S, v5.4S, v25.4S
        FMLA        v0.4S, v6.4S, v26.4S
        FMLA        v0.4S, v7.4S, v27.4S
        FMLA        v0.4S, v16.4S, v28.4S
        FMLA        v0.4S, v17.4S, v29.4S

        FMAX        v0.4S, v0.4S, v30.4S
        FMIN        v0.4S, v0.4S, v31.4S

        TBZ         x20, 1, 3f

        STR         d0, [x4], 8
        DUP         d0, v0.D[1]
        TBZ         x20, 0, 4f
3:
        STR         s0, [x4], 4
4:
        # output_width -= 1
        SUBS        x1, x1, 1
        # output += output_increment
        ADD         x4, x4, x6
        # process next pixel if output_width != 0
        B.NE        0b

        # Restore x19,x20 from stack
        LDP         x19, x20, [sp], 16
        RET

END_FUNCTION xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
