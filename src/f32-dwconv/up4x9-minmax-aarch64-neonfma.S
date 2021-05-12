// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma(
#     size_t channels,                   x0
#     size_t output_width,               x1
#     const float** input,               x2
#     const float* weights,              x3
#     float* output,                     x4
#     size_t input_stride,               x5
#     size_t output_increment,           x6
#     size_t input_offset,               (x7) -> x20
#     const float* zero,                 [sp + 80] -> x19
#     const xnn_f32_minmax_params params [sp + 88] -> (x20)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

BEGIN_FUNCTION xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma

        # Save x19-x20,d8-d15 on stack
        STP     x19, x20, [sp, -80]!
        STP     d8,  d9, [sp, 16]
        STP     d10, d11, [sp, 32]
        STP     d12, d13, [sp, 48]
        STP     d14, d15, [sp, 64]

        # Load zero, params pointer
        LDP     x19, x20, [sp, 80]

        # Load min/max values
        LD2R    {v30.4s, v31.4s}, [x20]
        MOV     x20, x7                 // input_offset

0:
        #  x7 := i0
        #  x8 := i1
        LDP     x7, x8, [x2]
        #  x9 := i2
        # x10 := i3
        LDP     x9, x10, [x2, 16]
        # x11 := i4
        # x12 := i5
        LDP     x11, x12, [x2, 32]
        # x13 := i6
        # x14 := i7
        LDP     x13, x14, [x2, 48]
        # x15 := i8
        LDR     x15, [x2, 64]

        CMP     x7, x19                 // if i0 == zero
        ADD     x7, x7, x20             // i0 += input_offset
        CSEL    x7, x19, x7, EQ         //   i0 = zero, else += i0 + input_offset
        CMP     x8, x19                 // if i1 == zero
        ADD     x8, x8, x20             // i1 += input_offset
        CSEL    x8, x19, x8, EQ         //   i1 = zero, else += i1 + input_offset
        CMP     x9, x19                 // if i2 == zero
        ADD     x9, x9, x20             // i2 += input_offset
        CSEL    x9, x19, x9, EQ         //   i2 = zero, else += i2 + input_offset
        CMP     x10, x19                // if i3 == zero
        ADD     x10, x10, x20           // i3 += input_offset
        CSEL    x10, x19, x10, EQ       //   i3 = zero, else += i3 + input_offset
        CMP     x11, x19                // if i4 == zero
        ADD     x11, x11, x20           // i4 += input_offset
        CSEL    x11, x19, x11, EQ       //   i4 = zero, else += i4 + input_offset
        CMP     x12, x19                // if i5 == zero
        ADD     x12, x12, x20           // i5 += input_offset
        CSEL    x12, x19, x12, EQ       //   i5 = zero, else += i5 + input_offset
        CMP     x13, x19                // if i6 == zero
        ADD     x13, x13, x20           // i6 += input_offset
        CSEL    x13, x19, x13, EQ       //   i6 = zero, else += i6 + input_offset
        CMP     x14, x19                // if i7 == zero
        ADD     x14, x14, x20           // i7 += input_offset
        CSEL    x14, x19, x14, EQ       //   i7 = zero, else += i7 + input_offset
        CMP     x15, x19                // if i8 == zero
        ADD     x15, x15, x20           // i8 += input_offset
        CSEL    x15, x19, x15, EQ       //   i8 = zero, else += i8 + input_offset

        # input += input_stride
        ADD     x2, x2, x5

        # x16 := c = channels
        # c -= 4
        SUBS    x16, x0, 4
        # x17 := w = weights
        MOV     x17, x3

        # skip main loop if c <= 4
        B.LO    2f
1:
        LDP     q0, q1, [x17], 32
        LDP     q2, q3, [x17], 32
        LDP     q4, q5, [x17], 32
        LDP     q6, q7, [x17], 32
        LDP     q8, q9, [x17], 32
        LDR     q10, [x7], 16
        LDR     q11, [x8], 16
        LDR     q12, [x9], 16
        LDR     q13, [x10], 16
        LDR     q14, [x11], 16
        LDR     q15, [x12], 16
        LDR     q16, [x13], 16
        LDR     q17, [x14], 16
        LDR     q18, [x15], 16

        FMLA    v0.4S, v1.4S, v10.4S
        FMLA    v0.4S, v2.4S, v11.4S
        FMLA    v0.4S, v3.4S, v12.4S
        FMLA    v0.4S, v4.4S, v13.4S
        FMLA    v0.4S, v5.4S, v14.4S
        FMLA    v0.4S, v6.4S, v15.4S
        FMLA    v0.4S, v7.4S, v16.4S
        FMLA    v0.4S, v8.4S, v17.4S
        FMLA    v0.4S, v9.4S, v18.4S

        FMAX    v0.4S, v0.4S, v30.4S
        FMIN    v0.4S, v0.4S, v31.4S

        STR     q0, [x4], 16
        SUBS    x16, x16, 4
        B.HS    1b

2:
        # restore actual c value
        ADD     x16, x16, 4
        # skip processing remainder channels unless c != 0
        CBZ     x16, 4f

        LDP     q0, q1, [x17], 32
        LDP     q2, q3, [x17], 32
        LDP     q4, q5, [x17], 32
        LDP     q6, q7, [x17], 32
        LDP     q8, q9, [x17], 32
        LDR     q10, [x7], 16
        LDR     q11, [x8], 16
        LDR     q12, [x9], 16
        LDR     q13, [x10], 16
        LDR     q14, [x11], 16
        LDR     q15, [x12], 16
        LDR     q16, [x13], 16
        LDR     q17, [x14], 16
        LDR     q18, [x15], 16

        FMLA    v0.4S, v1.4S, v10.4S
        FMLA    v0.4S, v2.4S, v11.4S
        FMLA    v0.4S, v3.4S, v12.4S
        FMLA    v0.4S, v4.4S, v13.4S
        FMLA    v0.4S, v5.4S, v14.4S
        FMLA    v0.4S, v6.4S, v15.4S
        FMLA    v0.4S, v7.4S, v16.4S
        FMLA    v0.4S, v8.4S, v17.4S
        FMLA    v0.4S, v9.4S, v18.4S

        FMAX    v0.4S, v0.4S, v30.4S
        FMIN    v0.4S, v0.4S, v31.4S

        TBZ     x16, 1, 3f

        ST1     {v0.2S}, [x4], 8
        DUP     d0, v0.D[1]

3:
        TBZ     x16, 0, 4f

        ST1     {v0.S}[0], [x4], 4

4:
        # output_width -= 1
        SUBS    x1, x1, 1
        # output += output_increment
        ADD     x4, x4, x6
        # process next pixel if output_width != 0
        B.NE    0b

        # Restore x19-x20,d8-d15 from stack
        LDP     d14, d15, [sp, 64]
        LDP     d12, d13, [sp, 48]
        LDP     d10, d11, [sp, 32]
        LDP     d8,  d9, [sp, 16]
        LDP     x19, x20, [sp], 80
        RET

END_FUNCTION xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
