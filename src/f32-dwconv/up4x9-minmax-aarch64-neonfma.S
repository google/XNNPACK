// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma(
#     size_t channels,
#     size_t output_width,
#     const float** input,
#     const float* weights,
#     float* output,
#     size_t input_stride,
#     size_t output_increment,
#     const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
BEGIN_FUNCTION xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma

        # Save d8-d15 on stack
        STP  d8,  d9, [sp, -64]!
        STP d10, d11, [sp, 16]
        STP d12, d13, [sp, 32]
        STP d14, d15, [sp, 48]

        # Load min/max values
        LD2R {v30.4s, v31.4s}, [x7]

0:
        #  x7 := i0
        #  x8 := i1
        LDP x7, x8, [x2]
        #  x9 := i2
        # x10 := i3
        LDP x9, x10, [x2, 16]
        # x11 := i4
        # x12 := i5
        LDP x11, x12, [x2, 32]
        # x13 := i6
        # x14 := i7
        LDP x13, x14, [x2, 48]
        # x15 := i8
        LDR x15, [x2, 64]
        # input += input_stride
        ADD x2, x2, x5

        # x16 := c = channels
        # c -= 4
        SUBS x16, x0, 4
        # x17 := w = weights
        MOV x17, x3

        # skip main loop if c <= 4
        B.LO 2f
1:
        LDP q0, q1, [x17], 32
        LDP q2, q3, [x17], 32
        LDP q4, q5, [x17], 32
        LDP q6, q7, [x17], 32
        LDP q8, q9, [x17], 32
        LDR q10, [x7], 16
        LDR q11, [x8], 16
        LDR q12, [x9], 16
        LDR q13, [x10], 16
        LDR q14, [x11], 16
        LDR q15, [x12], 16
        LDR q16, [x13], 16
        LDR q17, [x14], 16
        LDR q18, [x15], 16

        FMLA v0.4S, v1.4S, v10.4S
        FMLA v0.4S, v2.4S, v11.4S
        FMLA v0.4S, v3.4S, v12.4S
        FMLA v0.4S, v4.4S, v13.4S
        FMLA v0.4S, v5.4S, v14.4S
        FMLA v0.4S, v6.4S, v15.4S
        FMLA v0.4S, v7.4S, v16.4S
        FMLA v0.4S, v8.4S, v17.4S
        FMLA v0.4S, v9.4S, v18.4S

        FMAX v0.4S, v0.4S, v30.4S
        FMIN v0.4S, v0.4S, v31.4S

        STR q0, [x4], 16
        SUBS x16, x16, 4
        B.HS 1b

2:
        # restore actual c value
        ADD x16, x16, 4
        # skip processing remainder channels unless c != 0
        CBZ x16, 4f

        LDP q0, q1, [x17], 32
        LDP q2, q3, [x17], 32
        LDP q4, q5, [x17], 32
        LDP q6, q7, [x17], 32
        LDP q8, q9, [x17], 32
        LDR q10, [x7], 16
        LDR q11, [x8], 16
        LDR q12, [x9], 16
        LDR q13, [x10], 16
        LDR q14, [x11], 16
        LDR q15, [x12], 16
        LDR q16, [x13], 16
        LDR q17, [x14], 16
        LDR q18, [x15], 16

        FMLA v0.4S, v1.4S, v10.4S
        FMLA v0.4S, v2.4S, v11.4S
        FMLA v0.4S, v3.4S, v12.4S
        FMLA v0.4S, v4.4S, v13.4S
        FMLA v0.4S, v5.4S, v14.4S
        FMLA v0.4S, v6.4S, v15.4S
        FMLA v0.4S, v7.4S, v16.4S
        FMLA v0.4S, v8.4S, v17.4S
        FMLA v0.4S, v9.4S, v18.4S

        FMAX v0.4S, v0.4S, v30.4S
        FMIN v0.4S, v0.4S, v31.4S

        TBZ x16, 1, 3f

        ST1 {v0.2S}, [x4], 8
        DUP d0, v0.D[1]

3:
        TBZ x16, 0, 4f

        ST1 {v0.S}[0], [x4], 4

4:
        # output_width -= 1
        SUBS x1, x1, 1
        # output += output_increment
        ADD x4, x4, x6
        # process next pixel if output_width != 0
        B.NE 0b

        # Restore d8-d15 from stack
        LDP d14, d15, [sp, 48]
        LDP d12, d13, [sp, 32]
        LDP d10, d11, [sp, 16]
        LDP  d8,  d9, [sp], 64
        RET

END_FUNCTION xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
