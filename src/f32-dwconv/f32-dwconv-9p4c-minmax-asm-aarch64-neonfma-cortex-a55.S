// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma_cortex_a55(
#     size_t channels,                   x0, x20
#     size_t output_width,               x1
#     const float** input,               x2
#     const float* weights,              x3, x19
#     float* output,                     x4
#     intptr_t input_stride,             x5
#     size_t output_increment,           x6
#     size_t input_offset,               x7
#     const float* zero,                 [sp + 64] -> x17
#     const xnn_f32_minmax_params params [sp + 72] -> (x16)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# inputs
# i0  x8
# i1  x9
# i2 x10
# i3 x11
# i4 x12
# i5 x13
# i6 x14
# i7 x15
# i8 x16

# weights.  Bias and 9 weights.
# x19

# accumulators
# v0-v3

# Input and weight paired values.
# Inputs are even and weights are odd registers
# v4  v5
# v6  v7
# v10 v11
# v12 v13
# v14 v15
# v16 v17
# v18 v19
# v20 v21
# v22 v23
# v24 v25
# v26 v27
# v28 v29

# Clamp v30 v31

# unused v8 v9

BEGIN_FUNCTION xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma_cortex_a55

        # Load zero, params pointer
        LDP         x17, x16, [sp]

        # Save x19-x20,d10-d15 on stack
        STP         x19, x20, [sp, -64]!
        STP         d10, d11, [sp, 16]
        STP         d12, d13, [sp, 32]
        STP         d14, d15, [sp, 48]

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
        # c -= 8
        SUBS        x20, x0, 8
        # x19 := w = weights
        MOV         x19, x3

        # skip main loop if c < 8
        B.LO        3f

          # SWP prologue

          # Load vbias.lo
          LD1         {v0.2S}, [x19], 8

          # Load vbias.hi
          LD1         {v1.2S}, [x19], 8

          # Load vi0.lo
          LD1         {v4.2S}, [x8], 8

          # Load vk0.lo
          LD1         {v5.2S}, [x19], 8

          # Load vi0.hi
          LD1         {v6.2S}, [x8], 8

          # Load vk0.hi
          LD1         {v7.2S}, [x19], 8

          # Load vi1.lo
          LD1         {v28.2S}, [x9], 8

          # Load vk1.lo
          LD1         {v29.2S}, [x19], 8

          # Load vi1.hi
          LD1         {v10.2S}, [x9], 8

          # Load vk1.hi
          LD1         {v11.2S}, [x19], 8

          # Load vi2.lo
          LD1         {v12.2S}, [x10], 8

          # Load vk2.lo
          LD1         {v13.2S}, [x19], 8

          # Load vi2.hi
          LD1         {v14.2S}, [x10], 8

          # Load vk2.hi
          LD1         {v15.2S}, [x19], 8

          # Load vi3.lo
          LD1         {v16.2S}, [x11], 8

          # Load vk3.lo
          LD1         {v17.2S}, [x19], 8

          # Load vi3.hi
          LD1         {v18.2S}, [x11], 8

          # Load vk3.hi
          LD1         {v19.2S}, [x19], 8

          # Load vi4.lo
          LD1         {v20.2S}, [x12], 8

          # Load vk4.lo
          LD1         {v21.2S}, [x19], 8

          # Load vi4.hi
          LD1         {v22.2S}, [x12], 8

          # Load vk4.hi
          LD1         {v23.2S}, [x19], 8

          # Load vi5.lo
          LD1         {v24.2S}, [x13], 8

          # Load vk5.lo
          LD1         {v25.2S}, [x19], 8

          # Load vi5.hi
          LD1         {v26.2S}, [x13], 8

          # Load vk5.hi
          LD1         {v27.2S}, [x19], 8

          # vacc.lo += vi0.lo * vk0.lo
          FMLA        v0.2S, v4.2S, v5.2S
          # Load vi6.lo
          LD1         {v4.2S}, [x14], 8

          # Load vk6.lo
          LD1         {v5.2S}, [x19], 8

          # vacc.hi += vi0.hi * vk0.hi
          FMLA        v1.2S, v6.2S, v7.2S
          # Load vi6.hi
          LD1         {v6.2S}, [x14], 8

          # Load vk6.hi
          LD1         {v7.2S}, [x19], 8

          # vacc.lo += vi1.lo * vk0.lo
          FMLA        v0.2S, v28.2S, v29.2S
          # Load vi7.lo
          LD1         {v28.2S}, [x15], 8

          # Load vk7.lo
          LD1         {v29.2S}, [x19], 8

          # vacc.hi += vi1.hi * vk0.hi
          FMLA        v1.2S, v10.2S, v11.2S
          # Load vi7.hi
          LD1         {v10.2S}, [x15], 8

          # Load vk7.hi
          LD1         {v11.2S}, [x19], 8

          # vacc.lo += vi2.lo * vk2.lo
          FMLA        v0.2S, v12.2S, v13.2S
          # Load vi8.lo
          LD1         {v12.2S}, [x16], 8

          # Load vk8.lo
          LD1         {v13.2S}, [x19], 8

          # vacc.hi += vi2.hi * vk2.hi
          FMLA        v1.2S, v14.2S, v15.2S
          # Load vi8.hi
          LD1         {v14.2S}, [x16], 8

          # Load vk8.hi
          LD1         {v15.2S}, [x19], 8

          # Load vbias_next.lo
          LD1         {v2.2S}, [x19], 8

          # Load vbias_next.hi
          LD1         {v3.2S}, [x19], 8

          # vacc.lo += vi3.lo * vk3.lo
          FMLA        v0.2S, v16.2S, v17.2S
          # Load vi0_next.lo
          LD1         {v16.2S}, [x8], 8

          # Load vk0_next.lo
          LD1         {v17.2S}, [x19], 8

          # vacc.hi += vi3.hi * vk3.hi
          FMLA        v1.2S, v18.2S, v19.2S
          # Load vi0_next.hi
          LD1         {v18.2S}, [x8], 8

          # Load vk0_next.hi
          LD1         {v19.2S}, [x19], 8

          # vacc.lo += vi4.lo * vk4.lo
          FMLA        v0.2S, v20.2S, v21.2S
          # Load vi1_next.lo
          LD1         {v20.2S}, [x9], 8

          # Load vk1_next.lo
          LD1         {v21.2S}, [x19], 8

          # vacc.hi += vi4.hi * vk4.hi
          FMLA        v1.2S, v22.2S, v23.2S
          # Load vi1_next.hi
          LD1         {v22.2S}, [x9], 8

          # Load vk1_next.hi
          LD1         {v23.2S}, [x19], 8

          # vacc.lo += vi5.lo * vk5.lo
          FMLA        v0.2S, v24.2S, v25.2S
          # Load vi2_next.lo
          LD1         {v24.2S}, [x10], 8

          # Load vk2_next.lo
          LD1         {v25.2S}, [x19], 8

          # vacc.hi += vi5.hi * vk5.hi
          FMLA        v1.2S, v26.2S, v27.2S
          # Load vi2_next.hi
          LD1         {v26.2S}, [x10], 8

          # Load vk2_next.hi
          LD1         {v27.2S}, [x19], 8

          # vacc.lo += vi6.lo * vk6.lo
          FMLA        v0.2S, v4.2S, v5.2S
          # Load vi3_next.lo
          LD1         {v4.2S}, [x11], 8

          # Load vk3_next.lo
          LD1         {v5.2S}, [x19], 8

          # vacc.hi += vi6.hi * vk6.hi
          FMLA        v1.2S, v6.2S, v7.2S
          # Load vi3_next.hi
          LD1         {v6.2S}, [x11], 8

          # Load vk3_next.hi
          LD1         {v7.2S}, [x19], 8

          # vacc.lo += vi7.lo * vk7.lo
          FMLA        v0.2S, v28.2S, v29.2S
          # Load vi4_next.lo
          LD1         {v28.2S}, [x12], 8

          # Load vk4_next.lo
          LD1         {v29.2S}, [x19], 8

          # vacc.hi += vi7.hi * vk7.hi
          FMLA        v1.2S, v10.2S, v11.2S
          # Load vi4_next.hi
          LD1         {v10.2S}, [x12], 8

          # Load vk4_next.hi
          LD1         {v11.2S}, [x19], 8

          # vacc.lo += vi8.lo * vk8.lo
          FMLA        v0.2S, v12.2S, v13.2S
          # Load vi5_next.lo
          LD1         {v12.2S}, [x13], 8

          # Load vk5_next.lo
          LD1         {v13.2S}, [x19], 8

          # vacc.hi += vi8.hi * vk8.hi
          FMLA        v1.2S, v14.2S, v15.2S
          # Load vi5_next.hi
          LD1         {v14.2S}, [x13], 8

          # Load vk5_next.hi
          LD1         {v15.2S}, [x19], 8

          # vacc_next.lo += vi0_next.lo * vk0_next.lo
          FMLA        v2.2S, v16.2S, v17.2S
          # Load vi6_next.lo
          LD1         {v16.2S}, [x14], 8

          # vacc.lo = min(vacc.lo, vmin)
          FMAX        v0.2S, v0.2S, v30.2S
          # Load vk6_next.lo
          LD1         {v17.2S}, [x19], 8

          # vacc_next.hi += vi0_next.hi * vk0_next.hi
          FMLA        v3.2S, v18.2S, v19.2S
          # Load vi6_next.hi
          LD1         {v18.2S}, [x14], 8

          # vacc.hi = min(vacc.hi, vmin)
          FMAX        v1.2S, v1.2S, v30.2S
          # Load vk6_next.hi
          LD1         {v19.2S}, [x19], 8

          # vacc_next.lo += vi1_next.lo * vk1_next.lo
          FMLA        v2.2S, v20.2S, v21.2S
          # Load vi7_next.lo
          LD1         {v20.2S}, [x15], 8

          # vacc.lo = max(vacc.lo, vmax)
          FMIN        v0.2S, v0.2S, v31.2S
          # Load vk7_next.lo
          LD1         {v21.2S}, [x19], 8

          # vacc_next.hi += vi1_next.hi * vk1_next.hi
          FMLA        v3.2S, v22.2S, v23.2S
          # Load vi7_next.hi
          LD1         {v22.2S}, [x15], 8

          # vacc.hi = max(vacc.hi, vmax)
          FMIN        v1.2S, v1.2S, v31.2S
          # Load vk7_next.hi
          LD1         {v23.2S}, [x19], 8

          # vacc_next.lo += vi2_next.lo * vk2_next.lo
          FMLA        v2.2S, v24.2S, v25.2S
          # Load vi8_next.lo
          LD1         {v24.2S}, [x16], 8

          # Load vk8_next.lo
          LD1         {v25.2S}, [x19], 8

          # vacc_next.hi += vi2_next.hi * vk2_next.hi
          FMLA        v3.2S, v26.2S, v27.2S
          # Load vi8_next.hi
          LD1         {v26.2S}, [x16], 8

          # Store vacc
          STP         d0, d1, [x4], 16

          # c -= 8
          SUBS        x20, x20, 8
          # Load vk8_next.hi
          LD1         {v27.2S}, [x19], 8

          B.LO        2f

1:
            # SWP iteration

            # Load vbias.lo
            LD1         {v0.2S}, [x19], 8

            # Load vbias.hi
            LD1         {v1.2S}, [x19], 8

            # vacc_prev.lo += vi3_prev.lo * vk3_prev.lo
            FMLA        v2.2S, v4.2S, v5.2S
            # Load vi0.lo
            LD1         {v4.2S}, [x8], 8

            # Load vk0.lo
            LD1         {v5.2S}, [x19], 8

            # vacc_prev.hi += vi3_prev.hi * vk3_prev.hi
            FMLA        v3.2S, v6.2S, v7.2S
            # Load vi0.hi
            LD1         {v6.2S}, [x8], 8

            # Load vk0.hi
            LD1         {v7.2S}, [x19], 8

            # vacc_prev.lo += vi4_prev.lo * vk4_prev.lo
            FMLA        v2.2S, v28.2S, v29.2S
            # Load vi1.lo
            LD1         {v28.2S}, [x9], 8

            # Load vk1.lo
            LD1         {v29.2S}, [x19], 8

            # vacc_prev.hi += vi4_prev.hi * vk4_prev.hi
            FMLA        v3.2S, v10.2S, v11.2S
            # Load vi1.hi
            LD1         {v10.2S}, [x9], 8

            # Load vk1.hi
            LD1         {v11.2S}, [x19], 8

            # vacc_prev.lo += vi5_prev.lo * vk5_prev.lo
            FMLA        v2.2S, v12.2S, v13.2S
            # Load vi2.lo
            LD1         {v12.2S}, [x10], 8

            # Load vk2.lo
            LD1         {v13.2S}, [x19], 8

            # vacc_prev.hi += vi5_prev.hi * vk5_prev.hi
            FMLA        v3.2S, v14.2S, v15.2S
            # Load vi2.hi
            LD1         {v14.2S}, [x10], 8

            # Load vk2.hi
            LD1         {v15.2S}, [x19], 8

            # vacc_prev.lo += vi6_prev.lo * vk6_prev.lo
            FMLA        v2.2S, v16.2S, v17.2S
            # Load vi3.lo
            LD1         {v16.2S}, [x11], 8

            # Load vk3.lo
            LD1         {v17.2S}, [x19], 8

            # vacc_prev.hi += vi6_prev.hi * vk6_prev.hi
            FMLA        v3.2S, v18.2S, v19.2S
            # Load vi3.hi
            LD1         {v18.2S}, [x11], 8

            # Load vk3.hi
            LD1         {v19.2S}, [x19], 8

            # vacc_prev.lo += vi7_prev.lo * vk7_prev.lo
            FMLA        v2.2S, v20.2S, v21.2S
            # Load vi4.lo
            LD1         {v20.2S}, [x12], 8

            # Load vk4.lo
            LD1         {v21.2S}, [x19], 8

            # vacc_prev.hi += vi7_prev.hi * vk7_prev.hi
            FMLA        v3.2S, v22.2S, v23.2S
            # Load vi4.hi
            LD1         {v22.2S}, [x12], 8

            # Load vk4.hi
            LD1         {v23.2S}, [x19], 8

            # vacc_prev.lo += vi8_prev.lo * vk8_prev.lo
            FMLA        v2.2S, v24.2S, v25.2S
            # Load vi5.lo
            LD1         {v24.2S}, [x13], 8

            # Load vk5.lo
            LD1         {v25.2S}, [x19], 8

            # vacc_prev.hi += vi8_prev.hi * vk8_prev.hi
            FMLA        v3.2S, v26.2S, v27.2S
            # Load vi5.hi
            LD1         {v26.2S}, [x13], 8

            # Load vk5.hi
            LD1         {v27.2S}, [x19], 8

            # vacc.lo += vi0.lo * vk0.lo
            FMLA        v0.2S, v4.2S, v5.2S
            # Load vi6.lo
            LD1         {v4.2S}, [x14], 8

            # vacc_prev.lo = min(vacc_prev.lo, vmin)
            FMAX        v2.2S, v2.2S, v30.2S
            # Load vk6.lo
            LD1         {v5.2S}, [x19], 8

            # vacc.hi += vi0.hi * vk0.hi
            FMLA        v1.2S, v6.2S, v7.2S
            # Load vi6.hi
            LD1         {v6.2S}, [x14], 8

            # vacc_prev.hi = min(vacc_prev.hi, vmin)
            FMAX        v3.2S, v3.2S, v30.2S
            # Load vk6.hi
            LD1         {v7.2S}, [x19], 8

            # vacc.lo += vi1.lo * vk0.lo
            FMLA        v0.2S, v28.2S, v29.2S
            # Load vi7.lo
            LD1         {v28.2S}, [x15], 8

            # vacc_prev.lo = max(vacc_prev.lo, vmax)
            FMIN        v2.2S, v2.2S, v31.2S
            # Load vk7.lo
            LD1         {v29.2S}, [x19], 8

            # vacc.hi += vi1.hi * vk0.hi
            FMLA        v1.2S, v10.2S, v11.2S
            # Load vi7.hi
            LD1         {v10.2S}, [x15], 8

            # vacc_prev.lo = max(vacc_prev.lo, vmax)
            FMIN        v3.2S, v3.2S, v31.2S
            # Load vk7.hi
            LD1         {v11.2S}, [x19], 8

            # vacc.lo += vi2.lo * vk2.lo
            FMLA        v0.2S, v12.2S, v13.2S
            # Load vi8.lo
            LD1         {v12.2S}, [x16], 8

            # Load vk8.lo
            LD1         {v13.2S}, [x19], 8

            # vacc.hi += vi2.hi * vk2.hi
            FMLA        v1.2S, v14.2S, v15.2S
            # Load vi8.hi
            LD1         {v14.2S}, [x16], 8

            # Store vacc_prev
            STP         d2, d3, [x4], 16

            # Load vk8.hi
            LD1         {v15.2S}, [x19], 8

            # Load vbias_next.lo
            LD1         {v2.2S}, [x19], 8

            # Load vbias_next.hi
            LD1         {v3.2S}, [x19], 8

            # vacc.lo += vi3.lo * vk3.lo
            FMLA        v0.2S, v16.2S, v17.2S
            # Load vi0_next.lo
            LD1         {v16.2S}, [x8], 8

            # Load vk0_next.lo
            LD1         {v17.2S}, [x19], 8

            # vacc.hi += vi3.hi * vk3.hi
            FMLA        v1.2S, v18.2S, v19.2S
            # Load vi0_next.hi
            LD1         {v18.2S}, [x8], 8

            # Load vk0_next.hi
            LD1         {v19.2S}, [x19], 8

            # vacc.lo += vi4.lo * vk4.lo
            FMLA        v0.2S, v20.2S, v21.2S
            # Load vi1_next.lo
            LD1         {v20.2S}, [x9], 8

            # Load vk1_next.lo
            LD1         {v21.2S}, [x19], 8

            # vacc.hi += vi4.hi * vk4.hi
            FMLA        v1.2S, v22.2S, v23.2S
            # Load vi1_next.hi
            LD1         {v22.2S}, [x9], 8

            # Load vk1_next.hi
            LD1         {v23.2S}, [x19], 8

            # vacc.lo += vi5.lo * vk5.lo
            FMLA        v0.2S, v24.2S, v25.2S
            # Load vi2_next.lo
            LD1         {v24.2S}, [x10], 8

            # Load vk2_next.lo
            LD1         {v25.2S}, [x19], 8

            # vacc.hi += vi5.hi * vk5.hi
            FMLA        v1.2S, v26.2S, v27.2S
            # Load vi2_next.hi
            LD1         {v26.2S}, [x10], 8

            # Load vk2_next.hi
            LD1         {v27.2S}, [x19], 8

            # vacc.lo += vi6.lo * vk6.lo
            FMLA        v0.2S, v4.2S, v5.2S
            # Load vi3_next.lo
            LD1         {v4.2S}, [x11], 8

            # Load vk3_next.lo
            LD1         {v5.2S}, [x19], 8

            # vacc.hi += vi6.hi * vk6.hi
            FMLA        v1.2S, v6.2S, v7.2S
            # Load vi3_next.hi
            LD1         {v6.2S}, [x11], 8

            # Load vk3_next.hi
            LD1         {v7.2S}, [x19], 8

            # vacc.lo += vi7.lo * vk7.lo
            FMLA        v0.2S, v28.2S, v29.2S
            # Load vi4_next.lo
            LD1         {v28.2S}, [x12], 8

            # Load vk4_next.lo
            LD1         {v29.2S}, [x19], 8

            # vacc.hi += vi7.hi * vk7.hi
            FMLA        v1.2S, v10.2S, v11.2S
            # Load vi4_next.hi
            LD1         {v10.2S}, [x12], 8

            # Load vk4_next.hi
            LD1         {v11.2S}, [x19], 8

            # vacc.lo += vi8.lo * vk8.lo
            FMLA        v0.2S, v12.2S, v13.2S
            # Load vi5_next.lo
            LD1         {v12.2S}, [x13], 8

            # Load vk5_next.lo
            LD1         {v13.2S}, [x19], 8

            # vacc.hi += vi8.hi * vk8.hi
            FMLA        v1.2S, v14.2S, v15.2S
            # Load vi5_next.hi
            LD1         {v14.2S}, [x13], 8

            # Load vk5_next.hi
            LD1         {v15.2S}, [x19], 8

            # vacc_next.lo += vi0_next.lo * vk0_next.lo
            FMLA        v2.2S, v16.2S, v17.2S
            # Load vi6_next.lo
            LD1         {v16.2S}, [x14], 8

            # vacc.lo = min(vacc.lo, vmin)
            FMAX        v0.2S, v0.2S, v30.2S
            # Load vk6_next.lo
            LD1         {v17.2S}, [x19], 8

            # vacc_next.hi += vi0_next.hi * vk0_next.hi
            FMLA        v3.2S, v18.2S, v19.2S
            # Load vi6_next.hi
            LD1         {v18.2S}, [x14], 8

            # vacc.hi = min(vacc.hi, vmin)
            FMAX        v1.2S, v1.2S, v30.2S
            # Load vk6_next.hi
            LD1         {v19.2S}, [x19], 8

            # vacc_next.lo += vi1_next.lo * vk1_next.lo
            FMLA        v2.2S, v20.2S, v21.2S
            # Load vi7_next.lo
            LD1         {v20.2S}, [x15], 8

            # vacc.lo = max(vacc.lo, vmax)
            FMIN        v0.2S, v0.2S, v31.2S
            # Load vk7_next.lo
            LD1         {v21.2S}, [x19], 8

            # vacc_next.hi += vi1_next.hi * vk1_next.hi
            FMLA        v3.2S, v22.2S, v23.2S
            # Load vi7_next.hi
            LD1         {v22.2S}, [x15], 8

            # vacc.hi = max(vacc.hi, vmax)
            FMIN        v1.2S, v1.2S, v31.2S
            # Load vk7_next.hi
            LD1         {v23.2S}, [x19], 8

            # vacc_next.lo += vi2_next.lo * vk2_next.lo
            FMLA        v2.2S, v24.2S, v25.2S
            # Load vi8_next.lo
            LD1         {v24.2S}, [x16], 8

            # Load vk8_next.lo
            LD1         {v25.2S}, [x19], 8

            # vacc_next.hi += vi2_next.hi * vk2_next.hi
            FMLA        v3.2S, v26.2S, v27.2S
            # Load vi8_next.hi
            LD1         {v26.2S}, [x16], 8

            # Store vacc
            STP         d0, d1, [x4], 16

            # c -= 8
            SUBS        x20, x20, 8
            # Load vk8_next.hi
            LD1         {v27.2S}, [x19], 8

            B.HS        1b

2:
          # SWP epilogue

          # vacc_prev.lo += vi3_prev.lo * vk3_prev.lo
          FMLA        v2.2S, v4.2S, v5.2S

          # vacc_prev.hi += vi3_prev.hi * vk3_prev.hi
          FMLA        v3.2S, v6.2S, v7.2S

          # vacc_prev.lo += vi4_prev.lo * vk4_prev.lo
          FMLA        v2.2S, v28.2S, v29.2S

          # vacc_prev.hi += vi4_prev.hi * vk4_prev.hi
          FMLA        v3.2S, v10.2S, v11.2S

          # vacc_prev.lo += vi5_prev.lo * vk5_prev.lo
          FMLA        v2.2S, v12.2S, v13.2S

          # vacc_prev.hi += vi5_prev.hi * vk5_prev.hi
          FMLA        v3.2S, v14.2S, v15.2S

          # vacc_prev.lo += vi6_prev.lo * vk6_prev.lo
          FMLA        v2.2S, v16.2S, v17.2S

          # vacc_prev.hi += vi6_prev.hi * vk6_prev.hi
          FMLA        v3.2S, v18.2S, v19.2S

          # vacc_prev.lo += vi7_prev.lo * vk7_prev.lo
          FMLA        v2.2S, v20.2S, v21.2S

          # vacc_prev.hi += vi7_prev.hi * vk7_prev.hi
          FMLA        v3.2S, v22.2S, v23.2S

          # vacc_prev.lo += vi8_prev.lo * vk8_prev.lo
          FMLA        v2.2S, v24.2S, v25.2S

          # vacc_prev.hi += vi8_prev.hi * vk8_prev.hi
          FMLA        v3.2S, v26.2S, v27.2S

          # vacc_prev.lo = min(vacc_prev.lo, vmin)
          FMAX        v2.2S, v2.2S, v30.2S

          # vacc_prev.hi = min(vacc_prev.hi, vmin)
          FMAX        v3.2S, v3.2S, v30.2S

          # vacc_prev.lo = max(vacc_prev.lo, vmax)
          FMIN        v2.2S, v2.2S, v31.2S

          # vacc_prev.lo = max(vacc_prev.lo, vmax)
          FMIN        v3.2S, v3.2S, v31.2S

          # Store vacc_prev
          STP         d2, d3, [x4], 16

3:
        # Is there a remainder? - 4 channels
        TBZ         x20, 2, 4f

        LDR         q10, [x8], 16           // load 9 inputs
        LDP         q0, q1, [x19], 32       // load bias and 9 weights
        LDR         q11, [x9], 16
        LDR         q12, [x10], 16
        LDR         q13, [x11], 16
        LDR         q14, [x12], 16
        LDR         q15, [x13], 16
        LDR         q16, [x14], 16
        LDR         q17, [x15], 16
        LDR         q18, [x16], 16
        LDP         q2, q3, [x19], 32
        LDP         q4, q5, [x19], 32
        LDP         q6, q7, [x19], 32
        LDP         q28, q29, [x19], 32

        FMLA        v0.4S, v1.4S, v10.4S
        FMLA        v0.4S, v2.4S, v11.4S
        FMLA        v0.4S, v3.4S, v12.4S
        FMLA        v0.4S, v4.4S, v13.4S
        FMLA        v0.4S, v5.4S, v14.4S
        FMLA        v0.4S, v6.4S, v15.4S
        FMLA        v0.4S, v7.4S, v16.4S
        FMLA        v0.4S, v28.4S, v17.4S
        FMLA        v0.4S, v29.4S, v18.4S

        FMAX        v0.4S, v0.4S, v30.4S
        FMIN        v0.4S, v0.4S, v31.4S

        STR         q0, [x4], 16

4:
        # Is there a remainder?- 1 to 3 channels
        TST         x20, 3
        B.EQ        6f

        LDR         q10, [x8], 16           // load 9 inputs
        LDP         q0, q1, [x19], 32       // load bias and 9 weights
        LDR         q11, [x9], 16
        LDR         q12, [x10], 16
        LDR         q13, [x11], 16
        LDR         q14, [x12], 16
        LDR         q15, [x13], 16
        LDR         q16, [x14], 16
        LDR         q17, [x15], 16
        LDR         q18, [x16], 16
        LDP         q2, q3, [x19], 32
        LDP         q4, q5, [x19], 32
        LDP         q6, q7, [x19], 32
        LDP         q28, q29, [x19], 32

        FMLA        v0.4S, v1.4S, v10.4S
        FMLA        v0.4S, v2.4S, v11.4S
        FMLA        v0.4S, v3.4S, v12.4S
        FMLA        v0.4S, v4.4S, v13.4S
        FMLA        v0.4S, v5.4S, v14.4S
        FMLA        v0.4S, v6.4S, v15.4S
        FMLA        v0.4S, v7.4S, v16.4S
        FMLA        v0.4S, v28.4S, v17.4S
        FMLA        v0.4S, v29.4S, v18.4S

        FMAX        v0.4S, v0.4S, v30.4S
        FMIN        v0.4S, v0.4S, v31.4S

        TBZ         x20, 1, 5f

        STR         d0, [x4], 8
        DUP         d0, v0.D[1]
        TBZ         x20, 0, 6f
5:
        STR         s0, [x4], 4
6:
        # output_width -= 1
        SUBS        x1, x1, 1
        # output += output_increment
        ADD         x4, x4, x6
        # process next pixel if output_width != 0
        B.NE        0b

        # Restore x19-x20,d10-d15 from stack
        LDP         d14, d15, [sp, 48]
        LDP         d12, d13, [sp, 32]
        LDP         d10, d11, [sp, 16]
        LDP         x19, x20, [sp], 64
        RET

END_FUNCTION xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma_cortex_a55

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
