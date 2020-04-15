// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/scalar-utils.h>
#include <xnnpack/dwconv.h>


void xnn_q8_dwconv_minmax_ukernel_up1x9__scalar(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_q8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  const int32_t vkernel_zero_point = params->scalar.kernel_zero_point;
  const int32_t vmultiplier = params->scalar.multiplier;
  const int32_t vq31rounding = INT32_C(0x40000000);
  const int32_t vremainder_mask = params->scalar.remainder_mask;
  const uint32_t vshift = params->scalar.shift;
  const int32_t vremainder_threshold = params->scalar.remainder_threshold;
  const int32_t vout_min = params->scalar.output_min_less_zero_point;
  const int32_t vout_max = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  do {
    const uint8_t* i0 = input[0];
    const uint8_t* i1 = input[1];
    const uint8_t* i2 = input[2];
    const uint8_t* i3 = input[3];
    const uint8_t* i4 = input[4];
    const uint8_t* i5 = input[5];
    const uint8_t* i6 = input[6];
    const uint8_t* i7 = input[7];
    const uint8_t* i8 = input[8];

    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    do {
      int32_t vacc = *((const int32_t*) w);

      const int32_t vi0 = (int32_t) (uint32_t) *i0++;
      const uint32_t vk0 = (uint32_t) ((const uint8_t*) w)[4];
      const int32_t vxk0 = (int32_t) vk0 - vkernel_zero_point;
      vacc += vi0 * vxk0;

      const int32_t vi1 = (int32_t) (uint32_t) *i1++;
      const uint32_t vk1 = (uint32_t) ((const uint8_t*) w)[5];
      const int32_t vxk1 = (int32_t) vk1 - vkernel_zero_point;
      vacc += vi1 * vxk1;

      const int32_t vi2 = (int32_t) (uint32_t) *i2++;
      const uint32_t vk2 = (uint32_t) ((const uint8_t*) w)[6];
      const int32_t vxk2 = (int32_t) vk2 - vkernel_zero_point;
      vacc += vi2 * vxk2;

      const int32_t vi3 = (int32_t) (uint32_t) *i3++;
      const uint32_t vk3 = (uint32_t) ((const uint8_t*) w)[7];
      const int32_t vxk3 = (int32_t) vk3 - vkernel_zero_point;
      vacc += vi3 * vxk3;

      const int32_t vi4 = (int32_t) (uint32_t) *i4++;
      const uint32_t vk4 = (uint32_t) ((const uint8_t*) w)[8];
      const int32_t vxk4 = (int32_t) vk4 - vkernel_zero_point;
      vacc += vi4 * vxk4;

      const int32_t vi5 = (int32_t) (uint32_t) *i5++;
      const uint32_t vk5 = (uint32_t) ((const uint8_t*) w)[9];
      const int32_t vxk5 = (int32_t) vk5 - vkernel_zero_point;
      vacc += vi5 * vxk5;

      const int32_t vi6 = (int32_t) (uint32_t) *i6++;
      const uint32_t vk6 = (uint32_t) ((const uint8_t*) w)[10];
      const int32_t vxk6 = (int32_t) vk6 - vkernel_zero_point;
      vacc += vi6 * vxk6;

      const int32_t vi7 = (int32_t) (uint32_t) *i7++;
      const uint32_t vk7 = (uint32_t) ((const uint8_t*) w)[11];
      const int32_t vxk7 = (int32_t) vk7 - vkernel_zero_point;
      vacc += vi7 * vxk7;

      const int32_t vi8 = (int32_t) (uint32_t) *i8++;
      const uint32_t vk8 = (uint32_t) ((const uint8_t*) w)[12];
      const int32_t vxk8 = (int32_t) vk8 - vkernel_zero_point;
      vacc += vi8 * vxk8;

      w = (const void*) ((uintptr_t) w + sizeof(int32_t) + 9 * sizeof(uint8_t));

      const int64_t vproduct = (int64_t) vacc * (int64_t) vmultiplier;
      const int32_t vq31product = (int32_t) (uint32_t) ((uint64_t) (vproduct + (int64_t) vq31rounding) >> 31);
      const int32_t vremainder = (vq31product & vremainder_mask) - (int32_t) (vq31product < 0);
      int32_t vout = asr_s32(vq31product, vshift) + (int32_t) (vremainder > vremainder_threshold);
      vout = vout < vout_min ? vout_min : vout;
      vout = vout > vout_max ? vout_max : vout;
      vout += voutput_zero_point;

      *output++ = vout;
    } while (--c != 0);

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
