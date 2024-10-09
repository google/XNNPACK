// Auto-generated file. Do not edit!
//   Template: src/qs8-prelu/scalar.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/prelu.h"


void xnn_qu8_prelu_ukernel__scalar_2x4(
    size_t rows,
    size_t channels,
    const uint8_t* restrict input,
    size_t input_stride,
    const void* restrict weights,
    uint8_t* restrict output,
    size_t output_stride,
    const struct xnn_qu8_prelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(rows != 0);
    assert(channels != 0);
    assert(channels % sizeof(uint8_t) == 0);
    assert(input != NULL);
    assert(output != NULL);

    const uint8_t* i0 = (const uint8_t*) input;
    uint8_t* o0 = (uint8_t*) output;
    const int32_t vinput_zero_point = params->scalar.input_zero_point;
    const int32_t vpositive_multiplier = params->scalar.positive_multiplier;
    const int32_t voutput_zero_point = params->scalar.output_zero_point;

    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
    uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);

    const size_t input_increment = input_stride * 2 - channels;
    const size_t output_increment = output_stride * 2 - channels;

    do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const int16_t* w = weights;
    size_t c = channels;

    for (; c >= 4 * sizeof(uint8_t); c -= 4 * sizeof(uint8_t)) {

      int32_t vi0x0 = (int32_t) i0[0];
      int32_t vi0x1 = (int32_t) i0[1];
      int32_t vi0x2 = (int32_t) i0[2];
      int32_t vi0x3 = (int32_t) i0[3];
      i0 += 4;
      int32_t vi1x0 = (int32_t) i1[0];
      int32_t vi1x1 = (int32_t) i1[1];
      int32_t vi1x2 = (int32_t) i1[2];
      int32_t vi1x3 = (int32_t) i1[3];
      i1 += 4;

      int32_t vnmulx0 = (int32_t) w[0];
      int32_t vnmulx1 = (int32_t) w[1];
      int32_t vnmulx2 = (int32_t) w[2];
      int32_t vnmulx3 = (int32_t) w[3];
      w += 4;

      vi0x0 -= vinput_zero_point;
      vi0x1 -= vinput_zero_point;
      vi0x2 -= vinput_zero_point;
      vi0x3 -= vinput_zero_point;
      vi1x0 -= vinput_zero_point;
      vi1x1 -= vinput_zero_point;
      vi1x2 -= vinput_zero_point;
      vi1x3 -= vinput_zero_point;

      int32_t vmul0x0 = XNN_UNPREDICTABLE(vi0x0 >= 0) ? vpositive_multiplier : vnmulx0;
      int32_t vmul0x1 = XNN_UNPREDICTABLE(vi0x1 >= 0) ? vpositive_multiplier : vnmulx1;
      int32_t vmul0x2 = XNN_UNPREDICTABLE(vi0x2 >= 0) ? vpositive_multiplier : vnmulx2;
      int32_t vmul0x3 = XNN_UNPREDICTABLE(vi0x3 >= 0) ? vpositive_multiplier : vnmulx3;
      int32_t vmul1x0 = XNN_UNPREDICTABLE(vi1x0 >= 0) ? vpositive_multiplier : vnmulx0;
      int32_t vmul1x1 = XNN_UNPREDICTABLE(vi1x1 >= 0) ? vpositive_multiplier : vnmulx1;
      int32_t vmul1x2 = XNN_UNPREDICTABLE(vi1x2 >= 0) ? vpositive_multiplier : vnmulx2;
      int32_t vmul1x3 = XNN_UNPREDICTABLE(vi1x3 >= 0) ? vpositive_multiplier : vnmulx3;

      vi0x0 = vi0x0 * (-vmul0x0);
      vi0x1 = vi0x1 * (-vmul0x1);
      vi0x2 = vi0x2 * (-vmul0x2);
      vi0x3 = vi0x3 * (-vmul0x3);
      vi1x0 = vi1x0 * (-vmul1x0);
      vi1x1 = vi1x1 * (-vmul1x1);
      vi1x2 = vi1x2 * (-vmul1x2);
      vi1x3 = vi1x3 * (-vmul1x3);

      int32_t vo0x0 = math_asr_s32(vi0x0, 8);
      int32_t vo0x1 = math_asr_s32(vi0x1, 8);
      int32_t vo0x2 = math_asr_s32(vi0x2, 8);
      int32_t vo0x3 = math_asr_s32(vi0x3, 8);
      int32_t vo1x0 = math_asr_s32(vi1x0, 8);
      int32_t vo1x1 = math_asr_s32(vi1x1, 8);
      int32_t vo1x2 = math_asr_s32(vi1x2, 8);
      int32_t vo1x3 = math_asr_s32(vi1x3, 8);

      vo0x0 += voutput_zero_point;
      vo0x1 += voutput_zero_point;
      vo0x2 += voutput_zero_point;
      vo0x3 += voutput_zero_point;
      vo1x0 += voutput_zero_point;
      vo1x1 += voutput_zero_point;
      vo1x2 += voutput_zero_point;
      vo1x3 += voutput_zero_point;

      vo0x0 = math_max_s32(vo0x0,  0);
      vo0x1 = math_max_s32(vo0x1,  0);
      vo0x2 = math_max_s32(vo0x2,  0);
      vo0x3 = math_max_s32(vo0x3,  0);
      vo1x0 = math_max_s32(vo1x0,  0);
      vo1x1 = math_max_s32(vo1x1,  0);
      vo1x2 = math_max_s32(vo1x2,  0);
      vo1x3 = math_max_s32(vo1x3,  0);
      
      vo0x0 = math_min_s32(vo0x0, 255);
      vo0x1 = math_min_s32(vo0x1, 255);
      vo0x2 = math_min_s32(vo0x2, 255);
      vo0x3 = math_min_s32(vo0x3, 255);
      vo1x0 = math_min_s32(vo1x0, 255);
      vo1x1 = math_min_s32(vo1x1, 255);
      vo1x2 = math_min_s32(vo1x2, 255);
      vo1x3 = math_min_s32(vo1x3, 255);

      o0[0] = (uint8_t) vo0x0;
      o0[1] = (uint8_t) vo0x1;
      o0[2] = (uint8_t) vo0x2;
      o0[3] = (uint8_t) vo0x3;
      o0 += 4;
      o1[0] = (uint8_t) vo1x0;
      o1[1] = (uint8_t) vo1x1;
      o1[2] = (uint8_t) vo1x2;
      o1[3] = (uint8_t) vo1x3;
      o1 += 4;
    }
    for (; c != 0; c -= sizeof(uint8_t)) {
      int32_t vi0 = (int32_t) *(uint8_t*)i0++;
      int32_t vi1 = (int32_t) *(uint8_t*)i1++;

      int32_t vnegative_multiplier = (int32_t) *w++;

      vi0 -= vinput_zero_point;
      vi1 -= vinput_zero_point;

      const int32_t vmul0 = XNN_UNPREDICTABLE(vi0 >= 0) ? vpositive_multiplier : vnegative_multiplier;
      const int32_t vmul1 = XNN_UNPREDICTABLE(vi1 >= 0) ? vpositive_multiplier : vnegative_multiplier;

      vi0 = vi0 * (-vmul0);
      vi1 = vi1 * (-vmul1);

      int32_t vo0 = math_asr_s32(vi0, 8);
      int32_t vo1 = math_asr_s32(vi1, 8);

      vo0 += voutput_zero_point;
      vo1 += voutput_zero_point;

      vo0 = math_max_s32(vo0, 0);
      vo1 = math_max_s32(vo1, 0);
      
      vo0 = math_min_s32(vo0, 255);
      vo1 = math_min_s32(vo1, 255);

      *(uint8_t*)o0++ = (uint8_t) vo0;
      *(uint8_t*)o1++ = (uint8_t) vo1;
    }
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
    } while (rows != 0);
}
