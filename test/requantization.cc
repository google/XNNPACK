// Copyright 2026 Google LLC
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/xnnpack/requantization.h"  // IWYU pragma: keep

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <vector>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"              // IWYU pragma: keep
#include "src/xnnpack/microparams-init.h"  // IWYU pragma: keep
#include "src/xnnpack/pack.h"              // IWYU pragma: keep
#include "xnnpack.h"

namespace {

// Largest `scale` accepted by the rndnu scalar contract.
constexpr float kMaxScale = 255.999f;
constexpr float kMinScale = 0x1.0p-32f;

// Reference for the rndnu scalar requantization, computed entirely in 64-bit so
// that it never suffers the int32 narrowing the implementation used to do.
// The 24-bit multiplier truncation is intentional, so the reference reproduces
// it rather than using an exact float product.
struct RndnuScalar {
  int32_t multiplier;
  uint32_t shift;
  int64_t rounding;
};

RndnuScalar ParseRndnuScalar(float scale) {
  const uint32_t scale_bits = float_as_uint32(scale);
  RndnuScalar p;
  p.multiplier =
      ((int32_t)scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  p.shift = 127 + 23 - (scale_bits >> 23);
  p.rounding = INT64_C(1) << (p.shift - 1);
  return p;
}

// Local 64-bit clamps. Deliberately not math_max_s64/math_min_s64 so that this
// test also compiles against a tree where the fix has not been applied, which
// is what makes it usable as a regression test.
inline int64_t ClampMax64(int64_t a, int64_t b) { return a > b ? a : b; }
inline int64_t ClampMin64(int64_t a, int64_t b) { return a < b ? a : b; }

int64_t ReferenceRequantize(int32_t input, const RndnuScalar& p, int32_t min,
                            int32_t max) {
  const int64_t prescaled = (int64_t)input * (int64_t)p.multiplier + p.rounding;
  int64_t output = prescaled >> p.shift;  // arithmetic shift right
  output = ClampMax64(output, (int64_t)min);
  output = ClampMin64(output, (int64_t)max);
  return output;
}

// Accumulator values that push the requantized result past INT32_MAX for a
// large `scale`: the overflow is in the *scaled* value, not the accumulator.
const int32_t kInputs[] = {
    0,
    1,
    -1,
    3,
    -3,
    255,
    -255,
    65535,
    -65535,
    1 << 20,
    -(1 << 20),
    1 << 24,
    -(1 << 24),
    1 << 30,
    -(1 << 30),
    2147483647,
    -2147483647 - 1,
};

TEST(Requantization, RndnuScalarQu8DoesNotOverflowInt32) {
  for (float scale = kMinScale; scale < kMaxScale; scale *= 2.0f) {
    const RndnuScalar p = ParseRndnuScalar(scale);
    // qu8 zero points are unsigned: 128 and 255 must not wrap through int8_t.
    for (const uint8_t zero_point :
         {uint8_t(0), uint8_t(1), uint8_t(128), uint8_t(255)}) {
      // Matches the output range passed to the function below.
      const int32_t min_less_zp = 0 - (int32_t)zero_point;
      const int32_t max_less_zp = 255 - (int32_t)zero_point;
      for (const int32_t input : kInputs) {
        const int64_t expected =
            ReferenceRequantize(input, p, min_less_zp, max_less_zp);
        const uint8_t actual =
            xnn_qu8_requantize_rndnu(input, scale, zero_point, 0, 255);
        ASSERT_EQ((int32_t)actual, (int32_t)expected + (int32_t)zero_point)
            << "scale=" << scale << " input=" << input
            << " zero_point=" << (int)zero_point;
      }
    }
  }
}

TEST(Requantization, RndnuScalarQs8DoesNotOverflowInt32) {
  for (float scale = kMinScale; scale < kMaxScale; scale *= 2.0f) {
    const RndnuScalar p = ParseRndnuScalar(scale);
    for (const int8_t zero_point :
         {int8_t(0), int8_t(1), int8_t(-128), int8_t(127)}) {
      const int32_t min_less_zp = -128 - zero_point;
      const int32_t max_less_zp = 127 - zero_point;
      for (const int32_t input : kInputs) {
        const int64_t expected =
            ReferenceRequantize(input, p, min_less_zp, max_less_zp);
        const int8_t actual =
            xnn_qs8_requantize_rndnu(input, scale, zero_point, -128, 127);
        ASSERT_EQ((int32_t)actual, (int32_t)expected + zero_point)
            << "scale=" << scale << " input=" << input
            << " zero_point=" << (int)zero_point;
      }
    }
  }
}

// End-to-end: a GEMM whose int32 accumulator scales past INT32_MAX must
// saturate to output_max, not wrap to output_min.
TEST(Requantization, RndnuScalarGemmSaturatesInsteadOfWrapping) {
  const size_t kc = 1024;
  const size_t nc = 2;
  const size_t mr = 1;
  const size_t nr = 2;
  const size_t kr = 1;
  const size_t sr = 1;

  // Full-scale activations and weights drive the accumulator to
  // kc * 255 * 127 = 33,154,560.
  std::vector<uint8_t> input(kc, 255);
  std::vector<uint8_t> weights(nc * kc, 127);
  std::vector<uint8_t> output(nc);
  std::vector<uint8_t> packed(nc * kc + 2 * sizeof(int32_t) + XNN_EXTRA_BYTES);

  struct xnn_qu8_packing_params packing_params = {0, 0};
  xnn_pack_qu8_gemm_goi_w(/*g=*/1, nc, kc, nr, kr, sr, /*n_stride=*/kc,
                          weights.data(), /*bias=*/nullptr, /*scale=*/nullptr,
                          packed.data(), /*extra_bytes=*/XNN_EXTRA_BYTES,
                          &packing_params);

  for (const float scale : {0.5f, 100.0f, 200.0f}) {
    union xnn_qu8_conv_minmax_params params;
    memset(&params, 0, sizeof(params));
    xnn_init_qu8_conv_minmax_rndnu_scalar_params(&params, /*kernel_zp=*/0,
                                                 scale,
                                                 /*output_zp=*/128, 0, 255);

    memset(output.data(), 0, output.size());
    xnn_qu8_gemm_minmax_rndnu_ukernel_1x2__scalar(
        mr, nc, kc, input.data(), kc, packed.data(), output.data(), nc,
        /*cn_stride=*/1, &params);

    // The exact product exceeds 127 - 128 for every scale tried, so the
    // correct answer saturates high regardless of the narrowing bug.
    ASSERT_EQ((int32_t)output[0], 255)
        << "scale=" << scale << " accumulator=" << (int64_t)kc * 255 * 127;
  }
}

}  // namespace
