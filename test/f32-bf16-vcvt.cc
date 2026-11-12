// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/vcvt.h"
#include "test/vunary-microkernel-tester.h"

namespace {

bool FlushesDenormals(uint64_t arch_flags) {
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  return (arch_flags & xnn_arch_x86_avx512bf16) != 0;
#else
  (void)arch_flags;
  return false;
#endif
}

template <typename UKernelFn>
void TestSpecialValues(uint64_t arch_flags, UKernelFn ukernel,
                       xnn_init_unary_uparams_fn init_params) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  // x86 AVX512-BF16 VCVTNEPS2BF16 flushes denormal float inputs to signed zero.
  const bool flushes_denormals = FlushesDenormals(arch_flags);
  VUnaryMicrokernelTester().Test<Convert, float, xnn_bfloat16>(
      ukernel, init_params,
      /*inputs=*/
      {
          0.0f,
          -0.0f,
          // Denormals and sub-denormal tie-to-even rounding.
          0x1.0p-134f,   // 0x00008000: tie to even -> +0.0f
          -0x1.0p-134f,  // 0x80008000: tie to even -> -0.0f
          0x1.0p-133f,   // 0x00010000: smallest positive bf16 denormal
          -0x1.0p-133f,  // 0x80010000: smallest negative bf16 denormal
          0x1.8p-133f,   // 0x00018000: tie to even -> 0x1.0p-132f
          -0x1.8p-133f,  // 0x80018000: tie to even -> -0x1.0p-132f
          0x1.0p-127f,   // 0x00400000: mid-range bf16 denormal
          -0x1.0p-127f,  // 0x80400000: mid-range bf16 denormal
          0x1.FCp-127f,  // 0x007F0000: largest bf16 denormal
          -0x1.FCp-127f,
          0x1.FEp-127f,  // 0x007F8000: rounds up to smallest normal 0x1.0p-126f
          -0x1.FEp-127f,
          // Smallest normal and round-to-nearest-even ties.
          0x1.0p-126f,
          -0x1.0p-126f,
          0x1.00p0f,
          0x1.01p0f,   // 0x3F808000: tie with even LSB -> 0x1.00p0f
          0x1.03p0f,   // 0x3F818000: tie with odd LSB -> 0x1.04p0f
          -0x1.01p0f,  // 0xBF808000: tie with even LSB -> -0x1.00p0f
          -0x1.03p0f,  // 0xBF818000: tie with odd LSB -> -0x1.04p0f
          // Max finite bf16 and overflow to infinity.
          0x1.FEp127f,
          -0x1.FEp127f,
          0x1.FFp127f,
          -0x1.FFp127f,
          std::numeric_limits<float>::max(),
          -std::numeric_limits<float>::max(),
          INFINITY,
          -INFINITY,
          NAN,
          -NAN,
      },
      /*expected=*/
      {
          xnn_bfloat16_from_bits(0x0000),
          xnn_bfloat16_from_bits(0x8000),
          xnn_bfloat16_from_bits(0x0000),
          xnn_bfloat16_from_bits(0x8000),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x0000 : 0x0001),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x8000 : 0x8001),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x0000 : 0x0002),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x8000 : 0x8002),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x0000 : 0x0040),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x8000 : 0x8040),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x0000 : 0x007F),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x8000 : 0x807F),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x0000 : 0x0080),
          xnn_bfloat16_from_bits(flushes_denormals ? 0x8000 : 0x8080),
          xnn_bfloat16_from_bits(0x0080),
          xnn_bfloat16_from_bits(0x8080),
          xnn_bfloat16_from_bits(0x3F80),
          xnn_bfloat16_from_bits(0x3F80),
          xnn_bfloat16_from_bits(0x3F82),
          xnn_bfloat16_from_bits(0xBF80),
          xnn_bfloat16_from_bits(0xBF82),
          xnn_bfloat16_from_bits(0x7F7F),
          xnn_bfloat16_from_bits(0xFF7F),
          xnn_bfloat16_from_bits(0x7F80),
          xnn_bfloat16_from_bits(0xFF80),
          xnn_bfloat16_from_bits(0x7F80),
          xnn_bfloat16_from_bits(0xFF80),
          xnn_bfloat16_from_bits(0x7F80),
          xnn_bfloat16_from_bits(0xFF80),
          xnn_bfloat16_from_bits(0x7FC0),
          xnn_bfloat16_from_bits(0xFFC0),
      },
      /*tolerance_ulp=*/0);
}

template <typename UKernelFn>
void TestExhaustiveBf16RoundTrip(uint64_t arch_flags, UKernelFn ukernel) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const bool flushes_denormals = FlushesDenormals(arch_flags);
  constexpr size_t kNumBf16Values = 65536;
  std::vector<float> inputs(kNumBf16Values + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<xnn_bfloat16> outputs(kNumBf16Values);
  for (uint32_t bits = 0; bits < kNumBf16Values; ++bits) {
    inputs[bits] = xnn_bfloat16_to_float(
        xnn_bfloat16_from_bits(static_cast<uint16_t>(bits)));
  }
  ukernel(kNumBf16Values * sizeof(float), inputs.data(), outputs.data(),
          nullptr);
  for (uint32_t bits = 0; bits < kNumBf16Values; ++bits) {
    const uint16_t actual_bits = xnn_bfloat16_to_bits(outputs[bits]);
    if ((bits & 0x7FFF) > 0x7F80) {
      ASSERT_GT(actual_bits & 0x7FFF, 0x7F80)
          << "bits = 0x" << std::hex << bits;
    } else if (flushes_denormals && (bits & 0x7F80) == 0) {
      ASSERT_EQ(actual_bits, static_cast<uint16_t>(bits & 0x8000))
          << "bits = 0x" << std::hex << bits;
    } else {
      ASSERT_EQ(actual_bits, static_cast<uint16_t>(bits))
          << "bits = 0x" << std::hex << bits;
    }
  }
}

}  // namespace

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out, params_type, init_params)                    \
  TEST(ukernel, batch_eq) {                                                    \
    TestBatchEq<Convert, datatype_in, datatype_out>(arch_flags, batch_tile,    \
                                                    ukernel, init_params);     \
  }                                                                            \
  TEST(ukernel, batch_div) {                                                   \
    TestBatchDiv<Convert, datatype_in, datatype_out>(arch_flags, batch_tile,   \
                                                     ukernel, init_params);    \
  }                                                                            \
  TEST(ukernel, batch_lt) {                                                    \
    TestBatchLT<Convert, datatype_in, datatype_out>(arch_flags, batch_tile,    \
                                                    ukernel, init_params);     \
  }                                                                            \
  TEST(ukernel, batch_gt) {                                                    \
    TestBatchGT<Convert, datatype_in, datatype_out>(arch_flags, batch_tile,    \
                                                    ukernel, init_params);     \
  }                                                                            \
  TEST(ukernel, special_values) {                                              \
    TestSpecialValues(arch_flags, ukernel, init_params);                       \
  }                                                                            \
  TEST(ukernel, exhaustive_bf16_round_trip) {                                  \
    TestExhaustiveBf16RoundTrip(arch_flags, ukernel);                          \
  }
#include "src/f32-bf16-vcvt/f32-bf16-vcvt.inc"
#undef XNN_UKERNEL
