// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/vcvt.h"
#include "test/vunary-microkernel-tester.h"

namespace {

struct ExactConvert : public Convert {
  float Tolerance(float, xnn_datatype) const override { return 0.0f; }
};

template <typename In, typename Out, typename UKernelFn>
void TestExactRounding(uint64_t arch_flags, size_t batch_tile,
                       UKernelFn ukernel,
                       xnn_init_unary_uparams_fn init_params) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  const xnn_quantization_params input_quantization = {-79, 0.0129482271f};
  const xnn_quantization_params output_quantization = {-12, 0.0054930891f};
  for (size_t batch_size : {size_t{1}, batch_tile, batch_tile + 1, size_t{256}}) {
    VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .input_quantization(input_quantization)
        .output_quantization(output_quantization)
        .Test<ExactConvert, In, Out>(ukernel, init_params);
  }
  for (float scale : {0x1.0p-8f, 0.3f, 1.0f, 128.0f}) {
    VUnaryMicrokernelTester()
        .batch_size(batch_tile + 1)
        .input_quantization({-10, scale})
        .output_quantization({20, 1.0f})
        .Test<Convert, In, Out>(ukernel, init_params);
  }
}

}  // namespace

#define XNN_QUANTIZED(T) xnnpack::quantized<T>
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile,           \
                                    vector_tile, datatype_in, datatype_out,    \
                                    params_type, init_params)                  \
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
  TEST(ukernel, output_scale) {                                                \
    TestOutputScale<Convert, datatype_in, datatype_out>(                       \
        arch_flags, batch_tile, ukernel, init_params);                         \
  }                                                                            \
  TEST(ukernel, output_zero_point) {                                           \
    TestOutputZeroPoint<Convert, datatype_in, datatype_out>(                   \
        arch_flags, batch_tile, ukernel, init_params);                         \
  }                                                                            \
  TEST(ukernel, input_scale) {                                                 \
    TestInputScale<Convert, datatype_in, datatype_out>(arch_flags, batch_tile, \
                                                       ukernel, init_params);  \
  }                                                                            \
  TEST(ukernel, input_zero_point) {                                            \
    TestInputZeroPoint<Convert, datatype_in, datatype_out>(                    \
        arch_flags, batch_tile, ukernel, init_params);                         \
  }                                                                            \
  TEST(ukernel, exact_rounding) {                                              \
    TestExactRounding<datatype_in, datatype_out>(arch_flags, batch_tile,       \
                                                 ukernel, init_params);        \
  }
#include "src/qs8-vcvt/qs8-vcvt.inc"
#undef XNN_UKERNEL
#undef XNN_QUANTIZED
