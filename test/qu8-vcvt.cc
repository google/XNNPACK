// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/vcvt.h"
#include "test/vunary-microkernel-tester.h"

struct ExactConvert : public Convert {
  float Tolerance(float, xnn_datatype) const override { return 0.0f; }
};

template <typename In, typename Out, typename UKernelFn>
void TestScalePrecision(uint64_t arch_flags, size_t batch_tile,
                        UKernelFn ukernel,
                        xnn_init_unary_uparams_fn init_params) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .input_quantization({127, 0.0129482271f})
        .output_quantization({137, 0.0054930891f})
        .Test<ExactConvert, In, Out>(ukernel, init_params);
  }
}

template <typename In, typename Out, typename UKernelFn>
void TestScaleBounds(uint64_t arch_flags, size_t batch_tile, UKernelFn ukernel,
                     xnn_init_unary_uparams_fn init_params) {
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (float input_scale : {0x1.0p-8f, 1.0f, 128.0f}) {
      xnn_quantization_params input_quantization =
          Convert().InputQuantizationParams(xnn_datatype_of<In>());
      xnn_quantization_params output_quantization =
          Convert().OutputQuantizationParams(xnn_datatype_of<Out>());
      input_quantization.scale = input_scale;
      VUnaryMicrokernelTester()
          .batch_size(batch_size)
          .input_quantization(input_quantization)
          .output_quantization(output_quantization)
          .Test<Convert, In, Out>(ukernel, init_params);
    }
  }
}

#define XNN_QUANTIZED(T) xnnpack::quantized<T>
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
  TEST(ukernel, scale_precision) {                                             \
    TestScalePrecision<datatype_in, datatype_out>(arch_flags, batch_tile,      \
                                                  ukernel, init_params);       \
  }                                                                            \
  TEST(ukernel, scale_bounds) {                                                \
    TestScaleBounds<datatype_in, datatype_out>(arch_flags, batch_tile,         \
                                               ukernel, init_params);          \
  }
#include "src/qu8-vcvt/qu8-vcvt.inc"
#undef XNN_UKERNEL
#undef XNN_QUANTIZED
