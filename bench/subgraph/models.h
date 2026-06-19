// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_BENCH_MODELS_MODELS_H_
#define XNNPACK_BENCH_MODELS_MODELS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "include/xnnpack.h"

namespace models {

xnn_subgraph_t FP32MobileNetV1();
xnn_subgraph_t FP32MobileNetV2();
xnn_subgraph_t FP32MobileNetV3Large();
xnn_subgraph_t FP32MobileNetV3Small();
xnn_subgraph_t QS8MobileNetV2();


xnn_subgraph_t FP32LayerNorm(size_t m, size_t n, size_t k, uint32_t norm_mask);


}  // namespace models

#endif  // XNNPACK_BENCH_MODELS_MODELS_H_
