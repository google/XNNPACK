// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "xnnpack.h"

namespace models {

xnn_subgraph_t FP32Attention(size_t b, size_t t, size_t h, size_t n, size_t s);
xnn_subgraph_t FP32MobileNetV1();
xnn_subgraph_t FP32MobileNetV2();
xnn_subgraph_t FP32MobileNetV3Large();
xnn_subgraph_t FP32MobileNetV3Small();
xnn_subgraph_t QS8MobileNetV2();

}  // namespace models
