// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>

#include <memory>
#include <vector>

namespace models {

typedef std::vector<std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>> ExecutionPlan;

ExecutionPlan MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan MobileNetV2(pthreadpool_t threadpool);

}  // namespace models
