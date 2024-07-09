// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

typedef std::vector<std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>> Operators;
typedef std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> Workspace;

// Helper class for holding a list of operators and associated workspace.
// Workspace needs to live as long as the operators.
class ExecutionPlan {
 public:
  ExecutionPlan() = default;
  // Takes ownership of operators and workspace.
  ExecutionPlan(Operators& operators, Workspace& workspace)
      : operators_(std::move(operators)), workspace_(std::move(workspace)) {}

  bool empty() const {
    return operators_.empty();
  }
  Operators::iterator begin() { return operators_.begin(); }
  Operators::iterator end() { return operators_.end(); }

 private:
  Operators operators_;
  std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace_;
};

typedef ExecutionPlan (*ExecutionPlanFactory)(pthreadpool_t threadpool);

ExecutionPlan FP32MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV2(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV3Large(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV3Small(pthreadpool_t threadpool);

ExecutionPlan FP32MobileNetV1Jit(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV2Jit(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV3LargeJit(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV3SmallJit(pthreadpool_t threadpool);
ExecutionPlan FP32MobileNetV3SmallFused(pthreadpool_t threadpool);

ExecutionPlan FP32SparseMobileNetV1(float sparsity, pthreadpool_t threadpool);
ExecutionPlan FP32SparseMobileNetV2(float sparsity, pthreadpool_t threadpool);
ExecutionPlan FP32SparseMobileNetV3Large(float sparsity, pthreadpool_t threadpool);
ExecutionPlan FP32SparseMobileNetV3Small(float sparsity, pthreadpool_t threadpool);

ExecutionPlan FP16MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan FP16MobileNetV2(pthreadpool_t threadpool);
ExecutionPlan FP16MobileNetV3Large(pthreadpool_t threadpool);
ExecutionPlan FP16MobileNetV3Small(pthreadpool_t threadpool);

ExecutionPlan FP16SparseMobileNetV1(float sparsity, pthreadpool_t threadpool);
ExecutionPlan FP16SparseMobileNetV2(float sparsity, pthreadpool_t threadpool);
ExecutionPlan FP16SparseMobileNetV3Large(float sparsity, pthreadpool_t threadpool);
ExecutionPlan FP16SparseMobileNetV3Small(float sparsity, pthreadpool_t threadpool);

ExecutionPlan QC8MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan QC8MobileNetV2(pthreadpool_t threadpool);

ExecutionPlan QS8MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan QS8MobileNetV2(pthreadpool_t threadpool);

ExecutionPlan QU8MobileNetV1(pthreadpool_t threadpool);
ExecutionPlan QU8MobileNetV2(pthreadpool_t threadpool);
ExecutionPlan QU8MobileNetV3Large(pthreadpool_t threadpool);
ExecutionPlan QU8MobileNetV3Small(pthreadpool_t threadpool);

}  // namespace models
