/* Copyright 2025 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef LITERT_TENSOR_INTERNAL_ARITHMETIC_HELPERS_H_
#define LITERT_TENSOR_INTERNAL_ARITHMETIC_HELPERS_H_

#include <array>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/internal/shape.h"
#include "litert/tensor/internal/utils.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/source_location.h"

namespace litert::tensor {

template <class T>
struct IsTensor : std::false_type {};

template <class... Mixins>
struct IsTensor<Tensor<Mixins...>> : std::true_type {};

template <>
struct IsTensor<TensorHandle> : std::true_type {};

// Adds the given tensors as inputs of an op.
template <class Op, class... Tensors,
          class = std::enable_if_t<std::conjunction_v<IsTensor<Tensors>...>>>
void AddInputs(std::shared_ptr<Op>& operation, Tensors&... tensors) {
  static_assert(std::is_base_of_v<graph::Operation, Op>,
                "The operation is not derived from graph::Operation.");
  operation->inputs.reserve(size(operation->inputs) + sizeof...(tensors));
  auto SetOpInput = [&operation](TensorHandle& t) {
    GetConsumers(t.GetRaw())->push_back(operation);
    operation->inputs.push_back(t.GetRaw());
  };
  (SetOpInput(tensors), ...);
}

template <class Op, class TensorSpecialization,
          class = std::enable_if_t<IsTensor<TensorSpecialization>::value>>
void AddInputs(std::shared_ptr<Op>& operation,
               absl::Span<TensorSpecialization> tensors) {
  operation->inputs.reserve(size(operation->inputs) + tensors.size());
  for (TensorSpecialization& t : tensors) {
    GetConsumers(t.GetRaw())->push_back(operation);
    operation->inputs.push_back(t.GetRaw());
  }
}

// Creates a new output tensor for the given op.
//
// `op_loc` should be the location of the creating operation function call.
template <class Op>
TensorHandle AddOutput(std::shared_ptr<Op>& operation, source_location op_loc) {
  static_assert(std::is_base_of_v<graph::Operation, Op>,
                "The operation is not derived from graph::Operation.");
  std::shared_ptr<graph::TensorGroup> group = operation->outputs_group.lock();
  if (!group) {
    group = graph::NewTensorGroup(1, std::move(op_loc));
    operation->outputs_group = group;
    group->producer = operation;
  } else {
    group->tensor_infos.emplace_back();
  }
  return TensorHandle(graph::GetTensor(group->tensor_infos.size() - 1, group));
}

template <class Op, class... Tensors>
TensorHandle ElementwiseOp(source_location loc, TensorHandle a,
                           Tensors&... tensors) {
  auto operation = std::make_shared<Op>();
  AddInputs(operation, a, tensors...);
  TensorHandle output = AddOutput(operation, loc);
  const graph::TensorInformation& a_info = *GetInfo(a.GetRaw());
  graph::TensorInformation& o_info = *GetInfo(output.GetRaw());

  const std::array<const TensorHandle*, sizeof...(tensors)> tensor_ptrs{
      &tensors...};

  std::vector<std::vector<int>> shapes;
  shapes.push_back(a_info.shape);

  for (const TensorHandle* tensor : tensor_ptrs) {
    shapes.push_back(GetInfo(tensor->GetRaw())->shape);
  }

  absl::StatusOr<std::vector<int>> broadcasted_shape = BroadcastShapes(shapes);
  if (!broadcasted_shape.ok()) {
    return TensorHandle(graph::ErrorTensor(broadcasted_shape.status()));
  }
  o_info.shape = *broadcasted_shape;
  if (IsComparisonOp(*operation)) {
    o_info.type = Type::kBOOL;
  } else {
    o_info.type = a_info.type;
  }

  const bool all_types_match =
      std::all_of(tensor_ptrs.begin(), tensor_ptrs.end(),
                  [a_type = a.GetType()](const TensorHandle* tensor) {
                    return tensor->GetType() == a_type;
                  });
  if (!all_types_match) {
    return TensorHandle(graph::ErrorTensor(absl::InvalidArgumentError(
        absl::StrCat("All tensors in an elementwise operation "
                     "must have the same type. op: ",
                     operation->GetName()))));
  }

  graph::OpDebugger::DebugOp(*operation);
  return output;
}

}  // namespace litert::tensor

#endif  // LITERT_TENSOR_INTERNAL_ARITHMETIC_HELPERS_H_
