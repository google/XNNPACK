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

#ifndef LITERT_TENSOR_BACKENDS_XNNPACK_UTILS_H_
#define LITERT_TENSOR_BACKENDS_XNNPACK_UTILS_H_

#include "include/xnnpack.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/utils/macros.h"

namespace litert::tensor {

template <>
struct ErrorStatusBuilder::ErrorConversion<xnn_status> {
  static constexpr bool IsError(xnn_status value) {
    return value != xnn_status_success;
  }
  static absl::Status AsError(xnn_status value) {
    return absl::UnknownError(
        absl::StrCat("xnn_status=", static_cast<int>(value)));
  }
};

inline absl::Status XnnStatusToAbsl(enum xnn_status status,
                                    absl::string_view label) {
  if (status == xnn_status_success) {
    return absl::OkStatus();
  }
  return absl::UnknownError(
      absl::StrCat("xnn_status=", static_cast<int>(status), ";", label));
}

inline xnn_datatype GetXnnpackType(const graph::TensorInformation& info) {
  switch (info.type) {
    case Type::kUnknown:
    case Type::kBOOL:
    case Type::kI2:
    case Type::kI4:
      if (info.quantization) {
        if (info.quantization->As<graph::PerChannelAffineQuantization>().ok()) {
          return xnn_datatype_qcint4;
        } else if (info.quantization->As<graph::BlockwiseQuantization>().ok()) {
          return xnn_datatype_qbint4;
        }
      }
      break;
    case Type::kI8:
      if (info.quantization) {
        if (auto it =
                info.quantization->As<graph::PerChannelAffineQuantization>();
            it.ok()) {
          return it->scales.size() > 1 ? xnn_datatype_qcint8
                                       : xnn_datatype_qint8;
        }
      }
      break;
    case Type::kI16:
    case Type::kI64:
    case Type::kU4:
    case Type::kU8:
    case Type::kU16:
    case Type::kU32:
    case Type::kU64:
    case Type::kFP16:
      return xnn_datatype_fp16;
    case Type::kI32:
      return xnn_datatype_int32;
    case Type::kFP32:
      return xnn_datatype_fp32;
    case Type::kFP64:
      break;
    case Type::kBF16:
      return xnn_datatype_bf16;
  }
  return xnn_datatype_invalid;
}

}  // namespace litert::tensor

#endif  // LITERT_TENSOR_BACKENDS_XNNPACK_UTILS_H_
