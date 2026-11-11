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

#include "litert/tensor/backends/xnnpack/conversion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "include/xnnpack.h"
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "litert/tensor/arithmetic_graph.h"
#include "litert/tensor/backends/common_nnpack/conversion.h"
#include "litert/tensor/backends/common_nnpack/graph.h"
#include "litert/tensor/backends/xnnpack/graph.h"
#include "litert/tensor/backends/xnnpack/utils.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/internal/graph_traversal.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/macros.h"

namespace litert::tensor {
namespace {

absl::once_flag g_xnn_init_once;
absl::NoDestructor<absl::Status> g_xnn_init_status(absl::OkStatus());

xnn_datatype GetXnnpackType(const NnpackValue& value) {
  switch (value.info.type) {
    case Type::kI2:
      if (value.info.quantization &&
          value.info.quantization->As<PerChannelAffineQuantization>().ok()) {
        return xnn_datatype_qcint2;
      }
      break;
    case Type::kI4:
      if (value.info.quantization) {
        if (value.info.quantization->As<PerChannelAffineQuantization>().ok()) {
          return xnn_datatype_qcint4;
        } else if (value.info.quantization->As<BlockwiseQuantization>().ok()) {
          return xnn_datatype_qbint4;
        }
      }
      break;
    case Type::kI8:
      if (value.info.quantization) {
        if (auto it =
                value.info.quantization->As<PerChannelAffineQuantization>();
            it.ok()) {
          return it->scales.size() > 1 ? xnn_datatype_qcint8
                                       : xnn_datatype_qint8;
        }
      }
      break;
    case Type::kFP16:
      return xnn_datatype_fp16;
    case Type::kBF16:
      return xnn_datatype_bf16;
    case Type::kI32:
      return xnn_datatype_int32;
    case Type::kFP32:
      return xnn_datatype_fp32;
    case Type::kUnknown:
    case Type::kBOOL:
    case Type::kI16:
    case Type::kI64:
    case Type::kU2:
    case Type::kU4:
    case Type::kU8:
    case Type::kU16:
    case Type::kU32:
    case Type::kU64:
    case Type::kFP64:
      break;
  }
  return xnn_datatype_invalid;
}

std::shared_ptr<const std::vector<bf16_t>> GetSharedWarmDummyBf16Scales(
    size_t min_blocks) {
  static absl::NoDestructor<absl::Mutex> mu;
  static absl::NoDestructor<std::shared_ptr<const std::vector<bf16_t>>> cached;
  absl::MutexLock lock(&*mu);
  if (*cached == nullptr || (*cached)->size() < min_blocks) {
    *cached =
        std::make_shared<const std::vector<bf16_t>>(min_blocks, bf16_t(1.0f));
  }
  return *cached;
}

}  // namespace

absl::Status XnnpackBuildContext::EnsureInitialized() {
  absl::call_once(g_xnn_init_once, []() {
    *g_xnn_init_status =
        XnnStatusToAbsl(xnn_initialize(nullptr), "xnn_initialize");
  });
  return *g_xnn_init_status;
}

std::unique_ptr<NnpackGraph> XnnpackBuildContext::CreateEmptyGraph() {
  return std::make_unique<XnnpackGraph>();
}

absl::Status XnnpackBuildContext::CreateSubgraph(size_t external_value_ids,
                                                 uint32_t flags) {
  LRT_TENSOR_RETURN_IF_ERROR(static_cast<XnnpackGraph*>(graph_.get())
                                 ->ResetSubgraph(external_value_ids, flags));
  if (!warm_fc_buffers_.empty()) {
    if (warm_max_rows_ > 0) {
      graph_->dequantized_buffers().emplace_back(warm_max_rows_, 1.0f);
      warm_dummy_f32_scales_ = graph_->dequantized_buffers().back().data();
    }
    if (warm_max_blocks_ > 0) {
      auto bf16_vec = GetSharedWarmDummyBf16Scales(warm_max_blocks_);
      warm_dummy_bf16_scales_ = bf16_vec->data();
      graph_->bf16_scale_refs().push_back(std::move(bf16_vec));
    }
  }
  return absl::OkStatus();
}

absl::Status XnnpackBuildContext::DefineTensorValue(const graph::Tensor& tensor,
                                                    NnpackValue& value) {
  const auto& info = value.info;
  const bool is_external =
      (value.flags & (FlagExternalInput() | FlagExternalOutput())) != 0;
  const bool is_warm_fc =
      !is_external && info.buffer != nullptr &&
      warm_fc_buffers_.contains(info.buffer.get());

  if (info.buffer && !is_external) {
    if (!is_warm_fc) {
      value.data = info.buffer->Lock();
    }
    graph_->keep_alive_buffers().push_back(info.buffer);
  }

  std::vector<size_t> dims;
  dims.reserve(info.shape.size());
  for (int dim : info.shape) {
    if (dim < -1) {
      return absl::InvalidArgumentError(
          absl::StrCat(info.name, ": invalid negative tensor dimension ", dim));
    }
    dims.push_back(dim == -1 ? 0 : static_cast<size_t>(dim));
  }
  const void* data_ptr =
      is_warm_fc ? reinterpret_cast<const void*>(
                       reinterpret_cast<uintptr_t>(info.buffer.get()) |
                       uintptr_t{1})
                 : value.data.data();
  uint32_t external_id = is_external ? value.id : XNN_INVALID_VALUE_ID;
  xnn_subgraph_t sg = subgraph();

  if (!info.quantization) {
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_define_tensor_value(sg, GetXnnpackType(value), dims.size(),
                                dims.empty() ? nullptr : dims.data(), data_ptr,
                                external_id, value.flags, &value.id))
        << "Could not define a new tensor value.";
  } else if (auto maybe_pcq =
                 info.quantization->As<PerChannelAffineQuantization>();
             maybe_pcq.ok()) {
    const auto& pcq = maybe_pcq.value();
    if (pcq.scales.size() == 1) {
      LRT_TENSOR_RETURN_IF_ERROR(xnn_define_quantized_tensor_value(
          sg, GetXnnpackType(value),
          pcq.zero_points.empty() ? 0 : pcq.zero_points[0], pcq.scales[0],
          dims.size(), dims.empty() ? nullptr : dims.data(), data_ptr,
          external_id, value.flags, &value.id))
          << "Could not define a new quantized tensor value.";
    } else {
      bool all_zeros = true;
      for (int64_t zp : pcq.zero_points) {
        if (zp != 0) {
          all_zeros = false;
          break;
        }
      }
      if (!all_zeros) {
        LRT_TENSOR_ASSIGN_OR_RETURN(
            std::vector<float> f32_data,
            DequantizeInt8ConstantTensor(
                info, absl::MakeSpan(value.data.data(), value.data.size())));
        graph_->dequantized_buffers().push_back(std::move(f32_data));
        data_ptr = graph_->dequantized_buffers().back().data();
        LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
            sg, xnn_datatype_fp32, dims.size(),
            dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
            value.flags, &value.id))
            << "Could not define a new tensor value after dequantization.";
      } else {
        if (pcq.quantized_dimension < 0 ||
            static_cast<size_t>(pcq.quantized_dimension) >= dims.size() ||
            pcq.scales.size() < dims[pcq.quantized_dimension]) {
          return absl::InvalidArgumentError(absl::StrCat(
              info.name, ": per-channel scale count (", pcq.scales.size(),
              ") is smaller than the channel dimension size (",
              pcq.quantized_dimension >= 0 &&
                      static_cast<size_t>(pcq.quantized_dimension) < dims.size()
                  ? dims[pcq.quantized_dimension]
                  : static_cast<size_t>(0),
              ")"));
        }
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_channelwise_quantized_tensor_value_v3(
                sg, GetXnnpackType(value), /*zero_point=*/0, pcq.scales.data(),
                dims.size(), pcq.quantized_dimension,
                dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
                value.flags, &value.id, /*channelwise_zero_point=*/nullptr))
            << "Could not define a new channelwise quantized tensor value.";
      }
    }
  } else if (auto maybe_bwq = info.quantization->As<BlockwiseQuantization>();
             maybe_bwq.ok()) {
    const auto& bwq = maybe_bwq.value();
    if (dims.size() < 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name,
          ": blockwise quantized tensor requires at least 2 dimensions"));
    }
    if (bwq.block_size == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": blockwise quantized tensor block_size must be > 0"));
    }
    const size_t expected_block_count = dims[0] * dims[1] / bwq.block_size;
    if (is_warm_fc) {
      if (bwq.NumBlocks() < expected_block_count) {
        return absl::InvalidArgumentError(absl::StrCat(
            info.name, ": blockwise scale count (", bwq.NumBlocks(),
            ") is smaller than block_count (", expected_block_count, ")"));
      }
      if (info.type == Type::kI8) {
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_channelwise_quantized_tensor_value_v3(
                sg, xnn_datatype_qcint8, /*zero_point=*/0,
                warm_dummy_f32_scales_, dims.size(), /*channel_dim=*/0,
                dims.data(), data_ptr, external_id, value.flags, &value.id,
                /*channelwise_zero_point=*/nullptr))
            << "Could not define warm channelwise fallback tensor for "
               "blockwise INT8.";
      } else {
        const bool uses_sym_fallback = bwq.MayHaveFloatZeroPoints() ||
                                       bwq.HasCachedCpuSymmetricFallback() ||
                                       bwq.zero_points.size() > 1;
        const int32_t zero_point =
            uses_sym_fallback
                ? 0
                : (bwq.zero_points.empty() ? 0
                                           : static_cast<int32_t>(
                                                 bwq.zero_points[0]));
        LRT_TENSOR_RETURN_IF_ERROR(
            xnn_define_blockwise_quantized_tensor_value_v2(
                sg, GetXnnpackType(value), zero_point, warm_dummy_bf16_scales_,
                dims.size(), bwq.quantized_dimension, bwq.block_size,
                dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
                value.flags, xnn_datatype_bf16, &value.id))
            << "Could not define warm blockwise quantized tensor value.";
      }
      return absl::OkStatus();
    }
    bwq.EnsureMaterialized();
    if (bwq.scales.size() < expected_block_count) {
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": blockwise scale count (", bwq.scales.size(),
          ") is smaller than block_count (", expected_block_count, ")"));
    }
    if (info.type == Type::kI8) {
      if (dims.size() != 2 || value.data.data() == nullptr) {
        return absl::InvalidArgumentError(absl::StrCat(
            info.name,
            ": blockwise INT8 tensor requires a 2D constant weight buffer"));
      }
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto fallback,
          bwq.GetOrCreateCpuSymmetricFallback(
              Type::kI8, static_cast<int>(dims[0]), static_cast<int>(dims[1]),
              value.data.data(), value.data.size()));
      const float* scales_ptr = fallback->scales.data();
      const void* sym_ptr = fallback->data->data();
      graph_->cpu_fallback_refs().push_back(std::move(fallback));
      LRT_TENSOR_RETURN_IF_ERROR(
          xnn_define_channelwise_quantized_tensor_value_v3(
              sg, xnn_datatype_qcint8, /*zero_point=*/0, scales_ptr,
              dims.size(), /*channel_dim=*/0, dims.data(), sym_ptr, external_id,
              value.flags, &value.id, /*channelwise_zero_point=*/nullptr))
          << "Could not define channelwise fallback tensor for blockwise INT8.";
    } else if (info.type == Type::kI4 &&
               (bwq.HasCachedCpuSymmetricFallback() ||
                bwq.HasNonZeroFloatZeroPoints() ||
                bwq.zero_points.size() > 1)) {
      if (dims.size() != 2 || value.data.data() == nullptr) {
        return absl::InvalidArgumentError(absl::StrCat(
            info.name,
            ": affine blockwise INT4 tensor requires a 2D constant weight "
            "buffer"));
      }
      LRT_TENSOR_ASSIGN_OR_RETURN(
          auto fallback,
          bwq.GetOrCreateCpuSymmetricFallback(
              Type::kI4, static_cast<int>(dims[0]), static_cast<int>(dims[1]),
              value.data.data(), value.data.size()));
      const void* scale_ptr = fallback->bf16_scales->data();
      const void* sym_ptr = fallback->data->data();
      graph_->cpu_fallback_refs().push_back(std::move(fallback));
      LRT_TENSOR_RETURN_IF_ERROR(xnn_define_blockwise_quantized_tensor_value_v2(
          sg, xnn_datatype_qbint4, /*zero_point=*/0, scale_ptr, dims.size(),
          bwq.quantized_dimension, bwq.block_size, dims.data(), sym_ptr,
          external_id, value.flags, xnn_datatype_bf16, &value.id))
          << "Could not define symmetric blockwise INT4 fallback tensor.";
    } else {
      std::shared_ptr<const std::vector<bf16_t>> bf16_scales =
          bwq.GetOrCreateBf16Scales();
      const void* scale_ptr = bf16_scales->data();
      graph_->bf16_scale_refs().push_back(std::move(bf16_scales));
      int32_t zero_point = bwq.zero_points.empty() ? 0 : bwq.zero_points[0];
      LRT_TENSOR_RETURN_IF_ERROR(xnn_define_blockwise_quantized_tensor_value_v2(
          sg, GetXnnpackType(value), zero_point, scale_ptr, dims.size(),
          bwq.quantized_dimension, bwq.block_size,
          dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
          value.flags, xnn_datatype_bf16, &value.id))
          << "Could not define a new blockwise quantized tensor value.";
    }
  } else {
    return absl::UnimplementedError("Unsupported quantization type.");
  }
  return absl::OkStatus();
}

absl::Status XnnpackBuildContext::DefineConstantTensor(
    Type datatype, absl::Span<const size_t> shape, const void* data,
    uint32_t* id) {
  xnn_datatype xnn_type = xnn_datatype_invalid;
  switch (datatype) {
    case Type::kFP32:
      xnn_type = xnn_datatype_fp32;
      break;
    case Type::kFP16:
      xnn_type = xnn_datatype_fp16;
      break;
    case Type::kI32:
      xnn_type = xnn_datatype_int32;
      break;
    case Type::kI8:
      xnn_type = xnn_datatype_qint8;
      break;
    default:
      return absl::InvalidArgumentError("Unsupported constant datatype");
  }

  return XnnStatusToAbsl(
      xnn_define_tensor_value(subgraph(), xnn_type, shape.size(),
                              shape.empty() ? nullptr : shape.data(), data,
                              XNN_INVALID_VALUE_ID, /*flags=*/0, id),
      "xnn_define_tensor_value");
}

absl::Status XnnpackBuildContext::LowerOp(const graph::Operation& op) {
  auto op_ext = op.GetExtension<XnnpackOperation>();
  if (op_ext == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Operation ", op.GetName(), " does not implement XNNPACK operation."));
  }
  return op_ext->ToXnnpack(op, *this);
}

namespace {

enum class WarmFcCacheMode {
  kDisabled,
  kEnabled,
};

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraphImpl(
    std::vector<TensorHandle> outputs, WarmFcCacheMode warm_fc_cache_mode) {
  const bool warm_fc_cache_only =
      (warm_fc_cache_mode == WarmFcCacheMode::kEnabled);
  uint32_t next_id = 0;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids;
  absl::flat_hash_set<const Buffer*> ineligible_buffers;
  for (const TensorHandle& out : outputs) {
    const graph::Tensor raw = out.GetRaw();
    auto [it, inserted] = external_ids.insert({raw, next_id});
    next_id += inserted;
    if (warm_fc_cache_only && raw.group != nullptr) {
      for (const graph::TensorInformation& info : raw.group->tensor_infos) {
        if (info.buffer != nullptr) {
          ineligible_buffers.insert(info.buffer.get());
        }
      }
    }
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto plan, GetExecutionPlan(outputs));
  absl::flat_hash_set<const Buffer*> candidate_fc_buffers;
  size_t max_warm_rows = 0;
  size_t max_warm_blocks = 0;
  for (const graph::Operation* op : plan) {
    const bool is_fc =
        warm_fc_cache_only && op->As<graph::FullyConnectedOperation>().ok();
    for (size_t i = 0; i < op->inputs.size(); ++i) {
      const graph::Tensor& t = op->inputs[i];
      if (auto info_or = graph::GetInfo(t);
          !info_or.ok() || info_or->buffer != nullptr) {
        if (warm_fc_cache_only && t.group != nullptr &&
            t.group->producer == nullptr) {
          if (t.group->tensor_infos.size() != 1) {
            for (const graph::TensorInformation& inf : t.group->tensor_infos) {
              if (inf.buffer != nullptr) {
                ineligible_buffers.insert(inf.buffer.get());
              }
            }
          } else if (info_or.ok() && info_or->buffer != nullptr) {
            bool eligible = false;
            size_t need_rows = 0;
            size_t need_blocks = 0;
            if (is_fc && i == 1 && op->inputs.size() >= 2 &&
                info_or->shape.size() == 2 && info_or->shape[0] > 0 &&
                info_or->shape[1] > 0 &&
                static_cast<uint64_t>(info_or->shape[0]) *
                        static_cast<uint64_t>(info_or->shape[1]) >
                    1ULL) {
              auto act_info = graph::GetInfo(op->inputs[0]);
              const bool valid_act =
                  act_info.ok() &&
                  (act_info->type == Type::kFP32 ||
                   act_info->type == Type::kFP16);
              const bool valid_bias =
                  op->inputs.size() < 3 || op->inputs[2].group == nullptr ||
                  op->inputs[2].group->producer == nullptr;
              if (valid_act && valid_bias) {
                if (info_or->quantization == nullptr) {
                  eligible = (info_or->type == Type::kFP32 ||
                              info_or->type == Type::kFP16 ||
                              info_or->type == Type::kBF16);
                } else if (auto pcq =
                               info_or->quantization
                                   ->As<PerChannelAffineQuantization>();
                           pcq.ok()) {
                  bool all_zps_zero = true;
                  for (int64_t zp : pcq->zero_points) {
                    if (zp != 0) {
                      all_zps_zero = false;
                      break;
                    }
                  }
                  const int qdim = pcq->quantized_dimension;
                  eligible =
                      (info_or->type == Type::kI8 ||
                       info_or->type == Type::kI4 ||
                       info_or->type == Type::kI2) &&
                      pcq->scales.size() > 1 && all_zps_zero && qdim >= 0 &&
                      static_cast<size_t>(qdim) < info_or->shape.size() &&
                      pcq->scales.size() >=
                          static_cast<size_t>(info_or->shape[qdim]);
                } else if (auto bwq =
                               info_or->quantization
                                   ->As<BlockwiseQuantization>();
                           bwq.ok()) {
                  const size_t rows = static_cast<size_t>(info_or->shape[0]);
                  const size_t cols = static_cast<size_t>(info_or->shape[1]);
                  if ((info_or->type == Type::kI8 ||
                       info_or->type == Type::kI4) &&
                      bwq->block_size > 0 &&
                      (rows * cols) % static_cast<size_t>(bwq->block_size) ==
                          0) {
                    const size_t blocks =
                        (rows * cols) / static_cast<size_t>(bwq->block_size);
                    if (bwq->NumBlocks() >= blocks) {
                      eligible = true;
                      if (info_or->type == Type::kI8) {
                        need_rows = rows;
                      } else {
                        need_blocks = blocks;
                      }
                    }
                  }
                }
              }
            }
            if (eligible) {
              candidate_fc_buffers.insert(info_or->buffer.get());
              max_warm_rows = std::max(max_warm_rows, need_rows);
              max_warm_blocks = std::max(max_warm_blocks, need_blocks);
            } else {
              ineligible_buffers.insert(info_or->buffer.get());
            }
          }
        }
        continue;
      }
      if (auto producer_or = graph::GetProducer(t);
          producer_or.ok() && *producer_or != nullptr) {
        continue;
      }
      auto [it, inserted] = external_ids.insert({t, next_id});
      next_id += inserted;
    }
  }

  XnnpackBuildContext ctx(std::move(outputs), std::move(external_ids));
  if (warm_fc_cache_only && !candidate_fc_buffers.empty()) {
    absl::flat_hash_set<const Buffer*> warm_fc_buffers;
    warm_fc_buffers.reserve(candidate_fc_buffers.size());
    for (const Buffer* buf : candidate_fc_buffers) {
      if (!ineligible_buffers.contains(buf)) {
        warm_fc_buffers.insert(buf);
      }
    }
    if (!warm_fc_buffers.empty()) {
      ctx.SetWarmFcBuffers(std::move(warm_fc_buffers), max_warm_rows,
                           max_warm_blocks);
    }
  }
  LRT_TENSOR_RETURN_IF_ERROR(BuildNnpackGraph(ctx));
  LRT_TENSOR_ASSIGN_OR_RETURN(auto graph, ctx.Finalize());
  return std::unique_ptr<XnnpackGraph>(
      static_cast<XnnpackGraph*>(graph.release()));
}

}  // namespace

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
    std::vector<TensorHandle> outputs) {
  return BuildXnnpackGraphImpl(std::move(outputs), WarmFcCacheMode::kDisabled);
}

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraphWithWarmFcCache(
    std::vector<TensorHandle> outputs) {
  return BuildXnnpackGraphImpl(std::move(outputs), WarmFcCacheMode::kEnabled);
}

}  // namespace litert::tensor
