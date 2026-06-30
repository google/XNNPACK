// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_COMPOSITES_UTIL_H_
#define XNNPACK_YNNPACK_COMPOSITES_UTIL_H_

#include <cstdint>
#include <memory>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

#define YNN_RETURN_IF_ERROR(status)      \
  do {                                   \
    const ynn_status _status = (status); \
    if (_status != ynn_status_success) { \
      return _status;                    \
    }                                    \
  } while (0)

namespace ynn {

using subgraph_ptr =
    std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>;
using runtime_ptr = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>;

inline subgraph_ptr create_subgraph(uint32_t external_value_ids,
                                    uint32_t flags) {
  ynn_subgraph_t subgraph = nullptr;
  ynn_status status = ynn_create_subgraph(external_value_ids, flags, &subgraph);
  if (status != ynn_status_success) {
    return subgraph_ptr(nullptr, ynn_delete_subgraph);
  }
  return subgraph_ptr(subgraph, ynn_delete_subgraph);
}

inline runtime_ptr create_runtime(ynn_subgraph_t subgraph,
                                  ynn_threadpool_t threadpool, uint32_t flags) {
  ynn_runtime_t runtime = nullptr;
  ynn_status status = ynn_create_runtime(subgraph, threadpool, flags, &runtime);
  if (status != ynn_status_success) {
    return runtime_ptr(nullptr, ynn_delete_runtime);
  }
  return runtime_ptr(runtime, ynn_delete_runtime);
}

inline runtime_ptr create_runtime(const subgraph_ptr& subgraph,
                                  ynn_threadpool_t threadpool, uint32_t flags) {
  return create_runtime(subgraph.get(), threadpool, flags);
}

template <typename T>
ynn_status define_constant(ynn_subgraph_t subgraph, T value, uint32_t& id) {
  return ynn_define_tensor(subgraph, type_of<T>(), 0, nullptr, &value,
                           YNN_VALUE_FLAG_COPY_DATA, &id);
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_COMPOSITES_UTIL_H_
