# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---[ Slinky thread pool.
ADD_LIBRARY(ynnpack_slinky_thread_pool STATIC
  slinky_thread_pool.cc
)

TARGET_LINK_LIBRARIES(ynnpack_slinky_thread_pool PUBLIC
  ynnpack_base
  slinky_base
  slinky_thread_pool_impl
)

# ---[ Subgraph library.
ADD_LIBRARY(ynnpack_subgraph STATIC
  broadcast.cc
  broadcast_like.cc
  concatenate.cc
  copy.cc
  dot.cc
  elementwise.cc
  even_split.cc
  fusion.cc
  fusion_lut.cc
  fusion_types.cc
  get_tensor_shape.cc
  reduce.cc
  runtime.cc
  slinky.cc
  stack.cc
  static_pad.cc
  static_slice.cc
  static_transpose.cc
  stencil_copy.cc
  subgraph.cc
  tensor.cc
  threadpool.cc
  utils.cc
)

TARGET_LINK_LIBRARIES(ynnpack_subgraph PUBLIC
  ynnpack_base
  ynnpack_slinky_thread_pool
  ynnpack_kernels_binary
  ynnpack_kernels_dot
  ynnpack_kernels_lut
  ynnpack_kernels_reduce
  ynnpack_kernels_ternary
  ynnpack_kernels_transpose
  ynnpack_kernels_unary
  slinky_builder
  slinky_runtime
)

# Handle optional Perfetto.
SET(YNN_ENABLE_PERFETTO OFF CACHE BOOL "Enable Perfetto tracing for YNNPACK")
IF(YNN_ENABLE_PERFETTO)
  # This would require finding/adding perfetto.
  # ADD_LIBRARY(ynnpack_perfetto STATIC perfetto.cc)
  # TARGET_COMPILE_DEFINITIONS(ynnpack_subgraph PRIVATE YNN_ENABLE_PERFETTO)
  # TARGET_LINK_LIBRARIES(ynnpack_subgraph PRIVATE ynnpack_perfetto)
ENDIF()

IF(YNNPACK_BUILD_TESTS)
  ADD_SUBDIRECTORY(test)
ENDIF()
