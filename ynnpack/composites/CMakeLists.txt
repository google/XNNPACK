# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_composites STATIC
  dot.cc
  dot_quantization.cc
  elementwise.cc
  normalization.cc
  pooling.cc
  reduction.cc
)

TARGET_INCLUDE_DIRECTORIES(ynnpack_composites PUBLIC .)
TARGET_LINK_LIBRARIES(ynnpack_composites PUBLIC
  ynnpack_base
)
