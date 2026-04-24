# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ADD_LIBRARY(ynnpack_kernels_lut STATIC lut.cc)
TARGET_LINK_LIBRARIES(ynnpack_kernels_lut PUBLIC ynnpack_base)
