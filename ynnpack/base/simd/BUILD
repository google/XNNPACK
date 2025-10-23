# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//ynnpack:__subpackages__"])

_COMPATIBLE_WITH = []

cc_library(
    name = "simd",
    hdrs = [
        "multi_vec.h",
        "vec.h",
    ] + select({
        "//ynnpack:x86": [
            "x86_avx.h",
            "x86_avx2.h",
            "x86_avx512.h",
            "x86_sse.h",
            "x86_sse41.h",
        ],
        "//ynnpack:arm": ["arm.h"],
        "//conditions:default": [],
    }),
    compatible_with = _COMPATIBLE_WITH,
    deps = ["//ynnpack/base"],
)
