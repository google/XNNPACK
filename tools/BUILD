# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name = "update_microkernels",
    srcs = [
        "update-microkernels.py",
    ],
    main = "update-microkernels.py",
    tags = [
        "nobuilder",
        "notap",
    ],
    target_compatible_with = select({
        "//build_config:emscripten": ["@platforms//:incompatible"],
        "//conditions:default": [],
    }),
    visibility = ["//:__subpackages__"],
)
