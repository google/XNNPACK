# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_python//python:py_binary.bzl", "py_binary")

py_binary(
    name = "update_microkernels",
    srcs = [
        "update-microkernels.py",
        "xnncommon.py",
    ],
    main = "tools/update-microkernels.py",
    tags = ["notap"],  # This should only be built as part of the related genrule.
    visibility = ["//:__subpackages__"],
)
