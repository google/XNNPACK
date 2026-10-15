# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load(
    "//:build_defs.bzl",
    "xnnpack_configurable_defines",
)

cc_binary(
    name = "probe",
    srcs = ["probe.cc"],
    defines = xnnpack_configurable_defines(),
    includes = ["../.."],
    linkopts = select({
        "//build_config:windows": [],
        "//build_config:android": [
            "-ldl",
            "-llog",
        ],
        "//conditions:default": ["-ldl"],
    }),
    linkstatic = True,
    tags = ["manual"],
    deps = [
        "//:XNNPACK",
        "//:operators",
        "//:subgraph",
        "@pthreadpool",
    ],
)
