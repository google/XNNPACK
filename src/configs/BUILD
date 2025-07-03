# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cc_library",
    "xnnpack_select_if",
)

package(default_visibility = ["//:__subpackages__"])

xnnpack_cc_library(
    name = "config_h",
    hdrs = [
        "//:src/xnnpack/config.h",
    ],
    deps = [
        ":hardware_config",
        "//:common",
        "//:microfnptr_h",
    ],
)

xnnpack_cc_library(
    name = "config_types",
    hdrs = [
        "//:src/xnnpack/config-types.h",
    ],
    deps = [
        ":hardware_config",
        "//:common",
        "//:microfnptr_h",
    ],
)

xnnpack_cc_library(
    name = "config_hdrs",
    hdrs = [
        "//:src/xnnpack/config.h",
        "//:src/xnnpack/config-types.h",
    ],
    deps = [
        ":hardware_config",
        "//:common",
        "//:microfnptr_h",
    ],
)

xnnpack_cc_library(
    name = "hardware_config",
    srcs = ["hardware-config.c"],
    hdrs = ["//:src/xnnpack/hardware-config.h"],
    deps = [
        "//:common",
        "//:init_once",
        "//:logging",
    ] + xnnpack_select_if(
        "//:cpuinfo_enabled",
        ["@cpuinfo"],
    ),
)

xnnpack_cc_library(
    name = "microkernel_configs",
    srcs = [
        "argmaxpool-config.c",
        "avgpool-config.c",
        "binary-elementwise-config.c",
        "cmul-config.c",
        "conv-hwc2chw-config.c",
        "dwconv-config.c",
        "dwconv2d-chw-config.c",
        "gemm-config.c",
        "ibilinear-chw-config.c",
        "ibilinear-config.c",
        "lut32norm-config.c",
        "maxpool-config.c",
        "pack-lh-config.c",
        "raddstoreexpminusmax-config.c",
        "reduce-config.c",
        "spmm-config.c",
        "transpose-config.c",
        "unary-elementwise-config.c",
        "unpool-config.c",
        "vmulcaddc-config.c",
        "x8-lut-config.c",
        "xx-fill-config.c",
        "xx-pad-config.c",
    ],
    hdrs = [
        "//:src/xnnpack/config.h",
    ],
    deps = [
        ":config_hdrs",
        ":hardware_config",
        "//:common",
        "//:indirection",
        "//:init_once",
        "//:logging",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
        "//:packing",
        "//:prod_microkernels",
        "//:xnnpack_h",
    ] + xnnpack_select_if(
        "//:cpuinfo_enabled",
        ["@cpuinfo"],
    ),
)
