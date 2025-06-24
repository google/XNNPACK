# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cc_library",
    "xnnpack_select_if",
)
load(
    "//:build_srcs.bzl",
    "MICROKERNEL_CONFIG_SRCS",
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
    name = "unary_elementwise_config",
    srcs = ["unary-elementwise-config.c"],
    deps = [
        ":config_hdrs",
        ":hardware_config",
        "//:common",
        "//:init_once",
        "//:logging",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
        "//:xnnpack_h",
    ],
)

xnnpack_cc_library(
    name = "reduce_config",
    srcs = ["reduce-config.c"],
    deps = [
        ":config_hdrs",
        ":hardware_config",
        "//:common",
        "//:init_once",
        "//:logging",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "microkernel_configs",
    hdrs = [
        "//:src/xnnpack/config.h",
    ],
    srcs = MICROKERNEL_CONFIG_SRCS,
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
        "//:pack_lh",
        "//:prod_microkernels",
        "//:xnnpack_h",
    ] + xnnpack_select_if(
        "//:cpuinfo_enabled",
        ["@cpuinfo"],
    ),
)
