# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load(
    "//:build_defs.bzl",
    "xnnpack_cc_library",
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
    name = "dwconv2d_chw_config",
    srcs = ["dwconv2d-chw-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "unary_elementwise_config",
    srcs = ["unary-elementwise-config.c"],
    deps = [
        ":config_hdrs",
        ":hardware_config",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "unpool_config",
    srcs = ["unpool-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "avgpool_config",
    srcs = ["avgpool-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "x8_lut_config",
    srcs = ["x8-lut-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "conv_hwc2chw_config",
    srcs = ["conv-hwc2chw-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "binary_elementwise_config",
    srcs = ["binary-elementwise-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "maxpool_config",
    srcs = ["maxpool-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "argmaxpool_config",
    srcs = ["argmaxpool-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "vmulcaddc_config",
    srcs = ["vmulcaddc-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "transpose_config",
    srcs = ["transpose-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "spmm_config",
    srcs = ["spmm-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "ibilinear_config",
    srcs = ["ibilinear-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:indirection",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "ibilinear_chw_config",
    srcs = ["ibilinear-chw-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:indirection",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
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
        "@cpuinfo",
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
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "dwconv_config",
    srcs = ["dwconv-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
        "@cpuinfo",
    ],
)

xnnpack_cc_library(
    name = "gemm_config",
    srcs = ["gemm-config.c"],
    deps = [
        ":config_hdrs",
        ":hardware_config",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
        "//:packing",
        "//:xnnpack_h",
        "@cpuinfo",
    ],
)

xnnpack_cc_library(
    name = "pack_lh_config",
    srcs = ["pack-lh-config.c"],
    deps = [
        ":config_hdrs",
        ":hardware_config",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "cmul_config",
    srcs = ["cmul-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "raddstoreexpminusmax_config",
    srcs = ["raddstoreexpminusmax-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
        "//:microparams_init",
    ],
)

xnnpack_cc_library(
    name = "lut32norm_config",
    srcs = ["lut32norm-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "xx_pad_config",
    srcs = ["xx-pad-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "xx_fill_config",
    srcs = ["xx-fill-config.c"],
    deps = [
        ":config_hdrs",
        "//:common",
        "//:init_once",
        "//:microfnptr_h",
        "//:microkernel_hdrs",
    ],
)

xnnpack_cc_library(
    name = "microkernel_configs",
    hdrs = [
        "//:src/xnnpack/config.h",
    ],
    copts = select({
        "//:cpuinfo_enabled": ["-DXNN_ENABLE_CPUINFO=1"],
        "//conditions:default": ["-DXNN_ENABLE_CPUINFO=0"],
    }),
    deps = [
        ":argmaxpool_config",
        ":avgpool_config",
        ":binary_elementwise_config",
        ":cmul_config",
        ":config_hdrs",
        ":conv_hwc2chw_config",
        ":dwconv2d_chw_config",
        ":dwconv_config",
        ":gemm_config",
        ":ibilinear_chw_config",
        ":ibilinear_config",
        ":lut32norm_config",
        ":maxpool_config",
        ":pack_lh_config",
        ":raddstoreexpminusmax_config",
        ":reduce_config",
        ":spmm_config",
        ":transpose_config",
        ":unary_elementwise_config",
        ":unpool_config",
        ":vmulcaddc_config",
        ":x8_lut_config",
        ":xx_fill_config",
        ":xx_pad_config",
        "//:common",
        "//:microfnptr_h",
        "//:prod_microkernels",
    ],
)
