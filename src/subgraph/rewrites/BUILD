load(
    "//:build_defs.bzl",
    "xnnpack_cxx_library",
)

package(default_visibility = ["//:__subpackages__"])

xnnpack_cxx_library(
    name = "fp16_to_fp32",
    srcs = ["fp16_to_fp32.cc"],
    hdrs = ["fp16_to_fp32.h"],
    deps = [
        "//:allocation_type",
        "//:allocator",
        "//:cache",
        "//:fp16",
        "//:internal",
        "//:logging",
        "//:node_type",
        "//:operator_h",
        "//:subgraph_h",
        "//:xnnpack_h",
        "//src/configs:config_hdrs",
        "//src/configs:microkernel_configs",  # build_cleaner: keep
    ],
)
