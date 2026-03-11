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
        "//:allocator",
        "//:cache",
        "//:internal",
        "//:logging",
        "//:node_type",
        "//:subgraph_h",
        "//:xnnpack_h",
    ],
)
