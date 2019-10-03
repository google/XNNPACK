licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "benchmark",
    srcs = glob(["src/*.h", "src/*.cc",]),
    hdrs = glob(["include/benchmark/*.h"]),
    copts = [
        "-DHAVE_POSIX_REGEX",
        "-Wno-deprecated-declarations",
    ],
    linkopts = select({
        ":linux_x86_64": ["-lm"],
        ":linux_arm64": ["-lm"],
        ":android": ["-lm"],
        "//conditions:default": [],
    }),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_arm64",
    values = {"cpu": "aarch64"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
    visibility = ["//visibility:public"],
)
