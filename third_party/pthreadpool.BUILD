# Description:
#   Portable pthread-based thread pool for C and C++

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "pthreadpool",
    srcs = [
        "src/threadpool-pthreads.c",
        "src/threadpool-atomics.h",
        "src/threadpool-utils.h",
    ],
    hdrs = [
        "include/pthreadpool.h",
    ],
    copts = [
        "-O2",
    ] + select({
        ":optimized_build": ["-O2"],
        "//conditions:default": [],
    }) + select({
        ":linux_aarch64": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":android_arm64": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":android_armv7": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        "//conditions:default": ["-DPTHREADPOOL_USE_CPUINFO=0"],
    }),
    defines = [
        "PTHREADPOOL_NO_DEPRECATED_API",
    ],
    includes = [
        "include",
    ],
    strip_include_prefix = "include",
    deps = [
        "@FXdiv",
    ] + select({
        ":linux_aarch64": ["@cpuinfo"],
        ":android_arm64": ["@cpuinfo"],
        ":android_armv7": ["@cpuinfo"],
        "//conditions:default": [],
    }),
)

############################# Build configurations #############################

config_setting(
    name = "optimized_build",
    values = {
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "linux_aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "android_armv7",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
)
