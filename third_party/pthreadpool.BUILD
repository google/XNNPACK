# Description:
#   Portable pthread-based thread pool for C and C++

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

INTERNAL_HDRS = [
    "src/threadpool-atomics.h",
    "src/threadpool-common.h",
    "src/threadpool-object.h",
    "src/threadpool-utils.h",
]

PORTABLE_SRCS = [
    "src/memory.c",
    "src/portable-api.c",
]

PTHREADS_IMPL_SRCS = PORTABLE_SRCS + ["src/pthreads.c"]

GCD_IMPL_SRCS = PORTABLE_SRCS + ["src/gcd.c"]

SHIM_IMPL_SRCS = ["src/shim.c"]

cc_library(
    name = "pthreadpool",
    srcs = select({
        ":pthreadpool_sync_primitive_explicit_condvar": INTERNAL_HDRS + PTHREADS_IMPL_SRCS,
        ":pthreadpool_sync_primitive_explicit_futex": INTERNAL_HDRS + PTHREADS_IMPL_SRCS,
        ":pthreadpool_sync_primitive_explicit_gcd": INTERNAL_HDRS + GCD_IMPL_SRCS,
        ":emscripten_with_threads": INTERNAL_HDRS + PTHREADS_IMPL_SRCS,
        ":emscripten": SHIM_IMPL_SRCS,
        ":macos_x86": INTERNAL_HDRS + GCD_IMPL_SRCS,
        ":macos_x86_64": INTERNAL_HDRS + GCD_IMPL_SRCS,
        ":ios": INTERNAL_HDRS + GCD_IMPL_SRCS,
        "//conditions:default": INTERNAL_HDRS + PTHREADS_IMPL_SRCS,
    }),
    copts = [
        "-std=gnu11",
        "-Wno-deprecated-declarations",
    ] + select({
        ":optimized_build": ["-O2"],
        "//conditions:default": [],
    }) + select({
        ":linux_arm": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":linux_aarch64": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":android_armv7": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        ":android_arm64": ["-DPTHREADPOOL_USE_CPUINFO=1"],
        "//conditions:default": ["-DPTHREADPOOL_USE_CPUINFO=0"],
    }) + select({
        ":pthreadpool_sync_primitive_explicit_condvar": [
            "-DPTHREADPOOL_USE_CONDVAR=1",
            "-DPTHREADPOOL_USE_FUTEX=0",
            "-DPTHREADPOOL_USE_GCD=0",
        ],
        ":pthreadpool_sync_primitive_explicit_futex": [
            "-DPTHREADPOOL_USE_CONDVAR=0",
            "-DPTHREADPOOL_USE_FUTEX=1",
            "-DPTHREADPOOL_USE_GCD=0",
        ],
        ":pthreadpool_sync_primitive_explicit_gcd": [
            "-DPTHREADPOOL_USE_CONDVAR=0",
            "-DPTHREADPOOL_USE_FUTEX=0",
            "-DPTHREADPOOL_USE_GCD=1",
        ],
        "//conditions:default": [],
    }),
    hdrs = [
        "include/pthreadpool.h",
    ],
    defines = [
        "PTHREADPOOL_NO_DEPRECATED_API",
    ],
    includes = [
        "include",
    ],
    linkopts = select({
        ":emscripten_with_threads": [
            "-s ALLOW_BLOCKING_ON_MAIN_THREAD=1",
            "-s PTHREAD_POOL_SIZE=8",
        ],
        "//conditions:default": [],
    }),
    strip_include_prefix = "include",
    deps = [
        "@FXdiv",
    ] + select({
        ":linux_arm": ["@cpuinfo"],
        ":linux_aarch64": ["@cpuinfo"],
        ":android_armv7": ["@cpuinfo"],
        ":android_arm64": ["@cpuinfo"],
        "//conditions:default": [],
    }),
)

############################# Build configurations #############################

# Synchronize workers using pthreads condition variable.
config_setting(
    name = "pthreadpool_sync_primitive_explicit_condvar",
    define_values = {"pthreadpool_sync_primitive": "condvar"},
)

# Synchronize workers using futex.
config_setting(
    name = "pthreadpool_sync_primitive_explicit_futex",
    define_values = {"pthreadpool_sync_primitive": "futex"},
)

# Synchronize workers using Grand Central Dispatch.
config_setting(
    name = "pthreadpool_sync_primitive_explicit_gcd",
    define_values = {"pthreadpool_sync_primitive": "gcd"},
)

config_setting(
    name = "optimized_build",
    values = {
        "compilation_mode": "opt",
    },
)

config_setting(
    name = "linux_arm",
    values = {"cpu": "arm"},
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

# Note: we need to individually match x86 and x86-64 macOS rather than use
# catch-all "apple_platform_type": "macos" because that option defaults to
# "macos" even when building on Linux!
config_setting(
    name = "macos_x86",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_x86_64",
    },
)

config_setting(
    name = "ios",
    values = {
        "crosstool_top": "@bazel_tools//tools/cpp:toolchain",
        "apple_platform_type": "ios",
    },
)

config_setting(
    name = "emscripten",
    values = {
        "crosstool_top": "//toolchain:emscripten",
    }
)

config_setting(
    name = "emscripten_with_threads",
    values = {
        "crosstool_top": "//toolchain:emscripten",
        "copt": "-pthread",
    }
)
