workspace(name = "xnnpack")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Bazel rule definitions
http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-master",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/master.zip"],
)

# Google Test framework, used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-master",
    urls = ["https://github.com/google/googletest/archive/master.zip"],
)

# Google Benchmark library, used in micro-benchmarks.
http_archive(
    name = "com_google_benchmark",
    strip_prefix = "benchmark-master",
    urls = ["https://github.com/google/benchmark/archive/master.zip"],
)

# FP16 library, used for half-precision conversions
http_archive(
    name = "FP16",
    strip_prefix = "FP16-3c54eacb74f6f5e39077300c5564156c424d77ba",
    sha256 = "0d56bb92f649ec294dbccb13e04865e3c82933b6f6735d1d7145de45da700156",
    urls = [
        "https://github.com/Maratyszcza/FP16/archive/3c54eacb74f6f5e39077300c5564156c424d77ba.zip",
    ],
    build_file = "@//third_party:FP16.BUILD",
)

# FXdiv library, used for repeated integer division by the same factor
http_archive(
    name = "FXdiv",
    strip_prefix = "FXdiv-f7dd0576a1c8289ef099d4fd8b136b1c4487a873",
    sha256 = "6e4b6e3c58e67c3bb090e286c4f235902c89b98cf3e67442a18f9167963aa286",
    urls = ["https://github.com/Maratyszcza/FXdiv/archive/f7dd0576a1c8289ef099d4fd8b136b1c4487a873.zip"],
)

# pthreadpool library, used for parallelization
http_archive(
    name = "pthreadpool",
    strip_prefix = "pthreadpool-da486afd0f9e2b42ccb90940e2dfba6cfed38708",
    sha256 = "8602a23e9d69cedf6054f428b3f8dbd935e1c5811df3adf38f0985ed1fed74ee",
    urls = ["https://github.com/Maratyszcza/pthreadpool/archive/da486afd0f9e2b42ccb90940e2dfba6cfed38708.zip"],
)

# clog library, used for logging
http_archive(
    name = "clog",
    strip_prefix = "cpuinfo-d5e37adf1406cf899d7d9ec1d317c47506ccb970",
    sha256 = "3f2dc1970f397a0e59db72f9fca6ff144b216895c1d606f6c94a507c1e53a025",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/d5e37adf1406cf899d7d9ec1d317c47506ccb970.tar.gz",
    ],
    build_file = "@//third_party:clog.BUILD",
)

# cpuinfo library, used for detecting processor characteristics
http_archive(
    name = "cpuinfo",
    strip_prefix = "cpuinfo-0cc563acb9baac39f2c1349bc42098c4a1da59e3",
    sha256 = "80625d0b69a3d69b70c2236f30db2c542d0922ccf9bb51a61bc39c49fac91a35",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/0cc563acb9baac39f2c1349bc42098c4a1da59e3.tar.gz",
    ],
    build_file = "@//third_party:cpuinfo.BUILD",
)

# psimd library, used for fallback 128-bit SIMD micro-kernels
http_archive(
    name = "psimd",
    strip_prefix = "psimd-88882f601f8179e1987b7e7cf4a8012c9080ad44",
    sha256 = "c621f9bb1ff9ab8f0fa4a04f3239d13b345a6e865318d7b464aa80531a1abb2c",
    urls = [
        "https://github.com/Maratyszcza/psimd/archive/88882f601f8179e1987b7e7cf4a8012c9080ad44.tar.gz",
    ],
    build_file = "@//third_party:psimd.BUILD",
)

# Ruy library, used to benchmark against
http_archive(
   name = "ruy",
   strip_prefix = "ruy-3d62e9545bd15c5df9ccfdd8453b93d64a6dd8eb",
   sha256 = "ca3024739684d5aba3612072511f508e88b34231ac8d19c8c3cc6598b6bb535d",
   urls = [
       "https://github.com/google/ruy/archive/3d62e9545bd15c5df9ccfdd8453b93d64a6dd8eb.zip",
   ],
)

# Android NDK location and version is auto-detected from $ANDROID_NDK_HOME environment variable
android_ndk_repository(name = "androidndk")

# Android SDK location and API is auto-detected from $ANDROID_HOME environment variable
android_sdk_repository(name = "androidsdk")
