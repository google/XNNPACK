workspace(name = "xnnpack")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Bazel rule definitions
http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-main",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/main.zip"],
)

# Bazel Skylib.
http_archive(
    name = "bazel_skylib",
    sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
    ],
)

# Google Test framework, used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-main",
    urls = ["https://github.com/google/googletest/archive/main.zip"],
)

# Google Benchmark library, used in micro-benchmarks.
http_archive(
    name = "com_google_benchmark",
    strip_prefix = "benchmark-main",
    urls = ["https://github.com/google/benchmark/archive/main.zip"],
)

# FP16 library, used for half-precision conversions
http_archive(
    name = "FP16",
    build_file = "@//third_party:FP16.BUILD",
    sha256 = "e66e65515fa09927b348d3d584c68be4215cfe664100d01c9dbc7655a5716d70",
    strip_prefix = "FP16-0a92994d729ff76a58f692d3028ca1b64b145d91",
    urls = [
        "https://github.com/Maratyszcza/FP16/archive/0a92994d729ff76a58f692d3028ca1b64b145d91.zip",
    ],
)

# FXdiv library, used for repeated integer division by the same factor
http_archive(
    name = "FXdiv",
    sha256 = "ab7dfb08829bee33dca38405d647868fb214ac685e379ec7ef2bebcd234cd44d",
    strip_prefix = "FXdiv-b408327ac2a15ec3e43352421954f5b1967701d1",
    urls = ["https://github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip"],
)

# pthreadpool library, used for parallelization
http_archive(
    name = "pthreadpool",
    sha256 = "e6370550a1abf1503daf3c2c196e0a1c2b253440c39e1a57740ff49af2d8bedf",
    strip_prefix = "pthreadpool-43edadc654d6283b4b6e45ba09a853181ae8e850",
    urls = ["https://github.com/Maratyszcza/pthreadpool/archive/43edadc654d6283b4b6e45ba09a853181ae8e850.zip"],
)

# clog library, used for logging
http_archive(
    name = "clog",
    build_file = "@//third_party:clog.BUILD",
    sha256 = "25843b5f21c32cba89f9b921c0500ab5cd0c2cb8fb0f345e5b5e4678329386c7",
    strip_prefix = "cpuinfo-49610f89b8b1eb52d75d1eda7a2c40c1e86a78e7",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/49610f89b8b1eb52d75d1eda7a2c40c1e86a78e7.zip",
    ],
)

# cpuinfo library, used for detecting processor characteristics
http_archive(
    name = "cpuinfo",
    build_file = "@//third_party:cpuinfo.BUILD",
    sha256 = "25843b5f21c32cba89f9b921c0500ab5cd0c2cb8fb0f345e5b5e4678329386c7",
    strip_prefix = "cpuinfo-49610f89b8b1eb52d75d1eda7a2c40c1e86a78e7",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/49610f89b8b1eb52d75d1eda7a2c40c1e86a78e7.zip",
    ],
)

# Ruy library, used to benchmark against
http_archive(
    name = "ruy",
    sha256 = "fe8345f521bb378745ebdd0f8c5937414849936851d2ec2609774eb2d7098e54",
    strip_prefix = "ruy-9f53ba413e6fc879236dcaa3e008915973d67a4f",
    urls = [
        "https://github.com/google/ruy/archive/9f53ba413e6fc879236dcaa3e008915973d67a4f.zip",
    ],
)

# Android NDK location and version is auto-detected from $ANDROID_NDK_HOME environment variable
android_ndk_repository(name = "androidndk")

# Android SDK location and API is auto-detected from $ANDROID_HOME environment variable
android_sdk_repository(name = "androidsdk")
