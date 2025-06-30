# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description:
#   XNNPACK - optimized floating-point neural network operators library

workspace(name = "xnnpack")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Bazel rule definitions
http_archive(
    name = "rules_cc",
    sha256 = "3868eab488bd5be37a6acedbd222a196bea14408a2857916f33cce7b4780897d",
    strip_prefix = "rules_cc-5e848c1434d3458018734238dbc4781f43992ea5",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/5e848c1434d3458018734238dbc4781f43992ea5.zip",
    ],
)

# Bazel Python rule definitions.
http_archive(
    name = "rules_python",
    sha256 = "4912ced70dc1a2a8e4b86cec233b192ca053e82bc72d877b98e126156e8f228d",
    strip_prefix = "rules_python-0.32.2",
    urls = [
        "https://github.com/bazelbuild/rules_python/releases/download/0.32.2/rules_python-0.32.2.tar.gz",
    ],
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# Bazel Skylib.
http_archive(
    name = "bazel_skylib",
    sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
    ],
)

# Bazel Platforms
http_archive(
    name = "platforms",
    sha256 = "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
    urls = ["https://github.com/bazelbuild/platforms/releases/download/0.0.6/platforms-0.0.6.tar.gz"],
)

# LINT.IfChange(googletest)
# Google Test framework, used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    sha256 = "648b9430fca63acc68c59ee98f624dcbcd9c24ea6b278c306ab6b7f49f62034a",
    strip_prefix = "googletest-d144031940543e15423a25ae5a8a74141044862f",
    urls = ["https://github.com/google/googletest/archive/d144031940543e15423a25ae5a8a74141044862f.zip"],
)
# LINT.ThenChange(cmake/DownloadGoogleTest.cmake,MODULE.bazel:googletest)

# LINT.IfChange(benchmark)
# Google Benchmark library, used in micro-benchmarks.
http_archive(
    name = "com_google_benchmark",
    sha256 = "28c7cac12cc25d87d3dcc8c5fb7d1bd0971b41a599a5c4787f8742cb39ca47db",
    strip_prefix = "benchmark-8d4fdd6e6e003867045e0bb3473b5b423818e4b7",
    urls = ["https://github.com/google/benchmark/archive/8d4fdd6e6e003867045e0bb3473b5b423818e4b7.zip"],
)
# LINT.ThenChange(cmake/DownloadGoogleBenchmark.cmake,MODULE.bazel:benchmark)

# LINT.IfChange(FXdiv)
# FXdiv library, used for repeated integer division by the same factor
http_archive(
    name = "FXdiv",
    sha256 = "ab7dfb08829bee33dca38405d647868fb214ac685e379ec7ef2bebcd234cd44d",
    strip_prefix = "FXdiv-b408327ac2a15ec3e43352421954f5b1967701d1",
    urls = ["https://github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip"],
)
# LINT.ThenChange(cmake/DownloadFXdiv.cmake,MODULE.bazel:FXdiv)

# LINT.IfChange(pthreadpool)
# pthreadpool library, used for parallelization
http_archive(
    name = "pthreadpool",
    sha256 = "516ba8d05c30e016d7fd7af6a7fc74308273883f857faf92bc9bb630ab6dba2c",
    strip_prefix = "pthreadpool-c2ba5c50bb58d1397b693740cf75fad836a0d1bf",
    urls = ["https://github.com/google/pthreadpool/archive/c2ba5c50bb58d1397b693740cf75fad836a0d1bf.zip"],
)
# LINT.ThenChange(cmake/DownloadPThreadPool.cmake,MODULE.bazel:pthreadpool)

# LINT.IfChange(cpuinfo)
# cpuinfo library, used for detecting processor characteristics
http_archive(
    name = "cpuinfo",
    sha256 = "ae356c4c0c841e20711b5e111a1ccdec9c2f3c1dd7bde7cfba1bed18d6d02459",
    strip_prefix = "cpuinfo-de0ce7c7251372892e53ce9bc891750d2c9a4fd8",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/de0ce7c7251372892e53ce9bc891750d2c9a4fd8.zip",
    ],
)
# LINT.ThenChange(cmake/DownloadCpuinfo.cmake,MODULE.bazel:cpuinfo)

# LINT.IfChange(kleidiai)
# KleidiAI library, used for ARM microkernels.
http_archive(
    name = "KleidiAI",
    sha256 = "a417305bacd09a3312c85617871c678115975cbab2490a7b5d4e25aab4830d0e",
    strip_prefix = "kleidiai-c80d18838af1ecf68f011625db103de497b5a840",
    urls = [
        "https://github.com/ARM-software/kleidiai/archive/c80d18838af1ecf68f011625db103de497b5a840.zip",
    ],
)
# LINT.ThenChange(cmake/DownloadKleidiAI.cmake,MODULE.bazel:kleidiai)

# Ruy library, used to benchmark against
http_archive(
    name = "ruy",
    sha256 = "fe8345f521bb378745ebdd0f8c5937414849936851d2ec2609774eb2d7098e54",
    strip_prefix = "ruy-9f53ba413e6fc879236dcaa3e008915973d67a4f",
    urls = [
        "https://github.com/google/ruy/archive/9f53ba413e6fc879236dcaa3e008915973d67a4f.zip",
    ],
