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

# LINT.IfChange
# Google Test framework, used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    sha256 = "5cb522f1427558c6df572d6d0e1bf0fd076428633d080e88ad5312be0b6a8859",
    strip_prefix = "googletest-e23cdb78e9fef1f69a9ef917f447add5638daf2a",
    urls = ["https://github.com/google/googletest/archive/e23cdb78e9fef1f69a9ef917f447add5638daf2a.zip"],
)
# LINT.ThenChange(cmake/DownloadGoogleTest.cmake)

# LINT.IfChange
# Google Benchmark library, used in micro-benchmarks.
http_archive(
    name = "com_google_benchmark",
    sha256 = "1ba14374fddcd9623f126b1a60945e4deac4cdc4fb25a5f25e7f779e36f2db52",
    strip_prefix = "benchmark-d2a8a4ee41b923876c034afb939c4fc03598e622",
    urls = ["https://github.com/google/benchmark/archive/d2a8a4ee41b923876c034afb939c4fc03598e622.zip"],
)
# LINT.ThenChange(cmake/DownloadGoogleBenchmark.cmake)

# LINT.IfChange
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
# LINT.ThenChange(cmake/DownloadFP16.cmake)

# LINT.IfChange
# FXdiv library, used for repeated integer division by the same factor
http_archive(
    name = "FXdiv",
    sha256 = "ab7dfb08829bee33dca38405d647868fb214ac685e379ec7ef2bebcd234cd44d",
    strip_prefix = "FXdiv-b408327ac2a15ec3e43352421954f5b1967701d1",
    urls = ["https://github.com/Maratyszcza/FXdiv/archive/b408327ac2a15ec3e43352421954f5b1967701d1.zip"],
)
# LINT.ThenChange(cmake/DownloadFXdiv.cmake)

# LINT.IfChange
# pthreadpool library, used for parallelization
http_archive(
    name = "pthreadpool",
    sha256 = "a4cf06de57bfdf8d7b537c61f1c3071bce74e57524fe053e0bbd2332feca7f95",
    strip_prefix = "pthreadpool-4fe0e1e183925bf8cfa6aae24237e724a96479b8",
    urls = ["https://github.com/Maratyszcza/pthreadpool/archive/4fe0e1e183925bf8cfa6aae24237e724a96479b8.zip"],
)
# LINT.ThenChange(cmake/DownloadPThreadPool.cmake)

# LINT.IfChange
# cpuinfo library, used for detecting processor characteristics
http_archive(
    name = "cpuinfo",
    sha256 = "5d7f00693e97bd7525753de94be63f99b0490ae6855df168f5a6b2cfc452e49e",
    strip_prefix = "cpuinfo-3c8b1533ac03dd6531ab6e7b9245d488f13a82a5",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/3c8b1533ac03dd6531ab6e7b9245d488f13a82a5.zip",
    ],
)
# LINT.ThenChange(cmake/DownloadCpuinfo.cmake)

# TODO: b/349993583 - Uncomment when KleidiAI has an official BUILD file.
# LINT.IfChange
# KleidiAI library, used for ARM microkernels.
# http_archive(
#     name = "KleidiAI",
#     sha256 = "39b26d8840ec719afaa480b0622a77952d0f22dbb8e8ba58ec9f93e39895a205",
#     strip_prefix = "kleidiai-1976f8661e8d5aa7d4cdca0f3d2a915e5ecb4c53",
#     urls = [
#         "https://gitlab.arm.com/kleidi/kleidiai/-/archive/1976f8661e8d5aa7d4cdca0f3d2a915e5ecb4c53/kleidiai-1976f8661e8d5aa7d4cdca0f3d2a915e5ecb4c53.zip",
#     ],
# )
# LINT.ThenChange(cmake/DownloadKleidiAI.cmake)

# Ruy library, used to benchmark against
http_archive(
    name = "ruy",
    sha256 = "fe8345f521bb378745ebdd0f8c5937414849936851d2ec2609774eb2d7098e54",
    strip_prefix = "ruy-9f53ba413e6fc879236dcaa3e008915973d67a4f",
    urls = [
        "https://github.com/google/ruy/archive/9f53ba413e6fc879236dcaa3e008915973d67a4f.zip",
    ],
)

http_archive(
    name = "rules_android_ndk",
    sha256 = "f238b4b0323f1e0028a4a3f1093574d70f087867f4b29626469a11eaaf9fd63f",
    strip_prefix = "rules_android_ndk-1ed5be3498d20c8120417fe73b6a5f2b4a3438cc",
    url = "https://github.com/bazelbuild/rules_android_ndk/archive/1ed5be3498d20c8120417fe73b6a5f2b4a3438cc.zip",
)

load("@rules_android_ndk//:rules.bzl", "android_ndk_repository")

android_ndk_repository(name = "androidndk")

register_toolchains("@androidndk//:all")

# Android SDK location and API is auto-detected from $ANDROID_HOME environment variable
android_sdk_repository(name = "androidsdk")
