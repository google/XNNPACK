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
    sha256 = "4f45c2213c1ad4de211b15721db4b05c31dc54e244ab26f397345631da327624",
    strip_prefix = "googletest-df1544bcee0c7ce35cd5ea0b3eb8cc81855a4140",
    urls = ["https://github.com/google/googletest/archive/df1544bcee0c7ce35cd5ea0b3eb8cc81855a4140.zip"],
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
    sha256 = "a57f2b93704fe49ad37fa3ef5ce8208443d6b589b89653341332bcf51aa6dde3",
    strip_prefix = "cpuinfo-dff2616ddd49122b63abcf44d2c097483b77f861",
    urls = [
        "https://github.com/pytorch/cpuinfo/archive/dff2616ddd49122b63abcf44d2c097483b77f861.zip",
    ],
)
# LINT.ThenChange(cmake/DownloadCpuinfo.cmake)

# LINT.IfChange
# KleidiAI library, used for ARM microkernels.
http_archive(
    name = "KleidiAI",
    sha256 = "6682b7a2795c711c1dd23ada552675b6514523e991043753648f2cad826f588f",
    strip_prefix = "kleidiai-382b07835c43fcb0401cb4dab3c8fb85eaf187b6",
    urls = [
        "https://gitlab.arm.com/kleidi/kleidiai/-/archive/382b07835c43fcb0401cb4dab3c8fb85eaf187b6/kleidiai-382b07835c43fcb0401cb4dab3c8fb85eaf187b6.zip"
    ],
)
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
