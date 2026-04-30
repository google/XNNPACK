# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "runner",
    srcs = ["runner.cc"],
    hdrs = ["runner.h"],
    deps = [
        "//:XNNPACK",
        "//litert/tensor",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/backends/xnnpack:arithmetic",
        "//litert/tensor/backends/xnnpack:conversion",
        "//litert/tensor/backends/xnnpack:utils",
        "//litert/tensor/internal:graph",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@pthreadpool",
    ],
)

cc_test(
    name = "runner_test",
    srcs = ["runner_test.cc"],
    deps = [
        ":runner",
        "//:XNNPACK",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor:datatypes",
        "//litert/tensor/backends/xnnpack:arithmetic",
        "//litert/tensor/utils:matchers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)
