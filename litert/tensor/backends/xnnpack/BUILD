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
    name = "arithmetic",
    srcs = ["arithmetic.cc"],
    hdrs = ["arithmetic.h"],
    deps = [
        ":utils",
        "//:XNNPACK",
        "//litert/tensor:arithmetic_graph",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/internal:mixin",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.cc"],
    deps = [
        ":arithmetic",
        ":conversion",
        "//:XNNPACK",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/runners/xnnpack:runner",
        "//litert/tensor/utils:matchers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        ":arithmetic",
        ":utils",
        "//:XNNPACK",
        "//litert/tensor",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/internal:graph_traversal",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
        "//:XNNPACK",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_test(
    name = "conversion_test",
    srcs = ["conversion_test.cc"],
    deps = [
        ":arithmetic",
        ":conversion",
        "//:XNNPACK",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor:datatypes",
        "//litert/tensor/utils:matchers",
        "@abseil-cpp//absl/status",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)
