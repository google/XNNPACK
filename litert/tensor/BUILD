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
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = [
        "tensor.h",
    ],
    # copybara:uncomment includes = ["utils"],
    deps = [
        ":buffer",
        ":datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/utils:macros",
        "//litert/tensor/utils:source_location",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    # copybara:uncomment includes = ["utils"],
    deps = [
        "//litert/tensor:datatypes",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "buffer_test",
    srcs = ["buffer_test.cc"],
    deps = [
        ":buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:matchers",
        "//litert/tensor/utils:matchers",
        "@abseil-cpp//absl/algorithm:container",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "datatypes",
    hdrs = ["datatypes.h"],
    deps = [
        "//litert/tensor/internal:fp16",
    ],
)

cc_test(
    name = "datatypes_test",
    srcs = ["datatypes_test.cc"],
    deps = [
        ":datatypes",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "arithmetic_graph",
    hdrs = ["arithmetic_graph.h"],
    deps = [
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/internal:mixin",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_library(
    name = "arithmetic",
    hdrs = ["arithmetic.h"],
    deps = [
        ":arithmetic_graph",
        ":buffer",
        ":datatypes",
        ":tensor",
        "//litert/tensor/internal:arithmetic_helpers",
        "//litert/tensor/internal:graph",
        "//litert/tensor/utils:macros",
        "//litert/tensor/utils:source_location",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:optional",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "tensor_test",
    srcs = ["tensor_test.cc"],
    deps = [
        ":arithmetic",
        ":tensor",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/utils:matchers",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.cc"],
    deps = [
        ":arithmetic",
        ":arithmetic_graph",
        ":tensor",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/utils:matchers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:optional",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)
