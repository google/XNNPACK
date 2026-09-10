# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//litert/tensor:__subpackages__"])

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    deps = [
        "//litert/tensor",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//litert/tensor:arithmetic_graph",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//litert/tensor:arithmetic_graph",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/utils:matchers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        ":graph",
        "//litert/tensor",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:graph",
        "//litert/tensor/internal:graph_traversal",
        "//litert/tensor/utils:macros",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "conversion_test",
    srcs = ["conversion_test.cc"],
    deps = [
        ":conversion",
        ":graph",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor:datatypes",
        "//litert/tensor/internal:arithmetic_helpers",
        "//litert/tensor/internal:graph",
        "//litert/tensor/internal:type_id",
        "//litert/tensor/utils:matchers",
        "//litert/tensor/utils:source_location",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)
