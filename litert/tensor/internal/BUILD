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
    name = "mixin",
    hdrs = ["mixin.h"],
)

cc_library(
    name = "arithmetic_helpers",
    hdrs = ["arithmetic_helpers.h"],
    deps = [
        ":graph",
        ":shape",
        ":utils",
        "//litert/tensor",
        "//litert/tensor:datatypes",
        "//litert/tensor/utils:macros",
        "//litert/tensor/utils:source_location",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "compile",
    hdrs = ["compile.h"],
    deps = [
        "//litert/tensor",
        "//litert/tensor:buffer",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_test(
    name = "compile_test",
    srcs = ["compile_test.cc"],
    deps = [
        ":compile",
        "//litert/tensor",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "graph",
    srcs = ["graph.cc"],
    hdrs = ["graph.h"],
    # copybara:uncomment includes = ["../utils"],
    defines = select({
        ":op_debugger_enabled": ["LITERT_OP_DEBUGGER_ENABLED=1"],
        "//conditions:default": [],
    }),
    deps = [
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/utils:macros",
        "//litert/tensor/utils:source_location",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
    ],
)

cc_test(
    name = "graph_test",
    srcs = ["graph_test.cc"],
    deps = [
        ":graph",
        "//litert/tensor:buffer",
        "//litert/tensor:datatypes",
        "//litert/tensor/utils:matchers",
        "//litert/tensor/utils:source_location",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        "@com_google_googletest//:gtest",  # lib test only
    ],
)

# This config_setting checks if the Bazel flag "OP_DEBUGGER" is set to "true"
config_setting(
    name = "op_debugger_enabled",
    define_values = {
        "OP_DEBUGGER": "true",  # The key is the flag name, the value is the expected value
    },
    visibility = ["//visibility:private"],
)

cc_library(
    name = "utils",
    hdrs = ["utils.h"],
    deps = [
        ":graph",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "graph_probe",
    srcs = ["graph_probe.cc"],
    hdrs = ["graph_probe.h"],
    deps = [
        ":graph",
        "//litert/tensor",
        "//litert/tensor/internal:graph_traversal",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "graph_traversal",
    srcs = ["graph_traversal.cc"],
    hdrs = ["graph_traversal.h"],
    deps = [
        ":graph",
        "//litert/tensor",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "graph_traversal_test",
    srcs = ["graph_traversal_test.cc"],
    deps = [
        ":arithmetic_helpers",
        ":graph",
        ":graph_traversal",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor/utils:source_location",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_test(
    name = "op_debugger_test",
    srcs = [
        "op_debugger_test.cc",
    ],
    copts = [
        "-DLITERT_OP_DEBUGGER_ENABLED=1",
    ],
    deps = [
        ":graph",
        "//litert/tensor",
        "//litert/tensor:arithmetic",
        "//litert/tensor:datatypes",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/log:scoped_mock_log",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "fp16",
    hdrs = ["fp16.h"],
)
