load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    # copybara:uncomment includes = ["../utils"],
    deps = [
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_builder",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:source_location",
    ],
)

cc_library(
    name = "mapped_file",
    srcs = ["mapped_file.cc"],
    hdrs = ["mapped_file.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_test(
    name = "mapped_file_test",
    srcs = ["mapped_file_test.cc"],
    deps = [
        ":mapped_file",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "parallel",
    srcs = ["parallel.cc"],
    hdrs = ["parallel.h"],
    linkopts = select({
        "@platforms//os:android": ["-llog"],
        "//conditions:default": [],
    }),
    deps = ["@pthreadpool"],
)

cc_test(
    name = "parallel_test",
    srcs = ["parallel_test.cc"],
    deps = [
        ":parallel",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_test(
    name = "macros_test",
    srcs = ["macros_test.cc"],
    deps = [
        ":macros",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_builder",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/types:source_location",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",  # gunit
    ],
)

cc_library(
    name = "matchers",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        ":macros",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@com_google_googletest//:gtest",  # lib test only
    ],
)

# This should mirror the :matchers target and is provided for compatibility
# with internal targets that are built exclusively for OSS.
cc_library(
    name = "matchers_no_g3",
    testonly = True,
    hdrs = ["matchers.h"],
    deps = [
        ":macros",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@com_google_googletest//:gtest",  # no g3
    ],
)
