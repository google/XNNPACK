load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])

cc_library(
    name = "test_main",
    testonly = True,
    srcs = [
        "test_main.cc",
    ],
    visibility = ["//:__subpackages__"],
    deps = ["@com_google_googletest//:gtest"],
    alwayslink = True,
)
