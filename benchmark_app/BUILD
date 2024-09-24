load("//tools/build_defs/apple:ios.bzl", "ios_application")
#load("//tools/build_defs/apple:macos.bzl", "macos_application")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

objc_library(
    name = "benchmark_lib",
    testonly = 1,
    srcs = ["main.mm"],
    deps = [
        "//:XNNPACK",
        "//:aligned_allocator",
        "//:common",
        "//:config_hdrs",
        "//:internal",
        "//:math",
        "//:microkernels_h",
        "//:node_type",
        "//:operators",
        "//:requantization",
        "//:subgraph",
        "@com_github_google_benchmark//:benchmark_main",
    ],
)

# Application for local benchmarking in XCode
ios_application(
    name = "iOSBenchmark",
    testonly = 1,
    bundle_id = "com.tensorflow.lite.delegates.gpu.metal.benchmark",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = "12.0",
    #provisioning_profile = "//tools/build_defs/apple:google_ios_development_profile",
    #provisioning_profile = "//googlemac/iPhone/Shared/ProvisioningProfiles:Google_Development.mobileprovision",
    deps = [":benchmark_lib"],
)

xcodeproj(
    name = "xcodeproj",
    project_name = "App",
    tags = ["manual"],
    top_level_targets = [
        top_level_target(
            ":iOSBenchmark",
            target_environments = [
                "device",
                "simulator",
            ],
        ),
    ],
)
#macos_application(
#    name = "MacOSBenchmark",
#    testonly = 1,
#    bundle_id = "com.tensorflow.lite.delegates.gpu.metal.benchmark",
#    infoplists = ["Info.plist"],
#    minimum_os_version = "10.13",
#    deps = [":benchmark_lib"],
#)
