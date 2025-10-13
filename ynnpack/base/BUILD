# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//ynnpack:__subpackages__"])

# Fully disable logging
config_setting(
    name = "log_level_explicit_none",
    define_values = {"ynn_log_level": "none"},
)

# Log fatal errors only
config_setting(
    name = "log_level_explicit_fatal",
    define_values = {"ynn_log_level": "fatal"},
)

# Log fatal and non-fatal errors
config_setting(
    name = "log_level_explicit_error",
    define_values = {"ynn_log_level": "error"},
)

# Log warnings and errors
config_setting(
    name = "log_level_explicit_warning",
    define_values = {"ynn_log_level": "warning"},
)

# Log information messages, warnings and errors
config_setting(
    name = "log_level_explicit_info",
    define_values = {"ynn_log_level": "info"},
)

# Log all messages, including debug messages
config_setting(
    name = "log_level_explicit_debug",
    define_values = {"ynn_log_level": "debug"},
)

# Builds with -c dbg
config_setting(
    name = "debug_build",
    values = {
        "compilation_mode": "dbg",
    },
)

_COMPATIBLE_WITH = []

cc_library(
    name = "log",
    hdrs = ["log.h"],
    compatible_with = _COMPATIBLE_WITH,
    defines = select({
        ":log_level_explicit_debug": ["YNN_LOG_LEVEL=YNN_LOG_LEVEL_DEBUG"],
        ":log_level_explicit_info": ["YNN_LOG_LEVEL=YNN_LOG_LEVEL_INFO"],
        ":log_level_explicit_warning": ["YNN_LOG_LEVEL=YNN_LOG_LEVEL_WARNING"],
        ":log_level_explicit_error": ["YNN_LOG_LEVEL=YNN_LOG_LEVEL_ERROR"],
        ":log_level_explicit_fatal": ["YNN_LOG_LEVEL=YNN_LOG_LEVEL_FATAL"],
        ":log_level_explicit_none": ["YNN_LOG_LEVEL=YNN_LOG_LEVEL_NONE"],
        ":debug_build": ["YNN_LOG_LEVEL=YNN_LOG_LEVEL_DEBUG"],
        "//conditions:default": ["YNN_LOG_LEVEL=YNN_LOG_LEVEL_WARNING"],
    }),
)

cc_library(
    name = "base",
    srcs = [
        "arch.cc",
        "to_string.cc",
        "type.cc",
    ],
    hdrs = [
        "arch.h",
        "arithmetic.h",
        "base.h",
        "bfloat16.h",
        "bit_cast.h",
        "build_config.h",
        "fp16.h",
        "half.h",
        "to_string.h",
        "type.h",
    ],
    compatible_with = _COMPATIBLE_WITH,
    deps = [
        ":log",
        "//ynnpack:ynnpack_h",
        "@cpuinfo",
    ],
)
