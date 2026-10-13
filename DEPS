# Copyright 2026 Google LLC
# Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file is used to manage the dependencies of the XNNPACK repo
# for use with the GN build system. It is losely based on what they do for V8:
# https://chromium.googlesource.com/v8/v8.git/+/refs/heads/main/DEPS
# Dependencies are static for now, automation for updating these will be added
# in the future. This is a fairly minimal config for getting GN builds working.
# Support for DEPS and GN is experimental and may be removed in the future.

# Keeps things together in the XNNPACK directory.
use_relative_paths = True

skip_child_includes = [
  'build',
  'third_party',
]

gclient_gn_args_file = 'build/config/gclient_args.gni'
gclient_gn_args = [
  'android_ndk_version',
  'build_with_chromium',
  'generate_location_tags',
]


# Expect everything to be up-to-date with submodules.
git_dependencies = 'SYNC'

vars = {
  'android_ndk_version': Str('2@30.0.16248370'),
  'chromium_url': 'https://chromium.googlesource.com',
  'generate_location_tags': False,
  # GN CIPD package version.
  'gn_version': 'git_revision:0035961837fbcfe875d855744b3eda4fd2eefb43',
  # ninja CIPD package.
  'ninja_package': 'infra/3pp/tools/ninja/',
  # ninja CIPD package version.
  # https://chrome-infra-packages.appspot.com/p/infra/3pp/tools/ninja
  'ninja_version': 'version:3@1.12.1.chromium.4',
  # This variable is overidden in Chromium's DEPS file.
  'build_with_chromium': False,
  # Fetch the prebuilt binaries for llvm-cov and llvm-profdata. Needed to
  # process the raw profiles produced by instrumented targets (built with
  # the gn arg 'use_clang_coverage').
  'checkout_clang_coverage_tools': False,
  # Fetch clang-tidy into the same bin/ directory as our clang binary.
  'checkout_clang_tidy': False,
  # Fetch clangd into the same bin/ directory as our clang binary.
  'checkout_clangd': False,
  # Fetch the KleidiAI project for additional kernels on Arm.
  'checkout_kleidiai': False,
  'rbe_instance': Str('projects/rbe-chrome-untrusted/instances/default_instance'),
  'siso_version': 'git_revision:b521b230cb7c56c821235e498285cf8e548a73ee',
  # RBE project to download rewrapper config files for. Only needed if
  # different from the project used in 'rbe_instance'
  'rewrapper_cfg_project': Str(''),
  # Fetch configuration files required for the 'use_remoteexec' gn arg
  'download_remoteexec_cfg': False,
  # reclient CIPD package version
  'reclient_version': 're_client_version:0.185.0.db415f21-gomaip',
}

deps = {
  'build':
    Var('chromium_url') + '/chromium/src/build.git' + '@' + 'aa268127b8cc571ebc60f200d052a09d27231843',
  'buildtools':
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '8790f824690e131a78f36e80ecc05a9272f061c8',
  'buildtools/linux64': {
    'packages': [
      {
        'package': 'gn/gn/linux-${{arch}}',
        'version': Var('gn_version'),
      }
    ],
    'dep_type': 'cipd',
    'condition': 'host_os == "linux" and host_cpu != "s390x" and host_os != "zos" and host_cpu != "ppc64"',
  },
  'buildtools/mac': {
    'packages': [
      {
        'package': 'gn/gn/mac-${{arch}}',
        'version': Var('gn_version'),
      }
    ],
    'dep_type': 'cipd',
    'condition': 'host_os == "mac"',
  },
  'buildtools/win': {
    'packages': [
      {
        'package': 'gn/gn/windows-amd64',
        'version': Var('gn_version'),
      }
    ],
    'dep_type': 'cipd',
    'condition': 'host_os == "win"',
  },
  'testing':
    Var('chromium_url') + '/chromium/src/testing' + '@' + 'afda462de52535e9cbb4df40c165d77643881319',
  'third_party/libpfm4':
    Var('chromium_url') + '/chromium/src/third_party/libpfm4.git' + '@' + 'cf4f1a05f7ec16fe8c82b1635b9978f1d7c5d976',
  'third_party/libpfm4/src':
    Var('chromium_url') + '/external/git.code.sf.net/p/perfmon2/libpfm4.git' + '@' + '995ad37079531b7c3a98cbd164b7619f3b0adc0d',
  'third_party/googletest/src':
    Var('chromium_url') + '/external/github.com/google/googletest.git' + '@' + '4fe3307fb2d9f86d19777c7eb0e4809e9694dde7',
  'third_party/google_benchmark': {
    'url': Var('chromium_url') + '/chromium/src/third_party/google_benchmark.git' + '@' + 'c3b654389bf74fac1f2d926d0439506f06c66751',
  },
  'third_party/google_benchmark/src': {
    'url': Var('chromium_url') + '/external/github.com/google/benchmark.git' + '@' + '8abf1e701fbd88c8170f48fe0558247e2e5f8e7d',
  },
  'third_party/abseil-cpp': {
    'url': Var('chromium_url') + '/chromium/src/third_party/abseil-cpp' + '@' + 'df548c50b2cda67158364d3d23c63043881b391d',
  },
  'third_party/kleidiai/src': {
    'url': 'https://gitlab.arm.com/kleidi/kleidiai@v1.30.0',
    'condition': 'checkout_kleidiai'
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + '97b436da4c33663581d394f4ee0a5977fc38c2f4',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '0e691bb69ac90450d8d53bc8ee5968c62f9c15a7',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + '97e60a1e7f7180965cc84e22ddde9f1d4ffd640f',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + '15a6644ba1c45f1acc16ac1e883efc3e56c6bed2',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',
  'third_party/cpuinfo/src':
    Var('chromium_url') + '/external/github.com/pytorch/cpuinfo.git' + '@' + '66ee79c038d70dad9f08705b2c9b3e58f6d8f512',
  'third_party/libunwind/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libunwind.git' + '@' + '9c0490ec858b3201b9aedbe219b530b95dc0be7e',
  'third_party/android_toolchain/ndk': {
    'packages': [
      {
        'package': 'chromium/third_party/android_toolchain/android_toolchain/${{platform}}',
        'version': 'version:' + Var('android_ndk_version'),
      },
    ],
    'condition': 'checkout_android',
    'dep_type': 'cipd',
  },
  'third_party/ninja': {
    'packages': [
      {
        'package': Var('ninja_package') + '${{platform}}',
        'version': Var('ninja_version'),
      }
    ],
    'condition': 'non_git_source',
    'dep_type': 'cipd',
  },
  'third_party/llvm-build/Release+Asserts': {
    'dep_type': 'gcs',
    'bucket': 'chromium-browser-clang',
    'objects': [
      {
        'object_name': 'Linux_x64/clang-android-runtime-library-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '2963958c20b4a199ece3b3687d0a6837034f9334e577a1b41f731286899529d0',
        'size_bytes': 6214188,
        'generation': 1791353531581542,
        'condition': 'checkout_android',
      },
      {
        'object_name': 'Linux_x64/clang-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'ab8ec1e75c7f74565e9f893212262780abc9174363ce5f1ff670c415e9cd4147',
        'size_bytes': 57258740,
        'generation': 1791353523054521,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '6cb141f7c28a13391c93536e00f41a95bd496d9fb0e2b6e75888580ec855c652',
        'size_bytes': 14939640,
        'generation': 1791353523580790,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'e15fc9da48194273e35112245a74b1ff692cc3f3323d952d4fc2629fd5cdaf72',
        'size_bytes': 15087948,
        'generation': 1791353523614177,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'b69495731e398beb837f5261f62408e9d2bb44f255b4dbd403b3deeb6d8b0646',
        'size_bytes': 2374572,
        'generation': 1791353524193645,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'c575a16aa6516a47ad58c6f66ea604d3ec183d9b7a6ca72139b10d01a3ade984',
        'size_bytes': 5924452,
        'generation': 1791353523760046,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '346a08e3a7891deb809cfbbe3f17c87a64145ac6d6a37f58fed8fe18f3e89913',
        'size_bytes': 53865192,
        'generation': 1791353549543090,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '01e3edd028295244eb2f18c96d00348a3d0b5269cef17d90730a15bc6ac0c6b2',
        'size_bytes': 982956,
        'generation': 1791353573947124,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '18e694a749e4cab12ccf9bcf17ca526188edf41aa7345d70cdbf311a99750f30',
        'size_bytes': 14229572,
        'generation': 1791353549836984,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '2409045110c36e6e2c0ecd7684d717f49220dbc9a31caa884cc99822da3c81eb',
        'size_bytes': 15130608,
        'generation': 1791353549992613,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'd7d165fe73ebea5d0c7cc43d55a4ca0b98a5d55465702b0b0de8db9fe953bdbd',
        'size_bytes': 2260504,
        'generation': 1791353550517983,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '18797773ddcd446afcfd9420b51c9f43e6eb805bb0ddeabcfc0fadfe967cd818',
        'size_bytes': 5525060,
        'generation': 1791353550148753,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '7ac4fb2a7fd4996bbade60ce75835d46f0bc3e89c9d994d0510c81d53b0bb2bf',
        'size_bytes': 44479696,
        'generation': 1791353576418315,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'ec772cde3504c162f0be72487e8cf7a1c971dd747d1573b0a5a8642ff8076f9d',
        'size_bytes': 12231584,
        'generation': 1791353576774358,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'c0f299d65d980089505e97367b7a5e08c211205c64ee6d9d87d798f24a602331',
        'size_bytes': 12228724,
        'generation': 1791353576955805,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'ed52f79362b287347e30deaeab32288af97be54c60e9c067505426d2b97c08b9',
        'size_bytes': 1861748,
        'generation': 1791353577847292,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '50392b193a67eedda0c6b27b228c9beb598ffe5beb4624831c8ba66b00f9208c',
        'size_bytes': 5225884,
        'generation': 1791353577184760,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': '194d833174c9ab6ea1d5e5f9902fcc946a4b713c5ef573e112dfc48450c849e4',
        'size_bytes': 51825724,
        'generation': 1791353603126632,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'abbc16b3fbcc2fe14e1fd78fa33da69b745ad3485becfd39ee291049d01894e6',
        'size_bytes': 15137260,
        'generation': 1791353603446047,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'b207042a5986ea8a684ebea9ea918220bc034d05dc7c86dd0a2712c368358e39',
        'size_bytes': 2653336,
        'generation': 1791353626758916,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'f3d4c9447d50f6f8c50828a321cc625dbc8345c3928c2a2950fdf50077b8fbce',
        'size_bytes': 15447340,
        'generation': 1791353603658278,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'c6ffba53da37b3d037bceea588a7cc5781361f018998733cc70c897e13f53168',
        'size_bytes': 2527436,
        'generation': 1791353604155593,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-24-init-7747-g62397f8b-67.tar.xz',
        'sha256sum': 'dbc2063e2ca76a1917a3347ff7d6807a450d14bb7d11bc3c98c3ef8d428f63fe',
        'size_bytes': 6010708,
        'generation': 1791353604178293,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + '47fbf574c4f14e7288da3e51ff6eca0990183377',
  'third_party/siso': {
    'packages': [
      {
        'package': 'build/siso/${{platform}}',
        'version': Var('siso_version'),
      }
    ],
    'dep_type': 'cipd',
    'condition': 'not build_with_chromium and host_cpu != "s390x" and host_os != "zos" and host_cpu != "ppc64"',
  },
}

hooks = [
  {
    # Update the Windows toolchain if necessary.
    'name': 'win_toolchain',
    'pattern': '.',
    'condition': 'checkout_win',
    'action': ['python3', 'build/vs_toolchain.py', 'update', '--force'],
  },
  {
    # Update the Mac toolchain if necessary.
    'name': 'mac_toolchain',
    'pattern': '.',
    'condition': 'checkout_mac',
    'action': ['python3', 'build/mac_toolchain.py'],
  },
  # Configure remote exec cfg files
  {
    'name': 'download_and_configure_reclient_cfgs',
    'pattern': '.',
    'condition': 'download_remoteexec_cfg and not build_with_chromium',
    'action': ['python3',
               'buildtools/reclient_cfgs/configure_reclient_cfgs.py',
               '--rbe_instance',
               Var('rbe_instance'),
               '--reproxy_cfg_template',
               'reproxy.cfg.template',
               '--rewrapper_cfg_project',
               Var('rewrapper_cfg_project'),
               '--quiet',
               ],
  },
  {
    'name': 'configure_reclient_cfgs',
    'pattern': '.',
    'condition': 'not download_remoteexec_cfg and not build_with_chromium',
    'action': ['python3',
               'buildtools/reclient_cfgs/configure_reclient_cfgs.py',
               '--rbe_instance',
               Var('rbe_instance'),
               '--reproxy_cfg_template',
               'reproxy.cfg.template',
               '--rewrapper_cfg_project',
               Var('rewrapper_cfg_project'),
               '--skip_remoteexec_cfg_fetch',
               ],
  },
  # Configure Siso for developer builds.
  {
    'name': 'configure_siso',
    'pattern': '.',
    'condition': 'not build_with_chromium',
    'action': ['python3',
               'build/config/siso/configure_siso.py',
               '--rbe_instance',
               Var('rbe_instance'),
               ],
  },
]

recursedeps = [
  'build',
  'buildtools',
]
