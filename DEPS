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
  'gn_version': 'git_revision:0e41dcd9b564dc1b48a4bc10e4a88eeb8724eea7',
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
  'siso_version': 'git_revision:3025be9c477069426f08b8235e9545d5b20afd0f',
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
    Var('chromium_url') + '/chromium/src/build.git' + '@' + 'bc2d66c42fa3ea6cdb5519072223a9c7f2f487e5',
  'buildtools':
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '245864014d8a0ce991bdc6c243dc4121cbdce3da',
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
    Var('chromium_url') + '/chromium/src/testing' + '@' + '49b0d934ea39933abf84d22b63a2503c5319da7c',
  'third_party/libpfm4':
    Var('chromium_url') + '/chromium/src/third_party/libpfm4.git' + '@' + 'e02968592ac424c7f45441249e7175d4549a5e3a',
  'third_party/libpfm4/src':
    Var('chromium_url') + '/external/git.code.sf.net/p/perfmon2/libpfm4.git' + '@' + '186a658c3e1e49b0ee3da414a7db8ba971a00cb5',
  'third_party/googletest/src':
    Var('chromium_url') + '/external/github.com/google/googletest.git' + '@' + 'a46a6012a133c7c0929fdae55b321c2fe65f3c5f',
  'third_party/google_benchmark': {
    'url': Var('chromium_url') + '/chromium/src/third_party/google_benchmark.git' + '@' + '57f4041d982b6900be1f525cfe909212a14d95c6',
  },
  'third_party/google_benchmark/src': {
    'url': Var('chromium_url') + '/external/github.com/google/benchmark.git' + '@' + 'a6b4a9f00b36c19af9997d68a1cf6e0b5b3d032c',
  },
  'third_party/abseil-cpp': {
    'url': Var('chromium_url') + '/chromium/src/third_party/abseil-cpp' + '@' + '0ae72b8d8f558a4e711d087951c107033ddfcc45',
  },
  'third_party/kleidiai/src': {
    'url': 'https://gitlab.arm.com/kleidi/kleidiai@v1.30.0',
    'condition': 'checkout_kleidiai'
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + 'ce8bfd89223f803e00f49de899e2b1e59db6e315',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '247ab3a16909851298b72c250c6b2ede98e7b00c',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + '9f914b40ed6ee85206fc628493e23d1284ebedf5',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + '15a6644ba1c45f1acc16ac1e883efc3e56c6bed2',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',
  'third_party/cpuinfo/src':
    Var('chromium_url') + '/external/github.com/pytorch/cpuinfo.git' + '@' + 'fd8b912165dfffccb4f652610a0a19afc51d98ee',
  'third_party/libunwind/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libunwind.git' + '@' + '9e44ea2ff4f53a60c2a3d2b6f511d7bab9ab12f0',
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
        'object_name': 'Linux_x64/clang-android-runtime-library-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '12d723081e648d3969ec52694b4399f9427f0da3c0f9c51cf3be4a4a1c7b1434',
        'size_bytes': 6201540,
        'generation': 1795640737947651,
        'condition': 'checkout_android',
      },
      {
        'object_name': 'Linux_x64/clang-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '65708ffb274d0666fbbdca62d16ed17e3e089cd617557807e03f2b31cdb14fa0',
        'size_bytes': 59684908,
        'generation': 1795640735326104,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '57ef751b286a05594285135168478d4d84f945b7781d65adacdff7e7fc1a8ff1',
        'size_bytes': 15400252,
        'generation': 1795640735422252,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'c6c29d6055e0417625339498a5ed1bb5fe52c98d43a8f02c95e988cf19da5d74',
        'size_bytes': 15762444,
        'generation': 1795640735437382,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'db39927fd7ad38f0896a52ee32ed73616b25f54dc5e8f66688ed256eca06fe60',
        'size_bytes': 2451160,
        'generation': 1795640735530279,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '8cb0e1f3445ee4417a75c041249e0c3bced53b0dccc3ff9b48ddcccd358f6ad3',
        'size_bytes': 6054508,
        'generation': 1795640735473868,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'ecbf823b86e196df99b72fc5e91cb97b0a35d458848ae12ff1457b3d1d8dae52',
        'size_bytes': 55629124,
        'generation': 1795640743919326,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '35bfb7ac229724e2ef5e22a6a9517da11bfe0454396b75846651c54bf7dbc744',
        'size_bytes': 1284544,
        'generation': 1795640751266252,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'a73724b73caf67947e7a5d1dfc9303ad96780e98612a5ce1b43e7c2453b60bf0',
        'size_bytes': 14547176,
        'generation': 1795640744039988,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'efdeb9bae50729fc6652334f605f69eb0d34a53520771d47d43b92f391345157',
        'size_bytes': 15637236,
        'generation': 1795640744085979,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '444a6cc1e45eec5fbbd9302906b8621d5f2d23a810cc178f93d89b776ac44428',
        'size_bytes': 2324768,
        'generation': 1795640744381467,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '8f6b9b4dd4c0feee0a01e0410e9999a351d31a6b15b090e329c8029dba921ef9',
        'size_bytes': 5626392,
        'generation': 1795640744171324,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '772d4f10a254cad33ede3f1e4443f3b7f7acc6cb942c2f914dcd4781342319a4',
        'size_bytes': 45652660,
        'generation': 1795640752551134,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'b32ccd659ea49902ffcdf96ee57698e7f52e6c345323fd6dd371ff9a46c1d68a',
        'size_bytes': 12414752,
        'generation': 1795640752736186,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '8be17c6c3aa6e447999be849f30353fbb8f12a0800842601671038ac3f686dfb',
        'size_bytes': 12420712,
        'generation': 1795640752946650,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '23e511965d1cb0763128169c421e72f8a3164a85ceb81f0a0f146b499e5cdc43',
        'size_bytes': 1910320,
        'generation': 1795640753077543,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'ad8cb0a56204fce45efb94755fbb83583974dacbcb2fa0c4a8aa055facecb6d6',
        'size_bytes': 5318704,
        'generation': 1795640752935417,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '2f0aaa8ed11fbba395e5fc2b9a235a92821012047991a107ca85e82f2eea5a81',
        'size_bytes': 53491412,
        'generation': 1795640761681803,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'fe50bafb7140505469d75a0c416210a69a6a81313740bc7d93b34a26b4269f85',
        'size_bytes': 15628240,
        'generation': 1795640761856433,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'f3bb9b9482b9db918fb492681ab7b170de186bc5565c439e56a0733d89e0302d',
        'size_bytes': 2722160,
        'generation': 1795640768914577,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '66566251012a604a3e539b5e1cc506e2a8931422fb686ea7b14d115df1dc7617',
        'size_bytes': 16104428,
        'generation': 1795640761900826,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': 'bd079fab99bf54c56b735b7bb366509ed3d4b9b8f23daaa1252fc8f0aacdf844',
        'size_bytes': 2620660,
        'generation': 1795640762057533,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-24-init-22088-g7ba4f04e-3.tar.xz',
        'sha256sum': '7733ee2a80e7ceaefb017602ff2f568d6a7e7a686aa0b12f3489cb3c7d1a2ec4',
        'size_bytes': 6164872,
        'generation': 1795640761889685,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + '6a879cd741b8ef93f5db20790d57686e8c9ffe1a',
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
