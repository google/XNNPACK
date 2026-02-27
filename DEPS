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
  'build_with_chromium',
  'generate_location_tags',
]


# Expect everything to be up-to-date with submodules.
git_dependencies = 'SYNC'

vars = {
  'chromium_url': 'https://chromium.googlesource.com',
  'generate_location_tags': False,
  # GN CIPD package version.
  'gn_version': 'git_revision:103f8b437f5e791e0aef9d5c372521a5d675fabb',
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
  'siso_version': 'git_revision:03ee208f9c31a303e1ba61f9bc7219158078bd50',
  # RBE project to download rewrapper config files for. Only needed if
  # different from the project used in 'rbe_instance'
  'rewrapper_cfg_project': Str(''),
  # Fetch configuration files required for the 'use_remoteexec' gn arg
  'download_remoteexec_cfg': False,
  # reclient CIPD package version
  'reclient_version': 're_client_version:0.185.0.db415f21-gomaip',
}

deps = {
  'buildtools':
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '6a18683f555b4ac8b05ac8395c29c84483ac9588',
  'build':
    Var('chromium_url') + '/chromium/src/build.git' + '@' + '483cecced32ce8b098d65eb08eb77925afa90bec',
  'testing':
    Var('chromium_url') + '/chromium/src/testing' + '@' + '4c3aba1a9edf0b61f30354e124a0cd99e33ecf6d',
  'third_party/libpfm4':
    Var('chromium_url') + '/chromium/src/third_party/libpfm4.git' + '@' + '25c29f04c9127e1ca09e6c1181f74850aa7f118b',
  'third_party/libpfm4/src':
    Var('chromium_url') + '/external/git.code.sf.net/p/perfmon2/libpfm4.git' + '@' + '964baf9d35d5f88d8422f96d8a82c672042e7064',
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
  'third_party/googletest/src':
    Var('chromium_url') + '/external/github.com/google/googletest.git' + '@' + '4fe3307fb2d9f86d19777c7eb0e4809e9694dde7',
  'third_party/google_benchmark': {
    'url': Var('chromium_url') + '/chromium/src/third_party/google_benchmark.git' + '@' + 'abeba5d5e6db5bdf85261045e148f1db3fdc40ad',
  },
  'third_party/google_benchmark/src': {
    'url': Var('chromium_url') + '/external/github.com/google/benchmark.git' + '@' + '7da00e8f6763d6e8c284d172c9cfcc5ae0ce9b7a',
  },
  'third_party/kleidiai/src': {
    'url': 'https://gitlab.arm.com/kleidi/kleidiai@v1.21.0',
    'condition': 'checkout_kleidiai'
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + '7ab65651aed6802d2599dcb7a73b1f82d5179d05',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '8f11bb1d4438d0239d0dfc1bd9456a9f31629dda',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + 'e81e859cfb7e78e70a58c3bfce859c509f45e1da',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + '9003ee6c137cea3b94161bd5c614fb43be523ee1',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',
  'third_party/cpuinfo/src':
    Var('chromium_url') + '/external/github.com/pytorch/cpuinfo.git' + '@' + 'f9a03241f8c3d4ed0c9728f5d70bff873d43d4e0',
  'third_party/libunwind/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libunwind.git' + '@' + 'ba19d93d6d4f467fba11ff20fe2fc7c056f79345',
  'third_party/android_toolchain/ndk': {
    'packages': [
      {
        'package': 'chromium/third_party/android_toolchain/android_toolchain',
        'version': 'KXOia11cm9lVdUdPlbGLu8sCz6Y4ey_HV2s8_8qeqhgC',
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
        'object_name': 'Linux_x64/clang-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '1c3c056427ab0db261c54c8fdf7c8404ff55e3de3e550520bcb1e1660ca05aad',
        'size_bytes': 57489092,
        'generation': 1768590901063677,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': 'e3f568bd57c7ab199eb384153eea8cbe3c0e0604b2d8bbb158985647709a9a9c',
        'size_bytes': 14391456,
        'generation': 1768590901188932,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '8762f3c6102eae568e3ca7a287e774514846f2bb2feda4cf7dc1c33d9f5f1c8d',
        'size_bytes': 14588900,
        'generation': 1768590901246745,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '4705feac592251ad0e2e1c41c855a6ecdc728908cbb459d4f68ca57f16bc4c5e',
        'size_bytes': 2321652,
        'generation': 1768590901407256,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '39d29ab3e708bcb6485181ac03123cfa3bac5b2365d5c441ab0cf5e7b25354b6',
        'size_bytes': 5802908,
        'generation': 1768590901316435,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '9aff2c8f9d941be0667dc3ad3d4c3591dccd70f7a3b8e80396a623364e752eeb',
        'size_bytes': 54613288,
        'generation': 1768590902935296,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '4f953afe4edebb54912719b437c78842978b0205792c069aa7529fd369d900be',
        'size_bytes': 1011040,
        'generation': 1768590912304306,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '62033f6ff5c1ea0a18ae581edfd1178f50ede19d84675eafb640d752e26b60ae',
        'size_bytes': 14444752,
        'generation': 1768590903234647,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '4a0365772b4eb7fe12fe595355e645f12c72255bf3f869941f2b4e5c5e2b76da',
        'size_bytes': 16398188,
        'generation': 1768590903283692,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '1c307bb206c5fc1e79deec1d12ce4ead53fa41574729deaa6d2b3a67f9540710',
        'size_bytes': 2352620,
        'generation': 1768590903522922,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '83767300f93707ba9a40d3d7e14f1149aaf587d5c1e3ae243be8742a008e5052',
        'size_bytes': 5682364,
        'generation': 1768590903341073,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '5475c6c38a6199276ff173665a27ab608aadb3118ac34f1a75391ee6dc226798',
        'size_bytes': 45585568,
        'generation': 1768590913838191,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '9893644917db71d520832ea9e276a98e2051acf53458efc6834973e076c6d36e',
        'size_bytes': 12429560,
        'generation': 1768590913986547,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': 'e67cfb9503c442bd29642b3a42b36c283346371b8143c81b895684df0fb09e69',
        'size_bytes': 12817084,
        'generation': 1768590914107805,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '5f17c695f24eadcc51edc720a5b0b2f2cc36413552038d93ae2bf361667d780a',
        'size_bytes': 1978756,
        'generation': 1768590914351005,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '5fd7fc88f00197a710aac7dd00a2b6947d5231c53a472ed38e1af0b9088cefc3',
        'size_bytes': 5418172,
        'generation': 1768590914206466,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': 'de6622ec6d9d22b00316c47b2eb59e6cb7dbcb2d5b59f04f18c94714d0b35066',
        'size_bytes': 48839256,
        'generation': 1768590925760667,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': 'bf6f38e7f5d06c8c3eff9ba85df1ddd1006828cc72b638d22a3e7562507a8a51',
        'size_bytes': 14353272,
        'generation': 1768590926006423,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': 'ad0dc3c686f63f40a35bd7a10f90935d08bcd9d1f23549c87cbdeb8cb503250c',
        'size_bytes': 2540656,
        'generation': 1768590935148812,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': 'bcf731c6ce9050067212fc4164893ce768809d73732459f2c1a0ddb8f124f5f2',
        'size_bytes': 14736740,
        'generation': 1768590926034327,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': 'bf86ab13e378a70f953d014cc7a37714dbaca0d8002cd638dd4d88df08231910',
        'size_bytes': 2416448,
        'generation': 1768590926285418,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-23-init-484-gf646b915-1.tar.xz',
        'sha256sum': '3ee3ece3cf0afa3536c39b15f818e5eab1ac4408cd31ad0ad82414d3a8aa1eca',
        'size_bytes': 5796552,
        'generation': 1768590926109316,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + 'd651bc848c45c945ecbc0c1a372b0b781e47c991',
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