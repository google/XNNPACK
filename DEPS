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
  'android_ndk_version': Str('2@30.0.14608247'),
  'chromium_url': 'https://chromium.googlesource.com',
  'generate_location_tags': False,
  # GN CIPD package version.
  'gn_version': 'git_revision:7395cecc9cd7e73181ce5262704f3323f356aadd',
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
  'siso_version': 'git_revision:0b5eb879ba9260eb6fd7dfe8f9437ef3396185c4',
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
    Var('chromium_url') + '/chromium/src/build.git' + '@' + 'd0b93c1274189c0bde86863c392612029ae8362c',
  'buildtools':
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '0d39be5a3f129cf1f35e7812108a2184e2193315',
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
    Var('chromium_url') + '/chromium/src/testing' + '@' + '00f8bbf1370bd2ae8519ddba54f87f17a4d4dcc6',
  'third_party/libpfm4':
    Var('chromium_url') + '/chromium/src/third_party/libpfm4.git' + '@' + '4a112befd93f8d90a9b6894b2ec4d320310e1178',
  'third_party/libpfm4/src':
    Var('chromium_url') + '/external/git.code.sf.net/p/perfmon2/libpfm4.git' + '@' + '6870a9f00412830ceaa7e4384bb92ee323e2a28f',
  'third_party/googletest/src':
    Var('chromium_url') + '/external/github.com/google/googletest.git' + '@' + '4fe3307fb2d9f86d19777c7eb0e4809e9694dde7',
  'third_party/google_benchmark': {
    'url': Var('chromium_url') + '/chromium/src/third_party/google_benchmark.git' + '@' + 'c3b654389bf74fac1f2d926d0439506f06c66751',
  },
  'third_party/google_benchmark/src': {
    'url': Var('chromium_url') + '/external/github.com/google/benchmark.git' + '@' + '8abf1e701fbd88c8170f48fe0558247e2e5f8e7d',
  },
  'third_party/kleidiai/src': {
    'url': 'https://gitlab.arm.com/kleidi/kleidiai@v1.25.0',
    'condition': 'checkout_kleidiai'
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + '5abc7f839700f0f17338434e1c1c6a8c87c00c11',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '8f11bb1d4438d0239d0dfc1bd9456a9f31629dda',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + '1cc8752379a5625586ffe65cf53c0411b0087865',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + 'a56dcd79c699366e7ac6466792c3025883ff7704',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',
  'third_party/cpuinfo/src':
    Var('chromium_url') + '/external/github.com/pytorch/cpuinfo.git' + '@' + '3681f0ce1446167d01dfe125d6db96ba2ac31c3c',
  'third_party/libunwind/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libunwind.git' + '@' + 'd6c7a21e978f0adaa43accaad53bc64f0b64f6ec',
  'third_party/android_toolchain/ndk': {
    'packages': [
      {
        'package': 'chromium/third_party/android_toolchain/android_toolchain',
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
        'object_name': 'Linux_x64/clang-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': 'e22e06c05fe1657f48f988b15804b8226e691addb00abba5b984a5c99ac98c42',
        'size_bytes': 59093892,
        'generation': 1781627382446731,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '96da04c2fff4e580ac81b840405edfa292108d0b593927a10750c0a1d8599c0a',
        'size_bytes': 14808648,
        'generation': 1781627382496605,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '1e6b6918bb270659ef517a3e8f80af8f371bc79a9d942241978db8faea22f152',
        'size_bytes': 15001372,
        'generation': 1781627382490030,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': 'bb63e33c20329c344a9dfa958f3742b316c0b9dd602647190fb0037d8d53a7e6',
        'size_bytes': 2332356,
        'generation': 1781627382888550,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': 'e43b98bce71d58bbe1b456f7b62997c4d017cf8362a0367592428ac5a7512f41',
        'size_bytes': 5875704,
        'generation': 1781627382574869,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '84b3934db2fb3c2e657d4a783a83ca6d2facaf598991490ae8ab712fcb03224b',
        'size_bytes': 56083264,
        'generation': 1781627385112611,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '5079e2ac3f0fa76e8b0218bf99a65a5439e8dcf6a750f886a355a66df34c69c1',
        'size_bytes': 1010816,
        'generation': 1781627394919334,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '1ca3dc160992d8c27765adaffcc8115aed34cdf7645a5354820e8c8b16b75dcd',
        'size_bytes': 14790996,
        'generation': 1781627384977799,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '4920275e6050dffbb81ba0210a627a61a20faa896aa50f5ecc2d765522790526',
        'size_bytes': 16720488,
        'generation': 1781627385229783,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '07e8bf3354e0c086e420ab8f1f376e48c8b6207788cf30b56d1b0ad1fd4b0f12',
        'size_bytes': 2372720,
        'generation': 1781627385661382,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '9b286eecf04996dfa82cd1434bceb19612fa0754e4390649ba8dcd846dca9c1e',
        'size_bytes': 5797028,
        'generation': 1781627385345493,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': 'e96c5e31ddc2a6e841bcb0f7278ae82b3a581799904276e6f7673213c9748c27',
        'size_bytes': 47058068,
        'generation': 1781627397208650,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '27eff49b49d393c903246d5e17ba620f62f493a8a39cb6dbdd2e3f5f06c592a2',
        'size_bytes': 12835944,
        'generation': 1781627397575694,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '9519e51006f45a797a351b616d92bff319793ab976eeb59df60901d6181f3c18',
        'size_bytes': 13205948,
        'generation': 1781627397237653,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '62084a5523a1c3b140166c2ed5a21bc110ceb65503f2759272a9f1feb00b503e',
        'size_bytes': 2000216,
        'generation': 1781627397451117,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': 'b6c6c0b1b4147ba4d51603713d3ed21f88ee2f506ff07fb432b3444f1d323295',
        'size_bytes': 5548680,
        'generation': 1781627397409118,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '9e3894b94d0d5e3d5904a559f590f12bd53aa5d1d9b6a902de2acb957825de46',
        'size_bytes': 51306328,
        'generation': 1781627409675428,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '8fa52c3d1707e73d0689aaa3c8d3789f13e85526bef029d942baf44ed980b442',
        'size_bytes': 14867152,
        'generation': 1781627409620988,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': 'f925c30e1f63291662b2ad91103592208ecc4354f669c98a07e35360d2561a13',
        'size_bytes': 2623388,
        'generation': 1781627419134531,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '54042464918d992a19efed3a5f2a170e6fc2e4551884b4870650e3a3a6a03b41',
        'size_bytes': 15262064,
        'generation': 1781627409708803,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '7c4fbafcf2741a036d1f4058fc31b52cb5b0951e617561fc3d51b7fbe0a3044b',
        'size_bytes': 2496336,
        'generation': 1781627409916932,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-23-init-19482-g53d18800-1.tar.xz',
        'sha256sum': '7ea19d03f21ef59d2e511af8d9cccff20ea72deb54923557173bb46a64244fa7',
        'size_bytes': 5934284,
        'generation': 1781627409787893,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + 'dd7362d6df176da8001a8195597968939b444ab1',
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