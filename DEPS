# This file is used to manage the dependencies of the XNNPACK repo
# for use with the GN build system. It is losely based on what they do for V8:
# https://chromium.googlesource.com/v8/v8.git/+/refs/heads/main/DEPS
# Dependencies are static for now, automation for updating these will be added
# in the future. This is a fairly minimal config for getting GN builds working.
# Support for DEPS and GN is experimental and may be removed in the future.

# No-op for testing.

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
  'gn_version': 'git_revision:1a310e88443018837759c952b113846b0096f65b',
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
  'siso_version': 'git_revision:8aacae89cf77656164b64fd9d24b3edb884b88ac',
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
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '136da69a1267b8db487354b96d44d0cc8add5aeb',
  'build':
    Var('chromium_url') + '/chromium/src/build.git' + '@' + 'a3b822ce095162ebe93e1da32848bc0faea3e531',
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
    'url': 'https://gitlab.arm.com/kleidi/kleidiai@v1.22.0',
    'condition': 'checkout_kleidiai'
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + '7ab65651aed6802d2599dcb7a73b1f82d5179d05',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '8f11bb1d4438d0239d0dfc1bd9456a9f31629dda',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + 'c365236fa3dbdaff431a582f4eb04f6772d1e3da',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + '9003ee6c137cea3b94161bd5c614fb43be523ee1',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',
  'third_party/cpuinfo/src':
    Var('chromium_url') + '/external/github.com/pytorch/cpuinfo.git' + '@' + '7364b490b5f78d58efe23ea76e74210fd6c3c76f',
  'third_party/libunwind/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libunwind.git' + '@' + '17ccf7d110c5526cb77e93cfd8330f491fb2bf18',
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
        'object_name': 'Linux_x64/clang-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '6d4796631177f3c070923242601a4e14425ffdc2ee84c95ce630c0198d755637',
        'size_bytes': 57997052,
        'generation': 1771832273904291,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '250f932ba1195eaa1ab8fd6e50f3c650ff3b74d28127d982ee79b96c4844bfb7',
        'size_bytes': 14392496,
        'generation': 1771832274267259,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '39258f20af9d459b8b032ee6bd38a79ae764fd950b79bace3227b8965d58d189',
        'size_bytes': 14618644,
        'generation': 1771832274470726,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'cf2fd97359cc982f56b0beb7e9abff5a4799a12ff8c3e22728ddf06e1053b867',
        'size_bytes': 2330268,
        'generation': 1771832275108482,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '720a9cbb8745d4ac4c3f887bc1f004fed44cfccb851adebaa765a1517531c952',
        'size_bytes': 5786008,
        'generation': 1771832274626197,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '189a5bca90f83e2dc9671fef5f149386452c215a532e94cdc98cc13461411fb2',
        'size_bytes': 54754892,
        'generation': 1771832277222794,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '3a0177f1cf5f88f00e86006cb9be42f5d86a8dc461d03daf2919dee2c632f619',
        'size_bytes': 1012728,
        'generation': 1771832301108033,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'c8a3f088c4d8bbb92eba2036bb68d1640c9c54ef7ec4686b4e2a5c49ff9b6bc3',
        'size_bytes': 14282996,
        'generation': 1771832277199632,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '4746a11e14fe96a00024a7b7d7b4ee1d61c8960171578efad263d1c54cfa6901',
        'size_bytes': 15449448,
        'generation': 1771832277304839,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'fbaa78a2d70a0872c2e0fd94cbe9f13c61b3077c6ba73f76a34413510a420ed0',
        'size_bytes': 2373980,
        'generation': 1771832277813116,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'f091b4127cd59792c2f278535b1a30ae2e0953c6e896f31a5b5c7beff707e388',
        'size_bytes': 5693600,
        'generation': 1771832277787419,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '4a091561d8ee6cfff59efffecbfc0122d233d23427e1cdbcee5080253bfc81f9',
        'size_bytes': 45966912,
        'generation': 1771832302845435,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '9248646928a75768d682b484e3a83f481152b8edce5f71fa6e8052d9ae210398',
        'size_bytes': 12473436,
        'generation': 1771832303183995,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'e5fde137e7372fedc8e2c4a8e4f9fe718c3bcc63c76acaa0c4a859b04e3f4762',
        'size_bytes': 12882364,
        'generation': 1771832303644261,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'd4e9d19346ef021be42a27b367efcdcfee03cc19bbceddb9ea4f92619ef8b06c',
        'size_bytes': 1993468,
        'generation': 1771832304157004,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'b01bce55530dbca2d6d25fa14ddbac8259184fdca1cc63365130953b9ebb517f',
        'size_bytes': 5442884,
        'generation': 1771832303592136,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '84c13e593b1532c0cf0b8fe0928482e6eea96408846a679ce68e53c90911c33e',
        'size_bytes': 49494200,
        'generation': 1771832332047169,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '2a5072bfa4a3eb9dd60321d955d6d4744b7b9f2f657f32dbc706774282787e9a',
        'size_bytes': 14455216,
        'generation': 1771832332462256,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'a9ddbb29b61eaac9f777b1fb532eebb49ab9e546bcd680129fcc7a2fd648cd27',
        'size_bytes': 2596776,
        'generation': 1771832355675158,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '5dad22f5da22e268e43476a28689ec7b9fb8d41ae55e53911a26e9c25a9c4c4b',
        'size_bytes': 14884728,
        'generation': 1771832332829491,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': 'f5a46dbdf979e31d35dd12db37d714bc06a36957883f3c11d727d56382465079',
        'size_bytes': 2478920,
        'generation': 1771832333093363,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-23-init-4965-g686acf63-1.tar.xz',
        'sha256sum': '0519107fdef9c730bb4f073c87dbd03995fd2da074aabef86347f62fb2b784f3',
        'size_bytes': 5838628,
        'generation': 1771832332695387,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + 'fa73233792740e161a6c0cf1e2155a39f72fb948',
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