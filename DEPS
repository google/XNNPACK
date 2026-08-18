# Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
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
  'android_ndk_version': Str('2@30.0.15729638'),
  'chromium_url': 'https://chromium.googlesource.com',
  'generate_location_tags': False,
  # GN CIPD package version.
  'gn_version': 'git_revision:d1996a79c64e0852e9a2559bf1596376eeebdabf',
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
  'siso_version': 'git_revision:1b1109fc6f5e177a439a195b87931224efc7a007',
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
    Var('chromium_url') + '/chromium/src/build.git' + '@' + '0d8b711d97d1ec64909af024363817e480640fb7',
  'buildtools':
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '9e7655f4ee433ef4c6efcffd57e379db8f8c0432',
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
    Var('chromium_url') + '/chromium/src/testing' + '@' + '2d2025cba7bafbf4db985c98c3bafaa26fc9fb3b',
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
  'third_party/abseil-cpp': {
    'url': Var('chromium_url') + '/chromium/src/third_party/abseil-cpp' + '@' + 'df548c50b2cda67158364d3d23c63043881b391d',
  },
  'third_party/kleidiai/src': {
    'url': 'https://gitlab.arm.com/kleidi/kleidiai@v1.30.0',
    'condition': 'checkout_kleidiai'
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + 'b16984ce99c702355a5b2b4c52574e82cec41fb9',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '2647155ea2bb555879a6f60d83d9e0137284b5a4',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + '6fd5620cc4fd3c55ee749e9bf71f52038431f76d',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + '02460584c6092e527c8b89f7df4de143d70e801f',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',
  'third_party/cpuinfo/src':
    Var('chromium_url') + '/external/github.com/pytorch/cpuinfo.git' + '@' + '66ee79c038d70dad9f08705b2c9b3e58f6d8f512',
  'third_party/libunwind/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libunwind.git' + '@' + '871ec683fbad86cadffff0a749d9befedda01248',
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
        'object_name': 'Linux_x64/clang-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '94940b8da1c3e7fec9020a75cfc4e852922a4d163dab8957e691a36ae5d49f0b',
        'size_bytes': 59100444,
        'generation': 1785366626567993,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '355a938fd3ab277dee7b2df50264416a0ebbdfd101e4b6beb82cc176f0ae300d',
        'size_bytes': 14809560,
        'generation': 1785366626548898,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '7eab6663df18976874c774eb2dee38f79287ac2dbdd336fcda2d955438fe131a',
        'size_bytes': 14988320,
        'generation': 1785366626796669,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '2cb5220dceb4be408c306537e83b323219ed0129cf3512fb4bbf49aa0c334b7f',
        'size_bytes': 2332356,
        'generation': 1785366626986503,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '61ac1326d1fd10fe3527f1ca0ce78aa357923157442056eaa53a027931db986e',
        'size_bytes': 5873792,
        'generation': 1785366626833938,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '51350c8f6da5c92ab19b0b17a3cbc7762b960b29fd68530e9feb52eb5079ae19',
        'size_bytes': 56088216,
        'generation': 1785366628543558,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '0105a7a2e8e6f4b88b6414756a5bfaa04e3cb1e678d29e3df367752a203edb6a',
        'size_bytes': 1010844,
        'generation': 1785366639043429,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '66d4a4b66e5ad9bee6b6235e9a8fee213afa4c321232e897881c8e4dcf752d61',
        'size_bytes': 14767916,
        'generation': 1785366628634961,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '9866482056c17d3f5b4c13a693c6e7e8417c54e943cf00e58a82a51b65dfd183',
        'size_bytes': 16714728,
        'generation': 1785366628706685,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': 'baec5b697375914d0ec1a377f66a6ac3826aad0103de2d6258b8a88cf7dd5c02',
        'size_bytes': 2372468,
        'generation': 1785366629000854,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': 'ab7600041b0db44338b42c55b7ed1224edc10c96283d71e53a373820be9e62f9',
        'size_bytes': 5790116,
        'generation': 1785366628828288,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': 'a9c5ab02bfdc07a6ad6c6711d884dccab1fdf52e05e71aa9328039cddaff3664',
        'size_bytes': 47045224,
        'generation': 1785366640567652,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '51eea196d67444cc33613791193acba52803e0b57cdcbecd8135b4578674145e',
        'size_bytes': 12837040,
        'generation': 1785366640711645,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '2e9c4180eaf40a5ffd4f26bfcb2c8b7172de1ad06bb81e98cfbec466cd5eabf9',
        'size_bytes': 13199252,
        'generation': 1785366640801569,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '9a2a764f9f2dbb46eba03ed29a87bf55b580a1f7e562ba4002056cb28e2c5902',
        'size_bytes': 1999328,
        'generation': 1785366641042033,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '39fad69229bd7f8b440809f74a711175f5f0ed3cc92c4d6f3652c0ca57e39c2e',
        'size_bytes': 5546692,
        'generation': 1785366640876543,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '0d42e6825a7b5cafb8b6d250e8aae3b9b1920f4cbc2fe4f5a8dd46154d95d25d',
        'size_bytes': 51315536,
        'generation': 1785366653133084,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': 'fbd7f3a0aa321c144542a3d97f702ae676306b837febfd7663a036929f8eb0d5',
        'size_bytes': 14855356,
        'generation': 1785366653263211,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': 'da5135a86ba5a585abc6dfb38c38803a14e9a83cd39ceaac85eedef50f193a1d',
        'size_bytes': 2622056,
        'generation': 1785366663332782,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '1e6d5c7e036b4740b7dabaca71e2e3e97a68477515ffc9faa10ba324781faa56',
        'size_bytes': 15260276,
        'generation': 1785366653344899,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': '6d21c3aebf0238eb31dc8fe41c01f079dde57a06bf628cedca7ef94f4e9cbdcb',
        'size_bytes': 2494100,
        'generation': 1785366653543218,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-23-init-19482-g53d18800-2.tar.xz',
        'sha256sum': 'ec5ac24513c7b650b27b638bc46c207b228072fd5cd7c45a86b988820c01ff95',
        'size_bytes': 5940460,
        'generation': 1785366653406139,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + 'e4816757c45a072f08279139108a0a937bbeb239',
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
