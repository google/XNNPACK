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
  'gn_version': 'git_revision:b2ac0e7a9089039e62b84d246eca83f84c540f76',
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
  'siso_version': 'git_revision:87bad442ede1c60700dfabef5862c4a584621734',
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
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '22e55595e15ebbbbb4bef118d5a654b185b0b30d',
  'build':
    Var('chromium_url') + '/chromium/src/build.git' + '@' + '07280efcb6346f2a90f8b85b0f23727e265869b7',
  'testing':
    Var('chromium_url') + '/chromium/src/testing' + '@' + '4c3aba1a9edf0b61f30354e124a0cd99e33ecf6d',
  'third_party/libpfm4':
    Var('chromium_url') + '/chromium/src/third_party/libpfm4.git' + '@' + '7b00b38c3a04258b69f720f34ac08ee5c16d2715',
  'third_party/libpfm4/src':
    Var('chromium_url') + '/external/git.code.sf.net/p/perfmon2/libpfm4.git' + '@' + '41878eab48c50bb9ec5f741a013e971bb5a9dff2',
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
    'url': 'https://gitlab.arm.com/kleidi/kleidiai@v1.23.0',
    'condition': 'checkout_kleidiai'
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + '7ab65651aed6802d2599dcb7a73b1f82d5179d05',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '8f11bb1d4438d0239d0dfc1bd9456a9f31629dda',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + '348df0ed1c3c8fe6c44dc0972accc250f448ac51',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + '9003ee6c137cea3b94161bd5c614fb43be523ee1',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',
  'third_party/cpuinfo/src':
    Var('chromium_url') + '/external/github.com/pytorch/cpuinfo.git' + '@' + '7364b490b5f78d58efe23ea76e74210fd6c3c76f',
  'third_party/libunwind/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libunwind.git' + '@' + '148e7893795b96b53f7b2ce23f9bb96c6e09a822',
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
        'object_name': 'Linux_x64/clang-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '750b331006635281d7d90696629f67db748ba62004c46675eccb8af144141847',
        'size_bytes': 58029996,
        'generation': 1772218390302503,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': 'd53439bb6ac13c8d2c30c20555ded434039802f70d4119c0138bd77d03552223',
        'size_bytes': 14392856,
        'generation': 1772218390323510,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': 'a24613fb7afce42c076bb95d1b671ac028746b379e88070c126f0aab17a4c34e',
        'size_bytes': 14635272,
        'generation': 1772218390330947,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '8dcd816a83361b7924093ccba92dfe6bd29af2cf8af58bf7ce785b38c5027a8b',
        'size_bytes': 2328908,
        'generation': 1772218390452408,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '0a15d6b8c2b774b0706618d2afa123b9c87af2ec12e74dc44346df4c4690b670',
        'size_bytes': 5780116,
        'generation': 1772218390340688,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '2661847eb275079358ab186eaf7f85d6139d44c7413a731dfac7f5ed1ec34a01',
        'size_bytes': 54827776,
        'generation': 1772218392155773,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '69918295c163ec5a20aede81d4100bbd41e01142d32e0555366bba05141f7bf2',
        'size_bytes': 1010608,
        'generation': 1772218399449599,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': 'b8013fe5d2410db4f365ec8779972415d1d0a08042a3a43f823a0da712108cff',
        'size_bytes': 14280488,
        'generation': 1772218392176137,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '508098b26e74bd7f5cdcc40a2ed2db24e2bdde15e0f1c14ce94f685f991b3dd6',
        'size_bytes': 15455912,
        'generation': 1772218392186146,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '46c33f13a68fc14005560c01a91215b5cab54c07e920a714264352e46af1350c',
        'size_bytes': 2376304,
        'generation': 1772218392292978,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '6a92e3f21b3a035f406313d24688bb1b312a9a0ec423ff808752b6638104aff3',
        'size_bytes': 5699700,
        'generation': 1772218392189830,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '909be0f896bcf140c710548ccda4673c0aea2480e28d10803c19b1689b36acd5',
        'size_bytes': 45847044,
        'generation': 1772218401088162,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '83dc8d90529730ae503e684ea0047a0baec2b0c4a81941d1bb4196feea6ba264',
        'size_bytes': 12444972,
        'generation': 1772218401143017,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '3b7ff06ccd41b0a1fb165e182a35bcd74ae49172f1720cd276eb5feac0e3dd9f',
        'size_bytes': 12816980,
        'generation': 1772218401144631,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '67148555d00427a3eaa8aeefb8c4c4e1271d585315bdbf0d28d20fd78957e309',
        'size_bytes': 1988008,
        'generation': 1772218401224240,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': 'a31075e7f46ed77c62ecec424722bec8335ef306a4701660f19b713229c49afa',
        'size_bytes': 5421552,
        'generation': 1772218401116635,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': 'f2c9d2a8accf7ed2e3c19b3f67fb94e60365411a536fb9d71391dd2d4e7e14bb',
        'size_bytes': 49546756,
        'generation': 1772218410442709,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '99e00bbb404557db32df4e7a183ac520c526fe0e143ca380dfb2d0c33a2025b5',
        'size_bytes': 14462056,
        'generation': 1772218410470169,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '62e9c022223e0fa6ff855c25dcee524818f04c570127ed7e74895b320a10100a',
        'size_bytes': 2597584,
        'generation': 1772218417651221,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '6a3ab3afb8d2e7f4a04eecd8073993586665ede3929308a0fa0119d9382b1e2d',
        'size_bytes': 14887416,
        'generation': 1772218410483998,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '4bd610d2fbcc6e2bd8fd2df8d8c23a915373f8c987701d295314e8b33d457075',
        'size_bytes': 2479300,
        'generation': 1772218410570017,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-23-init-5669-g8a0be0bc-1.tar.xz',
        'sha256sum': '2ee77b6240b76353840439b38e7009d9f1fb8e97930dbbef3b1ff805ee981c5f',
        'size_bytes': 5846184,
        'generation': 1772218410487302,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + 'd2e96d6073aff96eaf6128e4d8897af00193ccf8',
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