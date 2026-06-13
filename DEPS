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
  'android_ndk_version': Str('2@30.0.14608247'),
  'chromium_url': 'https://chromium.googlesource.com',
  'generate_location_tags': False,
  # GN CIPD package version.
  'gn_version': 'git_revision:1740f5c25bcac5a650ee3d1c1ec22bfa25fcd756',
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
  'siso_version': 'git_revision:8629cde6728d25cde0f6fc1e6ae6b5886e0ebe41',
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
    Var('chromium_url') + '/chromium/src/build.git' + '@' + '7b34ad40857f9d30f69ea3953c3025b3473aebb1',
  'buildtools':
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '5e7f45b2c8d0acf21ee52a32bb6ec83d4620eb59',
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
    Var('chromium_url') + '/chromium/src/testing' + '@' + '4c3aba1a9edf0b61f30354e124a0cd99e33ecf6d',
  'third_party/libpfm4':
    Var('chromium_url') + '/chromium/src/third_party/libpfm4.git' + '@' + 'd35e15a834ad4e05ba2f0a3c1a2b76fde99a0570',
  'third_party/libpfm4/src':
    Var('chromium_url') + '/external/git.code.sf.net/p/perfmon2/libpfm4.git' + '@' + '977a25bb3dfe45f653a6cee71ffaae9a92fc3095',
  'third_party/googletest/src':
    Var('chromium_url') + '/external/github.com/google/googletest.git' + '@' + '4fe3307fb2d9f86d19777c7eb0e4809e9694dde7',
  'third_party/google_benchmark': {
    'url': Var('chromium_url') + '/chromium/src/third_party/google_benchmark.git' + '@' + 'abeba5d5e6db5bdf85261045e148f1db3fdc40ad',
  },
  'third_party/google_benchmark/src': {
    'url': Var('chromium_url') + '/external/github.com/google/benchmark.git' + '@' + '8abf1e701fbd88c8170f48fe0558247e2e5f8e7d',
  },
  'third_party/kleidiai/src': {
    'url': 'https://gitlab.arm.com/kleidi/kleidiai@v1.25.0',
    'condition': 'checkout_kleidiai'
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + 'be1c391acca009d8d80535ce924e3d285451cdfa',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '8f11bb1d4438d0239d0dfc1bd9456a9f31629dda',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + '01f1fb50a55d022e351f52fb71a642547eb5c080',
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
        'object_name': 'Linux_x64/clang-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '6aa45b7398e915919ce07a5e1e15e8710327957f61c1d97d18e77e5f167e9d14',
        'size_bytes': 69605340,
        'generation': 1777639339258203,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '0f6db77d3bd03bd39add2f13a19072236ae00feddc45dc51dfe976d58dd843de',
        'size_bytes': 14615020,
        'generation': 1777639339339127,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'dbbb792a7893c3477fe2a9e378b6a09ed93dbff41f707941b7c84939ba74649c',
        'size_bytes': 14786588,
        'generation': 1777639339349447,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '1e1f962a845d5246e6b541e25ef6bae50b381ec64243f3cc2aee0b8c7d718a8e',
        'size_bytes': 2336008,
        'generation': 1777639339569555,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'f3701a93e5ad4eae4d3aa0c853dd71a4e776d8f9cec5db3e0c800e1843d5a9a4',
        'size_bytes': 5793432,
        'generation': 1777639339446398,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'd25493bb0c74e7aad55135b0aceed9dcca22d6477043cb005aa3ccce7d708b83',
        'size_bytes': 55341104,
        'generation': 1777639341141424,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '2f1677e29b78c584450ce6f5d72273f9584ac43d51bc9ca91619b045e93c2c12',
        'size_bytes': 1016564,
        'generation': 1777639349729752,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'f8a6449263f115c9f69371cf415899d186f11790dce0b6022a7b3a583b0716f5',
        'size_bytes': 14567332,
        'generation': 1777639341376334,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'dbd23694bd418555f8a65871f4b862830db1045f785e003822c4124f6da99484',
        'size_bytes': 15995436,
        'generation': 1777639341422572,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'c24cbc8695bb7ed7f5c9709b6ff2925d57c635b27375c374f23860dec9d633ac',
        'size_bytes': 2379552,
        'generation': 1777639341530758,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '98ac084036e387b7da7729fdb4f0dde9724897b52642f2b1f20782968765c24f',
        'size_bytes': 5722328,
        'generation': 1777639341526564,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'f6fd465945dae58eff22819cdc62ccae2271dd3cf624a2fa68d87f634f32dcb7',
        'size_bytes': 46106304,
        'generation': 1777639351295488,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '26d400934fee65cd9af87d0d7c23c3ec086adac93459759d44ced9b7519694a2',
        'size_bytes': 12619892,
        'generation': 1777639351445994,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'cd4aed6e6e89a9616c76de4faee90f335dea9fc69b45b0d00fb9b9eddb9f81b4',
        'size_bytes': 12980784,
        'generation': 1777639351661710,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'b2ff8a6e42fa14d4bb58a03a64962be7230a47dc2f8eeb83ad1d46cddc4d70da',
        'size_bytes': 1992948,
        'generation': 1777639351899002,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'd1e42c8009c5983bbb37867bba8522db876bd32bb76c6e5560bf5134a02f7aaf',
        'size_bytes': 5444568,
        'generation': 1777639351710887,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '7fa1aa9bf477f565687a2f516bf75ff6bfdd13da9b36bf521a75ac8ae2761d6a',
        'size_bytes': 50211688,
        'generation': 1777639362668478,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '36f00ca3c9ae9bbf226dbe5625b42e06ddf9cfe81586e89c6cd96fa1ed99cc7b',
        'size_bytes': 14723216,
        'generation': 1777639362810257,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'f25fb199c927f86083f5b6d2641daac60aa81de9ea1204f38e9301c2889a0dec',
        'size_bytes': 2611460,
        'generation': 1777639371151452,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '0ef91424212a6b4cb6da4fe6a97ffca044b9469a246ae5a2d3d58dd8e02f2e98',
        'size_bytes': 15101300,
        'generation': 1777639362859425,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': 'ab1fe0acdb40306cadde73474437a98483a914f1bfad95d8de4d5d0c3dae2d18',
        'size_bytes': 2500728,
        'generation': 1777639362926842,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-23-init-10931-g20b6ec66-8.tar.xz',
        'sha256sum': '9f3bf71579786784762963c50cdc908ad07213a8b8c7f81edca209c095b130a9',
        'size_bytes': 5868192,
        'generation': 1777639362899299,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + '3e520fe28bbeed4a87f5e575272933d359915989',
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