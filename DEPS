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
  'gn_version': 'git_revision:c5a0003bcc2ac3f8d128aaffd700def6068e9a76',
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
  'rbe_instance': Str('projects/rbe-chrome-untrusted/instances/default_instance'),
  'siso_version': 'git_revision:03ee208f9c31a303e1ba61f9bc7219158078bd50',
  # RBE project to download rewrapper config files for. Only needed if
  # different from the project used in 'rbe_instance'
  'rewrapper_cfg_project': Str(''),
  # Fetch configuration files required for the 'use_remoteexec' gn arg
  'download_remoteexec_cfg': False,
}

deps = {
  'buildtools':
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + '628cf12465dae2a157524a23608a58b525d30623',
  'build':
    Var('chromium_url') + '/chromium/src/build.git' + '@' + '2badee6987646d48d95e9db90caf8a91cc2719bd',
  'testing':
    Var('chromium_url') + '/chromium/src/testing' + '@' + '4c3aba1a9edf0b61f30354e124a0cd99e33ecf6d',
  'third_party/catapult':
    Var('chromium_url') + '/catapult.git' + '@' + '1c28f8a288ca1883a3555eaf29a9fc718afaad6e',
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
  'third_party/google_benchmark_chrome': {
    'url': Var('chromium_url') + '/chromium/src/third_party/google_benchmark.git' + '@' + 'fa1929c5500ccfc01852ba50ff9258303e93601e',
  },
  'third_party/google_benchmark_chrome/src': {
    'url': Var('chromium_url') + '/external/github.com/google/benchmark.git' + '@' + '761305ec3b33abf30e08d50eb829e19a802581cc',
  },
  'third_party/libc++/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxx.git' + '@' + 'ddfdbbc1ab109b4fc6171f3d8c38faf4586701d2',
  'third_party/libc++abi/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libcxxabi.git' + '@' + '9df28d1e6c6e831ad34a0cd354f667e3d54fc3a1',
  'third_party/llvm-libc/src':
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + '1f7cf83fb28c5bd777f4cdceed29bd52c69552b0',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + 'd90cd6f1493e09d12c407243f7f331a8cda55efb',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',

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
        'object_name': 'Linux_x64/clang-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'ffec41dcf83423532ea5cf11662b212d167b7a872b58bc4f731507d9b8fd1b7a',
        'size_bytes': 56109980,
        'generation': 1761337932156611,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'b7e9a0644956ea65a327b337f78d093fdf8ef10394cffd47e2777d0fc3eb8d97',
        'size_bytes': 14230956,
        'generation': 1761337932193057,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'a03330b69e92764a8861cb5cc0a3ad6a2514fe69f1474797f6ca46456e8a8cb6',
        'size_bytes': 14426688,
        'generation': 1761337932232398,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '327248b52b5160ddf440e0355d7abd5817d5144fc49a69a234bc7dd7174ae4b8',
        'size_bytes': 2292772,
        'generation': 1761337932297003,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '875b27a1b78f3b5ee97894cf3efa6faa9be5de646075f9fe1cceb8903405ffe5',
        'size_bytes': 5703936,
        'generation': 1761337932232851,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '3644769a80f6fbd182643a6cb9554e7270b5b571f62d546130f8786c5d47b581',
        'size_bytes': 53974704,
        'generation': 1761337933985979,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '5843537eae828493dbf3cea4f5ed0a014329ef3d48c0b65825848f377a7f7e93',
        'size_bytes': 1010116,
        'generation': 1761337941802663,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'a512248b32bf96356264d0004c483c31f9db0649e53ca2cf159db6283e97ec8b',
        'size_bytes': 14292684,
        'generation': 1761337933988162,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '3caffede54b9f16cf846b06caab23bd6fd71043ed68bb2b866bf2f37cc60c071',
        'size_bytes': 15805316,
        'generation': 1761337934002952,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '03da082317c7d2fa540b83f9d14f29ef0f42a7671c26e416a8d74c9fd37f4f43',
        'size_bytes': 2335124,
        'generation': 1761337934119552,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '37284075951953ebe8b273856b87148b596a7eb75ae619669c7cec0eb2fa72bd',
        'size_bytes': 5599068,
        'generation': 1761337933999423,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '9ccaa556dd7b4478fa9a90ed82b8c78dd2cc337e055e13654ee26d517760cdce',
        'size_bytes': 45058144,
        'generation': 1761337943479305,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'b9c37a6417648f9d4ccf8fbe4afb5448ef708c8e61291204d3ce4654c8427c15',
        'size_bytes': 12282788,
        'generation': 1761337943478346,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'dd0afd0386ea01f328db0d04841e53ad0740b691cd0744226223537de7fae38a',
        'size_bytes': 12677444,
        'generation': 1761337943476703,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'd426d5ac2159f94e8e3e8edb923cc465bd7969ad6df0a337aa923adcf52a1be3',
        'size_bytes': 1967576,
        'generation': 1761337943522567,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'a0ef0f87d8796dbf5d0ecec65b688a9afa8054c9a7bb620e32179555de2ded67',
        'size_bytes': 5347212,
        'generation': 1761337943489457,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'a3b528676d56cfb2c3df9b771ce75e2edc73346905416c81576710d20b6ae02c',
        'size_bytes': 48228992,
        'generation': 1761337953371073,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '5adfd4ec4b905f283b54d1f0f7969b21f3f82bc7d02dc430054cbe6ebb769387',
        'size_bytes': 14192380,
        'generation': 1761337953407064,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '1fad7810f49dba86ff46334813f3bd7d26771cf72a9ad0d8785442a83c823f27',
        'size_bytes': 2517344,
        'generation': 1761337961187718,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': '7160a3896b95a0a2c20fe4a7d7a64aea65ff833fa296d73b28cd238ae71e16b6',
        'size_bytes': 14604224,
        'generation': 1761337953407009,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'a873ab24e0b65523965ce4e27d74ba3eee447f262e72ba690d1eb98a76959dc6',
        'size_bytes': 2378244,
        'generation': 1761337953490821,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-22-init-12326-g8a5f1533-1.tar.xz',
        'sha256sum': 'c9ab59eb04490e9df3a477d354516c2ee16ca9fa456698b5ca99f689325a34ef',
        'size_bytes': 5706036,
        'generation': 1761337953424721,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + 'c32a3112f46745b6b0ec81b933bb3bd6303c7af0',
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