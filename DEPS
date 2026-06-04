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
  'gn_version': 'git_revision:6f8c0328ee29c76e3566a216f2f0cf2992daa6ed',
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
  'siso_version': 'git_revision:b18cb0f263cfcc2f17a925cb211972a32dc211f6',
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
    Var('chromium_url') + '/chromium/src/build.git' + '@' + 'f7c5798e4f219afaf2f9d88338e78f6c605858c0',
  'buildtools':
    Var('chromium_url') + '/chromium/src/buildtools.git' + '@' + 'e06be5db47ae15871b2c40c44d75a9fb91daf194',
  'third_party/depot_tools':
    Var('chromium_url') + '/chromium/tools/depot_tools.git' + '@' + 'ea198736a0bcaacc34b73b6218dd4de6e5d92fd3',
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
  'third_party/libpfm4':
    Var('chromium_url') + '/chromium/src/third_party/libpfm4.git' + '@' + 'c9516df91c2b4361d5e9f31ca14d23ad0bcbaabc',
  'third_party/libpfm4/src':
    Var('chromium_url') + '/external/git.code.sf.net/p/perfmon2/libpfm4.git' + '@' + 'fbcd0c258b79b28c3c9282f309b6465d78f47003',
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
    Var('chromium_url') + '/external/github.com/llvm/llvm-project/libc.git' + '@' + 'f720d21a0bd670e643a7919a6ba57e1e4fefca8b',
  'third_party/pthreadpool/src': Var('chromium_url') + '/external/github.com/google/pthreadpool.git' + '@' + '02460584c6092e527c8b89f7df4de143d70e801f',
  'third_party/fxdiv/src':
    Var('chromium_url') + '/external/github.com/Maratyszcza/FXdiv.git' + '@' + '63058eff77e11aa15bf531df5dd34395ec3017c8',
  'third_party/cpuinfo/src':
    Var('chromium_url') + '/external/github.com/pytorch/cpuinfo.git' + '@' + 'ea6b9f1bb6e1001d8b21574d5bc78ddef62e499d',
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
        'object_name': 'Linux_x64/clang-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': 'de584381536aa5ba2403033c4f8b70f3c39c2e5d7fa87c953b7fd8bfbba0ee2a',
        'size_bytes': 69635068,
        'generation': 1778518470595878,
        'condition': 'host_os == "linux"',
      },
      {
        'object_name': 'Linux_x64/clang-tidy-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '958f1354466c77c9be861aa7a6c261e35e03605ed6a0ebf577cacb7300ae73fe',
        'size_bytes': 14614028,
        'generation': 1778518470994342,
        'condition': 'host_os == "linux" and checkout_clang_tidy',
      },
      {
        'object_name': 'Linux_x64/clangd-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '9b2fc5ff65260a576ff6e35a152c84235b8b44999dea205b1acece8d41bdb11b',
        'size_bytes': 14791720,
        'generation': 1778518471166440,
        'condition': 'host_os == "linux" and checkout_clangd',
      },
      {
        'object_name': 'Linux_x64/llvm-code-coverage-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '8d0234c75afbbd5b94ce757fa010bddcc24327a7e6f5a954d4c88da7944c8e9e',
        'size_bytes': 2341388,
        'generation': 1778518471769630,
        'condition': 'host_os == "linux" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Linux_x64/llvmobjdump-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': 'a333dd084c2af4d5043bdb961ea81bdd11a200634732b24713a37ee2cb750a5d',
        'size_bytes': 5806280,
        'generation': 1778518472909665,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "linux"',
      },
      {
        'object_name': 'Mac/clang-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '11189402d8cf0209fa02fe5efb79e9f6bd363e4871faaeee37ea21d42f8c5c65',
        'size_bytes': 55337116,
        'generation': 1778518475287661,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac/clang-mac-runtime-library-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '1a5a032f901e079f639f4bf22ff5bee82025de5bf7f89bee6a1ad75ab2e54841',
        'size_bytes': 1017168,
        'generation': 1778518499789886,
        'condition': 'checkout_mac and not host_os == "mac"',
      },
      {
        'object_name': 'Mac/clang-tidy-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': 'fd2b4e5c13c8d4a0fd41341da2579072d3b5ec5aa7c4a7083f2a4246a17a89bc',
        'size_bytes': 14552028,
        'generation': 1778518476356552,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac/clangd-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '143638b0d1f41e7eac1f0a76066af63a63e8616ba3b1c79ef9e05873899f3602',
        'size_bytes': 15994340,
        'generation': 1778518476785587,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clangd',
      },
      {
        'object_name': 'Mac/llvm-code-coverage-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '93f2776d55ae6b3ccd80a7dbf84aeeb4b3f9985fd7ddf718f82b9aa851f12db9',
        'size_bytes': 2379612,
        'generation': 1778518477408748,
        'condition': 'host_os == "mac" and host_cpu == "x64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac/llvmobjdump-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '9f1df7279b8f27ef92c5ba6dd79c9b1146a0778a549dead7dca0a52c4d2fdf0d',
        'size_bytes': 5718128,
        'generation': 1778518476423529,
        'condition': 'host_os == "mac" and host_cpu == "x64"',
      },
      {
        'object_name': 'Mac_arm64/clang-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '94aac4f9e8a68559b579c76b863a5e0df6e80ba2900d2b7b5a6f1475a08a390c',
        'size_bytes': 46116112,
        'generation': 1778518502243133,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Mac_arm64/clang-tidy-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '25800d94785ff6eac01963a869215de1a1db072aee0add3ce3f6b1b47045f123',
        'size_bytes': 12622332,
        'generation': 1778518502560336,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_tidy',
      },
      {
        'object_name': 'Mac_arm64/clangd-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '69fb949b69ddcf968c5a71d0ad4c445cc42da16aa51f9801f768590fe368cb82',
        'size_bytes': 12967480,
        'generation': 1778518503028892,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clangd',
      },
      {
        'object_name': 'Mac_arm64/llvm-code-coverage-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '5d473bf300e95de41902ea85eb29ad7ada626187f89dcdb0b4897650a26712a8',
        'size_bytes': 1992380,
        'generation': 1778518503744736,
        'condition': 'host_os == "mac" and host_cpu == "arm64" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Mac_arm64/llvmobjdump-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': 'c658c070370514fab55d952e085b25fb4a1e573f59ca385b7097ab2b089a039a',
        'size_bytes': 5445304,
        'generation': 1778518502910688,
        'condition': 'host_os == "mac" and host_cpu == "arm64"',
      },
      {
        'object_name': 'Win/clang-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '855e4a23cfd89e0fa9a6899ccb343a7c28fd3d26f045a89cc40cc4c29cb6a85c',
        'size_bytes': 50194108,
        'generation': 1778518531637397,
        'condition': 'host_os == "win"',
      },
      {
        'object_name': 'Win/clang-tidy-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '6a3434b0a94ad1c20de247c2e5abf793e5653cea7d8c87a53b8c7959ec20f3c3',
        'size_bytes': 14705072,
        'generation': 1778518531980509,
        'condition': 'host_os == "win" and checkout_clang_tidy',
      },
      {
        'object_name': 'Win/clang-win-runtime-library-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '57d080966be03f17777a96d9c6decb03ac585d3233242743587bb601b46919d0',
        'size_bytes': 2610572,
        'generation': 1778518555380445,
        'condition': 'checkout_win and not host_os == "win"',
      },
      {
        'object_name': 'Win/clangd-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '48200b83edf568d3afbd72daf850eeff87f18e2350ace48acad6838eeebf0a16',
        'size_bytes': 15090972,
        'generation': 1778518532216019,
       'condition': 'host_os == "win" and checkout_clangd',
      },
      {
        'object_name': 'Win/llvm-code-coverage-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '81004ab7e9b141a5d0ca0240e8c52de21498e61867f05ebc9fda64220dbb18ff',
        'size_bytes': 2501464,
        'generation': 1778518532770750,
        'condition': 'host_os == "win" and checkout_clang_coverage_tools',
      },
      {
        'object_name': 'Win/llvmobjdump-llvmorg-23-init-10931-g20b6ec66-11.tar.xz',
        'sha256sum': '47e9d546f4c72a1c3327f4791663f3afe95c8f9a854a1081ea6c7fbd59b79e3b',
        'size_bytes': 5876436,
        'generation': 1778518532715015,
        'condition': '(checkout_linux or checkout_mac or checkout_android) and host_os == "win"',
      },
    ],
  },
  'tools/clang':
    Var('chromium_url') + '/chromium/src/tools/clang.git' + '@' + '93e6db4b344965912507450e0abac81551a2372f',
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
    # Ensure that the DEPS'd "depot_tools" has its self-update capability
    # disabled.
    'name': 'disable_depot_tools_selfupdate',
    'pattern': '.',
    'action': [
        'python3',
        'third_party/depot_tools/update_depot_tools_toggle.py',
        '--disable',
    ],
  },
  {
    # Update the Mac toolchain if necessary.
    'name': 'mac_toolchain',
    'pattern': '.',
    'condition': 'checkout_mac',
    'action': ['python3', 'build/mac_toolchain.py'],
  },
  {
    # Update LASTCHANGE.
    'name': 'lastchange',
    'pattern': '.',
    'action': ['python3', 'build/util/lastchange.py',
               '-o', 'build/util/LASTCHANGE'],
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