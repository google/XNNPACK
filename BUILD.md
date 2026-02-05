# Building XNNPACK

This is a quick-start guide for building XNNPACK in some common configurations.
XNNPACK currently supports these build systems:

* [Bazel](https://bazel.build/)
* [CMake](https://cmake.org/)

Some combinations of operating systems, compilers and CPU architectures are
tested regularly via GitHub Actions (see [build.yml](.github/workflows/build.yml)).

## Building with CMake, testing with CTest

### Build without cross-compilation, default compiler
`scripts/build-local.sh` automatically configures and runs CMake for the host
CPU and operating system.

```sh
# In the XNNPACK root directory
scripts/build-local.sh
# Build artifacts will be created in build/local
cd build/local
# Run the test suite
ctest --output-on-failure --parallel $(nproc)
```

#### Specifying an alternate compiler
Standard CMake arguments like `-DCMAKE_CXX_COMPILER` and
`-DCMAKE_C_COMPILER` are used like this:

```sh
scripts/build-local.sh -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_C_COMPILER=clang-19
cd build/local
ctest --output-on-failure --parallel $(nproc)
```

_Note_: `nproc` uses the number of CPU cores in your host system.

### Build for Android

* Building for Android requires setting the `ANDROID_NDK` environment variable.
* You can download the NDK [from here](https://developer.android.com/ndk/downloads).

```sh
# Set ANDROID_NDK to wherever the NDK is unpacked
export ANDROID_NDK=$HOME/bin/android-ndk-r27c
# Build for 32-bit Arm
./scripts/build-android-armv7.sh
# Push the compiled benchmarks onto an ADB-connected device
adb push ./build/android/armeabi-v7a/bench /data/local/tmp
```

### Combinations known to work
| Platform | Target     | Host CPU | Target CPU | Compiler           | Works  | Supported** | Notes  |
| -------- | -------    | -------- | ---------- |  ------------------|--------|-------------| -----  |
| Linux    | Linux      | x64      | x64        | Clang 19.1.7       | ✅     | ✅          | |
| Linux    | Linux      | x64      | x64        | Clang 18           | ✅     | ✅          | |
| Linux    | Linux      | x64      | x64        | GCC 14.2.0         | ✅     | ✅          | Various GCC versions are supported. |
| Linux    | Linux      | AArch64* | Armv7-A    | GCC                | ✅     | ✅          | |
| MacOS    | MacOS      | AArch64  | AArch64    | Clang              | ✅     | ✅          | |
| Linux    | Android    | x64      | Armv7-A    | (NDK r27c provided)| ✅     | ✅          | |
| Linux    | Android    | x64      | AArch64    | (NDK r27c provided)| ✅     | ✅          | |
| Linux    | Android    | x64      | x64        | (NDK r27c provided)| ✅     | ✅          | |
| MacOS    | iOS        | AArch64  | AArch64    | Clang              | ✅     | ✅          | |
| MacOS    | iOS        | AArch64  | x86        | Clang              | ✅     | ✅          | |
| Windows  | Windows    | x64      | x64        | Clang/MSVC         | ✅     | ✅          | |
| Windows  | Windows    | x64      | x86        | Clang/MSVC         | ✅     | ✅          | |
| Windows  | Windows    | x64      | AArch64    | Clang/MSVC         | ✅     | ✅          | |


(* Note: `AArch64` is the official name for the `arm64` CPU architecture).

(** Note: Supported combinations are regularly tested and should always work. If they don't, [raise a GitHub issue](https://github.com/google/XNNPACK/issues)).

## Building with Bazel

* Optional features are behind Bazel flags like
`--define=xnn_enable_avxvnni=false`.
* ⚠️ **Cross-compilation to another operating system (e.g. Android) or CPU
  architecture doesn't work on recent versions of Bazel**.
* Using Bazel's `:all` meta-target is not supported. Instead, use explicit
  targets like `//bench/...`.
* The `-c` parameter controls the optimization level:
   * `-c opt` enables `-O2`, with no assertions or debug information
     (recommended for production).
   * `-c fastbuild` means **no optimization**, minimal debug information, and
      assertions (for fast development and testing).
   * `-c dbg` means build with debug information enabled, **no optimization**
     (for debugging).

### Build benchmarks and test suites without cross-compilation, default compiler

```sh
bazel build -c fastbuild //bench/... //test/...
bazel test -c fastbuild --local_test_jobs=HOST_CPUS //bench/... //test/...
```

### Build benchmarks and test suites using an alternate compiler

Bazel listens to the `CC` and `CCX` environment variables, which specify
another compiler.

```sh
CC=clang-19 CXX=clang++-19 bazel build -c fastbuild //bench/... //test/...
CC=clang-19 CXX=clang++-19 bazel test -c fastbuild --local_test_jobs=HOST_CPUS //bench/... //test/...
```

### Combinations known to work
| Platform | CPU | Compiler     | `-c` config | Works | Supported | Notes |
| -------- | --- | -------------| ------------|-------|-----------| ----- |
| Linux    | x64 | GCC 14.2.0   | `fastbuild` |   ✅  |     ❌    | |
| Linux    | x64 | GCC 14.2.0   |       `opt` |   ❌  |     ❌    | |
| Linux    | x64 | Clang 19.1.7 | `fastbuild` |   ✅  |     ❌    | |
| Linux    | x64 | Clang 19.1.7 |       `opt` |   ✅  |     ❌    | |
| Linux    | x64 | Clang 18     | `fastbuild` |   ✅  |     ✅    | |
| Linux    | x64 | GCC 9        | `fastbuild` |       |     ✅    | With [appropriate defines](.github/workflows/build.yml). |
| Linux    | AArch64 | Clang 18 | `fastbuild` |       |     ✅    | |
| Linux    | AArch64 | Clang 20 | `fastbuild` |       |     ✅    | |
| Linux    | AArch64 | GCC 13   | `fastbuild` |       |     ✅    | |

## Building with GN

* GN is the build system used by the Chromium project.
* GN support is incomplete and experimental as of January 2026 and is not
  automatically tested yet. If these instructions do not work for you,
  [open an issue.](https://github.com/google/XNNPACK/issues)
* For more information on how GN works, see the 
  [upstream quick start guide.](https://gn.googlesource.com/gn/+/HEAD/docs/quick_start.md)

### Getting the code

XNNPACK uses Chromium's `depot_tools` to manage dependencies when building with
GN. To check out the code:

1. Install `depot_tools` according to the [upstream instructions.](https://commondatastorage.googleapis.com/chrome-infra-docs/flat/depot_tools/docs/html/depot_tools_tutorial.html#_setting_up)
2. Create an enter a new directory for XNNPACK with
   ```sh
   mkdir XNNPACK && cd XNNPACK
   ```
3. Create a file named `.gclient` with these contents:
   ```
    solutions = [
      { "name"        : 'xnnpack',
        "url"         : 'https://github.com/google/XNNPACK',
        "deps_file"   : 'DEPS',
        "managed"     : False,
        "custom_deps" : {
       },
       "custom_vars": {},
     },
   ]
   # Optionally, set target_os to include dependencies for an Android build
   target_os = ["android"]
   ```
4. Run `gclient sync` to get the source.

### Configuring a build

To set up a build for the first time, run `gn args out/Default`. `gn` will open
a text editor for configuring the build. Build arguments are given like this:

```
# Sample build configuration

# "x64", "x86" (32-bit), and "arm64" are supported
target_cpu = "x64"
# Asks for a release-optimized build
is_debug = false
# Enables assertions and some logging, but otherwise builds with release optimization
dcheck_always_on = true
# Retain some symbol information for profiling
symbol_level = 1

# Support for some instruction set extensions can be configured like this
# like this
xnnpack_enable_avx512 = true

# [Googlers-only], use Siso and Bazel remote execution to build faster
use_siso = true
use_remoteexec = true
```

After saving and exiting, GN will generate the build files.

To regenerate the build files after manually editing `out/Default/args.gn`,
run `gn gen out/Default`.

To see the available build arguments and what they're set to, use
`gn args --list out/Default`. Since many build arguments are shared
with the Chromium project, not all of them will affect XNNPACK.

You can have any number of build directories with different configurations
under the `out/` directory.

### Building

After configuring the build, run `autoninja -C out/Default` to build all XNNPACK
tests and benchmarks.

### Running tests

All GN-enabled tests are named `xnnpack_*_test`, e.g.
`out/Default/xnnpack_rminmax_test` and are runnable from the command line.

A helper script runs all the known unit tests:

```sh
vpython3 scripts/run-gn-tests.py out/Default
```

To run the tests on Android, they must first be pushed to the device.

```sh
adb push out/arm64/xnnpack_operators_test /data/local/tmp
# out/arm64/xnnpack_operators_test: 1 file pushed, 0 skipped. 299.7 MB/s (4525216 bytes in 0.014s)
adb shell /data/local/tmp/xnnpack_operators_test
```

### Supported configurations


| `target_os` | `target_cpu` | Works | Supported  | Notes |
| ----------- | ------------ | ----- | ---------- | ----- |
| `linux`     | `x64`        |  ✅   |   ❌       |       |
| `linux`     | `x86`        |  ✅   |   ❌       |       |
| `linux`     | `arm64`      |  ✅   |   ❌       |       |
| `linux`     | `arm64`      |  ✅   |   ❌       |       |
| `macos`     | `arm64`      |  ✅   |   ❌       |       |
| `android`   | `arm64`      |  ✅   |   ❌       |  [1]  |

[1] `target_os=["android"]` must be set in `.gclient` (see above).
    `is_component_build=true` is not supported.

Tests which involve YNNPACK are not supported yet.