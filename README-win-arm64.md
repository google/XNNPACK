# Native Windows on ARM64 (WoA) builds for XNNPACK

This branch (`chrisd/windows-arm64-native-all`) adds first-class support for
building XNNPACK **natively on a Windows on ARM64 host** (Snapdragon X /
Copilot+ class machines), instead of only cross-compiling from an x64 host.

It bundles three independent changes that are also available as separate
branches for upstream review (see [Upstreaming](#upstreaming) below).

## TL;DR — build it

From a clone of this branch on an ARM64 Windows machine (see
[Prerequisites](#prerequisites)):

```bat
:: Either run from an ARM64 VS Developer Command Prompt, or point VCVARSALL
:: at your VS 2022 install so the scripts can initialize one.
for /f "usebackq delims=" %i in (`vswhere -products * -latest -property installationPath`) do set "VCVARSALL=%i\VC\Auxiliary\Build\vcvarsall.bat"

:: Recommended: clang-cl path (ASM + FP16 micro-kernels enabled)
scripts\build-windows-arm64-clang.cmd

:: ...or the cl.exe baseline (smaller feature set, no ASM / FP16)
scripts\build-windows-arm64-native.cmd

:: ...or the armv8.4-a path (clang-cl + i8mm + bf16; needs a Snapdragon X / armv8.4+ host)
scripts\build-windows-arm64-clang-v84.cmd
```

Run the tests from the build directory:

```bat
:: clang-cl build
ctest --test-dir build\windows\arm64-clang -C Release --output-on-failure --parallel %NUMBER_OF_PROCESSORS%

:: clang-cl armv8.4-a build
ctest --test-dir build\windows\arm64-clang-v84 -C Release --output-on-failure --parallel %NUMBER_OF_PROCESSORS%

:: cl.exe build
ctest --test-dir build\windows\arm64 -C Release --output-on-failure --parallel %NUMBER_OF_PROCESSORS%
```

## Prerequisites

* An **ARM64 Windows host** (e.g. Snapdragon X Plus / Elite — Surface Laptop 7,
  Copilot+ PCs). Validated on Snapdragon X Plus (X1P-64-100).
* **Visual Studio 2022 17.14+** with the **ARM64 native** build tools
  (`v143` toolset). The native arm64 toolset ships in the image used by the
  GitHub `windows-11-arm` runner.
* **CMake** and **Ninja** (both included with VS 2022).
* For the clang-cl path only: **clang-cl** must be available. VS 2022 17.14
  ships it under the *"C++ Clang tools for Windows"* component
  (`<VC>\Tools\Llvm\ARM64\bin` or `<VC>\Tools\Llvm\bin`). A standalone
  LLVM 22.x install also works. The build script adds the LLVM `bin` directory
  to `PATH` automatically.

Run the scripts from an ARM64 VS Developer Command Prompt, or set `VCVARSALL`
to your VS 2022 `vcvarsall.bat`. If `VCINSTALLDIR` is already present the
clang-cl scripts reuse that environment; otherwise they call `VCVARSALL arm64`.

## Build profiles

| Script | Compiler | `-march` | ASM `.S` kernels | FP16 kernels | i8mm / bf16 | Output dir |
|---|---|---|---|---|---|---|
| `scripts\build-windows-arm64-native.cmd` | `cl.exe` (native arm64) | MSVC default | off | off | off | `build\windows\arm64` |
| `scripts\build-windows-arm64-clang.cmd` | `clang-cl` | `armv8.2-a+fp16+dotprod` | **on** | **on** | off | `build\windows\arm64-clang` |
| `scripts\build-windows-arm64-clang-v84.cmd` | `clang-cl` | `armv8.4-a+...+i8mm+bf16` | **on** | **on** | **on** | `build\windows\arm64-clang-v84` |

All three are *native* arm64-on-arm64 builds (no `CMAKE_TOOLCHAIN_FILE` cross),
so `ctest` can launch the produced binaries directly. The existing
`scripts\build-windows-arm64.cmd` (x64 → arm64 **cross**) is unchanged and
still works for x64 hosts; it just can't run the tests.

> The `-v84` profile raises the ISA baseline to `armv8.4-a`, so its binaries
> require an armv8.4+ host (Snapdragon X / Oryon and later) and will fault on
> older Windows-on-ARM SKUs (Snapdragon 8cx Gen 1/2/3). Use the plain
> `clang` profile for broad compatibility.

The Windows ARM64 scripts keep `XNNPACK_ENABLE_KLEIDIAI=OFF`. KleidiAI's
MSVC-compatible CMake path currently routes its ARM64 assembly through
`armasm64.exe`, which rejects the `/arch:armv8.2` source flags used by that
target.

### Why separate scripts?

`cl.exe` (VS 17.14) currently **refuses to assemble** the GNU-syntax `.S`
micro-kernels under `src/...-aarch*`, and **does not implement** the `__fp16`
scalar/vector intrinsics. So the cl.exe baseline forces
`XNNPACK_ENABLE_ASSEMBLY`, `XNNPACK_ENABLE_ARM_FP16_SCALAR` and
`_VECTOR` **off**.

`clang-cl` handles both, so the clang-cl path turns them **on** and is the
recommended way to get full performance. On a Snapdragon X Plus it measured
roughly **+17% throughput** on attention / transformer / mobilenet subgraph
benches versus the cl.exe baseline. The `-v84` variant additionally raises the
`-march` baseline to `armv8.4-a` and enables the i8mm / bf16 matrix-multiply
micro-kernels.

## What's in this branch

Three commits, each touching disjoint files:

1. **`CMakeLists.txt` — include Windows ARM64 in `AARCH64_DEFAULTS`.**
   The CMake `XNNPACK_TARGET_PROCESSOR_AARCH64_DEFAULTS` selector mirrored
   Bazel's `//build_config:aarch64`, which excludes Windows. That silently
   forced `XNNPACK_ENABLE_ARM_I8MM` / `SME` / `SME2` **off** on Windows ARM64
   even when requested on the command line. Snapdragon X / Copilot+ CPUs
   implement FEAT_I8MM and FEAT_BF16 (Windows reports them via
   `IsProcessorFeaturePresent`, and cpuinfo detects them), so the exclusion
   was a build-system artefact, not a hardware limit. With this change,
   `-DXNNPACK_ENABLE_ARM_I8MM=ON` actually takes effect.

2. **Native cl.exe build + CI.**
   * `scripts\build-windows-arm64-native.cmd` — native arm64 cl.exe build.
   * `.github/workflows/build.yml` — new `cmake-windows-arm64-native` job on
     the public `windows-11-arm` runner that builds and (when
     `inputs.run-tests` is set) runs `ctest`.

3. **clang-cl toolchains + builds.**
   * `cmake\clang-cl-arm64.toolchain` — clang-cl toolchain pinned to the
     conservative WoA baseline `armv8.2-a+fp16+dotprod` (supported on every
     shipping Windows-on-ARM SKU), with `-ffp-contract=off` so FMA
     contraction doesn't drift the `f32-vgelu` rational polynomial past test
     tolerance.
   * `scripts\build-windows-arm64-clang.cmd` — drives the toolchain and
     enables ASM + FP16.
   * `cmake\clang-cl-arm64-v84.toolchain` + `scripts\build-windows-arm64-clang-v84.cmd`
     — the same, walked up to `armv8.4-a` with `i8mm` + `bf16` enabled
     (requires an armv8.4+ host).

## Validation

On Snapdragon X Plus (X1P-64-100, Surface Laptop 7), Windows 11, LLVM 22.1.6:

* `scripts\build-windows-arm64-clang.cmd` builds and passes **494 / 494**
  ctest.
* Runtime probe confirms the i8mm micro-kernel is selected once
  `XNNPACK_ENABLE_ARM_I8MM=ON` (enabled by change #1).
* No perf regressions on subgraph attention / transformer / mobilenet benches.

> i8mm / bf16 need `armv8.4-a+`. Use `scripts\build-windows-arm64-clang-v84.cmd`
> (or pass a higher `-march` on the cmake line) to exercise them. SME/SME2
> are outside this profile and remain disabled by the helper script.

## Upstreaming

For sending to `google/XNNPACK`, the same work is split into three
single-purpose branches (preferred by upstream review). These branches carry
only the **armv8.2-a baseline** (the conservative Windows-on-ARM target
supported by every shipping WoA SKU):

| Branch | Scope | Baseline |
|---|---|---|
| `chrisd/cmake-aarch64-defaults-include-windows` | CMake `AARCH64_DEFAULTS` fix | n/a |
| `chrisd/ci-windows-arm64-native` | native cl.exe build + CI job | MSVC default |
| `chrisd/clang-cl-arm64-toolchain` | clang-cl toolchain + build script | `armv8.2-a+fp16+dotprod` |

The **armv8.4-a `-v84` profile** (`cmake/clang-cl-arm64-v84.toolchain` +
`scripts/build-windows-arm64-clang-v84.cmd`, enabling i8mm + bf16) currently
lives **only on this combined branch**. It is intended as a **future, separate
upstream split** — once the armv8.2 branches above land, it can be promoted to
its own single-purpose branch (e.g. `chrisd/clang-cl-arm64-i8mm`) and PR'd on
top, so reviewers get the higher ISA baseline as an isolated change.

This combined branch is intended for **internal sharing** so co-workers can
clone one branch and get everything.
