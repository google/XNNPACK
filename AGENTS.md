# Agent Instructions for XNNPACK

Guidelines for automated agents and developers contributing to XNNPACK.

## Overview & Architecture

XNNPACK is a highly optimized library of neural network inference operators
for ARM, x86, RISC-V, and WebAssembly architectures.

The codebase is organized into three primary layers:
1. **Graph & Runtime API** (`include/xnnpack.h`, `src/subgraph/`,
   `src/runtime.c`): Represents models as subgraphs, optimizes execution
   plans, manages tensor memory allocations, and schedules execution.
2. **Operator API** (`src/operators/`): Implements operator lifecycle:
   `create`, `reshape`, and `setup`. Computes workspace sizes, chooses
   microkernels, and packs weights.
3. **Microkernels & Hardware Dispatch** (`src/<datatype>-<op>/`,
   `src/configs/*-config.c`, `src/xnnpack/`): Hardware-specific inner loops
   optimized with architecture SIMD/vector intrinsics or assembly. Production
   dispatch is resolved dynamically based on CPU feature detection.

### YNNPACK (`ynnpack/`)

YNNPACK is the next-generation execution engine based on Slinky for cross-node
loop fusion (`ynnpack/README.md`). Keep changes to legacy XNNPACK (`src/`) and
YNNPACK (`ynnpack/`) strictly separated. When touching YNNPACK, maintain Slinky
buffer bounds, pipeline stage contracts, and compatibility shim constraints.

## What to Avoid in Pull Requests (Strict Anti-Patterns)

Pull requests violating these principles will be closed:

- **No mass "hardening" or speculative defensive sweeps**: Do not add
  redundant `NULL` checks, bounds checks, or integer overflow wrappers to
  internal `static` functions or hot loops where callers already guarantee
  invariants. Input validation belongs exclusively at public and subgraph
  boundaries. Any boundary validation PR must be scoped to a single operator
  and include a regression test in `test/subgraph/<op>.cc` demonstrating a
  real, triggerable issue.
- **Never replace `XNN_UNREACHABLE()` or assertions with fallbacks**:
  `XNN_UNREACHABLE()` and internal assertions are intentional fail-fast
  invariants and critical compiler optimization hints. Do not replace them
  with fallback return values, default strings (e.g., `"<unknown>"`), or
  soft errors.
- **Never zero-initialize padding or unused lanes to silence sanitizers**:
  Buffer padding bytes and unused SIMD tail lanes are intentionally left
  uninitialized so MemorySanitizer (MSan) can detect illicit out-of-bounds
  reads by microkernels. Do not zero-fill memory to silence sanitizers.
- **No microkernel or performance changes without hardware benchmarks**:
  Do not submit assembly, intrinsics, loop unrolling, or dispatch config
  (`src/configs/*-config.c`) changes without reproducible before/after
  benchmark tables from `bench/` on target physical hardware (including
  specific SoC/CPU model, compiler version, and thread pinning).
- **Never edit auto-generated files directly**: Files containing the header
  `Auto-generated file. Do not edit!` (`src/**/gen/*.c`, `gen/*.bzl`,
  `cmake/gen/*.cmake`, generated `test/*.cc`, `bench/*.cc`) must only be
  modified by updating their generators or templates.
- **No monolithic PRs or cosmetic sweeps**: Keep PRs atomic: one operator or
  one microkernel family per PR. Integrate tests into the canonical test file
  (`test/subgraph/<op>.cc` or `test/<op>.cc`) rather than creating new ad-hoc
  test files. Do not submit formatting sweeps, comment rewrites, or YAML-to-
  table refactoring churn.
- **No unnecessary or overly detailed code comments**: Default to **no
  comment** when code is self-explanatory (such as parameter validation,
  dimension/rank checks, status returns, or standard control flow). Never add
  multi-line comments explaining how a check works or restating the PR
  description inside the code. Only add a brief comment when a non-obvious
  hardware quirk, ISA constraint, or algorithmic invariant cannot be expressed
  in code.
- **No verbose AI-generated PR descriptions or review essays**: Keep PR titles,
  descriptions, and review replies concise (1-3 sentences stating what changed
  and why, plus test or benchmark output). Do not paste multi-section essays
  (`Problem`, `Root Cause Analysis`, `Solution Architecture`).
- **No temporary files or compilation artifacts**: Do not include temporary
  scripts, compilation artifacts (such as `.pyc` or `__pycache__`), or
  unrelated files in a pull request. Ensure the working tree is clean before
  submitting.

## Code Generation Workflows

Over 7,000 source, test, benchmark, and build files are generated from
templates and specification files.

### 1. Microkernel Templates (`src/**/*.c.in`)
- Templates use `tools/xngen.py` syntax (`$for`, `$if`, `${var}`).
- Edit the corresponding `.c.in` template under `src/<datatype>-<op>/`. When
  adding a new `.c.in` template, set the copyright date in its header to the
  current year before generating C sources from it.
- Regenerate C sources by running the operator script:
  ```bash
  ./scripts/generate-<datatype>-<op>.sh
  ```
  Generated C files are placed in `src/<datatype>-<op>/gen/`.

### 2. Registering Microkernels
When adding a new microkernel:
1. Follow naming rules in `doc/microkernel-naming-conventions.md`:
   `xnn_<datatype>_<op>_ukernel_<params>__<arch>_u<unroll>`
2. Declare function prototypes in `src/xnnpack/<op>.h`.
3. Add `XNN_UKERNEL(...)` entries to
   `src/<datatype>-<op>/<datatype>-<op>.inc` (see
   `doc/microkernel-enumerators.md`).
4. If enabling for production dispatch, update `src/configs/<op>-config.c`
   and validate configuration correctness:
   ```bash
   python3 scripts/check_config.py src/configs/<op>-config.c
   ```
5. Update Bazel (`gen/*.bzl`) and CMake (`cmake/gen/*.cmake`) file lists:
   ```bash
   python3 tools/update-microkernels.py
   ```

### 3. Tests and Benchmarks (`test/*.yaml`)
- Microkernel tests and benchmarks are defined in YAML files under `test/`.
- Edit `test/<datatype>-<op>.yaml` and regenerate test and benchmark sources:
  ```bash
  ./scripts/generate-tests.sh
  ```
  (or run the specific generator: `python3 tools/generate-*-test.py`).

## Build Systems & Verification

XNNPACK maintains multi-build parity across **CMake**, **Bazel**, and **GN**.
Any change adding source files, dependencies, or compiler flags must update all
three build definitions.

### Parity Verification
- Compare exported `compile_commands.json` across GN and CMake (see
  `BUILD.md` for generation steps):
  ```bash
  python3 scripts/compare-compile-commands.py \
    out/Default/compile_commands.json build/local/compile_commands.json
  ```

### Local CMake Build & Test
```bash
./scripts/build-local.sh
ctest --test-dir build/local --output-on-failure --parallel $(nproc)
```

### Local Bazel Build & Test
```bash
bazel test -c fastbuild //test/...
```

## Code Style & Microkernel Conventions

- **Copyright Notice on New Source Files**: When adding a new source file in a
  pull request (including `.c`, `.cc`, `.h`, `.c.in` templates, `.py`, and
  `.sh`) that will have a copyright notice, set the copyright date in the
  header to the current year (and regenerate any generated files derived from a
  new `.c.in` template). Only set the copyright date to the current year on
  newly added source files; do not modify copyright headers on existing files
  or existing templates.
- **Language Standards**: C99/C11 (`-std=c99` or `-std=c11`, no GNU
  extensions) for core library and microkernels. C++17 for `test/` and
  `bench/`. Python 3 for `tools/` and `scripts/`.
- **Formatting**: Google C++ style enforced via `.clang-format`.
- **Internal Primitives**:
  - Memory: `xnn_allocate_memory` and `xnn_allocate_simd_memory`
    (`src/xnnpack/allocator.h`).
  - Logging: `xnn_log_error`, `xnn_log_warning`, `xnn_log_info`
    (`src/xnnpack/log.h`).
  - Inlining: `static XNN_INLINE` on performance-critical helpers in inner
    loops.
  - Branch Hints: `XNN_LIKELY` and `XNN_UNLIKELY` on tail iterations and
    error paths.
- **Comments**: Explain *why* an unusual constraint or non-obvious algorithm
  is used. Do not explain *what* standard C code does, and avoid LLM commentary
  or conversational restatements.
- **RISC-V Vector (RVV)**:
  - Target optimization for SiFive X280.
  - Prefer `m2` or `m4` LMUL. Avoid `m1` and `m8`.
  - Hoist `vsetvl` configuration outside inner loops when `n >= vlmax`.
- **Testing**:
  - Never use `XNN_OPTNONE` (or `optnone` attributes) in tests unless
    specifically requested.
