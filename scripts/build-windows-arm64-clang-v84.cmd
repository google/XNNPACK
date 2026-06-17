mkdir build\windows
mkdir build\windows\arm64-clang-v84

rem Native ARM64 host build using clang-cl at the armv8.4-a baseline
rem (Profile 3: the armv8.2-a clang-cl baseline plus i8mm + bf16, FMA
rem contraction off). Pairs with cmake/clang-cl-arm64-v84.toolchain.
rem Compared to scripts/build-windows-arm64-clang.cmd, this additionally
rem enables the i8mm / bf16 micro-kernels. Requires an armv8.4+ host
rem (Snapdragon X / Oryon and later).
if defined VCINSTALLDIR (
  echo VCINSTALLDIR: %VCINSTALLDIR%
) else (
  echo VCVARSALL: %VCVARSALL%
  if "%VCVARSALL%"=="" (
    echo VCVARSALL is not set. Run VsDevCmd.bat or set VCVARSALL before invoking this script.
    exit /b 1
  )
  call "%VCVARSALL%" arm64
)

rem clang-cl must be on PATH. VS 2022 17.14 ships clang-cl inside the VC
rem LLVM component at <VC>\Tools\Llvm\bin (and ARM64 variant under
rem \Llvm\ARM64\bin). The VS arm64 environment does NOT always add
rem Llvm\bin to PATH by default; tack it on so the cmake compiler probe
rem finds clang-cl.exe.
set CLANG_DIR=%VCINSTALLDIR%Tools\Llvm\ARM64\bin
if not exist "%CLANG_DIR%\clang-cl.exe" set CLANG_DIR=%VCINSTALLDIR%Tools\Llvm\bin
set PATH=%CLANG_DIR%;%PATH%
echo clang-cl: %CLANG_DIR%\clang-cl.exe
"%CLANG_DIR%\clang-cl.exe" --version

rem Set up the CMake arguments.
set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static -G="Ninja" -DCMAKE_BUILD_TYPE=Release
set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_TOOLCHAIN_FILE=%cd%\cmake\clang-cl-arm64-v84.toolchain
set CMAKE_ARGS=%CMAKE_ARGS% -DXNNPACK_ENABLE_ASSEMBLY=ON -DXNNPACK_ENABLE_ARM_FP16_SCALAR=ON -DXNNPACK_ENABLE_ARM_FP16_VECTOR=ON
set CMAKE_ARGS=%CMAKE_ARGS% -DXNNPACK_ENABLE_ARM_I8MM=ON -DXNNPACK_ENABLE_ARM_BF16=ON -DXNNPACK_ENABLE_ARM_SME=OFF -DXNNPACK_ENABLE_ARM_SME2=OFF -DXNNPACK_ENABLE_KLEIDIAI=OFF
rem set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_VERBOSE_MAKEFILE=ON

rem User-specified CMake arguments go last to allow overriding defaults.
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo CMAKE_ARGS: %CMAKE_ARGS%

rem Configure the build.
cd build\windows\arm64-clang-v84
cmake ..\..\.. %CMAKE_ARGS%

rem Run the build.
cmake --build . --config Release -- -j %NUMBER_OF_PROCESSORS%
