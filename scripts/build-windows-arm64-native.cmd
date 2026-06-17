mkdir build\windows
mkdir build\windows\arm64

rem Native ARM64 host build (e.g. GHA windows-11-arm runner, or a Snapdragon
rem X dev box). Uses the arm64 native cl.exe rather than the x64 -> arm64
rem cross used by scripts\build-windows-arm64.cmd. No CMAKE_TOOLCHAIN_FILE
rem is needed because we are not cross-compiling, which also lets ctest run
rem the produced binaries directly.
echo VCVARSALL: %VCVARSALL%
call "%VCVARSALL%" arm64

rem Set up the CMake arguments. Same minimal feature set as
rem scripts\build-windows-arm64.cmd so the two cl.exe baselines stay
rem comparable; clang-cl + ASM + FP16 lives in a separate script.
set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static -G="Ninja" -DCMAKE_BUILD_TYPE=Release
set CMAKE_ARGS=%CMAKE_ARGS% -DXNNPACK_ENABLE_ASSEMBLY=OFF -DXNNPACK_ENABLE_ARM_FP16_SCALAR=OFF -DXNNPACK_ENABLE_ARM_FP16_VECTOR=OFF
set CMAKE_ARGS=%CMAKE_ARGS% -DXNNPACK_ENABLE_ARM_I8MM=OFF -DXNNPACK_ENABLE_ARM_BF16=OFF -DXNNPACK_ENABLE_KLEIDIAI=OFF
rem set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_VERBOSE_MAKEFILE=ON

rem User-specified CMake arguments go last to allow overriding defaults.
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo CMAKE_ARGS: %CMAKE_ARGS%

rem Configure the build.
cd build\windows\arm64
cmake ..\..\.. %CMAKE_ARGS%

rem Run the build.
cmake --build . --config Release -- -j %NUMBER_OF_PROCESSORS%
