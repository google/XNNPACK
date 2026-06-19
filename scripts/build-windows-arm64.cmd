mkdir build\windows
mkdir build\windows\arm64

rem Set up the Visual Studio environment for arm64 builds.
echo VCVARSALL: %VCVARSALL%
call "%VCVARSALL%" x64_arm64

rem Set up the CMake arguments.
set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static -G="Ninja" -DCMAKE_BUILD_TYPE=Release
set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_TOOLCHAIN_FILE=%cd%\cmake\x64_arm64.toolchain
set CMAKE_ARGS=%CMAKE_ARGS% -DXNNPACK_ENABLE_ASSEMBLY=OFF -DXNNPACK_ENABLE_ARM_FP16_SCALAR=OFF -DXNNPACK_ENABLE_ARM_BF16=OFF
rem set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_VERBOSE_MAKEFILE=ON

rem Use-specified CMake arguments go last to allow overridding defaults.
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo CMAKE_ARGS: %CMAKE_ARGS%

rem Configure the build.
cd build\windows\arm64
cmake ..\..\.. %CMAKE_ARGS%

rem Run the build.
cmake --build . --config Release -- -j %NUMBER_OF_PROCESSORS%
