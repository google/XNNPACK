mkdir build\windows
mkdir build\windows\arm64ec

set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static -DXNNPACK_ENABLE_ASSEMBLY=OFF -DXNNPACK_ENABLE_ARM_FP16=OFF -DXNNPACK_ENABLE_ARM_BF16=OFF
set CMAKE_ARGS=%CMAKE_ARGS% -G="Visual Studio 17 2022" -A=ARM64EC

rem Use-specified CMake arguments go last to allow overridding defaults
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo %CMAKE_ARGS%

cd build\windows\arm64ec && cmake ..\..\.. %CMAKE_ARGS%
cmake --build . -j %NUMBER_OF_PROCESSORS% --config Release
