mkdir build\windows
mkdir build\windows\x64

set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static
set CMAKE_ARGS=%CMAKE_ARGS% -G="Visual Studio 17 2022" -A=x64

rem Use-specified CMake arguments go last to allow overridding defaults
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo %CMAKE_ARGS%

cd build\windows\x64 && cmake ..\..\.. %CMAKE_ARGS%
cmake --build . -j %NUMBER_OF_PROCESSORS% --config Release
