mkdir build\windows
mkdir build\windows\x86

set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static
set CMAKE_ARGS=%CMAKE_ARGS% -G="Visual Studio 17 2022" -A=Win32 -DCMAKE_CXX_FLAGS="/MP"

rem Use-specified CMake arguments go last to allow overridding defaults
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo %CMAKE_ARGS%

cd build\windows\x86 && cmake ..\..\.. %CMAKE_ARGS%
cmake --build . --config Release -- -m:%NUMBER_OF_PROCESSORS%
