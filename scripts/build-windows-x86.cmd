mkdir build\windows
mkdir build\windows\x86

rem Set up the Visual Studio environment for x86 builds.
echo VCVARSALL: %VCVARSALL%
call "%VCVARSALL%" x86

rem Set up the CMake arguments.
set CMAKE_ARGS=-DXNNPACK_LIBRARY_TYPE=static -G="Ninja" -DCMAKE_BUILD_TYPE=Release
rem set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_VERBOSE_MAKEFILE=ON

rem Use-specified CMake arguments go last to allow overridding defaults.
set CMAKE_ARGS=%CMAKE_ARGS% %*

echo CMAKE_ARGS: %CMAKE_ARGS%

rem Configure the build.
cd build\windows\x86
cmake ..\..\.. %CMAKE_ARGS%

rem Run the build.
cmake --build . --config Release -- -j %NUMBER_OF_PROCESSORS%
