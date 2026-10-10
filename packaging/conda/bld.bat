:: SPDX-License-Identifier: MIT
cmake -S . -B build -G Ninja %CMAKE_ARGS% -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
    -DSPINGALETT_NATIVE_ARCH=OFF -DBUILD_WITH_OPENMP=ON -DSPINGALETT_VULKAN=ON -DBUILD_APPS=OFF -DBUILD_TESTS=OFF ^
    -DSPINGALETT_BIN_DIR="%SRC_DIR%\build\Bin" -DSPINGALETT_LIB_DIR="%SRC_DIR%\build\Lib"
if errorlevel 1 exit 1
cmake --build build --parallel %CPU_COUNT%
if errorlevel 1 exit 1
cmake --install build
if errorlevel 1 exit 1
