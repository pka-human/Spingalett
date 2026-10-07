cmake_minimum_required(VERSION 3.21)
project(Spingalett VERSION 0.8.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------------------
set(SPINGALETT_BIN_DIR ${CMAKE_SOURCE_DIR}/Bin CACHE PATH "Output directory for executables and shared libraries")
set(SPINGALETT_LIB_DIR ${CMAKE_SOURCE_DIR}/Lib CACHE PATH "Output directory for static/import libraries")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${SPINGALETT_BIN_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${SPINGALETT_BIN_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${SPINGALETT_LIB_DIR})

# ------------------------------------------------------------------------------
# Inference-only build: the engine of Spingalett.Inference.h and nothing else
# ------------------------------------------------------------------------------
option(SPINGALETT_INFERENCE_ONLY "Build only the inference engine: no training, file I/O, OpenMP or heap" OFF)
if(SPINGALETT_INFERENCE_ONLY)
    include(cmake/InferenceOnly.cmake)
    return()
endif()

# ------------------------------------------------------------------------------
# Feature flags (populated below, used for config header)
# ------------------------------------------------------------------------------
set(SPINGALETT_HAS_OPENMP OFF)
set(SPINGALETT_HAS_OPENBLAS OFF)
set(SPINGALETT_HAS_CUDA OFF)

# ------------------------------------------------------------------------------
# Library sources
# ------------------------------------------------------------------------------
file(GLOB SPINGALETT_SOURCES "Src/*.c")

add_library(spingalett SHARED ${SPINGALETT_SOURCES})
add_library(Spingalett::spingalett ALIAS spingalett)

target_include_directories(spingalett
    PRIVATE Src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/Spingalett>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(spingalett PRIVATE SPINGALETT_EXPORTS)

# Background threads (the data set reader decodes ahead): POSIX threads, or Win32 on Windows
if(NOT WIN32)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(spingalett PRIVATE Threads::Threads)
endif()

# ------------------------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------------------------
option(BUILD_WITH_OPENMP "Enable OpenMP" OFF)

if(BUILD_WITH_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_C_FOUND)
        target_link_libraries(spingalett PRIVATE OpenMP::OpenMP_C)
        set(SPINGALETT_HAS_OPENMP ON)
        message(STATUS "Spingalett: OpenMP enabled (FindOpenMP)")
    else()
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            find_library(OPENMP_omp_LIBRARY NAMES omp PATHS /usr/lib /usr/local/lib)
            if(OPENMP_omp_LIBRARY)
                include(CheckCCompilerFlag)
                check_c_compiler_flag("-fopenmp" HAVE_FOPENMP)
                if(HAVE_FOPENMP)
                    target_compile_options(spingalett PRIVATE -fopenmp)
                    target_link_libraries(spingalett PRIVATE -fopenmp)
                    set(SPINGALETT_HAS_OPENMP ON)
                    message(STATUS "Spingalett: OpenMP enabled (Clang fallback)")
                endif()
            endif()
        endif()

        if(NOT SPINGALETT_HAS_OPENMP)
            message(STATUS "Spingalett: OpenMP not found, single-threaded build")
        endif()
    endif()
else()
    message(STATUS "Spingalett: OpenMP disabled by option")
endif()

# ------------------------------------------------------------------------------
# OpenBLAS
# ------------------------------------------------------------------------------
option(BUILD_WITH_OPENBLAS "Use OpenBLAS for matrix ops" OFF)

if(BUILD_WITH_OPENBLAS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENBLAS_PKG QUIET openblas)
    endif()

    if(OPENBLAS_PKG_FOUND)
        target_link_libraries(spingalett PRIVATE ${OPENBLAS_PKG_LINK_LIBRARIES})
        target_include_directories(spingalett PRIVATE ${OPENBLAS_PKG_INCLUDE_DIRS})
        set(SPINGALETT_HAS_OPENBLAS ON)
        set(_spg_blas_inc ${OPENBLAS_PKG_INCLUDE_DIRS})
        set(_spg_blas_lib ${OPENBLAS_PKG_LINK_LIBRARIES})
        message(STATUS "Spingalett: OpenBLAS enabled via pkg-config (${OPENBLAS_PKG_LINK_LIBRARIES})")
    else()
        find_library(OPENBLAS_LIB NAMES openblas
            PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
        )
        find_path(OPENBLAS_INCLUDE NAMES cblas.h
            PATHS /usr/include /usr/local/include /usr/include/openblas
                  /usr/include/x86_64-linux-gnu
        )

        if(OPENBLAS_LIB AND OPENBLAS_INCLUDE)
            target_link_libraries(spingalett PRIVATE ${OPENBLAS_LIB})
            target_include_directories(spingalett PRIVATE ${OPENBLAS_INCLUDE})
            set(SPINGALETT_HAS_OPENBLAS ON)
            set(_spg_blas_inc ${OPENBLAS_INCLUDE})
            set(_spg_blas_lib ${OPENBLAS_LIB})
            message(STATUS "Spingalett: OpenBLAS enabled (${OPENBLAS_LIB})")
        else()
            message(WARNING "Spingalett: OpenBLAS requested but not found")
        endif()
    endif()

    if(SPINGALETT_HAS_OPENBLAS)
        message(STATUS "Spingalett: OpenBLAS uses its own thread pool (pthreads/OpenMP built-in)")

        # Other CBLAS providers lack the OpenBLAS thread API; probe for it.
        include(CheckSymbolExists)
        set(CMAKE_REQUIRED_INCLUDES ${_spg_blas_inc})
        set(CMAKE_REQUIRED_LIBRARIES ${_spg_blas_lib})
        check_symbol_exists(openblas_set_num_threads "cblas.h" SPINGALETT_OPENBLAS_THREAD_CONTROL)
        unset(CMAKE_REQUIRED_INCLUDES)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(SPINGALETT_OPENBLAS_THREAD_CONTROL)
            target_compile_definitions(spingalett PRIVATE SPINGALETT_OPENBLAS_THREAD_CONTROL)
        endif()
    endif()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/Spingalett/Spingalett.Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Spingalett/Spingalett.Config.h
    @ONLY
)

# ------------------------------------------------------------------------------
# Compiler flags
# ------------------------------------------------------------------------------
include(CheckCCompilerFlag)

# -march=native tunes the library for the build machine. Turn it off for binaries that must run
# elsewhere and pick a baseline through CMAKE_C_FLAGS instead (e.g. -march=x86-64-v3 for AVX2).
option(SPINGALETT_NATIVE_ARCH "Optimize for the build machine (-march=native)" ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(spingalett PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-missing-field-initializers
    )

    if(NOT WIN32)
        target_compile_options(spingalett PRIVATE -fvisibility=hidden)
    endif()

    target_compile_options(spingalett PRIVATE
        $<$<CONFIG:Release>:
            -O3
            -fno-math-errno
            -fno-signed-zeros
            -fno-trapping-math
            -fassociative-math
            -freciprocal-math
            -funroll-loops
            -fomit-frame-pointer
            -fno-plt
            -fno-stack-protector
            -ffunction-sections
            -fdata-sections
            -ftree-vectorize
        >
        $<$<CONFIG:RelWithDebInfo>:
            -O2
            -fno-math-errno
            -fno-signed-zeros
            -fno-trapping-math
            -fassociative-math
            -freciprocal-math
            -funroll-loops
            -ffunction-sections
            -fdata-sections
            -ftree-vectorize
        >
        $<$<CONFIG:Debug>:
            -O0 -g3
        >
    )

    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        target_compile_options(spingalett PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:
                -fno-semantic-interposition
                -fprefetch-loop-arrays
            >
        )
    endif()

    if(SPINGALETT_NATIVE_ARCH AND NOT WIN32)
        target_compile_options(spingalett PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:-march=native -mtune=native>
        )
    endif()

    if(APPLE)                                   # ld64: no --gc-sections or -s
        target_link_options(spingalett PRIVATE
            $<$<CONFIG:Release>:-Wl,-dead_strip -Wl,-x>
            $<$<CONFIG:RelWithDebInfo>:-Wl,-dead_strip>
        )
    elseif(NOT WIN32)
        target_link_options(spingalett PRIVATE
            $<$<CONFIG:Release>:-s -Wl,--gc-sections>
            $<$<CONFIG:RelWithDebInfo>:-Wl,--gc-sections>
        )
    endif()

elseif(MSVC)
    target_compile_options(spingalett PRIVATE /W4)
    target_compile_options(spingalett PRIVATE
        $<$<CONFIG:Release>:/O2 /Oi /GL /fp:fast>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Oi>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
    target_link_options(spingalett PRIVATE
        $<$<CONFIG:Release>:/LTCG>
    )

    check_c_compiler_flag("/arch:AVX2" MSVC_HAS_AVX2)
    if(MSVC_HAS_AVX2)
        target_compile_options(spingalett PRIVATE
            $<$<NOT:$<CONFIG:Debug>>:/arch:AVX2>
        )
        message(STATUS "Spingalett: MSVC AVX2+FMA enabled")
    else()
        check_c_compiler_flag("/arch:AVX" MSVC_HAS_AVX)
        if(MSVC_HAS_AVX)
            target_compile_options(spingalett PRIVATE
                $<$<NOT:$<CONFIG:Debug>>:/arch:AVX>
            )
            message(STATUS "Spingalett: MSVC AVX enabled")
        endif()
    endif()
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(spingalett PRIVATE m)
endif()

# x86-64 binaries that are not tuned for the build machine (release packages, MinGW) also carry
# GEMM kernels for AVX2 and AVX-512 (Src/Spingalett.GEMM.c) and batched INT8 kernels for AVX-512
# VNNI and AVX-VNNI (Src/Spingalett.Int8Tiles.c), and pick them at run time; AArch64 Linux ones
# carry INT8 kernels for the dot product instructions.
option(SPINGALETT_CPU_DISPATCH "Choose AVX2, AVX-512, VNNI and ARM dot product kernels at run time on builds without -march=native" ON)
set(_spg_native_flags OFF)
if(SPINGALETT_NATIVE_ARCH AND NOT WIN32 AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(_spg_native_flags ON)
endif()
if(SPINGALETT_CPU_DISPATCH AND NOT _spg_native_flags AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(SPINGALETT_GEMM_VARIANTS
        ${CMAKE_CURRENT_SOURCE_DIR}/Src/Kernels/Spingalett.GEMM.AVX2.c
        ${CMAKE_CURRENT_SOURCE_DIR}/Src/Kernels/Spingalett.GEMM.AVX512.c)
    target_sources(spingalett PRIVATE ${SPINGALETT_GEMM_VARIANTS}
                   ${CMAKE_CURRENT_SOURCE_DIR}/Src/Kernels/Spingalett.Int8Tiles.AVX512.c
                   ${CMAKE_CURRENT_SOURCE_DIR}/Src/Kernels/Spingalett.Int8Tiles.AVXVNNI.c)
    target_compile_definitions(spingalett PRIVATE SPINGALETT_GEMM_DISPATCH SPINGALETT_INT8_DISPATCH)
    # -fno-lto: link-time optimization must not move code between instruction sets
    set_source_files_properties(Src/Kernels/Spingalett.GEMM.AVX2.c
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-fno-lto")
    set_source_files_properties(Src/Kernels/Spingalett.GEMM.AVX512.c
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx2;-mfma;-fno-lto")
    set_source_files_properties(Src/Kernels/Spingalett.Int8Tiles.AVX512.c
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni;-mavx2;-mfma;-fno-lto")
    set_source_files_properties(Src/Kernels/Spingalett.Int8Tiles.AVXVNNI.c
        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mavxvnni;-fno-lto")
    message(STATUS "Spingalett: GEMM kernels for AVX2 and AVX-512, INT8 kernels for AVX-512 VNNI and AVX-VNNI, chosen at run time")
elseif(SPINGALETT_CPU_DISPATCH AND NOT _spg_native_flags AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang"
       AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # AArch64 Linux: batched INT8 kernels for the dot product instructions (ARMv8.2), where present
    target_sources(spingalett PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src/Kernels/Spingalett.Int8Tiles.DotProd.c)
    target_compile_definitions(spingalett PRIVATE SPINGALETT_INT8_DISPATCH)
    set_source_files_properties(Src/Kernels/Spingalett.Int8Tiles.DotProd.c
        PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod;-fno-lto")
    message(STATUS "Spingalett: INT8 kernels for the ARM dot product instructions chosen at run time")
endif()

# The inference engine and batched model inference compute in IEEE order (no reassociation,
# reciprocals or contraction), so the two agree bit for bit and match the standalone engine.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(SPINGALETT_STRICT_FP -fno-associative-math -fno-reciprocal-math -ffp-contract=off)
    set_source_files_properties(Src/Spingalett.Inference.c Src/Spingalett.Model.c
        PROPERTIES COMPILE_OPTIONS "${SPINGALETT_STRICT_FP}")
endif()

if(NOT MSVC)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_result)
    if(ipo_result)
        set_target_properties(spingalett PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        )
    endif()
endif()

# Before 1.0 a minor release may break the ABI, so the soname carries major.minor (.so.0.4).
if(PROJECT_VERSION_MAJOR EQUAL 0)
    set(SPINGALETT_SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
else()
    set(SPINGALETT_SOVERSION ${PROJECT_VERSION_MAJOR})
endif()
set_target_properties(spingalett PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SPINGALETT_SOVERSION}
)
# macOS packages carry their OpenMP runtime (@rpath/libomp.dylib) next to the library
if(APPLE)
    set_target_properties(spingalett PROPERTIES INSTALL_RPATH "@loader_path")
endif()

# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------
option(BUILD_EXAMPLE "Build examples" ON)

if(BUILD_EXAMPLE)
    file(GLOB EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Examples/*.c")
    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)

        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE spingalett)
        if(UNIX AND NOT APPLE)
            target_link_libraries(${EXAMPLE_NAME} PRIVATE m)       # stb_image in DatasetTool
        endif()

        if(UNIX AND NOT APPLE)
            set_target_properties(${EXAMPLE_NAME} PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
            )
        endif()
    endforeach()

    # DatasetTool and ModelTool are utilities rather than demos: install them next to the library.
    foreach(SPINGALETT_TOOL DatasetTool ModelTool)
        if(APPLE)
            set_target_properties(${SPINGALETT_TOOL} PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
        elseif(UNIX)
            set_target_properties(${SPINGALETT_TOOL} PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        endif()
        install(TARGETS ${SPINGALETT_TOOL} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endforeach()
endif()

# ------------------------------------------------------------------------------
# Apps: DigitPad (draw a digit, see the prediction; needs SDL2) and its trainer
# ------------------------------------------------------------------------------
option(BUILD_APPS "Build the DigitPad demo app and its trainer" OFF)

if(BUILD_APPS)
    set(DIGITPAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Apps/DigitPad)

    add_executable(DigitPadTrain ${DIGITPAD_DIR}/Train.c)
    target_link_libraries(DigitPadTrain PRIVATE spingalett)

    find_package(SDL2 CONFIG QUIET)
    if(TARGET SDL2::SDL2)
        add_executable(DigitPad ${DIGITPAD_DIR}/DigitPad.c)
        if(WIN32)
            # a GUI program (no console window) with an icon, version information and a manifest
            target_sources(DigitPad PRIVATE ${DIGITPAD_DIR}/Package/DigitPad.rc)
            set_target_properties(DigitPad PROPERTIES WIN32_EXECUTABLE ON)
        endif()
        if(TARGET SDL2::SDL2main)   # before SDL2, which resolves its references
            target_link_libraries(DigitPad PRIVATE SDL2::SDL2main)
        endif()
        target_link_libraries(DigitPad PRIVATE spingalett SDL2::SDL2)
        install(TARGETS DigitPad RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    else()
        message(STATUS "Spingalett: SDL2 not found, DigitPad app not built (DigitPadTrain is)")
    endif()

    foreach(DIGITPAD_TARGET DigitPadTrain DigitPad)
        if(TARGET ${DIGITPAD_TARGET} AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${DIGITPAD_TARGET} PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
        endif()
        if(TARGET ${DIGITPAD_TARGET} AND UNIX AND NOT APPLE)
            target_link_libraries(${DIGITPAD_TARGET} PRIVATE m)
            set_target_properties(${DIGITPAD_TARGET} PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
            )
        endif()
    endforeach()
endif()

# ------------------------------------------------------------------------------
# Tests (run with: ctest --test-dir <build> --output-on-failure)
# ------------------------------------------------------------------------------
option(BUILD_TESTS "Build the test suite" ON)

if(BUILD_TESTS)
    enable_testing()

    add_executable(SpingalettTests ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Spingalett.Tests.c)
    target_link_libraries(SpingalettTests PRIVATE spingalett)
    target_include_directories(SpingalettTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Src)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        # the deployment tests recompute the engine's arithmetic and compare it bit for bit
        target_compile_options(SpingalettTests PRIVATE -ffp-contract=off)
    endif()
    target_compile_definitions(SpingalettTests PRIVATE
        SPINGALETT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Tests/Data")
    if(UNIX AND NOT APPLE)
        target_link_libraries(SpingalettTests PRIVATE m)
        set_target_properties(SpingalettTests PROPERTIES BUILD_RPATH "$ORIGIN")
    endif()
    if(NOT WIN32)
        target_link_libraries(SpingalettTests PRIVATE Threads::Threads)     # predictions from several threads
    endif()

    # White-box GEMM test: compiles the kernel source itself with the library's ISA flags.
    add_executable(SpingalettGemmTests ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Spingalett.GemmTests.c
                   ${CMAKE_CURRENT_SOURCE_DIR}/Src/Spingalett.GEMM.c
                   ${CMAKE_CURRENT_SOURCE_DIR}/Src/Spingalett.Memory.c
                   ${CMAKE_CURRENT_SOURCE_DIR}/Src/Spingalett.Error.c)
    target_include_directories(SpingalettGemmTests PRIVATE Src ${CMAKE_CURRENT_SOURCE_DIR}/Include
                               ${CMAKE_CURRENT_BINARY_DIR}/Spingalett)
    target_compile_definitions(SpingalettGemmTests PRIVATE SPINGALETT_STATIC)
    if(SPINGALETT_GEMM_VARIANTS)              # every kernel set the processor supports
        target_sources(SpingalettGemmTests PRIVATE ${SPINGALETT_GEMM_VARIANTS})
        target_compile_definitions(SpingalettGemmTests PRIVATE SPINGALETT_GEMM_DISPATCH)
    endif()
    if(SPINGALETT_NATIVE_ARCH AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        target_compile_options(SpingalettGemmTests PRIVATE -O2 -march=native)
    endif()
    if(SPINGALETT_HAS_OPENMP AND TARGET OpenMP::OpenMP_C)
        target_link_libraries(SpingalettGemmTests PRIVATE OpenMP::OpenMP_C)
    endif()
    if(SPINGALETT_HAS_OPENBLAS)
        target_include_directories(SpingalettGemmTests PRIVATE ${_spg_blas_inc})
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(SpingalettGemmTests PRIVATE m)
    endif()
    add_test(NAME core.gemm COMMAND SpingalettGemmTests)

    foreach(group grad conv norm equiv cont optim sched dropout gen predict valid step data io model xor)
        add_test(NAME core.${group} COMMAND SpingalettTests ${group}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()

    # The inference engine on its own, also running models exported as C headers.
    include(cmake/SpingalettEngine.cmake)
    spingalett_engine_library(SpingalettEngine)
    spingalett_engine_tests(SpingalettEngine SpingalettTests)

    # A sanitizer-instrumented library cannot be loaded into an uninstrumented Python.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND AND NOT CMAKE_C_FLAGS MATCHES "-fsanitize" AND NOT CMAKE_CROSSCOMPILING)
        execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                        RESULT_VARIABLE SPINGALETT_NUMPY_MISSING OUTPUT_QUIET ERROR_QUIET)
        if(SPINGALETT_NUMPY_MISSING EQUAL 0)
            add_test(NAME python.bindings
                     COMMAND ${CMAKE_COMMAND} -E env
                             SPINGALETT_LIBRARY=$<TARGET_FILE:spingalett>
                             PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/Bindings/Python
                             ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Tests/test_python_bindings.py)

            add_executable(SpingalettLayout ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Spingalett.Layout.c)
            target_link_libraries(SpingalettLayout PRIVATE spingalett)
            add_test(NAME python.layout
                     COMMAND ${CMAKE_COMMAND} -E env
                             SPINGALETT_LIBRARY=$<TARGET_FILE:spingalett>
                             PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/Bindings/Python
                             ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Tests/test_python_layout.py
                             $<TARGET_FILE:SpingalettLayout>)
        else()
            message(STATUS "Spingalett: NumPy not found, skipping Python binding tests")
        endif()
    endif()
endif()

# ------------------------------------------------------------------------------
# Install (headers, library and a CMake package: find_package(Spingalett))
# ------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS spingalett
    EXPORT SpingalettTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY Include/Spingalett DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PATTERN "*.in" EXCLUDE)
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/Spingalett/Spingalett.Config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Spingalett
)

set(SPINGALETT_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/Spingalett)
install(EXPORT SpingalettTargets NAMESPACE Spingalett:: DESTINATION ${SPINGALETT_CMAKE_DIR})
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpingalettConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SpingalettConfig.cmake
    INSTALL_DESTINATION ${SPINGALETT_CMAKE_DIR}
)
# Before 1.0 a minor release may change the API, so only the same major.minor is compatible.
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SpingalettConfigVersion.cmake
    COMPATIBILITY SameMinorVersion
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SpingalettConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/SpingalettConfigVersion.cmake
        DESTINATION ${SPINGALETT_CMAKE_DIR})
