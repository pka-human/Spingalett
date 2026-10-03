cmake_minimum_required(VERSION 3.21)
project(Spingalett VERSION 0.4.0 LANGUAGES C)

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

    if(NOT WIN32)
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
            set_target_properties(${EXAMPLE_NAME} PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
            )
        endif()
    endforeach()

    # DatasetTool is a utility rather than a demo: install it next to the library.
    if(APPLE)
        set_target_properties(DatasetTool PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
    elseif(UNIX)
        set_target_properties(DatasetTool PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    install(TARGETS DatasetTool RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
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
        target_link_libraries(DigitPad PRIVATE spingalett SDL2::SDL2)
        if(TARGET SDL2::SDL2main)
            target_link_libraries(DigitPad PRIVATE SDL2::SDL2main)
        endif()
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
    target_compile_definitions(SpingalettTests PRIVATE
        SPINGALETT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Tests/Data")
    if(UNIX AND NOT APPLE)
        target_link_libraries(SpingalettTests PRIVATE m)
        set_target_properties(SpingalettTests PROPERTIES BUILD_RPATH "$ORIGIN")
    endif()

    # White-box GEMM test: compiles the kernel source itself with the library's ISA flags.
    add_executable(SpingalettGemmTests ${CMAKE_CURRENT_SOURCE_DIR}/Tests/Spingalett.GemmTests.c
                   ${CMAKE_CURRENT_SOURCE_DIR}/Src/Spingalett.GEMM.c
                   ${CMAKE_CURRENT_SOURCE_DIR}/Src/Spingalett.Memory.c
                   ${CMAKE_CURRENT_SOURCE_DIR}/Src/Spingalett.Error.c)
    target_include_directories(SpingalettGemmTests PRIVATE Src ${CMAKE_CURRENT_SOURCE_DIR}/Include
                               ${CMAKE_CURRENT_BINARY_DIR}/Spingalett)
    target_compile_definitions(SpingalettGemmTests PRIVATE SPINGALETT_STATIC)
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

    foreach(group grad equiv cont optim sched dropout gen predict valid step data io xor)
        add_test(NAME core.${group} COMMAND SpingalettTests ${group}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()

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
