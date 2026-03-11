cmake_minimum_required(VERSION 3.21)
project(Spingalett VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Output directories
# ------------------------------------------------------------------------------
set(SPINGALETT_BIN_DIR ${CMAKE_SOURCE_DIR}/Bin)
set(SPINGALETT_LIB_DIR ${CMAKE_SOURCE_DIR}/Lib)

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

target_include_directories(spingalett
    PRIVATE Src
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:Include>
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
            message(STATUS "Spingalett: OpenBLAS enabled (${OPENBLAS_LIB})")
        else()
            message(WARNING "Spingalett: OpenBLAS requested but not found")
        endif()
    endif()

    if(SPINGALETT_HAS_OPENBLAS)
        message(STATUS "Spingalett: OpenBLAS uses its own thread pool (pthreads/OpenMP built-in)")
    endif()
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include/Spingalett/Spingalett.Config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/Spingalett/Spingalett.Config.h
    @ONLY
)

target_include_directories(spingalett
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Spingalett

)

# ------------------------------------------------------------------------------
# Compiler flags
# ------------------------------------------------------------------------------
include(CheckCCompilerFlag)

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

    if(NOT WIN32)
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

set_target_properties(spingalett PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
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
        target_include_directories(${EXAMPLE_NAME} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/Include
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/Spingalett
        )
        target_link_libraries(${EXAMPLE_NAME} PRIVATE spingalett)

        if(UNIX AND NOT APPLE)
            set_target_properties(${EXAMPLE_NAME} PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
            )
        endif()
    endforeach()
endif()

# ------------------------------------------------------------------------------
# Install
# ------------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS spingalett
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY Include/Spingalett DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/Spingalett/Spingalett.Config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Spingalett
)
