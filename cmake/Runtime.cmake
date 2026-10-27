# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# The runtime (Include/Spingalett/Spingalett.Runtime.h): the library's deployment models without
# training, data sets, importers or the GPU backend, for programs that only run trained models. It is
# the part of the library's sources that loads, prepares and runs models, compiled with the library's
# flags and kernels (spingalett_build) and with SPINGALETT_RUNTIME, which leaves out what would need
# the rest, and it exports the functions of Src/Spingalett.Runtime.map. Built next to the library
# (SPINGALETT_RUNTIME, on by default) or alone (SPINGALETT_RUNTIME_ONLY), as libspingalett-runtime.

set(SPINGALETT_RUNTIME_SOURCES Inference Model Session Blocks Attention Image Gguf Tokenizer Json Chat IR IROps IROnnx IRFile IRFuse IRShapes IRLayout
    GEMM ConvGEMM Int8Tiles SIMD Conv Error Memory Settings Thread)
list(TRANSFORM SPINGALETT_RUNTIME_SOURCES REPLACE "^(.+)$" "${PROJECT_SOURCE_DIR}/Src/Spingalett.\\1.c")

function(spingalett_runtime_library)
    add_library(spingalett_runtime SHARED ${SPINGALETT_RUNTIME_SOURCES})
    add_library(Spingalett::runtime ALIAS spingalett_runtime)
    target_compile_definitions(spingalett_runtime PRIVATE SPINGALETT_RUNTIME)
    set_target_properties(spingalett_runtime PROPERTIES OUTPUT_NAME spingalett-runtime EXPORT_NAME runtime)
    spingalett_library(spingalett_runtime ${PROJECT_SOURCE_DIR}/Src/Spingalett.Runtime.map)
endfunction()

# Examples/Runtime: programs of the runtime alone. RunModel, a tool, is installed with it.
# The GPU's backends in the runtime (SPINGALETT_RUNTIME_GPU): the device layer, the matrix product's tiles and the
# executor of graphs of operators, with the backends Vulkan.cmake and Cuda.cmake found (their kernels embedded).
option(SPINGALETT_RUNTIME_GPU "Build the runtime with the GPU's backends (graphs of operators on CUDA and Vulkan)" ON)
function(spingalett_runtime_gpu)
    if(NOT TARGET spingalett_runtime OR NOT SPINGALETT_RUNTIME_GPU OR NOT (SPINGALETT_HAS_VULKAN OR SPINGALETT_HAS_CUDA))
        return()
    endif()
    set(gpu_dir ${PROJECT_SOURCE_DIR}/Src/Gpu)
    target_sources(spingalett_runtime PRIVATE ${gpu_dir}/Spingalett.Device.c ${gpu_dir}/Spingalett.GpuKernels.c
                                              ${gpu_dir}/Spingalett.GpuGraph.c ${gpu_dir}/Spingalett.GpuSession.c)
    target_compile_definitions(spingalett_runtime PRIVATE SPINGALETT_RUNTIME_GPU)
    target_include_directories(spingalett_runtime PRIVATE ${gpu_dir})
    if(NOT WIN32)
        target_link_libraries(spingalett_runtime PRIVATE ${CMAKE_DL_LIBS})
    endif()
    if(SPINGALETT_HAS_VULKAN)
        add_dependencies(spingalett_runtime spingalett_shaders)
        target_sources(spingalett_runtime PRIVATE ${gpu_dir}/Spingalett.Vulkan.c)
        target_include_directories(spingalett_runtime PRIVATE ${PROJECT_BINARY_DIR}/Gpu ${SPINGALETT_VULKAN_INCLUDE})
    endif()
    if(SPINGALETT_HAS_CUDA)
        add_dependencies(spingalett_runtime spingalett_cuda_kernels)
        target_sources(spingalett_runtime PRIVATE ${gpu_dir}/Spingalett.Cuda.c ${gpu_dir}/Spingalett.Gunzip.c)
        target_include_directories(spingalett_runtime PRIVATE ${PROJECT_BINARY_DIR}/Cuda)
    endif()
    message(STATUS "Spingalett: the runtime with the GPU's backends (SPINGALETT_RUNTIME_GPU)")
endfunction()

function(spingalett_runtime_examples)
    file(GLOB sources ${PROJECT_SOURCE_DIR}/Examples/Runtime/*.c)
    foreach(source ${sources})
        get_filename_component(name ${source} NAME_WE)
        add_executable(${name} ${source})
        target_link_libraries(${name} PRIVATE spingalett_runtime)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        if(APPLE)
            set_target_properties(${name} PROPERTIES INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
        elseif(UNIX)
            set_target_properties(${name} PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
        endif()
    endforeach()
    install(TARGETS RunModel RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endfunction()

# The runtime's tests: its API, the models of Tests/Data/runtime (written by earlier releases), and its
# exports (Tests/check_abi.py). With `exporter`, the full library's test program, also every kind of
# model it writes, whose results from the runtime must be the full library's bit for bit.
function(spingalett_runtime_tests exporter)
    add_executable(SpingalettRuntimeTests ${PROJECT_SOURCE_DIR}/Tests/Spingalett.RuntimeTests.c)
    target_link_libraries(SpingalettRuntimeTests PRIVATE spingalett_runtime)
    # the oldest C the runtime's header serves
    set_target_properties(SpingalettRuntimeTests PROPERTIES C_STANDARD 99 C_EXTENSIONS OFF)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(SpingalettRuntimeTests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(SpingalettRuntimeTests PRIVATE m)
        set_target_properties(SpingalettRuntimeTests PROPERTIES BUILD_RPATH "$ORIGIN")
    endif()
    add_test(NAME runtime.models COMMAND SpingalettRuntimeTests ${PROJECT_SOURCE_DIR}/Tests/Data)
    # (graphs on the GPU every node its kernels take, however small)
    set_tests_properties(runtime.models PROPERTIES ENVIRONMENT "SPINGALETT_GRAPH_GPU_MIN=0")
    if(exporter)
        set(dir ${PROJECT_BINARY_DIR}/runtime_models)
        file(MAKE_DIRECTORY ${dir})
        add_test(NAME runtime.export COMMAND ${exporter} export-models ${dir})
        add_test(NAME runtime.library COMMAND SpingalettRuntimeTests ${PROJECT_SOURCE_DIR}/Tests/Data ${dir})
        set_tests_properties(runtime.export PROPERTIES FIXTURES_SETUP runtime_models)
        set_tests_properties(runtime.library PROPERTIES FIXTURES_REQUIRED runtime_models)
    endif()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        set(symbols)
        if(UNIX AND NOT APPLE AND CMAKE_NM)
            set(symbols ${CMAKE_NM} $<TARGET_FILE:spingalett_runtime>)
        endif()
        add_test(NAME runtime.abi
                 COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/Tests/check_abi.py runtime ${symbols})
    endif()
endfunction()

# The runtime library and its pkg-config file; with `headers` (a build of the runtime alone), the
# headers it needs: Spingalett.Runtime.h, Spingalett.Inference.h and Spingalett.Config.h.
function(spingalett_runtime_install headers)
    install(TARGETS spingalett_runtime EXPORT SpingalettTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    if(headers)
        install(FILES ${PROJECT_SOURCE_DIR}/Include/Spingalett/Spingalett.Runtime.h
                      ${PROJECT_SOURCE_DIR}/Include/Spingalett/Spingalett.Inference.h
                      ${PROJECT_BINARY_DIR}/Spingalett/Spingalett.Config.h
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Spingalett)
    endif()
    spingalett_install_pkg_config(spingalett-runtime spingalett-runtime "Spingalett runtime"
        "Spingalett's deployment models without training: .slett models from FP32 to INT2 on every core")
endfunction()
