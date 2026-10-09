# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# The GPU backends of the library: Vulkan compute (cmake/Vulkan.cmake) and CUDA (cmake/Cuda.cmake), each
# where its compiler is found, under one executor (Spingalett.Gpu.c) and device interface
# (Spingalett.Device.c). Both open their drivers at run time.

include(${CMAKE_CURRENT_LIST_DIR}/Vulkan.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Cuda.cmake)

set(gpu_dir ${CMAKE_CURRENT_SOURCE_DIR}/Src/Gpu)
if(SPINGALETT_HAS_VULKAN OR SPINGALETT_HAS_CUDA)
    target_sources(spingalett PRIVATE ${gpu_dir}/Spingalett.Device.c ${gpu_dir}/Spingalett.GpuKernels.c
                                      ${gpu_dir}/Spingalett.Gpu.c)
    target_include_directories(spingalett PRIVATE ${gpu_dir})
    if(NOT WIN32)
        target_link_libraries(spingalett PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()
if(SPINGALETT_HAS_VULKAN)
    add_dependencies(spingalett spingalett_shaders)
    target_sources(spingalett PRIVATE ${gpu_dir}/Spingalett.Vulkan.c)
    set_source_files_properties(${gpu_dir}/Spingalett.Vulkan.c PROPERTIES OBJECT_DEPENDS "${spirv_files}")
    target_include_directories(spingalett PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Gpu ${SPINGALETT_VULKAN_INCLUDE})
endif()
if(SPINGALETT_HAS_CUDA)
    add_dependencies(spingalett spingalett_cuda_kernels)
    target_sources(spingalett PRIVATE ${gpu_dir}/Spingalett.Cuda.c)
    set_source_files_properties(${gpu_dir}/Spingalett.Cuda.c PROPERTIES OBJECT_DEPENDS "${ptx_files}")
    target_include_directories(spingalett PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/Cuda)
endif()
