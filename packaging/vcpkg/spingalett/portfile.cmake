# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# An overlay port of the source tree it is in:
#
#     vcpkg install spingalett --overlay-ports=<checkout>/packaging/vcpkg
#     vcpkg install "spingalett[vulkan]" --overlay-ports=<checkout>/packaging/vcpkg
#
# A port for vcpkg's registry takes a release instead of the checkout:
#     vcpkg_from_github(OUT_SOURCE_PATH SOURCE_PATH REPO pka-human/Spingalett REF "v${VERSION}"
#                       SHA512 <the archive's> HEAD_REF main)
get_filename_component(SOURCE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# The library is a shared one: its ABI is kept within a major version (Src/Spingalett.map).
vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        openmp BUILD_WITH_OPENMP
)
set(vulkan_options -DSPINGALETT_VULKAN=OFF)
if("vulkan" IN_LIST FEATURES)
    set(vulkan_options -DSPINGALETT_VULKAN=ON
        "-DSPINGALETT_GLSLC=${CURRENT_HOST_INSTALLED_DIR}/tools/shaderc/glslc${VCPKG_HOST_EXECUTABLE_SUFFIX}")
endif()

# Kernels chosen at run time (no -march=native); the outputs of each configuration in its own build
# tree (by default they go to Bin/ and Lib/ of the sources).
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        ${vulkan_options}
        -DSPINGALETT_NATIVE_ARCH=OFF
        -DBUILD_EXAMPLE=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_APPS=OFF
    OPTIONS_RELEASE
        "-DSPINGALETT_BIN_DIR=${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/Bin"
        "-DSPINGALETT_LIB_DIR=${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/Lib"
    OPTIONS_DEBUG
        "-DSPINGALETT_BIN_DIR=${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-dbg/Bin"
        "-DSPINGALETT_LIB_DIR=${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-dbg/Lib"
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME spingalett CONFIG_PATH lib/cmake/Spingalett)
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
