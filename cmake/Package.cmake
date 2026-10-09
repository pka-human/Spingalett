# SPDX-License-Identifier: MIT
# Copyright (c) 2026 pka_human (pka_human@proton.me)
#
# What every kind of build installs for the programs that use it: the CMake package
# (find_package(Spingalett): SpingalettConfig.cmake over the export set SpingalettTargets) and
# pkg-config files.
include(CMakePackageConfigHelpers)

function(spingalett_install_package)
    set(dir ${CMAKE_INSTALL_LIBDIR}/cmake/Spingalett)
    install(EXPORT SpingalettTargets NAMESPACE Spingalett:: DESTINATION ${dir})
    configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/SpingalettConfig.cmake.in
        ${PROJECT_BINARY_DIR}/SpingalettConfig.cmake INSTALL_DESTINATION ${dir})
    # 1.x keeps the API and ABI of 1.0, so any 1.x of at least the version asked for will do (before
    # 1.0 a minor release could change them, and only the same major.minor was compatible).
    if(PROJECT_VERSION_MAJOR EQUAL 0)
        set(compatibility SameMinorVersion)
    else()
        set(compatibility SameMajorVersion)
    endif()
    write_basic_package_version_file(${PROJECT_BINARY_DIR}/SpingalettConfigVersion.cmake
        COMPATIBILITY ${compatibility})
    install(FILES ${PROJECT_BINARY_DIR}/SpingalettConfig.cmake ${PROJECT_BINARY_DIR}/SpingalettConfigVersion.cmake
            DESTINATION ${dir})
endfunction()

# NAME.pc for the library `library` (-l), relocatable: the prefix is found from the file's own
# directory, so that unpacked archives work where they are.
function(spingalett_install_pkg_config name library title description)
    set(pc_dir ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
    cmake_path(ABSOLUTE_PATH pc_dir BASE_DIRECTORY ${CMAKE_INSTALL_PREFIX} OUTPUT_VARIABLE absolute)
    file(RELATIVE_PATH SPINGALETT_PC_PREFIX ${absolute} ${CMAKE_INSTALL_PREFIX})
    string(REGEX REPLACE "/$" "" SPINGALETT_PC_PREFIX "${SPINGALETT_PC_PREFIX}")
    foreach(dir LIBDIR INCLUDEDIR)
        if(IS_ABSOLUTE "${CMAKE_INSTALL_${dir}}")
            set(SPINGALETT_PC_${dir} "${CMAKE_INSTALL_${dir}}")
        else()
            set(SPINGALETT_PC_${dir} "\${prefix}/${CMAKE_INSTALL_${dir}}")
        endif()
    endforeach()
    set(SPINGALETT_PC_NAME "${title}")
    set(SPINGALETT_PC_DESCRIPTION "${description}")
    set(SPINGALETT_PC_LIBRARY ${library})
    configure_file(${PROJECT_SOURCE_DIR}/cmake/spingalett.pc.in ${PROJECT_BINARY_DIR}/${name}.pc @ONLY)
    install(FILES ${PROJECT_BINARY_DIR}/${name}.pc DESTINATION ${pc_dir})
endfunction()
