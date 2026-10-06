# SPINGALETT_INFERENCE_ONLY: the library is the inference engine alone, a static library with
# Spingalett.Inference.h, for microcontrollers and other targets without a heap, files or threads.
include(${CMAKE_CURRENT_LIST_DIR}/SpingalettEngine.cmake)
include(CMakePackageConfigHelpers)

spingalett_engine_library(spingalett)
add_library(Spingalett::spingalett ALIAS spingalett)

option(BUILD_TESTS "Build the test suite" ON)
if(BUILD_TESTS)
    enable_testing()
    spingalett_engine_tests(spingalett "")
endif()

install(TARGETS spingalett EXPORT SpingalettTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${PROJECT_SOURCE_DIR}/Include/Spingalett/Spingalett.Inference.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Spingalett)
set(SPINGALETT_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/Spingalett)
install(EXPORT SpingalettTargets NAMESPACE Spingalett:: DESTINATION ${SPINGALETT_CMAKE_DIR})
configure_package_config_file(${PROJECT_SOURCE_DIR}/cmake/SpingalettConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SpingalettConfig.cmake INSTALL_DESTINATION ${SPINGALETT_CMAKE_DIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/SpingalettConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SpingalettConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/SpingalettConfigVersion.cmake
        DESTINATION ${SPINGALETT_CMAKE_DIR})
