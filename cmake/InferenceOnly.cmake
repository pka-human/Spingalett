# SPINGALETT_INFERENCE_ONLY: the library is the inference engine alone, a static library with
# Spingalett.Inference.h, for microcontrollers and other targets without a heap, files or threads.
include(${CMAKE_CURRENT_LIST_DIR}/SpingalettEngine.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/Package.cmake)

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
spingalett_install_package()
