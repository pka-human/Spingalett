# The standalone inference engine (Include/Spingalett/Spingalett.Inference.h): Src/Spingalett.Inference.c
# compiled with SPINGALETT_INFERENCE_ONLY, which needs no heap, stdio, threads or generated headers.

function(spingalett_engine_library name)
    add_library(${name} STATIC ${PROJECT_SOURCE_DIR}/Src/Spingalett.Inference.c)
    target_compile_definitions(${name} PUBLIC SPINGALETT_INFERENCE_ONLY)
    target_include_directories(${name}
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE ${PROJECT_SOURCE_DIR}/Src)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(${name} PUBLIC m)
    endif()
endfunction()

# Tests of the engine on its own. With `exporter` (the full library's test program) they also run
# models that it exports as C headers at build time.
function(spingalett_engine_tests engine exporter)
    add_executable(SpingalettEngineTests ${PROJECT_SOURCE_DIR}/Tests/Spingalett.EngineTests.c)
    target_link_libraries(SpingalettEngineTests PRIVATE ${engine})
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(SpingalettEngineTests PRIVATE -ffp-contract=off)
    endif()
    if(exporter AND NOT CMAKE_CROSSCOMPILING)
        set(dir ${CMAKE_CURRENT_BINARY_DIR}/test_headers)
        set(headers ${dir}/test_model_int8.h ${dir}/test_model_int4.h ${dir}/test_model_fp16.h
                    ${dir}/test_model_conv_int8.h ${dir}/test_model_conv_f32.h ${dir}/test_model_norm_int8.h
                    ${dir}/test_model_expected.h)
        add_custom_command(OUTPUT ${headers}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${dir}
            COMMAND ${exporter} export-headers ${dir}
            DEPENDS ${exporter}
            COMMENT "Exporting the test models as C headers")
        target_sources(SpingalettEngineTests PRIVATE ${headers})
        target_include_directories(SpingalettEngineTests PRIVATE ${dir})
        target_compile_definitions(SpingalettEngineTests PRIVATE SPINGALETT_TEST_HEADERS)
    endif()
    add_test(NAME model.engine COMMAND SpingalettEngineTests)
    if(CMAKE_NM AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" AND NOT WIN32)
        add_test(NAME model.engine_symbols
                 COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} -DLIBRARY=$<TARGET_FILE:${engine}>
                         -P ${PROJECT_SOURCE_DIR}/cmake/CheckEngineSymbols.cmake)
    endif()
endfunction()
