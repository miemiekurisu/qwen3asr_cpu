cmake_minimum_required(VERSION 3.21)

project(qwen_asr_provider VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QASR_ENABLE_TESTS "Build unit tests" ${BUILD_TESTING})
option(QASR_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(WIN32)
    set(QASR_DEFAULT_CPU_BACKEND OFF)
else()
    set(QASR_DEFAULT_CPU_BACKEND ON)
endif()
option(QASR_ENABLE_CPU_BACKEND "Build the internal CPU inference backend" ${QASR_DEFAULT_CPU_BACKEND})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_library(qasr_core STATIC
    src/core/status.cc
    src/core/audio_types.cc
    src/core/timestamp.cc
    src/core/state_machine.cc
    src/storage/safetensors_loader.cc
    src/model/tokenizer.cc
    src/audio/frontend.cc
    src/inference/encoder.cc
    src/inference/decoder.cc
    src/inference/streaming_policy.cc
    src/service/realtime.cc
    src/runtime/blas.cc
    src/runtime/config.cc
    src/runtime/task.cc
    src/runtime/engine.cc
    src/runtime/model_bridge.cc
    src/runtime/session_manager.cc
    src/runtime/task_queue.cc
    src/runtime/realtime_session.cc
    src/protocol/openai.cc
    src/protocol/vllm.cc
)

target_include_directories(qasr_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(qasr_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(qasr_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(QASR_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(qasr_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(qasr_core PRIVATE -fsanitize=address,undefined)
endif()

include(QasrBlas)
qasr_configure_blas(qasr_core)

if(QASR_ENABLE_CPU_BACKEND AND NOT WIN32)
    find_package(Threads REQUIRED)

    add_library(qasr_cpu_c STATIC
        src/backend/qwen_cpu/qwen_asr.c
        src/backend/qwen_cpu/qwen_asr_stream.h
        src/backend/qwen_cpu/qwen_asr_audio.c
        src/backend/qwen_cpu/qwen_asr_decoder.c
        src/backend/qwen_cpu/qwen_asr_encoder.c
        src/backend/qwen_cpu/qwen_asr_kernels.c
        src/backend/qwen_cpu/qwen_asr_kernels_generic.c
        src/backend/qwen_cpu/qwen_asr_kernels_neon.c
        src/backend/qwen_cpu/qwen_asr_kernels_avx.c
        src/backend/qwen_cpu/qwen_asr_safetensors.c
        src/backend/qwen_cpu/qwen_asr_tokenizer.c
    )

    target_include_directories(qasr_cpu_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/qwen_cpu)
    target_compile_features(qasr_cpu_c PUBLIC c_std_11)

    if(MSVC)
        target_compile_options(qasr_cpu_c PRIVATE /O2)
    else()
        target_compile_options(qasr_cpu_c PRIVATE -O3 -ffast-math -Wall -Wextra)
    endif()

    if(APPLE)
        target_compile_definitions(qasr_cpu_c PRIVATE USE_BLAS ACCELERATE_NEW_LAPACK)
    else()
        target_compile_definitions(qasr_cpu_c PRIVATE USE_BLAS USE_OPENBLAS)
    endif()

    target_compile_definitions(qasr_core PUBLIC QASR_CPU_BACKEND_ENABLED=1)
    target_include_directories(qasr_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/qwen_cpu)
    target_link_libraries(qasr_cpu_c PUBLIC Threads::Threads)
    if(UNIX AND NOT APPLE)
        target_link_libraries(qasr_cpu_c PUBLIC m)
    endif()
    qasr_configure_blas(qasr_cpu_c)
    target_link_libraries(qasr_core PUBLIC qasr_cpu_c)
endif()

add_executable(qasr_cli
    app/qasr_cli.cc
    src/cli/options.cc
)
target_link_libraries(qasr_cli PRIVATE qasr_core)
target_include_directories(qasr_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(qasr_cli PRIVATE /W4 /permissive-)
else()
    target_compile_options(qasr_cli PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(QASR_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(qasr_cli PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(qasr_cli PRIVATE -fsanitize=address,undefined)
endif()

if(QASR_ENABLE_CPU_BACKEND AND NOT WIN32)
    add_executable(qasr_server
        app/qasr_server.cc
        src/service/server.cc
    )
    target_link_libraries(qasr_server PRIVATE qasr_core)
    target_include_directories(qasr_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/third_party
    )

    if(MSVC)
        target_compile_options(qasr_server PRIVATE /W4 /permissive-)
    else()
        target_compile_options(qasr_server PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    if(QASR_ENABLE_ASAN AND NOT MSVC)
        target_compile_options(qasr_server PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(qasr_server PRIVATE -fsanitize=address,undefined)
    endif()
endif()

if(QASR_ENABLE_TESTS)
    enable_testing()

    add_executable(qasr_unit_tests
        src/cli/options.cc
        src/service/server.cc
        tests/test_main.cc
        tests/status_test.cc
        tests/audio_types_test.cc
        tests/timestamp_test.cc
        tests/blas_test.cc
        tests/config_test.cc
        tests/task_test.cc
        tests/engine_test.cc
        tests/model_bridge_test.cc
        tests/cli_options_test.cc
        tests/server_test.cc
        tests/realtime_test.cc
        tests/openai_test.cc
        tests/vllm_test.cc
        tests/state_machine_test.cc
        tests/safetensors_loader_test.cc
        tests/tokenizer_test.cc
        tests/audio_frontend_test.cc
        tests/encoder_test.cc
        tests/decoder_test.cc
        tests/streaming_policy_test.cc
        tests/session_manager_test.cc
        tests/task_queue_test.cc
        tests/realtime_session_test.cc
    )

    target_link_libraries(qasr_unit_tests PRIVATE qasr_core)
    target_include_directories(qasr_unit_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/third_party
    )

    if(MSVC)
        target_compile_options(qasr_unit_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(qasr_unit_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    if(QASR_ENABLE_ASAN AND NOT MSVC)
        target_compile_options(qasr_unit_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(qasr_unit_tests PRIVATE -fsanitize=address,undefined)
    endif()

    add_test(NAME qasr_unit_tests COMMAND qasr_unit_tests)
endif()
