cmake_minimum_required(VERSION 3.21)

project(qwen3asr_cpu VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QASR_ENABLE_TESTS "Build unit tests" ${BUILD_TESTING})
option(QASR_ENABLE_ASAN "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
set(QASR_DEFAULT_CPU_BACKEND ON)
option(QASR_ENABLE_CPU_BACKEND "Build the internal CPU inference backend" ${QASR_DEFAULT_CPU_BACKEND})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_library(qasr_core STATIC
    src/core/status.cc
    src/core/audio_types.cc
    src/core/timestamp.cc
    src/core/state_machine.cc
    src/core/inference_arena.cc
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
    src/runtime/job_pool.cc
    src/runtime/realtime_session.cc
    src/protocol/openai.cc
    src/protocol/vllm.cc
    src/base/json.cc
    src/base/http_server.cc
    src/base/utf8.cc
    src/base/process_spawn.cc
    src/audio/audio_convert.cc
    src/subtitle/subtitle_writer.cc
    src/inference/aligner_types.cc
    src/inference/aligner_client.cc
)

# Engine v2 sources (GPU branch)
set(QASR_ENGINE_V2_SOURCES
    src/engine/types.cc
    src/engine/config.cc
    src/engine/asr_engine.cc
    src/engine/qwen_model.cc
    src/engine/qwen_session.cc
    src/engine/cpu_asr_engine.cc
    src/engine/cpu_asr_executor.cc
    src/engine/cuda_asr_engine.cc
    src/engine/perf_stats.cc
    src/backend/device_backend.cc
    src/backend/cpu_backend.cc
    src/backend/cuda_backend.cc
    src/scheduler/scheduler.cc
    src/backend/audio_segmentation.c
)
target_sources(qasr_core PRIVATE ${QASR_ENGINE_V2_SOURCES})

target_include_directories(qasr_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(qasr_core PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(qasr_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(QASR_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(qasr_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(qasr_core PRIVATE -fsanitize=address,undefined)
endif()

if(WIN32)
    target_link_libraries(qasr_core PUBLIC ws2_32)
endif()

include(QasrBlas)
qasr_configure_blas(qasr_core)

include(QasrOneDNN)

# Shared backend include path (audio_segmentation.h etc.)
target_include_directories(qasr_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend)

# libcurl for translation client
find_package(CURL QUIET)
if(CURL_FOUND)
    message(STATUS "libcurl: found (${CURL_INCLUDE_DIRS})")
    target_include_directories(qasr_core PRIVATE ${CURL_INCLUDE_DIRS})
    # server.cc (compiled in qasr_server and qasr_unit_tests) also includes curl.
    # Use a variable so both targets get the include path.
    set(QASR_CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIRS})
    target_link_libraries(qasr_core PRIVATE ${CURL_LIBRARIES})
    target_compile_definitions(qasr_core PUBLIC QASR_CURL_AVAILABLE=1)
    target_sources(qasr_core PRIVATE src/engine/translation.cc)
else()
    message(WARNING "libcurl not found; translation client disabled")
endif()

if(QASR_ENABLE_CPU_BACKEND)
    if(NOT WIN32)
        find_package(Threads REQUIRED)
    endif()

    set(QASR_CPU_SOURCES
        src/backend/qwen_cpu/qwen_asr.c
        src/backend/qwen_cpu/qwen_asr_stream.h
        src/backend/qwen_cpu/qwen_asr_alloc.c
        src/backend/qwen_cpu/qwen_asr_alloc.h
        src/backend/qwen_cpu/qwen_asr_audio.c
        src/backend/qwen_cpu/qwen_asr_decoder.c
        src/backend/qwen_cpu/qwen_asr_encoder.c
        src/backend/qwen_cpu/qwen_asr_kernels.c
        src/backend/qwen_cpu/qwen_asr_kernels_generic.c
        src/backend/qwen_cpu/qwen_asr_kernels_avx.c
        src/backend/qwen_cpu/qwen_asr_perf.c
        src/backend/qwen_cpu/qwen_asr_onednn.c
        src/backend/qwen_cpu/qwen_asr_safetensors.c
        src/backend/qwen_cpu/qwen_asr_tokenizer.c
        src/backend/qwen_cpu/qwen_asr_aligner.c
        src/backend/qwen_cpu/qwen_asr_utf8.c
        src/backend/qwen_cpu/qwen_silero_vad.c
        src/backend/qwen_cpu/qwen_silero_vad.h
    )

    # NEON kernels only on ARM
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
        list(APPEND QASR_CPU_SOURCES src/backend/qwen_cpu/qwen_asr_kernels_neon.c)
    endif()

    add_library(qasr_cpu_c STATIC ${QASR_CPU_SOURCES})

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86|X86")
        target_compile_definitions(qasr_cpu_c PRIVATE QWEN_X86_AVX2_AVAILABLE=1)
        if(MSVC)
            set_source_files_properties(
                src/backend/qwen_cpu/qwen_asr_kernels_avx.c
                PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        else()
            set_source_files_properties(
                src/backend/qwen_cpu/qwen_asr_kernels_avx.c
                PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
        endif()
    endif()

    target_include_directories(qasr_cpu_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/qwen_cpu)
    target_compile_features(qasr_cpu_c PUBLIC c_std_11)

    # Optional Silero VAD support via ONNX runtime.  Disabled by default
    # so the build is hermetic; enable with -DQASR_ENABLE_SILERO_VAD=ON
    # and ensure onnxruntime is on the search path (or set
    # QASR_ONNXRUNTIME_ROOT to its install dir).  When disabled, the
    # VAD code compiles to a no-op stub and legacy timeout-based
    # silence detection is used instead.
    option(QASR_ENABLE_SILERO_VAD "Enable Silero VAD (ONNX runtime) for streaming endpoint detection" OFF)
    set(QASR_SILERO_VAD_AVAILABLE 0)
    if(QASR_ENABLE_SILERO_VAD)
        if(QASR_ONNXRUNTIME_ROOT)
            find_path(QASR_ONNXRUNTIME_INCLUDE_DIR
                NAMES onnxruntime_c_api.h
                PATHS ${QASR_ONNXRUNTIME_ROOT} ${QASR_ONNXRUNTIME_ROOT}/include
                NO_DEFAULT_PATH)
            find_library(QASR_ONNXRUNTIME_LIBRARY
                NAMES onnxruntime
                PATHS ${QASR_ONNXRUNTIME_ROOT} ${QASR_ONNXRUNTIME_ROOT}/lib ${QASR_ONNXRUNTIME_ROOT}/lib64
                NO_DEFAULT_PATH)
        else()
            find_path(QASR_ONNXRUNTIME_INCLUDE_DIR
                NAMES onnxruntime_c_api.h)
            find_library(QASR_ONNXRUNTIME_LIBRARY
                NAMES onnxruntime)
        endif()
        if(QASR_ONNXRUNTIME_INCLUDE_DIR AND QASR_ONNXRUNTIME_LIBRARY)
            target_include_directories(qasr_cpu_c PRIVATE ${QASR_ONNXRUNTIME_INCLUDE_DIR})
            target_link_libraries(qasr_cpu_c PRIVATE ${QASR_ONNXRUNTIME_LIBRARY})
            target_compile_definitions(qasr_cpu_c PRIVATE QWEN_HAS_ONNXRUNTIME=1)
            set(QASR_SILERO_VAD_AVAILABLE 1)
            message(STATUS "Silero VAD: ONNX runtime found at ${QASR_ONNXRUNTIME_INCLUDE_DIR}")
        else()
            message(WARNING "Silero VAD requested but ONNX runtime not found; building stub")
        endif()
    endif()

    if(MSVC)
        # /O2 conflicts with /RTC1 and /Od that CMake adds in Debug mode.
        # Strip them for this perf-critical C library (the only C target).
        string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
        string(REPLACE "/Od" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
        target_compile_options(qasr_cpu_c PRIVATE /O2)
    else()
        target_compile_options(qasr_cpu_c PRIVATE -O3 -ffast-math -Wall -Wextra)
    endif()

    if(APPLE)
        target_compile_definitions(qasr_cpu_c PRIVATE USE_BLAS)
    else()
        target_compile_definitions(qasr_cpu_c PRIVATE USE_BLAS)
        # Backend-specific thread-control hooks: kernels.c and perf.c
        # call openblas_{get,set}_num_threads under this guard.
        if(QASR_BLAS_SELECTED STREQUAL "openblas")
            target_compile_definitions(qasr_cpu_c PRIVATE USE_OPENBLAS)
        endif()
    endif()

    qasr_configure_onednn(qasr_cpu_c)
    if(QASR_HAS_ONEDNN)
        target_compile_definitions(qasr_core PUBLIC QASR_ONEDNN_AVAILABLE=1)
    endif()

    target_compile_definitions(qasr_core PUBLIC QASR_CPU_BACKEND_ENABLED=1)
    target_include_directories(qasr_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/qwen_cpu)
    target_include_directories(qasr_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend)
    if(WIN32)
        # Win32 threading API + memory-mapped I/O are in kernel32 (linked by default)
    else()
        target_link_libraries(qasr_cpu_c PUBLIC Threads::Threads)
    endif()
    if(UNIX AND NOT APPLE)
        target_link_libraries(qasr_cpu_c PUBLIC m)
    endif()
    qasr_configure_blas(qasr_cpu_c)
    target_link_libraries(qasr_core PUBLIC qasr_cpu_c)

    add_executable(qasr_cpu_bench
        app/qasr_cpu_bench.cc
    )
    target_link_libraries(qasr_cpu_bench PRIVATE qasr_cpu_c)
    target_include_directories(qasr_cpu_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/qwen_cpu)

    if(MSVC)
        target_compile_options(qasr_cpu_bench PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(qasr_cpu_bench PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    if(QASR_ENABLE_ASAN AND NOT MSVC)
        target_compile_options(qasr_cpu_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(qasr_cpu_bench PRIVATE -fsanitize=address,undefined)
    endif()
endif()

# ─── CUDA Backend (Engine v2) ───
option(QASR_ENABLE_CUDA_BACKEND "Enable CUDA backend for Engine v2" OFF)
set(QASR_CUDA_PLATFORM "generic" CACHE STRING "CUDA target platform: generic, dgx_spark")
set(QASR_CUDA_BACKEND_AVAILABLE 0)

if(QASR_ENABLE_CUDA_BACKEND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(QASR_CUDA_PLATFORM STREQUAL "dgx_spark")
            set(CMAKE_CUDA_ARCHITECTURES "121")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "native")
        endif()
    endif()

    find_library(QASR_CUBLAS_LIBRARY cublas PATHS /usr/local/cuda/lib64 /usr/local/cuda-13.0/lib64)
    find_path(QASR_CUDA_INCLUDE_DIR cuda_runtime.h PATHS /usr/local/cuda/include /usr/local/cuda-13.0/include)

    if(QASR_CUBLAS_LIBRARY AND QASR_CUDA_INCLUDE_DIR)
        set(QASR_CUDA_BACKEND_AVAILABLE 1)
        message(STATUS "CUDA: platform=${QASR_CUDA_PLATFORM} arch=${CMAKE_CUDA_ARCHITECTURES}")
        message(STATUS "CUDA: cublas=${QASR_CUBLAS_LIBRARY} include=${QASR_CUDA_INCLUDE_DIR}")

        target_include_directories(qasr_core PUBLIC ${QASR_CUDA_INCLUDE_DIR})
        target_link_libraries(qasr_core PRIVATE ${QASR_CUBLAS_LIBRARY})
        target_compile_definitions(qasr_core PUBLIC QASR_CUDA_BACKEND_ENABLED=1)

        # CUDA kernel sources (compiled by nvcc)
        set(QASR_CUDA_KERNEL_SOURCES
            src/backend/rms_norm.cu
            src/backend/rope.cu
            src/backend/swiglu.cu
            src/backend/argmax.cu
            src/backend/cublas_gemm.cu
            src/backend/elementwise.cu
            src/backend/attention.cu
            src/backend/embedding.cu
            src/backend/layer_norm.cu
            src/backend/bidir_attention.cu
            src/backend/gelu.cu
            src/backend/fp32_transpose.cu
            src/backend/broadcast_add.cu
            src/backend/gemv.cu
            src/backend/conv2d.cu
            src/backend/reshape.cu
            src/backend/sinusoidal_pe.cu
            src/backend/cuda_graph.cu
            src/backend/decoder_fusion.cu
        )
        target_sources(qasr_core PRIVATE ${QASR_CUDA_KERNEL_SOURCES})

        if(QASR_CUDA_PLATFORM STREQUAL "dgx_spark")
            target_compile_definitions(qasr_core PRIVATE QASR_CUDA_PLATFORM_DGX_SPARK)
        endif()
    else()
        message(WARNING "CUDA backend requested but cuBLAS not found; CUDA disabled")
    endif()
endif()

add_executable(qasr_cli
    app/qasr_cli.cc
    src/cli/options.cc
)
target_link_libraries(qasr_cli PRIVATE qasr_core)
target_include_directories(qasr_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(qasr_cli PRIVATE /W4 /permissive- /utf-8)
    target_link_options(qasr_cli PRIVATE /DELAYLOAD:libopenblas.dll)
    target_link_libraries(qasr_cli PRIVATE delayimp)
else()
    target_compile_options(qasr_cli PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(QASR_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(qasr_cli PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(qasr_cli PRIVATE -fsanitize=address,undefined)
endif()

# qasr_v2_test — Engine v2 CPU/CUDA comparison test
add_executable(qasr_v2_test
    app/qasr_v2_test.cc
)
target_link_libraries(qasr_v2_test PRIVATE qasr_core)
target_include_directories(qasr_v2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
    target_compile_options(qasr_v2_test PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(qasr_v2_test PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

# qasr_blas_bench — real-time-stream BLAS benchmark
# Compares OpenBLAS / BLIS / MKL throughput on the project's actual hot path
# (qwen_transcribe_stream_live over a real long audio file).  See app/README.
add_executable(qasr_blas_bench
    app/qasr_blas_bench.cc
)
target_link_libraries(qasr_blas_bench PRIVATE qasr_core)
target_include_directories(qasr_blas_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
qasr_configure_blas_bench_tag(qasr_blas_bench)
if(MSVC)
    target_compile_options(qasr_blas_bench PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(qasr_blas_bench PRIVATE -Wall -Wextra -Wpedantic)
endif()
if(QASR_ENABLE_ASAN AND NOT MSVC)
    target_compile_options(qasr_blas_bench PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(qasr_blas_bench PRIVATE -fsanitize=address,undefined)
endif()

if(QASR_ENABLE_CPU_BACKEND OR WIN32)
    add_executable(qasr_server
        app/qasr_server.cc
        src/service/server.cc
    )
    target_link_libraries(qasr_server PRIVATE qasr_core)
    target_include_directories(qasr_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${QASR_CURL_INCLUDE_DIRS}
    )

    if(MSVC)
        target_compile_options(qasr_server PRIVATE /W4 /permissive- /utf-8)
        target_link_options(qasr_server PRIVATE /DELAYLOAD:libopenblas.dll)
        target_link_libraries(qasr_server PRIVATE delayimp)
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

    if(APPLE)
        find_program(QASR_OSASCRIPT_EXECUTABLE osascript)
    endif()

    set(QASR_UNIT_TEST_SOURCES
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
        tests/extract_indexed_safetensors_test.cc
        tests/cli_options_test.cc
        tests/server_test.cc
        tests/silero_vad_test.cc
        tests/utf8_truncate_test.cc
        tests/realtime_test.cc
        tests/openai_test.cc
        tests/vllm_test.cc
        tests/state_machine_test.cc
        tests/safetensors_loader_test.cc
        tests/safetensors_safety_test.cc
        tests/utf8_test.cc
        tests/qasr_alloc_test.cc
        tests/qasr_safety_microbench.cc
        tests/tokenizer_test.cc
        tests/audio_frontend_test.cc
        tests/encoder_test.cc
        tests/decoder_test.cc
        tests/streaming_policy_test.cc
        tests/session_manager_test.cc
        tests/task_queue_test.cc
        tests/job_pool_test.cc
        tests/realtime_session_test.cc
        tests/json_test.cc
        tests/http_server_test.cc
        tests/realfile_audio_test.cc
        tests/silence_gap_test.cc
        tests/inference_arena_test.cc
        tests/inference_backend_test.cc
        tests/subtitle_writer_test.cc
        tests/audio_convert_test.cc
        tests/ffmpeg_argv_test.cc
        tests/aligner_types_test.cc
        tests/aligner_client_test.cc
        tests/chunk_callback_test.cc
        tests/qwen_clone_shared_test.cc
        tests/audit_bug_replay_scheduler_race_test.cc
        tests/audit_bug_replay_integer_overflow_test.cc
        tests/audit_bug_replay_c_alloc_test.cc
        tests/audit_bug_replay_gpu_diagnose_test.cc
        tests/sentence_boundary_test.cc
    )

    if(QASR_ENABLE_CPU_BACKEND)
        list(APPEND QASR_UNIT_TEST_SOURCES tests/qwen_cpu_kernels_test.cc)
        list(APPEND QASR_UNIT_TEST_SOURCES tests/onednn_int8_test.cc)
        list(APPEND QASR_UNIT_TEST_SOURCES tests/engine_v2_test.cc)
        list(APPEND QASR_UNIT_TEST_SOURCES tests/session_test.cc)
        list(APPEND QASR_UNIT_TEST_SOURCES tests/cpu_concurrent_session_test.cc)
        list(APPEND QASR_UNIT_TEST_SOURCES tests/scheduler_test.cc)
        list(APPEND QASR_UNIT_TEST_SOURCES tests/cuda_backend_test.cc)
        list(APPEND QASR_UNIT_TEST_SOURCES tests/multi_session_analysis_test.cc)
    endif()

    add_executable(qasr_unit_tests ${QASR_UNIT_TEST_SOURCES})

    target_link_libraries(qasr_unit_tests PRIVATE qasr_core qasr_cpu_c)
    target_include_directories(qasr_unit_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${QASR_CURL_INCLUDE_DIRS}
    )
    target_compile_definitions(qasr_unit_tests PRIVATE
        QASR_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    if(MSVC)
        target_compile_options(qasr_unit_tests PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(qasr_unit_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()

    if(QASR_ENABLE_ASAN AND NOT MSVC)
        target_compile_options(qasr_unit_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(qasr_unit_tests PRIVATE -fsanitize=address,undefined)
    endif()

    add_test(NAME qasr_unit_tests COMMAND qasr_unit_tests)
    set_tests_properties(qasr_unit_tests PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
