# QasrOneDNN.cmake - Optional oneDNN integration for INT8 decoder acceleration.
#
# Strategy:
#   1. Try to find a system-installed oneDNN (find_package).
#   2. If not found, download the source tarball and use add_subdirectory.
#      (Avoids FetchContent subbuild which hits ninja recompaction bug on Win)
#
# Usage:
#   include(QasrOneDNN)           # at top-level, after project()
#   qasr_configure_onednn(<target>)
#
# Defines USE_ONEDNN on <target> if oneDNN is available.
# The variable QASR_HAS_ONEDNN is set to TRUE/FALSE for downstream logic.

option(QASR_ENABLE_ONEDNN "Enable oneDNN for INT8 decoder acceleration" ON)

# Cache: have we already fetched/found oneDNN in this configure run?
if(NOT DEFINED _QASR_ONEDNN_READY)
    set(_QASR_ONEDNN_READY FALSE)
    set(_QASR_ONEDNN_TARGET "")

    if(QASR_ENABLE_ONEDNN)
        # 1. Try system install first
        find_package(dnnl CONFIG QUIET)
        if(NOT dnnl_FOUND)
            find_package(dnnl QUIET)
        endif()

        if(dnnl_FOUND)
            set(_QASR_ONEDNN_READY TRUE)
            set(_QASR_ONEDNN_TARGET "DNNL::dnnl")
            message(STATUS "oneDNN: using system installation")
        else()
            # 2. Download source tarball and use add_subdirectory directly.
            #    This avoids FetchContent's subbuild which triggers a
            #    "ninja: error: failed recompaction: Permission denied"
            #    bug in CMake 3.27 + Ninja on Windows.
            set(_ONEDNN_VERSION "3.7")
            set(_ONEDNN_URL "https://github.com/oneapi-src/oneDNN/archive/refs/tags/v${_ONEDNN_VERSION}.tar.gz")
            set(_ONEDNN_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/onednn-src")
            set(_ONEDNN_TARBALL "${CMAKE_BINARY_DIR}/_deps/onednn-v${_ONEDNN_VERSION}.tar.gz")

            if(NOT EXISTS "${_ONEDNN_SRC_DIR}/CMakeLists.txt")
                message(STATUS "oneDNN: not found on system, downloading v${_ONEDNN_VERSION} source...")

                if(NOT EXISTS "${_ONEDNN_TARBALL}")
                    file(DOWNLOAD "${_ONEDNN_URL}" "${_ONEDNN_TARBALL}"
                         STATUS _dl_status
                         TIMEOUT 300
                         SHOW_PROGRESS)
                    list(GET _dl_status 0 _dl_code)
                    if(NOT _dl_code EQUAL 0)
                        list(GET _dl_status 1 _dl_msg)
                        message(WARNING "oneDNN: download failed (${_dl_code}: ${_dl_msg})")
                    endif()
                endif()

                if(EXISTS "${_ONEDNN_TARBALL}")
                    message(STATUS "oneDNN: extracting tarball...")
                    file(ARCHIVE_EXTRACT INPUT "${_ONEDNN_TARBALL}"
                         DESTINATION "${CMAKE_BINARY_DIR}/_deps")
                    # Tarball extracts to oneDNN-<version>/
                    set(_EXTRACTED_DIR "${CMAKE_BINARY_DIR}/_deps/oneDNN-${_ONEDNN_VERSION}")
                    if(EXISTS "${_EXTRACTED_DIR}/CMakeLists.txt" AND NOT EXISTS "${_ONEDNN_SRC_DIR}/CMakeLists.txt")
                        file(RENAME "${_EXTRACTED_DIR}" "${_ONEDNN_SRC_DIR}")
                    endif()
                endif()
            endif()

            if(EXISTS "${_ONEDNN_SRC_DIR}/CMakeLists.txt")
                # Minimal build settings: CPU only, no examples/tests
                set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
                set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
                set(DNNL_ENABLE_WORKLOAD INFERENCE CACHE STRING "" FORCE)
                set(DNNL_ENABLE_PRIMITIVE MATMUL CACHE STRING "" FORCE)
                set(DNNL_CPU_RUNTIME SEQ CACHE STRING "" FORCE)
                set(DNNL_GPU_RUNTIME NONE CACHE STRING "" FORCE)
                set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
                set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)

                add_subdirectory("${_ONEDNN_SRC_DIR}"
                                 "${CMAKE_BINARY_DIR}/_deps/onednn-build"
                                 EXCLUDE_FROM_ALL)

                if(TARGET dnnl)
                    set(_QASR_ONEDNN_READY TRUE)
                    set(_QASR_ONEDNN_TARGET "dnnl")
                    message(STATUS "oneDNN: built from source (v${_ONEDNN_VERSION}, static, CPU/SEQ, matmul-only)")
                else()
                    message(WARNING "oneDNN: add_subdirectory completed but 'dnnl' target not found")
                endif()
            else()
                message(WARNING "oneDNN: source not available after download attempt")
            endif()
        endif()
    else()
        message(STATUS "oneDNN: disabled by QASR_ENABLE_ONEDNN=OFF")
    endif()
endif()

function(qasr_configure_onednn target_name)
    set(QASR_HAS_ONEDNN FALSE PARENT_SCOPE)

    if(_QASR_ONEDNN_READY AND _QASR_ONEDNN_TARGET)
        target_link_libraries(${target_name} PUBLIC ${_QASR_ONEDNN_TARGET})
        target_compile_definitions(${target_name} PRIVATE USE_ONEDNN=1)
        set(QASR_HAS_ONEDNN TRUE PARENT_SCOPE)
        message(STATUS "oneDNN: INT8 decoder acceleration enabled for ${target_name}")
    else()
        message(STATUS "oneDNN: not available, ${target_name} will use BF16/OpenBLAS fallback")
    endif()
endfunction()
