# qasr BLAS backend selection
#
# Default: auto-detect.  Priority order (chosen by SIMD-feature & licence, NOT
# by any specific CPU model — see QASR_BENCH_NOTES at the bottom for how to
# override on a per-machine basis):
#   1. OpenBLAS (BSD-3)  ── default; broad SIMD coverage (SSE/AVX/AVX2/
#                          AVX-VNNI on x86, NEON/SVE on AArch64)
#                          apt: libopenblas-dev  /  source build at $QASR_DEPS_DIR
#   2. BLIS     (BSD-3)  ── drop-in alternative; strongest on AMD Zen and on
#                          Intel CPUs that expose AVX-512 (Xeon, older HEDT)
#                          apt: libblis-openmp-dev  /  source build at $HOME/.local/blis
#   3. MKL      (Intel Simplified Software License — proprietary)  ── last
#                          proprietary option; free for personal use; ships
#                          binary-only kernels; usually the fastest on Intel
#                          but not portable
#   4. Reference  ── netlib / pkg-config (slow fallback, no SIMD)
#
# Why OpenBLAS first, not BLIS, even though both are BSD-3:
#   * OpenBLAS explicitly added AVX-VNNI kernels (>=0.3.20).  That covers all
#     Intel chips with VNNI: Alder Lake (12th gen) onwards on the consumer
#     side, and Sapphire Rapids / Granite Rapids on the server side.  The
#     "VNNI present, AVX-512 absent" combination is exactly the modern
#     consumer-Intel sweet spot, and OpenBLAS is well tuned for it.
#   * BLIS is well tuned for AMD Zen (AMD engineers contribute to it) and
#     for AVX-512 Intel silicon.  On a CPU that lacks AVX-512 the BLIS
#     advantage is much smaller; on some AVX2-only Intel parts OpenBLAS
#     is faster.  Hence BLIS is a peer candidate, not the default.
#   * MKL is the fastest on Intel in many workloads, but is proprietary and
#     drops to "fallback" status by policy: we never want a default build
#     to silently pull in a non-redistributable binary.  Users who want it
#     opt in explicitly with -DQASR_BLAS=mkl.
#
# Override via -DQASR_BLAS=openblas|blis|mkl|auto|ref
#
# Strategy:
#   - If the user pinned a backend with -DQASR_BLAS=<name>, require it (or fail
#     loudly so they know to install it).
#   - Otherwise, walk the priority list in order; the first that finds both
#     headers and a library wins.
#
# QASR_BENCH_NOTES: hardware-specific tuning is intentionally out of scope for
# this file.  To pick the fastest BLAS for a particular machine, run
# `scripts/deprecated/blas_bench.py` (emits sgemm throughput across the realistic matmul
# sizes used by qwen_asr) and then set QASR_BLAS accordingly in your
# CMakePresets.json.  Do not encode a specific CPU model into CMake defaults.
#
# Output (per target):
#   target_compile_definitions(<tgt> PUBLIC QASR_BLAS_<NAME>=1)
#   target_link_libraries(<tgt>     PUBLIC <resolved BLAS>)

include_guard(GLOBAL)

# Cache the chosen backend name in QASR_BLAS_SELECTED so other parts of the
# build (e.g. USE_OPENBLAS thread hooks in qwen_asr_kernels.c) can branch on
# it.  We also set QASR_BLAS_<NAME> as a compile definition on the target.
set(QASR_BLAS_SELECTED "" CACHE STRING "BLAS backend actually selected" FORCE)

function(qasr_configure_blas target_name)
    if(APPLE)
        find_library(QASR_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        target_compile_definitions(${target_name} PUBLIC QASR_BLAS_ACCELERATE=1 ACCELERATE_NEW_LAPACK)
        target_link_libraries(${target_name} PUBLIC ${QASR_ACCELERATE_FRAMEWORK})
        return()
    endif()

    set(_backend "${QASR_BLAS}")

    # ------------------------------------------------------------------
    # 1. OpenBLAS (BSD-3)  ── project default
    # ------------------------------------------------------------------
    if(_backend STREQUAL "" OR _backend STREQUAL "auto" OR _backend STREQUAL "openblas")
        # Convenience: in default/auto/openblas modes, also probe the
        # project-preferred install path (controlled by QASR_DEPS_DIR, default
        # /opt/qasr-deps) so a source-built OpenBLAS works without
        # -DOpenBLAS_DIR=.
        if((_backend STREQUAL "auto" OR _backend STREQUAL ""
                OR _backend STREQUAL "openblas")
                AND NOT DEFINED OpenBLAS_DIR
                AND NOT DEFINED ENV{OpenBLAS_DIR})
            if(DEFINED QASR_DEPS_DIR)
                set(_deps_dir "${QASR_DEPS_DIR}")
            elseif(DEFINED ENV{QASR_DEPS_DIR})
                set(_deps_dir "$ENV{QASR_DEPS_DIR}")
            else()
                set(_deps_dir "/opt/qasr-deps")
            endif()
            if(EXISTS "${_deps_dir}/lib/cmake/openblas/OpenBLASConfig.cmake"
                    OR EXISTS "${_deps_dir}/lib/cmake/openblas/openBLASConfig.cmake")
                set(OpenBLAS_DIR "${_deps_dir}/lib/cmake/openblas" CACHE PATH "")
            endif()
        endif()
        find_package(OpenBLAS QUIET)
        if(OpenBLAS_FOUND)
            if(TARGET OpenBLAS::OpenBLAS)
                target_link_libraries(${target_name} PUBLIC OpenBLAS::OpenBLAS)
                target_compile_definitions(${target_name} PUBLIC QASR_BLAS_OPENBLAS=1)
                if(_backend STREQUAL "openblas")
                    message(STATUS "qasr BLAS: pinned to OpenBLAS (CMake target)")
                else()
                    message(STATUS "qasr BLAS: selected OpenBLAS (default, BSD-3)")
                endif()
        set(QASR_BLAS_SELECTED "openblas" CACHE STRING "" FORCE)
                return()
            endif()
            # The prebuilt OpenBLAS releases ship OpenBLASConfig.cmake that sets
            # *relative* paths (e.g. "win64/include", "win64/bin/libopenblas.dll").
            # Detect this case and resolve them against OpenBLAS_DIR.
            if(DEFINED OpenBLAS_DIR)
                get_filename_component(_openblas_prefix "${OpenBLAS_DIR}/../../.." ABSOLUTE)
            endif()

            set(_openblas_inc "")
            if(DEFINED OpenBLAS_INCLUDE_DIRS)
                set(_openblas_inc "${OpenBLAS_INCLUDE_DIRS}")
            elseif(DEFINED OpenBLAS_INCLUDE_DIR)
                set(_openblas_inc "${OpenBLAS_INCLUDE_DIR}")
            endif()
            if(_openblas_inc AND NOT IS_ABSOLUTE "${_openblas_inc}" AND DEFINED _openblas_prefix)
                set(_openblas_inc "${_openblas_prefix}/${_openblas_inc}")
            endif()
            if(_openblas_inc AND NOT EXISTS "${_openblas_inc}" AND DEFINED _openblas_prefix)
                set(_openblas_inc "${_openblas_prefix}/include")
            endif()
            if(_openblas_inc AND EXISTS "${_openblas_inc}")
                target_include_directories(${target_name} PUBLIC "${_openblas_inc}")
            endif()

            set(_openblas_lib "")
            if(DEFINED OpenBLAS_LIBRARIES)
                set(_openblas_lib "${OpenBLAS_LIBRARIES}")
            elseif(DEFINED OpenBLAS_LIBRARY)
                set(_openblas_lib "${OpenBLAS_LIBRARY}")
            endif()
            if(_openblas_lib AND NOT IS_ABSOLUTE "${_openblas_lib}" AND DEFINED _openblas_prefix)
                set(_openblas_lib "${_openblas_prefix}/${_openblas_lib}")
            endif()
            if(_openblas_lib AND EXISTS "${_openblas_lib}")
                target_link_libraries(${target_name} PUBLIC "${_openblas_lib}")
            elseif(_openblas_lib)
                target_link_libraries(${target_name} PUBLIC ${_openblas_lib})
            else()
                message(FATAL_ERROR "qasr BLAS: OpenBLAS found but no link target exported.")
            endif()
            target_compile_definitions(${target_name} PUBLIC QASR_BLAS_OPENBLAS=1)
            if(_backend STREQUAL "openblas")
                message(STATUS "qasr BLAS: pinned to OpenBLAS (CMake config)")
            else()
                message(STATUS "qasr BLAS: selected OpenBLAS (default, BSD-3)")
            endif()
        set(QASR_BLAS_SELECTED "openblas" CACHE STRING "" FORCE)
            return()
        endif()
        # Fallback: pkg-config (Debian/Ubuntu libopenblas-dev)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)
            target_compile_definitions(${target_name} PUBLIC QASR_BLAS_OPENBLAS=1)
            target_link_libraries(${target_name} PUBLIC PkgConfig::OPENBLAS)
            if(_backend STREQUAL "openblas")
                message(STATUS "qasr BLAS: pinned to OpenBLAS (pkg-config)")
            else()
                message(STATUS "qasr BLAS: selected OpenBLAS (default, BSD-3, pkg-config)")
            endif()
        set(QASR_BLAS_SELECTED "openblas" CACHE STRING "" FORCE)
            return()
        endif()
        if(_backend STREQUAL "openblas")
            message(FATAL_ERROR
                "qasr BLAS: -DQASR_BLAS=openblas but libopenblas not found.\n"
                "  Install: sudo apt-get install -y libopenblas-dev")
        endif()
    endif()

    # ------------------------------------------------------------------
    # 2. BLIS (BSD-3)  ── drop-in alternative
    #    apt: libblis-openmp-dev  /  source build into $HOME/.local/blis
    # ------------------------------------------------------------------
    if(_backend STREQUAL "" OR _backend STREQUAL "auto" OR _backend STREQUAL "blis")
        # BLIS ships cblas.h under <prefix>/include/blis/  (not <prefix>/include/)
        # and libblis.so under <prefix>/lib/.  Apt-installed Debian packages put
        # cblas.h directly under /usr/include/ though, so probe both layouts.
        set(_blis_search_inc
            /usr/include
            /usr/include/blis
            /usr/local/include
            /usr/local/include/blis
            $ENV{HOME}/.local/blis/include
            $ENV{HOME}/.local/blis/include/blis
            /opt/qasr-deps/blis/include
            /opt/qasr-deps/blis/include/blis)
        set(_blis_search_lib
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
            $ENV{HOME}/.local/blis/lib
            /opt/qasr-deps/blis/lib)
        set(_blis_hdr "cblas.h")
        find_path(_blis_inc_dir ${_blis_hdr}
            PATHS ${_blis_search_inc}
            NO_DEFAULT_PATH)
        find_library(_blis_lib NAMES blis
            PATHS ${_blis_search_lib}
            NO_DEFAULT_PATH)
        if(_blis_inc_dir AND _blis_lib)
            message(STATUS "qasr BLAS: BLIS found (lib=${_blis_lib}, hdr=${_blis_inc_dir})")
            target_include_directories(${target_name} PUBLIC "${_blis_inc_dir}")
            target_link_libraries(${target_name} PUBLIC "${_blis_lib}")
            target_compile_definitions(${target_name} PUBLIC QASR_BLAS_BLIS=1)
            if(_backend STREQUAL "blis")
                message(STATUS "qasr BLAS: pinned to BLIS")
            else()
                message(STATUS "qasr BLAS: selected BLIS (BSD-3, alternative)")
            endif()
        set(QASR_BLAS_SELECTED "blis" CACHE STRING "" FORCE)
            return()
        endif()
        if(_backend STREQUAL "blis")
            message(FATAL_ERROR
                "qasr BLAS: -DQASR_BLAS=blis requested but libblis not found.\n"
                "  Install: sudo apt-get install -y libblis-openmp-dev\n"
                "  Or build from source: cd /tmp/BLIS && ./configure --prefix=\\$HOME/.local/blis haswell && make -j\$(nproc) install\n"
                "  Or fall back to OpenBLAS: cmake -DQASR_BLAS=openblas ...")
        endif()
    endif()

    # ------------------------------------------------------------------
    # 3. MKL (Intel oneAPI)  ── last proprietary option
    #    Proprietary licence, free for personal use; binary-only kernels.
    # ------------------------------------------------------------------
    if(_backend STREQUAL "" OR _backend STREQUAL "auto" OR _backend STREQUAL "mkl")
        set(_mkl_hint OFF)
        if(DEFINED ENV{MKLROOT})
            set(_mkl_hint ON)
        elseif(EXISTS "/opt/intel/oneapi/mkl/latest")
            set(_mkl_hint ON)
        elseif(EXISTS "/opt/intel/oneapi")
            set(_mkl_hint ON)
        elseif(EXISTS "/opt/intel/mkl")
            set(_mkl_hint ON)
        elseif(EXISTS "/opt/miniforge3/envs/mkl")
            set(_mkl_hint ON)
        endif()
        if(_mkl_hint)
            find_package(MKL CONFIG QUIET)
            if(MKL_FOUND)
                target_link_libraries(${target_name} PUBLIC MKL::MKL)
                target_compile_definitions(${target_name} PUBLIC QASR_BLAS_MKL=1)
                if(_backend STREQUAL "mkl")
                    message(STATUS "qasr BLAS: pinned to MKL (CMake config)")
                else()
                    message(STATUS "qasr BLAS: selected MKL (priority 3, proprietary)")
                endif()
        set(QASR_BLAS_SELECTED "mkl" CACHE STRING "" FORCE)
                return()
            endif()
            # Manual fallback: pick up the standard oneAPI / conda layout
            set(_mkl_root "$ENV{MKLROOT}")
            if(_mkl_root STREQUAL "")
                if(EXISTS "/opt/intel/oneapi/mkl/latest")
                    set(_mkl_root "/opt/intel/oneapi/mkl/latest")
                elseif(EXISTS "/opt/intel/oneapi")
                    set(_mkl_root "/opt/intel/oneapi")
                elseif(EXISTS "/opt/intel/mkl")
                    set(_mkl_root "/opt/intel/mkl")
                elseif(EXISTS "/opt/miniforge3/envs/mkl")
                    set(_mkl_root "/opt/miniforge3/envs/mkl")
                endif()
            endif()
            if(_mkl_root AND EXISTS "${_mkl_root}/include/mkl.h")
                message(STATUS "qasr BLAS: MKL headers at ${_mkl_root}")
                target_include_directories(${target_name} PUBLIC "${_mkl_root}/include")
                if(EXISTS "${_mkl_root}/lib/intel64")
                    set(_mkl_lib_dir "${_mkl_root}/lib/intel64")
                else()
                    set(_mkl_lib_dir "${_mkl_root}/lib")
                endif()
                target_link_directories(${target_name} PUBLIC "${_mkl_lib_dir}")
                target_link_libraries(${target_name} PUBLIC mkl_rt)
                # MKL's CBLAS declarations live in mkl_cblas.h, but the project
                # uses the Netlib convention <cblas.h>.  Emit a tiny shim
                # header at configure time so the existing #include "cblas.h"
                # in qwen_asr_kernels.c resolves unchanged.
                if(NOT EXISTS "${_mkl_root}/include/cblas.h"
                        AND EXISTS "${_mkl_root}/include/mkl_cblas.h")
                    file(WRITE "${_mkl_root}/include/cblas.h"
                        "/* Auto-generated by QasrBlas.cmake: redirect to MKL CBLAS */\n"
                        "#include <mkl_cblas.h>\n")
                endif()
                target_compile_definitions(${target_name} PUBLIC QASR_BLAS_MKL=1)
                if(_backend STREQUAL "mkl")
                    message(STATUS "qasr BLAS: pinned to MKL (oneAPI, manual layout)")
                else()
                    message(STATUS "qasr BLAS: selected MKL (priority 3, proprietary, manual layout)")
                endif()
                set(QASR_BLAS_SELECTED "mkl" CACHE STRING "" FORCE)
                return()
            endif()
        endif()
        if(_backend STREQUAL "mkl")
            message(FATAL_ERROR
                "qasr BLAS: -DQASR_BLAS=mkl requested but no oneAPI MKL found.\n"
                "  Install oneAPI MKL and source setvars.sh, or set MKLROOT, e.g.:\n"
                "    source /opt/intel/oneapi/setvars.sh\n"
                "    cmake -DQASR_BLAS=mkl ...\n"
                "  Or use an open-source BLAS: cmake -DQASR_BLAS=blis|openblas ...")
        endif()
    endif()

    # ------------------------------------------------------------------
    # 4. Reference (netlib)  ── slow but always present
    # ------------------------------------------------------------------
    if(_backend STREQUAL "" OR _backend STREQUAL "auto" OR _backend STREQUAL "ref")
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(REF_BLAS REQUIRED IMPORTED_TARGET blas)
            target_compile_definitions(${target_name} PUBLIC QASR_BLAS_REF=1)
            target_link_libraries(${target_name} PUBLIC PkgConfig::REF_BLAS)
            message(WARNING "qasr BLAS: using reference BLAS — install libblis-openmp-dev or libopenblas-dev for speed")
        set(QASR_BLAS_SELECTED "ref" CACHE STRING "" FORCE)
            return()
        endif()
    endif()

    message(FATAL_ERROR
        "qasr BLAS: no usable BLAS found.\n"
        "  Install one of (open-source, BSD-3, recommended):\n"
        "    sudo apt-get install -y libblis-openmp-dev    # priority 2 (alternative)\n"
        "    sudo apt-get install -y libopenblas-dev       # priority 1 (default)\n"
        "  Or use the proprietary but free-for-personal oneAPI MKL:\n"
        "    source /opt/intel/oneapi/setvars.sh && cmake -DQASR_BLAS=mkl ...")
endfunction()

# Tag the bench target with the selected backend name (used by qasr_blas_bench
# to self-identify in its JSON output).  We do this lazily: if the variable is
# empty (e.g. project pulled in QasrBlas.cmake before configuring qasr_blas_bench),
# the bench falls back to "unknown".
function(qasr_configure_blas_bench_tag target_name)
    if(NOT QASR_BLAS_SELECTED STREQUAL "")
        target_compile_definitions(${target_name} PRIVATE
            "QASR_BENCH_BACKEND=\"${QASR_BLAS_SELECTED}\"")
    endif()
endfunction()
