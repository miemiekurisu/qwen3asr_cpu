function(qasr_configure_blas target_name)
    if(APPLE)
        find_library(QASR_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        target_compile_definitions(${target_name} PUBLIC QASR_BLAS_ACCELERATE=1)
        target_link_libraries(${target_name} PUBLIC ${QASR_ACCELERATE_FRAMEWORK})
        return()
    endif()

    find_package(OpenBLAS REQUIRED)
    target_compile_definitions(${target_name} PUBLIC QASR_BLAS_OPENBLAS=1)

    if(TARGET OpenBLAS::OpenBLAS)
        target_link_libraries(${target_name} PUBLIC OpenBLAS::OpenBLAS)
        return()
    endif()

    # The prebuilt OpenBLAS releases for Windows ship an OpenBLASConfig.cmake
    # that sets *relative* paths (e.g. "win64/include", "win64/bin/libopenblas.dll").
    # These do not resolve correctly when consumed by MSVC.  Detect this case and
    # resolve the paths relative to the installation prefix derived from
    # OpenBLAS_DIR (which points to <prefix>/lib/cmake/openblas).
    if(DEFINED OpenBLAS_DIR)
        get_filename_component(_openblas_prefix "${OpenBLAS_DIR}/../../.." ABSOLUTE)
    endif()

    # --- include dirs ---
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

    # --- library ---
    set(_openblas_lib "")
    if(DEFINED OpenBLAS_LIBRARIES)
        set(_openblas_lib "${OpenBLAS_LIBRARIES}")
    elseif(DEFINED OpenBLAS_LIBRARY)
        set(_openblas_lib "${OpenBLAS_LIBRARY}")
    endif()
    if(_openblas_lib AND NOT IS_ABSOLUTE "${_openblas_lib}" AND DEFINED _openblas_prefix)
        set(_openblas_lib "${_openblas_prefix}/${_openblas_lib}")
    endif()
    if(_openblas_lib AND NOT EXISTS "${_openblas_lib}" AND DEFINED _openblas_prefix)
        # Prefer the MSVC import library (.lib) when it exists next to the DLL.
        find_file(_openblas_implib NAMES libopenblas.lib openblas.lib
            PATHS "${_openblas_prefix}" "${_openblas_prefix}/lib"
            NO_DEFAULT_PATH)
        if(_openblas_implib)
            set(_openblas_lib "${_openblas_implib}")
        endif()
    endif()

    if(_openblas_lib AND EXISTS "${_openblas_lib}")
        target_link_libraries(${target_name} PUBLIC "${_openblas_lib}")
    elseif(_openblas_lib)
        # Fallback: pass whatever was exported (may work on other platforms).
        target_link_libraries(${target_name} PUBLIC ${_openblas_lib})
    else()
        message(FATAL_ERROR "OpenBLAS found, but no link target or library path was exported.")
    endif()
endfunction()
