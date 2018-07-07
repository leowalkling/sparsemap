if(NOT AD3_FOUND)
  set(_find_extra_args)
  if(AD3_FIND_REQUIRED)
    list(APPEND _find_extra_args REQUIRED)
  endif()
  if(AD3_FIND_QUIET)
    list(APPEND _find_extra_args QUIET)
  endif()

  if(NOT AD3_DIR)
    set(AD3_DIR $ENV{AD3_DIR})
  endif()

  find_package(PythonInterp ${_find_extra_args})
    if(PYTHON_EXECUTABLE)
      execute_process(COMMAND "${PYTHON_EXECUTABLE}"
        -c "import ad3; print(" ".join(ad3.__spec__.submodule_search_locations))"
        OUTPUT_VARIABLE AD3_PACKAGE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        )
      execute_process(COMMAND "${PYTHON_EXECUTABLE}"
        -c "import ad3; print(ad3.__version__)"
        OUTPUT_VARIABLE AD3_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        )
    endif()

endif()

set(AD3_INCLUDE_DIR ${AD3_DIR}/include)

if (WIN32)
  set(AD3_LIBRARY ${AD3_DIR}/lib/ad3.lib)
else()
  set(AD3_LIBRARY ${AD3_DIR}/lib/libad3.a)
endif()

# handle the QUIETLY and REQUIRED arguments and set NumPy_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AD3
                                  REQUIRED_VARS AD3_DIR
                                  VERSION_VAR AD3_VERSION)

mark_as_advanced(AD3_INCLUDE_DIR)
mark_as_advanced(AD3_LIBRARY)
mark_as_advanced(AD3_PACKAGE_DIR)
