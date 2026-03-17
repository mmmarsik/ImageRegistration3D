#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "NIFTI::znz" for configuration ""
set_property(TARGET NIFTI::znz APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(NIFTI::znz PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libznz.a"
  )

list(APPEND _cmake_import_check_targets NIFTI::znz )
list(APPEND _cmake_import_check_files_for_NIFTI::znz "${_IMPORT_PREFIX}/lib/libznz.a" )

# Import target "NIFTI::niftiio" for configuration ""
set_property(TARGET NIFTI::niftiio APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(NIFTI::niftiio PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libniftiio.a"
  )

list(APPEND _cmake_import_check_targets NIFTI::niftiio )
list(APPEND _cmake_import_check_files_for_NIFTI::niftiio "${_IMPORT_PREFIX}/lib/libniftiio.a" )

# Import target "NIFTI::nifti1_tool" for configuration ""
set_property(TARGET NIFTI::nifti1_tool APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(NIFTI::nifti1_tool PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/nifti1_tool"
  )

list(APPEND _cmake_import_check_targets NIFTI::nifti1_tool )
list(APPEND _cmake_import_check_files_for_NIFTI::nifti1_tool "${_IMPORT_PREFIX}/bin/nifti1_tool" )

# Import target "NIFTI::nifticdf" for configuration ""
set_property(TARGET NIFTI::nifticdf APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(NIFTI::nifticdf PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libnifticdf.a"
  )

list(APPEND _cmake_import_check_targets NIFTI::nifticdf )
list(APPEND _cmake_import_check_files_for_NIFTI::nifticdf "${_IMPORT_PREFIX}/lib/libnifticdf.a" )

# Import target "NIFTI::nifti_stats" for configuration ""
set_property(TARGET NIFTI::nifti_stats APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(NIFTI::nifti_stats PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/nifti_stats"
  )

list(APPEND _cmake_import_check_targets NIFTI::nifti_stats )
list(APPEND _cmake_import_check_files_for_NIFTI::nifti_stats "${_IMPORT_PREFIX}/bin/nifti_stats" )

# Import target "NIFTI::nifti2" for configuration ""
set_property(TARGET NIFTI::nifti2 APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(NIFTI::nifti2 PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libnifti2.a"
  )

list(APPEND _cmake_import_check_targets NIFTI::nifti2 )
list(APPEND _cmake_import_check_files_for_NIFTI::nifti2 "${_IMPORT_PREFIX}/lib/libnifti2.a" )

# Import target "NIFTI::nifti_tool" for configuration ""
set_property(TARGET NIFTI::nifti_tool APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(NIFTI::nifti_tool PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/nifti_tool"
  )

list(APPEND _cmake_import_check_targets NIFTI::nifti_tool )
list(APPEND _cmake_import_check_files_for_NIFTI::nifti_tool "${_IMPORT_PREFIX}/bin/nifti_tool" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
