#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "dynamorio" for configuration "RelWithDebInfo"
set_property(TARGET dynamorio APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(dynamorio PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib32/release/libdynamorio.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdynamorio.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS dynamorio )
list(APPEND _IMPORT_CHECK_FILES_FOR_dynamorio "${_IMPORT_PREFIX}/lib32/release/libdynamorio.so" )

# Import target "dynamorio_static" for configuration "RelWithDebInfo"
set_property(TARGET dynamorio_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(dynamorio_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "ASM;C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib32/release/libdynamorio_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS dynamorio_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_dynamorio_static "${_IMPORT_PREFIX}/lib32/release/libdynamorio_static.a" )

# Import target "drinjectlib" for configuration "RelWithDebInfo"
set_property(TARGET drinjectlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drinjectlib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib32/libdrinjectlib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drinjectlib )
list(APPEND _IMPORT_CHECK_FILES_FOR_drinjectlib "${_IMPORT_PREFIX}/lib32/libdrinjectlib.a" )

# Import target "drdecode" for configuration "RelWithDebInfo"
set_property(TARGET drdecode APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drdecode PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib32/release/libdrdecode.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drdecode )
list(APPEND _IMPORT_CHECK_FILES_FOR_drdecode "${_IMPORT_PREFIX}/lib32/release/libdrdecode.a" )

# Import target "drlibc" for configuration "RelWithDebInfo"
set_property(TARGET drlibc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drlibc PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "ASM;C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib32/libdrlibc.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drlibc )
list(APPEND _IMPORT_CHECK_FILES_FOR_drlibc "${_IMPORT_PREFIX}/lib32/libdrlibc.a" )

# Import target "drmemfuncs" for configuration "RelWithDebInfo"
set_property(TARGET drmemfuncs APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemfuncs PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "ASM;C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib32/libdrmemfuncs.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemfuncs )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemfuncs "${_IMPORT_PREFIX}/lib32/libdrmemfuncs.a" )

# Import target "drconfiglib" for configuration "RelWithDebInfo"
set_property(TARGET drconfiglib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drconfiglib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib32/libdrconfiglib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drconfiglib )
list(APPEND _IMPORT_CHECK_FILES_FOR_drconfiglib "${_IMPORT_PREFIX}/lib32/libdrconfiglib.a" )

# Import target "drfrontendlib" for configuration "RelWithDebInfo"
set_property(TARGET drfrontendlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drfrontendlib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/bin32/libdrfrontendlib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drfrontendlib )
list(APPEND _IMPORT_CHECK_FILES_FOR_drfrontendlib "${_IMPORT_PREFIX}/bin32/libdrfrontendlib.a" )

# Import target "minizip" for configuration "RelWithDebInfo"
set_property(TARGET minizip APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(minizip PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libminizip.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS minizip )
list(APPEND _IMPORT_CHECK_FILES_FOR_minizip "${_IMPORT_PREFIX}/tools/lib32/release/libminizip.a" )

# Import target "drmemtrace_reuse_distance" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_reuse_distance APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_reuse_distance PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_reuse_distance.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_reuse_distance )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_reuse_distance "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_reuse_distance.a" )

# Import target "drmemtrace_histogram" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_histogram APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_histogram PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_histogram.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_histogram )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_histogram "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_histogram.a" )

# Import target "drmemtrace_reuse_time" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_reuse_time APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_reuse_time PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_reuse_time.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_reuse_time )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_reuse_time "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_reuse_time.a" )

# Import target "drmemtrace_basic_counts" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_basic_counts APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_basic_counts PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_basic_counts.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_basic_counts )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_basic_counts "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_basic_counts.a" )

# Import target "drmemtrace_opcode_mix" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_opcode_mix APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_opcode_mix PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_opcode_mix.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_opcode_mix )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_opcode_mix "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_opcode_mix.a" )

# Import target "drmemtrace_syscall_mix" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_syscall_mix APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_syscall_mix PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_syscall_mix.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_syscall_mix )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_syscall_mix "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_syscall_mix.a" )

# Import target "drmemtrace_view" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_view APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_view PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_view.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_view )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_view "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_view.a" )

# Import target "drmemtrace_func_view" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_func_view APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_func_view PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_func_view.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_func_view )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_func_view "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_func_view.a" )

# Import target "drmemtrace_invariant_checker" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_invariant_checker APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_invariant_checker PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_invariant_checker.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_invariant_checker )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_invariant_checker "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_invariant_checker.a" )

# Import target "drmemtrace_schedule_stats" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_schedule_stats APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_schedule_stats PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_schedule_stats.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_schedule_stats )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_schedule_stats "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_schedule_stats.a" )

# Import target "drmemtrace_schedule_file" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_schedule_file APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_schedule_file PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_schedule_file.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_schedule_file )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_schedule_file "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_schedule_file.a" )

# Import target "drmemtrace_simulator" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_simulator APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_simulator PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_simulator.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_simulator )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_simulator "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_simulator.a" )

# Import target "drmemtrace_record_filter" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_record_filter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_record_filter PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_record_filter.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_record_filter )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_record_filter "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_record_filter.a" )

# Import target "directory_iterator" for configuration "RelWithDebInfo"
set_property(TARGET directory_iterator APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(directory_iterator PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdirectory_iterator.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS directory_iterator )
list(APPEND _IMPORT_CHECK_FILES_FOR_directory_iterator "${_IMPORT_PREFIX}/tools/lib32/release/libdirectory_iterator.a" )

# Import target "drmemtrace_raw2trace" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_raw2trace APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_raw2trace PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_raw2trace.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_raw2trace )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_raw2trace "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_raw2trace.a" )

# Import target "drmemtrace_analyzer" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_analyzer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_analyzer PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_analyzer.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_analyzer )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_analyzer "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_analyzer.a" )

# Import target "drmemtrace_empty" for configuration "RelWithDebInfo"
set_property(TARGET drmemtrace_empty APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmemtrace_empty PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_empty.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmemtrace_empty )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmemtrace_empty "${_IMPORT_PREFIX}/tools/lib32/release/libdrmemtrace_empty.a" )

# Import target "empty_creator" for configuration "RelWithDebInfo"
set_property(TARGET empty_creator APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(empty_creator PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/tools/lib32/release/libempty_creator.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libempty_creator.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS empty_creator )
list(APPEND _IMPORT_CHECK_FILES_FOR_empty_creator "${_IMPORT_PREFIX}/tools/lib32/release/libempty_creator.so" )

# Import target "drcontainers" for configuration "RelWithDebInfo"
set_property(TARGET drcontainers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drcontainers PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrcontainers.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drcontainers )
list(APPEND _IMPORT_CHECK_FILES_FOR_drcontainers "${_IMPORT_PREFIX}/ext/lib32/release/libdrcontainers.a" )

# Import target "drmgr" for configuration "RelWithDebInfo"
set_property(TARGET drmgr APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmgr PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrmgr.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrmgr.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmgr )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmgr "${_IMPORT_PREFIX}/ext/lib32/release/libdrmgr.so" )

# Import target "drmgr_static" for configuration "RelWithDebInfo"
set_property(TARGET drmgr_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drmgr_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrmgr_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drmgr_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drmgr_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrmgr_static.a" )

# Import target "drx" for configuration "RelWithDebInfo"
set_property(TARGET drx APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drx PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrx.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrx.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drx )
list(APPEND _IMPORT_CHECK_FILES_FOR_drx "${_IMPORT_PREFIX}/ext/lib32/release/libdrx.so" )

# Import target "drx_static" for configuration "RelWithDebInfo"
set_property(TARGET drx_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drx_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrx_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drx_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drx_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrx_static.a" )

# Import target "drwrap" for configuration "RelWithDebInfo"
set_property(TARGET drwrap APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drwrap PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrwrap.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrwrap.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drwrap )
list(APPEND _IMPORT_CHECK_FILES_FOR_drwrap "${_IMPORT_PREFIX}/ext/lib32/release/libdrwrap.so" )

# Import target "drwrap_static" for configuration "RelWithDebInfo"
set_property(TARGET drwrap_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drwrap_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "ASM;C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrwrap_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drwrap_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drwrap_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrwrap_static.a" )

# Import target "drreg" for configuration "RelWithDebInfo"
set_property(TARGET drreg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drreg PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrreg.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrreg.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drreg )
list(APPEND _IMPORT_CHECK_FILES_FOR_drreg "${_IMPORT_PREFIX}/ext/lib32/release/libdrreg.so" )

# Import target "drreg_static" for configuration "RelWithDebInfo"
set_property(TARGET drreg_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drreg_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrreg_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drreg_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drreg_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrreg_static.a" )

# Import target "drbbdup" for configuration "RelWithDebInfo"
set_property(TARGET drbbdup APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drbbdup PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrbbdup.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrbbdup.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drbbdup )
list(APPEND _IMPORT_CHECK_FILES_FOR_drbbdup "${_IMPORT_PREFIX}/ext/lib32/release/libdrbbdup.so" )

# Import target "drbbdup_static" for configuration "RelWithDebInfo"
set_property(TARGET drbbdup_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drbbdup_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrbbdup_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drbbdup_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drbbdup_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrbbdup_static.a" )

# Import target "elf_pic" for configuration "RelWithDebInfo"
set_property(TARGET elf_pic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(elf_pic PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libelf_pic.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS elf_pic )
list(APPEND _IMPORT_CHECK_FILES_FOR_elf_pic "${_IMPORT_PREFIX}/ext/lib32/release/libelf_pic.a" )

# Import target "dw_pic" for configuration "RelWithDebInfo"
set_property(TARGET dw_pic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(dw_pic PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdw_pic.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS dw_pic )
list(APPEND _IMPORT_CHECK_FILES_FOR_dw_pic "${_IMPORT_PREFIX}/ext/lib32/release/libdw_pic.a" )

# Import target "dwelf_pic" for configuration "RelWithDebInfo"
set_property(TARGET dwelf_pic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(dwelf_pic PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdwelf_pic.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS dwelf_pic )
list(APPEND _IMPORT_CHECK_FILES_FOR_dwelf_pic "${_IMPORT_PREFIX}/ext/lib32/release/libdwelf_pic.a" )

# Import target "ebl_pic" for configuration "RelWithDebInfo"
set_property(TARGET ebl_pic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(ebl_pic PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libebl_pic.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ebl_pic )
list(APPEND _IMPORT_CHECK_FILES_FOR_ebl_pic "${_IMPORT_PREFIX}/ext/lib32/release/libebl_pic.a" )

# Import target "drsyms" for configuration "RelWithDebInfo"
set_property(TARGET drsyms APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drsyms PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELWITHDEBINFO "dynamorio"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrsyms.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrsyms.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drsyms )
list(APPEND _IMPORT_CHECK_FILES_FOR_drsyms "${_IMPORT_PREFIX}/ext/lib32/release/libdrsyms.so" )

# Import target "drsyms_static" for configuration "RelWithDebInfo"
set_property(TARGET drsyms_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drsyms_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C;CXX"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrsyms_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drsyms_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drsyms_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrsyms_static.a" )

# Import target "drutil" for configuration "RelWithDebInfo"
set_property(TARGET drutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drutil PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrutil.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrutil.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drutil )
list(APPEND _IMPORT_CHECK_FILES_FOR_drutil "${_IMPORT_PREFIX}/ext/lib32/release/libdrutil.so" )

# Import target "drutil_static" for configuration "RelWithDebInfo"
set_property(TARGET drutil_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drutil_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrutil_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drutil_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drutil_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrutil_static.a" )

# Import target "drcovlib" for configuration "RelWithDebInfo"
set_property(TARGET drcovlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drcovlib PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrcovlib.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrcovlib.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drcovlib )
list(APPEND _IMPORT_CHECK_FILES_FOR_drcovlib "${_IMPORT_PREFIX}/ext/lib32/release/libdrcovlib.so" )

# Import target "drcovlib_static" for configuration "RelWithDebInfo"
set_property(TARGET drcovlib_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drcovlib_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrcovlib_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drcovlib_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drcovlib_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrcovlib_static.a" )

# Import target "drstatecmp" for configuration "RelWithDebInfo"
set_property(TARGET drstatecmp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drstatecmp PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrstatecmp.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrstatecmp.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drstatecmp )
list(APPEND _IMPORT_CHECK_FILES_FOR_drstatecmp "${_IMPORT_PREFIX}/ext/lib32/release/libdrstatecmp.so" )

# Import target "drstatecmp_static" for configuration "RelWithDebInfo"
set_property(TARGET drstatecmp_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drstatecmp_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrstatecmp_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drstatecmp_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drstatecmp_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrstatecmp_static.a" )

# Import target "drcallstack" for configuration "RelWithDebInfo"
set_property(TARGET drcallstack APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drcallstack PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrcallstack.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libdrcallstack.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS drcallstack )
list(APPEND _IMPORT_CHECK_FILES_FOR_drcallstack "${_IMPORT_PREFIX}/ext/lib32/release/libdrcallstack.so" )

# Import target "drcallstack_static" for configuration "RelWithDebInfo"
set_property(TARGET drcallstack_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(drcallstack_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELWITHDEBINFO "C"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/ext/lib32/release/libdrcallstack_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS drcallstack_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_drcallstack_static "${_IMPORT_PREFIX}/ext/lib32/release/libdrcallstack_static.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
