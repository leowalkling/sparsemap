include_directories(${CMAKE_CURRENT_LIST_DIR} ${AD3_PACKAGE_DIR})
add_cython_target(pyx_factors _factors.pyx CXX)
add_cython_target(pyx_sparsemap _sparsemap.pyx CXX)

add_library(_factors MODULE ${pyx_factors})
python_extension_module(_factors
                        LINKED_MODULES_VAR linked_modules_list
                        FORWARD_DECL_MODULES_VAR fdecl_module_list)
target_include_directories(_factors PRIVATE ${NumPy_INCLUDE_DIRS} ${AD3_INCLUDE_DIR})
target_link_libraries(_factors lapjv ${AD3_LIBRARY})

add_library(_sparsemap MODULE ${pyx_sparsemap})
python_extension_module(_sparsemap
                        LINKED_MODULES_VAR linked_modules_list
                        FORWARD_DECL_MODULES_VAR fdecl_module_list)
target_include_directories(_sparsemap PRIVATE ${NumPy_INCLUDE_DIRS} ${AD3_INCLUDE_DIR})
target_link_libraries(_sparsemap lapjv ${AD3_LIBRARY})

install(TARGETS _factors _sparsemap
        ARCHIVE DESTINATION sparsemap
        LIBRARY DESTINATION sparsemap)

python_modules_header(modules
                      FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
