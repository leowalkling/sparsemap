cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0023 NEW)

project(sparsemap)

find_package(PythonInterp)
find_package(PythonLibs)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonExtensions REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(AD3 REQUIRED MODULE)

set(CMAKE_CXX_STANDARD 11)

if(WIN32)
    set(CONDA_SUBDIR "Library/")
else()
    set(CONDA_SUBDIR "")
endif()

add_subdirectory(src)

include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(python)
