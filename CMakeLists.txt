cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

if( CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR )
    message( FATAL_ERROR "Please select another Build Directory ! (and give it a clever name, like bin_Visual2012_64bits/)" )
endif()
if( CMAKE_SOURCE_DIR MATCHES " " )
	message( "Your Source Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()
if( CMAKE_BINARY_DIR MATCHES " " )
	message( "Your Build Directory contains spaces. If you experience problems when compiling, this can be the cause." )
endif()

include(CheckLanguage)
include(CMakeDependentOption)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#project name
project(DMO_Templated LANGUAGES CXX)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED True)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	MESSAGE(STATUS "Generate with CUDA Version " ${CMAKE_CUDA_COMPILER_VERSION})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
	set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
	set(CMAKE_CUDA_ARCHITECTURES 86)
	
	find_package(CUDAToolkit)
endif()





set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH} )

########## required libraries ##########

#OpenMesh
find_package(OpenMesh REQUIRED)
set(LIBS ${LIBS} ${OPENMESH_LIBRARIES})
add_definitions(-D_USE_MATH_DEFINES)

#OpenMP
find_package(OpenMP)
if(NOT OpenMP_FOUND)
	message(WARNING "Building without OpenMP support. This still works but the CPU version of this code will be much slower without OpenMP.")
endif()

if(NOT MSVC)
	#X11
	MESSAGE(STATUS "Add X11 and stdc++fs for non windows systems")
	find_package(X11 REQUIRED)
	set(LIBS ${LIBS} X11)
	set(LIBS ${LIBS} stdc++fs)
endif()

########## executables ##########

set(SRC ${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
