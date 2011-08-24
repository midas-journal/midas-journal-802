

# Note set CUDA_BUILD_EMULATION to OFF to enable Device mode compilation.

cmake_minimum_required(VERSION 2.6.2)

project(Cuda-project)


set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})

option(CITK_USE_CUDA "Use cuda compiler and SDK - this is turned off by default to allow compilation in IJ")

IF (CITK_USE_CUDA)
# find_package(CUDA)
find_package(CUDA QUIET REQUIRED)
# find_package(CUDA 2.1)
# find_package(CUDA 2.2)
# find_package(CUDA 2.3)
# find_package(CUDA 2.1 REQUIRED)
# find_package(CUDA 2.2 REQUIRED)
# find_package(CUDA 2.3 REQUIRED)
# find_package(CUDA 2.1 EXACT)
# find_package(CUDA 2.2 EXACT)
# find_package(CUDA 2.3 EXACT)
# find_package(CUDA 2.1 EXACT REQUIRED)
# find_package(CUDA 2.2 EXACT REQUIRED)
# find_package(CUDA 2.3 EXACT REQUIRED)
# find_package(CUDA 2.2 EXACT QUIET REQUIRED)

#if (CUDA_FOUND)
#   message("CUDA found")
#else()
#   message("CUDA not found")
#endif()

SET (CUDA_USE_SDK ON)

ENDIF (CITK_USE_CUDA)

find_package (ITK)
if ( ITK_FOUND )
   include( ${ITK_USE_FILE} )
endif( ITK_FOUND )



ENABLE_TESTING()
INCLUDE(Dart)
FIND_PROGRAM(ITK_TEST_DRIVER itkTestDriver)
SET(TEST_COMMAND ${ITK_TEST_DRIVER} --add-before-env PATH ${CMAKE_CURRENT_BINARY_DIR})




OPTION(CITK_USE_THRUST "Use thrust widely, even in simple arithmetic filters - testing only")

IF (CITK_USE_THRUST)
   add_definitions(-DCITK_USE_THRUST)
ENDIF (CITK_USE_THRUST)

include_directories(
  ${myProject_SOURCE_DIR}
    )

add_subdirectory(
  src
  bin
  )

