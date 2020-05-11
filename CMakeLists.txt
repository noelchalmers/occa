cmake_minimum_required (VERSION 3.12 FATAL_ERROR)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

message(STATUS "Using CMake version ${CMAKE_VERSION}")
cmake_policy(SET CMP0054 NEW)
cmake_policy(SET CMP0056 NEW)

# Set project
project (occa
  DESCRIPTION "JIT Compilation for Multiple Architectures: C++, OpenMP, CUDA, HIP, OpenCL, Metal"
  HOMEPAGE_URL "https://github.com/libocca/occa/"
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(OS)

option(ALLOW_CUDA "Allow CUDA" OFF)
option(ALLOW_MPI  "Allow MPI" OFF)
option(ALLOW_OPENMP "Allow OpenMP" OFF)
option(ALLOW_OPENCL "Allow OpenCL" OFF)
option(ALLOW_METAL "Allow Metal" OFF)
option(ALLOW_HIP "Allow HIP" OFF)
#option(ALLOW_ONEAPI "Allow OneAPI" OFF)
option(ENABLE_TESTS "Enable tests" OFF)
option(ENABLE_EXAMPLES "Build simple examples" OFF)
option(BUILD_UTILITY "Build occa utility binary" OFF)

find_package(Threads REQUIRED)

if (ALLOW_OPENMP)
  include(OpenMP)
endif()

if (ALLOW_OPENCL)
  include(OpenCL)
endif()

# searches for CUDA if allowed
if (ALLOW_CUDA)
  include(CUDA)
endif()

if (ALLOW_HIP)
  include(HIP)
endif()

# searches for MPI if allowed
if (ALLOW_MPI)
  include(MPI)
endif()

if (ALLOW_METAL)
  include(Metal)
endif()

include(SetCompilerFlags)

if (ALLOW_OPENMP)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(OCCA_BUILD_DIR ${PROJECT_BINARY_DIR})
configure_file(scripts/compiledDefinesTemplate.hpp.in ${PROJECT_BINARY_DIR}/include/occa/defines/compiledDefines.hpp)
install(FILES ${PROJECT_BINARY_DIR}/include/occa/defines/compiledDefines.hpp DESTINATION include/occa/defines)

if (ENABLE_TESTS)
  include(CTest)
endif()

file(GLOB_RECURSE sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.cpp")
list(REMOVE_ITEM sources src/core/kernelOperators.cpp src/tools/runFunction.cpp)

if (WITH_METAL)
  file(GLOB_RECURSE sources_metal RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.mm")
  set(sources ${sources} ${sources_metal})
endif()

add_library(libocca SHARED ${sources})

# Without this, CMake will create liblibocca.so
set_target_properties(libocca PROPERTIES OUTPUT_NAME occa LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(LIBOCCA_LIBRARIES ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

if (WITH_CUDA)
  set(LIBOCCA_LIBRARIES ${CUDA_LIBRARIES} ${LIBOCCA_LIBRARIES})
endif()

if (WITH_HIP)
  set(LIBOCCA_LIBRARIES ${HIP_LIBRARIES} ${LIBOCCA_LIBRARIES})
endif()

if (WITH_OPENCL)
  set(LIBOCCA_LIBRARIES ${OpenCL_LIBRARIES} ${LIBOCCA_LIBRARIES})
endif()

if (WITH_MPI)
  set(LIBOCCA_LIBRARIES ${MPI_CXX_LIBRARIES} ${MPI_C_LIBRARIES} ${LIBOCCA_LIBRARIES})
endif()

if (WITH_METAL)
  set(LIBOCCA_LIBRARIES ${METAL_LIBRARY} ${CORE_SERVICES} ${APP_KIT} ${LIBOCCA_LIBRARIES})
endif()

target_link_libraries(libocca ${LIBOCCA_LIBRARIES})

target_include_directories(libocca PUBLIC
                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)
target_compile_definitions(libocca PRIVATE -DUSE_CMAKE)

install(TARGETS libocca DESTINATION lib)

if (BUILD_UTILITY)
  add_subdirectory(bin)
endif()

if (ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

install(DIRECTORY include/ DESTINATION include)
