# Copyright (c) 2017-2018 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ========================================= Name / Output settings =====================================

set(TARGET_NAME "test_iclBLAS")


# ========================================= Source/Header files ========================================
file(GLOB SOURCE_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/../gtest_main.cpp)

file(GLOB SOURCE_GTEST_FUSED ${ICLGPU__GTEST_DIR}/*.cc)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  )

file(GLOB HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)
# ===================================== Include/Link directories =======================================

include_directories(
    ${ICLGPU__API_INCLUDE_DIR}
    ${ICLGPU__API_BLAS_INCLUDE_DIR}
    ${ICLGPU__GTEST_DIR}
)

# =============================================== Filters ==============================================

source_group(test_cases    FILES ${SOURCES})
source_group(gtest_fused   FILES ${SOURCE_GTEST_FUSED})

# =================================== Link targets and dependencies ====================================

# Tests executable.
add_executable(${TARGET_NAME}
    ${HEADERS}
    ${SOURCES}
    ${SOURCE_MAIN}
    ${SOURCE_GTEST_FUSED}
)

# Set library dependencies
find_package (Threads)
target_link_libraries(${TARGET_NAME}
    ${ICLGPU__API_BLAS_NAME}
    ${CMAKE_THREAD_LIBS_INIT}
)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED)
# =================================== Custom pre- and post-steps =======================================

if(ICLGPU__RUN_TESTS)
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
      COMMAND ${TARGET_NAME}
      COMMENT "Executing tests..."
    )
endif()

# ======================================================================================================
