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

set(TARGET_NAME "iclBLAS")
set(API_INCLUDE_DIR ${ICLGPU__API_INCLUDE_DIR}/iclBLAS)

set(MODULE_ID API_BLAS)

set(ICLGPU__MODULE_ID                ${MODULE_ID}                PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_NAME        ${TARGET_NAME}              PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_INCLUDE_DIR ${API_INCLUDE_DIR}          PARENT_SCOPE)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB HEADERS
    ${API_INCLUDE_DIR}/*.h
    ${API_INCLUDE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

include_directories(
    ${API_INCLUDE_DIR}
    ${ICLGPU__CORE_INCLUDE_DIR}
    ${ICLGPU__BLAS_INCLUDE_DIR}
)

set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    EXPORT_ICLBLAS_SYMBOLS
)

if(MSVC)
add_compile_options("/bigobj")
endif()

add_library(${TARGET_NAME}
    SHARED
    ${SOURCES}
    ${HEADERS}
)

target_link_libraries(${TARGET_NAME} ${ICLGPU__CORE_NAME} ${ICLGPU__BLAS_NAME})
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED)
set_property(TARGET ${TARGET_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
