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

set(LIBRARY_FILENAME "iclBLAS")
set(API_INCLUDE_DIR ${ICLGPU__API_INCLUDE_DIR}/iclBLAS)

set(MODULE_ID API_BLAS)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

file(GLOB HEADERS
    ${API_INCLUDE_DIR}/*.h
    ${API_INCLUDE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

if(MSVC)
    add_compile_options("/bigobj")
endif()

add_library(${TARGET_NAME}
    SHARED
    ${SOURCES}
    ${HEADERS}
)

target_include_directories(${TARGET_NAME}
    PUBLIC ${API_INCLUDE_DIR}
)

target_compile_definitions(${TARGET_NAME}
    PRIVATE EXPORT_ICLBLAS_SYMBOLS
)

target_link_libraries(${TARGET_NAME}
    PRIVATE ${ICLGPU__CORE_NAME}
    PRIVATE ${ICLGPU__BLAS_NAME}
)

set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${LIBRARY_FILENAME}
    FOLDER ${TARGET_FOLDER_NAME}
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# ======================== Export module info =================================
set(MODULE_IDS                       ${MODULE_ID}                PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_NAME        ${TARGET_NAME}              PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_INCLUDE_DIR ${API_INCLUDE_DIR}          PARENT_SCOPE)
