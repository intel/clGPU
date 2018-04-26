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

set(TARGET_NAME "core_lib")
set(API_INCLUDE_DIR ${ICLGPU__API_INCLUDE_DIR}/iclgpu)

set(CODEGEN_TOOLSDIR ${CMAKE_CURRENT_SOURCE_DIR}/../tools)
list(APPEND CMAKE_MODULE_PATH "${CODEGEN_TOOLSDIR}")
include(Codegen)

set(MODULE_ID CORE)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
)

file(GLOB HEADERS
    ${API_INCLUDE_DIR}/*.hpp
    ${API_INCLUDE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)

file(GLOB_RECURSE OCL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.cpp
    ${API_INCLUDE_DIR}/ocl/*.hpp
    ${API_INCLUDE_DIR}/ocl/*.h
)

file(GLOB_RECURSE OCL_KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.cl
)

add_library(${TARGET_NAME} STATIC
    ${HEADERS}
    ${SOURCES}
    ${OCL_FILES}
    ${OCL_KERNELS}
)

source_group(ocl FILES ${OCL_FILES} ${OCL_KERNELS})

target_include_directories(${TARGET_NAME}
    PUBLIC ${API_INCLUDE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${OpenCL_INCLUDE_DIRS}
    PRIVATE ${ICLGPU__KHR_CLHPP_DIR}
)

target_link_libraries(${TARGET_NAME} OpenCL)
set_target_properties(${TARGET_NAME} PROPERTIES
    FOLDER ${TARGET_FOLDER_NAME}
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# ======================== Autogenerate kernels DB ============================
add_codegen_kernels_db("core")

set_target_properties(${CODEGEN_TARGET_NAME} PROPERTIES
    FOLDER ${TARGET_FOLDER_NAME}
)

target_include_directories(${TARGET_NAME}
    PRIVATE ${CODEGEN_INCDIR}
)

target_sources(${TARGET_NAME}
    PRIVATE "${CODEGEN_INCDIR}/${KERNELS_DB_INC}"
)

set_source_files_properties("${CODEGEN_INCDIR}/${KERNELS_DB_INC}" PROPERTIES
    GENERATED TRUE
)
source_group(codegen FILES "${CODEGEN_INCDIR}/${KERNELS_DB_INC}")

target_compile_definitions(${TARGET_NAME}
    PUBLIC CORE_OCL_KERNELS_DB="${KERNELS_DB_INC}"
)

add_dependencies(${TARGET_NAME} ${CODEGEN_TARGET_NAME})

# ======================== Export module info =================================
set(MODULE_IDS                       ${MODULE_ID}                PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_NAME        ${TARGET_NAME}              PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_INCLUDE_DIR ${API_INCLUDE_DIR}          PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_TOOLS_DIR   ${CODEGEN_TOOLSDIR}         PARENT_SCOPE)
