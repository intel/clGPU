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

set(MODULE_ID CORE)

set(ICLGPU__MODULE_ID                ${MODULE_ID}                PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_NAME        ${TARGET_NAME}              PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_INCLUDE_DIR ${API_INCLUDE_DIR}          PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_TOOLS_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/tools PARENT_SCOPE)

set(CODEGEN_DIR      ${CMAKE_CURRENT_BINARY_DIR}/codegen)
set(CODEGEN_INCDIR   ${CODEGEN_DIR}/include)
set(CODEGEN_CACHEDIR ${CODEGEN_DIR}/cache)
set(CODEGEN_TOOLSDIR ${CMAKE_CURRENT_SOURCE_DIR}/tools)
set(PRIMITIVE_DB_INC "ocl_kernels.inc")
set(PRIMITIVE_DB_INC_SOURCE ${CODEGEN_INCDIR}/${PRIMITIVE_DB_INC})

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)

file(GLOB HEADERS
    ${API_INCLUDE_DIR}/*.hpp
    ${API_INCLUDE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)

file(GLOB_RECURSE OCL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/*.cpp
    ${API_INCLUDE_DIR}/ocl/*.hpp
    ${API_INCLUDE_DIR}/ocl/*.h
)

file(GLOB_RECURSE OCL_KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ocl/*.cl
)

source_group(codegen FILES ${PRIMITIVE_DB_INC_SOURCE})
source_group(ocl FILES ${OCL_FILES} ${OCL_KERNELS})

include_directories(
    ${API_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCL_INCLUDE_DIRS}
    ${ICLGPU__KHR_CLHPP_DIR}
    ${CODEGEN_INCDIR}
)

add_library(${TARGET_NAME} STATIC
    ${HEADERS}
    ${SOURCES}
    ${OCL_FILES}
    ${PRIMITIVE_DB_INC_SOURCE}
)

target_link_libraries(${TARGET_NAME} OpenCL)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED)
set_property(TARGET ${TARGET_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# =================================== Custom pre- and post-steps =======================================

add_custom_command(OUTPUT ${CODEGEN_CACHEDIR}/${PRIMITIVE_DB_INC}
    COMMAND "${CMAKE_COMMAND}" -E make_directory ${CODEGEN_CACHEDIR}
    COMMAND "${PYTHON_EXECUTABLE}" "${CODEGEN_TOOLSDIR}/primitive_db_gen.py" "${CODEGEN_CACHEDIR}/${PRIMITIVE_DB_INC}" "${CMAKE_CURRENT_SOURCE_DIR}/src/ocl"
    DEPENDS ${OCL_KERNELS} ${CODEGEN_TOOLSDIR}/primitive_db_gen.py
    COMMENT "Generating ${PRIMITIVE_DB_INC} ..."
)

add_custom_command(OUTPUT ${PRIMITIVE_DB_INC_SOURCE}
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHEDIR}/${PRIMITIVE_DB_INC}" "${PRIMITIVE_DB_INC_SOURCE}"
    DEPENDS ${CODEGEN_CACHEDIR}/${PRIMITIVE_DB_INC} ${OCL_KERNELS} ${CODEGEN_TOOLSDIR}/primitive_db_gen.py
    COMMENT "Updating file if the file changed (${PRIMITIVE_DB_INC}) ..."
)
