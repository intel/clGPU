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

set(TARGET_NAME "blas_funcs")

set(MODULE_ID BLAS)

set(CODEGEN_DIR      ${CMAKE_CURRENT_BINARY_DIR}/codegen)
set(CODEGEN_INCDIR   ${CODEGEN_DIR}/include)
set(CODEGEN_CACHEDIR ${CODEGEN_DIR}/cache)
set(CODEGEN_TOOLSDIR ${ICLGPU__CORE_TOOLS_DIR})
set(PRIMITIVE_DB_INC "ocl_kernels.inc")
set(PRIMITIVE_DB_INC_SOURCE ${CODEGEN_INCDIR}/${PRIMITIVE_DB_INC})

set(FUNCTIONS_DEF_FILE ${CMAKE_CURRENT_SOURCE_DIR}/functions.def)
set(FUNCTIONS_DEF_CODEGEN ${CODEGEN_CACHEDIR}/functions.def)
set(FUNCTIONS_INCDIR ${CODEGEN_INCDIR}/functions)

set(ICLGPU__MODULE_ID                ${MODULE_ID}                PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_NAME        ${TARGET_NAME}              PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_INCLUDE_DIR ${CODEGEN_INCDIR}           PARENT_SCOPE)

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
)

file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

file(GLOB_RECURSE KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
)

if(CMAKE_VERSION VERSION_LESS 3.8)
message(WARNING "CMAKE v.3.8 is recommended for friendly sources grouping in the project")
source_group(functions FILES
    ${SOURCES}
    ${KERNELS}
)
else()
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX functions FILES
    ${SOURCES}
    ${KERNELS}
)
endif()

source_group(codegen FILES ${PRIMITIVE_DB_INC_SOURCE} ${FUNCTIONS_DEF_CODEGEN})

include_directories(
    ${ICLGPU__CORE_INCLUDE_DIR}
    ${CODEGEN_INCDIR}
)

add_library(${TARGET_NAME}
    STATIC
    ${SOURCES}
    ${HEADERS}
    ${KERNELS}
    ${PRIMITIVE_DB_INC_SOURCE}
    ${FUNCTIONS_DEF_FILE}
    ${FUNCTIONS_DEF_CODEGEN}
)

target_link_libraries(${TARGET_NAME} ${ICLGPU__CORE_NAME})
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 14)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD_REQUIRED)
set_property(TARGET ${TARGET_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

# =================================== Custom pre- and post-steps =======================================
add_custom_command(OUTPUT ${FUNCTIONS_DEF_CODEGEN}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${FUNCTIONS_INCDIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${FUNCTIONS_DEF_FILE} ${FUNCTIONS_DEF_CODEGEN}
    COMMAND "${PYTHON_EXECUTABLE}" "${CODEGEN_TOOLSDIR}/functions_hpp_gen.py" "${FUNCTIONS_DEF_CODEGEN}" "${CMAKE_CURRENT_SOURCE_DIR}" "${FUNCTIONS_INCDIR}"
    DEPENDS "${FUNCTIONS_DEF_FILE}" "${CODEGEN_TOOLSDIR}/functions_hpp_gen.py" "${CODEGEN_TOOLSDIR}/functions_def.py"
    COMMENT "Generating functions codes..."
)

add_custom_command(OUTPUT ${CODEGEN_CACHEDIR}/${PRIMITIVE_DB_INC}
    COMMAND "${CMAKE_COMMAND}" -E make_directory ${CODEGEN_CACHEDIR}
    COMMAND "${PYTHON_EXECUTABLE}" "${CODEGEN_TOOLSDIR}/primitive_db_gen.py" "${CODEGEN_CACHEDIR}/${PRIMITIVE_DB_INC}" "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS ${KERNELS} ${CODEGEN_TOOLSDIR}/primitive_db_gen.py
    COMMENT "Generating ${PRIMITIVE_DB_INC} ..."
)

add_custom_command(OUTPUT ${PRIMITIVE_DB_INC_SOURCE}
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHEDIR}/${PRIMITIVE_DB_INC}" "${PRIMITIVE_DB_INC_SOURCE}"
    DEPENDS ${CODEGEN_CACHEDIR}/${PRIMITIVE_DB_INC} ${KERNELS} ${CODEGEN_TOOLSDIR}/primitive_db_gen.py
    COMMENT "Updating file if the file changed (${PRIMITIVE_DB_INC}) ..."
)
