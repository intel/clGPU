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

set(MODULE_ID BLAS)

set(CODEGEN_TOOLSDIR ${ICLGPU__CORE_TOOLS_DIR})
set(FUNCTIONS_DEF_FILE ${CMAKE_CURRENT_SOURCE_DIR}/functions.fdef)

list(APPEND CMAKE_MODULE_PATH "${CODEGEN_TOOLSDIR}")
include(Codegen)

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

add_library(${TARGET_NAME}
    STATIC
    ${SOURCES}
    ${HEADERS}
    ${KERNELS}
    ${FUNCTIONS_DEF_FILE}
)

target_include_directories(${TARGET_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${TARGET_NAME} ${ICLGPU__CORE_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES
    FOLDER ${TARGET_FOLDER_NAME}
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# ======================== Autogenerate functions sources =====================
add_codegen_functions(${FUNCTIONS_DEF_FILE})

target_include_directories(${TARGET_NAME}
    PUBLIC ${CODEGEN_INCDIR}
)

target_sources(${TARGET_NAME}
    PRIVATE ${FUNCTION_SOURCES}
)

if(CMAKE_VERSION VERSION_LESS 3.8)
    message(WARNING "CMAKE v.3.8 is recommended for friendly sources grouping in the project")
    source_group(functions
        FILES ${FUNCTION_SOURCES} ${KERNELS} ${FUNCTIONS_DEF_FILE}
    )
else()
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}
        PREFIX functions
        FILES ${FUNCTION_SOURCES} ${KERNELS} ${FUNCTIONS_DEF_FILE}
    )
endif()

# ======================== Autogenerate kernels DB ============================
add_codegen_kernels_db("blas")

set_target_properties(${CODEGEN_TARGET_NAME} PROPERTIES
    FOLDER ${TARGET_FOLDER_NAME}
)

target_include_directories(${TARGET_NAME}
    PUBLIC ${CODEGEN_INCDIR}
)

target_sources(${TARGET_NAME}
    PRIVATE "${CODEGEN_INCDIR}/${KERNELS_DB_INC}"
)

set_source_files_properties("${CODEGEN_INCDIR}/${KERNELS_DB_INC}" PROPERTIES
    GENERATED TRUE
)
source_group(codegen FILES "${CODEGEN_INCDIR}/${KERNELS_DB_INC}")

target_compile_definitions(${TARGET_NAME}
    PUBLIC BLAS_OCL_KERNELS_DB="${KERNELS_DB_INC}"
)

add_dependencies(${TARGET_NAME} ${CODEGEN_TARGET_NAME})

# ======================== Export module info =================================
set(MODULE_IDS                       ${MODULE_ID}                PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_NAME        ${TARGET_NAME}              PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(ICLGPU__${MODULE_ID}_INCLUDE_DIR ${CODEGEN_INCDIR}           PARENT_SCOPE)
