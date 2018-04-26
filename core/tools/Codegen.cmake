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

cmake_minimum_required (VERSION 3.5 FATAL_ERROR)

macro(define_codegen_variables)
    if(NOT DEFINED PYTHON_EXECUTABLE)
        find_package(PythonInterp REQUIRED)
    endif()

    if(NOT DEFINED CODEGEN_DIR)
        set(CODEGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/codegen)
    endif()

    set(CODEGEN_INCDIR ${CODEGEN_DIR}/include)
    set(CODEGEN_CACHEDIR ${CODEGEN_DIR}/cache)

    if(NOT DEFINED CODEGEN_TOOLSDIR)
        set(CODEGEN_TOOLSDIR ${ICLGPU__CORE_TOOLS_DIR})
    endif()
endmacro()

#.rst:
# add_codegen_functions
# ------------------
#
# Generate functions headers and sources.
#
# add_codegen_functions(targetName def_file [sources_dir [headers_dir]])
#
# ::
#
#   targetName  - target name to which generated sources to be added
#
#   def_file    - functions definition file
#   sources_dir - directory where generated sources will be stored
#                   Default: ${CMAKE_CURRENT_SOURCE_DIR}
#   headers_dir - directory where generated headers will be stored
#                   Default: ${CMAKE_CURRENT_BINARY_DIR}/codegen/include

function(add_codegen_functions functionsDef)
    define_codegen_variables()

    set(FUNCTIONS_DEF_FILE ${functionsDef})
    set(FUNCTIONS_INCDIR ${CODEGEN_INCDIR}/functions)
    set(FUNCTION_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR})


    if(ARGV GREATER 1)
        set(FUNCTION_SOURCES_DIR ${ARGV1})
    endif()
    if(ARGV GREATER 2)
        set(FUNCTIONS_INCDIR ${ARGV2})
    endif()

    set(FUNCTION_SOURCES_LIST_NAME "functions.clist")
    set(FUNCTION_SOURCES_LIST "${CODEGEN_INCDIR}/${FUNCTION_SOURCES_LIST_NAME}")
    if(NOT EXISTS ${FUNCTION_SOURCES_LIST})
        message("Initially generating functions codes...")
        file(MAKE_DIRECTORY ${CODEGEN_INCDIR} ${FUNCTIONS_INCDIR})
        execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" "${CODEGEN_TOOLSDIR}/functions_hpp_gen.py" "${FUNCTIONS_DEF_FILE}" "${FUNCTION_SOURCES_DIR}" "${FUNCTIONS_INCDIR}"
            OUTPUT_VARIABLE FUNCTION_SOURCES_TMP
        )
        file(WRITE ${FUNCTION_SOURCES_LIST} "set(FUNCTION_SOURCES ${FUNCTION_SOURCES_TMP})")
    endif()

    set(FUNCTION_SOURCES_LIST_CACHE "${CODEGEN_CACHEDIR}/${FUNCTION_SOURCES_LIST_NAME}")
    add_custom_command(OUTPUT ${FUNCTION_SOURCES_LIST}
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CODEGEN_CACHEDIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${FUNCTIONS_INCDIR}"
        COMMAND "${CMAKE_COMMAND}" -E echo "set(FUNCTION_SOURCES" > "${FUNCTION_SOURCES_LIST_CACHE}"
        COMMAND "${PYTHON_EXECUTABLE}" "${CODEGEN_TOOLSDIR}/functions_hpp_gen.py" "${FUNCTIONS_DEF_FILE}" "${FUNCTION_SOURCES_DIR}" "${FUNCTIONS_INCDIR}" >> "${FUNCTION_SOURCES_LIST_CACHE}"
        COMMAND "${CMAKE_COMMAND}" -E echo ")" >> "${FUNCTION_SOURCES_LIST_CACHE}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${FUNCTION_SOURCES_LIST_CACHE}" "${FUNCTION_SOURCES_LIST}"
        DEPENDS "${FUNCTIONS_DEF_FILE}" "${CODEGEN_TOOLSDIR}/functions_hpp_gen.py" "${CODEGEN_TOOLSDIR}/functions_def.py"
        MAIN_DEPENDENCY "${FUNCTIONS_DEF_FILE}"
        COMMENT "Generating functions codes..."
    )

    include(${FUNCTION_SOURCES_LIST})

    set(CODEGEN_INCDIR ${CODEGEN_INCDIR} PARENT_SCOPE)
    set(FUNCTION_SOURCES ${FUNCTION_SOURCES} PARENT_SCOPE)
endfunction()

#.rst:
# add_codegen_kernels_db
# ------------------
#
# Generate kernels database include file.
#
# add_codegen_kernels_db(targetName [output_file [sources_dir]])
#
# ::
#
#   targetName    - target name to which generated source will be added
#
#   output_file - generated file path
#                   Default: ${CMAKE_CURRENT_BINARY_DIR}/codegen/include/ocl_kernels.inc
#   sources_dir - directory where kernels are stored
#                   Default: ${CMAKE_CURRENT_SOURCE_DIR}

function(add_codegen_kernels_db Prefix)
    define_codegen_variables()

    set(KERNELS_DB_INC "${Prefix}_ocl_kernels.inc")
    set(KERNELS_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    if(ARGC GREATER 1)
        set(KERNELS_SOURCES_DIR ${ARGV1})
    endif()

    set(CODEGEN_TARGET_NAME "${Prefix}_ocl_kernels_db")

    if(NOT EXISTS ${CODEGEN_INCDIR}/${KERNELS_DB_INC})
        message("Initially generating ${CODEGEN_INCDIR}/${KERNELS_DB_INC}...")
        file(MAKE_DIRECTORY ${CODEGEN_INCDIR})
        execute_process(
            COMMAND "${PYTHON_EXECUTABLE}" "${CODEGEN_TOOLSDIR}/primitive_db_gen.py" "${CODEGEN_INCDIR}/${KERNELS_DB_INC}" "${KERNELS_SOURCES_DIR}"
        )
    endif()

    add_custom_target(${CODEGEN_TARGET_NAME} ALL
        COMMAND "${CMAKE_COMMAND}" -E make_directory ${CODEGEN_CACHEDIR}
        COMMAND "${PYTHON_EXECUTABLE}" "${CODEGEN_TOOLSDIR}/primitive_db_gen.py" "${CODEGEN_CACHEDIR}/${KERNELS_DB_INC}" "${KERNELS_SOURCES_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${CODEGEN_CACHEDIR}/${KERNELS_DB_INC}" "${CODEGEN_INCDIR}/${KERNELS_DB_INC}"
        DEPENDS ${CODEGEN_TOOLSDIR}/primitive_db_gen.py
        COMMENT "Generating ${CODEGEN_CACHEDIR}/${KERNELS_DB_INC} ..."
    )

    set(CODEGEN_INCDIR ${CODEGEN_INCDIR} PARENT_SCOPE)
    set(KERNELS_DB_INC ${KERNELS_DB_INC} PARENT_SCOPE)
    set(CODEGEN_TARGET_NAME ${CODEGEN_TARGET_NAME} PARENT_SCOPE)
endfunction()
