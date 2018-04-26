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

macro(build_subdirs)
    if(DEFINED TARGET_NAME)
        set(SUBDIR_NAME_PREFIX "${TARGET_NAME}_")
    else()
        set(SUBDIR_NAME_PREFIX "")
    endif()

    get_filename_component(CURRENT_FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    if(DEFINED TARGET_FOLDER_NAME)
        set(TARGET_FOLDER_NAME "${TARGET_FOLDER_NAME}/${CURRENT_FOLDER_NAME}")
    else()
        set(TARGET_FOLDER_NAME "${CURRENT_FOLDER_NAME}")
    endif()

    unset(SUBMODULE_IDS)

    foreach(SUBDIR_NAME ${ARGV})
        set(TARGET_NAME ${SUBDIR_NAME_PREFIX}${SUBDIR_NAME})
        add_subdirectory(${SUBDIR_NAME})
        list(APPEND SUBMODULE_IDS ${MODULE_IDS})
    endforeach()

    set(MODULE_IDS ${SUBMODULE_IDS} PARENT_SCOPE)

    foreach(MODULE_ID ${SUBMODULE_IDS})
        set(ICLGPU__${MODULE_ID}_NAME        ${ICLGPU__${MODULE_ID}_NAME}        PARENT_SCOPE)
        set(ICLGPU__${MODULE_ID}_SOURCE_DIR  ${ICLGPU__${MODULE_ID}_SOURCE_DIR}  PARENT_SCOPE)
        set(ICLGPU__${MODULE_ID}_INCLUDE_DIR ${ICLGPU__${MODULE_ID}_INCLUDE_DIR} PARENT_SCOPE)
    endforeach()
endmacro()
