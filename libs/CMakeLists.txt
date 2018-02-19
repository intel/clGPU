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

foreach(DIR_NAME "blas/funcs" "blas/shlib")
    add_subdirectory(${DIR_NAME})

    set(ICLGPU__${ICLGPU__MODULE_ID}_NAME        ${ICLGPU__${ICLGPU__MODULE_ID}_NAME}        PARENT_SCOPE)
    set(ICLGPU__${ICLGPU__MODULE_ID}_SOURCE_DIR  ${ICLGPU__${ICLGPU__MODULE_ID}_SOURCE_DIR}  PARENT_SCOPE)
    set(ICLGPU__${ICLGPU__MODULE_ID}_INCLUDE_DIR ${ICLGPU__${ICLGPU__MODULE_ID}_INCLUDE_DIR} PARENT_SCOPE)

endforeach(DIR_NAME)
