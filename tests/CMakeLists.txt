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

# =========================================== Compiler options =========================================
include(BuildSubdirs)

intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
    SET
      WarnLevel3
  )
if (NOT MSVC)
  intel_config_flag_apply_settings(CompilerOptions CMAKE_CXX_FLAGS ALL_PATTERN ""
      SET_RAW
        "-Wno-error=conversion-null"
        "-Wno-error=type-limits"
        "-Wno-error=unused-variable"
    )
endif ()

include_directories(${ICLGPU__GTEST_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB SOURCE_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/gtest_main.cpp)
file(GLOB SOURCE_GTEST_FUSED ${ICLGPU__GTEST_DIR}/*.cc)

set(TARGET_NAME "test")
build_subdirs("blas_reference" "core" "blas_funcs" "iclBLAS")
