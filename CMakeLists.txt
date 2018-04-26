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
include(CheckCXXCompilerFlag)

# Register Intel helper modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/utils/build/cmake/modules")
include(IntelHelpers)

# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 1) ===================================
# ======================================================================================================
# The section must be before project definition which is a point for configurations generation
# for multi-configuration generators.

# Available configuration types:
# Two standard configurations.
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
  )
set(CMAKE_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} CACHE INTERNAL "Available build configurations.")

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

# Name of project (helper constant variable).
set(ICLGPU__PROJ_NAME "iclGPU")

# Default languages: C, C++.
project("${ICLGPU__PROJ_NAME}")


# ======================================================================================================
# ==================================== BUILD CONFIGURATIONS (part 2) ===================================
# ======================================================================================================
# The section must be after project definition and its supported languages because otherwise
# used variables are not available yet.

#intel_custom_build_add("<custom config>" "Release") # No custom configuration at the momeent.
# Populating global property with list of debug configurations.
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS "Debug")
# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# ======================================================================================================
# ====================================== HELPER CONSTANT VARIABLES =====================================
# ======================================================================================================

# Path which points to main directory of project.
set(ICLGPU__MAIN_DIR      "${CMAKE_CURRENT_SOURCE_DIR}")

# Path which points to directory with common dependencies (internal and 3-rd party).
set(ICLGPU__COMMON_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/common")

# Path which points to directory with binaries for Intel OpenCL SDK ICD (Installable Client Driver).
set(ICLGPU__IOCL_ICD_DIR  "${ICLGPU__COMMON_DIR}/intel_ocl_icd")

# Path which points to directory with C++ bindings for OpenCL (header files + wrapper that disables specific warnings).
set(ICLGPU__KHR_CLHPP_DIR "${ICLGPU__COMMON_DIR}/khronos_ocl_clhpp")

# Path which points to directory with fused version of googletest framework (with fused googlemock as well).
set(ICLGPU__GTEST_DIR     "${ICLGPU__COMMON_DIR}/googletest-fused")

# Path which points to directory with end-user API includes.
set(ICLGPU__API_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Path which points to default root directory for compilation output.
set(ICLGPU_BUILD__DEFAULT_OUT_ROOT "${CMAKE_BINARY_DIR}/out")

# Prefix for all targets in internal pass.
set(ICLGPU_BUILD__PROJ_NAME_PREFIX "")

# Single/multi-configuration generator helpers.
if(CMAKE_CFG_INTDIR STREQUAL ".")
  set(ICLGPU__TARGET_CFG_VAR "${CMAKE_BUILD_TYPE}")
  set(ICLGPU__MULTI_CFG_GEN  NO)
else()
  set(ICLGPU__TARGET_CFG_VAR "${CMAKE_CFG_INTDIR}")
  set(ICLGPU__MULTI_CFG_GEN  YES)
endif()

# Code generation settings.

# Path which points to root directory where code generated elements are created.
set(ICLGPU__CODEGEN_DIR     "${CMAKE_CURRENT_BINARY_DIR}/codegen/${ICLGPU__TARGET_CFG_VAR}")

# Path which points to automatically included directory with code generated elements
# (to support "copy-if-different" optimization).
set(ICLGPU__CODEGEN_INCDIR  "${ICLGPU__CODEGEN_DIR}/include")

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

# ======================================================================================================
# =========================================== HELPER FUNCTIONS =========================================
# ======================================================================================================

# Writes debug message in project (it will be presented only if debug is enabled).
#
# @param text Text to present.
function(iclgpu_dmsg text)
  if(ICLGPU__CMAKE_DEBUG)
    message(STATUS "[iclGPU][D] ${text}")
  endif()
endfunction()

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

# Detecting, setting and validating target architecture for compilation.

set(__ICLGPU_RequestedArch "")

intel_arch_detect(__ICLGPU_DetectedArch_Target __ICLGPU_DetectedArch_Host __ICLGPU_RequestedArch)
intel_arch_validate(__ICLGPU_DetectedArchValid_Target "${__ICLGPU_DetectedArch_Target}")
unset(__ICLGPU_DetectedArch_Host)
unset(__ICLGPU_RequestedArch)

if(DEFINED ICLGPU__ARCHITECTURE_TARGET)
  intel_arch_normalize(__ICLGPU_Arch_Target "${ICLGPU__ARCHITECTURE_TARGET}")
elseif(__ICLGPU_DetectedArchValid_Target)
  set(__ICLGPU_Arch_Target "${__ICLGPU_DetectedArch_Target}")
  message("[ICLGPU] ICLGPU__ARCHITECTURE_TARGET: Target architecture is not specified. Trying to deduce it from context.")
else()
  message(FATAL_ERROR "[ICLGPU] ICLGPU__ARCHITECTURE_TARGET: Target architecture is not specified and cannot be deduced from context.
      Please specify one, e.g. Windows32, Linux64, ..."
    )
endif()
set(ICLGPU__ARCHITECTURE_TARGET "${__ICLGPU_Arch_Target}")
unset(__ICLGPU_Arch_Target)
unset(__ICLGPU_DetectedArchValid_Target)

intel_arch_validate(__ICLGPU_ArchValid_Target "${ICLGPU__ARCHITECTURE_TARGET}")
if(NOT __ICLGPU_ArchValid_Target)
  message(FATAL_ERROR "[ICLGPU] ICLGPU__ARCHITECTURE_TARGET: Target architecture \"${ICLGPU__ARCHITECTURE_TARGET}\" is invalid.
      Please specify correct one, e.g. Windows32, Linux64, ..."
    )
endif()
unset(__ICLGPU_ArchValid_Target)

# ======================================================================================================
# ============================================ CMAKE OPTIONS ===========================================
# ======================================================================================================

# CMake: Enables debug trace messages in adapter project.
set(ICLGPU__CMAKE_DEBUG OFF CACHE BOOL "CMake: Enables debug trace messages in iclGPU project.")
mark_as_advanced(ICLGPU__CMAKE_DEBUG)

# ======================================================================================================
# ======================================================================================================
# ======================================================================================================


# ======================================================================================================

# Setting helper variables for component paths.
intel_arch_get_os(__ICLGPU_TargetOs "${ICLGPU__ARCHITECTURE_TARGET}")
string(TOLOWER "${__ICLGPU_TargetOs}" __ICLGPU_TargetOs)

intel_arch_get_cpu(__ICLGPU_TargetCpu "${ICLGPU__ARCHITECTURE_TARGET}")
string(TOLOWER "${__ICLGPU_TargetCpu}" __ICLGPU_TargetCpu)

if(__ICLGPU_TargetCpu STREQUAL "32")
  set(__ICLGPU_TargetCpuDir "x86")
elseif(__ICLGPU_TargetCpu STREQUAL "64")
  set(__ICLGPU_TargetCpuDir "x64")
else()
  set(__ICLGPU_TargetCpuDir "${__ICLGPU_TargetCpu}")
endif()

# ======================================================================================================
# Selecting Intel OpenCL SDK version and path.
set(ICLGPU__IOCL_ICD_ROOT "${ICLGPU__COMMON_DIR}/intel_ocl_icd" CACHE INTERNAL "Path to Intel OpenCL SDK ICD used to build ICLGPU framework.")

# Select SDK subdirectories with headers, binaries and libraries (based on architecture, cpu, generator and SDK type).
set(ICLGPU__IOCL_ICD_INCDIRS "${ICLGPU__IOCL_ICD_ROOT}/${__ICLGPU_TargetOs}/include" CACHE INTERNAL "Paths to interface headers for Intel OpenCL SDK ICD.")
set(ICLGPU__IOCL_ICD_STLDIRS "${ICLGPU__IOCL_ICD_ROOT}/${__ICLGPU_TargetOs}/${ICLGPU__TARGET_CFG_VAR}/lib/${__ICLGPU_TargetCpuDir}" CACHE INTERNAL "Paths to static libraries for Intel OpenCL SDK ICD.")
set(ICLGPU__IOCL_ICD_SHLDIRS "${ICLGPU__IOCL_ICD_ROOT}/${__ICLGPU_TargetOs}/${ICLGPU__TARGET_CFG_VAR}/bin/${__ICLGPU_TargetCpuDir}" CACHE INTERNAL "Paths to shared libraries for Intel OpenCL SDK ICD.")

# Select link directory based on targeted OS.
# - on Windows: static libraries directory.
# - on others:  shared libraries directory.
if(__ICLGPU_TargetOs MATCHES "^windows$")
  set(ICLGPU__IOCL_ICD_LIBDIRS ${ICLGPU__IOCL_ICD_STLDIRS} CACHE INTERNAL "Paths to libraries to link for Intel OpenCL SDK ICD.")
else()
  set(ICLGPU__IOCL_ICD_LIBDIRS ${ICLGPU__IOCL_ICD_SHLDIRS} CACHE INTERNAL "Paths to libraries to link for Intel OpenCL SDK ICD.")
endif()

# ======================================================================================================
# Check for python interpreter (required tool).
find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
  message(WARNING "[iclGPU] Project requires Python interpreter to build (with python loader). CMake could not detect it correctly.
      If you have installed this interpreter, please disregard this warning or specify PYTHON_EXECUTABLE in CMake command-line."
    )
endif()

# ================================================ Outputs =============================================

if(DEFINED ICLGPU__OUTPUT_DIR)
  set(ICLGPU__OUTPUT_BIN_DIR "${ICLGPU__OUTPUT_DIR}" CACHE PATH "Output directory path where the final exetuables, examples and tests will be stored.")
  set(ICLGPU__OUTPUT_LIB_DIR "${ICLGPU__OUTPUT_DIR}" CACHE PATH "Output directory path where the final libraries will be stored.")
else()
  # Output directory path where the final libraries, examples and tests will be stored.
  set(__ICLGPU_OutDir "${ICLGPU_BUILD__DEFAULT_OUT_ROOT}")
  set(ICLGPU__OUTPUT_BIN_DIR "${__ICLGPU_OutDir}" CACHE PATH "Output directory path where the final exetuables, examples and tests will be stored.")
  set(ICLGPU__OUTPUT_LIB_DIR "${ICLGPU__OUTPUT_BIN_DIR}" CACHE PATH "Output directory path where the final libraries will be stored.")
  unset(__ICLGPU_OutDir)
endif()

# ======================================================================================================
# Old.
set(EXECUTABLE_OUTPUT_PATH "${ICLGPU__OUTPUT_BIN_DIR}")
set(LIBRARY_OUTPUT_PATH    "${ICLGPU__OUTPUT_BIN_DIR}")
# New.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${ICLGPU__OUTPUT_LIB_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${ICLGPU__OUTPUT_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${ICLGPU__OUTPUT_BIN_DIR}")

# ============================== Abstraction of compiler and linker options ============================
include("${CMAKE_CURRENT_SOURCE_DIR}/CMakeCompilerLinkerOpts.txt" NO_POLICY_SCOPE)

# ===================================== Generic compiler flags ===========================================
foreach(__ICLGPU_CompilerFlagName IN ITEMS "CMAKE_CXX_FLAGS" "CMAKE_C_FLAGS")
  # Change some generic settings of compiler.
  # NOTE: Debug info generation is enabled for all build configuration, because it is separate on Windows
  #       and we will use "strip" command on Linux (to separate it).
  intel_config_flag_apply_settings(
      CompilerOptions
      "${__ICLGPU_CompilerFlagName}"
      ALL_PATTERN ""
      SET
        CompileAsDefault
        DebugInfoProgramDatabase
        ExceptionsEnabled
        MultiProcessorCompilation
        DeadCodeEliminate
        ExtensionsEnabled
        TreatWarnAsErrorEnabled
        WarnLevel4
        NoFastMath
        StackProtector
    )
  intel_config_flag_apply_settings(
      CompilerOptions
      "${__ICLGPU_CompilerFlagName}"
      ALL_PATTERN_NOINHERIT ""
      SET
        OptimizeSize
    )
  intel_config_flag_apply_settings(
      CompilerOptions
      "${__ICLGPU_CompilerFlagName}"
      PATTERN "^Debug"
      SET
        OptimizeDisabled
    )

  # Adding needed settings specific to MSVC.
  if(MSVC)
    if(CMAKE_COMPILER_IS_INTEL)
      intel_config_flag_apply_settings(
          CompilerOptions
          "${__ICLGPU_CompilerFlagName}"
          ALL_PATTERN ""
          SET_RAW
            "/wd177"
            "/wd367"
            "/wd411"
            "/wd2415"
            "/wd3280"
            "/wd3346"
            "/wd11074"
            "/wd11076"
        )
    else()
      intel_config_flag_apply_settings(
          CompilerOptions
          "${__ICLGPU_CompilerFlagName}"
          ALL_PATTERN ""
          SET_RAW
            "/arch:AVX"
            "/GS"
            "/sdl"
            "/wd4100"
        )
    endif()
  elseif(CMAKE_COMPILER_IS_INTEL)
    message(FATAL_ERROR "TODO Support native ICC")
  # Adding needed settings specific to GCC.
  # NOTE: Following options can be needed in the future (although some not recommended: NR):
  # [NR] -fno-short-enums
  # [NR] -fno-tree-pre
  #      -fno-omit-frame-pointer
  #      -Wa,--noexecstack
  #      -fkeep-inline-functions
  elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    intel_config_flag_apply_settings(
        CompilerOptions
        "${__ICLGPU_CompilerFlagName}"
        ALL_PATTERN ""
        SET_RAW
          -pipe
          -fmessage-length=0
          -fno-strict-aliasing
          -W
          -Wno-unknown-pragmas
          -Wwrite-strings
          -Wswitch
          -Wformat
          -Wformat-security
          -Wno-error=missing-field-initializers
          -Wno-error=unused-parameter
          -Wno-error=unused-function
          -march=corei7
          -mstackrealign
          -msse
          -msse2
          -msse3
          -mssse3
          -msse4
          -msse4.1
          -msse4.2
          -fvisibility=hidden
          -finline
          -finline-functions
          -finline-limit=300
          -funswitch-loops
          -fPIE
          -fPIC
          -Wl,--no-undefined
      )
  elseif((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
    intel_config_flag_apply_settings(
        CompilerOptions
        "${__ICLGPU_CompilerFlagName}"
        ALL_PATTERN ""
        SET_RAW
          -pipe
          -fvisibility=hidden
          -fvisibility-inlines-hidden
          -Wall
          -Wno-covered-switch-default
          -Wextra
          -Wno-unused-parameter
          -Wno-gnu
          -pedantic
          -finline
          -msse4.2
          -fPIE
          -fPIC
      )
  endif()
endforeach()

# C++ only options.
intel_config_flag_apply_settings(
  CompilerOptions
  CMAKE_CXX_FLAGS
  ALL_PATTERN ""
  SET
    StandardCxx14
)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  intel_config_flag_apply_settings(
      CompilerOptions
      CMAKE_CXX_FLAGS
      ALL_PATTERN ""
      SET_RAW
        -fno-operator-names
        -fpermissive
        -fvisibility-inlines-hidden
    )
endif()

# NOTE: When compiling with Clang, use the flags below for C++ files. These flags cannot be enabled when compiling
#       C files.
if((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
  intel_config_flag_apply_settings(
      CompilerOptions
      CMAKE_CXX_FLAGS
      ALL_PATTERN ""
      SET_RAW
        -stdlib=libc++
    )
endif()

unset(__ICLGPU_CompilerFlagName)

# ======================================== Generic linker options ======================================
# Additional libraries that needs to be linked to shared objects/executables.
set(ICLGPU__SYSTEM_LINK_LIBRARIES)

foreach(__ICLGPU_LinkerFlagName IN ITEMS "CMAKE_EXE_LINKER_FLAGS" "CMAKE_SHARED_LINKER_FLAGS")
  # Change some generic settings of linker.
  # NOTE: Debug info generation is enabled for all build configuration, because it is separate on Windows
  #       and we will use "strip" command on Linux (to separate it).
  intel_config_flag_apply_settings(
      LinkerOptions
      "${__ICLGPU_LinkerFlagName}"
      ALL_PATTERN ""
      SET
        DebugInfoProgramDatabase
        DeadCodeEliminate
        IdenticalCodeDataFold
        IncrementalDisabled
    )
endforeach()

unset(__ICLGPU_LinkerFlagName)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  intel_config_flag_apply_settings(
        LinkerOptions
        CMAKE_EXE_LINKER_FLAGS
        ALL_PATTERN ""
        SET_RAW
          -pie
          -Wl,-z,noexecstack,-z,relro,-z,now
      )
  intel_config_flag_apply_settings(
        LinkerOptions
        CMAKE_SHARED_LINKER_FLAGS
        ALL_PATTERN ""
        SET_RAW
          -Wl,-z,noexecstack,-z,relro,-z,now
      )
endif()

if((CMAKE_C_COMPILER_ID MATCHES "^Clang$") OR (CMAKE_CXX_COMPILER_ID MATCHES "^Clang$"))
  intel_config_flag_apply_settings(
      LinkerOptions
      CMAKE_SHARED_LINKER_FLAGS
      ALL_PATTERN ""
      SET_RAW
        -shared
        -Wl,-undefined,dynamic_lookup
        -Wl,-headerpad_max_install_names
    )

  list(APPEND ICLGPU__SYSTEM_LINK_LIBRARIES "c++" "c++abi" "supc++")
endif()

# ======================================================================================================

iclgpu_dmsg(STATUS " - compile/link flags:")
foreach(__ICLGPU_Flag IN ITEMS "CMAKE_CXX_FLAGS" "CMAKE_C_FLAGS" "CMAKE_EXE_LINKER_FLAGS" "CMAKE_SHARED_LINKER_FLAGS")
  iclgpu_dmsg("    + ${__ICLGPU_Flag}:     ${${__ICLGPU_Flag}}")
  foreach(__ICLGPU_CfgType ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${__ICLGPU_Flag}_${__ICLGPU_CfgType}" __ICLGPU_CfgFlag)
    if(DEFINED "${__ICLGPU_CfgFlag}")
      iclgpu_dmsg("    + ${__ICLGPU_CfgFlag}:     ${${__ICLGPU_CfgFlag}}")
    endif()
  endforeach()
  iclgpu_dmsg("")
endforeach()
unset(__ICLGPU_Flag)
unset(__ICLGPU_CfgType)
unset(__ICLGPU_CfgFlag)

# ============================== Generic compiler preprocessor definitions =============================

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS
    $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
  )
endif()

# ===================================== Include/Link directories =======================================

include_directories(
    ${ICLGPU__IOCL_ICD_INCDIRS}
)

link_directories(
    ${ICLGPU__IOCL_ICD_LIBDIRS}
)

# =================================== Link targets and dependencies ====================================
add_subdirectory(core)
add_subdirectory(libs)
add_subdirectory(tests)
