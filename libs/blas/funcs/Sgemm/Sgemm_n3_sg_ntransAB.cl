/* Copyright (c) 2018 Intel Corporation
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// Parameters:
#define TARG_KERNEL_NAME    Sgemm_n3_sg_ntransAB
#define TARG_SG_SIZE        16
#define TARG_DATA_TYPE      float

#define TARG_MATRIX_FMT_A   C
#define TARG_MATRIX_FMT_B   C

#define TARG_TILE_M    16
#define TARG_TILE_N     8
#define TARG_TILE_AK   16
#define TARG_TILE_BK   16

#define TARG_TILE_IDX_GDIM_M   0
#define TARG_TILE_IDX_GDIM_N   1


// Template instantiation:
#include <gemm_n3_template.h>

