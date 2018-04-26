/* Copyright (c) 2017-2018 Intel Corporation
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

#ifndef SROTM_HELPERS_H
#define SROTM_HELPERS_H

#define PRIMITIVE_CAT(a, b) a ## b
#define CAT(a, b) PRIMITIVE_CAT( a, b )

#define ROT_FULL 0
#define ROT_DIAGONAL_ONES 1
#define ROT_ANTI_DIAGONAL_ONES 2

// Calculation for ROT_FULL
#define ROT_IMPL_0(new_x, new_y, this_x, this_y, param)    \
    new_x = mad(param[1], this_x, param[2] * this_y);   \
    new_y = mad(param[3], this_x, param[4] * this_y)

// Calculation for ROT_DIAGONAL_ONES
#define ROT_IMPL_1(new_x, new_y, this_x, this_y, param)    \
    new_x = mad(param[2], this_y, this_x);              \
    new_y = mad(param[3], this_x, this_y)

// Calculation for ROT_ANTI_DIAGONAL_ONES
#define ROT_IMPL_2(new_x, new_y, this_x, this_y, param)    \
    new_x = mad(param[1], this_x, this_y);              \
    new_y = mad(param[4], this_y, -this_x)

#define ROT_IMPL(rot_type, new_x, new_y, this_x, this_y, param) \
     CAT( ROT_IMPL_, rot_type ) ( new_x, new_y, this_x, this_y, param )

#endif /* SROTM_HELPERS_H */
