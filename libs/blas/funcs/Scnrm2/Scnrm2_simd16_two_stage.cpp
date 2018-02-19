// Copyright (c) 2017-2018 Intel Corporation
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//      http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "functions/Scnrm2.hpp"

static const char* module_name = "Scnrm2_simd16_two_stage";
static const char* first_kernel_name = "Scnrm2_simd16_first_stage";
static const char* second_kernel_name = "Scnrm2_simd16_second_stage";

namespace iclgpu { namespace functions { namespace implementations {

#define SIMD 16
#define SIMD_PER_GROUP 16
#define WORK_GROUPS 256

#define TILE SIMD*SIMD_PER_GROUP*WORK_GROUPS

bool Scnrm2_simd16_two_stage::accept(const Scnrm2::params& params, Scnrm2::score& score)
{
    if (params.n >= TILE) score.n = 1.3f;
    return true;
}

event Scnrm2_simd16_two_stage::execute(const Scnrm2::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();

    // First stage
    auto first_kernel = engine->get_kernel(first_kernel_name, module_name);
    size_t x_buf_size = params.n * params.incx;
    const size_t group_sums_buf_size = WORK_GROUPS;

    first_kernel->set_arg(0, params.n);
    auto buf_x = engine->get_input_buffer(params.x, x_buf_size);
    first_kernel->set_arg(1, buf_x);
    first_kernel->set_arg(2, params.incx);
    auto buf_group_sums = engine->get_temp_buffer<float>(group_sums_buf_size);
    first_kernel->set_arg(3, buf_group_sums);

    auto gws = nd_range(SIMD, SIMD_PER_GROUP, WORK_GROUPS);
    auto lws = nd_range(SIMD, SIMD_PER_GROUP, 1);
    auto options = kernel_options(gws, lws);
    first_kernel->set_options(options);

    auto event = first_kernel->submit(dep_events);

    // Second stage
    auto second_kernel = engine->get_kernel(second_kernel_name, module_name);
    size_t res_buf_size = 1;

    second_kernel->set_arg(0, buf_group_sums);
    auto buf_res = engine->get_output_buffer(params.result, res_buf_size);
    second_kernel->set_arg(1, buf_res);

    auto gws2 = nd_range(SIMD, SIMD_PER_GROUP);
    auto lws2 = gws2;
    auto options2 = kernel_options(gws2, lws2);
    second_kernel->set_options(options2);

    event = second_kernel->submit({ event });

    return event;
}

} } } // namespace iclgpu::functions::implementations
