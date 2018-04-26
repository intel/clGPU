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

#include "functions/Isamax.hpp"

static const char* module_name = "Isamax_opt_simd16_2stage";
static const char* first_stage_kernel_name = "Isamax_opt_simd16_2stage";
static const char* second_stage_kernel_name = "Isamax_opt_simd16_2stage_2";

#define SIMD_WIDTH 16
#define LOCAL_GROUP_SIZE 256
#define LOCAL_GROUP_NUMBER 256

#define WORKING_THREADS (LOCAL_GROUP_SIZE * LOCAL_GROUP_NUMBER)

namespace iclgpu { namespace functions { namespace implementations {

bool Isamax_opt_simd16_2stage::accept(const Isamax::params& params, Isamax::score& score)
{
    if (params.n >= WORKING_THREADS && (params.incx > 0))
    {
        score.n = 1.85f;
        score.incx = 1.1f;
        return true;
    }
    else
    {
        return false;
    }
}

event Isamax_opt_simd16_2stage::execute(const Isamax::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(first_stage_kernel_name, module_name);
    auto vec_size = params.n * params.incx;

    kernel->set_arg(0, params.n);

    auto buf_x = engine->get_input_buffer(params.x, vec_size);
    kernel->set_arg(1, buf_x);

    kernel->set_arg(2, params.incx);

    auto a256_tempres_buf = engine->get_temp_buffer<float>(LOCAL_GROUP_NUMBER * 2);
    kernel->set_arg(3, a256_tempres_buf);


    auto gws = nd_range(WORKING_THREADS);
    auto lws = nd_range(256);

    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    auto event = kernel->submit(dep_events);

    /* ------------------------------- Second stage ------------------------------------ */
    auto second_kernel = engine->get_kernel(second_stage_kernel_name, module_name);

    second_kernel->set_arg(0, LOCAL_GROUP_NUMBER);

    second_kernel->set_arg(1, a256_tempres_buf);

    auto res_buffer = engine->get_output_buffer(params.result, 1);
    second_kernel->set_arg(2, res_buffer);

    auto gws2 = nd_range(LOCAL_GROUP_NUMBER);
    auto lws2 = nd_range(LOCAL_GROUP_NUMBER);
    auto options2 = kernel_options(gws2, lws2);
    second_kernel->set_options(options2);

    event = second_kernel->submit({ event });

    return event;
}

} } } // namespace iclgpu::functions::implementations
