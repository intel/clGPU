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

#include "functions/Isamin.hpp"

static const char* module_name = "Isamin_opt_simd16";
static const char* kernel_name = "Isamin_opt_simd16";

#define SIMD_WIDTH 16
#define LSM_WIDTH 16
#define WORKING_TILE SIMD_WIDTH * LSM_WIDTH

namespace iclgpu { namespace functions { namespace implementations {

bool Isamin_opt_simd16::accept(const Isamin::params& params, Isamin::score& score)
{
    if (params.n >= WORKING_TILE && params.incx > 0)
    {
        score.n = 1.7f;
        return false;
    }
    else
    {
        return false;
    }
}

event Isamin_opt_simd16::execute(const Isamin::params& params, const std::vector<event>& dep_events)
{
    auto engine = context()->get_engine();
    auto kernel = engine->get_kernel(kernel_name, module_name);
    auto vec_size = params.n * params.incx;

    kernel->set_arg(0, params.n);

    auto buf_x = engine->get_input_buffer(params.x, vec_size);
    kernel->set_arg(1, buf_x);

    kernel->set_arg(2, params.incx);

    auto buf_res = engine->get_output_buffer(params.result, 1);
    kernel->set_arg(3, buf_res);


    auto gws = nd_range(WORKING_TILE);
    auto lws = nd_range(WORKING_TILE);

    auto options = kernel_options(gws, lws);
    kernel->set_options(options);

    return kernel->submit(dep_events);
}

} } } // namespace iclgpu::functions::implementations
